#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceMixer {
  DeviceTensor norm;
  DeviceTensor down;
  DeviceTensor up;
  DeviceTensor inject;
};

struct DeviceLayer {
  bool linear{false};
  DeviceMixer hc_attn;
  DeviceMixer hc_ffn;

  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;
  /// qkv and gate rows stacked ([hidden -> conv channels + value dim]) when
  /// both are Q8_0; then ssm_qkv/ssm_gate are empty.
  DeviceTensor ssm_in;
  /// alpha and beta rows stacked: [hidden -> 2 * v_heads] F16.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm,
      indexer_q, indexer_k, indexer_q_norm, indexer_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  DeviceTensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv,
      ple_conv1d;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1] F16.
  DeviceTensor router;
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up,
      shexp_down;
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden;
  DeviceMixer nextn_head;
};

/// The trunk (and optionally the MTP draft block) uploaded to the GPU. The
/// n-gram table is never uploaded: it is read from disk per token.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, const core::GgufReader* mtp_reader,
      std::string* error_msg = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  /// The leading output() rows the MTP draft head scores in place (see
  /// draft_vocab.hpp); empty when the draft projects the full vocabulary.
  const DeviceTensor& draft_output() const noexcept { return draft_output_; }
  /// Special tokens above the prefix: the first rows of each session's
  /// draft table.
  std::span<const std::int32_t> draft_specials() const noexcept {
    return draft_specials_;
  }
  /// Bytes of one output() row.
  std::size_t output_row_bytes() const noexcept {
    return output_.rows == 0 ? 0 : output_bytes_ / output_.rows;
  }
  /// Drafts score output() rows [0, prefix_rows) in place plus a
  /// per-session table that starts with `specials`. Draft proposals only
  /// steer speculation: the target still verifies every token, so greedy
  /// text is unchanged. Sampled text can differ: a smaller head changes
  /// acceptance, and so the later random draws.
  [[nodiscard]] bool SetDraftVocab(std::uint32_t prefix_rows,
                                   std::vector<std::int32_t> specials,
                                   std::string* error_msg);
  /// Rows the draft head must never propose (1 = masked), by token ID: the
  /// medium level's default suppression (logit_bias_default.hpp). The device
  /// copy is read by MaskSuppressedLogits; nothing else changes.
  [[nodiscard]] bool SetSuppressedRows(std::vector<std::uint8_t> flags,
                                       std::string* error_msg);
  [[nodiscard]] const std::uint8_t* suppressed_device() const noexcept {
    return suppressed_device_;
  }
  [[nodiscard]] bool IsSuppressed(std::int32_t id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < suppressed_.size() &&
           suppressed_[static_cast<std::size_t>(id)] != 0;
  }
  /// The trimmed target head (head_trim.hpp): the set's rows of output() in a
  /// compact copy, and the token ID -> compact row map.
  struct HeadTrim {
    DeviceTensor head;                     ///< rows = padded compact rows
    const std::int32_t* inverse{nullptr};  ///< device, vocab entries
    const std::int32_t* ids{nullptr};      ///< device, compact row -> token ID
    std::uint32_t scored{0};               ///< compact rows that are real
    float threshold{0.0F};                 ///< fallback below this top prob
    bool timing{false};  ///< stamp device timestamps (--vocab-trim-timing)
    bool counters{
        false};  ///< count what the fallback buys (--vocab-trim-counters)
    bool discard_aware{false};  ///< --vocab-trim-discard-aware
    bool rowlog{false};         ///< --vocab-trim-rowlog
    /// --vocab-trim-tier2: a flagged row is first re-scored over the first
    /// `tier2_rows` token ids (0 = off), and goes on to the full matrix only
    /// if its top probability over those and the compact set is under
    /// `tier2_threshold`. `tier2_tail` is how many compact rows lie below
    /// `tier2_rows`.
    std::uint32_t tier2_rows{0};
    float tier2_threshold{0.0F};
    std::uint32_t tier2_tail{0};
    /// --vocab-trim-fallback-skip-scored: the full pass reads only the
    /// `n_unscored` head rows the compact set does not hold (`unscored`, ids
    /// ascending, device) and keeps the compact values of the scored ones.
    bool skip_scored{false};
    const std::int32_t* unscored{nullptr};
    std::uint32_t n_unscored{0};
  };
  /// Gathers output() rows `ids` (ascending token IDs, `padded_rows` >= their
  /// count) into a compact device copy and keeps `inverse` (one entry per
  /// output row, -1 for a row that is not scored) beside it. Needs a Q8_0
  /// head. The compact copy is read-only after this.
  [[nodiscard]] bool SetHeadTrim(std::span<const std::int32_t> ids,
                                 std::span<const std::int32_t> inverse,
                                 std::uint32_t padded_rows, float threshold,
                                 std::string* error_msg);
  /// Turns the head's device-side timing on (default off); needs SetHeadTrim.
  void SetHeadTrimTiming(bool on) noexcept { head_trim_.timing = on; }
  /// Turns the fallback's measurement counters on (default off); needs
  /// SetHeadTrim.
  void SetHeadTrimCounters(bool on) noexcept { head_trim_.counters = on; }
  /// Logs the rows of every head call by call site (default off); needs
  /// SetHeadTrim.
  void SetHeadTrimRowLog(bool on) noexcept { head_trim_.rowlog = on; }
  /// Discard-aware flagging (--vocab-trim-discard-aware; default off): chains
  /// do not pay a full-head read for rows that cannot be used. Needs
  /// SetHeadTrim and a threshold above 0.
  void SetHeadTrimDiscardAware(bool on) noexcept {
    head_trim_.discard_aware = on && head_trim_.threshold > 0.0F;
  }
  /// Turns the second tier of the fallback on (default off): `rows` leading
  /// token ids, the tier's threshold (negative: the fallback's own), the
  /// compact set's ids. Needs SetHeadTrim and a threshold above 0; `rows` must
  /// be below the vocabulary.
  [[nodiscard]] bool SetHeadTrimTier2(std::uint32_t rows, float threshold,
                                      std::span<const std::int32_t> ids,
                                      std::string* error_msg);
  /// Turns on skipping the scored rows in the full pass (default off); needs
  /// SetHeadTrim. `inverse` is the same map SetHeadTrim took.
  [[nodiscard]] bool SetHeadTrimSkipScored(
      bool on, std::span<const std::int32_t> inverse, std::string* error_msg);
  [[nodiscard]] const HeadTrim* head_trim() const noexcept {
    return head_trim_.head.empty() ? nullptr : &head_trim_;
  }
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  std::size_t output_bytes_{0};
  DeviceTensor draft_output_;
  std::vector<std::int32_t> draft_specials_;
  std::vector<std::uint8_t> suppressed_;
  const std::uint8_t* suppressed_device_{nullptr};
  HeadTrim head_trim_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
