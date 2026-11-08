#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/sampling.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::kernels {

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream);

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream);

/// Score scratch, in floats, that `LaunchDFlashNonCausalAttention` needs for
/// blocks of up to `block_size` tokens over a `history_capacity` ring.
[[nodiscard]] std::size_t DFlashAttentionScratchElements(
    std::uint32_t block_size, std::uint32_t num_q_heads,
    std::uint32_t num_kv_heads, std::uint32_t history_capacity) noexcept;

/// `scores` holds `scores_elements` floats of stream-ordered scratch, sized by
/// `DFlashAttentionScratchElements`.
void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* scores, std::size_t scores_elements);

/// Thread-per-key reference route of the call above. Exposed so an equivalence
/// test can run both routes in one process; the production entry point picks
/// between them.
void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Pair-sharing route of the call above: three passes that read every K and
/// V row once per KV head. Requires `head_dim` to be a multiple of four, at
/// most 128.
void LaunchDFlashNonCausalAttentionPairs(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* scores, std::size_t scores_elements);

/// Quantizes a BF16 weight matrix into a Q8_0 copy for the draft-only LM head.
/// `total_elements` must be a multiple of the Q8_0 block size.
void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream);

[[nodiscard]] std::size_t DFlashSelectorScratchElements(
    std::uint32_t vocab_size) noexcept;

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream);

struct DFlashSampling {
  float repeat_penalty{1.0F};
  float frequency_penalty{0.0F}, presence_penalty{0.0F};
  std::uint32_t repeat_last_n{0};
  const sampling::TokenPenalty* penalties{nullptr};
  std::uint32_t penalty_count{0};
  const sampling::TokenId* history{nullptr};
  std::uint32_t history_count{0};
  /// Draw a sampled slot from 2^24 integer masses (floor of each weight's
  /// share, the remainder on candidate 0, as SampleMtpProposal does) and export
  /// each candidate's probability as mass * 2^-24, exact in float. The exported
  /// q is then the distribution actually sampled, bit for bit, which
  /// Flash-Next's exact verifier requires. Off: the float draw the Qwen 27B
  /// path uses.
  std::uint32_t integer_masses{0};
  /// The draft head scores a compact subset of the vocabulary (Flash-Next's
  /// trimmed head, item 2.1): row r of the head is token row_ids[r]. The
  /// selector maps its candidates to token ids before the codebooks,
  /// penalties and output read them. Null: rows are token ids.
  const std::int32_t* row_ids{nullptr};
};

struct DFlashSelectorSequence {
  std::uint32_t count;
  float temperature;
  DFlashSampling sampling;
};

// Proposal rows are contiguous by sequence. Each token sequence includes its
// anchor before the proposal slots; all other buffers contain proposals only.
// Partial buffers need DFlashSelectorScratchElements(vocab_size) per proposal.
// Requested candidates include one-hot probabilities for greedy sequences.
void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream);

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
