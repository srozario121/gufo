#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_TRIE_SHADOW_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_TRIE_SHADOW_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace gufo::models::qwen38_flash_next {

/// Trie v3's shadow mode (srozario121/gufo_mtp_continuous_learning item
/// 1.32): where v3 would append a trie proposal after a finished MTP chain,
/// record the proposal and what the target emitted at that position, and
/// change nothing. GUFO_TRIE_SHADOW names the JSON-lines file; unset, no trie
/// is indexed and nothing is written.
///
/// The record's `match` estimates v3's acceptance without reading any row: a
/// point-mass proposal is accepted with the target's probability of its
/// token, and the emitted token is drawn from that same distribution, so
/// P(emitted == trie token) is that probability for sampled rows (and the
/// exact greedy acceptance for greedy rows).
enum class TrieShadowReason : std::uint8_t {
  kNone,
  kGate,   ///< the chain ended under the MTP confidence gate (gufo #882)
  kWidth,  ///< the chain filled the length controller's width
};

/// Where v3 would fire: a chain without prompt-lookup copies, with room left
/// under the cycle's cap, whose last token is not a stop token.
[[nodiscard]] constexpr TrieShadowReason TrieShadowFires(
    std::uint32_t lookup_copies, std::size_t chain, std::size_t cap,
    bool gate_stopped, bool last_is_stop) noexcept {
  if (lookup_copies != 0 || last_is_stop || chain == 0 || chain >= cap)
    return TrieShadowReason::kNone;
  return gate_stopped ? TrieShadowReason::kGate : TrieShadowReason::kWidth;
}

struct TrieShadowEvent {
  std::uint64_t position{0};  ///< the token index the proposal is for
  std::int32_t token{0};      ///< the trie's proposal
  std::uint32_t order{0};     ///< its longest matching context order
  double probability{0};      ///< the trie's blended probability
  TrieShadowReason reason{TrieShadowReason::kNone};
  bool sampled{false};
  std::uint32_t chain{0};  ///< tokens in the chain (anchor included)
  std::uint32_t keep{0};   ///< tokens verification kept
};

/// One JSON line. `emitted` is -1 when the position was not reached or the
/// request ended before its next anchor.
[[nodiscard]] inline std::string FormatTrieShadow(const TrieShadowEvent& e,
                                                  bool reached,
                                                  std::int32_t emitted) {
  char line[320];
  const int n = std::snprintf(
      line, sizeof line,
      "{\"pos\":%llu,\"reason\":\"%s\",\"sampled\":%s,\"chain\":%u,"
      "\"keep\":%u,\"reached\":%s,\"trie\":%d,\"order\":%u,\"p\":%.6g,"
      "\"emitted\":%d,\"match\":%s}\n",
      static_cast<unsigned long long>(e.position),
      e.reason == TrieShadowReason::kGate ? "gate" : "width",
      e.sampled ? "true" : "false", e.chain, e.keep,
      reached ? "true" : "false", e.token, e.order, e.probability, emitted,
      reached && emitted == e.token ? "true" : "false");
  return n > 0 ? std::string(line, static_cast<std::size_t>(n)) : std::string();
}

/// One session's shadow state: at most one event waits for the next anchor.
class TrieShadowTracker {
public:
  /// After verification. A chain verification did not keep whole never
  /// reached the trie's position: its line is returned now. A kept chain's
  /// event waits for the next anchor (Anchor); a replaced wait is flushed.
  [[nodiscard]] std::string Verified(const TrieShadowEvent& event) {
    std::string out = Flush();
    if (event.keep < event.chain)
      return out + FormatTrieShadow(event, false, -1);
    waiting_ = event;
    return out;
  }
  /// The next cycle's anchor at `position`: it is the emitted token when the
  /// position is the waiting event's; anything else flushes it unresolved.
  [[nodiscard]] std::string Anchor(std::uint64_t position,
                                   std::int32_t token) {
    if (!waiting_)
      return {};
    if (waiting_->position != position)
      return Flush();
    auto line = FormatTrieShadow(*waiting_, true, token);
    waiting_.reset();
    return line;
  }
  /// A reset, restore or end of request: the waiting event, unresolved.
  [[nodiscard]] std::string Flush() {
    if (!waiting_)
      return {};
    auto line = FormatTrieShadow(*waiting_, true, -1);
    waiting_.reset();
    return line;
  }
  [[nodiscard]] bool Waiting() const noexcept { return waiting_.has_value(); }

private:
  std::optional<TrieShadowEvent> waiting_;
};

/// The model-wide sink: one file, lines appended under a lock.
class TrieShadowLog {
public:
  explicit TrieShadowLog(std::FILE* file) : file_(file) {}
  ~TrieShadowLog() {
    if (file_ != nullptr)
      std::fclose(file_);
  }
  TrieShadowLog(const TrieShadowLog&) = delete;
  TrieShadowLog& operator=(const TrieShadowLog&) = delete;
  /// Opens GUFO_TRIE_SHADOW's path for appending; null when unset or the
  /// file cannot be opened (shadow mode then stays off).
  [[nodiscard]] static std::unique_ptr<TrieShadowLog> FromEnvironment();
  void Write(const std::string& lines) {
    if (lines.empty())
      return;
    const std::lock_guard lock(mutex_);
    std::fwrite(lines.data(), 1, lines.size(), file_);
    std::fflush(file_);
  }

private:
  std::mutex mutex_;
  std::FILE* file_;
};

inline std::unique_ptr<TrieShadowLog> TrieShadowLog::FromEnvironment() {
  const char* path = std::getenv("GUFO_TRIE_SHADOW");
  if (path == nullptr || *path == '\0')
    return nullptr;
  std::FILE* file = std::fopen(path, "a");
  return file == nullptr ? nullptr : std::make_unique<TrieShadowLog>(file);
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_TRIE_SHADOW_HPP_
