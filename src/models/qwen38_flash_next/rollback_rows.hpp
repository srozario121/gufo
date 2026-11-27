#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROLLBACK_ROWS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROLLBACK_ROWS_HPP_

#include <cstdint>
#include <iterator>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Proper prefixes a speculative pass may snapshot: a chain of at most
/// kMaxMtpDraftTokens drafts verifies kMaxMtpDraftTokens + 1 rows, so it
/// keeps one rollback row per draft (max_speculative - 1).
inline constexpr std::uint32_t kMaxRollbackRows = kMaxMtpDraftTokens;

/// Stable per-prefix rollback addresses. Growing the depth does not move
/// existing buffers or invalidate graphs captured for a smaller batch.
/// Kernels take it by value and read only rows[t] for t below the pass's
/// rollback depth; the rows past it stay null.
struct RollbackRows {
  float* rows[kMaxRollbackRows]{};
};
static_assert(std::size(RollbackRows{}.rows) >= kMaxMtpDraftTokens,
              "a speculative pass snapshots one row per draft");

/// Whether a session may grow its rollback rows to `depth` (EnsureRollback):
/// fewer rows than the pass width, within RollbackRows, and no speculative
/// batch pending.
[[nodiscard]] constexpr bool RollbackDepthAllowed(
    std::uint32_t depth, std::uint32_t max_speculative,
    std::uint32_t pending_spec_tokens) noexcept {
  return depth < max_speculative && depth <= std::size(RollbackRows{}.rows) &&
         pending_spec_tokens == 0;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROLLBACK_ROWS_HPP_
