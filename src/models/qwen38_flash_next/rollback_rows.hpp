#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROLLBACK_ROWS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROLLBACK_ROWS_HPP_

#include <cstdint>
#include <iterator>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Proper prefixes one verification segment may snapshot: a pass of at most
/// eight rows (the decode arithmetic) keeps one rollback row per row after
/// its first. A chain longer than eight rows (the depth probe's MTP chains,
/// or a wide copy) verifies through ForwardWide, whose later segments roll
/// back through executor-owned rows, so this stays at seven whatever
/// kMaxMtpDraftTokens is: ForwardWide's segment length is this plus one.
inline constexpr std::uint32_t kMaxRollbackRows = 7;
static_assert(kMaxRollbackRows == kMaxBatchedMtpDraftTokens,
              "a batched or narrow chain snapshots one row per draft");

/// Stable per-prefix rollback addresses. Growing the depth does not move
/// existing buffers or invalidate graphs captured for a smaller batch.
/// Kernels take it by value and read only rows[t] for t below the pass's
/// rollback depth; the rows past it stay null.
struct RollbackRows {
  float* rows[kMaxRollbackRows]{};
};
static_assert(std::size(RollbackRows{}.rows) + 1 == 8,
              "ForwardWide's segments are eight rows (Executor::Rollback)");

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
