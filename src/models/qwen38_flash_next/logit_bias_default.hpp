#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_LOGIT_BIAS_DEFAULT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_LOGIT_BIAS_DEFAULT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Default suppression of the output rows a workload does not use (medium
/// level of the vocab trim): every row whose text has a non-ASCII byte and
/// that is not a special token, except the rows of a keep list, gets a soft
/// negative bias. Soft on purpose: a request that really asks for another
/// script still wins, because the model's own margin is larger than the bias.
struct SuppressionPlan {
  /// Indexed by token ID: `value` for a suppressed row, 0 for any other.
  std::shared_ptr<const std::vector<float>> bias;
  /// Indexed by token ID: 1 for a suppressed row. The draft head masks these.
  std::vector<std::uint8_t> flags;
  std::size_t non_ascii{0};   ///< non-special rows with a non-ASCII byte
  std::size_t kept{0};        ///< of those, in the keep list
  std::size_t suppressed{0};  ///< of those, suppressed
};

/// `non_ascii(id)` is true when the token's text has a byte of 0x80 or more;
/// `special(id)` is true for the special tokens (never suppressed). Keep IDs
/// outside [0, vocab) are ignored.
template<typename NonAscii, typename Special>
[[nodiscard]] SuppressionPlan PlanSuppression(
    std::uint32_t vocab, float value, std::span<const std::int32_t> keep,
    NonAscii non_ascii, Special special) {
  SuppressionPlan plan;
  std::vector<bool> kept(vocab, false);
  for (const auto id : keep)
    if (id >= 0 && static_cast<std::uint32_t>(id) < vocab)
      kept[static_cast<std::size_t>(id)] = true;
  auto bias = std::make_shared<std::vector<float>>(vocab, 0.0F);
  plan.flags.assign(vocab, 0);
  for (std::uint32_t id = 0; id < vocab; ++id) {
    if (special(id) || !non_ascii(id))
      continue;
    ++plan.non_ascii;
    if (kept[id]) {
      ++plan.kept;
      continue;
    }
    (*bias)[id] = value;
    plan.flags[id] = 1;
    ++plan.suppressed;
  }
  plan.bias = std::move(bias);
  return plan;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
