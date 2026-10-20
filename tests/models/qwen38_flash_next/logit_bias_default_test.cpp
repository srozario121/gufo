#include "src/models/qwen38_flash_next/logit_bias_default.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A SYNTHETIC vocabulary, test data only: ids 0-99 are ASCII, 100-149 have a
// high byte, 150-159 have a high byte but are special tokens, 160-199 are
// ASCII again.
bool NonAscii(std::uint32_t id) {
  return id >= 100 && id < 160;
}
bool Special(std::uint32_t id) {
  return id >= 150 && id < 160;
}

}  // namespace

// Every non-ASCII, non-special row is suppressed at the value, except the keep
// list; ASCII rows and specials are untouched.
void CheckDefaultSet() {
  const std::vector<std::int32_t> keep{101, 120, 149};
  const auto plan = qfn::PlanSuppression(200, -8.0F, keep, NonAscii, Special);
  Require(plan.non_ascii == 50 && plan.kept == 3 && plan.suppressed == 47,
          "counts");
  Require(plan.bias->size() == 200 && plan.flags.size() == 200, "sizes");
  for (std::uint32_t id = 0; id < 200; ++id) {
    const bool expect =
        NonAscii(id) && !Special(id) && id != 101 && id != 120 && id != 149;
    Require((*plan.bias)[id] == (expect ? -8.0F : 0.0F), "bias value");
    Require(plan.flags[id] == (expect ? 1 : 0), "flag");
  }
}

// An empty keep list suppresses every non-ASCII row; out-of-range keep IDs and
// repeats are ignored.
void CheckKeepEdges() {
  const auto all = qfn::PlanSuppression(200, -4.0F, {}, NonAscii, Special);
  Require(all.suppressed == 50 && all.kept == 0, "no keep list");
  const std::vector<std::int32_t> keep{-1, 200, 5000, 101, 101};
  const auto some = qfn::PlanSuppression(200, -4.0F, keep, NonAscii, Special);
  Require(some.kept == 1 && some.suppressed == 49, "keep edges");
  Require((*some.bias)[101] == 0.0F && (*some.bias)[102] == -4.0F,
          "a kept row is untouched");
}

int main() {
  try {
    CheckDefaultSet();
    CheckKeepEdges();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
