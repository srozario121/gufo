#include "src/models/qwen38_flash_next/decode_copy.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// ggml's reference Q4_1 quantizer: levels over the block's own [min, max].
void ReferenceQ4_1(const float* x, float* d, float* m, std::uint8_t* levels) {
  float lo = x[0];
  float hi = x[0];
  for (int j = 1; j < 32; ++j) {
    lo = std::fmin(lo, x[j]);
    hi = std::fmax(hi, x[j]);
  }
  const float step = (hi - lo) / 15.0F;
  const float inv = step != 0.0F ? 1.0F / step : 0.0F;
  *d = step;
  *m = lo;
  for (int j = 0; j < 32; ++j) {
    levels[j] = static_cast<std::uint8_t>(
        std::fmin(15.0F, std::floor((x[j] - lo) * inv + 0.5F)));
  }
}

double Error(const float* x, float d, float m, const std::uint8_t* levels) {
  double e = 0;
  for (int j = 0; j < 32; ++j) {
    const double r = x[j] - (d * levels[j] + m);
    e += r * r;
  }
  return e;
}

void Unpack(const std::uint8_t* qs, std::uint8_t* levels) {
  for (int j = 0; j < 16; ++j) {
    levels[j] = qs[j] & 0x0F;
    levels[j + 16] = qs[j] >> 4;
  }
}

void CheckRoundToHalf() {
  Require(qfn::RoundToHalf(1.0F) == 1.0F, "1.0");
  Require(qfn::RoundToHalf(65504.0F) == 65504.0F, "max half");
  Require(std::isinf(qfn::RoundToHalf(70000.0F)), "overflow to inf");
  Require(qfn::RoundToHalf(0.1F) == 0.0999755859375F, "0.1");
  Require(qfn::RoundToHalf(-0.1F) == -0.0999755859375F, "sign");
  Require(qfn::RoundToHalf(std::ldexp(1.0F, -24)) == std::ldexp(1.0F, -24),
          "smallest subnormal");
  Require(qfn::RoundToHalf(std::ldexp(1.0F, -25)) == 0.0F, "tie to even zero");
  Require(qfn::RoundToHalf(2049.0F) == 2048.0F, "tie to even 2048");
  Require(qfn::RoundToHalf(2051.0F) == 2052.0F, "tie to even 2052");
}

void CheckQuantizerAgainstReference() {
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_real_distribution<float> scale_dist(1e-3F, 0.2F);
  double fitted_total = 0;
  double reference_total = 0;
  for (int block = 0; block < 2000; ++block) {
    // A Q8_0 block: scale * int8 codes, with an occasional outlier.
    const float scale = scale_dist(rng);
    float x[32];
    for (int j = 0; j < 32; ++j) {
      float v = normal(rng) * 30.0F;
      if (block % 5 == 0 && j == 3)
        v *= 3.0F;
      const float q = std::fmax(-127.0F, std::fmin(127.0F, std::round(v)));
      x[j] = qfn::RoundToHalf(scale) * q;
    }
    float d = 0, m = 0;
    std::uint8_t qs[16];
    qfn::QuantizeQ4_1Block(x, &d, &m, qs);
    std::uint8_t levels[32];
    Unpack(qs, levels);
    Require(qfn::RoundToHalf(d) == d && qfn::RoundToHalf(m) == m,
            "d and m are not halves");
    for (int j = 0; j < 32; ++j)
      Require(levels[j] <= 15, "level out of range");
    float rd = 0, rm = 0;
    std::uint8_t rlevels[32];
    ReferenceQ4_1(x, &rd, &rm, rlevels);
    const double e = Error(x, d, m, levels);
    const double re = Error(x, qfn::RoundToHalf(rd), qfn::RoundToHalf(rm),
                            rlevels);
    // The search starts from the plain mapping and keeps the best fit, so
    // it may not do materially worse on any block.
    Require(e <= re * 1.05 + 1e-9, "fitted block worse than the reference");
    fitted_total += e;
    reference_total += re;
  }
  Require(fitted_total < reference_total, "fit did not beat the reference");
}

void CheckExactAndConstantBlocks() {
  // A block already on a 16-level grid reproduces (within half rounding).
  float x[32];
  for (int j = 0; j < 32; ++j)
    x[j] = 0.5F * static_cast<float>(j % 16) - 2.0F;
  float d = 0, m = 0;
  std::uint8_t qs[16], levels[32];
  qfn::QuantizeQ4_1Block(x, &d, &m, qs);
  Unpack(qs, levels);
  Require(Error(x, d, m, levels) < 1e-6, "grid block not reproduced");
  // A constant block has no step.
  for (float& v : x)
    v = 0.25F;
  qfn::QuantizeQ4_1Block(x, &d, &m, qs);
  Unpack(qs, levels);
  Require(d == 0.0F && m == 0.25F, "constant block parameters");
  Require(Error(x, d, m, levels) == 0.0, "constant block not exact");
}

void CheckParse() {
  qfn::DecodeCopyMode mode = qfn::DecodeCopyMode::kBalanced;
  Require(qfn::ParseDecodeCopyMode(nullptr, &mode) &&
              mode == qfn::DecodeCopyMode::kOff,
          "unset is off");
  Require(qfn::ParseDecodeCopyMode("", &mode) &&
              mode == qfn::DecodeCopyMode::kOff,
          "empty is off");
  Require(qfn::ParseDecodeCopyMode("off", &mode) &&
              mode == qfn::DecodeCopyMode::kOff,
          "off");
  Require(qfn::ParseDecodeCopyMode("balanced", &mode) &&
              mode == qfn::DecodeCopyMode::kBalanced,
          "balanced");
  Require(!qfn::ParseDecodeCopyMode("q4", &mode) &&
              mode == qfn::DecodeCopyMode::kOff,
          "unknown text is rejected and stays off");
}

void CheckClassSelection() {
  using C = qfn::DecodeCopyClass;
  const auto make = [](C cls, bool q8, std::uint32_t rows, std::uint32_t cols) {
    return qfn::DecodeCopyCandidate{cls, q8, true, rows, cols};
  };
  const auto on = qfn::DecodeCopyMode::kBalanced;
  Require(qfn::ShouldCopyForDecode(on, make(C::kSsmIn, true, 16384, 2560)),
          "ssm_in");
  Require(qfn::ShouldCopyForDecode(on, make(C::kSsmOut, true, 2560, 6144)),
          "ssm_out at the size floor");
  Require(qfn::ShouldCopyForDecode(on, make(C::kAttnQkv, true, 13312, 2560)),
          "attn_qkv");
  Require(qfn::ShouldCopyForDecode(on, make(C::kShexp, true, 640, 2560)),
          "shared expert, small");
  Require(!qfn::ShouldCopyForDecode(on, make(C::kSsmOut, true, 2560, 4096)),
          "below the size floor");
  Require(!qfn::ShouldCopyForDecode(on, make(C::kSsmIn, false, 16384, 2560)),
          "not Q8_0");
  Require(!qfn::ShouldCopyForDecode(on, make(C::kSsmIn, true, 16384, 2570)),
          "cols not a multiple of 32");
  qfn::DecodeCopyCandidate empty = make(C::kSsmIn, true, 16384, 2560);
  empty.has_data = false;
  Require(!qfn::ShouldCopyForDecode(on, empty), "no data");
}

void CheckFlagOffIsNoOp() {
  using C = qfn::DecodeCopyClass;
  for (const C cls : {C::kSsmIn, C::kSsmOut, C::kAttnQkv, C::kShexp}) {
    Require(!qfn::DecodeCopyClassSelected(qfn::DecodeCopyMode::kOff, cls),
            "off selected a class");
    Require(!qfn::ShouldCopyForDecode(
                qfn::DecodeCopyMode::kOff,
                qfn::DecodeCopyCandidate{cls, true, true, 16384, 2560}),
            "off copied a tensor");
  }
}

void CheckBytes() {
  Require(qfn::DecodeCopyBytes(1, 32) == 20, "one block");
  Require(qfn::DecodeCopyBytes(16384, 2560) == 16384ULL * 80 * 20, "ssm_in");
  // Q8_0 is 34 bytes per block: the copy is 20/34 of it.
  Require(qfn::DecodeCopyBytes(2560, 6144) * 34 ==
              std::size_t{2560} * 6144 / 32 * 20 * 34,
          "ratio");
}

}  // namespace

int main() {
  try {
    CheckRoundToHalf();
    CheckQuantizerAgainstReference();
    CheckExactAndConstantBlocks();
    CheckParse();
    CheckClassSelection();
    CheckFlagOffIsNoOp();
    CheckBytes();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
