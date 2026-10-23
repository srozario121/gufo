#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DECODE_COPY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DECODE_COPY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#define GUFO_DECODE_COPY_HD __host__ __device__
#else
#define GUFO_DECODE_COPY_HD
#endif

namespace gufo::models::qwen38_flash_next {

/// Q4_1 decode copy of the Q8_0 dense trunk.
///
/// Decode reads every large dense matrix once per round, so its bytes set the
/// round's cost. A Q8_0 block is 34 bytes for 32 weights; a Q4_1 block is 20.
/// With GUFO_DECODE_COPY=balanced the model re-encodes the selected matrices
/// at load into a second, decode-only Q4_1 copy that the narrow (at most
/// eight row) projection kernels read instead of the Q8_0 rows. Prefill, wide
/// batches and the MTP draft block keep reading the Q8_0 weights.
///
/// This is an APPROXIMATION: the copy re-quantizes 8-bit weights, so decode
/// output changes. It is off unless asked for.
///
/// Design after the "balanced" decode copy of github.com/mikealanni/gufo
/// (branch fork-enhancements, MIT, Copyright (c) 2026 gufo contributors),
/// reimplemented here from the described change.
///
/// Everything here is host-compilable (and also `__host__ __device__` under
/// HIP) so the CPU tests exercise the same arithmetic the load-time kernel
/// runs.

enum class DecodeCopyMode { kOff, kBalanced };

/// The projection classes a copy may cover. Matrices outside this list
/// (attn_out, the hyper-connection projections, the MTP block) are never
/// copied: they cost the most quality per byte saved.
enum class DecodeCopyClass { kSsmIn, kSsmOut, kAttnQkv, kShexp };

/// Parses GUFO_DECODE_COPY. Unset, empty and "off" give kOff; "balanced"
/// gives kBalanced. Returns false, leaving *mode at kOff, for any other text
/// so the caller can report it and stay off.
[[nodiscard]] inline bool ParseDecodeCopyMode(const char* value,
                                              DecodeCopyMode* mode) {
  *mode = DecodeCopyMode::kOff;
  if (value == nullptr) {
    return true;
  }
  const std::string_view text(value);
  if (text.empty() || text == "off") {
    return true;
  }
  if (text == "balanced") {
    *mode = DecodeCopyMode::kBalanced;
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool DecodeCopyClassSelected(
    DecodeCopyMode mode, DecodeCopyClass cls) noexcept {
  switch (mode) {
    case DecodeCopyMode::kOff:
      return false;
    case DecodeCopyMode::kBalanced:
      return cls == DecodeCopyClass::kSsmIn ||
             cls == DecodeCopyClass::kSsmOut ||
             cls == DecodeCopyClass::kAttnQkv || cls == DecodeCopyClass::kShexp;
  }
  return false;
}

/// Smallest matrix of the large classes that is worth a copy (2560 x 6144
/// elements). The shared expert's matrices are small and always qualify.
inline constexpr std::size_t kDecodeCopyMinElements = std::size_t{2560} * 6144;

inline constexpr std::uint32_t kQ4_1BlockElements = 32;
inline constexpr std::size_t kQ4_1BlockBytes = 20;

/// What the load-time selection needs to know about one device tensor.
struct DecodeCopyCandidate {
  DecodeCopyClass cls;
  bool is_q8_0;
  bool has_data;
  std::uint32_t rows;
  std::uint32_t cols;
};

[[nodiscard]] constexpr bool ShouldCopyForDecode(
    DecodeCopyMode mode, const DecodeCopyCandidate& t) noexcept {
  if (!DecodeCopyClassSelected(mode, t.cls)) {
    return false;
  }
  if (!t.is_q8_0 || !t.has_data || t.cols == 0 ||
      t.cols % kQ4_1BlockElements != 0) {
    return false;
  }
  const std::size_t min_elements =
      t.cls == DecodeCopyClass::kShexp ? 1 : kDecodeCopyMinElements;
  return std::size_t{t.rows} * t.cols >= min_elements;
}

/// Bytes of the Q4_1 copy of a rows x cols Q8_0 matrix.
[[nodiscard]] constexpr std::size_t DecodeCopyBytes(
    std::uint32_t rows, std::uint32_t cols) noexcept {
  return std::size_t{rows} * cols / kQ4_1BlockElements * kQ4_1BlockBytes;
}

/// Rounds to the nearest IEEE half (ties to even), returned as a float. On the
/// device this is the hardware conversion; on the host an exact software one.
GUFO_DECODE_COPY_HD inline float RoundToHalf(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __half2float(__float2half(v));
#else
  if (!std::isfinite(v) || v == 0.0F) {
    return v;
  }
  const float magnitude = std::fabs(v);
  int exponent = 0;
  (void)std::frexp(magnitude, &exponent);  // magnitude = f * 2^exponent
  // A half keeps 11 significant bits; below 2^-14 the spacing stays 2^-24.
  const int quantum = std::max(exponent - 1 - 10, -24);
  const float scaled = std::ldexp(magnitude, -quantum);
  const float rounded = std::ldexp(std::nearbyint(scaled), quantum);
  if (rounded > 65504.0F) {
    return std::copysign(INFINITY, v);
  }
  return std::copysign(rounded, v);
#endif
}

/// Number of candidate step sizes the quantizer tries per block.
inline constexpr int kQ4_1Candidates = 24;

/// Re-encodes one 32-weight block (already dequantized from Q8_0) as Q4_1:
/// x[j] ~= d * level[j] + m with levels 0..15. The step is searched, because
/// the plain [min, max] mapping wastes range on outliers: for each candidate
/// step the levels are rounded, then the step and minimum are refitted to the
/// rounded levels by least squares (both stored as halves), and the fit with
/// the lowest squared error wins. The levels are then taken again as the
/// nearest to the winning step and minimum.
///
/// d and m are returned already rounded to half precision. qs holds element j
/// in the low nibble and element j + 16 in the high nibble of byte j, as ggml
/// stores Q4_1.
GUFO_DECODE_COPY_HD inline void QuantizeQ4_1Block(const float* x, float* d_out,
                                                  float* m_out,
                                                  std::uint8_t* qs) {
  float lo = x[0];
  float hi = x[0];
  for (int j = 1; j < 32; ++j) {
    lo = x[j] < lo ? x[j] : lo;
    hi = x[j] > hi ? x[j] : hi;
  }
  float best_d = RoundToHalf((hi - lo) / 15.0F);
  float best_m = RoundToHalf(lo);
  if (hi > lo) {
    float best_err = INFINITY;
    for (int c = 0; c < kQ4_1Candidates; ++c) {
      const float iscale =
          (15.0F + 0.1F * static_cast<float>(c - 8)) / (hi - lo);
      float sl = 0.0F;
      float sl2 = 0.0F;
      float sx = 0.0F;
      float slx = 0.0F;
      float level[32];
      for (int j = 0; j < 32; ++j) {
        float l = floorf((x[j] - lo) * iscale + 0.5F);
        l = l < 0.0F ? 0.0F : (l > 15.0F ? 15.0F : l);
        level[j] = l;
        sl += l;
        sl2 += l * l;
        sx += x[j];
        slx += l * x[j];
      }
      const float det = 32.0F * sl2 - sl * sl;
      if (det <= 0.0F) {
        continue;
      }
      const float d = RoundToHalf((32.0F * slx - sl * sx) / det);
      const float m = RoundToHalf((sx - d * sl) / 32.0F);
      float err = 0.0F;
      for (int j = 0; j < 32; ++j) {
        const float r = x[j] - (d * level[j] + m);
        err += r * r;
      }
      if (err < best_err) {
        best_err = err;
        best_d = d;
        best_m = m;
      }
    }
  }
  const float inv = best_d != 0.0F ? 1.0F / best_d : 0.0F;
  for (int j = 0; j < 16; ++j) {
    float q0 = floorf((x[j] - best_m) * inv + 0.5F);
    float q1 = floorf((x[j + 16] - best_m) * inv + 0.5F);
    q0 = q0 < 0.0F ? 0.0F : (q0 > 15.0F ? 15.0F : q0);
    q1 = q1 < 0.0F ? 0.0F : (q1 > 15.0F ? 15.0F : q1);
    qs[j] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(q0) |
                                      (static_cast<std::uint32_t>(q1) << 4U));
  }
  *d_out = best_d;
  *m_out = best_m;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DECODE_COPY_HPP_
