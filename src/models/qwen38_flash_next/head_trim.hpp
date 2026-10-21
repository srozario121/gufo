#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_HEAD_TRIM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_HEAD_TRIM_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Trimmed output head (high level of the vocab trim). The target head scores
/// only the rows of a token-ID set, gathered once at load into a compact
/// device copy; every other token is "not a candidate" (logit -infinity) for
/// the argmax, the sampler, penalties and logit_bias. When the best
/// probability over the scored rows is below a threshold, the full head is
/// re-run for that position, so the token is chosen over the whole
/// vocabulary.
///
/// The default threshold is 0.5. A token outside the set can only win over a
/// scored row Y whose renormalised probability q is at least 0.5 if the set
/// holds under 2/3 of the true mass (an outside token X needs p_X > q * M, and
/// p_X <= 1 - M); that is a position the offline held-out miss rate (about
/// 0.1% of tokens) says is rare, and a fallback there would add cost on every
/// confident token. Below 0.5 the scored rows are spread, which is where a
/// miss lives. A higher threshold (0.9) flags many more positions, and one
/// flagged row among a verification chain costs a whole extra head read, so
/// it gives the saving back (the share is a T12 measurement, not claimed
/// here); 0 turns the fallback off. A hit outside the set at high confidence is
/// not detectable by this rule; the divergence report measures it.
inline constexpr float kHeadTrimDefaultThreshold = 0.5F;

/// Rows of the compact copy are padded to a multiple of this, so every
/// projection kernel sees a tile-aligned head (the full head's 248,320 rows
/// are a multiple of 256). Padding rows are zero and never read back.
inline constexpr std::uint32_t kHeadTrimRowAlign = 256;

/// Most logit rows one head projection can flag (one decode batch of 8
/// sessions by 8 rows, or one verification chain).
inline constexpr std::uint32_t kHeadTrimMaxRows = 64;

struct HeadTrimPlan {
  /// Compact row -> token ID, ascending.
  std::vector<std::int32_t> ids;
  /// Token ID -> compact row, -1 for a row that is not scored.
  std::vector<std::int32_t> inverse;
  /// Compact rows including padding.
  std::uint32_t padded_rows{0};
  std::size_t from_set{0};       ///< distinct in-range IDs of the set file
  std::size_t added_special{0};  ///< special tokens the set did not list
};

/// `set` is the parsed ID file, `special` the vocabulary's special tokens
/// (chat, thinking, tool-call markers and the stop tokens), which are always
/// scored: a head without the stop token could never end a turn. IDs outside
/// [0, vocab) are ignored.
[[nodiscard]] inline HeadTrimPlan PlanHeadTrim(
    std::uint32_t vocab, std::span<const std::int32_t> set,
    std::span<const std::int32_t> special) {
  HeadTrimPlan plan;
  plan.inverse.assign(vocab, -1);
  std::vector<bool> in(vocab, false);
  for (const auto id : set) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= vocab ||
        in[static_cast<std::size_t>(id)])
      continue;
    in[static_cast<std::size_t>(id)] = true;
    ++plan.from_set;
  }
  for (const auto id : special) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= vocab ||
        in[static_cast<std::size_t>(id)])
      continue;
    in[static_cast<std::size_t>(id)] = true;
    ++plan.added_special;
  }
  for (std::uint32_t id = 0; id < vocab; ++id) {
    if (!in[id])
      continue;
    plan.inverse[id] = static_cast<std::int32_t>(plan.ids.size());
    plan.ids.push_back(static_cast<std::int32_t>(id));
  }
  const auto rows = static_cast<std::uint32_t>(plan.ids.size());
  plan.padded_rows =
      (rows + kHeadTrimRowAlign - 1) / kHeadTrimRowAlign * kHeadTrimRowAlign;
  return plan;
}

/// The best probability over the scored rows, 1 / sum(exp(x - max)): the
/// device kernel's quantity, at temperature 1. NaN when a logit is NaN or no
/// row is finite.
[[nodiscard]] inline float HeadTrimTopProbability(
    std::span<const float> scored) {
  float maximum = -std::numeric_limits<float>::infinity();
  bool bad = false;
  for (const float v : scored) {
    if (std::isnan(v))
      bad = true;
    else
      maximum = std::max(maximum, v);
  }
  if (bad || !std::isfinite(maximum))
    return std::numeric_limits<float>::quiet_NaN();
  double sum = 0.0;
  for (const float v : scored)
    sum += std::exp(static_cast<double>(v) - static_cast<double>(maximum));
  return static_cast<float>(1.0 / sum);
}

/// A position falls back to the full head when its top probability is below
/// the threshold or is not a number. A threshold of 0 or less never falls
/// back.
[[nodiscard]] inline bool HeadTrimNeedsFallback(float top_probability,
                                                float threshold) {
  return threshold > 0.0F && !(top_probability >= threshold);
}

/// Compact logits to full-vocabulary logits: a scored token keeps its value,
/// every other token is -infinity. The device kernel does the same.
inline void HeadTrimExpand(std::span<const float> scored,
                           std::span<const std::int32_t> inverse,
                           std::span<float> full) {
  for (std::size_t id = 0; id < full.size(); ++id) {
    const auto row = id < inverse.size() ? inverse[id] : -1;
    full[id] = row >= 0 ? scored[static_cast<std::size_t>(row)]
                        : -std::numeric_limits<float>::infinity();
  }
}

/// Head projections the trimmed target made and how many fell back; read from
/// the device counters.
struct HeadTrimStats {
  std::uint64_t rows{0};       ///< target logit rows scored on the compact head
  std::uint64_t fallbacks{0};  ///< of those, rows re-run on the full head
  std::uint64_t full_reads{0};  ///< full-head passes (one per projection with a
                                ///< fallback row, however many rows it holds)
  [[nodiscard]] double Rate() const noexcept {
    return rows == 0
               ? 0.0
               : static_cast<double>(fallbacks) / static_cast<double>(rows);
  }
};

/// Device-side timing of the trimmed head (--vocab-trim-timing), in
/// nanoseconds of the device's constant wall clock, summed over every
/// HeadProject call since the model loaded. Phases: a the compact projection,
/// b ExpandHeadLogits, c HeadConfidence, d the full-head fallback (quantize
/// and read), split by how many rows the call flagged; total spans the start
/// of (a) to the end of the last phase of the call. Each phase includes the
/// launch gap before the timestamp kernel that closes it; `gap_ns` sums the
/// same measure over two timestamp kernels with nothing between them (one per
/// call), the floor of what an empty phase reads.
inline constexpr std::size_t kHeadTimingBuckets = 4;  ///< 0, 1, 2, 3+ rows
struct HeadTimingStats {
  std::uint64_t calls{0};           ///< HeadProject calls timed
  std::uint64_t fallback_calls{0};  ///< of those, with the fallback stage
  std::uint64_t a_ns{0};
  std::uint64_t b_ns{0};
  std::uint64_t c_ns{0};
  std::uint64_t total_ns{0};
  std::uint64_t gap_ns{0};
  std::uint64_t d_calls[kHeadTimingBuckets]{};
  std::uint64_t d_ns[kHeadTimingBuckets]{};

  [[nodiscard]] static constexpr std::size_t Bucket(
      std::uint32_t flagged) noexcept {
    return flagged < kHeadTimingBuckets ? flagged : kHeadTimingBuckets - 1;
  }
  /// Device counters, in this order: calls,
  /// fallback_calls, a, b, c, total, gap, d_calls[], d_ticks[]. `ns_per_tick`
  /// converts the clock's ticks.
  static constexpr std::size_t kCounterCount = 7 + 2 * kHeadTimingBuckets;
  [[nodiscard]] static HeadTimingStats FromCounters(
      const unsigned long long* c, double ns_per_tick) noexcept {
    const auto ns = [&](std::uint64_t ticks) {
      return static_cast<std::uint64_t>(
          static_cast<double>(ticks) * ns_per_tick + 0.5);
    };
    HeadTimingStats s;
    s.calls = c[0];
    s.fallback_calls = c[1];
    s.a_ns = ns(c[2]);
    s.b_ns = ns(c[3]);
    s.c_ns = ns(c[4]);
    s.total_ns = ns(c[5]);
    s.gap_ns = ns(c[6]);
    for (std::size_t i = 0; i < kHeadTimingBuckets; ++i) {
      s.d_calls[i] = c[7 + i];
      s.d_ns[i] = ns(c[7 + kHeadTimingBuckets + i]);
    }
    return s;
  }
  HeadTimingStats& operator+=(const HeadTimingStats& o) noexcept {
    calls += o.calls;
    fallback_calls += o.fallback_calls;
    a_ns += o.a_ns;
    b_ns += o.b_ns;
    c_ns += o.c_ns;
    total_ns += o.total_ns;
    gap_ns += o.gap_ns;
    for (std::size_t i = 0; i < kHeadTimingBuckets; ++i) {
      d_calls[i] += o.d_calls[i];
      d_ns[i] += o.d_ns[i];
    }
    return *this;
  }
  /// Calls f(name, value) for each field, as reported: counts as they are,
  /// durations in microseconds. The caller adds its own prefix.
  template<typename F>
  void ForEachField(F&& f) const {
    const auto us = [](std::uint64_t ns) {
      return static_cast<double>(ns) / 1000.0;
    };
    static constexpr const char* kUs[kHeadTimingBuckets] = {
        "d_none_us", "d_one_us", "d_two_us", "d_many_us"};
    static constexpr const char* kCalls[kHeadTimingBuckets] = {
        "d_none_calls", "d_one_calls", "d_two_calls", "d_many_calls"};
    f("calls", static_cast<double>(calls));
    f("fallback_calls", static_cast<double>(fallback_calls));
    f("a_us", us(a_ns));
    f("b_us", us(b_ns));
    f("c_us", us(c_ns));
    for (std::size_t i = 0; i < kHeadTimingBuckets; ++i) {
      f(kUs[i], us(d_ns[i]));
      f(kCalls[i], static_cast<double>(d_calls[i]));
    }
    f("total_us", us(total_ns));
    f("gap_us", us(gap_ns));
  }
  /// This minus an earlier reading of the same counters.
  [[nodiscard]] HeadTimingStats Since(const HeadTimingStats& o) const noexcept {
    HeadTimingStats s = *this;
    s.calls -= o.calls;
    s.fallback_calls -= o.fallback_calls;
    s.a_ns -= o.a_ns;
    s.b_ns -= o.b_ns;
    s.c_ns -= o.c_ns;
    s.total_ns -= o.total_ns;
    s.gap_ns -= o.gap_ns;
    for (std::size_t i = 0; i < kHeadTimingBuckets; ++i) {
      s.d_calls[i] -= o.d_calls[i];
      s.d_ns[i] -= o.d_ns[i];
    }
    return s;
  }
};

/// Measurement counters of the fallback (--vocab-trim-counters), summed over
/// every flagged row since the model loaded. They ask what the full-head
/// reads buy: `argmax_changed` counts flagged rows whose greedy argmax over
/// the full head differs from the argmax over the scored rows; `mass_sum`
/// is the softmax mass (temperature 1, over the whole vocabulary after the
/// fallback) of the tokens that were not scored, summed over flagged rows,
/// with the number of rows above 0.05 and above 0.2. The `chain_*` fields come
/// from the verification loop (one record per speculative round): flagged
/// rows at or past the number of kept tokens were discarded unused.
struct HeadCounterStats {
  std::uint64_t flagged{0};         ///< flagged rows examined
  std::uint64_t argmax_changed{0};  ///< of those, full argmax != scored argmax
  double mass_sum{0.0};             ///< unscored softmax mass, summed
  std::uint64_t mass_over_005{0};
  std::uint64_t mass_over_02{0};
  std::uint64_t chains{0};             ///< verification rounds recorded
  std::uint64_t chain_rows{0};         ///< verification rows in those rounds
  std::uint64_t chain_flagged{0};      ///< flagged rows among them
  std::uint64_t discarded_flagged{0};  ///< flagged rows past the kept prefix
  std::uint64_t used_changed{0};       ///< argmax-changing flagged rows kept
  std::uint64_t discarded_changed{0};  ///< argmax-changing rows discarded

  /// The mass is accumulated on the device as fixed point, 2^-32 units.
  static constexpr double kMassScale = 4294967296.0;
  /// Device counters, in this order: flagged, argmax_changed, mass (fixed
  /// point), mass_over_005, mass_over_02, chains, chain_rows, chain_flagged,
  /// discarded_flagged, used_changed, discarded_changed.
  static constexpr std::size_t kCounterCount = 11;
  [[nodiscard]] static HeadCounterStats FromCounters(
      const unsigned long long* c) noexcept {
    HeadCounterStats s;
    s.flagged = c[0];
    s.argmax_changed = c[1];
    s.mass_sum = static_cast<double>(c[2]) / kMassScale;
    s.mass_over_005 = c[3];
    s.mass_over_02 = c[4];
    s.chains = c[5];
    s.chain_rows = c[6];
    s.chain_flagged = c[7];
    s.discarded_flagged = c[8];
    s.used_changed = c[9];
    s.discarded_changed = c[10];
    return s;
  }
  HeadCounterStats& operator+=(const HeadCounterStats& o) noexcept {
    flagged += o.flagged;
    argmax_changed += o.argmax_changed;
    mass_sum += o.mass_sum;
    mass_over_005 += o.mass_over_005;
    mass_over_02 += o.mass_over_02;
    chains += o.chains;
    chain_rows += o.chain_rows;
    chain_flagged += o.chain_flagged;
    discarded_flagged += o.discarded_flagged;
    used_changed += o.used_changed;
    discarded_changed += o.discarded_changed;
    return *this;
  }
  /// This minus an earlier reading of the same counters.
  [[nodiscard]] HeadCounterStats Since(
      const HeadCounterStats& o) const noexcept {
    HeadCounterStats s = *this;
    s.flagged -= o.flagged;
    s.argmax_changed -= o.argmax_changed;
    s.mass_sum -= o.mass_sum;
    s.mass_over_005 -= o.mass_over_005;
    s.mass_over_02 -= o.mass_over_02;
    s.chains -= o.chains;
    s.chain_rows -= o.chain_rows;
    s.chain_flagged -= o.chain_flagged;
    s.discarded_flagged -= o.discarded_flagged;
    s.used_changed -= o.used_changed;
    s.discarded_changed -= o.discarded_changed;
    return s;
  }
  /// Calls f(name, value) for each field; the caller adds its prefix.
  template<typename F>
  void ForEachField(F&& f) const {
    f("flagged", static_cast<double>(flagged));
    f("argmax_changed", static_cast<double>(argmax_changed));
    f("mass_sum", mass_sum);
    f("mass_over_0p05", static_cast<double>(mass_over_005));
    f("mass_over_0p2", static_cast<double>(mass_over_02));
    f("chains", static_cast<double>(chains));
    f("chain_rows", static_cast<double>(chain_rows));
    f("chain_flagged", static_cast<double>(chain_flagged));
    f("discarded_flagged", static_cast<double>(discarded_flagged));
    f("used_changed", static_cast<double>(used_changed));
    f("discarded_changed", static_cast<double>(discarded_changed));
  }
};

}  // namespace gufo::models::qwen38_flash_next

#endif
