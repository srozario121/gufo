#include "src/models/qwen38_flash_next/head_trim.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr float kInf = std::numeric_limits<float>::infinity();

}  // namespace

// The set is sorted, deduplicated and clipped to the vocabulary; the special
// tokens join it; the compact copy is padded to a whole tile.
void CheckPlan() {
  // A SYNTHETIC vocabulary of 1000 tokens, test data only.
  const std::vector<std::int32_t> set{500, 3, 3, -1, 1000, 77, 5000};
  const std::vector<std::int32_t> special{999, 77, 998};
  const auto plan = qfn::PlanHeadTrim(1000, set, special);
  Require(plan.from_set == 3, "distinct in-range set IDs");
  Require(plan.added_special == 2, "special tokens the set lacks");
  Require((plan.ids == std::vector<std::int32_t>{3, 77, 500, 998, 999}),
          "ascending compact rows");
  Require(plan.inverse.size() == 1000, "one inverse entry per token");
  for (std::size_t row = 0; row < plan.ids.size(); ++row)
    Require(plan.inverse[static_cast<std::size_t>(plan.ids[row])] ==
                static_cast<std::int32_t>(row),
            "inverse maps back to the compact row");
  Require(plan.inverse[4] == -1 && plan.inverse[0] == -1, "unscored rows");
  Require(plan.padded_rows == qfn::kHeadTrimRowAlign, "padded to a tile");

  std::vector<std::int32_t> many;
  for (std::int32_t id = 0; id < 600; ++id)
    many.push_back(id * 3);
  const auto wide = qfn::PlanHeadTrim(2000, many, {});
  Require(wide.ids.size() == 600 && wide.padded_rows == 768,
          "600 rows pad to 768");
  const auto exact = qfn::PlanHeadTrim(
      2000, std::vector<std::int32_t>(many.begin(), many.begin() + 512), {});
  Require(exact.padded_rows == 512, "a whole tile adds no padding");
}

// An empty set plus no specials is an empty plan; the device refuses it.
void CheckEmpty() {
  const auto plan = qfn::PlanHeadTrim(10, {}, {});
  Require(plan.ids.empty() && plan.padded_rows == 0, "empty plan");
}

// The top probability is 1 / sum(exp(x - max)); NaN when it is undefined.
void CheckTopProbability() {
  const std::array<float, 3> flat{1.0F, 1.0F, 1.0F};
  Require(std::fabs(qfn::HeadTrimTopProbability(flat) - 1.0F / 3.0F) < 1e-6F,
          "uniform logits");
  const std::array<float, 2> sharp{20.0F, 0.0F};
  Require(qfn::HeadTrimTopProbability(sharp) > 0.999999F, "one clear winner");
  const std::array<float, 2> half{3.0F, 3.0F};
  Require(std::fabs(qfn::HeadTrimTopProbability(half) - 0.5F) < 1e-6F,
          "two equal rows");
  const std::array<float, 2> nan{1.0F, std::nanf("")};
  Require(std::isnan(qfn::HeadTrimTopProbability(nan)), "a NaN logit");
  const std::array<float, 2> none{-kInf, -kInf};
  Require(std::isnan(qfn::HeadTrimTopProbability(none)), "no finite row");
}

// The rule: fall back below the threshold or when the probability is not a
// number; a threshold of 0 never falls back; the threshold itself is confident.
void CheckFallbackRule() {
  Require(qfn::kHeadTrimDefaultThreshold == 0.5F, "default threshold");
  Require(qfn::HeadTrimNeedsFallback(0.49F, 0.5F), "just below");
  Require(!qfn::HeadTrimNeedsFallback(0.5F, 0.5F), "at the threshold");
  Require(!qfn::HeadTrimNeedsFallback(0.99F, 0.5F), "confident");
  Require(qfn::HeadTrimNeedsFallback(std::nanf(""), 0.5F), "NaN falls back");
  Require(!qfn::HeadTrimNeedsFallback(0.0F, 0.0F) &&
              !qfn::HeadTrimNeedsFallback(std::nanf(""), 0.0F),
          "threshold 0 turns the fallback off");
  Require(qfn::HeadTrimNeedsFallback(0.999F, 1.0F), "threshold 1");
  // Two scored rows at 0.6 / 0.4 renormalised: confident at 0.5, not at 0.9.
  const std::array<float, 2> rows{std::log(0.6F), std::log(0.4F)};
  const float top = qfn::HeadTrimTopProbability(rows);
  Require(!qfn::HeadTrimNeedsFallback(top, 0.5F) &&
              qfn::HeadTrimNeedsFallback(top, 0.9F),
          "the same position under two thresholds");
}

// Expansion keeps scored values at their token IDs and everything else at
// -infinity; the argmax of the expansion is the compact argmax mapped back.
void CheckExpand() {
  const std::vector<std::int32_t> set{4, 9, 2};
  const auto plan = qfn::PlanHeadTrim(12, set, {});
  // compact rows are tokens 2, 4, 9
  const std::array<float, 3> compact{0.5F, 7.0F, 3.0F};
  std::array<float, 12> full{};
  qfn::HeadTrimExpand(compact, plan.inverse, full);
  std::size_t best = 0;
  for (std::size_t id = 0; id < full.size(); ++id) {
    const bool scored = id == 2 || id == 4 || id == 9;
    Require(scored ? std::isfinite(full[id]) : full[id] == -kInf,
            "only scored tokens are finite");
    if (full[id] > full[best])
      best = id;
  }
  Require(best == 4 && full[4] == 7.0F && full[2] == 0.5F && full[9] == 3.0F,
          "values land on their token IDs");
  // A token outside the set that the full head would prefer cannot win here;
  // the fallback exists for it.
  Require(full[11] == -kInf, "an unscored token cannot win");
}

// The counters' rate.
void CheckStats() {
  qfn::HeadTrimStats stats;
  Require(stats.Rate() == 0.0, "no rows, no rate");
  stats.rows = 200;
  stats.fallbacks = 30;
  stats.full_reads = 12;
  Require(std::fabs(stats.Rate() - 0.15) < 1e-12, "rate is fallbacks / rows");
}

// The device timing counters: ticks to nanoseconds, bucketed by flagged rows,
// differences between readings, and the reported fields.
void CheckTiming() {
  Require(qfn::HeadTimingStats::Bucket(0) == 0 &&
              qfn::HeadTimingStats::Bucket(1) == 1 &&
              qfn::HeadTimingStats::Bucket(2) == 2 &&
              qfn::HeadTimingStats::Bucket(3) == 3 &&
              qfn::HeadTimingStats::Bucket(64) == 3,
          "three or more flagged rows share the last bucket");
  unsigned long long c[qfn::HeadTimingStats::kCounterCount]{};
  c[0] = 10;   // calls
  c[1] = 8;    // fallback calls
  c[2] = 100;  // a, ticks
  c[3] = 20;
  c[4] = 30;
  c[5] = 400;  // total
  c[6] = 5;    // gap
  c[7] = 6;    // d calls, 0 rows
  c[8] = 2;    // 1 row
  c[11] = 60;  // d ticks, 0 rows
  c[12] = 300;
  // 10 ns per tick (a 100 MHz clock).
  const auto s = qfn::HeadTimingStats::FromCounters(c, 10.0);
  Require(s.calls == 10 && s.fallback_calls == 8 && s.a_ns == 1000 &&
              s.b_ns == 200 && s.c_ns == 300 && s.total_ns == 4000 &&
              s.gap_ns == 50 && s.d_calls[0] == 6 && s.d_calls[1] == 2 &&
              s.d_ns[0] == 600 && s.d_ns[1] == 3000 && s.d_ns[2] == 0,
          "counters convert to nanoseconds");
  auto sum = s;
  sum += s;
  Require(sum.calls == 20 && sum.d_ns[1] == 6000 && sum.total_ns == 8000,
          "readings add");
  const auto delta = sum.Since(s);
  Require(delta.calls == 10 && delta.a_ns == 1000 && delta.d_calls[0] == 6 &&
              delta.d_ns[1] == 3000 && delta.gap_ns == 50,
          "a later reading minus an earlier one");
  std::size_t fields = 0;
  double a_us = -1;
  double total_us = -1;
  s.ForEachField([&](const char* name, double value) {
    ++fields;
    if (std::string_view(name) == "a_us")
      a_us = value;
    if (std::string_view(name) == "total_us")
      total_us = value;
  });
  Require(fields == 5 + 2 * qfn::kHeadTimingBuckets + 2 && a_us == 1.0 &&
              total_us == 4.0,
          "fields are reported in microseconds");
}

// The fallback's measurement counters: fixed-point mass to a double,
// differences, sums and the reported field names.
void CheckCounters() {
  unsigned long long c[qfn::HeadCounterStats::kCounterCount]{};
  c[0] = 10;
  c[1] = 3;
  c[2] = 3ULL << 31;  // 1.5 in 2^-32 units
  c[3] = 4;
  c[4] = 2;
  c[5] = 7;
  c[6] = 40;
  c[7] = 9;
  c[8] = 5;
  c[9] = 1;
  c[10] = 2;
  const auto s = qfn::HeadCounterStats::FromCounters(c);
  Require(s.flagged == 10 && s.argmax_changed == 3 && s.mass_sum == 1.5 &&
              s.mass_over_005 == 4 && s.mass_over_02 == 2 && s.chains == 7 &&
              s.chain_rows == 40 && s.chain_flagged == 9 &&
              s.discarded_flagged == 5 && s.used_changed == 1 &&
              s.discarded_changed == 2,
          "counters convert");
  auto sum = s;
  sum += s;
  Require(
      sum.flagged == 20 && sum.mass_sum == 3.0 && sum.discarded_flagged == 10,
      "readings add");
  const auto delta = sum.Since(s);
  Require(delta.flagged == 10 && delta.mass_sum == 1.5 &&
              delta.discarded_changed == 2,
          "a later reading minus an earlier one");
  std::size_t fields = 0;
  double mass = -1;
  double over = -1;
  s.ForEachField([&](const char* name, double value) {
    ++fields;
    if (std::string_view(name) == "mass_sum")
      mass = value;
    if (std::string_view(name) == "mass_over_0p05")
      over = value;
  });
  Require(fields == qfn::HeadCounterStats::kCounterCount && mass == 1.5 &&
              over == 4,
          "one reported field per counter");
}

int main() {
  try {
    CheckCounters();
    CheckPlan();
    CheckEmpty();
    CheckTopProbability();
    CheckFallbackRule();
    CheckExpand();
    CheckStats();
    CheckTiming();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
