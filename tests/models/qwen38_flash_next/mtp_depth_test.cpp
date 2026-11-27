// The depth-14 gate-probe build raises kMaxMtpDraftTokens from seven to
// fourteen. At the default limit of seven every policy decision must stay
// exactly what the seven-draft controllers decided; at fourteen the
// controllers must stay within their limits and the measured cost widths,
// and the probe records must keep their format with longer lists.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/mtp_costs.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/rollback_rows.hpp"
#include "src/models/qwen38_flash_next/trie_shadow.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// The controllers as they were with kMaxMtpDraftTokens = 7 (14872adb),
// kept verbatim apart from names, as the reference for the default limit.
namespace seven {

inline constexpr std::uint32_t kMax = 7;

struct State {
  std::array<float, kMax> successes;
  std::array<float, kMax> failures;
  std::uint32_t retry_tokens{0};
  std::uint32_t probe_depth{0};
  std::uint32_t explored_depth{0};
  std::uint32_t probe_delay{0};
  std::uint32_t failed_depths{0};
};

class LengthController {
public:
  explicit LengthController(std::uint32_t limit, std::uint32_t concurrency = 1)
      : limit_(std::clamp(limit, 1U, kMax)),
        concurrency_(std::clamp(concurrency, 1U, 8U)) {
    Reset();
  }
  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget,
                                     std::uint32_t context = 0,
                                     std::uint32_t concurrency = 0,
                                     bool sampled_subset = false) const {
    const auto cap = std::min(budget, limit_);
    if (cap == 0 || state_.retry_tokens != 0)
      return 0;
    sampled_subset = sampled_subset && state_.explored_depth != 0;
    return std::min(
        cap,
        std::max({1U, BestLength(cap, context, concurrency, sampled_subset),
                  state_.probe_depth}));
  }
  [[nodiscard]] float ExpectedTokens(std::uint32_t drafts) const {
    float expected = 1.0F, survival = 1.0F;
    for (unsigned depth = 0; depth < std::min(drafts, limit_); ++depth) {
      survival *= Probability(depth);
      expected += survival;
    }
    return expected;
  }
  void Observe(std::size_t accepted, std::size_t drafted,
               std::uint32_t context = 0, std::uint32_t concurrency = 0,
               bool sampled_subset = false) {
    if (drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const auto observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t depth = 0; depth < observed; ++depth) {
      state_.successes[depth] =
          0.75F * state_.successes[depth] + (depth < accepted ? 1.0F : 0.0F);
      state_.failures[depth] =
          0.75F * state_.failures[depth] + (depth == accepted ? 1.0F : 0.0F);
    }
    state_.probe_depth = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= state_.explored_depth || state_.probe_delay == 0) {
        state_.probe_depth = std::min<std::uint32_t>(2 * drafted, limit_);
        state_.probe_delay = 16;
      } else {
        --state_.probe_delay;
      }
    }
    state_.explored_depth =
        std::max<std::uint32_t>(state_.explored_depth, observed);
    if (accepted < drafted)
      state_.failed_depths |= 1U << accepted;
    if (BestLength(limit_, context, concurrency, sampled_subset) == 0)
      state_.retry_tokens = 16;
  }
  void ObserveArToken() {
    if (state_.retry_tokens != 0)
      --state_.retry_tokens;
  }
  void Reset() {
    state_.successes.fill(1.5F);
    state_.failures.fill(0.5F);
    state_.retry_tokens = 0;
    state_.probe_depth = 0;
    state_.explored_depth = 0;
    state_.probe_delay = 0;
    state_.failed_depths = 0;
  }
  [[nodiscard]] const State& Get() const { return state_; }

private:
  [[nodiscard]] float Probability(unsigned depth) const {
    return depth < state_.explored_depth &&
                   (state_.failed_depths & (1U << depth)) == 0
               ? 1.0F
               : state_.successes[depth] /
                     (state_.successes[depth] + state_.failures[depth]);
  }
  [[nodiscard]] std::uint32_t BestLength(std::uint32_t cap,
                                         std::uint32_t context,
                                         std::uint32_t concurrency,
                                         bool sampled_subset) const {
    const auto costs = qfn::MtpCycleCosts(
        context, concurrency == 0 ? concurrency_ : concurrency, sampled_subset);
    float survival = 1.0F;
    float expected_tokens = 1.0F;
    float best_score = 1.0F;
    std::uint32_t best = 0;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      const auto depth = length - 1;
      const float probability = Probability(depth);
      survival *= probability;
      expected_tokens += survival;
      const float cost = costs[length] / costs[0];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }
  std::uint32_t limit_;
  std::uint32_t concurrency_;
  State state_{};
};

class BatchController {
public:
  struct Row {
    const LengthController* acceptance;
    std::uint32_t budget;
  };
  [[nodiscard]] std::uint32_t Choose(std::span<const Row> rows,
                                     std::uint32_t context) const {
    if (rows.empty() || rows.size() > 8)
      return 0;
    const auto& cell = CellFor(rows.size(), context);
    std::uint32_t cap = kMax;
    for (const auto& row : rows)
      cap = std::min(cap, row.budget);
    if (cap == 0)
      return 0;
    const bool stable = last_concurrency_ == rows.size() &&
                        last_context_bin_ == ContextBin(context) &&
                        stable_cycles_ >= 2;
    if (stable && (cell.samples[0] == 0 || cell.cycles % 32 < 2))
      return 0;
    if (stable && cell.samples[0] != 0 && cell.samples[1] < 2)
      return 1;
    auto costs = qfn::MtpCycleCosts(context, rows.size());
    const float scale = cell.samples[0] ? cell.ms[0] / costs[0] : 1.0F;
    for (unsigned width = 0; width <= cap; ++width)
      costs[width] =
          cell.samples[width] >= 2 ? cell.ms[width] : costs[width] * scale;
    float best_score = rows.size() / costs[0];
    unsigned best = 0;
    for (unsigned width = 1; width <= cap; ++width) {
      float expected = 0;
      for (const auto& row : rows)
        expected += row.acceptance->ExpectedTokens(width);
      const float score = expected / costs[width];
      if (score > best_score * 1.01F) {
        best_score = score;
        best = width;
      }
    }
    if (cell.cycles % 16 == 2)
      return std::min(cap, std::max(1U, best + 1));
    const auto probe_phase = cell.cycles % 16;
    if (stable && (probe_phase == 3 || probe_phase == 4) &&
        cell.last_drafts != 0 && cell.samples[cell.last_drafts] < 2)
      return std::min(cap, cell.last_drafts);
    return best;
  }
  void SkipCycle() {
    last_concurrency_ = 0;
    stable_cycles_ = 0;
  }
  void Observe(std::size_t concurrency, std::uint32_t context,
               std::uint32_t drafts, float ms) {
    if (concurrency == 0 || concurrency > 8 || drafts > kMax ||
        !std::isfinite(ms) || ms <= 0)
      return;
    const auto bin = ContextBin(context);
    const bool same_cohort =
        last_concurrency_ == concurrency && last_context_bin_ == bin;
    stable_cycles_ = same_cohort ? std::min(stable_cycles_ + 1, 1000000U) : 1;
    last_concurrency_ = concurrency;
    last_context_bin_ = bin;
    auto& cell = CellFor(concurrency, context);
    if (!same_cohort || drafts != cell.last_drafts) {
      cell.last_drafts = drafts;
      ++cell.cycles;
      return;
    }
    cell.last_drafts = drafts;
    auto& count = cell.samples[drafts];
    auto& estimate = cell.ms[drafts];
    if (count < 2)
      estimate = ms;
    else
      estimate += 0.25F * (ms - estimate);
    count = std::min(count + 1, 1000000U);
    ++cell.cycles;
  }

private:
  struct Cell {
    std::array<float, 8> ms{};
    std::array<unsigned, 8> samples{};
    std::uint64_t cycles{0};
    std::uint32_t last_drafts{0};
  };
  static unsigned ContextBin(std::uint32_t context) {
    unsigned bin = 0;
    for (unsigned bound = 4096; context >= bound && bin < 7; bound *= 2)
      ++bin;
    return bin;
  }
  const Cell& CellFor(std::size_t concurrency, std::uint32_t context) const {
    return cells_[concurrency - 1][ContextBin(context)];
  }
  Cell& CellFor(std::size_t concurrency, std::uint32_t context) {
    return cells_[concurrency - 1][ContextBin(context)];
  }
  std::array<std::array<Cell, 8>, 8> cells_{};
  std::size_t last_concurrency_{0};
  unsigned last_context_bin_{0};
  unsigned stable_cycles_{0};
};

}  // namespace seven

/// A fixed pseudo-random stream, so every run replays the same workload.
struct Stream {
  std::uint64_t state;
  std::uint32_t Next(std::uint32_t bound) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>((state >> 33) % bound);
  }
};

bool SameState(const qfn::MtpLengthState& actual, const seven::State& expected,
               std::uint32_t limit) {
  for (std::size_t depth = 0; depth < seven::kMax; ++depth)
    if (actual.successes[depth] != expected.successes[depth] ||
        actual.failures[depth] != expected.failures[depth])
      return false;
  // Depths past seven are never observed below a limit of seven: they keep
  // their reset prior, so a restored state validates the same way.
  for (std::size_t depth = seven::kMax; depth < qfn::kMaxMtpDraftTokens;
       ++depth)
    if (actual.successes[depth] != 1.5F || actual.failures[depth] != 0.5F)
      return false;
  return limit <= seven::kMax && actual.retry_tokens == expected.retry_tokens &&
         actual.probe_depth == expected.probe_depth &&
         actual.explored_depth == expected.explored_depth &&
         actual.probe_delay == expected.probe_delay &&
         actual.failed_depths == expected.failed_depths;
}

void CheckConstants() {
  static_assert(qfn::kMaxMtpDraftTokens == 14);
  static_assert(qfn::kDefaultMtpDraftTokens == 7);
  static_assert(qfn::kMtpCostWidths == 8 && qfn::kMtpPricedDraftTokens == 7);
  static_assert(qfn::kMaxBatchedMtpDraftTokens == 7);
  static_assert(std::tuple_size_v<decltype(qfn::MtpCycleCosts(0, 1))> == 8,
                "the measured cost curve grew past its measured widths");
  std::cout << "depth constants: limit 14, default 7, priced widths 1..8\n";
}

// A verification of kMaxMtpDraftTokens drafts snapshots one proper prefix
// per draft; the device rollback rows (RollbackRows) held seven, so a
// fourteen-draft pass failed EnsureRollback ("invalid rollback allocation
// depth") on the GPU.
void CheckRollbackRows() {
  namespace rocm = qfn::rocm;
  static_assert(std::size(rocm::RollbackRows{}.rows) >= qfn::kMaxMtpDraftTokens,
                "rollback rows must cover the longest chain");
  static_assert(rocm::kMaxRollbackRows == qfn::kMaxMtpDraftTokens);
  static_assert(rocm::kMaxRollbackRows >= qfn::kMaxBatchedMtpDraftTokens);
  // The engine verifies limit + 1 rows (max_speculative), so EnsureRollback
  // asks for one row per draft.
  for (std::uint32_t limit = 1; limit <= qfn::kMaxMtpDraftTokens; ++limit) {
    const auto max_speculative = limit + 1;
    for (std::uint32_t depth = 0; depth <= limit; ++depth)
      Require(rocm::RollbackDepthAllowed(depth, max_speculative, 0),
              "rollback depth " + std::to_string(depth) + " refused at limit " +
                  std::to_string(limit));
    Require(!rocm::RollbackDepthAllowed(limit + 1, max_speculative, 0),
            "rollback depth past the pass width accepted");
    Require(!rocm::RollbackDepthAllowed(limit, max_speculative, 1),
            "rollback grown under a pending speculative batch");
  }
  // Past the rows themselves, even a wider executor must refuse.
  Require(!rocm::RollbackDepthAllowed(rocm::kMaxRollbackRows + 1,
                                      rocm::kMaxRollbackRows + 3, 0),
          "rollback depth past RollbackRows accepted");
  // The default limit asks for exactly the seven rows it always used.
  Require(rocm::RollbackDepthAllowed(qfn::kDefaultMtpDraftTokens,
                                     qfn::kDefaultMtpDraftTokens + 1, 0) &&
              !rocm::RollbackDepthAllowed(qfn::kDefaultMtpDraftTokens + 1,
                                          qfn::kDefaultMtpDraftTokens + 1, 0),
          "the default limit's rollback depth changed");
  std::cout << "rollback rows: " << rocm::kMaxRollbackRows
            << " slots, depths 1..14 accepted\n";
}

// Every decision and state at limits 1..7 equals the seven-draft controller,
// across occupancies, contexts and both head curves.
void CheckLengthControllerAtSeven() {
  Stream stream{0x94};
  std::size_t decisions = 0;
  for (std::uint32_t limit = 1; limit <= 7; ++limit) {
    for (std::uint32_t concurrency : {1U, 2U, 4U, 6U, 8U}) {
      qfn::MtpLengthController actual(limit, concurrency);
      seven::LengthController expected(limit, concurrency);
      for (unsigned cycle = 0; cycle < 400; ++cycle) {
        const auto budget = stream.Next(12);
        const auto context = stream.Next(5) * 16384 + stream.Next(4096);
        const auto occupancy = stream.Next(3) == 0 ? stream.Next(9) : 0;
        const bool subset = stream.Next(2) == 0;
        const auto length = actual.Choose(budget, context, occupancy, subset);
        Require(length == expected.Choose(budget, context, occupancy, subset),
                "a limit of seven or less chose another width");
        ++decisions;
        for (std::uint32_t drafts = 0; drafts <= 9; ++drafts)
          Require(
              actual.ExpectedTokens(drafts) == expected.ExpectedTokens(drafts),
              "expected tokens changed at a limit of seven or less");
        if (length == 0) {
          actual.ObserveArToken();
          expected.ObserveArToken();
          continue;
        }
        // Drafted counts past the limit are clamped by both controllers.
        const auto drafted = length + stream.Next(3);
        const auto accepted = stream.Next(drafted + 1);
        actual.Observe(accepted, drafted, context, occupancy, subset);
        expected.Observe(accepted, drafted, context, occupancy, subset);
        Require(SameState(actual.State(), expected.Get(), limit),
                "a limit of seven or less changed the controller state");
        if (cycle % 97 == 96) {
          qfn::MtpLengthController restored(limit, concurrency);
          Require(restored.Restore(actual.State()),
                  "a limit-seven state no longer restores");
          actual = restored;
        }
      }
    }
  }
  std::cout << "length controller at limits 1..7 matches the seven-draft "
               "controller over "
            << decisions << " decisions\n";
}

void CheckBatchControllerAtSeven() {
  Stream stream{0x941};
  std::size_t decisions = 0;
  for (std::size_t sessions = 2; sessions <= 8; ++sessions) {
    std::vector<qfn::MtpLengthController> actual_rows;
    std::vector<seven::LengthController> expected_rows;
    for (std::size_t i = 0; i < sessions; ++i) {
      actual_rows.emplace_back(7, static_cast<std::uint32_t>(sessions));
      expected_rows.emplace_back(7, static_cast<std::uint32_t>(sessions));
    }
    qfn::MtpBatchController actual;
    seven::BatchController expected;
    for (unsigned cycle = 0; cycle < 300; ++cycle) {
      std::vector<qfn::MtpBatchController::Row> a;
      std::vector<seven::BatchController::Row> e;
      // DecodeBatchImpl's budgets at a limit of seven are at most seven.
      for (std::size_t i = 0; i < sessions; ++i) {
        const auto budget = stream.Next(8);
        a.push_back({&actual_rows[i], budget});
        e.push_back({&expected_rows[i], budget});
      }
      const auto context = stream.Next(4) * 8192 + stream.Next(2048);
      const auto width = actual.Choose(a, context);
      Require(width == expected.Choose(e, context),
              "the batch planner chose another width at a limit of seven");
      ++decisions;
      for (std::size_t i = 0; i < sessions; ++i) {
        const auto accepted = width == 0 ? 0 : stream.Next(width + 1);
        if (width != 0) {
          actual_rows[i].Observe(accepted, width, context);
          expected_rows[i].Observe(accepted, width, context);
        }
      }
      if (stream.Next(13) == 0) {
        actual.SkipCycle();
        expected.SkipCycle();
      }
      const float ms = 40.0F + 30.0F * width + stream.Next(20);
      actual.Observe(sessions, context, width, ms);
      expected.Observe(sessions, context, width, ms);
    }
  }
  std::cout << "batch planner at a limit of seven matches the seven-draft "
               "planner over "
            << decisions << " decisions\n";
}

void CheckLengthControllerAtFourteen() {
  // A perfect head reaches the full limit through the probe depth (twice the
  // last fully accepted chain): 3, 6, 12, 14. The priced choice then stays at
  // seven, the longest measured width, between probes. A nonzero choice only
  // selects drafting over AR; the engine drafts to its limit (PrepareDecode).
  qfn::MtpLengthController controller(14);
  std::vector<std::uint32_t> lengths;
  for (unsigned cycle = 0; cycle < 16; ++cycle) {
    const auto length = controller.Choose(14);
    Require(length >= 1 && length <= 14, "limit 14 chose outside 1..14");
    lengths.push_back(length);
    controller.Observe(length, length);
  }
  Require(std::ranges::equal(std::span(lengths).first(5),
                             std::array<std::uint32_t, 5>{3, 6, 12, 14, 7}),
          "a perfect head did not probe its way to fourteen drafts");
  Require(
      *std::ranges::max_element(lengths) == 14 && controller.Choose(14) == 7,
      "the priced width of a perfect head is not the measured maximum");
  Require(controller.Choose(5) == 5 && controller.Choose(0) == 0,
          "limit 14 ignored the output budget");
  Require(controller.ExpectedTokens(14) > controller.ExpectedTokens(7),
          "depths eight to fourteen add no expected tokens");

  // A failure at depth fourteen is recorded and survives a restore.
  controller.Observe(13, 14);
  auto state = controller.State();
  Require((state.failed_depths & (1U << 13)) != 0 &&
              state.explored_depth == 14 && state.failures[13] > 0.5F,
          "a rejection at depth fourteen was not observed");
  qfn::MtpLengthController restored(14);
  Require(restored.Restore(state), "a depth-14 state does not restore");
  for (std::uint32_t budget = 0; budget <= 16; ++budget)
    Require(restored.Choose(budget) == controller.Choose(budget),
            "a restored depth-14 state chose another width");
  auto invalid = state;
  invalid.failed_depths = 1U << 14;
  Require(!restored.Restore(invalid), "a failure past depth 14 restored");
  invalid = state;
  invalid.probe_depth = 15;
  Require(!restored.Restore(invalid), "a probe past depth 14 restored");
  invalid = state;
  invalid.successes[13] = -1.0F;
  Require(!restored.Restore(invalid), "an invalid depth-14 prior restored");
  // A limit-seven session rejects a depth-14 history rather than clamp it.
  qfn::MtpLengthController seven_limit(7);
  Require(!seven_limit.Restore(state),
          "a limit-seven session accepted a depth-14 history");

  // The AR backoff prices only measured widths: drafting at most seven, a
  // limit of fourteen backs off exactly when a limit of seven does.
  Stream stream{0x1414};
  for (std::uint32_t concurrency : {1U, 8U}) {
    qfn::MtpLengthController deep(14, concurrency), shallow(7, concurrency);
    for (unsigned cycle = 0; cycle < 500; ++cycle) {
      const auto context = stream.Next(4) * 8192;
      const bool deep_ar = deep.Choose(14, context) == 0;
      Require(deep_ar == (shallow.Choose(7, context) == 0),
              "limit 14 changed the AR backoff decision");
      if (deep_ar) {
        deep.ObserveArToken();
        shallow.ObserveArToken();
        continue;
      }
      // Rejections only, at or before depth seven: the chains agree.
      const auto drafted = 7U;
      const auto accepted = stream.Next(7);
      deep.Observe(accepted, drafted, context);
      shallow.Observe(accepted, drafted, context);
    }
  }

  // Wide budgets at every occupancy and context stay within the curve.
  for (std::uint32_t concurrency = 1; concurrency <= 8; ++concurrency) {
    qfn::MtpLengthController wide(14, concurrency);
    for (unsigned cycle = 0; cycle < 64; ++cycle) {
      const auto context = cycle * 4096;
      const bool subset = cycle % 2 == 0;
      const auto length = wide.Choose(14, context, 0, subset);
      Require(length <= 14, "limit 14 exceeded its limit");
      if (length == 0) {
        wide.ObserveArToken();
        continue;
      }
      wide.Observe(cycle % (length + 1), length, context, 0, subset);
    }
  }
  std::cout << "length controller at limit 14 reaches 14 drafts, restores, "
               "and backs off as at 7\n";
}

void CheckBatchControllerAtFourteen() {
  // A batch member with a deeper session limit still verifies at most seven
  // drafts in the eight-row decode batch.
  qfn::MtpLengthController a(14), b(14);
  for (unsigned i = 0; i < 12; ++i) {
    a.Observe(14, 14);
    b.Observe(14, 14);
  }
  std::array<qfn::MtpBatchController::Row, 2> rows{{{&a, 14}, {&b, 14}}};
  qfn::MtpBatchController policy, reference;
  for (unsigned cycle = 0; cycle < 200; ++cycle) {
    const auto width = policy.Choose(rows, 0);
    Require(width <= qfn::kMaxBatchedMtpDraftTokens,
            "a batched chain exceeded seven drafts");
    Require(width == reference.Choose(rows, 0),
            "an ignored observation changed the batch planner");
    policy.Observe(2, 0, width, 60.0F + 20.0F * width);
    reference.Observe(2, 0, width, 60.0F + 20.0F * width);
    // A lone cycle's longer chain is not a batched width: it is ignored.
    policy.Observe(2, 0, 8 + cycle % 7, 500.0F);
  }
  std::cout << "batch planner keeps batched chains at seven drafts\n";
}

// The probe-v2 record of a fourteen-draft chain keeps the key set and layout
// of a seven-draft one (score_gate.RECORD_KEYS | V2_KEYS); only its lists
// are longer.
void CheckProbeRecordAtFourteen() {
  qfn::GateProbeChain record{
      .position = 100, .sampled = false, .chain = 16, .keep = 12, .lookup = 1};
  for (int j = 0; j < 14; ++j) {
    record.probability.push_back(j % 2 == 0 ? 1.0F : 0.5F);
    record.agree.push_back(j % 3 - 1);
    record.order.push_back(static_cast<std::uint32_t>(j % 4));
    record.trie.push_back(j == 13 ? -1 : 1000 + j);
  }
  for (int j = 0; j < 12; ++j)
    record.kept.push_back(2000 + j);
  record.ms = 123.4567;
  const auto line = qfn::FormatGateProbe(record);
  const std::string expected =
      "{\"pos\":100,\"sampled\":false,\"chain\":16,\"keep\":12,\"lookup\":1,"
      "\"p\":[1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5],"
      "\"agree\":[-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0],"
      "\"order\":[0,1,2,3,0,1,2,3,0,1,2,3,0,1],\"ms\":123.457,"
      "\"trie\":[1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,"
      "1012,-1],"
      "\"kept\":[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011]}"
      "\n";
  Require(line == expected, "a depth-14 probe record changed format: " + line);
  std::cout << "probe-v2 record of a 14-draft chain keeps its format\n";
}

}  // namespace

int main() {
  try {
    CheckConstants();
    CheckRollbackRows();
    CheckLengthControllerAtSeven();
    CheckBatchControllerAtSeven();
    CheckLengthControllerAtFourteen();
    CheckBatchControllerAtFourteen();
    CheckProbeRecordAtFourteen();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
