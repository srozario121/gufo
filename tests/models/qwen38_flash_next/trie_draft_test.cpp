#include "src/models/qwen38_flash_next/trie_draft.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
using Trie = qfn::TrieDraft;
namespace policy = qfn::trie_policy;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Brute-force PPM-C over exact counts. With at most Trie::kSlots distinct
// tokens nothing is evicted, so the trie must agree exactly. Returns the
// highest blend probability and the longest order; `of`, when set, receives
// the probability of that token.
Trie::Step Reference(const std::vector<std::int32_t>& indexed,
                     const std::vector<std::int32_t>& context,
                     std::int32_t token = -1, double* of = nullptr) {
  std::vector<std::map<std::int32_t, std::uint32_t>> levels;
  for (const auto k : Trie::kOrders) {
    if (k > context.size())
      break;
    std::map<std::int32_t, std::uint32_t> counts;
    for (std::size_t i = k; i < indexed.size(); ++i) {
      bool same = true;
      for (std::size_t j = 0; j < k && same; ++j)
        same = indexed[i - k + j] == context[context.size() - k + j];
      if (same)
        ++counts[indexed[i]];
    }
    if (counts.empty())
      break;
    levels.push_back(counts);
  }
  if (levels.empty())
    return {.token = -1};
  std::vector<std::int32_t> candidates;
  for (const auto& level : levels)
    for (const auto& [token, count] : level)
      candidates.push_back(token);
  Trie::Step best{.token = -1, .probability = -1};
  for (const auto x : candidates) {
    double p = 1.0 / Trie::kBaseVocabulary;
    for (const auto& level : levels) {
      double total = 0;
      for (const auto& [token, count] : level)
        total += count;
      const double used = static_cast<double>(level.size());
      const auto found = level.find(x);
      const double c = found == level.end() ? 0 : found->second;
      p = c / (total + used) + used / (total + used) * p;
    }
    if (x == token && of != nullptr)
      *of = p;
    if (p > best.probability)
      best = {x, p, 0};
  }
  best.order = Trie::kOrders[levels.size() - 1];
  return best;
}

// Every draft step equals the exact blend: token, probability and order.
void CheckAgainstReference() {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 40; ++trial) {
    std::uniform_int_distribution<int> symbol(10, 13);  // 4 symbols
    std::vector<std::int32_t> tokens(60 + trial * 5);
    for (auto& t : tokens)
      t = symbol(rng);
    Trie trie;
    trie.Extend(tokens);
    Require(trie.IndexedTokens() == tokens.size(), "indexes every token");
    std::array<Trie::Step, 5> steps{};
    const std::int32_t tail[1] = {symbol(rng)};
    const auto n = trie.Draft(tokens, tail, steps);
    std::vector<std::int32_t> context(tokens);
    context.push_back(tail[0]);
    for (std::size_t s = 0; s < n; ++s) {
      double chosen = -1;
      const auto want = Reference(tokens, context, steps[s].token, &chosen);
      // Ties go to the first candidate in slot order; any maximiser is right.
      Require(std::abs(chosen - want.probability) < 1e-12,
              "the drafted token has the highest blend probability");
      Require(std::abs(want.probability - steps[s].probability) < 1e-12,
              "draft probability matches the blend");
      Require(want.order == steps[s].order, "longest matched order");
      context.push_back(steps[s].token);
    }
    Require(n == steps.size(), "a 4-symbol history always matches order 1");
  }
}

// A repeated passage is copied with high probability at long orders.
void CheckCopy() {
  std::vector<std::int32_t> tokens;
  for (std::int32_t i = 0; i < 50; ++i)
    tokens.push_back(1000 + i);
  for (std::int32_t i = 0; i < 20; ++i)
    tokens.push_back(1000 + i);
  Trie trie;
  trie.Extend(tokens);
  std::array<Trie::Step, 6> steps{};
  Require(trie.Draft(tokens, {}, steps) == steps.size(), "copy drafted");
  for (std::size_t s = 0; s < steps.size(); ++s) {
    Require(steps[s].token == 1020 + static_cast<std::int32_t>(s),
            "the copy continues the earlier passage");
    Require(steps[s].order >= 16 && steps[s].probability > 0.5,
            "long confident match");
  }
}

// Incremental indexing equals indexing at once; a rewritten history restarts.
void CheckIncrementalAndRewrite() {
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> symbol(1, 3);
  std::vector<std::int32_t> tokens(200);
  for (auto& t : tokens)
    t = symbol(rng);
  Trie whole;
  whole.Extend(tokens);
  Trie parts;
  for (std::size_t n = 0; n <= tokens.size(); n += 17)
    parts.Extend(std::span(tokens).first(n));
  parts.Extend(tokens);
  std::array<Trie::Step, 4> a{};
  std::array<Trie::Step, 4> b{};
  Require(whole.Draft(tokens, {}, a) == parts.Draft(tokens, {}, b),
          "same draft length");
  for (std::size_t i = 0; i < a.size(); ++i)
    Require(a[i].token == b[i].token && a[i].probability == b[i].probability,
            "incremental equals whole");
  Require(whole.Nodes() == parts.Nodes(), "same nodes");

  auto rewritten = tokens;
  rewritten.back() = 99;
  parts.Extend(rewritten);
  Trie fresh;
  fresh.Extend(rewritten);
  Require(parts.Nodes() == fresh.Nodes(), "a rewrite restarts the index");
  parts.Extend(std::span(rewritten).first(10));
  Require(parts.IndexedTokens() == 10, "a shorter history restarts");
}

void CheckNoMatch() {
  Trie trie;
  std::array<Trie::Step, 3> steps{};
  const std::vector<std::int32_t> none;
  Require(trie.Draft(none, none, steps) == 0, "empty index drafts nothing");
  const std::vector<std::int32_t> tokens{5, 6, 7};
  trie.Extend(tokens);
  const std::int32_t unseen[1] = {42};
  Require(trie.Draft(tokens, unseen, steps) == 0,
          "an unseen last token has no order-1 context");
}

void CheckPolicy() {
  // Bins follow numpy.searchsorted(..., side="left").
  Require(policy::HeadChance(0.0) == policy::kHeadChance[0], "lowest decile");
  Require(policy::HeadChance(0.4217758572543825) == policy::kHeadChance[0],
          "an edge value stays in the lower bin");
  Require(policy::HeadChance(0.5) == policy::kHeadChance[1], "second decile");
  Require(policy::HeadChance(1.0) == policy::kHeadChance[8], "top bin");
  Require(policy::Survival({.token = 0, .probability = 0.995, .order = 32}) ==
              policy::kSurvival[9][7],
          "order 32, p > 0.99");
  Require(policy::Survival({.token = 0, .probability = 0.3, .order = 5}) ==
              policy::kSurvival[4][0],
          "order 5 bins with 6, p = 0.3 in the first bin");
  Require(policy::Survival({.token = 0, .probability = 1.0, .order = 1}) == 0,
          "a sparse cell is never copied");

  policy::MtpCost cost;
  const double prior = (policy::kRoundMs + policy::kMtpRowMs * 1.86) / 2.62;
  Require(std::abs(cost.MsPerToken(0) - prior) < 1e-12, "the prior");
  for (int i = 0; i < 1000; ++i)
    cost.Observe(3, 4);
  Require(std::abs(cost.MsPerToken(0) -
                   (policy::kRoundMs + policy::kMtpRowMs * 3) / 4) < 0.05,
          "the request's own rounds dominate");
  cost.Reset();
  Require(cost.MsPerToken(0) == prior, "a new request starts at the prior");

  // A confident long copy pays; nothing to copy never does.
  std::array<Trie::Step, 6> sure{};
  for (auto& s : sure)
    s = {.token = 1, .probability = 1.0, .order = 32};
  Require(policy::PlanCopies(0.98, sure, prior, 1000) == 6,
          "six sure copies after a sure first token");
  Require(policy::PlanCopies(0.98, {}, prior, 1000) == 0, "no copies");
  std::array<Trie::Step, 6> weak{};
  for (auto& s : weak)
    s = {.token = 1, .probability = 0.2, .order = 1};
  Require(policy::PlanCopies(0.98, weak, prior, 1000) == 0,
          "unlikely copies do not pay");
  Require(policy::PlanCopies(0.05, sure, prior, 1000) == 0,
          "a first token that will miss gates every copy");
}

int main() {
  CheckAgainstReference();
  CheckCopy();
  CheckIncrementalAndRewrite();
  CheckNoMatch();
  CheckPolicy();
  std::cout << "trie_draft: ok\n";
  return 0;
}
