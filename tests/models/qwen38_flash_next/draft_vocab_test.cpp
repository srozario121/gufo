#include "src/models/qwen38_flash_next/draft_vocab.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

namespace {

// IDs below 100 stand for the prefix and the special tokens.
bool Common(std::int32_t id) {
  return id >= 0 && id < 100;
}

}  // namespace

// Text inside the prefix needs no table rows and keeps the small head.
void CheckCommonText() {
  qfn::MtpDraftVocab vocab(8);
  std::vector<std::int32_t> tokens(qfn::kMtpDraftVocabWindow + 10, 1);
  std::vector<std::int32_t> added;
  vocab.CatchUp(tokens, Common, &added);
  Require(added.empty() && vocab.rare().empty(), "common tokens were tabled");
  Require(vocab.Fits(2, Common), "common text left the small head");
  Require(qfn::MtpDraftVocab(8).Fits(2, Common), "empty history");
}

// A new rare token is tabled in first-use order. One miss in the window
// keeps the small head; a second one leaves it until the earlier of the two
// has left the window.
void CheckMissWindow() {
  qfn::MtpDraftVocab vocab(8);
  std::vector<std::int32_t> tokens{1, 500, 1};
  std::vector<std::int32_t> added;
  vocab.CatchUp(tokens, Common, &added);
  Require(vocab.Fits(1, Common) && vocab.Fits(500, Common),
          "a lone miss left the small head");
  tokens.insert(tokens.end(), {700, 500});
  vocab.CatchUp(tokens, Common, &added);
  Require(added == std::vector<std::int32_t>{500, 700}, "tabled order");
  Require(!vocab.Fits(1, Common), "small head right after a second miss");
  // The earlier miss (500, index 1) stays in the window until W more tokens.
  tokens.resize(1 + qfn::kMtpDraftVocabWindow, 500);
  added.clear();
  vocab.CatchUp(tokens, Common, &added);
  Require(added.empty(), "a known token was tabled again");
  Require(!vocab.Fits(1, Common), "miss left the window too early");
  tokens.push_back(1);
  vocab.CatchUp(tokens, Common, &added);
  Require(vocab.Fits(1, Common) && vocab.Fits(700, Common),
          "known text did not return to the small head");
  Require(!vocab.Fits(900, Common), "an untabled anchor drafted small");
}

// A full table turns every further rare token into a miss.
void CheckCapacity() {
  qfn::MtpDraftVocab vocab(2);
  std::vector<std::int32_t> tokens{500, 600, 700};
  tokens.resize(3 + qfn::kMtpDraftVocabWindow, 1);
  std::vector<std::int32_t> added;
  vocab.CatchUp(tokens, Common, &added);
  Require(added == std::vector<std::int32_t>{500, 600}, "table overfilled");
  Require(vocab.Fits(600, Common) && !vocab.Fits(700, Common),
          "untabled token proposed");
  tokens.insert(tokens.end(), {700, 700});
  vocab.CatchUp(tokens, Common, &added);
  Require(!vocab.Fits(1, Common), "overflow token is not a miss");
}

// Reset forgets the table and the history.
void CheckReset() {
  qfn::MtpDraftVocab vocab(8);
  std::vector<std::int32_t> tokens{500};
  vocab.CatchUp(tokens, Common, nullptr);
  vocab.Reset();
  Require(vocab.scanned() == 0 && vocab.rare().empty() && vocab.Fits(1, Common),
          "reset kept state");
  Require(!vocab.Fits(500, Common), "reset kept a tabled token");
}

// Sampled requests may use the small head at any session count, in or out of
// a batch, so a seeded request drafts the same way whatever overlaps it.
// Greedy eligibility is unchanged: plain greedy needs one session or a
// jointly planned batch, and greedy penalties/constraints never qualify.
void CheckEligibility() {
  for (const std::uint32_t sessions : {1U, 2U, 8U}) {
    for (const bool planned : {false, true}) {
      Require(qfn::MtpDraftSubsetAllowed(true, false, sessions, planned),
              "a sampled request lost the small head");
      Require(!qfn::MtpDraftSubsetAllowed(false, false, sessions, planned),
              "greedy penalties or constraints reached the small head");
      Require(qfn::MtpDraftSubsetAllowed(false, true, sessions, planned) ==
                  (sessions == 1 || planned),
              "plain greedy eligibility changed");
    }
  }
}

// The draft-set tests use small SYNTHETIC lists. They are test data only; the
// real sets are files given on the command line, never compiled in.

// One decimal ID per line; blanks, comments, CRLF and padding are tolerated,
// first place wins for a repeat, and anything else is refused with its line.
void CheckTokenIdList() {
  std::string error;
  auto ids =
      qfn::ParseTokenIdList("5\n# note\n\n  7 \r\n5\n0\n99", 100, &error);
  Require(ids && *ids == std::vector<std::int32_t>{5, 7, 0, 99},
          "token list order or duplicates");
  Require(qfn::ParseTokenIdList("", 100, &error)->empty(), "empty list");
  for (const char* bad :
       {"5\nx\n", "5\n100\n", "-1\n", "1 2\n", "5\n+3\n", "1234567890\n"}) {
    error.clear();
    Require(!qfn::ParseTokenIdList(bad, 100, &error) && !error.empty(),
            "a bad line was accepted");
  }
  error.clear();
  Require(!qfn::ParseTokenIdList("1\n2\nzz", 100, &error) &&
              error.find("line 3") != std::string::npos,
          "the error names the line");
}

// With no set the plan is exactly the specials: today's draft head.
void CheckPlanWithoutSet() {
  const auto plan = qfn::PlanMtpDraftVocab(1000, 1200, {1190, 1191});
  Require(plan.rows == std::vector<std::int32_t>{1190, 1191} &&
              plan.set_ids == 0 && plan.set_rows == 0,
          "plan without a set changed the table");
  Require(plan.member[1190] && !plan.member[1189] && !plan.member[5],
          "membership without a set");
}

// Set IDs below the prefix cost nothing; IDs above it become static rows in
// the set's order after the specials; specials and repeats are not doubled.
void CheckPlanWithSet() {
  const std::vector<std::int32_t> set{10, 1100, 1190, 1050, 999, 1100, 1199};
  const auto plan = qfn::PlanMtpDraftVocab(1000, 1200, {1190, 1191}, set);
  Require(plan.rows == std::vector<std::int32_t>{1190, 1191, 1100, 1050, 1199},
          "static rows or their order");
  Require(plan.set_ids == set.size() && plan.set_prefix == 2 &&
              plan.set_rows == 3 && plan.set_dropped == 0,
          "plan counts");
  Require(plan.member[1100] && plan.member[1050] && !plan.member[1060],
          "membership with a set");
}

// A set larger than the table leaves the request its reserve, in set order.
void CheckPlanCapacity() {
  std::vector<std::int32_t> set;
  for (std::int32_t id = 2000; id < 4000; ++id)
    set.push_back(id);
  const auto plan = qfn::PlanMtpDraftVocab(1000, 5000, {4990, 4991, 4992}, set);
  const std::size_t room =
      qfn::kMtpDraftExtraRows - qfn::kMtpDraftRequestReserve - 3;
  Require(plan.rows.size() == 3 + room && plan.set_rows == room &&
              plan.set_dropped == set.size() - room,
          "static rows exceed the table less the reserve");
  Require(plan.rows[3] == 2000 && plan.rows.back() == 2000 + room - 1 &&
              !plan.member[2000 + room],
          "the set's first IDs are the ones kept");
  Require(qfn::kMtpDraftExtraRows - plan.rows.size() >=
              qfn::kMtpDraftRequestReserve,
          "no room left for the request");
  // A request's table share starts where the static rows end.
  qfn::MtpDraftVocab vocab(qfn::kMtpDraftExtraRows - plan.rows.size());
  std::vector<std::int32_t> tokens(300);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = 3000 + static_cast<std::int32_t>(i);
  std::vector<std::int32_t> added;
  vocab.CatchUp(
      tokens,
      [&](std::int32_t id) {
        return id < 1000 || (id < 5000 && plan.member[id]);
      },
      &added);
  Require(added.size() == qfn::kMtpDraftRequestReserve,
          "the request's share is the reserve");
}

int main() {
  try {
    CheckCommonText();
    CheckMissWindow();
    CheckCapacity();
    CheckReset();
    CheckEligibility();
    CheckTokenIdList();
    CheckPlanWithoutSet();
    CheckPlanWithSet();
    CheckPlanCapacity();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
