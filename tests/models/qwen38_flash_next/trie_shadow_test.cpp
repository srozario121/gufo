// Trie v3 shadow mode (trie_shadow.hpp): where it fires, how a waiting event
// resolves, the JSON line, and the GUFO_TRIE_SHADOW sink.
#include "src/models/qwen38_flash_next/trie_shadow.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace q = gufo::models::qwen38_flash_next;

namespace {

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAILED: " << what << "\n";
    ++failures;
  }
}

bool Has(const std::string& s, const std::string& part) {
  return s.find(part) != std::string::npos;
}

q::TrieShadowEvent Event(std::uint32_t chain, std::uint32_t keep) {
  return {.position = 100 + chain,
          .token = 42,
          .order = 6,
          .probability = 0.25,
          .reason = q::TrieShadowReason::kGate,
          .sampled = true,
          .chain = chain,
          .keep = keep};
}

void TestFires() {
  using q::TrieShadowFires;
  using R = q::TrieShadowReason;
  Expect(TrieShadowFires(0, 3, 8, true, false) == R::kGate,
         "a gate-ended chain without copies fires as gate");
  Expect(TrieShadowFires(0, 3, 8, false, false) == R::kWidth,
         "a width-ended chain without copies fires as width");
  Expect(TrieShadowFires(2, 3, 8, true, false) == R::kNone,
         "a chain with prompt-lookup copies does not fire");
  Expect(TrieShadowFires(0, 8, 8, false, false) == R::kNone,
         "a chain at the cap has no room to fire");
  Expect(TrieShadowFires(0, 3, 8, false, true) == R::kNone,
         "a chain ending in a stop token does not fire");
  Expect(TrieShadowFires(0, 0, 8, false, false) == R::kNone,
         "an empty chain does not fire");
}

void TestTracker() {
  q::TrieShadowTracker t;
  // Not reached: verification kept 2 of 4.
  auto line = t.Verified(Event(4, 2));
  Expect(Has(line, "\"reached\":false") && Has(line, "\"emitted\":-1") &&
             Has(line, "\"match\":false") && !t.Waiting(),
         "a chain not kept whole is written at once, not reached");
  // Reached and matched by the next anchor at the event's position.
  Expect(t.Verified(Event(4, 4)).empty() && t.Waiting(),
         "a chain kept whole waits for the next anchor");
  line = t.Anchor(104, 42);
  Expect(Has(line, "\"reached\":true") && Has(line, "\"emitted\":42") &&
             Has(line, "\"match\":true") && !t.Waiting(),
         "the next anchor at the event's position resolves it as a match");
  // Reached, not matched.
  (void)t.Verified(Event(3, 3));
  line = t.Anchor(103, 7);
  Expect(Has(line, "\"emitted\":7") && Has(line, "\"match\":false"),
         "a different anchor is a reached miss");
  // A different position flushes the wait unresolved.
  (void)t.Verified(Event(3, 3));
  line = t.Anchor(999, 42);
  Expect(Has(line, "\"reached\":true") && Has(line, "\"emitted\":-1") &&
             !t.Waiting(),
         "an anchor at another position flushes the wait unresolved");
  // Flush on reset, and a second event replacing a wait flushes the first.
  (void)t.Verified(Event(3, 3));
  line = t.Verified(Event(5, 5));
  Expect(Has(line, "\"pos\":103") && t.Waiting(),
         "a new wait flushes the old one");
  line = t.Flush();
  Expect(Has(line, "\"pos\":105") && !t.Waiting() && t.Flush().empty(),
         "flush writes the wait once");
  Expect(t.Anchor(1, 1).empty(),
         "an anchor with nothing waiting writes nothing");
}

void TestFormat() {
  auto e = Event(4, 4);
  e.reason = q::TrieShadowReason::kWidth;
  e.sampled = false;
  const auto line = q::FormatTrieShadow(e, true, 42);
  Expect(line ==
             "{\"pos\":104,\"reason\":\"width\",\"sampled\":false,\"chain\":4,"
             "\"keep\":4,\"reached\":true,\"trie\":42,\"order\":6,\"p\":0.25,"
             "\"emitted\":42,\"match\":true,\"conf\":1,\"g_tok\":-1,"
             "\"g_order\":0,\"g_p\":0,\"g_agree\":false,\"g_acc\":false,"
             "\"lk_len\":0,\"lk_tok\":-1,\"lk_match\":false}\n",
         "the JSON line, no variant data: " + line);
}

void TestVariants() {
  // A: the confidence product is carried as given.
  auto e = Event(4, 4);
  e.confidence = 0.125;
  Expect(Has(q::FormatTrieShadow(e, false, -1), "\"conf\":0.125"),
         "A: the confidence product");
  // B: a gate prediction that agrees with the gated draft, accepted (kept
  // whole) and not accepted (kept short).
  e.gate_token = 9;
  e.gate_agree = true;
  e.gate_order = 3;
  e.gate_probability = 0.75;
  auto line = q::FormatTrieShadow(e, true, 1);
  Expect(Has(line,
             "\"g_tok\":9,\"g_order\":3,\"g_p\":0.75,"
             "\"g_agree\":true,\"g_acc\":true"),
         "B: an agreeing gate prediction on a chain kept whole: " + line);
  e.keep = 3;
  line = q::FormatTrieShadow(e, false, -1);
  Expect(Has(line, "\"g_agree\":true,\"g_acc\":false"),
         "B: ... and on a chain the gated draft was rejected in");
  e.gate_agree = false;
  Expect(Has(q::FormatTrieShadow(e, false, -1), "\"g_agree\":false"),
         "B: a disagreeing gate prediction");
  // D: a lookup proposal matches only when reached and equal to the emitted.
  e = Event(4, 4);
  e.lookup_length = 5;
  e.lookup_token = 77;
  Expect(Has(q::FormatTrieShadow(e, true, 77),
             "\"lk_len\":5,\"lk_tok\":77,\"lk_match\":true"),
         "D: a reached lookup proposal equal to the emitted token matches");
  Expect(Has(q::FormatTrieShadow(e, true, 78), "\"lk_match\":false"),
         "D: ... a different emitted token does not");
  Expect(Has(q::FormatTrieShadow(e, false, -1), "\"lk_match\":false"),
         "D: ... nor an unreached position");
  // The tracker carries the variant fields through to the resolved line.
  q::TrieShadowTracker t;
  (void)t.Verified(e);
  Expect(Has(t.Anchor(104, 77), "\"lk_match\":true"),
         "D: the tracker resolves the lookup match with the anchor");
}

void TestGateProbe() {
  q::GateProbeChain c{.position = 77,
                      .sampled = true,
                      .chain = 4,
                      .keep = 2,
                      .lookup = 0,
                      .probability = {0.5F, 0.25F, 0.125F},
                      .agree = {1, 0, -1},
                      .order = {6, 1, 0}};
  const auto line = q::FormatGateProbe(c);
  Expect(line ==
             "{\"pos\":77,\"sampled\":true,\"chain\":4,\"keep\":2,"
             "\"lookup\":0,\"p\":[0.5,0.25,0.125],\"agree\":[1,0,-1],"
             "\"order\":[6,1,0]}\n",
         "the gate probe line: " + line);
  c.probability.clear();
  c.agree.clear();
  c.order.clear();
  Expect(Has(q::FormatGateProbe(c), "\"p\":[],\"agree\":[],\"order\":[]"),
         "a chain without MTP drafts has empty arrays");
}

void TestSink() {
  char path[] = "/tmp/trie_shadow_testXXXXXX";
  const int fd = mkstemp(path);
  Expect(fd >= 0, "a temp file");
  if (fd < 0)
    return;
  close(fd);
  unsetenv("GUFO_TRIE_SHADOW");
  Expect(q::TrieShadowLog::FromEnvironment() == nullptr,
         "unset GUFO_TRIE_SHADOW: shadow mode off");
  setenv("GUFO_TRIE_SHADOW", path, 1);
  {
    auto log = q::TrieShadowLog::FromEnvironment();
    Expect(log != nullptr, "set GUFO_TRIE_SHADOW: a sink");
    if (log) {
      log->Write("a\n");
      log->Write("");
      log->Write("b\n");
    }
  }
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  Expect(text.str() == "a\nb\n", "lines appended in order: " + text.str());
  setenv("GUFO_TRIE_SHADOW", "/nonexistent-dir/x.jsonl", 1);
  Expect(q::TrieShadowLog::FromEnvironment() == nullptr,
         "an unopenable path leaves shadow mode off");
  unsetenv("GUFO_TRIE_SHADOW");
  setenv("GUFO_MTP_GATE_PROBE", path, 1);
  Expect(
      q::TrieShadowLog::FromEnvironment() == nullptr &&
          q::TrieShadowLog::FromEnvironment("GUFO_MTP_GATE_PROBE") != nullptr,
      "the gate probe's sink reads its own variable only");
  unsetenv("GUFO_MTP_GATE_PROBE");
  std::remove(path);
}

}  // namespace

int main() {
  TestFires();
  TestTracker();
  TestFormat();
  TestVariants();
  TestGateProbe();
  TestSink();
  if (failures != 0)
    return 1;
  std::cout << "trie_shadow_test: all checks passed\n";
  return 0;
}
