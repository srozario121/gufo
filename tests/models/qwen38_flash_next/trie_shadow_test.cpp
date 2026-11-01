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
             "\"emitted\":42,\"match\":true}\n",
         "the JSON line: " + line);
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
  std::remove(path);
}

}  // namespace

int main() {
  TestFires();
  TestTracker();
  TestFormat();
  TestSink();
  if (failures != 0)
    return 1;
  std::cout << "trie_shadow_test: all checks passed\n";
  return 0;
}
