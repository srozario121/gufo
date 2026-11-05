// DFlash-2 drafting on Flash-Next sessions: seeded replay is exact across
// fresh and reset sessions (reset keeps verify scratch and graphs), drafts are
// accepted, the committed state after rejected drafts matches a prefill of
// the same tokens, and a continuation-cache restore into a slot that ran
// another request replays exactly as the uninterrupted session.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/transfer_pool.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen38_flash_next/mtp_replay.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;
namespace sampling = gufo::sampling;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// The published DFlash-2 drafter's sliding window (its config), which the
// continuation fixtures below cross.
constexpr std::uint32_t kDflashWindow = 2048;

std::vector<std::int32_t> Prompt(const qfn::Model& model,
                                 const std::string& text) {
  namespace qt = gufo::tokenization;
  std::string error;
  const std::array messages{
      qt::ChatMessage{qt::ChatRole::kSystem,
                      "Follow the instruction and answer accurately."},
      qt::ChatMessage{qt::ChatRole::kUser, text}};
  const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
      model.tokenizer(), messages, {.enable_thinking = false}, &error);
  Require(encoded.has_value(), error);
  return {encoded->begin(), encoded->end()};
}

// Replays one configuration on a fresh session, then twice more on the same
// session after Reset (which keeps verify scratch and graphs): tokens, step
// widths, acceptance and RNG must not move.
void CheckReplay(const std::shared_ptr<qfn::Model>& model,
                 const std::vector<std::int32_t>& prompt,
                 const sampling::SamplingConfig& config, const char* name) {
  std::string error;
  const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  gufo::test::MtpReplay expected{};
  auto session =
      model->CreateSession(gufo::core::SessionMode::kSpeculative, 4096, &error);
  Require(session != nullptr, error);
  for (unsigned round = 0; round < 3; ++round) {
    if (round != 0)
      session->Reset();
    Require(session->Sync(prompt, &error), error);
    session->ResetDraftPolicy();
    sampling::SamplerState sampler(config, history);
    const auto replay = gufo::test::DecodeMtpReplay(*session, sampler, 200);
    Require(replay.drafted > 0 && replay.accepted > 0,
            std::string(name) + ": DFlash-2 drafted or accepted nothing");
    if (round == 0)
      expected = replay;
    else
      Require(replay == expected,
              std::string(name) +
                  ": DFlash-2 replay after Reset changed tokens, widths, "
                  "acceptance or RNG (round " +
                  std::to_string(round) + ")");
  }
  std::cout << "DFlash-2 replay " << name
            << " tokens=200 drafted=" << expected.drafted
            << " accepted=" << expected.accepted << " reset_replays_exact=2\n"
            << std::flush;
}

// After sampled decoding with rejected drafts, the session's next-token
// logits must agree with a fresh prefill of the same committed tokens. The
// verify and prefill shapes round differently, so this compares the argmax
// and a relative error bound; a wrong recurrent rollback after a partial
// accept moves the logits far beyond it.
// Decodes 160 sampled tokens in `mode`, then compares the session's
// next-token logits with a fresh prefill of the same committed tokens.
struct StateError {
  double relative;
  bool argmax_equal;
  std::uint64_t rejected;
};
StateError CommittedStateError(const std::shared_ptr<qfn::Model>& model,
                               const std::vector<std::int32_t>& prompt,
                               gufo::core::SessionMode mode) {
  std::string error;
  auto session = model->CreateSession(mode, 4096, &error);
  Require(session && session->Sync(prompt, &error), error);
  const sampling::SamplingConfig config{
      .temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .seed = 7};
  sampling::SamplerState sampler(
      config, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  const auto replay = gufo::test::DecodeMtpReplay(*session, sampler, 160);
  // Every committed token was verified; Logits() holds the next row.
  const std::vector<std::int32_t> committed(session->Tokens().begin(),
                                            session->Tokens().end());
  Require(committed.size() == prompt.size() + replay.tokens.size(),
          "committed tokens differ from the decoded ones");
  auto reference = model->CreateSession(
      gufo::core::SessionMode::kAutoregressive, 4096, &error);
  Require(reference && reference->Sync(committed, &error), error);
  const auto a = session->Logits(), b = reference->Logits();
  Require(a.size() == b.size() && !a.empty(), "logit rows differ in size");
  double diff = 0, norm = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
    norm += double(b[i]) * b[i];
  }
  const auto argmax = [](std::span<const float> x) {
    return std::ranges::max_element(x) - x.begin();
  };
  return {std::sqrt(diff / std::max(norm, 1e-30)), argmax(a) == argmax(b),
          replay.drafted - replay.accepted};
}

// A wrong recurrent rollback after a partial accept moves the logits far
// from a prefill of the committed tokens. Decode and prefill shapes round
// differently, so the bound is relative to autoregressive decoding's own
// distance from the same prefill.
void CheckCommittedState(const std::shared_ptr<qfn::Model>& model,
                         const std::vector<std::int32_t>& prompt) {
  const auto ar = CommittedStateError(model, prompt,
                                      gufo::core::SessionMode::kAutoregressive);
  const auto dflash =
      CommittedStateError(model, prompt, gufo::core::SessionMode::kSpeculative);
  std::cout << "committed state vs prefill: AR relative_error=" << ar.relative
            << " argmax_equal=" << ar.argmax_equal
            << " | DFlash-2 rejected=" << dflash.rejected
            << " relative_error=" << dflash.relative
            << " argmax_equal=" << dflash.argmax_equal << '\n';
  Require(dflash.rejected > 0, "committed-state check saw no rejected draft");
  Require(dflash.argmax_equal && dflash.relative < 2 * ar.relative + 1e-3,
          "DFlash-2 rollback left a state further from a prefill than "
          "autoregressive decoding");
}

// Component checkpoints as the cache holds them: append rows up to each
// frontier and every private component in full.
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
struct NoBorrowers final : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
};
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}
std::unique_ptr<gufo::hip::TransferStream> Acquire(
    gufo::hip::TransferPool& streams) {
  auto stream = streams.TryAcquire();
  Require(stream != nullptr, "transfer stream exhausted");
  return stream;
}
Checkpoint Capture(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams) {
  Checkpoint result{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    const auto stream = Acquire(streams);
    Wait(component.row_bytes
             ? adapter.CopyRowsOut(slot, component.id, 0, rows, bytes, *stream)
             : adapter.CapturePrivate(slot, component.id, bytes, *stream));
  }
  return result;
}
// Reverse component order: the metadata, which holds the drafter's injected
// length, arrives after the drafter's K/V and target rows.
void Load(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams) {
  adapter.BeginRestore(slot, checkpoint.positions);
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const std::span<const std::byte> bytes(checkpoint.bytes[i]);
    const auto stream = Acquire(streams);
    Wait(component.row_bytes
             ? adapter.CopyRowsIn(slot, component.id, 0,
                                  checkpoint.positions[i].valid_rows, bytes,
                                  *stream)
             : adapter.LoadPrivate(slot, component.id, bytes, *stream));
  }
  Require(adapter.Validate(slot, checkpoint.positions),
          "DFlash-2 component restore rejected");
}
void SameLogits(const qfn::Session& a, const qfn::Session& b,
                const std::string& name) {
  Require(a.Logits().size() == b.Logits().size() &&
              std::memcmp(a.Logits().data(), b.Logits().data(),
                          a.Logits().size_bytes()) == 0,
          name + ": restored next-step logits differ");
}
std::vector<std::int32_t> Repeated(const std::vector<std::int32_t>& head,
                                   const std::vector<std::int32_t>& pattern,
                                   std::size_t size) {
  std::vector<std::int32_t> tokens = head;
  while (tokens.size() < size)
    tokens.push_back(pattern[(tokens.size() - head.size()) % pattern.size()]);
  return tokens;
}

// The drafter's state at a checkpoint is its injected K/V window and the
// target rows it has not injected yet. Restoring them into a slot that ran
// another request must reproduce the uninterrupted session's next blocks:
// tokens, step widths, acceptance and RNG. `decoded` sampled tokens precede
// the checkpoint (zero: the drafter has injected nothing; a prompt longer
// than the window then needs a full window of target rows).
void CheckContinuation(const std::shared_ptr<qfn::Model>& model,
                       const std::vector<std::int32_t>& prompt,
                       const std::vector<std::int32_t>& other,
                       std::size_t decoded, const char* name) {
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(model, gufo::core::SessionMode::kSpeculative,
                                   4096, {'d', 'f', 'l', 'a', 's', 'h'});
  NoBorrowers source_guard, destination_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto destination = adapter.CreateSlot(destination_guard);
  auto& live = adapter.GetSession(*source);
  auto& restored = adapter.GetSession(*destination);
  std::string error;
  const sampling::SamplingConfig config{
      .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 11};
  // The destination's drafter and capture ring first hold another history.
  Require(restored.Sync(other, &error), error);
  sampling::SamplerState stale(
      config, std::vector<sampling::TokenId>(other.begin(), other.end()));
  (void)gufo::test::DecodeMtpReplay(restored, stale, 48);
  Require(live.Sync(prompt, &error), error);
  live.ResetDraftPolicy();
  sampling::SamplerState sampler(
      config, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  if (decoded != 0)
    Require(gufo::test::DecodeMtpReplay(live, sampler, decoded).drafted > 0,
            std::string(name) + ": no DFlash-2 draft before the checkpoint");
  const auto checkpoint = Capture(adapter, *source, streams);
  Load(adapter, *destination, checkpoint, streams);
  SameLogits(live, restored, name);
  auto replay_sampler = sampler;
  const auto expected = gufo::test::DecodeMtpReplay(live, sampler, 96);
  const auto replay = gufo::test::DecodeMtpReplay(restored, replay_sampler, 96);
  Require(expected.drafted > 0 && expected.accepted > 0,
          std::string(name) + ": continuation drafted or accepted nothing");
  Require(replay == expected,
          std::string(name) +
              ": DFlash-2 restore changed tokens, widths, acceptance or RNG");
  std::cout << "DFlash-2 continuation " << name
            << " position=" << checkpoint.positions.front().valid_rows
            << " tokens=96 drafted=" << expected.drafted
            << " accepted=" << expected.accepted << " exact=1\n"
            << std::flush;
}

// An in-pass boundary freezes the frontier while the pass's tail rows still
// write the capture ring. With the drafter's injected length a window behind
// the boundary, the checkpoint needs the full window of target rows below it.
// The restored slot must then decode exactly as a session prefilled to the
// boundary.
void CheckInPassContinuation(const std::shared_ptr<qfn::Model>& model,
                             const std::vector<std::int32_t>& prompt,
                             const std::vector<std::int32_t>& pattern,
                             const std::vector<std::int32_t>& other) {
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(model, gufo::core::SessionMode::kSpeculative,
                                   4096, {'i', 'n', 'p', 'a', 's', 's'});
  NoBorrowers source_guard, destination_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto destination = adapter.CreateSlot(destination_guard);
  auto& live = adapter.GetSession(*source);
  auto& restored = adapter.GetSession(*destination);
  std::string error;
  auto reference =
      model->CreateSession(gufo::core::SessionMode::kSpeculative, 4096, &error);
  Require(reference != nullptr, error);
  const sampling::SamplingConfig config{
      .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 13};
  Require(restored.Sync(other, &error), error);
  sampling::SamplerState stale(
      config, std::vector<sampling::TokenId>(other.begin(), other.end()));
  (void)gufo::test::DecodeMtpReplay(restored, stale, 48);
  // Both sessions inject the same drafted history before the extension.
  for (auto* session : {&live, reference.get()}) {
    Require(session->Sync(prompt, &error), error);
    session->ResetDraftPolicy();
    sampling::SamplerState first(
        config, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
    Require(gufo::test::DecodeMtpReplay(*session, first, 32).drafted > 0,
            "in-pass: no DFlash-2 draft before the extension");
  }
  const std::vector<std::int32_t> committed(live.Tokens().begin(),
                                            live.Tokens().end());
  Require(std::ranges::equal(committed, reference->Tokens()),
          "in-pass: the seeded histories differ");
  const auto boundary =
      static_cast<std::uint32_t>(committed.size() + kDflashWindow + 160);
  Require(boundary + 5 <= 4096, "in-pass fixture exceeds the context");
  const auto extended = Repeated(committed, pattern, boundary + 5);
  std::optional<Checkpoint> frozen;
  adapter.PrefillThrough(*source, extended, boundary,
                         [&] { frozen = Capture(adapter, *source, streams); });
  Require(frozen.has_value() && live.Position() == boundary + 5,
          "in-pass capture did not run");
  Require(reference->Sync(std::span(extended).first(boundary), &error), error);
  Load(adapter, *destination, *frozen, streams);
  SameLogits(*reference, restored, "in-pass");
  restored.ResetDraftPolicy();
  reference->ResetDraftPolicy();
  sampling::SamplerState sampler(
      config, std::vector<sampling::TokenId>(extended.begin(),
                                             extended.begin() + boundary));
  auto replay_sampler = sampler;
  const auto expected = gufo::test::DecodeMtpReplay(*reference, sampler, 96);
  const auto replay = gufo::test::DecodeMtpReplay(restored, replay_sampler, 96);
  Require(expected.drafted > 0,
          "in-pass: continuation drafted nothing after the boundary");
  Require(replay == expected,
          "in-pass: DFlash-2 restore changed tokens, widths, acceptance or "
          "RNG");
  std::cout << "DFlash-2 continuation in_pass boundary=" << boundary
            << " tokens=96 drafted=" << expected.drafted
            << " accepted=" << expected.accepted << " exact=1\n"
            << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--dflash-model") {
    std::cerr << "Usage: dflash_session_test --model FIRST.gguf "
                 "--dflash-model DFLASH.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2], {.max_context = 4096, .dflash_model_path = argv[4]}, &error);
    Require(model != nullptr, error);
    Require(model->HasDflash() && !model->HasMtp(),
            "expected a DFlash-2 model");
    const auto prompt = Prompt(
        *model,
        "Write a Python function that parses a CSV line with quoted fields, "
        "then explain it step by step.");
    CheckReplay(model, prompt, {.temperature = 0.0F, .seed = 1}, "greedy");
    CheckReplay(model, prompt,
                {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .seed = 3},
                "sampled");
    CheckCommittedState(model, prompt);
    const auto other = Prompt(
        *model, "List five prime numbers and explain why each one is prime.");
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. Strix Halo executes "
        "this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    CheckContinuation(model, prompt, other, 48, "short");
    // Past the drafter's window: its K/V ring wraps, and before any draft
    // the checkpoint holds a full window of target rows.
    std::string text;
    for (unsigned i = 0; i < 160; ++i)
      text += "Line " + std::to_string(i) +
              ": the quick brown fox jumps over the lazy dog. ";
    const auto long_prompt = Prompt(
        *model, text +
                    "\nCount the lines above, then write a Python function "
                    "that counts the words of each line.");
    Require(long_prompt.size() > kDflashWindow + 100 &&
                long_prompt.size() + 200 < 4096,
            "long continuation fixture length " +
                std::to_string(long_prompt.size()));
    CheckContinuation(model, long_prompt, other, 0, "long_prefill");
    CheckContinuation(model, long_prompt, other, 48, "long_decoded");
    CheckInPassContinuation(model, prompt, pattern, other);
    std::cout << "dflash_session_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dflash_session_test: " << e.what() << '\n';
    return 1;
  }
}
