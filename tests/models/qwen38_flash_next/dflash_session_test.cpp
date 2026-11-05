// DFlash-2 drafting on Flash-Next sessions: seeded replay is exact across
// fresh and reset sessions (reset keeps verify scratch and graphs), drafts are
// accepted, and the committed state after rejected drafts matches a prefill of
// the same tokens.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen38_flash_next/mtp_replay.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

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
    std::cout << "dflash_session_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dflash_session_test: " << e.what() << '\n';
    return 1;
  }
}
