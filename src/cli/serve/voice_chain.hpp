// A spoken conversation made of three parts: the ASR model this process
// serves transcribes each user turn, an OpenAI-compatible chat server (main
// gufo) answers over streaming HTTP, and a TTS model in this process speaks
// the answer sentence by sentence while it streams. Each part sits behind a
// small interface so tests replace it.
#ifndef GUFO_SERVER_VOICE_CHAIN_HPP_
#define GUFO_SERVER_VOICE_CHAIN_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/realtime_audio.hpp"

namespace gufo::server {

class AsrService;
class TtsService;

struct ChainToolCall {
  std::string id, name;
  std::string arguments;  ///< JSON object
};

/// One chat message in OpenAI form: system, user, assistant or tool.
struct ChainMessage {
  std::string role;
  std::string content;
  std::vector<ChainToolCall> tool_calls;  ///< assistant
  std::string tool_call_id;               ///< tool
};

struct ChainChatRequest {
  std::vector<ChainMessage> messages;
  /// Function objects (name, description, parameters), as JSON.
  std::vector<std::string> tools;
  std::size_t max_tokens{4096};
  bool reasoning{false};
  std::optional<std::uint64_t> seed;
};

/// Called on the thread that runs ChainChatClient::Complete. A callback
/// returning false stops the stream.
struct ChainChatEvents {
  std::function<bool(std::string_view)> on_content, on_reasoning;
  std::function<bool()> cancelled;
};

struct ChainChatResult {
  std::vector<ChainToolCall> calls;
  std::string finish_reason;
  std::size_t prompt_tokens{}, completion_tokens{};
  /// Stopped by a callback or by cancelled(); not an error.
  bool cancelled{false};
};

/// Streams one chat completion.
class ChainChatClient {
public:
  virtual ~ChainChatClient() = default;
  /// False with error when the request fails; a stopped stream returns true
  /// with result->cancelled.
  virtual bool Complete(const ChainChatRequest& request,
                        const ChainChatEvents& events, ChainChatResult* result,
                        std::string* error) = 0;
};

/// Transcribes one closed user turn of mono 16 kHz audio.
class ChainTranscriber {
public:
  virtual ~ChainTranscriber() = default;
  virtual bool Transcribe(std::span<const float> pcm16k, std::string* text,
                          std::string* error) = 0;
};

/// Speaks one sentence as 24 kHz mono samples. on_audio returning false, or
/// cancelled() turning true, stops it; the caller tells a stop from a
/// failure by checking cancelled().
class ChainSynthesizer {
public:
  virtual ~ChainSynthesizer() = default;
  virtual bool Synthesize(
      std::string_view text,
      const std::function<bool(std::span<const float>)>& on_audio,
      const std::function<bool()>& cancelled, std::string* error) = 0;
};

struct ChainParts {
  std::shared_ptr<ChainChatClient> chat;
  std::shared_ptr<ChainTranscriber> transcriber;
  std::shared_ptr<ChainSynthesizer> synthesizer;
};

/// A conversation whose user audio arrives at rate Hz.
std::unique_ptr<VoiceConversation> CreateChainConversation(ChainParts parts,
                                                           std::uint32_t rate,
                                                           std::string* error);

/// Splits a server-sent event stream into data payloads; lines may arrive
/// split across any number of Feed calls.
class SseParser {
public:
  /// Calls on_data with each event's data (data lines joined by '\n');
  /// false when on_data returned false.
  bool Feed(std::string_view bytes,
            const std::function<bool(std::string_view)>& on_data);
  /// The stream sent `data: [DONE]`.
  bool done() const noexcept { return done_; }

private:
  bool Line(std::string_view line,
            const std::function<bool(std::string_view)>& on_data);
  std::string line_, data_;
  bool has_data_{false}, done_{false};
};

/// Applies one chat.completion.chunk to result, calling the events; false
/// stops the stream, with error set when the chunk was an error or invalid.
bool DecodeChainChatChunk(std::string_view data, const ChainChatEvents& events,
                          ChainChatResult* result, std::string* error);

/// The JSON body of a streaming /v1/chat/completions request.
std::string ChainChatBody(const std::string& model,
                          const ChainChatRequest& request);

/// The end of the first complete sentence of text (past its terminator), or
/// npos.
std::size_t ChainSentenceEnd(std::string_view text);

/// libcurl client of base_url + /v1/chat/completions for model.
std::shared_ptr<ChainChatClient> CreateHttpChainChatClient(
    std::string base_url, std::string model);
/// Transcribes with the ASR service of this process; service must outlive
/// the transcriber.
std::shared_ptr<ChainTranscriber> CreateAsrChainTranscriber(
    AsrService* service);
/// Speaks with a TTS service of this process as speaker, resampling to
/// 24 kHz when its rate differs.
std::shared_ptr<ChainSynthesizer> CreateTtsChainSynthesizer(
    std::shared_ptr<TtsService> service, std::string speaker,
    std::string language = "english");

}  // namespace gufo::server

#endif  // GUFO_SERVER_VOICE_CHAIN_HPP_
