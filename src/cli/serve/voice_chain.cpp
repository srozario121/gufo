#include "src/cli/serve/voice_chain.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "gufo/ext/tts.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_buffer.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {

constexpr std::uint32_t kOutputRate = 24000;
constexpr std::uint32_t kAsrRate = 16000;
// An error body kept from a failed HTTP response.
constexpr std::size_t kErrorBodyBytes = 64U << 10;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

double Milliseconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - since)
      .count();
}

// Markup the chat model writes that a voice should not read aloud.
std::string Speakable(std::string_view sentence) {
  std::string text;
  text.reserve(sentence.size());
  for (const char c : sentence)
    if (c != '*' && c != '#' && c != '`')
      text += c;
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  text = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
  const bool words = std::ranges::any_of(text, [](char c) {
    const auto byte = static_cast<unsigned char>(c);
    return byte >= 0x80 || (byte >= '0' && byte <= '9') ||
           ((byte | 0x20) >= 'a' && (byte | 0x20) <= 'z');
  });
  return words ? text : std::string();
}

// One spoken sentence of a reply: its bytes in the reply text and its
// samples in the reply audio.
struct Spoken {
  std::size_t text_end{}, sample_begin{}, sample_end{};
};

class ChainConversation final : public VoiceConversation {
public:
  ChainConversation(ChainParts parts, std::uint32_t rate)
      : parts_(std::move(parts)), rate_(rate) {}

  bool Configure(const Settings& settings, std::string* error) override {
    return ValidTools(settings.tools, error);
  }

  bool AppendAudio(std::span<const float> samples,
                   std::string* error) override {
    (void)error;
    audio_.insert(audio_.end(), samples.begin(), samples.end());
    open_ = true;
    return true;
  }

  bool CommitAudio(std::size_t* samples, std::string* error) override {
    if (samples != nullptr)
      *samples = audio_.size();
    open_ = false;
    if (audio_.empty())
      return false;
    std::vector<float> pcm;
    try {
      pcm = rate_ == kAsrRate
                ? std::move(audio_)
                : ResampleMono(AudioBuffer{.sample_rate = rate_,
                                           .channels = 1,
                                           .samples = std::move(audio_)},
                               kAsrRate);
    } catch (const std::exception& e) {
      audio_.clear();
      SetError(error, e.what());
      return false;
    }
    audio_.clear();
    std::string text;
    if (!parts_.transcriber->Transcribe(pcm, &text, error))
      return false;
    // A turn with no words is an empty turn: nothing to answer.
    if (Speakable(text).empty())
      return false;
    history_.push_back({.role = "user", .content = std::move(text)});
    return true;
  }

  bool ClearAudio(std::string* error) override {
    (void)error;
    audio_.clear();
    open_ = false;
    return true;
  }

  bool audio_open() const override { return open_; }

  bool AddUserText(std::string_view text, std::string* error) override {
    (void)error;
    history_.push_back({.role = "user", .content = std::string(text)});
    return true;
  }

  bool AddToolOutputs(std::span<const std::string> outputs,
                      std::string* error) override {
    if (outputs.size() != pending_calls_.size()) {
      SetError(error, std::to_string(outputs.size()) + " tool outputs for " +
                          std::to_string(pending_calls_.size()) +
                          " function calls");
      return false;
    }
    for (std::size_t i = 0; i < outputs.size(); ++i)
      history_.push_back({.role = "tool",
                          .content = outputs[i],
                          .tool_call_id = pending_calls_[i]});
    pending_calls_.clear();
    return true;
  }

  bool Respond(const Settings& settings, const Callbacks& callbacks,
               Reply* reply, std::string* error) override {
    if (!ValidTools(settings.tools, error))
      return false;
    *reply = {};
    // Calls the client never answered leave the context: a chat server
    // refuses tool calls without results.
    if (!pending_calls_.empty() && reply_message_) {
      auto& message = history_[*reply_message_];
      message.tool_calls.clear();
      if (message.content.empty())
        history_.erase(history_.begin() +
                       static_cast<std::ptrdiff_t>(*reply_message_));
    }
    spoken_.clear();
    reply_message_.reset();
    pending_calls_.clear();
    const auto started = std::chrono::steady_clock::now();
    ChainChatRequest request{.messages = {},
                             .tools = settings.tools,
                             .max_tokens = settings.max_output_tokens,
                             .reasoning = settings.reasoning,
                             .seed = settings.seed};
    if (!settings.instructions.empty())
      request.messages.push_back(
          {.role = "system", .content = settings.instructions});
    request.messages.insert(request.messages.end(), history_.begin(),
                            history_.end());

    bool stopped = false;
    std::string failure;
    // Text not yet spoken starts at reply->text[unspoken].
    std::size_t unspoken = 0;
    const auto cancelled = [&] {
      if (!stopped && callbacks.cancelled && callbacks.cancelled())
        stopped = true;
      return stopped;
    };
    const auto speak = [&](std::size_t end) {
      const auto text = Speakable(
          std::string_view(reply->text).substr(unspoken, end - unspoken));
      unspoken = end;
      if (text.empty())
        return true;
      Spoken sentence{.text_end = end,
                      .sample_begin = reply->audio_samples,
                      .sample_end = reply->audio_samples};
      const auto on_audio = [&](std::span<const float> samples) {
        if (cancelled())
          return false;
        if (samples.empty())
          return true;
        if (reply->first_audio_ms < 0)
          reply->first_audio_ms = Milliseconds(started);
        reply->audio_samples += samples.size();
        sentence.sample_end = reply->audio_samples;
        if (callbacks.on_audio && !callbacks.on_audio(samples))
          stopped = true;
        return !stopped;
      };
      std::string synthesis_error;
      const bool ok = parts_.synthesizer->Synthesize(text, on_audio, cancelled,
                                                     &synthesis_error);
      spoken_.push_back(sentence);
      if (!ok && !cancelled()) {
        failure = synthesis_error.empty() ? "speech synthesis failed"
                                          : synthesis_error;
        return false;
      }
      return !stopped;
    };
    ChainChatEvents events;
    events.cancelled = cancelled;
    events.on_reasoning = [&](std::string_view text) {
      if (cancelled())
        return false;
      reply->reasoning += text;
      if (callbacks.on_reasoning && !callbacks.on_reasoning(text))
        stopped = true;
      return !stopped;
    };
    events.on_content = [&](std::string_view text) {
      if (cancelled())
        return false;
      if (text.empty())
        return true;
      if (reply->first_text_ms < 0)
        reply->first_text_ms = Milliseconds(started);
      reply->text += text;
      if (callbacks.on_text && !callbacks.on_text(text)) {
        stopped = true;
        return false;
      }
      if (!settings.audio_output)
        return true;
      // The server keeps generating into the socket while a sentence is
      // spoken; the next chunk is read once it has been.
      while (true) {
        const auto end =
            ChainSentenceEnd(std::string_view(reply->text).substr(unspoken));
        if (end == std::string_view::npos)
          return true;
        if (!speak(unspoken + end))
          return false;
      }
    };

    ChainChatResult result;
    std::string chat_error;
    const bool ok =
        parts_.chat->Complete(request, events, &result, &chat_error);
    if (!failure.empty()) {
      SetError(error, std::move(failure));
      reply->total_ms = Milliseconds(started);
      return false;
    }
    if (!ok) {
      if (cancelled()) {
        reply->cancelled = true;
      } else {
        SetError(error, chat_error.empty() ? "chat completion failed"
                                           : std::move(chat_error));
        reply->total_ms = Milliseconds(started);
        return false;
      }
    }
    reply->cancelled = reply->cancelled || result.cancelled || cancelled();
    if (!reply->cancelled && settings.audio_output &&
        unspoken < reply->text.size()) {
      if (!speak(reply->text.size()) && !failure.empty()) {
        SetError(error, std::move(failure));
        reply->total_ms = Milliseconds(started);
        return false;
      }
      reply->cancelled = cancelled();
    }
    reply->input_tokens = result.prompt_tokens;
    reply->output_tokens = result.completion_tokens;
    reply->incomplete = result.finish_reason == "length";

    ChainMessage message{.role = "assistant", .content = reply->text};
    // An interrupted spoken reply keeps the sentences that were spoken.
    if (reply->cancelled && settings.audio_output)
      message.content.resize(spoken_.empty() ? 0 : spoken_.back().text_end);
    // Calls of a cancelled reply are not made.
    if (!reply->cancelled) {
      for (std::size_t i = 0; i < result.calls.size(); ++i) {
        auto call = result.calls[i];
        if (call.id.empty())
          call.id = "call_" + std::to_string(++calls_);
        if (call.arguments.empty())
          call.arguments = "{}";
        reply->calls.push_back(
            {.name = call.name, .arguments = call.arguments});
        pending_calls_.push_back(call.id);
        message.tool_calls.push_back(std::move(call));
      }
    }
    if (!message.content.empty() || !message.tool_calls.empty()) {
      reply_message_ = history_.size();
      history_.push_back(std::move(message));
    }
    reply->total_ms = Milliseconds(started);
    return true;
  }

  // The user heard every sentence that started before samples: the one cut
  // off stays whole, since its words cannot be aligned to the audio.
  bool TruncateReply(std::size_t samples, std::string* error) override {
    (void)error;
    if (!reply_message_ || spoken_.empty())
      return true;
    const bool heard_all = samples >= spoken_.back().sample_end;
    if (heard_all)
      return true;
    std::size_t keep = 0;
    for (const auto& sentence : spoken_)
      if (sentence.sample_begin < samples)
        keep = sentence.text_end;
    auto& message = history_[*reply_message_];
    message.content.resize(std::min(keep, message.content.size()));
    const auto first = message.content.find_first_not_of(" \t\r\n");
    message.content =
        first == std::string::npos
            ? std::string()
            : message.content.substr(
                  0, message.content.find_last_not_of(" \t\r\n") + 1);
    if (message.content.empty() && message.tool_calls.empty()) {
      history_.erase(history_.begin() +
                     static_cast<std::ptrdiff_t>(*reply_message_));
      reply_message_.reset();
    }
    spoken_.clear();
    return true;
  }

private:
  static bool ValidTools(const std::vector<std::string>& tools,
                         std::string* error) {
    for (const auto& tool : tools) {
      try {
        const auto value = json::parse(tool);
        if (value.is_object() && !value.member_str("name").empty())
          continue;
      } catch (const std::exception&) {
      }
      SetError(error, "tools must be named function objects");
      return false;
    }
    return true;
  }

  ChainParts parts_;
  std::uint32_t rate_;
  std::vector<float> audio_;
  bool open_{false};
  std::vector<ChainMessage> history_;
  // The last reply: its message in history_, its spoken sentences and the
  // IDs of its calls awaiting outputs.
  std::optional<std::size_t> reply_message_;
  std::vector<Spoken> spoken_;
  std::vector<std::string> pending_calls_;
  std::size_t calls_{0};
};

// ---------------------------------------------------------------------------
// HTTP chat client.

class HttpChatClient final : public ChainChatClient {
public:
  HttpChatClient(std::string base_url, std::string model)
      : url_(Endpoint(std::move(base_url))), model_(std::move(model)) {
    static std::once_flag initialized;
    std::call_once(initialized,
                   [] { (void)curl_global_init(CURL_GLOBAL_DEFAULT); });
  }

  bool Complete(const ChainChatRequest& request, const ChainChatEvents& events,
                ChainChatResult* result, std::string* error) override {
    *result = {};
    const std::string body = ChainChatBody(model_, request);
    struct State {
      CURL* curl{};
      SseParser sse;
      const ChainChatEvents* events{};
      ChainChatResult* result{};
      std::string error, error_body;
      bool stopped{false};
    } state;
    const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
        curl_easy_init(), curl_easy_cleanup);
    if (!curl) {
      SetError(error, "cannot initialize libcurl");
      return false;
    }
    state.curl = curl.get();
    state.events = &events;
    state.result = result;
    curl_slist* list = nullptr;
    list = curl_slist_append(list, "Content-Type: application/json");
    list = curl_slist_append(list, "Accept: text/event-stream");
    const std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
        list, curl_slist_free_all);
    std::array<char, CURL_ERROR_SIZE> curl_error{};
    using Write = std::size_t (*)(char*, std::size_t, std::size_t, void*);
    const Write write = [](char* data, std::size_t size, std::size_t count,
                           void* user) -> std::size_t {
      auto& s = *static_cast<State*>(user);
      const std::size_t bytes = size * count;
      long status = 0;
      curl_easy_getinfo(s.curl, CURLINFO_RESPONSE_CODE, &status);
      if (status >= 400) {
        const std::size_t room =
            kErrorBodyBytes - std::min(kErrorBodyBytes, s.error_body.size());
        s.error_body.append(data, std::min(bytes, room));
        return bytes;
      }
      const bool more = s.sse.Feed(
          std::string_view(data, bytes), [&](std::string_view payload) {
            return DecodeChainChatChunk(payload, *s.events, s.result, &s.error);
          });
      if (more)
        return bytes;
      s.stopped = true;
      return 0;
    };
    using Progress =
        int (*)(void*, curl_off_t, curl_off_t, curl_off_t, curl_off_t);
    const Progress progress = [](void* user, curl_off_t, curl_off_t, curl_off_t,
                                 curl_off_t) -> int {
      auto& s = *static_cast<State*>(user);
      if (s.events->cancelled && s.events->cancelled()) {
        s.stopped = true;
        return 1;
      }
      return 0;
    };
    CURL* handle = curl.get();
    curl_easy_setopt(handle, CURLOPT_URL, url_.c_str());
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(body.size()));
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 10L);
    // A long prompt may take minutes to prefill; only a silent server for
    // ten minutes is dead.
    curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(handle, CURLOPT_LOW_SPEED_TIME, 600L);
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(handle, CURLOPT_USERAGENT, "gufo-voice-chain/1");
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &state);
    curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, progress);
    curl_easy_setopt(handle, CURLOPT_XFERINFODATA, &state);
    curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, curl_error.data());
    const CURLcode code = curl_easy_perform(handle);
    if (!state.error.empty()) {
      SetError(error, "chat completion failed: " + state.error);
      return false;
    }
    if (state.stopped &&
        (code == CURLE_WRITE_ERROR || code == CURLE_ABORTED_BY_CALLBACK ||
         code == CURLE_OK)) {
      result->cancelled = true;
      return true;
    }
    if (code != CURLE_OK) {
      const std::string cause = curl_error[0] != '\0'
                                    ? std::string(curl_error.data())
                                    : std::string(curl_easy_strerror(code));
      SetError(error,
               "chat completion request to " + url_ + " failed: " + cause);
      return false;
    }
    long status = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
    if (status >= 400) {
      std::string message;
      try {
        const auto value = json::parse(state.error_body);
        if (const auto* detail = value.find("error"))
          message = detail->is_string() ? detail->str()
                                        : detail->member_str("message");
      } catch (const std::exception&) {
      }
      if (message.empty())
        message = state.error_body.substr(0, 200);
      SetError(error, "chat server returned HTTP " + std::to_string(status) +
                          (message.empty() ? std::string() : ": " + message));
      return false;
    }
    if (!state.sse.done() && result->finish_reason.empty()) {
      SetError(error, "chat completion stream ended early");
      return false;
    }
    return true;
  }

private:
  static std::string Endpoint(std::string base) {
    while (!base.empty() && base.back() == '/')
      base.pop_back();
    if (base.ends_with("/v1"))
      return base + "/chat/completions";
    return base + "/v1/chat/completions";
  }
  std::string url_, model_;
};

// ---------------------------------------------------------------------------
// In-process models.

class AsrTranscriber final : public ChainTranscriber {
public:
  explicit AsrTranscriber(AsrService* service) : service_(service) {}
  bool Transcribe(std::span<const float> pcm16k, std::string* text,
                  std::string* error) override {
    ext::TranscriptionRequest request{.wav = {},
                                      .context = {},
                                      .language = std::nullopt,
                                      .max_new_tokens = 256,
                                      .pcm16k = pcm16k,
                                      .on_text = {}};
    ext::TranscriptionResult result;
    if (!service_->Transcribe(request, [] { return false; }, &result, error))
      return false;
    *text = std::move(result.text);
    return true;
  }

private:
  AsrService* service_;
};

class TtsSynthesizer final : public ChainSynthesizer {
public:
  TtsSynthesizer(std::shared_ptr<TtsService> service, std::string speaker,
                 std::string language)
      : service_(std::move(service)),
        speaker_(std::move(speaker)),
        language_(std::move(language)) {}

  bool Synthesize(std::string_view text,
                  const std::function<bool(std::span<const float>)>& on_audio,
                  const std::function<bool()>& cancelled,
                  std::string* error) override {
    const std::uint32_t rate = service_->sample_rate();
    // Another rate is resampled once per sentence, so chunk edges do not
    // click.
    std::vector<float> buffered;
    const ext::CancellationCheck check = [&] {
      return cancelled && cancelled();
    };
    ext::SynthesisRequest request{
        .text = std::string(text), .speaker = speaker_, .language = language_};
    request.on_audio = [&](std::span<const float> samples) {
      if (rate == kOutputRate)
        return on_audio(samples);
      buffered.insert(buffered.end(), samples.begin(), samples.end());
      return !check();
    };
    ext::SynthesisResult result;
    if (!service_->Synthesize(request, check, &result, error))
      return false;
    if (rate == kOutputRate || buffered.empty())
      return true;
    try {
      const auto resampled =
          ResampleMono(AudioBuffer{.sample_rate = rate,
                                   .channels = 1,
                                   .samples = std::move(buffered)},
                       kOutputRate);
      (void)on_audio(resampled);
    } catch (const std::exception& e) {
      SetError(error, e.what());
      return false;
    }
    return true;
  }

private:
  std::shared_ptr<TtsService> service_;
  std::string speaker_, language_;
};

}  // namespace

std::unique_ptr<VoiceConversation> CreateChainConversation(ChainParts parts,
                                                           std::uint32_t rate,
                                                           std::string* error) {
  if (!parts.chat || !parts.transcriber || !parts.synthesizer) {
    SetError(error, "the voice chain is not configured");
    return nullptr;
  }
  if (rate == 0) {
    SetError(error, "invalid input sample rate");
    return nullptr;
  }
  return std::make_unique<ChainConversation>(std::move(parts), rate);
}

bool SseParser::Feed(std::string_view bytes,
                     const std::function<bool(std::string_view)>& on_data) {
  while (!bytes.empty()) {
    const auto newline = bytes.find('\n');
    if (newline == std::string_view::npos) {
      line_.append(bytes);
      return true;
    }
    line_.append(bytes.substr(0, newline));
    bytes.remove_prefix(newline + 1);
    if (!line_.empty() && line_.back() == '\r')
      line_.pop_back();
    const std::string line = std::move(line_);
    line_.clear();
    if (!Line(line, on_data))
      return false;
  }
  return true;
}

bool SseParser::Line(std::string_view line,
                     const std::function<bool(std::string_view)>& on_data) {
  if (line.empty()) {
    if (!has_data_)
      return true;
    const std::string data = std::move(data_);
    data_.clear();
    has_data_ = false;
    if (data == "[DONE]") {
      done_ = true;
      return true;
    }
    return on_data(data);
  }
  if (line.front() == ':')
    return true;
  const auto colon = line.find(':');
  const auto field = line.substr(0, colon);
  if (field != "data")
    return true;
  auto value = colon == std::string_view::npos ? std::string_view()
                                               : line.substr(colon + 1);
  if (value.starts_with(' '))
    value.remove_prefix(1);
  if (has_data_)
    data_ += '\n';
  data_.append(value);
  has_data_ = true;
  return true;
}

bool DecodeChainChatChunk(std::string_view data, const ChainChatEvents& events,
                          ChainChatResult* result, std::string* error) {
  json::Value chunk;
  try {
    chunk = json::parse(data);
  } catch (const std::exception& e) {
    SetError(error, std::string("invalid stream chunk: ") + e.what());
    return false;
  }
  if (!chunk.is_object()) {
    SetError(error, "invalid stream chunk");
    return false;
  }
  if (const auto* detail = chunk.find("error")) {
    std::string message =
        detail->is_string() ? detail->str() : detail->member_str("message");
    SetError(error, message.empty() ? "the chat server reported an error"
                                    : std::move(message));
    return false;
  }
  if (const auto* usage = chunk.find("usage"); usage && usage->is_object()) {
    result->prompt_tokens = usage->member_size("prompt_tokens");
    result->completion_tokens = usage->member_size("completion_tokens");
  }
  const auto* choices = chunk.find("choices");
  if (!choices || !choices->is_array() || choices->items().empty())
    return true;
  const auto& choice = choices->items().front();
  if (const auto* finish = choice.find("finish_reason");
      finish && finish->is_string())
    result->finish_reason = finish->str();
  const auto* delta = choice.find("delta");
  if (!delta || !delta->is_object())
    return true;
  for (const char* field : {"reasoning_content", "reasoning"}) {
    const auto* reasoning = delta->find(field);
    if (reasoning && reasoning->is_string() && !reasoning->str().empty() &&
        events.on_reasoning && !events.on_reasoning(reasoning->str()))
      return false;
  }
  if (const auto* calls = delta->find("tool_calls");
      calls && calls->is_array()) {
    for (const auto& call : calls->items()) {
      const std::size_t index = call.member_size("index", result->calls.size());
      if (index > 64) {
        SetError(error, "invalid tool call index");
        return false;
      }
      if (result->calls.size() <= index)
        result->calls.resize(index + 1);
      auto& target = result->calls[index];
      if (const auto id = call.member_str("id"); !id.empty())
        target.id = id;
      if (const auto* function = call.find("function");
          function && function->is_object()) {
        if (const auto name = function->member_str("name"); !name.empty())
          target.name = name;
        target.arguments += function->member_str("arguments");
      }
    }
  }
  if (const auto* content = delta->find("content");
      content && content->is_string() && !content->str().empty() &&
      events.on_content && !events.on_content(content->str()))
    return false;
  return true;
}

std::string ChainChatBody(const std::string& model,
                          const ChainChatRequest& request) {
  auto body = json::Value::object();
  body["model"] = model;
  body["stream"] = true;
  body["stream_options"] = json::Value::object();
  body["stream_options"]["include_usage"] = true;
  body["max_tokens"] = request.max_tokens;
  auto messages = json::Value::array();
  for (const auto& message : request.messages) {
    auto entry = json::Value::object();
    entry["role"] = message.role;
    if (message.role == "assistant" && message.content.empty() &&
        !message.tool_calls.empty())
      entry["content"] = nullptr;
    else
      entry["content"] = message.content;
    if (!message.tool_calls.empty()) {
      auto calls = json::Value::array();
      for (const auto& call : message.tool_calls) {
        auto item = json::Value::object();
        item["id"] = call.id;
        item["type"] = "function";
        item["function"] = json::Value::object();
        item["function"]["name"] = call.name;
        item["function"]["arguments"] = call.arguments;
        calls.push_back(std::move(item));
      }
      entry["tool_calls"] = std::move(calls);
    }
    if (message.role == "tool")
      entry["tool_call_id"] = message.tool_call_id;
    messages.push_back(std::move(entry));
  }
  body["messages"] = std::move(messages);
  if (!request.tools.empty()) {
    auto tools = json::Value::array();
    for (const auto& tool : request.tools) {
      auto item = json::Value::object();
      item["type"] = "function";
      item["function"] = json::parse(tool);
      tools.push_back(std::move(item));
    }
    body["tools"] = std::move(tools);
  }
  body["chat_template_kwargs"] = json::Value::object();
  body["chat_template_kwargs"]["enable_thinking"] = request.reasoning;
  if (request.seed)
    body["seed"] = static_cast<unsigned long long>(*request.seed);
  return body.dump();
}

std::size_t ChainSentenceEnd(std::string_view text) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\n')
      return i + 1;
    if ((c == '.' || c == '!' || c == '?') && i + 1 < text.size() &&
        (text[i + 1] == ' ' || text[i + 1] == '\n' || text[i + 1] == '\t' ||
         text[i + 1] == '\r'))
      return i + 1;
    // Full-width 。！？ end a sentence without a space.
    if (i + 2 < text.size()) {
      const auto a = static_cast<unsigned char>(text[i]);
      const auto b = static_cast<unsigned char>(text[i + 1]);
      const auto d = static_cast<unsigned char>(text[i + 2]);
      if ((a == 0xE3 && b == 0x80 && d == 0x82) ||
          (a == 0xEF && b == 0xBC && (d == 0x81 || d == 0x9F)))
        return i + 3;
    }
  }
  return std::string_view::npos;
}

std::shared_ptr<ChainChatClient> CreateHttpChainChatClient(std::string base_url,
                                                           std::string model) {
  return std::make_shared<HttpChatClient>(std::move(base_url),
                                          std::move(model));
}

std::shared_ptr<ChainTranscriber> CreateAsrChainTranscriber(
    AsrService* service) {
  return std::make_shared<AsrTranscriber>(service);
}

std::shared_ptr<ChainSynthesizer> CreateTtsChainSynthesizer(
    std::shared_ptr<TtsService> service, std::string speaker,
    std::string language) {
  return std::make_shared<TtsSynthesizer>(
      std::move(service), std::move(speaker), std::move(language));
}

}  // namespace gufo::server
