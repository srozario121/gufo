// The voice chain with fake models: transcription, the chat stream (through
// the real SSE parser and chunk decoder), speech by sentence, tool turns,
// cancellation and truncation; plus the libcurl client against a canned
// local HTTP server.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/voice_chain.hpp"
#include "src/core/json.hpp"

namespace {
using namespace gufo::server;
namespace json = gufo::json;

void Check(bool condition, const char* what, int line) {
  if (!condition) {
    std::fprintf(stderr, "voice_chain_test:%d: check failed: %s\n", line, what);
    std::abort();
  }
}
#define CHECK(condition) \
  Check(static_cast<bool>(condition), #condition, __LINE__)

// SSE for one chat reply, split into network-sized pieces.
std::string Chunk(const json::Value& delta, const char* finish = nullptr) {
  auto chunk = json::Value::object();
  chunk["object"] = "chat.completion.chunk";
  auto choice = json::Value::object();
  choice["index"] = 0;
  choice["delta"] = delta;
  choice["finish_reason"] = finish ? json::Value(finish) : json::Value();
  chunk["choices"] = json::Value::array();
  chunk["choices"].push_back(std::move(choice));
  return "data: " + chunk.dump() + "\n\n";
}
std::string Content(const std::string& text) {
  auto delta = json::Value::object();
  delta["content"] = text;
  return Chunk(delta);
}
std::string Reasoning(const std::string& text) {
  auto delta = json::Value::object();
  delta["reasoning_content"] = text;
  return Chunk(delta);
}
std::string Finish(const char* reason = "stop") {
  return Chunk(json::Value::object(), reason) +
         "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,"
         "\"completion_tokens\":7}}\n\ndata: [DONE]\n\n";
}

// Plays scripted SSE replies through SseParser and DecodeChainChatChunk,
// feeding a few bytes at a time.
class FakeChat final : public ChainChatClient {
public:
  std::vector<std::string> replies;
  std::vector<ChainChatRequest> requests;
  std::string failure;
  // Pieces of the current reply delivered before the latest callback.
  std::size_t delivered{0};
  bool finished_stream{false};

  bool Complete(const ChainChatRequest& request, const ChainChatEvents& events,
                ChainChatResult* result, std::string* error) override {
    requests.push_back(request);
    *result = {};
    finished_stream = false;
    if (!failure.empty()) {
      *error = failure;
      return false;
    }
    CHECK(!replies.empty());
    const std::string sse = replies.front();
    replies.erase(replies.begin());
    SseParser parser;
    std::string decode_error;
    delivered = 0;
    for (std::size_t at = 0; at < sse.size(); at += 7) {
      if (events.cancelled && events.cancelled()) {
        result->cancelled = true;
        return true;
      }
      ++delivered;
      if (!parser.Feed(std::string_view(sse).substr(at, 7),
                       [&](std::string_view data) {
                         return DecodeChainChatChunk(data, events, result,
                                                     &decode_error);
                       })) {
        if (!decode_error.empty()) {
          *error = decode_error;
          return false;
        }
        result->cancelled = true;
        return true;
      }
    }
    CHECK(parser.done());
    finished_stream = true;
    return true;
  }
};

class FakeTranscriber final : public ChainTranscriber {
public:
  std::string text{"what is the weather"};
  std::size_t samples{0};
  bool Transcribe(std::span<const float> pcm16k, std::string* out,
                  std::string*) override {
    samples = pcm16k.size();
    *out = text;
    return true;
  }
};

// 100 samples per byte of text, in two chunks.
class FakeSynth final : public ChainSynthesizer {
public:
  std::vector<std::string> texts;
  std::vector<std::size_t> chat_progress;
  const FakeChat* chat{nullptr};
  bool Synthesize(std::string_view text,
                  const std::function<bool(std::span<const float>)>& on_audio,
                  const std::function<bool()>& cancelled,
                  std::string*) override {
    texts.emplace_back(text);
    chat_progress.push_back(chat ? chat->delivered : 0);
    const std::vector<float> samples(text.size() * 50, 0.25F);
    for (int half = 0; half < 2; ++half) {
      if (cancelled && cancelled())
        return false;
      if (!on_audio(samples))
        return false;
    }
    return true;
  }
};

struct Harness {
  std::shared_ptr<FakeChat> chat = std::make_shared<FakeChat>();
  std::shared_ptr<FakeTranscriber> transcriber =
      std::make_shared<FakeTranscriber>();
  std::shared_ptr<FakeSynth> synth = std::make_shared<FakeSynth>();
  std::unique_ptr<VoiceConversation> conversation;
  std::string text, reasoning;
  std::size_t audio{0};
  Harness() {
    synth->chat = chat.get();
    std::string error;
    conversation = CreateChainConversation(
        {.chat = chat, .transcriber = transcriber, .synthesizer = synth}, 24000,
        &error);
    CHECK(conversation != nullptr);
  }
  VoiceConversation::Callbacks Callbacks(std::function<bool()> cancelled = {}) {
    VoiceConversation::Callbacks callbacks;
    callbacks.on_text = [this](std::string_view delta) {
      text += delta;
      return true;
    };
    callbacks.on_reasoning = [this](std::string_view delta) {
      reasoning += delta;
      return true;
    };
    callbacks.on_audio = [this](std::span<const float> samples) {
      audio += samples.size();
      return true;
    };
    callbacks.cancelled =
        cancelled ? cancelled : std::function<bool()>([] { return false; });
    return callbacks;
  }
  bool Respond(VoiceConversation::Reply* reply, std::string* error,
               VoiceConversation::Settings settings = {},
               std::function<bool()> cancelled = {}) {
    text.clear();
    reasoning.clear();
    audio = 0;
    synth->texts.clear();
    synth->chat_progress.clear();
    return conversation->Respond(settings, Callbacks(std::move(cancelled)),
                                 reply, error);
  }
  // The assistant message of the latest request, by position from the end.
  const ChainMessage& Sent(std::size_t from_end) const {
    const auto& messages = chat->requests.back().messages;
    CHECK(messages.size() > from_end);
    return messages[messages.size() - 1 - from_end];
  }
};

void SpokenTurnStreamsTextAndSpeech() {
  Harness h;
  std::string error;
  std::size_t samples = 0;
  CHECK(!h.conversation->audio_open());
  CHECK(h.conversation->AppendAudio(std::vector<float>(24000, 0.1F), &error));
  CHECK(h.conversation->audio_open());
  CHECK(h.conversation->CommitAudio(&samples, &error));
  CHECK(samples == 24000);
  CHECK(h.transcriber->samples == 16000);  // resampled for the ASR model
  CHECK(!h.conversation->audio_open());
  h.chat->replies = {Reasoning("thinking") + Content("Hello there. ") +
                     Content("It is sunny") + Content(" today!") + Finish()};
  VoiceConversation::Settings settings;
  settings.instructions = "Be brief.";
  settings.max_output_tokens = 99;
  VoiceConversation::Reply reply;
  CHECK(h.Respond(&reply, &error, settings));
  CHECK(h.text == "Hello there. It is sunny today!");
  CHECK(reply.text == h.text);
  // Reasoning is passed on but never spoken.
  CHECK(h.reasoning == "thinking" && reply.reasoning == "thinking");
  CHECK(h.synth->texts.size() == 2);
  CHECK(h.synth->texts[0] == "Hello there.");
  CHECK(h.synth->texts[1] == "It is sunny today!");
  // The first sentence is spoken while the reply still streams.
  CHECK(h.synth->chat_progress[0] < h.synth->chat_progress[1]);
  CHECK(h.audio == (12 + 18) * 100 && reply.audio_samples == h.audio);
  CHECK(reply.first_text_ms >= 0 &&
        reply.first_audio_ms >= reply.first_text_ms);
  CHECK(reply.input_tokens == 12 && reply.output_tokens == 7);
  CHECK(!reply.cancelled && !reply.incomplete && reply.calls.empty());
  const auto& request = h.chat->requests.back();
  CHECK(request.max_tokens == 99);
  CHECK(request.messages.size() == 2);
  CHECK(request.messages[0].role == "system" &&
        request.messages[0].content == "Be brief.");
  CHECK(request.messages[1].role == "user" &&
        request.messages[1].content == "what is the weather");

  // Negative control: a text reply is not spoken.
  h.chat->replies = {Content("No speech. At all.") + Finish("length")};
  settings.audio_output = false;
  CHECK(h.conversation->AddUserText("again", &error));
  CHECK(h.Respond(&reply, &error, settings));
  CHECK(h.text == "No speech. At all.");
  CHECK(h.synth->texts.empty() && h.audio == 0 && reply.audio_samples == 0);
  CHECK(reply.first_audio_ms < 0);
  CHECK(reply.incomplete);
  // The previous reply is in the context.
  CHECK(h.Sent(1).role == "assistant" &&
        h.Sent(1).content == "Hello there. It is sunny today!");
}

void TextTurn() {
  Harness h;
  std::string error;
  CHECK(h.conversation->AddUserText("hi", &error));
  h.chat->replies = {Content("Hi!") + Finish()};
  VoiceConversation::Reply reply;
  CHECK(h.Respond(&reply, &error));
  CHECK(h.text == "Hi!");
  // A last sentence without a trailing space is spoken at the end.
  CHECK(h.synth->texts.size() == 1 && h.synth->texts[0] == "Hi!");
  CHECK(h.chat->requests.back().messages.size() == 1);
  CHECK(h.Sent(0).role == "user" && h.Sent(0).content == "hi");
  // Without instructions there is no system message.
  CHECK(h.chat->requests.back().messages.front().role != "system");
}

void ToolCallThenAnswer() {
  Harness h;
  std::string error;
  CHECK(h.conversation->AddUserText("weather in Rome?", &error));
  auto first = json::Value::object();
  first["tool_calls"] = json::parse(
      R"([{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}])");
  auto rest = json::Value::object();
  rest["tool_calls"] =
      json::parse(R"([{"index":0,"function":{"arguments":"\"Rome\"}"}}])");
  h.chat->replies = {Chunk(first) + Chunk(rest) + Finish("tool_calls"),
                     Content("It is sunny in Rome.") + Finish()};
  VoiceConversation::Settings settings;
  settings.tools = {
      R"({"name":"get_weather","description":"Weather","parameters":{"type":"object"}})"};
  VoiceConversation::Reply reply;
  CHECK(h.Respond(&reply, &error, settings));
  CHECK(reply.calls.size() == 1);
  CHECK(reply.calls[0].name == "get_weather");
  CHECK(reply.calls[0].arguments == R"({"city":"Rome"})");
  CHECK(h.text.empty() && h.audio == 0);
  CHECK(h.chat->requests.back().tools == settings.tools);

  // Negative control: outputs must match the calls.
  const std::vector<std::string> two{"a", "b"};
  CHECK(!h.conversation->AddToolOutputs(two, &error) && !error.empty());
  error.clear();
  const std::vector<std::string> one{R"({"sky":"sunny"})"};
  CHECK(h.conversation->AddToolOutputs(one, &error));
  CHECK(h.Respond(&reply, &error, settings));
  CHECK(h.text == "It is sunny in Rome.");
  CHECK(h.synth->texts.size() == 1);
  CHECK(reply.calls.empty());
  const auto& tool = h.Sent(0);
  CHECK(tool.role == "tool" && tool.tool_call_id == "call_abc" &&
        tool.content == R"({"sky":"sunny"})");
  const auto& call = h.Sent(1);
  CHECK(call.role == "assistant" && call.tool_calls.size() == 1 &&
        call.tool_calls[0].id == "call_abc" &&
        call.tool_calls[0].name == "get_weather");
  // The request body carries the tool turn in OpenAI form.
  const auto body = json::parse(ChainChatBody("m", h.chat->requests.back()));
  CHECK(body.find("stream")->as_bool());
  CHECK(body.find("tools")->items().size() == 1);
  CHECK(body.find("tools")->items()[0].member_str("type") == "function");
  const auto& messages = body.find("messages")->items();
  const auto& assistant = messages[messages.size() - 2];
  CHECK(assistant.find("content")->is_null());
  CHECK(assistant.find("tool_calls")->items()[0].find("function")->member_str(
            "arguments") == R"({"city":"Rome"})");
  CHECK(messages.back().member_str("tool_call_id") == "call_abc");
  CHECK(!body.find("chat_template_kwargs")
             ->find("enable_thinking")
             ->as_bool(true));

  // Invalid tools are refused before any request.
  const auto requests = h.chat->requests.size();
  settings.tools = {"not json"};
  CHECK(!h.conversation->Configure(settings, &error) && !error.empty());
  CHECK(!h.Respond(&reply, &error, settings));
  CHECK(h.chat->requests.size() == requests);
}

void CancelMidReply() {
  Harness h;
  std::string error;
  CHECK(h.conversation->AddUserText("talk", &error));
  h.chat->replies = {Content("One. ") + Content("Two. ") + Content("Three. ") +
                     Content("Four.") + Finish()};
  VoiceConversation::Reply reply;
  bool cancel = false;
  // The user interrupts once the first audio arrives.
  auto callbacks = h.Callbacks([&] { return cancel; });
  callbacks.on_audio = [&](std::span<const float> samples) {
    h.audio += samples.size();
    cancel = true;
    return true;
  };
  CHECK(h.conversation->Respond({}, callbacks, &reply, &error));
  CHECK(reply.cancelled);
  CHECK(!h.chat->finished_stream);
  CHECK(h.synth->texts.size() == 1);
  CHECK(h.audio == 200);  // the second half of "One." never played
  CHECK(h.text.find("Four") == std::string::npos);
  CHECK(reply.calls.empty());
  // The context keeps what was spoken.
  h.chat->replies = {Content("Ok.") + Finish()};
  CHECK(h.conversation->AddUserText("go on", &error));
  CHECK(h.Respond(&reply, &error));
  CHECK(h.Sent(1).role == "assistant" && h.Sent(1).content == "One.");

  // Negative control: without the cancel, the same reply runs to the end.
  Harness full;
  CHECK(full.conversation->AddUserText("talk", &error));
  full.chat->replies = {Content("One. ") + Content("Two. ") +
                        Content("Three. ") + Content("Four.") + Finish()};
  CHECK(full.Respond(&reply, &error));
  CHECK(!reply.cancelled && full.chat->finished_stream);
  CHECK(full.synth->texts.size() == 4);
}

// Sentences of 10, 16 and 10 bytes: [0, 1000), [1000, 2600), [2600, 3600).
const char* kThree = "First one. Second one here. Third one.";

Harness Spoke() {
  Harness h;
  std::string error;
  CHECK(h.conversation->AddUserText("go", &error));
  h.chat->replies = {Content(kThree) + Finish(),
                     Content("Next.") + Finish()};
  VoiceConversation::Reply reply;
  CHECK(h.Respond(&reply, &error));
  CHECK(reply.audio_samples == 3600);
  return h;
}
std::string HeardAfter(std::size_t samples) {
  Harness h = Spoke();
  std::string error;
  CHECK(h.conversation->TruncateReply(samples, &error));
  CHECK(h.conversation->AddUserText("more", &error));
  VoiceConversation::Reply reply;
  CHECK(h.Respond(&reply, &error));
  const auto& previous = h.Sent(1);
  return previous.role == "assistant" ? previous.content : "<none>";
}

void TruncateKeepsHeardSentences() {
  // Inside sentence 2: sentence 1 and the cut-off sentence 2 stay.
  CHECK(HeardAfter(1500) == "First one. Second one here.");
  // Inside sentence 1.
  CHECK(HeardAfter(500) == "First one.");
  // Nothing heard: the reply leaves the context.
  CHECK(HeardAfter(0) == "<none>");
  // Negative control: everything heard keeps everything.
  CHECK(HeardAfter(3600) == kThree);
}

void ChatErrorFailsRespond() {
  Harness h;
  std::string error;
  CHECK(h.conversation->AddUserText("hi", &error));
  h.chat->failure = "chat server returned HTTP 500: boom";
  VoiceConversation::Reply reply;
  CHECK(!h.Respond(&reply, &error));
  CHECK(error == "chat server returned HTTP 500: boom");
  CHECK(h.audio == 0);

  // An error event inside the stream fails the reply too.
  Harness stream;
  CHECK(stream.conversation->AddUserText("hi", &error));
  stream.chat->replies = {
      Content("Partial. ") +
      "data: {\"error\":{\"message\":\"generation failed\"}}\n\n"
      "data: [DONE]\n\n"};
  error.clear();
  CHECK(!stream.Respond(&reply, &error));
  CHECK(error.find("generation failed") != std::string::npos);
}

void EmptyCommit() {
  Harness h;
  std::string error;
  std::size_t samples = 7;
  CHECK(!h.conversation->CommitAudio(&samples, &error));
  CHECK(error.empty() && samples == 0);
  // A turn without words is empty too.
  h.transcriber->text = "  ";
  CHECK(h.conversation->AppendAudio(std::vector<float>(2400, 0.0F), &error));
  CHECK(!h.conversation->CommitAudio(&samples, &error));
  CHECK(error.empty() && samples == 2400);
  // Cleared audio leaves nothing to commit.
  h.transcriber->text = "hello";
  CHECK(h.conversation->AppendAudio(std::vector<float>(2400, 0.0F), &error));
  CHECK(h.conversation->ClearAudio(&error));
  CHECK(!h.conversation->audio_open());
  CHECK(!h.conversation->CommitAudio(&samples, &error) && error.empty());
  // Negative control: real audio commits.
  CHECK(h.conversation->AppendAudio(std::vector<float>(2400, 0.0F), &error));
  CHECK(h.conversation->CommitAudio(&samples, &error));
}

void SseParsing() {
  const std::string stream =
      ": keep-alive\r\n"
      "event: message\r\n"
      "data: {\"a\":1}\r\n\r\n"
      "data: first\ndata: second\n\n"
      "data:nospace\n\n"
      "data: [DONE]\n\n";
  for (std::size_t step : {1U, 3U, 5U, 1000U}) {
    SseParser parser;
    std::vector<std::string> events;
    for (std::size_t at = 0; at < stream.size(); at += step)
      CHECK(parser.Feed(std::string_view(stream).substr(at, step),
                        [&](std::string_view data) {
                          events.emplace_back(data);
                          return true;
                        }));
    CHECK(events.size() == 3);
    CHECK(events[0] == "{\"a\":1}");
    CHECK(events[1] == "first\nsecond");
    CHECK(events[2] == "nospace");
    CHECK(parser.done());
  }
  // Negative control: no terminator, no done; an unfinished event is held.
  SseParser parser;
  std::size_t count = 0;
  CHECK(parser.Feed("data: x\n\ndata: y\n", [&](std::string_view) {
    ++count;
    return true;
  }));
  CHECK(count == 1 && !parser.done());
  // A callback returning false stops the stream.
  SseParser stop;
  CHECK(!stop.Feed("data: x\n\ndata: y\n\n",
                   [](std::string_view) { return false; }));

  CHECK(ChainSentenceEnd("Hello. World") == 6);
  CHECK(ChainSentenceEnd("Pi is 3.14 today") == std::string_view::npos);
  CHECK(ChainSentenceEnd("Hello.") == std::string_view::npos);
  CHECK(ChainSentenceEnd("line\nnext") == 5);
}

// One HTTP exchange on 127.0.0.1 with a canned response.
class CannedServer {
public:
  explicit CannedServer(std::vector<std::string> response)
      : response_(std::move(response)) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd_ >= 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    CHECK(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    CHECK(::bind(fd_, reinterpret_cast<sockaddr*>(&address),
                 sizeof(address)) == 0);
    CHECK(::listen(fd_, 1) == 0);
    socklen_t size = sizeof(address);
    CHECK(::getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &size) ==
          0);
    port_ = ntohs(address.sin_port);
    thread_ = std::thread([this] { Serve(); });
  }
  ~CannedServer() {
    if (thread_.joinable())
      thread_.join();
    ::close(fd_);
  }
  // The request it received, once the exchange is over.
  const std::string& Request() {
    if (thread_.joinable())
      thread_.join();
    return request_;
  }
  std::string url() const {
    return "http://127.0.0.1:" + std::to_string(port_);
  }
private:
  void Serve() {
    const int client = ::accept(fd_, nullptr, nullptr);
    CHECK(client >= 0);
    char buffer[4096];
    std::size_t expected = std::string::npos;
    while (expected == std::string::npos || request_.size() < expected) {
      const auto got = ::recv(client, buffer, sizeof(buffer), 0);
      if (got <= 0)
        break;
      request_.append(buffer, std::size_t(got));
      const auto end = request_.find("\r\n\r\n");
      if (expected == std::string::npos && end != std::string::npos) {
        const auto length = request_.find("Content-Length: ");
        CHECK(length != std::string::npos);
        expected = end + 4 + std::stoul(request_.substr(length + 16));
      }
    }
    for (const auto& piece : response_)
      (void)::send(client, piece.data(), piece.size(), MSG_NOSIGNAL);
    ::shutdown(client, SHUT_RDWR);
    ::close(client);
  }
  std::vector<std::string> response_;
  std::string request_;
  int fd_{-1};
  int port_{0};
  std::thread thread_;
};

std::string HexChunk(std::string_view data) {
  char size[16];
  std::snprintf(size, sizeof(size), "%zx\r\n", data.size());
  return size + std::string(data) + "\r\n";
}

void HttpClientStreams() {
  const std::string sse = Reasoning("hmm") + Content("Hel") + Content("lo.") +
                          Finish();
  std::vector<std::string> pieces{
      "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
      "Transfer-Encoding: chunked\r\n\r\n"};
  // Chunk edges fall inside lines.
  for (std::size_t at = 0; at < sse.size(); at += 11)
    pieces.push_back(HexChunk(std::string_view(sse).substr(at, 11)));
  pieces.push_back("0\r\n\r\n");
  CannedServer server(pieces);
  auto client = CreateHttpChainChatClient(server.url() + "/", "voice-model");
  ChainChatRequest request;
  request.messages = {{.role = "user", .content = "hi"}};
  request.max_tokens = 64;
  std::string content, reasoning, error;
  ChainChatEvents events;
  events.on_content = [&](std::string_view text) {
    content += text;
    return true;
  };
  events.on_reasoning = [&](std::string_view text) {
    reasoning += text;
    return true;
  };
  events.cancelled = [] { return false; };
  ChainChatResult result;
  CHECK(client->Complete(request, events, &result, &error));
  CHECK(content == "Hello." && reasoning == "hmm");
  CHECK(result.finish_reason == "stop" && result.prompt_tokens == 12);
  CHECK(!result.cancelled);
  const auto& sent = server.Request();
  CHECK(sent.starts_with("POST /v1/chat/completions HTTP/1.1"));
  const auto body = json::parse(sent.substr(sent.find("\r\n\r\n") + 4));
  CHECK(body.member_str("model") == "voice-model");
  CHECK(body.find("stream")->as_bool());
  CHECK(body.member_size("max_tokens") == 64);
}

void HttpClientErrors() {
  CannedServer server({
      "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
      "Content-Length: 43\r\n\r\n",
      R"({"error":{"message":"model is not loaded"}})"});
  auto client = CreateHttpChainChatClient(server.url() + "/v1", "m");
  ChainChatRequest request;
  request.messages = {{.role = "user", .content = "hi"}};
  ChainChatEvents events;
  bool any = false;
  events.on_content = [&](std::string_view) {
    any = true;
    return true;
  };
  ChainChatResult result;
  std::string error;
  CHECK(!client->Complete(request, events, &result, &error));
  CHECK(error.find("HTTP 500") != std::string::npos);
  CHECK(error.find("model is not loaded") != std::string::npos);
  CHECK(!any);

  // Through a conversation, a refused connection fails the reply.
  auto transcriber = std::make_shared<FakeTranscriber>();
  auto synth = std::make_shared<FakeSynth>();
  int closed = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  CHECK(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
  CHECK(::bind(closed, reinterpret_cast<sockaddr*>(&address),
               sizeof(address)) == 0);
  socklen_t size = sizeof(address);
  CHECK(::getsockname(closed, reinterpret_cast<sockaddr*>(&address), &size) ==
        0);
  const auto port = ntohs(address.sin_port);
  ::close(closed);  // bound once, never listening
  auto conversation = CreateChainConversation(
      {.chat = CreateHttpChainChatClient(
           "http://127.0.0.1:" + std::to_string(port), "m"),
       .transcriber = transcriber,
       .synthesizer = synth},
      24000, &error);
  CHECK(conversation != nullptr);
  CHECK(conversation->AddUserText("hi", &error));
  VoiceConversation::Reply reply;
  VoiceConversation::Callbacks callbacks;
  callbacks.cancelled = [] { return false; };
  error.clear();
  CHECK(!conversation->Respond({}, callbacks, &reply, &error));
  CHECK(error.find("failed") != std::string::npos);
  CHECK(synth->texts.empty());
}

}  // namespace

int main() {
  SpokenTurnStreamsTextAndSpeech();
  TextTurn();
  ToolCallThenAnswer();
  CancelMidReply();
  TruncateKeepsHeardSentences();
  ChatErrorFailsRespond();
  EmptyCommit();
  SseParsing();
  HttpClientStreams();
  HttpClientErrors();
  std::puts("voice_chain_test: ok");
  return 0;
}
