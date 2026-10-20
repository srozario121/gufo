#ifndef GUFO_SERVER_SAMPLING_REQUEST_HPP_
#define GUFO_SERVER_SAMPLING_REQUEST_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

struct SamplingRequestError {
  std::string message;
  std::string code;
};

namespace detail {

inline constexpr std::int64_t kMaxExactJsonInteger = INT64_C(9007199254740991);
inline constexpr std::size_t kMaxSamplingWindow =
    std::numeric_limits<std::uint32_t>::max();

inline std::string SamplingErrorCode(std::string_view field) {
  std::string code = "invalid_";
  code.append(field);
  for (char& character : code) {
    if (character == '-') {
      character = '_';
    }
  }
  return code;
}

inline std::optional<SamplingRequestError> ReadSamplingFloat(
    const json::Value& body, std::string_view field, float* output,
    bool nullable = false) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr || (nullable && value->is_null())) {
    return std::nullopt;
  }
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() <
          -static_cast<double>(std::numeric_limits<float>::max()) ||
      value->as_double() >
          static_cast<double>(std::numeric_limits<float>::max())) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be a finite number",
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<float>(value->as_double());
  return std::nullopt;
}

template<typename Integer>
inline std::optional<SamplingRequestError> ReadSamplingInteger(
    const json::Value& body, std::string_view field, Integer minimum,
    Integer maximum, Integer* output, bool nullable = false) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr || (nullable && value->is_null())) {
    return std::nullopt;
  }
  const double number = value->is_number() ? value->as_double() : 0.0;
  if (!value->is_number() || !std::isfinite(number) ||
      std::floor(number) != number || number < static_cast<double>(minimum) ||
      number > static_cast<double>(maximum)) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be an integer between " +
                   std::to_string(minimum) + " and " + std::to_string(maximum),
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<Integer>(number);
  return std::nullopt;
}

/// Most `logit_bias` entries one request may carry (OpenAI's own limit).
inline constexpr std::size_t kMaxLogitBiasEntries = 300;
inline constexpr double kMaxLogitBiasMagnitude = 100.0;

/// Reads a request's `logit_bias`: an object of decimal token IDs (no sign,
/// no leading zeros, below the vocabulary) to finite numbers in [-100, 100],
/// at most kMaxLogitBiasEntries of them. The request's entries replace the
/// server default for their tokens. Called only when the server accepts it.
inline std::optional<SamplingRequestError> ReadLogitBias(
    const json::Value& body, const sampling::LogitBias& server,
    sampling::SamplingConfig* output) {
  const json::Value* value = body.find("logit_bias");
  if (value == nullptr || value->is_null() ||
      (value->is_object() && value->empty())) {
    return std::nullopt;
  }
  const auto invalid = [](std::string message) {
    return SamplingRequestError{.message = std::move(message),
                                .code = "invalid_logit_bias"};
  };
  if (!value->is_object()) {
    return invalid("'logit_bias' must be an object of token IDs to numbers");
  }
  sampling::LogitBias bias;
  bias.vocab = server.vocab;
  bias.defaults = server.defaults;
  for (const auto& [key, entry] : value->members()) {
    if (bias.entries.size() >= kMaxLogitBiasEntries) {
      return invalid("'logit_bias' has more than " +
                     std::to_string(kMaxLogitBiasEntries) + " entries");
    }
    const bool canonical =
        !key.empty() && key.size() <= 9 && (key == "0" || key.front() != '0') &&
        std::ranges::all_of(key, [](char c) { return c >= '0' && c <= '9'; });
    const std::uint64_t token = canonical
                                    ? std::stoull(key)
                                    : std::numeric_limits<std::uint64_t>::max();
    if (!canonical || token >= server.vocab) {
      return invalid("'logit_bias' keys must be token IDs below " +
                     std::to_string(server.vocab));
    }
    if (!entry.is_number() || !std::isfinite(entry.as_double()) ||
        std::abs(entry.as_double()) > kMaxLogitBiasMagnitude) {
      return invalid(
          "'logit_bias' values must be finite numbers between -100 "
          "and 100");
    }
    bias.entries.emplace_back(static_cast<sampling::TokenId>(token),
                              static_cast<float>(entry.as_double()));
  }
  std::ranges::sort(bias.entries, {},
                    &std::pair<sampling::TokenId, float>::first);
  if (std::ranges::adjacent_find(bias.entries, {},
                                 &std::pair<sampling::TokenId, float>::first) !=
      bias.entries.end()) {
    return invalid("'logit_bias' repeats a token ID");
  }
  output->logit_bias =
      std::make_shared<const sampling::LogitBias>(std::move(bias));
  return std::nullopt;
}

}  // namespace detail

/// Reads all shared sampling controls, using defaults for omitted fields.
inline std::optional<SamplingRequestError> ParseSamplingConfig(
    const json::Value& body, const sampling::SamplingConfig& defaults,
    sampling::SamplingConfig* output) {
  if (output == nullptr) {
    return SamplingRequestError{
        .message = "sampling output must not be null",
        .code = "invalid_sampling",
    };
  }
  *output = defaults;

  // These controls alter proposal/target probabilities. Reject unsupported
  // spellings instead of accepting a request with a different distribution.
  // A server that accepts `logit_bias` (a non-null default) reads it below.
  const bool accepts_logit_bias = defaults.logit_bias != nullptr;
  for (const auto& [field, value] : body.members()) {
    if (field == "logit_bias" && (accepts_logit_bias || value.is_null() ||
                                  (value.is_object() && value.empty())))
      continue;
    const bool draft_control = field.starts_with("draft_") ||
                               field.ends_with("_draft") || field == "draft" ||
                               field == "speculative";
    if (draft_control || field == "samplers" || field == "typical_p" ||
        field == "tfs_z" || field == "mirostat" || field == "mirostat_eta" ||
        field == "mirostat_tau" || field == "dynatemp_range" ||
        field == "dynatemp_exponent" || field.starts_with("xtc_") ||
        field.starts_with("dry_") || field == "top_n_sigma" ||
        field == "logit_bias") {
      return SamplingRequestError{
          .message = "request field '" + field + "' is not supported",
          .code = "unsupported_sampling",
      };
    }
  }

  if (accepts_logit_bias) {
    if (auto error =
            detail::ReadLogitBias(body, *defaults.logit_bias, output)) {
      return error;
    }
  }
  if (auto error = detail::ReadSamplingFloat(body, "temperature",
                                             &output->temperature, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "top_k", std::int32_t{0},
          std::numeric_limits<std::int32_t>::max(), &output->top_k)) {
    return error;
  }
  if (auto error =
          detail::ReadSamplingFloat(body, "top_p", &output->top_p, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "min_p", &output->min_p)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "min_keep", std::size_t{0},
                                               detail::kMaxSamplingWindow,
                                               &output->min_keep)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "seed", std::int64_t{-1},
                                               detail::kMaxExactJsonInteger,
                                               &output->seed, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "repeat_penalty",
                                             &output->repeat_penalty)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "repeat_last_n", std::size_t{0}, detail::kMaxSamplingWindow,
          &output->repeat_last_n)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(
          body, "frequency_penalty", &output->frequency_penalty, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "presence_penalty",
                                             &output->presence_penalty, true)) {
    return error;
  }

  // Check the original JSON number, not the narrowed float: 1.00000001 must
  // not round into the permitted top-p range. Native sampler controls may
  // be broader; these bounds implement the public API contract.
  for (const auto& [field, minimum, maximum] :
       {std::tuple{"temperature", 0.0, 2.0},
        {"top_p", 0.0, 1.0},
        {"min_p", 0.0, 1.0},
        {"frequency_penalty", -2.0, 2.0},
        {"presence_penalty", -2.0, 2.0}}) {
    const auto* value = body.find(field);
    if (value && value->is_number() &&
        (value->as_double() < minimum || value->as_double() > maximum)) {
      return SamplingRequestError{
          .message = "'" + std::string(field) + "' must be between " +
                     std::to_string(minimum) + " and " +
                     std::to_string(maximum),
          .code = detail::SamplingErrorCode(field),
      };
    }
  }

  try {
    output->Validate();
  } catch (const std::invalid_argument& exception) {
    return SamplingRequestError{
        .message = exception.what(),
        .code = "invalid_sampling",
    };
  }
  if (output->temperature > 2.0F) {
    return SamplingRequestError{
        .message = "'temperature' must be between 0 and 2",
        .code = "invalid_temperature",
    };
  }
  for (const auto& [field, value] :
       {std::pair{"frequency_penalty", output->frequency_penalty},
        std::pair{"presence_penalty", output->presence_penalty}}) {
    if (value < -2.0F || value > 2.0F) {
      return SamplingRequestError{
          .message = "'" + std::string(field) + "' must be between -2 and 2",
          .code = detail::SamplingErrorCode(field),
      };
    }
  }
  return std::nullopt;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SAMPLING_REQUEST_HPP_
