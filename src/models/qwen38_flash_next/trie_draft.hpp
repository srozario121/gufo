#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_TRIE_DRAFT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_TRIE_DRAFT_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Smoothed n-gram drafter for sampled requests: a count trie over the
/// session's committed tokens (prompt and output), queried with a PPM-C blend
/// of every matching context order.
///
/// Each committed token is counted under the contexts of orders kOrders that
/// end just before it. A node keeps the kSlots most frequent next tokens with
/// space-saving counters. A query blends the matching orders from the
/// shortest up: P_k(x) = c_k(x) / (n_k + u_k) + u_k / (n_k + u_k) P_{k-1}(x),
/// from a uniform base, where u_k is the number of slots in use. The draft is
/// greedy: the most likely token, then the query again with it appended.
///
/// The index is a function of the committed tokens alone, so proposals depend
/// only on the prefix. They are verified as point masses, which keeps the
/// target distribution exact.
class TrieDraft {
public:
  static constexpr std::size_t kSlots = 4;
  static constexpr std::array<std::size_t, 10> kOrders{1, 2,  3,  4,  6,
                                                       8, 12, 16, 24, 32};
  static constexpr std::size_t kMaxOrder = 32;
  /// Qwen3.8's vocabulary (v1/v2 carried Qwen2's 151,936; it only sets the
  /// blend's uniform floor).
  static constexpr double kBaseVocabulary = 248320.0;

  struct Step {
    std::int32_t token{0};
    double probability{0};  ///< the blend's probability of `token`
    std::size_t order{0};   ///< longest matching context order
  };

  void Clear() {
    keys_.clear();
    slots_.clear();
    nodes_.clear();
    used_ = 0;
    indexed_ = 0;
    last_ = 0;
  }

  [[nodiscard]] std::size_t IndexedTokens() const { return indexed_; }
  [[nodiscard]] std::size_t Nodes() const { return nodes_.size(); }

  /// Counts the committed tokens appended since the last call. A shorter
  /// history, or one whose last indexed token changed, restarts the index.
  void Extend(std::span<const std::int32_t> committed) {
    if (committed.size() < indexed_ ||
        (indexed_ != 0 && committed[indexed_ - 1] != last_)) {
      Clear();
    }
    std::array<std::uint64_t, kOrders.size()> keys{};
    for (std::size_t i = indexed_; i < committed.size(); ++i) {
      const std::size_t n = Keys(committed.first(i), {}, keys);
      for (std::size_t k = 0; k < n; ++k)
        nodes_[Insert(keys[k])].Add(committed[i]);
    }
    indexed_ = committed.size();
    if (indexed_ != 0)
      last_ = committed[indexed_ - 1];
  }

  /// Greedy draft for the context `committed` + `tail`: fills up to
  /// out.size() steps and returns how many. `committed` should be the indexed
  /// history; `tail` holds uncommitted tokens (anchor and drafts).
  std::size_t Draft(std::span<const std::int32_t> committed,
                    std::span<const std::int32_t> tail,
                    std::span<Step> out) const {
    // The last kMaxOrder context tokens, then the draft itself.
    std::vector<std::int32_t> context;
    context.reserve(kMaxOrder + tail.size() + out.size());
    const std::size_t total = committed.size() + tail.size();
    for (std::size_t i = total - std::min(total, kMaxOrder); i < total; ++i)
      context.push_back(i < committed.size() ? committed[i]
                                             : tail[i - committed.size()]);
    std::array<std::uint64_t, kOrders.size()> keys{};
    std::array<const Node*, kOrders.size()> matched{};
    std::array<std::int32_t, kSlots * kOrders.size()> candidates{};
    for (std::size_t step = 0; step < out.size(); ++step) {
      const std::size_t n = Keys(context, {}, keys);
      std::size_t m = 0;
      for (; m < n; ++m) {
        const Node* node = Find(keys[m]);
        if (node == nullptr)
          break;
        matched[m] = node;
      }
      if (m == 0)
        return step;
      std::size_t count = 0;
      for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t s = 0; s < kSlots; ++s) {
          if (matched[j]->count[s] == 0)
            continue;
          const auto token = matched[j]->token[s];
          if (std::find(candidates.begin(), candidates.begin() + count,
                        token) == candidates.begin() + count)
            candidates[count++] = token;
        }
      }
      Step best{};
      best.probability = -1;
      for (std::size_t c = 0; c < count; ++c) {
        const double p = Blend(std::span<const Node* const>(matched).first(m),
                               candidates[c]);
        if (p > best.probability)
          best = {candidates[c], p, 0};
      }
      best.order = kOrders[m - 1];
      out[step] = best;
      context.push_back(best.token);
    }
    return out.size();
  }

private:
  struct Node {
    std::uint32_t total{0};
    std::array<std::int32_t, kSlots> token{};
    std::array<std::uint32_t, kSlots> count{};

    /// Space-saving top-kSlots counter over next tokens.
    void Add(std::int32_t t) {
      ++total;
      std::size_t min_i = 0;
      for (std::size_t i = 0; i < kSlots; ++i) {
        if (count[i] > 0 && token[i] == t) {
          ++count[i];
          return;
        }
        if (count[i] < count[min_i])
          min_i = i;
      }
      token[min_i] = t;
      ++count[min_i];
    }
  };

  static double Blend(std::span<const Node* const> nodes, std::int32_t x) {
    double p = 1.0 / kBaseVocabulary;
    for (const Node* n : nodes) {
      double used = 0;
      double c = 0;
      for (std::size_t s = 0; s < kSlots; ++s) {
        if (n->count[s] == 0)
          continue;
        used += 1;
        if (n->token[s] == x)
          c = n->count[s];
      }
      const double d = static_cast<double>(n->total) + used;
      p = c / d + (used / d) * p;
    }
    return p;
  }

  static std::uint64_t Mix(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
  }

  /// Keys of the contexts of each order in kOrders that fits, ending at the
  /// end of `a` + `b`, shortest first. Returns how many.
  static std::size_t Keys(std::span<const std::int32_t> a,
                          std::span<const std::int32_t> b,
                          std::array<std::uint64_t, kOrders.size()>& out) {
    const std::size_t len = a.size() + b.size();
    const std::size_t top = std::min(kMaxOrder, len);
    std::uint64_t h = 0x9e3779b97f4a7c15ULL;
    std::size_t o = 0;
    for (std::size_t k = 1; k <= top; ++k) {
      const std::size_t i = len - k;
      const auto token =
          static_cast<std::uint32_t>(i < a.size() ? a[i] : b[i - a.size()]);
      h = std::rotl((h ^ token) * 0x100000001b3ULL, 17);
      if (o < kOrders.size() && kOrders[o] == k)
        out[o++] = Mix(h ^ (static_cast<std::uint64_t>(k) << 56));
    }
    return o;
  }

  static constexpr std::uint32_t kEmpty =
      std::numeric_limits<std::uint32_t>::max();

  [[nodiscard]] std::size_t Slot(std::uint64_t key) const {
    return static_cast<std::size_t>(key * 0x9E3779B97F4A7C15ULL) &
           (keys_.size() - 1);
  }

  [[nodiscard]] const Node* Find(std::uint64_t key) const {
    if (used_ == 0)
      return nullptr;
    for (std::size_t i = Slot(key);; i = (i + 1) & (keys_.size() - 1)) {
      if (slots_[i] == kEmpty)
        return nullptr;
      if (keys_[i] == key)
        return &nodes_[slots_[i]];
    }
  }

  /// Node index for `key`, created when absent. The table stays at most half
  /// full.
  std::uint32_t Insert(std::uint64_t key) {
    if (2 * (used_ + 1) > keys_.size())
      Grow();
    std::size_t i = Slot(key);
    while (slots_[i] != kEmpty && keys_[i] != key)
      i = (i + 1) & (keys_.size() - 1);
    if (slots_[i] == kEmpty) {
      keys_[i] = key;
      slots_[i] = static_cast<std::uint32_t>(nodes_.size());
      nodes_.emplace_back();
      ++used_;
    }
    return slots_[i];
  }

  void Grow() {
    const std::size_t capacity = keys_.empty() ? 4096 : 2 * keys_.size();
    std::vector<std::uint64_t> keys(capacity);
    std::vector<std::uint32_t> slots(capacity, kEmpty);
    keys.swap(keys_);
    slots.swap(slots_);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (slots[i] == kEmpty)
        continue;
      std::size_t j = Slot(keys[i]);
      while (slots_[j] != kEmpty)
        j = (j + 1) & (keys_.size() - 1);
      keys_[j] = keys[i];
      slots_[j] = slots[i];
    }
  }

  std::vector<std::uint64_t> keys_;
  std::vector<std::uint32_t> slots_;
  std::vector<Node> nodes_;
  std::size_t used_{0};
  std::size_t indexed_{0};
  std::int32_t last_{0};
};

/// Trie-first rounds for sampled requests. gufo's own rounds (the head's
/// token, then prompt-lookup copies) are unchanged; the trie only replaces the
/// head's first token where the two disagree and the trie is likely right,
/// which is mostly where the head is about to leave a verbatim copy.
///
/// The calibration is an offline replay of logged agent sessions at
/// temperature 1: the chance a trie step is right given the steps before it
/// were, by (longest matching order, blend probability) cell.
namespace trie_policy {

/// The trie's token replaces the head's when its calibrated chance is at
/// least this. In the replay, at disagreements in this band the trie was right
/// 82% of the time and the head 16%.
inline constexpr double kSwitchChance = 0.9;
/// The trie's own copies continue while their joint chance stays above this.
inline constexpr double kCopyChance = 0.5;
/// One first token plus kMaxCopies stays inside --draft-tokens 7.
inline constexpr std::size_t kMaxCopies = 6;

inline constexpr std::array<std::size_t, 10> kOrderEdges{1, 2,  3,  4,  6,
                                                         8, 12, 16, 24, 32};
inline constexpr std::array<double, 8> kProbabilityEdges{0.3, 0.5,  0.7,  0.8,
                                                         0.9, 0.95, 0.99, 1.01};
/// Chance a copied step is right given the steps before it were, by cell
/// [order bin][probability bin]; 0 where fewer than 30 replayed steps fell.
inline constexpr std::array<std::array<double, 8>, 10> kSurvival{{
    {0.08187927375618957, 0.20178755586112065, 0.2870760769935839,
     0.5566546762589928, 0.7063106796116505, 0.8253119429590018,
     0.903448275862069, 0.0},
    {0.10568070519098922, 0.23993589139247667, 0.3113465037691708,
     0.5006321112515802, 0.7189242374549032, 0.8328095537397863,
     0.910288582183187, 0.9792284866468842},
    {0.09694258016405667, 0.21247304693979102, 0.3264980353634578,
     0.4659335920655455, 0.6019731246810682, 0.8133230054221534,
     0.9113962725328445, 0.9796491228070175},
    {0.10594989561586639, 0.2476501838986514, 0.3887187340479837,
     0.5138676566936012, 0.56645074826893, 0.6862718340611353,
     0.8413417951042611, 0.9648103632478633},
    {0.12072072072072072, 0.20852095600969864, 0.44769403824521936,
     0.5336183443451225, 0.5930087390761548, 0.6759807549962991,
     0.8167880413951706, 0.9657253078787279},
    {0.18829516539440203, 0.2949260042283298, 0.49259013492590137,
     0.5987460815047022, 0.6545864045864046, 0.7211033274956217,
     0.8048407288913618, 0.9438400352627677},
    {0.15, 0.3021370670596905, 0.5354330708661418, 0.5835913312693498,
     0.7609962904080552, 0.7382645803698435, 0.8069506032793647,
     0.9485747977815857},
    {0.36231884057971014, 0.4457395498392283, 0.6009842116054952,
     0.7676261994159366, 0.7748463114754098, 0.8587843967342002,
     0.8344630071599045, 0.9686639217605789},
    {0.1455223880597015, 0.3445615839547442, 0.4113132461625924,
     0.6074074074074074, 0.6596916299559471, 0.8242027800490597,
     0.8979457682826623, 0.9774651430561871},
    {0.4266666666666667, 0.3256880733944954, 0.5556179775280898,
     0.6419491525423728, 0.7719467956469166, 0.8672953856517346,
     0.8861218021281413, 0.96859909062402},
}};

/// Bins as numpy.searchsorted(edges, x) (side left): edges strictly below x.
template<typename T, std::size_t N>
[[nodiscard]] std::size_t Bin(const std::array<T, N>& edges, T x) {
  return static_cast<std::size_t>(
      std::lower_bound(edges.begin(), edges.end(), x) - edges.begin());
}

[[nodiscard]] inline double Survival(const TrieDraft::Step& step) {
  const auto o =
      std::min<std::size_t>(Bin(kOrderEdges, step.order), kSurvival.size() - 1);
  const auto p = std::min<std::size_t>(Bin(kProbabilityEdges, step.probability),
                                       kSurvival[o].size() - 1);
  return kSurvival[o][p];
}

}  // namespace trie_policy
}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_TRIE_DRAFT_HPP_
