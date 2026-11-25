#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace ygopro {

// Knowledge of ONE hand from the other seat. Slot numbers are used only
// between shuffles to deduplicate public events and resolve a public choice.
class PublicHand {
 public:
  using Code = std::uint32_t;

  void arrive(unsigned slot, Code publicly_known) {
    // A public hand entry can be reordered before its next reveal.
    // The added identity is certain even when its eventual slot is not.
    shuffle();
    if (publicly_known) ++unlocated_[publicly_known];
    else slots_[slot] = 0;  // a newly arrived unidentified card
  }

  void reveal(unsigned slot, Code code) {
    if (!code) return;
    auto it = slots_.find(slot);
    if (it != slots_.end()) {
      if (!it->second || it->second == code) {
        it->second = code;
        return;
      }
      // A public reveal contradicts the old slot. Preserve identities and rematch them.
      shuffle();
    }
    // This could be a previously seen copy whose position was shuffled.
    consume(unlocated_, code);
    slots_[slot] = code;
  }

  Code at(unsigned slot) const {
    auto it = slots_.find(slot);
    return it == slots_.end() ? 0 : it->second;
  }

  void shuffle() {
    for (const auto &entry : slots_) {
      if (entry.second) ++unlocated_[entry.second];
    }
    slots_.clear();
  }

  void public_draw(Code code) {
    shuffle();  // the draw message carries identities but no slot numbers
    ++unlocated_[code];
  }

  void leave(Code publicly_known) {
    shuffle();  // the remaining hand can be renumbered
    if (publicly_known) {
      consume(unlocated_, publicly_known);
    } else {
      // Any one name might have left. Keep only guaranteed copies of each.
      for (auto it = unlocated_.begin(); it != unlocated_.end();) {
        if (--it->second == 0) it = unlocated_.erase(it);
        else ++it;
      }
    }
  }

  std::vector<Code> names() const {
    std::vector<Code> result;
    for (const auto &entry : unlocated_) {
      result.insert(result.end(), entry.second, entry.first);
    }
    for (const auto &entry : slots_) {
      if (entry.second) result.push_back(entry.second);
    }
    std::sort(result.begin(), result.end());
    return result;
  }

 private:
  static void consume(std::map<Code, int> &pool, Code code) {
    auto it = pool.find(code);
    if (it != pool.end() && --it->second == 0) pool.erase(it);
  }

  std::map<Code, int> unlocated_;
  std::map<unsigned, Code> slots_;
};

}  // namespace ygopro
