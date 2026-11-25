#ifndef YGO_BATTLE_FEATURES_PROTOTYPE_H_
#define YGO_BATTLE_FEATURES_PROTOTYPE_H_
#include <array>
#include <vector>
#include <algorithm>
namespace ygo_battle {
// A public view only. Hidden attack/defense values must never affect output.
struct PublicCard {
  bool visible = false, attack_position = false, defense_position = false;
  int attack = -1, defense = -1, sequence = 0;
};
inline float scaled(int value) {
  return std::max(-1.0f, std::min(1.0f, value / 12000.0f));
}
inline int combat(const PublicCard &c) {
  if (!c.visible) return -1;
  return c.attack_position ? c.attack : (c.defense_position ? c.defense : -1);
}
// 24 features; comparisons describe numbers, not guaranteed battle outcomes.
inline std::array<float,24> encode(const PublicCard *attacker,
    const PublicCard *target, const std::vector<PublicCard> &opponents,
    bool target_window, bool direct) {
  std::array<float,24> f{};
  if (!attacker || !attacker->visible) return f;
  f[0]=1; f[1]=target_window; f[2]=direct;
  const bool ak=attacker->attack >= 0;
  f[3]=ak; f[5]=ak ? scaled(attacker->attack) : 0;
  f[17]=std::clamp(attacker->sequence,0,6)/6.0f;
  if (target) {
    f[4]=target->visible;
    f[18]=std::clamp(target->sequence,0,6)/6.0f;
    f[12]=!target->visible;
    if (target->visible) {
      f[6]=target->attack>=0 ? scaled(target->attack) : 0;
      f[7]=target->defense>=0 ? scaled(target->defense) : 0;
      f[10]=target->attack_position; f[11]=target->defense_position;
      const int value=combat(*target);
      if (value>=0) f[8]=scaled(value);
      if (ak && value>=0) {
        f[13]=1; f[9]=scaled(attacker->attack-value);
        f[14]=attacker->attack>value; f[15]=attacker->attack==value;
        f[16]=attacker->attack<value;
      }
    }
  }
  for (const auto &c:opponents) {
    if (!c.visible) { f[20]+=1.0f/7; continue; }
    f[19]+=1.0f/7;
    const int value=combat(c);
    if (!ak || value<0) continue;
    if (attacker->attack>value) f[21]+=1.0f/7;
    else if (attacker->attack==value) f[22]+=1.0f/7;
    else f[23]+=1.0f/7;
  }
  return f;
}
}
#endif
