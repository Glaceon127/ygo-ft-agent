#ifndef YGOENV_YGOPRO_YGOPRO_H_
#define YGOENV_YGOPRO_YGOPRO_H_

// clang-format off
#include <algorithm>
#include "battle_features.h"
#include "public_hand.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>


#include <fmt/core.h>
#include <fmt/ranges.h>
#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <ankerl/unordered_dense.h>
#include <unordered_set>

#include "ygoenv/core/BS_thread_pool.h"

#include "ygoenv/core/async_envpool.h"
#include "ygoenv/core/env.h"

#include "ygopro-core/common.h"
#include "ygopro-core/card_data.h"
#include "ygopro-core/duel.h"
#include "ygopro-core/card.h"
#include "ygopro-core/field.h"
#include "ygopro-core/group.h"
#include "ygopro-core/ocgapi.h"

// clang-format on

namespace ygopro {
int g_draw_source_index(intptr_t handle, int player, int count, int index, uint32_t code);

// ===== 外部对手（WindBot 桥接）=====
// 默认 false，不影响训练。开启后，非 AI 玩家的回合不调用内置 bot，
// 而是把控制权交回 Python（g_waiting_external = true）。
inline bool g_external_opponent = false;
inline bool g_waiting_external = false;
inline std::vector<uint8_t> g_external_buffer;   // 对手回合的原始 MSG buffer
// 每条消息单独存一项：WindBot 的 OnPacket 处理一条就 return，
// 因此必须一帧一条地发，不能拼接。
inline std::vector<std::vector<uint8_t>> g_mirror_msgs;
inline int g_num_legal_actions = 0;   // 当前决策点的合法动作数
// 是否给外部客户端补发 MSG_UPDATE_DATA（默认关闭，训练不受影响）
inline bool g_update_data_enabled = false;


inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int i, int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  const auto &w_ = w[ind[i]];
  if (i == n - 1) {
    if (w_.size() == 1) {
      return w_[0] == r;
    } else {
      return w_[0] == r || w_[1] == r;
    }
  }
  if (w_.size() == 1) {
    return sum_to2(w, ind, i + 1, r - w_[0]);
  } else {
    return sum_to2(w, ind, i + 1, r - w_[0]) ||
           sum_to2(w, ind, i + 1, r - w_[1]);
  }
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int r) {
  return sum_to2(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight2(
  const std::vector<std::vector<int>> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to2(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline std::vector<uint32_t>
parse_codes_from_opcodes(const std::vector<uint32_t> &opcodes) {
  int n = opcodes.size();
  std::vector<uint32_t> codes;

  if (n == 2) {
    codes.push_back(opcodes[0]);
    return codes;
  }

  if (((n - 2) % 3) != 0) {
    for (int i = 0; i < n; i++) {
      fmt::println("{}: {}", i, opcodes[i]);
    }
    throw std::runtime_error("invalid format of opcodes");
  }

  for (int i = 2; i < n; i += 3) {
    codes.push_back(opcodes[i]);
    if ((opcodes[i + 1] != 1073742080) || (opcodes[i + 2] != 1073741829)) {
      for (int i = 0; i < n; i++) {
        fmt::println("{}: {}", i, opcodes[i]);
      }
      auto err = fmt::format("invalid format of opcodes starting from {}", i);
      throw std::runtime_error(err);
    }
  }
  return codes;
}

static std::string msg_to_string(int msg) {
  switch (msg) {
  case MSG_RETRY:
    return "retry";
  case MSG_HINT:
    return "hint";
  case MSG_WIN:
    return "win";
  case MSG_SELECT_BATTLECMD:
    return "select_battlecmd";
  case MSG_SELECT_IDLECMD:
    return "select_idlecmd";
  case MSG_SELECT_EFFECTYN:
    return "select_effectyn";
  case MSG_SELECT_YESNO:
    return "select_yesno";
  case MSG_SELECT_OPTION:
    return "select_option";
  case MSG_SELECT_CARD:
    return "select_card";
  case MSG_SELECT_CHAIN:
    return "select_chain";
  case MSG_SELECT_PLACE:
    return "select_place";
  case MSG_SELECT_POSITION:
    return "select_position";
  case MSG_SELECT_TRIBUTE:
    return "select_tribute";
  case MSG_SELECT_COUNTER:
    return "select_counter";
  case MSG_SELECT_SUM:
    return "select_sum";
  case MSG_SELECT_DISFIELD:
    return "select_disfield";
  case MSG_SORT_CARD:
    return "sort_card";
  case MSG_SELECT_UNSELECT_CARD:
    return "select_unselect_card";
  case MSG_CONFIRM_DECKTOP:
    return "confirm_decktop";
  case MSG_CONFIRM_CARDS:
    return "confirm_cards";
  case MSG_SHUFFLE_DECK:
    return "shuffle_deck";
  case MSG_SHUFFLE_HAND:
    return "shuffle_hand";
  case MSG_SWAP_GRAVE_DECK:
    return "swap_grave_deck";
  case MSG_SHUFFLE_SET_CARD:
    return "shuffle_set_card";
  case MSG_REVERSE_DECK:
    return "reverse_deck";
  case MSG_DECK_TOP:
    return "deck_top";
  case MSG_SHUFFLE_EXTRA:
    return "shuffle_extra";
  case MSG_NEW_TURN:
    return "new_turn";
  case MSG_NEW_PHASE:
    return "new_phase";
  case MSG_CONFIRM_EXTRATOP:
    return "confirm_extratop";
  case MSG_MOVE:
    return "move";
  case MSG_POS_CHANGE:
    return "pos_change";
  case MSG_SET:
    return "set";
  case MSG_SWAP:
    return "swap";
  case MSG_FIELD_DISABLED:
    return "field_disabled";
  case MSG_SUMMONING:
    return "summoning";
  case MSG_SUMMONED:
    return "summoned";
  case MSG_SPSUMMONING:
    return "spsummoning";
  case MSG_SPSUMMONED:
    return "spsummoned";
  case MSG_FLIPSUMMONING:
    return "flipsummoning";
  case MSG_FLIPSUMMONED:
    return "flipsummoned";
  case MSG_CHAINING:
    return "chaining";
  case MSG_CHAINED:
    return "chained";
  case MSG_CHAIN_SOLVING:
    return "chain_solving";
  case MSG_CHAIN_SOLVED:
    return "chain_solved";
  case MSG_CHAIN_END:
    return "chain_end";
  case MSG_CHAIN_NEGATED:
    return "chain_negated";
  case MSG_CHAIN_DISABLED:
    return "chain_disabled";
  case MSG_RANDOM_SELECTED:
    return "random_selected";
  case MSG_BECOME_TARGET:
    return "become_target";
  case MSG_DRAW:
    return "draw";
  case MSG_DAMAGE:
    return "damage";
  case MSG_RECOVER:
    return "recover";
  case MSG_EQUIP:
    return "equip";
  case MSG_LPUPDATE:
    return "lpupdate";
  case MSG_CARD_TARGET:
    return "card_target";
  case MSG_CANCEL_TARGET:
    return "cancel_target";
  case MSG_PAY_LPCOST:
    return "pay_lpcost";
  case MSG_ADD_COUNTER:
    return "add_counter";
  case MSG_REMOVE_COUNTER:
    return "remove_counter";
  case MSG_ATTACK:
    return "attack";
  case MSG_BATTLE:
    return "battle";
  case MSG_ATTACK_DISABLED:
    return "attack_disabled";
  case MSG_DAMAGE_STEP_START:
    return "damage_step_start";
  case MSG_DAMAGE_STEP_END:
    return "damage_step_end";
  case MSG_MISSED_EFFECT:
    return "missed_effect";
  case MSG_TOSS_COIN:
    return "toss_coin";
  case MSG_TOSS_DICE:
    return "toss_dice";
  case MSG_ROCK_PAPER_SCISSORS:
    return "rock_paper_scissors";
  case MSG_HAND_RES:
    return "hand_res";
  case MSG_ANNOUNCE_RACE:
    return "announce_race";
  case MSG_ANNOUNCE_ATTRIB:
    return "announce_attrib";
  case MSG_ANNOUNCE_CARD:
    return "announce_card";
  case MSG_ANNOUNCE_NUMBER:
    return "announce_number";
  case MSG_CARD_HINT:
    return "card_hint";
  case MSG_TAG_SWAP:
    return "tag_swap";
  case MSG_RELOAD_FIELD:
    return "reload_field";
  case MSG_AI_NAME:
    return "ai_name";
  case MSG_SHOW_HINT:
    return "show_hint";
  case MSG_PLAYER_HINT:
    return "player_hint";
  case MSG_MATCH_KILL:
    return "match_kill";
  case MSG_CUSTOM_MSG:
    return "custom_msg";
  default:
    return "unknown_msg";
  }
}

// system string
static const std::map<int, std::string> system_strings = {
    // announce type
    {1050, "Monster"},
    {1051, "Spell"},
    {1052, "Trap"},
    {1054, "Normal"},
    {1055, "Effect"},
    {1056, "Fusion"},
    {1057, "Ritual"},
    {1058, "Trap Monsters"},
    {1059, "Spirit"},
    {1060, "Union"},
    {1061, "Gemini"},
    {1062, "Tuner"},
    {1063, "Synchro"},
    {1064, "Token"},
    {1066, "Quick-Play"},
    {1067, "Continuous"},
    {1068, "Equip"},
    {1069, "Field"},
    {1070, "Counter"},
    {1071, "Flip"},
    {1072, "Toon"},
    {1073, "Xyz"},
    {1074, "Pendulum"},
    {1075, "Special Summon"},
    {1076, "Link"},
    {1080, "(N/A)"},
    {1081, "Extra Monster Zone"},
    // announce type end
    // actions
    {1150, "Activate"},
    {1151, "Normal Summon"},
    {1152, "Special Summon"},
    {1153, "Set"},
    {1154, "Flip Summon"},
    {1155, "To Defense"},
    {1156, "To Attack"},
    {1157, "Attack"},
    {1158, "View"},
    {1159, "S/T Set"},
    {1160, "Put in Pendulum Zone"},
    {1161, "Do Effect"},
    {1162, "Reset Effect"},
    {1163, "Pendulum Summon"},
    {1164, "Synchro Summon"},
    {1165, "Xyz Summon"},
    {1166, "Link Summon"},
    {1167, "Tribute Summon"},
    {1168, "Ritual Summon"},
    {1169, "Fusion Summon"},
    {1190, "Add to hand"},
    {1191, "Send to GY"},
    {1192, "Banish"},
    {1193, "Return to Deck"},
    // actions end
    {1, "Normal Summon"},
    {30, "Replay rules apply. Continue this attack?"},
    {31, "Attack directly with this monster?"},
    {80, "Start Step of the Battle Phase."},
    {81, "During the End Phase."},
    {90, "Conduct this Normal Summon without Tributing?"},
    {91, "Use additional Summon?"},
    {92, "Tribute your opponent's monster?"},
    {93, "Continue selecting Materials?"},
    {94, "Activate this card's effect now?"},
    {95, "Use the effect of [%ls]?"},
    {96, "Use the effect of [%ls] to avoid destruction?"},
    {97, "Place [%ls] to a Spell & Trap Zone?"},
    {98, "Tribute a monster(s) your opponent controls?"},
    {200, "From [%ls], activate [%ls]?"},
    {203, "Chain another card or effect?"},
    {210, "Continue selecting?"},
    {218, "Pay LP by Effect of [%ls], instead?"},
    {219, "Detach Xyz material by Effect of [%ls], instead?"},
    {220, "Remove Counter(s) by Effect of [%ls], instead?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {222, "Activate Trigger Effect?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {1621, "Attack Negated"},
    {1622, "[%ls] Missed timing"}
};

static std::string get_system_string(int desc) {
  auto it = system_strings.find(desc);
  if (it != system_strings.end()) {
    return it->second;
  }
  // 新卡池（BabelCDB 2026）会引入硬编码表里没有的系统字符串 id（如 1213）。
  // 这些只用于「选项文案」显示，找不到时用占位符即可，绝不能让整局 abort。
  return "system string " + std::to_string(desc);
}

static std::string ltrim(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}


inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos) {
  std::string spec;
  if (loc & LOCATION_HAND) {
    spec += "h";
  } else if (loc & LOCATION_MZONE) {
    spec += "m";
  } else if (loc & LOCATION_SZONE) {
    spec += "s";
  } else if (loc & LOCATION_GRAVE) {
    spec += "g";
  } else if (loc & LOCATION_REMOVED) {
    spec += "r";
  } else if (loc & LOCATION_EXTRA) {
    spec += "x";
  }
  spec += std::to_string(seq + 1);
  if (loc & LOCATION_OVERLAY) {
    spec.push_back('a' + pos);
  }
  return spec;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos, bool opponent) {
  std::string spec = ls_to_spec(loc, seq, pos);
  if (opponent) {
    spec.insert(0, 1, 'o');
  }
  return spec;
}

inline std::tuple<uint8_t, uint8_t, uint8_t>
spec_to_ls(const std::string spec) {
  uint8_t loc;
  uint8_t seq;
  uint8_t pos = 0;
  int offset = 1;
  if (spec[0] == 'h') {
    loc = LOCATION_HAND;
  } else if (spec[0] == 'm') {
    loc = LOCATION_MZONE;
  } else if (spec[0] == 's') {
    loc = LOCATION_SZONE;
  } else if (spec[0] == 'g') {
    loc = LOCATION_GRAVE;
  } else if (spec[0] == 'r') {
    loc = LOCATION_REMOVED;
  } else if (spec[0] == 'x') {
    loc = LOCATION_EXTRA;
  } else if (std::isdigit(spec[0])) {
    loc = LOCATION_DECK;
    offset = 0;
  } else {
    std::string s = fmt::format("Invalid spec {}", spec);
    throw std::runtime_error(s);
  }
  int end = offset;
  while (end < spec.size() && std::isdigit(spec[end])) {
    end++;
  }
  seq = std::stoi(spec.substr(offset, end - offset)) - 1;
  if (end < spec.size()) {
    pos = spec[end] - 'a';
  }
  return {loc, seq, pos};
}


inline std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>
spec_to_ls(uint8_t player, const std::string spec) {
  uint8_t controller = player;
  int offset = 0;
  if (spec[0] == 'o') {
    controller = 1 - player;
    offset++;
  }
  auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
  return {controller, loc, seq, pos};
}


static std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> read_decks(const std::string &fp) {
  std::ifstream file(fp);
  std::string line;
  std::vector<uint32_t> main_deck, extra_deck, side_deck;
  bool found_extra = false;

  if (file.is_open()) {
    // Read the main deck
    while (std::getline(file, line)) {
      if (line.find("side") != std::string::npos) {
        break;
      }
      if (line.find("extra") != std::string::npos) {
        found_extra = true;
        break;
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        main_deck.push_back(std::stoul(line));
      }
    }

    if (main_deck.size() < 40) {
      std::string err = fmt::format("Main deck must contain at least 40 cards, found: {}, file: {}", main_deck.size(), fp);
      throw std::runtime_error(err);
    }

    // Read the extra deck
    if (found_extra) {
      while (std::getline(file, line)) {
        if (line.find("side") != std::string::npos) {
          break;
        }
        // Check if line contains only digits
        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
          extra_deck.push_back(std::stoul(line));
        }
      }
    }

    // Read the side deck
    while (std::getline(file, line)) {
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        side_deck.push_back(std::stoul(line));
      }
    }

    file.close();
  } else {
    throw std::runtime_error(fmt::format("Unable to open deck file: {}", fp));
  }

  return std::make_tuple(main_deck, extra_deck, side_deck);
}

template <class K = uint8_t>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::map<K, std::string> &m, int id_offset = 0,
         int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  int i = 0;
  for (const auto &[k, v] : m) {
    if (i < m_offset) {
      i++;
      continue;
    }
    m2[k] = i - m_offset + id_offset;
    i++;
  }
  return m2;
}

template <class K = char>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::vector<K> &cmds, int id_offset = 0, int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  for (int i = m_offset; i < cmds.size(); i++) {
    m2[cmds[i]] = i - m_offset + id_offset;
  }
  return m2;
}

static std::string reason_to_string(uint8_t reason) {
  // !victory 0x0 Surrendered
  // !victory 0x1 LP reached 0
  // !victory 0x2 Cards can't be drawn
  // !victory 0x3 Time limit up
  // !victory 0x4 Lost connection
  switch (reason) {
  case 0x0:
    return "Surrendered";
  case 0x1:
    return "LP reached 0";
  case 0x2:
    return "Cards can't be drawn";
  case 0x3:
    return "Time limit up";
  case 0x4:
    return "Lost connection";
  default:
    return "Unknown";
  }
}

#define DEFINE_X_TO_ID_FUN(name, x_map) \
inline uint8_t name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  throw std::runtime_error( \
    fmt::format("[" #name "] cannot find id: {}", x)); \
}

#define DEFINE_X_TO_STRING_FUN(name, x_map) \
inline std::string name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  return "unknown"; \
}

static const ankerl::unordered_dense::map<int, uint8_t> system_string2id =
    make_ids(system_strings, 16);
// 新卡池（BabelCDB 2026）引入了硬编码表里没有的系统字符串 id（如 1213）。
// 原宏遇到未知 id 会 throw，导致整个 env core dump；这里改为映射到 0（None），
// 只损失该动作的效果提示信息，不影响候选生成与训练。
inline uint8_t system_string_to_id(int x) {
  auto it = system_string2id.find(x);
  if (it != system_string2id.end()) {
    return it->second;
  }
  return 0;
}


static const std::map<uint8_t, std::string> location2str = {
    {LOCATION_DECK, "Deck"},
    {LOCATION_HAND, "Hand"},
    {LOCATION_MZONE, "Main Monster Zone"},
    {LOCATION_SZONE, "Spell & Trap Zone"},
    {LOCATION_GRAVE, "Graveyard"},
    {LOCATION_REMOVED, "Banished"},
    {LOCATION_EXTRA, "Extra Deck"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> location2id =
    make_ids(location2str, 1);
DEFINE_X_TO_ID_FUN(location_to_id, location2id)


#define POS_NONE 0x0 // xyz materials (overlay) ???

static const std::map<uint8_t, std::string> position2str = {
    {POS_NONE, "none"},
    {POS_FACEUP_ATTACK, "face-up attack"},
    {POS_FACEDOWN_ATTACK, "face-down attack"},
    {POS_ATTACK, "attack"},
    {POS_FACEUP_DEFENSE, "face-up defense"},
    {POS_FACEUP, "face-up"},
    {POS_FACEDOWN_DEFENSE, "face-down defense"},
    {POS_FACEDOWN, "face-down"},
    {POS_DEFENSE, "defense"},
};
DEFINE_X_TO_STRING_FUN(position_to_string, position2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> position2id =
    make_ids(position2str);
DEFINE_X_TO_ID_FUN(position_to_id, position2id)


#define ATTRIBUTE_NONE 0x0 // token

static const std::map<uint8_t, std::string> attribute2str = {
    {ATTRIBUTE_NONE, "None"},   {ATTRIBUTE_EARTH, "Earth"},
    {ATTRIBUTE_WATER, "Water"}, {ATTRIBUTE_FIRE, "Fire"},
    {ATTRIBUTE_WIND, "Wind"},   {ATTRIBUTE_LIGHT, "Light"},
    {ATTRIBUTE_DARK, "Dark"},   {ATTRIBUTE_DEVINE, "Divine"},
};
DEFINE_X_TO_STRING_FUN(attribute_to_string, attribute2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> attribute2id =
    make_ids(attribute2str);
DEFINE_X_TO_ID_FUN(attribute_to_id, attribute2id)


#define RACE_NONE 0x0 // token

static const std::map<uint32_t, std::string> race2str = {
    {RACE_NONE, "None"},
    {RACE_WARRIOR, "Warrior"},
    {RACE_SPELLCASTER, "Spellcaster"},
    {RACE_FAIRY, "Fairy"},
    {RACE_FIEND, "Fiend"},
    {RACE_ZOMBIE, "Zombie"},
    {RACE_MACHINE, "Machine"},
    {RACE_AQUA, "Aqua"},
    {RACE_PYRO, "Pyro"},
    {RACE_ROCK, "Rock"},
    {RACE_WINDBEAST, "Winged Beast"},
    {RACE_PLANT, "Plant"},
    {RACE_INSECT, "Insect"},
    {RACE_THUNDER, "Thunder"},
    {RACE_DRAGON, "Dragon"},
    {RACE_BEAST, "Beast"},
    {RACE_BEASTWARRIOR, "Beast Warrior"},
    {RACE_DINOSAUR, "Dinosaur"},
    {RACE_FISH, "Fish"},
    {RACE_SEASERPENT, "Sea Serpent"},
    {RACE_REPTILE, "Reptile"},
    {RACE_PSYCHO, "Psychic"},
    {RACE_DEVINE, "Divine"},
    {RACE_CREATORGOD, "Creator God"},
    {RACE_WYRM, "Wyrm"},
    {RACE_CYBERSE, "Cyberse"},
    {RACE_ILLUSION, "Illusion"}};

static const ankerl::unordered_dense::map<uint32_t, uint8_t> race2id =
    make_ids(race2str);
DEFINE_X_TO_ID_FUN(race_to_id, race2id)


static const std::map<uint32_t, std::string> type2str = {
    {TYPE_MONSTER, "Monster"},
    {TYPE_SPELL, "Spell"},
    {TYPE_TRAP, "Trap"},
    {TYPE_NORMAL, "Normal"},
    {TYPE_EFFECT, "Effect"},
    {TYPE_FUSION, "Fusion"},
    {TYPE_RITUAL, "Ritual"},
    {TYPE_TRAPMONSTER, "Trap Monster"},
    {TYPE_SPIRIT, "Spirit"},
    {TYPE_UNION, "Union"},
    {TYPE_DUAL, "Gemini"},
    {TYPE_TUNER, "Tuner"},
    {TYPE_SYNCHRO, "Synchro"},
    {TYPE_TOKEN, "Token"},
    {TYPE_QUICKPLAY, "Quick-play"},
    {TYPE_CONTINUOUS, "Continuous"},
    {TYPE_EQUIP, "Equip"},
    {TYPE_FIELD, "Field"},
    {TYPE_COUNTER, "Counter"},
    {TYPE_FLIP, "Flip"},
    {TYPE_TOON, "Toon"},
    {TYPE_XYZ, "Xyz"},
    {TYPE_PENDULUM, "Pendulum"},
    {TYPE_SPSUMMON, "Special"},
    {TYPE_LINK, "Link"},
};

inline std::vector<uint8_t> type_to_ids(uint32_t type) {
  std::vector<uint8_t> ids;
  ids.reserve(type2str.size());
  for (const auto &[k, v] : type2str) {
    ids.push_back(std::min(1u, type & k));
  }
  return ids;
}

static const std::map<int, std::string> phase2str = {
    {PHASE_DRAW, "draw phase"},
    {PHASE_STANDBY, "standby phase"},
    {PHASE_MAIN1, "main1 phase"},
    {PHASE_BATTLE_START, "battle start phase"},
    {PHASE_BATTLE_STEP, "battle step phase"},
    {PHASE_DAMAGE, "damage phase"},
    {PHASE_DAMAGE_CAL, "damage calculation phase"},
    {PHASE_BATTLE, "battle phase"},
    {PHASE_MAIN2, "main2 phase"},
    {PHASE_END, "end phase"},
};
DEFINE_X_TO_STRING_FUN(phase_to_string, phase2str)

static const ankerl::unordered_dense::map<int, uint8_t> phase2id =
    make_ids(phase2str);
DEFINE_X_TO_ID_FUN(phase_to_id, phase2id)


static const std::vector<int> _msgs = {
    MSG_SELECT_IDLECMD,  MSG_SELECT_CHAIN,     MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,  MSG_SELECT_POSITION,  MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,    MSG_SELECT_BATTLECMD, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_OPTION,   MSG_SELECT_PLACE,     MSG_SELECT_SUM,
    MSG_SELECT_DISFIELD, MSG_ANNOUNCE_ATTRIB,  MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_CARD,
};

static const ankerl::unordered_dense::map<int, uint8_t> msg2id =
    make_ids(_msgs, 1);
DEFINE_X_TO_ID_FUN(msg_to_id, msg2id)


enum class ActionAct {
  None,
  Set,
  Repo,
  SpSummon,
  Summon,
  MSet,
  Attack,
  DirectAttack,
  Activate,
  Cancel,
};

inline std::string action_act_to_string(ActionAct act) {
  switch (act) {
  case ActionAct::None:
    return "None";
  case ActionAct::Set:
    return "Set";
  case ActionAct::Repo:
    return "Repo";
  case ActionAct::SpSummon:
    return "SpSummon";
  case ActionAct::Summon:
    return "Summon";
  case ActionAct::MSet:
    return "MSet";
  case ActionAct::Attack:
    return "Attack";
  case ActionAct::DirectAttack:
    return "DirectAttack";
  case ActionAct::Activate:
    return "Activate";
  case ActionAct::Cancel:
    return "Cancel";
  default:
    return "Unknown";
  }
}

enum class ActionPhase {
  None,
  Battle,
  Main2,
  End,
};

inline std::string action_phase_to_string(ActionPhase phase) {
  switch (phase) {
  case ActionPhase::None:
    return "None";
  case ActionPhase::Battle:
    return "Battle";
  case ActionPhase::Main2:
    return "Main2";
  case ActionPhase::End:
    return "End";
  default:
    return "Unknown";
  }
}

enum class ActionPlace {
  None,
  MZone1,
  MZone2,
  MZone3,
  MZone4,
  MZone5,
  MZone6,
  MZone7,
  SZone1,
  SZone2,
  SZone3,
  SZone4,
  SZone5,
  SZone6,
  SZone7,
  SZone8,
  OpMZone1,
  OpMZone2,
  OpMZone3,
  OpMZone4,
  OpMZone5,
  OpMZone6,
  OpMZone7,
  OpSZone1,
  OpSZone2,
  OpSZone3,
  OpSZone4,
  OpSZone5,
  OpSZone6,
  OpSZone7,
  OpSZone8,
};


inline std::vector<ActionPlace> flag_to_usable_places(
  uint32_t flag, bool reverse = false) {
  std::vector<ActionPlace> places;
  for (int j = 0; j < 4; j++) {
    uint32_t value = (flag >> (j * 8)) & 0xff;
    for (int i = 0; i < 8; i++) {
      bool avail = (value & (1 << i)) == 0;
      if (reverse) {
        avail = !avail;
      }
      if (avail) {
        ActionPlace place;
        if (j == 0) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::MZone1));
        } else if (j == 1) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::SZone1));
        } else if (j == 2) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpMZone1));
        } else if (j == 3) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpSZone1));
        }
        places.push_back(place);
      }
    }
  }
  return places;
}

inline std::string action_place_to_string(ActionPlace place) {
  int i = static_cast<int>(place);
  if (i == 0) {
    return "None";
  }
  else if (i >= static_cast<int>(ActionPlace::MZone1) && i <= static_cast<int>(ActionPlace::MZone7)) {
    return fmt::format("m{}", i - static_cast<int>(ActionPlace::MZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::SZone1) && i <= static_cast<int>(ActionPlace::SZone8)) {
    return fmt::format("s{}", i - static_cast<int>(ActionPlace::SZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpMZone1) && i <= static_cast<int>(ActionPlace::OpMZone7)) {
    return fmt::format("om{}", i - static_cast<int>(ActionPlace::OpMZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpSZone1) && i <= static_cast<int>(ActionPlace::OpSZone8)) {
    return fmt::format("os{}", i - static_cast<int>(ActionPlace::OpSZone1) + 1);
  }
  else {
    return "Unknown";
  }
}


inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  x = x % 65536;
  return {
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x & 0xff),
  };
}

static std::vector<int> find_substrs(const std::string &str,
                                     const std::string &substr) {
  std::vector<int> res;
  int pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    res.push_back(pos);
    pos += substr.length();
  }
  return res;
}

inline std::string time_now() {
  // strftime %Y-%m-%d %H-%M-%S
  time_t now = time(0);
  tm *ltm = localtime(&now);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H-%M-%S", ltm);
  return std::string(buffer);
}

// from ygopro/gframe/replay.h

// replay flag
#define REPLAY_COMPRESSED	0x1
#define REPLAY_TAG			0x2
#define REPLAY_DECODED		0x4
#define REPLAY_SINGLE_MODE	0x8
#define REPLAY_UNIFORM		0x10

// max size
#define MAX_REPLAY_SIZE	0x20000


struct ReplayHeader {
	unsigned int id;
	unsigned int version;
	unsigned int flag;
	unsigned int seed;
	unsigned int datasize;
	unsigned int start_time;
	unsigned char props[8];

	ReplayHeader()
		: id(0), version(0), flag(0), seed(0), datasize(0), start_time(0), props{ 0 } {}
};

// from ygopro/gframe/replay.h

using PlayerId = uint8_t;
using CardCode = uint32_t;
using CardId = uint16_t;

const int DESCRIPTION_LIMIT = 10000;
const int CARD_EFFECT_OFFSET = 10010;

class LegalAction {
public:
  std::string spec_ = "";
  ActionAct act_ = ActionAct::None;
  ActionPhase phase_ = ActionPhase::None;
  bool finish_ = false;
  uint8_t position_ = 0;
  int effect_ = -1;
  uint8_t number_ = 0;
  ActionPlace place_ = ActionPlace::None;
  uint8_t attribute_ = 0;

  int spec_index_ = 0;
  CardId cid_ = 0;
  uint32_t src_code_ = 0;  // 该动作在消息里的真实卡号（仅选卡类消息）
  int msg_ = 0;
  // 规则上合法、但被环境策略屏蔽的候选（例如无触发 tag 且必亏的攻击）。
  // 保留它只是为了不改动 legal_actions_ 的索引，观测里会把 msg 写成 0，
  // 模型侧的 a_mask 因此不会选中它。
  bool masked_ = false;
  uint32_t response_ = 0;

  static LegalAction from_spec(const std::string &spec) {
    LegalAction la;
    la.spec_ = spec;
    return la;
  }

  static LegalAction act_spec(ActionAct act, const std::string &spec) {
    LegalAction la;
    la.act_ = act;
    la.spec_ = spec;
    return la;
  }

  static LegalAction finish() {
    LegalAction la;
    la.finish_ = true;
    return la;
  }

  static LegalAction cancel() {
    LegalAction la;
    la.act_ = ActionAct::Cancel;
    return la;
  }

  static LegalAction activate_spec(int effect_idx, const std::string &spec) {
    LegalAction la;
    la.act_ = ActionAct::Activate;
    la.effect_ = effect_idx;
    la.spec_ = spec;
    return la;
  }

  static LegalAction phase(ActionPhase phase) {
    LegalAction la;
    la.phase_ = phase;
    return la;
  }

  static LegalAction number(uint8_t number) {
    LegalAction la;
    la.number_ = number;
    return la;
  }

  static LegalAction place(ActionPlace place) {
    LegalAction la;
    la.place_ = place;
    return la;
  }

  static LegalAction attribute(int attribute) {
    LegalAction la;
    la.attribute_ = attribute;
    return la;
  }
};

class SpecInfo {
public:
  uint16_t index;
  CardId cid;
};

inline uint8_t compute_effect_tags(const std::string &desc) {
  std::string d;
  d.reserve(desc.size());
  for (char ch : desc) {
    d.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  // 同时匹配英文与中文卡文（cdb 可能是任意语言）
  auto hit = [&](const char *en, const char *zh) {
    // std::string::find("") 恒返回 0，必须跳过空关键词，
    // 否则「只用中文关键词」的规则会退化成无条件命中。
    return (en != nullptr && en[0] != '\0' &&
            d.find(en) != std::string::npos) ||
           (zh != nullptr && zh[0] != '\0' &&
            d.find(zh) != std::string::npos);
  };
  // bit0-5: 这张卡被打掉之后可能带来的触发收益（劣势攻击的合理性来源）
  // bit6  : 这只怪兽可以直击，用于和「打对方怪兽」做收益比较
  // bit7  : 效果会把手牌/墓地资源拿回手牌，代表可能的后继收益
  uint8_t t = 0;
  if (hit("when this card is destroyed", "这张卡被破坏的场合") ||
      hit("", "这张卡被破坏送去墓地")) t |= 1 << 0;
  if (hit("destroyed by battle", "被战斗破坏")) t |= 1 << 1;
  // 只认「这张卡被送墓时」的触发写法，避免把主动送墓也算进来
  if (hit("if this card is sent to the gy", "这张卡被送去墓地的场合") ||
      hit("when this card is sent to the gy", "这张卡被送去墓地时") ||
      hit("", "这张卡被送入墓地的场合")) {
    t |= 1 << 2;
  }
  if (hit("this card battles", "这张卡进行战斗")) t |= 1 << 3;
  if (hit("piercing", "穿刺")) t |= 1 << 4;
  if (hit("battle damage", "战斗伤害")) t |= 1 << 5;
  if (hit("attacks directly", "直接攻击") ||
      hit("attack directly", "可以直接攻击")) t |= 1 << 6;
  if (hit("to your hand", "加入手卡") ||
      hit("", "加入手牌")) t |= 1 << 7;
  return t;
}

inline uint16_t compute_effect_features(const std::string &desc) {
  std::string d;
  d.reserve(desc.size());
  for (char ch : desc) {
    d.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  // 同时匹配英文与中文卡文（cdb 可能是任意语言）
  auto hit = [&](const char *en, const char *zh) {
    // std::string::find("") 恒返回 0，必须跳过空关键词，
    // 否则「只用中文关键词」的规则会退化成无条件命中。
    return (en != nullptr && en[0] != '\0' &&
            d.find(en) != std::string::npos) ||
           (zh != nullptr && zh[0] != '\0' &&
            d.find(zh) != std::string::npos);
  };
  // bit0-5  资源 / 展开
  // bit6-10 场面交互
  // bit11-15 PSCT 结构 / 时点
  uint16_t f = 0;
  if (hit("from your deck to your hand", "从卡组把") ||
      hit("from the deck to your hand", "从卡组将") ||
      hit("", "从卡组选") || hit("", "从卡组把1张")) f |= 1 << 0;
  if (hit("special summon", "特殊召唤")) f |= 1 << 1;
  if (hit("fusion summon", "融合召唤") || hit("synchro summon", "同调召唤") ||
      hit("xyz summon", "超量召唤") || hit("link summon", "连接召唤") ||
      hit("ritual summon", "仪式召唤")) f |= 1 << 2;
  if (hit("to the gy", "送去墓地") || hit("to the graveyard", "送入墓地") ||
      hit("", "送去墓地的")) f |= 1 << 3;
  if (hit("from your gy", "从墓地") || hit("in your gy", "在墓地") ||
      hit("from your graveyard", "从自己墓地") || hit("in your graveyard", "自己墓地的")) f |= 1 << 4;
  if (hit("draw 1 card", "抽1张") || hit("draw 2 cards", "抽2张") ||
      hit("draw 3 cards", "抽3张") || hit("", "抽卡")) f |= 1 << 5;
  if (hit("destroy", "破坏")) f |= 1 << 6;
  if (hit("banish", "除外")) f |= 1 << 7;
  if (hit("negate", "无效")) f |= 1 << 8;
  if ((hit("gains", "上升") || hit("loses", "下降") || hit("becomes", "变成")) &&
      (hit("atk", "攻击力") || hit("def", "守备力"))) f |= 1 << 9;
  if (hit("battle position", "表示形式") || hit("control of", "控制权")) f |= 1 << 10;
  if (d.find(':') != std::string::npos || d.find(';') != std::string::npos ||
      d.find("\xef\xbc\x9a") != std::string::npos ||       // 全角冒号 ：
      d.find("\xef\xbc\x9b") != std::string::npos) {        // 全角分号 ；
    f |= 1 << 11;
  }
  if (d.find(';') != std::string::npos ||
      d.find("\xef\xbc\x9b") != std::string::npos) f |= 1 << 12;
  if (hit("target 1", "以") || hit("target 2", "为对象") ||
      hit("target 3", "作为对象") || hit("targets 1", "取为对象")) f |= 1 << 13;
  if (hit("quick effect", "对方回合") ||
      hit("during either player's turn", "双方回合") ||
      hit("during your opponent's turn", "对手回合")) f |= 1 << 14;
  if (hit("once per turn", "1回合1次") || hit("", "一回合一次")) f |= 1 << 15;
  return f;
}

// category 与 Card 解耦：cards_ 是 ankerl::unordered_dense::map，元素连续存储，
// Card 的 sizeof 即 stride，给它加字段会破坏整张表的布局。
static ankerl::unordered_dense::map<uint32_t, uint32_t> g_card_category;

// cdb datas.category 的位（与 ocgcore effect_category 一致）。
// NEGATE = 无效「发动」，DISABLE = 效果「无效化」，两者合起来才覆盖常见的无效系卡。
static constexpr uint32_t kCategoryDestroy = 0x1;
static constexpr uint32_t kCategoryDisable = 0x4000;
static constexpr uint32_t kCategoryNegate = 0x10000000;

class Card {
  friend class YGOProEnvImpl;

protected:
  CardCode code_ = 0;
  uint32_t alias_;
  uint64_t setcode_;
  uint32_t type_;
  uint32_t level_;
  uint32_t lscale_;
  uint32_t rscale_;
  int32_t attack_;
  int32_t defense_;
  uint32_t race_;
  uint32_t attribute_;
  uint32_t link_marker_;
  // uint32_t category_;
  std::string name_;
  std::string desc_;
  uint8_t tags_ = 0;
  uint16_t effects_ = 0;
  std::vector<std::string> strings_;

  uint32_t data_ = 0;

  uint32_t status_ = 0;
  PlayerId controler_ = 0;
  uint32_t location_ = 0;
  uint32_t sequence_ = 0;
  uint32_t position_ = 0;
  uint32_t counter_ = 0;

public:
  Card() = default;

  Card(CardCode code, uint32_t alias, uint64_t setcode, uint32_t type,
       uint32_t level, uint32_t lscale, uint32_t rscale, int32_t attack,
       int32_t defense, uint32_t race, uint32_t attribute, uint32_t link_marker,
       const std::string &name, const std::string &desc,
       const std::vector<std::string> &strings)
      : code_(code), alias_(alias), setcode_(setcode), type_(type),
        level_(level), lscale_(lscale), rscale_(rscale), attack_(attack),
        defense_(defense), race_(race), attribute_(attribute),
        link_marker_(link_marker), name_(name), desc_(desc), strings_(strings),
        tags_(compute_effect_tags(desc)), effects_(compute_effect_features(desc)) {
  }

  ~Card() = default;

  void set_location(uint32_t location) {
    controler_ = location & 0xff;
    location_ = (location >> 8) & 0xff;
    sequence_ = (location >> 16) & 0xff;
    position_ = (location >> 24) & 0xff;
  }

  const std::string &name() const { return name_; }
  const std::string &desc() const { return desc_; }
  const uint32_t &type() const { return type_; }
  const uint32_t &level() const { return level_; }
  const std::vector<std::string> &strings() const { return strings_; }

  std::string get_spec(bool opponent) const {
    return ls_to_spec(location_, sequence_, position_, opponent);
  }

  std::string get_spec(PlayerId player) const {
    return get_spec(player != controler_);
  }

  std::string get_position() const { return position_to_string(position_); }

  std::string get_effect_description(CardCode code, int effect_idx) const {
    if (code == 0) {
      return get_system_string(effect_idx);
    }
    if (effect_idx == 0) {
      return "default";
    }
    effect_idx -= CARD_EFFECT_OFFSET;
    if (effect_idx < 0) {
      throw std::runtime_error(
          fmt::format("Invalid effect index: {}", effect_idx));
    }
    auto s = strings_[effect_idx];
    if (s.empty()) {
      return "effect " + std::to_string(effect_idx);
    }
    return s;
  }
};

struct MDuel {
  intptr_t pduel;
  uint64_t seed;
  std::vector<CardCode> main_deck0;
  std::vector<CardCode> extra_deck0;
  std::string deck_name0;
  std::vector<CardCode> main_deck1;
  std::vector<CardCode> extra_deck1;
  std::string deck_name1;
};

inline Card db_query_card(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query1(db, "SELECT * FROM datas WHERE id=?");
  query1.bind(1, code);
  bool found = query1.executeStep();
  if (!found) {
    std::string msg = "[db_query_card] Card not found: " + std::to_string(code);
    throw std::runtime_error(msg);
  }

  uint32_t alias = query1.getColumn("alias");
  uint64_t setcode = query1.getColumn("setcode").getInt64();
  uint32_t type = query1.getColumn("type");
  uint32_t level_ = query1.getColumn("level");
  uint32_t level = level_ & 0xff;
  uint32_t lscale = (level_ >> 24) & 0xff;
  uint32_t rscale = (level_ >> 16) & 0xff;
  int32_t attack = query1.getColumn("atk");
  int32_t defense = query1.getColumn("def");
  uint32_t link_marker = 0;
  if (type & TYPE_LINK) {
    defense = 0;
    link_marker = defense;
  }
  uint32_t race = query1.getColumn("race");
  uint32_t attribute = query1.getColumn("attribute");
  uint32_t category = static_cast<uint32_t>(query1.getColumn("category").getInt64());

  SQLite::Statement query2(db, "SELECT * FROM texts WHERE id=?");
  query2.bind(1, code);
  query2.executeStep();

  std::string name = query2.getColumn(1);
  std::string desc = query2.getColumn(2);
  std::vector<std::string> strings;
  for (int i = 3; i < query2.getColumnCount(); ++i) {
    std::string str = query2.getColumn(i);
    strings.push_back(str);
  }
  Card card(code, alias, setcode, type, level, lscale, rscale, attack,
            defense, race, attribute, link_marker, name, desc, strings);
  g_card_category[static_cast<uint32_t>(code)] = category;  // 旁路存储
  return card;
}

// Match YGOMobile DataManager::ReadDB. The normal Black Luster Soldier is
// rule-named for the ritual card despite having an adjacent print number.
inline CardCode rule_name_for(CardCode code, CardCode alias, uint32_t type) {
  constexpr CardCode artwork_range = 20;
  const CardCode distance = code >= alias ? code - alias : alias - code;
  return (code == 5405695 || (!(type & TYPE_TOKEN) && distance >= artwork_range))
             ? alias : 0;
}

inline card_data db_query_card_data(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query(db,
      "SELECT c.*, base.alias AS base_alias, base.type AS base_type "
      "FROM datas AS c LEFT JOIN datas AS base ON base.id = c.alias WHERE c.id = ?");
  query.bind(1, code);
  query.executeStep();
  card_data card;
  card.code = code;
  card.alias = query.getColumn("alias");
  // Fluorohydride 的 card_data.setcode 是 uint16_t[SIZE_SETCODE]
  int64_t setcode_raw = query.getColumn("setcode").getInt64();
  write_setcode(card.setcode, static_cast<uint64_t>(setcode_raw));
  card.type = query.getColumn("type");
  card.rule_code = rule_name_for(code, card.alias, card.type);
  if (card.rule_code) {
    card.alias = 0;  // load this printed card's own script
  } else if (card.alias && !(card.type & TYPE_TOKEN)) {
    // An alternate artwork uses its base print's script and inherits its rule name.
    card.rule_code = rule_name_for(card.alias, query.getColumn("base_alias"),
                                   query.getColumn("base_type"));
  }
  uint32_t level_ = query.getColumn("level");
  card.level = level_ & 0xff;
  card.lscale = (level_ >> 24) & 0xff;
  card.rscale = (level_ >> 16) & 0xff;
  card.attack = query.getColumn("atk");
  card.defense = query.getColumn("def");
  if (card.type & TYPE_LINK) {
    card.link_marker = card.defense;
    card.defense = 0;
  } else {
    card.link_marker = 0;
  }
  card.race = static_cast<uint32_t>(query.getColumn("race").getInt64());
  card.attribute = static_cast<uint32_t>(query.getColumn("attribute").getInt64());
  return card;
}

struct card_script {
  uint8_t *buf;
  int len;
};

static ankerl::unordered_dense::map<CardCode, Card> cards_;
static ankerl::unordered_dense::map<CardCode, CardId> card_ids_;
// 反向表：action.cid_ 是内部 CardId，而 g_card_category 的 key 是官方 code
static ankerl::unordered_dense::map<CardId, CardCode> g_id_to_code;
static ankerl::unordered_dense::map<CardCode, card_data> cards_data_;
static ankerl::unordered_dense::map<std::string, card_script> cards_script_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    main_decks_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    extra_decks_;
static std::vector<std::string> deck_names_;
static ankerl::unordered_dense::map<std::string, int> deck_names_ids_;

inline const Card &c_get_card(CardCode code) {
  auto it = cards_.find(code);
  if (it != cards_.end()) {
    return it->second;
  }
  // 卡池外的卡（新版引擎运行时生成的虚拟卡/衍生物）：返回空卡，不中断对局
  static const Card unknown_card{};
  return unknown_card;
}

inline CardId &c_get_card_id(CardCode code) {
  auto it = card_ids_.find(code);
  if (it != card_ids_.end()) {
    return it->second;
  }
  // 卡池外的卡（新版引擎运行时生成的衍生物/虚拟卡）：映射到 0（unknown）
  static CardId unknown_card_id = 0;
  return unknown_card_id;
}

inline void sort_extra_deck(std::vector<CardCode> &deck) {
  std::vector<CardCode> c;
  std::vector<std::pair<CardCode, int>> fusion, xyz, synchro, link;

  for (auto code : deck) {
    const Card &cc = c_get_card(code);
    if (cc.type() & TYPE_FUSION) {
      fusion.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_XYZ) {
      xyz.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_SYNCHRO) {
      synchro.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_LINK) {
      link.push_back({code, cc.level()});
    } else {
      throw std::runtime_error("Not extra deck card");
    }
  }

  auto cmp = [](const std::pair<CardCode, int> &a,
                const std::pair<CardCode, int> &b) {
    return a.second < b.second;
  };
  std::sort(fusion.begin(), fusion.end(), cmp);
  std::sort(xyz.begin(), xyz.end(), cmp);
  std::sort(synchro.begin(), synchro.end(), cmp);
  std::sort(link.begin(), link.end(), cmp);

  for (const auto &tc : fusion) {
    c.push_back(tc.first);
  }
  for (const auto &tc : xyz) {
    c.push_back(tc.first);
  }
  for (const auto &tc : synchro) {
    c.push_back(tc.first);
  }
  for (const auto &tc : link) {
    c.push_back(tc.first);
  }

  deck = c;
}

inline void preload_deck(const SQLite::Database &db,
                         const std::vector<CardCode> &deck) {
  for (const auto &code : deck) {
    auto it = cards_.find(code);
    if (it == cards_.end()) {
      cards_[code] = db_query_card(db, code);
      if (card_ids_.find(code) == card_ids_.end()) {
        throw std::runtime_error("Card not found in code list: " +
                                 std::to_string(code));
      }
    }

    auto it2 = cards_data_.find(code);
    if (it2 == cards_data_.end()) {
      cards_data_[code] = db_query_card_data(db, code);
    }
  }
}

inline uint32_t card_reader_callback(CardCode code, card_data *card) {
  auto it = cards_data_.find(code);
  if (it == cards_data_.end()) {
    // 容错：脚本里 AddCodeList 可能引用未收录的卡，返回空数据不中断对局
    *card = card_data{};
    card->code = code;
    return 0;
  }
  *card = it->second;
  return 0;
}

inline uint8_t *read_card_script(const std::string &path, int *lenptr) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *lenptr = 0;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  int len = file.tellg();
  file.seekg(0, std::ios::beg);
  uint8_t *buf = new uint8_t[len];
  file.read((char *)buf, len);
  *lenptr = len;
  return buf;
}

inline uint8_t *script_reader_callback(const char *name, int *lenptr) {
  std::string path(name);
  auto it = cards_script_.find(path);
  if (it == cards_script_.end()) {
    fmt::println("[script_reader_callback] Script not found: " + path);
    throw std::runtime_error("[script_reader_callback] Script not found: " + path);
  }
  *lenptr = it->second.len;
  return it->second.buf;
}

// ocgcore 的 Lua 错误回调：msg_type==1 时错误文本在 duel::strbuffer 里。
// 之前从没设置过 handler，所以所有 Lua 报错都被静默丢弃。
inline uint32_t message_handler_callback(intptr_t pduel, uint32_t msg_type) {
  if (msg_type == 1) {
    auto *p = reinterpret_cast<duel *>(pduel);
    fmt::println("[LUA ERROR] {}", p->strbuffer);
  }
  return 0;
}

static void init_module(const std::string &db_path,
                        const std::string &code_list_file,
                        const std::map<std::string, std::string> &decks) {
  // parse code from code_list_file
  SQLite::Database db(db_path, SQLite::OPEN_READONLY);

  auto start = std::chrono::steady_clock::now();

  std::ifstream file(code_list_file);
  std::string line;
  int i = 0;
  CardCode code;
  int has_script, script_len;
  while (std::getline(file, line)) {
    i++;
    std::istringstream iss(line);
    if (!(iss >> code >> has_script)) {
        std::cerr << "Failed to parse line in code_list: " << line << std::endl;
        continue;
    }
    card_ids_[code] = i;
    g_id_to_code[i] = code;
    cards_[code] = db_query_card(db, code);
    cards_data_[code] = db_query_card_data(db, code);
    if (has_script) {
      std::string path = "./script/c" + std::to_string(code) + ".lua";
      uint8_t *buf = read_card_script(path, &script_len);
      cards_script_[path] = {buf, script_len};
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  // fmt::println("load {} cards in {}ms", cards_data_.size(), milliseconds);

  for (const auto &[name, deck] : decks) {
    auto [main_deck, extra_deck, side_deck] = read_decks(deck);
    main_decks_[name] = main_deck;
    extra_decks_[name] = extra_deck;
    if (name[0] != '_') {
      deck_names_.push_back(name);
      deck_names_ids_[name] = deck_names_.size() - 1;
    }
  }

  for (auto &[name, deck] : extra_decks_) {
    sort_extra_deck(deck);
  }

  card_data card;
  cards_data_[0] = card;

  std::vector<std::string> preload = {
    // 旧 procedure.lua 仍会被引擎/旧卡脚本请求，先加载；随后新版 proc_*.lua 覆盖同名函数
    "./script/procedure.lua",
    "./script/constant.lua",
    "./script/utility.lua",
    "./script/archetype_setcode_constants.lua",
    "./script/card_counter_constants.lua",
    "./script/cards_specific_functions.lua",
    "./script/proc_normal.lua",
    "./script/proc_equip.lua",
    "./script/proc_fusion.lua",
    "./script/proc_fusion_spell.lua",
    "./script/proc_gemini.lua",
    "./script/proc_link.lua",
    "./script/proc_maximum.lua",
    "./script/proc_pendulum.lua",
    "./script/proc_persistent.lua",
    "./script/proc_ritual.lua",
    "./script/proc_rush.lua",
    "./script/proc_skill.lua",
    "./script/proc_spirit.lua",
    "./script/proc_synchro.lua",
    "./script/proc_union.lua",
    "./script/proc_workaround.lua",
    "./script/proc_xyz.lua",
    "./script/chain.lua",
    "./script/deprecated_functions.lua",
  };
  for (const auto &path : preload) {
    uint8_t *buf = read_card_script(path, &script_len);
    cards_script_[path] = {buf, script_len};
  }
  cards_script_["./script/c0.lua"] = {nullptr, 0};

  set_card_reader(card_reader_callback);
  set_script_reader(script_reader_callback);
  set_message_handler(message_handler_callback);
}

inline std::string getline() {
  char *line = nullptr;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, stdin);

  if (read != -1) {
    // Remove line ending character(s)
    if (line[read - 1] == '\n')
      line[read - 1] = '\0'; // Replace newline character with null terminator
    else if (line[read - 2] == '\r' && line[read - 1] == '\n') {
      line[read - 2] = '\0'; // Replace carriage return and newline characters
                             // with null terminator
      line[read - 1] = '\0';
    }

    std::string input(line);
    free(line);
    return input;
  } else {
    exit(0);
  }

  free(line);
  return "";
}

class Player {
  friend class YGOProEnvImpl;

protected:
  const std::string nickname_;
  const int init_lp_;
  const PlayerId duel_player_;
  const bool verbose_;

  bool seen_waiting_ = false;

public:
  Player(const std::string &nickname, int init_lp, PlayerId duel_player,
         bool verbose = false)
      : nickname_(nickname), init_lp_(init_lp), duel_player_(duel_player),
        verbose_(verbose) {}
  virtual ~Player() = default;

  void notify(const std::string &text) {
    if (verbose_) {
      fmt::println("{} {}", duel_player_, text);
    }
  }

  const int &init_lp() const { return init_lp_; }

  virtual int think(const std::vector<LegalAction> &actions) = 0;
};

class GreedyAI : public Player {
protected:
public:
  GreedyAI(const std::string &nickname, int init_lp, PlayerId duel_player,
           bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override { return 0; }
};

class RandomAI : public Player {
protected:
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dist_;

public:
  RandomAI(int max_options, int seed, const std::string &nickname, int init_lp,
           PlayerId duel_player, bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose), gen_(seed),
        dist_(0, max_options - 1) {}

  int think(const std::vector<LegalAction> &actions) override {
    return dist_(gen_) % actions.size();
  }
};

class HumanPlayer : public Player {
protected:
public:
  HumanPlayer(const std::string &nickname, int init_lp, PlayerId duel_player,
              bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override {
    while (true) {
      std::string input = getline();
      if (input == "quit") {
        exit(0);
      }
      int idx = -1;
      try {
        idx = std::stoi(input) - 1;
      } catch (std::invalid_argument &e) {
        fmt::println("{} Invalid input: {}", duel_player_, input);
        continue;
      }
      if (idx >= 0 && idx < actions.size()) {
        return idx;
      } else {
        fmt::println("{} Choose from {} actions", duel_player_, actions.size());
      }
    }
  }
};

class YGOProEnvFns {
public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("deck1"_.Bind(std::string("OldSchool")),
                    "deck2"_.Bind(std::string("OldSchool")), "player"_.Bind(-1),
                    "play_mode"_.Bind(std::string("bot")),
                    "verbose"_.Bind(false), "max_options"_.Bind(16),
                    "max_cards"_.Bind(80), "n_history_actions"_.Bind(16),
                    "record"_.Bind(false), "async_reset"_.Bind(false),
                    "greedy_reward"_.Bind(true), "timeout"_.Bind(600),
                    "oppo_info"_.Bind(false), "battle_features"_.Bind(false),
                    "chain_effect_probe"_.Bind(false), "chain_target_probe"_.Bind(false),
                    "max_steps"_.Bind(1000));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config &conf) {
    int n_action_feats = 14;   // 12 原有 + effect_semantics + target_relation
    return MakeDict(
        "obs:cards_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 73})),
        "obs:global_"_.Bind(Spec<uint8_t>({33})),
        // 连锁内容：(8 环 × 13 列) —— 0 link_no, 1 ctrl(0=me,1=opp),
        // 2-3 code, 4-7 category(32bit LE), 8 negated, 9 loc, 10 seq,
        // 11-12 effects(16bit LE)。训练侧由 ChainEncoder 编码，默认零初始化
        // 输出层，因此热启动旧权重时前向完全等价。
        "obs:chain_"_.Bind(Spec<uint8_t>({8, 13})),
        "obs:battle_"_.Bind(Spec<float>({conf["max_options"_], 24}, {-1.0f, 1.0f})),
        "obs:actions_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], n_action_feats})),
        "obs:h_actions_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], n_action_feats + 2})),
        "obs:mask_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 14})),
        "info:negation_preview"_.Bind(Spec<uint8_t>({conf["max_options"_]}, {0, 2})),
        "info:opening_ecclesia"_.Bind(Spec<int>({}, {0,1})),
        "info:droll_event"_.Bind(Spec<int>({conf["max_options"_], 3})),
        "info:chain_effect_slots"_.Bind(Spec<uint8_t>({8}, {0, 14})),
        "info:chain_target_context"_.Bind(Spec<uint8_t>({4})),
        "info:num_options"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})),
        "info:to_play"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_selfplay"_.Bind(Spec<int>({}, {0, 1})),
        "info:win_reason"_.Bind(Spec<int>({}, {-1, 1})),
        "info:step_time"_.Bind(Spec<double>({2})),
        "info:deck"_.Bind(Spec<int>({2})),
        // 单步「错误动作」标记：0=无 1=手坑通召 2=手坑自伤 3=自残攻击 4=手坑正确连锁
        // 只走 info 通道，不进 obs、不改 ret_reward_，因此不影响 done 判定与权重形状。
        "info:g_draws"_.Bind(Spec<int>({2,4}, {0,255})),
        "info:bad_action"_.Bind(Spec<int>({}, {0, 4})),
        // 阶段0：辉夜效果发动次数 / 简易融合产物被消耗 / 被自毁（本步计数）
        "info:kaguya_effect"_.Bind(Spec<int>({}, {0, 16})),
        "info:sf_consumed"_.Bind(Spec<int>({}, {0, 14})),
        "info:sf_destroyed"_.Bind(Spec<int>({}, {0, 14})),
        "info:wicat_convert"_.Bind(Spec<int>({}, {0, 7})),
        "info:start_used"_.Bind(Spec<int>({}, {0, 7})),
        "info:tail_used"_.Bind(Spec<int>({}, {0, 7})),
        "info:kaguya_summon0"_.Bind(Spec<int>({}, {0, 7})),
        "info:kaguya_summon1"_.Bind(Spec<int>({}, {0, 7})),
        "info:kaguya_search0"_.Bind(Spec<int>({}, {0, 7})),
        "info:kaguya_search1"_.Bind(Spec<int>({}, {0, 7})),
        "info:kaguya_bounce0"_.Bind(Spec<int>({}, {0, 7})),
        "info:kaguya_bounce1"_.Bind(Spec<int>({}, {0, 7})),
        "info:tail_fusion0"_.Bind(Spec<int>({}, {0, 7})),
        "info:tail_fusion1"_.Bind(Spec<int>({}, {0, 7})),
        "info:tail_oppmat0"_.Bind(Spec<int>({}, {0, 7})),
        "info:tail_oppmat1"_.Bind(Spec<int>({}, {0, 7})),
        "info:gyoku_grave0"_.Bind(Spec<int>({}, {0, 7})),
        "info:gyoku_grave1"_.Bind(Spec<int>({}, {0, 7})),
        "info:story_summon0"_.Bind(Spec<int>({}, {0, 7})),
        "info:story_summon1"_.Bind(Spec<int>({}, {0, 7})),
        "info:rename0"_.Bind(Spec<int>({}, {0, 7})),
        "info:rename1"_.Bind(Spec<int>({}, {0, 7})),
        "info:sum_sel"_.Bind(Spec<int>({}, {0, 31})),
        "info:sum_opp"_.Bind(Spec<int>({}, {0, 31})),
        "info:sum_gyoku"_.Bind(Spec<int>({}, {0, 31})),
        "info:sel_gyoku"_.Bind(Spec<int>({}, {0, 31})),
        "info:pick_gyoku"_.Bind(Spec<int>({}, {0, 31})),
        "info:freeze_cand"_.Bind(Spec<int>({}, {0, 31})),
        "info:story_cand"_.Bind(Spec<int>({}, {0, 31})),
        "info:extra_cand"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_ma"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_ma_all"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_fld"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_fld_all"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_wr"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_wr_all"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_ma_i0"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_ma_i1"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_ma_i2"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_ma_i3"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_fld_i0"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_fld_i1"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_fld_i2"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_fld_i3"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_wr_i0"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_wr_i1"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_wr_i2"_.Bind(Spec<int>({}, {0, 31})),
        "info:dsc_wr_i3"_.Bind(Spec<int>({}, {0, 31})),
        "info:rename_win"_.Bind(Spec<int>({}, {0, 31})),
        "info:rename_hit"_.Bind(Spec<int>({}, {0, 31})),
        "info:rename_clr"_.Bind(Spec<int>({}, {0, 31})),
        "info:obs_gyoku"_.Bind(Spec<int>({}, {0, 31})),
        "info:obs_spec_fill"_.Bind(Spec<int>({}, {0, 255})),
        "info:opp_fusion_mat"_.Bind(Spec<int>({}, {0, 7})),
        "info:tail_chain"_.Bind(Spec<int>({}, {0, 7})),
        "info:gyoku_id"_.Bind(Spec<int>({}, {0, 65535})),
        "info:rn_live"_.Bind(Spec<int>({}, {0, 31})),
        "info:tail_equip"_.Bind(Spec<int>({}, {0, 7})),
        "info:atk_tgt_total"_.Bind(Spec<int>({}, {0, 255})),
        "info:atk_tgt_zero"_.Bind(Spec<int>({}, {0, 255})),
        "info:def_tgt_total"_.Bind(Spec<int>({}, {0, 255})),
        "info:def_tgt_zero"_.Bind(Spec<int>({}, {0, 255})),
        "info:atk_seen"_.Bind(Spec<int>({}, {0, 65535})),
        "info:atk_changed"_.Bind(Spec<int>({}, {0, 65535})),
        "info:def_changed"_.Bind(Spec<int>({}, {0, 65535})),
        "info:start_set"_.Bind(Spec<int>({}, {0, 7})),
        "info:mill_id_a"_.Bind(Spec<int>({}, {0, 15049})),
        "info:mill_id_b"_.Bind(Spec<int>({}, {0, 15049})),
        "info:mill_cand_sel"_.Bind(Spec<int>({}, {0, 4})),
        "info:mill_cand_size"_.Bind(Spec<int>({}, {0, 64})),
        "info:mill_cand_start"_.Bind(Spec<int>({}, {0, 2})),
        "info:mill_cand_tail"_.Bind(Spec<int>({}, {0, 2})),
        "info:mill_cand_dsel"_.Bind(Spec<int>({}, {0, 4})),
        "info:mill_cand_dsize"_.Bind(Spec<int>({}, {0, 128})),
        "info:mill_cand_one"_.Bind(Spec<int>({}, {0, 2})),
        "info:mill_any_sel"_.Bind(Spec<int>({}, {0, 4})),
        "info:mill_any_size"_.Bind(Spec<int>({}, {0, 128})),
        "info:mill_any_start"_.Bind(Spec<int>({}, {0, 2})),
        "info:mill_any_tail"_.Bind(Spec<int>({}, {0, 2})),
        "info:pair_start_cand"_.Bind(Spec<int>({}, {0, 2})),
        "info:pair_start_pick"_.Bind(Spec<int>({}, {0, 2})),
        "info:pair_tail_cand"_.Bind(Spec<int>({}, {0, 2})),
        "info:pair_tail_pick"_.Bind(Spec<int>({}, {0, 2})),
        "info:pair_kg_cand"_.Bind(Spec<int>({}, {0, 4})),
        "info:pair_fd_cand"_.Bind(Spec<int>({}, {0, 4})),
        "info:pair_mg_cand"_.Bind(Spec<int>({}, {0, 4})),
        "info:pair_sn_cand"_.Bind(Spec<int>({}, {0, 4})),
        "info:nz_win"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_act"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_win_1"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_win_2"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_win_3"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_act_1"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_act_2"_.Bind(Spec<int>({}, {0, 255})),
        "info:nz_act_3"_.Bind(Spec<int>({}, {0, 255})),
        "info:hist_11"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_22"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_12"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_xx"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_p1"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_p2"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_p3"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_p4p"_.Bind(Spec<int>({}, {0, 64})),
        "info:hist_unk"_.Bind(Spec<int>({}, {0, 64})),
        "info:w12_all_win"_.Bind(Spec<int>({}, {0, 64})),
        "info:w12_all_p1"_.Bind(Spec<int>({}, {0, 64})),
        "info:w12_all_p2"_.Bind(Spec<int>({}, {0, 64})),
        "info:w12_m7_win"_.Bind(Spec<int>({}, {0, 64})),
        "info:w12_m7_p1"_.Bind(Spec<int>({}, {0, 64})),
        "info:w12_m7_p2"_.Bind(Spec<int>({}, {0, 64}))
      );
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config &conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})));
  }
};

using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;

enum PlayMode { kHuman, kSelfPlay, kRandomBot, kGreedyBot, kCount };

// parse play modes seperated by '+'
inline std::vector<PlayMode> parse_play_modes(const std::string &play_mode) {
  std::vector<PlayMode> modes;
  std::istringstream ss(play_mode);
  std::string token;
  while (std::getline(ss, token, '+')) {
    if (token == "human") {
      modes.push_back(kHuman);
    } else if (token == "self") {
      modes.push_back(kSelfPlay);
    } else if (token == "bot") {
      modes.push_back(kGreedyBot);
    } else if (token == "random") {
      modes.push_back(kRandomBot);
    } else {
      throw std::runtime_error("Unknown play mode: " + token);
    }
  }
  // human mode can't be combined with other modes
  if (std::find(modes.begin(), modes.end(), kHuman) != modes.end() &&
      modes.size() > 1) {
    throw std::runtime_error("Human mode can't be combined with other modes");
  }
  return modes;
}

// rules = 1, Traditional
// rules = 0, Default
// rules = 4, Link
// rules = 5, MR5
constexpr int32_t rules_ = 5;
constexpr int32_t duel_options_ = ((rules_ & 0xFF) << 16) + (0 & 0xFFFF);


class YGOProEnvImpl {
protected:
  const EnvSpec<YGOProEnvFns> spec_;

  constexpr static int init_lp_ = 8000;
  constexpr static int startcount_ = 5;
  constexpr static int drawcount_ = 1;

  const std::string deck1_;
  const std::string deck2_;

  std::vector<uint32_t> main_deck0_;
  std::vector<uint32_t> main_deck1_;
  std::vector<uint32_t> extra_deck0_;
  std::vector<uint32_t> extra_deck1_;

  std::string deck_name_[2] = {"", ""};
  std::string nickname_[2] = {"Alice", "Bob"};

  const std::vector<PlayMode> play_modes_;

  // if play_mode_ == 'bot' or 'human', player_ is the order of the ai player
  // -1 means random, 0 and 1 means the first and second player respectively
  const int player_;

  PlayMode play_mode_;
  bool verbose_ = false;

  PlayerId ai_player_;

  intptr_t pduel_ = 0;
  std::unique_ptr<Player> players_[2]; //  abstract class must be pointer

  std::uniform_int_distribution<uint64_t> dist_int_;
  bool done_{true};
  long step_count_{0};
  bool duel_started_{false};
  uint32_t eng_flag_{0};

  PlayerId winner_;
  uint8_t win_reason_;
  const bool greedy_reward_;

  int lp_[2];

  // turn player
  PlayerId tp_;
  int current_phase_;
  int turn_count_;

  int msg_;
  std::vector<LegalAction> legal_actions_;
  PlayerId to_play_;
  std::function<void(int)> callback_;
  int last_decision_msg_ = 0;

  uint8_t data_[4096];
  int dp_ = 0;
  int dl_ = 0;

  uint8_t query_buf_[4096];
  int qdp_ = 0;

  uint8_t resp_buf_[128];

  using IdleCardSpec = std::tuple<CardCode, std::string, uint32_t>;

  // chain
  PlayerId chaining_player_;

  // 连锁栈：MSG_CHAINING 逐条 push，MSG_CHAIN_END 清空。
  // 之前非 verbose 时整条消息被丢弃，模型完全看不到连锁状态。
  struct ChainEntry {
    PlayerId controler = 0;
    uint16_t effects = 0;     // 该卡 effects_ 全 16 位（bit8=无效, bit6=破坏）
    bool negated = false;
    uint32_t category = 0;    // cdb datas.category，权威的效果分类位
    CardCode code = 0;        // 发动源的卡号
    uint32_t desc = 0;        // 该效果的描述 id（原样保留，便于后续对齐）
    uint8_t loc = 0;          // 发动源所在区域（LOCATION_*）
    uint8_t seq = 0;          // 发动源所在序号
    uint8_t link_no = 0;      // 这是第几层连锁（1-based）
    int id = 0;               // 单调递增的连锁 id
  };
  std::vector<ChainEntry> chain_stack_;
  bool update_negation_source(ChainEntry& entry, uint32_t from, uint32_t to) {
    if (entry.loc != LOCATION_MZONE ||
        (from & 0xffffffu) != (uint32_t(entry.controler) | (LOCATION_MZONE << 8) | (uint32_t(entry.seq) << 16)))
      return false;
    entry.seq = (to & 0xffffu) == (from & 0xffffu) ? ((to >> 16) & 255) : 74;
    return true;
  }
  int chain_seq_ = 0;
  int rename_pending_chain_ = 0;
  std::map<int, std::string> pending_rename_;
  std::string last_opp_sel_spec_;
  uint32_t last_hint_value_ = 0;   // 最近一次 HINT_SELECTMSG 的 value
  bool chain_forced_ = false;     // 本次连锁里是否出现强制项
  uint8_t spe_count_ = 0;         // MSG_SELECT_CHAIN 头部的「还能响应数」

  const int n_history_actions_;

  // circular buffer for history actions
  TArray<uint8_t> history_actions_1_;
  TArray<uint8_t> history_actions_2_;
  int ha_p_1_ = 0;
  int ha_p_2_ = 0;

  std::unordered_set<std::string> revealed_;
  std::array<PublicHand, 2> public_hands_;

  // multi select
  int ms_idx_ = -1;
  int ms_mode_ = 0;
  // 攻击目标选择上下文：MSG_HINT(HINT_SELECTMSG, 549) 置位，
  // 随后 MSG_SELECT_CARD(min=max=1) 消费；pending_attacker_atk_ 为攻击者 ATK。
  bool attack_target_pending_ = false;
  int pending_attacker_atk_ = -1;
  int pending_attacker_ctrl_ = -1;   // 攻击者坐标（用于判定时实时查攻）
  int pending_attacker_loc_ = -1;
  int pending_attacker_seq_ = -1;
  // Public MSG_ATTACK source for target reselection, independent of reward bookkeeping.
  static constexpr uint32_t kBattleSourceMask = 0x00ffffff;
  uint32_t battle_attack_source_ = 0;
  int atk_tgt_total_ = 0;   // 攻击表侧攻击目标次数
  int atk_tgt_zero_ = 0;    // 其中 atk==0（看不到有效攻击力）
  int def_tgt_total_ = 0;   // 攻击表侧守备目标次数
  int def_tgt_zero_ = 0;    // 其中 def==0
  // 场上攻守变化统计
  int atk_seen_ = 0;        // 扫到的场上卡次数
  int atk_changed_ = 0;     // 其中 atk 与上次不同
  int def_changed_ = 0;
  int prev_atk_[2][16][16] = {};
  int prev_def_[2][16][16] = {};
  // 本次 multi_select 中被策略屏蔽的 spec（观测里 msg=0，模型不选）
  std::set<std::string> ms_masked_specs_;
  int ms_min_ = 0;
  int ms_max_ = 0;
  int ms_must_ = 0;
  std::vector<std::string> ms_specs_;
  std::vector<std::vector<int>> ms_combs_;
  ankerl::unordered_dense::map<std::string, int> ms_spec2idx_;
  std::vector<int> ms_r_idxs_;

  // discard hand cards
  bool discard_hand_ = false;

  // replay
  bool record_ = false;
  // 响应流在回放文件中的起始偏移，用于结束时回填 datasize
  long replay_data_start_ = 0;
  FILE* fp_ = nullptr;
  bool is_recording = false;

  // MSG_SELECT_COUNTER
  int n_counters_ = 0;

  std::mt19937 gen_;

  std::mt19937 duel_gen_;


public:
  // step return
  float ret_reward_ = 0;
  // 0=无 1=手坑通召 2=手坑自伤 3=自残攻击 4=手坑正确连锁 5=发动已被无效的卡
  std::array<std::array<int,4>,2> g_draws_{};
  int bad_action_ = 0;
  uint8_t pending_attacker_tags_ = 0;  // 本次攻击者的 tag，供目标选择阶段豁免

  // ---- 阶段0：简易融合「使用中」状态与辉夜效果计数 ----
  // c1845204.lua: tc:RegisterFlagEffect(1845204, RESET_EVENT+RESETS_STANDARD, 0, 1)
  // descon 用 GetFlagEffect(1845204)~=0 判断回合结束破坏。core 的 flag effect
  // 不经 MSG_QUERY 输出，所以按 (controler, mzone sequence) 自行追踪；离场即清。
  bool simple_fusion_pending_ = false;      // 简易融合已发动，等待 extra->mzone
  bool simple_fusion_on_mzone_[2][7] = {}; // [player][mzone seq]
  int kaguya_effect_ = 0;                  // 本步辉夜效果发动次数
  int sf_consumed_ = 0;                    // 本步：flag 卡在非 END 阶段离场（正确利用）
  int sf_destroyed_ = 0;                   // 本步：flag 卡在 END 阶段离场（自毁）
  // 威凯特当融合素材：mzone->grave 之后 <=2 步内，同一方终端 extra->mzone
  int wicat_convert_ = 0;                  // 本步完成次数
  long wicat_grave_step_ = -1;             // 威凯特最后一次送墓的 step
  PlayerId actor_ = 0;                     // 本步行动方（step 入口的 to_play_）
  // START / TAIL 的效果发动次数（我方），里程碑 2/3 改由事件驱动
  int start_used_ = 0;
  int tail_used_ = 0;
  int kaguya_summon_[2] = {0, 0};  // 通召辉夜次数（按 controler）
  int kaguya_search_[2] = {0, 0};  // e1（检索）发动次数（按 controler）
  int kaguya_bounce_[2] = {0, 0};  // e2（弹回）发动次数（按 controler）
  int tail_fusion_[2] = {0, 0};     // TAIL 融合发动次数（按 controler）
  int tail_oppmat_[2] = {0, 0};     // 其中用了「对方场上怪兽」的次数
  int gyoku_grave_[2] = {0, 0};     // 19144623（改名的妖精王子）从场上进墓
  int story_summon_[2] = {0, 0};    // 讲述者 4026187 登场
  int rename_[2] = {0, 0};          // 三种改卡名效果发动合计
  int sum_sel_ = 0;                 // MSG_SELECT_SUM 次数
  int sum_opp_ = 0;                 // 其中候选含对方场上的卡
  int sum_gyoku_ = 0;               // 其中候选含「改名的妖精王子」
  int sel_gyoku_ = 0;               // MSG_SELECT_CARD 候选含 19144623
  int pick_gyoku_ = 0;              // 实际选中 19144623
  int freeze_cand_ = 0;             // 候选全对方MZONE+单选（疑似改卡名目标）
  int story_cand_ = 0;              // 融合怪候选含讲述者
  int extra_cand_ = 0;              // 候选来自 EXTRA（融合怪选择）
  int dsc_ma_ = 0, dsc_ma_all_ = 0;     // 玛奇莉勒 e4 desc 命中 / 总数
  int dsc_fld_ = 0, dsc_fld_all_ = 0;   // 场地 e3
  int dsc_wr_ = 0, dsc_wr_all_ = 0;     // 编写者 e2
  int dsc_ma_idx_[4] = {0, 0, 0, 0};    // 玛奇莉勒 desc&0xF 分布
  int dsc_fld_idx_[4] = {0, 0, 0, 0};   // 场地 desc&0xF
  int dsc_wr_idx_[4] = {0, 0, 0, 0};    // 编写者 desc&0xF
  bool rename_pending_ = false;         // 三家改名效果刚发动
  int rename_ctrl_ = 0;                 // 该次改名的控制者
  int rename_win_ = 0;                  // 检测到改名目标选择
  int rename_hit_ = 0;                  // 成功解析并写入映射
  int rename_clr_ = 0;                  // 映射被清除
  int obs_gyoku_ = 0;                   // actions_ 里出现妖精王子
  int rn_live_ = 0;                     // renamed_ 当前非零条目数
  int rn_mismatch_ = 0;                 // 读到 renamed_ 但坐标对不上
  uint32_t renamed_[2][16][16] = {};    // [ctrl][loc][seq] -> 19144623
  bool tail_fusion_pending_ = false;
  bool tail_fusion_active_ = false;      // 正在结算 TAIL 的融合效果
  int opp_fusion_mat_ = 0;               // 本步：用对方怪兽当融合素材的次数
  int mill_tail_turn_ = -1;              // 堆墓选中 TAIL 的回合号
  int tail_chain_ = 0;                   // 本步：同回合链路达成（每局一次，按玩家分）
  bool tail_chain_done_[2] = {false, false};  // 每方各一次，避免对手吃掉名额
  int tail_fusion_ctrl_ = 0;
  int kaguya_desc_[4] = {0, 0, 0, 0};  // 辉夜 desc 的低位桶（调试）
  int tail_equip_ = 0;   // TAIL 从墓地装到魔陷区（e3 生效）
  int start_set_ = 0;    // START 从墓地盖放到魔陷区（e3 生效）
  // 本步 DECK -> GRAVE 的卡（内部 id，最多 2 张），用于统计堆墓分布
  int mill_id_a_ = 0;
  int mill_id_b_ = 0;
  int mill_cnt_ = 0;
  int mill_mine_cnt_ = 0;   // 本步「我方」DECK -> GRAVE 计数（奖励用，最多 2 张）
  // 堆墓选择（从卡组选妖精传姬卡）的候选统计
  int mill_cand_sel_ = 0;
  int mill_cand_size_ = 0;
  int mill_cand_start_ = 0;
  int mill_cand_tail_ = 0;
  bool sel_has_start_ = false;   // 最近一次卡组选卡是否含 START
  bool sel_has_tail_ = false;
  bool pend_pair_ = false;       // 已发生一次卡组选卡，等 DECK->GRAVE 确认是堆墓
  bool pend_s_ = false;
  bool pend_t_ = false;
  bool pend_kg_ = false;   // 窗口含辉夜 86937530
  bool pend_fd_ = false;   // 窗口含舞踏会场地 56725612
  bool pend_mg_ = false;   // 窗口含玛奇莉勒 19144622
  bool pend_sn_ = false;   // 窗口含白雪 55623480
  int pair_kg_cand_ = 0;   // 含辉夜的堆墓窗口数（条件选取率分母）
  int pair_fd_cand_ = 0;
  int pair_mg_cand_ = 0;
  int pair_sn_cand_ = 0;
  // 连锁对抗窗口：连锁上有对方环，且我方有可发动的无效/禁制类效果
  int nz_win_ = 0, nz_act_ = 0;
  int nz_win_by_len_[4] = {0, 0, 0, 0};
  int nz_act_by_len_[4] = {0, 0, 0, 0};
  bool nz_pend_ = false;
  int nz_pend_len_ = 0;
  int pair_start_cand_ = 0;
  int pair_start_pick_ = 0;
  int pair_tail_cand_ = 0;
  int pair_tail_pick_ = 0;
  int mill_any_sel_ = 0;      // 全在卡组 + 至少一张妖精传姬
  int mill_any_size_ = 0;
  int mill_any_start_ = 0;
  int mill_any_tail_ = 0;
  int mill_cand_dsel_ = 0;    // 只要求「全在卡组」
  int mill_cand_dsize_ = 0;
  int mill_cand_one_ = 0;     // 妖精传姬选择里 size==1 的次数
  // 卡组选卡直方图（口径：全部候选在卡组 + 至少一张妖精传姬）
  int hist_11_ = 0;           // (min,max) = (1,1)
  int hist_22_ = 0;           // (min,max) = (2,2)
  int hist_12_ = 0;           // (min,max) = (1,2)
  int hist_xx_ = 0;           // 其他 (min,max)
  int hist_p1_ = 0;           // 候选池 == 1
  int hist_p2_ = 0;           // 候选池 == 2
  int hist_p3_ = 0;           // 候选池 == 3
  int hist_p4p_ = 0;          // 候选池 >= 4
  int hist_unk_ = 0;          // 候选里存在无法解析的卡号
  int w12_all_win_ = 0;       // (min,max)==(1,2) 且 size>=2 的窗口
  int w12_all_p1_ = 0;        // 其中只选 1 张
  int w12_all_p2_ = 0;        // 其中选 2 张
  int w12_m7_win_ = 0;        // 含 m7 区候选（威凯特素材取除）的 (1,2) 窗口
  int w12_m7_p1_ = 0;
  int w12_m7_p2_ = 0;
  bool w12_pend_ = false;
  bool w12_m7_ = false;
  int ret_win_reason_ = 0;
  std::vector<uint32_t> last_sel_codes_;  // 最近一次 MSG_SELECT_CARD 的真实卡号
  std::map<std::string, uint32_t> cur_win_code_;  // 当前选择窗口 spec→真实卡号
  int cur_win_player_ = -1;  // 当前窗口归属玩家
  int obs_spec_fill_ = 0;  // 观测里仍需 spec 反查的候选数（应为 0）

  // 用消息真实卡号覆盖 action 的占位 cid（spec_to_card_id 对卡组位置会取错卡）
  void set_action_cid_from_code(const std::string &spec, uint32_t code) {
    if (code == 0) return;
    CardId cid = c_get_card_id(code);
    if (cid == 0) return;
    bool hit = false;
    for (auto &la : legal_actions_) {
      if (!la.finish_ && la.spec_ == spec) {
        la.cid_ = cid;
        la.src_code_ = code;
        hit = true;
        break;
      }
    }
    if (!hit && std::getenv("YGO_CIDMISS")) {
      static int miss_n = 0;
      if (miss_n++ < 15) {
        fmt::print(stderr, "[CIDMISS] msg={} spec='{}' code={} nact={}\n",
                   int(msg_), spec, code, int(legal_actions_.size()));
        for (auto &la : legal_actions_) {
          fmt::print(stderr, "    act spec='{}' cid={} finish={}\n", la.spec_,
                     int(la.cid_), la.finish_);
        }
      }
    }
  }
  std::vector<std::string> last_sel_specs_;  // 与之对应的 spec

  YGOProEnvImpl();

  YGOProEnvImpl(const EnvSpec<YGOProEnvFns> &spec, uint64_t env_seed)
      : spec_(spec), dist_int_(0, 0xffffffff),
        deck1_(spec.config["deck1"_]), deck2_(spec.config["deck2"_]),
        player_(spec.config["player"_]), players_{nullptr, nullptr},
        play_modes_(parse_play_modes(spec.config["play_mode"_])),
        verbose_(spec.config["verbose"_]), record_(spec.config["record"_]),
        n_history_actions_(spec.config["n_history_actions"_]),
        greedy_reward_(spec.config["greedy_reward"_]) {
    if (record_) {
      if (!verbose_) {
        throw std::runtime_error("record mode must be used with verbose mode and num_envs=1");
      }
    }
    // fmt::println("env_id: {}, seed: {}, x: {}", env_id_, seed_, dist_int_(gen_));

    gen_ = std::mt19937(env_seed);
    duel_gen_ = std::mt19937(dist_int_(gen_));

    int max_options = spec.config["max_options"_];
    int n_action_feats = spec.state_spec["obs:actions_"_].shape[1];
    history_actions_1_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
    history_actions_2_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
  }

  int max_options() const { return spec_.config["max_options"_]; }

  int max_cards() const { return spec_.config["max_cards"_]; }

  bool done() const { return done_; }

  bool random_mode() const { return play_modes_.size() > 1; }

  bool self_play() const {
    return std::find(play_modes_.begin(), play_modes_.end(), kSelfPlay) !=
           play_modes_.end();
  }

  void update_time_stat(const clock_t& start, uint64_t time_count, double& time_stat) {
    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    time_stat = time_stat * (static_cast<double>(time_count) /
      (time_count + 1)) + seconds / (time_count + 1);
  }

  // void update_time_stat(const std::string& deck, double seconds) {
  //   uint64_t& time_count = deck_time_count_[deck];
  //   double& time_stat = deck_time_[deck];
  //   time_stat = time_stat * (static_cast<double>(time_count) /
  //     (time_count + 1)) + seconds / (time_count + 1);
  //   time_count++;
  // }

  MDuel new_duel(uint32_t seed) {
    auto pduel = YGO_CreateDuel(seed);
    MDuel mduel{pduel, seed};

    for (PlayerId i = 0; i < 2; i++) {
      YGO_SetPlayerInfo(pduel, i, init_lp_, startcount_, drawcount_);
      auto [main_deck, extra_deck, deck_name] = load_deck(pduel, i, duel_gen_);
      if (i == 0) {
        mduel.main_deck0 = main_deck;
        mduel.extra_deck0 = extra_deck;
        mduel.deck_name0 = deck_name;
      } else {
        mduel.main_deck1 = main_deck;
        mduel.extra_deck1 = extra_deck;
        mduel.deck_name1 = deck_name;
      }
    }
    YGO_StartDuel(pduel, duel_options_);
    return mduel;
  }

  void reset() {
    opening_ecclesia_stage_=0; outside_draw_adds_ = {}; search_additions_.clear(); resolving_search_=0;
    g_draws_ = {};
    // clock_t start = clock();

    if (random_mode()) {
      play_mode_ = play_modes_[dist_int_(gen_) % play_modes_.size()];
    } else {
      play_mode_ = play_modes_[0];
    }

    if (play_mode_ != kSelfPlay) {
      if (player_ == -1) {
        ai_player_ = dist_int_(gen_) % 2;
      } else {
        ai_player_ = player_;
      }
    }

    turn_count_ = 0;
    ms_idx_ = -1;
    simple_fusion_pending_ = false;
    for (auto &row : simple_fusion_on_mzone_) {
      for (auto &v : row) v = false;
    }
    kaguya_effect_ = 0;
    sf_consumed_ = 0;
    sf_destroyed_ = 0;
    wicat_convert_ = 0;
    wicat_grave_step_ = -1;
    start_used_ = 0;
    tail_used_ = 0;
    for (int i = 0; i < 2; ++i) {
      kaguya_summon_[i] = 0;
      kaguya_search_[i] = 0;
      kaguya_bounce_[i] = 0;
      tail_fusion_[i] = 0;
      tail_oppmat_[i] = 0;
      gyoku_grave_[i] = 0;
      story_summon_[i] = 0;
      rename_[i] = 0;
    }
    sum_sel_ = 0;
    sum_opp_ = 0;
    sum_gyoku_ = 0;
    sel_gyoku_ = 0;
    pick_gyoku_ = 0;
    freeze_cand_ = 0;
    story_cand_ = 0;
    extra_cand_ = 0;
    dsc_ma_ = 0; dsc_ma_all_ = 0;
    dsc_fld_ = 0; dsc_fld_all_ = 0;
    dsc_wr_ = 0; dsc_wr_all_ = 0;
    for (int z = 0; z < 4; ++z) {
      dsc_ma_idx_[z] = 0;
      dsc_fld_idx_[z] = 0;
      dsc_wr_idx_[z] = 0;
    }
    rename_pending_ = false;
    opp_fusion_mat_ = 0;
    tail_chain_ = 0;
    mill_tail_turn_ = -1;
    for (int i = 0; i < 2; ++i) tail_chain_done_[i] = false;
    rename_win_ = 0;
    rename_hit_ = 0;
    rename_clr_ = 0;
    // obs_gyoku_ 累加，不清零
    rn_live_ = 0;
    rn_mismatch_ = 0;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 16; ++b)
        for (int c3 = 0; c3 < 16; ++c3)
          if (renamed_[a][b][c3] != 0) ++rn_live_;
    tail_fusion_pending_ = false;
    tail_equip_ = 0;
    start_set_ = 0;
    mill_id_a_ = 0;
    mill_id_b_ = 0;
    mill_cnt_ = 0;
    mill_mine_cnt_ = 0;
    mill_cand_sel_ = 0;
    mill_cand_size_ = 0;
    mill_cand_start_ = 0;
    mill_cand_tail_ = 0;
    mill_cand_dsel_ = 0;
    mill_cand_dsize_ = 0;
    mill_cand_one_ = 0;
    hist_11_ = 0;
    hist_22_ = 0;
    hist_12_ = 0;
    hist_xx_ = 0;
    hist_p1_ = 0;
    hist_p2_ = 0;
    hist_p3_ = 0;
    hist_p4p_ = 0;
    hist_unk_ = 0;
    w12_all_win_ = 0;
    w12_all_p1_ = 0;
    w12_all_p2_ = 0;
    w12_m7_win_ = 0;
    w12_m7_p1_ = 0;
    w12_m7_p2_ = 0;
    w12_pend_ = false;
    w12_m7_ = false;
    mill_any_sel_ = 0;
    mill_any_size_ = 0;
    mill_any_start_ = 0;
    mill_any_tail_ = 0;
    sel_has_start_ = false;
    sel_has_tail_ = false;
    pend_pair_ = false;
    pend_s_ = false;
    pend_t_ = false;
    pair_start_cand_ = 0;
    pair_start_pick_ = 0;
    pair_tail_cand_ = 0;
    pair_tail_pick_ = 0;
    pair_kg_cand_ = 0;
    pair_fd_cand_ = 0;
    pair_mg_cand_ = 0;
    pair_sn_cand_ = 0;
    actor_ = 0;
    nz_win_ = 0;
    nz_act_ = 0;
    nz_pend_ = false;
    nz_pend_len_ = 0;
    for (int _i = 0; _i < 4; ++_i) { nz_win_by_len_[_i] = 0; nz_act_by_len_[_i] = 0; }
    chain_stack_.clear();
    chain_forced_ = false;
    spe_count_ = 0;
    attack_target_pending_ = false;
    pending_attacker_atk_ = -1;
    pending_attacker_ctrl_ = -1;
    pending_attacker_loc_ = -1;
    pending_attacker_seq_ = -1;
    battle_attack_source_ = 0;
    atk_tgt_total_ = 0;
    atk_tgt_zero_ = 0;
    def_tgt_total_ = 0;
    def_tgt_zero_ = 0;
    atk_seen_ = 0;
    atk_changed_ = 0;
    def_changed_ = 0;
    for (int a2 = 0; a2 < 2; ++a2)
      for (int b2 = 0; b2 < 16; ++b2)
        for (int c2 = 0; c2 < 16; ++c2) {
          prev_atk_[a2][b2][c2] = 0;
          prev_def_[a2][b2][c2] = 0;
        }
    ms_masked_specs_.clear();

    public_hands_ = {};
    revealed_.clear();
    history_actions_1_.Zero();
    history_actions_2_.Zero();
    ha_p_1_ = 0;
    ha_p_2_ = 0;

    // clock_t _start = clock();

    intptr_t old_duel = pduel_;
    if (duel_started_) {
      YGO_EndDuel(pduel_);
    }
    MDuel mduel;
    mduel = new_duel(dist_int_(gen_));

    auto duel_seed = mduel.seed;
    pduel_ = mduel.pduel;

    deck_name_[0] = mduel.deck_name0;
    deck_name_[1] = mduel.deck_name1;
    main_deck0_ = mduel.main_deck0;
    extra_deck0_ = mduel.extra_deck0;
    main_deck1_ = mduel.main_deck1;
    extra_deck1_ = mduel.extra_deck1;

    for (PlayerId i = 0; i < 2; i++) {
      std::string nickname = i == 0 ? "Alice" : "Bob";
      if (i == ai_player_) {
        nickname = "Agent";
      }
      nickname_[i] = nickname;
      if ((play_mode_ == kHuman) && (i != ai_player_)) {
        players_[i] = std::make_unique<HumanPlayer>(nickname, init_lp_, i, verbose_);
      } else if (play_mode_ == kRandomBot) {
        players_[i] = std::make_unique<RandomAI>(max_options(), dist_int_(gen_), nickname, init_lp_, i, verbose_);
      } else {
        players_[i] = std::make_unique<GreedyAI>(nickname, init_lp_, i, verbose_);
      }
      lp_[i] = players_[i]->init_lp_;
    }

    if (record_) {
      if (is_recording && fp_ != nullptr) {
        fclose(fp_);
      }
      auto time_str = time_now();
      // Use last 4 digits of seed as unique id
      auto seed_ = duel_seed % 10000;
      std::string fname;
      while (true) {
        fname = fmt::format("./replay/a{} {:04d}.yrp", time_str, seed_);
        // check existence
        if (std::filesystem::exists(fname)) {
          seed_ = (seed_ + 1) % 10000;
        } else {
          break;
        } 
      }
      fp_ = fopen(fname.c_str(), "wb");
      if (!fp_) {
        throw std::runtime_error("Failed to open file for replay: " + fname);
      }

      is_recording = true;

      // YRP2 ExtendedReplayHeader（80 字节）：
      //   ReplayHeader base(32) + seed_sequence[8](32) + header_version(1) + value[15](15)
      // 数据保持不压缩（flag 不设 REPLAY_COMPRESSED），与客户端的读取逻辑兼容。
      ReplayHeader rh;
      rh.id = 0x32707279;              // YRP2
      rh.version = 0x00001362;
      rh.flag = REPLAY_UNIFORM;        // 0x10：REPLAY_UNIFORM，无压缩位
      rh.seed = 0;                     // YRP2 的 seed 固定 0，真实种子在 seed_sequence
      rh.datasize = 0;                 // 结束时回填
      rh.start_time = (unsigned int)time(nullptr);
      fwrite(&rh, sizeof(rh), 1, fp_);

      uint32_t seed_seq[SEED_COUNT] = {0};
      make_seed_seq(duel_seed, seed_seq);   // 与建局时完全一致
      fwrite(seed_seq, sizeof(seed_seq), 1, fp_);   // 32 字节
      uint8_t header_version = 1;
      fwrite(&header_version, 1, 1, fp_);           // 1 字节
      uint8_t pad15[15] = {0};
      fwrite(pad15, 15, 1, fp_);                    // 15 字节

      for (PlayerId i = 0; i < 2; i++) {
        uint16_t name[20];
        memset(name, 0, 40);
        std::string name_str = fmt::format("{} {}", nickname_[i], deck_name_[i]);
        if (name_str.size() > 20) {
          // truncate
          name_str = name_str.substr(0, 20);
        }
        fmt::println("name: {}", name_str);
        str_to_uint16(name_str.c_str(), name);
        fwrite(name, 40, 1, fp_);
      }

      ReplayWriteInt32(init_lp_);
      ReplayWriteInt32(startcount_);
      ReplayWriteInt32(drawcount_);
      ReplayWriteInt32(duel_options_);

      for (PlayerId i = 0; i < 2; i++) {
        auto &main_deck = i == 0 ? main_deck0_ : main_deck1_;
        auto &extra_deck = i == 0 ? extra_deck0_ : extra_deck1_;
        // 与 YGOMobile single_duel.cpp 一致：主卡组和额外卡组都按逆序写入
        ReplayWriteInt32(main_deck.size());
        for (int j = int(main_deck.size()) - 1; j >= 0; --j) {
          ReplayWriteInt32(main_deck[j]);
        }
        ReplayWriteInt32(extra_deck.size());
        for (int j = int(extra_deck.size()) - 1; j >= 0; --j) {
          ReplayWriteInt32(extra_deck[j]);
        }
      }
      replay_data_start_ = ftell(fp_);

    }

    duel_started_ = true;
    eng_flag_ = 0;
    winner_ = 255;
    win_reason_ = 255;
    discard_hand_ = false;

    done_ = false;
    step_count_ = 0;

    // update_time_stat(_start, reset_time_count_, reset_time_2_);
    // _start = clock();

    next();

    ret_reward_ = 0;
    ret_win_reason_ = 0;
  }

  void init_multi_select(
    int min, int max, int must, const std::vector<std::string> &specs,
    int mode = 0, const std::vector<std::vector<int>> &combs = {},
    const std::set<std::string> &masked_specs = {}) {
    ms_idx_ = 0;
    ms_mode_ = mode;
    ms_min_ = min;
    ms_max_ = max;
    ms_must_ = must;
    ms_specs_ = specs;
    ms_masked_specs_ = masked_specs;
    ms_r_idxs_.clear();
    ms_spec2idx_.clear();

    for (int j = 0; j < ms_specs_.size(); ++j) {
      const auto &spec = ms_specs_[j];
      ms_spec2idx_[spec] = j;
    }

    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        const auto &spec = ms_specs_[j];
        auto la = LegalAction::from_spec(spec);
        la.masked_ = ms_masked_specs_.count(spec) > 0;
        legal_actions_.push_back(la);
      }
    } else {
      // 只保留选卡数落在 [ms_min_, ms_max_] 内的组合。
      // combinations_with_weight2 只保证权重(等级)和等于目标值，不保证张数，
      // 原实现把全部组合都放进来，会造成两类死局：
      //   1) 组合长度 < min：最后一张选完就 finish，张数不足，core 回 MSG_RETRY；
      //   2) 组合长度 > max：一路推进超过 max，core 同样回 MSG_RETRY。
      ms_combs_.clear();
      for (const auto &c : combs) {
        if (static_cast<int>(c.size()) >= ms_min_ &&
            static_cast<int>(c.size()) <= ms_max_) {
          ms_combs_.push_back(c);
        }
      }
      if (ms_combs_.empty()) {
        // 目标值已由 must 卡满足（或本来就没有可选卡）：选 0 张即完成。
        // 交给模型作为唯一候选，避免在这里直接 YGO_SetResponseb 打乱 step 流程。
        if (ms_min_ == 0) {
          legal_actions_.push_back(LegalAction::finish());
          return;
        }
        throw std::runtime_error(fmt::format(
          "no comb in [{}, {}] for multi_select_2, combs: {}",
          ms_min_, ms_max_, combs.size()));
      }
      _callback_multi_select_2_prepare();
    }
  }

  void handle_multi_select() {
    legal_actions_.clear();
    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        if (ms_spec2idx_.find(ms_specs_[j]) != ms_spec2idx_.end()) {
          auto la = LegalAction::from_spec(ms_specs_[j]);
          la.masked_ = ms_masked_specs_.count(ms_specs_[j]) > 0;
          legal_actions_.push_back(la);
        }
      }
      // 门控实验开关：YGO_M7_FORCE=1 只取 1 个素材；=2 强制取满 2 个。
      // 仅作用于 m7 区（威凯特素材取除）的 (1,2) 窗口，默认关闭。
      const char *m7f_env = getenv("YGO_M7_FORCE");
      const char *m7fp_env = getenv("YGO_M7_FORCE_PLAYER");
      const int m7fp = (m7fp_env != nullptr) ? std::atoi(m7fp_env) : -1;
      const bool m7_side_ok = (m7fp < 0 || static_cast<int>(to_play_) == m7fp);
      const int m7force = (w12_pend_ && w12_m7_ && m7f_env != nullptr &&
                           m7_side_ok)
                              ? std::atoi(m7f_env)
                              : 0;
      if (m7force == 1 && ms_idx_ == ms_max_ - 1 && ms_idx_ >= ms_min_) {
        legal_actions_.clear();
        legal_actions_.push_back(LegalAction::finish());
      } else if (m7force == 2 && ms_idx_ == ms_max_ - 1 &&
                 ms_idx_ >= ms_min_) {
        for (auto it = legal_actions_.begin(); it != legal_actions_.end();) {
          if (it->finish_) {
            it = legal_actions_.erase(it);
          } else {
            ++it;
          }
        }
        if (legal_actions_.empty()) {
          legal_actions_.push_back(LegalAction::finish());
        }
      }
      if (ms_idx_ == ms_max_ - 1) {
        if (ms_idx_ >= ms_min_ && !legal_actions_.empty() &&
            legal_actions_.back().finish_) {
          legal_actions_.pop_back();
        }
        if (ms_idx_ >= ms_min_ && (m7force != 2 || legal_actions_.empty())) {
          legal_actions_.push_back(LegalAction::finish());
        }
        callback_ = [this](int idx) {
          _callback_multi_select(idx, true);
        };
      } else if (ms_idx_ >= ms_min_) {
        legal_actions_.push_back(LegalAction::finish());
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      } else {
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };    
      }
    } else {
      _callback_multi_select_2_prepare();
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    }
  }

  int get_ms_spec_idx(const std::string &spec) const {
    auto it = ms_spec2idx_.find(spec);
    if (it != ms_spec2idx_.end()) {
      return it->second;
    }
    // TODO(2): find the root cause
    // print ms_spec2idx
    show_deck(0);
    show_deck(1);
    show_buffer();
    show_turn();
    fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}, r_idx: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_, ms_r_idxs_);
    fmt::print("ms_spec2idx: ");
    for (const auto &[k, v] : ms_spec2idx_) {
      fmt::print("({}, {}), ", k, v);
    }
    fmt::print("\n");
    return -1;
    // throw std::runtime_error("Spec not found: " + spec);
  }

  void _callback_multi_select_2(int idx) {
    const auto &action = legal_actions_[idx];
    if (action.finish_) {
      // select_sum 的 min == 0 且没有可凑组合时，选 0 张就是唯一合法动作
      _callback_multi_select_2_finish();
      return;
    }
    idx = get_ms_spec_idx(action.spec_);
    if (idx == -1) {
      // TODO(2): find the root cause
      std::vector<std::string> specs;
      for (const auto &la : legal_actions_) {
        specs.push_back(la.spec_);
      }
      fmt::println("specs: {}, idx: {}, spec: {}", specs, idx, action.spec_);
      throw std::runtime_error("Spec not found");
    }
    ms_r_idxs_.push_back(idx);
    std::vector<std::vector<int>> combs;
    for (auto &c : ms_combs_) {
      if (c[0] == idx) {
        c.erase(c.begin());
        if (c.empty()) {
          // 组合长度已在 init_multi_select 里限定在 [min,max]，
          // 这里再确认一次，避免把张数不足的响应发给 core（core 会回 RETRY）。
          if (static_cast<int>(ms_r_idxs_.size()) >= ms_min_) {
            _callback_multi_select_2_finish();
            return;
          }
        } else {
          combs.push_back(c);
        }
      }
    }
    ms_idx_++;
    ms_combs_ = combs;
    if (ms_combs_.empty()) {
      throw std::runtime_error("multi_select_2 exhausted before finish");
    }
  }

  void _callback_multi_select_2_prepare() {
    std::set<int> comb;
    for (const auto &c : ms_combs_) {
      comb.insert(c[0]);
    }
    for (auto &i : comb) {
      const auto &spec = ms_specs_[i];
      legal_actions_.push_back(LegalAction::from_spec(spec));
    }
  }

  void _callback_multi_select_2_finish() {
    ms_idx_ = -1;
    resp_buf_[0] = ms_r_idxs_.size() + ms_must_;
    for (int i = 0; i < ms_must_; ++i) {
      resp_buf_[i + 1] = 0;
    }
    for (int i = 0; i < ms_r_idxs_.size(); ++i) {
      resp_buf_[i + ms_must_ + 1] = ms_r_idxs_[i];
    }
    YGO_SetResponseb(pduel_, resp_buf_);
  }

  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];
    if (!action.finish_ && !action.spec_.empty()) {
      {
        auto [pkc, pkl, pks, pkp] = spec_to_ls(to_play_, action.spec_);
        if (pkc >= 0 && pkc < 2 && pkl < 16 && pks < 16 &&
            renamed_[pkc][pkl][pks] != 0) {
          ++pick_gyoku_;
        }
      }
      if (rename_pending_ && !chain_stack_.empty()) {
        pending_rename_[chain_stack_.back().id] = action.spec_;
        rename_pending_ = false;
      } else {
        auto [qc, ql, qs, qp] = spec_to_ls(to_play_, action.spec_);
        if (ql == LOCATION_MZONE && qc != static_cast<int>(to_play_) &&
            (last_hint_value_ == 514u || last_hint_value_ == 573u)) {
          last_opp_sel_spec_ = action.spec_;
        }
      }
    }
    // fmt::println("Select card: {}, finish: {}", option, finish);
    if (action.finish_) {
      finish = true;
    } else {
      idx = get_ms_spec_idx(action.spec_);
      if (idx != -1) {
        ms_r_idxs_.push_back(idx);
      } else {
        // TODO(2): find the root cause
        std::vector<std::string> specs;
        for (const auto &la : legal_actions_) {
          specs.push_back(la.spec_);
        }
        fmt::println("specs: {}, idx: {}, spec: {}", specs, idx, action.spec_);
        ms_idx_ = -1;
        resp_buf_[0] = ms_min_;
        for (int i = 0; i < ms_min_; ++i) {
          resp_buf_[i + 1] = i;
        }
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
    }
    if (finish) {
      if (w12_pend_) {
        const int k = static_cast<int>(ms_r_idxs_.size());
        if (k == 1) ++w12_all_p1_;
        else if (k == 2) ++w12_all_p2_;
        if (w12_m7_) {
          if (k == 1) ++w12_m7_p1_;
          else if (k == 2) ++w12_m7_p2_;
        }
        w12_pend_ = false;
      }
      ms_idx_ = -1;
      resp_buf_[0] = ms_r_idxs_.size();
      for (int i = 0; i < ms_r_idxs_.size(); ++i) {
        resp_buf_[i + 1] = ms_r_idxs_[i];
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else {
      ms_idx_++;
      ms_spec2idx_.erase(action.spec_);
    }
  }

  void update_history_actions(PlayerId player, const LegalAction& action) {
    if (action.act_ == ActionAct::Cancel) {
      return;
    }
    auto& ha_p = player == 0 ? ha_p_1_ : ha_p_2_;
    auto& history_actions = player == 0 ? history_actions_1_ : history_actions_2_;
    ha_p--;
    if (ha_p < 0) {
      ha_p = n_history_actions_ - 1;
    }
    history_actions[ha_p].Zero();
    _set_obs_action(history_actions, ha_p, action);
    _set_obs_action_effect_semantics(history_actions, ha_p, action);
    _set_obs_action_target_relation(history_actions, ha_p, action);
    // Spec index not available in history actions
    history_actions[ha_p](0) = 0;
    // turn/phase 的列号必须跟随实际宽度（最后两列），不能硬编码
    const int nf = static_cast<int>(history_actions.Shape()[1]) - 2;
    // 与 features.py:encode_history_actions 一致：历史列存 turn_diff，clip 到 16，
    // 否则绝对 turn 超过 embed(20) 的词表会越界产生 NaN。
    history_actions[ha_p](nf) = static_cast<uint8_t>(std::min<int>(turn_count_, 16));
    history_actions[ha_p](nf + 1) = static_cast<uint8_t>(phase_to_id(current_phase_));
  }

  void show_deck(const std::vector<CardCode> &deck, const std::string &prefix) const {
    fmt::print("{} deck: [", prefix);
    for (int i = 0; i < deck.size(); i++) {
      fmt::print(" '{}'", c_get_card(deck[i]).name());
    }
    fmt::print(" ]\n");
  }

  void show_turn() const {
    fmt::println("turn: {}, phase: {}, tplayer: {}", turn_count_, phase_to_string(current_phase_), tp_);
  }

  void show_buffer() const {
    fmt::println("msg: {}, dp: {}, dl: {}", msg_to_string(msg_), dp_, dl_);
    for (int i = 0; i < dl_; ++i) {
      fmt::print("{:02x} ", data_[i]);
    }
    fmt::print("\n");
  }

  void show_deck(PlayerId player) const {
    fmt::print("Player {}'s deck: {}\n", player, deck_name_[player]);
    // show_deck(player == 0 ? main_deck0_ : main_deck1_, "Main");
    // show_deck(player == 0 ? extra_deck0_ : extra_deck1_, "Extra");
  }

  void show_history_actions(PlayerId player) const {
    const auto &ha = player == 0 ? history_actions_1_ : history_actions_2_;
    // print card ids of history actions
    for (int i = 0; i < n_history_actions_; ++i) {
      fmt::print("history {}\n", i);
      uint8_t msg_id = uint8_t(ha(i, 3));
      int msg = _msgs[msg_id - 1];
      fmt::print("msg: {},", msg_to_string(msg));
      uint8_t v1 = ha(i, 1);
      uint8_t v2 = ha(i, 2);
      CardId card_id = (static_cast<CardId>(v1) << 8) + static_cast<CardId>(v2);
      fmt::print(" {};", card_id);
      for (int j = 4; j < ha.Shape()[1]; j++) {
        fmt::print(" {}", uint8_t(ha(i, j)));
      }
      fmt::print("\n");
    }
  }


  // ---- 错误动作检测（只上报，不惩罚）----
  // 三类：手坑被当怪兽用、手坑无效自己的卡、自残攻击。
  // 全部在 callback_ 执行前判断，此时 legal_actions_ / msg_ 仍是「刚被选中的那个决策点」。
  // 「无效型」手坑：只有这几张才有「连锁对手=正确 / 连锁自己=错误」的语义。
  // 增殖的G / 小丑与锁鸟 / 尼比鲁 属于「限制对手特召检索」型，发动即有作用、
  // 除锁鸟外基本无副作用，不存在「乱发」这个错误类别，故不参与此判定。
  bool _is_response_handtrap(CardId cid) const {
    if (cid == 0) return false;
    auto it = g_id_to_code.find(cid);
    if (it == g_id_to_code.end()) return false;
    switch (static_cast<uint32_t>(it->second)) {
      case 14558127:  // 灰流丽
      case 97268402:  // 效果遮蒙者
      case 10045474:  // 无限泡影
      case 24224830:  // 墓穴的指名者
      case 65681983:  // 抹杀之指名者
        return true;
      default:
        return false;
    }
  }

  // 只有「怪兽类」手坑才可能被通召；泡影/指名者是魔陷，不在召唤候选里。
  bool _is_summonable_handtrap(CardId cid) const {
    if (cid == 0) return false;
    auto it = g_id_to_code.find(cid);
    if (it == g_id_to_code.end()) return false;
    switch (static_cast<uint32_t>(it->second)) {
      case 14558127:  // 灰流丽
      case 23434538:  // 增殖的G
      case 97268402:  // 效果遮蒙者
      case 94145021:  // 小丑与锁鸟
        return true;
      default:
        return false;
    }
  }

  void _detect_bad_action(int idx) {
    bad_action_ = 0;
    if (idx < 0 || idx >= static_cast<int>(legal_actions_.size())) return;
    const auto &la = legal_actions_[idx];

    // bad=5：发动效果时该卡已处于「被无效/禁止」状态。
    // 只在 step() 入口看发动前的 status（与观测 x[11] 的 negated 位同判据），
    // 所以「发动后被连锁泡影等无效」不计入。
    // 用 la.spec_ 定位到具体卡实例，避免同卡号多实例混淆。
    if (la.act_ == ActionAct::Activate && !la.spec_.empty()) {
      auto [dc, dloc, dseq, dpos] = spec_to_ls(to_play_, la.spec_);
      // get_card() 的 QUERY flags 不含 QUERY_STATUS，status_ 恒为 0；
      // 必须用 get_cards_in_location()（带 QUERY_STATUS）按区域+序列定位实例。
      for (const auto &dc_card : get_cards_in_location(dc, dloc)) {
        if (dc_card.sequence_ == dseq && dc_card.location_ == dloc) {
          if (dc_card.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) {
            bad_action_ = 5;
          }
          break;
        }
      }
      if (bad_action_ == 5) return;
    }

    if (msg_ == MSG_SELECT_IDLECMD &&
        (la.act_ == ActionAct::Summon || la.act_ == ActionAct::MSet)) {
      if (_is_summonable_handtrap(la.cid_)) bad_action_ = 1;
      return;
    }

    if (msg_ == MSG_SELECT_CHAIN && la.act_ == ActionAct::Activate &&
        _is_response_handtrap(la.cid_)) {
      // 链上最后一条是自己发的 -> 手坑在无效自己
      if (!chain_stack_.empty() && chain_stack_.back().controler == to_play_) {
        bad_action_ = 2;
      } else if (!chain_stack_.empty()) {
        bad_action_ = 4;   // 连锁对手，正例
      }
      return;
    }

    if (msg_ == MSG_SELECT_BATTLECMD && la.act_ == ActionAct::Attack) {
      auto [ac, aloc, aseq, apos] = spec_to_ls(to_play_, la.spec_);
      Card attacker = get_card(ac, aloc, aseq);
      if (attacker.code_ == 0) return;
      pending_attacker_tags_ = attacker.tags_;   // 供目标选择阶段豁免
      constexpr uint8_t BATTLE_TAGS = 0x7F;
      if (attacker.tags_ & BATTLE_TAGS) return;
      bool has_faceup = false;
      bool safe = false;
      for (int tseq = 0; tseq < 7; ++tseq) {
        Card target = get_card(1 - to_play_, LOCATION_MZONE, tseq);
        if (target.code_ == 0) continue;
        if (target.position_ & POS_FACEDOWN) continue;
        has_faceup = true;
        if ((target.position_ & POS_ATTACK) && attacker.attack_ >= target.attack_) {
          safe = true;
          break;
        }
        if ((target.position_ & POS_DEFENSE) && attacker.attack_ >= target.defense_) {
          safe = true;
          break;
        }
      }
      if (has_faceup && !safe) bad_action_ = 3;
      return;
    }

    // 修复 3：攻击目标选择阶段（core 用 HINT_SELECTMSG==549 标记）。
    // BATTLECMD 只能判「所有表侧都打不过」；当场上同时有「打得过」和「打不过」
    // 时，模型挑哪个只有这里才知道。
    // DirectAttack（含场地魔法「妖精传姬的舞蹈会」赋予的直击）不产生
    // MSG_SELECT_CARD，所以「打不过怪兽但能直击」永远不会进这个分支。
    if (msg_ == MSG_SELECT_CARD) {
      if (attack_target_pending_ && pending_attacker_atk_ >= 0) {
        constexpr uint8_t BATTLE_TAGS = 0x7F;
        bool exempt = (pending_attacker_tags_ & BATTLE_TAGS) != 0;
        // 实时查询攻击者 ATK（BATTLECMD 时的快照可能已被效果改变）
        int atk = pending_attacker_atk_;
        if (pending_attacker_ctrl_ >= 0 && pending_attacker_loc_ >= 0 &&
            pending_attacker_seq_ >= 0) {
          atk = get_card(pending_attacker_ctrl_, pending_attacker_loc_,
                         pending_attacker_seq_).attack_;
        }
        attack_target_pending_ = false;
        pending_attacker_atk_ = -1;
        pending_attacker_ctrl_ = -1;
        pending_attacker_loc_ = -1;
        pending_attacker_seq_ = -1;
        pending_attacker_tags_ = 0;
        if (!exempt && !la.spec_.empty()) {
          auto [tc, tloc, tseq, tpos] = spec_to_ls(to_play_, la.spec_);
          Card target = get_card(tc, tloc, tseq);
          if (target.code_ != 0 && !(target.position_ & POS_FACEDOWN)) {
            bool lose = false;
            if (target.position_ & POS_ATTACK) {
              ++atk_tgt_total_;
              if (target.attack_ == 0) ++atk_tgt_zero_;
              lose = atk < target.attack_;
            } else if (target.position_ & POS_DEFENSE) {
              ++def_tgt_total_;
              if (target.defense_ == 0) ++def_tgt_zero_;
              lose = atk < target.defense_;
            }
            if (lose) bad_action_ = 3;
          }
        }
      } else {
        attack_target_pending_ = false;
        pending_attacker_atk_ = -1;
        pending_attacker_tags_ = 0;
      }
      return;
    }
  }

  void step(int idx) {
    if (idx>=0 && idx<int(legal_actions_.size()) && opening_ecclesia_stage_>=0) {
      const auto& a=legal_actions_[idx];
      if (a.act_!=ActionAct::None && a.act_!=ActionAct::Cancel) {
        const bool name=a.cid_==c_get_card_id(55273560);
        if (!opening_ecclesia_stage_ && to_play_==tp_ && name && a.act_==ActionAct::Summon)
          opening_ecclesia_stage_=1;
        else if (opening_ecclesia_stage_==1 && to_play_==tp_ && name && a.act_==ActionAct::Activate)
          opening_ecclesia_stage_=2;
        else opening_ecclesia_stage_=-1;
      }
    }
    g_draws_ = {};
    g_waiting_external = false;
    kaguya_effect_ = 0;      // 阶段0：本步计数清零
    sf_consumed_ = 0;
    sf_destroyed_ = 0;
    wicat_convert_ = 0;
    start_used_ = 0;
    tail_used_ = 0;
    for (int i = 0; i < 2; ++i) {
      kaguya_summon_[i] = 0;
      kaguya_search_[i] = 0;
      kaguya_bounce_[i] = 0;
      tail_fusion_[i] = 0;
      tail_oppmat_[i] = 0;
      gyoku_grave_[i] = 0;
      story_summon_[i] = 0;
      rename_[i] = 0;
    }
    sum_sel_ = 0;
    sum_opp_ = 0;
    sum_gyoku_ = 0;
    sel_gyoku_ = 0;
    pick_gyoku_ = 0;
    freeze_cand_ = 0;
    story_cand_ = 0;
    extra_cand_ = 0;
    dsc_ma_ = 0; dsc_ma_all_ = 0;
    dsc_fld_ = 0; dsc_fld_all_ = 0;
    dsc_wr_ = 0; dsc_wr_all_ = 0;
    for (int z = 0; z < 4; ++z) {
      dsc_ma_idx_[z] = 0;
      dsc_fld_idx_[z] = 0;
      dsc_wr_idx_[z] = 0;
    }
    rename_pending_ = false;
    opp_fusion_mat_ = 0;
    tail_chain_ = 0;
    rename_win_ = 0;
    rename_hit_ = 0;
    rename_clr_ = 0;
    // obs_gyoku_ 累加，不清零
    rn_live_ = 0;
    rn_mismatch_ = 0;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 16; ++b)
        for (int c3 = 0; c3 < 16; ++c3)
          if (renamed_[a][b][c3] != 0) ++rn_live_;
    tail_fusion_pending_ = false;
    tail_equip_ = 0;
    start_set_ = 0;
    mill_id_a_ = 0;
    mill_id_b_ = 0;
    mill_cnt_ = 0;
    mill_mine_cnt_ = 0;
    mill_cand_sel_ = 0;
    mill_cand_size_ = 0;
    mill_cand_start_ = 0;
    mill_cand_tail_ = 0;
    mill_cand_dsel_ = 0;
    mill_cand_dsize_ = 0;
    mill_cand_one_ = 0;
    hist_11_ = 0;
    hist_22_ = 0;
    hist_12_ = 0;
    hist_xx_ = 0;
    hist_p1_ = 0;
    hist_p2_ = 0;
    hist_p3_ = 0;
    hist_p4p_ = 0;
    hist_unk_ = 0;
    mill_any_sel_ = 0;
    mill_any_size_ = 0;
    mill_any_start_ = 0;
    mill_any_tail_ = 0;
    // 注意：sel_has_start_/sel_has_tail_ 必须跨步保留
    // （MSG_SELECT_CARD 在上一 step 产生、响应在下一 step 执行）
    pair_start_cand_ = 0;
    pair_start_pick_ = 0;
    pair_tail_cand_ = 0;
    pair_tail_pick_ = 0;
    pair_kg_cand_ = 0;
    pair_fd_cand_ = 0;
    pair_mg_cand_ = 0;
    pair_sn_cand_ = 0;
    actor_ = to_play_;       // 本步行动方，用于判定「我方」
    // 简易融合发动：以「当前选中动作」为准（比 MSG_CHAINING 更可靠）
    if (idx >= 0 && idx < static_cast<int>(legal_actions_.size())) {
      const auto &la0 = legal_actions_[idx];
      if (la0.act_ == ActionAct::Activate) {
        auto it0 = g_id_to_code.find(la0.cid_);
        if (it0 != g_id_to_code.end() && it0->second == 1845204) {
          simple_fusion_pending_ = true;
        }
      }
    }
    if (nz_pend_ && idx >= 0 && idx < static_cast<int>(legal_actions_.size()) &&
        legal_actions_[idx].act_ == ActionAct::Activate) {
      ++nz_act_;
      ++nz_act_by_len_[nz_pend_len_];
    }
    nz_pend_ = false;
    // ---- 攻击目标决策 dump（YGO_ATKDUMP=路径，仅观测）----
    bool _atk_win = false;
    if (attack_target_pending_ && msg_ == MSG_SELECT_CARD) {
      for (const auto &_la : legal_actions_) {
        if (!_la.spec_.empty() && _la.spec_[0] == 'o') { _atk_win = true; break; }
      }
    }
    if (_atk_win) {
      const char *_ad = std::getenv("YGO_ATKDUMP");
      if (_ad) {
        std::vector<Card> _field;
        for (int _pi = 0; _pi < 2; ++_pi) {
          for (auto _loc : {LOCATION_MZONE, LOCATION_SZONE}) {
            auto _cs = get_cards_in_location(_pi, _loc);
            for (auto &_c : _cs) _field.push_back(_c);
          }
        }
        FILE *_fp = std::fopen(_ad, "a");
        if (_fp) {
          int _aatk_live = -1;
          if (pending_attacker_ctrl_ >= 0 && pending_attacker_loc_ >= 0) {
            auto _acs = get_cards_in_location(pending_attacker_ctrl_,
                                              pending_attacker_loc_);
            for (auto &_c : _acs) {
              if ((int)_c.sequence_ == pending_attacker_seq_) {
                _aatk_live = _c.attack_;
                break;
              }
            }
          }
          std::fprintf(_fp, "A %d %d %d %zu msg=%d actrl=%d aloc=%d aseq=%d alive=%d", turn_count_,
                       pending_attacker_atk_, idx, legal_actions_.size(), (int)msg_,
                       (int)pending_attacker_ctrl_, (int)pending_attacker_loc_,
                       (int)pending_attacker_seq_, _aatk_live);
          for (size_t _i = 0; _i < legal_actions_.size(); ++_i) {
            const auto &_la = legal_actions_[_i];
            auto _it = g_id_to_code.find(_la.cid_);
            uint32_t _code = (_it == g_id_to_code.end()) ? 0 : _it->second;
            const Card *_live = nullptr;
            for (const auto &_c : _field) {
              if (_c.code_ != _code) continue;
              if (_la.spec_.empty()) { _live = &_c; break; }
              if (_c.get_spec(false) == _la.spec_ || _c.get_spec(true) == _la.spec_) {
                _live = &_c;
                break;
              }
            }
            if (_live) {
              std::fprintf(_fp, " %u:%d:%d:%d:%d:%d:%d:%s", _code, _live->attack_,
                           _live->defense_, (int)_live->position_,
                           (int)_live->level(), (int)(_live->type() & 0xffff),
                           (int)_live->controler_, _la.spec_.c_str());
            } else {
              std::fprintf(_fp, " %u:-1:-1:-1:-1:-1:-1:%s", _code, _la.spec_.c_str());
            }
          }
          std::fprintf(_fp, "\n");
          std::fclose(_fp);
        }
      }
    }

    _detect_bad_action(idx);
    callback_(idx);
    update_history_actions(to_play_, legal_actions_[idx]);

    PlayerId player = to_play_;

    if (verbose_) {
      show_decision(idx);
    }

    if (ms_idx_ != -1) {
      handle_multi_select();
    } else {
      next();
    }

    step_count_++;
    if (!done_ && (step_count_ >= spec_.config["max_steps"_])) {
      PlayerId winner = lp_[0] > lp_[1] ? 0 : 1;
      _duel_end(winner, 0x01);
      done_ = true;
      legal_actions_.clear();
    }

    float reward = 0;
    int reason = 0;
    if (done_) {
      float base_reward;
      if (greedy_reward_) {
        if (winner_ == 0) {
          if (turn_count_ <= 1) {
            // FTK
            base_reward = 16.0;
          } else if (turn_count_ <= 3) {
            base_reward = 8.0;
          } else if (turn_count_ <= 5) {
            base_reward = 4.0;
          } else if (turn_count_ <= 7) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 7);
          }
        } else {
          if (turn_count_ <= 1) {
            base_reward = 8.0;
          } else if (turn_count_ <= 3) {
            base_reward = 4.0;
          } else if (turn_count_ <= 5) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 5);
          }
        }
      } else {
        base_reward = 1.0;
      }

      if (play_mode_ == kSelfPlay) {
        // if (spec_.config["oppo_info"_]) {
        if (false) {
          reward = winner_ == 0 ? base_reward : -base_reward;
        } else {
          // to_play_ is the previous player
          reward = winner_ == player ? base_reward : -base_reward;
        }
      } else {
        reward = winner_ == ai_player_ ? base_reward : -base_reward;
      }

      if (win_reason_ == 0x01) {
        reason = 1;
      } else if (win_reason_ == 0x02) {
        reason = -1;
      }

      if (record_) {
        if (!is_recording || fp_ == nullptr) {
          throw std::runtime_error("Recording is not started");
        }
        // 回填 datasize：响应流字节数（不含头部/名字/卡组）。
        long replay_end = ftell(fp_);
        if (replay_end > replay_data_start_) {
          unsigned int datasize =
              static_cast<unsigned int>(replay_end - replay_data_start_);
          fseek(fp_, static_cast<long>(offsetof(ReplayHeader, datasize)),
                SEEK_SET);
          fwrite(&datasize, sizeof(datasize), 1, fp_);
          fflush(fp_);
        }
        fclose(fp_);
        is_recording = false;
      }
    }


    // update_time_stat(start, step_time_count_, step_time_);
    // step_time_count_++;

    // double step_time = 0;
    // if (done_) {
    //   step_time = step_time_;
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }

    // if (done_) {
    //   update_time_stat(deck_name_[0], step_time_);
    //   update_time_stat(deck_name_[1], step_time_);
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }
    // if (step_time_count_ % 3000 == 0) {
    //   fmt::println("Step time: {:.3f}", step_time_ * 1000);
    // }
    ret_reward_ = reward;
    ret_win_reason_ = reason;
  }

  using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;
  using State =
      Dict<typename YGOProEnvSpec::StateKeys,
           typename SpecToTArray<typename YGOProEnvSpec::StateSpec::Values>::Type>;

  void WriteState(State &state) {
    for (int i=0; i<max_options(); ++i)
      for (int j=0; j<24; ++j) state["obs:battle_"_](i,j)=0;
    float reward = ret_reward_;
    int win_reason = ret_win_reason_;
    int n_options = legal_actions_.size();
    state["info:opening_ecclesia"_] = int(opening_ecclesia_stage_==2);
    droll_event(state["info:droll_event"_]);
    negation_preview(state["info:negation_preview"_]);
    auto &effect_slots = state["info:chain_effect_slots"_];
    for (int i = 0; i < 8; ++i) effect_slots(i) = 0;
    if (spec_.config["chain_effect_probe"_] && n_options > 0) {
      for (size_t i = 0; i < std::min<size_t>(8, chain_stack_.size()); ++i) {
        const auto &entry = chain_stack_[i];
        const uint32_t index = entry.desc & 0xf;
        // Zero means padding or no unambiguous card-local description.
        // Match the activated card before treating the low bits as a slot.
        if (entry.desc >= DESCRIPTION_LIMIT && (entry.desc >> 4) == entry.code &&
            index < 14 && c_get_card_id(entry.code) != 0) {
          effect_slots(i) = static_cast<uint8_t>(index + 1);
        }
      }
    }
    auto &target_context = state["info:chain_target_context"_];
    for (int i = 0; i < 4; ++i) target_context(i) = 0;
    if (spec_.config["chain_target_probe"_] && n_options > 0) {
      _set_chain_target_context(target_context);
    }
    state["reward"_] = reward;
    state["info:to_play"_] = int(to_play_);
    for (int s=0; s<2; ++s) for (int g=0; g<4; ++g) state["info:g_draws"_](s,g) = g_draws_[s][g];
    state["info:bad_action"_] = bad_action_;
    state["info:kaguya_effect"_] = kaguya_effect_;
    state["info:sf_consumed"_] = sf_consumed_;
    state["info:sf_destroyed"_] = sf_destroyed_;
    state["info:wicat_convert"_] = wicat_convert_;
    state["info:start_used"_] = start_used_;
    state["info:tail_used"_] = tail_used_;
    state["info:kaguya_summon0"_] = kaguya_summon_[0];
    state["info:kaguya_summon1"_] = kaguya_summon_[1];
    state["info:kaguya_search0"_] = kaguya_search_[0];
    state["info:kaguya_search1"_] = kaguya_search_[1];
    state["info:kaguya_bounce0"_] = kaguya_bounce_[0];
    state["info:kaguya_bounce1"_] = kaguya_bounce_[1];
    state["info:tail_fusion0"_] = tail_fusion_[0];
    state["info:tail_fusion1"_] = tail_fusion_[1];
    state["info:tail_oppmat0"_] = tail_oppmat_[0];
    state["info:tail_oppmat1"_] = tail_oppmat_[1];
    state["info:gyoku_grave0"_] = gyoku_grave_[0];
    state["info:gyoku_grave1"_] = gyoku_grave_[1];
    state["info:story_summon0"_] = story_summon_[0];
    state["info:story_summon1"_] = story_summon_[1];
    state["info:rename0"_] = rename_[0];
    state["info:rename1"_] = rename_[1];
    state["info:sum_sel"_] = sum_sel_;
    state["info:sum_opp"_] = sum_opp_;
    state["info:sum_gyoku"_] = sum_gyoku_;
    state["info:sel_gyoku"_] = sel_gyoku_;
    state["info:pick_gyoku"_] = pick_gyoku_;
    state["info:freeze_cand"_] = freeze_cand_;
    state["info:story_cand"_] = story_cand_;
    state["info:extra_cand"_] = extra_cand_;
    state["info:dsc_ma"_] = dsc_ma_;
    state["info:dsc_ma_all"_] = dsc_ma_all_;
    state["info:dsc_fld"_] = dsc_fld_;
    state["info:dsc_fld_all"_] = dsc_fld_all_;
    state["info:dsc_wr"_] = dsc_wr_;
    state["info:dsc_wr_all"_] = dsc_wr_all_;
    state["info:dsc_ma_i0"_] = dsc_ma_idx_[0];
    state["info:dsc_ma_i1"_] = dsc_ma_idx_[1];
    state["info:dsc_ma_i2"_] = dsc_ma_idx_[2];
    state["info:dsc_ma_i3"_] = dsc_ma_idx_[3];
    state["info:dsc_fld_i0"_] = dsc_fld_idx_[0];
    state["info:dsc_fld_i1"_] = dsc_fld_idx_[1];
    state["info:dsc_fld_i2"_] = dsc_fld_idx_[2];
    state["info:dsc_fld_i3"_] = dsc_fld_idx_[3];
    state["info:dsc_wr_i0"_] = dsc_wr_idx_[0];
    state["info:dsc_wr_i1"_] = dsc_wr_idx_[1];
    state["info:dsc_wr_i2"_] = dsc_wr_idx_[2];
    state["info:dsc_wr_i3"_] = dsc_wr_idx_[3];
    state["info:rename_win"_] = rename_win_;
    state["info:rename_hit"_] = rename_hit_;
    state["info:rename_clr"_] = rename_clr_;
    state["info:obs_gyoku"_] = obs_gyoku_;
    state["info:obs_spec_fill"_] = obs_spec_fill_;
    state["info:opp_fusion_mat"_] = opp_fusion_mat_;
    state["info:tail_chain"_] = tail_chain_;
    state["info:gyoku_id"_] = static_cast<int>(c_get_card_id(19144623u));
    state["info:rn_live"_] = rn_live_;
    state["info:tail_equip"_] = tail_equip_;
    state["info:atk_tgt_total"_] = atk_tgt_total_;
    state["info:atk_tgt_zero"_] = atk_tgt_zero_;
    state["info:def_tgt_total"_] = def_tgt_total_;
    state["info:def_tgt_zero"_] = def_tgt_zero_;
    state["info:atk_seen"_] = atk_seen_;
    state["info:atk_changed"_] = atk_changed_;
    state["info:def_changed"_] = def_changed_;
    state["info:start_set"_] = start_set_;
    state["info:mill_id_a"_] = mill_id_a_;
    state["info:mill_id_b"_] = mill_id_b_;
    state["info:mill_cand_sel"_] = mill_cand_sel_;
    state["info:mill_cand_size"_] = mill_cand_size_;
    state["info:mill_cand_start"_] = mill_cand_start_;
    state["info:mill_cand_tail"_] = mill_cand_tail_;
    state["info:mill_cand_dsel"_] = mill_cand_dsel_;
    state["info:mill_cand_dsize"_] = mill_cand_dsize_;
    state["info:mill_cand_one"_] = mill_cand_one_;
    state["info:mill_any_sel"_] = mill_any_sel_;
    state["info:mill_any_size"_] = mill_any_size_;
    state["info:mill_any_start"_] = mill_any_start_;
    state["info:mill_any_tail"_] = mill_any_tail_;
    state["info:pair_start_cand"_] = pair_start_cand_;
    state["info:pair_start_pick"_] = pair_start_pick_;
    state["info:pair_tail_cand"_] = pair_tail_cand_;
    state["info:pair_tail_pick"_] = pair_tail_pick_;
    state["info:pair_kg_cand"_] = pair_kg_cand_;
    state["info:pair_fd_cand"_] = pair_fd_cand_;
    state["info:pair_mg_cand"_] = pair_mg_cand_;
    state["info:pair_sn_cand"_] = pair_sn_cand_;
    state["info:nz_win"_] = nz_win_;
    state["info:nz_act"_] = nz_act_;
    state["info:nz_win_1"_] = nz_win_by_len_[1];
    state["info:nz_win_2"_] = nz_win_by_len_[2];
    state["info:nz_win_3"_] = nz_win_by_len_[3];
    state["info:nz_act_1"_] = nz_act_by_len_[1];
    state["info:nz_act_2"_] = nz_act_by_len_[2];
    state["info:nz_act_3"_] = nz_act_by_len_[3];
    state["info:hist_11"_] = hist_11_;
    state["info:hist_22"_] = hist_22_;
    state["info:hist_12"_] = hist_12_;
    state["info:hist_xx"_] = hist_xx_;
    state["info:hist_p1"_] = hist_p1_;
    state["info:hist_p2"_] = hist_p2_;
    state["info:hist_p3"_] = hist_p3_;
    state["info:hist_p4p"_] = hist_p4p_;
    state["info:hist_unk"_] = hist_unk_;
    state["info:w12_all_win"_] = w12_all_win_;
    state["info:w12_all_p1"_] = w12_all_p1_;
    state["info:w12_all_p2"_] = w12_all_p2_;
    state["info:w12_m7_win"_] = w12_m7_win_;
    state["info:w12_m7_p1"_] = w12_m7_p1_;
    state["info:w12_m7_p2"_] = w12_m7_p2_;
    state["info:is_selfplay"_] = int(play_mode_ == kSelfPlay);
    state["info:win_reason"_] = win_reason;
    if (reward != 0.0) {
      state["info:step_time"_][0] = 0;
      state["info:step_time"_][1] = 0;
      state["info:deck"_][0] = deck_names_ids_[deck_name_[0]];
      state["info:deck"_][1] = deck_names_ids_[deck_name_[1]];
    }

    if (n_options == 0) {
      state["info:num_options"_] = 1;
      state["obs:global_"_][32] = uint8_t(1);
      // if (step_count_ >= spec_.config["max_steps"_]) {
      //   fmt::println("Max steps reached return");
      // }
      return;
    }

    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;

    if (spec_.config["oppo_info"_]) {
      _set_obs_g_cards(state["obs:cards_"_], to_play_);
      auto [spec_infos_, loc_n_cards_] = _set_obs_mask(state["obs:mask_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    } else {
      auto [spec_infos_, loc_n_cards_] = _set_obs_cards(state["obs:cards_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    }

    // ATK/DEF causal ablation: YGO_ATK_ABL=zero|shuf (diagnostic only, default off)
    {
      static const int atk_abl = [] {
        const char *e = std::getenv("YGO_ATK_ABL");
        if (!e) return 0;
        std::string s(e);
        if (s == "zero") return 1;
        if (s == "shuf") return 2;
        return 0;
      }();
      if (atk_abl != 0) {
        auto &fc = state["obs:cards_"_];
        std::vector<int> mz;
        for (int i = 0; i < static_cast<int>(fc.Shape(0)); ++i) {
          if (fc(i, 2) != location_to_id(LOCATION_MZONE)) continue;
          if ((fc(i, 12) | fc(i, 13) | fc(i, 14) | fc(i, 15)) == 0) continue;
          mz.push_back(i);
        }
        if (atk_abl == 1) {
          for (int i : mz) {
            for (int j = 12; j <= 15; ++j) fc(i, j) = 0;
          }
        } else if (atk_abl == 2 && mz.size() > 1) {
          std::vector<std::array<uint8_t, 4>> vals;
          vals.reserve(mz.size());
          for (int i : mz) vals.push_back({fc(i, 12), fc(i, 13), fc(i, 14), fc(i, 15)});
          for (size_t k = 0; k < mz.size(); ++k) {
            const auto &v = vals[(k + 1) % vals.size()];
            for (int j = 0; j < 4; ++j) fc(mz[k], j + 12) = v[j];
          }
        }
      }
    }

    _set_obs_chain(state["obs:chain_"_], to_play_);
    _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);

    // we can't shuffle because idx must be stable in callback
    if (n_options > max_options()) {
      legal_actions_.resize(max_options());
    }

    n_options = legal_actions_.size();
    state["info:num_options"_] = n_options;

    for (int i = 0; i < n_options; ++i) {
      auto &action = legal_actions_[i];
      action.msg_ = msg_;
      const auto &spec = action.spec_;
      if (!spec.empty()) {
        auto [owner, loc, seq, pos] = spec_to_ls(to_play_, spec);
        if (owner != to_play_ && loc == LOCATION_HAND) {
          auto it = spec_infos.find(spec);
          action.spec_index_ = it == spec_infos.end() ? 0 : it->second.index;
          action.cid_ = it == spec_infos.end() ? 0 : it->second.cid;
        } else {
          const auto& spec_info = find_spec_info(spec_infos, spec);
          action.spec_index_ = spec_info.index;
          if (action.cid_ == 0) action.cid_ = spec_info.cid;
        }
      }
    }

    _set_obs_actions(state["obs:actions_"_], legal_actions_);
    _set_obs_battle(state["obs:battle_"_]);

    // ---- 连锁对抗窗口统计（不含训练、纯观测）----
    nz_pend_ = false;
    if (!chain_stack_.empty()) {
      int op_link = 0;
      for (const auto &ce : chain_stack_) {
        if (ce.controler != to_play_) ++op_link;
      }
      if (op_link > 0) {
        bool has_neg = false;
        for (int _i = 0; _i < n_options; ++_i) {
          if (legal_actions_[_i].masked_) continue;
          if (legal_actions_[_i].act_ != ActionAct::Activate) continue;
          uint8_t _sem = state["obs:actions_"_](_i, 12);
          if (_sem & ((1 << 1) | (1 << 2))) { has_neg = true; break; }
        }
        if (has_neg) {
          int _b = std::min(op_link, 3);
          ++nz_win_;
          ++nz_win_by_len_[_b];
          nz_pend_ = true;
          nz_pend_len_ = _b;
        }
      }
    }

    // write history actions

    auto ha_p = to_play_ == 0 ? ha_p_1_ : ha_p_2_;
    auto &history_actions = to_play_ == 0 ? history_actions_1_ : history_actions_2_;

    int offset = n_history_actions_ - ha_p;
    int n_h_action_feats = history_actions.Shape()[1];

    state["obs:h_actions_"_].Assign(
      (uint8_t *)history_actions[ha_p].Data(), n_h_action_feats * offset);
    state["obs:h_actions_"_][offset].Assign(
      (uint8_t *)history_actions.Data(), n_h_action_feats * ha_p);
    
    for (int i = 0; i < n_history_actions_; ++i) {
      if (uint8_t(state["obs:h_actions_"_](i, 3)) == 0) {
        break;
      }
      // turn 列号必须跟随实际宽度（最后两列），不能硬编码：
      // actions_ 由 12 列扩到 14 列（+ semantics + relation）后，
      // turn 从列 12 移到了列 14。写死 12 会覆盖 semantics 列。
      const int hnf = static_cast<int>(state["obs:h_actions_"_].Shape()[1]) - 2;
      int turn_diff = std::min(16, turn_count_ - uint8_t(state["obs:h_actions_"_](i, hnf)));
      state["obs:h_actions_"_](i, hnf) = static_cast<uint8_t>(turn_diff);
    }
  }

private:
  using SpecInfos = ankerl::unordered_dense::map<std::string, SpecInfo>;


  std::vector<CardCode> public_hand_rows(PlayerId owner, int count,
                                          int offset, SpecInfos &infos) {
    auto names = public_hands_[owner].names();
    if (names.size() > static_cast<size_t>(count))
      throw std::runtime_error("Public hand exceeds hand size");
    // A candidate may refer to a slot only while that slot was explicitly
    // identified. Rows themselves are always in canonical card-code order.
    if (!names.empty()) {
      for (int slot = 0; slot < count; ++slot) {
        auto code = public_hands_[owner].at(slot);
        if (!code) continue;
        auto row = std::lower_bound(names.begin(), names.end(), code) - names.begin();
        infos[ls_to_spec(LOCATION_HAND, slot, 0, true)] = {
            static_cast<uint16_t>(offset + row + 1), c_get_card_id(code)};
      }
    }
    return names;
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        // check this
        if (opponent && location != LOCATION_HAND && !revealed_.empty()) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          const auto names = location == LOCATION_HAND
              ? public_hand_rows(player, n_cards, offset, spec_infos) : std::vector<CardCode>{};
          for (auto i = 0; i < n_cards; i++) {
            if (static_cast<size_t>(i) < names.size()) {
              auto id = c_get_card_id(names[i]);
              f_cards(offset, 0) = id >> 8;
              f_cards(offset, 1) = id & 0xff;
            }
            f_cards(offset, 2) = location_to_id(location);
            f_cards(offset, 4) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_card_(f_cards, offset, c, hide, card_id);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_g_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      std::vector<uint8_t> configs = {
          LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE,
          LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED,
          LOCATION_EXTRA,
      };
      for (auto location : configs) {
        std::vector<Card> cards = get_cards_in_location(player, location);
        if (pi == 1 && location == LOCATION_HAND) {
          // The privileged critic keeps all physical cards. Align its ID mask
          // with the same canonical prefix used by the ordinary actor.
          auto dest = cards.begin();
          for (auto code : public_hands_[player].names()) {
            auto found = std::find_if(dest, cards.end(),
                                      [code](const Card &c) { return c.code_ == code; });
            if (found == cards.end()) throw std::runtime_error("Missing public hand card");
            std::rotate(dest, found, found + 1);
            ++dest;
          }
        }
        int n_cards = cards.size();
        for (int i = 0; i < n_cards; ++i) {
          const auto &c = cards[i];
          CardId card_id = c_get_card_id(c.code_);
          _set_obs_card_(f_cards, offset, c, false, card_id, false);
          offset++;
          if (offset == (spec_.config["max_cards"_] * 2 - 1)) {
            return;
          }
        }
      }
    }
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_mask(TArray<uint8_t> &mask, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        // check this
        if (opponent && location != LOCATION_HAND && !revealed_.empty()) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          const auto names = location == LOCATION_HAND
              ? public_hand_rows(player, n_cards, offset, spec_infos) : std::vector<CardCode>{};
          for (auto i = 0; i < n_cards; i++) {
            if (static_cast<size_t>(i) < names.size() && c_get_card_id(names[i]))
              mask(offset, 0) = 1;
            mask(offset, 1) = 1;
            mask(offset, 3) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_mask_(mask, offset, c, hide);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_card_(TArray<uint8_t> &f_cards, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      CardId cid_out = card_id;
      int rcc = static_cast<int>(c.controler_);
      int rll = static_cast<int>(location & 0x7f);
      int rss = static_cast<int>(c.sequence_);
      if (rcc >= 0 && rcc < 2 && rll < 16 && rss < 16 &&
          renamed_[rcc][rll][rss] != 0) {
        cid_out = renamed_[rcc][rll][rss];
      }
      f_cards(offset, 0) = static_cast<uint8_t>(cid_out >> 8);
      f_cards(offset, 1) = static_cast<uint8_t>(cid_out & 0xff);
    }
    f_cards(offset, 2) = location_to_id(location);

    uint8_t seq = 0;
    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      seq = c.sequence_ + 1;
    }
    f_cards(offset, 3) = seq;
    f_cards(offset, 4) = global ? c.controler_ : ((c.controler_ != to_play_) ? 1 : 0);
    if (overlay) {
      f_cards(offset, 5) = position_to_id(POS_FACEUP);
      f_cards(offset, 6) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          f_cards(offset, 5) = position_to_id(POS_FACEDOWN);
        }
        // else {
        //   fmt::println("location: {}, position: {}", location2str.at(location), position_to_string(c.position_));
        // }
      } else {
        f_cards(offset, 5) = position_to_id(c.position_);
      }
    }
    if (!hide) {
      f_cards(offset, 7) = attribute_to_id(c.attribute_);
      f_cards(offset, 8) = race_to_id(c.race_);
      f_cards(offset, 9) = c.level_;
      f_cards(offset, 10) = std::min(c.counter_, static_cast<uint32_t>(15));
      uint8_t card_status = static_cast<uint8_t>(
          (c.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) != 0);
      // 阶段0：x[11] 三值 —— 0=正常, 1=disabled/forbidden,
      // 2=「简易融合使用中」（回合结束会被破坏，应作为素材消耗掉）。
      // 网络侧这一列是 embed(3)，恰好第 3 个值可用，不扩维、不改权重形状。
      if (card_status == 0 && location == LOCATION_MZONE &&
          c.controler_ < 2 && c.sequence_ < 7 &&
          simple_fusion_on_mzone_[c.controler_][c.sequence_]) {
        card_status = 2;
      }
      f_cards(offset, 11) = card_status;
      if (location == LOCATION_MZONE && c.controler_ < 2 &&
          c.sequence_ < 16) {
        int pc = static_cast<int>(c.controler_);
        int ps = static_cast<int>(c.sequence_);
        ++atk_seen_;
        if (prev_atk_[pc][location][ps] != 0 &&
            prev_atk_[pc][location][ps] != c.attack_) {
          ++atk_changed_;
        }
        if (prev_def_[pc][location][ps] != 0 &&
            prev_def_[pc][location][ps] != c.defense_) {
          ++def_changed_;
        }
        prev_atk_[pc][location][ps] = c.attack_;
        prev_def_[pc][location][ps] = c.defense_;
      }
      // 原设计：高低字节，网络侧 bytes_to_bin 会重组成真实值
      auto [atk1, atk2] = float_transform(c.attack_);
      f_cards(offset, 12) = atk1;
      f_cards(offset, 13) = atk2;

      auto [def1, def2] = float_transform(c.defense_);
      f_cards(offset, 14) = def1;
      f_cards(offset, 15) = def2;

      auto type_ids = type_to_ids(c.type_);
      for (int j = 0; j < type_ids.size(); ++j) {
        f_cards(offset, 16 + j) = type_ids[j];
      }
      // 41..72 = cdb category 的 32 位，每位一列（与原版 type_ids 的表示一致）
      uint32_t cat = 0;
      {
        auto cit = g_card_category.find(static_cast<uint32_t>(c.code_));
        if (cit != g_card_category.end()) cat = cit->second;
      }
      for (int j = 0; j < 32; ++j) {
        f_cards(offset, 41 + j) = static_cast<uint8_t>((cat >> j) & 1);
      }
    }
  }

  void _set_obs_mask_(TArray<uint8_t> &mask, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      if (card_id != 0) {
        mask(offset, 0) = 1;
      }
    }
    mask(offset, 1) = 1;

    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      mask(offset, 2) = 1;
    }
    mask(offset, 3) = 1;
    if (overlay) {
      mask(offset, 4) = 1;
      mask(offset, 5) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          mask(offset, 4) = 1;
        }
      } else {
        mask(offset, 4) = 1;
      }
    }
    if (!hide) {
      mask(offset, 6) = 1;
      mask(offset, 7) = 1;
      mask(offset, 8) = 1;
      mask(offset, 9) = 1;
      mask(offset, 10) = 1;
      mask(offset, 11) = 1;
      mask(offset, 12) = 1;
      mask(offset, 13) = 1;
    }
  }

  void track_battle_source_move(uint32_t from, uint32_t to) {
    if (!battle_attack_source_ || (from & kBattleSourceMask) != battle_attack_source_) {
      return;
    }
    // Follow a move within the same player's monster zone. Leaving that zone
    // ends this binding, even if another monster later takes the same slot.
    battle_attack_source_ = (from & 0xffff) == (to & 0xffff) ? to & kBattleSourceMask : 0;
  }

  void _set_obs_battle(TArray<float> &f) {
    if (!spec_.config["battle_features"_]) return;
    const bool target_window = msg_ == MSG_SELECT_CARD && attack_target_pending_;
    if (msg_ != MSG_SELECT_BATTLECMD && !target_window) return;
    // Query current instances once: pending attack-time snapshots may be stale.
    const auto own = get_cards_in_location(to_play_, LOCATION_MZONE);
    const auto opposing = get_cards_in_location(1-to_play_, LOCATION_MZONE);
    auto view = [](const Card &c) {
      ygo_battle::PublicCard v;
      v.visible = !(c.position_ & POS_FACEDOWN);
      v.sequence = c.sequence_;
      if (v.visible) {
        v.attack_position = c.position_ & POS_ATTACK;
        v.defense_position = c.position_ & POS_DEFENSE;
        v.attack = c.attack_;
        // Link monsters have no DEF; the engine uses this field for link markers.
        v.defense = (c.type_ & TYPE_LINK) ? -1 : c.defense_;
      }
      return v;
    };
    auto find_seq = [](const std::vector<Card> &cards, int seq) -> const Card* {
      for (const auto &c: cards) if (c.location_==LOCATION_MZONE && static_cast<int>(c.sequence_)==seq) return &c;
      return nullptr;
    };
    const Card *target_attacker = nullptr;
    if (target_window) {
      if (pending_attacker_ctrl_ == to_play_ && pending_attacker_loc_ == LOCATION_MZONE) {
        target_attacker = find_seq(own, pending_attacker_seq_);
      } else if (battle_attack_source_ &&
                 (battle_attack_source_ & 0xff) == to_play_ &&
                 ((battle_attack_source_ >> 8) & 0xff) == LOCATION_MZONE) {
        target_attacker = find_seq(own, battle_attack_source_ >> 16);
      }
    }
    std::vector<ygo_battle::PublicCard> opponents;
    for (const auto &c: opposing) if (c.location_==LOCATION_MZONE) opponents.push_back(view(c));
    for (int i=0; i<static_cast<int>(legal_actions_.size()) && i<max_options(); ++i) {
      const auto &la=legal_actions_[i];
      if (la.masked_ || la.spec_.empty()) continue;
      const Card *ac=nullptr, *tc=nullptr;
      const bool direct=la.act_==ActionAct::DirectAttack;
      if (msg_==MSG_SELECT_BATTLECMD &&
          (la.act_==ActionAct::Attack || direct)) {
        auto [ctrl,loc,seq,pos]=spec_to_ls(to_play_,la.spec_);
        if (ctrl!=to_play_ || loc!=LOCATION_MZONE) continue;
        ac=find_seq(own,seq);
      } else if (target_window) {
        if (!target_attacker) continue;
        auto [ctrl,loc,seq,pos]=spec_to_ls(to_play_,la.spec_);
        if (ctrl==to_play_ || loc!=LOCATION_MZONE) continue;
        ac=target_attacker;
        tc=find_seq(opposing,seq);
        if (!tc) continue;
      } else continue;
      if (!ac) continue;
      const auto av=view(*ac);
      const auto tv=tc ? view(*tc) : ygo_battle::PublicCard{};
      const auto values=ygo_battle::encode(&av,tc ? &tv : nullptr,opponents,target_window,direct);
      for (int j=0;j<24;++j) f(i,j)=values[j];
    }
  }

  void droll_event(TArray<int>& result);
  std::array<int,2> outside_draw_adds_{};
  int opening_ecclesia_stage_ = 0;
  struct SearchAddition { int player, slot; uint32_t cause, shown; };
  std::vector<SearchAddition> search_additions_;
  uint32_t resolving_search_ = 0;
  void negation_preview(TArray<uint8_t>& result);

  void _set_chain_target_context(TArray<uint8_t> &context) {
    constexpr CardCode kKaguya = 86937530u;
    if (msg_ != MSG_SELECT_CHAIN || !duel_started_ || chain_stack_.empty()) return;
    const auto &observed = chain_stack_.back();
    if (observed.code != kKaguya || observed.desc != kKaguya * 16u + 1u ||
        observed.negated || observed.link_no == 0 || observed.link_no > 8) return;

    auto *field = reinterpret_cast<duel *>(pduel_)->game_field;
    if (field->core.current_chain.empty()) return;
    const auto &link = field->core.current_chain.back();
    if (link.chain_count != observed.link_no || link.triggering_player > 1 ||
        !link.target_cards || link.target_cards->container.size() != 1) return;
    auto *target = *link.target_cards->container.begin();
    const auto defender = 1 - link.triggering_player;
    if (target->current.location != LOCATION_MZONE || target->current.controler != defender ||
        !(target->current.position & POS_FACEUP) || !target->is_has_relation(link)) return;

    // The selected face-up monster is public. Only its controller can see
    // whether their own deck or extra deck can pay the script's alternative.
    context(0) = observed.link_no;
    context(1) = to_play_ == defender ? 1 : 2;
    context(2) = target->current.sequence + 1;
    if (to_play_ != defender) return;
    if (link.triggering_controler > 1 || link.triggering_location != LOCATION_MZONE) return;
    const auto &source_zone = field->player[link.triggering_controler].list_mzone;
    if (link.triggering_sequence >= source_zone.size()) return;
    auto *source = source_zone[link.triggering_sequence];
    if (!source || !source->is_has_relation(link) || source->get_status(STATUS_DISABLED | STATUS_FORBIDDEN)) {
      return;
    }
    context(3) = 1;
    if (!field->is_chain_disablable(0)) return;
    const uint32_t code = target->get_code();
    const auto matches = [code, &link](::card *candidate) {
      return (candidate->get_code() == code || candidate->get_another_code() == code) &&
             candidate->is_capable_cost_to_grave(link.triggering_player);
    };
    const auto &player = field->player[defender];
    if (std::any_of(player.list_main.begin(), player.list_main.end(), matches) ||
        std::any_of(player.list_extra.begin(), player.list_extra.end(), matches)) {
      context(3) = 2;
    }
  }

  void _set_obs_chain(TArray<uint8_t> &f, PlayerId to_play) {
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 13; ++j) f(i, j) = 0;
    int k = 0;
    for (const auto &ce : chain_stack_) {
      if (k >= 8) break;
      f(k, 0) = static_cast<uint8_t>(std::min<int>(ce.link_no, 7));
      f(k, 1) = ce.controler != to_play ? 1 : 0;
      // Use the same code_list CardId as cards_/actions_ and id_embed.
      // ce.code is the full official card code, not an embedding index.
      // Unknown/generated cards map to 0 via c_get_card_id().
      const CardId cid = c_get_card_id(ce.code);
      f(k, 2) = static_cast<uint8_t>(cid >> 8);
      f(k, 3) = static_cast<uint8_t>(cid & 0xff);
      for (int b = 0; b < 4; ++b)
        f(k, 4 + b) = static_cast<uint8_t>((ce.category >> (8 * b)) & 0xff);
      f(k, 8) = ce.negated ? 1 : 0;
      f(k, 9) = static_cast<uint8_t>(location_to_id(ce.loc & 0x7f));
      f(k, 10) = ce.seq;
      f(k, 11) = static_cast<uint8_t>(ce.effects >> 8);
      f(k, 12) = static_cast<uint8_t>(ce.effects & 0xff);
      ++k;
    }
  }

  void _set_obs_global(TArray<uint8_t> &feat, PlayerId player, const std::vector<int> &loc_n_cards) {
    uint8_t me = player;
    uint8_t op = 1 - player;

    // 原设计：高低字节，网络侧 bytes_to_bin 重组成真实 LP
    auto [me_lp_1, me_lp_2] = float_transform(lp_[me]);
    feat(0) = me_lp_1;
    feat(1) = me_lp_2;

    auto [op_lp_1, op_lp_2] = float_transform(lp_[op]);
    feat(2) = op_lp_1;
    feat(3) = op_lp_2;

    feat(4) = std::min(turn_count_, 16);
    feat(5) = phase_to_id(current_phase_);
    feat(6) = (me == 0) ? 1 : 0;
    feat(7) = (me == tp_) ? 1 : 0;

    for (int i = 0; i < loc_n_cards.size(); i++) {
      feat(8 + i) = static_cast<uint8_t>(loc_n_cards[i]);
    }

    // ---- 22-31: 连锁栈 + 可用性 ----
    int my_chain = 0, op_chain = 0, n_negate = 0, n_destroy = 0, n_negated = 0;
    for (const auto &ce : chain_stack_) {
      if (ce.controler == me) ++my_chain; else ++op_chain;
      // 统一到 cdb category：NEGATE 覆盖「无效发动」，DISABLE 覆盖「效果无效化」
      if (ce.category & (kCategoryNegate | kCategoryDisable)) ++n_negate;
      if (ce.category & kCategoryDestroy) ++n_destroy;
      if (ce.negated) ++n_negated;
    }
    int chain_count = static_cast<int>(chain_stack_.size());
    feat(22) = static_cast<uint8_t>(std::min(chain_count, 7));
    feat(23) = static_cast<uint8_t>(std::min(my_chain, 7));
    feat(24) = static_cast<uint8_t>(std::min(op_chain, 7));
    feat(25) = (!chain_stack_.empty() && chain_stack_.back().controler != me) ? 1 : 0;
    feat(26) = n_negate > 0 ? 1 : 0;
    feat(27) = n_destroy > 0 ? 1 : 0;
    feat(28) = chain_forced_ ? 1 : 0;
    feat(29) = n_negated > 0 ? 1 : 0;
    feat(30) = static_cast<uint8_t>(std::min<int>(spe_count_, 7));
    int n_act = 0;
    for (const auto &la : legal_actions_) {
      if (la.act_ == ActionAct::Activate && !la.masked_) ++n_act;
    }
    feat(31) = static_cast<uint8_t>(std::min(n_act, 15));
  }

  const SpecInfo& find_spec_info(SpecInfos &spec_infos, const std::string &spec) {
    auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) {
      // TODO(2): find the root cause
      // print spec2index
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_);
      fmt::println("Spec: {}, Spec2index:", spec);
      for (auto &[k, v] : spec_infos) {
        fmt::print("{}: {} {}, ", k, v.index, v.cid);
      }
      fmt::print("\n");
      // throw std::runtime_error("Spec not found: " + spec);
      spec_infos[spec] = {0, 0};
      return spec_infos[spec];
    }
    return it->second;
  }

  void _set_obs_action_spec(
    TArray<uint8_t> &feat, int i, int idx) {
    feat(i, 0) = static_cast<uint8_t>(idx);
  }

  void _set_obs_action_card_id(
    TArray<uint8_t> &feat, int i, CardId cid) {
    feat(i, 1) = static_cast<uint8_t>(cid >> 8);
    feat(i, 2) = static_cast<uint8_t>(cid & 0xff);
  }

  void _set_obs_action_msg(TArray<uint8_t> &feat, int i, int msg) {
    auto it = msg2id.find(msg);
    if (it == msg2id.end()) {
      static int warned = 0;
      if (warned < 8) {
        fmt::println("[WARN] unknown msg={} action_idx={} -> 0", msg, i);
        warned++;
      }
      feat(i, 3) = 0;
      return;
    }
    feat(i, 3) = it->second;
  }

  void _set_obs_action_act(TArray<uint8_t> &feat, int i, ActionAct act) {
    feat(i, 4) = static_cast<uint8_t>(act);
  }

  void _set_obs_action_finish(TArray<uint8_t> &feat, int i) {
    feat(i, 5) = 1;
  }

  void _set_obs_action_effect(TArray<uint8_t> &feat, int i, int effect) {
    // 0: None
    // 1: default
    // 2-15: card effect
    // 16+: system
    if (effect == -1) {
      effect = 0;
    } else if (effect == 0) {
      effect = 1;
    } else if (effect >= CARD_EFFECT_OFFSET) {
      effect = effect - CARD_EFFECT_OFFSET + 2;
    } else {
      effect = system_string_to_id(effect);
    }
    feat(i, 6) = static_cast<uint8_t>(effect);
  }

  void _set_obs_action_phase(TArray<uint8_t> &feat, int i, ActionPhase phase){
    feat(i, 7) = static_cast<uint8_t>(phase);
  }

  void _set_obs_action_position(TArray<uint8_t> &feat, int i, uint8_t position) {
    feat(i, 8) = position_to_id(position);
  }

  void _set_obs_action_number(TArray<uint8_t> &feat, int i, uint8_t number) {
    feat(i, 9) = number;
  }

  void _set_obs_action_place(TArray<uint8_t> &feat, int i, ActionPlace place) {
    feat(i, 10) = static_cast<uint8_t>(place);
  }

  void _set_obs_action_attrib(TArray<uint8_t> &feat, int i, uint8_t attrib) {
    feat(i, 11) = attribute_to_id(attrib);
  }

  void _set_obs_action(TArray<uint8_t> &feat, int i, const LegalAction &action) {
    auto msg = action.msg_;
    if (action.masked_) {
      // msg=0 会被模型的 a_mask 判为无效候选
      feat(i, 3) = 0;
    } else {
      _set_obs_action_msg(feat, i, msg);
    }
    _set_obs_action_card_id(feat, i, action.cid_);
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
        msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD) {
      if (action.finish_) {
        _set_obs_action_finish(feat, i);
      } else {
        _set_obs_action_spec(feat, i, action.spec_index_);
      }
    } else if (msg == MSG_SELECT_POSITION) {
      _set_obs_action_position(feat, i, action.position_);
    } else if (msg == MSG_SELECT_EFFECTYN) {
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_YESNO || msg == MSG_SELECT_OPTION) {
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (
      msg == MSG_SELECT_BATTLECMD ||
      msg == MSG_SELECT_IDLECMD ||
      msg == MSG_SELECT_CHAIN) {
      _set_obs_action_phase(feat, i, action.phase_);
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_PLACE || msg == MSG_SELECT_DISFIELD) {
      _set_obs_action_place(feat, i, action.place_);
    } else if (msg == MSG_ANNOUNCE_CARD) {
      // card id, already set
    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, action.attribute_);
    } else if (msg == MSG_ANNOUNCE_NUMBER) {
      _set_obs_action_number(feat, i, action.number_);
    } else {
      static int warned2 = 0;
      if (warned2 < 8) {
        fmt::println("[WARN] unsupported action msg={} action_idx={}", msg, i);
        warned2++;
      }
      feat(i, 3) = 0;
      return;
    }
  }

  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    int offset = 0;
    bool opponent = false;
    if (spec[0] == 'o') {
      player = 1 - player;
      opponent = true;
      offset++;
    }
    auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
    if (opponent && loc == LOCATION_HAND)
      return c_get_card_id(public_hands_[player].at(seq));
    if (opponent) {
      bool hidden_for_opponent = true;
      if (
        loc == LOCATION_MZONE || loc == LOCATION_SZONE ||
        loc == LOCATION_GRAVE || loc == LOCATION_REMOVED) {
        hidden_for_opponent = false;
      }
      if (revealed_.size() != 0) {
        hidden_for_opponent = false;
      }
      if (hidden_for_opponent) {
        return 0;
      }
      Card c = get_card(player, loc, seq);
      bool hide = c.position_ & POS_FACEDOWN;
      if (revealed_.find(spec) != revealed_.end()) {
        hide = false;
      }
      CardId card_id = 0;
      if (!hide) {
        card_id = c_get_card_id(c.code_);
      }
    }
    return c_get_card_id(get_card_code(player, loc, seq));
  }

  void _set_obs_actions(TArray<uint8_t> &feat, const std::vector<LegalAction> &actions) {
    for (int i = 0; i < actions.size(); ++i) {
      // SELECT_CARD 等多候选场景不会填 cid_，这里统一用 spec 反查补上，
      // 否则模型看到的候选全是「0 号卡」，无法区分要选哪张。
      LegalAction la = actions[i];
      if (std::getenv("YGO_CIDTRACE") && !la.spec_.empty() &&
          (msg_ == MSG_SELECT_CARD || msg_ == MSG_SELECT_UNSELECT_CARD ||
           msg_ == MSG_SELECT_TRIBUTE || msg_ == MSG_SELECT_SUM)) {
        static int tr = 0;
        if (tr++ < 60) {
          auto tw = cur_win_code_.find(la.spec_);
          auto tcode = g_id_to_code.find(la.cid_);
          fmt::print(stderr,
                     "[CIDTRACE] msg={} spec='{}' obs_code={} win_code={} src={}\n",
                     int(msg_), la.spec_,
                     tcode == g_id_to_code.end() ? 0u : tcode->second,
                     tw == cur_win_code_.end() ? 0u : tw->second,
                     int(la.src_code_));
        }
      }
      bool filled_from_spec = false;
      if (la.cid_ == 0 && !la.spec_.empty()) {
        la.cid_ = spec_to_card_id(la.spec_, to_play_);
        filled_from_spec = true;
        if (std::getenv("YGO_OBSCHK")) ++obs_spec_fill_;
      }
      bool renamed_hit = false;
      if (!la.spec_.empty()) {
        auto [mc, ml, ms2, mp] = spec_to_ls(to_play_, la.spec_);
        if (mc >= 0 && mc < 2 && ml < 16 && ms2 < 16 &&
            renamed_[mc][ml][ms2] != 0) {
          la.cid_ = renamed_[mc][ml][ms2];
          renamed_hit = true;
        }
      }
      _set_obs_action(feat, i, la);
      if (std::getenv("YGO_OBSCHK") && !filled_from_spec && !renamed_hit &&
          !la.finish_ && !la.spec_.empty() && to_play_ == cur_win_player_) {
        auto itc2 = g_id_to_code.find(la.cid_);
        uint32_t c2 = (itc2 == g_id_to_code.end())
                          ? 0u
                          : static_cast<uint32_t>(itc2->second);
        uint32_t want = 0u;
        auto wit = cur_win_code_.find(la.spec_);
        if (wit != cur_win_code_.end()) want = wit->second;
        if (la.src_code_ != 0 && c2 != la.src_code_) {
          static int nbad = 0;
          if (nbad++ < 12) {
            fmt::print(stderr,
                       "[OBSCHK] msg={} tp={} spec={} cid={} code={} src={}\n",
                       int(msg_), to_play_, la.spec_, int(la.cid_), c2,
                       int(la.src_code_));
          }
        }
      }
      if (la.cid_ == 2937u) ++obs_gyoku_;
      if (!la.spec_.empty()) {
        auto [qc, ql, qs, qp] = spec_to_ls(to_play_, la.spec_);
        if (qc >= 0 && qc < 2 && ql < 16 && qs < 16 &&
            renamed_[qc][ql][qs] == 0) {
          // 该坐标没有映射；如果是「候选含妖精王子」却没命中，就是坐标问题
        }
      }
      _set_obs_action_effect_semantics(feat, i, la);
      _set_obs_action_target_relation(feat, i, la);
    }
  }

  // 12: 该动作的卡在 cdb category 下的粗语义（整卡级，训练/推理都能复现）
  void _set_obs_action_effect_semantics(
    TArray<uint8_t> &feat, int i, const LegalAction &action) {
    feat(i, 12) = 0;
    auto iit = g_id_to_code.find(action.cid_);
    if (iit == g_id_to_code.end()) return;
    auto cit = g_card_category.find(static_cast<uint32_t>(iit->second));
    if (cit == g_card_category.end()) return;
    uint32_t cat = cit->second;
    uint8_t sem = 0;
    if (cat & kCategoryDestroy) sem |= 1 << 0;
    if (cat & kCategoryDisable) sem |= 1 << 1;
    if (cat & kCategoryNegate) sem |= 1 << 2;
    if (cat & 0x20000) sem |= 1 << 3;   // CATEGORY_SEARCH
    if (cat & 0x200) sem |= 1 << 4;     // CATEGORY_SPECIAL_SUMMON
    if (cat & 0x10000) sem |= 1 << 5;   // CATEGORY_DRAW
    if (cat & 0x20) sem |= 1 << 6;      // CATEGORY_TOGRAVE
    if (cat & 0x8) sem |= 1 << 7;       // CATEGORY_TOHAND
    feat(i, 12) = sem;
  }

  // 13: 本候选与连锁发动源的关系（0 无连锁 / 1 是场上发动源 / 2 发动源在场外
  //     / 3 同侧其它 / 4 对侧其它）
  void _set_obs_action_target_relation(
    TArray<uint8_t> &feat, int i, const LegalAction &action) {
    feat(i, 13) = 0;
    if (chain_stack_.empty() || action.spec_.empty()) return;
    bool any_on_field = false;
    for (const auto &ce : chain_stack_) {
      const bool onfield = (ce.loc & (LOCATION_MZONE | LOCATION_SZONE)) != 0;
      if (onfield) any_on_field = true;
      auto sp = ls_to_spec(ce.loc, ce.seq, 0, ce.controler != to_play_);
      if (sp == action.spec_) {
        feat(i, 13) = onfield ? 1 : 2;
        return;
      }
    }
    if (!any_on_field) { feat(i, 13) = 2; return; }
    feat(i, 13) = (action.spec_[0] == 'o') ? 4 : 3;
  }


  void str_to_uint16(const char* src, uint16_t* dest) {
      for (int i = 0; i < strlen(src); i += 1) {
        dest[i] = src[i];
      }

      // Add null terminator
      dest[strlen(src) + 1] = '\0';
  }

  void ReplayWriteInt8(int8_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt32(int32_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  // ygopro-core API
  // 与客户端 ReplayMode::StartDuel 的 create_duel_v2(seed_sequence) 保持一致：
  // 用 std::seed_seq(8 个 uint32) 初始化，而不是单值 seed。
  // 同一个 duel_seed 派生出的 8 个值也会原样写进回放头。
  static void make_seed_seq(uint32_t seed, uint32_t out[SEED_COUNT]) {
    std::mt19937 gen(seed);
    for (int i = 0; i < SEED_COUNT; ++i) {
      out[i] = static_cast<uint32_t>(gen());
    }
  }

  intptr_t YGO_CreateDuel(uint32_t seed) {
    duel* pduel = new duel();
    uint32_t seq[SEED_COUNT] = {0};
    make_seed_seq(seed, seq);
    pduel->random.seed(seq, SEED_COUNT);
    pduel->rng_version = 2;   // 与 create_duel_v2 一致
    return (intptr_t)pduel;
  }

  void YGO_SetPlayerInfo(intptr_t pduel, int32_t playerid, int32_t lp, int32_t startcount, int32_t drawcount) const {
    set_player_info(pduel, playerid, lp, startcount, drawcount);
  }

  void YGO_NewCard(intptr_t pduel, uint32_t code, uint8_t owner, uint8_t playerid, uint8_t location, uint8_t sequence, uint8_t position) const {
    new_card(pduel, code, owner, playerid, location, sequence, position);
  }

  void YGO_StartDuel(intptr_t pduel, int32_t options) const {
    start_duel(pduel, options);
  }

  void YGO_EndDuel(intptr_t pduel) const {
    // end_duel(pduel);
    duel* pd = (duel*)pduel;
    delete pd;
  }

  int32_t YGO_GetMessage(intptr_t pduel, uint8_t* buf) {
    return get_message(pduel, buf);
  }

  uint32_t YGO_Process(intptr_t pduel) {
    return process(pduel);
  }

  int32_t YGO_QueryCard(intptr_t pduel, uint8_t playerid, uint8_t location, uint8_t sequence, int32_t query_flag, uint8_t* buf) {
    return query_card(pduel, playerid, location, sequence, query_flag, buf, 0);
  }

  int32_t YGO_QueryFieldCount(intptr_t pduel, uint8_t playerid, uint8_t location) {
    return query_field_count(pduel, playerid, location);
  }

  int32_t YGO_QueryFieldCard(intptr_t pduel, uint8_t playerid, uint8_t location, uint32_t query_flag, uint8_t* buf) {
    return query_field_card(pduel, playerid, location, query_flag, buf, 0);
  }

  void YGO_SetResponsei(intptr_t pduel, int32_t value) {
    last_decision_msg_ = msg_;
    if (record_) {
      ReplayWriteInt8(4);
      ReplayWriteInt32(value);
    }
    set_responsei(pduel, value);
  }

  void YGO_SetResponseb(intptr_t pduel, uint8_t* buf) {
    last_decision_msg_ = msg_;
    if (record_) {
      switch (msg_) {
        case MSG_SORT_CARD:
          ReplayWriteInt8(1);
          fwrite(buf, 1, 1, fp_);
          break;
        case MSG_SELECT_COUNTER:
          ReplayWriteInt8(2 * n_counters_);
          fwrite(buf, 2 * n_counters_, 1, fp_);
          break;
        case MSG_SELECT_PLACE:
        case MSG_SELECT_DISFIELD:
          ReplayWriteInt8(3);
          fwrite(buf, 3, 1, fp_);
          break;
        default:
          ReplayWriteInt8(buf[0] + 1);
          fwrite(buf, buf[0] + 1, 1, fp_);
          break;
      }
    }
    set_responseb(pduel, buf);
  }

  // ygopro-core API

  void show_decision(int idx) {
    std::string s;
    const auto& a = legal_actions_[idx];
    if (!a.spec_.empty()) {
      s = a.spec_;
    } else if (a.place_ != ActionPlace::None) {
      s = action_place_to_string(a.place_);
    } else if (a.position_ != 0) {
      s = position_to_string(a.position_);
    } else {
      s = fmt::format("{}", a);
    }
    fmt::print("Player {} chose \"{}\" in {}\n", to_play_, s, legal_actions_);
  }

  std::tuple<std::vector<CardCode>, std::vector<CardCode>, std::string>
  load_deck(
    intptr_t pduel, PlayerId player, std::mt19937& gen, bool shuffle = true) const {
    std::string deck_name = player == 0 ? deck1_ : deck2_;

    if (deck_name == "random") {
      // generate random deck name
      std::uniform_int_distribution<uint64_t> dist_int(0,
                                                       deck_names_.size() - 1);
      deck_name = deck_names_[dist_int(gen)];
    }

    std::vector<CardCode> main_deck = main_decks_.at(deck_name);
    std::vector<CardCode> extra_deck = extra_decks_.at(deck_name);

    if (verbose_) {
      fmt::println("{} {}: {}, main({}), extra({})", player, nickname_[player],
        deck_name, main_deck.size(), extra_deck.size());
    }

    if (shuffle) {
      std::shuffle(main_deck.begin(), main_deck.end(), gen);
    }

    // add main deck in reverse order following ygopro
    // but since we have shuffled deck, so just add in order

    // 与 YGOMobile 一致：逆序放入卡组（配合 new_card 的插入语义）
    for (int i = int(main_deck.size()) - 1; i >= 0; --i) {
      YGO_NewCard(pduel, main_deck[i], player, player, LOCATION_DECK, 0,
               POS_FACEDOWN_DEFENSE);
    }

    // add extra deck in reverse order following ygopro
    for (int i = int(extra_deck.size()) - 1; i >= 0; --i) {
      YGO_NewCard(pduel, extra_deck[i], player, player, LOCATION_EXTRA, 0,
               POS_FACEDOWN_DEFENSE);
    }

    return {main_deck, extra_deck, deck_name};
  }

  // 按区域取 query_flag，与 srvpro2 的 refresh-query.ts 完全一致
  static uint32_t zone_query_flag(uint8_t loc) {
    if (loc == LOCATION_MZONE) return 0x881fff;
    if (loc == LOCATION_SZONE) return 0xe81fff;
    if (loc == LOCATION_HAND) return 0x681fff;
    if (loc == LOCATION_GRAVE) return 0x081fff;
    if (loc == LOCATION_EXTRA) return 0xe81fff;
    return 0xf81fff;
  }

  // 从 query data 里取 position（用于判断是否盖放）
  static uint32_t q_get_position(const uint8_t* d, uint32_t clen) {
    if (clen < 8) return 0;
    uint32_t flag = 0;
    std::memcpy(&flag, d + 4, 4);
    if (!(flag & 0x2)) return 0;  // QUERY_POSITION
    size_t off = 8;
    if (flag & 0x1) off += 4;  // 跳过 QUERY_CODE
    if (off + 4 > clen) return 0;
    uint32_t pos = 0;
    std::memcpy(&pos, d + off, 4);
    return pos;
  }

  // 只把 code 置 0，保留 position/atk/def 等（对手仍能知道有卡、攻守）
  static void q_clear_code(uint8_t* d, uint32_t clen) {
    if (clen < 12) return;
    uint32_t flag = 0;
    std::memcpy(&flag, d + 4, 4);
    if (!(flag & 0x1)) return;  // 无 QUERY_CODE
    std::memset(d + 8, 0, 4);
  }

  // 参照 YGOPro 服务端 / srvpro：每批消息后补一组全量场地同步，
  // 让 WindBot 的 ClientField 维持准确状态。
  void append_update_data_msgs() {
    if (!g_update_data_enabled || pduel_ == 0) {
      return;
    }
    // EXTRA 必须包含：同调候选都在额外卡组，缺了它占位卡永远是 Id=0，
    // SwordsoulExecutor 的同调条件无法匹配 CardId。
    static const uint8_t locs[] = {
        LOCATION_MZONE, LOCATION_SZONE, LOCATION_HAND, LOCATION_EXTRA};
    static std::vector<uint8_t> buf;
    buf.resize(0x10000);
    for (uint8_t p = 0; p < 2; ++p) {
      for (uint8_t loc : locs) {
        int32_t len = YGO_QueryFieldCard(pduel_, p, loc, zone_query_flag(loc),
                                        buf.data());
        if (len <= 0) {
          continue;
        }
        std::vector<uint8_t> msg;
        msg.reserve(3 + len);
        msg.push_back(6);  // MSG_UPDATE_DATA
        msg.push_back(p);
        msg.push_back(loc);
        msg.insert(msg.end(), buf.begin(), buf.begin() + len);
        // 对手(player=1)隐藏：按区域与表示形式，只清 code 保留其他
        if (p == 1) {
          size_t off = 3;
          while (off + 4 <= msg.size()) {
            uint32_t clen = 0;
            std::memcpy(&clen, msg.data() + off, 4);
            if (clen < 4 || off + clen > msg.size()) {
              break;
            }
            bool hide = false;
            if (loc == LOCATION_GRAVE) {
              hide = false;  // 墓地永不隐藏
            } else if (loc == LOCATION_HAND) {
              hide = true;  // 手牌隐藏
            } else if (loc & (LOCATION_MZONE | LOCATION_SZONE)) {
              // 场上：只有盖放才隐藏，表侧完全可见
              hide = (q_get_position(msg.data() + off, clen) &
                      POS_FACEDOWN) != 0;
            } else {
              hide = true;  // DECK / EXTRA / REMOVED
            }
            if (hide) {
              q_clear_code(msg.data() + off, clen);
            }
            off += clen;
          }
        }
        g_mirror_msgs.emplace_back(std::move(msg));
      }
    }
  }

  void next() {
    while (duel_started_) {
      if (eng_flag_ == PROCESSOR_END) {
        break;
      }
      uint32_t res = YGO_Process(pduel_);
      dl_ = res & PROCESSOR_BUFFER_LEN;
      eng_flag_ = res & PROCESSOR_FLAG;

      if (dl_ == 0) {
        continue;
      }
      YGO_GetMessage(pduel_, data_);
      // 关键：桥接逻辑只在「外部对手（WindBot）」模式下执行。
      // 训练时 g_external_opponent=false，绝不能让 g_mirror_msgs 被写入：
      // 它是全局无锁 std::vector，而 num_envs 个工作线程会并发调用 next()，
      // 并发 emplace_back 会破坏 vector 内部指针 → 扩容时 delete 坏地址
      // → "double free or corruption (out)"。
      // 11/8 16:39 那次训练能跑，正是因为当时 next() 里还没有这段代码。
      if (g_external_opponent) {
        // 每条消息单独入列（一帧一条，WindBot 只处理帧内第一条）
        // 时序对齐 srvpro2 的 shouldRefreshFirst：
        // 对需要客户端响应的 SELECT_*/ANNOUNCE_* 消息，必须先刷新场地再发请求，
        // 否则 WindBot 会基于上一帧的旧状态做决策（不同调、不召唤、时点错）。
        const uint8_t first_msg = data_[0];
        const bool is_response_msg =
            (first_msg >= 10 && first_msg <= 16) ||   // SELECT_BATTLECMD..SELECT_CHAIN
            first_msg == 18 || first_msg == 19 ||      // SELECT_PLACE / POSITION
            first_msg == 20 || first_msg == 22 ||      // SELECT_TRIBUTE / COUNTER
            first_msg == 23 || first_msg == 26 ||      // SELECT_SUM / UNSELECT
            first_msg == 140 || first_msg == 141 ||    // ANNOUNCE_RACE / ATTRIB
            first_msg == 142 || first_msg == 143;      // ANNOUNCE_CARD / NUMBER
        if (is_response_msg) {
          append_update_data_msgs();
        }
        g_mirror_msgs.emplace_back(data_, data_ + dl_);
        if (!is_response_msg) {
          append_update_data_msgs();
        }
      }
      dp_ = 0;
      while ((dp_ != dl_) || (ms_idx_ != -1)) {
        if (ms_idx_ != -1) {
          handle_multi_select();
        } else {
          handle_message();
          if (legal_actions_.empty()) {
            continue;
          }
        }
        g_num_legal_actions = static_cast<int>(legal_actions_.size());
        if ((play_mode_ == kSelfPlay) || (to_play_ == ai_player_)) {
          if (legal_actions_.size() == 1) {
            callback_(0);
            auto la = legal_actions_[0];
            la.msg_ = msg_;
            if (la.cid_ == 0 && !la.spec_.empty()) {
              la.cid_ = spec_to_card_id(la.spec_, to_play_);
            }
            update_history_actions(to_play_, la);
            if (verbose_) {
              show_decision(0);
            }
          } else {
            return;
          }
        } else if (g_external_opponent) {
          // 对手回合：保存原始 buffer 并交回 Python（WindBot 需要协议原文）
          g_external_buffer.assign(data_, data_ + dl_);
          g_waiting_external = true;
          return;
        } else {
          auto idx = players_[to_play_]->think(legal_actions_);
          callback_(idx);
          if (verbose_) {
            show_decision(idx);
          }
        }
      }
    }
    done_ = true;
    legal_actions_.clear();
  }

  uint8_t read_u8() { return data_[dp_++]; }

  uint16_t read_u16() {
    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);
    dp_ += 2;
    return v;
  }

  uint32_t read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(data_ + dp_);
    dp_ += 4;
    return v;
  }

  uint32_t q_read_u8() {
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint32_t q_read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  CardCode get_card_code(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card_code] Invalid card");
    }
    qdp_ += 8;
    return q_read_u32();
  }

  Card get_card(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_POSITION |
                    QUERY_LEVEL | QUERY_RANK | QUERY_LSCALE | QUERY_RSCALE |
                    QUERY_LINK;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      show_deck(0);
      show_deck(1);
      show_turn();
      show_buffer();
      auto s = fmt::format("[get_card] Invalid card (bl <= 0), player: {}, loc: {}, seq: {}", player, loc, seq);
      throw std::runtime_error(s);
    }
    uint32_t f = q_read_u32();
    if (f == LEN_EMPTY) {
      return Card();
    }
    f = q_read_u32();
    CardCode code = q_read_u32();
    Card c = c_get_card(code);
    uint32_t position = q_read_u32();
    c.set_location(position);
    uint32_t level = q_read_u32();
    if ((level & 0xff) > 0) {
      c.level_ = level & 0xff;
    }
    uint32_t rank = q_read_u32();
    if ((rank & 0xff) > 0) {
      c.level_ = rank & 0xff;
    }
    c.attack_ = q_read_u32();
    c.defense_ = q_read_u32();
    c.lscale_ = q_read_u32();
    c.rscale_ = q_read_u32();
    uint32_t link = q_read_u32();
    uint32_t link_marker = q_read_u32();
    if ((link & 0xff) > 0) {
      c.level_ = link & 0xff;
    }
    if (link_marker > 0) {
      c.defense_ = link_marker;
    }
    return c;
  }

  std::vector<Card> get_cards_in_location(PlayerId player, uint8_t loc) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_EQUIP_CARD |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_STATUS |
                    QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
    int32_t bl = YGO_QueryFieldCard(pduel_, player, loc, flags, query_buf_);
    qdp_ = 0;
    std::vector<Card> cards;
    while (true) {
      if (qdp_ >= bl) {
        break;
      }
      uint32_t f = q_read_u32();
      if (f == LEN_EMPTY) {
        continue;
        ;
      }
      f = q_read_u32();
      CardCode code = q_read_u32();
      Card c = c_get_card(code);

      uint8_t controller = q_read_u8();
      uint8_t location = q_read_u8();
      uint8_t sequence = q_read_u8();
      uint8_t position = q_read_u8();
      c.controler_ = controller;
      c.location_ = location;
      c.sequence_ = sequence;
      c.position_ = position;

      uint32_t level = q_read_u32();
      if ((level & 0xff) > 0) {
        c.level_ = level & 0xff;
      }
      uint32_t rank = q_read_u32();
      if ((rank & 0xff) > 0) {
        c.level_ = rank & 0xff;
      }
      c.attack_ = q_read_u32();
      c.defense_ = q_read_u32();

      // TODO(2): equip_target
      if (f & QUERY_EQUIP_CARD) {
        q_read_u32();
      }

      uint32_t n_xyz = q_read_u32();
      for (int i = 0; i < n_xyz; ++i) {
        auto code = q_read_u32();
        Card c_ = c_get_card(code);
        c_.controler_ = controller;
        c_.location_ = location | LOCATION_OVERLAY;
        c_.sequence_ = sequence;
        c_.position_ = i;
        cards.push_back(c_);
      }

      // TODO(2): counters
      uint32_t n_counters = q_read_u32();
      for (int i = 0; i < n_counters; ++i) {
        if (i == 0) {
          c.counter_ = q_read_u32();
        }
        else {
          q_read_u32();
        }
      }

      c.status_ = q_read_u32();
      c.lscale_ = q_read_u32();
      c.rscale_ = q_read_u32();

      uint32_t link = q_read_u32();
      uint32_t link_marker = q_read_u32();
      if ((link & 0xff) > 0) {
        c.level_ = link & 0xff;
      }
      if (link_marker > 0) {
        c.defense_ = link_marker;
      }
      cards.push_back(c);
    }
    return cards;
  }

  std::vector<Card> read_cardlist(bool extra = false, bool extra8 = false) {
    std::vector<Card> cards;
    auto count = read_u8();
    cards.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto card = get_card(controller, loc, seq);
      if (extra) {
        if (extra8) {
          card.data_ = read_u8();
        } else {
          card.data_ = read_u32();
        }
      }
      cards.push_back(card);
    }
    return cards;
  }

  std::vector<IdleCardSpec> read_cardlist_spec(PlayerId player, bool extra = false, bool extra8 = false) {
    std::vector<IdleCardSpec> card_specs;
    auto count = read_u8();
    card_specs.reserve(count);
    for (int i = 0; i < count; ++i) {
      CardCode code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      uint32_t data = 0;
      if (extra) {
        if (extra8) {
          data = read_u8();
        } else {
          data = read_u32();
        }
      }
      card_specs.push_back({code, ls_to_spec(loc, seq, 0, player != controller), data});
    }
    return card_specs;
  }

  std::tuple<CardCode, int> unpack_desc(CardCode code, uint32_t desc) {
    if (desc < DESCRIPTION_LIMIT) {
      return {0, desc};
    }
    CardCode code_ = desc >> 4;
    int idx = desc & 0xf;
    if (idx < 0 || idx >= 14) {
      fmt::print("Code: {}, Code_: {}, Desc: {}\n", code, code_, desc);
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      throw std::runtime_error("Invalid effect index: " + std::to_string(idx));
    }
    return {code_, idx + CARD_EFFECT_OFFSET};
  }

  std::string cardlist_info_for_player(const Card &card, PlayerId pl) {
    std::string spec = card.get_spec(pl);
    if (card.location_ == LOCATION_DECK) {
      spec = "deck";
    }
    if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
      return position_to_string(card.position_) + "card (" + spec + ")";
    }
    return card.name_ + " (" + spec + ")";
  }

  // This function does the following:
  // 1. read msg_ from data_ and update dp_
  // 2. (optional) print information if verbose_ is true
  // 3. update to_play_ and options_ if need action
  void handle_message() {
    msg_ = int(data_[dp_++]);
    legal_actions_ = {};

    if (verbose_) {
      fmt::println("Message {}, length {}, dp {}", msg_to_string(msg_), dl_, dp_);
    }

    if (msg_ == MSG_DRAW) {
      auto player = read_u8();
      auto drawed = read_u8();
      if (player < 2 && current_phase_ && current_phase_ != PHASE_DRAW) outside_draw_adds_[player] += drawed;
      std::vector<uint32_t> codes;
      for (int i = 0; i < drawed; ++i) {
        uint32_t code = read_u32();
        if (player < 2 && (code & 0x80000000u))
          public_hands_[player].public_draw(code & 0x7fffffffu);
        if (turn_count_ > 0 && player < 2 && tp_ == 1-player) {
          int g = g_draw_source_index(pduel_,player,drawed,i,code & 0x7fffffffu);
          if (g >= 0) ++g_draws_[tp_][g];
        }
        if (verbose_) codes.push_back(code & 0x7fffffffu);
      }
      if (!verbose_) return;
      const auto &pl = players_[player];
      pl->notify(fmt::format("Drew {} cards:", drawed));
      for (int i = 0; i < drawed; ++i) {
        const auto &c = c_get_card(codes[i]);
        pl->notify(fmt::format("{}: {}", i + 1, c.name_));
      }
      const auto &op = players_[1 - player];
      op->notify(fmt::format("Opponent drew {} cards.", drawed));
    } else if (msg_ == MSG_NEW_TURN) {
      opening_ecclesia_stage_=0; outside_draw_adds_ = {}; search_additions_.clear(); resolving_search_=0;
      battle_attack_source_ = 0;
      tp_ = int(read_u8());
      turn_count_++;
      if (!verbose_) {
        return;
      }
      auto& player = players_[tp_];
      player->notify("Your turn.");
      players_[1 - tp_]->notify(fmt::format("{}'s turn.", player->nickname_));
    } else if (msg_ == MSG_NEW_PHASE) {
      current_phase_ = int(read_u16());
      if (!verbose_) {
        return;
      }
      auto phase_str = phase_to_string(current_phase_);
      for (int i = 0; i < 2; ++i) {
        players_[i]->notify(fmt::format("Entering {} phase.", phase_str));
      }
    } else if (msg_ == MSG_MOVE) {
      // 阶段0：无论 verbose 与否都解析 MOVE 头部，追踪简易融合产物。
      // 保存并恢复 dp_，保证下面 verbose 分支仍从报文头开始读。
      {
        auto saved_dp = dp_;
        CardCode mv_code = read_u32();
        uint32_t from = read_u32();
        uint32_t to = read_u32();
        uint32_t mv_reason = read_u32();
        for (auto& entry : chain_stack_) update_negation_source(entry, from, to);
        uint8_t f_ctrl = from & 0xff, f_loc = (from >> 8) & 0xff,
                f_seq = (from >> 16) & 0xff;
        uint8_t t_ctrl = to & 0xff, t_loc = (to >> 8) & 0xff,
                t_seq = (to >> 16) & 0xff;
        auto public_zone = [](uint8_t loc, uint8_t pos) {
          return loc == LOCATION_GRAVE ||
                 ((loc == LOCATION_MZONE || loc == LOCATION_SZONE ||
                   loc == LOCATION_REMOVED || loc == LOCATION_EXTRA) &&
                  (pos & POS_FACEUP));
        };
        const bool known_move = public_zone(f_loc, from >> 24) ||
                                public_zone(t_loc, to >> 24);
        if (f_loc == LOCATION_HAND)
          for (auto& a: search_additions_) if (a.player==f_ctrl && !a.shown) a.slot=-1;
        if (f_loc == LOCATION_DECK && t_loc == LOCATION_HAND && t_ctrl < 2 && current_phase_ && current_phase_ != PHASE_DRAW) {
          ++outside_draw_adds_[t_ctrl];
          if (resolving_search_) search_additions_.push_back({t_ctrl,t_seq,resolving_search_,0});
        }
        if (f_loc == LOCATION_HAND && f_ctrl < 2)
          public_hands_[f_ctrl].leave(known_move ? mv_code : 0);
        if (t_loc == LOCATION_HAND && t_ctrl < 2)
          public_hands_[t_ctrl].arrive(t_seq, known_move ? mv_code : 0);
        track_battle_source_move(from, to);
        if (simple_fusion_pending_ && f_loc == LOCATION_EXTRA &&
            t_loc == LOCATION_MZONE && t_ctrl < 2 && t_seq < 7) {
          simple_fusion_on_mzone_[t_ctrl][t_seq] = true;
          simple_fusion_pending_ = false;
        }
        // 带 flag 的卡离开 mzone：END 阶段是自毁，其余是当素材/叠放（正确利用）
        if (f_loc == LOCATION_MZONE && f_ctrl < 2 && f_seq < 7 &&
            t_loc != LOCATION_MZONE &&
            simple_fusion_on_mzone_[f_ctrl][f_seq]) {
          // 只统计我方；对方的仍要清标记，否则残留在格子上被后续误判
          if (f_ctrl == actor_) {
            if (current_phase_ == PHASE_END) ++sf_destroyed_;
            else ++sf_consumed_;
          }
          simple_fusion_on_mzone_[f_ctrl][f_seq] = false;
        }
        // 事件级：用对方场上的怪兽当融合素材
        // 判据：改名卡从 MZONE 因融合进墓，且融合方 != 该卡所有者
        if (f_loc == LOCATION_MZONE && t_loc == LOCATION_GRAVE &&
            (mv_reason & 0x40u) && tail_fusion_ctrl_ >= 0 &&
            tail_fusion_ctrl_ < 2 &&
            tail_fusion_ctrl_ == static_cast<int>(actor_) &&
            tail_fusion_ctrl_ != static_cast<int>(f_ctrl) &&
            f_ctrl >= 0 && f_ctrl < 2 && f_loc < 16 && f_seq < 16 &&
            renamed_[f_ctrl][f_loc][f_seq] != 0) {
          ++opp_fusion_mat_;
        }
        // 改名卡（renamed_ 非零）离场统计 —— 必须在清除之前
        bool was_renamed = (f_ctrl >= 0 && f_ctrl < 2 && f_loc < 16 &&
                            f_seq < 16 && renamed_[f_ctrl][f_loc][f_seq] != 0);
        if (was_renamed) {
          int gc = static_cast<int>(f_ctrl);
          // 从 MZONE 进墓 = 被当素材消耗
          if (f_loc == LOCATION_MZONE && t_loc == LOCATION_GRAVE) {
            ++gyoku_grave_[gc];
            // reason 带 REASON_FUSION(0x40) = 融合素材
            if (mv_reason & 0x40u) ++tail_oppmat_[gc];
          }
          renamed_[f_ctrl][f_loc][f_seq] = 0;
          ++rename_clr_;
        }
        // 墓地 -> 魔陷区：TAIL 的 e3 装备 / START 的 e3 盖放（只看我方）
        if (f_loc == LOCATION_GRAVE && t_loc == LOCATION_SZONE &&
            f_ctrl == actor_) {
          if (mv_code == 82119326) {
            ++tail_equip_;
            const int sc = static_cast<int>(f_ctrl);
            if (sc >= 0 && sc < 2 && !tail_chain_done_[sc] &&
                mill_tail_turn_ == turn_count_) {
              ++tail_chain_;
              tail_chain_done_[sc] = true;
            }
          } else if (mv_code == 19326613) {
            ++start_set_;
          }
        }
        // 统计用：本步 DECK -> GRAVE 的卡（最多记 2 张）
        if (f_loc == LOCATION_DECK && t_loc == LOCATION_GRAVE) {
          int mid = static_cast<int>(c_get_card_id(mv_code));
          // 奖励用：只认我方堆墓（mill_id_* / mill_tail_turn_ 供 shaping 使用）
          if (f_ctrl == actor_ && mill_mine_cnt_ < 2) {
            if (mill_mine_cnt_ == 0) mill_id_a_ = mid;
            else mill_id_b_ = mid;
            if (mid == 12451) mill_tail_turn_ = turn_count_;   // TAIL 进墓的回合
            ++mill_mine_cnt_;
          }
          // 统计用：候选/选中计数保持双方口径（combo 分析用）
          if (mill_cnt_ < 2) {
            ++mill_cnt_;
            if (pend_pair_ && mill_cnt_ == 1) {
              // 真堆墓：候选数在这一刻才计入（检索不会走到这里）
              if (pend_s_) ++pair_start_cand_;
              if (pend_t_) ++pair_tail_cand_;
              if (pend_kg_) ++pair_kg_cand_;
              if (pend_fd_) ++pair_fd_cand_;
              if (pend_mg_) ++pair_mg_cand_;
              if (pend_sn_) ++pair_sn_cand_;
              pend_pair_ = false;
            }
          }
          if (sel_has_start_ && mv_code == 19326613) ++pair_start_pick_;
          if (sel_has_tail_ && mv_code == 82119326) ++pair_tail_pick_;
        }
        // 阶段0+：威凯特当融合素材
        //   A 威凯特 mzone->grave（我方）
        //   B 终端 extra->mzone（我方），且距 A <= 2 步
        {
          auto itc = card_ids_.find(mv_code);
          const int cid = (itc == card_ids_.end())
                              ? -1
                              : static_cast<int>(itc->second);
          if (cid == 4219 && f_ctrl == actor_ &&
              f_loc == LOCATION_MZONE && t_loc == LOCATION_GRAVE) {
            wicat_grave_step_ = step_count_;
          } else if ((cid == 638 || cid == 11847) && t_ctrl == actor_ &&
                     f_loc == LOCATION_EXTRA && t_loc == LOCATION_MZONE &&
                     wicat_grave_step_ >= 0 &&
                     step_count_ - wicat_grave_step_ <= 2) {
            ++wicat_convert_;
            wicat_grave_step_ = -1;
          }
        }
        dp_ = saved_dp;
      }
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      uint32_t newloc = read_u32();
      uint32_t reason = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      Card cnew = c_get_card(code);
      cnew.set_location(newloc);
      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];

      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto plnewspec = cnew.get_spec(false);
      auto opnewspec = cnew.get_spec(true);

      auto getspec = [&](auto& p) { return p.get() == pl.get() ? plspec : opspec; };
      auto getnewspec = [&](auto& p) {
        return p.get() == pl.get() ? plnewspec : opnewspec;
      };
      bool card_visible = true;
      if ((card.position_ & POS_FACEDOWN) && (cnew.position_ & POS_FACEDOWN)) {
        card_visible = false;
      }
      auto getvisiblename = [&](auto& p) {
        return card_visible ? card.name_ : "Face-down card";
      };

      if ((reason & REASON_DESTROY) && (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("Card {} ({}) destroyed.", plspec, card.name_));
        op->notify(fmt::format("Card {} ({}) destroyed.", opspec, card.name_));
      } else if ((card.location_ == cnew.location_) &&
                 (card.location_ & LOCATION_ONFIELD)) {
        if (card.controler_ != cnew.controler_) {
          pl->notify(
              fmt::format("Your card {} ({}) changed controller to {} and is "
                          "now located at {}.",
                          plspec, card.name_, op->nickname_, plnewspec));
          op->notify(
              fmt::format("You now control {}'s card {} ({}) and it's located "
                          "at {}.",
                          pl->nickname_, opspec, card.name_, opnewspec));
        } else {
          pl->notify(fmt::format("Your card {} ({}) switched its zone to {}.",
                                 plspec, card.name_, plnewspec));
          op->notify(fmt::format("{}'s card {} ({}) switched its zone to {}.",
                                 pl->nickname_, opspec, card.name_, opnewspec));
        }
      } else if ((reason & REASON_DISCARD) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You discarded {} ({})", plspec, card.name_));
        op->notify(fmt::format("{} discarded {} ({})", pl->nickname_, opspec,
                               card.name_));
      } else if ((card.location_ == LOCATION_REMOVED) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your banished card {} ({}) returns to the field at "
                        "{}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s banished card {} ({}) returns to the field at "
                        "{}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((card.location_ == LOCATION_GRAVE) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((cnew.location_ == LOCATION_HAND) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(
            fmt::format("Card {} ({}) returned to hand.", plspec, card.name_));
      } else if ((reason & (REASON_RELEASE | REASON_SUMMON)) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You tribute {} ({}).", plspec, card.name_));
        op->notify(fmt::format("{} tributes {} ({}).", pl->nickname_, opspec,
                               getvisiblename(op)));
      } else if ((card.location_ == (LOCATION_OVERLAY | LOCATION_MZONE)) &&
                 (cnew.location_ & LOCATION_GRAVE)) {
        pl->notify(fmt::format("You detached {}.", card.name_));
        op->notify(fmt::format("{} detached {}.", pl->nickname_, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_GRAVE)) {
        pl->notify(fmt::format("Your card {} ({}) was sent to the graveyard.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was sent to the graveyard.",
                               pl->nickname_, opspec, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_REMOVED)) {
        pl->notify(
            fmt::format("Your card {} ({}) was banished.", plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was banished.", pl->nickname_,
                               opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_DECK)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your deck.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) returned to their deck.",
                               pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_EXTRA)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your extra deck.",
                               plspec, card.name_));
        op->notify(
            fmt::format("{}'s card {} ({}) returned to their extra deck.",
                        pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ == LOCATION_DECK) &&
                 (cnew.location_ == LOCATION_SZONE) &&
                 (cnew.position_ != POS_FACEDOWN)) {
        pl->notify(fmt::format("Activating {} ({})", plnewspec, card.name_));
        op->notify(fmt::format("{} activating {} ({})", pl->nickname_, opspec,
                               cnew.name_));
      }
    } else if (msg_ == MSG_SWAP) {
      CardCode code1 = read_u32();
      uint32_t loc1 = read_u32();
      CardCode code2 = read_u32();
      uint32_t loc2 = read_u32();
      for (auto& entry : chain_stack_)
        if (!update_negation_source(entry, loc1, loc2)) update_negation_source(entry, loc2, loc1);
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      Card cards[2];
      cards[0] = c_get_card(code1);
      cards[1] = c_get_card(code2);
      cards[0].set_location(loc1);
      cards[1].set_location(loc2);

      for (PlayerId pl = 0; pl < 2; pl++) {
        for (int i = 0; i < 2; i++) {
          auto c = cards[i];
          auto spec = c.get_spec(pl);
          auto plname = players_[1 - c.controler_]->nickname_;
          players_[pl]->notify("Card " + c.name_ + " swapped control towards " +
                               plname + " and is now located at " + spec + ".");
        }
      }
    } else if (msg_ == MSG_SET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      auto c = card.controler_;
      auto& cpl = players_[c];
      auto& opl = players_[1 - c];
      cpl->notify(fmt::format("You set {} ({}) in {} position.", card.name_,
                              card.get_spec(c), card.get_position()));
      opl->notify(fmt::format("{} sets {} in {} position.", cpl->nickname_,
                              card.get_spec(PlayerId(1 - c)),
                              card.get_position()));
    } else if (msg_ == MSG_EQUIP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      Card card = get_card(c, loc, seq);
      c = read_u8();
      loc = read_u8();
      seq = read_u8();
      pos = read_u8();
      Card target = get_card(c, loc, seq);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto c = cardlist_info_for_player(card, pl);
        auto t = cardlist_info_for_player(target, pl);
        players_[pl]->notify(fmt::format("{} equipped to {}.", c, t));
      }
    } else if (msg_ == MSG_HINT) {
      auto hint_type = read_u8();
      auto player = read_u8();
      auto value = read_u32();

      if (hint_type == HINT_SELECTMSG) {
        last_hint_value_ = value;
      }
      if (hint_type == HINT_SELECTMSG && value == 501) {
        discard_hand_ = true;
      }
      // 549 = 「选择攻击目标」，core 在 PROCESSOR_SELECT_CARD 前发出，
      // 用于把「战斗目标选择」和普通效果选卡区分开。
      if (hint_type == HINT_SELECTMSG && value == 549) {
        attack_target_pending_ = true;
      }
      // non-GUI don't need hint
      return;
      if (hint_type == HINT_SELECTMSG) {
        if (value > 2000) {
          CardCode code = value;
          players_[player]->notify(fmt::format("{} select {}",
                                               players_[player]->nickname_,
                                               c_get_card(code).name_));
        } else {
          players_[player]->notify(get_system_string(value));
        }
      } else if (hint_type == HINT_NUMBER) {
        players_[1 - player]->notify(
            fmt::format("Choice of player: {}", value));
      } else {
        fmt::println("Unknown hint type {} with value {}", hint_type, value);
      }
    } else if (msg_ == MSG_CARD_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      uint8_t player = read_u8();
      uint8_t loc = read_u8();
      uint8_t seq = read_u8();
      uint8_t pos = read_u8();
      uint8_t type = read_u8();
      uint32_t value = read_u32();
      if (type == CHINT_RACE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string races_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           races_str));
        }
      } else if (type == CHINT_ATTRIBUTE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string attributes_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           attributes_str));
        }
      } else {
        fmt::println("Unknown card hint type {} with value {}", type, value);
      }
    } else if (msg_ == MSG_POS_CHANGE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      uint8_t prevpos = card.position_;
      card.position_ = read_u8();

      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];
      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto prevpos_str = position_to_string(prevpos);
      auto pos_str = position_to_string(card.position_);
      pl->notify("The position of card " + plspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
      op->notify("The position of card " + opspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
    } else if (msg_ == MSG_BECOME_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto u = read_u8();
      uint32_t target = read_u32();
      uint8_t tc = target & 0xff;
      uint8_t tl = (target >> 8) & 0xff;
      uint8_t tseq = (target >> 16) & 0xff;
      Card card = get_card(tc, tl, tseq);
      auto name = players_[chaining_player_]->nickname_;
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto tcname = card.name_;
        if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
          tcname = position_to_string(card.position_) + " card";
        }
        players_[pl]->notify(name + " targets " + spec + " (" + tcname + ")");
      }
    } else if (msg_ == MSG_CONFIRM_DECKTOP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        if (pl == player) {
          p->notify(fmt::format("You reveal {} cards from your deck:", size));
        } else {
          p->notify(fmt::format("{} reveals {} cards from their deck:",
                                players_[player]->nickname_, size));
        }
        for (int i = 0; i < size; ++i) {
          p->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
    } else if (msg_ == MSG_RANDOM_SELECTED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      std::vector<Card> cards;

      for (int i = 0; i < count; ++i) {
        auto c = read_u8();
        auto loc = read_u8();
        if (loc & LOCATION_OVERLAY) {
          throw std::runtime_error("Overlay not supported for random selected");
        }
        auto seq = read_u8();
        auto pos = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto s = "card is";
        if (count > 1) {
          s = "cards are";
        }
        if (pl == player) {
          p->notify(fmt::format("Your {} {} randomly selected:", s, count));
        } else {
          p->notify(fmt::format("{}'s {} {} randomly selected:",
                                players_[player]->nickname_, s, count));
        }
        for (int i = 0; i < count; ++i) {
          p->notify(fmt::format("{}: {}", cards[i].get_spec(pl), cards[i].name_));
        }
      }
    } else if (msg_ == MSG_PLAYER_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 6;
      // TODO(3): implement output
    } else if (msg_ == MSG_CARD_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c1 = read_u8();
      auto l1 = read_u8();
      auto s1 = read_u8();
      read_u8();
      auto c2 = read_u8();
      auto l2 = read_u8();
      auto s2 = read_u8();
      read_u8();

      Card card1 = get_card(c1, l1, s1);
      Card card2 = get_card(c2, l2, s2);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto spec1 = card1.get_spec(pl);
        auto spec2 = card2.get_spec(pl);
        auto c1name = card1.name_;
        auto c2name = card2.name_;
        if ((card1.controler_ != pl) && (card1.position_ & POS_FACEDOWN)) {
          c1name = position_to_string(card1.position_) + " card";
        }
        if ((card2.controler_ != pl) && (card2.position_ & POS_FACEDOWN)) {
          c2name = position_to_string(card2.position_) + " card";
        }
        p->notify(fmt::format(" {} ({}) targets {} ({})", spec1, c1name, spec2, c2name));
      }
    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      auto skip_panel = read_u8();  // 新版新增字段
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        auto code = read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        if (verbose_) {
          cards.push_back(get_card(c, loc, seq));
        }
        if (loc == LOCATION_HAND) {
          if (c < 2 && player == 1 - c)
            public_hands_[c].reveal(seq, code);
          if (c<2 && player==1-c && resolving_search_)
            for (auto& a: search_additions_)
              if (a.player==c && a.slot==seq && a.cause==resolving_search_ && !a.shown) a.shown=code;
        } else {
          revealed_.insert(ls_to_spec(loc, seq, 0, c == player));
        }
      }
      if (!verbose_) {
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];

      op->notify(fmt::format("{} shows you {} cards.", pl->nickname_, size));
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }
    } else if (msg_ == MSG_MISSED_EFFECT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 4;
      CardCode code = read_u32();
      Card card = c_get_card(code);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto str = get_system_string(1622);
        std::string fmt_str = "[%ls]";
        str = str.replace(str.find(fmt_str), fmt_str.length(), card.name_);
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_SORT_CARD) {
      // TODO(3): implement action
      if (!verbose_) {
        dp_ = dl_;
        resp_buf_[0] = 255;
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }
      auto& pl = players_[player];
      pl->notify(
          "Sort " + std::to_string(size) +
          " cards by entering numbers separated by spaces (c = cancel):");
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }

      fmt::println("sort card action not implemented");
      resp_buf_[0] = 255;
      YGO_SetResponseb(pduel_, resp_buf_);

      // // generate all permutations
      // std::vector<int> perm(size);
      // std::iota(perm.begin(), perm.end(), 0);
      // std::vector<std::vector<int>> perms;
      // do {
      //   auto option = std::accumulate(perm.begin(), perm.end(),
      //   std::string(),
      //                                 [&](std::string &acc, int i) {
      //                                   return acc + std::to_string(i + 1) +
      //                                   " ";
      //                                 });
      //   options_.push_back(option);
      // } while (std::next_permutation(perm.begin(), perm.end()));
      // options_.push_back("c");
      // callback_ = [this](int idx) {
      //   const auto &option = options_[idx];
      //   if (option == "c") {
      //     resp_buf_[0] = 255;
      //     YGO_SetResponseb(pduel_, resp_buf_);
      //     return;
      //   }
      //   std::istringstream iss(option);
      //   int x;
      //   int i = 0;
      //   while (iss >> x) {
      //     resp_buf_[i] = uint8_t(x);
      //     i++;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      // };
    } else if (msg_ == MSG_ADD_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      // TODO(3): counter type to string
      pl->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_REMOVE_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      pl->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_ATTACK_DISABLED) {
      battle_attack_source_ = 0;
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      for (PlayerId pl = 0; pl < 2; pl++) {
        players_[pl]->notify(get_system_string(1621));
      }
    } else if (msg_ == MSG_SHUFFLE_SET_CARD) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      // TODO(3): implement output
      dp_ = dl_;
    } else if (msg_ == MSG_SHUFFLE_DECK) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your deck.");
      op->notify(pl->nickname_ + " shuffled their deck.");
    } else if (msg_ == MSG_SHUFFLE_EXTRA) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) {
        read_u32();
      }
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify(fmt::format("You shuffled your extra deck ({}).", count));
      op->notify(fmt::format("{} shuffled their extra deck ({}).", pl->nickname_, count));
    } else if (msg_ == MSG_SHUFFLE_HAND) {
      auto player = read_u8();
      for (auto& a: search_additions_) if (a.player==player && !a.shown) a.slot=-1;
      if (player < 2) public_hands_[player].shuffle();
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      dp_ = dl_;

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your hand.");
      op->notify(pl->nickname_ + " shuffled their hand.");
    } else if (msg_ == MSG_SUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_SUMMONING) {
      if (!verbose_) {
        CardCode code = read_u32();
        if (code == 86937530) {
          Card card = c_get_card(code);
          int kc = static_cast<int>(card.controler_);
          if (kc >= 0 && kc < 2) ++kaguya_summon_[kc];
        }
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (auto& pl : players_) {
        pl->notify(nickname + " summoning " + card.name_ + " (" +
                   std::to_string(card.attack_) + "/" +
                   std::to_string(card.defense_) + ") in " +
                   card.get_position() + " position.");
      }
    } else if (msg_ == MSG_SPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto code = read_u32();
      auto location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);

      auto& cpl = players_[card.controler_];
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        players_[1 - pl]->notify(cpl->nickname_ + " flip summons " + spec +
                                 " (" + card.name_ + ")");
      }
    } else if (msg_ == MSG_SPSUMMONING) {
      if (!verbose_) {
        CardCode code = read_u32();
        if (code == 4026187u) {
          Card card = c_get_card(code);
          int sc = static_cast<int>(card.controler_);
          if (sc >= 0 && sc < 2) ++story_summon_[sc];
        }
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      if (code == 4026187u) {
        int sc2 = static_cast<int>(card.controler_);
        if (sc2 >= 0 && sc2 < 2) ++story_summon_[sc2];
      }
      const auto &nickname = players_[card.controler_]->nickname_;
      for (PlayerId p = 0; p < 2; p++) {
        auto& pl = players_[p];
        auto pos = card.get_position();
        auto atk = std::to_string(card.attack_);
        auto def = std::to_string(card.defense_);
        std::string name = p == card.controler_ ? "You" : nickname;
        if (card.type_ & TYPE_LINK) {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + ") in " + pos + " position.");
        } else {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + "/" + def + ") in " + pos + " position.");
        }
      }
    } else if (msg_ == MSG_CHAIN_NEGATED || msg_ == MSG_CHAIN_DISABLED) {
      const auto link_no = read_u8();
      auto link = std::find_if(chain_stack_.rbegin(), chain_stack_.rend(),
                               [link_no](const ChainEntry &entry) { return entry.link_no == link_no; });
      if (link_no != 0 && link != chain_stack_.rend()) {
        link->negated = true;
      }
      // Rename bookkeeping uses the existing resolution path.
      if (!chain_stack_.empty()) {
        pending_rename_.erase(chain_stack_.back().id);
        rename_pending_chain_ = 0;
      }
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_SOLVED) {
      resolving_search_=0;
      if (!chain_stack_.empty()) {
        int cid2 = chain_stack_.back().id;
        auto it2 = pending_rename_.find(cid2);
        if (it2 != pending_rename_.end()) {
          auto [pc, pl, ps, pp] = spec_to_ls(to_play_, it2->second);
          if (pl == LOCATION_MZONE && pc != static_cast<int>(to_play_) &&
              pc >= 0 && pc < 2 && ps < 16) {
            renamed_[pc][pl][ps] = 2937u;
            ++rename_hit_;
          }
          pending_rename_.erase(it2);
        }
        rename_pending_chain_ = 0;
      }
      tail_fusion_active_ = false;
      dp_ = dl_;
      revealed_.clear();
    } else if (msg_ == MSG_CHAIN_SOLVING) {
      const auto resolving_link = read_u8();
      resolving_search_=0;
      for (const auto& link: chain_stack_) if (link.link_no == resolving_link) resolving_search_=link.code;
      if (rename_pending_chain_ > 0 && !chain_stack_.empty() &&
          chain_stack_.back().id == rename_pending_chain_) {
        rename_pending_ = true;
      }
      if (!chain_stack_.empty() &&
          chain_stack_.back().code == 82119326u &&
          chain_stack_.back().controler == actor_) {
        tail_fusion_active_ = true;
        tail_fusion_ctrl_ = static_cast<int>(chain_stack_.back().controler);
      }
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_END) {
      chain_stack_.clear();
      chain_forced_ = false;
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      CardCode code = read_u32();
      Card card = c_get_card(code);
      // 阶段0：辉夜（86937530）效果发动 -> 本步计数，供交卷 K2 使用（只看我方）
      // 阶段0+：START / TAIL 的效果发动（只看我方）
      if (card.controler_ == actor_) {
        if (code == 86937530) ++kaguya_effect_;
        else if (code == 19326613) ++start_used_;
        else if (code == 82119326) ++tail_used_;
      }
      card.set_location(read_u32());
      if (card.location_ == LOCATION_HAND && card.controler_ < 2)
        public_hands_[card.controler_].reveal(card.sequence_, code);
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();
      if (code == 19144622u || code == 56725612u || code == 78021082u) {
        rename_pending_ = true;
        rename_ctrl_ = static_cast<int>(card.controler_);
      }
      if (code == 19144622u) {
        ++dsc_ma_idx_[desc & 0xFu];
        ++dsc_ma_all_;
        if (desc == 19144622u * 16u + 2u) ++dsc_ma_;
      } else if (code == 56725612u) {
        ++dsc_fld_idx_[desc & 0xFu];
        ++dsc_fld_all_;
        if (desc == 56725612u * 16u + 1u) ++dsc_fld_;
      } else if (code == 78021082u) {
        ++dsc_wr_idx_[desc & 0xFu];
        ++dsc_wr_all_;
        if (desc == 78021082u * 16u + 1u) ++dsc_wr_;
      }
      if ((code == 19144622u || code == 56725612u || code == 78021082u) &&
          card.controler_ == actor_) {
        int rc = static_cast<int>(card.controler_);
        if (rc >= 0 && rc < 2) ++rename_[rc];
      }
      if (code == 82119326 && desc == 82119326u * 16u) {
        int tc2 = static_cast<int>(card.controler_);
        if (tc2 >= 0 && tc2 < 2) ++tail_fusion_[tc2];
        tail_fusion_pending_ = true;
        // 只记录我方发起的 TAIL 融合，避免对手的融合覆盖归属
        if (tc2 == static_cast<int>(actor_)) tail_fusion_ctrl_ = tc2;
      }
      if (code == 86937530) {
        int kc = static_cast<int>(card.controler_);
        if (kc >= 0 && kc < 2) {
          if (desc == 86937530u * 16u) ++kaguya_search_[kc];
          else if (desc == 86937530u * 16u + 1u) ++kaguya_bounce_[kc];
          else if (desc < 4) ++kaguya_desc_[desc];
        }
      }
      auto c = card.controler_;
      PlayerId o = 1 - c;
      chaining_player_ = c;
      // 记录连锁项（训练时也需要，之前在这里直接 return 丢弃了）
      ChainEntry ce;
      ce.controler = tc;
      ce.effects = card.effects_;
      ce.code = code;
      ce.desc = desc;
      ce.loc = tl;
      ce.seq = tl == LOCATION_MZONE
          ? (card.location_ == LOCATION_MZONE && card.controler_ == tc ? card.sequence_ : 74)
          : ts;
      ce.link_no = cs;
      {
        auto cit = g_card_category.find(static_cast<uint32_t>(code));
        if (cit != g_card_category.end()) ce.category = cit->second;
      }
      ++chain_seq_;
      ce.id = chain_seq_;
      if (chain_stack_.empty()) search_additions_.clear();
      chain_stack_.push_back(ce);
      if (code == 19144622u || code == 56725612u || code == 78021082u) {
        if (!last_opp_sel_spec_.empty()) {
          pending_rename_[chain_seq_] = last_opp_sel_spec_;
          last_opp_sel_spec_.clear();
        } else {
          rename_pending_chain_ = chain_seq_;
        }
      }
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      players_[c]->notify("Activating " + card.get_spec(c) + " (" + card.name_ +
                          ")");
      players_[o]->notify(players_[c]->nickname_ + " activating " +
                          card.get_spec(o) + " (" + card.name_ + ")");
    } else if (msg_ == MSG_DAMAGE) {
      auto player = read_u8();
      auto amount = read_u32();
      _damage(player, amount);
    } else if (msg_ == MSG_RECOVER) {
      auto player = read_u8();
      auto amount = read_u32();
      _recover(player, amount);
    } else if (msg_ == MSG_LPUPDATE) {
      auto player = read_u8();
      auto lp = read_u32();
      if (lp >= lp_[player]) {
        _recover(player, lp - lp_[player]);
      } else {
        _damage(player, lp_[player] - lp);
      }
    } else if (msg_ == MSG_PAY_LPCOST) {
      auto player = read_u8();
      auto cost = read_u32();
      lp_[player] -= cost;
      if (!verbose_) {
        return;
      }
      auto& pl = players_[player];
      pl->notify("You pay " + std::to_string(cost) + " LP. Your LP is now " +
                 std::to_string(lp_[player]) + ".");
      players_[1 - player]->notify(
          pl->nickname_ + " pays " + std::to_string(cost) + " LP. " +
          pl->nickname_ + "'s LP is now " + std::to_string(lp_[player]) + ".");
    } else if (msg_ == MSG_ATTACK) {
      auto attacker = read_u32();
      battle_attack_source_ = attacker & kBattleSourceMask;
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      PlayerId ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;
      auto apos = (attacker >> 24) & 0xff;
      auto target = read_u32();
      PlayerId tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      if ((ac == 0) && (aloc == 0) && (aseq == 0) && (apos == 0)) {
        return;
      }

      Card acard = get_card(ac, aloc, aseq);
      auto name = players_[ac]->nickname_;
      if ((tc == 0) && (tloc == 0) && (tseq == 0) && (tpos == 0)) {
        for (PlayerId i = 0; i < 2; i++) {
          players_[i]->notify(name + " prepares to attack with " +
                              acard.get_spec(i) + " (" + acard.name_ + ")");
        }
        return;
      }

      Card tcard = get_card(tc, tloc, tseq);
      for (PlayerId i = 0; i < 2; i++) {
        auto aspec = acard.get_spec(i);
        auto tspec = tcard.get_spec(i);
        auto tcname = tcard.name_;
        if ((tcard.controler_ != i) && (tcard.position_ & POS_FACEDOWN)) {
          tcname = tcard.get_position() + " card";
        }
        players_[i]->notify(name + " prepares to attack " + tspec + " (" +
                            tcname + ") with " + aspec + " (" + acard.name_ +
                            ")");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_START) {
      battle_attack_source_ = 0;
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("begin damage");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_END) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("end damage");
      }
    } else if (msg_ == MSG_BATTLE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      auto aa = read_u32();
      auto ad = read_u32();
      auto bd0 = read_u8();
      auto target = read_u32();
      auto da = read_u32();
      auto dd = read_u32();
      auto bd1 = read_u8();

      auto ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;

      auto tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      Card acard = get_card(ac, aloc, aseq);
      Card tcard;
      if (tloc != 0) {
        tcard = get_card(tc, tloc, tseq);
      }
      for (int i = 0; i < 2; i++) {
        auto& pl = players_[i];
        std::string attacker_points;
        if (acard.type_ & TYPE_LINK) {
          attacker_points = std::to_string(aa);
        } else {
          attacker_points = std::to_string(aa) + "/" + std::to_string(ad);
        }
        if (tloc != 0) {
          std::string defender_points;
          if (tcard.type_ & TYPE_LINK) {
            defender_points = std::to_string(da);
          } else {
            defender_points = std::to_string(da) + "/" + std::to_string(dd);
          }
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks " +
                     tcard.name_ + " (" + defender_points + ")");
        } else {
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks");
        }
      }
    } else if (msg_ == MSG_WIN) {
      auto player = read_u8();
      auto reason = read_u8();
      auto& winner = players_[player];
      auto& loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }
    } else if (msg_ == MSG_RETRY) {
      // 引擎拒绝了上一次响应。旧实现直接 throw，会让整个训练进程 abort。
      // 改为：结束当前这一局（envpool 会自动 reset 该 env），不影响其它 env。
      if (getenv("YGOPRO_DBG_RETRY")) {
        fmt::println("[retry] last_decision={} step={} player={} dp={} dl={}",
                     msg_to_string(last_decision_msg_), step_count_,
                     int(to_play_), dp_, dl_);
      }
      dp_ = dl_;
      ms_idx_ = -1;
      legal_actions_.clear();
      YGO_EndDuel(pduel_);
      duel_started_ = false;
      done_ = true;
      return;
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      battle_attack_source_ = 0;
      auto player = read_u8();
      auto activatable = read_cardlist_spec(player, true);
      auto attackable = read_cardlist_spec(player, true, true);
      bool to_m2 = read_u8();
      bool to_ep = read_u8();

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Battle menu:");
      }
      for (const auto [code_t, spec, desc] : activatable) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: activate {}({}) [{}/{}] ({})",
            cmd_idx, c.name_, spec, c.attack_, c.defense_, c.get_effect_description(code_d, eff_idx));
        }
      }
      std::vector<int> attack_original_indices;
      attack_original_indices.reserve(attackable.size());
      for (int attack_idx = 0; attack_idx < static_cast<int>(attackable.size());
           ++attack_idx) {
        const auto [code, spec, data] = attackable[attack_idx];
        bool direct_attackable = data & 0x1;

        auto act = direct_attackable ? ActionAct::DirectAttack : ActionAct::Attack;

        auto la = LegalAction::act_spec(act, spec);
        legal_actions_.push_back(la);
        attack_original_indices.push_back(attack_idx);
        if (verbose_) {
          auto [controller, loc, seq, pos] = spec_to_ls(player, spec);
          auto c = get_card(controller, loc, seq);
          int cmd_idx = legal_actions_.size();
          auto attack_str = direct_attackable ? "direct attack" : "attack";
          std::string s = fmt::format(
            "{}: {} {}({}) ", cmd_idx, attack_str, c.name_, spec);
          if (c.type_ & TYPE_LINK) {
            s += fmt::format("[{}]", c.attack_);
          } else {
            s += fmt::format("[{}/{}]", c.attack_, c.defense_);
          }
          pl->notify(s);
        }
      }
      if (to_m2) {
        legal_actions_.push_back(
          LegalAction::phase(ActionPhase::Main2));
        int cmd_idx = legal_actions_.size();
        if (verbose_) {
          pl->notify(fmt::format("{}: Main phase 2.", cmd_idx));
        }
      }
      if (to_ep) {
        legal_actions_.push_back(
          LegalAction::phase(ActionPhase::End));
        int cmd_idx = legal_actions_.size();
        if (verbose_) {
          pl->notify(fmt::format("{}: End phase.", cmd_idx));
        }
      }
      int n_activatables = activatable.size();
      // 过滤后实际 push 的数量，必须用它而不是 attackable.size()
      int n_attackables = static_cast<int>(attack_original_indices.size());
      // 每个攻击候选对应的攻击者 ATK，供后续目标选择阶段过滤使用
      std::vector<int> attack_atks;
      attack_atks.reserve(n_attackables);
      // 攻击者坐标（供判定时实时查攻，避免 BATTLECMD 快照过期）
      std::vector<std::array<int, 3>> attack_pos;
      attack_pos.reserve(n_attackables);
      for (int k = 0; k < n_attackables; ++k) {
        const auto &[a_code, a_spec, a_data] = attackable[attack_original_indices[k]];
        auto [ac, aloc, aseq, apos] = spec_to_ls(player, a_spec);
        attack_atks.push_back(get_card(ac, aloc, aseq).attack_);
        attack_pos.push_back({static_cast<int>(ac), static_cast<int>(aloc),
                              static_cast<int>(aseq)});
      }
      to_play_ = player;
      callback_ = [this, n_activatables, n_attackables, to_ep, to_m2,
                   attack_original_indices, attack_atks, attack_pos](int idx) {
        const auto &la = legal_actions_[idx];
        if (idx < n_activatables) {
          pending_attacker_atk_ = -1;
          attack_target_pending_ = false;
          YGO_SetResponsei(pduel_, idx << 16);
        } else if (idx < (n_activatables + n_attackables)) {
          int rel = idx - n_activatables;
          int original = attack_original_indices[rel];
          pending_attacker_atk_ = attack_atks[rel];
          pending_attacker_ctrl_ = attack_pos[rel][0];
          pending_attacker_loc_ = attack_pos[rel][1];
          pending_attacker_seq_ = attack_pos[rel][2];
          YGO_SetResponsei(pduel_, (original << 16) + 1);
        } else if ((la.phase_ == ActionPhase::End) && to_ep) {
          YGO_SetResponsei(pduel_, 3);
        } else if ((la.phase_ == ActionPhase::Main2) && to_m2) {
          YGO_SetResponsei(pduel_, 2);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_UNSELECT_CARD) {
      // TODO: add feature of selected cards (also for multi select)
      auto player = read_u8();
      bool finishable = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto select_size = read_u8();

      cur_win_code_.clear();
      cur_win_player_ = player;
      std::vector<uint32_t> sel_codes;
      sel_codes.reserve(select_size);
      std::vector<std::string> select_specs;
      select_specs.reserve(select_size);
      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards:");
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          sel_codes.push_back(code);
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          auto s = fmt::format("{}: {}({})", i + 1, card.name_, spec);
          pl->notify(s);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          sel_codes.push_back(read_u32());
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          select_specs.push_back(spec);
        }
      }

      auto unselect_size = read_u8();

      // unselect not allowed (no regrets)
      dp_ += 8 * unselect_size;

      for (int j = 0; j < select_specs.size(); ++j) {
        legal_actions_.push_back(LegalAction::from_spec(select_specs[j]));
      }
      for (int j = 0; j < static_cast<int>(select_specs.size()) &&
                      j < static_cast<int>(sel_codes.size()); ++j) {
        set_action_cid_from_code(select_specs[j], sel_codes[j]);
        cur_win_code_[select_specs[j]] = sel_codes[j];
      }

      if (finishable) {
        legal_actions_.push_back(LegalAction::finish());
      }

      // cancelable and finishable not needed

      to_play_ = player;
      callback_ = [this](int idx) {
        if (legal_actions_[idx].finish_) {
          YGO_SetResponsei(pduel_, -1);
        } else {
          resp_buf_[0] = 1;
          resp_buf_[1] = idx;
          YGO_SetResponseb(pduel_, resp_buf_);
        }
      };

    } else if (msg_ == MSG_SELECT_CARD) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      cur_win_code_.clear();
      std::vector<std::string> specs;
      specs.reserve(size);
      // 消息里每张候选都带 4 字节卡号；非 verbose 路径过去只 `dp_ += 4` 跳过，
      // 统计时再用位置反查卡号，结果反查出来的是「另一张卡」（卡组顺序被引擎洗过），
      // 导致 mill_cand / pair 系列统计全错（2026-11-19 定位）。这里统一把卡号读出来。
      std::vector<uint32_t> sel_codes;
      sel_codes.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          sel_codes.push_back(code);
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          cards.push_back(card);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          int i = specs.size();
          if (card.controler_ != player && card.position_ & POS_FACEDOWN) {
            pl->notify(
              fmt::format("{}: {} card ({})", i, card.get_position(), spec));
          } else {
            pl->notify(
              fmt::format("{}: {} ({})", i, card.name_, spec));
          }
        }
      } else {
        for (int i = 0; i < size; ++i) {
          sel_codes.push_back(read_u32());
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          specs.push_back(spec);
        }
      }

      // 统计：(min,max)==(1,2) 窗口实际选了几张（威凯特素材取除窗口 = m7 区）
      if (min == 1 && max == 2 && size >= 2) {
        ++w12_all_win_;
        bool has_m7 = false;
        for (const auto &sp : specs) {
          if (sp.size() >= 2 && sp[0] == 'm' && sp[1] == '7') {
            has_m7 = true;
            break;
          }
        }
        if (has_m7) ++w12_m7_win_;
        w12_pend_ = true;
        w12_m7_ = has_m7;
      } else {
        w12_pend_ = false;
        w12_m7_ = false;
      }

      // 改名效果的目标选择窗口
      if (rename_pending_ && size > 0 && min == 1 && max == 1) {
        bool all_opp_mz = true;
        for (const auto &sp : specs) {
          if (sp.empty() || sp[0] != 'o') { all_opp_mz = false; break; }
          auto [c2, l2, s2, p2] = spec_to_ls(player, sp);
          if (l2 != LOCATION_MZONE || c2 == player) {
            all_opp_mz = false;
            break;
          }
        }
        if (all_opp_mz) ++rename_win_;
        else rename_pending_ = false;
      }
      // 统计：融合怪候选（EXTRA）里是否含讲述者
      {
        CardId story_id = c_get_card_id(4026187u);
        bool all_extra = (size > 0);
        bool has_story = false;
        for (const auto &sp : specs) {
          auto [c2, l2, s2, p2] = spec_to_ls(player, sp);
          if (l2 != LOCATION_EXTRA) { all_extra = false; break; }
          if (spec_to_card_id(sp, player) == story_id) has_story = true;
        }
        if (all_extra) {
          ++extra_cand_;
          if (has_story) ++story_cand_;
        }
      }
      // 统计：候选全是对方 MZONE 且只选 1 只（疑似改卡名选目标）
      if (size > 0 && min == 1 && max == 1) {
        bool all_opp_mz = true;
        for (const auto &sp : specs) {
          if (sp.empty() || sp[0] != 'o') { all_opp_mz = false; break; }
          auto [c2, l2, s2, p2] = spec_to_ls(player, sp);
          if (l2 != LOCATION_MZONE || c2 == player) {
            all_opp_mz = false;
            break;
          }
        }
        if (all_opp_mz) ++freeze_cand_;
      }
      // 统计：从卡组选「妖精传姬」卡的候选构成（威凯特/START 的堆墓选择）
      if (size > 0) {
        // 妖精传姬 = setcode 0x1db 的全部 12 张卡（与 lua 的 IsSetCard(0x1db) 一致）。
        // 2026-11-19 修正：旧实现只有 8 个卡号，漏了白雪/塔利娅/辛德瑞拉/罗奇卡，
        // 且任一候选解析失败就把整条窗口判成「非纯妖精传姬」，导致 mill_cand
        // 只剩小池子残渣（size1 虚高到 68.6%）。这里改用全量卡号表，
        // 并把「解析不到卡号」单独计 hist_unk，不再与「非妖精传姬」混为一谈。
        static const uint32_t kFT[] = {
            4026187u,  19144622u, 19326613u, 27632520u, 42921475u,
            52022648u, 55623480u, 56725612u, 65953423u, 78021082u,
            82119326u, 86937530u};
        bool all_deck = true, all_ft = true, any_ft = false, any_unk = false;
        bool has_s = false, has_t = false;
        bool has_kg = false, has_fd = false, has_mg = false, has_sn = false;
        for (int i = 0; i < size; ++i) {
          if (specs[i].empty() ||
              !std::isdigit(static_cast<unsigned char>(specs[i][0]))) {
            all_deck = false;
            break;
          }
          CardId cid = spec_to_card_id(specs[i], player);
          auto iit = g_id_to_code.find(cid);
          uint32_t code = 0;
          if (i < static_cast<int>(sel_codes.size()) && sel_codes[i] != 0) {
            // 消息自带的卡号（准确）
            code = sel_codes[i];
          } else if (iit != g_id_to_code.end() && iit->second != 0) {
            // 隐藏卡/无卡号时退回位置反查（可能不准，仅供统计参考）
            code = static_cast<uint32_t>(iit->second);
          } else {
            any_unk = true;
            all_ft = false;
            continue;
          }
          bool ft = false;
          for (uint32_t k : kFT) {
            if (k == code) { ft = true; break; }
          }
          if (ft) { any_ft = true; } else { all_ft = false; }
          if (code == 19326613u) has_s = true;
          if (code == 82119326u) has_t = true;
          if (code == 86937530u) has_kg = true;
          if (code == 56725612u) has_fd = true;
          if (code == 19144622u) has_mg = true;
          if (code == 55623480u) has_sn = true;
        }
        {
          bool has_g = false;
          for (const auto &sp : specs) {
            auto [gc, gl, gs2, gp] = spec_to_ls(player, sp);
            if (gc >= 0 && gc < 2 && gl < 16 && gs2 < 16 &&
                renamed_[gc][gl][gs2] != 0) {
              has_g = true;
              break;
            }
          }
          if (has_g) ++sel_gyoku_;
        }
        if (std::getenv("YGO_DBG_SEL")) {
          static int dbg_n = 0;
          static int dbg_tot = 0;
          if (dbg_n < 20 && size >= 2 && all_deck) {
            for (int i = 0; i < size && i < static_cast<int>(sel_codes.size());
                 ++i) {
              CardId rc = spec_to_card_id(specs[i], player);
              auto rit = g_id_to_code.find(rc);
              uint32_t rcode = (rit == g_id_to_code.end())
                                   ? 0u
                                   : static_cast<uint32_t>(rit->second);
              ++dbg_tot;
              if (rcode != sel_codes[i]) {
                fmt::print(stderr,
                           "[MISMATCH] spec={} msg={} resolved={} ({} of {})\n",
                           specs[i], sel_codes[i], rcode, dbg_tot,
                           int(dbg_n) + 1);
                ++dbg_n;
              }
            }
          }
        }
        if (all_deck && any_ft) {
          ++mill_any_sel_;
          mill_any_size_ += size;
          if (has_s) ++mill_any_start_;
          if (has_t) ++mill_any_tail_;
          // 直方图：口径 =「全部候选在卡组 + 至少一张妖精传姬」
          if (min == 1 && max == 1) ++hist_11_;
          else if (min == 2 && max == 2) ++hist_22_;
          else if (min == 1 && max == 2) ++hist_12_;
          else ++hist_xx_;
          if (size == 1) ++hist_p1_;
          else if (size == 2) ++hist_p2_;
          else if (size == 3) ++hist_p3_;
          else ++hist_p4p_;
          if (any_unk) ++hist_unk_;
          sel_has_start_ = has_s;
          sel_has_tail_ = has_t;
          // 先挂起，等 MSG_MOVE 的 DECK->GRAVE 确认这是堆墓而不是检索
          pend_pair_ = true;
          pend_s_ = has_s;
          pend_t_ = has_t;
          pend_kg_ = has_kg;
          pend_fd_ = has_fd;
          pend_mg_ = has_mg;
          pend_sn_ = has_sn;
        }
        if (all_deck) {
          ++mill_cand_dsel_;
          mill_cand_dsize_ += size;
        }
        if (all_deck && all_ft) {
          ++mill_cand_sel_;
          mill_cand_size_ += size;
          if (size == 1) ++mill_cand_one_;
          if (has_s) ++mill_cand_start_;
          if (has_t) ++mill_cand_tail_;
        }
      }

      // ── 门控实验：强制堆墓优先级（YGO_MILL_ARM=A/B，默认关闭）──
      {
        static const int mill_arm = []() {
          const char *s = std::getenv("YGO_MILL_ARM");
          if (!s) return 0;
          if (s[0] == 'A' || s[0] == 'a') return 1;
          if (s[0] == 'B' || s[0] == 'b') return 2;
          return 0;
        }();
        static const int arm_player = []() {
          const char *s = std::getenv("YGO_MILL_ARM_PLAYER");
          if (!s) return 0;
          return (s[0] == '1') ? 1 : 0;
        }();
        // 只对「威凯特 2 张送墓」窗口生效（min=max=2 且候选全是妖精传姬），
        // 避免误伤检索窗口（上一版没有这个限制，导致策略崩盘、实验无效）
        static const uint32_t kFTall[] = {
            4026187u,  19144622u, 19326613u, 27632520u, 42921475u,
            52022648u, 55623480u, 56725612u, 65953423u, 78021082u,
            82119326u, 86937530u};
        bool ft_only = !sel_codes.empty();
        for (uint32_t c : sel_codes) {
          bool f = false;
          for (uint32_t k : kFTall) {
            if (k == c) { f = true; break; }
          }
          if (!f) { ft_only = false; break; }
        }
        if (mill_arm != 0 && min == 2 && max == 2 && ft_only &&
            !sel_codes.empty() && player == arm_player) {
          static const uint32_t kArmA[] = {82119326u, 19326613u, 56725612u,
                                           86937530u, 19144622u, 55623480u};
          static const uint32_t kArmB[] = {56725612u, 86937530u, 19144622u,
                                           55623480u, 19326613u, 82119326u};
          const uint32_t *pri = (mill_arm == 1) ? kArmA : kArmB;
          // min==max==2：引擎要求一次回传 min 个下标，回 1 个会被判非法
          std::vector<int> picks;
          for (int k = 0; k < 6 && static_cast<int>(picks.size()) < min; ++k) {
            for (int i = 0; i < static_cast<int>(sel_codes.size()); ++i) {
              if (sel_codes[i] != pri[k]) continue;
              bool dup = false;
              for (int q : picks) {
                if (q == i) { dup = true; break; }
              }
              if (dup) continue;
              picks.push_back(i);
              break;
            }
          }
          if (static_cast<int>(picks.size()) == min) {
            std::string dbg;
            for (int q : picks) {
              dbg += fmt::format("{}:{} ", q, sel_codes[q]);
            }
            fmt::print(stderr, "[ARM] min={} max={} size={} picks={}\n",
                       min, max, size, dbg);
            resp_buf_[0] = min;
            for (int i = 0; i < min; ++i) resp_buf_[i + 1] = picks[i];
            YGO_SetResponseb(pduel_, resp_buf_);
            return;
          }
        }
      }

      last_sel_codes_ = sel_codes;
      last_sel_specs_ = specs;
      if (discard_hand_) {
        discard_hand_ = false;
        if (current_phase_ == PHASE_END) {
          // random discard
          std::vector<int> comb(size);
          std::iota(comb.begin(), comb.end(), 0);
          std::shuffle(comb.begin(), comb.end(), gen_);
          resp_buf_[0] = min;
          for (int i = 0; i < min; ++i) {
            resp_buf_[i + 1] = comb[i];
          }
          YGO_SetResponseb(pduel_, resp_buf_);
          return;
        }
      }

      // 注意：attack_target_pending_ / pending_attacker_atk_ 不在这里清，
      // 否则 step() 里的 _detect_bad_action 拿不到攻击者信息。
      // 清除动作移到 _detect_bad_action 的 MSG_SELECT_CARD 分支。

      // TODO(1): use this when added to history actions
      // if ((min == max) && (max == specs.size())) {
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs, 0, {}, {});

      // 修复（2026-11-19 §159.9）：卡组候选的卡号必须用引擎消息里的真实卡号。
      // spec_to_card_id() 按位置反查 YGO_QueryCard 对卡组位置返回的是另一张卡，
      // 会让模型在「从卡组选卡」时看到错误的候选身份，训练/推理都受影响。
      for (int i = 0; i < static_cast<int>(specs.size()) &&
                      i < static_cast<int>(sel_codes.size()); ++i) {
        set_action_cid_from_code(specs[i], sel_codes[i]);
        cur_win_code_[specs[i]] = sel_codes[i];
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_TRIBUTE) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      cur_win_code_.clear();
      std::vector<uint32_t> sel_codes;
      sel_codes.reserve(size);
      std::vector<int> release_params;
      release_params.reserve(size);
      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          sel_codes.push_back(code);
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();
          Card card = get_card(controller, loc, seq);
          cards.push_back(card);
          release_params.push_back(release_param);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) +
                   " cards to tribute separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < size; ++i) {
          sel_codes.push_back(read_u32());
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          specs.push_back(spec);

          release_params.push_back(release_param);
        }
      }

      bool has_weight =
          std::any_of(release_params.begin(), release_params.end(),
                      [](int i) { return i != 1; });

      if (has_weight) {
        throw std::runtime_error("weight not implemented for select tribute");
        // combs = combinations_with_weight(release_params, min);
      }

      // TODO(1): use this when added to history actions
      // if (max == specs.size()) {
      //   // tribute all
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);
      for (int i = 0; i < static_cast<int>(specs.size()) &&
                      i < static_cast<int>(sel_codes.size()); ++i) {
        set_action_cid_from_code(specs[i], sel_codes[i]);
        cur_win_code_[specs[i]] = sel_codes[i];
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_SUM) {
      // ritual summoning mode 1 (max)
      auto mode = read_u8();
      auto player = read_u8();
      auto val = read_u32();
      int _min = read_u8();
      int _max = read_u8();
      auto must_select_size = read_u8();

      if (mode == 0) {
        if (must_select_size > 2) {
          throw std::runtime_error(
              " must select size: " + std::to_string(must_select_size) +
              " not implemented for MSG_SELECT_SUM");
        }
      } else {
        throw std::runtime_error("mode: " + std::to_string(mode) +
                                 " not implemented for MSG_SELECT_SUM");
      }

      cur_win_code_.clear();
      cur_win_player_ = player;
      std::vector<uint32_t> sum_codes;
      std::vector<int> select_params;
      std::vector<std::string> select_specs;

      int expected = val;
      if (verbose_) {
        std::vector<Card> must_select;
        must_select.reserve(must_select_size);
        for (int i = 0; i < must_select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          must_select.push_back(card);
          expected -= (param & 0xff);
        }
        auto& pl = players_[player];
        pl->notify("Select cards with a total value of " +
                   std::to_string(expected) + ", seperated by spaces.");
        for (const auto &card : must_select) {
          auto spec = card.get_spec(player);
          pl->notify(card.name_ + " (" + spec +
                     ") must be selected, automatically selected.");
        }
      } else {
        for (int i = 0; i < must_select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          expected -= (param & 0xff);
        }
      }

      uint8_t select_size = read_u8();
      select_params.reserve(select_size);
      select_specs.reserve(select_size);

      if (verbose_) {
        std::vector<Card> select;
        select.reserve(select_size);
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          sum_codes.push_back(code);
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          select.push_back(card);
          select_params.push_back(param);
        }
        auto& pl = players_[player];
        for (const auto &card : select) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", select_specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          sum_codes.push_back(read_u32());
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          select_specs.push_back(spec);
          select_params.push_back(param);
        }
      }

      {
        ++sum_sel_;
        bool has_opp = false;
        bool has_gyoku = false;
        for (const auto &sp : select_specs) {
          if (!sp.empty() && sp[0] == 'o') {
            has_opp = true;
            // spec 形如 "o5a"，用 spec_to_card_id 反查
            CardId cid = spec_to_card_id(sp, player);
            if (cid == 19144623u) has_gyoku = true;
          }
        }
        if (has_opp) ++sum_opp_;
        if (has_gyoku) ++sum_gyoku_;
      }
      std::vector<std::vector<int>> card_levels;
      for (int i = 0; i < select_size; ++i) {
        std::vector<int> levels;
        int level1 = select_params[i] & 0xff;
        int level2 = (select_params[i] >> 16);
        if (level1 > 0) {
          levels.push_back(level1);
        }
        if (level2 > 0) {
          levels.push_back(level2);
        }
        card_levels.push_back(levels);
      }

      // We assume any card_level can be the first

      std::vector<std::vector<int>> combs =
          combinations_with_weight2(card_levels, expected);
      
      for (auto &comb : combs) {
        std::sort(comb.begin(), comb.end());
      }

      init_multi_select(
        _min, _max, must_select_size, select_specs, 1, combs);
      for (int i = 0; i < static_cast<int>(select_specs.size()) &&
                      i < static_cast<int>(sum_codes.size()); ++i) {
        set_action_cid_from_code(select_specs[i], sum_codes[i]);
        cur_win_code_[select_specs[i]] = sum_codes[i];
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };

    } else if (msg_ == MSG_SELECT_CHAIN) {
      auto player = read_u8();
      auto size = read_u8();
      auto spe_count = read_u8();
      spe_count_ = spe_count;
      dp_ += 8;  // hint_timing + other_timing
      // 新版把 forced 从头部移到每个连锁项里，位置信息打包成 uint32

      std::vector<CardCode> codes;
      std::vector<uint32_t> descs;
      std::vector<std::string> specs;
      // core: 只要有任意一项 CHAIN_FORCED，-1(取消) 就会被拒并回 MSG_RETRY
      bool forced = false;
      for (int i = 0; i < size; ++i) {
        auto edesc = read_u8();
        auto chain_forced = read_u8();
        if (chain_forced) {
          forced = true;
        }
        CardCode code = read_u32();
        codes.push_back(code);
        uint32_t loc_packed = read_u32();
        PlayerId c = static_cast<PlayerId>(loc_packed & 0xff);
        uint8_t loc = static_cast<uint8_t>((loc_packed >> 8) & 0xff);
        uint8_t seq = static_cast<uint8_t>((loc_packed >> 16) & 0xff);
        uint8_t pos = static_cast<uint8_t>((loc_packed >> 24) & 0xff);
        specs.push_back(ls_to_spec(loc, seq, pos, c != player));
        uint32_t desc = read_u32();
        descs.push_back(desc);
      }

      chain_forced_ = forced;
      if ((size == 0) && (spe_count == 0)) {
        // non-GUI don't need this
        // if (verbose_) {
        //   fmt::println("keep processing");
        // }
        YGO_SetResponsei(pduel_, -1);
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      chaining_player_ = player;
      if (!op->seen_waiting_) {
        if (verbose_) {
          op->notify("Waiting for opponent.");
        }
        op->seen_waiting_ = true;
      }

      if (verbose_) {
        pl->notify("Select chain:");
      }

      for (int i = 0; i < size; i++) {
        CardCode code = codes[i];
        uint32_t desc = descs[i];
        auto spec = specs[i];
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          std::string s = fmt::format(
            "{}: {}({}) ({})",
            i + 1, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (!forced) {
        legal_actions_.push_back(LegalAction::cancel());
        if (verbose_) {
          pl->notify(fmt::format("{}: cancel", size + 1));
        }
      }
      to_play_ = player;
      callback_ = [this, forced](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.act_ == ActionAct::Cancel) {
          if (forced) {
            // 理论上 Cancel 不会出现在 legal_actions_ 里；留作保险：
            // 强制连锁必须选一个效果，选第 0 项而不是 -1
            YGO_SetResponsei(pduel_, 0);
            return;
          }
          YGO_SetResponsei(pduel_, -1);
          return;
        }
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_YESNO) {
      auto player = read_u8();
      auto desc = read_u32();
      auto [code, eff_idx] = unpack_desc(0, desc);
      if (desc == 0) {
        show_buffer();
        auto s = fmt::format("Unknown desc {} in select_yesno", desc);
        throw std::runtime_error(s);
      }
      auto la = LegalAction::activate_spec(eff_idx, "");
      if (code != 0) {
        la.cid_ = c_get_card_id(code);
      }
      legal_actions_.push_back(la);
      if (verbose_) {
        auto& pl = players_[player];
        std::string s;
        if (code == 0) {
          s = get_system_string(eff_idx);
        } else {
          Card c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          eff_idx -= CARD_EFFECT_OFFSET;
          if (c.strings_.empty() ||
              eff_idx >= static_cast<int>(c.strings_.size())) {
            // 卡池外的虚拟卡 / 效果索引越界：用占位符，不中断对局
            s = fmt::format("option (cid {})", code);
          } else {
            auto str = c.strings_[eff_idx];
            if (str.empty()) {
              str = "effect " + std::to_string(eff_idx);
            }
            s = fmt::format("{} ({})", c.name_, str);
          }
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }
      // TODO: maybe add card id to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      auto player = read_u8();

      CardCode code = read_u32();
      auto ct = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      auto desc = read_u32();
      std::string spec = ls_to_spec(loc, seq, pos, ct != player);
      auto [code_d, eff_idx] = unpack_desc(code, desc);
      if (desc == 0) {
        code_d = code;
      }
      auto la = LegalAction::activate_spec(eff_idx, spec);
      if (code_d != 0) {
        la.cid_ = c_get_card_id(code_d);
      }
      legal_actions_.push_back(la);

      if (verbose_) {
        Card c = c_get_card(code);
        auto& pl = players_[player];
        auto name = c.name_;
        std::string s;
        if (code_d == 0) {
          s = get_system_string(desc);
          std::string fmt_str = "[%ls]";
          auto pos = find_substrs(s, fmt_str);
          if (pos.size() == 0) {
            // nothing to replace
          } else if (pos.size() == 1) {
            auto p = pos[0];
            s = s.substr(0, p) + name + s.substr(p + fmt_str.size());
          } else if (pos.size() == 2) {
            auto p1 = pos[0];
            auto p2 = pos[1];
            s = s.substr(0, p1) + spec +
                s.substr(p1 + fmt_str.size(), p2 - p1 - fmt_str.size()) + name +
                s.substr(p2 + fmt_str.size());
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        } else {
          s = fmt::format(
            "{}({}) ({})", c.name_, spec, c.get_effect_description(code_d, eff_idx));
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }

      // TODO: maybe add card info to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_OPTION) {
      auto player = read_u8();
      auto size = read_u8();
      if (verbose_) {
        players_[player]->notify("Select an option:");
      }
      for (int i = 0; i < size; ++i) {
        auto desc = read_u32();
        auto [code, eff_idx] = unpack_desc(0, desc);
        if (desc == 0) {
          show_buffer();
          auto s = fmt::format("Unknown desc {} in select_option", desc);
          throw std::runtime_error(s);
        }
        auto la = LegalAction::activate_spec(eff_idx, "");
        if (code != 0) {
          la.cid_ = c_get_card_id(code);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          std::string s;
          if (code == 0) {
            s = get_system_string(eff_idx);
          } else {
            Card c = c_get_card(code);
            int cmd_idx = legal_actions_.size();
            eff_idx -= CARD_EFFECT_OFFSET;
            if (c.strings_.empty() ||
                eff_idx >= static_cast<int>(c.strings_.size())) {
              // 卡池外的虚拟卡没有效果文本，用占位符，不中断对局
              s = fmt::format("option {} (cid {})", i + 1, code);
            } else {
              auto str = c.strings_[eff_idx];
              if (str.empty()) {
                str = "effect " + std::to_string(eff_idx);
              }
              s = fmt::format("{} ({})", c.name_, str);
            }
          }
          players_[player]->notify(std::to_string(i + 1) + ": " + s);
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_IDLECMD) {
      battle_attack_source_ = 0;
      int32_t player = read_u8();
      auto summonable_ = read_cardlist_spec(player);
      auto spsummon_ = read_cardlist_spec(player);
      auto repos_ = read_cardlist_spec(player);
      auto idle_mset_ = read_cardlist_spec(player);
      auto idle_set_ = read_cardlist_spec(player);
      auto idle_activate_ = read_cardlist_spec(player, true);
      bool to_bp_ = read_u8();
      bool to_ep_ = read_u8();
      read_u8(); // can_shuffle

      int offset = 0;

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Select a card and action to perform.");
      }
      for (const auto &[code, spec, data] : summonable_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Summon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-up attack position", cmd_idx, name));
        }
      }
      offset += summonable_.size();
      int spsummon_offset = offset;
      for (const auto &[code, spec, data] : spsummon_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::SpSummon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Special summon {}", cmd_idx, name));
        }
      }
      offset += spsummon_.size();
      int repos_offset = offset;
      for (const auto &[code, spec, data] : repos_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Repo, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Change position of {}", cmd_idx, name));
        }
      }
      offset += repos_.size();
      int mset_offset = offset;
      for (const auto &[code, spec, data] : idle_mset_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::MSet, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-down defense position", cmd_idx, name));
        }
      }
      offset += idle_mset_.size();
      int set_offset = offset;
      for (const auto &[code, spec, data] : idle_set_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Set, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Set {}", cmd_idx, name));
        }
      }
      offset += idle_set_.size();
      int activate_offset = offset;
      for (const auto &[code_t, spec, desc] : idle_activate_) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: Activate {}({}) ({})",
            cmd_idx, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (to_bp_) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::Battle));
        if (verbose_) {
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format("{}: Enter the battle phase.", cmd_idx));
        }
      }
      if (to_ep_) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::End));
        if (verbose_) {
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format("{}: End phase.", cmd_idx));
        }
      }

      to_play_ = player;
      callback_ = [this, spsummon_offset, repos_offset, mset_offset, set_offset,
                   activate_offset](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.phase_ == ActionPhase::Battle) {
          YGO_SetResponsei(pduel_, 6);
        } else if (action.phase_ == ActionPhase::End) {
          YGO_SetResponsei(pduel_, 7);
        } else {
          auto act = action.act_;
          if (act == ActionAct::Summon) {
            uint32_t idx_ = idx;
            YGO_SetResponsei(pduel_, idx_ << 16);
          } else if (act == ActionAct::SpSummon) {
            uint32_t idx_ = idx - spsummon_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 1);
          } else if (act == ActionAct::Repo) {
            uint32_t idx_ = idx - repos_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 2);
          } else if (act == ActionAct::MSet) {
            uint32_t idx_ = idx - mset_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 3);
          } else if (act == ActionAct::Set) {
            uint32_t idx_ = idx - set_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 4);
          } else if (act == ActionAct::Activate) {
            uint32_t idx_ = idx - activate_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 5);
          }
        }
      };
    } else if (msg_ == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      // TODO(1): add card informaton to select place
      auto player = read_u8();
      auto count = read_u8();
      if (count == 0) {
        count = 1;
      }
      if (count != 1) {
        auto s = fmt::format("Select place count {} not implemented for {}",
                              count, msg_ == MSG_SELECT_PLACE ? "place" : "disfield");
        throw std::runtime_error(s);
      }
      auto flag = read_u32();
      auto places = flag_to_usable_places(flag);
      if (verbose_) {
        auto place_s = msg_ == MSG_SELECT_PLACE ? "place" : "disfield";
        auto s = fmt::format("Select {} for card, one of:", place_s);
        players_[player]->notify(s);
      }
      for (int i = 0; i < places.size(); ++i) {
        legal_actions_.push_back(LegalAction::place(places[i]));
        if (verbose_) {
          auto s = fmt::format("{}: {}", i + 1, action_place_to_string(places[i]));
          players_[player]->notify(s);
        }
      }
      to_play_ = player;
      callback_ = [this, player](int idx) {
        auto place = legal_actions_[idx].place_;
        int i = static_cast<int>(place);
        uint8_t plr = player;
        uint8_t loc;
        uint8_t seq;
        if (
          i >= static_cast<int>(ActionPlace::MZone1) &&
          i <= static_cast<int>(ActionPlace::MZone7)) {
          loc = LOCATION_MZONE;
          seq = i - static_cast<int>(ActionPlace::MZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::SZone1) &&
          i <= static_cast<int>(ActionPlace::SZone8)) {
          loc = LOCATION_SZONE;
          seq = i - static_cast<int>(ActionPlace::SZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::OpMZone1) &&
          i <= static_cast<int>(ActionPlace::OpMZone7)) {
          plr = 1 - player;
          loc = LOCATION_MZONE;
          seq = i - static_cast<int>(ActionPlace::OpMZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::OpSZone1) &&
          i <= static_cast<int>(ActionPlace::OpSZone8)) {
          plr = 1 - player;
          loc = LOCATION_SZONE;
          seq = i - static_cast<int>(ActionPlace::OpSZone1);
        }
        resp_buf_[0] = plr;
        resp_buf_[1] = loc;
        resp_buf_[2] = seq;
        YGO_SetResponseb(pduel_, resp_buf_);
      };
    } else if (msg_ == MSG_SELECT_COUNTER) {
      auto player = read_u8();
      auto counter_type = read_u16();
      int counter_count = read_u16();
      int count = read_u8();
      if (count > 2) {
        throw std::runtime_error("Select counter count " +
                                 std::to_string(count) + " not implemented");
      }
      auto& pl = players_[player];
      if (verbose_) {
        pl->notify(fmt::format("Type new {} for {} card(s), separated by spaces.", "UNKNOWN_COUNTER", count));
      }
      std::vector<int> counters;
      counters.reserve(count);
      for (int i = 0; i < count; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto counter = read_u16();
        counters.push_back(counter & 0xffff);

        if (verbose_) {
          pl->notify(c_get_card(code).name_ + ": " + std::to_string(counter));
        }
        // auto spec = ls_to_spec(loc, seq, 0, controller != player);
        // options_.push_back(spec);
      }
      // TODO(2): implement action
      n_counters_ = count;
      uint16_t resp1 = static_cast<uint16_t>(std::min(counter_count, counters[0]));
      memcpy(resp_buf_, &resp1, 2);
      counter_count -= counters[0];
      if (count == 2) {
        uint16_t resp2 = 0;
        if (counter_count > 0) {
          resp2 = static_cast<uint16_t>(counter_count);
        }
        memcpy(resp_buf_ + 2, &resp2, 2);
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
      auto player = read_u8();
      int count = read_u8();
      std::vector<int> numbers;
      for (int i = 0; i < count; ++i) {
        int number = read_u32();
        if (number <= 0 || number > 12) {
          throw std::runtime_error("Number " + std::to_string(number) +
                                   " not implemented for announce number");
        }
        numbers.push_back(number);
        legal_actions_.push_back(LegalAction::number(number));
      }
      if (verbose_) {
        auto& pl = players_[player];
        std::string str = "Select a number, one of:";
        pl->notify(str);
        for (int i = 0; i < count; ++i) {
          pl->notify(fmt::format("{}: {}", i + 1, numbers[i]));
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {
      auto player = read_u8();
      int count = read_u8();
      auto flag = read_u32();

      int n_attrs = 7;

      std::vector<uint8_t> attrs;
      for (int i = 0; i < n_attrs; i++) {
        if (flag & (1 << i)) {
          attrs.push_back(i + 1);
        }
      }
      // TODO(2): implement action
      if (count != 1) {
        throw std::runtime_error("Announce attrib count " +
                                 std::to_string(count) + " not implemented");
      }

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(count) +
                   " attributes separated by spaces:");
        for (int i = 0; i < attrs.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, attribute_to_string(1 << (attrs[i] - 1))));
        }
      }

      // auto combs = combinations(attrs.size(), count);
      for (int i = 0; i < attrs.size(); i++) {
        legal_actions_.push_back(LegalAction::attribute(1 << (attrs[i] - 1)));
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        uint32_t resp = 0;
        resp |= action.attribute_;
        YGO_SetResponsei(pduel_, resp);
      };
    } else if (msg_ == MSG_ANNOUNCE_CARD) {
      auto player = read_u8();
      int count = read_u8();

      std::vector<uint32_t> opcodes;
      opcodes.reserve(count);
      for (int i = 0; i < count; i++) {
        opcodes.push_back(read_u32());
      }

      auto codes = parse_codes_from_opcodes(opcodes);

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select 1 card from the following cards:");
        for (int i = 0; i < codes.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, c_get_card(codes[i]).name_));
        }
      }

      for (auto code : codes) {
        LegalAction la;
        la.cid_ = c_get_card_id(code);
        la.response_ = code;
        legal_actions_.push_back(la);
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        uint32_t resp = action.response_;
        YGO_SetResponsei(pduel_, resp);
      };
    } else if (msg_ == MSG_SELECT_POSITION) {
      auto player = read_u8();
      auto code = read_u32();
      auto valid_pos = read_u8();
      CardId cid = c_get_card_id(code);

      if (verbose_) {
        auto& pl = players_[player];
        auto card = c_get_card(code);
        pl->notify("Select position for " + card.name_ + ":");
      }

      for (auto pos : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                       POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE}) {
        if (valid_pos & pos) {
          LegalAction la;
          la.cid_ = cid;
          la.position_ = pos;
          legal_actions_.push_back(la);
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            auto& pl = players_[player];
            pl->notify(fmt::format("{}: {}", cmd_idx, position_to_string(pos)));
          }
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        uint8_t pos = legal_actions_[idx].position_;
        YGO_SetResponsei(pduel_, pos);
      };
    } else {
      show_deck(0);
      show_deck(1);
      show_buffer();
      throw std::runtime_error(
        fmt::format("Unknown message {} (msg_id={}), length {}, dp {}",
        msg_to_string(msg_), static_cast<int>(msg_), dl_, dp_));
    }
  }

  void _damage(uint8_t player, uint32_t amount) {
    lp_[player] -= amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp decreased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp decreased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _recover(uint8_t player, uint32_t amount) {
    lp_[player] += amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp increased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp increased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _duel_end(uint8_t player, uint8_t reason) {
    winner_ = player;
    win_reason_ = reason;
    YGO_EndDuel(pduel_);

    duel_started_ = false;
  }
};

class YGOProEnv : public Env<YGOProEnvSpec> {
protected:
  const int max_episode_steps_;
  const int timeout_;

  int elapsed_step_;

  std::uniform_int_distribution<uint64_t> dist_int_;

  // The pool can't be in vector, so we create multiple pools manually
  BS::thread_pool pool0_;
  BS::thread_pool pool1_;
  BS::thread_pool pool2_;
  BS::thread_pool pool3_;
  BS::thread_pool pool4_;

  const int max_timeout_{5};
 
  // YGOProEnvImpl env_impl0_;
  // YGOProEnvImpl env_impl1_;
  // YGOProEnvImpl env_impl2_;
  // YGOProEnvImpl env_impl3_;
  // YGOProEnvImpl env_impl4_;
  std::vector<YGOProEnvImpl> env_impls_;

  bool done_{true};

public:
  YGOProEnv(const Spec &spec, int env_id)
      : Env<YGOProEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        timeout_(spec.config["timeout"_]),
        pool0_(1), pool1_(1), pool2_(1), pool3_(1), pool4_(1),
        dist_int_(0, 0xffffffff) {
    env_impls_.reserve(max_timeout_);
    env_impls_.emplace_back(spec, dist_int_(gen_));
  }

  bool IsDone() override { return done_; }

  BS::thread_pool& get_pool(int idx) {
    switch (idx) {
      case 0: return pool0_;
      case 1: return pool1_;
      case 2: return pool2_;
      case 3: return pool3_;
      case 4: return pool4_;
      default: throw std::runtime_error("Invalid pool index");
    }
  }

  void handle_timeout() {
    env_impls_.emplace_back(spec_, dist_int_(gen_));
    if (env_impls_.capacity() > max_timeout_) {
      throw std::runtime_error("Too many timeouts");
    }
    done_ = true;
    State state = Allocate();
    state["reward"_] = 1.0;
    state["info:to_play"_] = 1;
    state["info:is_selfplay"_] = 1;
    state["info:win_reason"_] = 1;
    state["info:num_options"_] = 1;
    state["info:kaguya_effect"_] = 0;
    state["info:sf_consumed"_] = 0;
    state["info:sf_destroyed"_] = 0;
    state["info:wicat_convert"_] = 0;
    state["info:start_used"_] = 0;
    state["info:tail_used"_] = 0;
    state["info:kaguya_summon0"_] = 0;
    state["info:kaguya_summon1"_] = 0;
    state["info:kaguya_search0"_] = 0;
    state["info:kaguya_search1"_] = 0;
    state["info:kaguya_bounce0"_] = 0;
    state["info:kaguya_bounce1"_] = 0;
    state["info:tail_fusion0"_] = 0;
    state["info:tail_fusion1"_] = 0;
    state["info:tail_oppmat0"_] = 0;
    state["info:tail_oppmat1"_] = 0;
    state["info:gyoku_grave0"_] = 0;
    state["info:gyoku_grave1"_] = 0;
    state["info:story_summon0"_] = 0;
    state["info:story_summon1"_] = 0;
    state["info:rename0"_] = 0;
    state["info:rename1"_] = 0;
    state["info:sum_sel"_] = 0;
    state["info:sum_opp"_] = 0;
    state["info:sum_gyoku"_] = 0;
    state["info:sel_gyoku"_] = 0;
    state["info:pick_gyoku"_] = 0;
    state["info:freeze_cand"_] = 0;
    state["info:story_cand"_] = 0;
    state["info:extra_cand"_] = 0;
    state["info:dsc_ma"_] = 0;
    state["info:dsc_ma_all"_] = 0;
    state["info:dsc_fld"_] = 0;
    state["info:dsc_fld_all"_] = 0;
    state["info:dsc_wr"_] = 0;
    state["info:dsc_wr_all"_] = 0;
    state["info:dsc_ma_i0"_] = 0;
    state["info:dsc_ma_i1"_] = 0;
    state["info:dsc_ma_i2"_] = 0;
    state["info:dsc_ma_i3"_] = 0;
    state["info:dsc_fld_i0"_] = 0;
    state["info:dsc_fld_i1"_] = 0;
    state["info:dsc_fld_i2"_] = 0;
    state["info:dsc_fld_i3"_] = 0;
    state["info:dsc_wr_i0"_] = 0;
    state["info:dsc_wr_i1"_] = 0;
    state["info:dsc_wr_i2"_] = 0;
    state["info:dsc_wr_i3"_] = 0;
    state["info:rename_win"_] = 0;
    state["info:rename_hit"_] = 0;
    state["info:rename_clr"_] = 0;
    state["info:obs_gyoku"_] = 0;
    state["info:obs_spec_fill"_] = 0;
    state["info:opp_fusion_mat"_] = 0;
    state["info:tail_chain"_] = 0;
    state["info:gyoku_id"_] = 0;
    state["info:rn_live"_] = 0;
    state["info:tail_equip"_] = 0;
    state["info:atk_tgt_total"_] = 0;
    state["info:atk_tgt_zero"_] = 0;
    state["info:def_tgt_total"_] = 0;
    state["info:def_tgt_zero"_] = 0;
    state["info:atk_seen"_] = 0;
    state["info:atk_changed"_] = 0;
    state["info:def_changed"_] = 0;
    state["info:start_set"_] = 0;
    state["info:mill_id_a"_] = 0;
    state["info:mill_id_b"_] = 0;
    state["info:mill_cand_sel"_] = 0;
    state["info:mill_cand_size"_] = 0;
    state["info:mill_cand_start"_] = 0;
    state["info:mill_cand_tail"_] = 0;
    state["info:mill_cand_dsel"_] = 0;
    state["info:mill_cand_dsize"_] = 0;
    state["info:mill_cand_one"_] = 0;
    state["info:mill_any_sel"_] = 0;
    state["info:mill_any_size"_] = 0;
    state["info:mill_any_start"_] = 0;
    state["info:mill_any_tail"_] = 0;
    state["info:pair_start_cand"_] = 0;
    state["info:pair_start_pick"_] = 0;
    state["info:pair_tail_cand"_] = 0;
    state["info:pair_tail_pick"_] = 0;
    state["info:opening_ecclesia"_]=0;
    for (int i=0; i<spec_.config["max_options"_]; ++i) for (int j=0;j<3;++j) state["info:droll_event"_](i,j)=0;
    for (int i = 0; i < spec_.config["max_options"_]; ++i) state["info:negation_preview"_](i) = 0;
    for (int i = 0; i < 8; ++i) state["info:chain_effect_slots"_](i) = 0;
    for (int i = 0; i < 4; ++i) state["info:chain_target_context"_](i) = 0;
    state["obs:global_"_][32] = uint8_t(1);
  }

  void Reset() override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    auto fut = pool.submit_task([this, idx]() {
      env_impls_[idx].reset();
    });
    if (fut.wait_for(std::chrono::seconds(timeout_)) != std::future_status::ready) {
      throw std::runtime_error("Reset timeout");
    }

    auto &env_impl = env_impls_[idx];
    elapsed_step_ = 0;
    done_ = false;
    State state = Allocate();
    env_impl.WriteState(state);
  }

  void Step(const Action &action) override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    int action_idx = action["action"_];
    pool.detach_task([this, action_idx, idx]() {
      // Test timeout: random sleep with probability 0.01
      // if (dist_int_(gen_) % 10000 == 0) {
      //   fmt::println("Env {} sleep {}", env_id_, env_impls_.capacity());
      //   std::this_thread::sleep_for(std::chrono::seconds(5));
      //   fmt::println("Env {} after {}", env_id_, env_impls_.capacity());
      //   auto& env_impl = env_impls_[idx];
      //   env_impl.step(action_idx);
      //   std::this_thread::sleep_for(std::chrono::seconds(1));
      //   return;
      // }
      env_impls_[idx].step(action_idx);
    });
    if (!pool.wait_for(std::chrono::seconds(timeout_))) {
      handle_timeout();
      fmt::println("Env {} timeout, new env created", env_id_);
    } else {
      auto& env_impl = env_impls_[idx];
      done_ = env_impl.ret_reward_ != 0;
      State state = Allocate();
      env_impl.WriteState(state);
    }
  }

};

using YGOProEnvPool = AsyncEnvPool<YGOProEnv>;

} // namespace ygopro

template <>
struct fmt::formatter<ygopro::LegalAction>: formatter<string_view> {

    // Format the LegalAction object
    template <typename FormatContext>
    auto format(const ygopro::LegalAction& action, FormatContext& ctx) const {
        std::stringstream ss;
        ss << "{";
        if (!action.spec_.empty()) {
          ss << "spec='" << action.spec_ << "', ";
        }
        if (action.cid_ != 0) {
          ss << "cid=" << action.cid_ << ", ";
        }
        if (action.act_ != ygopro::ActionAct::None) {
          ss << "act=" << ygopro::action_act_to_string(action.act_) << ", ";
        }
        if (action.phase_ != ygopro::ActionPhase::None) {
          ss << "phase=" << ygopro::action_phase_to_string(action.phase_) << ", ";
        }
        if (action.finish_) {
          ss << "finish=true, ";
        }
        if (action.position_ != 0) {
          ss << "position=" << ygopro::position_to_string(action.position_) << ", ";
        }
        if (action.effect_ != -1) {
          ss << "effect=" << action.effect_ << ", ";
        }
        if (action.number_ != 0) {
          ss << "number=" << int(action.number_) << ", ";
        }
        if (action.place_ != ygopro::ActionPlace::None) {
          ss << "place=" << ygopro::action_place_to_string(action.place_) << ", ";
        }
        if (action.attribute_ != 0) {
          ss << "attribute=" << ygopro::attribute_to_string(action.attribute_) << ", ";
        }
        std::string s = ss.str();
        if (s.back() == ' ') {
          s.pop_back();
          s.pop_back();
        }
        s.push_back('}');
        return format_to(ctx.out(), "{}", s);
    }
};

#endif // YGOENV_YGOPRO_YGOPRO_H_
