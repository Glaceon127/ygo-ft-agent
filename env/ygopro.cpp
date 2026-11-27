#include "ygoenv/ygopro/ygopro.h"
#include "ygoenv/core/py_envpool.h"
#include "ygopro-core/effect.h"
namespace ygopro {
int g_draw_source_index(intptr_t handle, int player, int count, int index, uint32_t code) {
  const auto& hand = reinterpret_cast<duel*>(handle)->game_field->player[player].list_hand;
  if (static_cast<int>(hand.size()) < count) throw std::runtime_error("short hand at draw");
  auto* card = hand[hand.size()-count+index];
  if (card->data.code != code || !(card->current.reason & REASON_DRAW))
    throw std::runtime_error("draw message and engine state disagree");
  auto* effect = card->current.reason_effect;
  auto* owner = effect ? effect->get_owner() : nullptr;
  if (!owner) return -1;
  switch (owner->data.code) {
    case 23434538: return 0;
    case 42141493: return 1;
    case 84192580: return 2;
    case 87126721: return 3;
    default: return -1;
  }
}
}


using YGOProEnvSpec = PyEnvSpec<ygopro::YGOProEnvSpec>;
using YGOProEnvPool = PyEnvPool<ygopro::YGOProEnvPool>;

PYBIND11_MODULE(ygopro_ygoenv, m) {
  REGISTER(m, YGOProEnvSpec, YGOProEnvPool)

  m.def("init_module", &ygopro::init_module);

  m.def("set_external_opponent", [](bool v) {
    ygopro::g_external_opponent = v;
    ygopro::g_waiting_external = false;
  });
  m.def("is_waiting_external", []() {
    return ygopro::g_waiting_external;
  });
  m.def("get_num_legal_actions", []() {
    return ygopro::g_num_legal_actions;
  });
  m.def("set_update_data_enabled", [](bool v) {
    ygopro::g_update_data_enabled = v;
  });
  m.def("pop_mirror_msgs", []() {
    pybind11::list out;
    for (const auto &m : ygopro::g_mirror_msgs) {
      out.append(pybind11::bytes(reinterpret_cast<const char *>(m.data()), m.size()));
    }
    ygopro::g_mirror_msgs.clear();
    return out;
  });
  m.def("get_external_buffer", []() {
    return pybind11::bytes(
        reinterpret_cast<const char *>(ygopro::g_external_buffer.data()),
        ygopro::g_external_buffer.size());
  });
}

#include "ygopro-core/effect.h"
#include "ygopro-core/interpreter.h"

uint8_t ygopro::YGOProEnvImpl::opponent_spell_counters() const {
  if (!duel_started_) return 0;
  const auto& field = reinterpret_cast<duel*>(pduel_)->game_field->player[1-to_play_];
  unsigned total = 0;
  for (const auto& zone : {&field.list_mzone, &field.list_szone})
    for (auto* c : *zone) {
      if (c && c->is_position(POS_FACEUP)) total += c->get_counter(1);
      if (total >= 3) return 3;
    }
  return total;
}

void ygopro::YGOProEnvImpl::public_effect_uses(TArray<uint8_t>& result) const {
  const int other = 1 - to_play_;
  for (int z=0; z<7; ++z) {
    uint16_t bits = 0;
    if (duel_started_ && !observed_effects_[other].empty()) {
      auto& monsters = reinterpret_cast<duel*>(pduel_)->game_field->player[other].list_mzone;
      auto* m = z < static_cast<int>(monsters.size()) ? monsters[z] : nullptr;
      if (m && m->is_position(POS_FACEUP)) {
        auto it = observed_effects_[other].find(m->data.code);
        if (it != observed_effects_[other].end()) bits = it->second;
      }
    }
    result(4+2*z) = bits & 255;
    result(5+2*z) = bits >> 8;
  }
}

void ygopro::YGOProEnvImpl::public_negation_targets(TArray<uint8_t>& result) const {
  for (int i=0; i<8; ++i) result(18+i) = 0;
  if (!duel_started_ || chain_stack_.empty() || chain_stack_.size()>8) return;
  const auto& links = reinterpret_cast<duel*>(pduel_)->game_field->core.current_chain;
  for (const auto& link : links) {
    if (!link.chain_count || link.chain_count>chain_stack_.size() ||
        link.triggering_player!=to_play_ || !link.target_cards || link.target_cards->container.size()!=1) continue;
    const auto& observed = chain_stack_[link.chain_count-1];
    if (observed.link_no!=link.chain_count || observed.negated) continue;
    auto* target = *link.target_cards->container.begin();
    if (target->current.location==LOCATION_MZONE && target->current.controler==1-to_play_ &&
        target->current.sequence<7 && target->is_position(POS_FACEUP) && target->is_has_relation(link))
      result(17+link.chain_count) = 1u << target->current.sequence;
  }
}

void ygopro::YGOProEnvImpl::negation_preview(TArray<uint8_t>& result, TArray<uint8_t>& zones) {
    for (int i = 0; i < max_options(); ++i) { result(i) = 0; zones(i) = 0; }
    const bool idle = msg_ == MSG_SELECT_IDLECMD;
    if (!duel_started_ || (!idle && msg_ != MSG_SELECT_CHAIN && msg_ != MSG_SELECT_EFFECTYN)) return;
    auto* game = reinterpret_cast<duel*>(pduel_);
    auto& core = game->game_field->core;
    const bool guide = !chain_stack_.empty() && chain_stack_.size() <= 8 &&
        chain_stack_.back().controler != to_play_ && tp_ != to_play_ &&
        core.current_chain.size() == chain_stack_.size();
    const auto& monsters = game->game_field->player[1-to_play_].list_mzone;
    unsigned faceup = 0;

    for (auto* m : monsters) if (m && m->is_position(POS_FACEUP)) {
      ++faceup;
    }
    // A mixed field needs exact target legality. The public policy identifies
    // the effect slot and consumes these bits only for a relevant decision.
    const bool mixed = faceup >= 2;
    if ((!guide && !mixed) || (msg_ == MSG_SELECT_EFFECTYN && core.select_chains.size() != 1)) return;
    size_t offset = 0;
    if (idle) {
      while (offset < legal_actions_.size() && legal_actions_[offset].act_ != ActionAct::Activate) ++offset;
    }
    for (size_t j = 0; j < core.select_chains.size() && offset+j < legal_actions_.size() &&
                       offset+j < size_t(max_options()); ++j) {
      const size_t i = offset+j;
      const auto& offer = core.select_chains[j];
      auto* e = offer.triggering_effect;
      auto* self = e->get_handler();
      auto code = self->data.code;
      if (idle && (code != 10045474 || self->current.location != LOCATION_HAND)) continue;
      if (code != 97268402 && code != 10045474 &&
          !((code == 56725612 || code == 78021082) && e->description == code * 16u + 1)) continue;
      const auto& action = legal_actions_[i];
      if (!e->target || self->current.controler != to_play_ ||
          action.act_ != ActionAct::Activate || (action.cid_ && action.cid_ != c_get_card_id(code)) ||
          action.spec_ != ls_to_spec(self->current.location, self->current.sequence, 0, false)) continue;
      auto legal_target = [&](::card* target) {
        if (!target || !target->is_position(POS_FACEUP) ||
            !target->is_capable_be_effect_target(e, to_play_)) return false;
        auto* lua = game->lua;
        lua->add_param(e, PARAM_TYPE_EFFECT);
        lua->add_param(to_play_, PARAM_TYPE_INT);
        lua->add_param(offer.evt.event_cards, PARAM_TYPE_GROUP);
        lua->add_param(offer.evt.event_player, PARAM_TYPE_INT);
        lua->add_param(offer.evt.event_value, PARAM_TYPE_INT);
        lua->add_param(offer.evt.reason_effect, PARAM_TYPE_EFFECT);
        lua->add_param(offer.evt.reason, PARAM_TYPE_INT);
        lua->add_param(offer.evt.reason_player, PARAM_TYPE_INT);
        lua->add_param(0, PARAM_TYPE_INT);
        lua->add_param(target, PARAM_TYPE_CARD);
        return bool(lua->check_condition(e->target, 10));
      };
      if (mixed) {
        zones(i) = 128;
        for (size_t z=0; z<monsters.size(); ++z)
          if (legal_target(monsters[z])) zones(i) |= 1u << z;
      }
      if (!guide) continue;
      result(i) = 1;
      for (size_t j = 0; j < chain_stack_.size(); ++j) {
        const auto& observed = chain_stack_[j];
        const auto& link = core.current_chain[j];
        if (observed.controler == to_play_ || observed.loc != LOCATION_MZONE ||
            observed.seq >= 7 || observed.link_no != link.chain_count) continue;
        auto* target = game->game_field->get_field_card(observed.controler, LOCATION_MZONE, observed.seq);
        if (!target || target != link.triggering_effect->get_handler() ||
            !target->is_has_relation(link)) continue;
        if (legal_target(target)) { result(i) = 2; break; }
      }
    }
  }

#include "ygopro-core/group.h"
void ygopro::YGOProEnvImpl::droll_event(TArray<int>& result) {
  for (int i=0; i<max_options(); ++i) for (int j=0;j<3;++j) result(i,j)=0;
  if (!duel_started_ || (msg_ != MSG_SELECT_CHAIN && msg_ != MSG_SELECT_EFFECTYN)) return;
  const auto& offers = reinterpret_cast<duel*>(pduel_)->game_field->core.select_chains;
  if (msg_ == MSG_SELECT_EFFECTYN && offers.size()!=1) return;
  for (size_t i=0;i<offers.size() && i<legal_actions_.size() && i<size_t(max_options());++i) {
    const auto& offer = offers[i];
    auto* e = offer.triggering_effect;
    auto* self = e->get_handler();
    const auto& action = legal_actions_[i];
    const auto& ev = offer.evt;
    if (self->data.code != 94145021 || self->current.controler != to_play_ ||
        action.act_ != ActionAct::Activate || action.cid_ != c_get_card_id(94145021) ||
        action.spec_ != ls_to_spec(self->current.location,self->current.sequence,0,false) ||
        ev.event_code != e->code || !ev.reason_effect || ev.reason_player != 1-to_play_) continue;
    // The guide matches the resolving effect. The drawn card's private name
    // never enters the feature. Only an already publicly revealed hand slot
    // may supply the optional identity for advice about a specific addition.
    auto* cause = ev.reason_effect->get_handler();
    if (!cause) continue;
    result(i,0) = cause->data.code;
    result(i,2) = outside_draw_adds_[1-to_play_];
    if (ev.event_cards && ev.event_cards->container.size()==1) {
      auto* added = *ev.event_cards->container.begin();
      for (const auto& a: search_additions_)
        if (a.player == 1-to_play_ && a.cause==cause->data.code && a.shown && a.shown==added->data.code)
          result(i,1) = a.shown;
    }
  }
}
