"""Eligibility for historical counter-guide supervision, independent of action legality."""
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def original_effect_monsters():
    path = Path(__file__).resolve().parent.parent/'assets/locale/en/cards.cdb'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        return frozenset(code for code, in db.execute('SELECT id FROM datas WHERE (type & 32) != 0'))


class CounterGuideRule:
    def __init__(self, ids, sword_slots, candidate_guides=False):
        self.single = [ids[97268402], ids[10045474]]
        self.dual = [ids[56725612], ids[78021082]]
        self.ballroom = ids[56725612]
        # Native script registrations of single-monster EFFECT_DESTROY_REPLACE
        # that spend exactly this monster's materials, with no usage limit.
        self.replace_overlay = {ids[c]: n for c,n in (
            (9608555,1),(2419596,1),(6906306,1),(98986900,1),(48739166,1),
            (62541668,1),(65305468,1),(8387138,1),(67173574,2)) if c in ids}
        self.target_destruction = [ids[c] for c in (83555666,4178474,79759861) if c in ids]
        self.field_destruction = [ids[c] for c in (12580477,53129443) if c in ids]
        self.ecclesia, self.raye, self.selene, self.beatrice = (
            ids[c] for c in (55273560, 26077387, 45819647, 27552504))
        catalog = json.loads((Path(__file__).resolve().parent/'ft_counter_data/quick_effects.json').read_text(encoding='utf-8'))
        # Slot 0 also encodes unknown/system descriptions. Never classify it.
        self.quick_slots = np.zeros((65536, 15), dtype=bool)
        self.advance_rules = {}
        for printed, effects in catalog.items():
            cid = ids.get(int(printed))
            if cid:
                self.quick_slots[cid, effects['quick']] = True
                if 'wait' in effects:
                    self.advance_rules[cid] = [(int(i), *rule) for i, rule in effects['wait'].items()]
        self.quick_monsters = (self.ecclesia, self.raye, self.selene, self.beatrice, *self.advance_rules)
        self.original_effect = np.zeros(65536, dtype=bool)
        original = original_effect_monsters()
        self.original_effect[[index for code,index in ids.items() if code in original]] = True
        self.any_slot = [ids[44362883], ids[32548318]]
        # Only the named on-field effects can open the guided target filter.
        from _popular8_counter import FIELD_NEGATE, RULES, verified_windows
        direct,_=verified_windows()
        if candidate_guides:
            scopes={(c,slot) for _,c,slot,z,cs in RULES if z==3 and set(cs).intersection(FIELD_NEGATE)}
        else:
            scopes={(c,slot) for _,c,slot,z,r in direct if z==3 and r in FIELD_NEGATE}
        self.field_guides={(ids[c],slot) for c,slot in scopes}
        self.slotted = [(ids[c], s) for c, s in {
            73819701: 2, 59921227: 2, 41350417: 1, 31425736: 2, **sword_slots
        }.items()]

    def _waiting_targets(self, state, field, code, face, slot, context):
        enemy_card = field[:,4] == 1
        field_mon = (field[:,2] == 3) & (field[:,6] == 0)
        spent_names, spent_zones = (0, 0) if context is None else map(int,context[:2])

        main = state[5] in (2, 8)
        damage = state[5] in (5, 6)
        # Hidden names are never inspected. A nonempty private zone may
        # contain a summon target, but does not establish that it does.
        wait = (code == self.ecclesia) & main & ((spent_names & 1) == 0) & bool(state[15] or state[16])
        own_emz = slot[field_mon & enemy_card & (slot >= 5)]
        other_emz = slot[field_mon & ~enemy_card & (slot >= 5)]
        # Raye needs an extra monster zone after releasing herself.
        # A shared zone occupied by the other player has reversed indexing.
        free_emz = len(set(own_emz) | {11 - int(z) for z in other_emz}) < 2
        wait |= ((code == self.raye) & (not damage) & ((spent_names & 2) == 0) &
                 (state[21] > 0) & ((slot >= 5) | free_emz))
        ready = ((spent_zones >> np.minimum(slot, 7)) & 1) == 0
        has_material = np.isin(field[:,3],field[enemy_card & (field[:,2] == 3) & (field[:,6] != 0),3])
        wait |= (code == self.beatrice) & (not damage) & ready & has_material & (state[15] > 0)
        if main and (face & (code == self.selene)).any():
            # Older archives only have the count of the first counter type.
            has_counters = (context is None or len(context) < 4 or context[3] >= 3)
            possible_caster = state[16] > 0 or (enemy_card & (field[:,2] == 5) &
                              (field[:,8] == 2) & (field[:,40] == 0)).any()
            if has_counters and possible_caster:
                occupied = set(slot[enemy_card & field_mon])
                for queen in np.flatnonzero(face & (code == self.selene) & ready):
                    # Selene's three bottom arrows point to own main zones
                    # only from an extra monster zone.
                    start = {5:0, 6:2}.get(slot[queen])
                    if start is not None and any(z not in occupied for z in range(start, start+3)):
                        wait[queen] = True
        if not damage:
            for cid in set(code[face]).intersection(self.advance_rules):
                for index, timing, zone, counted in self.advance_rules[cid]:
                    if (timing == 'opponent_turn' and not state[7]) or (timing == 'controller_turn' and state[7]):
                        continue
                    if timing == 'main' and not main:
                        continue
                    eligible = (code == cid) & face
                    if counted:
                        if context is None or len(context) < 18:
                            continue
                        low = context[4:18:2].astype(int)
                        high = context[5:18:2].astype(int)
                        used = (low | (high << 8)) & (1 << (index-1))
                        eligible &= used[np.minimum(slot,6)] == 0
                    if zone == 'none':
                        wait |= eligible
                    elif zone in ('material', 'material_deck'):
                        wait |= eligible & has_material & (zone == 'material' or state[15] > 0)
                    elif zone == 'other_hand':
                        wait |= eligible & (state[9] > 0)
                    elif state[{'deck':15, 'hand':16, 'extra':21}[zone]] > 0:
                        occupied = slot[enemy_card & field_mon]
                        wait |= eligible & ((slot < 5) | (np.count_nonzero(occupied < 5) < 5))
        return wait & face

    def _allow_replacement(self, obs, blocked, use, select, targeting, on_field, active, sources, last):
        chain, cards, menu = (obs[k] for k in ('chain_', 'cards_', 'actions_'))
        rows = np.arange(len(chain))
        parent = np.maximum(last-targeting.astype(int),0)
        link = chain[rows,parent]
        source = sources[rows,parent]
        single = np.isin(source,self.target_destruction)
        windows = (single | np.isin(source,self.field_destruction)) & active[rows,parent] & (
            link[:,1] == 0) & (link[:,8] == 0) & (link[:,9] == 4) & ~on_field.any(1)
        ctx = obs.get('negation_context_')
        zones = obs.get('negation_zones_')
        for b in np.flatnonzero(windows & (use.any(1) | (targeting & select.any(1)))):
            field = cards[b]
            enemy = (field[:,2] == 3) & (field[:,4] == 1)
            host = enemy & (field[:,6] == 0) & np.isin(field[:,5], (1,4,5))
            seq = np.maximum(field[:,3].astype(int)-1,0)
            code = field[:,0].astype(int)*256 + field[:,1]
            overlay = np.bincount(seq[enemy & (field[:,6] != 0)], minlength=7)
            candidates = np.zeros(len(field),bool)
            for i in np.flatnonzero(host & (field[:,11] != 1) & (field[:,20] != 0)):
                cost = self.replace_overlay.get(code[i])
                if cost and overlay[seq[i]] >= cost:
                    candidates[i] = True
            if single[b]:
                target = int(ctx[b,18+parent[b]]) if ctx is not None and ctx.shape[1]>=26 else 0
                candidates &= (target & (1 << seq)) != 0
            if not candidates.any():
                continue
            allowed = int(np.bitwise_or.reduce(1 << seq[candidates]))
            if targeting[b]:
                ref = menu[b,:,0].astype(int)
                offered = select[b] & (ref>0)
                eligible = candidates[np.maximum(0,ref-1)]
                if (offered & eligible).any():
                    blocked[b] |= offered & ~eligible
            else:
                # With a mixed field the engine only guarantees *some* legal
                # target. Check the actual candidate zones for each negator.
                verified = False if zones is None else ((zones[b]&128)!=0) & ((zones[b]&allowed)!=0)
                if np.all(~host | candidates):
                    verified = True
                blocked[b] &= ~(use[b] & verified)

    def blocked(self, obs, *, quick_timing=True):
        """Historical counter-only mask for reproducible before/after audits."""
        menu, chain, cards = (np.asarray(obs[k]) for k in ('actions_', 'chain_', 'cards_'))
        slots = obs['chain_effect_']
        def card_id(arr, offset):
            return arr[..., offset].astype(np.int32)*256 + arr[..., offset+1]
        action_ids = card_id(menu, 1)
        use = (menu[..., 4] == 8) & (np.isin(action_ids, self.single) |
              (np.isin(action_ids, self.dual) & (menu[..., 6] == 3)))
        blocked = np.zeros_like(use)
        select = np.isin(menu[...,3], [3,9])
        # A face-up Ballroom or Writer has only one optional on-summon trigger.
        # SELECT_EFFECTYN can encode a system prompt instead of its local slot.
        row = np.arange(len(cards))[:, None]
        source_ref = menu[..., 0].astype(int)
        idx = np.maximum(source_ref - 1, 0)
        source_zone = cards[row, idx, 2]
        trigger_source = (((action_ids == self.ballroom) & (source_zone == 4)) |
                          ((action_ids == self.dual[1]) & (source_zone == 3)))
        optional_trigger = (trigger_source & (source_ref > 0) & (cards[row, idx, 4] == 0) &
                            np.isin(cards[row, idx, 5], (1, 4, 5)) & (menu[..., 3] == 6) &
                            (menu[..., 4] == 8) & (menu[..., 6] > 15))
        quick_use = use | optional_trigger
        if not (quick_use.any() or select.any()):
            return blocked
        sources = card_id(chain, 2)
        active = chain[..., 0] != 0
        enemy = active & (chain[..., 1] == 1)
        on_field = enemy & np.isin(chain[..., 9], [3, 4])
        blocked |= use & ~on_field.any(1)[:, None]

        # During target selection the negator itself is already on the chain.
        last = np.maximum(0, np.where(active, np.arange(chain.shape[1]), -1).max(1))
        rows = np.arange(len(chain))
        own = active[rows,last] & ~enemy[rows,last]
        targeting = own & (np.isin(sources[rows,last], self.single) |
                    (np.isin(sources[rows,last], self.dual) & (slots[rows,last] == 2)))
        # A Trap can have a zone-selection prompt between activation and
        # target selection. A later unrelated card-selection prompt during
        # resolution must not inherit this restriction.
        history = obs['h_actions_']
        recent = np.argmax(history[...,3] != 11,axis=1)
        choice = history[rows,recent]
        targeting &= ((choice[:,4] == 8) & (card_id(choice,1) == sources[rows,last]))
        self._allow_replacement(obs, blocked, use, select, targeting, on_field, active, sources, last)
        guided = np.isin(sources, self.any_slot)
        for source, slot in self.slotted:
            guided |= (sources == source) & (slots == slot)
        for source, slot in self.field_guides:
            guided |= (sources == source) & (slots == slot) & (chain[...,9] == 3)
        trigger = np.maximum(last - targeting.astype(int), 0)
        guide_window = ((obs['global_'][:,7] == 0) &
                        guided[rows,trigger] & enemy[rows,trigger])
        ballroom_offer = quick_use & (action_ids == self.ballroom)
        ballroom_select = targeting & (sources[rows,last] == self.ballroom) & select.any(1)
        relevant = guide_window & (use.any(1) | (select.any(1) & targeting))
        for b in np.flatnonzero(relevant | ballroom_offer.any(1) | ballroom_select):
            field = cards[b]
            possible = ((field[:,2] == 3) & (field[:,4] == 1) & (field[:,6] == 0) &
                        np.isin(field[:,5], [1,4,5]))
            matches = np.zeros(len(field),dtype=bool)
            for j in np.flatnonzero(on_field[b] & (chain[b,:,9] == 3)):
                matches |= ((card_id(field,0) == sources[b,j]) &
                            (field[:,3] == int(chain[b,j,10])+1))
            matches &= possible
            disabled = field[:,11] == 1
            if ballroom_offer[b].any():
                candidates = matches if guide_window[b] else possible
                if not (candidates & (field[:,20] != 0) & ~disabled).any():
                    blocked[b] |= ballroom_offer[b]
            if not relevant[b] and not ballroom_select[b]:
                continue
            if relevant[b] and use[b].any():
                preview = obs.get('negation_preview_')
                verdict = preview[b] if preview is not None else np.zeros_like(use[b], dtype=np.uint8)
                # Before paying the cost the engine only certifies that *some*
                # target is legal. If all possible targets are chain sources,
                # its legal menu is guaranteed to contain an allowed target.
                effect = field[:,20] != 0
                enabled = field[:,11] != 1
                for group, candidates in (
                        ([self.single[0]], possible & enabled & effect),
                        ([self.single[1]], possible & enabled &
                         (effect | self.original_effect[card_id(field,0)])),
                        (self.dual, possible & effect)):
                    uncertain = not (candidates & matches).any() or (candidates & ~matches).any()
                    deny = (verdict == 1) | ((verdict != 2) & uncertain)
                    blocked[b] |= use[b] & np.isin(action_ids[b], group) & (deny | ~matches.any())
            if relevant[b] and targeting[b]:
                ref = menu[b,:,0].astype(int)
                blocked[b] |= select[b] & ~((ref > 0) & matches[np.maximum(ref-1,0)])
            if ballroom_select[b]:
                # The engine supplies the actual legal targets only after activation.
                # Keep a mandatory choice if every remaining target is disabled.
                ref = menu[b,:,0].astype(int)
                available = select[b] & (ref > 0) & ~blocked[b]
                already_disabled = disabled[np.maximum(ref-1,0)]
                if (available & ~already_disabled).any():
                    blocked[b] |= available & already_disabled
        if not quick_timing:
            return blocked
        context = obs.get('negation_context_')
        previous = card_id(choice,1)
        disable_prompt = (np.zeros(len(cards),bool) if context is None else context[:,2].astype(bool))
        pending_target = disable_prompt & (choice[:,4] == 8) & (
            np.isin(previous,self.single) | (np.isin(previous,self.dual) &
            ((choice[:,6] == 3) | ((choice[:,3] == 6) & (choice[:,6] > 15)))))
        for b in np.flatnonzero(quick_use.any(1) | (select.any(1) & (targeting | pending_target))):
            field = cards[b]
            code = card_id(field,0)
            enemy_card = field[:,4] == 1
            field_mon = (field[:,2] == 3) & (field[:,6] == 0)
            face = enemy_card & field_mon & np.isin(field[:,5], (1,4,5))
            top = last[b] - int(targeting[b])
            departed = top >= 0 and enemy[b,top] and chain[b,top,9] == 3 and chain[b,top,10] >= 7
            active_quick = (top >= 0 and enemy[b,top] and chain[b,top,9] == 3 and
                            self.quick_slots[sources[b,top], slots[b,top]])
            if not (departed or active_quick or np.isin(code[face],self.quick_monsters).any()):
                continue
            slot = np.maximum(field[:,3].astype(int) - 1, 0)
            wait = self._waiting_targets(obs['global_'][b], field, code, face, slot,
                                         None if context is None else context[b])
            # A source which paid a leave-field cost has no live monster slot.
            # Keep earlier on-field links eligible, never substitute a new occupant.
            links = np.flatnonzero(on_field[b] & (chain[b,:,9] == 3))
            bound = np.zeros(len(field),bool)
            for j in links:
                same = face & (code==sources[b,j]) & (slot==chain[b,j,10])
                bound |= same
                if (sources[b,j] in (self.selene,self.beatrice) and slots[b,j]==2):
                    wait &= ~same
            restrict = departed or active_quick
            if not (wait.any() or restrict):
                continue
            valid = face & ~wait
            if restrict and chain[b,top,10] < 7:
                valid &= (code == sources[b,top]) & (slot == chain[b,top,10])
            elif restrict or guide_window[b]:
                valid &= bound
            for negator in self.single+self.dual:
                effect = field[:,20]!=0
                if negator == self.single[1]:
                    effect |= self.original_effect[code]
                possible = face & effect
                if negator in self.single or negator == self.ballroom:
                    possible &= field[:,11]!=1
                candidates = valid & possible
                if candidates.any() and (possible & ~valid).any():
                    # The activation prompt only promises some legal target. If
                    # the only safe-looking alternative is protected, pass now.
                    allowed_zones = int(np.bitwise_or.reduce(1 << slot[candidates]))
                    zones = obs.get('negation_zones_')
                    verified = False if zones is None else ((zones[b] & 128) != 0) & ((zones[b] & allowed_zones) != 0)
                    blocked[b] |= quick_use[b] & (action_ids[b] == negator) & np.logical_not(verified)
                if restrict and 'negation_preview_' in obs:
                    blocked[b] |= quick_use[b] & (action_ids[b] == negator) & (obs['negation_preview_'][b] == 1)
                if not candidates.any():
                    blocked[b] |= quick_use[b] & (action_ids[b]==negator)
            if targeting[b] or pending_target[b]:
                ref = menu[b,:,0].astype(int)
                offered = select[b] & (ref>0) & ~blocked[b]
                negator = previous[b] if pending_target[b] else sources[b,last[b]]
                if negator == self.ballroom:
                    valid &= field[:,11] != 1
                allowed = valid[np.maximum(ref-1,0)]
                if (offered & allowed).any():
                    blocked[b] |= offered & ~allowed
        # On our own main phase a hand Imperm can expire as a response when we
        # commit a monster. Keep the timing choice open when the real target
        # menu includes a face-up quick-effect monster. Never force activation.
        imp = self.single[1]
        bait = use & (menu[...,3]==1) & (action_ids==imp) & (source_zone==2)
        idle_turn = (obs['global_'][:,7]!=0) & np.isin(obs['global_'][:,5],(2,8))
        open_offer = bait.any(1) & ~active.any(1) & (menu[...,4]==4).any(1)
        select_bait = (targeting | pending_target) & (last==0) & (previous==imp) & (choice[:,3]==1)
        for b in np.flatnonzero(idle_turn & (open_offer | (select_bait & select.any(1)))):
            field = cards[b]
            face = (field[:,2]==3) & (field[:,4]==1) & (field[:,6]==0) & np.isin(field[:,5],(1,4,5))
            fast = face & (field[:,11]!=1) & (field[:,20]!=0) & self.quick_slots[card_id(field,0)].any(1)
            if not fast.any():
                continue
            if open_offer[b]:
                bits = int(np.bitwise_or.reduce(1 << (field[fast,3].astype(int)-1)))
                zones = obs.get('negation_zones_')
                verified = False if zones is None else ((zones[b]&128)!=0) & ((zones[b]&bits)!=0)
                if np.all(~face | fast):
                    verified = True
                blocked[b] &= ~(bait[b] & verified)
            else:
                ref = menu[b,:,0].astype(int)
                eligible = (ref>0) & fast[np.maximum(0,ref-1)]
                if (select[b] & eligible).any():
                    blocked[b,select[b]] = ~eligible[select[b]]
        return blocked
