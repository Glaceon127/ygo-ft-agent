"""Public effect facts for the optional learned timing policy."""
import json
from pathlib import Path

import numpy as np


COSTS = ('none', 'self_release', 'self_grave', 'self_banish', 'detach_one', 'three_spell_counters', 'unknown')
FEATURES = ('activate', 'select_monster', 'faceup_monster', 'disabled', 'has_material',
            'quick_registered', 'quick_seen_by_name', 'tracked_use', 'history_available',
            'opponent_spell_counters_up_to_three', 'latest_enemy_link_mzone', 'latest_enemy_link_known_quick',
            'latest_enemy_link_departed', 'target_is_latest_enemy_source', *('has_cost_' + k for k in COSTS),
            *('latest_enemy_link_cost_' + k for k in COSTS),
            'quick_monster_with_material', 'quick_monster_without_material',
            'own_field_empty', 'negator_in_hand', 'negator_set', 'normal_summon_offered',
            'own_turn', 'negator_faceup_on_field', 'target_zones_verified',
            'legal_target_has_self_departure_cost', 'legal_target_without_known_self_departure_cost')


def card_ids(x, offset):
    return x[..., offset].astype(np.int32) * 256 + x[..., offset+1]


def with_timing_projection(checkpoint):
    """Return a candidate in memory. Preserve any already-trained projection."""
    params = checkpoint['params']
    width = params['FiLMActor_0']['Dense_0']['kernel'].shape[0]
    shape = (len(FEATURES), width + 1)
    projection = params.get('quick_timing_projection')
    if projection is not None:
        if projection.shape in ((28,width+1),(30,width+1),(34,width+1)):
            expanded = np.pad(projection, ((0,len(FEATURES)-projection.shape[0]),(0,0)))
            return {**checkpoint, 'params': {**params, 'quick_timing_projection': expanded}}
        if projection.shape != shape:
            raise ValueError('Incompatible quick timing projection')
        return checkpoint
    return {**checkpoint, 'params': {**params, 'quick_timing_projection': np.zeros(shape, np.float32)}}


class QuickTimingInputs:
    def __init__(self, ids, catalogue):
        self.single = [ids[c] for c in (97268402, 10045474)]
        self.dual = [ids[c] for c in (56725612, 78021082)]
        self.tracked = [ids[c] for c in (55273560, 26077387, 45819647, 27552504)]
        self.costs = np.zeros((65536, 15), np.uint8)
        for code, slots in json.loads(Path(catalogue).read_text()).items():
            if int(code) in ids:
                for slot, cost in slots.items():
                    self.costs[ids[int(code)], int(slot)] = 1 + COSTS.index(cost)

    def __call__(self, obs):
        menu, cards, chain = (np.asarray(obs[k]) for k in ('actions_', 'cards_', 'chain_'))
        out = np.zeros((*menu.shape[:2], len(FEATURES)), np.float32)
        ctx = obs.get('negation_context_')
        for b, actions in enumerate(menu):
            ids = card_ids(actions, 1)
            negator = np.isin(ids, self.single) | (np.isin(ids, self.dual) &
                      ((actions[:,6] == 3) | ((actions[:,3] == 6) & (actions[:,6] > 15))))
            activate = negator & (actions[:,4] == 8)
            select = np.isin(actions[:,3], (3, 9)) & (actions[:,0] > 0)
            if not (activate.any() or select.any()):
                continue
            field = cards[b]
            printed = card_ids(field, 0)
            enemy_zone = (field[:,2] == 3) & (field[:,4] == 1)
            face = enemy_zone & (field[:,6] == 0) & np.isin(field[:,5], (1, 4, 5))
            facts = np.zeros((len(field), len(FEATURES)), np.float32)
            facts[face,2] = 1
            facts[face,3] = field[face,11] == 1
            material_zones = field[enemy_zone & (field[:,6] != 0),3]
            facts[face,4] = np.isin(field[face,3], material_zones)
            slots = self.costs[printed[face]]
            facts[face,5] = slots.any(1)
            # Aggregate these per monster before pooling an activation menu.
            # An overlay on a different Xyz monster cannot pay this one's cost.
            facts[face,28] = slots.any(1) & (facts[face,4] == 1)
            facts[face,29] = slots.any(1) & (facts[face,4] == 0)
            zones = field[face,3].astype(int) - 1
            if ctx is not None and ctx.shape[1] >= 18:
                bits = ctx[b,4+2*zones].astype(int) | (ctx[b,5+2*zones].astype(int) << 8)
                facts[face,6] = ((slots[:,1:] != 0) & ((bits[:,None] >> np.arange(14)) & 1)).any(1)
                tracked = ((printed[face] == self.tracked[0]) & bool(ctx[b,0] & 1) |
                           (printed[face] == self.tracked[1]) & bool(ctx[b,0] & 2) |
                           np.isin(printed[face], self.tracked[2:]) & (((int(ctx[b,1]) >> zones) & 1) != 0))
                facts[face,7] = tracked
                facts[:,8] = 1
                facts[:,9] = min(int(ctx[b,3]), 3) / 3
            for i in range(len(COSTS)):
                facts[face,14+i] = (slots == i+1).any(1)
            links = np.flatnonzero((chain[b,:,0] != 0) & (chain[b,:,1] == 1))
            if len(links):
                j = links[-1]
                link = chain[b,j]
                if link[1] == 1 and link[9] == 3:
                    source = int(card_ids(link,2))
                    slot = int(obs['chain_effect_'][b,j])
                    cost = self.costs[source,slot] if 0 < slot < 15 else 0
                    facts[:,10:13] = (1, cost != 0, link[10] >= 7)
                    facts[face,13] = (printed[face] == source) & (zones == link[10])
                    if cost:
                        facts[:,14+len(COSTS)+int(cost)-1] = 1
            out[b,activate] = facts.max(0)
            out[b,activate,0] = 1
            refs = np.maximum(actions[:,0].astype(int)-1, 0)
            own_board = (field[:,4]==0) & np.isin(field[:,2],(3,4)) & (field[:,6]==0)
            out[b,activate,30] = not own_board.any()
            offered = field[refs[activate]]
            out[b,activate,31] = offered[:,2]==2
            out[b,activate,32] = (offered[:,2]==4) & np.isin(offered[:,5],(2,6,7))
            out[b,activate,33] = ((actions[:,3]==1) & (actions[:,4]==4)).any()
            out[b,activate,34] = obs['global_'][b,7]!=0
            out[b,activate,35] = np.isin(offered[:,2],(3,4)) & np.isin(offered[:,5],(1,4,5))
            previews = obs.get('negation_zones_')
            if previews is not None:
                masks = previews[b,activate].astype(int)
                verified = (masks & 128)!=0
                legal = verified[:,None] & (((masks[:,None] >> zones) & 1)!=0)
                # A registered cost is public. Its presence does not establish
                # that all other conditions of this effect are currently met.
                self_leaves = facts[face,15:18].any(1)
                out[b,activate,36] = verified
                out[b,activate,37] = (legal & self_leaves).any(1)
                out[b,activate,38] = (legal & ~self_leaves).any(1)
            targets = select & face[refs]
            out[b,targets] = facts[refs[targets]]
            out[b,targets,1] = 1
        return out
