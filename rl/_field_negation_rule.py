"""Model action mask for targeted field negation, using public information only."""
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def original_effect_monsters():
    path = Path(__file__).resolve().parent.parent/'assets/locale/en/cards.cdb'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        return frozenset(code for code, in db.execute('SELECT id FROM datas WHERE (type & 32) != 0'))


class FieldNegationRule:
    def __init__(self, ids, sword_slots, candidate_guides=False):
        self.single = [ids[97268402], ids[10045474]]
        self.dual = [ids[56725612], ids[78021082]]
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

    def blocked(self, obs):
        menu, chain, cards = (np.asarray(obs[k]) for k in ('actions_', 'chain_', 'cards_'))
        slots = obs['chain_effect_']
        def card_id(arr, offset):
            return arr[..., offset].astype(np.int32)*256 + arr[..., offset+1]
        action_ids = card_id(menu, 1)
        use = (menu[..., 4] == 8) & (np.isin(action_ids, self.single) |
              (np.isin(action_ids, self.dual) & (menu[..., 6] == 3)))
        blocked = np.zeros_like(use)
        select = np.isin(menu[...,3], [3,9])
        if not (use.any() or select.any()):
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
        guided = np.isin(sources, self.any_slot)
        for source, slot in self.slotted:
            guided |= (sources == source) & (slots == slot)
        for source, slot in self.field_guides:
            guided |= (sources == source) & (slots == slot) & (chain[...,9] == 3)
        trigger = np.maximum(last - targeting.astype(int), 0)
        guide_window = ((obs['global_'][:,7] == 0) &
                        guided[rows,trigger] & enemy[rows,trigger])
        relevant = guide_window & (use.any(1) | (select.any(1) & targeting))
        for b in np.flatnonzero(relevant):
            field = cards[b]
            possible = ((field[:,2] == 3) & (field[:,4] == 1) & (field[:,6] == 0) &
                        np.isin(field[:,5], [1,4,5]))
            matches = np.zeros(len(field),dtype=bool)
            for j in np.flatnonzero(on_field[b] & (chain[b,:,9] == 3)):
                matches |= ((card_id(field,0) == sources[b,j]) &
                            (field[:,3] == int(chain[b,j,10])+1))
            matches &= possible
            if use[b].any():
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
            if targeting[b]:
                ref = menu[b,:,0].astype(int)
                blocked[b] |= select[b] & ~((ref > 0) & matches[np.maximum(ref-1,0)])
        return blocked
