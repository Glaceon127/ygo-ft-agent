"""Public observations for activation and target ranking of field negation."""
import json
from pathlib import Path

import numpy as np
from _quick_timing_inputs import card_ids


TRAITS = ('EFFECT_DIRECT_ATTACK', 'EFFECT_PIERCE', 'EFFECT_EXTRA_ATTACK',
          'EFFECT_CANNOT_SPECIAL_SUMMON', 'EFFECT_CANNOT_TO_HAND')
FEATURES = (*TRAITS, 'attack_position', 'attack_scaled', 'own_lp_scaled', 'enemy_lp_scaled',
            'first_turn', 'own_turn', 'main_phase', 'own_public_attack_restriction',
            'own_defense_position_monster', 'enemy_chain', 'negator_in_hand', 'negator_faceup_szone')
HEAD = 'passive_negation_projection'
DEADLINE_FEATURES = ('main_end_direct_attack_at_least_own_lp',
                     'main_end_direct_attack_at_least_own_lp_under_public_attack_lock')


class PassiveNegationInputs:
    def __init__(self, ids, catalogue, phase_deadline=False):
        self.single = [ids[c] for c in (97268402,10045474)]
        self.main_only = ids[97268402]  # Veiler's printed activation condition.
        self.phase_deadline = phase_deadline
        self.dual = [ids[c] for c in (56725612,78021082)]
        self.effects = np.zeros((65536,len(TRAITS)+1),bool)
        names=(*TRAITS,'EFFECT_CANNOT_ATTACK_ANNOUNCE')
        for code,traits in json.loads(Path(catalogue).read_text()).items():
            if int(code) in ids:
                self.effects[ids[int(code)]] = [t in traits for t in names]

    def __call__(self,obs):
        menu,cards,chain=(obs[k] for k in ('actions_','cards_','chain_'))
        context=obs.get('negation_context_')
        previews=obs.get('negation_zones_')
        width = len(FEATURES) + (len(DEADLINE_FEATURES) if self.phase_deadline else 0)
        output=np.zeros((*menu.shape[:2],2,width),np.float32)
        for b,actions in enumerate(menu):
            ids=card_ids(actions,1)
            use=(actions[:,4]==8) & (np.isin(ids,self.single) | (np.isin(ids,self.dual) &
                ((actions[:,6]==3) | ((actions[:,3]==6) & (actions[:,6]>15)))))
            select=(actions[:,0]>0) & np.isin(actions[:,3],(3,9))
            target_prompt=False
            if select.any() and context is not None and context[b,2]:
                links=np.flatnonzero(chain[b,:,0])
                if len(links):
                    j=links[-1]
                    top=chain[b,j]
                    code=int(card_ids(top,2))
                    target_prompt=top[1]==0 and (code in self.single or
                        (code in self.dual and obs['chain_effect_'][b,j]==2))
            if not (use.any() or target_prompt):
                continue
            field=cards[b]
            face=np.isin(field[:,5],(1,4,5)) & (field[:,6]==0)
            tags=self.effects[card_ids(field,0)]
            hosts=np.flatnonzero(face & (field[:,2]==3) & (field[:,4]==1) &
                                 (field[:,11]!=1) & tags[:,:len(TRAITS)].any(axis=1))
            if not len(hosts):
                continue
            public=obs['global_'][b]
            own=(field[:,4]==0) & face
            own_lock=np.any(own & np.isin(field[:,2],(3,4)) & (field[:,11]!=1) & tags[:,-1])
            facts=np.zeros((len(field),len(FEATURES)),np.float32)
            facts[hosts,:len(TRAITS)]=tags[hosts,:len(TRAITS)]
            facts[hosts,5]=field[hosts,5]==1
            facts[hosts,6]=np.minimum(card_ids(field[hosts],12),8000)/8000
            facts[hosts,7:15]=(min(int(card_ids(public,0)),8000)/8000,
                min(int(card_ids(public,2)),8000)/8000, public[4]==1, bool(public[7]),
                public[5] in (2,8), own_lock, np.any(own & (field[:,2]==3) & (field[:,5]==4)),
                np.any((chain[b,:,0]!=0)&(chain[b,:,1]==1)))
            refs=np.maximum(actions[:,0].astype(int)-1,0)
            for i in np.flatnonzero(use):
                candidates=hosts
                if previews is not None and int(previews[b,i]) & 128:
                    zones=field[hosts,3].astype(int)-1
                    candidates=hosts[((int(previews[b,i])>>zones)&1)!=0]
                if len(candidates):
                    output[b,i,0,:len(FEATURES)]=facts[candidates].max(0)
                    source=field[refs[i]]
                    output[b,i,0,15:17]=(source[2]==2,source[2]==4 and source[5] in (1,4,5))
                    if (self.phase_deadline and ids[i]==self.main_only and public[5]==2
                            and not public[7] and context is not None
                            and context.shape[1]>26 and context[b,26]):
                        direct = candidates[tags[candidates,0] & (field[candidates,5]==1)]
                        if len(direct):
                            atk = card_ids(field[direct],12)
                            lp = int(card_ids(public,0))
                            reaches_lp = bool(np.any((atk<65535) & (atk>=lp)))
                            output[b,i,0,len(FEATURES):] = (reaches_lp, reaches_lp and own_lock)
            if target_prompt:
                output[b,select,1,:len(FEATURES)]=facts[refs[select]]
        return output
