"""Keep all legal targeted negations available for learned threat assessment."""
import numpy as np


class FieldNegationRule:
    def __init__(self, ids, sword_slots, candidate_guides=False):
        self.ballroom = ids[56725612]

    def blocked(self, obs):
        """Leave threat assessment to the actor, including during unrelated chains."""
        menu, cards = np.asarray(obs['actions_']), np.asarray(obs['cards_'])
        blocked = np.zeros(menu.shape[:2], dtype=bool)
        def ids(array, offset):
            return array[...,offset].astype(np.int32)*256+array[...,offset+1]
        ballot = (ids(menu,1)==self.ballroom) & (menu[...,4]==8)
        ballroom = ballot & ((menu[...,6]==3) | ((menu[...,3]==6) & (menu[...,6]>15)))
        select = np.isin(menu[...,3], (3,9)) & (menu[...,0]>0)
        if not (ballroom.any() or select.any()):
            return blocked
        # Only Ballroom retains its separate, previously requested rule about
        # spending the once-per-turn effect on an already disabled monster.
        face = (cards[...,2]==3) & (cards[...,4]==1) & (cards[...,6]==0) & np.isin(cards[...,5],(1,4,5))
        disabled = cards[...,11]==1
        zones = obs.get('negation_zones_')
        for b in np.flatnonzero(ballroom.any(1)):
            enabled = face[b] & ~disabled[b] & (cards[b,:,20]!=0)
            if not enabled.any():
                blocked[b] |= ballroom[b]
            elif zones is not None:
                live = np.bitwise_or.reduce(1 << (cards[b,enabled,3].astype(int)-1))
                blocked[b] |= ballroom[b] & ((zones[b]&128)!=0) & ((zones[b]&live)==0)
        if select.any():
            history = np.asarray(obs['h_actions_'])
            rows = np.arange(len(menu))
            last = history[rows,np.argmax(history[...,3]!=11,axis=1)]
            negated = (ids(last,1)==self.ballroom) & (last[:,4]==8) & (
                (last[:,6]==3) | ((last[:,3]==6) & (last[:,6]>15)))
            chain = np.asarray(obs['chain_'])
            top = np.maximum(np.where(chain[...,0]!=0,np.arange(chain.shape[1]),-1).max(1),0)
            link = chain[rows,top]
            targeting = (link[:,0]!=0) & (link[:,1]==0) & (ids(link,2)==self.ballroom) & (
                obs['chain_effect_'][rows,top]==2)
            context = obs.get('negation_context_')
            if context is not None:
                targeting |= context[:,2].astype(bool)
            for b in np.flatnonzero(negated & targeting & select.any(1)):
                ref = np.maximum(menu[b,:,0].astype(int)-1,0)
                if (select[b] & ~disabled[b,ref]).any():
                    blocked[b] |= select[b] & disabled[b,ref]
        return blocked
