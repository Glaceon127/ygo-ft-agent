"""Joint counter inputs and optional actor heads used by the u586/u1172 checkpoints."""
from pathlib import Path

import numpy as np

from _field_negation_rule import FieldNegationRule

WORK = Path(__file__).resolve().parent
SWORD = (20001443, 55273560, 56465981)


def replace(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f'Incompatible installed code: {before}')
    return source.replace(before,after)


def make_masks(candidate_guides=False):
    ids={int(line.split()[0]):i for i,line in enumerate((WORK/'code_list.txt').read_text().splitlines(),1) if line.strip()}
    sword_slots = {20001443: 1, 55273560: 1, 56465981: 1}
    negation=FieldNegationRule(ids,sword_slots,candidate_guides)
    def decode(x, offset):
        return x[...,offset].astype(np.int32)*256+x[...,offset+1]
    def masks(obs):
        menu,cards,chain=(np.asarray(obs[k]) for k in ('actions_','cards_','chain_'))
        batch=np.arange(menu.shape[0])
        links=chain[...,0]!=0
        i=np.argmax(links,axis=1)
        link=chain[batch,i]
        common=(obs['global_'][:,7]==0)&(links.sum(1)==1)&(link[:,1]==1)
        source=decode(link,2)[:,None]
        slot=obs['chain_effect_'][batch,i,None]
        ref=menu[...,0].astype(np.int32)
        row=cards[batch[:,None],np.maximum(0,ref-1)]
        card=decode(menu,1)
        kind=menu[...,3]
        act=common[:,None]&(ref!=0)&(row[...,4]==0)&(menu[...,4]==8)
        zone=row[...,2]
        ash=(card==ids[14558127])&(zone==2)
        story=(card==ids[4026187])&(zone==3)&(menu[...,6]==3)
        ball=(card==ids[56725612])&(zone==4)&(menu[...,6]==3)
        field=act&((kind==2)|(kind==6)|(kind==7))&(
            ((source==ids[44362883])&(ash|story))|
            ((source==ids[32548318])&ash)|
            ((source==ids[73819701])&(slot==2)&ball))
        popular=act&(kind==2)&(ash|story)&(
            ((source==ids[59921227])&(slot==2))|
            ((source==ids[41350417])&(slot==1))|
            ((source==ids[31425736])&(slot==2)))
        droll=((decode(cards,0)==ids[94145021])&(cards[...,2]==2)&(cards[...,4]==0)).any(1)
        imperm=(card==ids[10045474])&((zone==2)|(zone==4))
        sword=np.stack([act&(kind==2)&(source==ids[c])&(slot==sword_slots[c])&
                         (story|(imperm if n==0 else ash&~droll[:,None]))
                         for n,c in enumerate(SWORD)],axis=-1)
        blocked=negation.blocked(obs)
        flags=np.concatenate((field[...,None],popular[...,None],sword),axis=-1)
        return {**obs,'joint_counter_':flags & ~blocked[...,None], 'negation_blocked_':blocked,
                'joint_ids_':card}
    return masks


def install():
    from ygoai.rl.jax import agent
    def masks(obs):
        flags=obs['joint_counter_'].astype(bool)
        return flags[...,0],flags[...,1],flags[...,2:]
    src=Path(agent.__file__).read_text()
    src=replace(src,'class RNNAgent(nn.Module):','class RNNAgent(nn.Module):\n    joint_counter: bool = False')
    src=replace(src,'def __call__(self, f_state, f_actions, mask, defense_mask=None):',
                'def __call__(self, f_state, f_actions, mask, defense_mask=None, counter_rows=None):')
    src=replace(src,'        if defense_mask is not None:', '''        if counter_rows is not None:
            for name, enabled in zip(("field_delta","popular_delta"), counter_rows):
                delta = self.param(name,nn.initializers.zeros,(f_actions.shape[-1],))
                logits = logits + jnp.where(enabled[:,None], f_actions.astype(jnp.float32) @ delta, 0)
        if defense_mask is not None:''')
    src=replace(src,'            logits = actor(f_state_r, f_actions, mask)\n','''            if self.joint_counter:
                field,popular,sword = joint_masks(x)
                logits = actor(f_state_r,f_actions,mask,counter_rows=(field.any(1),popular.any(1)))
            else:
                logits = actor(f_state_r, f_actions, mask)
''')
    src=replace(src,'        if self.q_head:\n            return rstate, logits, valid',
                '''        if self.joint_counter:
            bias = self.param("sword_bias", nn.initializers.zeros, (3,))
            logits = logits + jnp.sum(sword*bias,axis=-1)
            logits = jnp.where(x["negation_blocked_"], -1e30, logits)
        if self.q_head:
            return rstate, logits, valid''')
    agent.__dict__['joint_masks']=masks
    exec(compile(src,agent.__file__,'exec'),agent.__dict__)
    return masks
