"""Joint counter inputs and optional timing and negation heads."""
from pathlib import Path

import numpy as np

from _field_negation_rule import FieldNegationRule
from _counter_guide_rule import CounterGuideRule

WORK = Path(__file__).resolve().parent
SWORD = (20001443, 55273560, 56465981)


def replace(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f'Incompatible installed code: {before}')
    return source.replace(before,after)


def make_masks(candidate_guides=False, *, learned_timing=False, keep_field_rules=False, for_supervision=False,
               passive_negation=False, phase_deadline=False):
    ids={int(line.split()[0]):i for i,line in enumerate((WORK/'code_list.txt').read_text().splitlines(),1) if line.strip()}
    sword_slots = {20001443: 1, 55273560: 1, 56465981: 1}
    negation=FieldNegationRule(ids,sword_slots,candidate_guides)
    guides=CounterGuideRule(ids,sword_slots,candidate_guides)
    if learned_timing:
        from _quick_timing_inputs import QuickTimingInputs
        timing_inputs = QuickTimingInputs(ids, WORK / 'ft_counter_data/quick_costs.json')
    if passive_negation:
        from _passive_negation_inputs import PassiveNegationInputs
        passive_inputs=PassiveNegationInputs(ids,WORK/'ft_counter_data/passive_effects.json',phase_deadline)
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
        if learned_timing:
            obs = {**obs, 'quick_timing_': timing_inputs(obs)}
        if passive_negation:
            obs={**obs,'passive_negation_':passive_inputs(obs)}
        if learned_timing and not keep_field_rules:
            blocked = np.zeros(menu.shape[:2], dtype=bool)
        else:
            blocked=negation.blocked(obs)
        flags=np.concatenate((field[...,None],popular[...,None],sword),axis=-1)
        guide_blocked=guides.blocked(obs) if for_supervision or flags.any() else blocked
        obs={k:v for k,v in obs.items() if k not in ('negation_context_', 'negation_zones_')}
        if for_supervision:
            obs['counter_guide_blocked_']=guide_blocked | blocked
        return {**obs,'joint_counter_':flags & ~(guide_blocked | blocked)[...,None], 'negation_blocked_':blocked,
                'joint_ids_':card}
    return masks


def install(*, initialize_deadline=False):
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
            if "quick_timing_" in x:
                facts = x["quick_timing_"].astype(jnp.float32)
                state = f_state_r.astype(jnp.float32)
                state = jnp.concatenate((state, jnp.ones_like(state[:,:1])), axis=-1)
                projection = self.param("quick_timing_projection", nn.initializers.zeros,
                                        (facts.shape[-1], state.shape[-1]))
                # This head was trained on activate/pass pairs. Leave target
                # ranking and unrelated activations to the existing actor.
                timing_offer = (facts[...,0] > 0) & ((facts[...,5] > 0) | (facts[...,11] > 0))
                delta = jnp.einsum("baf,bf->ba", facts, state @ projection.T)
                logits += jnp.where(timing_offer, delta, 0.)
            if "passive_negation_" in x:
                facts = x["passive_negation_"].astype(jnp.float32)
                state = f_state_r.astype(jnp.float32)
                state = jnp.concatenate((state, jnp.ones_like(state[:,:1])), axis=-1)
                projection = self.param("passive_negation_projection", nn.initializers.zeros,
                                        (*facts.shape[-2:], state.shape[-1]))
                context = jnp.einsum("bw,sfw->bsf", state, projection)
                logits += jnp.einsum("basf,bsf->ba", facts, context)
                if self.has_variable("params", "phase_deadline_calibration") or (joint_init_deadline and self.is_initializing()):
                    calibration = self.param("phase_deadline_calibration", nn.initializers.zeros, (3,))
                    deadline, locked = facts[:,:,0,-2], facts[:,:,0,-1]
                    passes = x["actions_"][:,:,4] == 9
                    pass_score = jnp.take_along_axis(logits, jnp.argmax(passes, axis=1)[:,None], axis=1)
                    prior = jnp.clip(logits - pass_score, -40., 40.)
                    shift = calibration[0]*deadline + calibration[1]*locked + calibration[2]*deadline*prior
                    logits += jnp.where(deadline.astype(bool) & passes.any(axis=1)[:,None], shift, 0.)
            logits = jnp.where(x["negation_blocked_"], -1e30, logits)
        if self.q_head:
            return rstate, logits, valid''')
    agent.__dict__['joint_masks']=masks
    agent.__dict__['joint_init_deadline']=initialize_deadline
    exec(compile(src,agent.__file__,'exec'),agent.__dict__)
    return masks
