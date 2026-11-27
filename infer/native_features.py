"""Additional actor inputs used by Fairy Tail self-play."""
import numpy as np
from .features import get_code_id, get_spec
from .models import Controller, Location, Position
from .ygoinf_effects import card_effects, chain_effects


def extra_inputs(context, cards, actions, size):
    chain = np.zeros((8,13),np.uint8)
    slots = np.zeros(8,np.uint8)
    for i,entry in enumerate(context.get('entries',[])[:8]):
        code,desc = entry['code'],entry['desc']
        cid,category,effects = get_code_id(code),card_effects(code),chain_effects(code)
        loc = entry['location'] & 127
        chain[i] = [min(entry['link'],7),entry['opponent'],cid>>8,cid&255,
                    *((category>>(8*j))&255 for j in range(4)),entry['negated'],
                    loc.bit_length(),entry['sequence'],effects>>8,effects&255]
        if cid and desc>=10000 and desc>>4==code and (desc&15)<14:
            slots[i]=(desc&15)+1
    battle=np.zeros((size,24),np.float32)
    field={get_spec(c):c for c in cards if c.location==Location.mzone and c.overlay_sequence==-1}
    opponents=[c for c in field.values() if c.controller==Controller.opponent]
    def visible(c):
        return c.position in (Position.faceup_attack,Position.faceup_defense,Position.faceup)
    def defense(c):
        return -1 if any(t.value=='link' for t in c.types) else c.defense
    def combat(c):
        return c.attack if c.position==Position.faceup_attack else defense(c)
    def scale(x):
        return max(-1,min(1,x/12000))
    for i,action in enumerate(actions[:size]):
        target=None
        if action.msg.value=='select_battlecmd' and action.act.value in ('attack','direct_attack'):
            attacker=field.get(action.spec)
        elif action.msg.value=='select_card' and context.get('attack_target'):
            attacker=field.get(context.get('attacker'))
            target=field.get(action.spec)
            if not target or target.controller!=Controller.opponent: continue
        else: continue
        if not attacker or attacker.controller!=Controller.me or not visible(attacker): continue
        f=battle[i]
        atk=attacker.attack
        f[0:4]=[1,target is not None,action.act.value=='direct_attack',atk>=0]
        f[5]=scale(atk) if atk>=0 else 0
        f[17]=min(attacker.sequence,6)/6
        if target:
            f[4],f[12],f[18]=visible(target),not visible(target),min(target.sequence,6)/6
            if visible(target):
                val=combat(target)
                f[6:9]=[scale(v) if v>=0 else 0 for v in (target.attack,defense(target),val)]
                f[10:12]=[target.position==Position.faceup_attack,target.position==Position.faceup_defense]
                if min(val,atk)>=0:
                    f[9]=scale(atk-val)
                    f[13:17]=[1,atk>val,atk==val,atk<val]
        for other in opponents:
            f[19 if visible(other) else 20]+=np.float32(1/7)
            if visible(other) and min(atk,combat(other))>=0:
                val=combat(other)
                f[21 if atk>val else 22 if atk==val else 23]+=np.float32(1/7)
    result = {'chain_':chain,'chain_effect_':slots,'battle_':battle}
    if 'negation_context' in context:
        result['negation_context_'] = np.asarray(context['negation_context'], dtype=np.uint8)
    offered = context.get('negation_preview', {})
    if offered:
        preview = np.zeros(size,np.uint8)
        zones = np.zeros(size,np.uint8)
        for i, action in enumerate(actions[:size]):
            if action.act.value == 'activate' and action.msg.value in ('select_chain','select_effectyn','select_idlecmd'):
                value = offered.get(str(action.response))
                zones[i] = offered.get(f'zones_{action.response}', 0)
                if value in (1,2):
                    preview[i] = value
        result['negation_preview_'] = preview
        result['negation_zones_'] = zones
    return result
