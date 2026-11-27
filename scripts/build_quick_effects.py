"""Compile explicit public monster-effect registrations from the deployed YGOPro scripts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3


def functions(script):
    return {m[1]: m[2] for m in re.finditer(
        r'^function ([\w.]+)\([^\n]*\)[ \t]*\n(.*?)^end[ \t]*$', script, re.M | re.S)}


def registrations(body):
    effects = {}
    indent = re.search(r'(?m)^([ \t]+)\S', body)
    if not indent:
        return
    for line in body.splitlines():
        nested = not line.startswith(indent[1]) or line[len(indent[1]):].startswith((' ', '\t'))
        line = line.strip()
        create = re.fullmatch(r'local (\w+)\s*=\s*Effect\.CreateEffect\(c\)', line)
        clone = re.fullmatch(r'local (\w+)\s*=\s*(\w+):Clone\(\)', line)
        setter = re.fullmatch(r'(\w+):Set(\w+)\((.*)\)', line)
        register = re.fullmatch(r'c:RegisterEffect\((\w+)\)', line)
        if create:
            effects[create[1]] = {'unknown': True} if nested else {}
        elif clone:
            effects[clone[1]] = dict(effects[clone[2]]) if clone[2] in effects else {'unknown': True}
        elif setter and setter[1] in effects:
            if nested:
                effects[setter[1]]['unknown'] = True
            effects[setter[1]][setter[2]] = setter[3]
        elif register and register[1] in effects:
            yield dict(effects[register[1]])


def advance_rule(effect, funcs, code):
    if effect.get('Code') != 'EVENT_FREE_CHAIN':
        return
    cost = re.sub(r'\s+', '', funcs.get(effect.get('Cost'), ''))
    cost = cost.replace('localc=e:GetHandler()', '').replace('e:GetHandler()', 'c')
    self_leaves = cost in {
        'ifchk==0thenreturnc:IsReleasable()endDuel.Release(c,REASON_COST)',
        'ifchk==0thenreturnc:IsAbleToGraveAsCost()endDuel.SendtoGrave(c,REASON_COST)',
        'ifchk==0thenreturnc:IsAbleToRemoveAsCost()endDuel.Remove(c,POS_FACEUP,REASON_COST)',
    }
    condition = re.sub(r'\s+', '', funcs.get(effect.get('Condition'), ''))
    if 'Condition' not in effect:
        timing = 'any'
    elif condition in ('returnDuel.GetTurnPlayer()~=tp','returnDuel.GetTurnPlayer()==1-tp'):
        timing = 'opponent_turn'
    elif condition == 'returnDuel.GetTurnPlayer()==tp':
        timing = 'controller_turn'
    elif condition in ('returnDuel.IsMainPhase()',
                       'returnDuel.GetCurrentPhase()==PHASE_MAIN1orDuel.GetCurrentPhase()==PHASE_MAIN2'):
        timing = 'main'
    else:
        return
    body = re.sub(r'\s+', '', funcs.get(effect.get('Target'), ''))
    detach_one = cost == ('ifchk==0thenreturnc:CheckRemoveOverlayCard(tp,1,REASON_COST)end'
                          'c:RemoveOverlayCard(tp,1,1,REASON_COST)')
    if detach_one:
        operation = funcs.get(effect.get('Operation'), '')
        useful = {'CATEGORY_SPECIAL_SUMMON','CATEGORY_DRAW','CATEGORY_SEARCH','CATEGORY_DESTROY',
                  'CATEGORY_REMOVE','CATEGORY_DISABLE','CATEGORY_TOGRAVE','CATEGORY_TOHAND'}
        if not (set(effect.get('Category','').split('+')) & useful or 'EFFECT_CANNOT_ACTIVATE' in operation):
            return
        if 'Target' not in effect or body.startswith('ifchk==0thenreturntrueend') or body in (
                f'ifchk==0thenreturnDuel.GetFlagEffect(tp,{code})==0end',
                'ifchk==0thenreturnDuel.GetFlagEffect(tp,id)==0end'):
            return [timing, 'material']
        if body.startswith('ifchk==0thenreturnDuel.IsPlayerCanDraw(tp,1)end'):
            return [timing, 'material_deck']
        return
    if not self_leaves:
        # Self-banishment at resolution leaves a window to chain negation.
        # The opponent's hand count is public; check no private card identity.
        operation = re.sub(r'\s+', '', funcs.get(effect.get('Operation'), ''))
        if ('Cost' not in effect and
                body.startswith('ifchk==0thenreturne:GetHandler():IsAbleToRemove()and'
                                'Duel.IsExistingMatchingCard(Card.IsAbleToRemove,tp,0,LOCATION_HAND,1,nil)end') and
                'localc=e:GetHandler()' in operation and 'rg=Group.FromCards(c,rs:GetFirst())' in operation and
                'Duel.Remove(rg,POS_FACEUP,REASON_EFFECT+REASON_TEMPORARY)' in operation):
            return [timing, 'other_hand']
        return
    if 'Target' not in effect or body.startswith('ifchk==0thenreturntrueend'):
        return [timing, 'none']
    # Match the entire feasibility predicate. A hidden summon candidate may
    # exist in the indicated zone; its identity never enters the decision.
    predicate = re.match(r'ifchk==0thenreturn(.+?)end', body)
    if not predicate:
        return
    match = re.fullmatch(
        r'(?:Duel\.(?:GetMZoneCount\(tp,e:GetHandler\(\)\)>0|GetLocationCount\(tp,LOCATION_MZONE\)>-1)and)?'
        r'Duel\.IsExistingMatchingCard\([\w.]+,tp,LOCATION_(DECK|HAND|EXTRA),0,1,nil,e,tp\)',
        predicate[1])
    if match:
        return [timing, match[1].lower()]


def public_cost(effect, funcs):
    if 'Cost' not in effect:
        return 'none'
    body = re.sub(r'\s+', '', funcs.get(effect['Cost'], ''))
    body = body.replace('localc=e:GetHandler()', '').replace('e:GetHandler()', 'c')
    known = {
        'ifchk==0thenreturnc:IsReleasable()endDuel.Release(c,REASON_COST)': 'self_release',
        'ifchk==0thenreturnc:IsAbleToGraveAsCost()endDuel.SendtoGrave(c,REASON_COST)': 'self_grave',
        'ifchk==0thenreturnc:IsAbleToRemoveAsCost()endDuel.Remove(c,POS_FACEUP,REASON_COST)': 'self_banish',
        'ifchk==0thenreturnc:CheckRemoveOverlayCard(tp,1,REASON_COST)endc:RemoveOverlayCard(tp,1,1,REASON_COST)': 'detach_one',
        'ifchk==0thenreturnDuel.IsCanRemoveCounter(tp,1,0,0x1,3,REASON_COST)endDuel.RemoveCounter(tp,1,0,0x1,3,REASON_COST)': 'three_spell_counters',
    }
    return known.get(body, 'unknown')


def catalogue(script_dir, database, *, facts_only=False):
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as db:
        codes = [row[0] for row in db.execute('SELECT id FROM datas WHERE type & 1 ORDER BY id')]
    output = {}
    for code in codes:
        path = script_dir / f'c{code}.lua'
        if not path.is_file():
            continue
        script = re.sub(r'--\[\[.*?\]\]|--[^\n]*', '', path.read_text(encoding='utf-8-sig', errors='replace'), flags=re.S)
        funcs = functions(script)
        body = funcs.get('s.initial_effect', funcs.get(f'c{code}.initial_effect', ''))
        slots = defaultdict(list)
        declared = list(registrations(body))
        for e in declared:
            desc = re.fullmatch(rf'aux\.Stringid\(\s*(?:id|{code})\s*,\s*(\d+)\s*\)', e.get('Description',''))
            if desc and int(desc[1]) < 14:
                slots[int(desc[1]) + 1].append(e)
        quick = []
        pre = {}
        costs = {}
        for slot, effects in sorted(slots.items()):
            if not all(e.get('Type') in ('EFFECT_TYPE_QUICK_O','EFFECT_TYPE_QUICK_F') and
                       e.get('Range') == 'LOCATION_MZONE' and 'unknown' not in e for e in effects):
                continue
            quick.append(slot)
            kinds = {public_cost(e, funcs) for e in effects}
            costs[str(slot)] = kinds.pop() if len(kinds) == 1 else 'unknown'
            rules = []
            for e in effects:
                rule = advance_rule(e, funcs, code)
                limit = e.get('CountLimit')
                if limit and limit != '1':
                    key = limit.replace(' ', '')
                    if not re.fullmatch(r'1,(?:id|\d+)(?:\+\d+)?', key) or any(
                            other.get('CountLimit', '').replace(' ', '') == key and
                            other.get('Description') != e.get('Description') for other in declared):
                        rule = None
                rules.append([*rule, bool(limit)] if rule else None)
            if rules[0] and all(rule == rules[0] for rule in rules):
                pre[str(slot)] = rules[0]
        if quick:
            if facts_only:
                output[str(code)] = costs
                continue
            output[str(code)] = {'quick': quick}
            if pre:
                output[str(code)]['wait'] = pre
    return output


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('ygopro', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--facts-only', action='store_true')
    args = cli.parse_args()
    rows = catalogue(args.ygopro / 'script', (args.ygopro / 'cards.cdb').resolve(), facts_only=args.facts_only)
    lines = [json.dumps(k)+':'+json.dumps(v,separators=(',',':')) for k,v in rows.items()]
    args.output.write_bytes(('{\n'+',\n'.join(lines)+'\n}\n').encode())
    if args.facts_only:
        print(len(rows), 'monsters;', sum(map(len, rows.values())), 'quick-effect slots')
    else:
        print(len(rows), 'monsters;', sum(len(v['quick']) for v in rows.values()), 'quick-effect slots;',
              sum(len(v.get('wait',{})) for v in rows.values()), 'advance timing rules')
