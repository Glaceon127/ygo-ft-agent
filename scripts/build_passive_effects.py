"""Extract printed, public continuous-effect registrations for the negation policy."""
import argparse
import json
import re
import sqlite3
from pathlib import Path

from build_quick_effects import functions, registrations


def traits(script, monster):
    funcs = functions(re.sub(r'--\[\[.*?\]\]|--[^\n]*', '', script, flags=re.S))
    tags = set()
    for name, body in funcs.items():
        if not name.endswith('.initial_effect'):
            continue
        for effect in registrations(body):
            if 'unknown' in effect or 'EFFECT_FLAG_CANNOT_DISABLE' in effect.get('Property',''):
                continue
            code, kind = effect.get('Code'), effect.get('Type')
            if monster and kind=='EFFECT_TYPE_SINGLE' and effect.get('Range','LOCATION_MZONE')=='LOCATION_MZONE':
                if code in ('EFFECT_DIRECT_ATTACK','EFFECT_PIERCE','EFFECT_EXTRA_ATTACK'):
                    tags.add(code)
            if kind!='EFFECT_TYPE_FIELD':
                continue
            opponent=effect.get('TargetRange','').replace(' ','').split(',')[-1]
            if monster and effect.get('Range')=='LOCATION_MZONE' and opponent=='1':
                if code in ('EFFECT_CANNOT_SPECIAL_SUMMON','EFFECT_CANNOT_TO_HAND'):
                    tags.add(code)
            if code=='EFFECT_CANNOT_ATTACK_ANNOUNCE' and opponent=='LOCATION_MZONE':
                if effect.get('Range')==('LOCATION_MZONE' if monster else 'LOCATION_SZONE'):
                    tags.add(code)
    return sorted(tags)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scripts',type=Path)
    parser.add_argument('database',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result={}
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for code,kind in db.execute('SELECT id,type FROM datas ORDER BY id'):
            path=args.scripts/f'c{code}.lua'
            if path.is_file():
                tags=traits(path.read_text(encoding='utf-8-sig',errors='replace'),bool(kind&1))
                if tags: result[code]=tags
    args.output.write_text(json.dumps(result,indent=1)+'\n',encoding='utf-8')
    print(len(result),'cards')


if __name__=='__main__':
    main()
