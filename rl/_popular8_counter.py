"""Guide-backed responses for the eight MD matchups on the learner's opponent turn."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'ft_counter_data'
ASH, OGRE, IMPERM, VEILER = 14558127, 59438930, 10045474, 97268402
FIELD_NEGATE = (IMPERM, VEILER)
DROLL, MAXX = 94145021, 23434538
# Post suffix, actual chain source, public effect slot, activation zone, response.
# None accepts any slot only for a card with a single matching activation in that zone.
RULES = (
    ('241fd',59921227,2,4,(ASH,)), ('88bf5',37654623,2,4,(OGRE,)),
    ('965d7',53286626,None,4,(ASH,)), ('965d7',21200905,1,3,(ASH,)),
    ('965d7',99153051,2,3,(ASH,)), ('965d7',24689197,2,3,(ASH,)),
    ('197b2',32548318,None,4,(ASH,)), ('197b2',6153210,None,4,(ASH,)),
    ('6c335',33760966,1,3,FIELD_NEGATE),
    ('c0723',31425736,2,3,(ASH,)), ('a5b91',30581601,1,3,FIELD_NEGATE),
    ('9deb0',31603289,1,3,(ASH,)), ('94b76',66975205,2,4,(OGRE,)),
    ('079a2',9603252,1,3,(IMPERM,)), ('079a2',9603252,2,3,(IMPERM,ASH)),
    ('079a2',9603252,3,3,(IMPERM,)), ('615bd',32353566,1,4,(OGRE,)),
    ('ff246',86120751,2,3,FIELD_NEGATE), ('ff246',57294268,2,3,FIELD_NEGATE),
    ('492ad',41350417,1,4,(ASH,)), ('9fd03',44362883,None,4,(ASH,)),
    ('d0d64',29948294,1,4,(ASH,)), ('d0d64',1984618,None,4,(ASH,)),
    ('f523d',73819701,2,3,FIELD_NEGATE), ('d0d64',73819701,2,3,(ASH,)),
    ('06d74',33907039,0,4,(ASH,)), ('f523d',45883110,1,3,FIELD_NEGATE),
    ('a7049',1498449,1,2,(MAXX,)), ('a7049',32548318,0,4,(MAXX,)),
    ('78bbe',78397661,1,3,(OGRE,)),
    ('d0d64',55273560,1,3,(ASH,)),
    ('def92',43129357,1,3,(OGRE,)), ('def92',92266279,2,4,(OGRE,)),
)

# Source of the addition, required publicly revealed addition (0 = unspecified),
# and whether this must be the opponent's first addition outside the draw phase.
AFTER_SEARCH = (
    ('0091b',2501624,0,False), ('3d382',33907039,0,False), ('3d382',31425736,0,False),
    ('47395',31603289,0,True),
    ('10812',6153210,75003700,False), ('10812',75003700,0,False),
    ('2d363',22623509,0,False), ('2d363',96228804,34041788,False),
    ('2d363',6071005,32353566,False), ('2d363',57916305,22623509,False),
    ('2d363',34041788,22623509,False), ('2d363',32353566,0,False),
    ('01f6b',1984618,0,True), ('01f6b',25451383,0,True),
    ('01f6b',45484331,0,True), ('01f6b',76666602,0,True),
)


def verified_windows():
    first=json.loads((DATA/'popular8_window_check.json').read_text())
    direct={(f,r['source'],r['slot'],r['zone'],c) for f,r in first.items() for c in r['counters']}
    for r in json.loads((DATA/'popular8_additional_openings.json').read_text()):
        direct.add((r['family'],r['source'],r['slot'],r['zone'],r['response']))
    post_search=set()
    path=DATA/'popular8_extended_windows.json'
    if path.exists():
        for r in json.loads(path.read_text())['windows']:
            if r['timing']=='direct':
                direct.add((r['family'],r['source'],r['slot'],r['zone'],r['response']))
            else:
                post_search.add((r['family'],r['source'],r['required_add'],r['first_only']))
    return direct,post_search

def labels_for(ids, verified_only=False):
    posts = json.loads((DATA/'popular8_sources.json').read_text(encoding='utf8'))['posts']
    verified, verified_search = verified_windows() if verified_only else (set(),set())
    names = {ASH:'Ash Blossom & Joyous Spring', OGRE:'Ghost Ogre & Snow Rabbit',
             IMPERM:'Infinite Impermanence', VEILER:'Effect Veiler', MAXX:'Maxx "C"', DROLL:'Droll & Lock Bird'}
    scopes=[]
    response_codes={ids[c]:c for c in names}
    raw_ids={idx:code for code,idx in ids.items()}
    for suffix,source,slot,zone,counters in RULES:
        post, = [r for r in posts if r['id'].endswith(suffix) and not r['archived']]
        allowed={c.get('card') for c in post['counter'] if c.get('impact')!='Backfire'}
        assert all(names[c] in allowed for c in counters)
        family=post['family'].replace(' ','')
        if verified_only:
            counters=[c for c in counters if (family,source,slot or 0,zone,c) in verified]
            if not counters: continue
        scopes.append((family,ids[source],slot or 0,zone,[ids[c] for c in counters],source==6153210))
    searches=[]
    for suffix,source,added,first in AFTER_SEARCH:
        post,=[r for r in posts if r['id'].endswith(suffix) and not r['archived']]
        assert any(c.get('card')==names[DROLL] and c.get('impact')!='Backfire' for c in post['counter'])
        family=post['family'].replace(' ','')
        if not verified_only or (family,source,added,first) in verified_search:
            searches.append((family,source,added,first))

    def labels(obs, families=None, droll_event=None, opening_ecclesia=None, offers=None):
        menu, chain = obs['actions_'],obs['chain_']
        result=np.zeros(menu.shape[:2],bool)
        active=chain[...,0]!=0
        batch=np.arange(len(chain))
        last=np.maximum(0,np.where(active,np.arange(chain.shape[1]),-1).max(1))
        top=chain[batch,last]
        source=top[:,2].astype(int)*256+top[:,3]
        effect=obs['chain_effect_'][batch,last]
        action_id=menu[...,1].astype(int)*256+menu[...,2]
        offer=(menu[...,3]==2)&(menu[...,4]==8)&(menu[...,0]!=0)&~obs['negation_blocked_'].astype(bool)
        window=(obs['global_'][:,7]==0)&(top[:,0]!=0)&(top[:,1]==1)&(top[:,8]==0)
        for family,cid,slot,zone,responses,requires_monster in scopes:
            rows=window&(source==cid)&(top[:,9]==zone)&(effect==slot)
            if cid==ids[55273560]:
                if opening_ecclesia is None: continue
                rows &= opening_ecclesia.astype(bool)
            if families is not None: rows &= families==family
            if requires_monster and rows.any():
                field=obs['cards_']
                rows &= ((field[...,2]==3)&(field[...,4]==0)&(field[...,6]==0)).any(1)
            if rows.any():
                eligible=offer[rows] & np.isin(action_id[rows],responses)
                result[rows] |= eligible
                if offers is not None and eligible.any():
                    batch_rows=np.flatnonzero(rows)
                    for response in responses:
                        rr,aa=np.nonzero(eligible & (action_id[rows]==response))
                        if len(rr):
                            key=f'{family}/direct/{raw_ids[cid]}/{slot}/{zone}/{response_codes[response]}'
                            offers[key]=(batch_rows[rr],aa)
        if droll_event is not None and searches:
            droll=offer&(action_id==ids[DROLL])&(obs['global_'][:,7,None]==0)
            for family,cause,added,first in searches:
                hits=droll&(droll_event[...,0]==cause)
                if families is not None: hits &= (families==family)[:,None]
                if added: hits &= droll_event[...,1]==added
                # Cupsy Way adds two cards at once. Restrict the low-impact
                # recommendation to its opening double search, before Cupsy
                # could already have searched Surprise.
                if first: hits &= droll_event[...,2]==(2 if cause==31603289 else 1)
                result |= hits
                if offers is not None and hits.any():
                    offers[f'{family}/after_search/{cause}/{added}/{int(first)}/{DROLL}']=np.nonzero(hits)
        return result
    return labels
