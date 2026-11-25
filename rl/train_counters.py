"""Run full PPO on the assembled counter policy, with signed draw costs."""
import copy
import json
import os
import sys
import threading
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from _joint_counter_model import WORK, install, replace, make_masks

FAIRY='FAIRY_TAIL_Magistus2'
SOURCES=(23434538,42141493,84192580,87126721)
OPP=(FAIRY,'GMaxx','GBird','GWater','GCat','Branded','Dracotail','Blitzclique','DarkMagician','Swordsoul')
POPULAR8=True
HOT=('Blitzclique','Rikka','Dracotail','Yummy','Witchcrafter','Invoked','DarkMagician','Branded')
if POPULAR8:
    OPP=(*OPP[:5],'Swordsoul')*2+HOT
FEE=.01


def cost_for_actor(counts,seat,actor):
    # rewards use the player who chose the action, including opponent responses
    # during the learner's turn. The learner's absolute return is always negative.
    charge=np.asarray(counts)[:,seat,:].sum(-1)*FEE
    return np.where(actor==seat,-charge,charge).astype(np.float32)


class Pools(gym.Env):
    def __init__(self,train,args,seed,size,threads,**kwargs):
        self.pools=[]
        groups=2*len(OPP)
        assert size%groups==0 and threads%groups==0
        for seat in (0,1):
            for name in OPP:
                local=copy.copy(args)
                local.deck1,local.deck2=(FAIRY,name) if seat==0 else (name,FAIRY)
                local.num_actor_threads*=groups  # independent annealing clocks
                self.pools.append(train.make_env(local,seed+10007*len(self.pools),size//groups,threads//groups,**kwargs))
        self.observation_space=self.pools[0].observation_space
        self.action_space=self.pools[0].action_space
        self.num_envs=size
        self.families=np.repeat(OPP*2,size//groups)
        self.learner_seats=np.repeat([0,1],size//2)
        self.draws=np.zeros((2,4),np.int64)
        self.coverage=np.zeros(3,np.int64)
        self.hot_windows=np.zeros(len(HOT),np.int64)
        self.window_counts={}

    @staticmethod
    def combine(rows):
        return jax.tree.map(lambda *xs:np.concatenate(xs,axis=0),*rows)

    def reset(self,**kwargs):
        obs,info=self.combine([pool.reset(**kwargs) for pool in self.pools])
        self.actors=np.array_split(info['to_play'].copy(),len(self.pools))
        return obs,info

    def step(self,action):
        rows=[]
        for i,(pool,moves) in enumerate(zip(self.pools,np.split(action,len(self.pools)))):
            obs,reward,term,trunc,info=pool.step(moves)
            counts=info['g_draws']
            seat=i//len(OPP)
            self.draws[seat]+=counts[:,seat,:].sum(axis=0)
            delta=cost_for_actor(counts,seat,self.actors[i])
            self.actors[i]=info['to_play'].copy()
            rows.append((obs,reward+delta,term,trunc,info))
        return self.combine(rows)

    def close(self):
        for env in self.pools: env.close()


def public_features(obs,info,add_public,popular8,families=None,offers=None):
    if 'negation_preview' in info:
        obs={**obs,'negation_preview_':info['negation_preview']}
    result=add_public(obs)
    if popular8 and families is not None:
        result['popular8_counter_']=popular8(result,families,info['droll_event'],info.get('opening_ecclesia'),offers)
    else:
        result['popular8_counter_']=np.zeros_like(result['negation_blocked_'])
    result.pop('negation_preview_',None)
    return result


def guide_loss(obs,logits,valid):
    yes=obs['joint_counter_'].astype(bool).any(-1)|obs['popular8_counter_'].astype(bool)
    active=yes.any(1)&valid.astype(bool)
    dummy=(~active[:,None])&(jnp.arange(yes.shape[1])==0)
    logp=jax.nn.log_softmax(logits.astype(jnp.float32))
    mass=jax.scipy.special.logsumexp(jnp.where(yes|dummy,logp,-1e30),axis=1)
    return -.05*jnp.where(active,mass,0).sum()/jnp.maximum(active.sum(),1)


def run():
    os.chdir(WORK)
    sys.path.insert(0,str(WORK))
    masks=install()
    add_public=make_masks()
    from _popular8_counter import labels_for
    ids={int(s.split()[0]):i for i,s in enumerate((WORK/'code_list.txt').read_text().splitlines(),1) if s.strip()}
    popular8=labels_for(ids,verified_only=True) if POPULAR8 else None
    class PublicFeatures(gym.Wrapper):
        def __init__(self,env):
            super().__init__(env)
            self.num_envs=env.num_envs
            spaces=dict(env.observation_space.spaces)
            n=spaces['actions_'].shape[0]
            spaces['joint_counter_']=gym.spaces.Box(0,1,(n,5),np.uint8)
            spaces['joint_ids_']=gym.spaces.Box(0,65535,(n,),np.int32)
            spaces['negation_blocked_']=gym.spaces.Box(0,1,(n,),np.uint8)
            spaces['popular8_counter_']=gym.spaces.Box(0,1,(n,),np.uint8)
            self.observation_space=gym.spaces.Dict(spaces)
        def reset(self,**kwargs):
            obs,info=self.env.reset(**kwargs)
            return self.features(obs,info),info
        def features(self,obs,info):
            families=None
            self.offers={}
            if isinstance(self.env,Pools):
                families=np.where(info['to_play']==self.env.learner_seats,self.env.families,'')
            return public_features(obs,info,add_public,popular8,families,self.offers)
        def step(self,action):
            for key,(rows,choices) in self.offers.items():
                counts=self.env.window_counts.setdefault(key,[0,0])
                counts[0]+=len(np.unique(rows))
                counts[1]+=len(np.unique(rows[np.asarray(action)[rows]==choices]))
            obs,*rest=self.env.step(action)
            return (self.features(obs,rest[-1]),*rest)
    import cleanba as train
    run_dir=Path(os.environ['JOINT_RUN']).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cards={int(s.split()[0]):i for i,s in enumerate((WORK/'code_list.txt').read_text().splitlines(),1) if s.strip()}
    def guide(logits,obs):
        # First mirror group uses the normal frozen policy. Other groups prompt
        # the opponent's available G or guide starter to expose training windows.
        m=obs['actions_']
        ids=obs['joint_ids_']
        group=(jnp.arange(m.shape[0])//(m.shape[0]//(2*len(OPP))))%len(OPP)
        allowed=jnp.zeros_like(ids,dtype=bool)
        openers={**dict(zip(OPP[1:5], [(c,) for c in SOURCES])),FAIRY:(),
                 'Branded':(44362883,62962630),'Dracotail':(6153210,32548318),'Blitzclique':(59921227,37654623) if POPULAR8 else (59921227,),
                 'DarkMagician':(41350417,), 'Swordsoul':(20001443,55273560,56465981),
                 'Yummy':(31425736,30581601),'Rikka':(53286626,99153051,21200905),
                 'Witchcrafter':(32353566,9603252),'Invoked':(86120751,57294268)}
        for i,name in enumerate(OPP):
            targets=openers[name]
            for code in targets:
                match=(m[...,4]==8)
                if name not in OPP[:5]: match=match|((m[...,3]==1)&(m[...,4]==4))
                if name in OPP[1:5]: match=match&(obs['global_'][:,7,None]==0)
                allowed=allowed|((group[:,None]==i)&(ids==cards[code])&match)
        return jnp.where(allowed.any(1,keepdims=True)&~allowed,jnp.finfo(logits.dtype).min,logits)
    def coverage(obs,main):
        f,p,s=masks(obs)
        return jnp.array([jnp.sum(a&main) for a in (f.any(1),p.any(1),s.any((1,2)))])
    def hot_coverage(obs,main):
        hit=(obs['popular8_counter_'].any(-1)&main).reshape(2,len(OPP),-1)
        return hit[:,-len(HOT):,:].sum(axis=(0,2))
    active_pools=[]
    def training_env(args,seed,size,threads,**kwargs):
        pool=Pools(train,args,seed,size,threads,**kwargs)
        active_pools.append(pool)
        return pool
    def record(args,update):
        pool=active_pools[0]
        path=run_dir/(args.exp_name+'.samples.jsonl')
        with path.open('a') as f:
            row=dict(update=update,draws=pool.draws.tolist(),counter=pool.coverage.tolist())
            if POPULAR8:
                row['popular8_windows']=dict(zip(HOT,pool.hot_windows.tolist()))
                row['popular8_offered_chosen']=pool.window_counts.copy()
            f.write(json.dumps(row)+'\n')
        pool.window_counts.clear()
        pool.coverage[:]=0
        pool.draws[:]=0
        pool.hot_windows[:]=0
    src=Path(train.__file__).read_text()
    for a,b in (
        ('    args = tyro.cli(Args)','    args = tyro.cli(Args)\n    args.deck=str(joint_dir)\n    args.deck1=args.deck2=joint_fairy'),
        ('return RNNAgent(\n            embedding_shape=args.num_embeddings,\n            dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,\n            param_dtype=jnp.float32,\n            ash_g_ids=args.m2',
         'return RNNAgent(\n            joint_counter=True, embedding_shape=args.num_embeddings,\n            dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,\n            param_dtype=jnp.float32,\n            ash_g_ids=args.m2'),
        ('            freeze_id=args.freeze_id,','            joint_counter=True, freeze_id=args.freeze_id,'),
        ('    envs = make_env(\n        args,\n        local_seed,','    envs = training_env(\n        args,\n        local_seed,'),
        ('        np.random.shuffle(main_player)','        assert np.all(main_player[:args.local_num_envs//2] == 0)'),
        ('    envs = EnvPreprocess(envs, skip_mask=True)','    envs = EnvPreprocess(PublicFeatures(envs), skip_mask=True)'),
        ('    eval_envs = EnvPreprocess(eval_envs, skip_mask=True)','    eval_envs = EnvPreprocess(PublicFeatures(eval_envs), skip_mask=True)'),
        ('    envs = make_env(args, 0, 2, 1)','    envs = PublicFeatures(make_env(args, 0, 2, 1))'),
        ('            logits = jnp.where(main[:, None], logits_a, logits_b)','            logits = jnp.where(main[:, None], logits_a, joint_guide(logits_b,next_obs))'),
        ('        ratios = reshape_time_series(ratios)','        ratios = reshape_time_series(jnp.where(switch_or_mains,ratios,1.0))'),
        ('    for update in range(1, args.num_updates + 2):','    for update in range(1, args.num_updates + 1):'),
        ('            cpu_action = np.array(action)','            cpu_action = np.array(action)\n            active_pools[0].coverage += np.asarray(joint_coverage(next_obs,main))'+
         ('\n            active_pools[0].hot_windows += np.asarray(hot_coverage(next_obs,main))' if POPULAR8 else '')),
        ('        rollout_time.append(time.time() - rollout_time_start)','        joint_record(args,update)\n        rollout_time.append(time.time() - rollout_time_start)'),
        ('        with open(path, "wb") as f:\n            f.write(flax.serialization.to_bytes(obj))','        tmp=str(path)+".tmp"\n        with open(tmp,"wb") as f: f.write(flax.serialization.to_bytes(obj))\n        os.replace(tmp,path)'),
    ):
        src=replace(src,a,b)
    for signature in ('mask, num_steps=None)','mask, num_steps=dones.shape[0] // num_envs)'):
        marker='            '+signature+'\n\n        loss = jnp.where'
        src=replace(src,marker,'            '+signature+'\n        loss += joint_loss(obs,new_logits,mask)\n\n        loss = jnp.where')
    train.__dict__.update(PublicFeatures=PublicFeatures,training_env=training_env,joint_guide=guide,joint_loss=guide_loss,
                          joint_coverage=jax.jit(coverage),hot_coverage=jax.jit(hot_coverage),active_pools=active_pools,joint_record=record,
                          joint_fairy=FAIRY,joint_dir=WORK.parent/"assets/joint_decks")
    exec(compile(src,train.__file__,'exec'),train.__dict__)
    train.main()


if __name__=='__main__':
    def failed(event):
        threading.__excepthook__(event)
        os._exit(1)
    threading.excepthook=failed
    run()
