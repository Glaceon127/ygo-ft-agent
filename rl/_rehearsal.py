"""Auxiliary imitation loss on verified native choices for the live PPO agent."""
import jax
import jax.numpy as jnp
import numpy as np
import hashlib
from pathlib import Path


def load_material_loss(run):
    paths = {
        'material.npz': '9fe3a99e5fbcf581f6eaaa3ae55e3f92591d5985a9b419a078fd892435d07e30',
        'snow_dracotail.npz': 'a2d45ee654a0b4a19998e05d2623b11a4ef46df982c8c861db43bc7c9699461b',
        'snow_witchcrafter.npz': '118415c22fd338602787dd7f9d8c3b55050a545f6277ad5f3185025358f2bc72',
    }
    parts = []
    for name, expected in paths.items():
        path = Path(run) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected training examples: {path}')
        with np.load(path, allow_pickle=False) as file:
            part = dict(file)
            width = part['actions_'].shape[1]
            if width == 24:
                # frame_from stores 24 slots only when all native options fit.
                for key in ('actions_', 'battle_', 'joint_counter_', 'negation_blocked_', 'joint_ids_'):
                    arr = part[key]
                    assert arr.shape[1] == width
                    part[key] = np.pad(arr, [(0, 0), (0, 64-width)] + [(0, 0)] * (arr.ndim-2))
            parts.append(part)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    chosen, rejected = data['chosen'], data['rejected']
    size = len(chosen)
    assert size == len(set(data['seed'])) == 55 and np.all(chosen != rejected)
    for idx in (chosen, rejected):
        assert np.all((idx >= 0) & (idx < data['actions_'].shape[1]))
        assert data['actions_'][np.arange(size), idx, 3].all()
    print(f'Material preference examples: {size}, minibatch 4, coefficient 0.01', flush=True)
    return pair_loss(data, [np.arange(size)], 4, 251)


def load_veiler_loss(path):
    with np.load(path, allow_pickle=False) as file:
        data=dict(file)
    groups=[np.flatnonzero(data['group']==i) for i in range(4)]
    assert [len(g) for g in groups]==[30,30,12,38]
    for idx in (data['chosen'],data['rejected']):
        assert data['actions_'][np.arange(len(idx)),idx,3].all()
    print('Veiler rehearsal: 110 development pairs, one from each of four groups per minibatch, coefficient 0.01',flush=True)
    return pair_loss(data,groups,1,253)


def pair_loss(data,groups,per_group,salt):
    tensors = {k: jnp.asarray(v) for k, v in data.items()}

    def loss(agent, variables, step):
        key = jax.random.fold_in(jax.random.PRNGKey(salt), step)
        keys=jax.random.split(key,len(groups)) if len(groups)>1 else [key]
        idx=jnp.concatenate([jnp.asarray(g)[jax.random.randint(k,(per_group,),0,len(g))] for g,k in zip(groups,keys)])
        obs = {k: v[idx] for k, v in tensors.items() if k.endswith('_')}
        state = jax.tree.unflatten(jax.tree.structure(agent.init_rnn_state(1)),
                                   [tensors['state0'][idx], tensors['state1'][idx]])
        logits = agent.apply(variables, {**obs, 'mask_': None}, state)[1].astype(jnp.float32)
        row = jnp.arange(len(idx))
        margin = logits[row, tensors['chosen'][idx]] - logits[row, tensors['rejected'][idx]]
        return 0.01 * jax.nn.softplus(-margin).mean()
    return loss


def load_loss(path):
    with np.load(path,allow_pickle=False) as file:
        groups = [np.flatnonzero(file['source'] == code) for code in (19144622,78021082,56725612)]
        assert all(len(g) > 0 for g in groups)
        labels = jnp.asarray(file['labels'])
        states = [jnp.asarray(file[f'state{i}']) for i in range(sum(k.startswith('state') for k in file))]
        obs = {k:jnp.asarray(file[k]) for k in file if k.endswith('_')}
    print('Rename demo loss: 2 decisions per source per minibatch, coefficient 0.02, '
          f'training pool sizes {[len(g) for g in groups]}',flush=True)
    def loss(agent, variables, step):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(249),step),len(groups))
        idx = jnp.concatenate([jnp.asarray(g)[jax.random.randint(k,(2,),0,len(g))]
                               for k,g in zip(keys,groups)])
        frames = {k:v[idx] for k,v in obs.items()}
        state = jax.tree.unflatten(jax.tree.structure(agent.init_rnn_state(1)),[v[idx] for v in states])
        logits = agent.apply(variables,{**frames,'mask_':None},state)[1].astype(jnp.float32)
        log_prob = jax.nn.log_softmax(logits)
        return -0.02 * jax.scipy.special.logsumexp(jnp.where(labels[idx],log_prob,-jnp.inf),axis=-1).mean()
    return loss
