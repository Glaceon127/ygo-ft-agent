import os
import numpy as np

import jax
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

import jax.numpy as jnp
import flax
from ygoai.rl.jax.agent import RNNAgent

# md_v4 checkpoint 的 Encoder_0/Embed_0 为 (15050, 1024)；
# ygoenv 训练时传入 (15049, 1024)，embed() 内部 +1。
# 可通过 YGOINF_EMBED_N / YGOINF_EMBED_DIM 覆盖，兼容旧版小 embedding 模型。
EMBED_N = int(os.environ.get("YGOINF_EMBED_N", "15049"))
EMBED_DIM = int(os.environ.get("YGOINF_EMBED_DIM", "1024"))


def create_agent():
    return RNNAgent(
        num_layers=2,
        rnn_channels=512,
        use_history=True,
        rnn_type='lstm',
        num_channels=128,
        film=True,
        noam=True,
        version=2,
        embedding_shape=(EMBED_N, EMBED_DIM),
    )


@jax.jit
def get_probs_and_value(params, rstate, obs):
    agent = create_agent()
    next_rstate, logits, value = agent.apply(params, obs, rstate)[:3]
    probs = jax.nn.softmax(logits, axis=-1)
    return next_rstate, probs, value


def predict_fn(params, rstate, obs):
    obs = jax.tree.map(lambda x: jnp.array([x]), obs)
    rstate, probs, value = get_probs_and_value(params, rstate, obs)
    return rstate, np.array(probs)[0].tolist(), float(np.array(value)[0])

def load_model(checkpoint, rstate, sample_obs, **kwargs):
    agent = create_agent()
    key = jax.random.PRNGKey(0)
    key, agent_key = jax.random.split(key, 2)
    sample_obs_ = jax.tree.map(lambda x: jnp.array([x]), sample_obs)
    params = jax.jit(agent.init)(agent_key, sample_obs_, rstate)
    with open(checkpoint, "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())

    params = jax.device_put(params)
    return params
