"""Serve a joint-counter checkpoint using the feature layout of native self-play."""
import argparse
import os
from pathlib import Path

import flax.serialization
import jax
import numpy as np
import uvicorn
from scipy.special import softmax

from _joint_counter_model import WORK, install, make_masks


def joint_predictor(checkpoint, num_threads):
    from battle import Args, create_agent1
    from ygoinf.features import Predictor

    config = Args(num_embeddings=15049)
    config.m1.battle_features = True
    config.m1.chain_effect_features = True
    config.m1.chain_response_features = True
    config.m1.own_turn_ash_g = True
    agent = create_agent1(config).clone(joint_counter=True)
    weights = flax.serialization.msgpack_restore(Path(checkpoint).read_bytes())
    if "sword_bias" not in weights["params"]:
        raise ValueError("Expected a joint-counter checkpoint with sword_bias")
    timing = "quick_timing_projection" in weights["params"]
    if timing:
        from _quick_timing_inputs import with_timing_projection
        weights = with_timing_projection(weights)
    from _passive_negation_inputs import FEATURES, HEAD
    passive = weights["params"].get(HEAD)
    mask = make_masks(learned_timing=timing, keep_field_rules=True, passive_negation=passive is not None,
                      phase_deadline=passive is not None and passive.shape[1]>len(FEATURES))
    predict = jax.jit(lambda params, obs, state: agent.apply(params, obs, state)[:2])

    def score(params, state, frame):
        obs = mask({key: value[None] for key, value in frame.items()})
        obs.pop("negation_preview_", None)
        obs["mask_"] = None
        state, logits = predict(params, obs, state)
        return state, softmax(np.asarray(logits)[0]).tolist(), None

    predictor = Predictor(weights, score)
    predictor.native_policy = True
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--port", default=3000, type=int)
    args = parser.parse_args()
    os.environ["CHECKPOINT"] = str(args.checkpoint.resolve(strict=True))
    os.chdir(WORK)
    install()
    from ygoinf.features import Predictor
    Predictor.load = staticmethod(joint_predictor)
    uvicorn.run("ygoinf.server:app", host="127.0.0.1", port=args.port)
