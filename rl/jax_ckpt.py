"""宽松 checkpoint 加载：允许 target 比 checkpoint 多出新子树。"""
import os as _os
_STATS = {"copied": 0, "shape_mismatch": 0, "missing": 0, "leaves": 0}
import flax
import flax.core


def _copy_into(dst, src):
    """把 src 中形状一致的叶子值覆盖到 dst；缺失或形状不符的保留 dst 的初始化值。"""
    if isinstance(dst, dict) and isinstance(src, dict):
        for k in list(dst.keys()):
            if k in src:
                dst[k] = _copy_into(dst[k], src[k])
        return dst
    if isinstance(dst, (list, tuple)) and isinstance(src, (list, tuple)) \
            and len(dst) == len(src):
        return type(dst)(_copy_into(d, s) for d, s in zip(dst, src))
    if hasattr(dst, "shape") and hasattr(src, "shape"):
        _STATS["leaves"] += 1
        if tuple(dst.shape) != tuple(src.shape):
            _STATS["shape_mismatch"] += 1
            if _os.environ.get("YGO_CKPT_DEBUG"):
                print("[ckpt] SHAPE MISMATCH dst=%s src=%s" % (tuple(dst.shape), tuple(src.shape)), flush=True)
            return dst
        _STATS["copied"] += 1
        return src
    if isinstance(dst, (int, float)) and isinstance(src, (int, float)):
        return src
    return dst


def load_checkpoint_into(target, path, extra_prefix=None):
    """加载 checkpoint 到 target 参数树。

    与 flax.serialization.from_bytes 的区别：checkpoint 里缺失的键（例如给模型
    新增的 ChainEncoder_0 子树）不会报错，而是保留 target 里的初始化值；
    形状不一致的键同样保留初始化值。
    """
    with open(path, "rb") as f:
        raw = f.read()
    state = flax.serialization.msgpack_restore(raw)

    if "params" in target:
        ckpt_params = state.get("params", state)
        trained = "cross_deck_defense" in ckpt_params.get("FiLMActor_0", {})
        enabled = "cross_deck_defense" in target["params"].get("FiLMActor_0", {})
        if trained != enabled:
            raise ValueError("Cross-deck checkpoint and model flag do not match")
        if ("own_turn_ash_g_bias" in ckpt_params) != ("own_turn_ash_g_bias" in target["params"]):
            raise ValueError("Ash-on-G checkpoint and model flag do not match")
        merged = flax.core.unfreeze(target)
        merged = {k: v for k, v in merged.items()}
        merged["params"] = _copy_into(
            flax.core.unfreeze(target["params"]), ckpt_params)
        if "batch_stats" in target and "batch_stats" in state:
            merged["batch_stats"] = _copy_into(
                flax.core.unfreeze(target["batch_stats"]),
                flax.core.unfreeze(state["batch_stats"]))
        return flax.serialization.from_bytes(
            target, flax.serialization.msgpack_serialize(merged))

    merged = _copy_into(flax.core.unfreeze(target), state)
    return flax.serialization.from_bytes(
        target, flax.serialization.msgpack_serialize(merged))


def stats():
    return dict(_STATS)
