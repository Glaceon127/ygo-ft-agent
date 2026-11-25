"""只读落盘：把 ygoinf 每次 /predict 的请求与结果写成 gzip JSONL。

不改模型、不改维度、不改返回结构。由环境变量 OBS_LOG 控制开关，
未设置时零开销。任何异常都被吞掉，绝不影响推理。
"""
import gzip
import json
import os
import threading
import time

_PATH = os.environ.get("OBS_LOG", "")
# 完整模式会把 cards_(73 维 x N 张) 全量落盘，44 局就要 GB 级；审计真正需要的是
# 「这一步是哪种消息、有几个候选、模型 top5 是谁、概率多少」。OBS_LOG_COMPACT=1
# 时只落这部分，体积约 1KB/步。
_COMPACT = os.environ.get("OBS_LOG_COMPACT", "") not in ("", "0")
_LOCK = threading.Lock()


def _plain(x):
    if hasattr(x, "model_dump"):
        return _plain(x.model_dump(mode="json"))
    if hasattr(x, "dict"):
        return _plain(x.dict())
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _compact(duel_id, body, predict_results):
    """只保留审计必需的字段：哪类消息、几个候选、top5、模型自己选了什么。"""
    gi = getattr(body.input, "global_", None)
    am = body.input.action_msg
    amd = _plain(am) if am is not None else {}
    if not isinstance(amd, dict):
        amd = {"raw": amd}
    pr = _plain(predict_results) or {}
    preds = pr.get("action_preds") or pr.get("actionPreds") or []
    top = sorted(preds, key=lambda p: -(p.get("prob") or -9))[:5]
    return {
        "ts": round(time.time(), 3),
        "duel_id": duel_id,
        "index": body.index,
        "prev_action_idx": body.prev_action_idx,
        "msg": amd.get("msg_name") or amd.get("msgName") or amd.get("type") or "?",
        "desc": amd.get("desc"),
        "min": amd.get("min"),
        "max": amd.get("max"),
        "n_actions": len(preds),
        "win_rate": pr.get("win_rate"),
        "turn": getattr(gi, "turn", None),
        "phase": getattr(gi, "phase", None),
        "n_chain": len((getattr(gi, "chain_info", None) or {}).get("entries") or []),
        "top": [{"p": round(float(p.get("prob") or 0), 4),
                 "r": p.get("response"),
                 "f": p.get("can_finish")} for p in top],
    }


def log_predict(duel_id, body, predict_results):
    if not _PATH:
        return
    try:
        rec = _compact(duel_id, body, predict_results) if _COMPACT else {
            "ts": time.time(),
            "duel_id": duel_id,
            "index": body.index,
            "prev_action_idx": body.prev_action_idx,
            "input": _plain(body.input),
            "predict_results": _plain(predict_results),
        }
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with _LOCK:
            with gzip.open(_PATH, "at", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass
