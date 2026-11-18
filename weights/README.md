# 权重

权重文件（`.flax_model`，单个约 75 MB）不存放在仓库里，从 GitHub Release 下载后放入本目录。

## 训练（续训）

```bash
python cleanba.py \
  --checkpoint weights/xxx.flax_model \
  --ckpt-dir checkpoints_xxx \
  --deck assets/FairyTail.ydk \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset <上一轮终点的 global_step> \
  --learning-rate 0.0002 --seed 1 \
  ...
```

`--tb-offset` 填上一轮的终点，续训后 TensorBoard 的横轴才是连续的全局步数。

## 推理

```bash
# cwd 必须是 scripts/ —— features.py 用相对路径读 code_list.txt
cd scripts

CHECKPOINT=../weights/xxx.flax_model \
JAX_PLATFORMS=cuda \
  python -m uvicorn ygoinf.server:app --host 127.0.0.1 --port 3000
```

启动后可访问 `http://127.0.0.1:3000/docs` 确认服务就绪。

## 评估

### 同一个卡组

```bash
python battle.py \
  --checkpoint1 weights/a.flax_model \
  --checkpoint2 weights/b.flax_model \
  --deck assets/FAIRY_TAIL_Magistus2.ydk \
  --num-episodes 1024 --num-envs 32
```

### 两个不同卡组

**关键两点**：

1. `--deck` 指向**目录**，`--deck1` / `--deck2` 传**卡组名（不含 `.ydk`）**，不是路径
2. 目录用 **`assets/deck_md`**（训练/评估共用，文件稳定）

```bash
python battle.py \
  --deck        assets/deck_md \
  --deck1       FAIRY_TAIL_Magistus2 \
  --deck2       FairyTail \
  --checkpoint1 weights/a.flax_model \
  --checkpoint2 weights/b.flax_model \
  --agent1-player 0 \
  --num-episodes 1024 --num-envs 32
```

`battle.py` 的参数注释写的是 “the **deck name** for the first player”，传完整路径会让 C++ 层抛 `std::bad_function_call`。

加载机制（`ygoai/utils.py:30-36`）：

```python
if deck_fp.is_dir():
    decks = {f.stem: str(f) for f in deck_fp.glob("*.ydk")}   # 目录下全部卡组注册进 C++
    deck_name = 'random'                                       # 不指定时每局随机抽
else:
    decks = {deck_name: deck}
```

**因此目录里每一份 `.ydk` 都必须可加载** —— 有一份坏的（如 CRLF 行尾导致主卡组 0 张）会让整个评估在初始化时就抛异常。所以 `deck_md` 里已移除 `elfnote_2nd_raw.ydk`。

> **不要用 `assets/deck_eval`**：该目录是后来新建的，实测其中 `.ydk` 会被清成 0 字节，导致 `RuntimeError: Main deck must contain at least 40 cards, found: 0`。原因未定位，用 `deck_md` 可完全规避。

**更省事的方式**：用 `scripts/eval/eval_dual.sh`，它会校验卡组并自动用 `deck_md`：

```bash
bash scripts/eval/eval_dual.sh \
  <ckpt1> FAIRY_TAIL_Magistus2 \
  <ckpt2> FairyTail 1024 1
```

### `--agent1-player` 与 `nan`

| 取值 | 行为 |
|---|---|
| 不传（默认 `-1`） | 每局翻转座位，`--deck1/--deck2` 会跟着换人 |
| `0` / `1` | 固定座位，`--deck1` 恒定由 agent1 使用 |

**固定座位时 Payoff 矩阵有一半是 `nan`**，因为另一侧没有对局数据（`0/0`），属正常现象，不影响整体 `win_rate`。

### 训练期评估的局数

`--local-eval-episodes N` 是**每个 actor 线程**的局数，实际总局数 = `N × num_actor_threads`。

```
--local-eval-episodes 8 --num-actor-threads 8  →  64 局
```

所以训练日志里的 `eval_win_rate` 是 `k/64`，不是 `k/8`。

## 校验

```bash
sha256sum xxx.flax_model
```
