# YGO Agent — FairyTail

基于 [SBL1996/YGO-Agent](https://github.com/SBL1996/ygo-agent) 的卡组特化训练分支。用 PPO 自对弈训练「妖精传姬 / 伟魔 / 恩底弥翁」卡组。

> **文档声明**：本 README 与 [OBSERVATION.md](OBSERVATION.md) 由 AI 生成。其中的命令、参数名、列索引均经过实际运行核对，但受验证范围限制仍可能有疏漏——尤其是观测布局细节，请以 `ygopro.h` 与 `features.py` 源码为准。发现问题欢迎提 issue。

- **观测**：73 列卡牌特征 + 33 列全局 + 14 列动作 + 16 列历史动作，训练与推理共用同一套布局
- **训练**：8 环境并行、本地自对弈、JAX/Flax
- **部署**：权重可直接接入 `ygoinf` + `srvpro2` 插件对局

> 仅支持 Linux / WSL2。Windows 原生不可用（C++ 扩展依赖 POSIX）。

---

## 快速开始

### 1. 环境

| 项 | 要求 |
|---|---|
| OS | Ubuntu 22.04（WSL2 可） |
| Python | 3.10+ |
| GPU | 显存 ≥ 8 GB（训练），推理可 CPU |
| 编译 | [xmake](https://xmake.io/) + gcc/g++ |
| 其他 | git、sqlite3 |

```bash
# xmake（若未安装）
curl -fsSL https://xmake.io/shget.text | bash

# 虚拟环境
python3 -m venv ~/venv
source ~/venv/bin/activate
```

### 2. 安装 Python 依赖

```bash
git clone https://github.com/SBL1996/YGO-Agent.git
cd YGO-Agent

pip install -e .

# 训练/推理核心依赖
pip install "jax[cuda12]==0.4.28" "flax==0.8.4" "optax==0.2.2" \
            "gymnasium==1.4.0" "numpy==1.26.4" "distrax==0.1.5"
```

### 3. 覆盖本仓库的环境改动

本仓库的 C++ 改动集中在**单个头文件**，直接覆盖即可：

```bash
cp /path/to/ygo-ft-agent/env/ygopro.h ygoenv/ygoenv/ygopro/ygopro.h
```

同时把 `scripts/`、`ygoai/rl/`、`ygoinf/ygoinf/` 下的文件按仓库结构覆盖。

### 4. 编译 C++ 环境

```bash
xmake build -r ygopro_ygoenv
```

编译产物在 `build/`，首次约 3–5 分钟。

**验证**：

```bash
python -c "import ygoenv; e = ygoenv.make_gymnasium('YGOPro-v1'); print('env ok:', type(e).__name__)"
# 期望输出：env ok: YGOProGymnasiumEnvPool
```

> `ygoenv` 不是标准 gymnasium 环境，必须用 `ygoenv.make_gymnasium(task_id)` 或 `ygoenv.make(task_id, "gymnasium")`，`gymnasium.make()` 找不到它。

### 5. 准备卡组与卡表

```bash
# 卡组
cp assets/deck/FairyTail.ydk <repo>/assets/deck_md/FairyTail.ydk

# 卡号表（embedding 用，缺了会维度不匹配）
cp assets/code_list.txt <repo>/scripts/code_list.txt
```

### 6. 训练

```bash
cd scripts
python -u cleanba.py \
  --exp-name my_run \
  --ckpt-dir checkpoints_my_run \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --learning-rate 0.0002 --seed 42 \
  --shaping \
  --shaping-tail-chain-bonus 1.00 \
  --shaping-mill-tail-bonus 1.00 \
  --shaping-opp-material-bonus 0.80 \
  --local-eval-episodes 8 --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_my_run
```

**耗时参考**（RTX 4060 Laptop）：约 1300 SPS，300 万步 ≈ 45 分钟。

#### 常用参数

| 参数 | 说明 |
|---|---|
| `--total-timesteps` | 本次训练的步数（1 update = 8192 步） |
| `--checkpoint` | 续训起点（`.flax_model`），只有模型参数，不含优化器状态 |
| `--tb-offset` | TensorBoard 的全局步数偏移，续训时填上一轮的终点 |
| `--num-embeddings` | 卡号 embedding 表行数。可省略（默认从环境读取 `n_embed = 1 + embedding_shape[0]`）；显式指定时取 **`code_list.txt` 行数 + 30** |
| `--shaping-*` | 奖励塑形，见 [OBSERVATION.md](OBSERVATION.md#奖励参数) |

> 本仓库的 `code_list.txt` 为 **15019** 行，对应 `--num-embeddings 15049`。

#### 续训

```bash
python -u cleanba.py \
  --checkpoint checkpoints_my_run/xxxx_50M_u600.flax_model \
  --tb-offset 53826656 \
  --total-timesteps 3000000 \
  ...  # 其余参数同上
```

> 注意：`--total-timesteps` 会影响 `--shaping-turn-task-weight` 的退火曲线（分母 = total_timesteps / n_actors），续训会让它重新从 1.0 开始退。学习率**不退火**（`anneal_lr` 默认关闭）。

### 7. 评估

```bash
cd scripts

# 自对弈（同权重）
python -u battle.py \
  --checkpoint1 checkpoints_my_run/xxxx.flax_model \
  --checkpoint2 checkpoints_my_run/xxxx.flax_model \
  --deck ../assets/deck_md/FairyTail.ydk \
  --num-embeddings 15049 --num-episodes 512 --num-envs 32 --seed 1 \
  --no-accurate
```

**输出指标**：

| 指标 | 含义 | 健康区间 |
|---|---|---|
| `win_rate` | 胜率（先手视角） | 自对弈应 ≈ 0.5 |
| `stable_term` | 正常终局比例（未超时/未投降） | > 80% |
| `bad_action` | 惩罚性动作统计 | 合计 < 1.0/局 |
| `tail_used` | 故事尾巴使用次数/局 | > 1.5 |
| `start_used` | Once Upon 使用次数/局 | > 0.5 |

**跨权重对比**（双向 1024 局，消除先手偏差）：

```bash
python -u battle.py --checkpoint1 A.flax_model --checkpoint2 B.flax_model ... 
python -u battle.py --checkpoint1 B.flax_model --checkpoint2 A.flax_model ...
# A 的真实胜率 ≈ (win_rate_AB + (1 - win_rate_BA)) / 2
```

---

## 项目结构

```
ygo-ft-agent/
├── env/ygopro.h            # C++ 环境改动（单头文件）
├── rl/
│   ├── shaping.py          # 奖励塑形
│   ├── agent.py            # JAX 模型（观测编码）
│   ├── cleanba.py          # 训练入口
│   └── battle.py           # 评估
├── infer/
│   ├── features.py         # 推理侧观测编码（必须与训练一致）
│   ├── models.py
│   ├── jax_inf.py
│   ├── ygoinf_effects.py
│   └── input-builder.ts    # srvpro2 插件的数据构造
├── assets/
│   ├── code_list.txt       # 卡号表（15019 行）
│   ├── FairyTail.ydk       # 卡组（旧，无白枪管之前）
│   └── FAIRY_TAIL_Magistus2.ydk  # 卡组（新）
├── docs/
│   └── deck-constraints.md # 卡组约束（换卡组前必读）
├── scripts/
│   ├── launch/             # 训练启动脚本（可追溯每轮配置）
│   └── eval/               # 评估脚本
└── weights/                # 权重（Release 附件，仓库内仅占位）
```

---

## 观测与奖励

四张特征表的逐列布局、编码规则、奖励参数含义，见 **[OBSERVATION.md](OBSERVATION.md)**。

**换卡组前必读 [docs/deck-constraints.md](docs/deck-constraints.md)**：行尾必须是 LF（CRLF 会让主卡组读成 0 张）、主卡组 ≥40、额外 ≤15 且只能是融合/同调/XYZ/连接、卡号必须在 `code_list.txt` / `cards.cdb` / `c*.lua` 三处齐全、`code_list.txt` 行号即 CardId 不可重排。

一条关键约定：

> **数值型特征必须单调可比较**。攻守、LP 等值经 `float_transform`（`{x>>8, x&0xff}`）写入，由网络侧 `bytes_to_bin` 重组为真实值再软分桶。改编码前必须走完 `C++ 写入 → 网络读取` 的完整数据流。

---

## 推理部署

```
权重(.flax_model)
   ↓ ygoinf（features.py 构造 73/33/14 列观测）
HTTP
   ↓ srvpro2 插件 model-player（input-builder.ts）
ocgcore 对局
```

**启动顺序**：

```bash
# 1) 推理服务（FastAPI + uvicorn，默认端口 3000）
#    必须 cd 到 scripts/ —— features.py 用相对路径读 code_list.txt
cd scripts
CHECKPOINT=/path/to/xxx.flax_model \
JAX_PLATFORMS=cuda \
  python -m uvicorn ygoinf.server:app --host 127.0.0.1 --port 3000

# 可选：记录每次决策的观测与动作
#   OBS_LOG=/tmp/obs.jsonl.gz

# 2) srvpro2（另开终端）
cd srvpro2
npm run build && npm start      # start = node dist/index.js
```

**两个容易踩的点**：

- 权重通过**环境变量 `CHECKPOINT`** 传入，不是命令行参数
- 工作目录**必须是 `scripts/`**，否则 `features.py` 找不到 `code_list.txt`

**必须核对**：推理侧 `features.py` 与训练侧 `ygopro.h` 的观测编码**逐字段一致**，否则模型行为会漂移。核对清单见 [OBSERVATION.md](OBSERVATION.md#训练--推理一致性核对)。

---

## 成绩（自对弈 512 局 vs 早期基线 13M）

| 权重 | 训练步数 | 胜率 | `stable_term` | `bad`/局 |
|---|---|---|---|---|
| ft37 | 45M | 61.04% | 81.6% | 1.029 |
| **ft41** | **51M** | **63.28%** | **85.7%** | **0.869** |

（均为同一卡组、同一编码体系下的可比数字）

---

## 致谢

本项目建立在以下开源工作之上：

- [SBL1996/YGO-Agent](https://github.com/SBL1996/YGO-Agent) — PPO 训练框架、`ygoenv` 的 Python/JAX 绑定
- [edo9300/ygopro-core](https://github.com/edo9300/ygopro-core) — 规则引擎（EDOPro 分支，`ygoenv` 编译时所依赖的 ocgcore）
- [Fluorohydride/ygopro-core](https://github.com/Fluorohydride/ygopro-core) — 规则引擎原始实现
- [purerosefallen/srvpro2](https://github.com/purerosefallen/srvpro2) — 对局服务端（推理部署用）

推理侧的独立工具链（**本仓库不包含**，仅作参考）：

- [purerosefallen/koishipro-core.js](https://github.com/purerosefallen/koishipro-core.js) — WASM 版 ocgcore，可用于不依赖 `ygoenv` 的规则验证

## License

跟随上游仓库。
