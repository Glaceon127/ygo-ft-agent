# 示例权重

仓库随附原始 v8 全局续训后的 [u196.flax_model](u196.flax_model)。同种子换先后手的 WindBot 对照中，相剑从 v8 的 28/50 到 u196 的 31/50，Yummy 从 31/50 到 33/50。独立战斗威胁窗口中，及时发动无效并点对目标从 105/108 降至 96/108。[评估与权重记录](../docs/u196-evaluation.json)

| 字节数 | SHA-256 | 对应的全局环境步数 |
|---:|---|---:|
| 77,300,358 | `afa514db2fa6a5ac49a7b3e6156be33223621eb754fce7be1cb14a483c7a9f39` | 85,798,272 |

## 校验与使用

在覆盖文件仓库根目录运行以下 Bash 命令。保留这些变量，在同一终端按主 README 准备工作副本后，即可使用其中的推理或续训命令。覆盖脚本会安装策略入口，示例权重直接从本仓库读取。[工作副本准备及启动命令](../README.md#使用本项目)

```bash
FT="$(pwd)"
CHECKPOINT="$FT/weights/u196.flax_model"
START_STEP=85798272
sha256sum "$CHECKPOINT"
```

推理使用 `serve_counters.py`，它启用联合策略及匹配的观测和动作筛选。服务启动后，访问 `http://127.0.0.1:3000/` 应返回 `OK`。续训使用 `train_joint.sh`，将 `START_STEP` 作为日志偏移，并设置本次预算 `TRAIN_STEPS`。[推理入口](../rl/serve_counters.py) · [训练入口](../scripts/launch/train_joint.sh)

## 早期普通策略的评估

以下示例适用于采用 `battle.py` 默认策略结构、embedding 配置为 15,049 的旧权重。将 `CKPT_A` 和 `CKPT_B` 设为两个模型的绝对路径。两个模型使用同一牌表时，`--deck` 可直接传文件。

```bash
cd "$UP/scripts"
python battle.py --checkpoint1 "${CKPT_A:?请设置模型 A 的绝对路径}" \
  --checkpoint2 "${CKPT_B:?请设置模型 B 的绝对路径}" \
  --num-embeddings 15049 --deck ../assets/joint_decks/FAIRY_TAIL_Magistus2.ydk \
  --num-episodes 32 --num-envs 32
```

使用不同卡组时，将 `--deck` 改为牌表目录，并传 `--deck1` 和 `--deck2` 指定 P0、P1 的牌表名（不含扩展名）。`--agent1-player 0` 将模型 A 固定在 P0，使用第一副牌表；取 1 则固定在 P1，使用第二副。默认值 −1 允许换座。目录中的所有牌表都会在初始化时读入。[评估实现](../rl/battle.py) · [牌表读入](../docs/deck-constraints.md)

训练日志里的 `eval_win_rate` 是训练期的 8 局抽查。示例权重另有独立窗口、内战和 WindBot 配对对照。[联合启动器](../scripts/launch/train_joint.sh) · [评估记录](../docs/u196-evaluation.json)
