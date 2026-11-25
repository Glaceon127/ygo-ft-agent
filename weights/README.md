# 示例权重

仓库随附 [u1172.flax_model](u1172.flax_model)。它是八系列打断与 G 送抽联合续训后的参数文件，已由 Git 跟踪，普通克隆即可取得。[评估与权重记录](../docs/u1172-evaluation.json)

| 字节数 | SHA-256 | 对应的全局环境步数 |
|---:|---|---:|
| 77,142,196 | `ae99ee79b3519c9ffcd22f73776f237f3de59102f19cfe402bcd5fea063b32e6` | 78,292,352 |

## 校验与使用

在覆盖文件仓库根目录运行以下 Bash 命令。保留这些变量，在同一终端按主 README 准备工作副本后，即可使用其中的推理或续训命令。覆盖脚本会安装策略入口，示例权重直接从本仓库读取。[工作副本准备及启动命令](../README.md#使用本项目)

```bash
FT="$(pwd)"
CHECKPOINT="$FT/weights/u1172.flax_model"
START_STEP=78292352
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

训练日志里的 `eval_win_rate` 还取决于实际启动配置。仓库的联合启动器使用一个 actor 线程和 `--local-eval-episodes 8`，每次训练期评估为 8 局。固定窗口、模型互战和 WindBot 实战的样本另见随权重保存的评估记录。[联合启动器](../scripts/launch/train_joint.sh) · [评估记录](../docs/u1172-evaluation.json)
