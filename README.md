# YGO Agent · 妖精传姬

> 本 README 由 AI 根据原作者项目备份与本分支源码整理生成。

这个分支训练「妖精传姬」在引擎提供的合法动作中完成展开、选择终端、响应对方连锁，并通过 YGOPro 客户端与 WindBot 对局。[基础牌表](assets/FAIRY_TAIL_Magistus2.ydk)

## 从原作者项目继承了什么

本项目基于 SBL1996 的 YGO-Agent。原作者已实现基于 EnvPool 和 YGOPro 的并行对局环境、合法动作编码、JAX/Flax 循环策略网络、强化学习训练、模型互战以及 ygoinf 推理服务。本仓库按**覆盖文件**组织，需要与上游源码配合使用。下面的对照以原始 `ygo-agent-main` 备份中实际存在的代码为基准。[原作者仓库](https://github.com/sbl1996/ygo-agent) · [MIT 许可证](LICENSE)

| 部分 | 原作者备份 | 本分支的改进 |
|---|---|---|
| 局面与合法动作 | 卡牌、全局、动作、历史动作每行分别为 41、23、12、14 列 | 扩展为 **73、33、14、16 列**。卡牌和动作增加效果语义、目标关系等信息，显式跟踪公开手牌、连锁中的发动者和已无效状态。[环境编码](env/ygopro.h) · [公开手牌](env/public_hand.h) |
| 连锁与战斗决策 | 策略从局面与历史编码选择引擎给出的动作 | 增加 **8 × 13** 连锁表、公开效果槽和每个候选动作的 **24 项**战斗特征。当公开描述能映射到效果槽时，可区分同卡的不同发动效果。攻击决策可直接使用公开攻守差、表示形式及候选目标信息。[策略网络](rl/agent.py) · [战斗特征](env/battle_features.h) |
| 妖精传姬展开 | 通用对局回报及自对弈训练 | 从观测统计讲述者、编写者等终端，并用引擎事件追踪威凯特送墓后终端登场、故事尾巴堆墓后同回合进入魔陷区。训练可分别调节这些奖励，并对自损攻击、错误使用手坑等动作计数。评估区分终端登场、临时怪兽和终端留场。[奖励](rl/shaping.py) · [评估](rl/battle.py) |
| 跨卡组打断 | 上游支持不同牌表与模型互战 | 新增可复现的攻略窗口、锁鸟检索后窗口和场上无效的合法目标预检。模型在混合场面衡量打断发动者与处理其他威胁的价值，混合对局训练另对攻略认可的应答加入辅助损失。[攻略映射](rl/_popular8_counter.py) · [攻略监督](rl/_counter_guide_rule.py) · [联合训练](rl/train_counters.py) |
| G 系手坑 | 对局引擎处理卡片效果与抽牌 | 根据引擎的抽牌原因追踪增殖的 G、山雀、水母和猫。当己方在自己回合让对方因这些效果实际抽牌，每张扣 **0.01**。终端和胜负回报继续参与权衡，未送抽的特殊召唤没有此项扣分。[抽牌归因](env/ygopro.cpp) · [奖励计算](rl/train_counters.py) |
| 客户端对局与核验 | ygoinf 已支持通过 HTTP 预测动作 | 补齐 srvpro2 协议消息到训练观测的映射，包括连锁发动地点、目标、战斗候选和多选菜单。模型玩家插件将预测转换回合法响应，记录重试、空预测、兜底和终局原因，支持导出 YRP 回放。[推理编码](infer/native_features.py) · [模型玩家插件](infer/index.ts) |

**续训与卡号兼容。** 分支保留卡号到 embedding 的固定映射，读入带标记列的卡号表。仓库随附 15,019 条卡号，配套配置使用 `--num-embeddings 15049`。早期权重扩展网络时可以复用形状匹配的参数，新层保留初始化值。续训日志支持全局步数偏移；联合训练按更新编号区分检查点，先写临时文件，再原子替换，末轮也会保存。[卡号表](assets/code_list.txt) · [参数加载](rl/jax_ckpt.py) · [训练循环](rl/cleanba.py)

## 跨卡训练怎样使用攻略

仓库内的打断规则覆盖 **Blitzclique、Rikka、Dracotail、Yummy、Witchcrafter、Invoked、Dark Magician、Branded** 八个系列，另保留相剑回归窗口。八系列打断条目来自 Master Duel Meta 攻略，按系列、发动地点和效果槽匹配。锁鸟使用检索结算后的窗口。给泡影、效遮标注直接打断偏好时，还会核对发动者是否仍可被取为目标。模型可另选同场的持续威胁。[攻略来源](https://www.masterduelmeta.com/combos-and-counters#counters) · [规则来源索引](assets/counter/popular8_sources.json) · [原生引擎复现记录](assets/counter/popular8_extended_windows.json)

学习方始终操作妖精传姬，分别在 P0 和 P1 的并行环境中采样。训练池混合本家镜像、四种 G、相剑及八系列对手。对手使用冻结的起点策略，部分对局引导它打出指定起动卡或 G，以便出现可以学习的决策。指南标签用于训练，模型可以在多个攻略允许的应答之间分配概率。整局应对能力另用专用 WindBot 检验。[训练牌表](assets/joint_decks) · [训练池与损失函数](rl/train_counters.py)

## 怎样评估模型

- **打断窗口**：在同一批合法局面和新起手中，统计模型首选动作命中攻略应答的比例。相剑回归还会让连锁结算，核对目标效果是否实际被阻止。
- **G 系应对**：对照相同起手下对方的实际抽牌数、己方终端留场和最终胜负，观察模型在减少送抽与维持场面之间的取舍。
- **完整对局**：做配对比较时，分别记录种子和模型座位。普通评估入口会分配或切换座位；需要相同发牌的对照时，应分别固定座位运行并核对起手。与专用 WindBot 对战时分别统计先后手胜率，并从日志、回放核对终局原因、推理失败和兜底次数。

## 使用本项目

**准备工作副本。** 在 Linux 或 WSL2 中准备一份上游工作副本、Python 3.10 环境及上游所需的 JAX/Flax、Optax、EnvPool 构建与 ygoinf 依赖。以下命令从本仓库根目录运行，`UP` 指向可覆盖的工作副本。覆盖脚本会替换其中对应的文件。保留原始备份供比对。[上游安装说明](https://github.com/sbl1996/ygo-agent#installation) · [覆盖清单](scripts/apply_overlay.sh)

```bash
FT="$(pwd)"
UP=/path/to/ygo-agent-work
bash "$FT/scripts/apply_overlay.sh" "$UP"
cd "$UP"
python -m pip install -e ./ygoenv -e ./ygoinf -e .
```

本次验证使用 Python 3.10、JAX/jaxlib 0.4.28、Flax 0.8.4、Optax 0.2.2、NumPy 1.26.4、Gymnasium 1.4.0 和 tyro 0.8.14。

**准备引擎与卡片资源。** 本分支使用 YGOMobile 版 ocgcore。将配套的头文件和 `lib/libocgcore_mobile.a` 放在同一个目录，头文件可由 `ygopro-core/ocgapi.h` 等相对路径访问。已有项目环境可直接复用这套 core；从对应的 YGOMobile `Classes/ocgcore` 源码重建时可对照本项目保留的补丁。Lua 的头文件和库须与构建该静态库时的版本一致。另需在工作副本中准备包含这些卡片的 `assets/locale/en/cards.cdb` 和 `scripts/script/` Lua 卡片脚本。上游早期快照中的资源可能缺少新卡。[mobile core 差异](env/mobile-core.patch) · [引擎构建配置](env/fairy.xmake.lua) · [牌表与卡号约束](docs/deck-constraints.md)

```bash
cd "$UP"
export FT_CORE_DIR=/path/to/ocgcore_mobile
export FT_LUA_DIR=/path/to/lua-prefix  # 含 include/lua/ 和 lib/
xmake f -y -F fairy.xmake.lua
xmake build -F fairy.xmake.lua ygopro_ygoenv
cd scripts
python - <<'PY'
import ygoenv
from ygoenv.ygopro import ygopro_ygoenv

print(ygopro_ygoenv.__file__)
required = {"obs:battle_", "info:g_draws", "info:droll_event",
            "info:negation_preview", "info:chain_effect_slots", "info:opening_ecclesia"}
missing = required.difference(ygoenv.make_spec("YGOPro-v1")._state_keys)
if missing:
    raise SystemExit(f"当前引擎缺少联合训练字段：{sorted(missing)}")
PY
```

自检如果报缺少字段，请核对打印的扩展路径、当前 Python 和 `PYTHONPATH`。已安装的旧版也能创建 `YGOPro-v1`；联合训练须加载本分支构建的扩展。

**启动推理服务。** 将 `CHECKPOINT` 设为兼容联合策略网络的模型文件绝对路径。随仓库提供的示例和校验值见[权重说明](weights/README.md)。专用入口会加载连锁、战斗及打断策略所需的参数。[推理入口](rl/serve_counters.py)

```bash
cd "$UP/scripts"
JAX_PLATFORMS=cuda python serve_counters.py "${CHECKPOINT:?请设置模型文件的绝对路径}" --port 3000
```

外部对局还需要配套的 srvpro2 和 WindBot。在 srvpro2 源码的 `plugins/model-player/` 放入本仓库 `infer/` 中的 TypeScript 文件及 `effect_tags.json`、`quick_effects.json`，重新构建插件，并设置 `MODEL_PLAYER_TAKEOVER=1`、`MODEL_PLAYER_NAME=model`、`MODEL_PLAYER_URL=http://127.0.0.1:3000`。由插件接管名为 `model` 的客户端。构建产物的插件目录需包含这两个 JSON 文件。如需导出回放，启动 srvpro2 时再设置 `MODEL_PLAYER_YRP=1`，可用 `MODEL_PLAYER_YRP_DIR` 指定输出目录。Yummy 对手使用 WindBot 参数 `Deck=Neko`，对应的牌表文件是 `AI_Neko.ydk`。[插件入口](infer/index.ts)

**继续训练。** 将 `START_STEP` 设为起点模型对应的全局步数，将 `TRAIN_STEPS` 设为本次计划增加的环境步数。启动器会重新初始化优化器，在终端实时输出日志，并将 TensorBoard 事件、窗口统计及检查点保存到本次运行目录。[训练启动器](scripts/launch/train_joint.sh)

```bash
cd "$UP/scripts"
bash "$FT/scripts/launch/train_joint.sh" "${CHECKPOINT:?请设置模型文件的绝对路径}" \
  "${START_STEP:?请设置起始全局步数}" "${TRAIN_STEPS:?请设置本次训练预算}"
```

从当前示例权重续训时，每个小批次都复习改名融合、素材取舍和效遮发动时机。效遮样本在普通战斗威胁与主要阶段结束窗口中分别抽取「发动」和「保留」。攻略打断仍由对局中的合法窗口触发，G 系抽牌按实际张数扣分。默认冻结起点权重作为对手，也可通过 `OPPONENT_CHECKPOINT` 指定固定对手。[专项损失](rl/_rehearsal.py)

旧阶段的启动参数保存在 [历史训练脚本](scripts/launch)，基础观测表和塑形参数见 [观测说明](OBSERVATION.md)。检查旧实验时请保留对应的权重、牌表、卡号表和启动参数。
