# 观测与奖励

训练（`ygoenv/ygopro.h`）和推理（`ygoinf/features.py`）必须生成**完全一致**的特征。

---

## 四张特征表

| 表 | 形状 | 内容 |
|---|---|---|
| `obs:cards_` | `(max_cards*2, 73)` | 双方场上/手牌/墓地的卡 |
| `obs:global_` | `(33,)` | 全局状态 |
| `obs:actions_` | `(max_options, 14)` | 当前合法动作候选 |
| `obs:h_actions_` | `(n_history, 16)` | 历史动作（14 + turn + phase） |

`max_cards = 80`（每方），故 `cards_` 实际 `(160, 73)`。

---

## `cards_` 73 列

| 列 | 字段 | 编码 | 类型 |
|---|---|---|---|
| 0-1 | `card_id` | 高低字节 | 分类（embedding） |
| 2 | `location` | 区域枚举 | 分类 |
| 3 | `sequence` | 位置序号 | 分类 |
| 4 | `controller` | 0=我方 1=对方 | 分类 |
| 5 | `position` | 表示形式 | 分类 |
| 6 | `overlay` | 超量素材数 | 分类 |
| 7 | `attribute` | 属性 | 分类 |
| 8 | `race` | 种族 | 分类 |
| 9 | `level` | 等级/阶级/连接值 | 分类（1–13） |
| 10 | `counter` | 指示物 | 分类 |
| 11 | `status` | 三值状态位 | 分类 |
| **12-13** | **`atk`** | **`float_transform` 高低字节** | **数值** |
| **14-15** | **`def`** | **`float_transform` 高低字节** | **数值** |
| 16-40 | `type_ids` | 25 个布尔位（含 `TYPE_LINK`） | 分类 |
| 41-72 | `category` | 32 个效果种类位 | 分类 |

### 关键约定

**① 数值特征用 `float_transform`，由网络侧重组成真实值**

```cpp
// C++ 写入
inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  return {x >> 8, x & 0xff};
}
```

```python
# 网络读取（jax/modules.py）
def bytes_to_bin(x, points, intervals):
    x = x[..., 0] * 256 + x[..., 1]        # 重组
    return jnp.clip((x - points + intervals) / intervals, 0, 1)   # 软分桶
```

**两个字节不能单独解释**，必须 `高字节*256 + 低字节`。

重组后再软分桶（thermometer 编码，JAX 版 `make_bin_params`）：

| 区间 | 桶数 | 步长 |
|---|---|---|
| 0 – 8000 | 24 | **333** |
| 8000 – 12000 | 8 | **500** |

所以攻守在 0–8000 的分辨率是 333；LP（上限 8000）正好落在细分区。超过 12000 的值全部饱和为同一向量。

**② Link 怪的 `def` 列存的是连接箭头掩码**

规则上 Link 怪没有守备力，`defense_` 恒为 0。本项目沿用游戏王 cdb 的约定：Link 怪的 `def` 字段存箭头位掩码。

```cpp
if (link_marker > 0) {
  c.defense_ = link_marker;    // 只有 Link 怪进这里
}
```

箭头位：⬆1 ↗2 ➡4 ↘8 ⬇16 ↙32 ⬅64 ↖128。例如「↙」=32、「↖⬆↗」=131。

推理侧必须做同样的替换，否则 Link 怪的 `def` 会恒为 0：

```ts
// srvpro2/plugins/model-player/input-builder.ts
defense: (q.linkMarker ?? 0) > 0 ? q.linkMarker : (q.defense ?? 0),
```

---

## `global_` 33 列

| 列 | 字段 | 编码 |
|---|---|---|
| 0-1 | 我方 LP | `float_transform` |
| 2-3 | 对方 LP | `float_transform` |
| 4 | `turn` | `min(turn, 16)` |
| 5 | `phase` | 阶段枚举 |
| 6 | `is_first` | 座位号（0 = 先手座） |
| 7 | `is_my_turn` | 0/1 |
| 8-21 | 各区域卡数 | 14 个单列（0–60） |
| 22-31 | 连锁栈 | 10 个单列（0–7） |
| 32 | `leaved` | 占位，不参与编码 |

> `is_first` 实际是**座位号**（`me == 0`），不是猜拳结果。训练时 `duel_options` 固定 P0 先手，因此等价于「是否先手」；但在 srvpro2 里座位与先后攻独立，语义仍是座位号。

---

## `actions_` 14 列

| 列 | 字段 | 编码 |
|---|---|---|
| 0 | `spec_index` | 目标卡在 `specs` 里的下标 |
| 1-2 | `card_id` | 高低字节 |
| **3** | **`msg`** | 消息类型映射；**0 = 无效候选（被 mask）** |
| 4 | `act` | 动作大类（召唤/发动/攻击/结束…） |
| 5 | `finish` | 1 = 提交选择 |
| 6 | `effect` | 0=None 1=default 2-15=卡效果 16+=系统 |
| 7 | `phase` | 阶段 |
| 8 | `position` | 表示形式 |
| 9 | `number` | 数值参数（数量/骰子等） |
| 10 | `place` | 区域 |
| 11 | `attrib` | 属性 |
| 12 | `semantics` | **效果语义**（256 类） |
| 13 | `relation` | **目标在连锁中的位置**：0=不在连锁栈 / 1=在连锁栈且在场上 / 2=在连锁栈但已离场 |

**`a_mask` 靠列 3 判断**：`msg == 0` 的动作在 softmax 前被排除。

`h_actions_` 在 14 列之外再加 `turn` 和 `phase` 两列。

---

## 训练 / 推理一致性核对

改任何一列的编码后，逐个核对：

| # | 字段 | 训练侧 | 推理侧 |
|---|---|---|---|
| 1 | `cards_[12:14]` atk | `float_transform(attack_)` | `float_transform(card.attack)` |
| 2 | `cards_[14:16]` def | `float_transform(defense_)`，Link 怪填 marker | 同上，需读 `q.linkMarker` |
| 3 | `global_[0:4]` LP | `float_transform(lp_)` | `float_transform(g.my_lp)` |
| 4 | `global_[4]` turn | `min(turn, 16)` | `min(max(turn,0), 16)` |
| 5 | `global_[6]` is_first | `me == 0` | `myDuelPos === 0` |
| 6 | `actions_[12]` semantics | 效果语义表 | 同名映射 |
| 7 | `MAX_CARDS` | 80 | 80 |

**快速自检**：跑一局真实对局，打印 Link 怪的 `def` 值。训练/推理一致时应为 **7 / 12 / 45 / 128** 之类，而不是全 0。

---

## 奖励参数

全部通过 `--shaping-*` 传入（下划线参数名换成连字符），值为 0 即关闭。共 19 项：

| 参数 | 日志缩写 | 含义 | 本仓库取值 |
|---|---|---|---|
| `--shaping-terminal-scale` | `tsc` | **终局奖励缩放**（默认 1.0） | 1.0 |
| `--shaping-lp-coef` | `lp` | LP 差 | 0.0002 |
| `--shaping-mon-coef` | `mon` | 场面怪兽数 | 0.02 |
| `--shaping-ms-scale` | `ms` | 里程碑缩放 | 1.0 |
| `--shaping-term-coef` | `term` | 终局达成系数 | 0 |
| `--shaping-turn-task-weight` | `turn` | turn_task 权重（**随训练线性退火到 0**） | 1.0 |
| `--shaping-tail-chain-bonus` | `tch` | 尾巴「装备后融合」同回合完成 | **1.00** |
| `--shaping-mill-tail-bonus` | `millT` | 尾巴从卡组进墓 | **1.00** |
| `--shaping-mill-field-penalty` | `millF` | 场地进墓（浪费） | 1.00 |
| `--shaping-mill-start-bonus` | `millS` | Once Upon 从卡组进墓 | 0.30 |
| `--shaping-opp-material-bonus` | `omm` | 用对方怪兽当融合素材 | **0.80** |
| `--shaping-wicat-convert-bonus` | `wic` | 威凯特把检索转成融合素材 | 0.30 |
| `--shaping-ht-summon-penalty` | `ht_summon` | 手坑通召 | **0.30** |
| `--shaping-ht-self-penalty` | `ht_self` | 手坑打自己 | 2.50 |
| `--shaping-self-attack-penalty` | `self_attack` | 攻击必亏（打不过/打不穿） | **0.80** |
| `--shaping-ht-correct-bonus` | `ht_correct` | 手坑命中关键时点 | 0.20 |
| `--shaping-tail-grave-bonus` | `msT` | 尾巴进墓地 | 0.20 |
| `--shaping-start-grave-bonus` | `msS` | Once Upon 进墓地 | 0 |
| `--shaping-negated-activate-penalty` | `nbad` | 发动被无效 | 0 |

启动时日志会打印一行 `[shaping] ...`，可直接核对是否与预期一致。

### 关于「选卡类」奖励

`mill_field_penalty`（惩罚堆场地）与 `mill_tail_bonus`（奖励堆尾巴）属于 `MSG_SELECT_CARD` 场景的奖励。实测：

> 这类奖励对行为分布的影响很弱。堆墓选择长期接近均匀（场地 26%、辉夜 24%、玛奇尔 15%、尾巴 10%），加惩罚没有明显改变。

可能原因：单次选择的收益（~0.5）相对终局 reward（±4）只占 2.5%，在 128 步 episode 的噪声里难以传导。

---

## 环境可复现性限制

`ygoenv` 写出的 `.yrp` 录像**不含随机种子**（只写 `init_lp / startcount / drawcount / duel_options / 卡表`），因此：

- ✅ 录像可用于**路线复盘**（每步做了什么）
- ❌ 录像**无法精确重放**（洗牌结果不可复现）

需要精确复现某个局面时，用 `resetGame(seed=..., openingCards=[...])` 重建，而不是读录像。
