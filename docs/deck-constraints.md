# 卡组约束

替换 `assets/*.ydk` 前请对照本文。**违反硬性约束会直接抛异常或读成 0 张卡**。

约束依据：`env/ygopro.h` 的 `read_decks`（L535-585）与 `sort_extra_deck`（L1440）。

## 当前卡号版本

| 资源 | 条目 | 最大卡号 | 最后更新 |
|---|---|---|---|
| `scripts/code_list.txt` | **15019** | 99995595 | 2026-11-16 |
| `assets/locale/en/cards.cdb` | **15019** | 99995595 | 2026-11-18 |
| `scripts/script/*.lua` | **13575** | 99995595 | — |

**三者条目数与最大卡号一致**（`9999xxxx` 段是占位/特殊卡，真实卡号最高在 `9xxxxxxxx`）。

对应 `--num-embeddings = 15019 + 30 = 15049`。

```bash
# 自检：三者是否同步
wc -l < scripts/code_list.txt                     # 15019
sqlite3 assets/locale/en/cards.cdb "SELECT COUNT(*) FROM texts;"   # 15019
ls scripts/script/*.lua | wc -l                   # 13575（部分卡无脚本）
```

## 一、文件格式

### 必须使用 LF 换行（硬性）

```cpp
// read_decks: 逐字符判断是否为数字
if (std::all_of(line.begin(), line.end(), ::isdigit)) {
    main_deck.push_back(std::stoul(line));
}
```

**CRLF 行尾的 `\r` 不是数字** → 整行被静默跳过 → 主卡组读到 0 张 → 触发下面的 `< 40` 异常。

Windows 上编辑的 `.ydk` 默认是 CRLF，务必转换：

```bash
tr -d '\r' < input.ydk > output.ydk
# 或
dos2unix input.ydk
```

### 分段标记

```
#created by ...     ← 可选，非数字行会被跳过
#main               ← 主卡组开始（标记行本身被跳过）
40041559
19144622
...
#extra              ← 额外卡组开始
37260677
...
!side               ← side 开始（本环境不使用）
```

**解析规则**（`line.find()` 子串匹配，不要求精确等于 `#extra`）：

- 遇到含 `"extra"` 的行 → 主卡组结束
- 遇到含 `"side"` 的行 → 额外卡组结束
- 其余纯数字行按当前段归属

## 二、数量约束

| 约束 | 值 | 是否硬性 | 依据 |
|---|---|---|---|
| 主卡组下限 | **≥ 40** | ✅ **抛异常** | `ygopro.h:557` |
| 主卡组上限 | 60 | ❌ 代码不查（游戏规则） | — |
| 额外卡组上限 | 15 | ❌ 代码不查（游戏规则） | — |
| 额外卡组卡种 | 只能是融合 / 同调 / XYZ / 连接 | ✅ **抛 `Not extra deck card`** | `sort_extra_deck` |
| 主 + 额外合计 | **≤ 80** | ⚠️ 观测槽位上限 | `max_cards = 80` |

**观测槽位说明**：

```cpp
"obs:cards_".Bind(Spec<uint8_t>({conf["max_cards"] * 2, 73}))   // (160, 73)
```

`max_cards * 2 = 160`，主卡组和额外卡组各占 80 个槽位。**主 + 额外超过 80 会溢出/截断**。

## 三、卡号约束（硬性）

每个卡号必须**同时**存在于三处，缺任一个都会抛异常：

| 资源 | 用途 | 缺失后果 |
|---|---|---|
| `scripts/code_list.txt` | CardId 分配、embedding 索引 | `c_get_card_id` 抛异常 |
| `assets/locale/en/cards.cdb` | 卡的属性/效果分类 | `db_query_card` 抛异常 |
| `scripts/script/c<code>.lua` | 效果实现 | `script_reader_callback` 抛异常 |

**因此要使用新卡，需同步升级这三项资源**（`code_list.txt` 末尾追加、`cards.cdb` 补行、`c<code>.lua` 补脚本）。

## 四、`code_list.txt` 的顺序不可变（硬性）

```
行号 = CardId = 卡号 embedding 的索引（1 起始，0 保留给隐藏卡）
```

- **只能在末尾追加**新卡
- **绝不能重排或插入**已有卡
- 重排会让所有旧 checkpoint 的卡嵌入语义**整体错位**（模型认不出卡）

`--num-embeddings` 必须等于：

```
code_list.txt 行数 + 30        （本仓库 15019 + 30 = 15049）
```

## 五、验证

替换卡组后，至少跑一遍：

```bash
# 静态检查：行尾、张数、卡号覆盖
python scripts/validate_deck.py <卡组文件>
```

或手动核对：

```bash
# 1) 行尾
file deck.ydk | grep -q CRLF && echo "需要转 LF"

# 2) 张数
sed -n '/#main/,/#extra/p' deck.ydk | grep -cE '^[0-9]+$'    # 应为 40-60
sed -n '/#extra/,/!side/p' deck.ydk | grep -cE '^[0-9]+$'    # 应为 0-15

# 3) 卡号覆盖（在 WSL 内，需 cdb）
python - <<'PY'
import sqlite3
DECK, CL = "deck.ydk", "scripts/code_list.txt"
CDB = "assets/locale/en/cards.cdb"
cl = {int(l.split()[0]) for l in open(CL) if l.split() and l.split()[0].isdigit()}
codes = {int(s) for s in open(DECK) if s.strip().isdigit()}
con = sqlite3.connect(CDB)
print("不在 code_list:", sorted(codes - cl))
print("不在 cdb:", [c for c in codes
                  if not con.execute("SELECT 1 FROM texts WHERE id=?", (c,)).fetchone()])
PY
```

## 六、常见错误

| 现象 | 原因 | 修法 |
|---|---|---|
| `Main deck must contain at least 40 cards, found: 0` | **CRLF 行尾** | 转 LF |
| `Not extra deck card` | 额外卡组里放了主卡组才有的卡 | 移出 |
| `c_get_card_id` 相关异常 | 卡号不在 `code_list.txt` | 补进 code_list 末尾 |
| `db_query_card` 相关异常 | 卡号不在 `cards.cdb` | 升级 cdb |
| 模型对某张卡行为异常 | `code_list.txt` 被重排过 | 恢复原顺序 |
