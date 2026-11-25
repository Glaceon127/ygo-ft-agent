# 牌表与卡号约束

本仓库的联合训练读取覆盖后的 `$UP/assets/joint_decks/`。单个 `.ydk` 文件也可传给普通评估入口。传目录时，初始化会依次读取该目录下所有 `*.ydk`，以文件名去掉扩展名作为牌表名。[资源安装](../scripts/apply_overlay.sh) · [牌表注册](../rl/ygoai_utils.py)

## 牌表格式

- 在 Linux 或 WSL 使用 LF 换行。每张卡单独写一行十进制卡号。读入器只接受整行数字，行尾空格或 CR 会使该行被跳过。空行会进入 `stoul` 并抛异常。仓库的 Git 属性已为 `.ydk` 指定 LF。
- 用 `#main`、`#extra`、`!side` 分段。读入从主卡组开始，遇到包含小写 `extra` 或 `side` 的行就切换，因此注释里也要避开这两个词。当前环境注册主卡组和额外卡组，副卡组不参加对局。
- 主卡组少于 40 张会报错。读入器没有检查主卡组 60 张、额外卡组 15 张的上限或禁限表。额外卡组排序时会检查卡片类型，只有带融合、同调、超量或连接类型位的卡能通过。
- 默认卡片观测有 `2 × 80 = 160` 行。编码按双方及区域逐行写入，主卡组和额外卡组共用观测容量。读入器没有“主卡组加额外卡组不得超过 80 张”的规则。[引擎读入与观测](../env/ygopro.h) · [Git 换行设置](../.gitattributes)

## 卡号、数据库和脚本

[卡号表](../assets/code_list.txt) 有 **15,019** 个不同卡号。每行包含卡号和脚本标记，其中 **13,573** 行标记为 1，**1,446** 行标记为 0。卡号表由覆盖脚本安装到 `$UP/scripts/code_list.txt`。英文卡库位于 `$UP/assets/locale/en/cards.cdb`，卡片脚本位于 `$UP/scripts/script/`，后两项由运行环境提供。

| 资源 | 引擎的实际行为 |
|---|---|
| 卡号表与模型 | 从 1 起的物理行号用作内部 ID，未知卡号查询返回 0。续用旧权重时保留原有行号，新卡追加到末尾。当前联合训练与推理配置均使用 15,049 个 embedding 配置项；额外的 30 个是本配置的余量。 |
| `cards.cdb` | 初始化遍历整张卡号表，逐卡查询 `datas` 和 `texts`。例如，`datas` 缺行会报 `[db_query_card] Card not found`，即使当前牌表没用到该卡。应检查卡号覆盖，数据库可以包含额外的卡。 |
| Lua | 初始化按第二列标记为 1 的行读取 `./script/c<卡号>.lua`，还会读公共脚本。标记为 0 的卡无需按该行加载专属文件。文件缺失时读取函数返回空指针和长度 0；引擎请求一个未登记的脚本路径时，回调才会抛出 `Script not found`。 |

[引擎初始化](../env/ygopro.h) · [联合训练配置](../scripts/launch/train_joint.sh) · [推理配置](../rl/serve_counters.py)

## 替换牌表后的自检

在已安装覆盖文件并准备好卡库和 Lua 的工作副本运行。以下命令使用实际训练环境的读入器检查整个牌表目录，还会列出缺失的专属脚本。通过这一检查只说明资源可加载，具体效果仍需在对局中验证。

```bash
cd "$UP/scripts"
python - <<'PY'
from pathlib import Path
from ygoai.utils import init_ygopro

rows = [line.split() for line in Path("code_list.txt").read_text().splitlines()]
missing = [code for code, flag in rows
           if int(flag) and not Path(f"script/c{code}.lua").is_file()]
if missing:
    print(f"有 {len(missing)} 个标记为 1 的卡号缺少专属脚本，前十个为 {missing[:10]}")
pool = Path("../assets/joint_decks")
if not any(pool.glob("*.ydk")):
    raise SystemExit(f"牌表目录为空或不存在：{pool}")
known = {code for code, flag in rows}
for deck in sorted(pool.glob("*.ydk")):
    codes = {line for line in deck.read_text().splitlines() if line.isdecimal()}
    if codes - known:
        raise SystemExit(f"{deck.name} 中有未收录的卡号：{sorted(codes - known)}")
_, names = init_ygopro("YGOPro-v1", "english", pool, "code_list.txt",
                       return_deck_names=True)
print("已加载牌表", sorted(names))
PY
```
