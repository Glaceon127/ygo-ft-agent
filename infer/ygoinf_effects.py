"""把 ygoenv 的 compute_effect_tags / compute_effect_features 移植到 Python。

逐条对齐 ygopro.h:1125 / :1163 的双语关键词匹配逻辑。
"""
import sqlite3
from functools import lru_cache
from pathlib import Path
import os

CDB = os.environ.get("YGOINF_CDB", str(Path("../assets/locale/en/cards.cdb").resolve()))


def _hit(d, en, zh):
    # std::string::find("") 恒返回 0，必须跳过空关键词
    if en and en in d:
        return True
    if zh and zh in d:
        return True
    return False


def compute_effect_tags(desc):
    d = desc.lower()
    t = 0
    if (_hit(d, "when this card is destroyed", "这张卡被破坏的场合") or
            _hit(d, "", "这张卡被破坏送去墓地")):
        t |= 1 << 0
    if _hit(d, "destroyed by battle", "被战斗破坏"):
        t |= 1 << 1
    if (_hit(d, "if this card is sent to the gy", "这张卡被送去墓地的场合") or
            _hit(d, "when this card is sent to the gy", "这张卡被送去墓地时") or
            _hit(d, "", "这张卡被送入墓地的场合")):
        t |= 1 << 2
    if _hit(d, "this card battles", "这张卡进行战斗"):
        t |= 1 << 3
    if _hit(d, "piercing", "穿刺"):
        t |= 1 << 4
    if _hit(d, "battle damage", "战斗伤害"):
        t |= 1 << 5
    if (_hit(d, "attacks directly", "直接攻击") or
            _hit(d, "attack directly", "可以直接攻击")):
        t |= 1 << 6
    if (_hit(d, "to your hand", "加入手卡") or
            _hit(d, "", "加入手牌")):
        t |= 1 << 7
    return t


def compute_effect_features(desc):
    d = desc.lower()
    f = 0
    if (_hit(d, "from your deck to your hand", "从卡组把") or
            _hit(d, "from the deck to your hand", "从卡组将") or
            _hit(d, "", "从卡组选") or _hit(d, "", "从卡组把1张")):
        f |= 1 << 0
    if _hit(d, "special summon", "特殊召唤"):
        f |= 1 << 1
    if (_hit(d, "fusion summon", "融合召唤") or
            _hit(d, "synchro summon", "同调召唤") or
            _hit(d, "xyz summon", "超量召唤") or
            _hit(d, "link summon", "连接召唤") or
            _hit(d, "ritual summon", "仪式召唤")):
        f |= 1 << 2
    if (_hit(d, "to the gy", "送去墓地") or
            _hit(d, "to the graveyard", "送入墓地") or
            _hit(d, "", "送去墓地的")):
        f |= 1 << 3
    if (_hit(d, "from your gy", "从墓地") or
            _hit(d, "in your gy", "在墓地") or
            _hit(d, "from your graveyard", "从自己墓地") or
            _hit(d, "in your graveyard", "自己墓地的")):
        f |= 1 << 4
    if (_hit(d, "draw 1 card", "抽1张") or
            _hit(d, "draw 2 cards", "抽2张") or
            _hit(d, "draw 3 cards", "抽3张") or
            _hit(d, "", "抽卡")):
        f |= 1 << 5
    if _hit(d, "destroy", "破坏"):
        f |= 1 << 6
    if _hit(d, "banish", "除外"):
        f |= 1 << 7
    if _hit(d, "negate", "无效"):
        f |= 1 << 8
    if ((_hit(d, "gains", "上升") or _hit(d, "loses", "下降") or
         _hit(d, "becomes", "变成")) and
            (_hit(d, "atk", "攻击力") or _hit(d, "def", "守备力"))):
        f |= 1 << 9
    if (_hit(d, "battle position", "表示形式") or
            _hit(d, "control of", "控制权")):
        f |= 1 << 10
    if (':' in d or ';' in d or '：' in d or '；' in d):
        f |= 1 << 11
    if (';' in d or '；' in d):
        f |= 1 << 12
    if (_hit(d, "target 1", "以") or
            _hit(d, "target 2", "为对象") or
            _hit(d, "target 3", "作为对象") or
            _hit(d, "targets 1", "取为对象")):
        f |= 1 << 13
    if (_hit(d, "quick effect", "对方回合") or
            _hit(d, "during either player's turn", "双方回合") or
            _hit(d, "during your opponent's turn", "对手回合")):
        f |= 1 << 14
    if (_hit(d, "once per turn", "1回合1次") or
            _hit(d, "", "一回合一次")):
        f |= 1 << 15
    return f


_con = None


def _get_con():
    global _con
    if _con is None:
        _con = sqlite3.connect(CDB, check_same_thread=False)
    return _con


@lru_cache(maxsize=20000)
def chain_effects(code):
    row = _get_con().execute("SELECT desc FROM texts WHERE id=?", (code,)).fetchone()
    return compute_effect_features(row[0] or '') if row else 0


@lru_cache(maxsize=20000)
def card_effects(code):
    """返回 cdb datas.category 的完整 32 位整数。

    对齐 ygopro.h::_set_obs_card_ 的 41..72 列：
        41 + j = (category >> j) & 1，j = 0..31

    category 是官方效果分类位掩码，不再从 desc 关键词推断。
    """
    if not code:
        return 0
    try:
        row = _get_con().execute(
            "SELECT category FROM datas WHERE id=?", (code,)).fetchone()
    except Exception:
        return 0
    if not row:
        return 0
    return int(row[0] or 0)
