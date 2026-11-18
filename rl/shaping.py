"""结果导向的奖励整形（reward shaping）。

设计原则：**不惩罚「攻击」这个动作本身**，只惩罚净损失。

- LP 相对差：自己掉血比对手多 -> 负；靠效果主动支付 LP 的代价也因此被计入，
  但系数很小，不会压过「成功的功能性攻击」带来的收益。
- 场上怪兽净差：自己少一只、对手没少 -> 负；有效交换（各少一只）约为 0。

只在**同一玩家视角的连续决策**之间计算差值。视角切换（回合交接）时跳过，
避免 me/op 翻转导致符号错乱；对手回合造成的变化本来也不该由本策略负责。

不改 C++、不改终局胜负奖励，只在训练时叠加。评估路径不包这一层。
"""

from __future__ import annotations

import os

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

LOC_MZONE = 3       # location_to_id 中怪兽区的编号
COL_CARD_ID_HI = 0  # cards_ 第 0 列：card id 高字节
COL_CARD_ID_LO = 1  # cards_ 第 1 列：card id 低字节
COL_LOCATION = 2    # cards_ 第 2 列：位置
COL_CONTROLLER = 4  # cards_ 第 4 列：0=自己, 1=对手（相对当前决策方）
COL_VIEW_IS_P0 = 6  # global_ 第 6 列：1 表示当前视角是玩家 0

# 终端怪：妖精传姬的讲述者 / 编写者。它们是展开的终点、应该留场，
# 所以适合做势能。威凯特(27632520) 刻意排除 —— 它注定被当素材消耗，
# 做势能会先 +后 −、净零，还会惩罚正确用法。
TERMINAL_CODES = (4026187, 78021082)

# ---- 展开链里程碑用的 card id（= code_list.txt 行号）----
MS_WICAT_ID = 4219      # 威凯特 27632520
MS_START_ID = 2971      # START 19326613
MS_TAIL_ID = 12451      # TAIL 82119326
MS_TERM_IDS = (638, 11847)   # 讲述者 / 编写者
MILL_START_ID = 2971         # START（19326613）
MILL_TAIL_ID = 12451         # TAIL（82119326）
MILL_FIELD_ID = 8702         # 舞踏会（56725612，场地魔法，墓地无效果）
LOC_SZONE = 4
LOC_GRAVE = 5

# 五个里程碑的默认权重（总和 0.50），由 ms_scale 统一缩放
MS_WEIGHTS = (0.10, 0.08, 0.07, 0.20, 0.05)

# ---- 阶段0 回合交卷 ----
# 终端（真应留场）/ 辉夜（半个终端）/ x[11] 列 / global_ 的 turn 列
COL_X11 = 11
COL_TURN = 4
TT_TERM_IDS = (638, 11847)      # 讲述者 / 编写者
TT_KAGUYA_ID = 13145            # 妖精传姬-辉夜
TT_X11_TEMPORARY = 2            # 简易融合使用中
TT_K_CAP = 2
TT_K_UNIT = 0.10
TT_K_MAX = 0.40      # K 项合计上限 = 主奖励 1.0 的 40%，不盖过主奖励


def _field_milestones(cards: np.ndarray) -> np.ndarray:
    """返回 (5, batch) 的 bool 矩阵，表示本步各里程碑是否「当前成立」。

    只判断当前快照；调用方负责做成 running OR（每局一次）。
    """
    ids = cards[:, :, 0].astype(np.int32) * 256 + cards[:, :, 1].astype(np.int32)
    loc = cards[:, :, 2]
    ctrl = cards[:, :, 4]
    x11 = cards[:, :, COL_X11]
    on_mz = loc == 3
    in_grave = loc == LOC_GRAVE
    in_sz = loc == LOC_SZONE
    me = ctrl == 0          # 相对当前决策方：0 = 我方

    # 终端必须是「我方的、且不是简易融合使用中的临时物」
    # —— 不加 ctrl 过滤时，对手做出终端也会给我方 +0.20。
    term_mask = (ids == MS_TERM_IDS[0]) | (ids == MS_TERM_IDS[1])
    term_on_mz = term_mask & on_mz & me & (x11 != TT_X11_TEMPORARY)

    launch = ((ids == MS_WICAT_ID) & on_mz & me).any(axis=1)
    term = term_on_mz.any(axis=1)
    me_back = (in_sz & me).any(axis=1)
    close = term & me_back

    # START / TAIL 不再按「墓地里有卡」判定 —— 堆墓只是手段，
    # 真正要的是「发动它们的效果」（START 进墓后盖放 / TAIL 在魔陷区融合）。
    # 这两项改由 info:start_used / info:tail_used 驱动（见 step）。
    none = np.zeros_like(launch)
    return np.stack([launch, none, none, term, close], axis=0)


def _build_terminal_lut(code_list_file: str, codes) -> np.ndarray | None:
    """按 code_list.txt 的行号(= card id) 建 bool 查找表。"""
    want = set(int(c) for c in codes)
    ids = []
    total = 0
    with open(code_list_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            if int(s.split()[0]) in want:
                ids.append(total)
    if not ids:
        return None
    lut = np.zeros(total + 1, dtype=bool)
    lut[np.asarray(ids, dtype=np.int64)] = True
    return lut


def _decode_lp(glob: np.ndarray, hi: int, lo: int) -> np.ndarray:
    return glob[..., hi].astype(np.int32) * 256 + glob[..., lo].astype(np.int32)


class RewardShaping(gym.Wrapper):
    """把 LP 差与怪兽净损失作为逐步奖励叠加到原始奖励上。"""

    def __init__(
        self,
        env: Any,
        lp_coef: float = 0.0002,
        mon_coef: float = 0.05,
        ht_summon_penalty: float = 0.0,
        ht_self_penalty: float = 0.0,
        self_attack_penalty: float = 0.0,
        ht_correct_bonus: float = 0.0,
        term_coef: float = 0.0,
        code_list_file: str | None = None,
        ms_scale: float = 0.0,
        start_grave_bonus: float = 0.08,
        tail_grave_bonus: float = 0.07,
        mill_start_bonus: float = 0.0,
        mill_tail_bonus: float = 0.0,
        mill_field_penalty: float = 0.0,
        wicat_convert_bonus: float = 0.0,
        opp_material_bonus: float = 0.0,
        tail_chain_bonus: float = 0.0,
        terminal_scale: float = 1.0,
        negated_activate_penalty: float = 0.0,
        turn_task_weight: float = 0.0,
        total_timesteps: int = 0,
    ) -> None:
        super().__init__(env)
        self.lp_coef = float(lp_coef)
        self.mon_coef = float(mon_coef)
        # 事件级奖励/惩罚：针对「明确的错误动作」，不是势能。
        # 势能保证最优策略不变，事件惩罚会改变策略 —— 这正是目的。
        self.ht_summon_penalty = float(ht_summon_penalty)
        self.ht_self_penalty = float(ht_self_penalty)
        self.self_attack_penalty = float(self_attack_penalty)
        self.ht_correct_bonus = float(ht_correct_bonus)
        # potential-based 势能：终端怪在场数之差。系数 0 时不读卡表。
        self.term_coef = float(term_coef)
        self._term_lut = None
        self._term_hits = 0
        if self.term_coef and code_list_file:
            try:
                self._term_lut = _build_terminal_lut(code_list_file, TERMINAL_CODES)
                n = int(self._term_lut.sum()) if self._term_lut is not None else 0
                print("[shaping] terminal lut: %d cards, file=%s" % (
                    n, code_list_file), flush=True)
            except OSError as e:
                print("[shaping] WARN terminal lut disabled:", e, flush=True)
                self._term_lut = None
        # 展开链里程碑：每局各触发一次，done 时清零
        self.ms_scale = float(ms_scale)
        # 威凯特当融合素材：C++ 上报 wicat_convert（威凯特送墓后 <=5 步内终端登场）
        self.wicat_convert_bonus = float(wicat_convert_bonus)
        # 用对方场上的怪兽当融合素材（事件级，C++ 上报 opp_fusion_mat）
        self.opp_material_bonus = float(opp_material_bonus)
        # 同回合链路：堆墓选中 TAIL → TAIL 从墓地装备（每局一次）
        self.tail_chain_bonus = float(tail_chain_bonus)
        # 终局 ±1 -> ±terminal_scale（只作用于 done 那一步的原始 reward）
        self.terminal_scale = float(terminal_scale)
        # bad=5：发动已被无效/禁止的卡（默认 0 = 不惩罚）
        self.negated_activate_penalty = float(negated_activate_penalty)
        # 事件级：本步把 START / TAIL 从卡组送进墓地
        self.mill_start_bonus = float(mill_start_bonus)
        self.mill_tail_bonus = float(mill_tail_bonus)
        self.mill_field_penalty = float(mill_field_penalty)
        import os as _o
        self._dbg_on = bool(_o.environ.get("SHAPING_DEBUG"))
        self._dbg_n = 0
        self._dbg_hits = 0
        self._dbg_sum = 0.0
        self._dbg_seen_key = 0
        _mw = np.asarray(MS_WEIGHTS, dtype=np.float32).copy()
        # index 1 = start_grave, index 2 = tail_grave（两项可独立加权）
        _mw[1] = float(start_grave_bonus)
        _mw[2] = float(tail_grave_bonus)
        self._ms_weights = _mw * self.ms_scale
        self._ms = None
        self._ms_tag = None              # (5, n) 每里程碑最后触发的局号
        self._ep_id = None               # (n,) 每 env 当前局号
        self._prev_es = None             # (n,) 上一步的 elapsed_step
        self._ms_episodes = 0            # 实际结束的局数，用于核对触发率
        self._ms_counts = np.zeros(len(MS_WEIGHTS), dtype=np.int64)
        self._bad_counts = {}
        self._prev = None
        # 阶段0：回合交卷
        self.turn_task_weight = float(turn_task_weight)
        self.total_timesteps = int(total_timesteps)
        self._tt_steps = 0
        self._tt_prev_turn = None
        self._tt_prev_cards = None
        self._tt_sf_destroyed = None    # (n,) 本回合自毁次数
        self._tt_sf_consumed = None     # (n,) 本回合正确利用次数
        self._tt_bad = None             # (n,) 本回合 bad 1/2/3 次数
        self._tt_kag_effect = None      # (n,) 本回合辉夜效果发动次数
        self._tt_stats = {"pass": 0, "fail_h2": 0, "neutral": 0,
                          "k_sum": 0.0, "w_sum": 0.0, "w_n": 0,
                          "h1": 0, "k1_sum": 0.0}
        self._tt_print_at = 0

    @property
    def num_envs(self) -> int:
        """gymnasium.Wrapper 不会转发该属性，这里显式暴露给 EnvPreprocess。"""
        return int(getattr(self.env, "num_envs", 1))

    def _state(self, obs: Mapping[str, Any]):
        glob = np.asarray(obs["global_"])
        cards = np.asarray(obs["cards_"])
        view_p0 = glob[:, COL_VIEW_IS_P0] == 1

        me_lp = _decode_lp(glob, 0, 1)
        op_lp = _decode_lp(glob, 2, 3)
        p0_lp = np.where(view_p0, me_lp, op_lp)
        p1_lp = np.where(view_p0, op_lp, me_lp)

        loc = cards[:, :, COL_LOCATION]
        ctrl = cards[:, :, COL_CONTROLLER]
        me_mon = ((loc == LOC_MZONE) & (ctrl == 0)).sum(axis=1).astype(np.int32)
        op_mon = ((loc == LOC_MZONE) & (ctrl == 1)).sum(axis=1).astype(np.int32)
        p0_mon = np.where(view_p0, me_mon, op_mon)
        p1_mon = np.where(view_p0, op_mon, me_mon)

        if self._term_lut is not None:
            ids = (cards[:, :, COL_CARD_ID_HI].astype(np.int32) * 256
                   + cards[:, :, COL_CARD_ID_LO].astype(np.int32))
            np.clip(ids, 0, self._term_lut.size - 1, out=ids)
            is_term = self._term_lut[ids]
            me_term = ((loc == LOC_MZONE) & (ctrl == 0) & is_term).sum(axis=1).astype(np.int32)
            op_term = ((loc == LOC_MZONE) & (ctrl == 1) & is_term).sum(axis=1).astype(np.int32)
            p0_term = np.where(view_p0, me_term, op_term)
            p1_term = np.where(view_p0, op_term, me_term)
        else:
            p0_term = np.zeros(glob.shape[0], dtype=np.int32)
            p1_term = np.zeros(glob.shape[0], dtype=np.int32)
        return p0_lp, p1_lp, p0_mon, p1_mon, view_p0, p0_term, p1_term

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._prev = self._state(result[0])
        n = np.asarray(result[0]["cards_"]).shape[0]
        self._ms = np.zeros((len(MS_WEIGHTS), n), dtype=bool)
        self._ms_tag = np.full((len(MS_WEIGHTS), n), -1, dtype=np.int64)
        self._ep_id = np.zeros(n, dtype=np.int64)
        self._prev_es = np.full(n, -1, dtype=np.int64)
        self._ms_episodes = 0
        g = np.asarray(result[0]["global_"])
        self._tt_prev_turn = g[:, COL_TURN].astype(np.int32).copy()
        self._tt_prev_cards = np.asarray(result[0]["cards_"]).copy()
        self._tt_sf_destroyed = np.zeros(n, dtype=np.int32)
        self._tt_sf_consumed = np.zeros(n, dtype=np.int32)
        self._tt_bad = np.zeros(n, dtype=np.int32)
        self._tt_kag_effect = np.zeros(n, dtype=np.int32)
        return result

    def step(self, actions):
        result = self.env.step(actions)
        obs = result[0]
        state = self._state(obs)
        prev, self._prev = self._prev, state

        rewards = np.asarray(result[1], dtype=np.float32)
        if self.terminal_scale != 1.0:
            _done = np.logical_or(result[2], result[3])
            rewards = rewards * np.where(
                _done, self.terminal_scale, 1.0).astype(np.float32)
        turn_delta = self._turn_task_accumulate(result)
        if prev is not None:
            # p0 绝对视角下的净变化
            delta_lp = (state[0] - prev[0]) - (state[1] - prev[1])
            delta_mon = (state[2] - prev[2]) - (state[3] - prev[3])
            delta_term = (state[5] - prev[5]) - (state[6] - prev[6])
            # 归因到「行动者」：prev[4] 表示行动前的决策方是否为 p0。
            # 不能因为 to_play 切换就跳过 —— 自残攻击几乎总伴随视角切换，
            # 跳过会把负反馈整段丢掉，这正是之前学不会自残的原因。
            sign = np.where(prev[4], 1.0, -1.0).astype(np.float32)
            shaped = sign * (
                self.lp_coef * delta_lp
                + self.mon_coef * delta_mon
                + self.term_coef * delta_term
            )
            done = np.logical_or(result[2], result[3])
            shaped = np.where(done, 0.0, shaped).astype(np.float32)
            rewards = rewards + shaped

        # ---- 展开链里程碑（每局各一次）----
        # 去重靠局号 tag：_ms_tag[m,i] 记录里程碑 m 在 env i 上最后触发的局号，
        # 与 done 检测解耦（done 漏报只会更严格，不会重复加分）。
        if self.ms_scale and self._ms is not None:
            cards = np.asarray(obs["cards_"])
            cur = _field_milestones(cards)
            if cur.shape[1] == self._ms.shape[1]:
                # 新局检测：elapsed_step 归零/回退。一步到位把该 env 的标志清空，
                # 不再依赖 terminated（长对局下会漏报）。
                _infos_ms = result[4] if len(result) > 4 else None
                if isinstance(_infos_ms, dict) and "elapsed_step" in _infos_ms:
                    es = np.asarray(_infos_ms["elapsed_step"], dtype=np.int64).reshape(-1)
                    if es.shape[0] == cur.shape[1] and self._prev_es is not None:
                        # 只在「上一步不是 1、这一步是 1」时判新局：
                        # 最严格，避免 es 卡住或抖动造成重复清零。
                        new_ep = (es == 1) & (self._prev_es != 1)
                        if np.any(new_ep):
                            self._ms[:, new_ep] = False
                            self._ep_id = self._ep_id + new_ep.astype(np.int64)
                            self._ms_episodes += int(new_ep.sum())
                        self._prev_es = es
                fresh = cur & ~self._ms
                if np.any(fresh):
                    bonus = (fresh * self._ms_weights[:, None]).sum(axis=0)
                    rewards = rewards + bonus.astype(np.float32)
                    self._ms_counts += fresh.sum(axis=1).astype(np.int64)
                    self._ms |= cur

        # ---- START / TAIL 效果发动（里程碑 2/3，每局一次）----
        # 由 C++ 的 info:start_used / info:tail_used 驱动（只统计我方）。
        if self.ms_scale and self._ms is not None:
            _inf = result[4] if len(result) > 4 else None
            if isinstance(_inf, dict):
                for ms_i, key in ((1, "start_used"), (2, "tail_used")):
                    v = _inf.get(key)
                    if v is None:
                        continue
                    v = np.asarray(v, dtype=np.int32).reshape(-1)
                    if v.size != self._ms.shape[1]:
                        continue
                    hit = v > 0
                    fresh = hit & ~self._ms[ms_i]
                    if np.any(fresh):
                        rewards = rewards + (fresh.astype(np.float32)
                                             * self._ms_weights[ms_i])
                        self._ms_counts[ms_i] += int(fresh.sum())
                        self._ms[ms_i] |= hit

        # ---- 事件级奖励/惩罚（读 info:bad_action）----
        # 0=无 1=手坑通召 2=手坑自伤 3=自残攻击 4=手坑正确连锁对手
        infos = result[4] if len(result) > 4 else None
        bad = infos.get("bad_action") if isinstance(infos, dict) else None
        if bad is not None:
            bad = np.asarray(bad, dtype=np.int32).reshape(-1)
            n = rewards.shape[0]
            if bad.size == 1 and n > 1:
                bad = np.full(n, int(bad[0]), dtype=np.int32)
            if bad.size == n:
                event = np.zeros(n, dtype=np.float32)
                if self.ht_summon_penalty:
                    event += np.where(bad == 1, -self.ht_summon_penalty, 0.0)
                if self.ht_self_penalty:
                    event += np.where(bad == 2, -self.ht_self_penalty, 0.0)
                if self.self_attack_penalty:
                    event += np.where(bad == 3, -self.self_attack_penalty, 0.0)
                if self.ht_correct_bonus:
                    event += np.where(bad == 4, self.ht_correct_bonus, 0.0)
                if self.negated_activate_penalty:
                    event += np.where(
                        bad == 5, -self.negated_activate_penalty, 0.0)
                rewards = rewards + event.astype(np.float32)
                for code in (1, 2, 3, 4):
                    k = int((bad == code).sum())
                    if k:
                        self._bad_counts[code] = self._bad_counts.get(code, 0) + k

        # 威凯特当融合素材（事件级，独立于 bad_action）
        if self.wicat_convert_bonus and isinstance(infos, dict):
            _wic = infos.get("wicat_convert")
            if _wic is not None:
                _wic = np.asarray(_wic, dtype=np.int32).reshape(-1)
                n_r = rewards.shape[0]
                if _wic.size == 1 and n_r > 1:
                    _wic = np.full(n_r, int(_wic[0]), dtype=np.int32)
                if _wic.size == n_r:
                    rewards = rewards + (_wic.astype(np.float32)
                                         * self.wicat_convert_bonus)

        # 同回合链路：堆墓选中 TAIL → TAIL 从墓地装备（事件级，每局一次）
        if self.tail_chain_bonus and isinstance(infos, dict):
            _tc = infos.get("tail_chain")
            if _tc is not None:
                _tc = np.asarray(_tc, dtype=np.int32).reshape(-1)
                n_r3 = rewards.shape[0]
                if _tc.size == 1 and n_r3 > 1:
                    _tc = np.full(n_r3, int(_tc[0]), dtype=np.int32)
                if _tc.size == n_r3:
                    rewards = rewards + (_tc.astype(np.float32)
                                         * self.tail_chain_bonus)

        # 用对方场上的怪兽当融合素材（事件级，跨多步，GAE 可捕捉）
        if self.opp_material_bonus and isinstance(infos, dict):
            _ofm = infos.get("opp_fusion_mat")
            if _ofm is not None:
                _ofm = np.asarray(_ofm, dtype=np.int32).reshape(-1)
                n_r2 = rewards.shape[0]
                if _ofm.size == 1 and n_r2 > 1:
                    _ofm = np.full(n_r2, int(_ofm[0]), dtype=np.int32)
                if _ofm.size == n_r2:
                    rewards = rewards + (_ofm.astype(np.float32)
                                         * self.opp_material_bonus)

        # 堆墓事件：本步把 START / TAIL 从卡组送进墓地（info:mill_id_a/_b）
        if ((self.mill_start_bonus or self.mill_tail_bonus
             or self.mill_field_penalty) and isinstance(infos, dict)):
            if self._dbg_on:
                self._dbg_seen_key += 0 if "mill_id_a" in infos else 1
                self._dbg_n += 1
            for _k in ("mill_id_a", "mill_id_b"):
                _v = infos.get(_k)
                if _v is None:
                    continue
                _arr = np.asarray(_v).reshape(-1)
                if _arr.size == 0:
                    continue
                if _arr.size == 1 and rewards.shape[0] > 1:
                    _arr = np.full(rewards.shape[0], int(_arr[0]), dtype=_arr.dtype)
                if self.mill_start_bonus:
                    _hit = np.where(_arr == MILL_START_ID,
                                    self.mill_start_bonus, 0.0).astype(np.float32)
                    rewards = rewards + _hit
                    if self._dbg_on:
                        self._dbg_hits += int((_hit > 0).sum())
                        self._dbg_sum += float(_hit.sum())
                if self.mill_field_penalty:
                    rewards = rewards - np.where(
                        _arr == MILL_FIELD_ID,
                        self.mill_field_penalty, 0.0).astype(np.float32)
                if self.mill_tail_bonus:
                    _hit2 = np.where(_arr == MILL_TAIL_ID,
                                     self.mill_tail_bonus, 0.0).astype(np.float32)
                    rewards = rewards + _hit2
                    if self._dbg_on:
                        self._dbg_hits += int((_hit2 > 0).sum())
                        self._dbg_sum += float(_hit2.sum())
            if self._dbg_on and self._dbg_n % 500 == 0:
                print(f"[mill_dbg] steps={self._dbg_n} 缺mill_id={self._dbg_seen_key} "
                      f"命中={self._dbg_hits} 累计奖励={self._dbg_sum:.3f}",
                      flush=True)

        rewards = rewards + turn_delta
        return (obs, rewards) + tuple(result[2:])

    # ---------------- 阶段0 回合交卷 ----------------
    def _anneal_weight(self) -> float:
        """1.0 (前40%) -> 0.5 (40~80%) -> 0 (最后20%)。"""
        if not self.turn_task_weight:
            return 0.0
        if self.total_timesteps <= 0:
            return self.turn_task_weight
        p = min(1.0, self._tt_steps / float(self.total_timesteps))
        if p < 0.4:
            f = 1.0
        elif p < 0.8:
            f = 1.0 - 0.5 * (p - 0.4) / 0.4
        else:
            f = 0.5 * (1.0 - (p - 0.8) / 0.2)
        return self.turn_task_weight * max(0.0, f)

    def _turn_task_accumulate(self, result) -> np.ndarray:
        """累计本回合信号；在回合边界发一次性交卷奖励。"""
        obs = result[0]
        infos = result[4] if len(result) > 4 else None
        glob = np.asarray(obs["global_"])
        cards = np.asarray(obs["cards_"])
        n = glob.shape[0]
        out = np.zeros(n, dtype=np.float32)
        if self._tt_prev_turn is None or not self.turn_task_weight:
            self._tt_prev_turn = glob[:, COL_TURN].astype(np.int32).copy()
            self._tt_prev_cards = cards.copy()
            return out

        self._tt_steps += n
        w = self._anneal_weight()
        self._tt_stats["w_sum"] += w
        self._tt_stats["w_n"] += 1

        def iv(key):
            if not isinstance(infos, dict) or key not in infos:
                return np.zeros(n, dtype=np.int32)
            return np.asarray(infos[key], dtype=np.int32).reshape(-1)[:n]

        sf_d = iv("sf_destroyed")
        sf_c = iv("sf_consumed")
        kag = iv("kaguya_effect")
        bad = iv("bad_action")
        self._tt_sf_destroyed += sf_d
        self._tt_sf_consumed += sf_c
        self._tt_kag_effect += kag
        self._tt_bad += np.isin(bad, (1, 2, 3, 5)).astype(np.int32)

        turn = glob[:, COL_TURN].astype(np.int32)
        es = iv("elapsed_step")
        # 严格新局判定（沿用里程碑里验证过的口径）：es==1 且上一步不是 1
        if self._prev_es is not None:
            new_ep = (es == 1) & (self._prev_es != 1)
        else:
            new_ep = (es == 1)
        # 只认 turn 变化；new_ep 不算边界（新局第一步的 obs 已换局，
        # 在此交卷会把上一局的赏罚记到新局首个动作上）。
        # turn 列是 min(turn_count_, 16)，长局饱和后不再产生边界，
        # 所以 new_ep 时必须无条件清空累计量，避免带入下一局误判。
        if np.any(new_ep):
            for key in ("_tt_sf_destroyed", "_tt_sf_consumed",
                        "_tt_bad", "_tt_kag_effect"):
                getattr(self, key)[new_ep] = 0
        boundary = (turn != self._tt_prev_turn) & (~new_ep)

        if np.any(boundary) and self._tt_prev_cards is not None:
            pc = self._tt_prev_cards
            ids = pc[:, :, COL_CARD_ID_HI].astype(np.int32) * 256 + \
                  pc[:, :, COL_CARD_ID_LO].astype(np.int32)
            loc = pc[:, :, COL_LOCATION]
            ctrl = pc[:, :, COL_CONTROLLER]
            x11 = pc[:, :, COL_X11]
            on_mz = loc == LOC_MZONE
            mine = ctrl == 0        # 上一帧的「自己」= 刚结束回合的玩家
            is_term = (ids == TT_TERM_IDS[0]) | (ids == TT_TERM_IDS[1])
            h1 = (on_mz & mine & is_term & (x11 != TT_X11_TEMPORARY)).any(axis=1)
            k1 = (on_mz & mine & (ids == TT_KAGUYA_ID)).sum(axis=1)
            h2 = self._tt_sf_destroyed == 0
            h3 = self._tt_bad == 0
            k2 = self._tt_kag_effect
            # K 项独立计分：不再依赖主奖励是否通过，否则 main==0 的回合
            # （约 80% 的边界）永远拿不到辉夜分，K 等于没有。
            # 上限：两项合计 <= TT_K_MAX(0.4)，即主奖励 1.0 的 40%。
            # H2 失败的回合不给 K，避免「自毁 -1 + 辉夜 +0.4」互相抵消。
            k = TT_K_UNIT * (np.minimum(k1, TT_K_CAP) +
                             np.minimum(k2, TT_K_CAP))
            k = np.minimum(k, TT_K_MAX).astype(np.float32)

            main = np.where(h1 & h2 & h3, 1.0,
                            np.where(~h2, -1.0, 0.0)).astype(np.float32)
            bonus = np.where(main < 0, 0.0, k).astype(np.float32)
            out = np.where(boundary, (main + bonus) * w, 0.0).astype(np.float32)

            b = boundary.astype(bool)
            self._tt_stats["pass"] += int((b & (main > 0)).sum())
            self._tt_stats["fail_h2"] += int((b & (main < 0)).sum())
            self._tt_stats["neutral"] += int((b & (main == 0)).sum())
            self._tt_stats["k_sum"] += float(bonus[b].sum())
            self._tt_stats["h1"] += int((b & h1).sum())
            self._tt_stats["k1_sum"] += float(k1[b].sum())

            for key in ("_tt_sf_destroyed", "_tt_sf_consumed",
                        "_tt_bad", "_tt_kag_effect"):
                arr = getattr(self, key)
                arr[b] = 0
            out = np.where(np.isnan(out), 0.0, out)

        self._tt_prev_turn = turn
        self._tt_prev_cards = cards.copy()
        _every = int(os.environ.get("TT_PRINT_EVERY", "50000"))
        if self._tt_steps - self._tt_print_at >= _every:
            self._tt_print_at = self._tt_steps
            st = self._tt_stats
            tot = st["pass"] + st["fail_h2"] + st["neutral"]
            if tot:
                print("[turn_task] step=%d w=%.2f n=%d pass=%.1f%% "
                      "h2fail=%.1f%% neutral=%.1f%% h1=%.1f%% "
                      "kaguya/boundary=%.2f k=%.2f"
                      % (self._tt_steps, self._anneal_weight(), tot,
                         100.0 * st["pass"] / tot,
                         100.0 * st["fail_h2"] / tot,
                         100.0 * st["neutral"] / tot,
                         100.0 * st["h1"] / tot,
                         st["k1_sum"] / tot, st["k_sum"]),
                      flush=True)
        return out
