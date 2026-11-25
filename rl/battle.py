import sys
import time
import os
import random
from collections import Counter
from typing import Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from functools import partial

import ygoenv
import numpy as np

import tyro

import jax
import jax.numpy as jnp
import flax


# ---- 做场统计用的 card id（= code_list.txt 行号）----
# 讲述者 4026187 / 编写者 78021082 / 威凯特 27632520
TERM_IDS = (638, 11847)
WICAT_ID = 4219
LOC_MZONE = 3
COL_ID_HI = 0
COL_ID_LO = 1
COL_LOCATION = 2
COL_X11 = 11
KAGUYA_ID = 13145
X11_TEMPORARY = 2


def _track_field(obs, seen_term, seen_wicat, term_now_max,
                 seen_stable, seen_temp, seen_kaguya):
    """每步更新做场标志；阶段0 另判「真终端/临时物/辉夜」。"""
    cards = np.asarray(obs["cards_"])
    ids = cards[:, :, COL_ID_HI].astype(np.int32) * 256 + cards[:, :, COL_ID_LO].astype(np.int32)
    on_mz = cards[:, :, COL_LOCATION] == LOC_MZONE
    x11 = cards[:, :, COL_X11]
    term_on = on_mz & ((ids == TERM_IDS[0]) | (ids == TERM_IDS[1]))
    wicat_on = on_mz & (ids == WICAT_ID)
    stable = term_on & (x11 != X11_TEMPORARY)
    temp = on_mz & (x11 == X11_TEMPORARY)
    kaguya = on_mz & (ids == KAGUYA_ID)
    n_term = term_on.sum(axis=1)
    seen_term |= n_term > 0
    seen_wicat |= wicat_on.any(axis=1)
    np.maximum(term_now_max, n_term, out=term_now_max)
    seen_stable |= stable.any(axis=1)
    seen_temp |= temp.any(axis=1)
    seen_kaguya |= kaguya.any(axis=1)

def _track_counts(obs, acc):
    """次数口径：帧级指标按绝对玩家累加，并记录峰值。

    cards_ 的 controller 是「相对当前决策方」，用 global_[6]（视角是否 P0）
    折成绝对玩家，避免视角翻转把双方混在一起。
    """
    cards = np.asarray(obs["cards_"])
    glob = np.asarray(obs["global_"])
    ids = cards[:, :, COL_ID_HI].astype(np.int32) * 256 + cards[:, :, COL_ID_LO].astype(np.int32)
    loc = cards[:, :, COL_LOCATION]
    ctrl = cards[:, :, 4]
    x11 = cards[:, :, COL_X11]
    mz = loc == LOC_MZONE
    view_p0 = glob[:, 6] == 1
    p0 = (ctrl == 0) == view_p0[:, None]

    term = (ids == TERM_IDS[0]) | (ids == TERM_IDS[1])
    groups = {
        "term": mz & term & (x11 != X11_TEMPORARY),
        "wic": mz & (ids == WICAT_ID),
        "kag": mz & (ids == KAGUYA_ID),
    }
    for tag, mask in groups.items():
        a = (mask & p0).sum(axis=1)
        b = (mask & ~p0).sum(axis=1)
        acc[tag + "_p0"] += float(a.sum())
        acc[tag + "_p1"] += float(b.sum())
        acc[tag + "_peak"] = max(acc[tag + "_peak"], int(a.max()), int(b.max()))


from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics, EnvPreprocess
from ygoai.rl.env import ChainEffectObservation
from ygoai.rl.jax.agent import RNNAgent, ModelArgs


@dataclass
class Args:
    seed: int = 1
    """the random seed"""

    env_id: str = "YGOPro-v1"
    """the id of the environment"""
    deck: str = "../assets/deck"
    """the deck file to use"""
    deck1: Optional[str] = None
    """the deck name for the first player, for example, `Hero`"""
    deck2: Optional[str] = None
    """the deck name for the second player, for example, `CyberDragon`"""
    agent1_player: int = -1
    """agent1 固定控制的玩家：0=P0，1=P1，-1=一半一半交替（默认）。
    设为 0 或 1 时，每局结束不再翻转，配合 --deck1/--deck2 可让某一方固定卡组。"""
    code_list_file: str = "code_list.txt"
    """the code list file for card embeddings"""
    lang: str = "english"
    """the language to use"""
    max_options: int = 24
    """the maximum number of options"""
    n_history_actions: int = 32
    """the number of history actions to use for the environment"""
    oppo_info: bool = False
    """whether to use opponent information"""
    num_embeddings: Optional[int] = None
    """the number of embeddings of the agent"""
    accurate: bool = True
    """whether to do accurate evaluation. If not, there will be more short games"""
    reverse: bool = False
    """whether to reverse the order of the agents"""
    use_belief1: bool = False
    """agent1 是否加载信念层（BeliefEncoder + belief_ 注入）"""
    use_belief2: bool = False
    """agent2 是否加载信念层；False 时输入全零 belief_，等价于原模型"""
    truncate: float = 0.0
    """the truncation ratio of games"""

    verbose: bool = False
    """whether to print debug information"""
    record: bool = False
    """whether to record the game as YGOPro replays"""

    num_episodes: int = 1024
    """the number of episodes to run""" 
    num_envs: int = 64
    """the number of parallel game environments"""

    m1: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent1"""
    m2: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent2"""

    checkpoint1: str = "checkpoints/agent.pt"
    """the checkpoint to load for the first agent, must be a `flax_model` file"""
    checkpoint2: str = "checkpoints/agent.pt"
    """the checkpoint to load for the second agent, must be a `flax_model` file"""
    
    xla_device: Optional[str] = None
    """the XLA device to use, `cpu` for forcing running on CPU"""

    env_threads: Optional[int] = None
    """the number of threads to use for envpool, defaults to `num_envs`"""

    cards_w2: Optional[int] = None
    """agent2 的 cards_ 宽度。None = 与 env 相同。
    用来让 58 列旧模型（v4）和 73 列新模型在同一局里对打：
    CardEncoder 吃 cards_[:, :, 2:]，宽度不同则 Dense 权重形状不同，
    因此 agent2 要用自己的宽度初始化，并在推理时截断 obs。"""


import ygoai.rl.jax.ckpt as _ckpt_mod


def load_checkpoint_into(target, path, extra_prefix=None):
    """宽松加载：missing/形状不符的键保留初始化值（见 ygoai.rl.jax.ckpt）。"""
    return _ckpt_mod.load_checkpoint_into(target, path)


def create_agent1(args):
    return RNNAgent(
        ash_g_ids=args.m1.ash_g_ids(args.code_list_file),
        **asdict(args.m1),
        embedding_shape=args.num_embeddings,
    )


def create_agent2(args):
    return RNNAgent(
        ash_g_ids=args.m2.ash_g_ids(args.code_list_file),
        **asdict(args.m2),
        embedding_shape=args.num_embeddings,
    )


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

    args = tyro.cli(Args)

    use_belief = args.use_belief1 or args.use_belief2
    if use_belief:
        from yai.belief import DEFAULT_KEY_CARDS, BeliefEngine
        from yai.cards import CardIndex
        from yai.jax_belief import install_belief_fusion
        from yai.obs import BeliefBatchAdapter, BeliefEvalWrapper
        install_belief_fusion()
        print(f"belief fusion installed (dim={len(DEFAULT_KEY_CARDS)})")

    if args.record:
        args.num_envs = 1
        args.verbose = True
        print("Set num_envs=1 and verbose=True for recording")
        if not os.path.exists("replay"):
            os.makedirs("replay")

    args.env_threads = min(args.env_threads or args.num_envs, args.num_envs)

    deck = init_ygopro(args.env_id, args.lang, args.deck, args.code_list_file)

    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    seed = args.seed + 100000
    random.seed(seed)
    seed = random.randint(0, int(1e8))
    random.seed(seed)
    np.random.seed(seed)

    if args.xla_device is not None:
        os.environ.setdefault("JAX_PLATFORMS", args.xla_device)

    if args.accurate:
        if args.num_envs != args.num_episodes:
            args.num_envs = args.num_episodes
            print("Set num_envs to num_episodes for accurate evaluation")

    num_envs = args.num_envs

    env_option = dict(
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=args.env_threads,
        seed=seed,
        player=-1,
        max_options=args.max_options,
        play_mode='self',
        async_reset=False,
        verbose=args.verbose,
        record=args.record,
    )
    chain_effect_features = args.m1.uses_chain_effect_slots or args.m2.uses_chain_effect_slots
    if chain_effect_features:
        env_option["chain_effect_probe"] = True
    envs = ygoenv.make(
        task_id=args.env_id,
        n_history_actions=args.n_history_actions,
        deck1=args.deck1,
        deck2=args.deck2,
        oppo_info=args.oppo_info,
        battle_features=args.m1.battle_features or args.m2.battle_features,
        **env_option,
    )
    envs.num_envs = num_envs
    if chain_effect_features:
        envs = ChainEffectObservation(envs)
    envs = EnvPreprocess(envs, skip_mask=not args.oppo_info)

    key = jax.random.PRNGKey(seed)

    obs_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    if use_belief:
        # 必须放在 RecordEpisodeStatistics 之后：BeliefEvalWrapper 是普通类，
        # 不能作为 gymnasium.Wrapper 的内层 env。
        adapter = BeliefBatchAdapter(
            BeliefEngine(), CardIndex.from_code_list(args.code_list_file))
        envs = BeliefEvalWrapper(envs, adapter)
    sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
    sample_obs["mask_"] = None
    if use_belief:
        sample_obs["belief_"] = jnp.zeros(
            (1, len(DEFAULT_KEY_CARDS)), dtype=jnp.float32)
    agent1 = create_agent1(args)
    rstate1 = agent1.init_rnn_state(1)
    params1 = jax.jit(agent1.init)(key, sample_obs, rstate1)
    params1 = load_checkpoint_into(params1, args.checkpoint1)

    if args.checkpoint1 == args.checkpoint2 and args.m1 == args.m2 and not args.cards_w2:
        agent2 = agent1
        params2 = params1
    else:
        w1 = int(sample_obs["cards_"].shape[-1])
        w2 = int(args.cards_w2) if args.cards_w2 else w1
        if w2 != w1:
            sample_obs2 = dict(sample_obs)
            sample_obs2["cards_"] = jnp.zeros(
                (*sample_obs["cards_"].shape[:-1], w2),
                dtype=sample_obs["cards_"].dtype)
            print(f"[battle] agent1 cards_={w1} 列，agent2 cards_={w2} 列（异构）")
        else:
            sample_obs2 = sample_obs
        agent2 = create_agent2(args)
        rstate2 = agent2.init_rnn_state(1)
        params2 = jax.jit(agent2.init)(key, sample_obs2, rstate2)
        params2 = load_checkpoint_into(params2, args.checkpoint2)
    
    params1 = jax.device_put(params1)
    params2 = jax.device_put(params2)
    
    def _obs_for(obs, want_belief, card_w=None):
        """按 agent 裁剪 obs：cards_ 截到指定宽度；无信念一方把 belief_ 置零。"""
        out = dict(obs)
        if card_w is not None and int(out["cards_"].shape[-1]) != int(card_w):
            out["cards_"] = out["cards_"][..., : int(card_w)]
        if not use_belief or want_belief:
            return out
        out["belief_"] = jnp.zeros_like(obs["belief_"])
        return out

    @partial(jax.jit, static_argnums=(4,))
    def get_probs(params, rstate, obs, done=None, model_id=1):
        if model_id == 1:
            agent = create_agent1(args)
        else:
            agent = create_agent2(args)
        next_rstate, logits = agent.apply(params, obs, rstate)[:2]
        probs = jax.nn.softmax(logits, axis=-1)
        if done is not None:
            next_rstate = jax.tree.map(
                lambda x: jnp.where(done[:, None], 0, x), next_rstate)
        return next_rstate, probs

    if num_envs != 1:
        @jax.jit
        def get_probs2(params1, params2, rstate1, rstate2, obs1, obs2, main, done):
            next_rstate1, probs1 = get_probs(params1, rstate1, obs1, None, 1)
            next_rstate2, probs2 = get_probs(params2, rstate2, obs2, None, 2)
            probs = jnp.where(main[:, None], probs1, probs2)
            rstate1 = jax.tree.map(
                lambda x1, x2: jnp.where(main[:, None], x1, x2), next_rstate1, rstate1)
            rstate2 = jax.tree.map(
                lambda x1, x2: jnp.where(main[:, None], x2, x1), next_rstate2, rstate2)
            rstate1, rstate2 = jax.tree.map(
                lambda x: jnp.where(done[:, None], 0, x), (rstate1, rstate2))
            return rstate1, rstate2, probs

        def predict_fn(rstate1, rstate2, obs, main, done):
            rstate1, rstate2, probs = get_probs2(
                params1, params2, rstate1, rstate2,
                _obs_for(obs, args.use_belief1, None),
                _obs_for(obs, args.use_belief2, args.cards_w2),
                main, done)
            return rstate1, rstate2, np.array(probs)
    else:
        def predict_fn(rstate1, rstate2, obs, main, done):
            if main[0]:
                rstate1, probs = get_probs(
                    params1, rstate1, _obs_for(obs, args.use_belief1, None), done, 1)
            else:
                rstate2, probs = get_probs(
                    params2, rstate2, _obs_for(obs, args.use_belief2, args.cards_w2), done, 2)
            return rstate1, rstate2, np.array(probs)

    obs, infos = envs.reset()
    next_to_play = infos['to_play']

    dones = np.zeros(num_envs, dtype=np.bool_)

    episode_rewards = []
    episode_lengths = []
    win_rates = []
    win_reasons = []
    win_players = []
    win_agents = []

    step = 0
    start = time.time()
    start_step = step

    # 三类错误动作的逐步计数（1=手坑通召 2=手坑自伤 3=自残攻击 4=手坑正确）
    bad_counts = Counter()
    rn = {}
    bad_decisions = 0

    # 做场统计（每局）
    seen_term = np.zeros(num_envs, dtype=np.bool_)
    seen_wicat = np.zeros(num_envs, dtype=np.bool_)
    term_now_max = np.zeros(num_envs, dtype=np.int32)
    seen_stable = np.zeros(num_envs, dtype=np.bool_)
    seen_temp = np.zeros(num_envs, dtype=np.bool_)
    seen_kaguya = np.zeros(num_envs, dtype=np.bool_)
    ep_term = []
    ep_wicat = []
    ep_both = []
    ep_alive = []
    ep_stable = []
    ep_temp = []
    ep_kaguya = []
    ep_has_start = np.zeros(num_envs, dtype=np.bool_)
    ep_has_tail = np.zeros(num_envs, dtype=np.bool_)
    ep_start_flag, ep_tail_flag = [], []
    ep_term_peak, ep_len_vals = [], []
    sf_consumed_total = 0
    sf_destroyed_total = 0
    kaguya_effect_total = 0
    wicat_convert_total = 0
    mill_counter = Counter()
    mill_total = 0
    mc_sel = mc_size = mc_start = mc_tail = 0
    mc_dsel = mc_dsize = mc_one = 0
    mn1 = mn2 = 0          # 堆 1 张 / 2 张的事件数
    mn2_tail = mn2_start = 0
    mn1_tail = mn1_start = 0
    ma_sel = ma_size = ma_start = ma_tail = 0
    hist = Counter()
    ps_cand = ps_pick = pt_cand = pt_pick = 0
    pk_kg = pk_fd = pk_mg = pk_sn = 0
    nz_win = nz_act = 0
    nz_ep_win = []
    nz_ep_act = []
    nz_win_b = {1: 0, 2: 0, 3: 0}
    nz_act_b = {1: 0, 2: 0, 3: 0}
    _id2code = {}
    try:
        for _i, _l in enumerate(open("code_list.txt", encoding="utf-8", errors="replace"), 1):
            _c = _l.strip()
            if _c:
                # code_list.txt 每行是 "code has_script"，只取卡号
                _id2code[_i] = _c.split()[0]
    except OSError:
        pass
    _CARD_NAME = {
        "27632520": "威凯特", "19326613": "START", "82119326": "TAIL",
        "4026187": "讲述者", "78021082": "编写者", "56725612": "舞踏会",
        "19144622": "玛奇莉尔", "86937530": "辉夜",
    }
    start_used_total = 0
    tail_used_total = 0
    cnt = {k: 0.0 for k in ("term_p0", "term_p1", "wic_p0", "wic_p1",
                            "kag_p0", "kag_p1")}
    cnt.update({k: 0 for k in ("term_peak", "wic_peak", "kag_peak")})

    if args.agent1_player >= 0:
        # 固定 agent1 控制的玩家，避免每局翻转导致卡组跟着换。
        main_player = np.full(num_envs, args.agent1_player, dtype=np.int64)
    elif num_envs == 1:
        # --record 会强制 num_envs=1，此时 first_player 为空数组，
        # 原逻辑会让 main_player 恒为 [1]（agent1 永远后手）。
        # 这里改成按 --reverse 决定：默认后手，加 --reverse 则先手。
        main_player = np.array([0 if args.reverse else 1], dtype=np.int64)
    else:
        first_player = np.zeros(num_envs // 2, dtype=np.int64)
        second_player = np.ones(num_envs - num_envs // 2, dtype=np.int64)
        if args.reverse:
            main_player = np.concatenate([second_player, first_player])
        else:
            main_player = np.concatenate([first_player, second_player])
    rstate1 = agent1.init_rnn_state(num_envs)
    rstate2 = agent2.init_rnn_state(num_envs)
    collected = np.zeros((args.num_episodes,), dtype=np.bool_)

    if not args.verbose:
        pbar = tqdm(total=args.num_episodes)

    model_time = env_time = 0
    while True:
        if start_step == 0 and len(episode_lengths) > int(args.num_episodes * 0.1):
            start = time.time()
            start_step = step
            model_time = env_time = 0

        _track_field(obs, seen_term, seen_wicat, term_now_max,
                     seen_stable, seen_temp, seen_kaguya)
        _track_counts(obs, cnt)
        if isinstance(infos, dict):
            for _k in ("rename_win", "rename_hit", "rename_clr",
                       "sel_gyoku", "pick_gyoku", "obs_gyoku",
                       "obs_spec_fill",
                       "gyoku_id", "rn_live",
                       "gyoku_grave0", "gyoku_grave1",
                       "story_summon0", "story_summon1",
                       "tail_oppmat0", "tail_oppmat1",
                       "opp_fusion_mat", "tail_chain",
                       "tail_equip", "start_set",
                       "atk_tgt_total", "atk_tgt_zero",
                       "def_tgt_total", "def_tgt_zero",
                       "atk_seen", "atk_changed", "def_changed"):
                if _k in infos:
                    rn[_k] = rn.get(_k, 0) + int(np.asarray(infos[_k]).sum())
            for _k in ("rename_win", "rename_hit", "rename_clr"):
                if _k in infos:
                    rn[_k] = rn.get(_k, 0) + int(np.asarray(infos[_k]).sum())

            if "sf_consumed" in infos:
                sf_consumed_total += int(np.asarray(infos["sf_consumed"]).sum())
            if "sf_destroyed" in infos:
                sf_destroyed_total += int(np.asarray(infos["sf_destroyed"]).sum())
            if "kaguya_effect" in infos:
                kaguya_effect_total += int(np.asarray(infos["kaguya_effect"]).sum())
            if "wicat_convert" in infos:
                wicat_convert_total += int(np.asarray(infos["wicat_convert"]).sum())
            if "start_used" in infos:
                start_used_total += int(np.asarray(infos["start_used"]).sum())
            if "tail_used" in infos:
                tail_used_total += int(np.asarray(infos["tail_used"]).sum())
            for _k in ("mill_id_a", "mill_id_b"):
                if _k in infos:
                    _arr = np.asarray(infos[_k]).reshape(-1)
                    for _v in _arr:
                        if int(_v) > 0:
                            mill_counter[int(_v)] += 1
                            mill_total += 1
                    if _arr.size == num_envs:
                        ep_has_start |= (_arr == 2971)
                        ep_has_tail |= (_arr == 12451)
            mc_sel += int(np.asarray(infos.get("mill_cand_sel", 0)).sum())
            mc_size += int(np.asarray(infos.get("mill_cand_size", 0)).sum())
            mc_start += int(np.asarray(infos.get("mill_cand_start", 0)).sum())
            mc_tail += int(np.asarray(infos.get("mill_cand_tail", 0)).sum())
            _ia = np.asarray(infos.get("mill_id_a", 0)).reshape(-1)
            _ib = np.asarray(infos.get("mill_id_b", 0)).reshape(-1)
            if _ia.size == _ib.size:
                _one = (_ia > 0) & (_ib == 0)
                _two = (_ia > 0) & (_ib > 0)
                mn1 += int(_one.sum())
                mn2 += int(_two.sum())
                mn1_tail += int((_one & (_ia == 12451)).sum())
                mn1_start += int((_one & (_ia == 2971)).sum())
                mn2_tail += int((_two & ((_ia == 12451) | (_ib == 12451))).sum())
                mn2_start += int((_two & ((_ia == 2971) | (_ib == 2971))).sum())
            mc_dsel += int(np.asarray(infos.get("mill_cand_dsel", 0)).sum())
            mc_dsize += int(np.asarray(infos.get("mill_cand_dsize", 0)).sum())
            mc_one += int(np.asarray(infos.get("mill_cand_one", 0)).sum())
            ma_sel += int(np.asarray(infos.get("mill_any_sel", 0)).sum())
            ma_size += int(np.asarray(infos.get("mill_any_size", 0)).sum())
            ma_start += int(np.asarray(infos.get("mill_any_start", 0)).sum())
            ma_tail += int(np.asarray(infos.get("mill_any_tail", 0)).sum())
            ps_cand += int(np.asarray(infos.get("pair_start_cand", 0)).sum())
            ps_pick += int(np.asarray(infos.get("pair_start_pick", 0)).sum())
            pt_cand += int(np.asarray(infos.get("pair_tail_cand", 0)).sum())
            pt_pick += int(np.asarray(infos.get("pair_tail_pick", 0)).sum())
            pk_kg += int(np.asarray(infos.get("pair_kg_cand", 0)).sum())
            pk_fd += int(np.asarray(infos.get("pair_fd_cand", 0)).sum())
            pk_mg += int(np.asarray(infos.get("pair_mg_cand", 0)).sum())
            pk_sn += int(np.asarray(infos.get("pair_sn_cand", 0)).sum())
            nz_win += int(np.asarray(infos.get("nz_win", 0)).sum())
            nz_act += int(np.asarray(infos.get("nz_act", 0)).sum())
            for _b in (1, 2, 3):
                nz_win_b[_b] += int(np.asarray(infos.get(f"nz_win_{_b}", 0)).sum())
                nz_act_b[_b] += int(np.asarray(infos.get(f"nz_act_{_b}", 0)).sum())
            for _hk in ("hist_11", "hist_22", "hist_12", "hist_xx",
                        "hist_p1", "hist_p2", "hist_p3", "hist_p4p",
                        "hist_unk",
                        "w12_all_win", "w12_all_p1", "w12_all_p2",
                        "w12_m7_win", "w12_m7_p1", "w12_m7_p2"):
                hist[_hk] += int(np.asarray(infos.get(_hk, 0)).sum())

        _start = time.time()
        main = next_to_play == main_player
        rstate1, rstate2, probs = predict_fn(rstate1, rstate2, obs, main, dones)

        actions = probs.argmax(axis=1)
        model_time += time.time() - _start

        to_play = next_to_play

        _start = time.time()
        obs, rewards, dones, infos = envs.step(actions)
        next_to_play = infos['to_play']

        bad = infos.get('bad_action')
        if bad is not None:
            bad_arr = np.asarray(bad, dtype=np.int32).reshape(-1)
            if bad_arr.size == 1 and num_envs > 1:
                bad_arr = np.full(num_envs, int(bad_arr[0]), dtype=np.int32)
            for v in bad_arr:
                bad_counts[int(v)] += 1
            bad_decisions += 1

        env_time += time.time() - _start

        step += 1

        for idx, d in enumerate(dones):
            if not d or (args.accurate and collected[idx]):
                continue
            # c1 = collected[idx]
            collected[idx] = True
            win_reason = infos['win_reason'][idx]
            pl = 1 if main[idx] else -1
            episode_length = infos['l'][idx]
            episode_reward = infos['r'][idx]
            main_reward = episode_reward * pl
            win = int(main_reward > 0)
            _nzw = infos.get('nz_win')
            _nza = infos.get('nz_act')
            if _nzw is not None:
                nz_ep_win.append(int(np.asarray(_nzw).reshape(-1)[idx]))
                nz_ep_act.append(int(np.asarray(_nza).reshape(-1)[idx]))

            win_player = 0 if (to_play[idx] == 0 and episode_reward > 0) or (to_play[idx] == 1 and episode_reward < 0) else 1
            win_players.append(win_player)
            win_agent = 1 if main_reward > 0 else 2
            win_agents.append(win_agent)
            # if not c1:
            #     print(f"{len(episode_lengths)}: {episode_length}, {main_reward}")
            ep_term.append(bool(seen_term[idx]))
            ep_wicat.append(bool(seen_wicat[idx]))
            ep_alive.append(int(term_now_max[idx]) > 0)
            ep_stable.append(bool(seen_stable[idx]))
            ep_temp.append(bool(seen_temp[idx]))
            ep_start_flag.append(bool(ep_has_start[idx]))
            ep_tail_flag.append(bool(ep_has_tail[idx]))
            ep_term_peak.append(int(term_now_max[idx]))
            ep_len_vals.append(int(episode_length))
            ep_has_start[idx] = False
            ep_has_tail[idx] = False
            ep_kaguya.append(bool(seen_kaguya[idx]))
            seen_term[idx] = False
            seen_wicat[idx] = False
            term_now_max[idx] = 0
            seen_stable[idx] = False
            seen_temp[idx] = False
            seen_kaguya[idx] = False

            episode_lengths.append(episode_length)
            episode_rewards.append(main_reward)
            win_rates.append(win)
            win_reasons.append(1 if win_reason == 1 else 0)
            if args.verbose:
                sys.stderr.write(f"Episode {len(episode_lengths)}: length={episode_length}, reward={main_reward}, win={win}, win_reason={win_reason}\n")
            else:
                pbar.set_postfix(len=np.mean(episode_lengths), reward=np.mean(episode_rewards), win_rate=np.mean(win_rates))
                pbar.update(1)

            # Only when num_envs=1, we switch the player here.
            # agent1_player 固定时不能翻转，否则卡组会跟着玩家换。
            if args.agent1_player < 0:
                if args.verbose:
                    main_player = 1 - main_player
                else:
                    main_player[idx] = 1 - main_player[idx]

        if len(episode_lengths) >= int(args.num_episodes * (1 - args.truncate)):
            break

    if not args.verbose:
        pbar.close()
    print(f"len={np.mean(episode_lengths)}, reward={np.mean(episode_rewards)}, win_rate={np.mean(win_rates)}, win_reason={np.mean(win_reasons)}")

    if ep_term:
        _n = len(ep_term)
        print(f"field: games={_n}")
        print(f"  terminal_on_field {sum(ep_term):>5}/{_n}  ({100.0*sum(ep_term)/_n:.1f}%)")
        print(f"  wicat_on_field    {sum(ep_wicat):>5}/{_n}  ({100.0*sum(ep_wicat)/_n:.1f}%)")
        print(f"  terminal_alive_end{sum(ep_alive):>5}/{_n}  ({100.0*sum(ep_alive)/_n:.1f}%)")
        print(f"  stable_term        {sum(ep_stable):>5}/{_n}  ({100.0*sum(ep_stable)/_n:.1f}%)")
        print(f"  temporary_seen     {sum(ep_temp):>5}/{_n}  ({100.0*sum(ep_temp)/_n:.1f}%)")
        print(f"  kaguya_on_field    {sum(ep_kaguya):>5}/{_n}  ({100.0*sum(ep_kaguya)/_n:.1f}%)")
        print(f"  sf_consumed={sf_consumed_total}  sf_destroyed={sf_destroyed_total}  "
              f"kaguya_effect={kaguya_effect_total}")
        print(f"  wicat_convert={wicat_convert_total}  "
              f"({wicat_convert_total / max(1, _n):.3f}/局)")
        _g = max(1, _n)
        print("  --- 次数/局（帧数按绝对玩家）---")
        print(f"  term_frames   P0={cnt['term_p0']/_g:.2f}  P1={cnt['term_p1']/_g:.2f}  "
              f"peak={cnt['term_peak']}")
        print(f"  wicat_frames  P0={cnt['wic_p0']/_g:.2f}  P1={cnt['wic_p1']/_g:.2f}  "
              f"peak={cnt['wic_peak']}")
        print(f"  kaguya_frames P0={cnt['kag_p0']/_g:.2f}  P1={cnt['kag_p1']/_g:.2f}  "
              f"peak={cnt['kag_peak']}")
        print(f"  start_used={start_used_total/_g:.3f}/局  "
              f"tail_used={tail_used_total/_g:.3f}/局  "
              f"kaguya_eff={kaguya_effect_total/_g:.2f}/局")

    print(f"mill(deck->grave): total={mill_total}")
    for _mid, _n in mill_counter.most_common(8):
        _code = _id2code.get(_mid, "?")
        _nm = _CARD_NAME.get(_code, "")
        print(f"  id={_mid:<6} code={_code:<10} {_nm:<8} {_n:>6}  "
              f"({100.0 * _n / max(1, mill_total):.1f}%)")

    if mc_sel:
        print(f"mill_cand: sel={mc_sel}  avg_size={mc_size / mc_sel:.1f}  "
              f"含START={mc_start} ({100.0 * mc_start / mc_sel:.1f}%)  "
              f"含TAIL={mc_tail} ({100.0 * mc_tail / mc_sel:.1f}%)  "
              f"size1={mc_one} ({100.0 * mc_one / mc_sel:.1f}%)")
    if mc_dsel:
        print(f"mill_cand_deck: sel={mc_dsel}  avg_size={mc_dsize / mc_dsel:.1f}")
    if ma_sel:
        print(f"mill_any: sel={ma_sel}  avg_size={ma_size / ma_sel:.1f}  "
              f"含START={ma_start} ({100.0 * ma_start / ma_sel:.1f}%)  "
              f"含TAIL={ma_tail} ({100.0 * ma_tail / ma_sel:.1f}%)")
    _h_sel = hist["hist_11"] + hist["hist_22"] + hist["hist_12"] + hist["hist_xx"]
    if _h_sel:
        _p1, _p2, _p3, _p4 = (hist["hist_p1"], hist["hist_p2"],
                              hist["hist_p3"], hist["hist_p4p"])
        _multi = hist["hist_22"] + hist["hist_12"] + hist["hist_xx"]
        print(f"hist(卡组选卡, 含妖精传姬): sel={_h_sel}  "
              f"(1,1)={hist['hist_11']} ({100.0 * hist['hist_11'] / _h_sel:.1f}%)  "
              f"(2,2)={hist['hist_22']} ({100.0 * hist['hist_22'] / _h_sel:.1f}%)  "
              f"(1,2)={hist['hist_12']}  other={hist['hist_xx']}  "
              f"unknown={hist['hist_unk']}")
        print(f"  pool: 1={_p1} ({100.0 * _p1 / _h_sel:.1f}%)  2={_p2}  "
              f"3={_p3}  >=4={_p4} ({100.0 * _p4 / _h_sel:.1f}%)  |  "
              f"多选(max>=2)={_multi} ({100.0 * _multi / max(1, _h_sel):.1f}%)")
    if hist["w12_all_win"] or hist["w12_m7_win"]:
        _w = hist["w12_all_win"]
        _m = hist["w12_m7_win"]
        print(f"w12(1,2窗口): 全部 win={_w} 选1={hist['w12_all_p1']} "
              f"选2={hist['w12_all_p2']}  |  m7素材 win={_m} "
              f"选1={hist['w12_m7_p1']} 选2={hist['w12_m7_p2']}")
    if nz_ep_win:
        _n = len(nz_ep_win)
        _tot = sum(nz_ep_win)
        print(f"nz 每局: 窗口={_tot / max(1, _n):.2f}  发动={sum(nz_ep_act) / max(1, _n):.2f} "
              f"({100.0 * sum(nz_ep_act) / max(1, _tot):.1f}%)  局数={_n}")

    if nz_win:
        _nz = "  ".join(
            f"{_b}环 {nz_act_b[_b]}/{nz_win_b[_b]}="
            f"{100.0 * nz_act_b[_b] / max(1, nz_win_b[_b]):.1f}%"
            for _b in (1, 2, 3) if nz_win_b[_b])
        print(f"nz(连锁对抗窗口): 窗口={nz_win} 发动={nz_act} "
              f"({100.0 * nz_act / max(1, nz_win):.1f}%)  |  {_nz}")

    if ps_cand or pt_cand:
        print(f"pair: START 可选={ps_cand} 选中={ps_pick} "
              f"({100.0 * ps_pick / max(1, ps_cand):.1f}%)  |  "
              f"TAIL 可选={pt_cand} 选中={pt_pick} "
              f"({100.0 * pt_pick / max(1, pt_cand):.1f}%)")

    _mills = {}
    for _mid, _cnt in mill_counter.items():
        _cd = _id2code.get(_mid, "?")
        _mills[_cd] = _mills.get(_cd, 0) + _cnt
    _rows = [("辉夜", "86937530", pk_kg), ("场地", "56725612", pk_fd),
             ("玛奇莉", "19144622", pk_mg), ("TAIL", "82119326", pt_cand),
             ("START", "19326613", ps_cand), ("白雪", "55623480", pk_sn)]
    _parts = []
    for _nm, _cd, _ca in _rows:
        _pk = _mills.get(_cd, 0)
        _rate = (100.0 * _pk / _ca) if _ca else 0.0
        _parts.append(f"{_nm} {_pk}/{_ca}={_rate:.1f}%")
    print("条件选取率: " + "  ".join(_parts))

    for _name, _flag in (("START", ep_start_flag), ("TAIL", ep_tail_flag)):
        _a = [p for f, p in zip(_flag, ep_term_peak) if f]
        _b = [p for f, p in zip(_flag, ep_term_peak) if not f]
        _al = [p for f, p in zip(_flag, ep_len_vals) if f]
        _bl = [p for f, p in zip(_flag, ep_len_vals) if not f]
        if _a and _b:
            print(f"pair_ep {_name}: 堆到 {len(_a)} 局  终局终端峰值均值 "
                  f"{sum(_a) / len(_a):.2f}  长度 {sum(_al) / len(_al):.1f}  |  "
                  f"未堆 {len(_b)} 局  {sum(_b) / len(_b):.2f}  "
                  f"长度 {sum(_bl) / len(_bl):.1f}")

    _n = mn1 + mn2
    if _n:
        print(f"mill_count: 1张={mn1} ({100.0 * mn1 / _n:.1f}%)  "
              f"2张={mn2} ({100.0 * mn2 / _n:.1f}%)  "
              f"平均 {(_n + mn2) / _n:.2f} 张/次")
        if mn1:
            print(f"  1张时: TAIL {mn1_tail} ({100.0 * mn1_tail / mn1:.1f}%)  "
                  f"START {mn1_start} ({100.0 * mn1_start / mn1:.1f}%)")
        if mn2:
            print(f"  2张时: TAIL {mn2_tail} ({100.0 * mn2_tail / mn2:.1f}%)  "
                  f"START {mn2_start} ({100.0 * mn2_start / mn2:.1f}%)")

    if rn:
        _n = max(1, args.num_episodes)
        print("rename_map: 目标选择 %d (%.3f/局)  写入映射 %d (%.3f/局)  "
              "清除 %d" % (
                  rn.get("rename_win", 0), rn.get("rename_win", 0) / _n,
                  rn.get("rename_hit", 0), rn.get("rename_hit", 0) / _n,
                  rn.get("rename_clr", 0)))

    if rn.get("obs_gyoku") or rn.get("sel_gyoku") or rn.get("pick_gyoku"):
        _n = max(1, args.num_episodes)
        print("gyoku_obs: actions_含2936 %d (%.3f/局) | 候选含 %d (%.3f/局) | 选中 %d (%.3f/局)" % (
            rn.get("obs_gyoku", 0), rn.get("obs_gyoku", 0) / _n,
            rn["sel_gyoku"], rn["sel_gyoku"] / _n,
            rn.get("pick_gyoku", 0), rn.get("pick_gyoku", 0) / _n))
        print("gyoku_id(+2937 映射): %d   rn_live(映射条目数): %d" % (
            rn.get("gyoku_id", 0), rn.get("rn_live", 0)))
    if rn.get("obs_spec_fill"):
        print("obs_spec_fill: 观测里仍需 spec 反查的候选 %d 条（§159.9 复核应为 0）" %
              rn.get("obs_spec_fill", 0))
    _as = rn.get("atk_seen", 0)
    _ac = rn.get("atk_changed", 0)
    _dc = rn.get("def_changed", 0)
    if _as:
        print("atk_change: 场上卡快照 %d 次  攻变 %d (%.2f%%)  守变 %d (%.2f%%)" % (
            _as, _ac, 100.0 * _ac / _as, _dc, 100.0 * _dc / _as))
    _at = rn.get("atk_tgt_total", 0)
    _az = rn.get("atk_tgt_zero", 0)
    _dt = rn.get("def_tgt_total", 0)
    _dz = rn.get("def_tgt_zero", 0)
    if _at or _dt:
        print("atk_visible: 攻目标 %d 次(其中atk=0 %d, %.1f%%)  "
              "守目标 %d 次(其中def=0 %d, %.1f%%)" % (
                  _at, _az, 100.0 * _az / max(1, _at),
                  _dt, _dz, 100.0 * _dz / max(1, _dt)))
    _te = rn.get("tail_equip", 0)
    _ss2 = rn.get("start_set", 0)
    if _te or _ss2:
        _n0 = max(1, args.num_episodes)
        print("grave->szone: TAIL装备 %d (%.3f/局)  START盖放 %d (%.3f/局)" % (
            _te, _te / _n0, _ss2, _ss2 / _n0))
    _tc = rn.get("tail_chain", 0)
    if _tc:
        print("tail_chain: %d 次 (%.3f/局)  ← 同回合「堆墓TAIL→墓地装备」链路" % (
            _tc, _tc / max(1, args.num_episodes)))
    _om = rn.get("opp_fusion_mat", 0)
    if _om:
        print("opp_material: %d 次 (%.3f/局)  ← 事件级奖励信号" % (
            _om, _om / max(1, args.num_episodes)))
    _gg = rn.get("gyoku_grave0", 0) + rn.get("gyoku_grave1", 0)
    _ss = rn.get("story_summon0", 0) + rn.get("story_summon1", 0)
    _to = rn.get("tail_oppmat0", 0) + rn.get("tail_oppmat1", 0)
    if _gg or _ss or _to:
        print("judge: 妖精王子进墓 %d  讲述者登场 %d  TAIL用对方素材 %d" % (
            _gg, _ss, _to))

    _tot = sum(bad_counts.values())
    print(f"bad_action: steps={bad_decisions}, entries={_tot}")
    _names = {0: 'none', 1: 'ht_summon', 2: 'ht_self', 3: 'self_attack',
              4: 'ht_correct', 5: 'negated_activate'}
    for _k in (0, 1, 2, 3, 4, 5):
        _n = bad_counts.get(_k, 0)
        _pct = 100.0 * _n / max(1, _tot)
        print(f"  {_k} {_names[_k]:<12} {_n:>7}  ({_pct:.3f}%)")

    episode_lengths = np.array(episode_lengths)
    win_players = np.array(win_players)
    win_agents = np.array(win_agents)
    N = len(win_players)

    mask1 = (win_players == 0) & (win_agents == 1)
    mask2 = (win_players == 0) & (win_agents == 2)
    mask3 = (win_players == 1) & (win_agents == 1)
    mask4 = (win_players == 1) & (win_agents == 2)

    N1, N2, N3, N4 = [np.sum(m) for m in [mask1, mask2, mask3, mask4]]

    print(f"Payoff matrix:")
    w1 = N1 / N
    w2 = N2 / N
    w3 = N3 / N
    w4 = N4 / N
    print(f"   agent1  agent2")
    print(f"0  {w1:.4f}  {w2:.4f}")
    print(f"1  {w3:.4f}  {w4:.4f}")

    print(f"0/1 matrix, win rates of agentX as playerY")
    w1 = N1 / (N1 + N4)
    w2 = N2 / (N2 + N3)
    w3 = 1 - w2
    w4 = 1 - w1
    print(f"   agent1  agent2")
    print(f"0  {w1:.4f}  {w2:.4f}")
    print(f"1  {w3:.4f}  {w4:.4f}")

    print(f"Length matrix, length of games of agentX as playerY")
    l1 = np.mean(episode_lengths[mask1 | mask4])
    l2 = np.mean(episode_lengths[mask2 | mask3])
    print(f"   agent1  agent2")
    print(f"0  {l1:3.2f}  {l2:3.2f}")
    print(f"1  {l2:3.2f}  {l1:3.2f}")

    total_time = time.time() - start
    total_steps = (step - start_step) * num_envs
    print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
    print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
