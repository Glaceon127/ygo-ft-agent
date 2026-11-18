#!/bin/bash
# 阶段0 第十四轮：ft26 起点，把 TAIL/START 的「获取 + 使用」链路整体加权
#   核心目标：让模型在「卡组有 TAIL、墓地没有」时优先把 TAIL 堆进墓地
#   ft26 实测堆墓分布仍接近随机（TAIL 10.5% vs 基准 9.1%），说明信号被 value 吸收
#
#   改动（相对 ft26）：
#     mill_tail   0.15 -> 0.60   （4 倍，单次 = 终局胜负的 60%）
#     mill_start  0.10 -> 0.30   （3 倍，保持「START = TAIL 的一半」）
#     tail_grave  0.05 -> 0.20   （4 倍，使用效果，与堆墓 1:3）
#     start_grave 0    -> 0      （不新增：START 使用效果不给奖励）
#     ms_scale    1.0  -> 1.0    （不上调：launch/term 已是 90%+ 触发，属白送）
#
#   每局期望：TAIL = 0.60x0.436 + 0.20x0.65 = 0.262 + 0.130 = 0.392
#             START = 0.30x0.387 = 0.116（只奖励堆墓）
#             TAIL : START ≈ 3.4 : 1
#   shaping 合计约 4.10，终局占比约 19.6%
#   步数 300 万（366 updates，约 45 分钟）
#   注意：ft25 只跑 150 万步时无效、ft26 跑 900 万步才显现，300 万步若看不到变化
#         不能立刻判定无效，可能只是步数不够
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft27_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft27_meta.txt

START=checkpoints_min0_ft26_1115_112217/1794712941_25M_u1098.flax_model
TB_OFFSET=26826656          # ft26 终点 = 17826656 + 9000000

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft27" > /dev/null 2>&1; then
  echo "ERROR: min0_ft27 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft27" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft27 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 27 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 0.80 \
  --shaping-ht-correct-bonus 0.20 \
  --shaping-ms-scale 1.0 \
  --shaping-start-grave-bonus 0.0 \
  --shaping-tail-grave-bonus 0.20 \
  --shaping-mill-start-bonus 0.30 \
  --shaping-mill-tail-bonus 0.60 \
  --shaping-wicat-convert-bonus 0.30 \
  --shaping-turn-task-weight 1.0 \
  --local-eval-episodes 8 \
  --eval-checkpoint "$START" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft27 \
  > /tmp/train_min0_ft27.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft27" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft27.log"
echo "updates=$((3000000 / 8192))"
