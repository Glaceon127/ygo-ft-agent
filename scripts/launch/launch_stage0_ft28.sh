#!/bin/bash
# 阶段0 第十五轮：ft27 起点，millT 0.60 -> 1.00（单次 = 终局胜负），其余全不动
#   背景：ft27 实测 69.4% 的堆墓是 2 张，其中 TAIL 命中率 23.8%（随机 18.2%）
#         ——模型已有偏好但不够强，本轮测试「奖励能否继续推高」
#   单变量：只改 mill_tail，mill_start 0.30 / tail_grave 0.20 / start_grave 0 均不变
#   期望：TAIL = 1.00x0.436 + 0.20x0.65 = 0.566/局，shaping 合计约 4.28，终局占比 18.9%
#   判据：2 张场景 TAIL 命中率 23.8% -> 40%+
#   盯：wicat_convert（当前 0.391，若掉说明威凯特被浪费在无谓堆墓）
#       stable_term（当前 85.2%，若掉说明为堆墓牺牲做场）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft28_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft28_meta.txt

START=checkpoints_min0_ft27_1115_164117/1794732081_28M_u366.flax_model
TB_OFFSET=29826656          # ft27 终点 = 26826656 + 3000000

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft28" > /dev/null 2>&1; then
  echo "ERROR: min0_ft28 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft28" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft28 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 28 \
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
  --shaping-mill-tail-bonus 1.00 \
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
  --tb-dir /tmp/tb_min0_ft28 \
  > /tmp/train_min0_ft28.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft28" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft28.log"
echo "updates=$((3000000 / 8192))"
