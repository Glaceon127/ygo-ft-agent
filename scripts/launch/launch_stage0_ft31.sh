#!/bin/bash
# 阶段0 第十八轮（B2）：压正例 + 小幅加惩罚，让 bad 不被其他奖励淹没
#   起点 ft29（内部 61.33% 的最强版），保留 millF
#   改动三项：
#     ht_correct   0.20 -> 0.10   （正例减半，减少抵消）
#     self_attack  0.80 -> 1.00
#     negated      0.90 -> 1.00
#   预期：惩罚 0.626 -> 0.718，正例 0.491 -> 0.246，净 -0.135 -> -0.472
#         shaping 总幅值 0.477 -> 1.084，bad 净值占比 28.3% -> 43.5%
#   步数 150 万（183 updates，约 23 分钟）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft31_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft31_meta.txt

START=checkpoints_min0_ft29_1115_203108/1794745872_34M_u366.flax_model
TB_OFFSET=35826656          # ft29 终点

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft31" > /dev/null 2>&1; then
  echo "ERROR: min0_ft31 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft31" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft31 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 31 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 1.00 \
  --shaping-ht-correct-bonus 0.10 \
  --shaping-negated-activate-penalty 1.00 \
  --shaping-ms-scale 1.0 \
  --shaping-start-grave-bonus 0.0 \
  --shaping-tail-grave-bonus 0.20 \
  --shaping-mill-start-bonus 0.30 \
  --shaping-mill-tail-bonus 1.00 \
  --shaping-mill-field-penalty 1.00 \
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
  --tb-dir /tmp/tb_min0_ft31 \
  > /tmp/train_min0_ft31.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name min0_ft31" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft31.log"
echo "updates=$((1500000 / 8192))"
