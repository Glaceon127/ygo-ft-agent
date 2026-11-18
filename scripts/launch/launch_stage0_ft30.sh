#!/bin/bash
# 阶段0 第十七轮：以 ft29 为起点续训 300 万步，配置完全沿用 ft29
#   ft29 已验证：场地惩罚(-1.0) 对选择无效（场地占比 25.8% -> 25.5%），
#               但 term_frames 32.54 -> 39.78、对 13M 60.6% -> 61.3%（内部口径）
#   本轮目的：确认 ft29 的做场提升是平台还是仍在上升
#   注意：外部基准（WindBot 相剑）已证明 ft29 与 13M 同档（8.8% vs 9.4%），
#         内部提升不代表外部提升
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft30_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft30_meta.txt

START=checkpoints_min0_ft29_1115_203108/1794745872_34M_u366.flax_model
TB_OFFSET=35826656          # ft29 终点 = 32826656 + 3000000

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft30" > /dev/null 2>&1; then
  echo "ERROR: min0_ft30 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft30" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft30 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 30 \
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
  --tb-dir /tmp/tb_min0_ft30 \
  > /tmp/train_min0_ft30.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name min0_ft30" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft30.log"
echo "updates=$((3000000 / 8192))"
