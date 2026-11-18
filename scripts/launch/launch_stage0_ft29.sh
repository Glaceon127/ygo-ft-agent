#!/bin/bash
# 阶段0 第十六轮：堆到场地魔法（舞踏会）给 -1.0 惩罚
#   用户判断：场地魔法进墓完全无收益，应避免
#   实测数据：堆舞踏会的局胜率 60.6% ≈ 全局 60.5%（没有证据表明有害），
#             但本轮按用户要求执行，同时观察是否出现「回避发动威凯特」
#   风险：堆场地 0.652 次/局 × (-1.0) = -0.652/局，会把 avg_return 拉负；
#         模型可能学成「不发动威凯特 e2」（不堆就不扣），需盯 mill 总量
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft29_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft29_meta.txt

START=checkpoints_min0_ft28_1115_174837/1794736121_31M_u366.flax_model
TB_OFFSET=32826656          # ft28 终点 = 29826656 + 3000000

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft29" > /dev/null 2>&1; then
  echo "ERROR: min0_ft29 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft29" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft29 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 29 \
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
  --tb-dir /tmp/tb_min0_ft29 \
  > /tmp/train_min0_ft29.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft29" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft29.log"
echo "updates=$((3000000 / 8192))"
