#!/bin/bash
# 阶段0 第十九轮（ft32）：首次带「改卡名可见」观测的训练
#   起点 ft29（34M_u366，内部 61.33% 的最强版）
#   配置与 ft29 完全一致（millT 1.00 / millS 0.30 / millF 1.00 / tail_grave 0.20）
#   唯一变化在 C++ 观测层：
#     1) 三家改名效果（玛奇莉勒 e4 / 场地 e3 / 编写者 e2）选中对方怪兽后，
#        该卡在 cards_ 与 actions_ 的 code 列变成 19144623（内部 id 2937）
#     2) 卡离场时清除映射
#   目的：让模型第一次能「看见」妖精王子，从而学会用对方怪兽融合出讲述者
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft32_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft32_meta.txt

START=checkpoints_min0_ft29_1115_203108/1794745872_34M_u366.flax_model
TB_OFFSET=35826656

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft32" > /dev/null 2>&1; then
  echo "ERROR: min0_ft32 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft32" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft32 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 32 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 0.80 \
  --shaping-negated-activate-penalty 0.50 \
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
  --tb-dir /tmp/tb_min0_ft32 \
  > /tmp/train_min0_ft32.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name min0_ft32" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft32.log"
echo "updates=$((3000000 / 8192))"
