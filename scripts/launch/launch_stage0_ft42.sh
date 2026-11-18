#!/bin/bash
# 阶段0 ft42：从 ft41 起，300 万步，唯一变量 = 卡组多一张白枪管
#
# 本轮设计（单变量）：
#   ft41 起点：在「桑德里永版」卡组上训了 600 万步
#   当前卡组：桑德里永 → 白枪管（27548199）
#   → 模型对新卡的 card_id embedding 未训练过，需要适应
#
# 观测/奖励/超参与 ft41 完全一致，只改卡组。
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=min0_ft42

# ---- 防重复启动 ----
if pgrep -f "cleanba.py --exp-name ${EXP_NAME}" > /dev/null 2>&1; then
  echo "ERROR: ${EXP_NAME} 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name ${EXP_NAME}" | head -3
  exit 1
fi

# ---- 卡组校验（必须有白枪管、不能有桑德里永）----
DECK=../assets/deck_md/FairyTail.ydk
if ! grep -q '^27548199$' "$DECK"; then
  echo "ERROR: 卡组里没有白枪管 27548199"
  exit 1
fi
if grep -q '^74689476$' "$DECK"; then
  echo "ERROR: 卡组里还有桑德里永 74689476"
  exit 1
fi
echo "deck ok: 白枪管 x$(grep -c '^27548199$' "$DECK")"

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft42_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft42_meta.txt

START=checkpoints_min0_ft41_1117_111737/1794885461_51M_u732.flax_model
TB_OFFSET=53826656          # ft41 终点

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name "$EXP_NAME" \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck "$DECK" \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 42 \
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
  --shaping-opp-material-bonus 0.80 \
  --shaping-tail-chain-bonus 1.00 \
  --shaping-turn-task-weight 1.0 \
  --local-eval-episodes 8 \
  --eval-checkpoint "$START" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft42 \
  > /tmp/train_min0_ft42.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name ${EXP_NAME}" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft42.log"
echo "updates=$((3000000 / 8192))"
