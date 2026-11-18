#!/bin/bash
# 从 scratch12m/u1404（1196 万步）续训 900 万步 → 总约 2096 万步
# 卡组：FAIRY_TAIL_Magistus2（新）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=scratch12m_plus9m
DECK="../assets/deck_md/FAIRY_TAIL_Magistus2.ydk"
START=checkpoints_min0_scratch12m_1118_094758/1794966480_11M_u1404.flax_model
EVAL_CKPT=checkpoints_min0_ft9_1112_111510/1794453314_13M_u488.flax_model
TB_OFFSET=11960320
STEPS=9000000

if pgrep -f "cleanba.py --exp-name ${EXP_NAME}" > /dev/null 2>&1; then
  echo "ERROR: ${EXP_NAME} 已在运行"; exit 1
fi
OTHER=$(pgrep -f "cleanba.py" | head -3 || true)
if [ -n "$OTHER" ]; then
  echo "ERROR: 还有其他 cleanba 进程："
  ps -eo pid,etime,cmd | grep cleanba | grep -v grep | cut -c1-90
  exit 1
fi

[ -f "$DECK" ] || { echo "ERROR: 卡组不存在: $DECK"; exit 1; }
[ -f "$START" ] || { echo "ERROR: 起点不存在: $START"; exit 1; }
[ -f "$EVAL_CKPT" ] || { echo "ERROR: 评估权重不存在"; exit 1; }

MAIN=$(sed -n '/#main/,/#extra/p' "$DECK" | grep -cE '^[0-9]+$' || true)
EXTRA=$(sed -n '/#extra/,/!side/p' "$DECK" | grep -cE '^[0-9]+$' || true)
echo "deck ok: main=$MAIN extra=$EXTRA"
[ "$MAIN" = "40" ] || { echo "ERROR: main != 40"; exit 1; }

echo "start=$START"
echo "tb_offset=$TB_OFFSET  steps=$STEPS  target_total=$((TB_OFFSET + STEPS))"
echo "GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_${EXP_NAME}_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_${EXP_NAME}_meta.txt

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
  --total-timesteps "$STEPS" \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 51 \
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
  --eval-checkpoint "$EVAL_CKPT" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_${EXP_NAME} \
  > /tmp/train_${EXP_NAME}.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name ${EXP_NAME}" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_${EXP_NAME}.log"
echo "updates=$((STEPS / 8192))"
