#!/bin/bash
# 阶段0 ft44：从 ft43/u732 续训 150 万步
#
# 动机：
#   ft42 是在 h_actions 列号 bug 未修时训出来的（观测有误），
#   ft43 是修复后、在错误观测分布上重建的第一段。
#   ft43 结果：vs13M −1.04 点，stable_term +5.1 但 tail/kaguya 下滑，
#   判断可能仍处于「适应正确观测」的过程中，再给 150 万步观察是否继续改善。
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=min0_ft44

if pgrep -f "cleanba.py --exp-name ${EXP_NAME}" > /dev/null 2>&1; then
  echo "ERROR: ${EXP_NAME} 已在运行"; exit 1
fi

OTHER=$(pgrep -f "cleanba.py" | head -3)
if [ -n "$OTHER" ]; then
  echo "ERROR: 还有其他 cleanba 进程："
  ps -eo pid,etime,cmd | grep cleanba | grep -v grep | cut -c1-90
  exit 1
fi

DECK=../assets/deck_md/FairyTail.ydk
if ! grep -q '^27548199$' "$DECK"; then echo "ERROR: 卡组缺白枪管"; exit 1; fi
if grep -q '^74689476$' "$DECK"; then echo "ERROR: 卡组仍有桑德里永"; exit 1; fi
echo "deck ok: 白枪管 x$(grep -c '^27548199$' "$DECK")"

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft44_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft44_meta.txt

START=checkpoints_min0_ft43_1117_165947/1794905991_59M_u732.flax_model
TB_OFFSET=62755936          # ft43 终点

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
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 44 \
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
  --tb-dir /tmp/tb_min0_ft44 \
  > /tmp/train_min0_ft44.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name ${EXP_NAME}" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft44.log"
echo "updates=$((1500000 / 8192))"
