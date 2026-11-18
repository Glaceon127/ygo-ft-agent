#!/bin/bash
# 阶段0 ft48：从 ft44/u183 续训 300 万步
#
# 观测变化：cdb 的 8 个语义位已用 lua SetCategory 替换（9818 张卡）。
#   与 ft47 的区别：ft47 是 cdb|lua（OR 合并，混合噪声），
#   ft48 是纯 lua（单一可靠来源，与 API 一致率 88-100%）。
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=min0_ft48

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
grep -q '^27548199$' "$DECK" || { echo "ERROR: 卡组缺白枪管"; exit 1; }
grep -q '^74689476$' "$DECK" && { echo "ERROR: 卡组仍有桑德里永"; exit 1; }
echo "deck ok: 白枪管 x$(grep -c '^27548199$' "$DECK")"

# 校验 cdb：辉夜应只有 TOHAND|SEARCH（0x20008），不应含 SPSUMMON(0x200)
CAT=$(/home/ygo/venv/bin/python -c "
import sqlite3
c = sqlite3.connect('../assets/locale/en/cards.cdb')
r = c.execute('SELECT category FROM datas WHERE id=86937530').fetchone()
print(r[0] if r else 0)
")
SEM=$(( (CAT >> 0 & 1) | ((CAT >> 13) & 2) | ((CAT >> 28) & 4) | ((CAT >> 16) & 8) | ((CAT >> 4) & 16) | ((CAT >> 11) & 32) | ((CAT >> 1) & 64) | ((CAT >> 0) & 128) ))
echo "cdb ok: 辉夜 category=0x$(printf %08x $CAT)"
if [ "$((CAT & 131072))" -eq 0 ]; then
  echo "ERROR: 缺 SEARCH 位"; exit 1
fi

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft48_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft48_meta.txt

START=checkpoints_min0_ft44_1117_183801/1794911885_61M_u183.flax_model
TB_OFFSET=64230496

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
  --learning-rate 0.0002 --seed 48 \
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
  --tb-dir /tmp/tb_min0_ft48 \
  > /tmp/train_min0_ft48.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name ${EXP_NAME}" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft48.log"
echo "updates=$((3000000 / 8192))"
