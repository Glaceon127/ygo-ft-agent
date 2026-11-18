#!/bin/bash
# 从零训练 900 万步（1099 updates），使用新卡组 FAIRY TAIL Magistus2
# 评估对手：ft9（13M 步）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=min0_scratch9m
DECK="../assets/deck_md/FAIRY_TAIL_Magistus2.ydk"
EVAL_CKPT=checkpoints_min0_ft9_1112_111510/1794453314_13M_u488.flax_model

# ---- 前置校验 ----
if pgrep -f "cleanba.py --exp-name ${EXP_NAME}" > /dev/null 2>&1; then
  echo "ERROR: ${EXP_NAME} 已在运行"; exit 1
fi
OTHER=$(pgrep -f "cleanba.py" | head -3)
if [ -n "$OTHER" ]; then
  echo "ERROR: 还有其他 cleanba 进程："
  ps -eo pid,etime,cmd | grep cleanba | grep -v grep | cut -c1-90
  exit 1
fi

[ -f "$DECK" ] || { echo "ERROR: 卡组不存在: $DECK"; exit 1; }
[ -f "$EVAL_CKPT" ] || { echo "ERROR: 评估权重不存在: $EVAL_CKPT"; exit 1; }

MAIN=$(sed -n '/#main/,/#extra/p' "$DECK" | grep -cE '^[0-9]+$' || true)
EXTRA=$(sed -n '/#extra/,/!side/p' "$DECK" | grep -cE '^[0-9]+$' || true)
echo "deck ok: main=$MAIN extra=$EXTRA"
[ "$MAIN" = "40" ] || { echo "ERROR: 主卡组应为 40，实际 $MAIN"; exit 1; }
[ "$EXTRA" = "15" ] || { echo "ERROR: 额外卡组应为 15，实际 $EXTRA"; exit 1; }

# 卡号必须在 code_list 里，否则 embedding 越界
MISS=$(/home/ygo/venv/bin/python - <<'PY'
DECK = "/mnt/h/YGOPro/deck/FAIRY TAIL Magistus2.ydk"
CL = "/home/ygo/ygo-agent/scripts/code_list.txt"
cl = set()
for line in open(CL, encoding="utf-8", errors="ignore"):
    p = line.split()
    if p and p[0].isdigit():
        cl.add(int(p[0]))
codes = set()
for line in open(DECK, encoding="utf-8", errors="ignore"):
    s = line.strip()
    if s.isdigit():
        codes.add(int(s))
print(len(codes - cl))
PY
)
echo "codes missing in code_list: $MISS"
[ "$MISS" = "0" ] || { echo "ERROR: 有卡号不在 code_list"; exit 1; }

# cdb 校验：新卡组含白枪管，且 category 已按 lua 修过（可选，仅提示）
CAT=$(/home/ygo/venv/bin/python -c "
import sqlite3
c = sqlite3.connect('../assets/locale/en/cards.cdb')
r = c.execute('SELECT category FROM datas WHERE id=86937530').fetchone()
print(r[0] if r else 0)
")
echo "cdb: 辉夜 category=0x$(printf %08x "$CAT")"

GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "GPU used: ${GPU} MiB"

# ---- 启动 ----
STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_${EXP_NAME}_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_${EXP_NAME}_meta.txt

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

# 注意：不传 --checkpoint => 从零初始化
setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name "$EXP_NAME" \
  --ckpt-dir "$OUT" \
  --deck "$DECK" \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 9000000 \
  --tb-offset 0 \
  --learning-rate 0.0002 --seed 50 \
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
  --local-eval-episodes 32 \
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
echo "updates=$((9000000 / 8192))"
