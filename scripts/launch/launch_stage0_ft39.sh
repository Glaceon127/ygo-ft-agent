#!/bin/bash
# 阶段0 第 N 轮（ft39）：LP 单调编码（与 ft38 的 atk/def 同源修复）
#   起点 ft38（阶段0 最强：vs ft37 双向 57.52%，vs 13M 65.62% 历史最高）
#   观测变化（唯一）：
#     global_ 列 0/2 = LP/32（0..8160 单调），列 1/3 置 0
#     旧：LP 高低字节（8000→{31,64}，4000→{15,160}，1000→{3,232}，低字节反序）
#   附带：turn_count 上限 16 → 60（原截断过早）
#   其余与 ft38 完全一致
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=min0_ft39

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name ${EXP_NAME}" > /dev/null 2>&1; then
  echo "ERROR: ${EXP_NAME} 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name ${EXP_NAME}" | head -3
  exit 1
fi

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft39_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft39_meta.txt

START=checkpoints_min0_ft38_1116_221717/1794838641_48M_u366.flax_model
TB_OFFSET=50826656          # ft38 终点

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name "$EXP_NAME" \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 39 \
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
  --tb-dir /tmp/tb_min0_ft39 \
  > /tmp/train_min0_ft39.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name ${EXP_NAME}" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft39.log"
echo "updates=$((3000000 / 8192))"
