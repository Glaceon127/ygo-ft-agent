#!/bin/bash
# 阶段0 第 N 轮（ft36）：同回合链路奖励「堆墓选中 TAIL → TAIL 从墓地装备」
#   起点 ft35（当前最强：vs ft34 双向 52.98%，bad 0.795 历史最低）
#   唯一新增：--shaping-tail-chain-bonus 1.00
#     判据（C++ info:tail_chain）：同回合内，mill_id 选中 TAIL(12451) 后
#       TAIL 从 GRAVE→SZONE（e3 装备）→ +1（每局最多一次）
#     实测 0.320/局（85% 的堆墓 TAIL 都在同回合被用上）
#     权重 1.00 → 每局期望 +0.320（占 shaping 6.7%）
#   其余与 ft35 完全一致（含 omm 0.80，按用户要求不改）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft36_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft36_meta.txt

START=checkpoints_min0_ft35_1116_181309/1794823993_39M_u366.flax_model
TB_OFFSET=41826656          # ft35 终点

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft36" > /dev/null 2>&1; then
  echo "ERROR: min0_ft36 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft36" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft36 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 36 \
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
  --tb-dir /tmp/tb_min0_ft36 \
  > /tmp/train_min0_ft36.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name min0_ft36" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft36.log"
echo "updates=$((3000000 / 8192))"
