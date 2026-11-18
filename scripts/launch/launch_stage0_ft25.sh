#!/bin/bash
# 阶段0 第十二轮：START 只留堆墓奖励，TAIL 保留「发动 + 堆墓」，两卡权重约 1 : 2.5
#   起点 ft16，其余参数与 ft16 一致 → 变量 = 这两张卡的奖励结构
#   START：--shaping-start-grave-bonus 0（发动奖励关闭）
#          --shaping-mill-start-bonus 0.10（只奖励从卡组堆墓）
#   TAIL ：--shaping-tail-grave-bonus 0.05（发动/融合）
#          --shaping-mill-tail-bonus 0.15（从卡组堆墓）
#   每局期望幅值：START = 0.10×0.387 ≈ 0.039
#                 TAIL  = 0.05×0.65 + 0.15×0.436 ≈ 0.033 + 0.065 = 0.098
#                 → 比值 ≈ 1 : 2.5
#   预估：ms+事件 0.387 -> 0.433，shaping 3.562 -> 3.608，终局占比 21.9% -> 21.7%
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft25_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft25_meta.txt

START=checkpoints_min0_ft16_from13M_2M/1794552901_15M_u244.flax_model
TB_OFFSET=16326656

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft25" > /dev/null 2>&1; then
  echo "ERROR: min0_ft25 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft25" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft25 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 25 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 0.80 \
  --shaping-ht-correct-bonus 0.20 \
  --shaping-ms-scale 1.0 \
  --shaping-start-grave-bonus 0.0 \
  --shaping-tail-grave-bonus 0.05 \
  --shaping-mill-start-bonus 0.10 \
  --shaping-mill-tail-bonus 0.15 \
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
  --tb-dir /tmp/tb_min0_ft25 \
  > /tmp/train_min0_ft25.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft25" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft25.log"
