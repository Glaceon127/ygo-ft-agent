#!/bin/bash
# 阶段0 第十一轮：ft16 起点，只提高堆墓目标权重（start_grave / tail_grave）
#   起点与 ft16 相同，其余 11 个 shaping 参数与 ft16 一致 → 单变量对照
#   变化：start_grave 0.08 -> 0.30；tail_grave 0.07 -> 0.30（各约 4 倍）
#   依据：录像显示威凯特堆墓常选舞踏会（池子 25%，实测 26.4% = 无偏好），
#         而 START/TAIL 各仅 1 张、且 TAIL 必须先入墓才能 e3 装备 -> e2 融合
#   预估：ms 幅值 0.387 -> 0.656，占总奖励 8.5% -> 13.6%，终局 21.9% -> 20.7%
#   评估对手 = ft16 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft24_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft24_meta.txt

START=checkpoints_min0_ft16_from13M_2M/1794552901_15M_u244.flax_model
TB_OFFSET=16326656

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft24" > /dev/null 2>&1; then
  echo "ERROR: min0_ft24 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft24" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft24 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 24 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 0.80 \
  --shaping-ht-correct-bonus 0.20 \
  --shaping-ms-scale 1.0 \
  --shaping-start-grave-bonus 0.30 \
  --shaping-tail-grave-bonus 0.30 \
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
  --tb-dir /tmp/tb_min0_ft24 \
  > /tmp/train_min0_ft24.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft24" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft24.log"
