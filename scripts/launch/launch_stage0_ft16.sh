#!/bin/bash
# 阶段0 第三轮：200 万步（244 updates）
#   起点 13M，评估对手同为 13M（冻结）
#   相对 ft14 的四个新变量：
#     1) 里程碑判定修复（ctrl==0 + x[11]!=2）
#     2) START / TAIL 改为「效果发动」驱动
#     3) --shaping-wicat-convert-bonus 0.30
#     4) battle.py 新增次数口径（不影响训练）
#   其余与 ft14 完全一致
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft16_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft16_meta.txt

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft16" > /dev/null 2>&1; then
  echo "ERROR: min0_ft16 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft16" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft16 \
  --ckpt-dir "$OUT" \
  --checkpoint checkpoints_min0_ft9_1112_111510/1794453314_13M_u488.flax_model \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 2000000 \
  --tb-offset 18827808 \
  --learning-rate 0.0002 --seed 16 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 0.80 \
  --shaping-ht-correct-bonus 0.20 \
  --shaping-ms-scale 1.0 \
  --shaping-wicat-convert-bonus 0.30 \
  --shaping-turn-task-weight 1.0 \
  --local-eval-episodes 8 \
  --eval-checkpoint checkpoints_min0_ft9_1112_111510/1794453314_13M_u488.flax_model \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft16 \
  > /tmp/train_min0_ft16.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft16" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft16.log"
