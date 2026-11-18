#!/bin/bash
# 阶段0 第六轮：ft16 起点，续训 150 万步（等比缩放版）
#   奖励结构（未缩放）：turn 1.0；bad_action ×3；里程碑 ×3；mon ×2；wic ×10；
#                      新增 bad=5（发动已被无效的卡，0.90）；终局 tsc=6.7 → 占比 50%
#   等比缩放：全部奖励 ÷6.7 → tsc=1.0，所有 shaping 系数 ÷6.7
#   相对比例与 policy 梯度不变（advantage 归一化），value 目标回到 ±1 量级，v_loss 不涨
#   未缩放版备份：/tmp/launch_stage0_ft19_unscaled.bak
#   评估对手 = ft16 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft19_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft19_meta.txt

START=checkpoints_min0_ft16_from13M_2M/1794552901_15M_u244.flax_model
TB_OFFSET=16326656          # ft16(15M_u244) 累计步数；与 ft18 同起点，用不同 tb-dir 区分

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft19" > /dev/null 2>&1; then
  echo "ERROR: min0_ft19 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft19" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft19 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 19 \
  --shaping \
  --shaping-lp-coef 0.00002985 \
  --shaping-mon-coef 0.00597015 \
  --shaping-ht-summon-penalty 0.13432836 \
  --shaping-ht-self-penalty 1.11940299 \
  --shaping-self-attack-penalty 0.35820896 \
  --shaping-ht-correct-bonus 0.08955224 \
  --shaping-ms-scale 0.44776119 \
  --shaping-wicat-convert-bonus 0.44776119 \
  --shaping-turn-task-weight 0.14925373 \
  --shaping-negated-activate-penalty 0.13432836 \
  --shaping-terminal-scale 1.0 \
  --local-eval-episodes 8 \
  --eval-checkpoint "$START" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft19 \
  > /tmp/train_min0_ft19.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft19" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft19.log"
