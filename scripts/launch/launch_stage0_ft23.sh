#!/bin/bash
# 阶段0 第十轮：ft16 起点，配置完全保持 ft16，续训 300 万步
#   目的：检验「同参数纯续训能否变强」——之前 ft19/ft21 都改了参数，不能回答这个问题
#   配置与 launch_stage0_ft16.sh 逐项一致（11 个 shaping 参数一个不改）
#   唯一差异：turn_task 的 H3 现在包含 bad=5（ft16 时只有 1/2/3），见 §113 说明
#   评估对手 = ft16 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft23_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft23_meta.txt

START=checkpoints_min0_ft16_from13M_2M/1794552901_15M_u244.flax_model
TB_OFFSET=16326656          # ft16(15M_u244) 累计步数

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft23" > /dev/null 2>&1; then
  echo "ERROR: min0_ft23 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft23" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft23 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 23 \
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
  --eval-checkpoint "$START" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft23 \
  > /tmp/train_min0_ft23.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft23" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft23.log"
