#!/bin/bash
# 阶段0 第七轮：ft16 起点，终局占比 25%，续训 150 万步
#   与 ft19（同起点、终局占比 50%、150 万步）形成干净对照：唯一变量 = 终局占比
#   做法：保持终局 ±1（tsc=1.0），把 ft19 的全部 shaping 系数 ×3
#     → 终局 1.0 / shaping 3.0 = 25%；shaping 内部比例不变
#   不采用「终局 ÷3」是因为那会让 return 尺度再缩 3 倍、value 项进一步欠拟合
#   评估对手 = ft16 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft20_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft20_meta.txt

START=checkpoints_min0_ft16_from13M_2M/1794552901_15M_u244.flax_model
TB_OFFSET=16326656          # ft16(15M_u244) 累计步数

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft20" > /dev/null 2>&1; then
  echo "ERROR: min0_ft20 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft20" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft20 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 20 \
  --shaping \
  --shaping-lp-coef 0.00008955 \
  --shaping-mon-coef 0.01791045 \
  --shaping-ht-summon-penalty 0.40298508 \
  --shaping-ht-self-penalty 3.35820897 \
  --shaping-self-attack-penalty 1.07462688 \
  --shaping-ht-correct-bonus 0.26865672 \
  --shaping-ms-scale 1.34328357 \
  --shaping-wicat-convert-bonus 1.34328357 \
  --shaping-turn-task-weight 0.44776119 \
  --shaping-negated-activate-penalty 0.40298508 \
  --shaping-terminal-scale 1.0 \
  --local-eval-episodes 8 \
  --eval-checkpoint "$START" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft20 \
  > /tmp/train_min0_ft20.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft20" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft20.log"
