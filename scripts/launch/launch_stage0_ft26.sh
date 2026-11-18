#!/bin/bash
# 阶段0 第十三轮：在 ft25 基础上续训 900 万步（排除「训练量不足」这个解释）
#   起点：ft25/17M_u183（START 只留堆墓 0.10、TAIL 0.05+0.15 的那版）
#   配置：与 ft25 完全一致，一个字都不改
#   目的：ft25 只跑了 150 万步就出现行为回落（stable_term 87.1% -> 82.6%），
#         本轮用 6 倍步数看它是「还没学会」还是「方向本身有害」
#   turn 退火会重置（anneal_total = 9e6/8 = 1125000），前 40% 仍有 turn 信号
#   预计 1098 updates / 约 130 分钟（SPS ~1150）
#   评估对手 = ft25 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft26_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft26_meta.txt

START=checkpoints_min0_ft25_1115_102803/1794709687_17M_u183.flax_model
TB_OFFSET=17826656          # ft25 终点累计步数 = 16326656 + 1500000

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft26" > /dev/null 2>&1; then
  echo "ERROR: min0_ft26 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft26" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft26 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 9000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 26 \
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
  --tb-dir /tmp/tb_min0_ft26 \
  > /tmp/train_min0_ft26.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft26" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft26.log"
echo "updates=$((9000000 / 8192))"
