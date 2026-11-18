#!/bin/bash
# 阶段0 第九轮：从 ft21 续训 100 万步（方案 A 的修补版）
#   起点：ft21/17M_u183（做场历史最好，但纠错变差、手坑萎缩）
#   改动（相对 ft21）：bad 惩罚恢复 ft16 量级
#     ht_summon 0.90 -> 0.30、ht_self 3.00 -> 2.50、self_attack 2.40 -> 0.80
#     保留 bad5 0.90（唯一被验证有效的纠错项）
#   保留 ft21 的有效项：turn 0.4（仍会退火）、ms 2.0、wic 1.00
#   注意：turn 退火按本次训练重置，会在 122 updates 内再走一遍 0.4 -> 0
#   评估对手 = ft21 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft22_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft22_meta.txt

START=checkpoints_min0_ft21_1114_205541/1794660946_17M_u183.flax_model
TB_OFFSET=17826656          # ft21 终点累计步数 = 16326656 + 1500000

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft22" > /dev/null 2>&1; then
  echo "ERROR: min0_ft22 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft22" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft22 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 22 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.30 \
  --shaping-ht-self-penalty 2.50 \
  --shaping-self-attack-penalty 0.80 \
  --shaping-ht-correct-bonus 0.20 \
  --shaping-ms-scale 2.0 \
  --shaping-wicat-convert-bonus 1.00 \
  --shaping-turn-task-weight 0.4 \
  --shaping-negated-activate-penalty 0.90 \
  --shaping-terminal-scale 1.0 \
  --local-eval-episodes 8 \
  --eval-checkpoint "$START" \
  --eval-interval 50 \
  --no-concurrency \
  --local-num-envs 8 --local-env-threads 8 --num-actor-threads 8 \
  --num-steps 128 --num-minibatches 16 \
  --save-interval 50 \
  --actor-device-ids 0 --learner-device-ids 0 \
  --tb-dir /tmp/tb_min0_ft22 \
  > /tmp/train_min0_ft22.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft22" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft22.log"
