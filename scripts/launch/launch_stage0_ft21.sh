#!/bin/bash
# 阶段0 第八轮：ft16 起点，150 万步，方案 A（压低 turn、抬高纠错/里程碑/wic）
#   相对 ft16：
#     turn 1.0 -> 0.4（会退火到 0）
#     bad 惩罚 ht_summon 3x / ht_self 1.2x / self_attack 3x，新增 bad5 0.90
#     ht_correct 保持 0.20（正例奖励，不跟着抬）
#     ms 1.0 -> 2.0；wic 0.30 -> 1.00
#     lp / mon / terminal_scale 均不变（终局 ±1）
#   预测占比：turn 9.3%、bad惩罚 15.1%、里程碑 16.1%、wic 8.6%、终局 19.1%
#   评估对手 = ft16 起点（冻结）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft21_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft21_meta.txt

START=checkpoints_min0_ft16_from13M_2M/1794552901_15M_u244.flax_model
TB_OFFSET=16326656          # ft16(15M_u244) 累计步数

# ---- 防重复启动：同名 exp 已在运行则直接退出 ----
if pgrep -f "cleanba.py --exp-name min0_ft21" > /dev/null 2>&1; then
  echo "ERROR: min0_ft21 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name min0_ft21" | head -3
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name min0_ft21 \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 1500000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 21 \
  --shaping \
  --shaping-lp-coef 0.0002 \
  --shaping-mon-coef 0.02 \
  --shaping-ht-summon-penalty 0.90 \
  --shaping-ht-self-penalty 3.00 \
  --shaping-self-attack-penalty 2.40 \
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
  --tb-dir /tmp/tb_min0_ft21 \
  > /tmp/train_min0_ft21.log 2>&1 < /dev/null &

sleep 3
PGID=$(pgrep -f "cleanba.py --exp-name min0_ft21" | head -1)
echo "pid=$PGID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft21.log"
