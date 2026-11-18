#!/bin/bash
# 阶段0 ft41：从 ft37 起，600 万步（编码已回滚，配置同 ft37）
#
# 为什么从 ft37 而不是 ft39：
#   ft38/ft39 是在「新编码」下训练的（atk/32、def 归零、LP 饱和）
#   现在编码已回滚到原作者设计（float_transform 高低字节）
#   ft37 是唯一「训练编码 == 当前编码」的权重（61.04% vs 13M）
#   ft39 在当前编码下会掉 3.5 点
#
# 相对 ft38/ft39 的差异：
#   - atk/def/LP 恢复 float_transform（网络侧 bytes_to_bin 重组成真实值）
#   - turn 上限回到 16
#   - Link 怪 def 列仍存箭头掩码（cdb 标准约定，推理侧已同步）
set -e
cd /home/ygo/ygo-agent/scripts || exit 1

EXP_NAME=min0_ft41

# ---- 防重复启动 ----
if pgrep -f "cleanba.py --exp-name ${EXP_NAME}" > /dev/null 2>&1; then
  echo "ERROR: ${EXP_NAME} 已在运行，拒绝重复启动"
  pgrep -af "cleanba.py --exp-name ${EXP_NAME}" | head -3
  exit 1
fi

STAMP=$(date +%m%d_%H%M%S)
OUT=checkpoints_min0_ft41_$STAMP
mkdir -p "$OUT"
echo "OUT=$OUT" > /tmp/train_min0_ft41_meta.txt

START=checkpoints_min0_ft37_1116_201339/1794831223_45M_u366.flax_model
TB_OFFSET=47826656          # ft37 终点

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=0

setsid nohup /home/ygo/venv/bin/python -u cleanba.py \
  --exp-name "$EXP_NAME" \
  --ckpt-dir "$OUT" \
  --checkpoint "$START" \
  --deck ../assets/deck_md/FairyTail.ydk \
  --code-list-file code_list.txt \
  --num-embeddings 15049 \
  --total-timesteps 6000000 \
  --tb-offset $TB_OFFSET \
  --learning-rate 0.0002 --seed 41 \
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
  --tb-dir /tmp/tb_min0_ft41 \
  > /tmp/train_min0_ft41.log 2>&1 < /dev/null &

sleep 3
PID=$(pgrep -f "cleanba.py --exp-name ${EXP_NAME}" | head -1)
echo "pid=$PID"
echo "out=$OUT"
echo "log=/tmp/train_min0_ft41.log"
echo "updates=$((6000000 / 8192))"
