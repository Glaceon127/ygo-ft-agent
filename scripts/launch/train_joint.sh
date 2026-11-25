#!/usr/bin/env bash
# Run in the scripts directory of the prepared upstream working copy.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
    echo "Usage: bash train_joint.sh CHECKPOINT GLOBAL_STEP [ENVIRONMENT_STEPS]" >&2
    exit 2
fi
checkpoint=$(realpath -e "$1")
offset=$2
budget=${3:-6000000}
[[ $offset =~ ^[0-9]+$ && $budget =~ ^[1-9][0-9]*$ ]]
test -f train_counters.py
steps=$(( (budget + 5119) / 5120 * 5120 ))
export JOINT_RUN=$(mktemp -d "$PWD/run_joint_XXXXXX")
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}
echo "Training log: $JOINT_RUN/training.log"
python -u train_counters.py --exp-name popular8 --seed 248 --log-frequency 1 --save-interval 20 \
    --ckpt-dir "$JOINT_RUN/checkpoints" --tb-dir "$JOINT_RUN/tb" --tb-offset "$offset" \
    --checkpoint "$checkpoint" --opponent-checkpoint "$checkpoint" --eval-checkpoint "$checkpoint" \
    --num-embeddings 15049 --no-switch --asymmetric --no-concurrency --max-options 64 \
    --m1.battle-features --m2.battle-features --m1.chain-effect-features --m2.chain-effect-features \
    --m1.chain-response-features --m2.chain-response-features --m1.own-turn-ash-g --m2.own-turn-ash-g \
    --local-num-envs 40 --local-env-threads 40 --num-actor-threads 1 --num-steps 128 --num-minibatches 20 \
    --actor-device-ids 0 --learner-device-ids 0 --total-timesteps "$steps" --learning-rate 0.00005 \
    --anneal-lr --eval-interval 50 --local-eval-episodes 8 --shaping-turn-task-anneal-timesteps "$budget" \
    --shaping --shaping-lp-coef 0.0002 --shaping-mon-coef 0.01 --shaping-ht-summon-penalty 0.3 \
    --shaping-ht-self-penalty 1.0 --shaping-self-attack-penalty 0.5 --shaping-ht-correct-bonus 0.1 \
    --shaping-term-coef 0.0 --shaping-ms-scale 1.0 --shaping-mill-start-bonus 0.3 \
    --shaping-mill-field-penalty 0.5 --shaping-mill-tail-bonus 0.4 --shaping-start-grave-bonus 0.0 \
    --shaping-tail-grave-bonus 0.2 --shaping-sf-consumed-bonus 0.2 --shaping-sf-destroyed-penalty 0.2 \
    --shaping-terminal-scale 1.0 --shaping-wicat-convert-bonus 0.2 --shaping-opp-material-bonus 0.4 \
    --shaping-tail-chain-bonus 0.0 --shaping-negated-activate-penalty 0.0 \
    --shaping-turn-task-weight 1.0 --shaping-turn-task-main 0.5 2>&1 | tee "$JOINT_RUN/training.log"
