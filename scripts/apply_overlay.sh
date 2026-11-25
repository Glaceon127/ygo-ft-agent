#!/usr/bin/env bash
# Apply to a working copy of SBL1996/ygo-agent. The editable Python packages
# and native build will then both use files under that copy.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
upstream=$(cd "${1:?usage: bash apply_overlay.sh UPSTREAM_WORKTREE}" && pwd)
test -f "$upstream/ygoenv/ygoenv/ygopro/ygopro.h"
test -f "$upstream/ygoinf/ygoinf/features.py"
cd "$root"
install -m 644 env/*.h env/ygopro.cpp "$upstream/ygoenv/ygoenv/ygopro/"
install -m 644 env/fairy.xmake.lua "$upstream/"
install -m 644 rl/agent.py "$upstream/ygoai/rl/jax/"
install -m 644 rl/jax_ckpt.py "$upstream/ygoai/rl/jax/ckpt.py"
install -m 644 rl/{env,utils,shaping}.py "$upstream/ygoai/rl/"
install -m 644 rl/ygoai_utils.py "$upstream/ygoai/utils.py"
install -m 644 rl/{battle,cleanba,train_counters,serve_counters}.py rl/_*.py assets/code_list.txt "$upstream/scripts/"
install -m 644 infer/*.py "$upstream/ygoinf/ygoinf/"
install -d "$upstream/scripts/ft_counter_data" "$upstream/assets/joint_decks"
install -m 644 assets/counter/*.json "$upstream/scripts/ft_counter_data/"
install -m 644 assets/joint_decks/*.ydk "$upstream/assets/joint_decks/"
