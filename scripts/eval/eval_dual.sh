#!/bin/bash
# 双卡组评估 wrapper
#
# 用法:
#   bash scripts/eval/eval_dual.sh <ckpt1> <deck1> <ckpt2> <deck2> [episodes] [seed]
#
# 例:
#   bash scripts/eval/eval_dual.sh \
#     checkpoints_xxx/u1404.flax_model FAIRY_TAIL_Magistus2 \
#     checkpoints_ft9/13M_u488.flax_model FairyTail 1024 1
#
# 两个要点:
#   1. --deck 传**目录**，--deck1/--deck2 传**卡组名**（不含 .ydk），
#      传完整路径会让 C++ 层抛 std::bad_function_call
#   2. 目录必须用 assets/deck_md —— 后来新建的 deck_eval 目录里，
#      .ydk 会被清成 0 字节（原因未定位），导致 "Main deck must contain 40" 崩溃
set -e

AGENT_HOME=${AGENT_HOME:-/home/ygo/ygo-agent}
SCRIPTS="$AGENT_HOME/scripts"
ASSETS="$AGENT_HOME/assets"
SRC="$ASSETS/deck_md"
POOL="$SRC"

# 只做校验，不重建：deck_md 是训练/评估共用的稳定目录
check_deck_pool() {
  local d1=$1 d2=$2
  for name in "$d1" "$d2"; do
    local f="$POOL/$name.ydk"
    if [ ! -f "$f" ]; then
      echo "[error] 卡组不存在: $f"; return 1
    fi
    local sz main
    sz=$(stat -c %s "$f")
    main=$(sed -n '/#main/,/#extra/p' "$f" | grep -cE '^[0-9]+$' || true)
    if [ "$sz" -eq 0 ] || [ "$main" -lt 40 ]; then
      echo "[error] 卡组异常: $name (size=$sz main=$main)"; return 1
    fi
    echo "[deck] $name: size=$sz main=$main"
  done
}

main() {
  if [ $# -lt 4 ]; then
    echo "usage: $0 <ckpt1> <deck1> <ckpt2> <deck2> [episodes=1024] [seed=1]"
    exit 1
  fi
  local ckpt1=$1 deck1=$2 ckpt2=$3 deck2=$4
  local episodes=${5:-1024}
  local seed=${6:-1}

  check_deck_pool "$deck1" "$deck2" || exit 1

  # 校验权重
  for c in "$ckpt1" "$ckpt2"; do
    [ -f "$c" ] || { echo "[error] 权重不存在: $c"; exit 1; }
  done

  # num_envs 不能超过 num_episodes（battle.py 里 battle() 有 assert）
  local envs=32
  [ "$episodes" -lt "$envs" ] && envs=$episodes

  echo "[run] $deck1 vs $deck2   episodes=$episodes envs=$envs seed=$seed"
  cd "$SCRIPTS"
  timeout 3600 /home/ygo/venv/bin/python -u battle.py \
    --deck        "../assets/deck_md" \
    --deck1       "$deck1" \
    --deck2       "$deck2" \
    --checkpoint1 "$ckpt1" \
    --checkpoint2 "$ckpt2" \
    --num-embeddings 15049 \
    --num-episodes "$episodes" --num-envs "$envs" --seed "$seed" \
    --no-accurate > "/tmp/eval_dual_$(date +%H%M%S).log" 2>&1
  local rc=$?

  local latest
  latest=$(ls -t /tmp/eval_dual_*.log 2>/dev/null | head -1)
  echo "[rc=$rc] log=$latest"
  if [ -f "$latest" ]; then
    grep -E '^len=|stable_term|terminal_on_field' "$latest" | tail -3 || true
  fi
  return $rc
}

main "$@"
