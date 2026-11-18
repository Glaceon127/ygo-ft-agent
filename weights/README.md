# 权重

权重文件（`.flax_model`，单个约 75 MB）不存放在仓库里，从 GitHub Release 下载后放入本目录。

## 训练（续训）

```bash
python cleanba.py \
  --checkpoint weights/xxx.flax_model \
  --ckpt-dir checkpoints_xxx \
  --deck assets/FairyTail.ydk \
  --num-embeddings 15049 \
  --total-timesteps 3000000 \
  --tb-offset <上一轮终点的 global_step> \
  --learning-rate 0.0002 --seed 1 \
  ...
```

`--tb-offset` 填上一轮的终点，续训后 TensorBoard 的横轴才是连续的全局步数。

## 推理

```bash
# cwd 必须是 scripts/ —— features.py 用相对路径读 code_list.txt
cd scripts

CHECKPOINT=../weights/xxx.flax_model \
JAX_PLATFORMS=cuda \
  python -m uvicorn ygoinf.server:app --host 127.0.0.1 --port 3000
```

启动后可访问 `http://127.0.0.1:3000/docs` 确认服务就绪。

## 评估

```bash
python battle.py \
  --checkpoint1 weights/a.flax_model --deck1 assets/FAIRY_TAIL_Magistus2.ydk \
  --checkpoint2 weights/b.flax_model --deck2 assets/FairyTail.ydk \
  --agent1-player 0 \
  --num-episodes 1024 --num-envs 32
```

`--deck1` / `--deck2` 让两侧用不同卡组；`--agent1-player 0`（或 1）固定座位，否则每局翻转会导致卡组跟着换人。

## 校验

```bash
sha256sum xxx.flax_model
```
