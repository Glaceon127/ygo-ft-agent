# 权重

本仓库不直接存放 `.flax_model`（单个约 75 MB）。权重通过 **GitHub Release** 分发，下载后放入本目录。

## 下载

| 版本 | 说明 | vs 13M（双向 1024 局） | 状态 |
|---|---|---|---|
| `v1-ft44` | 阶段 0 最强（旧卡组 FairyTail） | **67.43%** | 可用 |
| `v2-scratch9m` | 从零 1200 万步（新卡组 Magistus2） | 待评估 | 训练中 |

## 使用

```bash
# 续训
python cleanba.py --checkpoint weights/xxx.flax_model --total-timesteps 3000000 ...

# 推理（注意 cwd 必须是 scripts/）
cd scripts
CHECKPOINT=../weights/xxx.flax_model \
JAX_PLATFORMS=cuda \
  python -m uvicorn ygoinf.server:app --host 127.0.0.1 --port 3000
```

## 校验

下载后建议核对 sha256，每个 Release 页面会给出：

```bash
sha256sum xxx.flax_model
```
