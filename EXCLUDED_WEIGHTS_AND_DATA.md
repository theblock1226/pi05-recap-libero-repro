# 未纳入本仓库的大文件（务必另存或可再下载）

## 可再下载（公开）
- HF Dataset: `RLinf/RECAP-Libero10-Task0-48succ-Data` → 原路径 `/root/rlinf_recap/datasets/...` (~82G)
- π0.5 / SigLIP2 / Gemma3 底模（HF 或 openpi assets）
- RLinf 源码：GitHub `RLinf/RLinf`

## 本次训练产物（公司机本地，离职前请另拷）
这些是「不能从公开网重新长出来」的核心权重，**本包故意不含**：

| 说明 | 原路径（本机） | 约大小 |
|------|----------------|--------|
| Value@3000 | `/root/rlinf_recap/RLinf/logs/value_sft/recap_value_model_sft-20260921-10:01:14/value_sft/checkpoints/global_step_3000/` | 数 G |
| CFG@6000 | `/root/rlinf_recap/RLinf/logs/cfg_rl/cfg_rl_openpi-20260929-00:58:47/cfg_sft/checkpoints/global_step_6000/` | ~32G 目录 |
| CFG@9000 | 同上 `.../global_step_9000/` | 类似 |
| Deploy 6000/9000 safetensors | `/root/rlinf_recap/deploy/cfg_rl_step{6000,9000}/model.safetensors` | 各 ~7.8G |
| CFG@30000 训练检查点 | `/root/rlinf_recap/RLinf/logs/cfg_rl/cfg_rl_openpi-20261005-07:56:43/cfg_sft/checkpoints/global_step_30000/` | ~32G |
| Deploy 12000–30000 safetensors | `/root/rlinf_recap/deploy/cfg_rl_step{12000,15000,18000,21000,24000,27000,30000}/model.safetensors` | 各 ~7.8G |

**建议**：把上表 4 项拷到移动硬盘 / 对象存储 / 私人 HF dataset，路径写进私人笔记。
本仓库已包含评测数字、曲线图、TensorBoard、日志，可证明做过什么；但没有权重无法在新机器直接 rollout 同一策略。
