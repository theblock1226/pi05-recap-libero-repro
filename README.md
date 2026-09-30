# π0.5 + RLinf RECAP（Libero Task0）复现证据包

本仓库是 **配置 / 脚本 / 训练日志 / TensorBoard / 评测结果 / 图表** 的完整归档（**不含模型权重与 82G 轨迹数据**）。
目标：离职或换机器后仍能对照实验结论，并按文档重新拉权重/数据复现。

## 包内有什么

| 路径 | 内容 |
|------|------|
| `rlinf_overlay/` | 相对 RLinf upstream 的 `examples/offline_rl` 全量拷贝 + `offline_rl.patch` + commit pin |
| `artifacts/logs/` | 评测脚本、plots 图集、summary.json、评测视频、step1–4 文本日志 |
| `artifacts/rlinf_train_logs/` | Value SFT / CFG 的 `run_*.log` + TensorBoard events（已去掉 `.pt`/`.distcp`） |
| `artifacts/deploy_templates/` | 导出目录的 `config.json` + `norm_stats.json`（无 safetensors） |
| `artifacts/evidence/` | 评测与指标 JSON 速查副本 |
| `artifacts/canvases/` | 复盘 Canvas |
| `meta/` | `recap_paths.yaml` 等元数据 |

## 包外需自行下载（换机复现）

1. **代码**：RLinf @ `rlinf_overlay/RLinf_PIN.txt` 所记 commit，再打上 `offline_rl.patch`
2. **数据**：Hugging Face `RLinf/RECAP-Libero10-Task0-48succ-Data`（约 82G）
3. **底模权重**：π0.5 PyTorch、SigLIP2-so400m、Gemma-3-270M（见原 `recap_paths.yaml` / 配置）
4. **本次 CFG 权重**（若要直接评测而非重训）：本机曾有
   - `.../cfg_sft/checkpoints/global_step_6000|9000/actor/model_state_dict/full_weights.pt`
   - 或 deploy：`model.safetensors`（各约 7.8G）——**未纳入本仓库**

## 关键结论（vanilla 仿真，无 CFG guidance）

- CFG 停于 **9000**（计划 30000）；保留 6000
- Task0×10：**6000 = 40%**，**9000 = 30%**
- 9000 × libero_10×10：overall **3%**（仅 Task0 非零）

详见 `artifacts/evidence/*.json` 与 `artifacts/logs/plots_*`。

## GitHub 说明

整包约数十 MB，单文件均远小于 100MB 限制，可直接 push。
权重请用 Git LFS / HF / 网盘另存，并在私人笔记记下原绝对路径。
