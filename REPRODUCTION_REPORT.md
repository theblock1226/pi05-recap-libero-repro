# π0.5 RLinf RECAP 复现报告（证据包内嵌）

## 结论摘要
- 四步管线（returns → value → advantages → CFG）在 Libero Task0 官方数据上跑通
- CFG 计划 30000，工程停于 **9000**；保留 **6000** 对照
- 仿真评测为 **vanilla π0.5 前向（无 CFG guidance）**
- Task0×10：6000 **40%** vs 9000 **30%**（同档噪声）
- 9000 × libero_10×10：overall **3%**，仅 Task0=30%，其余任务 0

## Value@3000（训练日志）
Spearman ≈ 0.76，MAE ≈ 0.096（见 `artifacts/evidence/value_metrics_midckpts.json` 与 `plots_value_midckpts/`）

## CFG@3000/6000/9000
loss ≈ 0.012 平台；见 `artifacts/evidence/cfg_metrics_3ckpts.json` 与 `plots_cfg_compare_3ckpts/`

## 图表目录
- `artifacts/logs/plots/` — returns / value targets
- `artifacts/logs/plots_value_midckpts/` — value 曲线
- `artifacts/logs/plots_cfg3000|6000|9000/` — 各 ckpt CFG 曲线
- `artifacts/logs/plots_cfg_compare_3ckpts/` — 三 ckpt 对比
- TensorBoard：`artifacts/rlinf_train_logs/**/tensorboard/`

## 评测 JSON
- `artifacts/evidence/eval_cfg6000_task0_x10_summary.json`
- `artifacts/evidence/eval_cfg9000_libero10_x10_summary.json`
- `artifacts/evidence/eval_cfg9000_smoke_summary.json`

## 未包含但必须另存的权重
见 `EXCLUDED_WEIGHTS_AND_DATA.md`。

## 评测图表（补全）
目录：`artifacts/logs/plots_eval/`
- `01_task0_success_6000_vs_9000.png`
- `02_task0_trial_heatmap_6000_vs_9000.png`
- `03_task0_running_success_rate.png`
- `04_cfg9000_libero10_per_task.png`
- `05_cfg9000_overall_mix.png`
- `06_recap_dashboard_train_and_eval.png`
- `metrics_eval.json` / `index.html`
