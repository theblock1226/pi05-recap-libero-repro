"""LIBERO-10 Task 0 success: RLinf doc vs this step-30000 run."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# Published tutorial numbers are percentages, not a remeasured 50-trial run.
# This run: same step-30000 weights, Task 0 x 50.
bars = [
    ("文档\nfew-shot SFT", 48.8, None, "#b0b7c3"),
    ("文档\nRECAP", 66.5, None, "#d64545"),
    ("本次\n普通前向", 50.0, "25/50", "#4c6ef5"),
    ("本次\nCFG 正条件", 64.0, "32/50", "#2f9e44"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=160)
xs = [0, 1, 2.55, 3.55]
for x, (label, pct, frac, color) in zip(xs, bars):
    ax.bar(x, pct / 100.0, width=0.72, color=color, zorder=3)
    text = f"{pct:.1f}%" if frac is None else f"{pct:.0f}%  {frac}"
    ax.text(x, pct / 100.0 + 0.015, text, ha="center", va="bottom", fontsize=11)

ax.text(0.5, 0.90, "+17.7 个百分点", ha="center", va="center", fontsize=11, color="#495057")
ax.text(3.05, 0.90, "+14 个百分点", ha="center", va="center", fontsize=11, color="#495057")

ax.set_xticks(xs, [b[0] for b in bars], fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Task 0 成功率", fontsize=12)
ax.set_title("LIBERO-10 Task 0：文档结果与本次 30000 步复现", fontsize=14, pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.01,
    0.01,
    "文档：RLinf RECAP 教程公布的 Task 0 成功率，未在本机 50 次协议上重测。\n"
    "本次：同一份 cfg_rl_step30000 权重；普通前向不加 Advantage 后缀，CFG 提示为 “Advantage: positive”。",
    fontsize=8.5,
    color="#495057",
    va="bottom",
)
fig.tight_layout(rect=(0, 0.07, 1, 1))

out = "/root/rlinf_recap/logs/plots_eval/09_task0_doc_vs_cfg30000.png"
fig.savefig(out)
fig.savefig("/root/rlinf_recap/logs/plots_cfg30000/05_task0_success_doc_vs_ours.png")
print(out)
