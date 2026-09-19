# -*- coding: utf-8 -*-
"""问题2 队友协议论文图（需先跑 run_noise_validity.py 与 run_kernel_nested.py）。

产出 output/figs/：
  figP1_noise_validity.png : 噪声有效性 N0/N05/N1 柱状 + NS 打乱噪声分布（真实噪声竟劣于随机噪声）
  figP2_feature_kernel.png : 四特征 × 五核 OOF mae_ord 分组柱状（放缩后核被激活 + geo 最优）
  figP3_kernel_significance.png : 非线性核相对 K1 的 Δmae 森林图（95%CI, 判定显著性）
"""
import os
import io
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config_p2 as C2                                        # noqa: E402

plt.rcParams.update({
    "font.family": ["SimSun"], "font.serif": ["SimSun"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "figure.dpi": 100, "savefig.dpi": 300, "axes.unicode_minus": False,
})
OUT = C2.OUTPUT_DIR
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)


def figP1():
    m = pd.read_csv(os.path.join(OUT, "noise_cv_metrics.csv"))
    shuf = pd.read_csv(os.path.join(OUT, "noise_shuffle_results.csv"))
    st = m[m["scope"] == "stratified-5fold-OOF"].set_index("scheme")
    ns_med = float(m.loc[m["scheme"] == "NS_shuffled", "mae_ord"].iloc[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    names = ["N0", "N05", "N1"]
    labels = [r"$\theta{=}0$ (N0)", r"$\theta{=}0.5$ (N05)", r"$\theta{=}1$ (N1)"]
    vals = [st.loc[k, "mae_ord"] for k in names]
    cols = ["#d62728", "#4c9bd1", "#9aa5b1"]
    ax1.bar(labels, vals, color=cols, width=0.6)
    ax1.axhline(ns_med, ls="--", c="#555555", lw=1)
    ax1.text(2.0, ns_med + 0.004, f"随机打乱噪声中位 = {ns_med:.3f}", ha="right", fontsize=8)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax1.set_ylabel("Ordinal MAE (OOF)")
    ax1.set_ylim(min(vals + [ns_med]) - 0.03, max(vals) + 0.05)
    ax1.text(0.02, 0.05, r"$\Delta_{10}$=MAE(N1)$-$MAE(N0)=+0.106",
             transform=ax1.transAxes, fontsize=8)
    ax1.text(0.02, -0.02, "95%CI=[+0.100,+0.112] 完全>0", transform=ax1.transAxes, fontsize=8)
    ax2.hist(shuf["mae_ord"], bins=12, color="#aab5bf", edgecolor="white")
    n1 = st.loc["N1", "mae_ord"]
    ax2.axvline(n1, color="#d62728", lw=1.6)
    ax2.axvline(ns_med, color="#555555", ls="--", lw=1)
    ax2.set_xlabel(r"随机打乱噪声 (NS) 的 Ordinal MAE")
    ax2.set_ylabel("频次 (30 种子)")
    ax2.text(n1, ax2.get_ylim()[1] * 0.9, " 真实噪声 N1", color="#d62728", fontsize=8, va="top")
    ax2.set_title("真实噪声劣于全部 30 次随机噪声", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP1_noise_validity.png"))
    plt.close(fig)


def figP2():
    c = pd.read_csv(os.path.join(OUT, "kernel_scaled_comparison.csv"))
    sel = pd.read_csv(os.path.join(OUT, "kernel_selection.csv"))
    feats = ["lam1", "lam2", "geo", "fro"]
    kerns = ["K1_lin", "K2_cap", "K3_eesm", "K4_pow", "K5_plus"]
    klab = ["K1", "K2", "K3", "K4", "K5"]
    adopted = {(r.feature, r.kernel): r.adopted for r in sel.itertuples()}
    x = np.arange(len(feats))
    w = 0.15
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    palette = ["#9aa5b1", "#dd8452", "#4c9bd1", "#55a868", "#b279a2"]
    for i, kn in enumerate(kerns):
        vals, edges = [], []
        for f in feats:
            row = c[(c.feature == f) & (c.kernel == kn)]
            vals.append(float(row["mae_ord"].iloc[0]))
            edges.append(bool(adopted.get((f, kn), False)))
        bars = ax.bar(x + (i - 2) * w, vals, w, color=palette[i], label=klab[i],
                      edgecolor=["#d62728" if e else "none" for e in edges],
                      linewidth=[1.6 if e else 0 for e in edges])
    ax.axhline(0.7560, ls=":", c="#888888", lw=1)
    ax.text(-0.45, 0.7755, "v2 $\\lambda_1$ 基线 0.756", fontsize=7.5, ha="left", color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\sqrt{\lambda_1\lambda_2}$", r"$\|H\|_F^2$"])
    ax.set_ylabel("Ordinal MAE (OOF, 越低越好)")
    ax.set_ylim(0.69, 0.79)
    ax.legend(ncol=5, fontsize=8, loc="upper center", frameon=False)
    ax.set_title("红框=§6.3 采纳核；$\\sqrt{\\lambda_1\\lambda_2}$ 最优，放缩激活非线性核", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP2_feature_kernel.png"))
    plt.close(fig)


def figP3():
    sel = pd.read_csv(os.path.join(OUT, "kernel_selection.csv"))
    nl = sel[sel["kernel"] != "K1_lin"].dropna(subset=["delta"]).copy()
    nl = nl.sort_values(["feature", "kernel"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ypos = np.arange(len(nl))
    for i, r in enumerate(nl.itertuples()):
        col = "#d62728" if r.significant else "#9aa5b1"
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.6)
        ax.plot(r.delta, i, "o", color=col, ms=5)
    ax.axvline(-0.005, ls="--", c="#555555", lw=1)
    ax.axvline(0, ls=":", c="#999999", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.feature} · {r.kernel.replace('_lin','').replace('_cap','').replace('_eesm','').replace('_pow','').replace('_plus','')}"
                        for r in nl.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$=MAE(kernel)$-$MAE(K1)  （红=显著, 虚线=$-0.005$ 门槛）")
    ax.set_title("非线性核相对线性核 K1 的配对改进（95% Bootstrap CI）", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figP3_kernel_significance.png"))
    plt.close(fig)


if __name__ == "__main__":
    figP1()
    figP2()
    figP3()
    print("已生成 figP1/figP2/figP3 到 output/figs/")
