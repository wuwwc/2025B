# -*- coding: utf-8 -*-
"""问题2 论文图表生成器（一次性运行, 需先执行 run_problem2.py）。

产出（problem2/output/figs/）:
  fig1_cross_carrier_relstd.png : 主特征模态特征 vs gamma_F 跨载波相对标准差（核不退化证据）
  fig2_kernel_cv_curves.png     : 三个参数化核 × 三个 theta 的 CV mae_ord-beta/q 曲线
  fig3_model_comparison.png     : 五核(theta 寻优) + 问题1口径参照 的三项序数指标对照柱状图
  fig4_ablation.png             : 逐层消融（聚合层/特征层/归一化强度 theta）mae_ord 阶梯图

复用 run_problem2 的 CV 明细 CSV, 保证与主流程数字一致。
"""
import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config_p2 as C2                                        # noqa: E402
import config as C1                                           # noqa: E402
import data_loader                                            # noqa: E402
import eigen_features as EF                                   # noqa: E402

# 科研论文风格：SimSun 单字体 + STIX 数学字体（matplotlib 3.7.2 Windows 下 CJK 回退失效, 见问题1同款配置说明）
plt.rcParams.update({
    "font.family": ["SimSun"],
    "font.serif": ["SimSun"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})

FIG_DIR = os.path.join(C2.OUTPUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def fig1():
    """每样本 122 载波 rel-std 分布: u(theta=1) vs gamma_F（对数坐标, 创新点2证据）。"""
    train = data_loader.load_cache("train")
    _, _, x = EF.principal_eigenmode_sinr(train["H"], train["noise_floor"])
    gF = EF.total_power_sinr(train["H"], train["noise_floor"])
    rx = EF.rel_std_per_sample(x)
    rf = EF.rel_std_per_sample(gF)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    lo = min(np.log10(np.maximum(rf.min(), 1e-17)), np.log10(np.maximum(rx.min(), 1e-17)))
    bins = np.linspace(min(lo, -16.5) - 0.5, 0.7, 80)      # 覆盖机器精度簇(~-15.6)与 lambda1 簇(~-0.9)
    ax.hist([np.log10(np.maximum(rf, 1e-17)), np.log10(np.maximum(rx, 1e-17))],
            bins=bins, label=[f"$\\gamma_F=\\|H\\|_F^2/\\sigma^2$（问题1口径）",
                               "$u(\\theta{=}1)=\\lambda_1/\\sigma^2$（主特征模态 SINR）"],
            color=["#9aa5b1", "#d62728"])
    ax.set_xlabel("每样本跨载波相对标准差  $\\log_{10}(\\mathrm{std}/\\mathrm{mean})$")
    ax.set_ylabel("样本数")
    ax.axvline(-12, color="k", ls=":", lw=1.0)
    ax.text(-11.6, ax.get_ylim()[1] * 0.55, "机器精度分界 $10^{-12}$", rotation=90, fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig1_cross_carrier_relstd.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig2():
    """K3/K4/K5 的 CV mae_ord 参数曲线, 每个核按 theta 分线型（主流程选定的权重设置）。"""
    df = pd.read_csv(os.path.join(C2.OUTPUT_DIR, "kernel_cv_results.csv"))
    comp = pd.read_csv(os.path.join(C2.OUTPUT_DIR, "model_comparison_p2.csv"))
    weights = comp[comp["feature"] == "eig"]["weights"].iloc[0]
    styles = {"K3_eesm": ("EESM指数核 $\\beta$", "#d62728", "o"),
              "K4_pow":  ("幂函数核 $q$", "#1f77b4", "s"),
              "K5_plus": ("正指数核 $\\beta$", "#2ca02c", "^")}
    lsys = {1.0: ":", 0.5: "--", 0.0: "-"}          # theta 越小线越实
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for kname, (lab, color, mk) in styles.items():
        for th in (1.0, 0.5, 0.0):
            sub = df[(df["feature"] == "eig") & (df["kernel"] == kname)
                     & (df["theta"] == th) & (df["weights"] == weights)]
            if kname == "K4_pow":
                xs = sub["param_value"].to_numpy(float)
                ax.plot(xs, sub["mae_ord"], marker=mk, color=color, ls=lsys[th],
                        label=f"{lab},  $\\theta={th:g}$")
                ax.set_xscale("symlog", linthresh=0.5)
                ax.set_xticks([-2, -1, -0.5, 1, 2])
                ax.set_xticklabels(["$-2$", "$-1$", "$-0.5$", "$1$", "$2$"])  # ASCII 减号, 避免 U+2212 缺字形
            else:
                xs = np.log10(sub["param_value"].to_numpy(float))
                ax.plot(xs, sub["mae_ord"], marker=mk, color=color, ls=lsys[th],
                        label=f"{lab},  $\\theta={th:g}$")
            ax.errorbar(xs, sub["mae_ord"], yerr=sub["mae_ord_std"], ls="none",
                        capsize=2, color=color, alpha=0.35)
    ax.set_xlabel("核参数（K3/K5: $\\log_{10}\\beta$;  K4: $q$, 对称对数轴）")
    ax.set_ylabel("CV Ordinal MAE")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig2_kernel_cv_curves.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig3():
    """五核最优(theta 寻优) + 参照(gamma_F) 三指标对照柱状图。"""
    comp = pd.read_csv(os.path.join(C2.OUTPUT_DIR, "model_comparison_p2.csv"))
    names = [f"{r.kernel.split('_')[1]} $\\theta$={r.theta:g}" for r in comp.itertuples()]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
    specs = [("acc", "Accuracy ↑", "#1f77b4"), ("mae_ord", "Ordinal MAE ↓（主指标）", "#d62728"),
             ("acc1", "Acc$_{\\pm1}$ ↑", "#2ca02c")]
    x = np.arange(len(comp))
    for ax, (col, title, color) in zip(axes, specs):
        vals = comp[col].to_numpy(float)
        errs = comp[f"{col}_std"].to_numpy(float) if f"{col}_std" in comp else np.zeros_like(vals)
        colors = [color] * (len(comp) - 1) + ["#9aa5b1"]     # 参照行灰色
        ax.bar(x, vals, 0.62, yerr=errs, capsize=2, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig3_model_comparison.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig4():
    """逐层消融: 每步只动一层, mae_ord 阶梯下降（读 ablation_p2.csv）。"""
    ab = pd.read_csv(os.path.join(C2.OUTPUT_DIR, "ablation_p2.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(ab))
    vals = ab["mae_ord"].to_numpy(float)
    errs = ab["mae_ord_std"].to_numpy(float)
    colors = ["#9aa5b1", "#9aa5b1", "#dd8452", "#4c9bd1", "#d62728"]
    ax.bar(x, vals, 0.62, yerr=errs, capsize=2, color=colors[: len(ab)])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("（", "\n（").replace(": ", ":\n") for s in ab["step"]], fontsize=7)
    ax.set_ylabel("CV Ordinal MAE ↓")
    ax.set_ylim(0.68, 0.92)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig4_ablation.png")
    fig.savefig(p); plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4):
        print(f"[fig] {fn()}")
    print("done.")
