# -*- coding: utf-8 -*-
"""问题1 论文图表生成器（一次性运行）。

产出:
  图2-6 PNG  -> problem1/output/figures/
  表1-4 CSV  -> problem1/output/tables/
  问题1_图表.md -> 项目根目录（内联表格 + 引用图片 + 简要题目/简介）

复用 problem1 各模块，与 run_problem1.py 结果一致（同 seed / folds / 口径）。
"""
import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl

# Windows GBK 控制台无法打印部分 Unicode（如组合字符 β̂），统一将 stdout 包为 UTF-8。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))          # problem1/
sys.path.insert(0, HERE)
import config as C                                           # noqa: E402
sys.path.append(C.PROJECT_DIR)
from common.cv import iter_folds, cross_validate             # noqa: E402
import data_loader                                           # noqa: E402
import sinr                                                  # noqa: E402
import eesm                                                  # noqa: E402
import ordinal                                               # noqa: E402
import rate_map                                              # noqa: E402
import metrics as M                                          # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(C.OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(C.OUTPUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
RNG = np.random.default_rng(0)


# ---------------- CV 辅助（与 run_problem1 一致） ----------------
def class_weights(cls, n_levels):
    N = len(cls)
    counts = np.bincount(cls, minlength=n_levels).astype(float)
    counts_safe = np.where(counts > 0, counts, 1.0)
    return (N / (n_levels * counts_safe))[cls]


def cv_ordinal(Z, cls, levels, folds, w=None):
    n_levels = len(levels)

    def fpe(tr, va):
        mdl = ordinal.fit_ordinal(Z[tr], cls[tr], n_levels, None if w is None else w[tr])
        cp = ordinal.predict_ordinal(Z[va], mdl)
        return M.ordinal_metrics(cls[va], cp, levels)
    _, agg = cross_validate(fpe, folds=folds)
    return agg


def cv_linear(Z, cls, levels, folds, w=None):
    n_levels = len(levels)

    def fpe(tr, va):
        mdl = rate_map.MonotonicLinearModel().fit(Z[tr], cls[tr], n_levels,
                                                  None if w is None else w[tr])
        cp = mdl.predict_cls(Z[va])
        return M.ordinal_metrics(cls[va], cp, levels)
    _, agg = cross_validate(fpe, folds=folds)
    return agg


def cv_isotonic(Z, y_val, cls, levels, folds):
    def fpe(tr, va):
        mdl = rate_map.IsotonicRateModel().fit(Z[tr], y_val[tr])
        mdl.levels = levels
        cp = M.values_to_cls(mdl.predict(Z[va]), levels)
        return M.ordinal_metrics(cls[va], cp, levels)
    _, agg = cross_validate(fpe, folds=folds)
    return agg


def oof_predictions(Z, cls, n_levels, folds, w_sel):
    """返回有序阈值主模型与单调线性基线的折外(OOF)类别预测。"""
    n = len(cls)
    oof_ord = np.zeros(n, int)
    oof_lin = np.zeros(n, int)
    for tr, va in folds:
        wt = None if w_sel is None else w_sel[tr]
        mo = ordinal.fit_ordinal(Z[tr], cls[tr], n_levels, wt)
        oof_ord[va] = ordinal.predict_ordinal(Z[va], mo)
        ml = rate_map.MonotonicLinearModel().fit(Z[tr], cls[tr], n_levels, wt)
        oof_lin[va] = ml.predict_cls(Z[va])
    return oof_ord, oof_lin


def read_valid_csi_time():
    """按与缓存相同的顺序/过滤读取 valid 的 csi_time（轻量, 只取两列）。"""
    paths = data_loader._list_xlsx("valid")
    csi, term, rowi = [], [], []
    for p in paths:
        wb = openpyxl.load_workbook(p, read_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        hdr = list(next(it))
        ci = hdr.index("csi_time")
        bi = hdr.index("beamforming_en")
        t = os.path.basename(os.path.dirname(os.path.dirname(p)))
        r = 0
        for row in it:
            if int(row[bi]) == 0:
                csi.append(row[ci]); term.append(t); rowi.append(r); r += 1
        wb.close()
        print(f"[csi_time] {t}: {r} rows")
    return np.array(csi, dtype=object), np.array(term), np.array(rowi, int)


# ---------------- 主计算 ----------------
def compute():
    print("=== load cache ===")
    train = data_loader.load_cache("train")
    valid = data_loader.load_cache("valid")
    y_tr = np.asarray(train["mcs"], float)
    levels = np.unique(y_tr)
    n_levels = len(levels)
    cls_tr = np.searchsorted(levels, y_tr).astype(int)
    y_val_tr = levels[cls_tr]
    n = len(y_tr)

    gamma_tr = sinr.collapse_flat(sinr.compute_sinr_linear(train["H"], train["noise_floor"])[0])[0]
    gamma_va = sinr.collapse_flat(sinr.compute_sinr_linear(valid["H"], valid["noise_floor"])[0])[0]
    folds = iter_folds(n, C.K_FOLDS, C.RANDOM_SEED, cls_tr if C.CV_STRATIFY else None)
    w_samp = class_weights(cls_tr, n_levels)
    betas = np.asarray(C.BETA_GRID, float)
    beta_rep = float(np.median(betas))

    # 类权重开关（mirror run）
    Zrep = eesm.eesm_linear(gamma_tr, beta_rep, C.ALPHA_FIXED)
    agg_off = cv_ordinal(Zrep, cls_tr, levels, folds, None)
    agg_on = cv_ordinal(Zrep, cls_tr, levels, folds, w_samp)
    use_w = (agg_on["mae_ord"][0] < agg_off["mae_ord"][0]) or \
            (np.isclose(agg_on["mae_ord"][0], agg_off["mae_ord"][0]) and agg_on["acc"][0] > agg_off["acc"][0])
    w_sel = w_samp if use_w else None
    print(f"[cv] use_weights={use_w}")

    # beta 不可识别 -> beta* 任取 median
    mae_lo = cv_ordinal(eesm.eesm_linear(gamma_tr, float(betas[0]), C.ALPHA_FIXED), cls_tr, levels, folds, None)["mae_ord"][0]
    mae_hi = cv_ordinal(eesm.eesm_linear(gamma_tr, float(betas[-1]), C.ALPHA_FIXED), cls_tr, levels, folds, None)["mae_ord"][0]
    spread = max(agg_off["mae_ord"][0], mae_lo, mae_hi) - min(agg_off["mae_ord"][0], mae_lo, mae_hi)
    order_invariant = bool(spread < C.TOL_INVARIANT)
    beta_star = beta_rep
    print(f"[beta] order_invariant={order_invariant} spread={spread:.2e} beta*={beta_star:g}")

    Z_star = eesm.eesm_linear(gamma_tr, beta_star, C.ALPHA_FIXED)
    Z_va = eesm.eesm_linear(gamma_va, beta_star, C.ALPHA_FIXED)

    # 全量重拟合三模型
    mdl_ord = ordinal.fit_ordinal(Z_star, cls_tr, n_levels, w_sel)
    mdl_lin = rate_map.MonotonicLinearModel().fit(Z_star, cls_tr, n_levels, w_sel)
    mdl_iso = rate_map.IsotonicRateModel().fit(Z_star, y_val_tr); mdl_iso.levels = levels

    # 表3: CV 性能（mean±std）
    agg_ord = cv_ordinal(Z_star, cls_tr, levels, folds, w_sel)
    agg_lin = cv_linear(Z_star, cls_tr, levels, folds, w_sel)
    agg_iso = cv_isotonic(Z_star, y_val_tr, cls_tr, levels, folds)

    # beta 曲线（图4）: 两模型 CV ordinal MAE 的 mean±std
    print("=== beta CV curve (both models) ===")
    brow = []
    for b in betas:
        Zb = eesm.eesm_linear(gamma_tr, float(b), C.ALPHA_FIXED)
        ao = cv_ordinal(Zb, cls_tr, levels, folds, w_sel)
        al = cv_linear(Zb, cls_tr, levels, folds, w_sel)
        brow.append((float(b), ao["mae_ord"][0], ao["mae_ord"][1], al["mae_ord"][0], al["mae_ord"][1]))

    # OOF 预测（图5/图6）
    oof_ord, oof_lin = oof_predictions(Z_star, cls_tr, n_levels, folds, w_sel)

    # valid 预测 + csi_time（表4）
    mcs_va = levels[ordinal.predict_ordinal(Z_va, mdl_ord)]
    print("=== read valid csi_time ===")
    csi, term, rowi = read_valid_csi_time()
    assert len(csi) == len(mcs_va) == len(valid["mcs"]), f"len mismatch {len(csi)} vs {len(mcs_va)}"
    assert np.array_equal(term, np.asarray(valid["terminal"])), "terminal order mismatch"

    return dict(levels=levels, n_levels=n_levels, cls_tr=cls_tr, y_val_tr=y_val_tr, n=n,
                Z_star=Z_star, Z_va=Z_va, beta_star=beta_star, order_invariant=order_invariant,
                use_w=use_w, mdl_ord=mdl_ord, mdl_lin=mdl_lin, mdl_iso=mdl_iso,
                agg_ord=agg_ord, agg_lin=agg_lin, agg_iso=agg_iso, brow=brow,
                oof_ord=oof_ord, oof_lin=oof_lin, mcs_va=mcs_va,
                valid=valid, csi=csi, term=term, rowi=rowi)


def db(z):
    return 10.0 * np.log10(np.maximum(np.asarray(z, float), 1e-300))


# ---------------- 图 ----------------
def fig2(d):
    levels, cls_tr, Z_star, n_levels, n = d["levels"], d["cls_tr"], d["Z_star"], d["n_levels"], d["n"]
    zdb = db(Z_star)
    data = [zdb[cls_tr == l] for l in range(n_levels)]
    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, positions=range(n_levels), widths=0.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe2ff")
    idx = RNG.choice(n, size=min(4000, n), replace=False)
    ax.scatter(np.asarray(cls_tr)[idx] + RNG.uniform(-0.16, 0.16, len(idx)), zdb[idx],
               s=3, alpha=0.22, color="#3b6fb6", zorder=0)
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels([f"{levels[l]:.1f}" for l in range(n_levels)], rotation=30)
    ax.set_xlabel("MCS 等级（mcs 数值）")
    ax.set_ylabel("等效 SINR  $z$  (dB)")
    ax.set_title("图2  不同 MCS 等级的等效 SINR 分布")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig2_sinr_by_mcs_boxplot.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig3(d):
    levels, cls_tr, Z_star, mdl_ord = d["levels"], d["cls_tr"], d["Z_star"], d["mdl_ord"]
    n = d["n"]
    zdb = db(Z_star)
    yv = levels[cls_tr]
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = RNG.choice(n, size=min(6000, n), replace=False)
    ax.scatter(zdb[sub], yv[sub] + RNG.uniform(-0.09, 0.09, len(sub)),
               s=4, alpha=0.18, color="#888888", label="训练样本 (z, mcs)")
    grid = np.linspace(Z_star.min(), Z_star.max(), 3000)
    q = levels[ordinal.predict_ordinal(grid, mdl_ord)]
    ax.step(db(grid), q, where="post", color="#d62728", lw=2.0, label="有序阈值模型  $Q_\\tau(z)$")
    tau = mdl_ord["thresholds"]
    for i, t in enumerate(tau):
        ax.axvline(db(t), color="#1f77b4", ls="--", lw=1.0, alpha=0.75,
                   label="阈值 $\\tau$" if i == 0 else None)
    ax.set_xlabel("等效 SINR  $z$  (dB)")
    ax.set_ylabel("MCS")
    ax.set_title("图3  等效 SINR—MCS 散点及最优有序阈值")
    ax.set_yticks(levels)
    ax.set_yticklabels([f"{l:.1f}" for l in levels])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig3_sinr_mcs_thresholds.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig4(d):
    brow, beta_star = d["brow"], d["beta_star"]
    lb = [np.log10(r[0]) for r in brow]
    om = [r[1] for r in brow]; osd = [r[2] for r in brow]
    lm = [r[3] for r in brow]; lsd = [r[4] for r in brow]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(lb, om, yerr=osd, marker="o", ms=3, capsize=2, color="#d62728", label="有序阈值主模型")
    ax.errorbar(lb, lm, yerr=lsd, marker="s", ms=3, capsize=2, color="#1f77b4", label="单调线性基线")
    ax.axvline(np.log10(beta_star), color="k", ls=":", lw=1.2, label=f"$\\hat\\beta$={beta_star:g}（任取）")
    ax.set_xlabel("$\\log_{10}\\beta$")
    ax.set_ylabel("CV Ordinal MAE")
    ax.set_title("图4  EESM 参数 β 的交叉验证损失曲线")
    ax.annotate("曲线水平 → 频域平坦下 β 不可识别\n（等效 SINR z=γ/β，正缩放不改变样本次序）",
                xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                fontsize=9, color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#e0c060", alpha=0.9))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig4_beta_cv_curve.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig5(d):
    cls_tr, oof_ord, oof_lin, n_levels = d["cls_tr"], d["oof_ord"], d["oof_lin"], d["n_levels"]
    e_ord = oof_ord - cls_tr
    e_lin = oof_lin - cls_tr
    lo = int(min(e_ord.min(), e_lin.min())); hi = int(max(e_ord.max(), e_lin.max()))
    lo = max(lo, -5); hi = min(hi, 5)
    vals = np.arange(lo, hi + 1)
    po = [np.mean(e_ord == v) for v in vals]
    pl = [np.mean(e_lin == v) for v in vals]
    x = np.arange(len(vals)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, po, w, color="#d62728", label="有序阈值主模型")
    ax.bar(x + w / 2, pl, w, color="#1f77b4", label="单调线性基线")
    ax.set_xticks(x); ax.set_xticklabels(vals)
    ax.set_xlabel("等级预测误差  $e=c(\\hat y)-c(y)$")
    ax.set_ylabel("样本比例")
    ax.set_title("图5  两种速率映射模型的 MCS 等级预测误差分布")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig5_error_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig6(d):
    levels, cls_tr, oof_ord, n_levels = d["levels"], d["cls_tr"], d["oof_ord"], d["n_levels"]
    cm = np.zeros((n_levels, n_levels))
    for t, pr in zip(cls_tr, oof_ord):
        cm[t, pr] += 1
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_levels)); ax.set_yticks(range(n_levels))
    ax.set_xticklabels([f"{levels[l]:.0f}" for l in range(n_levels)], rotation=45, ha="right")
    ax.set_yticklabels([f"{levels[l]:.0f}" for l in range(n_levels)])
    for i in range(n_levels):
        for j in range(n_levels):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xlabel("预测 MCS"); ax.set_ylabel("真实 MCS")
    ax.set_title("图6  有序阈值主模型归一化混淆矩阵（按真实类别行归一化）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig6_confusion_matrix.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------- 表 ----------------
def tables(d):
    levels, mdl_ord, mdl_lin = d["levels"], d["mdl_ord"], d["mdl_lin"]
    beta_star, n_levels = d["beta_star"], d["n_levels"]
    agg_ord, agg_lin, agg_iso = d["agg_ord"], d["agg_lin"], d["agg_iso"]

    # 表1 参数标定
    t1 = pd.DataFrame({
        "模型": ["单调线性基线", "有序阈值主模型"],
        "最优EESM参数 β̂": [f"{beta_star:g}（不可识别,任取）", f"{beta_star:g}（不可识别,任取）"],
        "第三层参数": [f"a={mdl_lin.a:.6e}, b={mdl_lin.b:.6f}", "τ 见表2"],
    })
    t1.to_csv(os.path.join(TAB_DIR, "table1_params.csv"), index=False, encoding="utf-8-sig")

    # 表2 有序阈值
    tau = mdl_ord["thresholds"]; blv = mdl_ord["block_levels"]
    rows = []
    for i, t in enumerate(tau):
        lo = levels[blv[i]]; hi = levels[blv[i + 1]]
        rows.append((f"τ_{i+1}", f"{lo:.1f} / {hi:.1f}", f"{t:.6e}", f"{10*np.log10(t):.4f}"))
    t2 = pd.DataFrame(rows, columns=["阈值", "相邻MCS等级 m_ℓ/m_ℓ+1", "线性尺度 τ", "dB (10log10τ)"])
    t2.to_csv(os.path.join(TAB_DIR, "table2_thresholds.csv"), index=False, encoding="utf-8-sig")

    # 表3 性能比较
    def row(name, agg):
        return {"模型": name,
                "Accuracy↑": f"{agg['acc'][0]:.4f}±{agg['acc'][1]:.4f}",
                "Ordinal MAE↓": f"{agg['mae_ord'][0]:.4f}±{agg['mae_ord'][1]:.4f}",
                "Acc±1↑": f"{agg['acc1'][0]:.4f}±{agg['acc1'][1]:.4f}"}
    t3 = pd.DataFrame([row("单调线性基线", agg_lin), row("有序阈值主模型", agg_ord),
                       row("L2等渗(额外对照)", agg_iso)])
    t3.to_csv(os.path.join(TAB_DIR, "table3_performance.csv"), index=False, encoding="utf-8-sig")

    # 表4 valid 预测
    nvalid = len(d["mcs_va"])
    t4 = pd.DataFrame({
        "样本编号": np.arange(nvalid),
        "sta(终端)": d["term"],
        "行号": d["rowi"],
        "csi_time": d["csi"],
        "SINR_eff(线性)": d["Z_va"],
        "SINR_eff(dB)": db(d["Z_va"]),
        "预测MCS": d["mcs_va"],
    })
    t4.to_csv(os.path.join(TAB_DIR, "table4_valid_predictions.csv"), index=False, encoding="utf-8-sig")
    return t1, t2, t3, t4


# ---------------- Markdown 文档 ----------------
def write_md(d, figs, t1, t2, t3, t4):
    def rel(p):
        return os.path.relpath(p, C.PROJECT_DIR).replace("\\", "/")

    def md_table(df):
        cols = list(df.columns)
        head = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(v) for v in rec) + " |" for rec in df.itertuples(index=False)]
        return "\n".join([head, sep] + body)

    agg_ord, agg_lin = d["agg_ord"], d["agg_lin"]
    prev4 = t4.head(10).copy()
    prev4["SINR_eff(线性)"] = prev4["SINR_eff(线性)"].map(lambda x: f"{x:.4e}")
    prev4["SINR_eff(dB)"] = prev4["SINR_eff(dB)"].map(lambda x: f"{x:.3f}")

    L = []
    L.append("# 问题1 图表汇总（关闭 TxBF · EESM + 有序阈值模型）\n")
    L.append("> 由 `problem1/make_figures.py` 自动生成。图片位于 `problem1/output/figures/`，")
    L.append("> 完整表格 CSV 位于 `problem1/output/tables/`。所有数值来自 5-fold 分层交叉验证（seed=42）。\n")
    L.append("> **关键前提**：数据频域平坦（每样本 122 载波 SINR 恒定到机器精度），等效 SINR 退化为 "
             "$z=\\gamma/\\beta$，故 **EESM 参数 β 不可识别**——图4 曲线水平、表1 的 β 为任取值，"
             "这是数据的固有性质而非拟合失败。\n")
    L.append("---\n")

    # 图
    L.append("## 一、图\n")
    L.append("### 图2  不同 MCS 等级的等效 SINR 分布\n")
    L.append(f"![图2]({rel(figs['fig2'])})\n")
    L.append("**简介**：按真实 MCS 分组的等效 SINR（dB）箱线图，叠加抖动散点。各等级中心位置总体右移、"
             "相邻等级部分重叠，说明等效 SINR 对 MCS 有显著且有序的区分能力，但非一一对应，需数据驱动标定阈值。\n")
    L.append("### 图3  等效 SINR—MCS 散点及最优有序阈值\n")
    L.append(f"![图3]({rel(figs['fig3'])})\n")
    L.append("**简介**：训练样本 $(z,\\text{mcs})$ 散点、最优阈值 $\\tau$（蓝色虚线）与主模型阶梯函数 $Q_\\tau(z)$（红色）。"
             "阈值间距明显不均，直观解释了为何简单线性模型不足以刻画 $z\\to\\text{MCS}$ 关系。\n")
    L.append("### 图4  EESM 参数 β 的交叉验证损失曲线\n")
    L.append(f"![图4]({rel(figs['fig4'])})\n")
    L.append("**简介**：主模型与线性基线的 CV Ordinal MAE 随 $\\log_{10}\\beta$ 变化（含各折标准差误差棒）。"
             "两条曲线均水平 → β 不可识别；主模型曲线整体低于基线，佐证其更优。\n")
    L.append("### 图5  两种模型的 MCS 等级预测误差分布（折外预测）\n")
    L.append(f"![图5]({rel(figs['fig5'])})\n")
    L.append("**简介**：等级误差 $e=c(\\hat y)-c(y)$ 的样本比例。主模型误差高度集中在 $e=0,\\pm1$，"
             "线性基线在 $e=0$ 处比例明显更低、大幅误判更多。\n")
    L.append("### 图6  有序阈值主模型归一化混淆矩阵（折外预测）\n")
    L.append(f"![图6]({rel(figs['fig6'])})\n")
    L.append("**简介**：按真实类别行归一化。质量集中于主对角线及相邻位置，说明误判主要是相邻速率等级间的局部错误，"
             "与 Ordinal MAE、Acc±1 的评价一致。\n")
    L.append("---\n")

    # 表
    L.append("## 二、表\n")
    L.append("### 表1  问题1 两类模型参数标定结果\n")
    L.append("**简介**：外层 β 与第三层参数的最优标定值。因频域平坦，β 不可识别（任取）；"
             "线性基线给出解析斜率/截距 $a,b$，主模型阈值见 表2。\n")
    L.append(md_table(t1) + "\n")
    L.append("### 表2  有序阈值模型最优阈值表\n")
    L.append("**简介**：主模型各相邻 MCS 等级间的等效 SINR 分界点 $\\tau_\\ell$（线性尺度与 dB 对照）。"
             "模型内部实际使用**线性尺度**阈值，dB 列仅供展示。\n")
    L.append(md_table(t2) + "\n")
    L.append("### 表3  主模型与基线性能比较（5-fold CV，mean±std）\n")
    L.append("**简介**：以 Accuracy、Ordinal MAE、Acc±1 三项序数指标比较。有序阈值主模型 Accuracy 最高、"
             "Ordinal MAE 最低，显著优于单调线性基线；L2 等渗为额外内部对照。\n")
    L.append(md_table(t3) + "\n")
    L.append(f"### 表4  验证集 B（关闭 TxBF）最终 MCS 预测结果（共 {len(t4)} 条，下为前 10 条预览）\n")
    L.append("**简介**：题目要求的最终预测答案。完整结果见 "
             f"`{rel(os.path.join(TAB_DIR, 'table4_valid_predictions.csv'))}`。"
             "注：原始数据无 `sta_mac` 列，STA 标识取自终端目录名；`SINR_eff` 为 β* 下的等效 SINR。\n")
    L.append(md_table(prev4) + "\n")
    L.append("---\n")
    L.append("## 三、结论摘要\n")
    L.append(f"- 主模型（EESM + 有序阈值）：Accuracy = **{agg_ord['acc'][0]:.4f}**，"
             f"Ordinal MAE = **{agg_ord['mae_ord'][0]:.4f}**，Acc±1 = **{agg_ord['acc1'][0]:.4f}**。\n")
    L.append(f"- 单调线性基线：Accuracy = {agg_lin['acc'][0]:.4f}，Ordinal MAE = {agg_lin['mae_ord'][0]:.4f}，"
             f"Acc±1 = {agg_lin['acc1'][0]:.4f}。主模型全面优于基线。\n")
    L.append("- 频域平坦导致 β 不可识别；模型对噪声口径（×N_RX、mW/W）不变，已在 "
             "`problem1/output/noise_convention_check.txt` 留证。\n")

    out = os.path.join(C.PROJECT_DIR, "问题1_图表.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[md] saved {out}")
    return out


def main():
    d = compute()
    print("=== figures ===")
    figs = {"fig2": fig2(d), "fig3": fig3(d), "fig4": fig4(d), "fig5": fig5(d), "fig6": fig6(d)}
    for k, v in figs.items():
        print(f"  {k}: {v}")
    print("=== tables ===")
    t1, t2, t3, t4 = tables(d)
    print(t1.to_string(index=False))
    print(t3.to_string(index=False))
    print(f"  table4 rows={len(t4)}")
    write_md(d, figs, t1, t2, t3, t4)
    print("done.")


if __name__ == "__main__":
    main()
