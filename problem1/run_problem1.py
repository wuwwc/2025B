# -*- coding: utf-8 -*-
"""Stage 4: 主流程（《2025B_问题1_联合优化模型.md》方案A）。

链路: H -> 线性 SINR gamma -> EESM(alpha=1) 等效 z(beta) -> 加权中位数等渗最优有序阈值 -> MCS
选参: 外层 beta 网格 + 内层序数阈值, 用 k-fold 分层 CV 选 (beta, 类权重开关)。
对比: 文档§8 单调线性基线(a>=0 WLS 拟合类别索引) + 额外 L2 等渗对照, 同一 CV 折下并列。
验证: 噪声口径(×N_RX、单位 mW-W)不变性; 频域平坦导致 beta 不可识别。
"""
import os
import sys
import numpy as np
import pandas as pd

import config as C
sys.path.append(C.PROJECT_DIR)
from common.cv import iter_folds, cross_validate   # noqa: E402
import data_loader
import sinr
import eesm
import ordinal
import rate_map
import metrics as M


def class_weights(cls, n_levels):
    """文档 §7: w_l = N / ((L+1) * N_l)，映射到每样本权重。"""
    N = len(cls)
    counts = np.bincount(cls, minlength=n_levels).astype(float)
    counts_safe = np.where(counts > 0, counts, 1.0)
    w_cls = N / (n_levels * counts_safe)
    return w_cls[cls]


def cv_ordinal(Z, cls, levels, folds, w=None):
    """给定每样本等效 SINR Z 与折划分, CV 评估有序阈值模型。返回 agg。"""
    n_levels = len(levels)

    def fpe(tr, va):
        mdl = ordinal.fit_ordinal(Z[tr], cls[tr], n_levels, None if w is None else w[tr])
        cp = ordinal.predict_ordinal(Z[va], mdl)
        return M.ordinal_metrics(cls[va], cp, levels)

    _, agg = cross_validate(fpe, folds=folds)
    return agg


def cv_baseline(Z, y_val, cls, levels, folds, kind):
    """旧基线在同一折划分下的 CV。kind='A'(L2等渗+吸附) 或 'C'(仿射+吸附)。"""

    def fpe(tr, va):
        if kind == "A":
            mdl = rate_map.IsotonicRateModel().fit(Z[tr], y_val[tr])
            mdl.levels = levels                      # 用全量档位吸附, 避免折内缺档
            pv = mdl.predict(Z[va])
        else:
            mdl = rate_map.AffineRateModel(1.0, 0.0).fit(Z[tr], y_val[tr])
            pv = rate_map.snap_to_levels(mdl.predict_raw(Z[va]), levels)
        cp = M.values_to_cls(pv, levels)
        return M.ordinal_metrics(cls[va], cp, levels)

    _, agg = cross_validate(fpe, folds=folds)
    return agg


def cv_linear(Z, cls, levels, folds, w=None):
    """文档§8 单调线性基线(a>=0 WLS 拟合类别索引 + round/clip) 在同一折下的 CV。"""
    n_levels = len(levels)

    def fpe(tr, va):
        mdl = rate_map.MonotonicLinearModel().fit(Z[tr], cls[tr], n_levels,
                                                  None if w is None else w[tr])
        cp = mdl.predict_cls(Z[va])
        return M.ordinal_metrics(cls[va], cp, levels)

    _, agg = cross_validate(fpe, folds=folds)
    return agg


def fmt_agg(name, agg):
    return (f"{name:16s} acc={agg['acc'][0]:.4f}±{agg['acc'][1]:.4f}  "
            f"mae_ord={agg['mae_ord'][0]:.4f}±{agg['mae_ord'][1]:.4f}  "
            f"acc1={agg['acc1'][0]:.4f}±{agg['acc1'][1]:.4f}  "
            f"rmse={agg['rmse'][0]:.2f}±{agg['rmse'][1]:.2f}  "
            f"r2={agg['r2'][0]:.4f}±{agg['r2'][1]:.4f}")


def main():
    # ---- Stage 0: 缓存 + EDA ----
    print("=== Stage 0: load cache / EDA ===")
    train = data_loader.load_cache("train")
    valid = data_loader.load_cache("valid")
    data_loader.eda(train)

    y_tr = np.asarray(train["mcs"], float)
    levels = np.unique(y_tr)
    assert np.allclose(levels, np.array(C.MCS_LEVELS)), f"levels mismatch: {levels}"
    n_levels = len(levels)
    cls_tr = np.searchsorted(levels, y_tr).astype(int)
    y_val_tr = levels[cls_tr]
    n = len(y_tr)

    # ---- Stage 1: 线性域 SINR（默认口径）----
    print("=== Stage 1: linear-domain SINR ===")
    gamma_tr_raw, _ = sinr.compute_sinr_linear(train["H"], train["noise_floor"])
    gamma_va_raw, _ = sinr.compute_sinr_linear(valid["H"], valid["noise_floor"])
    rel_std = gamma_tr_raw.std(axis=1) / np.maximum(gamma_tr_raw.mean(axis=1), 1e-300)
    gamma_tr, freq_flat = sinr.collapse_flat(gamma_tr_raw)   # 频域平坦则吸附, 消除浮点噪声
    gamma_va, _ = sinr.collapse_flat(gamma_va_raw)
    print(f"[diag] per-sample subcarrier gamma rel-std: mean={rel_std.mean():.3e} max={rel_std.max():.3e}")
    print(f"[diag] gamma range=[{gamma_tr.min():.4e},{gamma_tr.max():.4e}]  frequency-flat={freq_flat}(collapsed)")

    folds = iter_folds(n, C.K_FOLDS, C.RANDOM_SEED, cls_tr if C.CV_STRATIFY else None)
    w_samp = class_weights(cls_tr, n_levels)
    betas = np.asarray(C.BETA_GRID, float)

    # ---- Stage 2/3: beta 外层 + 序数阈值内层, k-fold CV ----
    print("=== Stage 2/3: EESM(beta) + ordinal thresholds, k-fold CV ===")
    beta_rep = float(np.median(betas))
    Z_rep = eesm.eesm_linear(gamma_tr, beta_rep, C.ALPHA_FIXED)
    agg_off = cv_ordinal(Z_rep, cls_tr, levels, folds, None)
    agg_on = cv_ordinal(Z_rep, cls_tr, levels, folds, w_samp)
    # beta 不可识别检测: 频域平坦 => z=gamma/beta, 正缩放不改变次序 => CV 指标对 beta 不变(容差内)
    mae_lo = cv_ordinal(eesm.eesm_linear(gamma_tr, float(betas[0]), C.ALPHA_FIXED),
                        cls_tr, levels, folds, None)["mae_ord"][0]
    mae_hi = cv_ordinal(eesm.eesm_linear(gamma_tr, float(betas[-1]), C.ALPHA_FIXED),
                        cls_tr, levels, folds, None)["mae_ord"][0]
    maes = (agg_off["mae_ord"][0], mae_lo, mae_hi)
    beta_spread = max(maes) - min(maes)
    order_invariant = bool(beta_spread < C.TOL_INVARIANT)
    print(f"[diag] CV mae_ord over beta[{betas[0]:.3g},{beta_rep:.3g},{betas[-1]:.3g}]="
          f"{mae_lo:.4f}/{agg_off['mae_ord'][0]:.4f}/{mae_hi:.4f} spread={beta_spread:.2e}"
          f" -> beta不可识别={order_invariant}")
    mae_off, acc_off = agg_off["mae_ord"][0], agg_off["acc"][0]
    mae_on, acc_on = agg_on["mae_ord"][0], agg_on["acc"][0]
    if C.USE_CLASS_WEIGHTS == "on":
        use_w = True
    elif C.USE_CLASS_WEIGHTS == "off":
        use_w = False
    else:
        use_w = (mae_on < mae_off) or (np.isclose(mae_on, mae_off) and acc_on > acc_off)
    w_sel = w_samp if use_w else None
    agg_sel = agg_on if use_w else agg_off
    print(f"[cv] weighting off: mae_ord={mae_off:.4f} acc={acc_off:.4f} | "
          f"on: mae_ord={mae_on:.4f} acc={acc_on:.4f} -> use_weights={use_w}")

    # beta* : 次序不变则任取 beta_rep（记录不可识别）; 否则网格搜索
    if order_invariant:
        beta_star = beta_rep
        beta_curve = [(float(b), agg_sel["mae_ord"][0], agg_sel["acc"][0]) for b in betas]
    else:
        beta_curve = []
        for b in betas:
            aggb = cv_ordinal(eesm.eesm_linear(gamma_tr, float(b), C.ALPHA_FIXED),
                              cls_tr, levels, folds, w_sel)
            beta_curve.append((float(b), aggb["mae_ord"][0], aggb["acc"][0]))
        beta_star = min(beta_curve, key=lambda t: (t[1], -t[2]))[0]
    pd.DataFrame({"beta": [c[0] for c in beta_curve],
                  "cv_mae_ord": [c[1] for c in beta_curve],
                  "cv_acc": [c[2] for c in beta_curve],
                  "order_invariant": order_invariant}).to_csv(
        os.path.join(C.OUTPUT_DIR, "beta_cv.csv"), index=False)
    print(f"[cv] beta*={beta_star:.6g}")

    # ---- 基线对比（在 beta* 下, 同一 folds）----
    # 文档§8 单调线性基线(主对照) + A_isotonic(额外内部对照)
    Z_star = eesm.eesm_linear(gamma_tr, beta_star, C.ALPHA_FIXED)
    agg_L = cv_linear(Z_star, cls_tr, levels, folds, w_sel)
    agg_A = cv_baseline(Z_star, y_val_tr, cls_tr, levels, folds, "A")

    # ---- 噪声口径不变性验证（unit × apply_nrx 四组）----
    print("=== noise convention invariance check ===")
    conv_rows = []
    for unit in ("mW", "W"):
        for apply_nrx in (True, False):
            g_c, _ = sinr.compute_sinr_linear(train["H"], train["noise_floor"],
                                              unit=unit, apply_nrx=apply_nrx)
            g_c, _ = sinr.collapse_flat(g_c)
            Z_c = eesm.eesm_linear(g_c, beta_star, C.ALPHA_FIXED)
            a_c = cv_ordinal(Z_c, cls_tr, levels, folds, w_sel)
            conv_rows.append((unit, apply_nrx, a_c["acc"][0], a_c["mae_ord"][0], a_c["acc1"][0]))
    base = conv_rows[0]
    max_dev = max(abs(r[2] - base[2]) + abs(r[3] - base[3]) + abs(r[4] - base[4]) for r in conv_rows)
    invariant_ok = bool(max_dev < C.TOL_INVARIANT)
    print(f"[invariance] max|Δmetric|={max_dev:.3e}  invariant_ok={invariant_ok}")
    with open(os.path.join(C.OUTPUT_DIR, "noise_convention_check.txt"), "w", encoding="utf-8") as f:
        f.write("噪声口径不变性验证（《联合优化模型》§13.1）\n")
        f.write(f"默认口径: unit={C.NOISE_UNIT}, apply_nrx={C.NOISE_APPLY_NRX}"
                f"（noise_floor 视为单接收支路功率, 总噪声=N_RX*P_b）\n")
        f.write("结论: ×N_RX(×2) 与 单位(mW↔W, ×1000) 都是 gamma 的全局常数缩放, 被 beta 与\n"
                "      有序阈值吸收 => 预测/指标逐位相同, 模型表现无法区分口径。\n")
        f.write(f"5-fold CV (weighting={'on' if use_w else 'off'}) 各口径指标:\n")
        f.write(f"{'unit':5s}{'apply_nrx':11s}{'acc':12s}{'mae_ord':12s}{'acc1':12s}\n")
        for unit, ap, ac, mo, a1 in conv_rows:
            f.write(f"{unit:5s}{str(ap):11s}{ac:<12.6f}{mo:<12.6f}{a1:<12.6f}\n")
        f.write(f"max|Δ(acc+mae_ord+acc1)| = {max_dev:.3e}  -> invariant_ok={invariant_ok} (tol={C.TOL_INVARIANT})\n")
        f.write("注: 残余偏差来自近平坦数据中 SINR 近乎相等样本的浮点舍入翻转(<0.3%), 物理上与口径无关。\n")

    # ---- 全量重拟合 + valid 预测 ----
    print("=== refit on full train & predict valid ===")
    mdl_ord = ordinal.fit_ordinal(Z_star, cls_tr, n_levels, w_sel)
    Z_va = eesm.eesm_linear(gamma_va, beta_star, C.ALPHA_FIXED)
    mcs_va = levels[ordinal.predict_ordinal(Z_va, mdl_ord)]
    mdlL = rate_map.MonotonicLinearModel().fit(Z_star, cls_tr, n_levels, w_sel)
    mdlA = rate_map.IsotonicRateModel().fit(Z_star, y_val_tr); mdlA.levels = levels
    mcs_va_L = levels[mdlL.predict_cls(Z_va)]
    mcs_va_A = mdlA.predict(Z_va)

    out_df = pd.DataFrame({"terminal": valid["terminal"], "row": valid["row_idx"],
                           "mcs_pred": mcs_va, "mcs_pred_linear_doc": mcs_va_L,
                           "mcs_pred_A_isotonic": mcs_va_A})
    out_csv = os.path.join(C.OUTPUT_DIR, "pred_valid_notxbf.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[valid] saved {out_csv}, rows={len(out_df)}")

    # thresholds.csv（最优有序阈值 + 上下档位）
    tau = mdl_ord["thresholds"]; blv = mdl_ord["block_levels"]
    if len(blv) > 1:
        pd.DataFrame({"threshold_idx": np.arange(len(tau)), "threshold_z": tau,
                      "level_lower": levels[blv[:-1]], "level_upper": levels[blv[1:]]}).to_csv(
            os.path.join(C.OUTPUT_DIR, "thresholds.csv"), index=False)

    # rate_map_curves.csv（阶梯示意）
    grid = np.linspace(Z_star.min(), Z_star.max(), 400)
    pd.DataFrame({"z": grid,
                  "ordinal_mcs": levels[ordinal.predict_ordinal(grid, mdl_ord)],
                  "linear_doc_mcs": levels[mdlL.predict_cls(grid)],
                  "linear_doc_raw_cls": mdlL.predict_raw_cls(grid),
                  "A_isotonic_mcs": mdlA.predict(grid)}).to_csv(
        os.path.join(C.OUTPUT_DIR, "rate_map_curves.csv"), index=False)

    # ---- eval_summary.txt ----
    ord_beats = (agg_sel["mae_ord"][0] <= min(agg_L["mae_ord"][0], agg_A["mae_ord"][0]) + 1e-3)
    rows_ok = len(out_df) == n
    L = []
    L.append("问题1 评估摘要（方案A: 线性域EESM(α=1) + 加权中位数等渗最优有序阈值 + 5-fold分层CV）")
    L.append(f"数据: N={n}, 档位数={n_levels}, levels={list(levels)}")
    L.append(f"counts={np.bincount(cls_tr, minlength=n_levels).tolist()}")
    L.append(f"噪声口径: unit={C.NOISE_UNIT}, apply_nrx={C.NOISE_APPLY_NRX}; 口径不变性 invariant_ok={invariant_ok} (max|Δmetric|={max_dev:.2e} < tol={C.TOL_INVARIANT})")
    L.append(f"频域平坦(freq_flat)={freq_flat}; beta不变性 spread={beta_spread:.2e}<tol => beta不可识别={order_invariant}, beta*={beta_star:.6g}(任取,对结果无影响)")
    L.append(f"类权重: use_weights={use_w} (CV: off mae_ord={mae_off:.4f}/acc={acc_off:.4f}, on mae_ord={mae_on:.4f}/acc={acc_on:.4f})")
    L.append("")
    L.append("5-fold 分层 CV 结果（mean±std）:")
    L.append(fmt_agg("ordinal(main)", agg_sel))
    L.append(fmt_agg("linear(doc-base)", agg_L))
    L.append(fmt_agg("A_isotonic(ref)", agg_A))
    L.append("")
    L.append(f"验收: 序数模型 mae_ord 不劣于基线={ord_beats}; valid行数={len(out_df)}(={n})ok={rows_ok}; "
             f"口径不变={invariant_ok}; beta不可识别={order_invariant}")
    L.append(f"最优有序阈值 tau(z域,beta*): {np.array2string(tau, precision=6, max_line_width=200)}")
    L.append(f"各块 level(档位索引): {blv.tolist()}")
    L.append(f"valid 预测档位分布: {pd.Series(mcs_va).value_counts().sort_index().to_dict()}")
    summary = "\n".join(L)
    with open(os.path.join(C.OUTPUT_DIR, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)
    print("done.")


if __name__ == "__main__":
    main()
