# -*- coding: utf-8 -*-
"""问题2 第二阶段：θ=0 下尺度放缩 + 嵌套CV 五核比较 + 特征维度对比（队友协议 §4-§9）。

固定 θ=0（第一阶段判定 noise_floor 无效）。对每个载波级特征 base∈{lam1,lam2,geo=√(λ1λ2),fro=||H||²}：
  - 外层 5-fold（seed=42, 分层）报告 OOF 泛化性能；
  - 内层 4-fold 在 outer-train 内选 K3 β / K4 q / K5 β（每内折重算尺度 s_inner, 只用内层训练）；
  - 每外折尺度 s_f 只由该外折训练部分算, 应用于一维 u=base/s_f 后聚合；
  - 类权重固定 off（避免与核共变, §6.2）。
模型选择（§6.3）：非线性核须相对 K1 满足 Δmae<=-0.005 且配对Bootstrap 95%CI上界<0 且 >=4/5折更优,
  否则回落 K1；再跨特征取 OOF mae 最小者为最终模型。最后 §7 全量重拟合 + 数据集B一次性预测。
输出 §8.2 全部文件 + experiment_summary.md, 并执行 §9 自检。
"""
import io
import os
import sys
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config_p2 as C2                                        # noqa: E402
from common.cv import iter_folds                              # noqa: E402
import data_loader                                            # noqa: E402
import ordinal                                                # noqa: E402
import eigen_features as EF                                   # noqa: E402
import p2_protocol as P                                       # noqa: E402

OUT = C2.OUTPUT_DIR
FEATS = ["lam1", "lam2", "geo", "fro"]
KERNEL_ORDER = ["K1_lin", "K2_cap", "K3_eesm", "K4_pow", "K5_plus"]
NONLIN = ["K2_cap", "K3_eesm", "K4_pow", "K5_plus"]
N_INNER = 4
DELTA_THR = 0.005


def inner_select_param(base, kname, grid, outer_train, cls, levels, seed):
    """内层 n_inner-fold 选参：返回 (best_param, mean_curve over grid)。尺度只由内层训练算。"""
    inner = iter_folds(len(outer_train), N_INNER, seed, cls[outer_train])
    oof = {p: np.zeros(len(outer_train), int) for p in grid}
    for itr, iva in inner:
        gtr, gva = outer_train[itr], outer_train[iva]
        u = EF.scale_input(base, EF.fold_scale(base, gtr))
        for p in grid:
            m = ordinal.fit_ordinal(EF.aggregate(u[gtr], kname, p), cls[gtr], len(levels))
            oof[p][iva] = ordinal.predict_ordinal(EF.aggregate(u[gva], kname, p), m)
    yt = cls[outer_train]
    curve = np.array([float(np.mean(np.abs(oof[p] - yt))) for p in grid])
    return grid[int(np.argmin(curve))], curve


def outer_oof(base, kname, cls, levels, folds, seed):
    """外层 OOF：返回 (oof_pred, scales, chosen_params, per_fold_mae, curves)。"""
    grid = EF.KERNELS[kname][2]
    N = len(cls)
    oofp = np.zeros(N, int)
    scales, chosen, pfmae, curves = [], [], [], []
    for f, (tr, va) in enumerate(folds):
        s = EF.fold_scale(base, tr)
        scales.append(s)
        u = EF.scale_input(base, s)
        if len(grid) == 1:
            p = grid[0]
        else:
            p, curve = inner_select_param(base, kname, grid, tr, cls, levels, seed + 100 * f)
            chosen.append(p)
            curves.append(curve)
        m = ordinal.fit_ordinal(EF.aggregate(u[tr], kname, p), cls[tr], len(levels))
        oofp[va] = ordinal.predict_ordinal(EF.aggregate(u[va], kname, p), m)
        pfmae.append(float(np.mean(np.abs(oofp[va] - cls[va]))))
    return oofp, scales, chosen, pfmae, curves


def main():
    t0 = time.time()
    tr = data_loader.load_cache("train")
    y = np.asarray(tr["mcs"], float)
    levels = np.unique(y)
    cls = np.searchsorted(levels, y).astype(int)
    terminal = np.asarray(tr["terminal"])
    n = len(y)
    folds = iter_folds(n, C2.K_FOLDS, C2.RANDOM_SEED, cls if C2.CV_STRATIFY else None)
    term_idx = np.searchsorted(np.unique(terminal), terminal)
    strata = term_idx * len(levels) + cls
    feats = EF.carrier_features(tr["H"])

    comp_rows, scale_rows, inner_rows, oof_store, sens_store, res = [], [], [], {}, {}, {}
    for feat in FEATS:
        base = feats[feat]
        res[feat] = {}
        for kname in KERNEL_ORDER:
            oofp, scales, chosen, pfmae, curves = outer_oof(
                base, kname, cls, levels, folds, C2.RANDOM_SEED)
            m = P.M.ordinal_metrics(cls, oofp, levels)
            res[feat][kname] = dict(oof=oofp, pfmae=pfmae, chosen=chosen, curves=curves, m=m)
            oof_store[(feat, kname)] = oofp
            comp_rows.append({"feature": feat, "kernel": kname, "mae_ord": m["mae_ord"],
                              "acc": m["acc"], "acc1": m["acc1"], "rmse": m["rmse"], "r2": m["r2"],
                              "chosen_params": ",".join(f"{c:g}" for c in chosen),
                              "scale_mean": float(np.mean(scales))})
            for f, (s, pm) in enumerate(zip(scales, pfmae)):
                scale_rows.append({"feature": feat, "kernel": kname, "outer_fold": f,
                                   "s_f": s, "fold_mae_ord": pm})
            if curves:
                mean_curve = np.mean(np.array(curves), axis=0)
                grid = EF.KERNELS[kname][2]
                sens_store[(feat, kname)] = mean_curve
                for p, cv in zip(grid, mean_curve):
                    inner_rows.append({"feature": feat, "kernel": kname, "param": float(p),
                                       "inner_oof_mae": cv})
    pd.DataFrame(scale_rows).to_csv(os.path.join(OUT, "kernel_scaled_outer_cv.csv"),
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(scale_rows).drop_duplicates(["feature", "outer_fold"]).to_csv(
        os.path.join(OUT, "scale_by_fold.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_rows).to_csv(os.path.join(OUT, "kernel_scaled_inner_cv.csv"),
                                    index=False, encoding="utf-8-sig")
    # 参数敏感性即 inner_rows（feature,kernel,param,inner_oof_mae）
    pd.DataFrame(inner_rows).to_csv(os.path.join(OUT, "kernel_parameter_sensitivity.csv"),
                                    index=False, encoding="utf-8-sig")

    # ---------- §6.3 逐特征：非线性核 vs K1 配对显著性 ----------
    sel_rows = []
    feat_best = {}
    for feat in FEATS:
        k1 = res[feat]["K1_lin"]
        e1 = np.abs(k1["oof"] - cls)
        elig = []
        for kname in KERNEL_ORDER:
            r = res[feat][kname]
            row = {"feature": feat, "kernel": kname, "mae_ord": r["m"]["mae_ord"],
                   "acc": r["m"]["acc"], "acc1": r["m"]["acc1"]}
            if kname == "K1_lin":
                row.update(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, folds_better=np.nan,
                           significant=False, adopted=(kname == "K1_lin"))
            else:
                ek = np.abs(r["oof"] - cls)
                lo, hi, dmean, _ = P.paired_bootstrap_diff(ek, e1, strata, n_boot=2000,
                                                           seed=C2.RANDOM_SEED)
                fb = sum(1 for a, b in zip(r["pfmae"], k1["pfmae"]) if a < b)
                sig = (dmean <= -DELTA_THR) and (hi < 0) and (fb >= 4)
                row.update(delta=dmean, ci_lo=lo, ci_hi=hi, folds_better=fb, significant=sig)
                if sig:
                    elig.append((r["m"]["mae_ord"], kname))
            sel_rows.append(row)
        # 采纳核：显著非线性核中最优, 否则 K1
        if elig:
            adopted = min(elig)[1]
        else:
            adopted = "K1_lin"
        feat_best[feat] = adopted
        for row in sel_rows:
            if row["feature"] == feat:
                row["adopted"] = (row["kernel"] == adopted)
    comp_rows = comp_rows
    pd.DataFrame(comp_rows).to_csv(os.path.join(OUT, "kernel_scaled_comparison.csv"),
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "kernel_selection.csv"),
                                  index=False, encoding="utf-8-sig")

    print("=== 各特征采纳核（§6.3 显著性判据）===")
    cand = []
    for feat in FEATS:
        ad = feat_best[feat]
        m = res[feat][ad]["m"]
        cand.append((m["mae_ord"], feat, ad))
        print(f"  {feat:5s} -> {ad:8s} mae={m['mae_ord']:.4f} acc={m['acc']:.4f} acc1={m['acc1']:.4f}")
    cand.sort()
    final_mae, final_feat, final_kernel = cand[0]
    print(f"\n最终模型: feature={final_feat} kernel={final_kernel} OOF mae={final_mae:.4f}")

    # ---------- §7 全量重拟合 + 数据集 B 预测 ----------
    base_tr = feats[final_feat]
    grid = EF.KERNELS[final_kernel][2]
    if len(grid) == 1:
        final_param = grid[0]
    else:  # 全量 A 五折选最终参数
        final_param, _ = inner_select_param(base_tr, final_kernel, grid, np.arange(n),
                                            cls, levels, C2.RANDOM_SEED)
    s_A = EF.fold_scale(base_tr, np.arange(n))
    u_tr = EF.scale_input(base_tr, s_A)
    z_tr = EF.aggregate(u_tr, final_kernel, final_param)
    model = ordinal.fit_ordinal(z_tr, cls, len(levels))

    va = data_loader.load_cache("valid")
    base_va = EF.carrier_features(va["H"])[final_feat]
    z_va = EF.aggregate(EF.scale_input(base_va, s_A), final_kernel, final_param)
    pred_cls = ordinal.predict_ordinal(z_va, model)
    pred_df = pd.DataFrame({"terminal": va["terminal"], "row_idx": va["row_idx"],
                            "scale_s_A": s_A, "agg_feature": z_va, "pred_mcs": levels[pred_cls]})
    pred_df.to_csv(os.path.join(OUT, "pred_valid_p2_scaled.csv"), index=False, encoding="utf-8-sig")

    thr = model["thresholds"]
    bl = model["block_levels"]
    lower = np.concatenate([[-np.inf], thr])                 # 每个块 [lower, upper)
    upper = np.concatenate([thr, [np.inf]])
    pd.DataFrame({"block_index": np.arange(len(bl)), "cls_index": bl,
                  "mcs_level": levels[bl], "lower_u": lower, "upper_u": upper}).to_csv(
        os.path.join(OUT, "final_thresholds_p2_scaled.csv"), index=False, encoding="utf-8-sig")

    # ---------- §9 自检 ----------
    checks = {}
    checks["finite_features"] = bool(np.all(np.isfinite(base_tr)) and np.all(np.isfinite(u_tr)))
    checks["scale_positive"] = bool(s_A > 0)
    q1 = EF.aggregate(u_tr, "K4_pow", 1.0)
    checks["K4_q1_eq_K1"] = bool(np.allclose(q1, EF.aggregate(u_tr, "K1_lin", None), rtol=1e-9))
    uu = EF.scale_input(base_tr * 1e6, EF.fold_scale(base_tr * 1e6, np.arange(n)))
    checks["scale_invariance"] = bool(np.allclose(uu, u_tr, rtol=1e-9))
    bad = 0
    for kn in ("K3_eesm", "K5_plus"):
        for p in EF.KERNELS[kn][2]:
            if not np.all(np.isfinite(EF.aggregate(u_tr, kn, p))):
                bad += 1
    checks["no_overflow_grid"] = (bad == 0)
    checks["B_rows_match"] = bool(len(pred_df) == len(va["mcs"]))
    checks["pred_in_levels"] = bool(np.all(np.isin(levels[pred_cls], levels)))
    print("=== §9 自检 ===")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    all_ok = all(checks.values())

    # ---------- experiment_summary.md ----------
    with open(os.path.join(OUT, "experiment_summary.md"), "w", encoding="utf-8") as f:
        f.write("# 问题2 实验总结（队友协议：噪声有效性 + 尺度放缩 + 嵌套五核）\n\n")
        f.write("## 噪声有效性（第一阶段）\n- 判定 noise_floor 无效/口径不匹配，主模型固定 θ=0（详见 noise_validity_summary.md）。\n\n")
        f.write("## 尺度放缩 + 嵌套五核（第二阶段）\n")
        f.write(f"- 最终尺度 s_A = {s_A:.4e}；最终模型 feature={final_feat}, kernel={final_kernel}, "
                f"param={final_param if final_param is not None else 'N/A'}。\n")
        f.write(f"- 最终 OOF mae_ord = {final_mae:.4f}。\n\n")
        f.write("## 各特征 OOF 指标（采纳核）\n\n| feature | adopted kernel | mae_ord | acc | acc±1 |\n|---|---|---|---|---|\n")
        for feat in FEATS:
            ad = feat_best[feat]
            m = res[feat][ad]["m"]
            f.write(f"| {feat} | {ad} | {m['mae_ord']:.4f} | {m['acc']:.4f} | {m['acc1']:.4f} |\n")
        f.write("\n## 自检（§9）\n")
        for k, v in checks.items():
            f.write(f"- {'OK' if v else 'FAIL'} {k}\n")
        f.write(f"\n全部自检通过：{all_ok}\n")

    # OOF 预测宽表
    oof_df = pd.DataFrame({"terminal": terminal, "cls_true": cls})
    for (feat, kname), op in oof_store.items():
        oof_df[f"{feat}_{kname}"] = op
    oof_df.to_csv(os.path.join(OUT, "kernel_scaled_oof_predictions.csv"),
                  index=False, encoding="utf-8-sig")

    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
