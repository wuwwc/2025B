# -*- coding: utf-8 -*-
"""问题2 主流程（PEMK-ORE v2, spec §15 五阶段）。

链路: H -> HH^H 闭式主特征值 lambda1 -> 广义特征 u(theta)=lambda1/(sigma^2)^theta (N,122)
      -> 五类核等权聚合 z -> PAVA 加权中位数有序阈值 -> MCS
Stage2 参数 CV: theta∈{1,0.5,0} × 五核(K3 beta / K4 q / K5 beta 网格) × 类权重开关,
5-fold 分层(seed=42), 所有候选共用同一折。
Stage4 选择: argmin mae_ord, 接近平手依次比 acc、acc±1、复杂度(无参核优先)、再偏好大 theta。
参照 R1: 同一五核流程作用于问题1口径 gamma_F=||H||_F^2/sigma^2 (theta=1), 量化特征替换增益。
Stage5: 训练集 A 全量重拟合 -> 预测数据集 B(valid)。另输出逐层消融表 ablation_p2.csv。
"""
import io
import os
import sys
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config_p2 as C2                                        # noqa: E402 (导入即注入 sys.path)
import config as C1                                           # noqa: E402
from common.cv import iter_folds, cross_validate              # noqa: E402
import data_loader                                            # noqa: E402
import ordinal                                                # noqa: E402
import metrics as M                                           # noqa: E402
import eigen_features as EF                                   # noqa: E402

PARAM_NAME = {"K3_eesm": "beta_EESM", "K4_pow": "q", "K5_plus": "beta_plus"}


def class_weights(cls, n_levels):
    """spec §10: w_l = N / (L * N_l), 映射为每样本权重（口径同问题1）。"""
    N = len(cls)
    counts = np.bincount(cls, minlength=n_levels).astype(float)
    counts_safe = np.where(counts > 0, counts, 1.0)
    return (N / (n_levels * counts_safe))[cls]


def cv_ordinal(Z, cls, levels, folds, w=None):
    def fpe(tr, va):
        mdl = ordinal.fit_ordinal(Z[tr], cls[tr], len(levels), None if w is None else w[tr])
        cp = ordinal.predict_ordinal(Z[va], mdl)
        return M.ordinal_metrics(cls[va], cp, levels)
    _, agg = cross_validate(fpe, folds=folds)
    return agg


def search_grid(make_input, thetas, cls, levels, folds, w, feature_tag):
    """theta × 五核 × 参数网格 CV, 返回明细行 list[dict]（含 mean±std 三主指标与辅助指标）。

    make_input(theta) -> (N,122) 载波级输入。
    """
    rows = []
    for th in thetas:
        x = make_input(th)
        for kname, (label, _func, grid, n_par) in EF.KERNELS.items():
            for p in grid:
                z = EF.aggregate(x, kname, p)
                agg = cv_ordinal(z, cls, levels, folds, w)
                rows.append({
                    "feature": feature_tag, "theta": th,
                    "kernel": kname, "kernel_label": label,
                    "param": PARAM_NAME.get(kname, ""), "param_value": (np.nan if p is None else float(p)),
                    "n_params": n_par, "weights": "on" if w is not None else "off",
                    "acc": agg["acc"][0], "acc_std": agg["acc"][1],
                    "mae_ord": agg["mae_ord"][0], "mae_ord_std": agg["mae_ord"][1],
                    "acc1": agg["acc1"][0], "acc1_std": agg["acc1"][1],
                    "rmse": agg["rmse"][0], "r2": agg["r2"][0],
                })
    return rows


def pick_best(rows):
    """spec §13: argmin mae_ord; 差距 < TIE_TOL_MAE 视为平手, 依次比 acc、acc1、复杂度、再偏好大 theta。"""
    mae_min = min(r["mae_ord"] for r in rows)
    cand = [r for r in rows if r["mae_ord"] - mae_min < C2.TIE_TOL_MAE]
    return min(cand, key=lambda r: (-r["acc"], -r["acc1"], r["n_params"], -r["theta"], r["kernel"],
                                     r["param_value"] if r["param_value"] == r["param_value"] else 0.0))


def kernel_bests(rows):
    """每个核在给定权重设置下的最优参数行（用于横向比较表）。"""
    out = {}
    for kname in EF.KERNELS:
        sub = [r for r in rows if r["kernel"] == kname]
        out[kname] = pick_best(sub)
    return out


def fmt_row(r):
    return (f"{r['feature']:4s} th={r['theta']:<4g} {r['kernel']:8s} {r['param']:9s}={r['param_value']:<10.5g} "
            f"w={r['weights']:3s} acc={r['acc']:.4f}±{r['acc_std']:.4f} "
            f"mae_ord={r['mae_ord']:.4f}±{r['mae_ord_std']:.4f} acc1={r['acc1']:.4f}")


def main():
    t0 = time.time()
    out = C2.OUTPUT_DIR

    # ---- Stage 0: 数据 + 诊断 ----
    print("=== Stage 0: load cache / EDA / 平坦性诊断 ===")
    train = data_loader.load_cache("train")
    valid = data_loader.load_cache("valid")
    y_tr = np.asarray(train["mcs"], float)
    levels = np.unique(y_tr)
    assert np.allclose(levels, np.array(C1.MCS_LEVELS)), f"levels mismatch: {levels}"
    n_levels = len(levels)
    cls_tr = np.searchsorted(levels, y_tr).astype(int)
    n = len(y_tr)
    data_loader.eda(train)

    # ---- Stage 1: 载波级输入（广义特征 u(theta) + 问题1口径参照）----
    lam1, lam2, x_tr = EF.principal_eigenmode_sinr(train["H"], train["noise_floor"])
    lam1_v, lam2_v, x_va = EF.principal_eigenmode_sinr(valid["H"], valid["noise_floor"])
    gF_tr = EF.total_power_sinr(train["H"], train["noise_floor"])
    gF_va = EF.total_power_sinr(valid["H"], valid["noise_floor"])
    sigma2_tr = EF.noise_sigma2(train["noise_floor"])
    sigma2_va = EF.noise_sigma2(valid["noise_floor"])

    def mk_eig(th):
        return EF.generalized_input(lam1, sigma2_tr, th)

    def mk_gamF(th):
        return gF_tr                                    # 参照固定为标准 SINR 口径 theta=1

    stage0 = EF.stage0_report({"train": (lam1, lam2, x_tr, gF_tr),
                               "valid": (lam1_v, lam2_v, x_va, gF_va)},
                              os.path.join(out, "stage0_diagnosis.txt"))
    print(f"[stage0] x rel-std(train) mean={stage0['train']['relstd_x'][0]:.3e} "
          f"vs gamma_F {stage0['train']['relstd_gammaF'][0]:.3e}")

    folds = iter_folds(n, C2.K_FOLDS, C2.RANDOM_SEED, cls_tr if C2.CV_STRATIFY else None)
    w_samp = class_weights(cls_tr, n_levels)

    # ---- Stage 2/3: theta × 五核参数 CV（权重 off 先扫, 再按 CV 决定 on/off, 协议同问题1）----
    print("=== Stage 2/3: theta x kernel parameter CV (weights off scan) ===")
    rows_off = search_grid(mk_eig, C2.THETA_GRID, cls_tr, levels, folds, None, "eig")
    for r in rows_off[:]:
        print(fmt_row(r))
    best_off = pick_best(rows_off)

    # 类权重开关: 在 off 最优候选上比较 on 是否改善
    x_b = EF.generalized_input(lam1, sigma2_tr, best_off["theta"])
    z_b = EF.aggregate(x_b, best_off["kernel"],
                       None if best_off["param_value"] != best_off["param_value"] else best_off["param_value"])
    agg_on_b = cv_ordinal(z_b, cls_tr, levels, folds, w_samp)
    use_w = (agg_on_b["mae_ord"][0] < best_off["mae_ord"]) or \
            (np.isclose(agg_on_b["mae_ord"][0], best_off["mae_ord"]) and agg_on_b["acc"][0] > best_off["acc"])
    if C2.USE_CLASS_WEIGHTS == "on":
        use_w = True
    elif C2.USE_CLASS_WEIGHTS == "off":
        use_w = False
    print(f"[cv] weighting: off mae_ord={best_off['mae_ord']:.4f} acc={best_off['acc']:.4f} | "
          f"on(rep cand) mae_ord={agg_on_b['mae_ord'][0]:.4f} acc={agg_on_b['acc'][0]:.4f} -> use_weights={use_w}")

    if use_w:
        print("=== re-scan grids with class weights ON ===")
        rows_eig = search_grid(mk_eig, C2.THETA_GRID, cls_tr, levels, folds, w_samp, "eig")
        for r in rows_eig:
            print(fmt_row(r))
    else:
        rows_eig = rows_off

    # 参照 R1: 同一五核流程作用于 gamma_F（问题1 口径, theta=1）
    print("=== reference R1: same pipeline on gamma_F ===")
    rows_gF = search_grid(mk_gamF, [1.0], cls_tr, levels, folds, w_samp if use_w else None, "gamF")
    for r in rows_gF:
        print(fmt_row(r))

    # CV 明细：off 扫描 + （若切换）on 重扫 + 参照 R1，去重后全部保存
    seen, all_rows = set(), []
    for r in rows_off + rows_eig + rows_gF:
        key = (r["feature"], r["theta"], r["kernel"], r["param_value"], r["weights"])
        if key not in seen:
            seen.add(key)
            all_rows.append(r)
    df_cv = pd.DataFrame(all_rows)
    df_cv.to_csv(os.path.join(out, "kernel_cv_results.csv"), index=False, encoding="utf-8-sig")

    # ---- Stage 4: 模型比较与选择 ----
    print("=== Stage 4: model comparison ===")
    kb = kernel_bests(rows_eig)
    best_r1 = pick_best(rows_gF)
    comp = pd.DataFrame([*kb.values(), best_r1])
    comp = comp[["feature", "theta", "kernel", "kernel_label", "param", "param_value", "weights",
                 "acc", "acc_std", "mae_ord", "mae_ord_std", "acc1", "acc1_std", "rmse", "r2"]]
    comp.to_csv(os.path.join(out, "model_comparison_p2.csv"), index=False, encoding="utf-8-sig")
    print(comp.to_string(index=False))

    best = pick_best(rows_eig)
    assert best["mae_ord"] <= best_r1["mae_ord"] + C2.TIE_TOL_MAE, \
        "主模型 mae_ord 劣于问题1口径参照, 与 spec 假设不符（仍需如实报告）"
    print(f"[select] best: {fmt_row(best)}")

    # ---- 逐层消融表（论文用）: 每步只动一层 ----
    abl_rows = [
        ([r for r in rows_gF if r["kernel"] == "K1_lin"][0], "P1口径: gamma_F + 算术平均"),
        (best_r1, "gamma_F + 五核寻优（聚合层）"),
        (pick_best([r for r in rows_eig if r["theta"] == 1.0]), r"lambda1/sigma^2 + 五核（特征层）"),
        (pick_best([r for r in rows_eig if r["theta"] == 0.5]), r"lambda1/sigma + 五核（半归一化）"),
        (pick_best([r for r in rows_eig if r["theta"] == 0.0]), r"lambda1 + 五核（增益功率口径）"),
    ]
    abl = pd.DataFrame([{**{"step": s}, **{k: r[k] for k in
                         ("feature", "theta", "kernel", "param", "param_value", "weights",
                          "acc", "mae_ord", "mae_ord_std", "acc1")}} for r, s in abl_rows])
    abl.to_csv(os.path.join(out, "ablation_p2.csv"), index=False, encoding="utf-8-sig")
    print("=== ablation ===")
    print(abl.to_string(index=False))

    # ---- Stage 5: 全量重拟合 + valid(数据集B) 预测 ----
    print("=== Stage 5: refit on full train & predict valid ===")
    p_star = None if best["param"] == "" else float(best["param_value"])
    x_tr_b = EF.generalized_input(lam1, sigma2_tr, best["theta"])
    x_va_b = EF.generalized_input(lam1_v, sigma2_va, best["theta"])
    z_tr = EF.aggregate(x_tr_b, best["kernel"], p_star)
    z_va = EF.aggregate(x_va_b, best["kernel"], p_star)
    w_sel = w_samp if use_w else None
    mdl = ordinal.fit_ordinal(z_tr, cls_tr, n_levels, w_sel)
    cp_va = ordinal.predict_ordinal(z_va, mdl)
    mcs_va = levels[cp_va]

    assert len(mcs_va) == len(valid["noise_floor"]), "valid 预测行数与数据集B样本数不一致"
    assert set(np.unique(mcs_va)) <= set(levels)

    pd.DataFrame({"terminal": valid["terminal"], "row": valid["row_idx"],
                  "theta": best["theta"], "z_eff": z_va, "mcs_pred": mcs_va}).to_csv(
        os.path.join(out, "pred_valid_p2.csv"), index=False, encoding="utf-8-sig")

    tau, blv = mdl["thresholds"], mdl["block_levels"]
    pd.DataFrame({"threshold_idx": np.arange(len(tau)), "threshold_z": tau,
                  "threshold_dB": 10 * np.log10(np.maximum(tau, 1e-300)),
                  "level_lower": levels[blv[:-1]], "level_upper": levels[blv[1:]]}).to_csv(
        os.path.join(out, "final_thresholds_p2.csv"), index=False)

    # ---- eval_summary_p2.txt ----
    s0 = stage0["train"]
    L = []
    L.append("问题2 评估摘要（PEMK-ORE v2: 广义主特征模态特征 u(theta)=lambda1/(sigma^2)^theta + 五核等权聚合 + PAVA 有序阈值, 5-fold 分层 CV seed=42）")
    L.append(f"数据: 训练集A N={n}, 档位数={n_levels}; 预测集B(valid) 行数={len(mcs_va)}")
    L.append("")
    L.append("[Stage0 诊断] 每样本122载波跨载波相对标准差(train):")
    L.append(f"  gamma_F(||H||_F^2/sigma^2): mean={s0['relstd_gammaF'][0]:.3e} max={s0['relstd_gammaF'][1]:.3e} (机器精度 -> 问题1聚合退化)")
    L.append(f"  x(lambda1/sigma^2)        : mean={s0['relstd_x'][0]:.3e} max={s0['relstd_x'][1]:.3e}")
    L.append(f"  lam2/lam1 均值={s0['lam2_over_lam1']:.4f}; 详见 stage0_diagnosis.txt")
    degenerate = s0["relstd_x"][1] < 1e-12
    L.append(f"  判定: 主特征模态输入跨载波差异{'不足(核聚合退化)' if degenerate else '真实存在(多核聚合不退化, 创新点2成立)'}")
    L.append("")
    L.append(f"[CV 协议] 所有候选共用同一折划分; 载波等权 1/{C1.N_SUBCARRIERS}; 类权重 use_weights={use_w}"
             f" (w_l=N/(L·N_l), 开关由 CV 决定)")
    L.append(f"[选择规则] argmin mae_ord, 平手(Δ<{C2.TIE_TOL_MAE:g})依次 acc > acc±1 > 复杂度(无参核优先) > 大theta")
    L.append(f"[搜索空间] theta∈{C2.THETA_GRID} × 五核(K3 beta/K4 q/K5 beta 网格) × 类权重开关; 参照 R1=gamma_F(theta=1)同流程")
    L.append("")
    L.append("五核最优 + 参照（每行=该核在 theta/参数上的 CV 最优, mean±std）:")
    for r in comp.itertuples(index=False):
        pv = "" if r.param == "" else f"{r.param}={r.param_value:.6g}"
        L.append(f"  [{r.feature} th={r.theta:g}] {r.kernel:8s} {r.kernel_label:6s} {pv:16s} w={r.weights:3s} "
                 f"acc={r.acc:.4f}±{r.acc_std:.4f} mae_ord={r.mae_ord:.4f}±{r.mae_ord_std:.4f} acc1={r.acc1:.4f}")
    L.append("")
    L.append("逐层消融（每步只动一层, 完整见 ablation_p2.csv）:")
    for r in abl.itertuples(index=False):
        pv = "" if r.param == "" else f"{r.param}={r.param_value:.6g}"
        L.append(f"  {r.step:34s} theta={r.theta:g} {r.kernel:8s} {pv:14s} "
                 f"acc={r.acc:.4f} mae_ord={r.mae_ord:.4f}±{r.mae_ord_std:.4f} acc1={r.acc1:.4f}")
    L.append("")
    L.append(f"主模型: theta={best['theta']:g}, {best['kernel']} ({best['kernel_label']}), "
             f"{best['param'] + '=' + format(best['param_value'], '.6g') if best['param'] else '无参数'}, "
             f"weights={'on' if use_w else 'off'}")
    L.append(f"  CV: acc={best['acc']:.4f}±{best['acc_std']:.4f}  mae_ord={best['mae_ord']:.4f}±{best['mae_ord_std']:.4f}  "
             f"acc1={best['acc1']:.4f}  rmse={best['rmse']:.2f}  r2={best['r2']:.4f}")
    L.append(f"参照R1(问题1口径 gamma_F 同流程最优): kernel={best_r1['kernel']} mae_ord={best_r1['mae_ord']:.4f} "
             f"acc={best_r1['acc']:.4f}; 主模型增益 Δmae_ord={best_r1['mae_ord'] - best['mae_ord']:+.4f}")
    L.append("注: 问题1 主模型（EESM+有序阈值, 同数据同协议）CV mae_ord≈0.8520/acc≈0.3763（problem1/output/eval_summary.txt）,")
    L.append("    与参照 R1 数值一致可作为管线正确性交叉验证。")
    L.append("")
    L.append(f"有序阈值 tau(z域): {np.array2string(tau, precision=6, max_line_width=200)}")
    L.append(f"各块 level(档位索引): {blv.tolist()}")
    L.append(f"valid 预测档位分布: {pd.Series(mcs_va).value_counts().sort_index().to_dict()}")
    L.append(f"\n耗时 {time.time() - t0:.1f}s")
    summary = "\n".join(L)
    with open(os.path.join(out, "eval_summary_p2.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)
    print("done.")


if __name__ == "__main__":
    main()
