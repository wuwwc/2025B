# -*- coding: utf-8 -*-
"""问题2 特征方案对比（探索性, 不改动 run_problem2 的正式产出）。

背景：特征探针显示 λ2、sqrt(λ1λ2)=|detH| 对 MCS 的排序能力优于 λ1, 且控制 λ1 后偏相关~0.42
（存在正交增量）。本脚本量化两条升级路线能否把 mae_ord 压到现主模型(λ1,θ=0)的 0.7560 以下：

  方案①  单特征替换：载波级特征 base∈{λ1, λ2, sqrt(λ1λ2), ||H||_F^2}, 各扫 θ×五核×PAVA(同主流程)。
  方案②  二维有序模型：成对特征 (f1,f2) 先跨载波均值得 per-sample, 秩标准化后 z=w·r1+(1-w)·r2,
          扫 w∈[0,1], 再进 PAVA 有序阈值（保留可解释性, 用满两个空间模态）。

结果打印对比表并写 output/feature_comparison.csv。协议与主流程一致：5-fold 分层 seed=42, 共用同一折。
"""
import io
import os
import sys
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config_p2 as C2                                        # noqa: E402
import config as C1                                           # noqa: E402
from common.cv import iter_folds, cross_validate              # noqa: E402
import data_loader                                            # noqa: E402
import ordinal                                                # noqa: E402
import metrics as M                                           # noqa: E402
import eigen_features as EF                                   # noqa: E402


def cv_ordinal(Z, cls, levels, folds, w=None):
    def fpe(tr, va):
        mdl = ordinal.fit_ordinal(Z[tr], cls[tr], len(levels), None if w is None else w[tr])
        return M.ordinal_metrics(cls[va], ordinal.predict_ordinal(Z[va], mdl), levels)
    _, agg = cross_validate(fpe, folds=folds)
    return agg


def rank_norm(v):
    r = pd.Series(v).rank(method="average").to_numpy(float)
    return (r - 0.5) / len(r)


def search_single(base, sigma2, cls, levels, folds):
    """方案①：给定载波级特征 base(N,K) 扫 θ×五核, 返回 CV 明细行。"""
    rows = []
    for th in C2.THETA_GRID:
        x = EF.generalized_input(base, sigma2, th)
        for kname, (label, _f, grid, _np_) in EF.KERNELS.items():
            for p in grid:
                z = EF.aggregate(x, kname, p)
                agg = cv_ordinal(z, cls, levels, folds)
                rows.append({"theta": th, "kernel": kname,
                             "param": ("" if p is None else float(p)),
                             "mae_ord": agg["mae_ord"][0], "mae_ord_std": agg["mae_ord"][1],
                             "acc": agg["acc"][0], "acc1": agg["acc1"][0]})
    return rows


def best_row(rows):
    return min(rows, key=lambda r: r["mae_ord"])


def search_2d(f1, f2, cls, levels, folds, ws=None):
    """方案②：per-sample 双特征秩标准化线性组合, 扫权重 w, 返回最优 w 的 CV 指标。"""
    r1, r2 = rank_norm(f1), rank_norm(f2)
    ws = np.linspace(0, 1, 21) if ws is None else ws
    best = None
    for w in ws:
        z = w * r1 + (1.0 - w) * r2
        agg = cv_ordinal(z, cls, levels, folds)
        m = agg["mae_ord"][0]
        if best is None or m < best["mae_ord"]:
            best = {"w": w, "mae_ord": m, "mae_ord_std": agg["mae_ord"][1],
                    "acc": agg["acc"][0], "acc1": agg["acc1"][0]}
    return best


def main():
    t0 = time.time()
    out = C2.OUTPUT_DIR
    train = data_loader.load_cache("train")
    y = np.asarray(train["mcs"], float)
    levels = np.unique(y)
    cls = np.searchsorted(levels, y).astype(int)
    n = len(y)
    folds = iter_folds(n, C2.K_FOLDS, C2.RANDOM_SEED, cls if C2.CV_STRATIFY else None)

    lam1, lam2, _x = EF.principal_eigenmode_sinr(train["H"], train["noise_floor"])
    sigma2 = EF.noise_sigma2(train["noise_floor"])
    geo = np.sqrt(lam1 * lam2)                                # 逐载波 |detH| 型几何平均
    fro = lam1 + lam2                                         # ||H||_F^2

    print("=== 方案①: 单特征替换 (θ×五核×PAVA) ===")
    bases = {"lam1(现主模型)": lam1, "lam2": lam2, "sqrt(lam1*lam2)": geo, "||H||_F^2": fro}
    single_rows = []
    for name, base in bases.items():
        rows = search_single(base, sigma2, cls, levels, folds)
        b = best_row(rows)
        rec = {"scheme": "①单特征", "name": name, "theta": b["theta"],
               "detail": f"{b['kernel']}" + (f" p={b['param']:.4g}" if b["param"] != "" else ""),
               "mae_ord": b["mae_ord"], "mae_ord_std": b["mae_ord_std"],
               "acc": b["acc"], "acc1": b["acc1"]}
        single_rows.append(rec)
        print(f"  {name:18s} θ={b['theta']:<4g} {rec['detail']:22s} "
              f"mae_ord={b['mae_ord']:.4f}±{b['mae_ord_std']:.4f} acc={b['acc']:.4f} acc1={b['acc1']:.4f}")

    print("=== 方案②: 二维有序模型 (per-sample 秩组合×PAVA) ===")
    a1, a2, ag, af = lam1.mean(1), lam2.mean(1), geo.mean(1), fro.mean(1)
    ratio = a2 / np.maximum(a1, 1e-300)                       # 逐样本条件性 λ2/λ1
    pairs = {"(lam1, lam2)": (a1, a2), "(lam1, sqrt)": (a1, ag), "(sqrt, ratio)": (ag, ratio),
             "(fro, ratio)": (af, ratio), "(lam1, ratio)": (a1, ratio), "(fro, lam2)": (af, a2)}
    two_rows = []
    for name, (f1, f2) in pairs.items():
        b = search_2d(f1, f2, cls, levels, folds)
        rec = {"scheme": "②二维", "name": name, "theta": 0.0, "detail": f"w={b['w']:.2f}",
               "mae_ord": b["mae_ord"], "mae_ord_std": b["mae_ord_std"],
               "acc": b["acc"], "acc1": b["acc1"]}
        two_rows.append(rec)
        print(f"  {name:16s} {rec['detail']:10s} "
              f"mae_ord={b['mae_ord']:.4f}±{b['mae_ord_std']:.4f} acc={b['acc']:.4f} acc1={b['acc1']:.4f}")

    df = pd.DataFrame(single_rows + two_rows).sort_values("mae_ord").reset_index(drop=True)
    df.to_csv(os.path.join(out, "feature_comparison.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 全部方案按 mae_ord 排序（越低越好, 现主模型基线 λ1=0.7560）===")
    print(df.to_string(index=False))
    top = df.iloc[0]
    print(f"\n最优: [{top.scheme}] {top.name} {top.detail} mae_ord={top.mae_ord:.4f} "
          f"(vs 基线 0.7560, Δ={top.mae_ord - 0.7560:+.4f})")
    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
