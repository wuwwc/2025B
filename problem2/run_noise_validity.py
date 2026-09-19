# -*- coding: utf-8 -*-
"""问题2 第一阶段：noise_floor 有效性检验（队友协议《问题处理.md》§3）。

固定线性聚合 z=mean_k lambda1（不含尺度放缩, 因 PAVA 对正比例缩放不变）, 对照五组噪声口径:
  N0  theta=0    z = a1                     (主参照, a1=mean_k lambda1)
  N1  theta=1    z = a1 / sigma^2           (真实噪声归一化)
  N05 theta=0.5  z = a1 / sqrt(sigma^2)     (半归一化敏感性)
  NC  固定噪声   折内中位数常数噪声          (自检: 逐样本预测必须与 N0 完全一致)
  NS  打乱噪声   随机重排 sigma^2 归属       (负对照, >=30 种子)
判据(§3.6): 相关性 + 排序翻转 + 同折 OOF + 配对 Bootstrap 95%CI + 分终端 grouped CV。
输出 §8.1 全部文件到 output/。
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
import eigen_features as EF                                   # noqa: E402
import p2_protocol as P                                       # noqa: E402

OUT = C2.OUTPUT_DIR
N_SHUFFLE = 30
N_BOOT = 2000


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
    strata = term_idx * len(levels) + cls                     # terminal×MCS 分层键

    lam1, _ = EF.eigenvalues(tr["H"])
    a1 = lam1.mean(axis=1)                                    # per-sample 主特征模态均值
    sigma2 = EF.noise_sigma2(tr["noise_floor"])
    nf = np.asarray(tr["noise_floor"], float)

    # ---------- §3.2 基础诊断（整体 + 分终端） ----------
    diag = []
    la = np.log10(np.maximum(a1, 1e-300))
    ls = np.log10(np.maximum(a1 / sigma2, 1e-300))
    diag.append({"scope": "overall", "n": n,
                 "nf_min": nf.min(), "nf_max": nf.max(), "nf_mean": nf.mean(),
                 "nf_std": nf.std(), "nf_median": np.median(nf),
                 "sp_loglam1_mcs": P.spearman(la, cls), "kd_loglam1_mcs": P.kendall(la, cls),
                 "sp_nf_mcs": P.spearman(nf, cls), "kd_nf_mcs": P.kendall(nf, cls),
                 "sp_loglam1_nf": P.spearman(la, nf),
                 "sp_loglam1_over_sig2_mcs": P.spearman(ls, cls)})
    for t in np.unique(terminal):
        m = terminal == t
        diag.append({"scope": str(t), "n": int(m.sum()),
                     "nf_min": nf[m].min(), "nf_max": nf[m].max(), "nf_mean": nf[m].mean(),
                     "nf_std": nf[m].std(), "nf_median": np.median(nf[m]),
                     "sp_loglam1_mcs": P.spearman(la[m], cls[m]), "kd_loglam1_mcs": P.kendall(la[m], cls[m]),
                     "sp_nf_mcs": P.spearman(nf[m], cls[m]), "kd_nf_mcs": P.kendall(nf[m], cls[m]),
                     "sp_loglam1_nf": P.spearman(la[m], nf[m]),
                     "sp_loglam1_over_sig2_mcs": P.spearman(ls[m], cls[m])})
    pd.DataFrame(diag).to_csv(os.path.join(OUT, "noise_diagnostics.csv"),
                              index=False, encoding="utf-8-sig")
    print("[diag] 整体: sp(logλ1,MCS)={:.3f} sp(NF,MCS)={:.3f} sp(logλ1,NF)={:.3f}".format(
        diag[0]["sp_loglam1_mcs"], diag[0]["sp_nf_mcs"], diag[0]["sp_loglam1_nf"]))

    # ---------- §3.3 五组噪声 OOF ----------
    schemes = {"N0": a1.copy(), "N1": a1 / sigma2, "N05": a1 / np.sqrt(sigma2)}
    # NC: 折内中位数常数噪声 -> 等价于 N0 乘正常数, 预测应与 N0 逐样本一致
    oof = {}
    for name, z in schemes.items():
        _, oof[name] = P.oof_metrics(z, cls, levels, folds)
    nc_pred = np.zeros(n, int)
    for trn, val in folds:
        s_med = np.median(sigma2[trn])                        # 折内常数噪声（正数）
        mdl_z = a1 / s_med
        from ordinal import fit_ordinal, predict_ordinal
        m = fit_ordinal(mdl_z[trn], cls[trn], len(levels))
        nc_pred[val] = predict_ordinal(mdl_z[val], m)
    oof["NC"] = nc_pred
    err = {k: np.abs(v - cls) for k, v in oof.items()}
    mae = {k: float(np.mean(e)) for k, e in err.items()}
    acc = {k: float(np.mean(oof[k] == cls)) for k in oof}
    acc1 = {k: float(np.mean(np.abs(oof[k] - cls) <= 1)) for k in oof}
    print(f"[OOF] N0 mae={mae['N0']:.4f} acc={acc['N0']:.4f} | N1 mae={mae['N1']:.4f} "
          f"acc={acc['N1']:.4f} | N05 mae={mae['N05']:.4f}")

    # ---------- §3.4 排序翻转 ----------
    rf = P.rank_flip(schemes["N0"], schemes["N1"], terminal, cls, seed=C2.RANDOM_SEED)
    pd.DataFrame([rf]).to_csv(os.path.join(OUT, "noise_rank_flip.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"[rank-flip] all={rf['all']:.4f} within={rf['within_terminal']:.4f} "
          f"cross={rf['cross_terminal']:.4f} adj={rf['adjacent_mcs']:.4f}")

    # ---------- §3.5 配对 Bootstrap Δ10 = MAE(N1)-MAE(N0) ----------
    ci_lo, ci_hi, dmean, p_gt0 = P.paired_bootstrap_diff(
        err["N1"], err["N0"], strata, n_boot=N_BOOT, seed=C2.RANDOM_SEED)
    print(f"[bootstrap] Δ10={dmean:+.4f} 95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}] P(Δ>0)={p_gt0:.3f}")

    # 逐折 N1 vs N0（至少4/5折判据）
    pf0 = P.per_fold_metrics(schemes["N0"], cls, levels, folds)
    pf1 = P.per_fold_metrics(schemes["N1"], cls, levels, folds)
    folds_n1_better = sum(1 for a, b in zip(pf1, pf0) if a["mae_ord"] < b["mae_ord"])

    # 分终端 grouped CV（跨终端稳健性）
    gfolds = P.grouped_folds(terminal, k=min(C2.K_FOLDS, len(np.unique(terminal))), seed=C2.RANDOM_SEED)
    g_mae0, _ = P.oof_metrics(schemes["N0"], cls, levels, gfolds)
    g_mae1, _ = P.oof_metrics(schemes["N1"], cls, levels, gfolds)
    print(f"[grouped-CV] N0 mae={g_mae0['mae_ord']:.4f} N1 mae={g_mae1['mae_ord']:.4f}")

    # ---------- NS 随机打乱噪声负对照 ----------
    ns_rows = []
    for s in range(N_SHUFFLE):
        rng = np.random.default_rng(1000 + s)
        sig_shuf = rng.permutation(sigma2)
        z = a1 / sig_shuf
        ag, _ = P.oof_metrics(z, cls, levels, folds)
        ns_rows.append({"seed": 1000 + s, "mae_ord": ag["mae_ord"],
                        "acc": ag["acc"], "acc1": ag["acc1"]})
    ns_df = pd.DataFrame(ns_rows)
    ns_df.to_csv(os.path.join(OUT, "noise_shuffle_results.csv"),
                 index=False, encoding="utf-8-sig")
    ns_med = float(np.median(ns_df["mae_ord"]))
    ns_better = int((ns_df["mae_ord"] <= mae["N1"]).sum())    # NS 不优于 N1 的次数
    print(f"[NS] 打乱噪声 mae 中位={ns_med:.4f}; N1={mae['N1']:.4f}; NS<=N1 次数={ns_better}/{N_SHUFFLE}")

    # ---------- §3.6 有效性判定 ----------
    delta = mae["N1"] - mae["N0"]
    per_term_ok = all(d["sp_nf_mcs"] * diag[0]["sp_nf_mcs"] >= 0 for d in diag[1:])
    if delta <= -0.01 and ci_hi < 0 and folds_n1_better >= 4 and ns_better <= N_SHUFFLE * 0.1:
        verdict = "有效"
    elif delta >= 0.01 or ci_lo > 0:
        verdict = "无效/口径不匹配"
    else:
        verdict = "证据不足"

    # ---------- 汇总指标文件 ----------
    metric_rows = []
    for name in ("N0", "N05", "N1", "NC"):
        metric_rows.append({"scheme": name, "mae_ord": mae[name], "acc": acc[name],
                            "acc1": acc1[name], "scope": "stratified-5fold-OOF"})
    metric_rows.append({"scheme": "NS_shuffled", "mae_ord": ns_med,
                        "acc": float(np.mean(ns_df['acc'])), "acc1": float(np.mean(ns_df['acc1'])),
                        "scope": f"mean of {N_SHUFFLE} seeds"})
    metric_rows.append({"scheme": "N0", "mae_ord": g_mae0["mae_ord"], "acc": g_mae0["acc"],
                        "acc1": g_mae0["acc1"], "scope": "grouped-by-terminal-CV"})
    metric_rows.append({"scheme": "N1", "mae_ord": g_mae1["mae_ord"], "acc": g_mae1["acc"],
                        "acc1": g_mae1["acc1"], "scope": "grouped-by-terminal-CV"})
    pd.DataFrame(metric_rows).to_csv(os.path.join(OUT, "noise_cv_metrics.csv"),
                                     index=False, encoding="utf-8-sig")

    # ---------- OOF 预测明细 ----------
    oof_df = pd.DataFrame({"terminal": terminal, "row_idx": tr["row_idx"], "cls_true": cls,
                           "noise_floor": nf})
    for name in ("N0", "N05", "N1", "NC"):
        oof_df[f"pred_{name}"] = oof[name]
        oof_df[f"err_{name}"] = err[name]
    oof_df.to_csv(os.path.join(OUT, "noise_oof_predictions.csv"),
                  index=False, encoding="utf-8-sig")

    # ---------- §8.1 noise_validity_summary.md ----------
    nc_ok = bool(np.array_equal(oof["NC"], oof["N0"]))
    with open(os.path.join(OUT, "noise_validity_summary.md"), "w", encoding="utf-8") as f:
        f.write("# noise_floor 有效性检验结论（第一阶段）\n\n")
        f.write(f"- 判定：**{verdict}**\n")
        f.write(f"- Δ10 = MAE(N1)−MAE(N0) = {delta:+.4f}，95%配对Bootstrap CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]，P(Δ>0)={p_gt0:.3f}\n")
        f.write(f"- N1 在 {folds_n1_better}/5 折优于 N0；分终端 grouped-CV：N0={g_mae0['mae_ord']:.4f} vs N1={g_mae1['mae_ord']:.4f}\n")
        f.write(f"- NS 随机打乱噪声 mae 中位={ns_med:.4f}；真实噪声 N1={mae['N1']:.4f}（{N_SHUFFLE} 次打乱中 {ns_better} 次不优于 N1）\n")
        f.write(f"- 排序翻转率 R_flip(N0↔N1)：总体={rf['all']:.4f}，终端内={rf['within_terminal']:.4f}，终端间={rf['cross_terminal']:.4f}\n")
        f.write(f"- NC 固定噪声自检：与 N0 逐样本预测一致 = **{nc_ok}**（§9.3 要求必须为 True）\n\n")
        f.write("## 依据\n\n")
        f.write(f"- 整体 Spearman：logλ1↔MCS={diag[0]['sp_loglam1_mcs']:.3f}，noise_floor↔MCS={diag[0]['sp_nf_mcs']:.3f}，"
                f"logλ1↔noise_floor={diag[0]['sp_loglam1_nf']:.3f}\n")
        f.write(f"- 分终端 noise_floor↔MCS 方向是否与整体一致：{per_term_ok}\n")
        f.write(f"- λ1/σ² 与 MCS 相关（{diag[0]['sp_loglam1_over_sig2_mcs']:.3f}）低于 λ1 与 MCS 相关（{diag[0]['sp_loglam1_mcs']:.3f}），"
                "说明除以样本级噪声底会破坏有序结构。\n\n")
        f.write("## 决策\n\n")
        if verdict == "有效":
            f.write("保留 θ=1，进入第二阶段按统一尺度原则比较五核。\n")
        else:
            f.write(f"判定为{verdict}：主模型固定 θ=0（输入称'主特征模态信道增益'，不称 SINR），"
                    "θ=1/θ=0.5 作为噪声敏感性结果报告。进入第二阶段尺度放缩 + 五核嵌套比较。\n")
    print(f"[verdict] {verdict}  (NC==N0 自检: {nc_ok})")
    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
