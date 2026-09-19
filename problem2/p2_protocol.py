# -*- coding: utf-8 -*-
"""问题2 队友协议（问题处理.md）复用的统计与验证工具。

被 run_protocol_v3.py 调用。核心：
- oof_predict / oof_metrics : 逐样本 OOF 预测与指标（不只折均值, 供配对检验）；
- paired_bootstrap_diff     : 两方案逐样本误差差值的配对 Bootstrap 95%CI（按 terminal×MCS 分层重采样）；
- rank_flip                 : N0/N1 得分的排序翻转率（总体/终端内/终端间/相邻档位）；
- grouped_folds             : 按 terminal 分组的交叉验证折（跨终端泛化稳健性）；
- spearman / kendall        : 相关系数。
所有随机过程显式传 seed, 保证可复现。
"""
import numpy as np
from scipy import stats

import ordinal
import metrics as M


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def kendall(a, b):
    return float(stats.kendalltau(a, b).statistic)


def oof_predict(z, cls, levels, folds, w=None):
    """返回每样本 OOF 预测类别索引 (N,)。z 为一维分数, folds 为 (tr,va) 列表。"""
    z = np.asarray(z, float)
    N = len(cls)
    oof = np.zeros(N, int)
    for tr, va in folds:
        mdl = ordinal.fit_ordinal(z[tr], cls[tr], len(levels), None if w is None else w[tr])
        oof[va] = ordinal.predict_ordinal(z[va], mdl)
    return oof


def oof_metrics(z, cls, levels, folds, w=None):
    """返回 (agg_dict, oof_pred)。agg_dict 为整体 OOF 指标（非折均值）。"""
    oof = oof_predict(z, cls, levels, folds, w)
    return M.ordinal_metrics(cls, oof, levels), oof


def per_fold_metrics(z, cls, levels, folds, w=None):
    """逐折 val 指标列表（供 '至少4/5折更优' 判据）。"""
    out = []
    for tr, va in folds:
        mdl = ordinal.fit_ordinal(z[tr], cls[tr], len(levels), None if w is None else w[tr])
        pred = ordinal.predict_ordinal(z[va], mdl)
        out.append(M.ordinal_metrics(cls[va], pred, levels))
    return out


def paired_bootstrap_diff(e_a, e_b, strata, n_boot=2000, seed=42):
    """配对 Bootstrap: 统计 Δ = mean(e_a - e_b) 的 95%CI（按 strata 分层, 层内有放回重采样）。

    e_a/e_b 为逐样本绝对误差（同长度）。返回 (ci_low, ci_high, mean_diff, p_gt0)。
    p_gt0 = Δ>0 的 Bootstrap 比例（近似单侧 p 值）。
    """
    d = np.asarray(e_a, float) - np.asarray(e_b, float)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    boots = np.empty(n_boot)
    for bi in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        boots[bi] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), float(d.mean()), float(np.mean(boots > 0))


def rank_flip(z0, z1, terminal, cls, n_pairs=2_000_000, seed=42):
    """N0 与 N1 得分的排序翻转率 R_flip = Pr[(z0_i-z0_j)(z1_i-z1_j)<0]（随机抽样样本对）。"""
    z0 = np.asarray(z0, float); z1 = np.asarray(z1, float)
    terminal = np.asarray(terminal); cls = np.asarray(cls)
    N = len(z0)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    flip = ((z0[i] - z0[j]) * (z1[i] - z1[j])) < 0
    same_term = terminal[i] == terminal[j]
    adj = np.abs(cls[i].astype(int) - cls[j].astype(int)) == 1
    return {
        "all": float(flip.mean()),
        "within_terminal": float(flip[same_term].mean()),
        "cross_terminal": float(flip[~same_term].mean()),
        "adjacent_mcs": float(flip[adj].mean()) if adj.any() else float("nan"),
        "n_pairs": int(ok.sum()),
    }


def grouped_folds(terminal, k=5, seed=42):
    """按 terminal 分组的 k-fold（同终端样本只出现在同一折）, 检验跨终端泛化。"""
    terminal = np.asarray(terminal)
    groups = np.unique(terminal)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    gid = {g: perm[r] % k for r, g in enumerate(groups)}
    fold_of = np.array([gid[t] for t in terminal])
    idx = np.arange(len(terminal))
    return [(idx[fold_of != f], idx[fold_of == f]) for f in range(k)]
