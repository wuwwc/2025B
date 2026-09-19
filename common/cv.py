# -*- coding: utf-8 -*-
"""项目级共享：k-fold 交叉验证工具（问题1/2/3 复用）。

- iter_folds: 产出每折 (train_idx, val_idx)，labels 非空则分层。
- cross_validate: 逐折调用评估函数，返回每折指标与 mean±std 聚合。
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def iter_folds(n, k=5, seed=42, labels=None):
    """产出每折 (train_idx, val_idx)。labels 非空则按其分层（StratifiedKFold）。"""
    idx = np.arange(n)
    if labels is not None:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(cv.split(idx, np.asarray(labels)))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(cv.split(idx))


def cross_validate(fit_predict_eval, n=None, k=5, seed=42, labels=None, folds=None):
    """fit_predict_eval(train_idx, val_idx) -> dict[str,float]（单折指标）。

    返回 (per_fold_list, agg)，agg[metric] = (mean, std)。
    可直接传 folds（预先算好的折划分）以复用同一划分。
    """
    if folds is None:
        folds = iter_folds(n, k, seed, labels)
    per = [fit_predict_eval(np.asarray(tr), np.asarray(va)) for tr, va in folds]
    keys = list(per[0].keys())
    agg = {m: (float(np.mean([p[m] for p in per])),
               float(np.std([p[m] for p in per]))) for m in keys}
    return per, agg
