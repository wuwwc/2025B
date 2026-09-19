# -*- coding: utf-8 -*-
"""评价指标（《联合优化模型》§10）：序数指标为主，值域指标为辅。

- acc      : Accuracy / top-1，预测档位==真实档位的比例（§10.1）
- mae_ord  : 有序平均绝对误差 mean|c(ŷ)-c(y)|，档位差（§10.2，联合优化主指标）
- acc1     : 相邻一级准确率 |c(ŷ)-c(y)|<=1（§10.3）
- rmse/mae_val/r2 : 在 mcs 数值域上的辅助指标
"""
import numpy as np


def values_to_cls(values, levels):
    """把（吸附后的）mcs 数值映射回类别索引；取最近档位。"""
    values = np.asarray(values, float)
    levels = np.asarray(levels, float)
    idx = np.abs(values[:, None] - levels[None, :]).argmin(axis=1)
    return idx.astype(int)


def ordinal_metrics(cls_true, cls_pred, levels):
    """cls_*: 类别索引; levels: 档位数值数组。返回序数+值域指标 dict。"""
    ct = np.asarray(cls_true, int)
    cp = np.asarray(cls_pred, int)
    levels = np.asarray(levels, float)
    d = np.abs(ct - cp)
    acc = float(np.mean(ct == cp))          # top-1 / Accuracy
    mae_ord = float(np.mean(d))             # 档位差 MAE（主指标）
    acc1 = float(np.mean(d <= 1))           # 相邻一级准确率
    vt = levels[ct]
    vp = levels[cp]
    err = vp - vt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae_val = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((vt - vt.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"acc": acc, "mae_ord": mae_ord, "acc1": acc1,
            "rmse": rmse, "mae_val": mae_val, "r2": r2}
