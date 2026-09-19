# -*- coding: utf-8 -*-
"""Stage 3: SINR_eff -> 速率索引 的映射模型。

主模型见 ordinal.py（加权中位数等渗 PAVA 最优有序阈值, 等价文档§5 DP）。
本模块提供基线/对照:
  MonotonicLinearModel : 《第一问算法.md》§8 单调线性基线(a>=0 WLS 拟合类别索引 + round/clip)
  IsotonicRateModel    : 额外内部对照(L2 等渗回归 + 吸附)
  AffineRateModel      : 旧无约束仿射基线(保留)
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class MonotonicLinearModel:
    """《第一问算法.md》§8 单调线性基线。

    模型 c(y_j) ~= a*z_j + b, a>=0；固定 beta 后为带 a>=0 约束的凸加权最小二乘, 解析解:
      a0 = sum w(z-zbar_w)(c-cbar_w) / sum w(z-zbar_w)^2;  a = max(0, a0);  b = cbar_w - a*zbar_w
    离散化: chat = clip(round(a*z+b), 0, L-1) -> yhat = levels[chat]。
    频域平坦下 z=gamma/beta, a 与 beta 不可同时辨识(a*z 对 beta 不变), 故 beta 任取。
    """

    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.n_levels = 0

    def fit(self, z, cls, n_levels, w=None):
        z = np.asarray(z, float)
        c = np.asarray(cls, float)
        w = np.ones(len(z)) if w is None else np.asarray(w, float)
        sw = w.sum()
        zbar = float((w * z).sum() / sw)
        cbar = float((w * c).sum() / sw)
        num = float((w * (z - zbar) * (c - cbar)).sum())
        den = float((w * (z - zbar) ** 2).sum())
        a0 = num / den if den > 1e-12 else 0.0
        self.a = max(0.0, a0)                 # 单调约束 a>=0
        self.b = cbar - self.a * zbar
        self.n_levels = int(n_levels)
        return self

    def predict_raw_cls(self, z):
        """连续类别预测 a*z+b（未取整）。"""
        return self.a * np.asarray(z, float) + self.b

    def predict_cls(self, z):
        """round+clip -> 类别索引 0..L-1。"""
        c = np.round(self.predict_raw_cls(z))
        return np.clip(c, 0, self.n_levels - 1).astype(int)


def snap_to_levels(pred, levels):
    """把连续预测值吸附到最近的离散 mcs 电平。"""
    pred = np.asarray(pred, dtype=float)
    levels = np.asarray(levels, dtype=float)
    idx = np.abs(pred[:, None] - levels[None, :]).argmin(axis=1)
    return levels[idx]


class IsotonicRateModel:
    def fit(self, x, y):
        self.levels = np.unique(y)
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.ir.fit(x, y)
        return self

    def predict_raw(self, x):
        return self.ir.predict(x)

    def predict(self, x):
        return snap_to_levels(self.predict_raw(x), self.levels)


class ThresholdRateModel:
    """序数门限模型：相邻 mcs 档位的切换门限取两档 SINR_eff 中点，
    预测时 rate_index = 门限划分的所在档位（单调阶梯，BLER 门限启发）。"""

    def fit(self, x, y):
        self.levels = np.unique(y)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        med = np.array([np.median(x[y == lv]) for lv in self.levels])
        # 相邻档位中点作为切换门限
        thr = 0.5 * (med[:-1] + med[1:])
        # 强制门限单调递增
        self.thresholds = np.maximum.accumulate(thr)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.thresholds, x, side="right")
        return self.levels[idx]

    def predict_raw(self, x):
        return self.predict(x)


class AffineRateModel:
    def __init__(self, alpha, c):
        self.alpha = alpha
        self.c = c
        self.levels = None

    def fit(self, x, y):
        from eesm import fit_affine
        self.alpha, self.c = fit_affine(x, y)
        self.levels = np.unique(y)
        return self

    def predict_raw(self, x):
        return self.alpha * np.asarray(x, dtype=float) + self.c

    def predict(self, x):
        return snap_to_levels(self.predict_raw(x), self.levels)
