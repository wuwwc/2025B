# -*- coding: utf-8 -*-
"""Stage 2: EESM 等效 SINR 映射（线性域, alpha=1, log-sum-exp 稳定）。

《联合优化模型》§4：phi(x)=1-exp(-x) 时
    z_j = -alpha * ln[(1/K) * sum_k exp(-gamma_{j,k}/beta)]
指数项使用**线性 SINR**（§3 注意点，不能把 dB 代入指数）。
§4.2：仅有 MCS 标签时 alpha 与阈值尺度不可辨识，取 alpha=1。
"""
import numpy as np


def eesm_linear(gamma_lin, beta, alpha=1.0):
    """等效 SINR z_j（线性域, 数值稳定）。

    gamma_lin: (M,K) 线性 SINR(>0); beta>0 -> z: (M,)
    频域平坦(gamma_{j,k} 与 k 无关)时退化为 z_j = alpha*gamma_j/beta（见 spec）。
    """
    a = -np.asarray(gamma_lin, float) / beta                 # (M,K)
    a_max = a.max(axis=1, keepdims=True)                     # (M,1)
    # ln mean exp(a) = a_max + ln mean exp(a - a_max)  （log-sum-exp 稳定化）
    lse = a_max[:, 0] + np.log(np.mean(np.exp(a - a_max), axis=1))
    return -alpha * lse


def fit_affine(x, y):
    """闭式一元线性回归 y = alpha*x + c。返回 (alpha, c)。（基线 C 使用）"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom < 1e-12:
        return 0.0, ym
    alpha = np.sum((x - xm) * (y - ym)) / denom
    return alpha, ym - alpha * xm
