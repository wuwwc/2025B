# -*- coding: utf-8 -*-
"""Stage 1: 向量化 SINR 计算（关闭 TxBF）。

《联合优化模型》§3：gamma_{j,k} = ||H_k^{(j)}||_F^2 / ||n_k^{(j)}||^2，返回**线性域** gamma。
EESM 的指数运算必须使用线性 SINR（§3 注意点）；dB 值仅供展示，不进入 EESM。
"""
import numpy as np

import config as C


def compute_sinr_linear(H, noise_floor, unit=None, apply_nrx=None):
    """关闭 TxBF 的每载波**线性** SINR。

    H: (M,122,2,4) complex; noise_floor: (M,) dBm
    unit: 'mW'|'W'（默认 C.NOISE_UNIT）; apply_nrx: bool（默认 C.NOISE_APPLY_NRX）
    返回 (gamma_lin, g_dB)，均 (M,122)；gamma_lin 进入 EESM，g_dB 仅展示。

    口径: noise_total = (N_RX if apply_nrx else 1) * P_b，P_b 由 unit 决定。
    ×N_RX 与单位都是全局常数缩放，会被下游 beta/阈值吸收（对预测无影响）。
    """
    unit = C.NOISE_UNIT if unit is None else unit
    apply_nrx = C.NOISE_APPLY_NRX if apply_nrx is None else apply_nrx

    # Frobenius 范数平方：对每个载波求 |h|^2 之和（跨 2rx × 4tx）-> (M,122)
    fro = (np.abs(H) ** 2).sum(axis=(2, 3))
    if unit == "mW":
        p_branch = 10.0 ** (noise_floor / 10.0)             # (M,) mW
    elif unit == "W":
        p_branch = 10.0 ** ((noise_floor - 30.0) / 10.0)    # (M,) W
    else:
        raise ValueError(f"unknown NOISE_UNIT: {unit}")
    noise_total = (C.N_RX if apply_nrx else 1.0) * p_branch  # (M,)
    gamma_lin = fro / noise_total[:, None]                  # (M,122)
    g_dB = 10.0 * np.log10(np.maximum(gamma_lin, 1e-300))
    return gamma_lin, g_dB


def collapse_flat(gamma_lin, tol=1e-9):
    """频域平坦检测 + 吸附。

    若每样本 122 载波 SINR 的相对离散度 < tol，判为频率非选择性，把该样本 122 个
    SINR 吸附到其均值（1e-16 级差异是 |h|^2 求和的浮点噪声，非真实频率选择性）。
    吸附后 EESM 精确退化为 z_j = gamma_j/beta，且对 beta、噪声口径严格不变。
    返回 (gamma_collapsed, is_flat)。
    """
    mu = gamma_lin.mean(axis=1, keepdims=True)
    rel_std = gamma_lin.std(axis=1, keepdims=True) / np.maximum(mu, 1e-300)
    is_flat = bool(np.all(rel_std < tol))
    if is_flat:
        return np.repeat(mu, gamma_lin.shape[1], axis=1), True
    return gamma_lin, False
