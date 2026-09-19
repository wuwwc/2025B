# -*- coding: utf-8 -*-
"""问题2 特征层 + 核聚合层（PEMK-ORE Stage 0/1/2）。

链路（spec v2 §3）：H -> G=HH^H -> 闭式主特征值 lambda1 -> 广义特征 u(theta)=lambda1/(sigma^2)^theta
              -> 五类核等权(1/K)跨载波聚合 -> z_(N,)
theta 与核函数/核参数一样由 CV 数据驱动选择（theta=1 即主特征模态 SINR, theta=0 即纯增益功率）。

噪声口径与问题1 一致：sigma_n^2 = N_RX * 10^(NF_dBm/10)（mW, ×N_RX），
2×2 Gram 矩阵特征值用闭式公式（spec §4.2），不做 SVD，全 numpy 向量化。
"""
import numpy as np

import config_p2 as C2          # noqa: F401  (导入即完成 sys.path 注入)
import config as C1             # problem1/config：系统常量与噪声口径


# ---------------- Stage 0/1：主特征模态 SINR ----------------
def eigenvalues(H):
    """逐载波闭式计算 G=HH^H 的两特征值（不做 SVD, 全向量化）。

    H: (N,K,2,4) complex -> (lam1, lam2): 各 (N,K), 满足 lam1>=lam2>=0, lam1+lam2=||H||_F^2。
    2x2 Gram 用闭式公式 lam=(tr±sqrt(tr^2-4det))/2, 根号内非负截断保证有限且实。
    """
    G = np.matmul(H, H.conj().transpose(0, 1, 3, 2))          # (N,K,2,2) = H H^H
    t = G[..., 0, 0].real + G[..., 1, 1].real                 # tr = ||H||_F^2
    d = G[..., 0, 0].real * G[..., 1, 1].real - np.abs(G[..., 0, 1]) ** 2   # det G (实)
    disc = np.sqrt(np.maximum(t * t - 4.0 * d, 0.0))
    lam1 = 0.5 * (t + disc)
    lam2 = np.maximum(0.5 * (t - disc), 0.0)
    return lam1, lam2


def principal_eigenmode_sinr(H, noise_floor):
    """逐载波闭式计算主特征模态 SINR。

    H: (N,122,2,4) complex; noise_floor: (N,) dBm
    返回 (lam1, lam2, x)：
      lam1/lam2: (N,122) 每载波 HH^H 的两特征值（lam1>=lam2>=0, lam1+lam2=||H||_F^2）
      x = lam1 / sigma_n^2: (N,122) 线性域主特征模态 SINR（spec §5）
    """
    lam1, lam2 = eigenvalues(H)
    sigma2 = noise_sigma2(noise_floor)
    x = lam1 / sigma2[:, None]
    return lam1, lam2, x


def carrier_features(H):
    """载波级候选特征 dict：lam1, lam2, geo=sqrt(lam1*lam2)=|detH|, fro=||H||_F^2。各 (N,K)。

    geo 为双模态几何平均（探针显示其对 MCS 排序能力优于 lam1, 且与 lam1 正交）。
    """
    lam1, lam2 = eigenvalues(H)
    return {"lam1": lam1, "lam2": lam2,
            "geo": np.sqrt(np.maximum(lam1 * lam2, 0.0)),
            "fro": lam1 + lam2}


def fold_scale(base, train_idx):
    """训练折内稳健正比例尺度 s_f = median_{n in train} mean_k base[n,k]（>0, 队友协议 §5.1）。

    只用 train_idx 计算, 严禁触碰验证折; 仅作正数缩放, 不产生负值、不改变样本间排序。
    """
    s = float(np.median(np.asarray(base)[train_idx].mean(axis=1)))
    assert s > 0.0, "尺度 s_f 必须为正"
    return s


def scale_input(base, s_f):
    """无量纲输入 u = base / s_f（队友协议 §5.1）。"""
    return np.asarray(base) / s_f


def noise_sigma2(noise_floor):
    """总噪声功率 sigma_n^2 = N_RX * 10^(NF/10) mW（与问题1 默认口径一致, spec §5）。"""
    return C1.N_RX * 10.0 ** (np.asarray(noise_floor, float) / 10.0)


def generalized_input(lam1, sigma2, theta):
    """载波级广义特征 u(theta) = lambda1 / (sigma_n^2)^theta（spec v2 §5）。

    theta=1: 主特征模态 SINR（问题1口径）; theta=0: 纯信道增益功率; theta=0.5: 中间口径。
    """
    if theta == 0.0:
        return lam1
    return lam1 / (sigma2 ** theta)[:, None]


def total_power_sinr(H, noise_floor):
    """问题1 口径 gamma_F = ||H||_F^2 / sigma_n^2（参照模型 R1 的载波级输入, spec §14）。"""
    fro = (np.abs(H) ** 2).sum(axis=(2, 3))                   # (N,122) = lam1+lam2
    return fro / noise_sigma2(noise_floor)[:, None]


def total_power_raw(H):
    """不除噪声的总增益功率 ||H||_F^2 = lam1+lam2, (N,K)。

    供广义口径 gamF(theta)=||H||_F^2/(sigma^2)^theta 复用（theta=1 即问题1 gamma_F,
    theta=0 即纯总增益功率）, 用于补全消融网格中问题1特征的去归一化分支。"""
    return (np.abs(H) ** 2).sum(axis=(2, 3))


def rel_std_per_sample(u):
    """每样本跨载波相对标准差 std/mean, (N,K) -> (N,)。"""
    mu = u.mean(axis=1)
    return u.std(axis=1) / np.maximum(mu, 1e-300)


# ---------------- Stage 2：五类核等权聚合（spec §6-§7） ----------------
def agg_linear(x):
    """K1 线性核：算术平均。x: (N,K) -> (N,)"""
    return x.mean(axis=1)


def agg_cap(x):
    """K2 香农容量核：2^{mean(log2(1+x))} - 1。"""
    return np.exp2(np.log1p(x).mean(axis=1) / np.log(2.0)) - 1.0


def agg_eesm(x, beta):
    """K3 EESM 指数核：-beta*ln(mean(exp(-x/beta)))。
    小 beta 时 exp(-x/beta) 会全体下溢到 0, 故先减去行内最大值移位（log-sum-exp 技巧）。"""
    v = -x / beta                       # v <= 0
    m = v.max(axis=1, keepdims=True)
    return -beta * (m[:, 0] + np.log(np.mean(np.exp(v - m), axis=1)))


def agg_pow(x, q):
    """K4 幂函数核：(mean(x^q))^{1/q}。x 下限截断防负 q 溢出。"""
    xs = np.maximum(x, C2.X_EPS)
    return (xs ** q).mean(axis=1) ** (1.0 / q)


def agg_expplus(x, beta):
    """K5 正指数核：beta*ln(mean(exp(x/beta)))，log-sum-exp 行内移最大值防溢出。"""
    u = x / beta
    m = u.max(axis=1, keepdims=True)
    return beta * (m[:, 0] + np.log(np.mean(np.exp(u - m), axis=1)))


# 核注册表：name -> (label, func, 参数网格, 自由参数个数)。参数网格元素 None 表示无参核。
KERNELS = {
    "K1_lin":  ("线性核",        agg_linear,  [None], 0),
    "K2_cap":  ("香农容量核",    agg_cap,     [None], 0),
    "K3_eesm": ("EESM指数核",    agg_eesm,    list(C2.BETA_EESM_GRID), 1),
    "K4_pow":  ("幂函数核",      agg_pow,     list(C2.Q_GRID), 1),
    "K5_plus": ("正指数核",      agg_expplus, list(C2.BETA_PLUS_GRID), 1),
}


def aggregate(x, kernel, param):
    """按核名与参数对载波级输入做等权 1/K 聚合。param=None 表示无参核。"""
    _, func, _, _ = KERNELS[kernel]
    return func(x) if param is None else func(x, param)


# ---------------- __main__ 自检 + Stage 0 诊断 ----------------
def stage0_stats(lam1, lam2, x, gF):
    """单 split 的跨载波平坦性统计（传入已算好的 (N,K) 数组, 避免重复 matmul）。"""
    r1, rx, rf = (rel_std_per_sample(lam1), rel_std_per_sample(x), rel_std_per_sample(gF))
    return dict(relstd_lam1=(r1.mean(), r1.max()),
                relstd_x=(rx.mean(), rx.max()),
                relstd_gammaF=(rf.mean(), rf.max()),
                lam2_over_lam1=float((lam2 / np.maximum(lam1, 1e-300)).mean()))


def stage0_report(splits, out_path=None):
    """打印/保存 lambda1、x、gamma_F 的跨载波平坦性诊断（创新点2 的实证依据）。

    splits: dict tag -> (lam1, lam2, x, gamma_F)，均由本模块函数计算得到。
    """
    rows = {}
    for tag, (lam1, lam2, x, gF) in splits.items():
        rows[tag] = stage0_stats(lam1, lam2, x, gF)
        r = rows[tag]
        print(f"[stage0/{tag}] rel-std lam1: mean={r['relstd_lam1'][0]:.3e} max={r['relstd_lam1'][1]:.3e} | "
              f"x: mean={r['relstd_x'][0]:.3e} max={r['relstd_x'][1]:.3e} | "
              f"gamma_F: mean={r['relstd_gammaF'][0]:.3e} max={r['relstd_gammaF'][1]:.3e}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("问题2 Stage0 数据诊断（spec §2 / §15：跨载波差异是多核聚合不退化的前提）\n")
            f.write("口径: sigma_n^2 = N_RX*10^(NF/10) mW（同问题1默认）。rel-std = 每样本122载波 std/mean。\n\n")
            for tag, r in rows.items():
                f.write(f"[{tag}]\n")
                f.write(f"  ||H||_F^2(=gamma_F输入)  rel-std mean/max = {r['relstd_gammaF'][0]:.3e} / {r['relstd_gammaF'][1]:.3e}\n")
                f.write(f"  lambda1               rel-std mean/max = {r['relstd_lam1'][0]:.3e} / {r['relstd_lam1'][1]:.3e}\n")
                f.write(f"  x=lambda1/sigma^2     rel-std mean/max = {r['relstd_x'][0]:.3e} / {r['relstd_x'][1]:.3e}\n")
                f.write(f"  mean(lam2/lam1)      = {r['lam2_over_lam1']:.4f}\n")
            f.write("\n结论判定：若 x 的 rel-std 显著大于 gamma_F（远超机器精度），则主特征模态输入下\n"
                    "      五类核聚合存在真实差异（创新点2成立）；若同为 1e-16 量级则核聚合退化，如实记录。\n")
    return rows


if __name__ == "__main__":
    import os
    import data_loader
    splits = {}
    for tag in ("train", "valid"):
        d = data_loader.load_cache(tag)
        lam1, lam2, x = principal_eigenmode_sinr(d["H"], d["noise_floor"])
        gF = total_power_sinr(d["H"], d["noise_floor"])
        splits[tag] = (lam1, lam2, x, gF)
    stage0_report(splits, os.path.join(C2.OUTPUT_DIR, "stage0_diagnosis.txt"))
