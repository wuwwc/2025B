# -*- coding: utf-8 -*-
"""问题2 Stage-0 EDA：核实 H_i 范数跨载波(信道)一致特性，并预研容量特征有效性。

产出诊断（供 问题2_spec.md 引用）:
  1) 平坦性: H_i 跨载波是整矩阵相同还是仅范数相同
  2) 特征与 MCS 的 Spearman 相关: gamma_dB / 总香农容量 Ctot / 主特征值 / 特征值比
  3) 5-fold CV: 有序阈值模型在不同一维特征上的序数指标（容量特征 vs 问题1 范数特征）
  4) eta 解析模型（rate ≈ η·Ctot·Δf）的样本内 sanity
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(os.path.dirname(HERE), "problem1")
sys.path.insert(0, P1)
import config as C                                                # noqa: E402
sys.path.append(C.PROJECT_DIR)
from common.cv import iter_folds, cross_validate                  # noqa: E402
import data_loader                                                # noqa: E402
import ordinal                                                    # noqa: E402
import rate_map                                                   # noqa: E402
import metrics as M                                               # noqa: E402

DF_SUB_MHZ = 0.078125          # 载波间隔 78.125 kHz -> MHz（附录1）

train = data_loader.load_cache("train")
H = train["H"]
nf = np.asarray(train["noise_floor"], float)
y = np.asarray(train["mcs"], float)
levels = np.unique(y)
n_levels = len(levels)
cls = np.searchsorted(levels, y).astype(int)
n = len(y)

# ---- 1) 平坦性 ----
dev_mat = float(np.abs(H - H[:, :1]).max())
fro = (np.abs(H) ** 2).sum(axis=(2, 3))                      # (N,122)
rel = fro.std(axis=1) / fro.mean(axis=1)
print(f"[flat] max|H_i-H_0|={dev_mat:.3e}  fro rel-std mean={rel.mean():.3e} max={rel.max():.3e}")

# ---- 2) 每载波 2x2 闭式 logdet -> 香农容量 ----
noise_total = C.N_RX * 10.0 ** (nf / 10.0)
gamma = fro / noise_total[:, None]                           # (N,122)
G = np.matmul(H, H.conj().transpose(0, 1, 3, 2))             # H H^H, (N,122,2,2)
t = G[..., 0, 0].real + G[..., 1, 1].real                    # tr = ||H||_F^2
dG = (G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]).real
rho = gamma / C.N_TX
detI = 1.0 + rho * t + rho ** 2 * dG                         # det(I + rho*G) (2x2 闭式)
cap = np.log2(np.maximum(detI, 1e-300))
Ctot = cap.sum(axis=1)                                       # 总容量 (bit/s/Hz over 122 subcarriers)
disc = np.sqrt(np.maximum(t[:, 0] ** 2 - 4 * dG[:, 0], 0.0))
lam1 = (t[:, 0] + disc) / 2.0
lam2 = (t[:, 0] - disc) / 2.0
g0 = gamma[:, 0]

feat = pd.DataFrame({
    "cls": cls,
    "gamma_dB": 10 * np.log10(g0),
    "Ctot": Ctot,
    "log_lam1": np.log10(np.maximum(lam1, 1e-300)),
    "lam2_over_lam1": lam2 / np.maximum(lam1, 1e-300),
})
print("[spearman vs cls]")
print(feat.corr(method="spearman")["cls"].round(4).to_string())

# ---- 3) 5-fold CV: 有序阈值 on 不同一维特征 ----
folds = iter_folds(n, C.K_FOLDS, C.RANDOM_SEED, cls)


def cv_feat(x):
    x = np.asarray(x, float)

    def fpe(tr, va):
        mdl = ordinal.fit_ordinal(x[tr], cls[tr], n_levels, None)
        cp = ordinal.predict_ordinal(x[va], mdl)
        return M.ordinal_metrics(cls[va], cp, levels)
    _, agg = cross_validate(fpe, folds=folds)
    return agg


for name, x in [("gamma(P1特征)", g0), ("Ctot(容量)", Ctot), ("lam1(主奇异值^2)", lam1)]:
    a = cv_feat(x)
    print(f"[cv] {name:18s} acc={a['acc'][0]:.4f}±{a['acc'][1]:.4f} "
          f"mae_ord={a['mae_ord'][0]:.4f}±{a['mae_ord'][1]:.4f} acc1={a['acc1'][0]:.4f}")

# ---- 4) eta 解析模型 sanity（样本内）----
Cp = Ctot * DF_SUB_MHZ                                       # Mbps 量纲候选
eta = float((y * Cp).sum() / (Cp * Cp).sum())
pred = rate_map.snap_to_levels(eta * Cp, levels)
cp = np.searchsorted(levels, pred).astype(int)
m = M.ordinal_metrics(cls, cp, levels)
print(f"[eta] eta={eta:.4f}  in-sample acc={m['acc']:.4f} mae_ord={m['mae_ord']:.4f} acc1={m['acc1']:.4f}")
