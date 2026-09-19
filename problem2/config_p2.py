# -*- coding: utf-8 -*-
"""问题2 全局配置：路径、核参数网格、CV 协议。

方法来源：《problem2/第二问数学模型_主特征模态多核聚合有序阈值.md》（PEMK-ORE）。
系统常量 / 噪声口径与问题1 一致，直接复用 problem1/config.py（N_RX=2, mW, ×N_RX）。

导入本模块即完成 sys.path 注入（problem1/ 与项目根目录），后续可直接
`import data_loader / ordinal / metrics / sinr` 及 `from common.cv import ...`。
"""
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # problem2/
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # 2025B/
P1_DIR = os.path.join(PROJECT_DIR, "problem1")

for _p in (P1_DIR, PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据与问题1 同源（notxbf），直接复用 problem1 已建好的 npz 缓存
TRAIN_CACHE = os.path.join(P1_DIR, "cache", "train_notxbf.npz")
VALID_CACHE = os.path.join(P1_DIR, "cache", "valid_notxbf.npz")

# ---- 五核聚合参数网格（§7 / §11）----
BETA_EESM_GRID = np.logspace(-2, 4, 21)   # K3 EESM 指数核 beta（线性域, 对数网格）
Q_GRID = [-2.0, -1.0, -0.5, 1.0, 2.0]     # K4 幂函数核 q（q=1 与 K1 线性核重合, 作一致性自检）
BETA_PLUS_GRID = np.logspace(-2, 4, 21)    # K5 正指数核 beta

# ---- 广义特征的噪声归一化指数（§5-v2）: u(theta) = lambda1 / (sigma^2)^theta ----
# theta=1 为 spec 原定义的主特征模态 SINR（问题1口径）; theta=0 为纯信道增益功率口径;
# 与核函数/核参数/类权重一样交给 CV 数据驱动选择。
THETA_GRID = [1.0, 0.5, 0.0]

# ---- 交叉验证协议（§11, 与问题1 相同：5-fold 分层, seed=42, 所有核共用同一折）----
K_FOLDS = 5
RANDOM_SEED = 42
CV_STRATIFY = True
USE_CLASS_WEIGHTS = "cv"    # 'on'/'off'/'cv'：类权重 w_l=N/(L·N_l) 开关由 CV 决定（口径同问题1）

# ---- 模型选择（§13）----
# 主规则 argmin mae_ord；mae_ord 差距小于该容差视为“接近”，依次比较 acc、acc±1、复杂度
#（无参核优先），再依次偏好更大的 theta（更接近标准 SINR 口径, 可解释性优先）。
TIE_TOL_MAE = 1e-3

# ---- 数值下限（幂核负 q 防 0 除 / 溢出）----
X_EPS = 1e-12
