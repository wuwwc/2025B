# -*- coding: utf-8 -*-
"""问题1 全局配置：路径、系统常量、EESM/噪声口径/序数模型/交叉验证超参。

方法来源：《2025B_问题1_联合优化模型.md》。
"""
import os
import numpy as np

# ---- 路径 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # problem1/
PROJECT_DIR = os.path.dirname(BASE_DIR)                        # 2025B/
DATA_DIR = os.path.join(PROJECT_DIR, "dataset", "notxbf_com_excels_f4")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TRAIN_CACHE = os.path.join(CACHE_DIR, "train_notxbf.npz")
VALID_CACHE = os.path.join(CACHE_DIR, "valid_notxbf.npz")

for _d in (CACHE_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- 系统常量 ----
N_SUBCARRIERS = 122      # 载波单元数 K
N_RX = 2                 # 接收天线
N_TX = 4                 # 发送天线

# CSI 列名（r=接收行, c=发送列），重组顺序：先 r 后 c
CSI_COLS = [f"csi_matrix_r{r}_c{c}" for r in range(N_RX) for c in range(N_TX)]

# ---- 速率档位（有序），运行时与训练数据 np.unique(mcs) 断言一致 ----
MCS_LEVELS = [103.2, 137.6, 206.5, 275.3, 309.7,
              344.1, 412.9, 458.8, 516.2, 573.5, 619.4]

# ---- 噪声功率口径（《联合优化模型》§2 / §13.1）----
# noise_floor(dBm) 视为“单接收支路”噪声功率 P_b；总噪声 ||n_k||^2 = N_RX * P_b。
# 注：×N_RX 与单位(mW/W)都是 gamma 的全局常数缩放，会被 beta 与阈值吸收，
#     对有序阈值模型的预测/指标无影响（run_problem1 会断言并留证 noise_convention_check.txt）。
NOISE_UNIT = "mW"        # 'mW': P_b=10**(nf/10);  'W': P_b=10**((nf-30)/10)
NOISE_APPLY_NRX = True   # True: 总噪声=N_RX*P_b;  False: 总噪声=P_b

# ---- EESM（线性域, alpha=1）----
ALPHA_FIXED = 1.0        # 仅有 MCS 标签时 alpha 与阈值尺度不可辨识，固定为 1（§4.2）
BETA_GRID = np.logspace(-2, 4, 31)   # 线性域 beta 对数网格（gamma 量级约 1e-4..1e3）

# ---- 序数速率模型 / 类不平衡（§7）----
USE_CLASS_WEIGHTS = "cv"  # 'on'/'off'/'cv'：是否启用类权重由 CV 决定

# ---- 交叉验证（§9）----
K_FOLDS = 5
CV_STRATIFY = True        # 按 MCS 档位分层（最小档计数 10 >= K_FOLDS，可行）
RANDOM_SEED = 42

# ---- 数值容差 ----
# 频域平坦数据中，少数 SINR 近乎相等的样本会因浮点舍入在不同 beta/噪声口径下发生
# 档位翻转，幅度 <1e-2（约 <0.3% 样本），物理上模型对 beta 与噪声口径不变。
# 用于判定“beta 不可识别”与“噪声口径不变”。
TOL_INVARIANT = 1e-2
