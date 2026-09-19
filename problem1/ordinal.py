# -*- coding: utf-8 -*-
"""Stage 3: 等效 SINR -> MCS 的有序阈值模型（《联合优化模型》§5-§8）。

用**加权中位数(L1)等渗回归(PAVA)** 求最优单调阈值，精确最小化类加权 ordinal MAE：
    min_{f 单调不减}  sum_j w_j * |cls_j - f(z_j)|
其解由 PAVA 以“加权中位数”池化相邻违例块得到；块边界即阈值 tau，块取值即 level，
预测 Q_tau(z) = 所在块的 level。等价于文档 §8.2 的“DP/坐标搜索最优有序阈值”，
但复杂度 O(N*L)、且是该目标的精确最优（放宽到允许并档，只会更低）。
"""
import numpy as np


def _median_from_hist(hist):
    """hist: list[float] 长度 n_levels，各类累计权重；返回加权中位数类别索引。

    加权中位数 = 使 cumsum >= total/2 的最小类别，是 sum_h hist[h]*|h-v| 的最小化点。
    """
    total = 0.0
    for v in hist:
        total += v
    if total <= 0.0:
        return 0
    half = total / 2.0
    cum = 0.0
    for h, v in enumerate(hist):
        cum += v
        if cum >= half:
            return h
    return len(hist) - 1


def fit_ordinal(z, cls, n_levels, w=None):
    """加权中位数等渗回归 -> 单调阈值模型。

    z: (N,) 分数(等效 SINR); cls: (N,) 类别索引 0..n_levels-1; w: (N,) 样本权重(None=全1)
    返回 dict(thresholds, block_levels, n_levels)。block_levels 严格递增（相邻同档已合并）。
    """
    z = np.asarray(z, float)
    cls = np.asarray(cls, int)
    N = len(z)
    w = np.ones(N) if w is None else np.asarray(w, float)
    order = np.argsort(z, kind="mergesort")
    z_l = z[order].tolist()
    c_l = cls[order].tolist()
    w_l = w[order].tolist()

    # PAVA：栈内每块保存“类别权重直方图 + z 范围 + 加权中位数”
    hist_stack, zmin_stack, zmax_stack, med_stack = [], [], [], []
    for i in range(N):
        h = [0.0] * n_levels
        h[c_l[i]] = w_l[i]
        hist_stack.append(h)
        zmin_stack.append(z_l[i])
        zmax_stack.append(z_l[i])
        med_stack.append(c_l[i])
        # 池化违例：顶部块中位数 < 下方块中位数
        while len(hist_stack) >= 2 and med_stack[-1] < med_stack[-2]:
            h2 = hist_stack.pop(); mn2 = zmin_stack.pop(); mx2 = zmax_stack.pop(); med_stack.pop()
            h1 = hist_stack.pop(); mn1 = zmin_stack.pop(); mx1 = zmax_stack.pop(); med_stack.pop()
            hm = [a + b for a, b in zip(h1, h2)]
            hist_stack.append(hm)
            zmin_stack.append(mn1 if mn1 < mn2 else mn2)
            zmax_stack.append(mx1 if mx1 > mx2 else mx2)
            med_stack.append(_median_from_hist(hm))

    # 合并相邻同 level 块，得到严格递增的 level 序列
    lv_c, mn_c, mx_c = [], [], []
    for lv, mn, mx in zip(med_stack, zmin_stack, zmax_stack):
        if lv_c and lv_c[-1] == lv:
            if mx > mx_c[-1]:
                mx_c[-1] = mx
        else:
            lv_c.append(lv); mn_c.append(mn); mx_c.append(mx)

    block_levels = np.array(lv_c, int)
    zmin_arr = np.array(mn_c, float)
    zmax_arr = np.array(mx_c, float)
    thresholds = (0.5 * (zmax_arr[:-1] + zmin_arr[1:])) if len(lv_c) > 1 else np.array([])
    assert np.all(np.diff(block_levels) >= 0), "block_levels 非单调，PAVA 有误"
    return {"thresholds": thresholds, "block_levels": block_levels, "n_levels": n_levels}


def predict_ordinal(z, model):
    """Q_tau(z) -> 类别索引（0..n_levels-1）。越界自然裁剪到首/末块。"""
    z = np.asarray(z, float)
    tau = model["thresholds"]
    lv = model["block_levels"]
    if len(tau) == 0:
        return np.full(z.shape, int(lv[0]), int)
    b = np.searchsorted(tau, z, side="right")   # 0..len(tau)
    return lv[b]
