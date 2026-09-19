# -*- coding: utf-8 -*-
"""Stage 0: 读取 xlsx、解析 CSI 字符串为复数张量、写/读 npz 缓存、EDA。"""
import os
import glob
import ast
import numpy as np
import pandas as pd

import config as C


def _parse_csi_cell(s):
    """解析形如 '[(a+bj), (c+dj), 1.4e-05j, ...]' 的字符串 -> (122,) complex 数组。

    注意：纯虚数(实部为0)的 repr 不带括号，故不能用 '), (' 分割，
    统一用 ast.literal_eval 整体解析最稳妥。
    """
    vals = ast.literal_eval(s.strip())
    return np.array(vals, dtype=np.complex128)


def parse_xlsx(path):
    """读取单个 xlsx，返回 dict: H(N,122,2,4) complex, mcs, noise_floor, terminal, row_idx。"""
    df = pd.read_excel(path, engine="openpyxl")
    n = len(df)

    # 过滤 beamforming_en == 0（关闭 TxBF）
    if "beamforming_en" in df.columns:
        keep = df["beamforming_en"].astype(int).values == 0
    else:
        keep = np.ones(n, dtype=bool)
    df = df[keep].reset_index(drop=True)
    n = len(df)

    # 解析 8 列 CSI -> (N, 122, 2, 4)
    H = np.empty((n, C.N_SUBCARRIERS, C.N_RX, C.N_TX), dtype=np.complex128)
    for col_idx, col in enumerate(C.CSI_COLS):
        r = col_idx // C.N_TX
        c = col_idx % C.N_TX
        arr = np.stack([_parse_csi_cell(v) for v in df[col].values])  # (N,122)
        H[:, :, r, c] = arr

    terminal = os.path.basename(os.path.dirname(os.path.dirname(path)))
    # valid 集无 mcs 标签，用 NaN 占位
    mcs = df["mcs"].astype(float).values if "mcs" in df.columns else np.full(n, np.nan)
    out = {
        "H": H,
        "mcs": mcs,
        "noise_floor": df["noise_floor"].astype(float).values,
        "terminal": np.array([terminal] * n),
        "row_idx": np.arange(n),
    }
    del df
    return out


def _list_xlsx(split):
    """返回某 split 下所有 xlsx 路径（按终端排序）。"""
    pattern = os.path.join(C.DATA_DIR, "*", split, "*.xlsx")
    return sorted(glob.glob(pattern))


def build_cache(split):
    """解析某 split 的所有 xlsx 并写入 npz 缓存。split in {'train','valid'}。"""
    paths = _list_xlsx(split)
    chunks = []
    for p in paths:
        print(f"[parse] {p}")
        chunks.append(parse_xlsx(p))
    data = {k: np.concatenate([ch[k] for ch in chunks], axis=0) for k in chunks[0]}
    cache_path = C.TRAIN_CACHE if split == "train" else C.VALID_CACHE
    np.savez(cache_path, **data)
    print(f"[cache] saved {cache_path}, N={len(data['mcs'])}, H shape={data['H'].shape}")
    return data


def load_cache(split):
    cache_path = C.TRAIN_CACHE if split == "train" else C.VALID_CACHE
    if not os.path.exists(cache_path):
        return build_cache(split)
    with np.load(cache_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def eda(data):
    """打印 mcs 分布、noise_floor 范围、样本数。"""
    mcs = data["mcs"]
    uniq, counts = np.unique(mcs, return_counts=True)
    print(f"[EDA] N={len(mcs)}")
    print(f"[EDA] mcs unique count={len(uniq)}")
    print(f"[EDA] mcs unique values={uniq}")
    print(f"[EDA] mcs counts={counts}")
    print(f"[EDA] noise_floor min/max={data['noise_floor'].min():.2f}/{data['noise_floor'].max():.2f}")
    return uniq


if __name__ == "__main__":
    for split in ("train", "valid"):
        d = load_cache(split)   # 已有缓存则直接读，避免重复解析
        if split == "train":
            eda(d)
