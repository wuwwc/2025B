# 问题2 实验总结（队友协议：噪声有效性 + 尺度放缩 + 嵌套五核）

## 噪声有效性（第一阶段）
- 判定 noise_floor 无效/口径不匹配，主模型固定 θ=0（详见 noise_validity_summary.md）。

## 尺度放缩 + 嵌套五核（第二阶段）
- 最终尺度 s_A = 1.6422e-09；最终模型 feature=geo, kernel=K1_lin, param=N/A。
- 最终 OOF mae_ord = 0.7129。

## 各特征 OOF 指标（采纳核）

| feature | adopted kernel | mae_ord | acc | acc±1 |
|---|---|---|---|---|
| lam1 | K1_lin | 0.7561 | 0.4214 | 0.8408 |
| lam2 | K5_plus | 0.7383 | 0.4398 | 0.8474 |
| geo | K1_lin | 0.7129 | 0.4498 | 0.8592 |
| fro | K1_lin | 0.7661 | 0.4111 | 0.8406 |

## 自检（§9）
- OK finite_features
- OK scale_positive
- OK K4_q1_eq_K1
- OK scale_invariance
- OK no_overflow_grid
- OK B_rows_match
- OK pred_in_levels

全部自检通过：True
