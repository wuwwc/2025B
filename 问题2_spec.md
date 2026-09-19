# 问题2 Spec：接收合并（特征向）SINR + 有序阈值速率估计模型

版本 v1（2026-09-19）　状态：待实现
方法来源：题面《题目提取结果.md》问题2；Stage-0 预研 `problem2/eda_capacity.py`（实测数字见 §5）

---

## 1. 题面重述

问题2：利用完备数据集 A 的 MIMO 信道 $H$（2×4×122）与关闭 TxBF 的实际传输速率 rate（标签），
**提出一种新的速率估计模型**，并对不完备数据集 B 的 $H$ 估计 rate。
与问题1的区别：不限于 EESM 框架，必须给出新模型；数据与问题1同源（notxbf）。

## 2. 关键数据特性：$\|H_i\|_F$ 跨信道（载波）一致

实测（train，N=20369）：

- 每样本 122 载波 $\|H_i\|_F^2$ 相对标准差：mean $2.85\times10^{-16}$、max $5.39\times10^{-16}$（机器精度）→ **范数跨信道一致**；
- 元素级 $\max|H_i-H_0|=4.98\times10^{-3}$：矩阵跨载波仅有相位级残差，范数严格平坦。

由此得到三条建模结论：

- **C1（ESM 层退化）**：$\gamma_i\equiv\gamma$ 时，任意生成元 $\Phi$、任意 $\alpha,\beta$ 的等效 SINR 映射恒等于 $\gamma$（恒等映射）。
  故一切 EESM/MIESM/LESM 类"改进"在本数据上收益恒为 0；新模型必须**丢掉 ESM 层**、在范数之外寻找信息源。
  问题1 的 β 不可识别问题在本模型中自然消失（模型无 β 参数）。
- **C2（充分统计量降维 + 2×2 闭式）**：$t=\|H\|_F^2=\lambda_1+\lambda_2$（平坦），$\det(HH^H)=\lambda_1\lambda_2$；
  每样本信道关于速率的信息压缩为二元统计量 $(t,\lambda_1)$。香农容量有 2×2 闭式
  $\det(I+\rho G)=1+\rho\,t+\rho^2\det G$，无需 SVD，全向量化。
- **C3（口径不变性）**：噪声口径的全局正缩放（×N_RX、mW/W）对一维特征只是正数乘；一维有序模型只依赖次序 → 预测不变（与问题1同口径结论）。

## 3. 为什么范数不够：特征向携带额外信息

题面口径 $\gamma_F=\|H\|_F^2/\sigma_n^2$ 只用 $t$，丢弃特征结构 $(\lambda_1,\lambda_2)$。
实际接收机做接收端合并（沿主特征向接收），合并后 SINR $\propto \lambda_1/\sigma_n^2$。
Spearman 相关（与类别索引）：$\gamma_{F,\text{dB}}$ 0.759 < $C_{tot}$ 0.790 < $\log\lambda_1$ **0.802**；$\lambda_2/\lambda_1$ 仅 −0.081（弱，不作主特征）。
⇒ 新模型信息源 = **特征向（接收合并后）SINR** 与 **流加权香农容量**。

## 4. 模型设计

### 4.1 特征层（`problem2/features.py`）

输入 $H$ (N,122,2,4)、noise_floor；噪声口径与问题1一致（mW、×N_RX，$\sigma_n^2=N_{RX}\cdot10^{nf/10}$）。

- $G_i=H_iH_i^H$，$t_i=\operatorname{tr}G_i$，$d_i=\det G_i$（2×2 闭式）；
- $\lambda_{i,1/2}=\tfrac12(t_i\pm\sqrt{t_i^2-4d_i})$，对载波取均值 $\lambda_1,\lambda_2$（同时输出跨载波 rel-std 作平坦证据）；
- $\rho_i=\gamma_i/N_{TX}$。

候选特征：

| 编号 | 特征 | 含义 |
|---|---|---|
| F1 | $\gamma_c=\lambda_1/\sigma_n^2$ | 主特征向（接收合并后）SINR |
| F2-w | $x_w=\sum_i[\log_2(1+\rho_i\lambda_{i,1})+w\log_2(1+\rho_i\lambda_{i,2})]$ | 流加权容量；$w=1$ 为全香农容量 $C_{tot}$，$w\to0$ 趋主流 |
| F0 | $\gamma_F$ | 问题1 特征（基线用） |

### 4.2 映射层

复用 `problem1/ordinal.py`：加权中位数等渗回归（PAVA）学单调阶梯映射 $x\to$ 档位索引；
类权重 $w_\ell=N/(L N_\ell)$ 的 on/off 由 CV 决定（协议同问题1）。预测 $\hat c=\operatorname{searchsorted}(\tau,x)$，$\hat y=mcs[\hat c]$。

### 4.3 候选集与选择（5-fold 分层 CV，seed=42，主指标 mae_ord）

- M1 = ordinal(F1)；M2-w = ordinal(F2-w)，$w\in\{0,0.25,0.5,0.75,1\}$；
- B1 = 问题1 主模型（EESM+ordinal on $\gamma_F$，参照）；B2 = ordinal(F0)；B3 = 单调线性基线（最优特征上，参照）。
- 选择规则：$\arg\min$ mae_ord，平手取 acc 高者。

### 4.4 已否决备选（附实证）

- 任意 EESM/Φ 变体：由 C1 收益恒 0；
- 线性解析 $\text{rate}\approx\eta\cdot C_{tot}\cdot\Delta f$ + 最近档吸附：样本内 acc=0.002（阈值结构强非线性，线性尺度失效）；
- 二维阈值/ML 黑箱：不作主链（保可解释与序数结构），可选作对照实验。

## 5. Stage-0 预研结果（`eda_capacity.py`，5-fold CV mean±std）

| 特征 | acc | mae_ord | acc1 |
|---|---|---|---|
| F0 $\gamma_F$（问题1口径） | 0.3769±0.0093 | 0.8482±0.0167 | 0.8098 |
| F2 w=1（$C_{tot}$） | 0.4058±0.0102 | 0.7804±0.0128 | 0.8351 |
| F1 $\lambda_1$ | **0.4254±0.0123** | **0.7508±0.0141** | **0.8425** |

参照：问题1 主模型 CV acc=0.3763 / mae_ord=0.8520 / acc1=0.8077。

## 6. 实现计划

文件：

- `problem2/features.py`：特征层 + 平坦性诊断（打印 C1/C2 证据）；
- `problem2/run_problem2.py`：Stage0 缓存+诊断 → Stage1 特征 → Stage2 CV 候选比较（含类权重开关）→
  Stage3 全量重拟合选中模型 → Stage4 valid 预测与摘要；
- 复用：`problem1/{config,data_loader,sinr,ordinal,rate_map,metrics}`、`common/cv`。

产出：

- `problem2/output/pred_valid_p2.csv`：terminal、row、SINR_eff(=$\gamma_c$)、预测 MCS（列口径对齐问题1表4）；
- `problem2/output/eval_summary_p2.txt`：CV 全表、选中模型、平坦性 rel-std、口径声明。

验收：

- A1 主模型 CV mae_ord < 0.848（胜同结构 B2）且 < 0.852（胜问题1主模型）；
- A2 平坦性 rel-std < 1e-12 写入 summary；
- A3 valid 预测行数 = 20369，取值 ⊆ 11 档位；
- A4 口径不变性：M1 在 ×N_RX on/off、mW/W 切换下预测逐条一致；M2 若被选中须在 summary 声明固定口径（$\rho$ 的绝对尺度进入次序）。

## 7. 风险与回退

- R1 $w$ 网格不胜 F1 → 主模型 = M1（最简且预研最优）；
- R2 train/valid 特征分布漂移（终端/点位）→ summary 输出分终端预测分布对照；
- R3 $\lambda$ 跨载波非严格平坦（元素级残差 5e-3）→ 用载波均值并报 rel-std；公式本身按载波计算，天然兼容非平坦情形。

## 8. 与问题1/3 的关系

- 问题1 = 范数特征 + EESM（退化恒等）+ 有序阈值；问题2 = 特征向特征 + 有序阈值（无 ESM 层），映射层与 CV 协议共享、结果可直接横向制表；
- 问题3（开启 TxBF）题面口径 $\sigma_1^2/\sigma_n^2$ 与 F1 同构 → 本 spec 的特征层可被问题3 直接复用（仅换数据源）。
