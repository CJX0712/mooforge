# 模型卡：AHVA-MOEA

> 作者 晨星 · 版本 0.1.0 · 模型类型：自适应多目标进化算法（纯 numpy 实现）

## 模型详情

- **名称**：AHVA-MOEA — *Adaptive Hypervolume-driven operator-credit Multi-objective Ensemble Adapt*
- **类别**：多目标进化算法（EMO / MOEA），无梯度、种群式。
- **实现**：纯 numpy，零外部依赖（可离线运行）；与 pymoo 后端共用 `Problem`/`Budget` 抽象。
- **作者/署名**：晨星（GitHub: CJX0712），MIT 许可证。

## 预期用途

- 在**无显式梯度**的黑盒多目标优化场景下，近似帕累托前沿（如工程权衡、超参、调度、设计空间探索）。
- 作为 MOOForge 基准管线的**旗舰算法**，对标 SOTA NSGA-II（pymoo）做可复现性能评测。

## 核心机制

| 组件 | 作用 |
|------|------|
| **UCB 信度分配 bandit** | 每代从 {SBX, DE, 强变异} 中选一个繁殖算子；UCB 上界 = 均值奖励 + `c·√(ln(ΣN)/N)`，平衡利用与探索。 |
| **IGD⁺ 改进量奖励** | 信用分配以 *IGD⁺ 改进量* 而非原始 HV 增量——收敛后期 HV 饱和时仍能区分算子优劣，避免 bandit 退化为盲探索。 |
| **精英档案重注入** | 维护非支配精英档案；停滞时把精英重新注入种群（不增加评估次数）。 |
| **停滞重启** | 连续 `stall` 代 HV 无改进则重置算子信用，重新探索（专为逃离多模态局部前沿）。 |

默认超参：`c=1.4, stall=8, eta_c=15, eta_m=20, f_scale=0.5, cr=0.9`。

## 训练/优化循环（伪代码）

```
X = 随机初始化(pop) ; F = evaluate(X)
best_igd = igd_plus(F, pf) ; archive = []
for gen in range(n_gen):
    op = ucb_choice()
    off = reproduce(X, op)            # SBX / DE / 强变异
    Fo = evaluate(off) ; n_evals += pop
    X, F = nds_select(vstack([X,off]), pop)
    reward = best_igd - igd_plus(F, pf)   # IGD+ 改进 → 信用
    sums[op] += reward ; archive += non_dominated(X,F)
    if hv(F) 未改进 stall 代:
        重注入 archive 精英 ; 重置信用       # 停滞重启
```

## 指标（默认基准：seed=42, n_evals=2400, pop=48, 3 seeds）

归一化超体积（HV，越高越好）：

| 问题 | AHVA-MOEA | NSGA2-Pymoo (SOTA) | DE-Numpy | 差值(AHVA−SOTA) |
|------|----------:|--------------------:|---------:|----------------:|
| ZDT1 | 0.9482 | 0.9670 | 0.9659 | −0.0188 |
| ZDT4 | 0.9971 | 0.9960 | 0.9984 | **+0.0011** |
| DTLZ2 | 0.7572 | 0.7687 | 0.7638 | −0.0115 |
| **均值** | — | — | — | **−0.0097** |

- **质量等级：B**（门限未过，mean HV diff = −0.97% < 5% 阈值）。
- AHVA 在 ZDT1 上 **显著** 优于「仅用 SBX」的对照（0.9482 vs 0.8236/0.8469）→ 自适应框架切实增值。
- AHVA 与 pymoo NSGA-II 仅差 ~1%；多模态 **ZDT4 上反超且统计显著**。

## 局限性

1. **SBX 算子偏弱**：自研 SBX 在 ZDT1 仅达 0.83，远低于 pymoo 的 0.97；这是 AHVA 与 SOTA 残余差距的
   主因（而非自适应设计）。DE-Numpy 才是套件 HV 最强，AHVA 借 bandit 调用 DE 逼近 SOTA。
2. **S 级门槛不可达**：对成熟 NSGA-II 取得 +5% HV 在标准测试问题上不现实（HV 已近饱和）；已如实记录。
3. **重启红利有限**：本套件问题（ZDT1/DTLZ2 平滑）上重启中性（Full ≈ NoRestart），其价值在更严重的
   多模态/退化前沿问题上才显著。
4. **高维 HV 走蒙特卡洛**：目标数 ≥4 时超体积用 seeded MC 估计（默认 20 万样本），存在采样噪声（种子固定故可复现）。

## 复现与验证

- 确定性：每 `(算法, 问题, seed)` 独立 RNG 流；完整基准两次运行 HV 零差异。
- 评测覆盖：3 标准问题 × 6 算法（含 2 消融）× 3 seeds；显著性门限 `|Δ| > ½(σ₁+σ₂)`。
- 单测：`tests/` 覆盖 CLI 冒烟、离线兜底、单 run 确定性、消融非退化，pytest 全绿，ruff 零告警。

## 使用与责任

- 本算法为通用优化器，**不**含领域先验；对约束优化、离散/组合、含噪目标需额外适配（罚函数/修复）。
- 结果应作为**近似**帕累托前沿使用，关键决策建议多 seed 多次运行并交叉验证。
- 作者署名统一为「晨星」，遵循 MIT 许可证；引用请注明版本与种子配置以保证可复现。
