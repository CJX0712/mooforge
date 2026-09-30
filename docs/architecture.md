# MOOForge 架构与设计

> 版本 0.1.0 · 作者 晨星

## 1. 设计原则

1. **单向无环依赖**：`cli → pipeline → {problems, hpo, optimizers, metrics, eval} → core`。
   上层只依赖下层接口，下层（`core`）不反向引用任何业务模块。
2. **确定性优先**：全局唯一种子入口 `set_all(seed)`，每个 `(算法, 问题, seed)` 派生独立 RNG 流，
   结果可逐位复现。
3. **依赖可降级**：核心仅依赖 numpy；`pymoo`（SOTA 后端）与 `optuna`（HPO）缺失时自动跳过，
   不抛未捕获异常。
4. **指标可比**：归一化超体积以「参考框体积」为分母 → HV ∈ [0,1]，跨问题、跨求解器可比较。

## 2. 目录结构

```
mooforge/
├── __init__.py            # 顶层 API 导出 (__version__, __author__, build, FLAGSHIP, ...)
├── cli.py                 # 命令行入口 (run / list-problems / list-algorithms / version)
├── core/                  # 不依赖任何业务模块
│   ├── config.py          # MOOForgeConfig + MOOFORGE_* 环境变量覆盖
│   ├── seed.py            # set_all / spawn / optuna_sampler
│   ├── types.py           # ProblemSpec / OptimizationResult / Budget
│   ├── errors.py          # E100~E500 错误码体系
│   └── interfaces.py      # Protocol: Problem / Optimizer / Indicator
├── problems/
│   ├── synthetic.py       # BaseProblem + ZDT1-6 / DTLZ1-3,7（含解析 PF）
│   └── registry.py        # make_problem / list_problems / default_problems
├── optimizers/
│   ├── base.py            # BaseOptimizer + 向量化算子 (SBX / 多项式变异 / DE)
│   ├── numpy_impl.py      # NSGA2-Numpy / DE-Numpy / AHVA-MOEA
│   ├── pymoo_backends.py  # NSGA2/NSGA3/MOEAD/SMS-EMOA/SPEA2 (pymoo 0.6.2)
│   └── factory.py         # build / available / list_algorithms / FLAGSHIP / PRIMARY_BASELINE
├── metrics/
│   ├── hypervolume.py     # 精确 HV(d≤3) + 蒙特卡洛(d≥4)
│   └── indicators.py      # normalized_hv / igd_plus / spacing / spread
├── hpo/
│   └── tune.py            # tune_flagship (Optuna TPE，独立问题/种子，无泄漏)
├── eval/
│   └── protocol.py        # run_single / aggregate / significant
└── pipeline/
    └── moo_pipeline.py    # MOOForgePipeline：编排 + 显著性 + 消融 + 评级
```

## 3. 数据流（一次 `run`）

```
MOOForgePipeline.run(problems?)
   │
   ├─ tune_flagship(cfg)                      # aggressive=False 时返回默认参数，used=False
   ├─ _algorithms()                          # 固定快集 + AHVA + 2 消融
   │     base_set = [NSGA2-Pymoo, NSGA2-Numpy, DE-Numpy]  (可用性探测)
   │     + AHVA-MOEA + [AHVA-NoRestart, AHVA-FixedSBX]
   │
   └─ for (algo, params) in algorithms:
        for pn in problems:
          for s in range(n_seeds):
            run_single(algo, pn, budget, s, params)
              └─ build(algo, **params)
                 make_problem(pn)
                 algo.run(problem, budget, spawn(seed, hash("algo|label|problem")))
                    └─ 每代: ucb_choice() → _reproduce() → evaluate → _nds_select()
                       → reward(IGD+ 改进) → 精英档案 / 停滞重启
                    └─ _result(): 计算 HV / IGD+ / Spacing / Spread
   │
   ├─ aggregate(records)                     # per (algo,problem) mean±std
   ├─ _flagship_vs_baseline(summary)        # AHVA vs NSGA2-Pymoo 显著性
   ├─ _ablation(summary)                    # Full / NoRestart / FixedSBX
   ├─ _failure_cases(records)              # 最低 HV 的 (algo,problem)
   └─ _grade(fvb, ablation)                 # S/A/B/C 自动评级
```

## 4. 确定性实现要点

- `set_all(seed)` 同时固定 `random` / `numpy` / `PYTHONHASHSEED`。
- `run_single` 用 `spawn(seed, _stable_hash("algo|label|problem"))` 派生**每算法独立**种子，
  再用 `np.random.seed(...)` 重置全局 RNG；算子（`sbx_crossover` / `polynomial_mutation` /
  `de_operator`）全部走模块级 `np.random.*`，因此每个 run 可逐位复现，且不同算法互不串流。
- pymoo 后端把同一派生种子喂入 `minimize(..., seed=...)`，同样可复现。
- `_stable_hash` 用 MD5 而非 Python `hash()`，避免 `PYTHONHASHSEED` 之外的进程间不一致。

## 5. 关键不变量（金丝标准）

- `normalized_hv` 返回值恒在 **[0,1]**（参考框体积归一 + 仅保留被 ref 支配的点）。
- `igd_plus(F, pf)` 的 PF 维数必须与 `F` 一致，否则抛 `E401_REFERENCE`。
- `fast_non_dominated_sort` + `crowding_distance` 与标准 Deb NSGA-II 语义一致。
- 完整基准两次运行：48 条记录 HV **零差异**（见 README 可复现性验证）。

## 6. 扩展指南

- **加问题**：在 `problems/synthetic.py` 继承 `BaseProblem`，实现 `_evaluate` 与 `_pareto_front`，
  并在 `spec.ref_point` 设「覆盖最差局部最优」的参考点（保证 HV 可比）；`registry` 自动收录。
- **加算法**：在 `optimizers/numpy_impl.py`（或 `pymoo_backends.py`）实现 `run(problem, budget, seed)`
  返回 `OptimizationResult`，在 `factory._REGISTRY` 注册；
  若需可用性探测，实现 `available()`。
- **换基线**：改 `factory.PRIMARY_BASELINE` 与 `FLAGSHIP` 即可切换显著性门限与旗舰。
