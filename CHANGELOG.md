# Changelog

All notable changes to MOOForge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
semantic versioning.

## [0.1.0] — 2026-09-30

### Added
- **MOOForge toolkit**: a deterministic, reproducible multi-objective EMO
  benchmarking framework with a single CLI entry point.
- **Tier-0 SOTA backend** via `pymoo` 0.6.2: NSGA-II, NSGA-III, MOEA-D,
  SMS-EMOA, SPEA2 (auto-detected; the toolkit degrades gracefully to pure
  numpy when `pymoo` is absent).
- **Pure-numpy optimizers** (zero external dependencies): NSGA-II, DE/rand/1/bin,
  and the **AHVA-MOEA** flagship.
- **AHVA-MOEA** (flagship): an upper-confidence-bound bandit for operator-credit
  assignment (SBX / DE / strong-mutation) combined with elite-archive
  re-injection and stagnation restart — all inside a fixed evaluation budget.
- **Rigorous quality indicators**: normalised hypervolume (exact for d≤3,
  seeded Monte-Carlo for d≥4, bounded in [0,1]), IGD⁺, Schott spacing, spread.
- **Deterministic benchmark pipeline**: ≥3 seeds, mean±std aggregation, a
  significance gate (|Δ| > ½(σ₁+σ₂)), flagship-vs-baseline comparison, an
  ablation study (NoRestart / FixedSBX), failure-case analysis, and an
  automated quality grade (S/A/B/C).
- **Synthetic problem suite**: ZDT1/2/3/4/6 and DTLZ1/2/3/7 with analytic
  Pareto fronts and per-problem reference points covering worst-case local
  optima (so normalised HV is comparable across solvers).
- **CLI**: `run`, `list-problems`, `list-algorithms`, `version`.
- **Packaging & ops**: `pyproject.toml`, `Dockerfile`, `Makefile`, `LICENSE`
  (MIT), `requirements.txt`, `tests/` (pytest), and GitHub Actions CI.

### Author
- 晨星 (Morning Star)
