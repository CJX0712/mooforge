"""End-to-end MOOForge benchmark pipeline.

Produces a single, deterministic ``benchmark.json`` with: per-run records, per
(algo, problem) mean+-std, flagship-vs-baseline significance, ablation study,
failure-case analysis, and a quality grade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.config import DEFAULT_CONFIG, MOOForgeConfig
from ..core.seed import set_all
from ..core.types import Budget
from ..eval.protocol import aggregate, run_single, significant
from ..hpo.tune import tune_flagship
from ..optimizers.factory import (
    FLAGSHIP,
    PRIMARY_BASELINE,
    available,
)
from ..problems.registry import default_problems

ABLATION_VARIANTS = [
    {"name": "AHVA-NoRestart", "params": {"no_restart": True}},
    {"name": "AHVA-FixedSBX", "params": {"fixed_op": "sbx"}},
]


@dataclass
class PipelineResult:
    meta: dict
    records: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    flagship_vs_baseline: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    failure_cases: list[dict] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "records": self.records,
            "summary": self.summary,
            "flagship_vs_baseline": self.flagship_vs_baseline,
            "ablation": self.ablation,
            "failure_cases": self.failure_cases,
            "quality": self.quality,
        }


class MOOForgePipeline:
    def __init__(self, config: MOOForgeConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        set_all(self.config.seed)

    def _algorithms(self) -> list[tuple[str, dict]]:
        # Fixed, fast, representative baseline set: one Tier-0 SOTA (pymoo
        # NSGA-II), two Tier-1 numpy baselines, the AHVA flagship, and two
        # ablations. SMSEMOA/SPEA2/NSGA3/MOEAD remain selectable via build().
        base_set = ["NSGA2-Pymoo", "NSGA2-Numpy", "DE-Numpy"]
        algos: list[tuple[str, dict]] = []
        for a in base_set:
            if available(a):
                algos.append((a, {}))
        algos.append((FLAGSHIP, {}))
        # ablations use fewer seeds (config.ablation_seeds)
        for v in ABLATION_VARIANTS:
            algos.append((FLAGSHIP, {**v["params"], "_label": v["name"]}))
        return algos

    def run(self, problems: list[str] | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        cfg = self.config
        problems = problems or default_problems()
        budget = Budget(n_evals=cfg.n_evals, pop_size=cfg.pop_size)

        # Optional HPO on a separate cheap problem/seed set (no leakage).
        hpo = tune_flagship(cfg)
        flagship_params = hpo.params if hpo.used else {}

        algorithms = self._algorithms()
        records: list[dict] = []
        for algo_name, params in algorithms:
            label = params.pop("_label", algo_name)
            is_abl = label != algo_name
            n_seeds = cfg.ablation_seeds if is_abl else cfg.n_seeds
            for pn in problems:
                for s in range(n_seeds):
                    eff_params = dict(params)
                    if algo_name == FLAGSHIP and not is_abl:
                        eff_params.update(flagship_params)
                    res = run_single(algo_name, pn, budget, s, eff_params)
                    if res is None:
                        continue
                    r = res.as_record()
                    r["algorithm"] = label
                    records.append(r)

        summary = aggregate(records)
        fvb = self._flagship_vs_baseline(summary, problems)
        ablation = self._ablation(summary)
        failures = self._failure_cases(records, problems)
        quality = self._grade(fvb, ablation)

        meta = {
            "system": "MOOForge",
            "seed": cfg.seed,
            "n_evals": cfg.n_evals,
            "pop_size": cfg.pop_size,
            "n_seeds": cfg.n_seeds,
            "ablation_seeds": cfg.ablation_seeds,
            "problems": problems,
            "hpo_used": hpo.used,
            "hpo_params": hpo.params,
            "runtime_sec": round(time.perf_counter() - t0, 3),
        }
        return PipelineResult(
            meta=meta,
            records=records,
            summary=summary,
            flagship_vs_baseline=fvb,
            ablation=ablation,
            failure_cases=failures,
            quality=quality,
        )

    def _flagship_vs_baseline(self, summary: dict, problems: list[str]) -> dict:
        out: dict = {}
        for pn in problems:
            f = summary.get(f"{FLAGSHIP}::{pn}")
            b = summary.get(f"{PRIMARY_BASELINE}::{pn}")
            if not f or not b or f["hv_mean"] is None or b["hv_mean"] is None:
                out[pn] = {"available": False}
                continue
            sig, _ = significant(
                f["hv_mean"], f["hv_std"] or 0.0, b["hv_mean"], b["hv_std"] or 0.0
            )
            out[pn] = {
                "available": True,
                "flagship_hv_mean": f["hv_mean"],
                "baseline_hv_mean": b["hv_mean"],
                "diff": f["hv_mean"] - b["hv_mean"],
                "flagship_hv_std": f["hv_std"],
                "baseline_hv_std": b["hv_std"],
                "significant": bool(sig),
            }
        return out

    def _ablation(self, summary: dict) -> dict:
        out: dict = {}
        for pn in default_problems():
            full = summary.get(f"{FLAGSHIP}::{pn}")
            nr = summary.get("AHVA-NoRestart::" + pn)
            fx = summary.get("AHVA-FixedSBX::" + pn)
            row = {}
            for tag, r in [("full", full), ("no_restart", nr), ("fixed_sbx", fx)]:
                row[tag] = (r["hv_mean"] if r and r["hv_mean"] is not None else None)
            out[pn] = row
        return out

    def _failure_cases(self, records: list[dict], problems: list[str]) -> list[dict]:
        # Derive from real data: find the three largest shortfalls relative to the
        # analytic front (lowest normalised HV -> hardest cases for the field).
        scored: list[tuple[float, dict]] = []
        for r in records:
            if r.get("hv") is None:
                continue
            scored.append((r["hv"], r))
        scored.sort(key=lambda x: x[0])
        cases = []
        seen = set()
        for hv, r in scored[:30]:
            tag = (r["algorithm"], r["problem"])
            if tag in seen:
                continue
            seen.add(tag)
            cases.append(
                {
                    "algorithm": r["algorithm"],
                    "problem": r["problem"],
                    "hv_norm": round(hv, 4),
                    "igd": round(r["igd"], 4) if r.get("igd") is not None else None,
                    "attribution": self._attribute(r["problem"], r["algorithm"]),
                }
            )
            if len(cases) >= 3:
                break
        return cases

    @staticmethod
    def _attribute(problem: str, algo: str) -> str:
        notes = {
            "ZDT4": "highly multimodal landscape traps gradient-like operators in local Pareto fronts",
            "DTLZ1": "1e5 local optima along the g function; premature convergence is common",
            "DTLZ7": "degenerate disconnected PF (2^m regions) is hard to cover uniformly",
            "ZDT3": "disconnected PF segments break continuity assumptions of crowding distance",
            "ZDT6": "thin, non-uniformly sampled front stresses diversity preservation",
        }
        base = notes.get(problem, "difficulty inherent to the objective geometry")
        return f"{algo} undershoots on {problem}: {base}."

    def _grade(self, fvb: dict, ablation: dict) -> dict:
        diffs = [v["diff"] for v in fvb.values() if v.get("available")]
        sigs = [v["significant"] for v in fvb.values() if v.get("available")]
        if not diffs:
            return {"grade": "C", "gate_passed": False, "note": "no comparable results"}
        mean_diff = float(sum(diffs) / len(diffs))
        all_positive = all(d > 0 for d in diffs)
        all_sig = all(sigs)
        passed = mean_diff >= 0.05 and all_positive
        grade = "S" if (passed and all_sig) else ("A" if passed else "B")
        return {
            "grade": grade,
            "gate_passed": bool(passed),
            "mean_hv_diff": round(mean_diff, 4),
            "min_hv_diff": round(min(diffs), 4),
            "all_positive": bool(all_positive),
            "all_significant": bool(all_sig),
            "threshold": 0.05,
        }


def run_pipeline(config: MOOForgeConfig | None = None, problems: list[str] | None = None) -> dict:
    return MOOForgePipeline(config).run(problems).to_dict()


__all__ = ["MOOForgePipeline", "PipelineResult", "run_pipeline"]
