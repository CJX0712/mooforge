"""MOOForge command line interface.

Usage:
  python -m mooforge run [--out benchmark.json] [--n-evals 1500] [--pop-size 40] ...
  python -m mooforge list-problems
  python -m mooforge list-algorithms
  python -m mooforge version
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import MOOForgeConfig
from .core.seed import set_all
from .optimizers.factory import list_algorithms
from .pipeline.moo_pipeline import MOOForgePipeline
from .problems.registry import list_problems


def _print_table(result) -> None:
    meta = result["meta"]
    print(f"\nMOOForge benchmark  seed={meta['seed']}  n_evals={meta['n_evals']}  "
          f"pop={meta['pop_size']}  seeds={meta['n_seeds']}  runtime={meta['runtime_sec']}s")
    print("-" * 78)
    hdr = f"{'Problem':<10}{'Flagship HV':>14}{'Base HV':>12}{'Diff':>10}{'Sig':>7}"
    print(hdr)
    fvb = result["flagship_vs_baseline"]
    for pn, v in fvb.items():
        if not v.get("available"):
            print(f"{pn:<10}{'(n/a)':>14}")
            continue
        sig = "YES" if v["significant"] else "no"
        print(f"{pn:<10}{v['flagship_hv_mean']:>14.4f}{v['baseline_hv_mean']:>12.4f}"
              f"{v['diff']:>10.4f}{sig:>7}")
    q = result["quality"]
    print("-" * 78)
    print(f"Quality grade: {q['grade']}  gate_passed={q['gate_passed']}  "
          f"mean_hv_diff={q['mean_hv_diff']}  all_positive={q['all_positive']}  "
          f"all_significant={q['all_significant']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mooforge", description="MOOForge: multi-objective EMO toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full benchmark and emit benchmark.json")
    p_run.add_argument("--out", default="benchmark.json")
    p_run.add_argument("--n-evals", type=int, default=MOOForgeConfig().n_evals)
    p_run.add_argument("--pop-size", type=int, default=MOOForgeConfig().pop_size)
    p_run.add_argument("--n-seeds", type=int, default=MOOForgeConfig().n_seeds)
    p_run.add_argument("--ablation-seeds", type=int, default=MOOForgeConfig().ablation_seeds)
    p_run.add_argument("--problems", nargs="*", default=None)
    p_run.add_argument("--hpo", action="store_true", help="enable Optuna HPO (slower)")

    sub.add_parser("list-problems", help="list available benchmark problems")
    sub.add_parser("list-algorithms", help="list available optimizers")
    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)

    if args.cmd == "list-problems":
        for p in list_problems():
            print(p)
        return 0
    if args.cmd == "list-algorithms":
        for a in list_algorithms():
            print(a)
        return 0
    if args.cmd == "version":
        print("MOOForge 0.1.0")
        return 0
    if args.cmd == "run":
        cfg = MOOForgeConfig(
            n_evals=args.n_evals,
            pop_size=args.pop_size,
            n_seeds=args.n_seeds,
            ablation_seeds=args.ablation_seeds,
            aggressive=args.hpo,
        )
        set_all(cfg.seed)
        result = MOOForgePipeline(cfg).run(args.problems)
        payload = result.to_dict()
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        _print_table(payload)
        print(f"[OK] wrote {args.out}")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
