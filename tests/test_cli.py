"""CLI smoke tests (run as subprocesses for realism)."""

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _run(args, **kw):
    return subprocess.run([PY, "-m", "mooforge.cli", *args], cwd=REPO,
                          capture_output=True, text=True, check=False, **kw)


def test_version():
    out = _run(["version"])
    assert out.returncode == 0
    assert "MOOForge" in out.stdout


def test_list_problems_and_algorithms():
    p = _run(["list-problems"])
    a = _run(["list-algorithms"])
    assert p.returncode == 0 and "ZDT1" in p.stdout
    assert a.returncode == 0 and "AHVA-MOEA" in a.stdout


def test_smoke_run_produces_benchmark(tmp_path):
    out = tmp_path / "bm.json"
    r = _run([
        "run", "--out", str(out),
        "--n-evals", "200", "--pop-size", "20",
        "--n-seeds", "2", "--ablation-seeds", "1",
        "--problems", "ZDT1",
    ])
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["problems"] == ["ZDT1"]
    assert data["quality"]["grade"] in ("S", "A", "B", "C")
    # Every reported HV must be in [0, 1].
    for rec in data["records"]:
        assert 0.0 <= rec["hv"] <= 1.0
