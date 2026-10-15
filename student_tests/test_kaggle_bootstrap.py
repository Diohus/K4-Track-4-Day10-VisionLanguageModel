"""Check that Kaggle commands use the notebook interpreter directly."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_uses_runtime_python(tmp_path, monkeypatch):
    nb = json.loads((ROOT/"lab_day10.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "PY = sys.executable" in "".join(c["source"]))
    assert "import venv" not in source and "/tmp/day10-venv" not in source
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    block = source[source.index("PY = sys.executable"):]
    exec(compile(block, "<kaggle-bootstrap>", "exec"),
         {"sys": sys, "os": os, "subprocess": subprocess, "ROOT": tmp_path})
    assert calls[0][0] == [sys.executable, "-m", "pip", "install", "-r", "requirements-kaggle.txt"]
    assert all(args[0] == sys.executable and kw["check"] for args, kw in calls)
    assert all(kw["cwd"] == tmp_path for _, kw in calls)
