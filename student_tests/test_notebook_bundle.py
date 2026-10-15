import ast
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_notebook_embeds_current_sources_without_secrets_or_weights():
    nb = json.loads((ROOT/"lab_day10.ipynb").read_text(encoding="utf-8"))
    payload = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PAYLOAD" for t in node.targets):
                payload = ast.literal_eval(node.value)
    # Source notebook inside extracted bundle intentionally has no recursive payload.
    if not payload:
        assert (ROOT/"student/run.py").is_file()
        return
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        names = archive.namelist()
        assert "lab_day10.ipynb" in names and "student/run.py" in names
        assert not any(n.startswith(("data/","work/",".venv/")) or
                       n.endswith((".pt",".bin",".safetensors")) or "kaggle.json" in n for n in names)
        for n in names:
            if n=="lab_day10.ipynb":
                continue
            assert archive.read(n) == (ROOT/n).read_bytes(), f"Stale notebook source: {n}"
        snapshot = json.loads(archive.read("lab_day10.ipynb"))
        assert len(snapshot["cells"]) == len(nb["cells"])
        for cell in snapshot["cells"]:
            if cell["cell_type"]=="code":
                ast.parse("".join(cell["source"]))
