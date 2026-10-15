"""Build a retry with verified E0 reuse and an early E3 smoke run."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
original = json.loads((root/'lab_day10.ipynb').read_text(encoding='utf-8'))
fixed = json.loads((root/'kaggle/resume/lab_day10.ipynb').read_text(encoding='utf-8'))
original['cells'][2] = fixed['cells'][2]
old = root/'kaggle-output/diagnosis/day10'
diagnostic = json.loads((old/'report/diagnostics/e0_test_inference.json').read_text())
predictions = (old/'work/e0_test.jsonl').read_bytes()
assert hashlib.sha256(predictions).hexdigest() == diagnostic['prediction_sha256']
assert len(predictions.splitlines()) == diagnostic['images']
version = json.loads((old/'data/splits/meta.json').read_text())['dataset_version']
buf = io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as bundle:
    for name in ['work/e0_test.jsonl','report/evidence/e0.json','report/diagnostics/e0_test_inference.json']:
        bundle.write(old/name,name)
payload = base64.b64encode(buf.getvalue()).decode()
original['cells'][8]['source'] = f'''assert json.loads((ROOT/'data/splits/meta.json').read_text())['dataset_version'] == {version!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as bundle:
    bundle.extractall(ROOT)
import hashlib
assert hashlib.sha256((ROOT/'work/e0_test.jsonl').read_bytes()).hexdigest() == {diagnostic['prediction_sha256']!r}
print('Reused verified E0 results from original run', flush=True)
'''.splitlines(True)
original['cells'][12]['source'] += ['\n# Catch E3 adapter/training failures before the long E1/E2 runs.\n',
    'run("-m","student.run","train","--setup","e3","--name","smoke_e3","--steps","2")\n']
for cell in original['cells']:
    if cell['cell_type']=='code':
        compile(''.join(cell['source']),'<retry>','exec')
        cell['outputs']=[]
        cell['execution_count']=None
folder=root/'kaggle/retry'
folder.mkdir(exist_ok=True)
(folder/'lab_day10.ipynb').write_text(json.dumps(original,ensure_ascii=False,indent=1),encoding='utf-8')
metadata=json.loads((root/'kaggle/kernel-metadata.json').read_text())
metadata.update(id='ngkhdio/day10-qwen3-vl-retry',title='Day10 Qwen3 VL Retry',kernel_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Prepared retry with verified E0 reuse and E3 smoke test')
