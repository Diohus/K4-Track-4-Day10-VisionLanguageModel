"""Prepare a separate Kaggle continuation, preserving the original run."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'lab_day10.ipynb').read_text(encoding='utf-8'))
cell = notebook['cells'][2]
source = ''.join(cell['source'])
source = source.replace('run("-m","pip","install","-r","requirements-kaggle.txt")', '''run("-m","pip","install","-r","requirements-kaggle.txt")
# This lab uses bitsandbytes; an old optional torchao breaks PEFT's Linear dispatcher.
run("-m","pip","uninstall","-y","torchao")
run("-c", "import torch; from peft import LoraConfig, get_peft_model; m=torch.nn.Sequential(torch.nn.Linear(4,4)); get_peft_model(m,LoraConfig(r=2,target_modules=['0'])); print('PEFT float Linear preflight passed')")''')
cell['source'] = source.splitlines(True)
restore = '''import shutil, hashlib
run('scripts/prepare_data.py','--source','/kaggle/input')
search_roots = [Path('/kaggle/input')]
for archive in Path('/kaggle/input').rglob('recovery.zip'):
    extracted = ROOT/'_recovery'
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            assert (extracted/member).resolve().is_relative_to(extracted.resolve())
        bundle.extractall(extracted)
    search_roots = [extracted]
candidates = [p.parent.parent.parent for base in search_roots for p in base.rglob('work/e2/run.json')]
if len(candidates) != 1:
    raise RuntimeError(f'Expected one previous run with E2, found {candidates}')
previous = candidates[0]
current_version = json.loads((ROOT/'data/splits/meta.json').read_text())['dataset_version']
assert json.loads((previous/'data/splits/meta.json').read_text())['dataset_version'] == current_version
for name in ('e1','e2'):
    folder = previous/'work'/name
    for filename in ('run.json','adapter_config.json','adapter_model.safetensors'):
        assert (folder/filename).is_file(), f'Missing {folder/filename}'
    info = json.loads((folder/'run.json').read_text())
    assert info['completed_steps'] == 300
    if name == 'e2':
        assert (folder/'projector.pt').is_file()
# Copy final adapters only; intermediate optimizer checkpoints are unnecessary.
for directory in ('data','report','work'):
    shutil.copytree(previous/directory, ROOT/directory, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('checkpoint-*'))
version = json.loads((ROOT/'data/splits/meta.json').read_text())['dataset_version']
for name in ('e1','e2'):
    assert json.loads((ROOT/f'work/{name}/run.json').read_text())['dataset_version'] == version
for name, split in [('e0','test'),('e1','val'),('e2','val')]:
    pred = ROOT/f'work/{name}_{split}.jsonl'
    diagnostic = json.loads((ROOT/f'report/diagnostics/{name}_{split}_inference.json').read_text())
    assert hashlib.sha256(pred.read_bytes()).hexdigest() == diagnostic['prediction_sha256']
    assert len(pred.read_text().splitlines()) == diagnostic['images']
print('RESTORED: E0 predictions, completed E1/E2 adapters and validation; starting E3 next', flush=True)
run('-m','student.run','audit')
'''
notebook['cells'][4]['source'] = restore.splitlines(True)
for index, message in [(8,'Reusing verified E0 test predictions'),(14,'Reusing completed E1 and validation'),(15,'Reusing completed E2 and validation')]:
    notebook['cells'][index]['source'] = [f'print({message!r})\n']
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), '<resume cell>', 'exec')
        cell['execution_count'] = None
        cell['outputs'] = []
out = root/'kaggle'/'resume'
out.mkdir(exist_ok=True)
(out/'lab_day10.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
metadata = json.loads((root/'kaggle/kernel-metadata.json').read_text())
metadata.update(id='ngkhdio/day10-qwen3-vl-resume-e3', title='Day10 Qwen3 VL Resume E3',
                kernel_sources=[], dataset_sources=[
                    'lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes',
                    'ngkhdio/day10-recovery-e1-e2-20261015'])
(out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Prepared continuation:', out)
