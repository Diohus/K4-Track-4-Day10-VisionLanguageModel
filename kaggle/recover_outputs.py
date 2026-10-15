import concurrent.futures
import json
import os
from pathlib import Path
import zipfile
import shutil
from dotenv import load_dotenv
from pip._vendor import truststore

truststore.inject_into_ssl()
load_dotenv('.env')
if os.environ.get('KAGGLE_KEY', '').startswith('KGAT_'):
    os.environ['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_KEY']
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

destination = Path('kaggle-output/recovery-files')
files = []
with api.build_kaggle_client() as client:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = 'ngkhdio'
    req.kernel_slug = 'day10-qwen3-vl-20261015'
    token = None
    while True:
        api._set_paging(req, 100, token)
        response = client.kernels.kernels_api_client.list_kernel_session_output(req)
        for f in response.files or []:
            name = f.file_name
            if '/report/' in name and not name.endswith(('.json', '.jsonl')):
                continue
            if name.startswith(('day10/report/', 'day10/work/e1/', 'day10/work/e2/', 'day10/data/splits/')) and 'checkpoint-' not in name:
                files.append((name, f.url))
            elif name in ['day10/work/e0_test.jsonl','day10/work/e1_val.jsonl','day10/work/e2_val.jsonl']:
                files.append((name, f.url))
        token = response.next_page_token
        if not token:
            break
print('Downloading', len(files), 'recovery files', flush=True)

def download(item):
    name, url = item
    target = destination/name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = Path('kaggle-output/diagnosis')/name
    if cached.is_file() and cached.stat().st_size:
        shutil.copyfile(cached, target)
        return
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
    with session.get(url, stream=True, timeout=(15,30)) as response:
        if response.status_code != 200:
            raise RuntimeError(f'Download failed for {name}: HTTP {response.status_code}')
        with target.with_suffix(target.suffix+'.partial').open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                stream.write(chunk)
        target.with_suffix(target.suffix+'.partial').replace(target)
    print('Saved', name, target.stat().st_size, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(download, files))
upload = Path('kaggle-output/recovery-dataset')
upload.mkdir(exist_ok=True)
with zipfile.ZipFile(upload/'recovery.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
    for path in destination.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(destination).as_posix())
(upload/'dataset-metadata.json').write_text(json.dumps({
    'id':'ngkhdio/day10-recovery-e1-e2-20261015',
    'title':'Day10 Recovery E1 E2 20261015',
    'licenses':[{'name':'other'}],
    'description':'Private recovery artifacts from my interrupted Day10 run. Final adapters, validation and E0 output only; no credentials.'
}), encoding='utf-8')
print(api.dataset_create_new(str(upload), public=False, convert_to_csv=False), flush=True)
