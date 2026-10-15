"""Build a self-contained Kaggle notebook from an explicit source-file allowlist."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = []
    for directory in ("student", "student_tests", "scripts", "tests", "configs"):
        files += sorted(p for p in (ROOT/directory).rglob("*") if p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts)
    files += [ROOT/n for n in ("README.md","GUIDE.md","RUBRIC.md","REPORT.md",
                              "requirements-colab.txt","requirements-kaggle.txt","pytest.ini",
                              ".gitignore","CHECKLIST.md","BAO_CAO_LOI_VA_LUU_Y.md","KAGGLE.md")]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(buf.getvalue()).decode()
    cells = []
    def md(text):
        cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
    def code(text):
        cells.append({"cell_type":"code","metadata":{},"source":text.splitlines(True),
                      "execution_count":None,"outputs":[]})
    md("# Lab Day 10 — Nguyễn Khánh Đô\nNotebook Kaggle tự chứa mã nguồn, không chứa dữ liệu/trọng số/khóa. GPU T4, bật Internet, gắn Road Damage Dataset.\nRepo: https://github.com/Diohus/K4-Track-4-Day10-VisionLanguageModel\nĐọc KAGGLE.md; chạy theo thứ tự. E0–E4 có thể cần nhiều phiên GPU. Mọi điểm số phải lấy từ lần chạy thật.")
    md("## 0. Môi trường Kaggle và mã nguồn\nCài thư viện trực tiếp vào runtime Kaggle, dùng torch/torchvision GPU có sẵn. Chỉ dùng .venv khi chạy local. Chạy các process riêng để giải phóng VRAM sau mỗi bước.")
    code('''import base64, io, json, os, subprocess, sys, zipfile
from pathlib import Path
ROOT = Path("/kaggle/working/day10")
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
# Source snapshot generated from repository; original scripts/tests preserved.
PAYLOAD = "''' + payload + '''"
if PAYLOAD:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
        for member in z.namelist():
            target = (ROOT/member).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(ROOT)
# Kaggle uses its notebook runtime directly; local work uses .venv.
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false",
           PYTHONUNBUFFERED="1", WANDB_DISABLED="true")
def run(*args):
    subprocess.run([PY,*map(str,args)],cwd=ROOT,env=ENV,check=True)
run("-m","pip","install","-r","requirements-kaggle.txt")
run("-m","pip","uninstall","-y","torchao")
run("-c", "import torch; from peft import LoraConfig, get_peft_model; m=torch.nn.Sequential(torch.nn.Linear(4,4)); get_peft_model(m,LoraConfig(r=2,target_modules=['0'])); print('PEFT float Linear preflight passed')")
run("-c","import torch, torchvision, transformers, peft, trl, bitsandbytes; print(torch.__version__,torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))")
run("-m","pytest","-q","-k","not test_report_template_has_student_markers_and_ten_questions")
run("-m","pytest","student_tests","-q")
''')
    md("## 1. Dữ liệu: split theo session\nAdd Input: lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes. Không cần API token khi đã gắn dataset.")
    code('''run("scripts/prepare_data.py","--source","/kaggle/input")
run("-m","student.run","audit")
run("-m","student.reporting","--preview")
from IPython.display import display, Image
for path in sorted((ROOT/"report/figures").glob("ground_truth_*.png")):
    display(Image(filename=str(path)))
''')
    md("## 2. Nhãn YOLO → JSON 0–1000\nHàm trong student/common.py có unit test biên, JSON rỗng và thứ tự box. Xem round_trip.json để đối chiếu pixel.")
    code('''run("-c","from student.common import yolo_to_box; print(yolo_to_box('pothole',[.5,.5,.2,.4]))")
print((ROOT/"report/diagnostics/round_trip.json").read_text())
''')
    md("## 3. E0: zero-shot\nSuy luận toàn bộ test đúng một lần, ghi raw output và chấm bằng scripts/score.py gốc. Không thay output lỗi bằng [].")
    code('''run("-m","student.run","infer","--name","e0","--split","test")
''')
    md("## 4. Chat template và mask loss\nKiểm tra trước khi train: prefix equality, 220 token ảnh, chỉ JSON + im_end được giám sát; pixel_values và image_grid_thw nối theo chiều 0.")
    code('''run("-c","from transformers import AutoProcessor; from scripts.labcommon import read_jsonl; from student.modeling import mask_audit, MODEL_ID; mask_audit(read_jsonl('data/splits/train.jsonl')[0], AutoProcessor.from_pretrained(MODEL_ID))")
''')
    md("## 5. Setup E1/E2/E3\nDùng chung batch=1 × grad_accum=8, 300 optimizer steps, seed=42, NF4. Mỗi setup bắt đầu từ model gốc; E3 khớp cả attn.proj, tổng 348 module. Full projector và LoRA trainables ở float32.")
    code('''run("-c","from student.run import config_for; import json; print(json.dumps({n:config_for(n) for n in ['e1','e2','e3']},indent=2))")
''')
    md("## 6. Huấn luyện và validation\nMỗi checkpoint lưu adapter, projector, optimizer, scheduler, scaler và RNG. Nếu hết phiên, xem KAGGLE.md để tải Output và tiếp tục checkpoint. Không chạy lại cell đã hoàn thành nếu chưa đổi tên run.")
    for setup in ("e1","e2","e3"):
        code(f'''run("-m","student.run","train","--setup","{setup}")
run("-m","student.run","infer","--name","{setup}","--adapter","work/{setup}","--split","val")
''')
    md("## 7. Chọn setup bằng val, suy luận test và chấm E1–E3\nKhóa lựa chọn trước khi chấm test. Metadata lấy từ run.json thật, không dùng con số minh họa trong work.txt.")
    code('''run("-m","student.run","select")
for setup in ("e1","e2","e3"):
    run("-m","student.run","infer","--name",setup,"--adapter",f"work/{setup}","--split","test")
''')
    md("## 8. E4: chỉ đổi kích thước dữ liệu\n10% (142 ảnh) so với 100% (1422 ảnh). Dùng lại run 100% của setup được chọn; cùng số bước, LR, rank, prompt và decoding. Không thay bất cứ biến nào khác.")
    code('''run("-m","student.run","e4")
run("-m","student.run","infer","--name","e4_data_size_0.1","--adapter","work/e4_data_size_0.1","--split","test")
run("-m","student.run","e4-score-full")
''')
    md("## 9. E5 — bonus GRPO tùy chọn\nBật sau khi E0–E4 đã hoàn thành và còn ngân sách GPU. Reward có test; 4 generations, 60 steps. Chưa xác nhận runtime GPU trước lần chạy của bạn.")
    code('''RUN_E5 = False
if RUN_E5:
    run("-m","pytest","student_tests/test_core.py","-q")
    run("-m","student.grpo")
    run("-m","student.run","infer","--name","e5","--adapter","work/e5","--split","test")
''')
    md("## 10. Báo cáo, tự chấm và tải kết quả\nREPORT.generated.md lấy số thật, CI theo ảnh/session và ≥6 ảnh. Đọc và bổ sung diễn giải cá nhân trước khi chép sang REPORT.md. Checker chưa qua là bình thường khi REPORT còn mẫu và Git chưa có lịch sử checkpoint.")
    code('''run("-m","student.reporting","--student","Nguyễn Khánh Đô")
run("-m","student.audit")
print((ROOT/"report/SELF_GRADE.md").read_text())
from IPython.display import Markdown
display(Markdown((ROOT/"REPORT.generated.md").read_text()))
''')
    code('''# Zip nộp bài chứa evidence, raw predictions, báo cáo và mã; không chứa data/trọng số.
run("-m","student.package")
from IPython.display import FileLink
display(FileLink("submission_bundle.zip"))
# work/ chứa adapter/projector/checkpoint. Dùng Save Version để lưu Output
# hoặc tải riêng checkpoint cần thiết trước khi kết thúc phiên.
''')
    notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                        "language_info":{"name":"python","version":"3.11"}},
                "nbformat":4,"nbformat_minor":5}
    # The notebook inside the source bundle uses its adjacent source files.
    # This avoids recursively embedding the upload notebook into itself.
    import copy
    source_notebook = copy.deepcopy(notebook)
    for cell in source_notebook["cells"]:
        cell["source"] = [line.replace(payload, "") for line in cell["source"]]
    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("lab_day10.ipynb", json.dumps(source_notebook, ensure_ascii=False))
    final_payload = base64.b64encode(buf2.getvalue()).decode()
    for cell in notebook["cells"]:
        cell["source"] = [line.replace(payload, final_payload) for line in cell["source"]]
    (ROOT/"lab_day10.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
    print("Built lab_day10.ipynb:",len(cells),"cells,",len(buf.getvalue()),"embedded source bytes")


if __name__ == "__main__":
    build()
