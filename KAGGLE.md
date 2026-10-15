# Chạy Kaggle — Nguyễn Khánh Đô

Repository: https://github.com/Diohus/K4-Track-4-Day10-VisionLanguageModel

## Cách khuyến nghị: upload notebook tự chứa mã

1. Tạo notebook trên Kaggle, **File → Import Notebook**, chọn `lab_day10.ipynb` từ bản đã sửa này. Notebook chứa snapshot mã nguồn, không cần chờ push GitHub.
2. Trong Settings chọn **GPU T4 ×2** nếu có, bật **Internet**. Mã chủ động dùng GPU 0, không gộp VRAM hai T4. Nếu tài khoản chưa cho bật GPU/Internet, hoàn tất xác minh mà Kaggle yêu cầu.
3. Add Input dataset [Road Damage Dataset](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes). Chỉ gắn một bản dataset này để tránh chọn nhầm thư mục nguồn. Mã dùng `--source /kaggle/input`, không cần khóa Kaggle.
4. Chạy cell 0: giải nén source vào `/kaggle/working/day10`; **cài thư viện trực tiếp vào runtime Kaggle** bằng `sys.executable -m pip`, dùng torch/torchvision CUDA có sẵn. Không tạo venv trên Kaggle; local vẫn dùng `.venv`.
5. Chạy lần lượt 1–8. Cell gọi subprocess với `check=True`: lỗi sẽ dừng, không chấm kết quả thiếu. E1/E2/E3 chạy từ cùng model gốc, 300 bước, batch 1 × accumulation 8.
6. E5 tùy chọn: đổi `RUN_E5 = True` khi E0–E4 đã xong và còn GPU. Chưa kiểm chứng GRPO trên GPU trong môi trường máy hiện tại; không coi code E5 là bằng chứng đã nhận bonus.
7. Cell 10 sinh `REPORT.generated.md`, ảnh và tự chấm. Tải `submission_bundle.zip` bằng FileLink hoặc phần Output. Đọc lại 10 câu, xem ảnh và bổ sung nhận xét cá nhân, sau đó thay nội dung `REPORT.md` bằng bản đã duyệt.
8. **Save Version / Save & Run All** để chạy nền và lưu Output sau khi hoàn thành. Với phiên tương tác đang có kết quả, tải ngay zip và adapter cần giữ; đừng bấm chạy lại tất cả trên cùng output đã hoàn thành. Không hứa giới hạn giờ/GPU cố định vì phụ thuộc tài khoản.

Notebook upload chứa toàn bộ source; notebook nằm trong thư mục giải nén dùng các source ở bên cạnh. Khi chỉnh source trên máy, chạy `.venv\Scripts\python.exe -m student.build_notebook` để cập nhật bản upload.

## Chạy từng phần, tránh mất công khi hết phiên

Mọi lệnh bên dưới chạy trong thư mục `/kaggle/working/day10`, bằng Python của runtime Kaggle:

```python
run("-m", "student.run", "train", "--setup", "e2")
run("-m", "student.run", "infer", "--name", "e2", "--adapter", "work/e2", "--split", "val")
```

Nếu đã có `work/e2/run.json`, training đã hoàn tất: bỏ qua train, tiếp tục inference. Inference từ chối ghi đè raw output hoàn chỉnh. Khi một lần inference lỗi, file `.partial` không được dùng để chấm; chạy lại cùng lệnh sẽ tạo lại inference từ đầu.

Checkpoint mỗi 100 bước chứa cả adapter, projector, optimizer, scheduler, scaler và RNG:

```python
run("-m", "student.run", "train", "--setup", "e2",
    "--resume", "work/e2/checkpoint-100")
```

Chỉ nạp checkpoint do chính bạn tạo. Khi chuyển sang phiên mới:
- Tải hoặc gắn Output của notebook trước làm Input.
- Chạy cell môi trường và chuẩn bị data; chép lại `work/<run>`, `report/` và prediction `work/*_test.jsonl`, `work/*_val.jsonl` từ Output cũ vào thư mục tương ứng trong working.
- Giữ nguyên cấu hình khi resume. Log tài nguyên cộng thời gian các phần đã train, không tính thời gian chờ giữa phiên.
- Mỗi 100 bước lưu checkpoint khá lớn do optimizer cho projector. Theo dõi dung lượng Output; sau khi tải an toàn chỉ giữ checkpoint cần resume. Không đưa checkpoint vào Git.

Nếu OOM ngay ở batch 1: kiểm tra GPU khác đang dùng bộ nhớ, khởi động lại session sạch rồi nạp checkpoint. Không tự đổi resolution/LoRA/LR riêng một setup và vẫn coi là so sánh cùng điều kiện. Nếu buộc đổi cấu hình, ghi rõ và chạy lại đối chứng phù hợp.

Có thể smoke-test 2 bước bằng tên riêng:
```python
run("-m","student.run","train","--setup","e1","--name","smoke_e1","--steps","2")
run("-m","student.run","infer","--name","smoke_e1","--adapter","work/smoke_e1","--split","val")
```
Không dùng smoke run làm E1 nộp bài. 300 bước là preset của lab; không đảm bảo F1 ≥ ngưỡng trên mọi môi trường. Nếu không đạt, kiểm tra mask, thang tọa độ, output bị cắt và loss bằng **val**; không sửa số JSON.

## Kaggle API (tùy chọn)

Đã chuẩn bị notebook tự chứa mã, nên API chỉ cần upload notebook và metadata. Username Kaggle có thể khác GitHub Diohus; không tự suy đoán.

Cài CLI **trong .venv** trên máy của bạn:
```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\kaggle.exe kernels init -p kaggle
```

Dùng phương thức xác thực trong hướng dẫn Kaggle CLI chính thức; không dán token vào notebook/Git. Sửa metadata để:
- `id`: `<kaggle-username>/day10-qwen3-vl`
- `code_file`: `lab_day10.ipynb` (chép notebook vào thư mục `kaggle/`)
- `language`: `python`; `kernel_type`: `notebook`
- `is_private`: `true`; `enable_gpu`: `true`; `enable_internet`: `true`
- `dataset_sources`: `["lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"]`

```powershell
.\.venv\Scripts\kaggle.exe kernels push -p kaggle
.\.venv\Scripts\kaggle.exe kernels status <kaggle-username>/day10-qwen3-vl
.\.venv\Scripts\kaggle.exe kernels output <kaggle-username>/day10-qwen3-vl -p kaggle-output
```

Push sẽ bắt đầu chạy GPU. Quyền GPU và quota vẫn phụ thuộc tài khoản. Tài liệu chính thức: [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
Phiên `ngkhdio/day10-qwen3-vl-retry` đã COMPLETE; bundle, evidence và raw predictions được tải về bằng API ngày 15/10/2026. Không lưu thông tin xác thực trong Git. Kết quả nghiệm thu ở `report/VERIFICATION.md`.

## Đưa kết quả về repository và nghiệm thu

Giải nén bundle vào repo, **không** giải nén toàn bộ Kaggle Output (chứa data/weights) để commit.
Raw predictions trong bundle nằm ở `report/predictions/`; nếu muốn sinh lại báo cáo, chép chúng về `work/` và chuẩn bị dataset gốc.

Sau khi hoàn thiện REPORT.md:
```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts/check_submission.py
.\.venv\Scripts\python.exe -m student.audit
git diff -- scripts tests configs
git status --short
git ls-files data work
git rev-list --count HEAD
```

Checker của đề **không kiểm tra** hai ngưỡng E1 valid/F1 mặc dù rubric yêu cầu; `student.audit` kiểm tra bổ sung. Checker 0 FAIL không tự đảm bảo 100 điểm.

Test `test_report_template_has_student_markers_and_ten_questions` kiểm tra **mẫu** REPORT còn dấu `<điền`, trái với yêu cầu báo cáo hoàn chỉnh. Trước khi hoàn thiện REPORT: chạy toàn bộ test. Sau khi hoàn thiện: giữ nguyên tests và chạy:
```powershell
.\.venv\Scripts\python.exe -m pytest -q -k "not test_report_template_has_student_markers_and_ten_questions"
.\.venv\Scripts\python.exe -m pytest student_tests -q
```
Ghi rõ lý do loại đúng test mẫu này; không sửa test để che lỗi.

Tạo ≥5 commit thực chất theo CHECKLIST.md sau từng checkpoint; đừng gộp mọi kết quả thành một commit cuối hoặc tạo commit rỗng. Push vào repo cá nhân đã nêu và đặt public khi sẵn sàng nộp theo rubric.

## Tài liệu kỹ thuật đã đối chiếu

- [Qwen3-VL Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_vl): processor/generation và trường multimodal.
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer): dataset ảnh + prompt, reward function và cấu hình group. Trang tài liệu đúng tag 1.9.2 không truy cập được lúc rà soát; mã vẫn ghim phiên bản đề và cần smoke-test GPU.

## Chuyển từ notebook cũ dùng venv

Import lại lab_day10.ipynb bản mới và chạy cell 0. Notebook mới dùng `PY = sys.executable`, không dùng `/tmp/day10-venv`. Có thể để nguyên thư mục venv cũ; nó không còn được sử dụng. Không cần xóa dataset hoặc kết quả.
