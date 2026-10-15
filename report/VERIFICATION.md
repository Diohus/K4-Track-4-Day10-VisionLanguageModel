# Nghiệm thu kết quả Kaggle

Ngày nhập và kiểm tra: 15/10/2026. Họ tên: Nguyễn Khánh Đô.

## Nguồn và tính nhất quán

- Notebook: https://www.kaggle.com/code/ngkhdio/day10-qwen3-vl-retry — API trả trạng thái COMPLETE.
- Bundle nguồn SHA256: `98127b910bb5d030ef0a7373b42e5a663c7e12bd67deb459545dd82da125a26a`.
- Dataset: `d10-21e4070301`; khôi phục meta và ba split từ chính output của notebook.
- Chỉ nhập report, evidence, raw predictions và ảnh minh họa; không nhập weights hoặc dữ liệu ảnh vào Git.
- Mã student, student_tests, scripts, tests và configs trên máy khớp bundle nguồn tại thời điểm nhập. Sau đó builder notebook được cập nhật để giữ cách xử lý torchao đã chạy thành công và loại test mẫu REPORT đã hoàn thiện.
- Chấm lại cả sáu bộ raw prediction, mỗi bộ đủ 311 image_id: micro, CI95, per_class và parse_valid_rate khớp evidence bằng so sánh trực tiếp giá trị JSON.
- Hash raw prediction khớp diagnostics của E0/E1/E2/E3/E4 10%. E4 100% tái sử dụng cùng run và prediction E3.
- Đã xem sáu ảnh prediction và bổ sung nhận xét trong REPORT.md. Có thêm sáu ảnh ground truth trong bundle.
- `git diff HEAD -- scripts tests configs` rỗng: không sửa mã chấm, tests hoặc cấu hình gốc.

## Kiểm thử

`python -m pytest -q -k "not test_report_template_has_student_markers_and_ten_questions"`: **67 passed, 1 skipped, 1 deselected**.

Chỉ loại test kiểm tra REPORT vẫn chứa placeholder vì REPORT đã hoàn thiện. Không sửa test này. Một test bỏ qua theo điều kiện sẵn có của suite.

`python -m pytest student_tests -q`: **40 passed**. Bao gồm kiểm tra notebook nhúng đúng nguồn hiện tại và không chứa dữ liệu/trọng số/credentials.

Audit xác nhận phần A: **40/40**. E1 parse=308/311=0,990354; ΔF1(E1−E0)=0,251106. Phần B/C và bonus cần giảng viên đánh giá; không tự khẳng định 100/100.

## Giới hạn và lịch sử

E3 F1=0,410256 là kết quả một seed. Paired CI theo session của E3−E2 chứa 0; test chỉ có ba session. E5 chưa chạy. Không cộng điểm GRPO từ mã nguồn hoặc unit test.

Các commit bổ sung ghi nhận các phần mã, kết quả, báo cáo và notebook tại lúc hoàn thiện. Không tạo commit rỗng hay thay ngày để giả lịch sử train. Bản nộp local không tự xác nhận repo GitHub đã công khai hoặc được push.

Kết quả checker và số commit cuối được ghi trong `SELF_GRADE.md` và `self_grade.json` cạnh tệp này.
