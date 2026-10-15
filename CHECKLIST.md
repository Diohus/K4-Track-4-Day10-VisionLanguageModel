# Nghiệm thu — Nguyễn Khánh Đô

Nguồn kết quả: `ngkhdio/day10-qwen3-vl-retry`, Kaggle COMPLETE, tải về ngày 15/10/2026. Dataset `d10-21e4070301`.

- [x] E1/E2/E3 và E4 10% hoàn thành 300 bước; E3 có 348 module LoRA. Nhật ký và cấu hình trong `report/runs/`.
- [x] Split chuẩn 1422/276/311 ảnh, 13/2/3 session; metadata khớp evidence.
- [x] E0 có đủ 311 raw output; E0 được retry tái sử dụng từ lần chạy trước, có hash nguồn.
- [x] Mask audit: 220 token ảnh, 369 token tổng, 57 token giám sát, prefix equality.
- [x] Chọn E3 bằng validation: F1 E1/E2/E3 = 0,201537/0,286758/0,396290.
- [x] Chấm lại sáu bộ prediction trên máy bằng scorer gốc: micro, CI95, từng lớp và parse rate khớp evidence.
- [x] E4 chỉ khác train_fraction: 142 và 1422 ảnh; 100% tái sử dụng E3. Có paired CI theo ảnh và session.
- [x] REPORT đủ 10 câu, bảng tài nguyên, sáu ảnh GT/pred đã xem và chú thích đúng/sai.
- [x] E1 parse rate 0,990354 ≥0,95; ΔF1(E1−E0)=0,251106 ≥0,10.
- [ ] E5 chưa chạy; không nhận kết quả thực nghiệm GRPO hoặc bonus chỉ từ mã nguồn.

Kết quả kiểm tra cuối và lịch sử Git xem `report/VERIFICATION.md`. Không sửa `scripts/`, `tests/` hoặc `configs/` gốc. Test yêu cầu REPORT còn placeholder được loại đúng tên vì báo cáo đã hoàn thiện; các test khác vẫn giữ nguyên.

Các commit hoàn thiện được tạo tại thời điểm nhập và kiểm tra kết quả; không mô tả thành commit lịch sử trong lúc train. Chưa xác minh trạng thái công khai của GitHub hoặc nộp bài từ xa.
