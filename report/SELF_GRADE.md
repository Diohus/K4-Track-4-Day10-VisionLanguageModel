# Tự chấm theo rubric
A tự động xác nhận được: **40/40** (chưa chạy thì không có điểm thực nghiệm).
Chi tiết: {'e0_e3': 12, 'meta_e1_e3': 6, 'e1_parse': 4, 'e1_gain': 6, 'e4': 6, 'consistent': 6}.

B /20: đối chiếu REPORT.generated.md với evidence; xem sáu ảnh đúng/sai, ablation.json và log tài nguyên.
C /30: tác giả cần đọc, bổ sung diễn giải cá nhân cho 10 câu; mã sinh báo cáo không thay thế đánh giá nội dung.
D /10: cần ≥5 commit thật theo checkpoint, repo sạch và checker 0 FAIL. Không tính lịch sử giả.
Bonus /10: reward có test là phần chuẩn bị; chỉ nhận điểm chạy GRPO khi có log/evidence thật và phân tích CI/reward hacking.

Không thể xác nhận điểm tối đa trước khi có kết quả GPU và đánh giá thủ công.

## Checker gốc
- [PASS] evidence e0: 
- [PASS] evidence e1: 
- [PASS] evidence e2: 
- [PASS] evidence e3: 
- [PASS] evidence e4: 2 file
- [PASS] e1 meta: 
- [PASS] e2 meta: 
- [PASS] e3 meta: 
- [PASS] dataset_version nhất quán: ['d10-21e4070301']
- [PASS] khớp data/splits/meta.json: 
- [PASS] REPORT.md đã điền: còn chuỗi '<điền'
- [PASS] git commits: 5/5
- [PASS] git forbidden files: 
