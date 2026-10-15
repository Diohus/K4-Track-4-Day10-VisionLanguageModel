# Bẫy kỹ thuật và giới hạn xác minh

1. Giữ nguyên scripts/ và tests/ (rubric trừ 10 điểm nếu sửa). configs/ cũng giữ bản đề; override batch 1 × 8 áp dụng đồng nhất trong mã riêng.
2. E3 phải có visual.blocks.*.attn.proj. Ví dụ ngắn trong work.txt thiếu target này; regex và assertion yêu cầu 348 module thay vì đếm nhầm.
3. Chuyển YOLO cực sát biên 1000: không chỉ tăng x2 rồi clamp; cần đảm bảo x1≤999 và x2>x1 sau làm tròn.
4. Mask prompt/ảnh/padding bằng −100. Token ảnh là placeholder của đặc trưng, không phải mục tiêu tái tạo pixel. Loss assistant vẫn truyền gradient về ViT khi mở train.
5. Không hard-code pad token ID. Collator giữ mm_token_type_ids nếu processor cung cấp và concatenate pixel_values/image_grid_thw.
6. Không lượng tử hóa visual/lm_head. Projector và LoRA trainables ở fp32; forward autocast fp16. Không gọi prepare_model_for_kbit_training một cách mù quáng làm toàn bộ vision fp32 tăng VRAM.
7. PEFT save_pretrained không tự lưu full projector. Lưu riêng projector.pt, kiểm tra key/shape khi nạp; inference thực hiện trong process mới để kiểm tra vòng lưu/nạp.
8. Train dừng nếu loss không hữu hạn; gradient overflow được giảm scale và thử lại tối đa 6 lần, sau đó dừng nếu vẫn lỗi. Không đếm retry thành optimizer step thành công. Checkpoint đầy đủ giúp tiếp tục cùng sampling và RNG.
9. Test chỉ dùng đánh giá cuối; chọn best E1–E3 bằng validation. E4 chỉ đổi train_fraction; không đồng thời đổi rank/LR/decoding.
10. Giữ 300 steps cho 10% train nghĩa là số epoch cao hơn; báo cáo phải nêu điều kiện này.
11. max_new_tokens=512 dùng đồng nhất E0–E4 để giảm rủi ro cắt JSON; log số output chạm giới hạn. Không “sửa” raw JSON lỗi trước chấm.
12. VRAM là allocated peak / 2**30 (GiB); training time tính vòng train và checkpoint giữa kỳ, không tính download/load model. E0 không train, tài nguyên train bằng 0.
13. CI chính thức bootstrap theo ảnh có thể lạc quan do frame cùng session tương quan; bổ sung paired image/session bootstrap. CI khác seed chưa được đo.
14. Checker gốc không kiểm tra hai ngưỡng E1 trong rubric và kiểm tra E4 khá lỏng; audit riêng kiểm tra split, đủ ảnh, version, missing, parse và gain.
15. Test mẫu REPORT yêu cầu dấu <điền còn tồn tại, checker lại yêu cầu hết placeholder: giữ nguyên test và ghi rõ loại đúng test mẫu đó khi REPORT đã hoàn thiện.
16. Không điền số minh họa trong work.txt vào meta. REPORT.generated.md chỉ sinh khi evidence khớp raw prediction và resource log.
17. Notebook dùng GPU 0 dù Kaggle cấp hai T4; memory không cộng thành 32 GB. GRPO bonus chưa được xác nhận GPU trong phiên chỉnh mã; không có log thì không tự nhận 10 điểm bonus.
18. Python hệ thống đã được hoàn nguyên: gỡ 8 gói mới cài trong phiên; mọi thư viện kiểm thử nằm trong .venv. Kaggle cài trực tiếp vào runtime theo yêu cầu; chỉ local dùng .venv.
19. Đã có kết quả GPU từ phiên retry COMPLETE và đã chấm lại raw predictions trên máy. Máy local chỉ khôi phục splits, không cần tải model hoặc toàn bộ ảnh để chấm lại. Không coi unit test là thay thế thí nghiệm GPU.

20. Dữ liệu thực tế có nhãn YOLO width/height bằng 0, loader gốc cho phép. Không chặn audit hay loại nhãn; target JSON mở rộng tối thiểu 1/1000 theo quy tắc sửa box của lab. Ghi zero_area_annotations.json, giữ nguyên GT của scorer; các GT này không thể đạt IoU 0.5.

21. Transformers 5.19.0 should_convert_module khớp prefix đầy đủ: skip "visual" không chặn model.visual.blocks.*. Dùng llm_int8_skip_modules=["model.visual", "lm_head"], giữ assertion để phát hiện vision bị lượng tử hóa. Không bỏ assertion hoặc dequantize weight đã nén để thay cho nạp lại weight fp16 gốc.
