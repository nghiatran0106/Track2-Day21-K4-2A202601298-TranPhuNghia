# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Trần Phú Nghĩa |
| MSSV | 2A202601298 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/nghiatran0106/Track2-Day21-K4-2A202601298-TranPhuNghia |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 200 | 0.05 | 5 | 0.7037 | 0.8720 |
| 5 | 100 | 0.2 | 5 | 0.7207 | 0.8760 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=5`.

**Lý do:** Bộ này đạt `f1_score` cao nhất trong năm lần chạy (0,7207), vượt xa ngưỡng 0,65 nên chắc chắn qua được quality gate. Điều đáng chú ý là lần chạy có accuracy cao nhất lại là lần 1 (0,8780) chứ không phải lần 5 — nếu tôi đặt ngưỡng trên accuracy thì đã chọn nhầm một mô hình có F1 thấp hơn 0,01. Về đánh đổi giữa hai tham số: lần 2 dùng `learning_rate=0.05` với chỉ 50 cây nên mô hình chưa kịp hội tụ, F1 rớt xuống 0,6051 và sẽ bị quality gate chặn. Khi tăng số cây lên 200 để bù cho `learning_rate` thấp (lần 4), F1 hồi phục lên 0,7037. Ngược lại, `learning_rate=0.2` cho phép đạt kết quả tốt nhất chỉ với 100 cây, tức là huấn luyện nhanh gấp đôi lần 3 mà vẫn tốt hơn.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult mất cân bằng lớp: chỉ 24,8% số mẫu thuộc lớp thu nhập trên 50K. Vì vậy một mô hình vô dụng, luôn trả lời "thu nhập thấp" cho mọi đầu vào, vẫn đạt accuracy 0,752 — con số này gây hiểu nhầm vì nó chỉ phản ánh tỷ lệ của lớp đa số chứ không chứng minh mô hình học được gì. Chính mô hình đó có `f1_score` bằng 0, và một quality gate đặt trên accuracy sẽ vui vẻ cho nó đi thẳng ra sản phẩm. F1 của lớp dương là trung bình điều hòa của precision và recall tính riêng cho nhóm thu nhập cao, nên nó đo đúng thứ ta quan tâm: mô hình bắt được bao nhiêu trường hợp thu nhập cao và trong số dự đoán dương có bao nhiêu là đúng. Kết quả thực nghiệm của tôi xác nhận điều này bằng số: accuracy giữa năm lần chạy chỉ dao động 0,032 (0,8460 - 0,8780) trong khi f1_score chênh nhau 0,1156 (0,6051 - 0,7207), tức là gấp hơn ba lần rưỡi. Cũng vì thế không được truyền `average="weighted"` hay `average="macro"`: cả hai đều cộng gộp điểm của lớp đa số vào kết quả, kéo giá trị lên cao một cách giả tạo và làm ngưỡng 0,65 mất hết ý nghĩa.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `import mlflow` báo lỗi `ModuleNotFoundError: No module named 'pkg_resources'` | Máy cá nhân chạy Python 3.12 với setuptools 84, bản này đã bỏ hẳn `pkg_resources` mà mlflow 2.13 vẫn cần | Hạ setuptools về bản dưới 81 trong môi trường ảo; CI runner dùng Python 3.10 nên không gặp lỗi này |
| Cài `dvc[s3]` báo xung đột phiên bản `fsspec` | Gói `gcsfs` còn sót lại từ lần cài thử `dvc[gs]` ghim chặt `fsspec==2025.12.0`, trong khi `s3fs` cần bản 2026.7.0 | Gỡ hẳn `dvc-gs`, `gcsfs`, `google-cloud-storage` khỏi môi trường ảo cho khớp với runner sạch của CI |
| Nguy cơ lộ AWS access key trong log của Actions | GitHub chỉ che giấu nguyên văn chuỗi JSON của secret `STORAGE_CREDENTIALS`, không tự che các trường được tách ra từ chuỗi đó | Gọi `echo "::add-mask::$ACCESS_KEY"` cho từng khóa ngay sau khi tách, trước khi ghi vào `$GITHUB_ENV` |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`, 22.361 mẫu) | 0.7207 | 0.8760 |
| Bước 3 (thêm `train_batch2`, 44.722 mẫu) | 0.7297 | 0.8800 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm f1_score nhích lên 0,009 và accuracy nhích lên 0,004 — một mức cải thiện rất nhỏ. Điều này đúng như dự đoán, vì hai nửa dữ liệu được chia ngẫu nhiên từ cùng một nguồn nên có cùng phân phối, và mô hình đã học gần hết những gì có thể học từ 22.361 mẫu đầu tiên. Giá trị thật sự của Bước 3 không nằm ở con số cao hơn, mà ở chỗ dữ liệu mới đi trọn một vòng từ commit đến API đang phục vụ mà không cần bất kỳ thao tác thủ công nào.
