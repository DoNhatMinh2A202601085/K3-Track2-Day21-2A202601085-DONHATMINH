# BÁO CÁO THỰC HÀNH MLOPS (REFLECTION REPORT)
## Lab: Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục (Day 21 - CI/CD cho AI Systems)

- **Học viên**: Đỗ Nhật Minh
- **Mã sinh viên / ID**: 2A202601085
- **Khóa học**: K3
- **GitHub Repository**: [https://github.com/DoNhatMinh2A202601085/K3-Track2-Day21-2A202601085-DONHATMINH](https://github.com/DoNhatMinh2A202601085/K3-Track2-Day21-2A202601085-DONHATMINH)
- **Hạ tầng Cloud**: Amazon Web Services (AWS S3 + EC2 `47.129.86.251`) & Terraform IaC

---

### 1. Bộ Siêu Tham Số Đã Chọn và Lý Do (Kết Quả Bước 1)

Trong quá trình thực nghiệm cục bộ với **MLflow** (`sqlite:///mlflow.db`) trên tập dữ liệu **Wine Quality** (12 đặc trưng hóa học, phân loại 3 mức chất lượng), tôi đã thử nghiệm nhiều cấu hình siêu tham số khác nhau cho mô hình `RandomForestClassifier`:

| Lần chạy | `n_estimators` | `max_depth` | `min_samples_split` | `accuracy` | `f1_score` (weighted) | Nhận xét |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Run 1 | 50 | 3 | 5 | 0.5580 | 0.5185 | Cây quá nông, mô hình bị underfitting. |
| Run 2 | 100 | 5 | 2 | 0.5640 | 0.5534 | Cấu hình mặc định ban đầu. |
| Run 3 | 200 | 10 | 5 | 0.6440 | 0.6417 | Độ chính xác tăng lên đáng kể. |
| **Run 4 (Tốt nhất)** | **200** | **20** | **2** | **0.6840** | **0.6830** | **Bộ tham số tối ưu được chọn lưu vào `params.yaml`.** |

- **Lý do lựa chọn**:
  - `n_estimators = 200` tạo đủ số lượng cây để giảm thiểu phương sai (variance) và ổn định kết quả dự đoán.
  - `max_depth = 20` và `min_samples_split = 2` cho phép cây phân nhánh đủ sâu để nắm bắt mối quan hệ phi tuyến phức tạp giữa các nồng độ hóa học (axit, đường, sunphat, độ cồn) mà vẫn kiểm soát tốt hiện tượng overfitting.
  - Cấu hình này đạt điểm số cao nhất trên tập dữ liệu đánh giá độc lập (`eval.csv`).

---

### 2. Tự Động Hóa CI/CD & Huấn Luyện Liên Tục (Bước 2 & 3)

- **Hạ tầng & CI/CD**: Xây dựng pipeline GitHub Actions gồm 4 jobs liên hoàn: **Unit Test** $\rightarrow$ **Train** (DVC pull từ AWS S3 `wine-mlops-donhatminh`) $\rightarrow$ **Eval Gate** ($\ge 0.65$) $\rightarrow$ **Deploy** (SSH restart FastAPI trên AWS EC2 `47.129.86.251`). Hạ tầng EC2 và Security Group được tự động hóa 100% bằng **Terraform**.
- **Hiệu quả Continuous Training**: Khi bổ sung 2998 mẫu mới từ `train_phase2.csv` (tổng 5996 mẫu), một `git push` duy nhất đã kích hoạt pipeline tự động huấn luyện lại:
  - **Accuracy**: Tăng từ **0.6840 lên 0.7540 (+7.00%)**.
  - **F1 Score**: Tăng từ **0.6830 lên 0.7534 (+7.04%)**.
  - Mô hình mới vượt qua Eval Gate ($\ge 0.70$) và được tự động cập nhật lên endpoint `POST /predict`.

---

### 3. Hoàn Thành 5 Thách Thức Nâng Cao (Bonus: +20 Điểm)

1. **Bonus 1 (Tracking MLflow Từ Xa & DagsHub - 4đ)**: Tích hợp biến môi trường `MLFLOW_TRACKING_URI` trong `src/train.py` cho phép chuyển đổi linh hoạt giữa backend SQLite cục bộ và remote tracking server (DagsHub/S3).
2. **Bonus 2 (Thí Nghiệm Nhiều Thuật Toán - 4đ)**: Bổ sung tham số `model_type` vào `params.yaml`, hỗ trợ `RandomForest`, `ExtraTrees` (0.7420), `GradientBoosting` (0.6980) và `LogisticRegression` (0.5240) có thể so sánh trực tiếp trên MLflow.
3. **Bonus 3 (Báo Cáo Hiệu Suất Tự Động - 4đ)**: Tự động tính toán Confusion Matrix, Precision, Recall, F1 theo từng lớp và xuất ra `outputs/report.txt`, được lưu trữ trong GitHub Artifacts sau mỗi lần chạy pipeline.
4. **Bonus 4 (Hoàn Trả / So Sánh Phiên Bản Trước - 4đ)**: Job `eval` trong `mlops.yml` tự động tải `metrics.json` của phiên bản đang chạy từ AWS S3, so sánh `new_accuracy` với `prev_accuracy` và cảnh báo/chặn deploy nếu mô hình mới bị suy giảm hiệu năng.
5. **Bonus 5 (Cảnh Báo Lệch Lạc Dữ Liệu - 4đ)**: Kiểm tra phân phối tỷ lệ các nhãn trước khi fit mô hình (Lớp 0: 36.86%, Lớp 1: 43.51%, Lớp 2: 19.63%), in cảnh báo nếu có lớp $< 10\%$ và ghi nhận `class_distribution` vào `outputs/metrics.json` và MLflow metrics.

---

### 4. Khó Khăn Gặp Phải và Cách Giải Quyết

1. **Đồng bộ DVC Remote với AWS S3 trong GitHub Actions**:
   - *Vấn đề*: Pipeline bị lỗi khi `dvc pull` do thiếu cấu hình remote hoặc credentials trên Runner sạch.
   - *Giải pháp*: Cấu hình AWS IAM credentials an toàn qua GitHub Secrets (`CLOUD_CREDENTIALS`), tích hợp thư viện `dvc-s3` và `boto3`, đồng thời xây dựng cơ chế fallback thông minh để đảm bảo pipeline luôn hoạt động trơn tru.
2. **Khởi tạo và quản lý hạ tầng Cloud**:
   - *Vấn đề*: Thao tác cấu hình EC2, Security Group (mở port 22 và 8000) và SSH Key thủ công tốn thời gian.
   - *Giải pháp*: Sử dụng **Terraform** (`terraform/main.tf`) tự động sinh cặp khóa RSA 4096-bit, cấp phát EC2 `t3.micro` tại region `ap-southeast-1` và thiết lập systemd service `mlops-serve` chỉ bằng một câu lệnh `terraform apply`.
3. **Xử lý kết nối SSH trong bước Deploy**:
   - *Vấn đề*: Secret `VM_HOST` bị dính ký tự xuống dòng và máy chủ EC2 mới cần thời gian sẵn sàng.
   - *Giải pháp*: Bổ sung bước làm sạch chuỗi IP và cơ chế SSH retry loop trong workflow bash script để đảm bảo kết nối 100% ổn định.
