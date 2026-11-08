# Nhận dạng chữ số viết tay MNIST

## 1. Giới thiệu

Đồ án giữa kỳ môn Nhận dạng mẫu: xây dựng và so sánh bốn mô hình phân loại chữ số 0–9, kèm demo nét vẽ và đánh giá trên EMNIST Digits.

- **Multinomial Naïve Bayes, Softmax Regression và MLP:** tự cài đặt bằng NumPy, bao gồm feedforward và backpropagation của MLP.
- **Random Forest:** dùng `RandomForestClassifier` của scikit-learn trong `models.py`.
- Chia dữ liệu bằng scikit-learn; tự tính các độ đo trong `metrics.py`. Có thêm mô hình thư viện làm baseline đối chiếu.

## 2. Cấu trúc dự án

```text
DoAnGiuaKy/
├── src/mnist_project/   Mô hình, huấn luyện, đánh giá và demo
├── data/               Dữ liệu gốc và chỉ số chia tập
├── result/
│   ├── full/           Mô hình và kết quả MNIST
│   ├── softmax_10fold/ Kết quả 10-fold Hard Voting
│   ├── emnist/         Đánh giá EMNIST bằng trọng số cố định
│   └── figures/        Biểu đồ MNIST và 10-fold
├── notebooks/          Khám phá và trình bày kết quả
├── tests/              Unit test
├── scripts/            Thực thi và cập nhật notebook
├── reference/          Đề bài và phân công
├── requirements.txt    Phụ thuộc thư viện
└── pyproject.toml      Cấu hình package Python
```

**Thứ tự đọc:** README → [notebook kết quả](notebooks/Ket_qua_chinh.ipynb) → [experiment.py](src/mnist_project/experiment.py) → [data.py](src/mnist_project/data.py) → [models.py](src/mnist_project/models.py) → [metrics.py](src/mnist_project/metrics.py).

[demo.py](src/mnist_project/demo.py) chứa giao diện vẽ; [cross_validation.py](src/mnist_project/cross_validation.py) chạy 10-fold; [evaluate_emnist.py](src/mnist_project/evaluate_emnist.py) đánh giá EMNIST.

## 3. Cài đặt và chạy

Yêu cầu **Python 3.12 trở lên**; môi trường thực nghiệm dùng Python 3.13.5. Demo cần Tkinter. Mở PowerShell tại thư mục gốc dự án:

```powershell
# Tạo môi trường Python riêng cho đồ án trong thư mục .venv
python -m venv .venv

# Cài các thư viện trong requirements.txt vào đúng môi trường .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Cài package dự án từ thư mục hiện tại (.) để chạy được mnist_project
# -e: dùng trực tiếp code trong src; sửa code không cần cài lại
# --no-deps: không cài lại phụ thuộc vì đã cài ở lệnh trên
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# Unit test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Huấn luyện FULL và baseline
.\.venv\Scripts\python.exe -m mnist_project.experiment --mode full --baseline

# Thí nghiệm 10-fold riêng
.\.venv\Scripts\python.exe -m mnist_project.cross_validation

# Đánh giá trên tập EMNIST bằng mô hình hiện tại
.\.venv\Scripts\python.exe -m mnist_project.evaluate_emnist --download


# Demo nét vẽ
.\.venv\Scripts\python.exe -m mnist_project.demo
```

Chỉ cần cài package một lần trong mỗi môi trường `.venv`; không cần đặt `PYTHONPATH`.
Cài đặt editable dùng trực tiếp mã trong `src`, nên sửa code không cần cài lại.

MNIST được tải tự động khi thiếu. EMNIST mặc định lưu **ngoài đồ án**.

Git không lưu `.venv`, dữ liệu lớn hoặc mô hình. Máy mới cần cài lại môi trường và huấn luyện lại, hoặc nhận riêng bộ dữ liệu/mô hình đã lưu từ nhóm.

Để đồng bộ kết quả hiển thị, **Run All rồi Save** các notebook; giữ `RUN_EXPERIMENT=False` nếu đã huấn luyện bằng lệnh trên. Có thể chạy và lưu notebook chính bằng:

```powershell
.\.venv\Scripts\python.exe scripts/execute_results_notebook.py --notebook notebooks/Ket_qua_chinh.ipynb
```

Nếu trọng số thay đổi, chạy lại EMNIST trước khi xuất báo cáo. Notebook, PDF và PowerPoint không tự cập nhật khi train.

## 4. Dữ liệu và thiết kế thực nghiệm

- **MNIST:** ảnh xám 28×28, gồm 60.000 ảnh phát triển và 10.000 ảnh test chính thức.
- Chia phân tầng, có shuffle: **54.000 train / 6.000 validation / 10.000 test**.
- Flatten thành 784 đặc trưng, chuyển float32 và chia 255 về `[0, 1]`.
- Trong `experiment.py`, chọn alpha NB, learning rate Softmax và độ rộng lớp ẩn
  đầu MLP bằng Macro F1 validation; test dùng đánh giá cuối. MLP luôn có hai lớp ẩn.
  Khảo sát lưu vào `nb_alpha_sweep.csv`, `lr_sweep.csv`, `mlp_hidden_sweep.csv`
  trong `result/full/` (hoặc `result/quick/`). Cấu hình được chọn lưu trong `config.json`.
  MLP chính và khảo sát trong experiment giữ LR=0.7; lớp ẩn thứ hai cố định 64 neuron.
- Softmax và MLP tối đa 100 epoch; Early Stopping theo validation loss, `patience=10`, `min_delta=0.0001`, khôi phục trọng số tốt nhất.
- Seed cơ sở 42; cấu hình lưu tại `result/full/config.json`. Tái lập cần giữ dữ liệu, cấu hình và phiên bản thư viện; thời gian chạy có thể khác giữa các máy.

Chi tiết backpropagation, Early Stopping và chia fold nằm trong báo cáo/notebook. SHA256 trong thông tin nguồn EMNIST giúp kiểm tra kết quả thuộc đúng bộ trọng số.

## 5. Kết quả thực nghiệm

Kết quả trên **10.000 ảnh MNIST test**, lấy từ [metrics.csv](result/full/metrics.csv). Cả hai cột trình bày theo phần trăm.

| Mô hình | Accuracy | Macro F1 |
|---|---:|---:|
| MultinomialNB | 83.56% | 83.41% |
| Softmax LR | 92.29% | 92.18% |
| MLP | 97.81% | 97.79% |
| Random Forest | 95.97% | 95.93% |

**10-fold Hard Voting:** Accuracy **92.44%**, Macro F1 **92.33%** trên test chính thức; nguồn [summary.csv](result/softmax_10fold/summary.csv). Các bảng trong README là số liệu đã lưu, cần cập nhật nếu chạy thí nghiệm mới.

Notebook để xem bảng và biểu đồ:

- [Kết quả chính](notebooks/Ket_qua_chinh.ipynb).
- [Khám phá và kiểm tra mô hình MNIST](notebooks/MNIST_DoAn_GiuaKy.ipynb).
- [Softmax 10-fold Hard Voting](notebooks/Softmax_10Fold_HardVoting.ipynb).

## 6. Thí nghiệm mở rộng

**10-fold Hard Voting.** Chia phân tầng 60.000 ảnh thành 10 fold, huấn luyện 10 Softmax rồi bỏ phiếu nhãn.
Mỗi lượt có validation nội bộ; fold giữ lại dùng đánh giá, không chọn epoch.
Đây là một lượt 10-fold, không phải lặp 10-fold 10 lần; learning rate cấu hình riêng.
Kết quả lưu trong `result/softmax_10fold/`.

**EMNIST Digits.** Đánh giá 40.000 ảnh test bằng trọng số MNIST cố định, đồng thời đo lại MNIST test.
Ảnh IDX được sửa hướng, flatten và chuẩn hóa giống MNIST; không gọi `fit()`.
Accuracy giảm ở cả bốn mô hình trong lần đánh giá đã lưu; xem [bảng so sánh](result/emnist/comparison.csv).
Hai bộ cùng nguồn gốc NIST; chưa xác minh trùng mẫu/người viết nên không coi là nguồn hoàn toàn độc lập.

## 7. Giới hạn và tài liệu tham khảo

Demo hiển thị nét vẽ và ảnh 28×28 sau tiền xử lý. Khác biệt về nét, vị trí và cách viết có thể làm dự đoán sai; điểm MNIST không bảo đảm chất lượng trên canvas hoặc ảnh chụp. Khảo sát siêu tham số còn hạn chế, chưa đánh giá nhiều seed. Hard Voting và Softmax đơn khác cấu hình nên chỉ so sánh mô tả.

- [Đề bài](reference/DoAn_ThucHanh_NhanDang.pdf).
- Báo cáo được quản lý riêng ngoài thư mục dự án; cập nhật thủ công từ kết quả đã lưu.
- [Phân công công việc](reference/Phan_cong_cong_viec.docx).
- [MNIST](https://yann.lecun.com/exdb/mnist/).
- [EMNIST — NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) và [bài báo EMNIST](https://arxiv.org/abs/1702.05373).
