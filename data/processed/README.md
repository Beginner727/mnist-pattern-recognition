# Dữ liệu sinh từ mã

`full_indices.npz`, `quick_indices.npz`: các chỉ số vào dữ liệu gốc. Sinh lại bằng `python -m mnist_project.experiment --mode full` hoặc `--mode quick`. Seed 42, phân tầng theo nhãn. Chuẩn hóa ảnh được thực hiện trong RAM bằng phép chia 255.
