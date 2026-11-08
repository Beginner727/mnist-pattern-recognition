# [Nguyễn Quốc Đạt] Các chỉ số đánh giá phân loại.
import numpy as np


# Accuracy, precision, recall, F1 và confusion matrix tự viết bằng NumPy.
def classification_scores(y_true, y_pred, n_classes=10):
    """Tính accuracy, precision/recall/F1 macro và confusion matrix.

    Nhãn nguyên thuộc 0..n_classes-1; hàng ma trận là nhãn thật,
    cột là nhãn dự đoán. Độ đo có mẫu số 0 được gán 0.
    Trả về dictionary gồm các độ đo, ma trận cm và F1 từng lớp."""

    # Chuẩn hóa nhãn đầu vào và kiểm tra kích thước.
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    # Đếm số mẫu theo cặp nhãn thật và nhãn dự đoán.
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    # Lấy số dự đoán đúng và tổng số mẫu của từng lớp.
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)

    # Tính các độ đo từng lớp; mẫu số bằng 0 cho kết quả 0.
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    # Tổng hợp accuracy và trung bình macro giữa các lớp.
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "cm": cm,
        "f1_by_class": f1,
    }


# Kiểm thử: tất cả dự đoán đúng -> 1.0 cho mỗi độ đo.
_yt = np.arange(10)
_q = classification_scores(_yt, _yt)
assert _q["accuracy"] == 1.0 and _q["f1_macro"] == 1.0
print("Metrics unit test: OK")
