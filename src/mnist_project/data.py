# [Nguyễn Duy Khánh] Tải, kiểm tra và chia dữ liệu MNIST.
from pathlib import Path
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw")

# Tải MNIST thật; không tự động thay bằng sklearn Digits nếu thiếu dữ liệu.
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist_npz(path=DATA_DIR / "mnist.npz"):
    """Trả ảnh gốc X_train_raw, X_test_raw và nhãn y_train_all, y_test_all.

    Ảnh có shape (60000, 28, 28) và (10000, 28, 28), chưa chuẩn hóa.
    """

    # Xác định file dữ liệu và tải về nếu chưa có sẵn.
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Đang tải MNIST lần đầu vào:", path)
        try:
            urllib.request.urlretrieve(MNIST_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Không tải được MNIST. Hãy tải mnist.npz và đặt vào data/raw/mnist.npz "
                f"rồi chạy lại notebook. URL: {MNIST_URL}"
            ) from exc

    # Đọc bốn mảng dữ liệu và kiểm tra định dạng MNIST.
    with np.load(path, allow_pickle=False) as mnist_archive:
        X_train_raw = mnist_archive["x_train"]
        y_train_all = mnist_archive["y_train"]
        X_test_raw = mnist_archive["x_test"]
        y_test_all = mnist_archive["y_test"]
    assert X_train_raw.shape == (60000, 28, 28), "File không đúng định dạng MNIST 60k."
    assert X_test_raw.shape == (10000, 28, 28), "File không đúng định dạng MNIST 10k."
    return (
        X_train_raw,
        y_train_all.astype(np.int64),
        X_test_raw,
        y_test_all.astype(np.int64),
    )


def stratified_train_validation(y, val_fraction=0.1, seed=42):
    """Chia chỉ số train/validation bằng sklearn, có shuffle và phân tầng.

    Chỉ truyền nhãn từ tập training chính thức; giữ test MNIST riêng.
    Trả hai mảng chỉ số để các pipeline dùng chung cách chia dữ liệu.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("Expected one-dimensional labels")

    # Phân tầng theo nhãn và cố định seed để tái lập cùng một cách chia.
    indices = np.arange(len(y))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return train_indices, validation_indices


def stratified_limit(indices, y_all, limit, seed=42):
    """Lấy tập con gần đúng tỷ lệ lớp; cần thiết cho chế độ thử code nhanh."""
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or len(indices) <= limit:
        return indices

    # Dùng seed cố định để tái lập việc xáo trộn chỉ số.
    rng = np.random.default_rng(seed)
    selected = []
    labels, counts = np.unique(y_all[indices], return_counts=True)

    # Phân bổ số mẫu mỗi lớp theo tỷ lệ và bù phần làm tròn.
    alloc_float = limit * counts / counts.sum()
    allocations = np.floor(alloc_float).astype(int)
    for k in np.argsort(-(alloc_float - allocations))[: limit - allocations.sum()]:
        allocations[k] += 1

    # Lấy mẫu không hoàn lại trong từng lớp.
    for cls, n in zip(labels, allocations):
        pool = indices[y_all[indices] == cls]
        selected.extend(rng.choice(pool, size=n, replace=False))
    return rng.permutation(selected)
