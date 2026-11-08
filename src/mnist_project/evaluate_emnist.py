# [Cả nhóm] Đánh giá mô hình MNIST đã lưu trên EMNIST, không huấn luyện lại.
"""Đánh giá EMNIST Digits test bằng trọng số cố định và so sánh MNIST test."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import os
import shutil
import subprocess
import urllib.error
import struct
import sys
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import classification_scores

ROOT = Path(__file__).resolve().parents[2]
URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
FILES = (
    "emnist-digits-test-images-idx3-ubyte.gz",
    "emnist-digits-test-labels-idx1-ubyte.gz",
)
SCORES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def file_hash(path):
    """Tính SHA256 theo khối để ghi nguồn dữ liệu và mô hình."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_archive(destination):
    """Tải HTTPS; dùng curl xác thực chứng chỉ nếu urllib gặp lỗi kết nối."""
    try:
        urllib.request.urlretrieve(URL, destination)
    except (urllib.error.URLError, OSError) as error:
        curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
        if not curl:
            raise RuntimeError(
                "Không tải được EMNIST qua HTTPS. Hãy tải ZIP từ NIST bằng trình "
                "duyệt, đặt vào --data-dir rồi chạy lại. Lỗi gốc: " + str(error)
            ) from error
        print("Python HTTPS failed; retrying with curl (TLS verification enabled).",
              flush=True)
        try:
            subprocess.run(
                [curl, "--location", "--fail", "--retry", "2",
                 "--connect-timeout", "30", "--max-time", "1800",
                 "--output", str(destination), URL],
                check=True,
            )
        except subprocess.CalledProcessError as curl_error:
            raise RuntimeError(
                "Cả Python và curl đều không tải được EMNIST. Kiểm tra mạng/ngày "
                "giờ máy hoặc tải ZIP từ NIST bằng trình duyệt vào --data-dir. "
                "File .part chưa hoàn tất không được dùng để đánh giá."
            ) from curl_error


def prepare_data(directory, download=False):
    """Tải ZIP chính thức nếu được yêu cầu, chỉ lấy hai tệp Digits test."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in FILES]
    if all(path.is_file() for path in paths):
        return paths
    archive = directory / "gzip.zip"
    if archive.exists() and not zipfile.is_zipfile(archive):
        if not download:
            raise ValueError("gzip.zip bị hỏng hoặc tải chưa đủ. Chạy lại với --download.")
        print("Invalid gzip.zip; downloading a complete replacement.", flush=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        if not download:
            raise FileNotFoundError(
                "Thiếu EMNIST Digits test. Chạy lại với --download hoặc đặt "
                f"gzip.zip chính thức vào {directory}."
            )
        partial = directory / "gzip.zip.part"
        print("Downloading official EMNIST archive (~536 MiB)...", flush=True)
        download_archive(partial)
        with zipfile.ZipFile(partial) as package:
            for name in FILES:
                package.getinfo("gzip/" + name)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as package:
        for path in paths:
            # Chỉ ghi tên tệp cố định, không giải nén đường dẫn từ ZIP.
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(package.read("gzip/" + path.name))
            temporary.replace(path)
    return paths


def read_test(images_path, labels_path):
    """Đọc IDX gốc và transpose ảnh để đưa chữ số về hướng đứng chuẩn."""
    with gzip.open(images_path, "rb") as stream:
        magic, count, height, width = struct.unpack(">IIII", stream.read(16))
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    with gzip.open(labels_path, "rb") as stream:
        label_magic, label_count = struct.unpack(">II", stream.read(8))
        labels = np.frombuffer(stream.read(), dtype=np.uint8)
    if (magic, height, width) != (2051, 28, 28) or label_magic != 2049:
        raise ValueError("Sai định dạng IDX ảnh hoặc nhãn.")
    if count != 40000 or label_count != count or len(labels) != count:
        raise ValueError("Cần đầy đủ 40.000 ảnh EMNIST Digits test.")
    if pixels.size != count * 784:
        raise ValueError("Kích thước dữ liệu ảnh không khớp header IDX.")
    if not np.array_equal(np.bincount(labels, minlength=10), [4000] * 10):
        raise ValueError("EMNIST Digits test phải có 4.000 mẫu mỗi nhãn 0–9.")
    X_test_raw = pixels.reshape(count, 28, 28).transpose(0, 2, 1).copy()
    return X_test_raw, labels.astype(np.int64)


def predict_batches(model, X_raw, batch_size):
    """Chuẩn hóa giống experiment.py và dự đoán theo batch để hạn chế RAM."""
    predictions = []
    for start in range(0, len(X_raw), batch_size):
        X = X_raw[start:start + batch_size].reshape(-1, 784).astype("float32") / 255
        predictions.append(model.predict(X))
    return np.concatenate(predictions)


def run(data_dir, download=False, batch_size=1024):
    """Đánh giá hai tập test bằng cùng models.pkl, chỉ ghi result/emnist."""
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn 0.")
    model_path = ROOT / "result/full/models.pkl"
    mnist_path = ROOT / "data/raw/mnist.npz"
    model_sha = file_hash(model_path)
    paths = prepare_data(data_dir, download)
    X_emnist_raw, y_emnist = read_test(*paths)
    with np.load(mnist_path, allow_pickle=False) as archive:
        X_mnist_raw = archive["x_test"]
        y_mnist = archive["y_test"]
    # Chỉ nạp pickle do chính dự án tạo ra.
    with model_path.open("rb") as stream:
        models = pickle.load(stream)
    out = ROOT / "result/emnist"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in models.items():
        for dataset, X_raw, labels in (
            ("MNIST", X_mnist_raw, y_mnist),
            ("EMNIST Digits", X_emnist_raw, y_emnist),
        ):
            print(f"Evaluating {name} on {dataset}: {len(labels)} images", flush=True)
            pred = predict_batches(model, X_raw, batch_size)
            scores = classification_scores(labels, pred)
            rows.append({"model": name, "dataset": dataset, "n": len(labels),
                         **{key: scores[key] for key in SCORES}})
            slug = name.replace(" ", "_")
            dataset_slug = dataset.replace(" ", "_")
            np.savetxt(out / f"cm_{slug}_{dataset_slug}.csv", scores["cm"],
                       delimiter=",", fmt="%d")
    metrics = pd.DataFrame(rows)
    comparison = metrics[metrics.dataset == "MNIST"].merge(
        metrics[metrics.dataset == "EMNIST Digits"], on="model",
        suffixes=("_mnist", "_emnist"), validate="one_to_one",
    )
    for key in SCORES:
        comparison[key + "_delta_pp"] = 100 * (
            comparison[key + "_emnist"] - comparison[key + "_mnist"]
        )
    metrics.to_csv(out / "metrics.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)

    # Lưu mẫu đã sửa hướng để kiểm tra đầu vào độc lập với điểm số.
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for digit in range(10):
        indices = np.flatnonzero(y_emnist == digit)[:2]
        for row, index in enumerate(indices):
            axes[row, digit].imshow(X_emnist_raw[index], cmap="gray", vmin=0, vmax=255)
            axes[row, digit].set_title(str(digit))
            axes[row, digit].axis("off")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=150)
    plt.close(fig)
    ax = comparison.set_index("model")[["accuracy_mnist", "accuracy_emnist"]].plot.bar(
        figsize=(8, 4), ylim=(0, 1), rot=0,
    )
    ax.set_ylabel("Accuracy")
    ax.legend(["MNIST test", "EMNIST Digits test"])
    ax.figure.tight_layout()
    ax.figure.savefig(out / "comparison.png", dpi=150)
    plt.close(ax.figure)
    if file_hash(model_path) != model_sha:
        raise RuntimeError("models.pkl đã thay đổi trong lúc đánh giá; hãy chạy lại.")
    provenance = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": URL, "split": "digits/test", "model_sha256": model_sha,
        "mnist_sha256": file_hash(mnist_path),
        "input_sha256": {path.name: file_hash(path) for path in paths},
        "preprocessing": "IDX transpose(0,2,1); float32 / 255; flatten 784",
        "training": False, "hyperparameter_selection": False,
        "comparison": "Both test sets re-evaluated using the same frozen models.pkl",
        "limitation": "Shared NIST origin; source-sample/writer overlap not verified",
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(comparison[["model", "accuracy_mnist", "accuracy_emnist",
                      "accuracy_delta_pp"]].to_string(index=False))
    return metrics


def main():
    """Đọc tham số CLI và khởi chạy đánh giá không huấn luyện."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    parser.add_argument(
        "--data-dir", type=Path, default=cache_root / "mnist-midterm/emnist",
        help="Thư mục EMNIST; mặc định cache ngoài đồ án.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    run(args.data_dir, args.download, args.batch_size)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
