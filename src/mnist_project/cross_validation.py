import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .models import SoftmaxRegressionScratch
from .data import load_mnist_npz, stratified_train_validation
from .metrics import classification_scores


# [Cả nhóm] Chia fold, validation nội bộ và Hard Voting.
def stratified_kfold_indices(y, n_splits=10, seed=42):
    """Yield cặp chỉ số (train, heldout) cho k fold phân tầng có shuffle.

    Mỗi mẫu thuộc heldout đúng một lần; seed giúp tái lập cách chia.
    Các fold cân bằng kích thước và gần giữ nguyên tỷ lệ từng lớp."""
    y = np.asarray(y)
    if y.ndim != 1 or n_splits < 2:
        raise ValueError("Expected 1D labels and at least two folds")

    # Kiểm tra mỗi lớp có đủ mẫu để chia thành các fold.
    labels, counts = np.unique(y, return_counts=True)
    if not len(labels) or counts.min() < n_splits:
        raise ValueError("Every class must have at least n_splits samples")

    # Xáo trộn có seed và phân phối từng lớp vào các fold.
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    offset = 0
    for label in labels:
        ids = np.flatnonzero(y == label)
        rng.shuffle(ids)
        assignments = (np.arange(len(ids)) + offset) % n_splits
        for f in range(n_splits):
            folds[f].extend(ids[assignments == f])
        offset = (offset + len(ids)) % n_splits

    # Lần lượt giữ một fold đánh giá, các fold còn lại làm train.
    all_ids = np.arange(len(y))
    for held in folds:
        held = rng.permutation(np.asarray(held, dtype=np.int64))
        mask = np.ones(len(y), dtype=bool)
        mask[held] = False
        yield rng.permutation(all_ids[mask]), held


def inner_validation_indices(train_indices, y, seed=42):
    """Tách validation từ 9 fold train; trả chỉ số gốc, không chạm held-out."""
    train_indices = np.asarray(train_indices)
    fit, stop = stratified_train_validation(np.asarray(y)[train_indices], 0.1, seed)
    return train_indices[fit], train_indices[stop]


def hard_vote(predictions, n_classes=10):
    """Đếm phiếu từ ma trận nhãn (n_mô_hình, n_mẫu).

    Trả về (nhãn thắng, bảng số phiếu); hòa phiếu chọn nhãn nhỏ nhất."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[0] == 0:
        raise ValueError("Expected at least one model's predictions")
    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Hard voting requires integer class labels")
    if np.any((predictions < 0) | (predictions >= n_classes)):
        raise ValueError("Class label outside the supported range")

    # Đếm phiếu từng nhãn; argmax xử lý hòa bằng nhãn nhỏ nhất.
    votes = np.zeros((predictions.shape[1], n_classes), dtype=np.int32)
    for p in predictions:
        np.add.at(votes, (np.arange(len(p)), p), 1)
    return votes.argmax(axis=1), votes


class SoftmaxHardVoting:
    def __init__(self, models, n_classes=10):
        """Giữ các mô hình đã train và số lớp; không huấn luyện lại mô hình."""
        if not models:
            raise ValueError("Cannot construct an empty ensemble")
        self.models = models
        self.n_classes = n_classes

    def predict(self, X):
        """Thu nhãn từ từng mô hình và trả về nhãn thắng bằng hard voting."""
        return hard_vote(np.stack([m.predict(X) for m in self.models]), self.n_classes)[
            0
        ]


# [Cả nhóm] Chạy thí nghiệm 10-fold, đánh giá và lưu kết quả.
def run(n_splits=10, epochs=100, lr=0.15, seed=42):
    """Train k mô hình Softmax độc lập trên các fold của 60.000 ảnh MNIST.

    Lưu điểm OOF, model, cấu hình, chỉ số chia tập và biểu đồ.
    Đánh giá tổ hợp trên test chính thức sau khi train; trả về tổ hợp."""

    # Chuẩn bị thư mục lưu kết quả của thí nghiệm 10-fold.
    root = Path(__file__).resolve().parents[2]
    out = root / "result/softmax_10fold"
    out.mkdir(parents=True, exist_ok=True)
    figdir = root / "result/figures/softmax_10fold"
    figdir.mkdir(parents=True, exist_ok=True)

    # Tải dữ liệu và chuẩn hóa cùng thang pixel với pipeline FULL.
    X_train_raw, y_train_all, X_test_raw, y_test_all = load_mnist_npz(
        root / "data/raw/mnist.npz"
    )
    X_development = X_train_raw.reshape(-1, 784).astype(np.float32) / 255
    X_test = X_test_raw.reshape(-1, 784).astype(np.float32) / 255

    # Tạo các fold và bộ đệm dự đoán OOF cho toàn bộ tập phát triển.
    folds = list(stratified_kfold_indices(y_train_all, n_splits, seed))
    oof = np.full(len(y_train_all), -1, dtype=np.int64)
    seen = np.zeros(len(y_train_all), dtype=np.int64)
    models = []
    rows = []
    indices = {}
    history = {}

    # Ghi giao thức chia tập, Early Stopping và seed của các mô hình.
    config = dict(
        n_splits=n_splits,
        n_repeats=1,
        n_models=n_splits,
        shuffle=True,
        stratified=True,
        split_seed=seed,
        model_seeds=[seed + i for i in range(n_splits)],
        epochs=epochs,
        learning_rate=lr,
        batch_size=256,
        weight_decay=1e-4,
        n_development=len(y_train_all),
        n_official_test=len(y_test_all),
        early_stopping=True,
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True,
        inner_validation_fraction=0.1,
        inner_validation_method="sklearn.train_test_split(stratify=y)",
        hyperparameter_selection=(
            "Early stopping on inner validation only; heldout and official test"
            " excluded"
        ),
        tie_break="smallest class label",
        python=platform.python_version(),
        numpy=np.__version__,
        sha256=hashlib.sha256((root / "data/raw/mnist.npz").read_bytes()).hexdigest(),
    )
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def stats_row(model, split, truth, pred):
        """Tính các độ đo, lưu confusion matrix và trả về một dòng kết quả."""
        s = classification_scores(truth, pred)
        # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
        np.savetxt(out / f"cm_{model}_{split}.csv", s["cm"], delimiter=",", fmt="%d")
        return dict(
            model=model,
            split=split,
            n=len(truth),
            **{
                k: s[k]
                for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            },
        )

    # Học mô hình riêng cho mỗi fold với validation nội bộ.
    for f, (train, held) in enumerate(folds, 1):
        assert not np.intersect1d(train, held).size
        indices[f"train_{f:02d}"] = train
        indices[f"heldout_{f:02d}"] = held

        # Tách validation từ phần train; held-out không chọn checkpoint.
        fit_ids, stop_ids = inner_validation_indices(train, y_train_all, seed + f - 1)
        indices[f"fit_{f:02d}"] = fit_ids
        indices[f"early_stop_{f:02d}"] = stop_ids
        model = SoftmaxRegressionScratch(lr=lr, epochs=epochs, seed=seed + f - 1)
        start = time.perf_counter()
        model.fit(
            X_development[fit_ids],
            y_train_all[fit_ids],
            X_development[stop_ids],
            y_train_all[stop_ids],
        )
        fit_seconds = time.perf_counter() - start

        # Đo riêng phần học, validation nội bộ và held-out.
        for split, ids in [
            ("train", fit_ids),
            ("early_stop", stop_ids),
            ("heldout", held),
        ]:
            start = time.perf_counter()
            pred = model.predict(X_development[ids])
            predict_seconds = time.perf_counter() - start
            row = stats_row(f"fold_{f:02d}", split, y_train_all[ids], pred)
            row.update(
                fold=f,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                epochs_run=model.n_epochs_,
                best_epoch=model.best_epoch_,
                best_val_loss=model.best_val_loss_,
            )
            rows.append(row)
            if split == "heldout":
                oof[ids] = pred
                seen[ids] += 1
        models.append(model)
        history[f"fold_{f:02d}"] = model.history_
        with (out / f"softmax_fold_{f:02d}.pkl").open("wb") as stream:
            # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
            pickle.dump(model, stream)
        print(
            f"Fold {f}/{n_splits} finished, heldout"
            f' accuracy={rows[-1]["accuracy"]:.5f}',
            flush=True,
        )
    assert np.all(seen == 1) and np.all(oof >= 0)

    # Lưu chỉ số chia tập, metrics và đường học của các fold.
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    np.savez_compressed(out / "fold_indices.npz", **indices)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")

    # All training is complete before evaluating the official test set.
    start = time.perf_counter()

    # Gộp nhãn của mười mô hình bằng hard voting trên test.
    predictions = np.stack([m.predict(X_test) for m in models])
    voted, votes = hard_vote(predictions, 10)
    predict_seconds = time.perf_counter() - start
    ensemble = SoftmaxHardVoting(models)
    with (out / "hard_voting.pkl").open("wb") as stream:
        # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
        pickle.dump(ensemble, stream)
    summary = [
        stats_row("single_excluded_model", "oof", y_train_all, oof),
        stats_row("hard_voting_10_softmax", "official_test", y_test_all, voted),
    ]
    summary[1]["predict_seconds"] = predict_seconds
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    pd.DataFrame(
        [
            stats_row(f"fold_{i+1:02d}", "official_test", y_test_all, p)
            for i, p in enumerate(predictions)
        ]
    ).to_csv(out / "individual_test_metrics.csv", index=False)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    np.savez_compressed(
        out / "predictions.npz",
        oof=oof,
        y_development=y_train_all,
        individual_test=predictions,
        hard_voting=voted,
        y_test=y_test_all,
        vote_counts=votes,
    )

    # Tổng hợp độ biến thiên giữa fold và số trường hợp hòa phiếu.
    held_rows = pd.DataFrame(rows).query("split == 'heldout'")
    ties = int(np.sum((votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1))
    aggregate = dict(
        heldout_accuracy_mean=float(held_rows.accuracy.mean()),
        heldout_accuracy_std=float(held_rows.accuracy.std(ddof=1)),
        heldout_f1_macro_mean=float(held_rows.f1_macro.mean()),
        test_vote_ties=ties,
        fit_seconds_total=sum(
            r["fit_seconds"] for r in rows if r["split"] == "heldout"
        ),
    )
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (out / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Vẽ accuracy held-out theo fold để đối chiếu kết quả.
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        held_rows.fold, held_rows.accuracy, marker="o", label="Held-out fold accuracy"
    )
    ax.set(xlabel="Fold", ylabel="Accuracy", xticks=range(1, n_splits + 1))
    ax.legend()
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figdir / "fold_accuracy.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.15)
    args = parser.parse_args()

    # Use the canonical module so pickle never records "__main__.SoftmaxHardVoting".
    from mnist_project.cross_validation import run as run_experiment

    run_experiment(epochs=args.epochs, lr=args.lr)
