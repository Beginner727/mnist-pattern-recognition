# [Cả nhóm] Tích hợp huấn luyện, đánh giá và lưu kết quả.
import argparse
import hashlib  # Dùng để tính giá trị băm SHA-256
import json
import pickle   # Lưu các mô hình đã train
import platform # Lấy thông tin phiên bản
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # Tạo và lưu hình ảnh mà không cần mở cửa sổ đồ họa
import matplotlib.pyplot as plt
from .data import load_mnist_npz, stratified_train_validation, stratified_limit
from .models import (
    MultinomialNBScratch,
    SoftmaxRegressionScratch,
    MLPScratch,
    RandomForestModel,
)
from .metrics import classification_scores


def survey_parameter(name, values, X_train, y_train, X_val, y_val, config):
    """Khảo sát NB/MLP trên cùng split; chọn giá trị có Macro F1 validation cao nhất."""
    rows = []
    for index, value in enumerate(values, start=1):
        if name == "MultinomialNB":
            model = MultinomialNBScratch(alpha=value)
            details = f"alpha={value}"
        elif name == "MLP":
            model = MLPScratch(
                hidden_layers=(value, 64), lr=config["mlp_lr"],
                epochs=config["epochs_mlp"], seed=config["seed"],
                early_stopping=config["early_stopping"],
                patience=config["patience"], min_delta=config["min_delta"],
                restore_best_weights=config["restore_best_weights"],
            )
            details = f"hidden=({value}, 64), lr={config['mlp_lr']}"
        else:
            raise ValueError(f"Không hỗ trợ khảo sát {name}")
        print(f"\n[{name} sweep {index}/{len(values)}] {details}", flush=True)
        start = time.perf_counter()
        if name == "MLP":
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        row = {"value": value, "train_seconds": train_seconds}
        for split, X, y in (("train", X_train, y_train),
                            ("validation", X_val, y_val)):
            scores = classification_scores(y, model.predict(X))
            row[f"{split}_accuracy"] = scores["accuracy"]
            row[f"{split}_f1_macro"] = scores["f1_macro"]
        if name == "MLP":
            row.update(hidden_1=value, hidden_2=64, lr=config["mlp_lr"],
                       epochs_run=model.n_epochs_, best_epoch=model.best_epoch_)
        else:
            row["alpha"] = value
        rows.append(row)
    if not rows:
        raise ValueError("Danh sách giá trị khảo sát không được rỗng.")
    best = max(rows, key=lambda row: row["validation_f1_macro"])
    return pd.DataFrame(rows), best["value"]


def run(mode="full", baseline=False):
    """Chạy ba mô hình NumPy và Random Forest scikit-learn; lưu kết quả.

    Chọn learning rate bằng validation; lưu model, metrics và biểu đồ.
    Nếu baseline=True, chạy thêm mô hình sklearn để đối chiếu."""

    # Xác định thư mục lưu mô hình, số liệu và hình kết quả.
    root = Path(__file__).resolve().parents[2]
    out = root / "result" / mode
    out.mkdir(parents=True, exist_ok=True)
    figs = root / "result/figures" / mode
    figs.mkdir(parents=True, exist_ok=True)

    # Tải MNIST và chia chỉ số train, validation, test.
    X_train_raw, y_train_all, X_test_raw, y_test_all = load_mnist_npz(
        root / "data/raw/mnist.npz"
    )
    train_idx, val_idx = stratified_train_validation(y_train_all)
    test_idx = np.arange(len(y_test_all))
    if mode == "quick":
        train_idx = stratified_limit(train_idx, y_train_all, 8000)
        val_idx = stratified_limit(val_idx, y_train_all, 1500)
        test_idx = stratified_limit(test_idx, y_test_all, 2000)

    # Lưu cách chia tập để có thể kiểm tra và tái lập thí nghiệm.
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "data/processed" / f"{mode}_indices.npz",
        train=train_idx,
        validation=val_idx,
        test=test_idx,
    )

    # Làm phẳng ảnh và chuẩn hóa pixel về khoảng [0, 1].
    datasets = {
        "train": (
            X_train_raw[train_idx].reshape(-1, 784).astype("float32") / 255,
            y_train_all[train_idx],
        ),
        "validation": (
            X_train_raw[val_idx].reshape(-1, 784).astype("float32") / 255,
            y_train_all[val_idx],
        ),
        "test": (
            X_test_raw[test_idx].reshape(-1, 784).astype("float32") / 255,
            y_test_all[test_idx],
        ),
    }
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["validation"]

    # Khai báo cấu hình huấn luyện và thông tin môi trường chạy.
    epochs = 100
    config = {
        "mode": mode,
        "seed": 42,
        "epochs_lr": epochs,
        "epochs_mlp": 100,
        "early_stopping": True,
        "patience": 10,
        "min_delta": 1e-4,
        "restore_best_weights": True,
        "hidden": [128, 64],
        "mlp_lr": 0.7,
        "nb_alpha_grid": [0.1, 1.0, 10.0],
        "mlp_hidden_1_grid": [64, 128, 256],
        "selection_metric": "validation_f1_macro",
        "trees": 40 if mode == "full" else 5,
        "nb_alpha": 1.0,
        "max_features": 28,
        "split_implementation": "sklearn.model_selection.train_test_split",
        "forest_n_jobs": -1,
        "forest_implementation": "sklearn.ensemble.RandomForestClassifier",
        "depth": 14 if mode == "full" else 8,
        "n": {k: len(v[1]) for k, v in datasets.items()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "sha256": (
            hashlib.sha256((root / "data/raw/mnist.npz").read_bytes()).hexdigest()
        ),
    }
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Khảo sát learning rate trên validation trước khi đánh giá test.
    trials = []
    candidates = []
    learning_rates = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    for trial_index, lr in enumerate(learning_rates, start=1):
        # Phân biệt từng lượt khảo sát khi epoch bắt đầu lại từ 1.
        print(
            f"\n[LR sweep {trial_index}/{len(learning_rates)}] "
            f"Softmax | lr={lr:g}",
            flush=True,
        )
        m = SoftmaxRegressionScratch(lr=lr, epochs=epochs).fit(
            X_train, y_train, X_val, y_val
        )
        f = classification_scores(y_val, m.predict(X_val))["f1_macro"]
        trials.append(
            {
                "lr": lr,
                "validation_f1_macro": f,
                "epochs_run": m.n_epochs_,
                "best_epoch": m.best_epoch_,
            }
        )
        candidates.append(m)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    pd.DataFrame(trials).to_csv(out / "lr_sweep.csv", index=False)
    best = int(np.argmax([r["validation_f1_macro"] for r in trials]))
    config["chosen_lr"] = trials[best]["lr"]

    # Khảo sát riêng alpha NB và lớp ẩn thứ nhất MLP; không sử dụng test.
    nb_trials, config["nb_alpha"] = survey_parameter(
        "MultinomialNB", config["nb_alpha_grid"],
        X_train, y_train, X_val, y_val, config,
    )
    nb_trials.to_csv(out / "nb_alpha_sweep.csv", index=False)
    mlp_trials, chosen_width = survey_parameter(
        "MLP", config["mlp_hidden_1_grid"],
        X_train, y_train, X_val, y_val, config,
    )
    mlp_trials.to_csv(out / "mlp_hidden_sweep.csv", index=False)
    config["hidden"] = [int(chosen_width), 64]
    config["learning_rate_grid"] = learning_rates
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Khởi tạo bốn mô hình bằng cấu hình đã chọn trên validation.
    models = {
        "MultinomialNB": MultinomialNBScratch(alpha=config["nb_alpha"]),
        "Softmax LR": SoftmaxRegressionScratch(lr=config["chosen_lr"], epochs=epochs),
        "MLP": MLPScratch(
            hidden_layers=config["hidden"], lr=config["mlp_lr"],
            epochs=config["epochs_mlp"], seed=config["seed"]
        ),
        "Random Forest": RandomForestModel(
            n_estimators=config["trees"],
            max_depth=config["depth"],
            max_features=config["max_features"],
            random_state=config["seed"],
            min_samples_leaf=2,
        ),
    }

    # Chuẩn bị nơi gom metrics, dự đoán và lịch sử học.
    rows = []
    predictions = {}
    histories = {}

    # Huấn luyện từng mô hình và đo thời gian fit.
    for name, m in models.items():
        # Báo mô hình chính thức và learning rate thực tế đang sử dụng.
        details = f" | lr={m.lr:g}" if name in ["Softmax LR", "MLP"] else ""
        print(f"\n[Final training] {name}{details}", flush=True)
        t = time.perf_counter()
        # Huấn luyện mạng với validation và Early Stopping.
        if name in ["Softmax LR", "MLP"]:
            m.fit(X_train, y_train, X_val, y_val)
            histories[name] = m.history_
        else:
            m.fit(X_train, y_train)
        train_sec = time.perf_counter() - t

        # Khảo sát F1 validation theo số cây của rừng.
        if name == "Random Forest":
            curve = [
                {
                    "trees": i + 1,
                    "validation_f1_macro": classification_scores(y_val, p)["f1_macro"],
                }
                for i, p in enumerate(m.staged_predict(X_val))
            ]
            # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
            pd.DataFrame(curve).to_csv(out / "forest_curve.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 2.4))
            ax.plot(
                [r["trees"] for r in curve], [r["validation_f1_macro"] for r in curve]
            )
            ax.set(xlabel="Number of trees", ylabel="Validation macro F1")
            fig.tight_layout()
            # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
            fig.savefig(figs / "forest_curve.png", dpi=160)
            plt.close(fig)

        # Đánh giá các tập và lưu confusion matrix cùng các độ đo.
        for split, (X_split, y_split) in datasets.items():
            t = time.perf_counter()
            pred = m.predict(X_split)
            sec = time.perf_counter() - t
            s = classification_scores(y_split, pred)
            predictions[(name, split)] = pred
            # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
            np.savetxt(
                out / f'cm_{name.replace(" ","_")}_{split}.csv',
                s["cm"],
                fmt="%d",
                delimiter=",",
            )
            rows.append(
                dict(
                    model=name,
                    split=split,
                    n=len(y_split),
                    **{
                        k: s[k]
                        for k in [
                            "accuracy",
                            "precision_macro",
                            "recall_macro",
                            "f1_macro",
                        ]
                    },
                    train_seconds=train_sec,
                    predict_seconds=sec,
                    predict_ms_per_image=sec * 1000 / len(y_split),
                )
            )
        print(name, rows[-1], flush=True)

    # Ghi số epoch thực chạy và checkpoint tốt nhất của mạng.
    config["training_summary"] = {
        name: dict(
            epochs_run=m.n_epochs_,
            best_epoch=m.best_epoch_,
            best_val_loss=m.best_val_loss_,
            stopped_epoch=m.stopped_epoch_,
        )
        for name, m in models.items()
        if name in ["Softmax LR", "MLP"]
    }

    # Lưu bảng so sánh, cấu hình, lịch sử và mô hình đã học.
    result = pd.DataFrame(rows)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    result.to_csv(out / "metrics.csv", index=False)
    comparison = result[result.split == "test"].copy()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    comparison.to_csv(out / "comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    comparison.set_index("model")[["accuracy", "f1_macro"]].plot.bar(ax=axes[0], rot=15)
    axes[0].set(ylabel="Test score", ylim=(0, 1))
    comparison.set_index("model")["train_seconds"].plot.bar(ax=axes[1], rot=15)
    axes[1].set(ylabel="Training seconds")
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "comparison.png", dpi=160)
    plt.close(fig)
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    (out / "history.json").write_text(json.dumps(histories), encoding="utf-8")
    with open(out / "models.pkl", "wb") as f:
        # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
        pickle.dump(models, f)

    # Minh họa ảnh MNIST đại diện cho từng chữ số.
    fig, ax = plt.subplots(2, 5, figsize=(9, 3.8))
    for c, a in enumerate(ax.flat):
        a.imshow(X_train_raw[np.flatnonzero(y_train_all == c)[0]], cmap="gray")
        a.set_title(str(c))
        a.axis("off")
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "samples.png", dpi=150)
    plt.close(fig)

    # Vẽ tỷ lệ nhãn để đối chiếu phân bố giữa các tập.
    fig, ax = plt.subplots(figsize=(9, 3.5))
    digits = np.arange(10)
    bar_width = 0.25
    for index, (k, (_, labels)) in enumerate(datasets.items()):
        # Ba cột cạnh nhau giúp phân biệt các tập có tỷ lệ gần bằng nhau.
        positions = digits + (index - (len(datasets) - 1) / 2) * bar_width
        ax.bar(
            positions,
            np.bincount(labels, minlength=10) / len(labels),
            width=bar_width,
            label=k,
        )
    ax.set(xlabel="Digit", ylabel="Class proportion", xticks=digits)
    ax.set_ylim(bottom=0, top=0.14)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "distribution.png", dpi=150)
    plt.close(fig)

    # Vẽ confusion matrix của bốn mô hình trên test.
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for a, name in zip(axes.flat, models):
        cm = np.loadtxt(out / f'cm_{name.replace(" ","_")}_test.csv', delimiter=",")
        a.imshow(cm, cmap="Blues")
        a.set(
            title=name,
            xlabel="Predicted",
            ylabel="True",
            xticks=range(10),
            yticks=range(10),
        )
    for a in list(axes.flat)[len(models):]:
        a.axis("off")
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(
        [r["lr"] for r in trials],
        [r["validation_f1_macro"] for r in trials],
        marker="o",
    )
    ax.set(xlabel="Learning rate", ylabel="Validation macro F1")
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "lr_sweep.png", dpi=160)
    plt.close(fig)

    # Vẽ train loss, validation accuracy và validation loss.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=name)
        axes[1].plot(h["val_accuracy"], label=name)
        axes[2].plot(h["val_loss"], label=name)
    axes[0].set(xlabel="Epoch (index 0)", ylabel="Minibatch mean cross entropy")
    axes[1].set(xlabel="Epoch (index 0)", ylabel="Validation accuracy")
    axes[2].set(xlabel="Epoch (index 0)", ylabel="Validation cross entropy")
    for a in axes:
        a.legend()
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "learning.png", dpi=150)
    plt.close(fig)

    # Chọn mô hình theo validation rồi minh họa các ảnh test dự đoán sai.
    best_name = (
        result[result.split == "validation"].sort_values("f1_macro").iloc[-1]["model"]
    )
    bad = np.flatnonzero(predictions[(best_name, "test")] != datasets["test"][1])
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for a in axes.flat:
        a.axis("off")
    for a, i in zip(axes.flat, bad[:10]):
        a.imshow(X_test_raw[test_idx[i]], cmap="gray")
        a.set_title(
            f'True {y_test_all[test_idx[i]]}, '
            f'pred {predictions[(best_name,"test")][i]}'
        )
    fig.tight_layout()
    # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
    fig.savefig(figs / "errors.png", dpi=150)
    plt.close(fig)

    # Chạy các mô hình sklearn tham chiếu khi được yêu cầu.
    if baseline:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier

        refs = {
        "MultinomialNB": MultinomialNB(alpha=config["nb_alpha"]),
        "Softmax LR": LogisticRegression(max_iter=200),
        "MLP": MLPClassifier(
                hidden_layer_sizes=tuple(config["hidden"]),
                max_iter=config["epochs_mlp"],
                random_state=42,
            ),
        "Random Forest": RandomForestClassifier(
                n_estimators=config["trees"],
                max_depth=config["depth"],
                min_samples_leaf=2,
                max_features=config["max_features"],
                random_state=42,
                n_jobs=2,
            ),
        }
        rr = []
        for name, m in refs.items():
            t = time.perf_counter()
            m.fit(X_train, y_train)
            sec = time.perf_counter() - t
            rr.append(
                {
                    "model": name,
                    "validation_f1_macro": classification_scores(
                        y_val, m.predict(X_val)
                    )["f1_macro"],
                    "train_seconds": sec,
                }
            )
        # Ghi kết quả/mô hình ra tệp để tổng hợp và tái sử dụng.
        pd.DataFrame(rr).to_csv(out / "baseline.csv", index=False)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="full")
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    run(args.mode, args.baseline)
