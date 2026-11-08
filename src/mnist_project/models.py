import numpy as np
from sklearn.ensemble import RandomForestClassifier


# [Nguyễn Duy Khánh] Multinomial Naïve Bayes từ đầu bằng NumPy.
class MultinomialNBScratch:
    def __init__(self, alpha=1.0):
        """Lưu alpha làm trơn Laplace cho đặc trưng không âm."""
        self.alpha = alpha

    def fit(self, X, y):
        """Học log prior và xác suất đặc trưng theo lớp; trả về self.

        X có dạng (n,d), hữu hạn và không âm. Pixel chuẩn hóa được dùng
        như lượng không âm, không phải số đếm đa thức nguyên thực sự.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or y.shape != (len(X),):
            raise ValueError("Expected nonempty X (n,d) and y (n,)")
        if not np.isfinite(X).all() or np.any(X < 0):
            raise ValueError("MultinomialNB requires finite nonnegative features")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        # Xác định các lớp và đếm số mẫu
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.class_count_ = np.bincount(encoded).astype(float)
        self.feature_count_ = np.stack(
            [X[encoded == c].sum(axis=0) for c in range(len(self.classes_))]
        )

        # Làm trơn lượng đặc trưng trước khi chuyển sang log xác suất.
        smoothed = self.feature_count_ + self.alpha

        # Làm trơn rồi chuẩn hóa theo tổng lượng đặc trưng của mỗi lớp.
        self.feature_log_prob_ = np.log(smoothed) - np.log(
            smoothed.sum(axis=1, keepdims=True)
        )
        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        return self

    def decision_function(self, X):
        """Trả về log prior + tổng x_j log theta_cj, dạng (n,K)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected feature dimension")
        if not np.isfinite(X).all() or np.any(X < 0):
            raise ValueError("MultinomialNB requires finite nonnegative features")
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        """Chuẩn hóa log điểm thành xác suất hậu nghiệm theo từng hàng."""
        return stable_softmax(self.decision_function(X))

    def predict(self, X):
        """Trả nhãn gốc có log điểm lớn nhất; hòa chọn nhãn nhỏ nhất."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]


# [Trần Tấn Đạt] Softmax, cross-entropy, Early Stopping và MLP.
# Các hàm này cũng được dùng trong MLP.
def stable_softmax(z):
    """Chuyển logits (n_mẫu, n_lớp) thành xác suất theo hàng, tránh tràn số."""
    z = z - np.max(z, axis=1, keepdims=True)  # chống tràn số e^z
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_from_probs(probs, labels):
    """Tính cross entropy trung bình từ xác suất và nhãn nguyên 0..K-1.

    Chặn xác suất dưới tại 1e-12 để tránh log(0)."""
    rows = np.arange(len(labels))
    return float(-np.log(np.maximum(probs[rows, labels], 1e-12)).mean())


# Early Stopping dùng chung cho Softmax và MLP, không dùng test.
class _EarlyStopping:
    def __init__(self, patience, min_delta):
        """Theo dõi loss nhỏ nhất và số epoch không cải thiện đáng kể."""
        if not isinstance(patience, (int, np.integer)) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = self.reference_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_parameters = None

    def update(self, loss, epoch, parameters):
        """Lưu bản sao ở loss nhỏ nhất; trả True khi hết patience.

        min_delta chỉ quyết định reset patience. Checkpoint luôn là epoch
        có loss nhỏ nhất, kể cả khi mức giảm nhỏ hơn min_delta.
        """
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_epoch = epoch
            self.best_parameters = [p.copy() for p in parameters]
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = float(loss)
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def _prepare_early_stopping(model, X_val, y_val, n_features, n_classes):
    """Kiểm tra validation và reset metadata cho một lần fit độc lập."""
    if not isinstance(model.epochs, (int, np.integer)) or model.epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    if (X_val is None) != (y_val is None):
        raise ValueError("Provide both X_val and y_val")
    if model.early_stopping and X_val is None:
        raise ValueError(
            "Early stopping requires validation; pass early_stopping=False to train"
            " without it"
        )
    monitor = _EarlyStopping(model.patience, model.min_delta)

    # Kiểm tra kích thước, kiểu nhãn và giá trị của tập validation.
    if X_val is not None:
        X_val = np.asarray(X_val, np.float32)
        y_val = np.asarray(y_val)
        if (
            X_val.ndim != 2
            or X_val.shape[1] != n_features
            or not len(X_val)
            or y_val.shape != (len(X_val),)
        ):
            raise ValueError("Invalid validation shape")
        if (
            not np.isfinite(X_val).all()
            or not np.issubdtype(y_val.dtype, np.integer)
            or np.any((y_val < 0) | (y_val >= n_classes))
        ):
            raise ValueError("Invalid validation features or labels")

    # Reset lịch sử và metadata để mỗi lần fit được độc lập.
    model.n_epochs_ = 0
    model.best_epoch_ = 0
    model.stopped_epoch_ = 0
    model.best_val_loss_ = None
    model.history_ = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    return X_val, y_val, monitor


class SoftmaxRegressionScratch:
    def __init__(
        self,
        lr=0.5,
        epochs=100,
        batch_size=256,
        weight_decay=1e-4,
        seed=42,
        early_stopping=True,
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True,
    ):
        """Lưu siêu tham số; epochs là giới hạn tối đa,
        patience tính theo validation loss.
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

    def fit(self, X, y, X_val=None, y_val=None):
        """Khởi tạo lại W, b và học Softmax bằng mini-batch gradient descent.

        X là ma trận đặc trưng, y thuộc 0..K-1; shuffle mỗi epoch.
        Validation chọn epoch bằng cross-entropy; không dùng tính gradient.
        Lưu loss cross entropy trung bình các batch (không gồm L2); trả về self."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        assert np.array_equal(self.classes_, np.arange(k)), "Yêu cầu nhãn 0..K-1"
        self.W = rng.normal(0, 0.01, size=(d, k)).astype(np.float32)
        self.b = np.zeros(k, dtype=np.float32)
        X_val, y_val, monitor = _prepare_early_stopping(
            self, X_val, y_val, X.shape[1], k
        )

        # Huấn luyện theo epoch; mỗi lượt xáo trộn lại thứ tự mẫu.
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            loss_sum = 0.0

            # Duyệt mini-batch và cập nhật tham số từ gradient.
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = X[idx]
                yb = y[idx]
                p = stable_softmax(xb @ self.W + self.b)
                loss_sum += cross_entropy_from_probs(p, yb) * len(yb)

                # dL/dZ = P - onehot(y), chia batch size để lấy gradient trung bình.
                delta = p.copy()
                delta[np.arange(len(yb)), yb] -= 1.0
                delta /= len(yb)
                grad_W = xb.T @ delta + self.weight_decay * self.W
                grad_b = delta.sum(axis=0)
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
            self.history_["train_loss"].append(loss_sum / n)
            self.n_epochs_ = epoch + 1
            stop = False

            # Đánh giá validation để theo dõi loss và chọn checkpoint.
            if X_val is not None:
                val_probs = self.predict_proba(X_val)
                val_loss = cross_entropy_from_probs(val_probs, y_val)
                val_acc = float(np.mean(val_probs.argmax(axis=1) == y_val))
                self.history_["val_loss"].append(val_loss)
                self.history_["val_accuracy"].append(val_acc)
                stop = monitor.update(val_loss, epoch + 1, [self.W, self.b])
                self.best_epoch_ = monitor.best_epoch
                self.best_val_loss_ = monitor.best_loss
            print(
                f"[Softmax] lr={self.lr:g} | epoch {epoch+1}/{self.epochs}:",
                f"loss={loss_sum/n:.4f}",
                f"val_loss={val_loss:.4f}" if X_val is not None else "",
                flush=True,
            )

            # Dừng khi số epoch không cải thiện đạt ngưỡng patience.
            if self.early_stopping and stop:
                self.stopped_epoch_ = self.n_epochs_
                break

        # Khôi phục checkpoint cả khi đạt giới hạn epoch, không chỉ khi dừng sớm.
        if (
            self.early_stopping
            and self.restore_best_weights
            and monitor.best_parameters is not None
        ):
            self.W, self.b = monitor.best_parameters
        return self

    def predict_proba(self, X):
        """Trả về xác suất các lớp, dạng (n_mẫu, n_lớp), từ trọng số đã học."""
        return stable_softmax(np.asarray(X, dtype=np.float32) @ self.W + self.b)

    def predict(self, X):
        """Trả về chỉ số lớp có xác suất Softmax lớn nhất cho từng mẫu."""
        return np.argmax(self.predict_proba(X), axis=1)


# MLP đa lớp: fully-connected + ReLU + Softmax, backprop bằng NumPy.
class MLPScratch:
    def __init__(
        self,
        hidden_layers=(64,),
        lr=0.5,
        epochs=100,
        batch_size=256,
        weight_decay=1e-4,
        seed=42,
        early_stopping=True,
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True,
    ):
        """Lưu kiến trúc MLP và giới hạn epoch; Early Stopping theo validation loss."""
        self.hidden_layers = tuple(hidden_layers)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

    def _initialize_parameters(self, n_features, n_classes, rng):
        """Dùng NumPy khởi tạo He cho trọng số và zero cho bias của từng lớp."""
        layers = [n_features, *self.hidden_layers, n_classes]
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / ni), size=(ni, no)).astype(np.float32)
            for ni, no in zip(layers[:-1], layers[1:])
        ]
        self.biases = [np.zeros(no, np.float32) for no in layers[1:]]

    def _backward(self, probs, labels, activations, z_hidden):
        """Tự lan truyền gradient ngược bằng chain rule, gồm đạo hàm L2.

        Nhận cache từ _forward; trả gradient trọng số và bias của mọi lớp.
        Không cập nhật tham số ở đây để tránh dùng lẫn trọng số cũ và mới.
        """
        # Đạo hàm kết hợp Softmax và cross-entropy ở lớp đầu ra.
        delta = probs.copy()
        delta[np.arange(len(labels)), labels] -= 1.0
        delta /= len(labels)
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        # Tự áp dụng chain rule từ lớp đầu ra về các lớp ẩn.
        for layer in reversed(range(len(self.weights))):
            grad_w[layer] = (
                activations[layer].T @ delta
                + self.weight_decay * self.weights[layer]
            )
            grad_b[layer] = delta.sum(axis=0)
            if layer > 0:
                # NumPy tính mask đạo hàm ReLU; quy ước ReLU'(0) = 0.
                relu_gradient = np.greater(z_hidden[layer - 1], 0)
                delta = (delta @ self.weights[layer].T) * relu_gradient
        return grad_w, grad_b

    def _forward(self, X):
        """Lan truyền thuận qua các lớp ReLU và lớp Softmax đầu ra.

        Trả về xác suất, các activation và logits lớp ẩn để dùng khi backprop."""
        activations = [X]  # A[0]=X; A[l] là input để tính dW[l]
        z_hidden = []
        a = X
        for w, b in zip(
            self.weights[:-1], self.biases[:-1]
        ):  # -1 vì phần tử cuối của weight thuộc output player
            z = a @ w + b
            z_hidden.append(z)
            a = np.maximum(z, 0.0)  # Dùng activation ReLU của NumPy.
            activations.append(a)
        z_last = a @ self.weights[-1] + self.biases[-1]
        p = stable_softmax(z_last)
        return p, activations, z_hidden

    def fit(self, X, y, X_val=None, y_val=None):
        """Khởi tạo lại mạng và học bằng backpropagation, mini-batch GD và L2.

        X có dạng (n_mẫu, n_đặc_trưng), y thuộc 0..K-1.
        Shuffle mỗi epoch; validation chọn epoch dừng, không dùng tính gradient.
        Lưu lịch sử loss theo batch, accuracy validation và trả về self."""
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.int64)
        rng = np.random.default_rng(self.seed)
        k = len(np.unique(y))
        assert np.array_equal(np.unique(y), np.arange(k))
        # Khởi tạo qua NumPy; giữ cùng RNG cho bước shuffle sau đó.
        self._initialize_parameters(X.shape[1], k, rng)
        X_val, y_val, monitor = _prepare_early_stopping(
            self, X_val, y_val, X.shape[1], k
        )

        # Huấn luyện theo epoch; mỗi lượt xáo trộn lại thứ tự mẫu.
        for epoch in range(self.epochs):
            order = rng.permutation(len(X))
            total = 0.0

            # Duyệt mini-batch và cập nhật tham số từ gradient.
            for start in range(0, len(X), self.batch_size):
                ids = order[start : start + self.batch_size]
                xb = X[ids]
                yb = y[ids]
                probs, A, Z = self._forward(xb)
                total += cross_entropy_from_probs(probs, yb) * len(ids)

                # Feedforward trả cache; backprop tự tính gradient từng lớp.
                grad_w, grad_b = self._backward(probs, yb, A, Z)

                # Cập nhật sau khi đã tính đủ gradient bằng trọng số hiện tại.
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.lr * grad_w[layer]
                    self.biases[layer] -= self.lr * grad_b[layer]
            train_loss = total / len(X)
            self.history_["train_loss"].append(train_loss)
            self.n_epochs_ = epoch + 1
            stop = False

            # Đánh giá validation để theo dõi loss và chọn checkpoint.
            if X_val is not None:
                val_probs = self.predict_proba(X_val)
                val_loss = cross_entropy_from_probs(val_probs, y_val)
                val_acc = float(np.mean(val_probs.argmax(axis=1) == y_val))
                self.history_["val_loss"].append(val_loss)
                self.history_["val_accuracy"].append(val_acc)
                stop = monitor.update(val_loss, epoch + 1, self.weights + self.biases)
                self.best_epoch_ = monitor.best_epoch
                self.best_val_loss_ = monitor.best_loss
            print(
                f"[MLP] lr={self.lr:g} | epoch {epoch+1}/{self.epochs}:",
                f"loss={train_loss:.4f}",
                f"val_loss={val_loss:.4f}" if X_val is not None else "",
                flush=True,
            )

            # Dừng khi số epoch không cải thiện đạt ngưỡng patience.
            if self.early_stopping and stop:
                self.stopped_epoch_ = self.n_epochs_
                break
        if (
            self.early_stopping
            and self.restore_best_weights
            and monitor.best_parameters is not None
        ):
            count = len(self.weights)
            self.weights = monitor.best_parameters[:count]
            self.biases = monitor.best_parameters[count:]
        return self

    def predict_proba(self, X, batch_size=1024):
        """Tính xác suất từng lớp theo các batch để giảm bộ nhớ khi dự đoán."""
        X = np.asarray(X, np.float32)
        parts = [
            self._forward(X[i : i + batch_size])[0]
            for i in range(0, len(X), batch_size)
        ]
        return np.vstack(parts)

    def predict(self, X):
        """Trả về nhãn có xác suất lớn nhất từ mạng đã huấn luyện."""
        return np.argmax(self.predict_proba(X), axis=1)


# [Nguyễn Quốc Đạt] Random Forest dùng scikit-learn để dựng cây và dự đoán.
class RandomForestModel(RandomForestClassifier):
    """Dùng Random Forest thư viện, bổ sung đường F1 theo số cây."""

    def __init__(
        self,
        n_estimators=40,
        max_depth=14,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ):
        """Cấu hình rừng bootstrap; thư viện đảm nhiệm fit và predict."""
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=True,
        )

    def staged_predict(self, X):
        """Trả nhãn theo xác suất trung bình sau mỗi cây để vẽ đường F1."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        probability_sum = np.zeros((len(X), len(self.classes_)))

        # Cây sklearn dùng cùng thứ tự lớp đã mã hóa trong rừng.
        for tree_count, tree in enumerate(self.estimators_, start=1):
            probability_sum += tree.predict_proba(X)
            mean_probability = probability_sum / tree_count
            yield self.classes_[mean_probability.argmax(axis=1)]
