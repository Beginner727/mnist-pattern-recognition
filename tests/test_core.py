# [Cả nhóm] Kiểm thử dữ liệu, các mô hình và chỉ số đánh giá.
import unittest
import numpy as np
from mnist_project.models import (
    MultinomialNBScratch,
    SoftmaxRegressionScratch,
    MLPScratch,
    RandomForestModel,
    stable_softmax,
)
from mnist_project.metrics import classification_scores
from mnist_project.data import stratified_train_validation, stratified_limit


class CoreTests(unittest.TestCase):
    def test_forest_bootstrap_and_reproducibility(self):
        """Kiểm tra bootstrap có hoàn lại và kết quả tái lập khi giữ cùng seed."""

        X = np.array([[0, 1], [0.1, 0.9], [1, 0], [0.9, 0.1]] * 8, dtype=np.float32)
        y = np.tile([3, 3, 8, 8], 8)
        a = RandomForestModel(n_estimators=5, max_features=2, random_state=7).fit(X, y)
        b = RandomForestModel(n_estimators=5, max_features=2, random_state=7).fit(X, y)
        np.testing.assert_array_equal(a.predict(X), y)
        for ia, ib in zip(a.estimators_samples_, b.estimators_samples_):
            np.testing.assert_array_equal(ia, ib)
            self.assertLess(len(np.unique(ia)), len(y))
        self.assertFalse(
            np.array_equal(a.estimators_samples_[0], a.estimators_samples_[1])
        )
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_forest_probability_average(self):
        """Đối chiếu xác suất trung bình cây, dự đoán từng bước và lưu/nạp rừng."""

        X = np.array([[0, 1], [1, 0], [0.4, 0.6], [0.6, 0.4]], dtype=np.float32)
        model = RandomForestModel(
            n_estimators=4, min_samples_leaf=1, max_features=2
        ).fit(X, np.array([2, 7, 2, 7]))
        probabilities = np.mean(
            [tree.predict_proba(X) for tree in model.estimators_], axis=0
        )
        expected = model.classes_[probabilities.argmax(axis=1)]
        np.testing.assert_allclose(model.predict_proba(X), probabilities)
        np.testing.assert_array_equal(model.predict(X), expected)
        np.testing.assert_array_equal(list(model.staged_predict(X))[-1], expected)

        # Kiểm tra lưu/nạp cho demo và đối chiếu trực tiếp bản thư viện.
        import pickle
        from sklearn.ensemble import RandomForestClassifier

        restored = pickle.loads(pickle.dumps(model))
        reference = RandomForestClassifier(**model.get_params()).fit(
            X, np.array([2, 7, 2, 7])
        )
        np.testing.assert_array_equal(restored.predict(X), expected)
        np.testing.assert_allclose(
            model.predict_proba(X), reference.predict_proba(X)
        )

    def test_new_models_reject_invalid_inputs(self):
        """Kiểm tra NB từ chối dữ liệu, alpha sai và rừng từ chối số cây bằng 0."""

        for model in [MultinomialNBScratch()]:
            for X in [np.array([[-1.0], [0.0]]), np.array([[np.nan], [0.0]])]:
                with self.assertRaises(ValueError):
                    model.fit(X, [0, 1])
        with self.assertRaises(ValueError):
            MultinomialNBScratch(alpha=0).fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            RandomForestModel(n_estimators=0).fit([[0], [1]], [0, 1])

    def test_split(self):
        """Kiểm tra chia tập không trùng, phân tầng, giới hạn mẫu và tái lập seed."""

        y = np.repeat(np.arange(10), 20)
        tr, va = stratified_train_validation(y)
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertEqual(len(tr) + len(va), len(y))
        np.testing.assert_array_equal(np.bincount(y[va]), np.repeat(2, 10))
        self.assertEqual(len(stratified_limit(tr, y, 53)), 53)

        # Đối chiếu đúng API sklearn và kiểm tra tính tái lập của seed.
        from sklearn.model_selection import train_test_split

        expected_train, expected_val = train_test_split(
            np.arange(len(y)), test_size=0.1, random_state=42,
            shuffle=True, stratify=y,
        )
        np.testing.assert_array_equal(tr, expected_train)
        np.testing.assert_array_equal(va, expected_val)
        repeated_train, repeated_val = stratified_train_validation(y)
        np.testing.assert_array_equal(tr, repeated_train)
        np.testing.assert_array_equal(va, repeated_val)


    def test_metrics_known(self):
        """Đối chiếu accuracy, F1 macro và confusion matrix với kết quả tính tay."""

        s = classification_scores(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n_classes=2
        )
        self.assertAlmostEqual(s["accuracy"], 0.75)
        self.assertAlmostEqual(s["f1_macro"], (2 / 3 + 0.8) / 2)
        np.testing.assert_array_equal(s["cm"], [[1, 1], [0, 2]])

    def test_softmax_extreme(self):
        """Kiểm tra Softmax cho xác suất hữu hạn và tổng bằng 1 với logits lớn."""

        p = stable_softmax(np.array([[10000.0, 10001.0, -10000.0]]))
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(p.sum(), 1.0)

    def test_learn_separable(self):
        """Kiểm tra bốn mô hình học được dữ liệu hai lớp dễ phân tách."""

        rng = np.random.default_rng(1)
        X = np.r_[
            rng.normal([0, 1], 0.05, (30, 2)), rng.normal([1, 0], 0.05, (30, 2))
        ].astype("float32")
        X = (X - X.min()) / (X.max() - X.min())
        y = np.repeat([0, 1], 30)
        models = [
            MultinomialNBScratch(),
            SoftmaxRegressionScratch(early_stopping=False, epochs=20, lr=0.2),
            MLPScratch(early_stopping=False, hidden_layers=(8,), epochs=30, lr=0.2),
            RandomForestModel(n_estimators=3, max_depth=3, min_samples_leaf=2),
        ]
        for m in models:
            with self.subTest(model=type(m).__name__):
                m.fit(X, y)
                self.assertGreater(np.mean(m.predict(X) == y), 0.95)

    def test_multinomial_counts_and_reference(self):
        """Đối chiếu làm trơn, prior và dự đoán MultinomialNB với scikit-learn."""

        from sklearn.naive_bayes import MultinomialNB

        X = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
        y = np.array([2, 2, 9])
        m = MultinomialNBScratch(alpha=1).fit(X, y)
        expected = np.array([[4 / 7, 2 / 7, 1 / 7], [1 / 6, 4 / 6, 1 / 6]])
        np.testing.assert_allclose(np.exp(m.feature_log_prob_), expected)
        np.testing.assert_allclose(np.exp(m.class_log_prior_), [2 / 3, 1 / 3])
        T = np.array([[0.0, 0.0, 0.0], [0.2, 0.7, 0.0], [3.0, 1.0, 0.0]])
        ref = MultinomialNB(alpha=1).fit(X, y)
        np.testing.assert_allclose(m.predict_proba(T), ref.predict_proba(T), rtol=1e-12)
        np.testing.assert_array_equal(m.predict(T), ref.predict(T))

    def test_mlp_numerical_gradient(self):
        """So sánh gradient MLP qua một bước cập nhật với sai phân hữu hạn."""

        X = np.array([[0.2, 0.7], [0.8, 0.1], [0.4, 0.6]], dtype="float32")
        y = np.array([0, 1, 0])
        initial = MLPScratch(
            early_stopping=False, hidden_layers=(3,), epochs=0, weight_decay=0, seed=5
        ).fit(X, y)
        updated = MLPScratch(
            early_stopping=False,
            hidden_layers=(3,),
            epochs=1,
            batch_size=3,
            lr=0.01,
            weight_decay=0,
            seed=5,
        ).fit(X, y)

        def loss():
            """Tính cross-entropy ở trọng số hiện tại để ước lượng gradient số."""

            return -np.log(initial._forward(X)[0][np.arange(3), y]).mean()

        for layer, w in enumerate(initial.weights):
            for index in [(0, 0), (w.shape[0] - 1, w.shape[1] - 1)]:
                old = w[index].copy()
                eps = 0.001
                w[index] = old + eps
                plus = loss()
                w[index] = old - eps
                minus = loss()
                w[index] = old
                numeric = (plus - minus) / (2 * eps)
                analytic = (old - updated.weights[layer][index]) / 0.01
                self.assertAlmostEqual(float(numeric), float(analytic), delta=0.001)


if __name__ == "__main__":
    unittest.main()
