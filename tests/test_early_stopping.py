# [Trần Tấn Đạt] Kiểm thử Early Stopping và khôi phục checkpoint.
import unittest
from unittest.mock import patch
import numpy as np
from mnist_project.models import SoftmaxRegressionScratch, MLPScratch, _EarlyStopping
from mnist_project.cross_validation import (
    stratified_kfold_indices,
    inner_validation_indices,
)


class EarlyStoppingTests(unittest.TestCase):
    def test_restore_best_weights_on_patience_and_epoch_limit(self):
        """Kiểm tra khôi phục trọng số tốt nhất khi hết patience hoặc giới hạn epoch."""

        X = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        for cls in (SoftmaxRegressionScratch, MLPScratch):
            for limit, patience, expected_epochs in [(100, 2, 3), (2, 10, 2)]:
                with self.subTest(model=cls.__name__, limit=limit):
                    expected = cls(epochs=1, early_stopping=False, seed=3).fit(X, y)
                    model = cls(epochs=limit, patience=patience, seed=3)
                    probabilities = [
                        np.array([[p, 1 - p], [1 - p, p]]) for p in [0.8, 0.7, 0.6]
                    ]
                    with patch.object(
                        model, "predict_proba", side_effect=probabilities
                    ):
                        model.fit(X, y, X[:2], y[:2])
                    self.assertEqual(model.n_epochs_, expected_epochs)
                    self.assertEqual(model.best_epoch_, 1)
                    self.assertEqual(model.stopped_epoch_, 3 if limit == 100 else 0)
                    np.testing.assert_allclose(
                        model.predict_proba(X), expected.predict_proba(X)
                    )
                    self.assertEqual(len(model.history_["val_loss"]), expected_epochs)

    def test_min_delta_and_checkpoint_are_independent(self):
        """Kiểm tra checkpoint vẫn lưu loss thấp nhất dù cải thiện chưa đạt min_delta."""

        monitor = _EarlyStopping(patience=2, min_delta=0.1)
        weights = np.array([1.0])
        self.assertFalse(monitor.update(1.0, 1, [weights]))
        weights[0] = 2
        self.assertFalse(monitor.update(0.95, 2, [weights]))
        weights[0] = 3
        self.assertTrue(monitor.update(0.96, 3, [weights]))
        self.assertEqual(monitor.best_epoch, 2)
        np.testing.assert_array_equal(monitor.best_parameters[0], [2.0])

    def test_validation_required_and_fit_resets_state(self):
        """Kiểm tra validation, patience và việc đặt lại trạng thái khi fit lần nữa."""

        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([0, 1])
        for cls in (SoftmaxRegressionScratch, MLPScratch):
            with self.assertRaises(ValueError):
                cls().fit(X, y)
            with self.assertRaises(ValueError):
                cls(patience=0).fit(X, y, X, y)
            with self.assertRaises(ValueError):
                cls().fit(X, y, X, None)
            m = cls(epochs=2).fit(X, y, X, y)
            predictions = m.predict_proba(X).copy()
            m.fit(X, y, X, y)
            self.assertEqual(len(m.history_["val_loss"]), 2)
            np.testing.assert_allclose(m.predict_proba(X), predictions)

    # [Nguyễn Quốc Đạt] Kiểm tra chia validation không dùng mẫu held-out.
    def test_inner_validation_never_uses_heldout(self):
        """Kiểm tra tập học và validation nội bộ tách biệt, không chứa mẫu held-out."""

        y = np.repeat(np.arange(10), 30)
        for train, held in stratified_kfold_indices(y):
            fit, stop = inner_validation_indices(train, y)
            self.assertFalse(np.intersect1d(fit, stop).size)
            self.assertFalse(np.intersect1d(np.r_[fit, stop], held).size)
            np.testing.assert_array_equal(np.sort(np.r_[fit, stop]), np.sort(train))
            self.assertEqual(len(np.unique(y[fit])), 10)
            self.assertEqual(len(np.unique(y[stop])), 10)


if __name__ == "__main__":
    unittest.main()
