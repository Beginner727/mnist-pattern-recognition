# [Nguyễn Quốc Đạt] Kiểm thử chia fold và Hard Voting.
import unittest
import numpy as np
from mnist_project.cross_validation import (
    stratified_kfold_indices,
    hard_vote,
    SoftmaxHardVoting,
)


class CrossValidationTests(unittest.TestCase):
    def test_partition_coverage_stratification_and_seed(self):
        """Kiểm tra 10 fold phủ đủ mẫu, phân tầng và tái lập theo seed."""

        y = np.repeat(np.arange(3), [21, 32, 47])
        folds = list(stratified_kfold_indices(y, 10, 42))
        again = list(stratified_kfold_indices(y, 10, 42))
        seen = np.zeros(len(y), dtype=int)
        trained = np.zeros(len(y), dtype=int)
        for (train, held), (train2, held2) in zip(folds, again):
            self.assertEqual(len(held), 10)
            self.assertEqual(len(train), 90)
            self.assertFalse(np.intersect1d(train, held).size)
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(held, held2)
            seen[held] += 1
            trained[train] += 1
        np.testing.assert_array_equal(seen, np.ones(len(y)))
        np.testing.assert_array_equal(trained, np.full(len(y), 9))
        per_class = np.array([np.bincount(y[h], minlength=3) for _, h in folds])
        self.assertTrue(np.all(np.ptp(per_class, axis=0) <= 1))
        different = list(stratified_kfold_indices(y, 10, 43))
        self.assertFalse(np.array_equal(folds[0][1], different[0][1]))

    def test_hard_vote_and_tie(self):
        """Kiểm tra đếm phiếu đa số và chọn nhãn nhỏ nhất khi hòa phiếu."""

        pred = np.array([[2, 1, 0], [2, 2, 1], [1, 2, 1], [0, 1, 2]])
        voted, counts = hard_vote(pred, 3)
        np.testing.assert_array_equal(voted, [2, 1, 1])
        np.testing.assert_array_equal(counts.sum(axis=1), [4, 4, 4])

    def test_invalid_inputs(self):
        """Kiểm tra chia fold và tổ hợp voting từ chối đầu vào không hợp lệ."""

        with self.assertRaises(ValueError):
            list(stratified_kfold_indices([0, 1], 10))
        with self.assertRaises(ValueError):
            hard_vote(np.empty((0, 2), dtype=int))
        with self.assertRaises(ValueError):
            hard_vote(np.array([[1.2]]))
        with self.assertRaises(ValueError):
            SoftmaxHardVoting([])


if __name__ == "__main__":
    unittest.main()
