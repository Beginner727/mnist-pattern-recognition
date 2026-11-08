# [Cả nhóm] Kiểm tra đầu vào và dự đoán EMNIST không huấn luyện.
import gzip
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np

from mnist_project.evaluate_emnist import predict_batches, read_test, download_archive


class EmnistTests(unittest.TestCase):
    """Kiểm chứng hướng ảnh IDX và tính nhất quán khi chia batch."""

    def test_download_tls_fallback(self):
        """Lỗi chứng chỉ Python chuyển sang curl mà không bỏ xác thực TLS."""
        with patch("mnist_project.evaluate_emnist.urllib.request.urlretrieve",
                   side_effect=urllib.error.URLError("CERTIFICATE_VERIFY_FAILED")), \
             patch("mnist_project.evaluate_emnist.shutil.which", return_value="curl.exe"), \
             patch("mnist_project.evaluate_emnist.subprocess.run") as command:
            download_archive(Path("archive.zip.part"))
        args = command.call_args.args[0]
        self.assertNotIn("--insecure", args)
        self.assertNotIn("-k", args)
        self.assertIn("--fail", args)
        self.assertTrue(command.call_args.kwargs["check"])

    def test_download_reports_missing_fallback(self):
        """Không có curl thì báo lỗi có hướng dẫn, không giả báo tải thành công."""
        with patch("mnist_project.evaluate_emnist.urllib.request.urlretrieve",
                   side_effect=urllib.error.URLError("SSL error")), \
             patch("mnist_project.evaluate_emnist.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                download_archive(Path("archive.zip.part"))

    def test_idx_orientation_and_labels(self):
        """Pixel bất đối xứng phải được transpose đúng, nhãn giữ nguyên."""
        images = np.zeros((40000, 28, 28), dtype=np.uint8)
        images[0, 2, 5] = 255
        labels = np.repeat(np.arange(10, dtype=np.uint8), 4000)
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "images.gz"
            label_path = Path(folder) / "labels.gz"
            with gzip.open(image_path, "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 40000, 28, 28))
                stream.write(images.tobytes())
            with gzip.open(label_path, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 40000))
                stream.write(labels.tobytes())
            actual, actual_labels = read_test(image_path, label_path)
            self.assertEqual(actual[0, 5, 2], 255)
            self.assertEqual(actual[0, 2, 5], 0)
            np.testing.assert_array_equal(actual_labels, labels)
            with gzip.open(label_path, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 2))
                stream.write(bytes([0, 1]))
            with self.assertRaises(ValueError):
                read_test(image_path, label_path)

    def test_batches_normalize_without_fit(self):
        """Batch cuối không bị bỏ; đầu vào float32 nằm trong [0, 1]."""
        class PredictOnly:
            """Mô hình giả chỉ hỗ trợ predict, không có fit."""

            def predict(self, X):
                """Kiểm tra dạng đầu vào và trả nhãn theo pixel đầu."""
                assert X.dtype == np.float32
                assert X.shape[1] == 784
                assert X.min() >= 0 and X.max() <= 1
                return (X[:, 0] > 0.5).astype(int)

        images = np.zeros((5, 28, 28), dtype=np.uint8)
        images[::2, 0, 0] = 255
        actual = predict_batches(PredictOnly(), images, 2)
        np.testing.assert_array_equal(actual, [1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
