# [Nguyễn Duy Khánh] Giao diện demo, nạp mô hình và xử lý ảnh vẽ.
from pathlib import Path
import pickle
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageTk


def main():
    """Mở giao diện vẽ chữ số và nạp các mô hình đã lưu để dự đoán."""

    # Tìm bộ mô hình đã huấn luyện trong thư mục kết quả.
    root = Path(__file__).resolve().parents[2]
    path = root / "result/full/models.pkl"
    if not path.exists():
        raise SystemExit("Run python -m mnist_project.experiment --mode full first.")

    with path.open("rb") as f:
        models = pickle.load(f)

    # Nạp thêm tổ hợp Softmax nếu có kết quả 10-fold.
    voting_path = root / "result/softmax_10fold/hard_voting.pkl"
    if voting_path.exists():
        with voting_path.open("rb") as f:
            models["10-fold Softmax voting"] = pickle.load(f)

    # Tạo cửa sổ, vùng vẽ và ảnh PIL dùng để nhận dạng.
    app = tk.Tk()
    app.title("MNIST NumPy demo")
    panels = tk.Frame(app)
    panels.pack(padx=12, pady=12)
    drawing_panel = tk.LabelFrame(panels, text="Nét vẽ của bạn")
    drawing_panel.pack(side=tk.LEFT, padx=(0, 12))
    input_panel = tk.LabelFrame(panels, text="Ảnh 28×28 đưa vào mô hình")
    input_panel.pack(side=tk.LEFT)
    canvas = tk.Canvas(drawing_panel, width=280, height=280, bg="black")
    canvas.pack()
    preview = tk.Label(input_panel, bg="black")
    preview.pack()
    image = Image.new("L", (280, 280))
    draw = ImageDraw.Draw(image)
    label = tk.Label(app, text="Vẽ một chữ số ở giữa khung", wraplength=500)
    label.pack()

    def show_model_input(frame):
        """Phóng ảnh đầu vào 28×28 để xem pixel, không thay đổi ảnh dự đoán."""
        enlarged = frame.resize((280, 280), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(enlarged, master=app)
        preview.config(image=photo)
        # Giữ tham chiếu để Tkinter không mất ảnh khi Python thu hồi bộ nhớ.
        preview.image = photo

    show_model_input(Image.new("L", (28, 28)))

    def paint(e):
        """Vẽ nét trắng theo vị trí chuột lên canvas và ảnh PIL đồng thời."""
        box = (e.x - 10, e.y - 10, e.x + 10, e.y + 10)
        canvas.create_oval(*box, fill="white", outline="white")
        draw.ellipse(box, fill=255)
        # Ảnh đầu vào và kết quả cũ không còn đại diện cho nét vẽ vừa thay đổi.
        show_model_input(Image.new("L", (28, 28)))
        label.config(text="Bấm Dự đoán để xem ảnh 28×28 và kết quả mới")

    def clear():
        """Xóa nét vẽ trên canvas, ảnh PIL và đặt lại thông báo hướng dẫn."""
        canvas.delete("all")
        draw.rectangle((0, 0, 280, 280), fill=0)
        show_model_input(Image.new("L", (28, 28)))
        label.config(text="Vẽ một chữ số")

    def predict():
        """Cắt vùng nét vẽ, thu về ảnh 28x28, chuẩn hóa và hiển thị dự đoán."""

        # Kiểm tra có nét vẽ trước khi tiền xử lý.
        bbox = image.getbbox()
        if bbox is None:
            label.config(text="Chưa có nét vẽ")
            return

        # Cắt nét vẽ, thu nhỏ và căn giữa khung ảnh 28 x 28.
        digit = image.crop(bbox)
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
        frame = Image.new("L", (28, 28))
        frame.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))

        # Hiển thị chính ảnh sẽ được flatten và chuẩn hóa bên dưới.
        show_model_input(frame)

        # Chuẩn hóa ảnh rồi hiển thị nhãn của từng mô hình.
        x = np.asarray(frame, dtype="float32").reshape(1, 784) / 255
        label.config(
            text=" | ".join(f"{name}: {m.predict(x)[0]}" for name, m in models.items())
        )

    # Gắn sự kiện chuột, các nút thao tác và chạy giao diện.
    canvas.bind("<B1-Motion>", paint)
    tk.Button(app, text="Dự đoán", command=predict).pack()
    tk.Button(app, text="Xóa", command=clear).pack()
    app.mainloop()


if __name__ == "__main__":
    main()
