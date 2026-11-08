# [Nguyễn Duy Khánh] Cập nhật nội dung hiển thị của notebook kết quả.
"""Chạy notebook Python thuần, không hỗ trợ magic hoặc widget của kernel."""

import argparse
import ast
import contextlib
import io
import os
import sys
from pathlib import Path

import IPython.display
import nbformat
from IPython.core.formatters import DisplayFormatter

ROOT = Path(__file__).resolve().parents[1]


class OutputStream(io.TextIOBase):
    """Thu stdout và stderr theo đúng thứ tự xen kẽ với display."""

    def __init__(self, outputs, name):
        """Giữ danh sách đầu ra và tên luồng cần thu."""
        self.outputs = outputs
        self.name = name

    def write(self, text):
        """Gộp các đoạn ghi liên tiếp trên cùng một luồng."""
        if text:
            if (self.outputs and self.outputs[-1].output_type == "stream"
                    and self.outputs[-1].name == self.name):
                self.outputs[-1].text += text
            else:
                self.outputs.append(nbformat.v4.new_output(
                    "stream", name=self.name, text=text,
                ))
        return len(text)


def execute_notebook(path, output=None):
    """Thực thi từ gốc dự án; chỉ lưu notebook khi mọi ô đều thành công."""
    path = (ROOT / path).resolve()
    output = (ROOT / output).resolve() if output else path
    notebook = nbformat.read(path, as_version=4)
    formatter = DisplayFormatter()
    namespace = {"__name__": "__main__"}
    old_display = IPython.display.display
    old_cwd = Path.cwd()
    old_sys_path = sys.path[:]
    count = 0

    try:
        os.chdir(ROOT)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            outputs = []

            def display(*objects, **kwargs):
                """Định dạng văn bản, bảng và ảnh bằng formatter của IPython."""
                for obj in objects:
                    if kwargs.get("raw"):
                        data = obj
                        metadata = kwargs.get("metadata", {})
                    else:
                        data, metadata = formatter.format(obj)
                    outputs.append(nbformat.v4.new_output(
                        "display_data", data=data, metadata=metadata,
                    ))

            IPython.display.display = display
            namespace["display"] = display
            with contextlib.redirect_stdout(OutputStream(outputs, "stdout")), \
                    contextlib.redirect_stderr(OutputStream(outputs, "stderr")):
                try:
                    tree = ast.parse(cell.source, filename=str(path))
                    last = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last = tree.body.pop().value
                    exec(compile(tree, str(path), "exec"), namespace)
                    if last is not None:
                        value = eval(compile(ast.Expression(last), str(path), "eval"),
                                     namespace)
                        if value is not None and not cell.source.rstrip().endswith(";"):
                            data, metadata = formatter.format(value)
                            outputs.append(nbformat.v4.new_output(
                                "execute_result", data=data, metadata=metadata,
                                execution_count=count,
                            ))
                except Exception as exc:
                    raise RuntimeError(f"Lỗi ô code {count} trong {path.name}") from exc
            cell.outputs = outputs
            cell.execution_count = count
        nbformat.validate(notebook)
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
    finally:
        IPython.display.display = old_display
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
    return count, output


def main():
    """Đọc tham số và chạy notebook được chọn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="notebooks/Ket_qua_chinh.ipynb")
    parser.add_argument("--output", type=Path, help="Mặc định cập nhật notebook gốc.")
    args = parser.parse_args()
    count, output = execute_notebook(Path(args.notebook), args.output)
    print(f"Executed {count} Python cells successfully: {output}")


if __name__ == "__main__":
    # Giữ tiếng Việt khi chạy qua pipe trên Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
