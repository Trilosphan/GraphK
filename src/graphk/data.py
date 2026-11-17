"""Tải và giải nén bộ dữ liệu Junyi Academy (bản 01/2015).

Nguồn gốc: PSLC DataShop, datasetId=1198 (cần đăng nhập).
Bản sao tải tự do: máy chủ dữ liệu của USTC (dùng trong thư viện EduData).
Điều kiện sử dụng: không dùng cho mục đích thương mại; trích dẫn
Chang, Hsu & Chen (2015), EDM 2015.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

JUNYI_URL = "http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar"
JUNYI_SIZE = 1_295_551_222  # byte, theo Content-Length của máy chủ
JUNYI_SHA256 = "7335386710db473a475ac47f6049ab8f1031bb703e6937f2d5529b46293ff35e"  # đo ngày 17/11/2026

EXERCISE_TABLE = "junyi_Exercise_table.csv"
PROBLEM_LOG = "junyi_ProblemLog_original.csv"
EXTRACT_MEMBERS = (
    EXERCISE_TABLE,
    PROBLEM_LOG,
    "relationship_annotation_training.csv",
    "relationship_annotation_testing.csv",
    "README.txt",
)


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_junyi(dest_dir: str | Path) -> Path:
    """Tải junyi.rar về dest_dir; tải tiếp được nếu bị ngắt giữa chừng."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "junyi.rar"
    if archive.exists() and archive.stat().st_size == JUNYI_SIZE:
        return archive
    subprocess.run(
        ["curl", "-L", "--fail", "-C", "-", "-o", str(archive), JUNYI_URL],
        check=True,
    )
    size = archive.stat().st_size
    if size != JUNYI_SIZE:
        raise RuntimeError(f"Kích thước tệp tải về {size} khác {JUNYI_SIZE}; hãy chạy lại để tải tiếp.")
    return archive


def _find_extractor() -> list[str]:
    for tool, args in (("unrar", ["x", "-o+"]), ("7z", ["x", "-y"]), ("bsdtar", ["-xf"])):
        if shutil.which(tool):
            return [tool, *args]
    raise RuntimeError("Không tìm thấy unrar/7z/bsdtar. Trên Colab: !apt-get -qq install unrar")


def extract_junyi(archive: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Giải nén bảng bài tập, nhật ký gốc và tệp gán nhãn quan hệ.

    Bỏ qua junyi_ProblemLog_for_PSLC.txt (9,7 GB) vì không dùng đến.
    """
    archive, out_dir = Path(archive), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {name: next(out_dir.rglob(name), None) for name in (EXERCISE_TABLE, PROBLEM_LOG)}
    if not all(found.values()):
        members = [*EXTRACT_MEMBERS]
        cmd = _find_extractor()
        if cmd[0] == "bsdtar":
            subprocess.run([*cmd, str(archive), "-C", str(out_dir), *members], check=True)
        elif cmd[0] == "7z":
            subprocess.run([*cmd, str(archive), f"-o{out_dir}", *members], check=True)
        else:
            subprocess.run([*cmd, str(archive), *members, f"{out_dir}/"], check=True)
        found = {name: next(out_dir.rglob(name), None) for name in (EXERCISE_TABLE, PROBLEM_LOG)}
    missing = [name for name, path in found.items() if path is None]
    if missing:
        raise FileNotFoundError(f"Không thấy {missing} sau khi giải nén {archive}")
    return {"exercise_table": found[EXERCISE_TABLE], "problem_log": found[PROBLEM_LOG]}
