"""Download the SECOM dataset from UCI into data/raw. Does nothing if the files already exist."""

import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from secom.config import RAW_DATA_DIR, SECOM_FILES, SECOM_URL

DOWNLOAD_TIMEOUT_SECONDS = 120


def missing_files(target_dir: Path) -> list[str]:
    """Return the expected dataset files that are absent from target_dir."""
    return [name for name in SECOM_FILES if not (target_dir / name).is_file()]


def extract_expected_files(archive_bytes: bytes, target_dir: Path) -> None:
    """Extract only the expected files by base name, so archive paths cannot escape target_dir."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist()}
        absent = [name for name in SECOM_FILES if name not in members]
        if absent:
            raise ValueError(f"Archive is missing expected files: {absent}")
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in SECOM_FILES:
            (target_dir / name).write_bytes(archive.read(members[name]))


def download_archive(url: str) -> bytes:
    """Fetch the raw archive bytes from url."""
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        return response.read()


def main() -> int:
    if not missing_files(RAW_DATA_DIR):
        print(f"SECOM data already present in {RAW_DATA_DIR}")
        return 0
    try:
        archive_bytes = download_archive(SECOM_URL)
        extract_expected_files(archive_bytes, RAW_DATA_DIR)
    except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, ValueError) as error:
        print(f"Could not download SECOM data from {SECOM_URL}: {error}", file=sys.stderr)
        return 1
    print(f"SECOM data written to {RAW_DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
