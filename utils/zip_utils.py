"""Utilities for zipping/unzipping directories to reduce inode count on shared filesystems."""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import Optional


def zip_and_remove_dir(
    dir_path: str,
    zip_path: Optional[str] = None,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 1,
) -> str:
    """
    Zip all files in *dir_path* into *zip_path*, verify the archive, then
    remove the original directory.

    Args:
        dir_path: Directory to zip (e.g. ``<digitnum>/images``).
        zip_path: Destination zip file.  Defaults to ``<dir_path>.zip``.
        compression: Compression method (default ZIP_DEFLATED).
        compresslevel: Speed / size trade-off (1 = fast, 9 = small).

    Returns:
        Absolute path to the created zip file.
    """
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if zip_path is None:
        zip_path = dir_path + ".zip"
    zip_path = os.path.abspath(zip_path)

    tmp_zip = zip_path + ".tmp"
    n_files = 0
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression, compresslevel=compresslevel) as zf:
            for root, _dirs, files in os.walk(dir_path):
                for fname in sorted(files):
                    full = os.path.join(root, fname)
                    arcname = os.path.relpath(full, dir_path)
                    zf.write(full, arcname)
                    n_files += 1

        # Quick integrity check — make sure we can open and list contents.
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"Corrupt entry in archive: {bad}")
            if len(zf.namelist()) != n_files:
                raise zipfile.BadZipFile(
                    f"Entry count mismatch: wrote {n_files}, archive has {len(zf.namelist())}"
                )

        os.replace(tmp_zip, zip_path)
    except BaseException:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise

    shutil.rmtree(dir_path)
    print(f"[zip_utils] Zipped {n_files} files: {dir_path} -> {zip_path}")
    return zip_path


def unzip_if_needed(
    zip_path: str,
    target_dir: Optional[str] = None,
) -> str:
    """
    Extract *zip_path* into *target_dir* **only** when the directory does not
    already exist (idempotent).

    Args:
        zip_path: Path to the zip archive.
        target_dir: Where to extract.  Defaults to the zip path minus ``.zip``.

    Returns:
        Absolute path of the (existing or newly created) directory.
    """
    zip_path = os.path.abspath(zip_path)
    if target_dir is None:
        if zip_path.endswith(".zip"):
            target_dir = zip_path[:-4]
        else:
            target_dir = zip_path + "_extracted"
    target_dir = os.path.abspath(target_dir)

    if os.path.isdir(target_dir):
        return target_dir

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Neither directory {target_dir} nor archive {zip_path} exists."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    n = len(os.listdir(target_dir))
    print(f"[zip_utils] Extracted {n} entries: {zip_path} -> {target_dir}")
    return target_dir


def cleanup_extracted_dir(dir_path: str, zip_path: Optional[str] = None) -> None:
    """
    Remove *dir_path* only when the corresponding zip archive already exists
    on disk (safety guard so we never delete the only copy).

    Args:
        dir_path: Directory to remove.
        zip_path: Expected zip archive.  Defaults to ``<dir_path>.zip``.
    """
    dir_path = os.path.abspath(dir_path)
    if zip_path is None:
        zip_path = dir_path + ".zip"
    zip_path = os.path.abspath(zip_path)

    if not os.path.isdir(dir_path):
        return

    if not os.path.isfile(zip_path):
        print(
            f"[zip_utils] Skipping cleanup of {dir_path}: "
            f"archive {zip_path} not found (keeping directory as safety measure)."
        )
        return

    shutil.rmtree(dir_path)
    print(f"[zip_utils] Cleaned up extracted directory: {dir_path}")
