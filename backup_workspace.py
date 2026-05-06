#!/usr/bin/env python3
"""
Cria um backup ZIP do workspace TurboQuant-Vulkan.

Uso:
    py backup_workspace.py
"""

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "backups"
EXCLUDE = {".git", ".venv", "backups", "__pycache__"}


def should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE:
            return False
    return True


def create_backup() -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = BACKUP_DIR / f"turboquant_vulkan_backup_{timestamp}.zip"

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in Path(ROOT).rglob("*"):
            if not should_include(path):
                continue
            if path.is_dir():
                continue
            archive.write(path, arcname=path.relative_to(ROOT))

    return archive_path


if __name__ == "__main__":
    archive = create_backup()
    print(f"Backup criado: {archive}")
