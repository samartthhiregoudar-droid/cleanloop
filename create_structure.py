"""Creates the CleanLoop folder structure. Run once, then you can delete it."""
from pathlib import Path

FOLDERS = [
    "vision",
    "backend/api",
    "frontend",
    "storage/snapshots",
    "storage/clips",
    "storage/audio",
    "tests",
    "eval/videos",
]

PACKAGES = ["vision", "backend", "backend/api", "tests"]

for folder in FOLDERS:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"created {folder}/")

for pkg in PACKAGES:
    (Path(pkg) / "__init__.py").touch(exist_ok=True)

print("\nDone! Folder structure is ready.")
