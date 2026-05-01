"""
Run this first! Creates all required project directories.
Usage: python scripts/setup_folders.py
"""
import os
from pathlib import Path

dirs = [
    "data/raw/train", "data/raw/val", "data/raw/test",
    "data/processed/frames", "data/processed/features",
    "models/saved", "models/checkpoints",
    "results/plots", "results/reports",
    "logs",
]

print("🗂️  Creating SHAR Project Folder Structure")
print("=" * 45)
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {d}/")

print("\n📌 Next steps:")
print("  1. Download dataset from Kaggle:")
print("     https://www.kaggle.com/datasets/daudshah/video-dataset")
print("  2. Place class folders inside:")
print("     data/raw/train/<ClassName>/")
print("     data/raw/val/<ClassName>/")
print("     data/raw/test/<ClassName>/")
print("  3. Install dependencies:  pip install -r requirements.txt")
print("  4. Open notebooks/ and run Phase 1→6 in order")
print("  5. Launch web app:  streamlit run webapp/app.py")
