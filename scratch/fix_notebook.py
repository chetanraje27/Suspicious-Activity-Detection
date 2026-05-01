import nbformat
import os

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/01_EDA_and_Preprocessing.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Target cell starts with "# Scan dataset folder structure"
target_text = "# Scan dataset folder structure"

for cell in nb.cells:
    if cell.cell_type == 'code' and target_text in cell.source:
        print("Found target cell. Updating...")
        new_source = """# Scan dataset folder structure
DATA_ROOT = Path("../data/raw")
splits = ["train", "val", "test"]

records = []
for split in splits:
    split_path = DATA_ROOT / split
    if not split_path.exists():
        print(f"⚠️  '{split}' folder not found at {split_path.absolute()}. Please place dataset there.")
        continue
    for cls_dir in sorted(split_path.iterdir()):
        if cls_dir.is_dir():
            videos = list(cls_dir.glob("*.mp4"))
            records.append({"split": split, "class": cls_dir.name, "count": len(videos)})

if not records:
    print("❌ Error: No video files found in the dataset directory. Please check the path and folder structure.")
    df = pd.DataFrame(columns=["split", "class", "count"])
else:
    df = pd.DataFrame(records)
    print("Dataset Summary:")
    print(df.groupby("split")["count"].sum())
    print(f"\\nTotal unique classes: {df['class'].nunique()}")
    df.head(25)"""
        cell.source = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
