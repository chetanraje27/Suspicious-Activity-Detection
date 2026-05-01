import nbformat
import os

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/01_EDA_and_Preprocessing.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Target cell starts with "# Scan dataset folder structure"
target_text = "# Scan dataset folder structure"

new_source = r"""# Scan dataset folder structure
DATA_ROOT = Path("../data/raw")
splits = ["train", "val", "test"]

# Helper to map filenames to class names
def get_class_from_name(name):
    name = name.lower()
    if "abuse" in name: return "Abuse"
    if "arrest" in name: return "Arrest"
    if "arson" in name: return "Arson"
    if "assault" in name: return "Assault"
    if "burglary" in name: return "Burglary"
    if "explosion" in name: return "Explosion"
    if "fighting" in name: return "Fighting"
    if "roadaccident" in name: return "RoadAccidents"
    if "robbery" in name: return "Robbery"
    if "shooting" in name: return "Shooting"
    if "shoplifting" in name: return "Shoplifting"
    if "stealing" in name: return "Stealing"
    if "vandalism" in name: return "Vandalism"
    if "normal" in name: return "Normal"
    if "walking_while_using_phone" in name or "walkingusingphone" in name: return "WalkingUsingPhone"
    if "walking_while_reading_book" in name or "walkingreadingbook" in name: return "WalkingReadingBook"
    if "walking" in name: return "Walking"
    if "standing_still" in name or "standingstill" in name: return "StandingStill"
    if "sitting" in name: return "Sitting"
    if "meet_and_split" in name or "meetandsplit" in name: return "MeetAndSplit"
    if "clapping" in name: return "Clapping"
    return "Unknown"

records = []
for split in splits:
    split_path = DATA_ROOT / split
    if not split_path.exists():
        print(f"⚠️  '{split}' folder not found at {split_path.absolute()}.")
        continue
    
    # Check if subdirectories exist
    subdirs = [d for d in split_path.iterdir() if d.is_dir()]
    
    if subdirs:
        # Subdirectory structure
        for cls_dir in sorted(subdirs):
            videos = list(cls_dir.glob("*.mp4"))
            records.append({"split": split, "class": cls_dir.name, "count": len(videos)})
    else:
        # Flat structure
        videos = list(split_path.glob("*.mp4"))
        if videos:
            class_counts = {}
            for v in videos:
                cls = get_class_from_name(v.name)
                class_counts[cls] = class_counts.get(cls, 0) + 1
            for cls, count in class_counts.items():
                records.append({"split": split, "class": cls, "count": count})

if not records:
    print("❌ Error: No video files found in the dataset directory. Please check the path and folder structure.")
    df = pd.DataFrame(columns=["split", "class", "count"])
else:
    df = pd.DataFrame(records)
    print("Dataset Summary:")
    summary = df.groupby("split")["count"].sum()
    print(summary)
    print(f"\nTotal unique classes found: {df['class'].nunique()}")
    print(f"Total videos across all splits: {summary.sum()}")
    display(df.head(25))
"""

for cell in nb.cells:
    if cell.cell_type == 'code' and target_text in cell.source:
        cell.source = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully with flat folder support.")
