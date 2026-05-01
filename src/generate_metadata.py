import os
from pathlib import Path

# Class mapping (same as in updated notebook)
CLASS_NAMES = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","RoadAccidents","Robbery","Shooting","Shoplifting",
    "Stealing","Vandalism","Normal","Walking","WalkingUsingPhone",
    "WalkingReadingBook","StandingStill","Sitting","MeetAndSplit","Clapping"
]

def get_class_idx(name):
    name = name.lower()
    if "abuse" in name: return 0
    if "arrest" in name: return 1
    if "arson" in name: return 2
    if "assault" in name: return 3
    if "burglary" in name: return 4
    if "explosion" in name: return 5
    if "fighting" in name: return 6
    if "roadaccident" in name: return 7
    if "robbery" in name: return 8
    if "shooting" in name: return 9
    if "shoplifting" in name: return 10
    if "stealing" in name: return 11
    if "vandalism" in name: return 12
    if "normal" in name: return 13
    if "walking_while_using_phone" in name or "walkingusingphone" in name: return 15
    if "walking_while_reading_book" in name or "walkingreadingbook" in name: return 16
    if "walking" in name: return 14
    if "standing_still" in name or "standingstill" in name: return 17
    if "sitting" in name: return 18
    if "meet_and_split" in name or "meetandsplit" in name: return 19
    if "clapping" in name: return 20
    return 13 # Default to Normal if unknown

def generate():
    data_root = Path("d:/Downloads/SHAR_Complete_Project/SHAR/data/raw")
    
    # 1. Generate lable.txt
    with open(data_root / "lable.txt", "w") as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")
    print(f"Generated lable.txt")

    # 2. Generate train.txt, valid.txt, test.txt
    splits = {
        "train": "train.txt",
        "val": "valid.txt",
        "test": "test.txt"
    }

    for folder, txt_file in splits.items():
        folder_path = data_root / folder
        if not folder_path.exists():
            print(f"Folder {folder} not found, skipping {txt_file}")
            continue
        
        videos = list(folder_path.glob("*.mp4"))
        with open(data_root / txt_file, "w") as f:
            for v in videos:
                idx = get_class_idx(v.name)
                # We save just the filename since dataset.py joins it with video_dir
                f.write(f"{v.name} {idx}\n")
        print(f"Generated {txt_file} with {len(videos)} entries")

if __name__ == "__main__":
    generate()
