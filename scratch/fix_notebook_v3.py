import nbformat
import os

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/01_EDA_and_Preprocessing.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Update Section 1.3 (Video Properties EDA)
target_1_3 = "## 1.3 — Video Properties EDA"
source_1_3 = r"""# Analyze video durations and frame counts
from tqdm.notebook import tqdm

sample_videos = []
for split in ["train"]:
    split_path = DATA_ROOT / split
    if not split_path.exists():
        break
    
    # Handle both structures
    all_videos = []
    subdirs = [d for d in split_path.iterdir() if d.is_dir()]
    if subdirs:
        for cls_dir in subdirs:
            all_videos.extend(list(cls_dir.glob("*.mp4")))
    else:
        all_videos = list(split_path.glob("*.mp4"))

    # Sample some videos for analysis (e.g., 20 videos)
    import random
    if len(all_videos) > 20:
        sampled = random.sample(all_videos, 20)
    else:
        sampled = all_videos

    for vid in tqdm(sampled, desc=f"Analyzing {split} samples"):
        cap = cv2.VideoCapture(str(vid))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if frames > 0 and fps > 0:
            sample_videos.append({
                "class": get_class_from_name(vid.name), "frames": frames,
                "fps": fps, "duration_s": frames/fps,
                "width": w, "height": h
            })

if not sample_videos:
    print("⚠️  No valid video samples found for EDA.")
    vid_df = pd.DataFrame(columns=["class", "frames", "fps", "duration_s", "width", "height"])
else:
    vid_df = pd.DataFrame(sample_videos)
    print("Video Statistics (sample):")
    display(vid_df[["frames", "fps", "duration_s", "width", "height"]].describe().round(2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(vid_df["duration_s"], bins=15, color="#3498db", edgecolor="black")
    axes[0].set_title("Video Duration Distribution (seconds)")
    axes[0].set_xlabel("Duration (s)")
    axes[1].hist(vid_df["frames"], bins=15, color="#e67e22", edgecolor="black")
    axes[1].set_title("Frame Count Distribution")
    axes[1].set_xlabel("Frame Count")
    plt.tight_layout(); plt.show()
"""

# 2. Update Section 1.4 (Frame Extraction)
target_1_4 = "## 1.4 — Frame Extraction & Visualization"
source_1_4 = r"""from dataset import extract_frames

# Show extracted frames from a sample video
sample_class = "Fighting"
# Find a video belonging to sample_class in the flat or subdir structure
sample_vid_path = None

# Search in train folder
train_path = DATA_ROOT / "train"
if train_path.exists():
    # Try flat structure first
    videos = list(train_path.glob("*.mp4"))
    for v in videos:
        if get_class_from_name(v.name) == sample_class:
            sample_vid_path = v
            break
    
    # Try subdirectory if not found
    if not sample_vid_path:
        class_dir = train_path / sample_class
        if class_dir.exists():
            vids = list(class_dir.glob("*.mp4"))
            if vids:
                sample_vid_path = vids[0]

if sample_vid_path:
    vid = str(sample_vid_path)
    frames = extract_frames(vid, num_frames=10)
    print(f"Extracted {len(frames)} frames from: {sample_vid_path.name}")
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for i, (ax, frame) in enumerate(zip(axes.flat, frames)):
        ax.imshow(frame)
        ax.set_title(f"Frame {i+1}")
        ax.axis("off")
    fig.suptitle(f"Sample Frames — Class: {sample_class}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../results/plots/sample_frames.png", dpi=100)
    plt.show()
else:
    print(f"⚠️  No videos found for class '{sample_class}'. Please check dataset.")
"""

# Apply updates
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        if target_1_3 in cell.source:
            # The next cell should be the code cell for 1.3
            if i + 1 < len(nb.cells) and nb.cells[i+1].cell_type == 'code':
                print(f"Updating code cell for {target_1_3}")
                nb.cells[i+1].source = source_1_3
        elif target_1_4 in cell.source:
             # The next cell should be the code cell for 1.4
            if i + 1 < len(nb.cells) and nb.cells[i+1].cell_type == 'code':
                print(f"Updating code cell for {target_1_4}")
                nb.cells[i+1].source = source_1_4

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully (Sections 1.3 and 1.4).")
