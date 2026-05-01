"""
Dataset & DataLoader for SHAR.
Supports flat folder structure with .txt annotation files.
Each video → 20 uniformly-sampled frames → tensor (20, 3, 224, 224).
"""
import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

# ─── Class Names (matches lable.txt order) ────────────────
CLASS_NAMES = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","RoadAccidents","Robbery","Shooting","Shoplifting",
    "Stealing","Vandalism","Normal","Walking","WalkingUsingPhone",
    "WalkingReadingBook","StandingStill","Sitting","MeetAndSplit","Clapping"
]

# ─── Transforms ───────────────────────────────────────────
def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

# ─── Video → Frames ───────────────────────────────────────
def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]

# ─── Read .txt annotation file ────────────────────────────
def load_annotations(txt_path, video_dir, label_names):
    """
    Reads train.txt / test.txt / valid.txt
    Each line is either:
      filename.mp4 classindex        (e.g. "Abuse001.mp4 0")
      OR
      classname/filename.mp4 classindex  (e.g. "Abuse/Abuse001.mp4 0")
    Returns list of (video_full_path, label_index)
    """
    samples = []
    missing = 0

    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue

        # Handle both "filename.mp4 0" and "class/filename.mp4 0"
        raw_path = parts[0]
        label_idx = int(parts[1])

        # Strip subfolder prefix if present (e.g. "Abuse/video.mp4" → "video.mp4")
        filename = Path(raw_path).name

        full_path = Path(video_dir) / filename

        if not full_path.exists():
            # Try with the raw path as-is
            full_path = Path(video_dir) / raw_path
            if not full_path.exists():
                missing += 1
                continue

        if label_idx >= len(label_names):
            continue

        samples.append((str(full_path), label_idx))

    if missing > 0:
        print(f"  ⚠️  {missing} videos listed in txt not found in folder (skipped)")

    return samples

# ─── Auto-detect class names from lable.txt ───────────────
def load_class_names(label_txt_path):
    if not os.path.exists(label_txt_path):
        print(f"⚠️  lable.txt not found at {label_txt_path}, using default class names")
        return CLASS_NAMES
    with open(label_txt_path, "r") as f:
        names = [l.strip() for l in f.readlines() if l.strip()]
    print(f"  ✅ Loaded {len(names)} class names from lable.txt")
    return names

# ─── Dataset Class ─────────────────────────────────────────
class SHARDataset(Dataset):
    def __init__(self, data_root, split="train", num_frames=20, transform=None):
        """
        data_root/
            train/    *.mp4  (flat, no subfolders)
            val/      *.mp4
            test/     *.mp4
            lable.txt
            train.txt
            valid.txt
            test.txt
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.transform = transform or get_transforms(split)

        # Map split name to folder and txt file
        split_map = {
            "train": ("train", "train.txt"),
            "val":   ("valid", "valid.txt"),
            "test":  ("test",  "test.txt"),
        }
        if split not in split_map:
            raise ValueError(f"split must be train/val/test, got: {split}")

        folder_name, txt_name = split_map[split]
        
        # Flexibility for validation folder naming
        if split == "val" and not (self.data_root / folder_name).exists():
            if (self.data_root / "val").exists():
                folder_name = "val"

        self.video_dir = self.data_root / folder_name
        self.txt_path  = self.data_root / txt_name
        self.label_txt = self.data_root / "lable.txt"

        # Load class names
        self.class_names = load_class_names(str(self.label_txt))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Load samples from txt file
        if not self.txt_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.txt_path}")
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video folder not found: {self.video_dir}")

        self.samples = load_annotations(
            str(self.txt_path),
            str(self.video_dir),
            self.class_names
        )
        print(f"  [{split}] {len(self.samples)} videos | "
              f"{len(set(s[1] for s in self.samples))} classes | "
              f"from {self.video_dir.name}/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = extract_frames(video_path, self.num_frames)
        if frames is None:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
        tensor_frames = torch.stack([self.transform(f) for f in frames])
        return tensor_frames, label, video_path

def get_dataloader(data_root, split, batch_size=8,
                   num_workers=4, num_frames=20):
    ds = SHARDataset(data_root, split, num_frames)
    shuffle = (split == "train")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader, ds