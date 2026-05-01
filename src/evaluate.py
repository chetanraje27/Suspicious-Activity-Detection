"""
Evaluation pipeline — confusion matrix, classification report, per-class accuracy.
Usage: python src/evaluate.py --model cnn_lstm --checkpoint models/saved/cnn_lstm_best.pth
"""
import os, sys, argparse
import torch
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_device, load_config, plot_confusion_matrix, CLASS_NAMES
from dataset import get_dataloader
from model import get_model

@torch.no_grad()
def evaluate(args):
    cfg = load_config("config.yaml")
    device = get_device()

    print(f"\n🔍 Evaluating: {args.model} | Checkpoint: {args.checkpoint}")
    test_loader, ds = get_dataloader(cfg["dataset"]["root"], "test", batch_size=8,
                                     num_workers=cfg["training"]["num_workers"])

    model = get_model(args.model, num_classes=cfg["dataset"]["num_classes"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    print("✅ Model loaded.")

    all_preds, all_labels, all_confs = [], [], []
    for frames, labels, paths in tqdm(test_loader, desc="Evaluating"):
        frames = frames.to(device)
        with autocast():
            outputs = model(frames)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confs.extend(confs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100 * (all_preds == all_labels).mean()
    print(f"\n🎯 Test Accuracy: {acc:.2f}%")

    names = CLASS_NAMES[:cfg["dataset"]["num_classes"]]
    print("\n" + classification_report(all_labels, all_preds, target_names=names))

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("results/plots", exist_ok=True)
    save_path = f"results/plots/{args.model}_confusion_matrix.png"
    plot_confusion_matrix(cm, names, save_path=save_path)
    print(f"📊 Confusion matrix saved → {save_path}")

    # Per-class accuracy
    print("\n📋 Per-Class Accuracy:")
    for i, name in enumerate(names):
        mask = all_labels == i
        if mask.sum() > 0:
            cls_acc = 100 * (all_preds[mask] == i).mean()
            tag = "🔴" if i < 13 else "🟢"
            print(f"  {tag} {name:<25} {cls_acc:.1f}%  ({mask.sum()} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn_lstm")
    parser.add_argument("--checkpoint", type=str, default="models/saved/cnn_lstm_best.pth")
    args = parser.parse_args()
    evaluate(args)
