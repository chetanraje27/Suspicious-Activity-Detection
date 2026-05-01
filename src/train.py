"""
Training pipeline for SHAR — GPU-accelerated with AMP.
Usage:
    cd SHAR_Project
    python src/train.py --model cnn_lstm --epochs 30 --batch_size 8
"""
import os, sys, argparse, time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, load_config, plot_training_curves, CLASS_NAMES
from dataset import get_dataloader
from model import get_model

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for frames, labels, _ in tqdm(loader, desc="  Train", leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100 * correct / total

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for frames, labels, _ in tqdm(loader, desc="  Val  ", leave=False):
        frames, labels = frames.to(device), labels.to(device)
        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100 * correct / total, all_preds, all_labels

def train(args):
    set_seed(42)
    cfg = load_config("config.yaml")
    device = get_device()

    print(f"\n🚀 Training: {args.model.upper()} | {args.epochs} epochs | BS={args.batch_size}")

    # DataLoaders
    print("\n📁 Loading datasets...")
    train_loader, _ = get_dataloader(cfg["dataset"]["root"], "train",
                                     args.batch_size, cfg["training"]["num_workers"])
    val_loader, _   = get_dataloader(cfg["dataset"]["root"], "val",
                                     args.batch_size, cfg["training"]["num_workers"])

    # Model
    print(f"\n🧠 Building model: {args.model}")
    model = get_model(args.model, num_classes=cfg["dataset"]["num_classes"],
                      dropout=cfg["model"]["dropout"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {total_params:,}")

    # Loss — weighted for class imbalance
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=cfg["training"]["learning_rate"],
                     weight_decay=cfg["training"]["weight_decay"])
    scheduler = StepLR(optimizer, step_size=cfg["training"]["scheduler_step"],
                       gamma=cfg["training"]["scheduler_gamma"])
    scaler = GradScaler()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    os.makedirs(cfg["model"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["model"]["checkpoint_dir"], exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    print("\n" + "="*60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        v_loss, v_acc, preds, labels = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"TLoss={t_loss:.4f} TAcc={t_acc:.1f}%  "
              f"VLoss={v_loss:.4f} VAcc={v_acc:.1f}%  "
              f"LR={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]")

        # Save best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_path = os.path.join(cfg["model"]["save_dir"], f"{args.model}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc,
                "model_name": args.model,
            }, save_path)
            print(f"  💾 Saved best model → {save_path}  (Val Acc: {best_val_acc:.2f}%)")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt = os.path.join(cfg["model"]["checkpoint_dir"], f"{args.model}_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)

    print("="*60)
    print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES[:cfg["dataset"]["num_classes"]]))

    # Save training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path=f"results/plots/{args.model}_training_curves.png")
    print(f"📈 Training curves saved → results/plots/{args.model}_training_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAR Training Script")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                        choices=["cnn_baseline", "cnn_lstm", "cnn_gru"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train(args)
