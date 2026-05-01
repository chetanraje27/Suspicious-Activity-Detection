"""Utility functions for SHAR project."""
import os, random, yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU")
    return dev

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def plot_class_distribution(label_counts, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c" if i < 13 else "#2ecc71" for i in range(len(label_counts))]
    bars = ax.bar(label_counts.keys(), label_counts.values(), color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Activity Class", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Class Distribution — SHAR Dataset", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Suspicious (13)"),
                       Patch(facecolor="#2ecc71", label="Normal (8)")]
    ax.legend(handles=legend_elements)
    for bar, val in zip(bars, label_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, str(val),
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — SHAR Model", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train Loss", color="#e74c3c", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", color="#3498db", linewidth=2)
    axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(train_accs, label="Train Acc", color="#e74c3c", linewidth=2)
    axes[1].plot(val_accs, label="Val Acc", color="#3498db", linewidth=2)
    axes[1].set_title("Accuracy Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

CLASS_NAMES = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","RoadAccidents","Robbery","Shooting","Shoplifting",
    "Stealing","Vandalism","Normal","Walking","WalkingUsingPhone",
    "WalkingReadingBook","StandingStill","Sitting","MeetAndSplit","Clapping"
]
SUSPICIOUS_CLASSES = CLASS_NAMES[:13]
NORMAL_CLASSES = CLASS_NAMES[13:]
