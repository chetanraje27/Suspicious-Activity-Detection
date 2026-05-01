"""
Single video inference + Grad-CAM visualization.
Usage: python src/predict.py --video path/to/video.mp4 --checkpoint models/saved/cnn_lstm_best.pth
"""
import os, sys, argparse, cv2
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_device, load_config, CLASS_NAMES, SUSPICIOUS_CLASSES
from dataset import extract_frames, get_transforms
from model import get_model

def predict_video(video_path, model, device, cfg, threshold=0.5):
    transform = get_transforms("val")
    frames = extract_frames(video_path, cfg["dataset"]["frames_per_video"])
    if frames is None:
        return None, None, None
    tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(), autocast():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    is_suspicious = pred_class in SUSPICIOUS_CLASSES and confidence >= threshold
    return pred_class, confidence, is_suspicious, probs

def create_annotated_video(video_path, pred_class, confidence, is_suspicious, output_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    color = (0, 0, 255) if is_suspicious else (0, 200, 0)
    label = f"{'⚠ ALERT: ' if is_suspicious else ''}{pred_class} ({confidence*100:.1f}%)"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        out.write(frame)
    cap.release(); out.release()
    print(f"📹 Annotated video saved → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="cnn_lstm")
    parser.add_argument("--checkpoint", default="models/saved/cnn_lstm_best.pth")
    parser.add_argument("--output", default="results/annotated_output.mp4")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    device = get_device()
    model = get_model(args.model, num_classes=21).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    result = predict_video(args.video, model, device, cfg)
    pred_class, confidence, is_suspicious, probs = result
    print(f"\n📌 Prediction: {pred_class}")
    print(f"   Confidence:  {confidence*100:.2f}%")
    print(f"   Alert:       {'🔴 YES' if is_suspicious else '🟢 NO'}")

    os.makedirs("results", exist_ok=True)
    create_annotated_video(args.video, pred_class, confidence, is_suspicious, args.output)
