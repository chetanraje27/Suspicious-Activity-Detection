"""
SHAR — Suspicious Human Activity Recognition
Streamlit Web Application
Run: streamlit run webapp/app.py
"""
import streamlit as st
import streamlit.components.v1 as components
import sys, os, cv2, time, tempfile
import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import CLASS_NAMES, SUSPICIOUS_CLASSES, get_device
from dataset import extract_frames, get_transforms
from model import get_model

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="SHAR — Suspicious Activity Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #252535);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .alert-red {
        background: linear-gradient(135deg, #3d0000, #5c0a0a);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .detection-log {
        background: #161b22;
        border-radius: 8px;
        padding: 10px;
        height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', Courier, monospace;
        border-left: 4px solid #e74c3c;
    }
    .log-entry { margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    .log-time { color: #8b949e; font-size: 0.8em; }
    .log-msg { color: #e74c3c; font-weight: bold; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(231,76,60,0.7); }
        70%  { box-shadow: 0 0 0 10px rgba(231,76,60,0); }
        100% { box-shadow: 0 0 0 0 rgba(231,76,60,0); }
    }
    .sidebar-title { font-size: 24px; font-weight: bold; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔍 SHAR System</div>', unsafe_allow_html=True)
    st.caption("Suspicious Human Activity Recognition")
    st.divider()

    st.subheader("⚙️ Settings")
    model_choice = st.selectbox("Model Architecture",
                                ["cnn_gru", "cnn_lstm", "cnn_baseline"],
                                index=0,
                                help="CNN-GRU is currently the default selected model")
    conf_threshold = st.slider("Alert Confidence Threshold", 0.3, 0.95, 0.60, 0.05,
                               help="Minimum confidence to trigger a SUSPICIOUS alert")
    num_frames = st.slider("Frames to Sample", 10, 30, 20,
                           help="Number of frames extracted per video")

    st.divider()
    st.subheader("📋 Dataset Info")
    st.info("**21 Classes:**\n- 🔴 13 Suspicious\n- 🟢 8 Normal\n- 1,334 total videos")

    st.divider()
    st.subheader("🔴 Alert Classes")
    for cls in SUSPICIOUS_CLASSES:
        st.markdown(f"• {cls}")

# ── Model Loading ────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    try:
        device = get_device()
        m = get_model(model_name, num_classes=21).to(device)
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "models", "saved", f"{model_name}_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            m.eval()
            return m, device, True
        else:
            return m, device, False
    except Exception as e:
        return None, None, False

def trigger_alert_audio(action_name):
    """Triggers a browser beep and text-to-speech alert."""
    components.html(
        f"""
        <script>
            // Clear any previous speech immediately
            window.speechSynthesis.cancel();
            if(window.parent) window.parent.speechSynthesis.cancel();

            // 1. Play Alarm Sound (Beep)
            function playBeep() {{
                const context = new (window.AudioContext || window.webkitAudioContext)();
                const osc = context.createOscillator();
                const gain = context.createGain();
                osc.connect(gain);
                gain.connect(context.destination);
                osc.type = "sawtooth"; 
                osc.frequency.value = 523.25; 
                gain.gain.value = 0.5;
                osc.start();
                setTimeout(() => {{ osc.stop(); }}, 400);
            }}
            playBeep();

            // 2. Text-to-Speech
            setTimeout(() => {{
                const msg = new SpeechSynthesisUtterance();
                msg.text = "Alert! Suspicious: {action_name}.";
                msg.rate = 1.1; 
                window.speechSynthesis.speak(msg);
            }}, 500);
        </script>
        """,
        height=0,
    )

def stop_audio():
    """Stops all ongoing browser speech alerts aggressively."""
    components.html(
        """
        <script>
            // Aggressively cancel all speech
            if ('speechSynthesis' in window) {
                window.speechSynthesis.pause();
                window.speechSynthesis.cancel();
                
                // Double check for window.parent if in Streamlit iframe
                if (window.parent && window.parent.speechSynthesis) {
                    window.parent.speechSynthesis.pause();
                    window.parent.speechSynthesis.cancel();
                }
            }
        </script>
        """,
        height=0,
    )

# ── Main UI ──────────────────────────────────────────────
st.title("🔍 Suspicious Human Activity Recognition")

# 🔕 Master Muzzle: Strictly stop any background speech on rerun if not supposed to be running
if not st.session_state.get("is_running", False):
    components.html("<script>window.speechSynthesis.cancel(); if(window.parent) window.parent.speechSynthesis.cancel();</script>", height=0)

st.caption("Deep Learning-powered real-time suspicious activity detection | 21-class classification")

# Status bar
model, device, model_loaded = load_model(model_choice)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", model_choice.upper().replace("_", "-"))
col2.metric("Device", str(device).upper() if device else "N/A")
col3.metric("Classes", "21 (13+8)")
col4.metric("Threshold", f"{conf_threshold:.0%}")

if not model_loaded:
    st.warning("⚠️ No trained model found. Please train a model first using the notebooks. "
               "The app will still run but predictions may be random.")

st.divider()

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📹 Video Analysis", "📊 Live Statistics", "ℹ️ About"])

# ─ TAB 1: Video Upload ───────────────────────────────────
with tab1:
    st.subheader("Upload a Video for Analysis")
    uploaded = st.file_uploader("Choose an MP4 video file", type=["mp4", "avi", "mov"],
                                 help="Upload any CCTV/surveillance style video")

    if uploaded:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()
        vid_path = tfile.name

        col_vid, col_res = st.columns([1.5, 1])

        with col_vid:
            monitor_placeholder = st.empty()
            st.caption("🔴 LIVE MONITOR SIMULATION")

        with col_res:
            alert_placeholder = st.empty()
            audio_placeholder = st.empty()
            st.subheader("📋 Activity Log")
            log_placeholder = st.empty()
            log_entries = []

            btn_col1, btn_col2 = st.columns(2)
            start_btn = btn_col1.button("🚀 Start Analysis", type="primary", use_container_width=True)
            stop_btn = btn_col2.button("🛑 Stop", type="secondary", use_container_width=True)

            if stop_btn:
                st.session_state.is_running = False
                audio_placeholder.empty()
                stop_audio()
                st.rerun()

            if start_btn:
                st.session_state.is_running = True
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: fps = 30
                
                window_frames = []
                frame_count = 0
                
                # Model Setup
                model.eval()
                transform = get_transforms("val")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    timestamp_sec = frame_count / fps
                    timestamp_str = time.strftime('%M:%S', time.gmtime(timestamp_sec))
                    
                    # Display Frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    monitor_placeholder.image(frame_rgb, use_container_width=True)
                    
                    # Buffer for inference
                    window_frames.append(frame_rgb)
                    
                    if len(window_frames) == num_frames:
                        # Prepare tensor
                        tensor = torch.stack([transform(f) for f in window_frames]).unsqueeze(0).to(device)
                        
                        # Inference
                        with torch.no_grad():
                            try:
                                from torch.cuda.amp import autocast
                                with autocast():
                                    logits = model(tensor)
                            except:
                                logits = model(tensor)
                        
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        pred_idx = probs.argmax()
                        pred_class = CLASS_NAMES[pred_idx]
                        confidence = float(probs[pred_idx])
                        
                        # Detection Logic
                        if pred_class in SUSPICIOUS_CLASSES and confidence >= conf_threshold:
                            # 🚨 ALERT
                            alert_placeholder.markdown(f"""
                            <div class="alert-red">
                                <h3>🚨 SUSPICIOUS: {pred_class.upper()}</h3>
                                <p>Confidence: {confidence:.1%} | Time: {timestamp_str}</p>
                            </div>""", unsafe_allow_html=True)
                            
                            # 🔊 Ring Sound (Only if still running)
                            if st.session_state.get("is_running", False):
                                with audio_placeholder:
                                    trigger_alert_audio(pred_class)
                            
                            # 📝 Log Entry
                            log_entries.append(f"<div class='log-entry'><span class='log-time'>[{timestamp_str}]</span> <span class='log-msg'>{pred_class} detected!</span></div>")
                        else:
                            alert_placeholder.markdown(f"""
                            <div class="alert-green" style="background: linear-gradient(135deg, #003d10, #0a5c1a); border: 2px solid #2ecc71; border-radius: 12px; padding: 20px; text-align: center;">
                                <h3>✅ STATUS: SAFE</h3>
                                <p>Activity: {pred_class} | Conf: {confidence:.1%}</p>
                            </div>""", unsafe_allow_html=True)
                        
                        # Update Log UI
                        log_html = f"<div class='detection-log'>{''.join(log_entries[::-1])}</div>"
                        log_placeholder.markdown(log_html, unsafe_allow_html=True)
                        
                        # Slide window (keep overlap or clear)
                        window_frames = window_frames[num_frames//2:] # 50% overlap for smoother detection
                    
                    # Small sleep to simulate real-time playback speed
                    time.sleep(0.01) 

                cap.release()
                st.success("✅ Analysis Complete")

        os.unlink(vid_path)

    else:
        st.info("👆 Upload a video to start activity detection")
        st.markdown("**Supported formats:** MP4, AVI, MOV")

        # Demo class list
        st.subheader("🏷️ Detectable Activity Classes")
        c1, c2 = st.columns(2)
        with c1:
            st.error("🔴 **Suspicious Classes (13)**")
            for cls in SUSPICIOUS_CLASSES:
                st.markdown(f"  • {cls}")
        with c2:
            st.success("🟢 **Normal Classes (8)**")
            from utils import NORMAL_CLASSES
            for cls in NORMAL_CLASSES:
                st.markdown(f"  • {cls}")

# ─ TAB 2: Statistics ─────────────────────────────────────
with tab2:
    st.subheader("📊 Dataset & Model Statistics")

    # Class distribution chart
    class_counts = {
        "Abuse":17,"Arrest":17,"Arson":17,"Assault":17,"Burglary":17,"Explosion":17,
        "Fighting":50,"RoadAccidents":50,"Robbery":17,"Shooting":17,"Shoplifting":17,
        "Stealing":17,"Vandalism":17,"Normal":140,"Walking":100,"WalkingUsingPhone":80,
        "WalkingReadingBook":80,"StandingStill":80,"Sitting":80,"MeetAndSplit":80,"Clapping":80
    }
    colors = ["#e74c3c" if c in SUSPICIOUS_CLASSES else "#2ecc71" for c in class_counts]

    fig3 = px.bar(x=list(class_counts.keys()), y=list(class_counts.values()),
                  title="Approximate Class Distribution (Training Set)",
                  color=list(class_counts.values()),
                  color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"])
    fig3.update_layout(template="plotly_dark", xaxis_tickangle=-45,
                       height=400, showlegend=False, margin=dict(b=120))
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Videos", "1,334")
    col_b.metric("Train / Val / Test", "940 / 194 / 200")
    col_c.metric("Total Classes", "21 (13 Suspicious + 8 Normal)")

# ─ TAB 3: About ──────────────────────────────────────────
with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ## Suspicious Human Activity Recognition (SHAR)

    ### 🎯 Objective
    Detect suspicious activities in video surveillance footage using deep learning.

    ### 🏗️ Architecture
    | Phase | Component | Technology |
    |-------|-----------|------------|
    | 1 | Dataset Preparation | OpenCV, Pandas |
    | 2 | Preprocessing & Feature Extraction | torchvision transforms |
    | 3 | CNN Baseline | ResNet-50 (ImageNet) |
    | 4 | CNN-LSTM (Main) | ResNet-50 + Bi-LSTM + Attention |
    | 5 | CNN-GRU (Lightweight) | MobileNetV3 + GRU |
    | 6 | Explainability | Grad-CAM + Attention Visualization |

    ### 📊 Dataset
    - **1,334 MP4 videos** across 21 activity classes
    - **13 Suspicious** + **8 Normal** categories
    - Split: 940 Train | 194 Val | 200 Test

    ### 🖥️ Hardware Used
    - GPU: NVIDIA GeForce RTX 3050 6GB
    - CPU: Intel Core i5-12450H
    - RAM: 16 GB

    ### 📚 References
    - UCF-Crime Dataset (Sultani et al., 2018)
    - LSTM for Video Classification (Donahue et al., 2015)
    - Grad-CAM (Selvaraju et al., 2017)
    """)
