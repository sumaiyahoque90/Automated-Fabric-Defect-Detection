"""
FabDetector - Smart Fabric Defect Detection System
Built with Streamlit + YOLOv8

Run with:  streamlit run app.py
"""

import os
import base64
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="FabDetector",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "/Users/bmshahriaalam/Desktop/Paper/Fabric/App/best.pt"
BG_IMAGE_PATH = "/Users/bmshahriaalam/Desktop/Paper/Fabric/App/Assets/B1.jpg"

CLASS_NAMES = ['defect_free', 'hole', 'knot_slub', 'spot', 'yarn']

# Colors per class (BGR won't matter, ultralytics handles its own palette,
# these are used only for our custom badges/metrics in the UI)
CLASS_COLORS = {
    "defect_free": "#2ecc71",
    "hole": "#e74c3c",
    "knot_slub": "#f39c12",
    "spot": "#9b59b6",
    "yarn": "#3498db",
}

DEFAULT_CONF = 0.50   # confidence threshold used to decide "real" detections
MIN_CONF_FOR_MISC = 0.50  # below this, treat frame/image as "Miscellaneous Items"


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it across reruns."""
    return YOLO(MODEL_PATH)


def img_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_welcome_background():
    b64 = img_to_base64(BG_IMAGE_PATH)
    if b64:
        bg_css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient(rgba(10,15,30,0.55), rgba(10,15,30,0.65)),
                               url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        </style>
        """
    else:
        # Fallback gradient if no background image is provided
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #16213e, #0f0f1a 70%);
        }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)


def set_main_background():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(160deg, #101c33 0%, #1b2a4a 45%, #22345c 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1524 0%, #14213d 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_global_css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        }

        /* Center-big start button on welcome page */
        /* Global button style */
        div.stButton > button {
            border-radius: 30px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            transition: all 0.25s ease-in-out;
            border: none !important;
        }


        /* Button text fix */
        div.stButton > button p {
            color: white !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
        }


        /* Hover effect */
        div.stButton > button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 25px rgba(0,0,0,0.45);
        }

        .fab-title {
            text-align: center;
            font-size: 3.1rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 4px 18px rgba(0,0,0,0.6);
            margin-bottom: 0.2rem;
            letter-spacing: 1px;
        }
        .fab-subtitle {
            text-align: center;
            font-size: 1.15rem;
            color: #d6e0f0;
            margin-bottom: 2.5rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }

        .glass-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            backdrop-filter: blur(6px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }

        .nav-header {
            color: #ffffff;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .nav-sub {
            color: #9fb3d1;
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }

        .misc-banner {
            background: linear-gradient(90deg, #ff6b6b, #ee5253);
            color: white;
            padding: 14px 20px;
            border-radius: 14px;
            font-size: 1.05rem;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 6px 18px rgba(238,82,83,0.35);
        }

        .class-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            margin-right: 6px;
        }

        h1, h2, h3, h4, p, label, span {
            color: #f0f4fa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_inference_on_image(model, image_bgr, conf_thresh):
    """Run YOLO on a single BGR image. Returns annotated BGR image + list of detections."""
    results = model.predict(image_bgr, conf=conf_thresh, verbose=False)
    r = results[0]
    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else CLASS_NAMES[cls_id]
            detections.append({"class": cls_name, "conf": conf})
        annotated = r.plot()  # BGR numpy array with boxes drawn
    else:
        annotated = image_bgr.copy()

    return annotated, detections


def is_miscellaneous(detections, conf_thresh=MIN_CONF_FOR_MISC):
    """Decide whether the image/frame should be treated as 'Miscellaneous Items'.

    Logic: since the model was trained on exactly 5 fabric-related classes, an
    out-of-distribution input (building, random object, etc.) will typically produce
    either NO boxes above the confidence threshold, or only weak/low-confidence boxes.
    If nothing survives the confidence filter, we label it Miscellaneous.
    """
    if len(detections) == 0:
        return True
    # extra safety: if best detection confidence is still weak, call it misc too
    best_conf = max(d["conf"] for d in detections)
    return best_conf < conf_thresh


def class_badge_html(cls_name, conf):
    color = CLASS_COLORS.get(cls_name, "#7f8c8d")
    return f'<span class="class-badge" style="background:{color}">{cls_name} · {conf*100:.1f}%</span>'


# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "nav" not in st.session_state:
    st.session_state.nav = "image"


# --------------------------------------------------------------------------------------
# WELCOME PAGE
# --------------------------------------------------------------------------------------

def welcome_page():
    set_welcome_background()
    inject_global_css()

    st.markdown(
        """
        <style>

        .welcome-container {
            text-align:center;
            padding-top:100px;
        }

        .fab-title {
            text-align:center;
            font-size:5rem;
            font-weight:900;
            color:white;
            text-shadow:
            0 4px 20px rgba(0,0,0,0.8);
            letter-spacing:1px;
        }


        .fab-subtitle {
            text-align:center;
            font-size:2rem;
            color:#e6edf5;
            margin-top:15px;
            margin-bottom:45px;
            text-shadow:
            0 3px 12px rgba(0,0,0,0.7);
        }


        /* Fancy Start Button */
div.stButton > button {
    width: 170px !important;
    height: 170px !important;
    border-radius: 50% !important;

    background: linear-gradient(
        135deg,
        #00C6FF 0%,
        #0072FF 100%
    ) !important;

    color: white !important;
    border: 3px solid rgba(255,255,255,0.25) !important;

    font-size: 30px !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px;

    box-shadow:
        0 12px 30px rgba(0,114,255,0.45),
        inset 0 3px 8px rgba(255,255,255,0.35);

    transition: all 0.35s ease;
}

/* Hover */
div.stButton > button:hover {

    background: linear-gradient(
        135deg,
        #4FACFE 0%,
        #3494e6 100%
    ) !important;

    transform: translateY(-6px) scale(1.08);

    box-shadow:
        0 20px 40px rgba(0,114,255,0.55),
        inset 0 4px 10px rgba(255,255,255,0.45);
}

/* Click */
div.stButton > button:active {
    transform: scale(0.96);
}

/* Button Text */
div.stButton > button p {
    color: white !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}


        div[data-testid="column"]:nth-of-type(2)
        div.stButton > button:hover {

            background:
            linear-gradient(
                135deg,
                #00c6ff,
                #0066ff
            ) !important;

            transform:
            translateY(-4px)
            scale(1.08);

        }


        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="welcome-container">

        <div class="fab-title">
        🧵 FabDetector
        </div>

        <div class="fab-subtitle">
        A smart system to detect fabric defects
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Center button
    left, center, right = st.columns([2,1,2])

    with center:

        if st.button(
            "🚀 Start",
            key="start_btn",
            use_container_width=True
            
        ):

            st.session_state.page="main"
            st.rerun()


# --------------------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------------------

def sidebar_nav():
    with st.sidebar:
        st.markdown('<div class="nav-header">🧵 FabDetector</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-sub">Fabric defect inspection panel</div>', unsafe_allow_html=True)

        if st.button("🖼️  Upload Picture", use_container_width=True):
            st.session_state.nav = "image"
        if st.button("🎥  Upload Video", use_container_width=True):
            st.session_state.nav = "video"

        st.markdown("---")
        conf = st.slider("Detection confidence threshold", 0.10, 0.95, DEFAULT_CONF, 0.05)

        st.markdown("---")
        st.markdown("**Classes detected**")
        for c in CLASS_NAMES:
            st.markdown(class_badge_html(c, 1.0).replace("· 100.0%", ""), unsafe_allow_html=True)

        st.markdown("---")
        if st.button("⬅️  Back to Welcome", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()

        return conf


def image_section(model, conf_thresh):
    st.header("🖼️ Upload Picture for Defect Detection")
    st.write("Upload a fabric image and FabDetector will highlight defects with bounding boxes, class names, and confidence scores.")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Running detection..."):
            annotated_bgr, detections = run_inference_on_image(model, image_bgr, conf_thresh)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**Detection Result**")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("### Results")
        if is_miscellaneous(detections, conf_thresh):
            st.markdown(
                '<div class="misc-banner">⚠️ Miscellaneous Item — no recognized fabric defect classes detected</div>',
                unsafe_allow_html=True,
            )
            st.caption("This image does not appear to contain fabric defects from the 5 trained classes "
                       "(defect_free, hole, knot_slub, spot, yarn). It has been classified as a Miscellaneous Item.")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            badges = "".join(class_badge_html(d["class"], d["conf"]) for d in detections)
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            for i, d in enumerate(detections, 1):
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{i}. {d['class']}**")
                c2.write(f"{d['conf']*100:.1f}%")


def video_section(model, conf_thresh):
    st.header("🎥 Upload Video for Defect Detection")
    st.write("Upload a fabric inspection video. FabDetector processes it frame-by-frame and draws bounding boxes "
             "with class labels and confidence scores throughout.")

    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_in.write(uploaded.read())
        tfile_in.close()

        st.markdown("**Original Video**")
        st.video(tfile_in.name)

        run_btn = st.button("▶️ Run Detection on Video")

        if run_btn:
            cap = cv2.VideoCapture(tfile_in.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0, text="Processing video frames...")
            status_placeholder = st.empty()
            frame_preview = st.empty()

            misc_frame_count = 0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                annotated_bgr, detections = run_inference_on_image(model, frame, conf_thresh)

                if is_miscellaneous(detections, conf_thresh):
                    misc_frame_count += 1
                    cv2.putText(
                        annotated_bgr, "Miscellaneous Item", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
                    )

                writer.write(annotated_bgr)

                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0),
                                       text=f"Processing frame {frame_idx}/{total_frames}")

                # live-ish preview every ~5 frames so it feels real-time without slowing things down too much
                if frame_idx % 5 == 0:
                    preview_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    frame_preview.image(preview_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

            cap.release()
            writer.release()
            progress.empty()
            status_placeholder.empty()
            frame_preview.empty()

            st.success("Detection complete!")

            if total_frames > 0 and misc_frame_count / total_frames > 0.6:
                st.markdown(
                    '<div class="misc-banner">⚠️ Most of this video does not contain recognized fabric defects — '
                    'likely a Miscellaneous video</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**Annotated Result Video**")
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button("⬇️ Download Annotated Video", data=video_bytes,
                                file_name="fabdetector_result.mp4", mime="video/mp4")


def main_app():
    set_main_background()
    inject_global_css()

    model = load_model()
    conf_thresh = sidebar_nav()

    st.markdown(
        '<h1 style="text-align:center;">FabDetector Inspection Console</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.nav == "image":
        image_section(model, conf_thresh)
    else:
        video_section(model, conf_thresh)


# --------------------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------------------

def main():
    if not os.path.exists(MODEL_PATH):
        st.session_state.page = st.session_state.get("page", "welcome")

    if st.session_state.page == "welcome":
        welcome_page()
    else:
        if not os.path.exists(MODEL_PATH):
            inject_global_css()
            set_main_background()
            st.error(
                f"Model file not found at '{MODEL_PATH}'. Place your trained `best.pt` "
                "in the same folder as app.py, then restart the app."
            )
            if st.button("⬅️ Back to Welcome"):
                st.session_state.page = "welcome"
                st.rerun()
        else:
            main_app()


if __name__ == "__main__":
    main()
