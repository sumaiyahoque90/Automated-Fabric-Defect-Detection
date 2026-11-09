# FabDetector

A Streamlit web app for fabric defect detection using your trained YOLOv8s model.

## 1. Folder setup

Put these files together like this:

```
FabDetector/
├── app.py
├── requirements.txt
├── best.pt                 <-- COPY YOUR TRAINED MODEL HERE
└── assets/
    └── background.jpg      <-- OPTIONAL: any fabric/textile photo for the welcome page
```

- **best.pt**: copy your trained YOLOv8s weights file into the `FabDetector` folder (same level as `app.py`).
- **assets/background.jpg**: optional. Drop any nice textile/fabric photo here for the welcome page background.
  If you skip this, the app will use a dark gradient background instead — nothing breaks.

## 2. Install dependencies

Open a terminal in the `FabDetector` folder in VS Code and run:

```bash
python -m venv venv
venv\Scripts\activate        # on Windows
# source venv/bin/activate   # on macOS/Linux

pip install -r requirements.txt
```

## 3. Run the app

```bash
streamlit run app.py
```

It will open in your browser at `http://localhost:8501`.

## How it works

- **Welcome page**: title + centered "Start" button + background image.
- **Main app**: left sidebar with "Upload Picture" / "Upload Video" buttons, plus a live
  confidence-threshold slider.
- **Image detection**: draws bounding boxes with class name + confidence score for
  `defect_free`, `hole`, `knot_slub`, `spot`, `yarn`.
- **Video detection**: processes the uploaded video frame-by-frame, draws boxes on every
  frame, then plays back the fully annotated result (with a live preview while it processes).
  Note: true frame-by-frame processing with YOLO is not instant, so for an uploaded video
  file the app processes it first and then streams back the annotated result — this is the
  standard, reliable approach in Streamlit for file uploads (as opposed to a live webcam feed).
- **Miscellaneous Items handling**: if an uploaded image/frame doesn't produce any
  confident detection from the 5 trained classes (e.g. your professor uploads a photo of a
  building), the app labels it **"Miscellaneous Item"** instead of forcing a wrong
  fabric-defect label. You can tune sensitivity with the confidence slider in the sidebar
  (default 0.50 — raise it if you get false positives on non-fabric images, lower it if
  real defects are being missed).

## Customizing

- **Confidence threshold default**: change `DEFAULT_CONF` in `app.py`.
- **Colors/theme**: edit the CSS inside `inject_global_css()`, `set_welcome_background()`,
  and `set_main_background()`.
- **Class colors**: edit the `CLASS_COLORS` dictionary.
