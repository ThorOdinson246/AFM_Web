# app.py
"""
Dash pipeline:
1) Upload image (JPG/PNG/TIFF)
2) Save to local folder: uploads/
3) Run CNN classifier (1.cnn_inference.py + cnn_classifier.pth)
4) Run U-Net segmentation (2.segmentation.py + best_quality_unet.pt)
5) Branch:
   - if dots or mixed -> Voronoi analysis on the U-Net mask (voronoi.py)
   - if lines -> Color wheel analysis on the ORIGINAL image (colorwheel.py)
6) Display:
   - Original image
   - U-Net mask
   - Extra outputs (two images)
"""

import base64
import io
import os
import uuid
from datetime import datetime
from pathlib import Path
import importlib.util

from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, no_update


# ----------------------------
# 1) PROJECT PATHS (same folder as this app.py)
# ----------------------------
BASE_DIR = Path(__file__).parent

CNN_SCRIPT = BASE_DIR / "1.cnn_inference.py"
UNET_SCRIPT = BASE_DIR / "2.segmentation.py"

# NEW: your algorithmic backends
VORONOI_SCRIPT = BASE_DIR / "2.voronoi.py"
COLORWHEEL_SCRIPT = BASE_DIR / "3.colorwheel.py"

CNN_WEIGHTS = BASE_DIR / "cnn_classifier.pth"
UNET_WEIGHTS = BASE_DIR / "best_quality_unet.pt"

UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ----------------------------
# 2) DYNAMIC IMPORT (works even with weird filenames)
# ----------------------------
def import_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


cnn_mod = import_module_from_file("cnn_backend", CNN_SCRIPT)
unet_mod = import_module_from_file("unet_backend", UNET_SCRIPT)

# Voronoi + Color wheel
vor_mod = import_module_from_file("voronoi_backend", VORONOI_SCRIPT)
cw_mod = import_module_from_file("colorwheel_backend", COLORWHEEL_SCRIPT)


# ----------------------------
# 3) LOAD MODELS ONCE
# ----------------------------
CNN_MODEL = cnn_mod.load_model(str(CNN_WEIGHTS))
UNET_MODEL, UNET_IMG_SIZE, UNET_DEVICE = unet_mod.load_model(str(UNET_WEIGHTS), device="cuda")


def run_unet_cached(image_path: str, job_dir: Path) -> str:
    """Run U-Net with the model loaded once; save mask into job_dir."""
    img_tensor, original_size = unet_mod.preprocess_image(
        image_path,
        img_size=UNET_IMG_SIZE,
        denoise=0,
        sharpen=0,
        invert=False,
    )

    mask = unet_mod.predict_mask(UNET_MODEL, img_tensor, UNET_DEVICE, threshold=0.5)

    out_path = job_dir / f"{Path(image_path).stem}_mask.png"
    unet_mod.save_mask(mask, str(out_path), original_size)

    return str(out_path)


# ----------------------------
# 4) HELPERS
# ----------------------------
def save_uploaded_image(contents: str, filename: str) -> str:
    if not contents or "," not in contents:
        raise ValueError("Upload data is not in expected base64 format.")

    _, b64data = contents.split(",", 1)
    raw = base64.b64decode(b64data)

    ext = Path(filename).suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED_EXTS)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:10]
    out_path = UPLOAD_DIR / f"upload_{stamp}_{uid}{ext}"

    out_path.write_bytes(raw)
    return str(out_path)


def image_path_to_data_url(image_path: str) -> str:
    if not image_path or not os.path.exists(image_path):
        return ""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def format_probs(probabilities: dict) -> str:
    parts = []
    for k, v in probabilities.items():
        parts.append(f"{k}: {v:.2f}")
    return " | ".join(parts)


def pick_best_voronoi_images(voronoi_root: Path, image_stem: str):
    """
    Your voronoi.py saves into: output_dir / image_name / <many files>
    We'll try to grab two useful PNGs for display.
    """
    folder = voronoi_root / image_stem
    if not folder.exists():
        return "", ""

    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        return "", ""

    # Heuristics: prefer anything with "overlay" then "hist" if present
    overlay = next((p for p in pngs if "overlay" in p.name.lower()), None)
    hist = next((p for p in pngs if "hist" in p.name.lower()), None)

    # fallback to first/second png
    if overlay is None:
        overlay = pngs[0]
    if hist is None:
        hist = pngs[1] if len(pngs) > 1 else ""

    return str(overlay) if overlay else "", str(hist) if hist else ""


# ----------------------------
# 5) DASH UI
# ----------------------------
app = dash.Dash(__name__)
app.title = "AFM Pipeline"

app.layout = html.Div(
    style={"maxWidth": "1300px", "margin": "30px auto", "fontFamily": "Arial"},
    children=[
        html.H2("AFM Pipeline (Upload → CNN → U-Net → Voronoi / Color Wheel)"),

        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag & drop or ", html.B("click"), " to upload an image"]),
            style={
                "width": "100%",
                "height": "120px",
                "lineHeight": "120px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "12px",
                "textAlign": "center",
                "marginBottom": "15px",
                "cursor": "pointer",
            },
            multiple=False,
        ),

        html.Div(id="status", style={"marginBottom": "10px", "whiteSpace": "pre-wrap"}),

        dcc.Loading(
            type="default",
            children=html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "18px",
                    "alignItems": "start",
                },
                children=[
                    html.Div(
                        children=[
                            html.H4("Original"),
                            html.Img(id="img-original", style={"width": "100%", "borderRadius": "10px"}),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H4("U-Net Mask"),
                            html.Img(id="img-mask", style={"width": "100%", "borderRadius": "10px"}),
                        ]
                    ),
                ],
            ),
        ),

        html.Hr(),

        dcc.Loading(
            type="default",
            children=html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "18px",
                    "alignItems": "start",
                },
                children=[
                    html.Div(
                        children=[
                            html.H4("Extra Output 1"),
                            html.Img(id="img-extra-1", style={"width": "100%", "borderRadius": "10px"}),
                            html.Div(id="extra-1-note", style={"marginTop": "6px", "color": "#444"}),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H4("Extra Output 2"),
                            html.Img(id="img-extra-2", style={"width": "100%", "borderRadius": "10px"}),
                            html.Div(id="extra-2-note", style={"marginTop": "6px", "color": "#444"}),
                        ]
                    ),
                ],
            ),
        ),

        html.Hr(),

        html.Div(
            children=[
                html.H4("Details"),
                html.Div(id="details-text", style={"whiteSpace": "pre-wrap"}),
            ]
        ),
    ],
)


# ----------------------------
# 6) CALLBACK
# ----------------------------
@app.callback(
    Output("status", "children"),
    Output("img-original", "src"),
    Output("img-mask", "src"),
    Output("img-extra-1", "src"),
    Output("extra-1-note", "children"),
    Output("img-extra-2", "src"),
    Output("extra-2-note", "children"),
    Output("details-text", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents or not filename:
        return "No file received.", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    try:
        # 1) Save upload
        saved_path = save_uploaded_image(contents, filename)

        # Per-upload job folder
        job_id = uuid.uuid4().hex[:10]
        job_dir = RESULTS_DIR / f"job_{job_id}"
        job_dir.mkdir(exist_ok=True)

        # 2) U-Net mask FIRST
        mask_path = run_unet_cached(saved_path, job_dir=job_dir)

        # 3) CNN classify ON THE MASK (so classifier sees structure, not texture)
        cls = cnn_mod.predict_image(CNN_MODEL, mask_path)

        predicted = cls.get("predicted_class", "unknown")
        confidence = float(cls.get("confidence", 0.0))
        probabilities = cls.get("probabilities", {})

        # Map irregular -> mixed (your earlier rule)
        predicted_for_ui = "mixed" if predicted == "irregular" else predicted

        # 4) Branch
        extra1_path, extra2_path = "", ""
        extra1_note, extra2_note = "", ""
        extra_details_lines = []

        if predicted_for_ui in ("dots", "mixed"):
            # Voronoi on the U-NET MASK
            vor_dir = job_dir / "voronoi_outputs"
            results = vor_mod.run_voronoi_analysis(
                image_path=mask_path,
                image_size=1.0,  # change if you want real scaling
                output_dir=str(vor_dir),
                threshold_edge=0.025,
                max_size=1024,
            )

            # Grab two images for display (overlay + histogram if present)
            stem = Path(mask_path).stem
            extra1_path, extra2_path = pick_best_voronoi_images(vor_dir, stem)
            extra1_note = "Voronoi output (from U-Net mask)"
            extra2_note = "Voronoi output (from U-Net mask)"

            if results is None:
                extra_details_lines.append("Voronoi: no results (analysis returned None).")
            else:
                extra_details_lines.append("Voronoi results:")
                for k, v in results.items():
                    extra_details_lines.append(f"  {k}: {v}")

        elif predicted_for_ui == "lines":
            # Color wheel on the ORIGINAL IMAGE (per your colorwheel.py docstring)
            cw_dir = job_dir / "colorwheel_output"
            results = cw_mod.analyze_image(
                image_path=saved_path,
                output_dir=str(cw_dir),
                num_clusters=8,
            )

            # colorwheel.py returns paths in dict
            extra1_path = results.get("color_wheel_image", "")
            extra2_path = results.get("one_phase_image", "")
            extra1_note = "Color wheel output (from original image)"
            extra2_note = "One-phase output (from original image)"

            extra_details_lines.append("Color wheel results:")
            for k, v in results.items():
                extra_details_lines.append(f"  {k}: {v}")

        else:
            extra_details_lines.append(f"No rule for class: {predicted_for_ui}")

        # 5) Convert to display
        original_src = image_path_to_data_url(saved_path)
        mask_src = image_path_to_data_url(mask_path)
        extra1_src = image_path_to_data_url(extra1_path) if extra1_path else ""
        extra2_src = image_path_to_data_url(extra2_path) if extra2_path else ""

        status = (
            f"✅ Saved: {saved_path}\n"
            f"✅ Mask: {mask_path}\n"
            f"📁 Job folder: {job_dir}"
        )

        details = (
            f"CNN predicted class: {predicted_for_ui}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Probabilities: {format_probs(probabilities)}\n\n"
            + "\n".join(extra_details_lines)
        )

        return status, original_src, mask_src, extra1_src, extra1_note, extra2_src, extra2_note, details

    except Exception as e:
        return f"❌ Error: {e}", "", "", "", "", "", "", ""


# ----------------------------
# 7) RUN
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)



