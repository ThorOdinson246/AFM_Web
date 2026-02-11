"""
AFM Analysis Pipeline - Dash Plotly Version
A modern web application using Dash Plotly components for AFM image analysis.

Features:
- Upload zone with drag & drop
- CNN Classification with probability charts
- U-Net Segmentation
- Voronoi Tessellation (for dots/mixed)
- Color Wheel Analysis (for lines)
- Interactive Plotly visualizations
"""

import base64
import io
import os
import uuid
from datetime import datetime
from pathlib import Path
import importlib.util
import json

import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

# ----------------------------
# 1) PROJECT PATHS
# ----------------------------
BASE_DIR = Path(__file__).parent

CNN_SCRIPT = BASE_DIR / "1.cnn_inference.py"
UNET_SCRIPT = BASE_DIR / "2.segmentation.py"
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
# 2) DYNAMIC IMPORT
# ----------------------------
def import_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

print("Loading modules...")
cnn_mod = import_module_from_file("cnn_backend", CNN_SCRIPT)
unet_mod = import_module_from_file("unet_backend", UNET_SCRIPT)
vor_mod = import_module_from_file("voronoi_backend", VORONOI_SCRIPT)
cw_mod = import_module_from_file("colorwheel_backend", COLORWHEEL_SCRIPT)

# ----------------------------
# 3) LOAD MODELS
# ----------------------------
print("Loading models...")
CNN_MODEL = cnn_mod.load_model(str(CNN_WEIGHTS))
print("✓ CNN model loaded")
UNET_MODEL, UNET_IMG_SIZE, UNET_DEVICE = unet_mod.load_model(str(UNET_WEIGHTS), device="cuda")
print(f"✓ U-Net model loaded (device: {UNET_DEVICE})")

# CNN input size
CNN_INPUT_SIZE = 217

# ----------------------------
# 4) HELPER FUNCTIONS
# ----------------------------
def run_unet(image_path: str, job_dir: Path) -> str:
    """Run U-Net segmentation."""
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


def save_uploaded_image(contents: str, filename: str) -> str:
    """Save uploaded base64 image to disk."""
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


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    if not image_path or not os.path.exists(image_path):
        return ""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_dots_from_mask(mask_path: str, output_path: str, 
                           min_circularity=0.6, max_aspect_ratio=1.8,
                           min_area=15, max_area=400, invert_mask=True):
    """Extract dot features from segmentation mask."""
    import cv2
    from skimage.measure import label, regionprops
    
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if invert_mask:
        img = 255 - img
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    labels_img = label(opened == 0)
    props = regionprops(labels_img)
    
    output = np.zeros_like(binary)
    stats = {'total': len(props), 'kept': 0}
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else float('inf')
        
        if (min_area <= area <= max_area and 
            circularity >= min_circularity and 
            aspect <= max_aspect_ratio):
            cy, cx = map(int, prop.centroid)
            y1, y2 = max(cy-2, 0), min(cy+3, output.shape[0])
            x1, x2 = max(cx-2, 0), min(cx+3, output.shape[1])
            output[y1:y2, x1:x2] = 255
            stats['kept'] += 1
    
    cv2.imwrite(output_path, output)
    return {
        'output_path': output_path,
        'total_features': stats['total'],
        'kept_features': stats['kept'],
        'rejected_features': stats['total'] - stats['kept']
    }


def pick_voronoi_images(voronoi_root: Path, image_stem: str):
    """Find Voronoi output images."""
    folder = voronoi_root / image_stem
    if not folder.exists():
        return "", "", "", ""
    
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        return "", "", "", ""
    
    voronoi_overlay = next((p for p in pngs if "voronoi_overlay" in p.name.lower()), None)
    morphology_map = next((p for p in pngs if "morphology_map" in p.name.lower()), None)
    snapshot = next((p for p in pngs if "snapshot" in p.name.lower()), None)
    original = next((p for p in pngs if "original" in p.name.lower()), None)
    
    return (
        str(voronoi_overlay) if voronoi_overlay else "",
        str(morphology_map) if morphology_map else "",
        str(snapshot) if snapshot else "",
        str(original) if original else ""
    )


def get_image_info(image_path: str):
    """Get image dimensions."""
    with Image.open(image_path) as im:
        width, height = im.size
        return {
            "original_width": width,
            "original_height": height,
            "cnn_input_size": CNN_INPUT_SIZE,
            "unet_input_size": UNET_IMG_SIZE,
            "will_resize": width != UNET_IMG_SIZE or height != UNET_IMG_SIZE,
        }


# ----------------------------
# 5) PLOTLY CHART FUNCTIONS
# ----------------------------
def create_probability_chart(probabilities: dict, predicted_class: str):
    """Create probability bar chart with Plotly."""
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    
    color_map = {
        'dots': '#22c55e',
        'lines': '#a855f7', 
        'mixed': '#fb923c',
        'irregular': '#fb923c',
    }
    
    colors = [color_map.get(c, '#3b82f6') if c == predicted_class else 'rgba(156, 163, 175, 0.5)' 
              for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[c.capitalize() for c in classes],
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis=dict(
            title='Probability (%)',
            range=[0, 105],
            gridcolor='rgba(229, 231, 235, 1)',
            linecolor='#d1d5db',
        ),
        xaxis=dict(
            gridcolor='rgba(229, 231, 235, 1)',
            linecolor='#d1d5db',
        ),
        font=dict(family='Inter, system-ui, sans-serif', color='#374151'),
        showlegend=False,
        bargap=0.4,
        height=250,
    )
    
    return fig


def create_confidence_gauge(confidence: float, predicted_class: str):
    """Create confidence gauge with Plotly."""
    color_map = {
        'dots': '#22c55e',
        'lines': '#a855f7',
        'mixed': '#fb923c',
        'irregular': '#fb923c',
    }
    color = color_map.get(predicted_class, '#3b82f6')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#d1d5db'},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 2,
            'bordercolor': '#e5e7eb',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [50, 75], 'color': 'rgba(251, 191, 36, 0.1)'},
                {'range': [75, 100], 'color': 'rgba(34, 197, 94, 0.1)'},
            ],
        },
        title={'text': 'Model Confidence', 'font': {'size': 14, 'color': '#6b7280'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, sans-serif'),
        margin=dict(l=30, r=30, t=50, b=30),
        height=200,
    )
    
    return fig


def create_metrics_chart(metrics: dict, title: str = "Analysis Metrics"):
    """Create metrics visualization chart."""
    # Filter numeric values only
    numeric_items = {k: v for k, v in metrics.items() 
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
    
    if not numeric_items:
        return go.Figure()
    
    labels = list(numeric_items.keys())
    values = list(numeric_items.values())
    
    # Normalize for radar chart
    max_val = max(abs(v) for v in values) if values else 1
    normalized = [v / max_val * 100 if max_val > 0 else 0 for v in values]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=normalized + [normalized[0]],  # Close the polygon
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='%{theta}: %{text}<extra></extra>',
        text=[f'{v:.4f}' for v in values] + [f'{values[0]:.4f}'],
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(229, 231, 235, 1)',
            ),
            angularaxis=dict(
                gridcolor='rgba(229, 231, 235, 1)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, sans-serif', color='#374151', size=10),
        margin=dict(l=60, r=60, t=40, b=40),
        height=300,
        showlegend=False,
    )
    
    return fig


# ----------------------------
# 6) DASH APP
# ----------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
)
app.title = "AFM Analysis Pipeline"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background: #ffffff;
                color: #1f2937;
                margin: 0;
                padding: 0;
            }
            .upload-zone {
                border: 2px dashed #d1d5db;
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #fafafa;
            }
            .upload-zone:hover {
                border-color: #2563eb;
                background: #eff6ff;
            }
            .card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .card-title {
                font-size: 14px;
                font-weight: 600;
                color: #374151;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .badge {
                display: inline-flex;
                align-items: center;
                padding: 8px 16px;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 14px;
            }
            .badge-dots { background: #dcfce7; color: #166534; }
            .badge-lines { background: #f3e8ff; color: #7c3aed; }
            .badge-mixed { background: #ffedd5; color: #c2410c; }
            .stat-box {
                background: #f9fafb;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            }
            .stat-label { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
            .stat-value { font-size: 18px; font-weight: 600; color: #111827; }
            .image-container {
                background: #f9fafb;
                border-radius: 8px;
                overflow: hidden;
                position: relative;
            }
            .image-container img {
                width: 100%;
                height: auto;
                display: block;
            }
            .image-title {
                position: absolute;
                top: 8px;
                left: 8px;
                background: rgba(0,0,0,0.6);
                color: white;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            .details-table {
                width: 100%;
                font-size: 13px;
            }
            .details-table td {
                padding: 6px 8px;
                border-bottom: 1px solid #f3f4f6;
            }
            .details-table td:first-child {
                color: #6b7280;
                font-weight: 500;
            }
            .details-table td:last-child {
                color: #111827;
                text-align: right;
                font-family: 'JetBrains Mono', monospace;
            }
            .pipeline-step {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 8px 12px;
                border-radius: 8px;
                margin-bottom: 8px;
                background: #f9fafb;
            }
            .pipeline-step.active { background: #eff6ff; }
            .pipeline-step.complete { background: #f0fdf4; }
            .pipeline-step.error { background: #fef2f2; }
            .step-icon {
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
            }
            .step-pending { background: #e5e7eb; color: #6b7280; }
            .step-active { background: #3b82f6; color: white; }
            .step-complete { background: #22c55e; color: white; }
            .step-error { background: #ef4444; color: white; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #f3f4f6; }
            ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    # Header Status Bar
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("●", style={'color': '#22c55e', 'marginRight': '8px'}),
                html.Span("Backend Connected", style={'fontSize': '14px', 'color': '#6b7280'}),
                html.Span(f" | CNN: cpu | U-Net: {UNET_DEVICE}", 
                         style={'fontSize': '12px', 'color': '#9ca3af', 'marginLeft': '12px'}),
            ], style={'textAlign': 'right', 'padding': '12px 0'})
        ])
    ], className="mb-3"),
    
    dbc.Row([
        # Left Column - Upload & Pipeline
        dbc.Col([
            # Upload Zone
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        html.Div("📁", style={'fontSize': '48px', 'marginBottom': '12px'}),
                        html.Div("Drop an AFM image here", style={'fontSize': '16px', 'fontWeight': '500', 'color': '#374151'}),
                        html.Div("or click to browse", style={'fontSize': '14px', 'color': '#6b7280', 'marginTop': '4px'}),
                        html.Div("Supports: JPG, PNG, TIFF", style={'fontSize': '12px', 'color': '#9ca3af', 'marginTop': '8px'}),
                    ]),
                    className='upload-zone',
                    multiple=False,
                )
            ], className='card'),
            
            # Pipeline Status
            html.Div(id='pipeline-status', className='card', style={'display': 'none'}),
            
            # Error Display
            html.Div(id='error-display'),
            
        ], md=4),
        
        # Right Column - Results
        dbc.Col([
            # Results Container
            html.Div(id='results-container', children=[
                # Placeholder when no results
                html.Div([
                    html.Div("📊", style={'fontSize': '64px', 'marginBottom': '16px', 'opacity': '0.5'}),
                    html.H4("Upload an Image", style={'color': '#374151', 'marginBottom': '8px'}),
                    html.P("Drop an AFM image to run the analysis pipeline.", 
                          style={'color': '#6b7280', 'fontSize': '14px'}),
                ], style={'textAlign': 'center', 'padding': '80px 20px'}, className='card')
            ]),
        ], md=8),
    ]),
    
    # Loading Spinner
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(id="loading-output"),
        color="#3b82f6",
    ),
    
    # Store for results data
    dcc.Store(id='analysis-results'),
    
], fluid=True, style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})


# ----------------------------
# 7) CALLBACKS
# ----------------------------
@callback(
    Output('analysis-results', 'data'),
    Output('loading-output', 'children'),
    Output('pipeline-status', 'children'),
    Output('pipeline-status', 'style'),
    Output('error-display', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call=True,
)
def process_upload(contents, filename):
    if not contents or not filename:
        return no_update, no_update, no_update, no_update, no_update
    
    # Pipeline steps UI
    def make_pipeline_ui(steps):
        icons = {'pending': '○', 'active': '◐', 'complete': '✓', 'error': '✗'}
        return html.Div([
            html.Div([
                html.Span(icons[s['status']], className=f"step-icon step-{s['status']}"),
                html.Span(s['label'], style={'fontSize': '13px', 'color': '#374151'}),
            ], className=f"pipeline-step {s['status']}")
            for s in steps
        ])
    
    steps = [
        {'id': 'upload', 'label': 'Image Upload', 'status': 'active'},
        {'id': 'segment', 'label': 'U-Net Segmentation', 'status': 'pending'},
        {'id': 'classify', 'label': 'CNN Classification', 'status': 'pending'},
        {'id': 'extract', 'label': 'Feature Extraction', 'status': 'pending'},
        {'id': 'analyze', 'label': 'Analysis', 'status': 'pending'},
    ]
    
    try:
        # 1) Save uploaded image
        saved_path = save_uploaded_image(contents, filename)
        steps[0]['status'] = 'complete'
        steps[1]['status'] = 'active'
        
        # Create job folder
        job_id = uuid.uuid4().hex[:10]
        job_dir = RESULTS_DIR / f"job_{job_id}"
        job_dir.mkdir(exist_ok=True)
        
        # 2) U-Net segmentation
        mask_path = run_unet(saved_path, job_dir)
        steps[1]['status'] = 'complete'
        steps[2]['status'] = 'active'
        
        # 3) CNN classification on mask
        cls = cnn_mod.predict_image(CNN_MODEL, mask_path)
        predicted = cls.get("predicted_class", "unknown")
        confidence = float(cls.get("confidence", 0.0))
        probabilities = cls.get("probabilities", {})
        
        predicted_for_ui = "mixed" if predicted == "irregular" else predicted
        steps[2]['status'] = 'complete'
        steps[3]['status'] = 'active'
        
        # 4) Get image info
        img_info = get_image_info(saved_path)
        
        # 5) Branch analysis
        extra_outputs = []
        analysis_details = {}
        voronoi_stats = None
        colorwheel_stats = None
        dot_extraction_stats = None
        
        if predicted_for_ui in ("dots", "mixed"):
            # Extract dots
            dots_output_path = str(job_dir / f"{Path(mask_path).stem}_DOTS_ONLY.png")
            dot_extraction_stats = extract_dots_from_mask(
                mask_path=mask_path,
                output_path=dots_output_path,
                invert_mask=True
            )
            steps[3]['status'] = 'complete'
            steps[4]['status'] = 'active'
            
            if dot_extraction_stats['kept_features'] >= 4:
                # Voronoi analysis
                vor_dir = job_dir / "voronoi_outputs"
                results = vor_mod.run_voronoi_analysis(
                    image_path=dots_output_path,
                    image_size=1.0,
                    output_dir=str(vor_dir),
                    threshold_edge=0.025,
                    max_size=1024,
                )
                
                stem = Path(dots_output_path).stem
                vor_overlay, vor_morph, vor_snap, vor_orig = pick_voronoi_images(vor_dir, stem)
                
                extra_outputs = [
                    {'path': dots_output_path, 'title': f'Extracted Dots ({dot_extraction_stats["kept_features"]})'},
                ]
                if vor_overlay:
                    extra_outputs.append({'path': vor_overlay, 'title': 'Voronoi Tessellation'})
                if vor_morph:
                    extra_outputs.append({'path': vor_morph, 'title': 'Morphology Map'})
                if vor_snap:
                    extra_outputs.append({'path': vor_snap, 'title': 'Analysis Snapshot'})
                
                if results:
                    analysis_details = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                                       for k, v in results.items()}
            else:
                extra_outputs = [
                    {'path': dots_output_path, 'title': f'Extracted Dots ({dot_extraction_stats["kept_features"]})'},
                ]
                analysis_details = {'note': 'Insufficient dots for Voronoi (need 4+)'}
        
        elif predicted_for_ui == "lines":
            steps[3]['status'] = 'complete'
            steps[4]['status'] = 'active'
            
            # Color wheel analysis
            cw_dir = job_dir / "colorwheel_output"
            results = cw_mod.analyze_image(
                image_path=saved_path,
                output_dir=str(cw_dir),
                num_clusters=8,
            )
            
            extra_outputs = [
                {'path': results.get("color_wheel_image", ""), 'title': 'Color Wheel'},
                {'path': results.get("one_phase_image", ""), 'title': 'Phase Separation'},
            ]
            
            colorwheel_stats = {
                "orientation_angle": results.get("orientation_angle", 0),
                "gpu_accelerated": results.get("gpu_accelerated", False),
                "grain_masks_count": len(results.get("grain_masks", [])),
            }
            analysis_details = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                               for k, v in results.items() if k not in ["grain_masks"]}
        
        steps[4]['status'] = 'complete'
        
        # Build result data
        result_data = {
            'job_id': job_id,
            'original_image': image_to_base64(saved_path),
            'mask_image': image_to_base64(mask_path),
            'predicted_class': predicted_for_ui,
            'confidence': confidence,
            'probabilities': probabilities,
            'image_info': img_info,
            'dot_extraction_stats': dot_extraction_stats,
            'extra_outputs': [{'image': image_to_base64(o['path']), 'title': o['title']} 
                             for o in extra_outputs if o.get('path')],
            'analysis_details': analysis_details,
            'colorwheel_stats': colorwheel_stats,
        }
        
        return result_data, "", make_pipeline_ui(steps), {'display': 'block'}, ""
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_ui = dbc.Alert(f"Error: {str(e)}", color="danger", className="mt-3")
        return no_update, no_update, no_update, no_update, error_ui


@callback(
    Output('results-container', 'children'),
    Input('analysis-results', 'data'),
    prevent_initial_call=True,
)
def update_results(data):
    if not data:
        return no_update
    
    # Classification badge
    badge_class = f"badge badge-{data['predicted_class']}"
    
    # Build results UI
    results = []
    
    # 1) Classification Card
    classification_card = html.Div([
        html.Div([
            html.Span("🧠", style={'fontSize': '18px'}),
            html.Span("Classification", style={'marginLeft': '8px'}),
            html.Span(f"Job: {data['job_id']}", 
                     style={'marginLeft': 'auto', 'fontSize': '12px', 'color': '#9ca3af', 'fontFamily': 'monospace'}),
        ], className='card-title', style={'display': 'flex', 'justifyContent': 'space-between'}),
        
        dbc.Row([
            dbc.Col([
                html.Span(data['predicted_class'].upper(), className=badge_class),
                html.Div(f"Confidence: {data['confidence']*100:.1f}%", 
                        style={'fontSize': '13px', 'color': '#6b7280', 'marginTop': '8px'}),
                
                # Probabilities text
                html.Div([
                    html.Div("Class Probabilities:", style={'fontSize': '12px', 'color': '#6b7280', 'marginTop': '12px', 'marginBottom': '4px'}),
                    html.Div([
                        html.Span(f"{k}: {v*100:.1f}%  ", style={'fontSize': '12px', 'marginRight': '8px'})
                        for k, v in data['probabilities'].items()
                    ])
                ])
            ], md=5),
            dbc.Col([
                dcc.Graph(
                    figure=create_probability_chart(data['probabilities'], data['predicted_class']),
                    config={'displayModeBar': False},
                    style={'height': '200px'}
                )
            ], md=7),
        ]),
    ], className='card')
    results.append(classification_card)
    
    # 2) Images Grid (Original + Mask)
    images_row = dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src=data['original_image'], style={'width': '100%'}),
                html.Div("Original Image", className='image-title'),
            ], className='image-container')
        ], md=6),
        dbc.Col([
            html.Div([
                html.Img(src=data['mask_image'], style={'width': '100%'}),
                html.Div("U-Net Mask", className='image-title'),
            ], className='image-container')
        ], md=6),
    ], className='mb-3')
    results.append(images_row)
    
    # 3) Image Info Card
    if data.get('image_info'):
        info = data['image_info']
        info_card = html.Div([
            html.Div([html.Span("📐"), html.Span("Image Information", style={'marginLeft': '8px'})], className='card-title'),
            dbc.Row([
                dbc.Col([html.Div([
                    html.Div("Original Size", className='stat-label'),
                    html.Div(f"{info['original_width']} × {info['original_height']}", className='stat-value'),
                ], className='stat-box')], md=3),
                dbc.Col([html.Div([
                    html.Div("CNN Input", className='stat-label'),
                    html.Div(f"{info['cnn_input_size']} × {info['cnn_input_size']}", className='stat-value'),
                ], className='stat-box')], md=3),
                dbc.Col([html.Div([
                    html.Div("U-Net Input", className='stat-label'),
                    html.Div(f"{info['unet_input_size']} × {info['unet_input_size']}", className='stat-value'),
                ], className='stat-box')], md=3),
                dbc.Col([html.Div([
                    html.Div("Resized", className='stat-label'),
                    html.Div("Yes" if info['will_resize'] else "No", className='stat-value'),
                ], className='stat-box')], md=3),
            ])
        ], className='card')
        results.append(info_card)
    
    # 4) Dot Extraction Stats
    if data.get('dot_extraction_stats'):
        stats = data['dot_extraction_stats']
        dot_card = html.Div([
            html.Div([html.Span("🔵"), html.Span("Dot Extraction", style={'marginLeft': '8px'})], className='card-title'),
            dbc.Row([
                dbc.Col([html.Div([
                    html.Div("Total Features", className='stat-label'),
                    html.Div(str(stats['total_features']), className='stat-value'),
                ], className='stat-box')], md=4),
                dbc.Col([html.Div([
                    html.Div("Dots Kept", className='stat-label'),
                    html.Div(str(stats['kept_features']), className='stat-value', style={'color': '#22c55e'}),
                ], className='stat-box', style={'background': '#f0fdf4'})], md=4),
                dbc.Col([html.Div([
                    html.Div("Rejected", className='stat-label'),
                    html.Div(str(stats['rejected_features']), className='stat-value', style={'color': '#f97316'}),
                ], className='stat-box', style={'background': '#fff7ed'})], md=4),
            ])
        ], className='card')
        results.append(dot_card)
    
    # 5) Extra Outputs (Voronoi/ColorWheel images)
    if data.get('extra_outputs'):
        extra_cols = []
        for output in data['extra_outputs']:
            if output.get('image'):
                extra_cols.append(
                    dbc.Col([
                        html.Div([
                            html.Img(src=output['image'], style={'width': '100%'}),
                            html.Div(output['title'], className='image-title'),
                        ], className='image-container')
                    ], md=6 if len(data['extra_outputs']) <= 2 else 4, className='mb-3')
                )
        if extra_cols:
            results.append(dbc.Row(extra_cols))
    
    # 6) Analysis Details with Chart
    if data.get('analysis_details'):
        details = data['analysis_details']
        
        # Table of details
        table_rows = []
        for k, v in details.items():
            if isinstance(v, float):
                v_str = f"{v:.6f}"
            else:
                v_str = str(v)
            table_rows.append(html.Tr([html.Td(k), html.Td(v_str)]))
        
        analysis_card = html.Div([
            html.Div([
                html.Span("📊"), 
                html.Span(f"{'Color Wheel' if data['predicted_class'] == 'lines' else 'Voronoi'} Analysis Details", 
                         style={'marginLeft': '8px'})
            ], className='card-title'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=create_metrics_chart(details),
                        config={'displayModeBar': False},
                    )
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.Table(table_rows, className='details-table')
                    ], style={'maxHeight': '300px', 'overflowY': 'auto'})
                ], md=6),
            ])
        ], className='card')
        results.append(analysis_card)
    
    # 7) ColorWheel Stats
    if data.get('colorwheel_stats'):
        cw = data['colorwheel_stats']
        cw_card = html.Div([
            html.Div([html.Span("🎨"), html.Span("Color Wheel Statistics", style={'marginLeft': '8px'})], className='card-title'),
            dbc.Row([
                dbc.Col([html.Div([
                    html.Div("Orientation Angle", className='stat-label'),
                    html.Div(f"{cw['orientation_angle']:.2f}°", className='stat-value'),
                ], className='stat-box')], md=4),
                dbc.Col([html.Div([
                    html.Div("GPU Accelerated", className='stat-label'),
                    html.Div("Yes" if cw['gpu_accelerated'] else "No", className='stat-value'),
                ], className='stat-box')], md=4),
                dbc.Col([html.Div([
                    html.Div("Grain Masks", className='stat-label'),
                    html.Div(str(cw['grain_masks_count']), className='stat-value'),
                ], className='stat-box')], md=4),
            ])
        ], className='card')
        results.append(cw_card)
    
    # 8) Confidence Gauge
    gauge_card = html.Div([
        html.Div([html.Span("🎯"), html.Span("Model Confidence", style={'marginLeft': '8px'})], className='card-title'),
        dcc.Graph(
            figure=create_confidence_gauge(data['confidence'], data['predicted_class']),
            config={'displayModeBar': False},
        )
    ], className='card')
    results.append(gauge_card)
    
    # 9) Raw Data (Collapsible)
    raw_data_card = html.Details([
        html.Summary("Raw API Response (Debug)", style={'cursor': 'pointer', 'padding': '12px', 'fontWeight': '500'}),
        html.Pre(
            json.dumps({k: v for k, v in data.items() if k not in ['original_image', 'mask_image', 'extra_outputs']}, indent=2),
            style={'background': '#f9fafb', 'padding': '12px', 'borderRadius': '8px', 'fontSize': '12px', 'overflowX': 'auto'}
        )
    ], className='card')
    results.append(raw_data_card)
    
    return results


# ----------------------------
# 8) RUN
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("AFM Analysis Pipeline - Dash Version")
    print("="*50)
    print(f"Open http://127.0.0.1:8050 in your browser")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=8050, debug=True)
