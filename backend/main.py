"""
FastAPI Backend for AFM Pipeline
Handles image upload, CNN classification, U-Net segmentation, 
Voronoi and Color Wheel analysis.
"""

import os
import sys
import uuid
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import importlib.util

from PIL import Image
import numpy as np

# Backend directory (where this file lives)
BACKEND_DIR = Path(__file__).parent
# Project root directory (parent of backend)
PROJECT_DIR = BACKEND_DIR.parent

# Add backend directory to path for imports
sys.path.insert(0, str(BACKEND_DIR))

# Model input sizes (from the model architectures)
CNN_INPUT_SIZE = 217  # CNN classifier expects 217x217
UNET_INPUT_SIZE = 256  # U-Net expects 256x256 (loaded from checkpoint)

# ----------------------------
# 1) PROJECT PATHS (all in backend folder now)
# ----------------------------
CNN_SCRIPT = BACKEND_DIR / "1.cnn_inference.py"
UNET_SCRIPT = BACKEND_DIR / "2.segmentation.py"
VORONOI_SCRIPT = BACKEND_DIR / "2.voronoi.py"
COLORWHEEL_SCRIPT = BACKEND_DIR / "3.colorwheel.py"

CNN_WEIGHTS = BACKEND_DIR / "cnn_classifier.pth"
UNET_WEIGHTS = BACKEND_DIR / "best_quality_unet.pt"

# Data directories stay in project root
UPLOAD_DIR = PROJECT_DIR / "uploads"
RESULTS_DIR = PROJECT_DIR / "results"

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


# Import modules
cnn_mod = import_module_from_file("cnn_backend", CNN_SCRIPT)
unet_mod = import_module_from_file("unet_backend", UNET_SCRIPT)
vor_mod = import_module_from_file("voronoi_backend", VORONOI_SCRIPT)
cw_mod = import_module_from_file("colorwheel_backend", COLORWHEEL_SCRIPT)


# ----------------------------
# 3) LOAD MODELS ONCE
# ----------------------------
print("Loading models...")
CNN_MODEL = cnn_mod.load_model(str(CNN_WEIGHTS))
print("✓ CNN model loaded")

UNET_MODEL, UNET_IMG_SIZE, UNET_DEVICE = unet_mod.load_model(str(UNET_WEIGHTS), device="cuda")
print(f"✓ U-Net model loaded (device: {UNET_DEVICE})")


# ----------------------------
# 4) HELPER FUNCTIONS
# ----------------------------
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


def image_path_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    if not image_path or not os.path.exists(image_path):
        return ""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pick_best_voronoi_images(voronoi_root: Path, image_stem: str) -> tuple:
    """
    Find best Voronoi output images for display.
    voronoi_v7 saves files as: {image_name}_voronoi_overlay.png, {image_name}_snapshot.png, etc.
    """
    folder = voronoi_root / image_stem
    if not folder.exists():
        print(f"Voronoi output folder does not exist: {folder}")
        return "", "", "", ""

    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in: {folder}")
        return "", "", "", ""
    
    print(f"Found {len(pngs)} PNG files in {folder}:")
    for p in pngs:
        print(f"  - {p.name}")

    # Find specific Voronoi output files by suffix
    voronoi_overlay = next((p for p in pngs if "voronoi_overlay" in p.name.lower()), None)
    morphology_map = next((p for p in pngs if "morphology_map" in p.name.lower()), None)
    snapshot = next((p for p in pngs if "snapshot" in p.name.lower()), None)
    original = next((p for p in pngs if "original" in p.name.lower()), None)
    light_defects = next((p for p in pngs if "light_defects" in p.name.lower()), None)
    dark_defects = next((p for p in pngs if "dark_defects" in p.name.lower()), None)

    # Return the 4 most useful images
    img1 = str(voronoi_overlay) if voronoi_overlay else ""
    img2 = str(morphology_map) if morphology_map else ""
    img3 = str(snapshot) if snapshot else ""
    img4 = str(original) if original else ""
    
    print(f"Selected images: voronoi={img1}, morphology={img2}, snapshot={img3}, original={img4}")

    return img1, img2, img3, img4


def get_voronoi_stats(voronoi_root: Path, image_stem: str) -> Dict[str, Any]:
    """Parse Voronoi output files for statistics."""
    folder = voronoi_root / image_stem
    stats = {}
    
    if not folder.exists():
        return stats
    
    # Look for pixel counts file
    txt_files = list(folder.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
                stats[txt_file.stem] = content
        except:
            pass
    
    return stats


def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get image dimensions and calculate resize info."""
    with Image.open(image_path) as im:
        width, height = im.size
        return {
            "original_width": width,
            "original_height": height,
            "cnn_input_size": CNN_INPUT_SIZE,
            "unet_input_size": UNET_IMG_SIZE,
            "will_resize_for_unet": width != UNET_IMG_SIZE or height != UNET_IMG_SIZE,
            "will_resize_for_cnn": width != CNN_INPUT_SIZE or height != CNN_INPUT_SIZE,
        }


def extract_dots_from_mask(
    mask_path: str,
    output_path: str,
    min_circularity: float = 0.6,
    max_aspect_ratio: float = 1.8,
    min_area: int = 15,
    max_area: int = 400,
    invert_mask: bool = True
) -> Dict[str, Any]:
    """
    Extract dots from segmentation mask by filtering for circular features.
    Creates standardized 5x5 dots at each feature centroid.
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save output
        min_circularity: Minimum circularity threshold (0-1)
        max_aspect_ratio: Maximum aspect ratio
        min_area: Minimum feature area in pixels
        max_area: Maximum feature area in pixels
        invert_mask: If True, invert the mask before processing (for white-on-black masks)
    
    Returns:
        Dict with 'output_path', 'total_features', 'kept_features'
    """
    import cv2
    from skimage.measure import label, regionprops
    
    # Read mask
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert mask if needed (U-Net outputs white features on black background)
    # But dot extraction expects black features on white background
    if invert_mask:
        img = 255 - img
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological opening to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find connected components (looking for WHITE features now after threshold)
    # Since we inverted, features are now black (0), background is white (255)
    # So we need to invert again for labeling to find the black regions
    labels_img = label(opened == 0)  # Find black regions (the features)
    props = regionprops(labels_img)
    
    # Filter by shape and create 5x5 dots
    # Output will have WHITE dots on BLACK background (for Voronoi)
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
            # Create consistent 5x5 dots (WHITE dots on BLACK background)
            y1, y2 = max(cy-2, 0), min(cy+3, output.shape[0])
            x1, x2 = max(cx-2, 0), min(cx+3, output.shape[1])
            output[y1:y2, x1:x2] = 255
            stats['kept'] += 1
    
    # Save output (WHITE dots on BLACK background)
    cv2.imwrite(output_path, output)
    
    return {
        'output_path': output_path,
        'total_features': stats['total'],
        'kept_features': stats['kept'],
        'rejected_features': stats['total'] - stats['kept']
    }


# ----------------------------
# 5) FASTAPI APP
# ----------------------------
app = FastAPI(
    title="AFM Analysis Pipeline",
    description="CNN Classification, U-Net Segmentation, Voronoi and Color Wheel Analysis",
    version="2.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from results directory
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ----------------------------
# 6) PYDANTIC MODELS
# ----------------------------
class ImageInfo(BaseModel):
    original_width: int
    original_height: int
    cnn_input_size: int
    unet_input_size: int
    will_resize_for_unet: bool
    will_resize_for_cnn: bool


class AnalysisResponse(BaseModel):
    success: bool
    job_id: str
    original_image: str
    mask_image: str
    dots_mask_image: Optional[str] = None
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    extra_outputs: List[Dict[str, Any]]
    analysis_details: Dict[str, Any]
    voronoi_stats: Optional[Dict[str, Any]] = None
    colorwheel_stats: Optional[Dict[str, Any]] = None
    dot_extraction_stats: Optional[Dict[str, Any]] = None
    image_info: Optional[ImageInfo] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cnn_device: str
    unet_device: str


# ----------------------------
# 7) API ENDPOINTS
# ----------------------------
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        cnn_device="cuda" if str(next(CNN_MODEL.parameters()).device) == "cuda" else "cpu",
        unet_device=str(UNET_DEVICE)
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Main analysis endpoint.
    1) Save uploaded image
    2) Run U-Net segmentation
    3) Run CNN classification on mask
    4) Branch to Voronoi or Color Wheel based on classification
    5) Return all results
    """
    try:
        # Validate file extension
        filename = file.filename or "unknown.png"
        ext = Path(filename).suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED_EXTS)}"
            )

        # Save uploaded file
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:10]
        saved_path = UPLOAD_DIR / f"upload_{stamp}_{uid}{ext}"
        
        content = await file.read()
        saved_path.write_bytes(content)

        # Create job directory
        job_id = uuid.uuid4().hex[:10]
        job_dir = RESULTS_DIR / f"job_{job_id}"
        job_dir.mkdir(exist_ok=True)

        # 1) U-Net segmentation
        mask_path = run_unet_cached(str(saved_path), job_dir=job_dir)

        # 2) CNN classification on mask
        cls = cnn_mod.predict_image(CNN_MODEL, mask_path)
        predicted = cls.get("predicted_class", "unknown")
        confidence = float(cls.get("confidence", 0.0))
        probabilities = cls.get("probabilities", {})

        # Map irregular -> mixed
        predicted_for_ui = "mixed" if predicted == "irregular" else predicted

        # 3) Branch analysis
        extra_outputs = []
        analysis_details = {}
        voronoi_stats = None
        colorwheel_stats = None
        dot_extraction_stats = None
        dots_mask_path = None

        if predicted_for_ui in ("dots", "mixed"):
            # Step 3a: Extract dots from U-Net mask (invert first since mask has white features on black)
            dots_output_path = str(job_dir / f"{Path(mask_path).stem}_DOTS_ONLY.png")
            dot_extraction_stats = extract_dots_from_mask(
                mask_path=mask_path,
                output_path=dots_output_path,
                min_circularity=0.6,
                max_aspect_ratio=1.8,
                min_area=15,
                max_area=400,
                invert_mask=True  # Invert because U-Net mask has white features on black background
            )
            dots_mask_path = dots_output_path
            
            # Only run Voronoi if we have enough dots (4+)
            if dot_extraction_stats['kept_features'] >= 4:
                # Step 3b: Voronoi on extracted dots mask
                vor_dir = job_dir / "voronoi_outputs"
                results = vor_mod.run_voronoi_analysis(
                    image_path=dots_mask_path,
                    image_size=1.0,
                    output_dir=str(vor_dir),
                    threshold_edge=0.025,
                    max_size=1024,
                )

                stem = Path(dots_mask_path).stem
                voronoi_overlay, morphology_map, snapshot, original = pick_best_voronoi_images(vor_dir, stem)
                
                extra_outputs = [
                    {
                        "image": image_path_to_base64(dots_mask_path),
                        "title": f"Extracted Dots ({dot_extraction_stats['kept_features']} features)",
                        "description": f"Filtered from {dot_extraction_stats['total_features']} total features"
                    }
                ]
                
                # Add all available voronoi output images
                if voronoi_overlay:
                    extra_outputs.append({
                        "image": image_path_to_base64(voronoi_overlay),
                        "title": "Voronoi Tessellation",
                        "description": "Voronoi cells overlaid on detected features"
                    })
                if morphology_map:
                    extra_outputs.append({
                        "image": image_path_to_base64(morphology_map),
                        "title": "Morphology Map",
                        "description": "Color-coded morphology analysis"
                    })
                if snapshot:
                    extra_outputs.append({
                        "image": image_path_to_base64(snapshot),
                        "title": "Analysis Snapshot",
                        "description": "Intermediate analysis visualization"
                    })

                if results:
                    analysis_details = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) 
                                       for k, v in results.items()}
                    voronoi_stats = get_voronoi_stats(vor_dir, stem)
            else:
                # Not enough dots for Voronoi
                extra_outputs = [
                    {
                        "image": image_path_to_base64(dots_mask_path),
                        "title": f"Extracted Dots ({dot_extraction_stats['kept_features']} features)",
                        "description": f"Insufficient dots for Voronoi analysis (need 4+)"
                    }
                ]
                analysis_details = {
                    "note": "Insufficient dots for Voronoi analysis",
                    "dots_found": dot_extraction_stats['kept_features'],
                    "minimum_required": 4
                }

        elif predicted_for_ui == "lines":
            # Color wheel on original image
            cw_dir = job_dir / "colorwheel_output"
            results = cw_mod.analyze_image(
                image_path=str(saved_path),
                output_dir=str(cw_dir),
                num_clusters=8,
            )

            extra_outputs = [
                {
                    "image": image_path_to_base64(results.get("color_wheel_image", "")),
                    "title": "Color Wheel",
                    "description": "Orientation-based color mapping"
                },
                {
                    "image": image_path_to_base64(results.get("one_phase_image", "")),
                    "title": "Phase Separation",
                    "description": "Isolated orientation phase"
                }
            ]

            colorwheel_stats = {
                "orientation_angle": results.get("orientation_angle", 0),
                "gpu_accelerated": results.get("gpu_accelerated", False),
                "grain_masks_count": len(results.get("grain_masks", []))
            }
            analysis_details = {
                k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                for k, v in results.items()
                if k not in ["grain_masks"]
            }

        # Get image info for frontend
        img_info = get_image_info(str(saved_path))

        # Build response
        return AnalysisResponse(
            success=True,
            job_id=job_id,
            original_image=image_path_to_base64(str(saved_path)),
            mask_image=image_path_to_base64(mask_path),
            dots_mask_image=image_path_to_base64(dots_mask_path) if dots_mask_path else None,
            predicted_class=predicted_for_ui,
            confidence=confidence,
            probabilities={k: float(v) for k, v in probabilities.items()},
            extra_outputs=extra_outputs,
            analysis_details=analysis_details,
            voronoi_stats=voronoi_stats,
            colorwheel_stats=colorwheel_stats,
            dot_extraction_stats=dot_extraction_stats,
            image_info=ImageInfo(**img_info)
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_jobs():
    """List all analysis jobs."""
    jobs = []
    for job_dir in RESULTS_DIR.glob("job_*"):
        if job_dir.is_dir():
            jobs.append({
                "job_id": job_dir.name.replace("job_", ""),
                "created": datetime.fromtimestamp(job_dir.stat().st_ctime).isoformat()
            })
    return {"jobs": sorted(jobs, key=lambda x: x["created"], reverse=True)}


@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get results for a specific job."""
    job_dir = RESULTS_DIR / f"job_{job_id}"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    files = list(job_dir.rglob("*"))
    return {
        "job_id": job_id,
        "files": [str(f.relative_to(job_dir)) for f in files if f.is_file()]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
