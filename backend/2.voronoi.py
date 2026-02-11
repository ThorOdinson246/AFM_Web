
import os
from pathlib import Path
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Configure matplotlib to use non-interactive backend (prevents plt.show() from blocking)
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot or voronoi_v7
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Matplotlib params to match run_image_analysis_knn
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import skimage
from skimage import color
import skimage.util

import voronoi_v7


def load_image(image_path, max_size=1024):
    """
    Load and process image using the SAME method as run_image_analysis_knn.ipynb
    
    Args:
        image_path: Path to image
        max_size: Maximum dimension (pixels). Images larger than this will be downsampled.
    
    Returns:
        Processed numpy array ready for voronoi_v7.analyze_image()
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Load image exactly like run_image_analysis_knn.ipynb
        im = Image.open(image_path)
        im = im.convert("RGB")
        
        # Key step: 1 - color.rgb2gray to invert (white dots become dark)
        data = 1 - color.rgb2gray(skimage.img_as_float(im))
        im.close()
        
        print(f"Loaded image: {image_path.name}")
        
        # Downsample if too large (speeds up processing dramatically)
        original_shape = data.shape
        if max_size is not None and (data.shape[0] > max_size or data.shape[1] > max_size):
            h, w = data.shape
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            from skimage.transform import resize
            data = resize(data, (new_h, new_w), anti_aliasing=True, preserve_range=True)
            print(f"  ⚠ Downsampled from {original_shape} to {data.shape} for faster processing")
        
        # Validate
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
        
        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {data.shape}")
        
        return data
        
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def run_voronoi_analysis(image_path, image_size=1.0, output_dir='voronoi_outputs', threshold_edge=0.025, max_size=1024):
    """
    Run Voronoi analysis matching run_image_analysis_knn.ipynb exactly.
    
    Args:
        image_path: Path to image file
        image_size: Real-world image size in micrometers
        output_dir: Directory to save results
        threshold_edge: Threshold for edge detection (0.025 = 2.5% tolerance, 0.65 = 65%)
        max_size: Max dimension for downsampling (None = no downsampling)
    
    Returns:
        Dictionary with analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"\n{'='*60}")
    print("Loading Image")
    print(f"{'='*60}")
    
    try:
        image_data = load_image(image_path, max_size=max_size)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Error: {e}")
        return None
    
    # Extract image name (without extension)
    image_name = Path(image_path).stem
    
    # Run analysis
    print(f"\n{'='*60}")
    print(f"Running Voronoi Analysis")
    print(f"{'='*60}")
    print(f"Image: {image_name}")
    print(f"Image size (real-world): {image_size} μm")
    print(f"Threshold edge: {threshold_edge}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Call analyze_image exactly like run_image_analysis_knn.ipynb
        results_dict = voronoi_v7.analyze_image(
            image_data=image_data,
            image_name=image_name,
            image_size=image_size,
            save_image=True,
            show_image=False,
            save_location=output_dir,
            threshold_edge=threshold_edge
        )
        
        # Display results
        print(f"\n{'='*60}")
        print("Analysis Results")
        print(f"{'='*60}")
        for key, value in results_dict.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")
        print(f"{'='*60}\n")
        
        # Success message
        output_path = Path(output_dir) / image_name
        print(f"✓ Analysis complete!")
        print(f"Results saved to: {output_path}/")
        print(f"\nGenerated files:")
        if output_path.exists():
            for file in sorted(output_path.glob('*')):
                print(f"  - {file.name}")
        
        return results_dict
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run analysis matching run_image_analysis_knn.ipynb
    results = run_voronoi_analysis(
        image_path='/home/newuser/Desktop/Mukesh_AFM/data/for_unet/Images/A_AC.png',
        image_size=1.0,  # Size in micrometers (1 = 1μm)
        output_dir='voronoi_outputs',
        threshold_edge=0.025,  # 2.5% tolerance (matches run_image_analysis_knn)
        max_size=1024  # Downsample for speed
    )

