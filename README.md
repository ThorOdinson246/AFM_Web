# AFM Analysis Pipeline v2.0

A modern web application for Atomic Force Microscopy (AFM) image analysis, featuring CNN classification, U-Net segmentation, Voronoi tessellation, and Color Wheel orientation analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   Upload    │  │   Plotly     │  │    Image Viewers       │  │
│  │   Zone      │  │   Charts     │  │    & Results           │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  CNN Model  │  │  U-Net Model │  │  Voronoi / ColorWheel  │  │
│  │  (PyTorch)  │  │  (PyTorch)   │  │      Analysis          │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Workflow

1. **Image Upload** - User uploads AFM image (JPG, PNG, or TIFF)
2. **U-Net Segmentation** - Neural network extracts features from the image
3. **CNN Classification** - Classifies the segmentation mask into:
   - `dots` - Dot-like structures
   - `lines` - Line patterns
   - `mixed` - Mixed morphology
4. **Feature Analysis** - Based on classification:
   - **Dots/Mixed**: Voronoi tessellation analysis
   - **Lines**: Color wheel orientation analysis

## Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Plotly.js** - Interactive charts
- **Lucide React** - Icons
- **React Dropzone** - File upload

### Backend
- **FastAPI** - Modern Python API framework
- **PyTorch** - Deep learning models
- **Uvicorn** - ASGI server
- **Pillow** - Image processing

## Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ with conda (recommended)
- CUDA-capable GPU (optional, for faster inference)

### 1. Backend Setup

```bash
cd backend

# Create/activate conda environment (if using conda)
conda activate your_env_name

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The backend will run at `http://127.0.0.1:8000`

### 2. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm run dev
```

The frontend will run at `http://localhost:3000`

### 3. Access the Application

Open your browser and navigate to `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/analyze` | Analyze uploaded image |
| GET | `/jobs` | List all analysis jobs |
| GET | `/results/{job_id}` | Get results for a job |

## Project Structure

```
AFM_Web/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css  # Global styles
│   │   │   ├── layout.tsx   # Root layout
│   │   │   └── page.tsx     # Main page
│   │   ├── components/      # React components
│   │   ├── lib/
│   │   │   └── api.ts       # API client
│   │   └── types/           # TypeScript types
│   ├── package.json
│   ├── tailwind.config.js
│   └── next.config.js
├── 1.cnn_inference.py       # CNN model code
├── 2.segmentation.py        # U-Net model code
├── 2.voronoi.py             # Voronoi analysis
├── 3.colorwheel.py          # Color wheel analysis
├── cnn_classifier.pth       # CNN weights
├── best_quality_unet.pt     # U-Net weights
└── voronoi_v7.py            # Voronoi utilities
```

## Features

- 📤 Drag & drop image upload
- 🖼️ Fullscreen image viewer
- 📊 Interactive Plotly charts for classification probabilities
- 📈 Confidence gauge visualization
- 📋 Detailed analysis metrics display
- 🔄 Real-time pipeline status tracking
- 🎨 Professional dark theme UI

## Color Palette

The application uses a professional, scientific color scheme:
- Background: `#0f1419` (dark navy)
- Cards: `#1a2332` (slate)
- Borders: `#2d3f56` (muted blue)
- Text: `#e8edf4` (off-white)
- Accents: Blue/Cyan gradient

## Notes

- Models are loaded once at startup for optimal performance
- Image analysis is performed synchronously
- Results are cached in the `results/` directory
- Each analysis job gets a unique ID for tracking
