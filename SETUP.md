# AFM Analysis Pipeline - Setup Guide

This web application provides a complete analysis pipeline for Atomic Force Microscopy (AFM) images, including CNN classification, U-Net segmentation, Voronoi tessellation, and Color Wheel analysis.

## 📋 Prerequisites

### Required Software
- **Python 3.10+** (with Conda recommended)
- **Node.js 18+** and **npm**
- **CUDA** (optional, for GPU acceleration)

---

## 🖥️ Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AFM_Web
```

---

### Step 2: Install Node.js and npm

#### **Linux (Ubuntu/Debian)**

```bash
# Update package list
sudo apt update

# Install Node.js 18.x (LTS)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version   # Should show v18.x.x
npm --version    # Should show 9.x.x or higher
```

#### **Linux (Fedora/RHEL)**

```bash
# Enable NodeSource repository
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo dnf install -y nodejs

# Verify installation
node --version
npm --version
```

#### **Windows**

1. Download Node.js installer from: https://nodejs.org/
2. Choose **LTS version** (18.x or 20.x)
3. Run the installer and follow the prompts
4. Check "Automatically install necessary tools" if prompted
5. Restart your terminal/PowerShell

```powershell
# Verify installation (in PowerShell or Command Prompt)
node --version
npm --version
```

#### **macOS**

```bash
# Using Homebrew
brew install node@18

# Or download from https://nodejs.org/

# Verify installation
node --version
npm --version
```

---

### Step 3: Setup Python Environment (Backend)

#### **Using Conda (Recommended)**

```bash
# Create conda environment
conda create -n colorwheel python=3.10 -y
conda activate colorwheel

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Or install manually:
pip install fastapi uvicorn python-multipart pydantic torch torchvision \
    scikit-image opencv-python pillow numpy scipy matplotlib
```

#### **Using venv (Alternative)**

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

---

### Step 4: Setup Frontend (Next.js)

```bash
cd frontend

# Install Node.js dependencies
npm install

# This will install:
# - next (React framework)
# - react, react-dom
# - tailwindcss (styling)
# - plotly.js (charts)
# - lucide-react (icons)
```

---

## 🚀 Running the Application

### Step 1: Start the Backend Server

Open a terminal and run:

#### **Linux/macOS**

```bash
cd backend

# If using Conda
conda activate colorwheel
uvicorn main:app --host 0.0.0.0 --port 8000

# Or using the Python path directly
/path/to/conda/envs/colorwheel/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

#### **Windows (PowerShell)**

```powershell
cd backend

# Activate conda environment
conda activate colorwheel

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:
```
Loading models...
✓ CNN model loaded
✓ U-Net model loaded (device: cuda)  # or cpu
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start the Frontend Server

Open a **new terminal** and run:

```bash
cd frontend
npm run dev
```

You should see:
```
▲ Next.js 14.x.x
- Local:        http://localhost:3000
✓ Ready in Xms
```

### Step 3: Access the Application

Open your browser and go to: **http://localhost:3000**

---

## 📁 Project Structure

```
AFM_Web/
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # Main API server
│   ├── 1.cnn_inference.py      # CNN classification module
│   ├── 2.segmentation.py       # U-Net segmentation module
│   ├── 2.voronoi.py            # Voronoi analysis wrapper
│   ├── 3.colorwheel.py         # Color wheel analysis module
│   ├── voronoi_v7.py           # Core Voronoi implementation
│   ├── cnn_classifier.pth      # CNN model weights
│   ├── best_quality_unet.pt    # U-Net model weights
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # Next.js React frontend
│   ├── src/
│   │   ├── app/                # Next.js app router pages
│   │   ├── components/         # React components
│   │   └── lib/                # API client and utilities
│   ├── package.json            # Node.js dependencies
│   └── next.config.js          # Next.js configuration
│
├── SETUP.md                    # This file
└── README.md                   # Project overview
```

---

## 🔧 Configuration

### Backend Port
The backend runs on port **8000** by default. To change:
```bash
uvicorn main:app --host 0.0.0.0 --port <PORT>
```

### Frontend Port
The frontend runs on port **3000** by default. To change:
```bash
npm run dev -- -p <PORT>
```

If you change the backend port, update `frontend/next.config.js`:
```javascript
destination: 'http://127.0.0.1:<NEW_PORT>/:path*',
```

---

## 🐛 Troubleshooting

### Backend Issues

**"Module not found" errors:**
```bash
pip install <missing-module>
```

**CUDA/GPU not detected:**
- Ensure CUDA is installed: https://developer.nvidia.com/cuda-downloads
- PyTorch will fall back to CPU if CUDA is unavailable

**Port already in use:**
```bash
# Find and kill process on port 8000
# Linux/macOS
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Frontend Issues

**npm install fails:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**"next: command not found":**
```bash
# Ensure you're in the frontend directory
cd frontend
npm install
```

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze` | POST | Analyze AFM image (multipart/form-data) |
| `/jobs` | GET | List all analysis jobs |
| `/results/{job_id}` | GET | Get results for specific job |

---

## 🎯 Usage

1. Open http://localhost:3000 in your browser
2. Drag and drop an AFM image (PNG, JPG, TIF)
3. Wait for the analysis pipeline to complete
4. View results:
   - **Classification**: dots, lines, mixed, or irregular
   - **U-Net Mask**: Segmentation output
   - **Voronoi Tessellation** (for dots): Cell analysis
   - **Color Wheel** (for lines): Orientation mapping

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

[Add contribution guidelines here]
