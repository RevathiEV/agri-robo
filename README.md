# 🍅 Agri ROBO - Tomato Disease Detection System

A web application for detecting tomato leaf diseases using AI, with robot control capabilities for Raspberry Pi deployment.

## Features

- 🔍 **Disease Detection**: Upload or capture images to detect tomato leaf diseases
- 🤖 **Robot Motor Control**: Control robot movement (Front, Back, Left, Right, Stop)
- 💧 **Fertilizer Dispenser**: Control servo motor for fertilizer dispensing
- 📷 **Camera Integration**: Desktop camera support (ready for Pi Camera)

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Node.js 16+** and npm
- **Trained model files** (`tomato_disease_model.h5` and `class_mapping.json`)

## Installation

### Backend Setup

1. **Create virtual environment:**
   ```bash
   python -m venv backend/venv
   ```

2. **Activate virtual environment:**
   ```bash
   # Windows
   backend\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source backend/venv/bin/activate
   ```

3. **Install dependencies (in virtual environment, NOT globally):**
   ```bash
   # Make sure virtual environment is activated first!
   pip install -r requirements.txt
   ```
   
   **Important:** Always install in virtual environment, never globally!
   
   **Note:** If you encounter NumPy installation issues on Python 3.13:
   ```bash
   pip install --only-binary :all: numpy
   pip install -r requirements.txt
   ```

4. **Place model files in project root:**
   - `tomato_disease_model.h5` (or `tomato_disease_model_best.h5`)
   - `class_mapping.json`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

## Running the Application

### Start Backend Server

```bash
# Make sure virtual environment is activated
cd backend
python main.py
```

Backend runs on: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on: `http://localhost:5173`

## Training the Model

If you need to train your own model:

```bash
# Activate virtual environment first
backend\venv\Scripts\Activate.ps1  # Windows
# or
source backend/venv/bin/activate  # Linux/Mac

# Train model (takes 1-3 hours)
python cnn_train.py
```

**Note:** Training data (`train/` and `val/` folders) are not included in the repository due to size.

## Project Structure

```
tomato/
├── backend/              # FastAPI backend
│   └── main.py          # API server
├── frontend/             # React frontend
│   ├── src/             # Source code
│   └── package.json     # Node dependencies
├── requirements.txt     # Python dependencies (install in virtual environment)
├── cnn_train.py         # Training script
└── *.h5                 # Model files (not in git)
```

## API Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `POST /api/detect-disease` - Detect disease from image
- `POST /api/motor/control?direction={direction}` - Control motors
- `POST /api/servo/control?action={action}` - Control servo

## Testing

**Test model loading:**
```bash
cd backend
python test_model.py
```

**Test API with image:**
```bash
python test_upload.py path/to/image.jpg
```

## Requirements

See `requirements.txt` for Python dependencies.  
See `frontend/package.json` for Node.js dependencies.

## Technologies

- **Backend**: FastAPI, TensorFlow, Keras, PIL
- **Frontend**: React, Vite, Tailwind CSS, Axios
- **ML**: TensorFlow/Keras for disease classification

## License

MIT License
