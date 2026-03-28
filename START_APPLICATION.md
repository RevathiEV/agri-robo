# 🍅 Agri ROBO - Application Startup Guide

## ✅ Files Verified
- ✅ `tomato_disease_model_best.h5` (111.50 MB) - Best model with 85.56% accuracy
- ✅ `tomato_disease_model.h5` - Final model
- ✅ `class_mapping.json` - 11 classes (10 diseases + Not_A_Leaf)
- ✅ Backend code updated to handle Not_A_Leaf class

## 🚀 How to Start the Application

### Option 1: Using Batch Files (Easiest)

**Start Backend:**
```powershell
.\start_backend.bat
```

**Start Frontend (in a new terminal):**
```powershell
cd frontend
npm install
npm run dev
```

### Option 2: Manual Commands

#### Step 1: Start Backend Server

**PowerShell:**
```powershell
# Navigate to project folder
cd "C:\Users\nithi\Desktop\Agri ROBO\tomato"

# Activate virtual environment
.\backend\venv\Scripts\Activate.ps1

# Navigate to backend
cd backend

# Start server
python main.py
```

**Command Prompt:**
```cmd
cd "C:\Users\nithi\Desktop\Agri ROBO\tomato"
backend\venv\Scripts\activate.bat
cd backend
python main.py
```

#### Step 2: Start Frontend (in a NEW terminal)

**PowerShell/Command Prompt:**
```powershell
cd "C:\Users\nithi\Desktop\Agri ROBO\tomato\frontend"
npm install
npm run dev
```

## 📍 Access Points

- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Frontend:** http://localhost:3000 (or the port shown in terminal)

## ✅ Verification

1. Backend should show:
   - "Model loaded successfully!"
   - "Application startup complete"
   - Server running on port 8000

2. Frontend should open in browser automatically

3. Test the API:
   - Visit http://localhost:8000/docs
   - Try the `/api/detect-disease` endpoint

## 🔧 Troubleshooting

**Port 8000 already in use:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**Model not loading:**
- Check that `tomato_disease_model_best.h5` is in project root
- Check that `class_mapping.json` is in project root
- Verify files are not corrupted

**Frontend not starting:**
- Make sure you ran `npm install` in frontend folder
- Check if port 3000 is available

## 📊 Model Information

- **Classes:** 11 (10 tomato diseases + Not_A_Leaf)
- **Input Size:** 128x128x3
- **Accuracy:** 85.56% (best validation accuracy)
- **Training Time:** 2h 30m 56s
