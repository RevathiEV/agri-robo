# Camera Integration Fix Plan

## Issue Identified
**CRITICAL:** Frontend (`DiseaseDetection.jsx`) tries to call `/api/camera/frame` endpoint which **does not exist** in the backend. This causes the camera preview to fail.

## Root Cause Analysis
- Frontend polls `/api/camera/frame` every 100ms to get individual frames
- Backend only has `/api/camera/stream` which is an MJPEG multipart stream endpoint
- These are incompatible approaches

## Fix Plan

### Step 1: Modify Frontend (`DiseaseDetection.jsx`)
- Replace polling mechanism with proper MJPEG stream handling
- Use the existing `/api/camera/stream` endpoint directly
- Display the stream in an `<img>` tag with the stream URL

### Step 2: Update Camera Start Logic
- Remove the `started` field check (backend returns success message instead)
- Use `response.data.success` to check camera status

### Step 3: Test the Fix
- Verify camera starts/stops correctly
- Verify live stream displays in the browser

## Files to Modify
1. `frontend/src/components/DiseaseDetection.jsx` - Fix camera stream handling

## Expected Outcome
- Camera live preview works on the frontend
- Proper error handling when camera is not available
- Compatible with the existing backend MJPEG stream endpoint

## Related Code (Backend)
The backend provides:
- `POST /api/camera/start` - Starts camera and returns `{"success": true, "message": "Camera started"}`
- `GET /api/camera/stream` - MJPEG multipart stream (`multipart/x-mixed-replace`)
- `POST /api/camera/stop` - Stops camera

## Requirements for Raspberry Pi
1. `picamera2` installed (via `sudo apt-get install python3-picamera2`)
2. Camera enabled in `sudo raspi-config`
3. Camera ribbon cable connected properly

## Frontend Fix Details
Replace the polling approach:
```javascript
// OLD (broken) - polls non-existent endpoint
streamIntervalRef.current = setInterval(async () => {
  const response = await axios.get('/api/camera/frame', { responseType: 'blob' })
  // ...
}, 100)

// NEW (working) - uses MJPEG stream directly
// Simply set videoRef.current.src to the stream URL
videoRef.current.src = `http://${apiBaseUrl}/api/camera/stream`
```

