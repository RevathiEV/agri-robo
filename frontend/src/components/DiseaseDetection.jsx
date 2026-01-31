import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'

function DiseaseDetection() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const [sprayRunning, setSprayRunning] = useState(false)
  const [sprayLoading, setSprayLoading] = useState(false)
  const videoRef = useRef(null)

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
      // Stop camera if active
      if (cameraActive) {
        stopCamera()
      }
    }
  }

  const startCamera = async () => {
    try {
      setError(null)
      const response = await axios.post('/api/camera/start')
      if (response.data.started) {
        setCameraActive(true)
        setPreview(null)
        setSelectedImage(null)
        setResult(null)
        
        // Stream URL will be set by useEffect when cameraActive changes
      }
    } catch (err) {
      console.error('Error starting camera:', err)
      setError(err.response?.data?.detail || 'Failed to start camera. Please try again.')
      setCameraActive(false)
    }
  }

  const stopCamera = async () => {
    try {
      await axios.post('/api/camera/stop')
      setCameraActive(false)
      if (videoRef.current) {
        videoRef.current.src = ''
      }
    } catch (err) {
      console.error('Error stopping camera:', err)
    }
  }

  const captureImage = async () => {
    if (!cameraActive) return
    
    try {
      setCapturing(true)
      setError(null)
      
      const response = await axios.post('/api/camera/capture', {}, {
        responseType: 'blob'
      })
      
      // Verify blob is valid
      if (!response.data || response.data.size === 0) {
        throw new Error('Received empty image data')
      }
      
      // Convert blob to file
      const blob = response.data
      const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' })
      
      // Create preview URL
      const previewUrl = URL.createObjectURL(blob)
      
      // Stop camera first
      setCameraActive(false)
      
      // Stop video stream
      if (videoRef.current) {
        videoRef.current.src = ''
      }
      
      // Stop camera on backend
      try {
        await axios.post('/api/camera/stop')
      } catch (stopErr) {
        console.warn('Error stopping camera:', stopErr)
      }
      
      // Set preview after a small delay to ensure UI updates
      setTimeout(() => {
        setSelectedImage(file)
        setPreview(previewUrl)
        setResult(null)
      }, 100)
      
    } catch (err) {
      console.error('Error capturing image:', err)
      setError(err.response?.data?.detail || err.message || 'Failed to capture image. Please try again.')
      setCameraActive(false)
      if (videoRef.current) {
        videoRef.current.src = ''
      }
    } finally {
      setCapturing(false)
    }
  }

  // Handle stream URL updates when camera becomes active
  useEffect(() => {
    if (cameraActive && videoRef.current) {
      // Wait a moment for camera to be ready
      const timer = setTimeout(() => {
        if (videoRef.current && cameraActive) {
          const streamUrl = '/api/camera/stream?t=' + Date.now()
          console.log('Setting stream URL:', streamUrl)
          videoRef.current.src = streamUrl
        }
      }, 800) // Wait 800ms for camera to start capturing frames
      
      return () => clearTimeout(timer)
    } else if (!cameraActive && videoRef.current) {
      videoRef.current.src = ''
    }
  }, [cameraActive])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraActive) {
        stopCamera()
      }
      // Clean up preview URLs
      if (preview) {
        URL.revokeObjectURL(preview)
      }
    }
  }, [])


  const detectDisease = async () => {
    if (!selectedImage) {
      setError('Please select or capture an image first')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await axios.post('/api/detect-disease', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
      // If new disease detected while spray is running, stop the spray
      if (sprayRunning) {
        setSprayRunning(false)
      }
    } catch (err) {
      console.error('Error detecting disease:', err)
      setError(err.response?.data?.detail || 'Failed to detect disease. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const startSpray = async () => {
    if (!result || !result.can_spray) {
      setError('Please detect a disease first before starting the dispenser.')
      return
    }

    setSprayLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/spray/start')
      if (response.data.success) {
        setSprayRunning(true)
        setError(null)
      }
    } catch (err) {
      console.error('Error starting spray:', err)
      setError(err.response?.data?.detail || 'Failed to start dispenser. Please try again.')
    } finally {
      setSprayLoading(false)
    }
  }

  const stopSpray = async () => {
    setSprayLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/spray/stop')
      if (response.data.success) {
        setSprayRunning(false)
        setError(null)
      }
    } catch (err) {
      console.error('Error stopping spray:', err)
      setError(err.response?.data?.detail || 'Failed to stop dispenser. Please try again.')
    } finally {
      setSprayLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Image Upload Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-700">Image Input</h3>
          
          {/* Camera Controls */}
          <div className="flex gap-2">
            {!cameraActive ? (
              <button
                onClick={startCamera}
                className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-all flex items-center justify-center gap-2"
              >
                📷 Open Camera
              </button>
            ) : (
              <button
                onClick={captureImage}
                disabled={capturing}
                className="flex-1 px-4 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {capturing ? '⏳ Capturing...' : '📸 Capture'}
              </button>
            )}
            
            {cameraActive && (
              <button
                onClick={stopCamera}
                className="px-4 py-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition-all"
              >
                ⏹️ Stop
              </button>
            )}
          </div>

          {/* Live Camera Stream or Preview */}
          {cameraActive ? (
            <div className="border-2 border-blue-400 rounded-lg overflow-hidden bg-black min-h-[300px] flex items-center justify-center relative">
              <img
                ref={videoRef}
                alt="Live Camera Stream"
                className="w-full h-auto max-h-[400px] object-contain"
                style={{ display: 'block' }}
                crossOrigin="anonymous"
                onError={(e) => {
                  console.error('Error loading camera stream:', e)
                  setError('Failed to load camera stream. Please try again.')
                }}
                onLoad={() => {
                  console.log('Camera stream image loaded')
                }}
              />
              {!videoRef.current?.complete && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto mb-2"></div>
                    <p>Loading camera stream...</p>
                  </div>
                </div>
              )}
            </div>
          ) : preview ? (
            <div className="space-y-3">
              <div className="border-2 border-green-400 rounded-lg overflow-hidden bg-gray-50 min-h-[300px] flex items-center justify-center p-2">
                <img
                  src={preview}
                  alt="Captured Image"
                  className="w-full h-auto max-h-[400px] object-contain rounded-lg shadow-lg"
                  style={{ display: 'block' }}
                  onError={(e) => {
                    console.error('Error loading captured image')
                    setError('Failed to load captured image. Please try capturing again.')
                    setPreview(null)
                    setSelectedImage(null)
                  }}
                  onLoad={() => {
                    console.log('Captured image loaded successfully')
                  }}
                />
              </div>
              <button
                onClick={() => {
                  if (preview) {
                    URL.revokeObjectURL(preview)
                  }
                  setPreview(null)
                  setSelectedImage(null)
                  setResult(null)
                }}
                className="w-full px-4 py-2 bg-gray-500 text-white rounded-lg text-sm hover:bg-gray-600 transition-all"
              >
                ✏️ Change Image
              </button>
            </div>
          ) : (
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors min-h-[200px] flex items-center justify-center">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                className="cursor-pointer flex flex-col items-center gap-2"
              >
                <span className="text-4xl">📁</span>
                <span className="text-gray-600 font-medium">Upload Image</span>
                <span className="text-sm text-gray-500">Click to select an image file</span>
              </label>
            </div>
          )}

          {/* Detect Button */}
          {preview && (
            <button
              onClick={detectDisease}
              disabled={loading}
              className="w-full px-6 py-4 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 text-lg shadow-lg"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin">⏳</span>
                  Analyzing Image...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  🔍 Detect Disease
                </span>
              )}
            </button>
          )}
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-700">Detection Results</h3>
          
          {error && (
            <div className="p-4 bg-red-50 border-2 border-red-200 rounded-lg">
              <p className="text-red-700 font-medium">❌ Error</p>
              <p className="text-red-600 text-sm mt-1">{error}</p>
            </div>
          )}

          {loading && (
            <div className="p-8 text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-primary-500 border-t-transparent"></div>
              <p className="mt-4 text-gray-600">Analyzing image...</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Main Result */}
              <div className={`p-6 rounded-lg border-2 ${
                result.is_healthy || result.is_not_a_leaf
                  ? 'bg-green-50 border-green-300'
                  : 'bg-yellow-50 border-yellow-300'
              }`}>
                <div className="flex items-center gap-3">
                  <span className="text-3xl">
                    {result.is_healthy || result.is_not_a_leaf ? '✅' : '⚠️'}
                  </span>
                  <div>
                    <h4 className="text-xl font-bold text-gray-800">
                      {result.disease}
                    </h4>
                    <p className="text-sm text-gray-600">
                      {result.is_healthy 
                        ? 'Healthy Leaf' 
                        : result.is_not_a_leaf 
                        ? 'Not A Leaf'
                        : 'Disease Detected'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Spray Control Buttons - Only show if disease is detected (not healthy/not a leaf) */}
              {result.can_spray && (
                <div className="space-y-3">
                  <div className="flex gap-2">
                    {!sprayRunning ? (
                      <button
                        onClick={startSpray}
                        disabled={sprayLoading || !result.can_spray}
                        className="flex-1 px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      >
                        {sprayLoading ? (
                          <>
                            <span className="animate-spin">⏳</span>
                            Starting...
                          </>
                        ) : (
                          <>
                            💧 Start Dispenser
                          </>
                        )}
                      </button>
                    ) : (
                      <button
                        onClick={stopSpray}
                        disabled={sprayLoading}
                        className="flex-1 px-6 py-3 bg-red-600 text-white rounded-lg font-semibold hover:bg-red-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      >
                        {sprayLoading ? (
                          <>
                            <span className="animate-spin">⏳</span>
                            Stopping...
                          </>
                        ) : (
                          <>
                            ⏹️ Stop Dispenser
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  {sprayRunning && (
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-sm text-blue-700 font-medium">
                        💧 Dispenser is running. Click "Stop Dispenser" to stop.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {!result && !loading && !error && (
            <div className="p-8 text-center text-gray-400 border-2 border-dashed border-gray-300 rounded-lg">
              <span className="text-4xl block mb-2">🔍</span>
              <p>Upload or capture an image to detect diseases</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DiseaseDetection

