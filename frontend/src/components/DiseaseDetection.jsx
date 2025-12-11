import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'

function DiseaseDetection() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [countdown, setCountdown] = useState(0)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const countdownIntervalRef = useRef(null)

  // Ensure video plays when stream is set
  useEffect(() => {
    if (videoRef.current && streamRef.current && cameraActive) {
      const video = videoRef.current
      video.srcObject = streamRef.current
      
      video.onloadedmetadata = () => {
        video.play().catch(err => {
          console.error('Error playing video:', err)
          setError('Error starting video. Please try again.')
        })
      }
    }
  }, [cameraActive])

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const startCamera = async () => {
    try {
      // Try to get camera with maximum quality for better accuracy
      const constraints = {
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1920, min: 1280 }, // Higher resolution for better accuracy
          height: { ideal: 1080, min: 720 },
          aspectRatio: { ideal: 16/9 }
        }
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      setCameraActive(true)
      setError(null) // Clear any previous errors
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        // Force play
        videoRef.current.play().catch(err => {
          console.error('Error playing video:', err)
          setError('Error starting video. Please try again.')
        })
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError('Unable to access camera. Please check permissions or use "Upload Image" instead.')
      setCameraActive(false)
    }
  }

  const stopCamera = () => {
    // Clear countdown if active
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current)
      countdownIntervalRef.current = null
    }
    setCountdown(0)
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setCameraActive(false)
  }

  const captureImageWithCountdown = () => {
    if (videoRef.current && canvasRef.current) {
      // Start countdown
      setCountdown(3)
      setError(null)
      
      let count = 3
      countdownIntervalRef.current = setInterval(() => {
        count--
        setCountdown(count)
        
        if (count <= 0) {
          clearInterval(countdownIntervalRef.current)
          setCountdown(0)
          performCapture()
        }
      }, 1000)
    } else {
      setError('Camera not initialized. Please close and reopen the camera.')
    }
  }

  const performCapture = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      
      // Check if video is ready
      if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        setError('Camera not ready. Please wait a moment and try again.')
        return
      }
      
      // Use higher resolution for better accuracy
      const scale = 2 // Capture at 2x resolution for better quality
      canvas.width = video.videoWidth * scale
      canvas.height = video.videoHeight * scale
      const ctx = canvas.getContext('2d')
      
      // Enable high-quality rendering
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      
      // Flip the image back (since we mirrored the video)
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)
      
      // Draw video at higher resolution
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Convert to blob with maximum quality
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
          setSelectedImage(file)
          setPreview(URL.createObjectURL(blob))
          setResult(null)
          setError(null)
          stopCamera()
        } else {
          setError('Failed to capture image. Please try again.')
        }
      }, 'image/jpeg', 0.98) // Maximum quality
    } else {
      setError('Camera not initialized. Please close and reopen the camera.')
    }
  }

  const captureImage = () => {
    // Immediate capture (no countdown)
    performCapture()
  }

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
    } catch (err) {
      console.error('Error detecting disease:', err)
      setError(err.response?.data?.detail || 'Failed to detect disease. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Image Upload Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload/Camera Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-700">Image Input</h3>
          
          {/* File Upload */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors">
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

          {/* Camera Section */}
          <div className="space-y-2">
            {!cameraActive ? (
              <button
                onClick={startCamera}
                className="w-full px-4 py-3 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 transition-all transform hover:scale-105 active:scale-95"
              >
                📷 Open Camera
              </button>
            ) : (
              <div className="space-y-3">
                <div className="relative bg-black rounded-lg overflow-hidden" style={{ minHeight: '300px', width: '100%', position: 'relative' }}>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full"
                    style={{ 
                      width: '100%',
                      height: 'auto',
                      minHeight: '300px',
                      maxHeight: '500px',
                      objectFit: 'contain',
                      display: 'block',
                      transform: 'scaleX(-1)',
                      backgroundColor: '#000',
                      zIndex: 1
                    }}
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  
                  {/* Countdown overlay */}
                  {countdown > 0 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-20">
                      <div className="text-center">
                        <div className="text-8xl font-bold text-white mb-4 animate-pulse">
                          {countdown}
                        </div>
                        <p className="text-white text-lg">Get ready...</p>
                      </div>
                    </div>
                  )}
                  
                  {/* Loading indicator */}
                  {cameraActive && !videoRef.current?.readyState && countdown === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center text-white z-10">
                      <div className="text-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-white border-t-transparent mx-auto mb-2"></div>
                        <p>Starting camera...</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={captureImageWithCountdown}
                    disabled={countdown > 0}
                    className="flex-1 px-4 py-3 bg-green-500 text-white rounded-lg font-semibold hover:bg-green-600 transition-all transform hover:scale-105 active:scale-95 text-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {countdown > 0 ? `⏳ ${countdown}` : '📸 Capture (3s countdown)'}
                  </button>
                  <button
                    onClick={captureImage}
                    disabled={countdown > 0}
                    className="flex-1 px-4 py-3 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 transition-all transform hover:scale-105 active:scale-95 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    📷 Instant Capture
                  </button>
                  <button
                    onClick={stopCamera}
                    className="px-4 py-3 bg-red-500 text-white rounded-lg font-semibold hover:bg-red-600 transition-all transform hover:scale-105 active:scale-95 shadow-lg"
                  >
                    ❌ Close
                  </button>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs font-semibold text-blue-800 mb-1">📋 Tips for Best Results:</p>
                  <ul className="text-xs text-blue-700 space-y-1 list-disc list-inside">
                    <li>Ensure good lighting (natural light is best)</li>
                    <li>Fill the frame with the leaf (close-up view)</li>
                    <li>Keep the camera steady</li>
                    <li>Focus on the leaf surface clearly</li>
                    <li>Avoid shadows and reflections</li>
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Preview */}
          {preview && (
            <div className="mt-4 space-y-2">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Captured Image Preview</h4>
              <img
                src={preview}
                alt="Preview"
                className="w-full h-auto rounded-lg border-2 border-gray-200 max-h-64 object-contain"
              />
              <button
                onClick={() => {
                  setPreview(null)
                  setSelectedImage(null)
                  setResult(null)
                }}
                className="w-full px-4 py-2 bg-gray-500 text-white rounded-lg text-sm hover:bg-gray-600 transition-all"
              >
                ✏️ Change Image
              </button>
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
                result.is_healthy
                  ? 'bg-green-50 border-green-300'
                  : 'bg-yellow-50 border-yellow-300'
              }`}>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-3xl">
                    {result.is_healthy ? '✅' : '⚠️'}
                  </span>
                  <div>
                    <h4 className="text-xl font-bold text-gray-800">
                      {result.disease}
                    </h4>
                    <p className="text-sm text-gray-600">
                      {result.is_healthy ? 'Healthy Leaf' : 'Disease Detected'}
                    </p>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">Confidence:</span>
                    <span className="text-lg font-bold text-gray-800">{result.confidence}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all ${
                        result.confidence >= 70
                          ? 'bg-green-500'
                          : result.confidence >= 50
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${result.confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Top Predictions */}
              {result.top_predictions && result.top_predictions.length > 0 && (
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h5 className="font-semibold text-gray-700 mb-3">Top Predictions:</h5>
                  <div className="space-y-2">
                    {result.top_predictions.map((pred, index) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-white rounded">
                        <span className="text-sm text-gray-700">
                          {index === 0 && '🥇 '}
                          {index === 1 && '🥈 '}
                          {index === 2 && '🥉 '}
                          {pred.name}
                        </span>
                        <span className="text-sm font-semibold text-gray-800">
                          {pred.confidence.toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Confidence Warning */}
              {result.confidence < 50 && (
                <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-sm text-yellow-700">
                    ⚠️ Low confidence prediction. Please verify with a clearer image.
                  </p>
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

