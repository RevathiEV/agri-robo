import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'

function DiseaseDetection({ refreshPumpStatus }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const videoRef = useRef(null)
  const streamIntervalRef = useRef(null)

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
      if (cameraActive) {
        stopCamera()
      }
    }
  }

  const startCamera = async () => {
    try {
      setError(null)
      const response = await axios.post('/api/camera/start')
      if (response.data.success) {
        setCameraActive(true)
        setPreview(null)
        setSelectedImage(null)
        setResult(null)
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

      if (!response.data || response.data.size === 0) {
        throw new Error('Received empty image data')
      }

      const blob = response.data
      const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' })
      const previewUrl = URL.createObjectURL(blob)

      setCameraActive(false)

      if (videoRef.current) {
        videoRef.current.src = ''
      }

      try {
        await axios.post('/api/camera/stop')
      } catch (stopErr) {
        console.warn('Error stopping camera:', stopErr)
      }

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

  useEffect(() => {
    if (cameraActive && videoRef.current) {
      const apiUrl = import.meta.env.VITE_API_URL || ''
      videoRef.current.src = `${apiUrl}/api/camera/stream`
    } else if (!cameraActive && videoRef.current) {
      videoRef.current.src = ''
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current)
        streamIntervalRef.current = null
      }
    }
  }, [cameraActive])

  useEffect(() => {
    return () => {
      if (cameraActive) {
        stopCamera()
      }
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
      await refreshPumpStatus()
    } catch (err) {
      console.error('Error detecting disease:', err)
      setError(err.response?.data?.detail || 'Failed to detect disease. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const resultTone = result?.is_healthy || result?.is_not_a_leaf
    ? 'border-emerald-300 bg-emerald-50'
    : 'border-amber-300 bg-amber-50'

  return (
    <div className="space-y-5 sm:space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {!cameraActive ? (
              <button
                onClick={startCamera}
                className="w-full rounded-2xl border border-sky-300 bg-sky-600 px-4 py-3 font-medium text-white transition-all hover:-translate-y-0.5 hover:bg-sky-700"
              >
                Open Camera
              </button>
            ) : (
              <button
                onClick={captureImage}
                disabled={capturing}
                className="w-full rounded-2xl border border-emerald-300 bg-emerald-600 px-4 py-3 font-medium text-white transition-all hover:-translate-y-0.5 hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {capturing ? 'Capturing...' : 'Capture'}
              </button>
            )}

            {cameraActive && (
              <button
                onClick={stopCamera}
                className="w-full rounded-2xl border border-rose-300 bg-rose-600 px-4 py-3 font-medium text-white transition-all hover:-translate-y-0.5 hover:bg-rose-700 sm:col-span-2"
              >
                Stop Camera
              </button>
            )}
          </div>

          {cameraActive ? (
            <div className="relative flex min-h-[240px] items-center justify-center overflow-hidden rounded-[28px] border border-slate-200 bg-slate-950 shadow-[0_24px_60px_rgba(15,23,42,0.16)] sm:min-h-[320px]">
              <img
                ref={videoRef}
                alt="Live Camera Stream"
                className="h-auto max-h-[300px] w-full object-contain sm:max-h-[420px]"
                style={{ display: 'block' }}
                crossOrigin="anonymous"
                onError={(e) => {
                  console.error('Stream error:', e)
                  setError('Failed to load camera stream. Check if camera is started.')
                }}
                onLoad={() => {
                  console.log('Stream loaded')
                }}
              />
              <div className="absolute left-4 top-4 rounded-full border border-white/15 bg-black/45 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-white backdrop-blur-sm">
                Live Pi camera stream
              </div>
            </div>
          ) : preview ? (
            <div className="space-y-3">
              <div className="flex min-h-[240px] items-center justify-center overflow-hidden rounded-[28px] border border-emerald-300 bg-white p-3 shadow-[0_24px_60px_rgba(16,39,35,0.08)] sm:min-h-[320px]">
                <img
                  src={preview}
                  alt="Captured Image"
                  className="h-auto max-h-[300px] w-full rounded-2xl object-contain shadow-lg sm:max-h-[420px]"
                  style={{ display: 'block' }}
                  onError={() => {
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
                className="w-full rounded-2xl border border-slate-300 bg-slate-100 px-4 py-3 text-sm font-medium text-slate-800 transition-all hover:bg-slate-200"
              >
                Change Image
              </button>
            </div>
          ) : (
            <div className="flex min-h-[240px] items-center justify-center rounded-[28px] border-2 border-dashed border-slate-300 bg-white/65 p-6 text-center transition-colors hover:border-emerald-500 hover:bg-white">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                className="flex cursor-pointer flex-col items-center gap-2"
              >
                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-emerald-800">
                  Upload zone
                </span>
                <span className="text-xl font-semibold text-slate-800">Select a tomato leaf image</span>
                <span className="text-sm text-slate-500">Click here to browse and prepare a sample for analysis</span>
              </label>
            </div>
          )}

          {preview && (
            <button
              onClick={detectDisease}
              disabled={loading}
              className="w-full rounded-[26px] border border-emerald-400 bg-gradient-to-r from-emerald-600 via-emerald-700 to-teal-800 px-5 py-4 text-base font-semibold text-white shadow-[0_22px_50px_rgba(21,128,61,0.25)] transition-all duration-200 hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50 sm:px-6 sm:text-lg"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin">◌</span>
                  Analyzing Image...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">Detect Disease</span>
              )}
            </button>
          )}
        </div>

        <div className="space-y-4">
          <div className="rounded-3xl border border-slate-200 bg-white/80 p-4 sm:p-5">
            <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">
              Detection results
            </p>
            <h3 className="mt-2 text-2xl font-bold text-slate-900">Model response and spray decision</h3>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Review classification output, confidence, and whether any automatic spray sequence was triggered.
            </p>
          </div>

          {error && (
            <div className="rounded-3xl border border-rose-200 bg-rose-50 p-4">
              <p className="font-medium text-rose-800">Error</p>
              <p className="mt-1 text-sm text-rose-700">{error}</p>
            </div>
          )}

          {loading && (
            <div className="rounded-3xl border border-slate-200 bg-white/70 p-8 text-center">
              <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent"></div>
              <p className="mt-4 text-slate-600">Analyzing image...</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className={`rounded-3xl border p-6 ${resultTone}`}>
                <div className="flex items-start gap-3 sm:items-center">
                  <div className={`flex h-12 w-12 items-center justify-center rounded-2xl text-xl font-bold ${
                    result.is_healthy || result.is_not_a_leaf
                      ? 'bg-emerald-600 text-white'
                      : 'bg-amber-500 text-white'
                  }`}>
                    {result.is_healthy || result.is_not_a_leaf ? 'OK' : '!'}
                  </div>
                  <div>
                    <h4 className="break-words text-lg font-bold text-slate-900 sm:text-xl">
                      {result.disease}
                    </h4>
                    <p className="text-sm text-slate-600">
                      {result.is_healthy
                        ? 'Healthy Leaf'
                        : result.is_not_a_leaf
                        ? 'Not A Leaf'
                        : 'Disease Detected'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                {result.is_healthy || result.is_not_a_leaf ? (
                  <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
                    <p className="font-medium text-emerald-800">
                      Healthy leaf or non-leaf sample detected. No spray action required.
                    </p>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
                    <p className="font-medium text-amber-800">
                      {result.auto_dispense_started
                        ? result.pump_message || 'Automatic spray started for 3 seconds.'
                        : `Disease detected. ${result.pump_message || 'Use Start / Stop Dispensing to control the pump.'}`}
                    </p>
                  </div>
                )}

              </div>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="rounded-[28px] border-2 border-dashed border-slate-300 bg-white/65 p-8 text-center text-slate-500">
              <span className="block text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                Awaiting sample
              </span>
              <p className="mt-3 text-lg font-semibold text-slate-700">
                Upload or capture an image to start disease detection
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DiseaseDetection
