import React, { useState } from 'react'
import axios from 'axios'

function ServoControl() {
  const [isRunning, setIsRunning] = useState(false)
  const [status, setStatus] = useState('Stopped')

  const handleServoControl = async (action) => {
    try {
      const endpoint = action === 'start' ? '/api/pump/start' : '/api/pump/stop'
      const response = await axios.post(endpoint)
      console.log('Pump control response:', response.data)
      
      setIsRunning(action === 'start')
      setStatus(action === 'start' ? 'Running' : 'Stopped')
    } catch (error) {
      console.error('Error controlling pump:', error)
      alert(`Pump control error: ${error.response?.data?.detail || error.message}`)
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-gray-600 mb-4">
        Control the water pump (Start / Stop dispensing)
      </p>
      
      {/* Status Display */}
      <div className={`p-4 rounded-lg text-center font-semibold ${
        isRunning 
          ? 'bg-green-100 text-green-800 border-2 border-green-500' 
          : 'bg-gray-100 text-gray-800 border-2 border-gray-300'
      }`}>
        <div className="flex items-center justify-center gap-2">
          <span className="text-2xl">{isRunning ? '🟢' : '🔴'}</span>
          <span>Status: {status}</span>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-4">
        <button
          onClick={() => handleServoControl('start')}
          disabled={isRunning}
          className="flex-1 px-6 py-4 rounded-lg font-semibold text-white bg-green-500 hover:bg-green-600 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          ▶️ Start Dispensing
        </button>
        <button
          onClick={() => handleServoControl('stop')}
          disabled={!isRunning}
          className="flex-1 px-6 py-4 rounded-lg font-semibold text-white bg-red-500 hover:bg-red-600 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          ⏹️ Stop Dispensing
        </button>
      </div>

      {/* Info */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-700">
          💡 Start Dispensing turns the pump ON; Stop Dispensing turns it OFF. No automatic spraying.
        </p>
      </div>
    </div>
  )
}

export default ServoControl

