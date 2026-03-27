import React, { useState } from 'react'
import axios from 'axios'

function MotorControl() {
  const [isMoving, setIsMoving] = useState(false)
  const [currentDirection, setCurrentDirection] = useState(null)

  const handleMotorControl = async (direction) => {
    setCurrentDirection(direction)
    setIsMoving(true)
    
    try {
      const response = await axios.post('/api/motor/control', null, {
        params: { direction }
      })
      console.log('Motor control response:', response.data)
    } catch (error) {
      console.error('Error controlling motor:', error)
      alert(`Motor control error: ${error.message}`)
    } finally {
      // Reset after a short delay for visual feedback
      setTimeout(() => {
        setIsMoving(false)
        setCurrentDirection(null)
      }, 500)
    }
  }

  const buttonClass = (direction) => {
    const baseClass = "w-full sm:w-auto px-4 sm:px-6 py-3 sm:py-4 rounded-lg font-semibold text-sm sm:text-base text-white transition-all duration-200 md:hover:scale-105 active:scale-95 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
    const activeClass = currentDirection === direction ? "ring-4 ring-offset-2" : ""
    
    switch(direction) {
      case 'front':
        return `${baseClass} ${activeClass} bg-green-500 hover:bg-green-600`
      case 'back':
        return `${baseClass} ${activeClass} bg-red-500 hover:bg-red-600`
      case 'left':
        return `${baseClass} ${activeClass} bg-blue-500 hover:bg-blue-600`
      case 'right':
        return `${baseClass} ${activeClass} bg-yellow-500 hover:bg-yellow-600`
      default:
        return `${baseClass} ${activeClass} bg-gray-500 hover:bg-gray-600`
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-gray-600 mb-4">
        Control the robot's movement direction
      </p>
      
      {/* Control Pad */}
      <div className="flex flex-col items-center gap-2 sm:gap-3">
        {/* Forward Button */}
        <button
          onClick={() => handleMotorControl('front')}
          disabled={isMoving}
          className={buttonClass('front')}
        >
          ⬆️ Forward
        </button>
        
        {/* Left and Right Row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-4 w-full sm:w-auto">
          <button
            onClick={() => handleMotorControl('left')}
            disabled={isMoving}
            className={buttonClass('left')}
          >
            ⬅️ Left
          </button>
          <button
            onClick={() => handleMotorControl('stop')}
            disabled={isMoving}
            className="w-full sm:w-auto px-4 sm:px-6 py-3 sm:py-4 rounded-lg font-semibold text-sm sm:text-base text-white bg-gray-600 hover:bg-gray-700 transition-all duration-200 md:hover:scale-105 active:scale-95 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ⏹️ Stop
          </button>
          <button
            onClick={() => handleMotorControl('right')}
            disabled={isMoving}
            className={buttonClass('right')}
          >
            ➡️ Right
          </button>
        </div>
        
        {/* Backward Button */}
        <button
          onClick={() => handleMotorControl('back')}
          disabled={isMoving}
          className={buttonClass('back')}
        >
          ⬇️ Backward
        </button>
      </div>

      {currentDirection && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg text-center">
          <p className="text-sm text-blue-700">
            Moving: <span className="font-bold">{currentDirection.toUpperCase()}</span>
          </p>
        </div>
      )}

      <div className="mt-4 p-3 bg-yellow-50 rounded-lg">
        <p className="text-xs text-yellow-700">
          ⚠️ Note: GPIO control will be implemented when Raspberry Pi is connected
        </p>
      </div>
    </div>
  )
}

export default MotorControl

