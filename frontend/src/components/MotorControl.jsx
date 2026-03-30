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
      setTimeout(() => {
        setIsMoving(false)
        setCurrentDirection(null)
      }, 500)
    }
  }

  const buttonClass = (direction) => {
    const baseClass = 'min-w-0 w-full rounded-2xl border px-4 py-4 text-center text-sm font-semibold transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-50 sm:px-5 sm:py-5 sm:text-base'
    const activeClass = currentDirection === direction
      ? 'scale-[0.98] ring-2 ring-offset-2 ring-offset-white'
      : 'hover:-translate-y-0.5'

    switch (direction) {
      case 'front':
        return `${baseClass} ${activeClass} border-emerald-300 bg-emerald-50 text-emerald-900 hover:bg-emerald-100`
      case 'back':
        return `${baseClass} ${activeClass} border-rose-300 bg-rose-50 text-rose-900 hover:bg-rose-100`
      case 'left':
        return `${baseClass} ${activeClass} border-sky-300 bg-sky-50 text-sky-900 hover:bg-sky-100`
      case 'right':
        return `${baseClass} ${activeClass} border-amber-300 bg-amber-50 text-amber-900 hover:bg-amber-100`
      default:
        return `${baseClass} ${activeClass} border-slate-300 bg-slate-100 text-slate-900 hover:bg-slate-200`
    }
  }

  return (
    <div className="space-y-5">
      <div className="rounded-[30px] border border-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950 px-4 py-6 shadow-[0_24px_60px_rgba(15,23,42,0.18)] sm:px-6">
        <div className="mx-auto flex max-w-md flex-col items-center gap-3 sm:gap-4">
          <button
            onClick={() => handleMotorControl('front')}
            disabled={isMoving}
            className={buttonClass('front')}
          >
            <span className="block text-lg">↑</span>
            <span className="mt-1 block">Forward</span>
          </button>

          <div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,0.88fr)_minmax(0,1fr)] sm:gap-4">
            <button
              onClick={() => handleMotorControl('left')}
              disabled={isMoving}
              className={buttonClass('left')}
            >
              <span className="block text-lg">←</span>
              <span className="mt-1 block">Left</span>
            </button>
            <button
              onClick={() => handleMotorControl('stop')}
              disabled={isMoving}
              className="min-w-0 w-full rounded-2xl border border-slate-300 bg-slate-100 px-4 py-4 text-center text-sm font-semibold text-slate-900 transition-all duration-200 hover:-translate-y-0.5 hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50 sm:px-5 sm:py-5 sm:text-base"
            >
              <span className="block text-lg">■</span>
              <span className="mt-1 block">Stop</span>
            </button>
            <button
              onClick={() => handleMotorControl('right')}
              disabled={isMoving}
              className={buttonClass('right')}
            >
              <span className="block text-lg">→</span>
              <span className="mt-1 block">Right</span>
            </button>
          </div>

          <button
            onClick={() => handleMotorControl('back')}
            disabled={isMoving}
            className={buttonClass('back')}
          >
            <span className="block text-lg">↓</span>
            <span className="mt-1 block">Backward</span>
          </button>
        </div>
      </div>

      {currentDirection && (
        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-center">
          <p className="text-sm font-medium text-sky-800">
            Active direction: <span className="font-bold uppercase">{currentDirection}</span>
          </p>
        </div>
      )}

      <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3">
        <p className="text-xs font-medium leading-5 text-amber-800">
          Hardware note: GPIO control is fully effective once the Raspberry Pi and motor driver are connected.
        </p>
      </div>
    </div>
  )
}

export default MotorControl
