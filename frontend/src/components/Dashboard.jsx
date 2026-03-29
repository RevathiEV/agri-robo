import React, { useEffect, useState } from 'react'
import axios from 'axios'
import MotorControl from './MotorControl'
import ServoControl from './ServoControl'
import DiseaseDetection from './DiseaseDetection'

function Dashboard() {
  const [pumpStatus, setPumpStatus] = useState({
    pump_running: false,
    mode: 'idle',
    initialized: false,
    gpio_available: false,
  })

  const refreshPumpStatus = async () => {
    try {
      const response = await axios.get('/api/pump/status')
      setPumpStatus(response.data.pump_status)
    } catch (error) {
      console.error('Error fetching pump status:', error)
    }
  }

  useEffect(() => {
    refreshPumpStatus()

    const intervalId = window.setInterval(() => {
      refreshPumpStatus()
    }, 1000)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [])

  return (
    <div className="min-h-screen p-3 sm:p-4 md:p-8">
      {/* Header */}
      <header className="mb-6 md:mb-8 text-center">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-2">
          🍅 Agri ROBO
        </h1>
        <p className="text-white/90 text-sm sm:text-base md:text-lg px-2">
          Tomato Disease Detection & Robot Control System
        </p>
      </header>

      {/* Dashboard Grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Motor Control Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-4 sm:p-6 transition-all md:hover:scale-[1.02]">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">🤖</span>
            Robot Motor Control
          </h2>
          <MotorControl />
        </div>

        {/* Servo Control Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-4 sm:p-6 transition-all md:hover:scale-[1.02]">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">💧</span>
            Fertilizer Dispenser
          </h2>
          <ServoControl
            pumpStatus={pumpStatus}
            refreshPumpStatus={refreshPumpStatus}
          />
        </div>

        {/* Disease Detection Card - Full Width */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-2xl p-4 sm:p-6 transition-all md:hover:scale-[1.01]">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <span className="text-2xl sm:text-3xl">🔍</span>
            Leaf Disease Detection
          </h2>
          <DiseaseDetection
            refreshPumpStatus={refreshPumpStatus}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-6 md:mt-8 text-center text-white/80 text-xs sm:text-sm px-2">
        <p>Agri ROBO System - Powered by AI & IoT</p>
      </footer>
    </div>
  )
}

export default Dashboard
