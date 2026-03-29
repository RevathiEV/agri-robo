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

  const modeLabel = pumpStatus.mode === 'manual'
    ? 'Manual control'
    : pumpStatus.mode === 'auto'
    ? 'Auto spray active'
    : 'Idle'

  return (
    <div className="app-shell min-h-screen px-4 py-5 sm:px-6 lg:px-8 lg:py-8">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-[radial-gradient(circle_at_top,rgba(31,143,97,0.2),transparent_55%)]" />
      <div className="pointer-events-none absolute right-0 top-20 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgba(241,181,92,0.32),transparent_62%)] blur-3xl" />

      <div className="mx-auto flex max-w-7xl flex-col gap-6 lg:gap-8">
        <header className="hero-panel px-5 py-6 sm:px-8 sm:py-8 lg:px-10">
          <div className="relative">
            <div>
              <span className="eyebrow border-white/15 bg-white/10 text-white/80">
                Smart Agriculture Demonstration Platform
              </span>
              <h1 className="mt-4 max-w-3xl text-4xl font-bold leading-tight sm:text-5xl lg:text-6xl">
                Agri ROBO
              </h1>
              <div className="mt-5 flex flex-wrap gap-2">
                <span className="info-chip border-white/15 bg-white/10 text-white/85">
                  AI disease analysis
                </span>
                <span className="info-chip border-white/15 bg-white/10 text-white/85">
                  Real-time motion control
                </span>
                <span className="info-chip border-white/15 bg-white/10 text-white/85">
                  Automated spray workflow
                </span>
              </div>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-4 sm:gap-6 lg:grid-cols-2">
          <div className="section-card soft-grid p-5 sm:p-6">
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <span className="eyebrow border-emerald-200 bg-emerald-50 text-emerald-800">
                  Mobility
                </span>
                <h2 className="mt-3 text-2xl font-bold text-slate-900">
                  Robot Motor Control
                </h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
                  Directional movement controls for navigating the robot during
                  inspection, demonstration, and field positioning.
                </p>
              </div>
            </div>
            <MotorControl />
          </div>

          <div className="section-card soft-grid p-5 sm:p-6">
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <span className="eyebrow border-sky-200 bg-sky-50 text-sky-800">
                  Actuation
                </span>
                <h2 className="mt-3 text-2xl font-bold text-slate-900">
                  Fertilizer Dispenser
                </h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
                  Manual spray controls with live pump feedback, ready for
                  controlled demos and precision dispensing scenarios.
                </p>
              </div>
              <div className="hidden rounded-2xl border border-slate-200 bg-white/70 px-4 py-3 text-right sm:block">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Live mode
                </p>
                <p className="mt-1 text-sm font-semibold text-slate-800">{modeLabel}</p>
              </div>
            </div>
            <ServoControl
              pumpStatus={pumpStatus}
              refreshPumpStatus={refreshPumpStatus}
            />
          </div>

          <div className="section-card soft-grid p-5 sm:p-6 lg:col-span-2">
            <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <span className="eyebrow border-amber-200 bg-amber-50 text-amber-800">
                  Computer Vision
                </span>
                <h2 className="mt-3 text-2xl font-bold text-slate-900">
                  Leaf Disease Detection
                </h2>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                  Capture or upload a leaf image, run the trained tomato disease model,
                  and review the classification result alongside spray action feedback.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="info-chip border-slate-200 bg-white/90 text-slate-700">
                  Image upload
                </span>
                <span className="info-chip border-slate-200 bg-white/90 text-slate-700">
                  Pi camera stream
                </span>
                <span className="info-chip border-slate-200 bg-white/90 text-slate-700">
                  Automated response
                </span>
              </div>
            </div>
            <DiseaseDetection refreshPumpStatus={refreshPumpStatus} />
          </div>
        </div>

        <footer className="pb-2 text-center text-xs font-medium uppercase tracking-[0.22em] text-slate-500 sm:text-sm">
          Agri ROBO • AI vision, robotics control, and precision spray interface
        </footer>
      </div>
    </div>
  )
}

export default Dashboard
