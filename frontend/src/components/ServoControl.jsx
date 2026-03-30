import React, { useState } from 'react'
import axios from 'axios'

function ServoControl({ pumpStatus, refreshPumpStatus }) {
  const [pendingAction, setPendingAction] = useState(null)

  const isRunning = pumpStatus?.pump_running || false
  const mode = pumpStatus?.mode || 'idle'
  const activeTargetLabel = pumpStatus?.active_target_label || 'No Pumps'
  const status = mode === 'manual'
    ? 'Running Manually'
    : mode === 'auto'
    ? 'Auto Spray Active'
    : 'Stopped'

  const handleServoControl = async (action) => {
    try {
      setPendingAction(action)
      const endpoint = action === 'start' ? '/api/pump/start' : '/api/pump/stop'
      const response = await axios.post(endpoint)
      console.log('Pump control response:', response.data)
      await refreshPumpStatus()
    } catch (error) {
      console.error('Error controlling pump:', error)
      alert(`Pump control error: ${error.response?.data?.detail || error.message}`)
    } finally {
      setPendingAction(null)
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-3 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-3xl border border-slate-200 bg-white/80 p-4 sm:p-5">
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">
            Pump control
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Start or stop the dispensing pump manually while keeping visibility on the current operating mode.
          </p>
        </div>
        <div className={`rounded-3xl border p-4 text-center font-semibold shadow-sm ${
          isRunning
            ? 'border-emerald-400 bg-emerald-50 text-emerald-900'
            : 'border-slate-300 bg-slate-100 text-slate-800'
        }`}>
          <p className="text-xs uppercase tracking-[0.22em] opacity-70">Live status</p>
          <div className="mt-2 flex items-center justify-center gap-2">
            <span className={`h-3 w-3 rounded-full ${isRunning ? 'bg-emerald-500' : 'bg-slate-400'}`} />
            <span>{status}</span>
          </div>
          <p className="mt-2 text-xs font-medium uppercase tracking-[0.18em] opacity-70">
            {activeTargetLabel}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <button
          onClick={() => handleServoControl('start')}
          disabled={isRunning || pendingAction === 'start'}
          className="w-full rounded-3xl border border-emerald-300 bg-gradient-to-br from-emerald-500 to-emerald-700 px-5 py-4 text-sm font-semibold text-white shadow-[0_20px_40px_rgba(22,163,74,0.22)] transition-all duration-200 hover:-translate-y-0.5 hover:from-emerald-600 hover:to-emerald-700 disabled:cursor-not-allowed disabled:opacity-50 sm:px-6 sm:py-5 sm:text-base"
        >
          {pendingAction === 'start' ? 'Starting...' : 'Start Dispensing'}
        </button>
        <button
          onClick={() => handleServoControl('stop')}
          disabled={!isRunning || pendingAction === 'stop'}
          className="w-full rounded-3xl border border-rose-300 bg-gradient-to-br from-rose-500 to-rose-700 px-5 py-4 text-sm font-semibold text-white shadow-[0_20px_40px_rgba(225,29,72,0.18)] transition-all duration-200 hover:-translate-y-0.5 hover:from-rose-600 hover:to-rose-700 disabled:cursor-not-allowed disabled:opacity-50 sm:px-6 sm:py-5 sm:text-base"
        >
          {pendingAction === 'stop' ? 'Stopping...' : 'Stop Dispensing'}
        </button>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white/80 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
            Manual mode
          </p>
          <p className="mt-2 text-sm text-slate-700">
            Start Dispensing keeps both pumps running until a stop command is issued.
          </p>
        </div>
        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-sky-700">
            Automation note
          </p>
          <p className="mt-2 text-sm text-sky-800">
            Disease detection can trigger Pump 1, Pump 2, or both pumps for a timed 3 second spray.
          </p>
        </div>
      </div>
    </div>
  )
}

export default ServoControl
