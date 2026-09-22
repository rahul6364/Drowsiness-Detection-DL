import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  Activity,
  Camera,
  Clock3,
  Eye,
  Gauge,
  Play,
  ShieldCheck,
  Square,
  TimerReset,
  Volume2,
  Zap,
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'

const API_BASE = 'http://localhost:8000'
const wsUrl = `ws://localhost:8000/ws`

const prettyStatus = (value) => {
  if (!value) return 'AWAKE'
  const upper = String(value).toUpperCase()
  if (upper.includes('DROWS')) return 'DROWSY'
  if (upper.includes('WARN')) return 'WARNING'
  return 'AWAKE'
}

const formatTime = (milliseconds) => {
  const minutes = Math.floor(milliseconds / 60000)
  const seconds = Math.floor((milliseconds % 60000) / 1000)
  return `${minutes}m ${seconds}s`
}

const getStatusClass = (status) => {
  const value = String(status || '').toUpperCase()
  if (value.includes('DROWS')) return 'status-drowsy'
  if (value.includes('WARN')) return 'status-warning'
  return 'status-awake'
}

function App() {
  const [status, setStatus] = useState('awake')
  const [drowsinessScore, setDrowsinessScore] = useState(0)
  const [faceScore, setFaceScore] = useState(0)
  const [eyeScore, setEyeScore] = useState(0)
  const [leftEye, setLeftEye] = useState('OPEN')
  const [rightEye, setRightEye] = useState('OPEN')
  const [fps, setFps] = useState(0)
  const [cameraFrame, setCameraFrame] = useState('')
  const [alertActive, setAlertActive] = useState(false)
  const [alertCount, setAlertCount] = useState(0)
  const [latestAlertTime, setLatestAlertTime] = useState(null)
  const [systemInfo, setSystemInfo] = useState({})
  const [monitoring, setMonitoring] = useState(false)
  const [muted, setMuted] = useState(false)
  const [sessionStarted, setSessionStarted] = useState(Date.now())
  const [framesProcessed, setFramesProcessed] = useState(0)
  const [device, setDevice] = useState('CPU')
  const [connectionState, setConnectionState] = useState('Disconnected')
  const [socketError, setSocketError] = useState('')
  const [chartData, setChartData] = useState([])
  const [loading, setLoading] = useState(true)
  const socketRef = useRef(null)
  const timerRef = useRef(null)

  const currentStatus = prettyStatus(status)
  const chartMax = useMemo(() => Math.max(100, ...chartData.map((item) => item.score), 70), [chartData])

  const startMonitoring = async () => {
    if (monitoring) return
    try {
      const res = await fetch(`${API_BASE}/api/start`, { method: 'POST' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.error || 'Unable to start monitoring')
      }
      setMonitoring(true)
      setSocketError('')
    } catch (error) {
      setSocketError(error.message)
    }
  }

  const stopMonitoring = async () => {
    if (!monitoring) return
    try {
      await fetch(`${API_BASE}/api/stop`, { method: 'POST' })
      setMonitoring(false)
    } catch (error) {
      setSocketError(error.message)
    }
  }

  const resetSession = async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' })
      setFramesProcessed(0)
      setChartData([])
      setAlertCount(0)
      setLatestAlertTime(null)
      setSessionStarted(Date.now())
      setDrowsinessScore(0)
      setStatus('awake')
    } catch (error) {
      setSocketError(error.message)
    }
  }

  const toggleMute = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/mute`, { method: 'POST' })
      const next = await res.json()
      setMuted(Boolean(next.muted))
    } catch (error) {
      setSocketError(error.message)
    }
  }

  useEffect(() => {
    const fetchSystem = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/system`)
        const data = await res.json()
        setSystemInfo(data)
      } catch (error) {
        console.error('System metadata fetch failed', error)
      }
    }

    fetchSystem()
  }, [])

  useEffect(() => {
    let unmounted = false
    const socket = new WebSocket(wsUrl)
    socketRef.current = socket

    socket.onopen = () => {
      if (unmounted) return
      setConnectionState('Connected')
      setLoading(false)
    }

    socket.onclose = () => {
      if (unmounted) return
      setConnectionState('Disconnected')
      setMonitoring(false)
    }

    socket.onerror = () => {
      if (unmounted) return
      setConnectionState('Error')
      setSocketError('WebSocket connection error')
    }

    socket.onmessage = (event) => {
      if (unmounted) return
      try {
        const payload = JSON.parse(event.data)
        if (payload.event === 'camera_unavailable') {
          setSocketError(payload.message)
          setMonitoring(false)
          return
        }
        if (payload.event === 'backend_error') {
          setSocketError(payload.message)
          return
        }

        const nextStatus = payload.status || 'awake'
        setStatus(nextStatus)
        setDrowsinessScore(Number(payload.drowsiness_score || 0))
        setFaceScore(Number(payload.face_score || 0))
        setEyeScore(Number(payload.eye_score || 0))
        setLeftEye(payload.left_eye || 'OPEN')
        setRightEye(payload.right_eye || 'OPEN')
        setFps(Number(payload.fps || 0))
        setAlertActive(Boolean(payload.alert))
        setLatestAlertTime(payload.latest_alert_time || payload.timestamp || null)
        setFramesProcessed(Number(payload.frames_processed || 0))
        setDevice(payload.device || 'CPU')
        setAlertCount(Number(payload.alert_count || 0))

        const entry = {
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          score: Number(payload.drowsiness_score || 0),
        }
        setChartData((prev) => [...prev.slice(-59), entry])

        if (payload.frame_data) {
          setCameraFrame(`data:image/jpeg;base64,${payload.frame_data}`)
        }
      } catch (error) {
        console.error('WebSocket payload parse failed:', error)
      }
    }

    return () => {
      unmounted = true
      if (socket.readyState === WebSocket.OPEN) {
        socket.close()
      } else if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => socket.close()
      }
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  useEffect(() => {
    const timer = setInterval(() => {
      setConnectionState((current) => current)
    }, 1000)
    timerRef.current = timer
    return () => clearInterval(timer)
  }, [])

  const statusClass = getStatusClass(currentStatus)
  const sessionDuration = Date.now() - sessionStarted
  const averageFps = chartData.length ? (chartData.reduce((sum, item) => sum + (fps || 0), 0) / chartData.length).toFixed(1) : '0.0'

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="card mb-6 px-5 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-2xl font-bold tracking-tight text-white">DriverGuard AI</p>
              <p className="text-sm text-slate-400">Real-Time Driver Monitoring System</p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <span className={`status-pill ${statusClass}`}>
                <span className="h-2 w-2 rounded-full bg-current" />
                SYSTEM ONLINE
              </span>
              <div className="rounded-full border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-xs text-slate-300">
                {connectionState}
              </div>
              <div className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-xs text-slate-300">
                <Clock3 size={14} />
                {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </div>
              <div className="rounded-full border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-xs text-slate-300">
                Model: {systemInfo.face_model || 'CNN / TensorFlow'}
              </div>
            </div>
          </div>
        </header>

        <div className="dashboard-grid">
          <main className="col-span-12 space-y-6 lg:col-span-8">
            <section className="card overflow-hidden">
              <div className="border-b border-slate-700/80 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
                    <Camera size={16} className="text-emerald-300" />
                    Live Driver Monitor
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-300">
                    <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
                    LIVE
                    <span className="ml-3">FPS {fps.toFixed(1)}</span>
                  </div>
                </div>
              </div>

              <div className="relative min-h-[420px] bg-slate-950/80 p-3">
                {cameraFrame ? (
                  <img src={cameraFrame} alt="Driver camera feed" className="h-full w-full rounded-xl object-cover" />
                ) : (
                  <div className="flex h-[420px] items-center justify-center rounded-xl border border-dashed border-slate-700 bg-slate-900/60 text-slate-400">
                    {loading ? 'Connecting to backend...' : socketError || (monitoring ? 'Waiting for camera frame...' : 'Click START MONITORING to open the camera')}
                  </div>
                )}

                {alertActive && (
                  <div className="absolute left-6 top-6 rounded-md border border-red-500/40 bg-red-500/20 px-3 py-2 text-sm text-red-100 shadow-lg">
                    DROWSINESS ALERT
                  </div>
                )}
              </div>
            </section>

            <section className="card p-5">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
                  <Gauge size={16} className="text-amber-300" />
                  Real-Time Drowsiness Graph
                </div>
                <div className="text-xs text-slate-400">Alert Threshold 70%</div>
              </div>
              <div className="h-52 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 100]} stroke="#94a3b8" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px' }} />
                    <ReferenceLine y={70} label={{ value: 'Alert Threshold', position: 'insideTopRight', fill: '#fbbf24', fontSize: 11 }} stroke="#fbbf24" strokeDasharray="5 5" />
                    <Line type="monotone" dataKey="score" stroke="#34d399" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </main>

          <aside className="col-span-12 space-y-6 lg:col-span-4">
            <section className="card p-5">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
                  <ShieldCheck size={16} className="text-emerald-300" />
                  Driver Status
                </div>
                <span className={`status-pill ${statusClass}`}>
                  {currentStatus}
                </span>
              </div>

              <div className="space-y-3">
                <div className="text-3xl font-bold text-white">{drowsinessScore.toFixed(1)}%</div>
                <p className="text-sm text-slate-300">
                  {alertActive ? 'Driver appears drowsy. Please take a break.' : 'Driver appears alert'}
                </p>
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  <Activity size={14} className="text-emerald-300" />
                  Drowsiness: {drowsinessScore.toFixed(1)}%
                </div>
              </div>
            </section>

            <section className="card p-5">
              <div className="mb-4 flex items-center gap-2 text-sm font-medium text-slate-200">
                <Zap size={16} className="text-cyan-300" />
                Model Analysis
              </div>

              <div className="space-y-4">
                <div className="rounded-xl border border-slate-700 bg-slate-800/80 p-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Face Analysis</div>
                  <div className="mt-2 text-lg font-semibold text-white">CNN</div>
                  <div className="text-2xl font-bold text-emerald-300">{faceScore.toFixed(1)}%</div>
                </div>
                <div className="rounded-xl border border-slate-700 bg-slate-800/80 p-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Eye Analysis</div>
                  <div className="mt-2 text-lg font-semibold text-white">MobileNetV2</div>
                  <div className="text-2xl font-bold text-amber-300">{eyeScore.toFixed(1)}%</div>
                </div>
                <div className="rounded-xl border border-slate-700 bg-slate-800/80 p-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Final Score</div>
                  <div className="mt-2 text-lg font-semibold text-white">Fusion</div>
                  <div className="text-2xl font-bold text-red-300">{drowsinessScore.toFixed(1)}%</div>
                </div>
              </div>
            </section>

            <section className="card p-5">
              <div className="mb-4 flex items-center gap-2 text-sm font-medium text-slate-200">
                <Eye size={16} className="text-violet-300" />
                Eye Status
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-3 text-center">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Left Eye</div>
                  <div className={`mt-3 inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-xs font-bold ${leftEye === 'CLOSED' ? 'bg-red-500/20 text-red-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                    {leftEye === 'CLOSED' ? 'CLOSED' : 'OPEN'}
                  </div>
                </div>
                <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-3 text-center">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Right Eye</div>
                  <div className={`mt-3 inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-xs font-bold ${rightEye === 'CLOSED' ? 'bg-red-500/20 text-red-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                    {rightEye === 'CLOSED' ? 'CLOSED' : 'OPEN'}
                  </div>
                </div>
              </div>
            </section>
          </aside>

          <section className="col-span-12 grid gap-6 lg:grid-cols-3">
            <div className={`card p-4 ${alertActive ? 'alert-panel' : ''}`}>
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
                <AlertTriangle size={16} className="text-red-300" />
                Alert Panel
              </div>
              <div className="text-xl font-bold text-white">{alertCount}</div>
              <div className="mt-2 text-xs text-slate-400">Latest alert: {latestAlertTime || 'No alert yet'}</div>
              <div className="mt-3 text-xs uppercase tracking-[0.2em] text-red-300">Severity: {alertActive ? 'High' : 'Normal'}</div>
            </div>

            <div className="card p-4">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
                <TimerReset size={16} className="text-cyan-300" />
                Session Information
              </div>
              <div className="space-y-2 text-sm text-slate-300">
                <div>Duration: {formatTime(sessionDuration)}</div>
                <div>Frames processed: {framesProcessed}</div>
                <div>Average FPS: {averageFps}</div>
                <div>Alerts: {alertCount}</div>
                <div>Current state: {currentStatus}</div>
              </div>
            </div>

            <div className="card p-4">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
                <Activity size={16} className="text-violet-300" />
                System Info
              </div>
              <div className="space-y-2 text-sm text-slate-300">
                <div>Detection Engine: OpenCV</div>
                <div>Face Model: CNN / TensorFlow</div>
                <div>Eye Model: MobileNetV2 / PyTorch</div>
                <div>Face Input: 64 × 64</div>
                <div>Eye Input: 224 × 224</div>
                <div>Fusion: 30% Face + 70% Eyes</div>
                <div>Alert Threshold: 70%</div>
                <div>Device: {device}</div>
                <div>FPS: {fps.toFixed(1)}</div>
              </div>
            </div>
          </section>

          <div className="col-span-12 mt-2 flex flex-wrap gap-3">
            <button onClick={startMonitoring} disabled={monitoring} className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-medium text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50">
              <Play size={16} />
              START MONITORING
            </button>
            <button onClick={stopMonitoring} disabled={!monitoring} className="inline-flex items-center gap-2 rounded-xl bg-slate-700 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-50">
              <Square size={16} />
              STOP MONITORING
            </button>
            <button onClick={toggleMute} className="inline-flex items-center gap-2 rounded-xl bg-amber-500 px-4 py-2.5 text-sm font-medium text-slate-950 hover:bg-amber-400">
              <Volume2 size={16} />
              {muted ? 'UNMUTE ALERT' : 'MUTE ALERT'}
            </button>
            <button onClick={resetSession} className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2.5 text-sm font-medium text-slate-900 hover:bg-white">
              <TimerReset size={16} />
              RESET SESSION
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
