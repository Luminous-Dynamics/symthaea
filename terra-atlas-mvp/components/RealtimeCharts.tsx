'use client'

import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { 
  TrendingUp, Activity, Zap, Globe, DollarSign, 
  Calendar, MapPin, Sun, Wind, Battery, Droplets,
  AlertTriangle, Wifi, WifiOff, RefreshCw, Eye,
  ArrowUp, ArrowDown, Minus, PauseCircle, PlayCircle,
  Users, Bell, TrendingDown, Clock, Award, CheckCircle
} from 'lucide-react'

interface ChartProps {
  data: any
  filters?: {
    type?: string
    status?: string
    country?: string
    capacityRange?: { min: number; max: number }
  }
  liveMode?: boolean
  refreshInterval?: number
}

// WebSocket connection for real-time data
const chartColorStyles: Record<string, { gradient: string; text: string }> = {
  emerald: {
    gradient: 'from-emerald-500 to-emerald-400',
    text: 'text-emerald-400'
  },
  cyan: {
    gradient: 'from-cyan-500 to-cyan-400',
    text: 'text-cyan-400'
  }
}

const useRealtimeData = (enabled: boolean = false, interval: number = 5000) => {
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [realtimeData, setRealtimeData] = useState<any>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const pollingIntervalRef = useRef<NodeJS.Timeout>()
  
  const connect = useCallback(() => {
    if (!enabled) return
    
    try {
      // Since Next.js App Router doesn't support WebSocket upgrades directly,
      // we'll use polling to the REST endpoint for now
      const fetchRealtimeData = async () => {
        try {
          const response = await fetch('/api/ws', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lastUpdate: lastUpdate?.toISOString() })
          })
          
          if (response.ok) {
            const data = await response.json()
            setRealtimeData({
              timestamp: new Date(data.timestamp),
              newProjects: data.newProjects,
              capacityAdded: data.capacityAdded,
              investmentFlow: data.investmentFlow,
              activeUsers: data.activeUsers,
              regionalActivity: data.regionalActivity,
              technologyTrends: data.technologyTrends,
              investmentTiers: data.investmentTiers,
              recentEvents: data.recentEvents
            })
            setLastUpdate(new Date())
            setIsConnected(true)
          }
        } catch (error) {
          console.error('Failed to fetch real-time data:', error)
          setIsConnected(false)
        }
      }
      
      // Initial fetch
      fetchRealtimeData()
      
      // Set up polling interval
      pollingIntervalRef.current = setInterval(fetchRealtimeData, interval)
      
      // Store cleanup function
      wsRef.current = { 
        close: () => {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
          }
        } 
      } as any
      
    } catch (error) {
      console.error('Real-time connection error:', error)
      setIsConnected(false)
      
      // Retry connection after 5 seconds
      reconnectTimeoutRef.current = setTimeout(connect, 5000)
    }
  }, [enabled, interval, lastUpdate])
  
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
  }, [])
  
  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])
  
  return {
    isConnected,
    lastUpdate,
    realtimeData,
    reconnect: connect,
    disconnect
  }
}

// Animated bar chart component
function AnimatedBarChart({ data, title, color = 'emerald' }: any) {
  const [animatedHeights, setAnimatedHeights] = useState<number[]>([])
  const colorStyle = chartColorStyles[color] || chartColorStyles.emerald
  
  useEffect(() => {
    // Animate bars on mount and data change
    const heights = data.map((d: any) => d.value)
    setAnimatedHeights(new Array(data.length).fill(0))
    
    const timer = setTimeout(() => {
      setAnimatedHeights(heights)
    }, 100)
    
    return () => clearTimeout(timer)
  }, [data])
  
  const maxValue = Math.max(...data.map((d: any) => d.value))
  
  return (
    <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      <div className="h-48 flex items-end gap-2">
        {data.map((item: any, idx: number) => (
          <div key={idx} className="flex-1 flex flex-col items-center">
            <div className="relative w-full flex flex-col items-center">
              <div 
                className={`w-full bg-gradient-to-t ${colorStyle.gradient} rounded-t-lg transition-all duration-700 ease-out hover:opacity-80 cursor-pointer`}
                style={{ 
                  height: `${(animatedHeights[idx] / maxValue) * 180}px`,
                  transitionDelay: `${idx * 50}ms`
                }}
              >
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 text-xs text-gray-400 whitespace-nowrap opacity-0 hover:opacity-100 transition-opacity">
                  {item.value.toLocaleString()}
                </div>
              </div>
              <span className="text-xs text-gray-500 mt-2">{item.label}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Radial progress chart component
function RadialProgress({ value, total, label, color = 'cyan' }: any) {
  const [animatedValue, setAnimatedValue] = useState(0)
  const colorStyle = chartColorStyles[color] || chartColorStyles.cyan
  const percentage = (value / total) * 100
  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (animatedValue / 100) * circumference
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValue(percentage)
    }, 100)
    return () => clearTimeout(timer)
  }, [percentage])
  
  return (
    <div className="relative inline-flex items-center justify-center">
      <svg className="w-32 h-32 transform -rotate-90">
        <circle
          cx="64"
          cy="64"
          r="45"
          stroke="currentColor"
          strokeWidth="8"
          fill="none"
          className="text-gray-700"
        />
        <circle
          cx="64"
          cy="64"
          r="45"
          stroke="currentColor"
          strokeWidth="8"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className={`${colorStyle.text} transition-all duration-1000 ease-out`}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-2xl font-bold text-white">{animatedValue.toFixed(1)}%</span>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
    </div>
  )
}

// Live ticker component
function LiveTicker({ projects }: { projects: any[] }) {
  const [currentIndex, setCurrentIndex] = useState(0)
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % projects.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [projects.length])
  
  if (projects.length === 0) return null
  
  const project = projects[currentIndex]
  
  return (
    <div className="bg-gradient-to-r from-emerald-950/30 to-cyan-950/30 border border-emerald-500/20 rounded-xl p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-gray-400">Latest Project</span>
        </div>
        <span className="text-xs text-gray-500">Live</span>
      </div>
      <div className="mt-3">
        <h4 className="font-semibold text-white">{project.name}</h4>
        <div className="flex items-center gap-4 mt-2 text-sm">
          <span className="text-emerald-400">{project.capacity_mw} MW</span>
          <span className="text-cyan-400">${(project.investment / 1000000).toFixed(1)}M</span>
          <span className="text-purple-400">{project.state}</span>
        </div>
      </div>
    </div>
  )
}

// Geographic distribution heatmap
function GeoHeatmap({ data }: { data: any[] }) {
  const stateData = useMemo(() => {
    const grouped = data.reduce((acc, project) => {
      const state = project.state || 'Unknown'
      if (!acc[state]) {
        acc[state] = { count: 0, capacity: 0, investment: 0 }
      }
      acc[state].count++
      acc[state].capacity += project.capacity_mw || 0
      acc[state].investment += project.investment || 0
      return acc
    }, {} as Record<string, any>)
    
    return Object.entries(grouped)
      .map(([state, stats]) => ({ state, ...stats }))
      .sort((a, b) => b.capacity - a.capacity)
      .slice(0, 10)
  }, [data])
  
  const maxCapacity = Math.max(...stateData.map(s => s.capacity))
  
  return (
    <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Geographic Distribution</h3>
        <MapPin className="h-5 w-5 text-gray-400" />
      </div>
      <div className="space-y-3">
        {stateData.map((state) => (
          <div key={state.state} className="relative">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-300">{state.state}</span>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-emerald-400">{state.count} projects</span>
                <span className="text-cyan-400">{(state.capacity / 1000).toFixed(1)} GW</span>
              </div>
            </div>
            <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-700"
                style={{ width: `${(state.capacity / maxCapacity) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Time series trend chart
function TrendChart({ data, metric = 'capacity' }: any) {
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null)
  
  const points = useMemo(() => {
    // Generate mock time series data
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    const values = months.map((month, idx) => ({
      month,
      value: Math.floor(Math.random() * 50000) + 20000 + idx * 5000,
      growth: idx > 0 ? Math.floor(Math.random() * 20) + 5 : 0
    }))
    return values
  }, [])
  
  const maxValue = Math.max(...points.map(p => p.value))
  
  return (
    <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Growth Trend</h3>
        <TrendingUp className="h-5 w-5 text-emerald-400" />
      </div>
      
      <div className="h-48 relative">
        <svg className="w-full h-full">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(y => (
            <line
              key={y}
              x1="0"
              y1={`${100 - y}%`}
              x2="100%"
              y2={`${100 - y}%`}
              stroke="rgba(255,255,255,0.1)"
              strokeDasharray="4 4"
            />
          ))}
          
          {/* Trend line */}
          <polyline
            fill="none"
            stroke="url(#gradient)"
            strokeWidth="2"
            points={points.map((p, idx) => 
              `${(idx / (points.length - 1)) * 100}%,${100 - (p.value / maxValue) * 100}%`
            ).join(' ')}
          />
          
          {/* Data points */}
          {points.map((point, idx) => (
            <g key={idx}>
              <circle
                cx={`${(idx / (points.length - 1)) * 100}%`}
                cy={`${100 - (point.value / maxValue) * 100}%`}
                r="4"
                fill="#10b981"
                className="cursor-pointer"
                onMouseEnter={() => setHoveredPoint(idx)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
              {hoveredPoint === idx && (
                <g>
                  <rect
                    x={`${(idx / (points.length - 1)) * 100 - 5}%`}
                    y={`${100 - (point.value / maxValue) * 100 - 10}%`}
                    width="60"
                    height="25"
                    fill="rgba(0,0,0,0.9)"
                    rx="4"
                  />
                  <text
                    x={`${(idx / (points.length - 1)) * 100}%`}
                    y={`${100 - (point.value / maxValue) * 100}%`}
                    fill="white"
                    fontSize="12"
                    textAnchor="middle"
                  >
                    {point.value.toLocaleString()}
                  </text>
                </g>
              )}
            </g>
          ))}
          
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
        </svg>
        
        {/* X-axis labels */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-500 mt-2">
          {points.map(p => (
            <span key={p.month}>{p.month}</span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function RealtimeCharts({ data, filters, liveMode = false, refreshInterval = 5000 }: ChartProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [isPaused, setIsPaused] = useState(false)
  const [showLivePanel, setShowLivePanel] = useState(false)
  
  // Use real-time data hook
  const { 
    isConnected, 
    lastUpdate, 
    realtimeData, 
    reconnect, 
    disconnect 
  } = useRealtimeData(liveMode && !isPaused, refreshInterval)
  
  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => setIsLoading(false), 500)
    return () => clearTimeout(timer)
  }, [filters])
  
  // Format time since last update
  const getTimeSinceUpdate = () => {
    if (!lastUpdate) return 'Never'
    const seconds = Math.floor((Date.now() - lastUpdate.getTime()) / 1000)
    if (seconds < 60) return `${seconds}s ago`
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    return `${hours}h ago`
  }
  
  // Process data based on filters
  const filteredData = useMemo(() => {
    if (!data?.projects) return []
    
    return data.projects.filter((project: any) => {
      if (filters?.type && filters.type !== 'all' && project.type !== filters.type) return false
      if (filters?.status && filters.status !== 'all' && project.status !== filters.status) return false
      if (filters?.country && filters.country !== 'all' && project.country !== filters.country) return false
      if (filters?.capacityRange) {
        if (project.capacity_mw < filters.capacityRange.min) return false
        if (project.capacity_mw > filters.capacityRange.max) return false
      }
      return true
    })
  }, [data, filters])
  
  // Prepare chart data
  const typeDistribution = useMemo(() => {
    const types = ['Solar', 'Wind', 'Battery', 'Hydro']
    return types.map(type => ({
      label: type,
      value: filteredData.filter((p: any) => p.type === type).length
    }))
  }, [filteredData])
  
  const capacityByMonth = useMemo(() => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    return months.map((month, idx) => ({
      label: month,
      value: Math.floor(Math.random() * 10000) + 5000 + idx * 1000
    }))
  }, [])
  
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-pulse">
        <div className="h-64 bg-gray-900/50 rounded-xl"></div>
        <div className="h-64 bg-gray-900/50 rounded-xl"></div>
        <div className="h-64 bg-gray-900/50 rounded-xl"></div>
        <div className="h-64 bg-gray-900/50 rounded-xl"></div>
      </div>
    )
  }
  
  return (
    <div className="space-y-6">
      {/* Live Status Bar */}
      {liveMode && (
        <div className="bg-gradient-to-r from-gray-900/50 to-gray-800/50 backdrop-blur-md border border-gray-700 rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                {isConnected ? (
                  <>
                    <Wifi className="h-4 w-4 text-emerald-400 animate-pulse" />
                    <span className="text-sm text-emerald-400">Live</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-4 w-4 text-red-400" />
                    <span className="text-sm text-red-400">Disconnected</span>
                  </>
                )}
              </div>
              
              {/* Last Update */}
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <RefreshCw className="h-3 w-3" />
                <span>Updated {getTimeSinceUpdate()}</span>
              </div>
              
              {/* Real-time Metrics */}
              {realtimeData && (
                <>
                  <div className="h-4 w-px bg-gray-700"></div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <ArrowUp className="h-3 w-3 text-emerald-400" />
                      <span className="text-xs text-gray-400">Projects:</span>
                      <span className="text-xs font-bold text-emerald-400">+{realtimeData.newProjects}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Zap className="h-3 w-3 text-yellow-400" />
                      <span className="text-xs text-gray-400">Capacity:</span>
                      <span className="text-xs font-bold text-yellow-400">+{realtimeData.capacityAdded}MW</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <DollarSign className="h-3 w-3 text-cyan-400" />
                      <span className="text-xs text-gray-400">Investment:</span>
                      <span className="text-xs font-bold text-cyan-400">${(realtimeData.investmentFlow / 1000000).toFixed(1)}M</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Users className="h-3 w-3 text-purple-400" />
                      <span className="text-xs text-gray-400">Active:</span>
                      <span className="text-xs font-bold text-purple-400">{realtimeData.activeUsers}</span>
                    </div>
                  </div>
                </>
              )}
            </div>
            
            {/* Controls */}
            <div className="flex items-center gap-2">
              {/* Pause/Play Button */}
              <button
                onClick={() => setIsPaused(!isPaused)}
                className={`p-2 rounded-lg transition ${
                  isPaused 
                    ? 'bg-orange-500/20 border border-orange-500/50 text-orange-400' 
                    : 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400'
                }`}
                title={isPaused ? 'Resume updates' : 'Pause updates'}
              >
                {isPaused ? <PlayCircle className="h-4 w-4" /> : <PauseCircle className="h-4 w-4" />}
              </button>
              
              {/* Reconnect Button (if disconnected) */}
              {!isConnected && (
                <button
                  onClick={reconnect}
                  className="px-3 py-1.5 bg-red-500/20 border border-red-500/50 text-red-400 rounded-lg hover:bg-red-500/30 transition text-sm"
                >
                  Reconnect
                </button>
              )}
              
              {/* Live Panel Toggle */}
              <button
                onClick={() => setShowLivePanel(!showLivePanel)}
                className={`p-2 rounded-lg transition ${
                  showLivePanel 
                    ? 'bg-purple-500/20 border border-purple-500/50 text-purple-400' 
                    : 'bg-gray-800/50 border border-gray-700 text-gray-400'
                }`}
                title="Toggle live metrics panel"
              >
                <Eye className="h-4 w-4" />
              </button>
            </div>
          </div>
          
          {/* Expandable Live Metrics Panel */}
          {showLivePanel && realtimeData && (
            <div className="mt-4 pt-4 border-t border-gray-700 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-emerald-500/10 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">New Projects</span>
                  <ArrowUp className="h-3 w-3 text-emerald-400" />
                </div>
                <div className="text-2xl font-bold text-emerald-400">+{realtimeData.newProjects}</div>
                <div className="text-xs text-gray-500 mt-1">Last 5 minutes</div>
              </div>
              
              <div className="bg-yellow-500/10 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">Capacity Added</span>
                  <Zap className="h-3 w-3 text-yellow-400" />
                </div>
                <div className="text-2xl font-bold text-yellow-400">{realtimeData.capacityAdded} MW</div>
                <div className="text-xs text-gray-500 mt-1">New generation</div>
              </div>
              
              <div className="bg-cyan-500/10 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">Investment Flow</span>
                  <DollarSign className="h-3 w-3 text-cyan-400" />
                </div>
                <div className="text-2xl font-bold text-cyan-400">${(realtimeData.investmentFlow / 1000000).toFixed(1)}M</div>
                <div className="text-xs text-gray-500 mt-1">Capital deployed</div>
              </div>
              
              <div className="bg-purple-500/10 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">Active Users</span>
                  <Users className="h-3 w-3 text-purple-400" />
                </div>
                <div className="text-2xl font-bold text-purple-400">{realtimeData.activeUsers}</div>
                <div className="text-xs text-gray-500 mt-1">Online now</div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Live ticker */}
      <LiveTicker projects={filteredData.slice(0, 10)} />
      
      {/* Charts grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AnimatedBarChart 
          data={typeDistribution} 
          title="Projects by Type" 
          color="emerald"
        />
        
        <TrendChart data={filteredData} metric="capacity" />
        
        <GeoHeatmap data={filteredData} />
        
        <AnimatedBarChart 
          data={capacityByMonth} 
          title="Monthly Capacity Added (MW)" 
          color="cyan"
        />
        
        {/* Regional Activity Heat Map */}
        {realtimeData?.regionalActivity && (
          <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Regional Activity</h3>
              <Globe className="h-5 w-5 text-emerald-400" />
            </div>
            <div className="space-y-3">
              {Object.entries(realtimeData.regionalActivity).map(([region, activity]) => (
                <div key={region}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-gray-300">{region}</span>
                    <span className="text-xs text-emerald-400">{(activity as number).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-700"
                      style={{ width: `${activity}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Technology Trends */}
        {realtimeData?.technologyTrends && (
          <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Technology Trends</h3>
              <TrendingUp className="h-5 w-5 text-purple-400" />
            </div>
            <div className="space-y-4">
              {Object.entries(realtimeData.technologyTrends).map(([tech, data]: [string, any]) => (
                <div key={tech} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {tech === 'solar' && <Sun className="h-4 w-4 text-yellow-400" />}
                    {tech === 'wind' && <Wind className="h-4 w-4 text-blue-400" />}
                    {tech === 'battery' && <Battery className="h-4 w-4 text-purple-400" />}
                    {tech === 'hydro' && <Droplets className="h-4 w-4 text-cyan-400" />}
                    {tech === 'nuclear' && <Zap className="h-4 w-4 text-orange-400" />}
                    <span className="text-sm text-gray-300 capitalize">{tech}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-white">{data.count}</span>
                    {data.trend === 'up' && <ArrowUp className="h-3 w-3 text-emerald-400" />}
                    {data.trend === 'down' && <ArrowDown className="h-3 w-3 text-red-400" />}
                    {data.trend === 'stable' && <Minus className="h-3 w-3 text-gray-400" />}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {/* Investment Tiers Section */}
      {realtimeData?.investmentTiers && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          {Object.entries(realtimeData.investmentTiers).map(([tier, data]: [string, any]) => (
            <div key={tier} className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400 capitalize">{tier} Tier</span>
                {tier === 'micro' && <Award className="h-4 w-4 text-blue-400" />}
                {tier === 'small' && <Award className="h-4 w-4 text-emerald-400" />}
                {tier === 'medium' && <Award className="h-4 w-4 text-purple-400" />}
                {tier === 'large' && <Award className="h-4 w-4 text-yellow-400" />}
              </div>
              <div className="text-lg font-bold text-white">{data.count}</div>
              <div className="text-xs text-gray-500">Investments</div>
              <div className="text-sm font-semibold text-cyan-400 mt-2">
                ${(data.amount / 1000000).toFixed(2)}M
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* Recent Events Feed */}
      {realtimeData?.recentEvents && (
        <div className="mt-6 bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Live Event Feed</h3>
            <Bell className="h-5 w-5 text-yellow-400 animate-pulse" />
          </div>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {realtimeData.recentEvents.map((event: any, idx: number) => (
              <div key={idx} className="flex items-start gap-3 p-3 bg-gray-800/30 rounded-lg border border-gray-700/50">
                <div className="mt-1">
                  {event.type === 'project_funded' && <DollarSign className="h-4 w-4 text-emerald-400" />}
                  {event.type === 'milestone_reached' && <CheckCircle className="h-4 w-4 text-blue-400" />}
                  {event.type === 'new_investor' && <Users className="h-4 w-4 text-purple-400" />}
                </div>
                <div className="flex-1">
                  <div className="text-sm text-white">
                    {event.type === 'project_funded' && (
                      <span><strong>{event.project}</strong> received ${(event.amount / 1000000).toFixed(1)}M funding</span>
                    )}
                    {event.type === 'milestone_reached' && (
                      <span><strong>{event.project}</strong> reached {event.milestone}</span>
                    )}
                    {event.type === 'new_investor' && (
                      <span><strong>{event.name}</strong> invested ${(event.amount / 1000).toFixed(0)}K</span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
                    {event.location && (
                      <span className="flex items-center gap-1">
                        <MapPin className="h-3 w-3" />
                        {event.location}
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Radial progress indicators */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 p-6 bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl">
        <RadialProgress 
          value={data?.stats?.operational || 0} 
          total={data?.stats?.total_projects || 1} 
          label="Operational" 
          color="emerald"
        />
        <RadialProgress 
          value={data?.stats?.construction || 0} 
          total={data?.stats?.total_projects || 1} 
          label="Construction" 
          color="yellow"
        />
        <RadialProgress 
          value={data?.stats?.solar_projects || 0} 
          total={data?.stats?.total_projects || 1} 
          label="Solar Share" 
          color="amber"
        />
        <RadialProgress 
          value={data?.stats?.wind_projects || 0} 
          total={data?.stats?.total_projects || 1} 
          label="Wind Share" 
          color="blue"
        />
      </div>
    </div>
  )
}
