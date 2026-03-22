'use client'

import { useState, useEffect } from 'react'
import { BarChart3, TrendingUp, Award, AlertCircle, Users, Shield, Activity } from 'lucide-react'

interface VerificationStats {
  totalDataPoints: number
  verifiedCount: number
  disputedCount: number
  flaggedCount: number
  averageTrustScore: number
  topContributors: Array<{
    id: string
    validations: number
    accuracy: number
  }>
  sourceReliability: Record<string, {
    trustScore: number
    dataPoints: number
  }>
}

export default function VerificationDashboard() {
  const [stats, setStats] = useState<VerificationStats>({
    totalDataPoints: 0,
    verifiedCount: 0,
    disputedCount: 0,
    flaggedCount: 0,
    averageTrustScore: 0,
    topContributors: [],
    sourceReliability: {}
  })

  const [timeRange, setTimeRange] = useState('24h')
  const [selectedSource, setSelectedSource] = useState<string | null>(null)

  useEffect(() => {
    // Calculate stats from localStorage (in production would fetch from API)
    calculateStats()
  }, [timeRange])

  const calculateStats = () => {
    // Simulated stats calculation
    // In production, this would aggregate data from the database
    const mockStats: VerificationStats = {
      totalDataPoints: 1247,
      verifiedCount: 892,
      disputedCount: 234,
      flaggedCount: 12,
      averageTrustScore: 78.5,
      topContributors: [
        { id: 'user_001', validations: 156, accuracy: 94.2 },
        { id: 'user_002', validations: 98, accuracy: 89.7 },
        { id: 'user_003', validations: 76, accuracy: 92.1 }
      ],
      sourceReliability: {
        'NOAA': { trustScore: 95.2, dataPoints: 342 },
        'NASA FIRMS': { trustScore: 92.8, dataPoints: 278 },
        'USGS': { trustScore: 94.5, dataPoints: 185 },
        'OpenAQ': { trustScore: 78.3, dataPoints: 156 },
        'NASA EONET': { trustScore: 91.2, dataPoints: 98 },
        'Community': { trustScore: 68.5, dataPoints: 188 }
      }
    }
    
    setStats(mockStats)
  }

  const getTrustColor = (score: number) => {
    if (score >= 90) return 'text-green-400'
    if (score >= 75) return 'text-blue-400'
    if (score >= 50) return 'text-yellow-400'
    return 'text-orange-400'
  }

  return (
    <div className="bg-gray-900/90 backdrop-blur-md rounded-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Shield className="w-6 h-6 mr-2 text-blue-400" />
          Verification Dashboard
        </h2>
        
        {/* Time Range Selector */}
        <div className="flex space-x-2">
          {['24h', '7d', '30d', 'all'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                timeRange === range
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-gray-500">Total</span>
          </div>
          <div className="text-2xl font-bold text-white">{stats.totalDataPoints}</div>
          <div className="text-xs text-gray-400">Data Points</div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <Shield className="w-5 h-5 text-green-400" />
            <span className="text-xs text-gray-500">Verified</span>
          </div>
          <div className="text-2xl font-bold text-green-400">{stats.verifiedCount}</div>
          <div className="text-xs text-gray-400">
            {((stats.verifiedCount / stats.totalDataPoints) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <AlertCircle className="w-5 h-5 text-orange-400" />
            <span className="text-xs text-gray-500">Disputed</span>
          </div>
          <div className="text-2xl font-bold text-orange-400">{stats.disputedCount}</div>
          <div className="text-xs text-gray-400">Under Review</div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-gray-500">Trust</span>
          </div>
          <div className={`text-2xl font-bold ${getTrustColor(stats.averageTrustScore)}`}>
            {stats.averageTrustScore.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400">Average Score</div>
        </div>
      </div>

      {/* Source Reliability */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Source Reliability</h3>
        <div className="space-y-2">
          {Object.entries(stats.sourceReliability).map(([source, data]) => (
            <div 
              key={source}
              className={`bg-gray-800/50 rounded-lg p-3 cursor-pointer transition-all ${
                selectedSource === source ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedSource(selectedSource === source ? null : source)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-white">{source}</span>
                <div className="flex items-center space-x-3">
                  <span className="text-xs text-gray-500">{data.dataPoints} points</span>
                  <span className={`text-sm font-bold ${getTrustColor(data.trustScore)}`}>
                    {data.trustScore.toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5">
                <div
                  className={`h-1.5 rounded-full transition-all ${
                    data.trustScore >= 90 ? 'bg-green-500' :
                    data.trustScore >= 75 ? 'bg-blue-500' :
                    data.trustScore >= 50 ? 'bg-yellow-500' :
                    'bg-orange-500'
                  }`}
                  style={{ width: `${data.trustScore}%` }}
                />
              </div>
              
              {selectedSource === source && (
                <div className="mt-3 pt-3 border-t border-gray-700 text-xs text-gray-400">
                  <div className="flex justify-between mb-1">
                    <span>Last Updated:</span>
                    <span>2 minutes ago</span>
                  </div>
                  <div className="flex justify-between mb-1">
                    <span>Validation Rate:</span>
                    <span>87%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Average Latency:</span>
                    <span>1.2s</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Top Contributors */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
          <Award className="w-4 h-4 mr-1 text-yellow-400" />
          Top Validators
        </h3>
        <div className="space-y-2">
          {stats.topContributors.map((contributor, index) => (
            <div key={contributor.id} className="flex items-center justify-between bg-gray-800/50 rounded-lg p-3">
              <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                  index === 0 ? 'bg-yellow-600' : 
                  index === 1 ? 'bg-gray-400' : 
                  'bg-orange-700'
                }`}>
                  {index + 1}
                </div>
                <div>
                  <div className="text-sm font-medium text-white">
                    Anonymous Validator #{contributor.id.slice(-3)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {contributor.validations} validations
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className={`text-sm font-bold ${getTrustColor(contributor.accuracy)}`}>
                  {contributor.accuracy}%
                </div>
                <div className="text-xs text-gray-500">accuracy</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Call to Action */}
      <div className="bg-blue-900/30 border border-blue-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Users className="w-5 h-5 text-blue-400 mt-0.5" />
          <div className="flex-1">
            <h4 className="text-sm font-semibold text-blue-300 mb-1">
              Help Improve Data Quality
            </h4>
            <p className="text-xs text-gray-400 mb-2">
              Join our community of validators to help verify environmental data and improve 
              the accuracy of our planetary intelligence system.
            </p>
            <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded transition-colors">
              Become a Validator
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}