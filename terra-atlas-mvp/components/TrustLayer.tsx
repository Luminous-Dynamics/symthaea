'use client'

import { useState, useEffect } from 'react'
import { Shield, AlertTriangle, CheckCircle, Users, ThumbsUp, ThumbsDown, Flag, Info } from 'lucide-react'

interface DataPoint {
  id: string
  type: string
  source: string
  quality_score: number
  data_lineage: string[]
  verification_status: string
  user_validations?: {
    confirms: number
    disputes: number
    flags: number
  }
}

interface TrustLayerProps {
  feature?: any // GeoJSON feature
  onValidate?: (validation: 'confirm' | 'dispute' | 'flag') => void
}

export default function TrustLayer({ feature, onValidate }: TrustLayerProps) {
  const [expanded, setExpanded] = useState(false)
  const [userVote, setUserVote] = useState<'confirm' | 'dispute' | 'flag' | null>(null)
  const [validations, setValidations] = useState({
    confirms: 0,
    disputes: 0,
    flags: 0
  })

  useEffect(() => {
    // Load any existing validations from localStorage or API
    if (feature?.properties?.id) {
      const stored = localStorage.getItem(`validation_${feature.properties.id}`)
      if (stored) {
        const data = JSON.parse(stored)
        setValidations(data.validations)
        setUserVote(data.userVote)
      }
    }
  }, [feature])

  if (!feature) return null

  const props = feature.properties || {}
  const qualityScore = props.quality_score || 0
  const dataLineage = props.data_lineage || []
  const verificationStatus = props.verification_status || 'unverified'

  // Calculate trust score based on multiple factors
  const calculateTrustScore = () => {
    let score = qualityScore * 100

    // Adjust based on verification status
    switch (verificationStatus) {
      case 'official':
      case 'official_government':
        score = Math.min(score + 20, 100)
        break
      case 'satellite_confirmed':
      case 'expert_verified':
        score = Math.min(score + 15, 100)
        break
      case 'sensor_network':
        score = Math.min(score + 10, 100)
        break
      case 'crowdsourced':
        score = Math.min(score + 5, 100)
        break
      case 'unverified':
      default:
        score = Math.max(score - 10, 0)
    }

    // Adjust based on user validations
    const totalVotes = validations.confirms + validations.disputes
    if (totalVotes > 0) {
      const confirmRatio = validations.confirms / totalVotes
      score = score * 0.7 + (confirmRatio * 100 * 0.3) // 30% weight to user validation
    }

    // Penalize heavily flagged content
    if (validations.flags > 5) {
      score = Math.max(score - (validations.flags * 2), 0)
    }

    return Math.round(score)
  }

  const trustScore = calculateTrustScore()

  // Determine trust level and color
  const getTrustLevel = (score: number) => {
    if (score >= 90) return { level: 'Very High', color: 'text-green-400', bgColor: 'bg-green-900/50' }
    if (score >= 75) return { level: 'High', color: 'text-blue-400', bgColor: 'bg-blue-900/50' }
    if (score >= 50) return { level: 'Moderate', color: 'text-yellow-400', bgColor: 'bg-yellow-900/50' }
    if (score >= 25) return { level: 'Low', color: 'text-orange-400', bgColor: 'bg-orange-900/50' }
    return { level: 'Very Low', color: 'text-red-400', bgColor: 'bg-red-900/50' }
  }

  const trust = getTrustLevel(trustScore)

  const handleValidation = (type: 'confirm' | 'dispute' | 'flag') => {
    const newValidations = { ...validations }
    
    // Remove previous vote if exists
    if (userVote) {
      if (userVote === 'confirm') newValidations.confirms--
      if (userVote === 'dispute') newValidations.disputes--
      if (userVote === 'flag') newValidations.flags--
    }

    // Add new vote
    if (type === 'confirm') newValidations.confirms++
    if (type === 'dispute') newValidations.disputes++
    if (type === 'flag') newValidations.flags++

    setValidations(newValidations)
    setUserVote(type)

    // Save to localStorage
    if (feature.properties?.id) {
      localStorage.setItem(`validation_${feature.properties.id}`, JSON.stringify({
        validations: newValidations,
        userVote: type
      }))
    }

    // Callback to parent
    if (onValidate) {
      onValidate(type)
    }

    // In production, would send to API
    // fetch('/api/validate', { method: 'POST', body: JSON.stringify({ featureId, type }) })
  }

  return (
    <div className="bg-gray-900/90 backdrop-blur-md rounded-lg p-4 border border-gray-700">
      {/* Trust Score Header */}
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center space-x-3">
          <Shield className={`w-5 h-5 ${trust.color}`} />
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-white font-semibold">Trust Score:</span>
              <span className={`font-bold ${trust.color}`}>{trustScore}%</span>
              <span className={`text-xs px-2 py-1 rounded ${trust.bgColor} ${trust.color}`}>
                {trust.level}
              </span>
            </div>
          </div>
        </div>
        <Info className="w-4 h-4 text-gray-400" />
      </div>

      {expanded && (
        <div className="mt-4 space-y-4">
          {/* Data Lineage */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Data Lineage</h4>
            <div className="flex flex-wrap gap-2">
              {dataLineage.map((source: string, i: number) => (
                <span 
                  key={i}
                  className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded"
                >
                  {source}
                  {i < dataLineage.length - 1 && <span className="ml-1">→</span>}
                </span>
              ))}
            </div>
          </div>

          {/* Verification Status */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Verification Status</h4>
            <div className="flex items-center space-x-2">
              {verificationStatus === 'official' || verificationStatus === 'official_government' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : verificationStatus === 'satellite_confirmed' || verificationStatus === 'expert_verified' ? (
                <CheckCircle className="w-4 h-4 text-blue-400" />
              ) : verificationStatus === 'sensor_network' ? (
                <Shield className="w-4 h-4 text-yellow-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-orange-400" />
              )}
              <span className="text-sm text-gray-400 capitalize">
                {verificationStatus.replace(/_/g, ' ')}
              </span>
            </div>
          </div>

          {/* Quality Metrics */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Quality Metrics</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Raw Quality Score</span>
                <span className="text-xs text-gray-300">{(qualityScore * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-600 to-blue-400 h-2 rounded-full"
                  style={{ width: `${qualityScore * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Community Validation */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2 flex items-center">
              <Users className="w-4 h-4 mr-1" />
              Community Validation
            </h4>
            
            {/* Validation Stats */}
            <div className="flex space-x-4 mb-3">
              <div className="text-center">
                <div className="text-green-400 font-semibold">{validations.confirms}</div>
                <div className="text-xs text-gray-500">Confirms</div>
              </div>
              <div className="text-center">
                <div className="text-orange-400 font-semibold">{validations.disputes}</div>
                <div className="text-xs text-gray-500">Disputes</div>
              </div>
              <div className="text-center">
                <div className="text-red-400 font-semibold">{validations.flags}</div>
                <div className="text-xs text-gray-500">Flags</div>
              </div>
            </div>

            {/* Validation Buttons */}
            <div className="flex space-x-2">
              <button
                onClick={() => handleValidation('confirm')}
                className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                  userVote === 'confirm' 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <ThumbsUp className="w-4 h-4 inline mr-1" />
                Confirm
              </button>
              <button
                onClick={() => handleValidation('dispute')}
                className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                  userVote === 'dispute' 
                    ? 'bg-orange-600 text-white' 
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <ThumbsDown className="w-4 h-4 inline mr-1" />
                Dispute
              </button>
              <button
                onClick={() => handleValidation('flag')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  userVote === 'flag' 
                    ? 'bg-red-600 text-white' 
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <Flag className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Trust Explanation */}
          <div className="text-xs text-gray-500 border-t border-gray-800 pt-3">
            <p>
              Trust Score combines multiple factors: data source reliability ({verificationStatus === 'official' ? '95%' : '70%'}), 
              quality metrics ({(qualityScore * 100).toFixed(0)}%), 
              and community validation ({validations.confirms + validations.disputes} votes).
            </p>
          </div>
        </div>
      )}
    </div>
  )
}