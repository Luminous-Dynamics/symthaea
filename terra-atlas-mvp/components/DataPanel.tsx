import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, AlertTriangle, Globe, TrendingUp, Shield, ChevronRight } from 'lucide-react'
import TrustLayer from './TrustLayer'
import VerificationDashboard from './VerificationDashboard'

interface DataPanelProps {
  data: any
  loading: boolean
  activeLayer: string
}

export default function DataPanel({ data, loading, activeLayer }: DataPanelProps) {
  const [showTrust, setShowTrust] = useState(false)
  const [showVerification, setShowVerification] = useState(false)
  const [selectedFeature, setSelectedFeature] = useState<any>(null)
  const stats = {
    total: data?.features?.length || 0,
    highConfidence: data?.features?.filter((f: any) => f.properties?.confidence >= 70).length || 0,
    lastUpdate: data?.metadata?.last_updated ? new Date(data.metadata.last_updated).toLocaleTimeString() : 'Never',
    source: data?.metadata?.source || 'Unknown'
  }

  return (
    <div className="w-80 bg-gray-900 border-l border-gray-800 p-4 overflow-y-auto">
      <h2 className="text-xl font-bold mb-4">Data Overview</h2>
      
      {/* Quick Stats */}
      <div className="space-y-3 mb-6">
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" />
              Active Events
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total.toLocaleString()}</div>
            <p className="text-xs text-gray-400">Last 48 hours</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-orange-400" />
              High Confidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.highConfidence.toLocaleString()}</div>
            <p className="text-xs text-gray-400">≥70% confidence</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Globe className="w-4 h-4 text-green-400" />
              Data Source
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm font-bold">{stats.source}</div>
            <p className="text-xs text-gray-400">Updated: {stats.lastUpdate}</p>
          </CardContent>
        </Card>
      </div>

      {/* Layer Info */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold mb-2 text-gray-400">Active Layer</h3>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              activeLayer === 'fires' ? 'bg-red-500' :
              activeLayer === 'earthquakes' ? 'bg-orange-500' :
              activeLayer === 'weather' ? 'bg-blue-500' :
              'bg-gray-500'
            }`} />
            <span className="capitalize">{activeLayer}</span>
          </div>
        </div>
      </div>

      {/* Attribution */}
      <div className="mt-auto pt-4 border-t border-gray-800">
        <h3 className="text-sm font-semibold mb-2 text-gray-400">Data Attribution</h3>
        <div className="text-xs text-gray-500 space-y-1">
          <p>• NASA FIRMS - Fire detection</p>
          <p>• USGS - Earthquake monitoring</p>
          <p>• OpenWeatherMap - Climate data</p>
          <p>• Carbon Monitor - Emissions</p>
        </div>
      </div>

      {/* Trust Badge */}
      <div className="mt-4 p-3 bg-gradient-to-r from-blue-900/20 to-green-900/20 rounded-lg border border-blue-800/30">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-400" />
          <span className="text-xs">Data Quality Score</span>
        </div>
        <div className="mt-1 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-blue-400 to-green-400"
            style={{ width: `${data?.metadata?.quality_score || 75}%` }}
          />
        </div>
      </div>

      {/* Trust & Verification Controls */}
      <div className="mt-4 space-y-2">
        <button
          onClick={() => setShowTrust(!showTrust)}
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-blue-400" />
            <span className="text-sm">Trust Layer</span>
          </div>
          <ChevronRight className={`w-4 h-4 transition-transform ${showTrust ? 'rotate-90' : ''}`} />
        </button>

        <button
          onClick={() => setShowVerification(!showVerification)}
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-green-400" />
            <span className="text-sm">Verification Stats</span>
          </div>
          <ChevronRight className={`w-4 h-4 transition-transform ${showVerification ? 'rotate-90' : ''}`} />
        </button>
      </div>

      {/* Show Trust Layer */}
      {showTrust && (
        <div className="mt-4">
          <TrustLayer 
            feature={data?.features?.[0] || null}
            onValidate={(type) => {
              console.log('Validation:', type)
              // In production, would send to API
            }}
          />
        </div>
      )}

      {/* Show Verification Dashboard */}
      {showVerification && (
        <div className="mt-4">
          <VerificationDashboard />
        </div>
      )}
    </div>
  )
}