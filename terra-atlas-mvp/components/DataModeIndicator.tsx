'use client'

import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle, XCircle } from 'lucide-react'

interface DataSource {
  name: string
  status: 'live' | 'demo' | 'error'
  apiKey?: 'yes' | 'no' | 'not_required'
}

export default function DataModeIndicator() {
  const [dataSources, setDataSources] = useState<DataSource[]>([])
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    // Check status of all data sources
    const checkDataSources = async () => {
      const sources = ['fires', 'earthquakes', 'weather', 'emissions']
      const statuses: DataSource[] = []

      for (const source of sources) {
        try {
          const response = await fetch(`/api/data/${source}`)
          const data = await response.json()
          
          statuses.push({
            name: source.charAt(0).toUpperCase() + source.slice(1),
            status: data.metadata?.dataMode || 'demo',
            apiKey: data.metadata?.apiKeyPresent
          })
        } catch {
          statuses.push({
            name: source.charAt(0).toUpperCase() + source.slice(1),
            status: 'error'
          })
        }
      }

      setDataSources(statuses)
    }

    checkDataSources()
  }, [])

  const hasRealData = dataSources.some(s => s.status === 'live')
  const allDemo = dataSources.every(s => s.status === 'demo')

  return (
    <div className="fixed bottom-4 left-4 z-50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`
          px-4 py-2 rounded-lg backdrop-blur-md shadow-lg
          transition-all duration-200 hover:scale-105
          ${hasRealData 
            ? 'bg-green-900/80 hover:bg-green-800/80' 
            : 'bg-yellow-900/80 hover:bg-yellow-800/80'}
        `}
      >
        <div className="flex items-center space-x-2">
          {hasRealData ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <AlertCircle className="w-5 h-5 text-yellow-400" />
          )}
          <span className="text-white font-medium">
            {hasRealData ? 'Live Data' : 'Demo Mode'}
          </span>
        </div>
      </button>

      {isExpanded && (
        <div className="mt-2 p-4 bg-gray-900/90 backdrop-blur-md rounded-lg shadow-xl">
          <h3 className="text-white font-semibold mb-3">Data Sources</h3>
          
          <div className="space-y-2 mb-3">
            {dataSources.map((source) => (
              <div key={source.name} className="flex items-center justify-between">
                <span className="text-gray-300">{source.name}:</span>
                <div className="flex items-center space-x-2">
                  {source.status === 'live' ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : source.status === 'error' ? (
                    <XCircle className="w-4 h-4 text-red-400" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-yellow-400" />
                  )}
                  <span className={`text-sm ${
                    source.status === 'live' ? 'text-green-400' :
                    source.status === 'error' ? 'text-red-400' :
                    'text-yellow-400'
                  }`}>
                    {source.status === 'live' ? 'Live' :
                     source.status === 'error' ? 'Error' : 'Demo'}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {allDemo && (
            <div className="pt-3 border-t border-gray-700">
              <p className="text-gray-400 text-sm mb-2">
                To use real data:
              </p>
              <ol className="text-gray-500 text-xs space-y-1">
                <li>1. Run: <code className="text-yellow-400">./scripts/setup-api-keys.sh</code></li>
                <li>2. Add API keys to <code className="text-yellow-400">.env.local</code></li>
                <li>3. Run: <code className="text-yellow-400">python3 scripts/fetch-real-data.py</code></li>
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  )
}