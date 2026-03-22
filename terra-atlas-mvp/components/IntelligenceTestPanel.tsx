'use client'

import { useState } from 'react'

export default function IntelligenceTestPanel() {
  const [testResults, setTestResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const runTests = async () => {
    setLoading(true)
    const results: any[] = []

    try {
      // Test 1: Fetch intelligence metrics
      const metricsStart = Date.now()
      const metricsRes = await fetch('/api/intelligence/metrics')
      const metricsTime = Date.now() - metricsStart
      const metricsData = await metricsRes.json()
      
      results.push({
        test: 'Intelligence Metrics API',
        status: metricsRes.ok ? 'PASS' : 'FAIL',
        time: `${metricsTime}ms`,
        data: metricsData
      })

      // Test 2: Fetch top projects with scoring
      const projectsStart = Date.now()
      const projectsRes = await fetch('/api/intelligence/top-projects?limit=5')
      const projectsTime = Date.now() - projectsStart
      const projectsData = await projectsRes.json()
      
      results.push({
        test: 'Top Projects API',
        status: projectsRes.ok && projectsData.projects ? 'PASS' : 'FAIL',
        time: `${projectsTime}ms`,
        projectCount: projectsData.projects?.length || 0,
        avgScore: projectsData.projects?.reduce((acc: number, p: any) => acc + p.score, 0) / (projectsData.projects?.length || 1)
      })

      // Test 3: Test filtering by minimum score
      const filteredRes = await fetch('/api/intelligence/top-projects?minScore=85')
      const filteredData = await filteredRes.json()
      
      results.push({
        test: 'Score Filtering',
        status: filteredData.projects?.every((p: any) => p.score >= 85) ? 'PASS' : 'FAIL',
        projectsAbove85: filteredData.projects?.length || 0
      })

      // Test 4: Test type filtering
      const solarRes = await fetch('/api/intelligence/top-projects?type=solar')
      const solarData = await solarRes.json()
      
      results.push({
        test: 'Type Filtering',
        status: solarData.projects?.every((p: any) => p.type === 'solar') ? 'PASS' : 'FAIL',
        solarProjects: solarData.projects?.length || 0
      })

      // Test 5: Performance test - rapid queries
      const perfStart = Date.now()
      const perfPromises = Array(10).fill(0).map(() => 
        fetch('/api/intelligence/top-projects?limit=1')
      )
      await Promise.all(perfPromises)
      const perfTime = Date.now() - perfStart
      
      results.push({
        test: 'Performance (10 concurrent)',
        status: perfTime < 1000 ? 'PASS' : 'FAIL',
        avgTime: `${perfTime/10}ms per request`,
        totalTime: `${perfTime}ms`
      })

    } catch (error) {
      results.push({
        test: 'Error',
        status: 'FAIL',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }

    setTestResults(results)
    setLoading(false)
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-md">
      <div className="bg-black/90 backdrop-blur-md border border-white/20 rounded-2xl p-4">
        <h3 className="text-white font-light mb-3 flex items-center gap-2">
          <span className="text-xl">🧪</span>
          Intelligence Test Panel
        </h3>
        
        <button
          onClick={runTests}
          disabled={loading}
          className="w-full px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 rounded-lg text-emerald-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed mb-3"
        >
          {loading ? 'Running Tests...' : 'Run Intelligence Tests'}
        </button>

        {testResults.length > 0 && (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {testResults.map((result, idx) => (
              <div
                key={idx}
                className={`p-2 rounded-lg text-xs ${
                  result.status === 'PASS' 
                    ? 'bg-green-500/10 border border-green-500/20' 
                    : 'bg-red-500/10 border border-red-500/20'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-white/80">{result.test}</span>
                  <span className={result.status === 'PASS' ? 'text-green-400' : 'text-red-400'}>
                    {result.status}
                  </span>
                </div>
                {Object.entries(result).filter(([key]) => !['test', 'status'].includes(key)).map(([key, value]) => (
                  <div key={key} className="text-white/50 mt-1">
                    {key}: {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}