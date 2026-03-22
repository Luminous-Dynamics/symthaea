'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, Battery, Zap, DollarSign, Globe, Users, Leaf, ChevronRight } from 'lucide-react'

interface Portfolio {
  investments: Investment[]
  totalValue: number
  totalReturns: number
  totalCO2Saved: number
  projectCount: number
}

interface Investment {
  id: string
  projectName: string
  type: string
  amount: number
  currentValue: number
  returns: number
  date: string
  status: 'Active' | 'Matured' | 'Pending'
}

export default function PortfolioDashboard() {
  const [portfolio, setPortfolio] = useState<Portfolio>({
    investments: [],
    totalValue: 0,
    totalReturns: 0,
    totalCO2Saved: 0,
    projectCount: 0
  })
  const [timeframe, setTimeframe] = useState('all')

  useEffect(() => {
    // In production, fetch from API
    // For now, show demo data
    loadDemoPortfolio()
  }, [timeframe])

  const loadDemoPortfolio = () => {
    const demoInvestments: Investment[] = [
      {
        id: '1',
        projectName: 'West Texas Solar Farm',
        type: 'Solar',
        amount: 25000,
        currentValue: 28500,
        returns: 3500,
        date: '2024-01-15',
        status: 'Active'
      },
      {
        id: '2',
        projectName: 'Columbia River Hydro Retrofit',
        type: 'Hydro',
        amount: 10000,
        currentValue: 11200,
        returns: 1200,
        date: '2024-03-22',
        status: 'Active'
      },
      {
        id: '3',
        projectName: 'Wyoming Wind Corridor',
        type: 'Wind',
        amount: 50000,
        currentValue: 54000,
        returns: 4000,
        date: '2024-05-10',
        status: 'Active'
      },
      {
        id: '4',
        projectName: 'Nevada Battery Storage',
        type: 'Battery',
        amount: 15000,
        currentValue: 15900,
        returns: 900,
        date: '2024-07-01',
        status: 'Pending'
      }
    ]

    const totalValue = demoInvestments.reduce((sum, inv) => sum + inv.currentValue, 0)
    const totalReturns = demoInvestments.reduce((sum, inv) => sum + inv.returns, 0)
    const totalCO2Saved = Math.round(totalValue / 1000 * 12.5) // Rough estimate

    setPortfolio({
      investments: demoInvestments,
      totalValue,
      totalReturns,
      totalCO2Saved,
      projectCount: demoInvestments.length
    })
  }

  const returnRate = portfolio.totalValue > 0 
    ? ((portfolio.totalReturns / (portfolio.totalValue - portfolio.totalReturns)) * 100).toFixed(1)
    : '0'

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/10 border border-emerald-500/20 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="h-5 w-5 text-emerald-400" />
            <span className="text-xs text-emerald-400">+{returnRate}%</span>
          </div>
          <p className="text-2xl font-bold text-white">${portfolio.totalValue.toLocaleString()}</p>
          <p className="text-xs text-gray-400 mt-1">Portfolio Value</p>
        </div>

        <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 border border-blue-500/20 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="h-5 w-5 text-blue-400" />
            <span className="text-xs text-green-400">↑</span>
          </div>
          <p className="text-2xl font-bold text-white">+${portfolio.totalReturns.toLocaleString()}</p>
          <p className="text-xs text-gray-400 mt-1">Total Returns</p>
        </div>

        <div className="bg-gradient-to-br from-green-500/10 to-green-600/10 border border-green-500/20 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Leaf className="h-5 w-5 text-green-400" />
            <span className="text-xs text-green-400">tons</span>
          </div>
          <p className="text-2xl font-bold text-white">{portfolio.totalCO2Saved.toLocaleString()}</p>
          <p className="text-xs text-gray-400 mt-1">CO₂ Saved</p>
        </div>

        <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/10 border border-purple-500/20 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Zap className="h-5 w-5 text-purple-400" />
            <span className="text-xs text-purple-400">active</span>
          </div>
          <p className="text-2xl font-bold text-white">{portfolio.projectCount}</p>
          <p className="text-xs text-gray-400 mt-1">Projects</p>
        </div>
      </div>

      {/* Investments List */}
      <div className="bg-gray-900/50 backdrop-blur-md rounded-xl border border-gray-800">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <h3 className="font-bold text-white">Your Investments</h3>
          <div className="flex gap-2">
            {['all', '1m', '3m', '1y'].map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                  timeframe === tf 
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50' 
                    : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
                }`}
              >
                {tf === 'all' ? 'All Time' : tf.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="divide-y divide-gray-800">
          {portfolio.investments.map((investment) => (
            <div key={investment.id} className="p-4 hover:bg-gray-800/30 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${
                      investment.type === 'Solar' ? 'bg-yellow-500/10' :
                      investment.type === 'Wind' ? 'bg-blue-500/10' :
                      investment.type === 'Hydro' ? 'bg-cyan-500/10' :
                      'bg-purple-500/10'
                    }`}>
                      {investment.type === 'Solar' ? <Zap className="h-4 w-4 text-yellow-400" /> :
                       investment.type === 'Wind' ? <Battery className="h-4 w-4 text-blue-400" /> :
                       investment.type === 'Hydro' ? <Globe className="h-4 w-4 text-cyan-400" /> :
                       <Battery className="h-4 w-4 text-purple-400" />}
                    </div>
                    <div>
                      <p className="font-semibold text-white">{investment.projectName}</p>
                      <div className="flex items-center gap-4 mt-1">
                        <span className="text-xs text-gray-400">{investment.type}</span>
                        <span className="text-xs text-gray-400">•</span>
                        <span className="text-xs text-gray-400">Invested {investment.date}</span>
                        <span className="text-xs text-gray-400">•</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          investment.status === 'Active' ? 'bg-green-500/20 text-green-400' :
                          investment.status === 'Matured' ? 'bg-blue-500/20 text-blue-400' :
                          'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {investment.status}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Invested</p>
                    <p className="font-semibold text-white">${investment.amount.toLocaleString()}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Current Value</p>
                    <p className="font-semibold text-white">${investment.currentValue.toLocaleString()}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Returns</p>
                    <p className={`font-semibold ${investment.returns > 0 ? 'text-green-400' : 'text-gray-400'}`}>
                      +${investment.returns.toLocaleString()}
                      <span className="text-xs ml-1">
                        ({((investment.returns / investment.amount) * 100).toFixed(1)}%)
                      </span>
                    </p>
                  </div>
                  <ChevronRight className="h-4 w-4 text-gray-400" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Add New Investment */}
      <button className="w-full bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/30 text-emerald-400 font-semibold py-3 rounded-xl hover:border-emerald-500/50 transition-all flex items-center justify-center gap-2">
        <Zap className="h-4 w-4" />
        Find New Investment Opportunities
      </button>
    </div>
  )
}