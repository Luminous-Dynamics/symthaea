'use client'

import { useState, useEffect } from 'react'
import { DollarSign, TrendingUp, Users, Leaf, Settings, Info, Calculator, Sparkles } from 'lucide-react'

interface CalculatorProps {
  project?: {
    id: number
    name: string
    capacity_mw: number
    type: string
    min_investment?: number
    status?: string
    location?: string
  }
}

interface AdvancedSettings {
  inflationRate: number
  taxRate: number
  reinvestDividends: boolean
  includeIncentives: boolean
}

export default function InvestmentCalculator({ project }: CalculatorProps) {
  const [investmentAmount, setInvestmentAmount] = useState(10000)
  const [years, setYears] = useState(7)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [advancedSettings, setAdvancedSettings] = useState<AdvancedSettings>({
    inflationRate: 2.5,
    taxRate: 15,
    reinvestDividends: false,
    includeIncentives: true
  })
  const [metrics, setMetrics] = useState({
    expectedReturn: 0,
    annualDividend: 0,
    co2Saved: 0,
    homesPowered: 0,
    communityShare: 0,
    communityBonus: 0,
    irr: 0,
    effectiveIRR: 0,
    totalValue: 0,
    monthlyIncome: 0
  })

  // Project-specific IRR rates based on type
  const getProjectIRR = (type: string | undefined, status: string | undefined) => {
    const baseRates: Record<string, number> = {
      'Solar': 0.125,      // 12.5%
      'Wind': 0.135,       // 13.5%
      'Battery': 0.15,     // 15%
      'Hydro': 0.11,       // 11%
      'Nuclear': 0.14,     // 14%
      'Geothermal': 0.13,  // 13%
    }
    
    let rate = baseRates[type || 'Solar'] || 0.12
    
    // Adjust for project status
    if (status === 'Operational') rate *= 0.95  // Lower risk, lower return
    if (status === 'Construction') rate *= 1.05  // Higher risk, higher return
    if (status === 'Development') rate *= 1.1    // Highest risk, highest return
    
    return rate
  }

  useEffect(() => {
    calculateReturns()
  }, [investmentAmount, years, advancedSettings, project])

  const calculateReturns = () => {
    // Get project-specific IRR
    const baseIRR = getProjectIRR(project?.type, project?.status)
    
    // Adjust for inflation
    const realIRR = baseIRR - (advancedSettings.inflationRate / 100)
    
    // Calculate returns with potential dividend reinvestment
    let totalReturn = 0
    let currentPrincipal = investmentAmount
    
    for (let year = 1; year <= years; year++) {
      const yearReturn = currentPrincipal * baseIRR
      
      if (advancedSettings.reinvestDividends) {
        currentPrincipal += yearReturn
        totalReturn = currentPrincipal - investmentAmount
      } else {
        totalReturn += yearReturn
      }
    }
    
    // Apply tax considerations
    const afterTaxReturn = totalReturn * (1 - advancedSettings.taxRate / 100)
    
    // Government incentives (ITC, PTC, etc.)
    const incentiveBonus = advancedSettings.includeIncentives ? investmentAmount * 0.26 : 0 // 26% ITC
    
    // Community transfer calculations
    const communityShareBase = years >= 7 ? investmentAmount * 0.1 : 0
    const communityBonus = years >= 7 ? totalReturn * 0.05 : 0 // Extra 5% bonus at transfer
    
    // Environmental calculations
    const capacityShare = investmentAmount / 1000000 // $1M per MW
    const mwShare = capacityShare * (project?.capacity_mw || 100)
    
    // More accurate environmental impact by project type
    const co2Factors: Record<string, number> = {
      'Solar': 450,
      'Wind': 550,
      'Battery': 300, // Enables more renewables
      'Hydro': 480,
      'Nuclear': 800,
      'Geothermal': 400
    }
    const co2PerMW = co2Factors[project?.type || 'Solar'] || 500
    
    const homesFactors: Record<string, number> = {
      'Solar': 600,
      'Wind': 900,
      'Battery': 500,
      'Hydro': 1200,
      'Nuclear': 2000,
      'Geothermal': 800
    }
    const homesPerMW = homesFactors[project?.type || 'Solar'] || 750
    
    // Calculate effective IRR including all factors
    const totalValue = investmentAmount + afterTaxReturn + incentiveBonus + communityBonus
    const effectiveIRR = Math.pow(totalValue / investmentAmount, 1 / years) - 1
    
    setMetrics({
      expectedReturn: afterTaxReturn,
      annualDividend: totalReturn / years,
      co2Saved: Math.round(mwShare * co2PerMW * years),
      homesPowered: Math.round(mwShare * homesPerMW),
      communityShare: communityShareBase,
      communityBonus: Math.round(communityBonus),
      irr: baseIRR * 100,
      effectiveIRR: effectiveIRR * 100,
      totalValue: Math.round(totalValue),
      monthlyIncome: Math.round((totalReturn / years) / 12)
    })
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-emerald-500/20">
      <h3 className="text-xl font-bold mb-6 text-emerald-400">
        Investment Calculator
        {project && <span className="text-sm text-gray-400 ml-2">for {project.name}</span>}
      </h3>

      <div className="space-y-6">
        {/* Investment Amount */}
        <div>
          <label className="block text-sm text-gray-400 mb-2">Investment Amount</label>
          <div className="relative">
            <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-emerald-400" />
            <input
              type="number"
              value={investmentAmount}
              onChange={(e) => setInvestmentAmount(Number(e.target.value))}
              min={project?.min_investment || 100}
              step={100}
              className="w-full bg-gray-800/50 border border-gray-700 rounded-lg py-3 pl-10 pr-4 text-white focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <div className="flex gap-2 mt-2">
            {[1000, 5000, 10000, 50000].map(amount => (
              <button
                key={amount}
                onClick={() => setInvestmentAmount(amount)}
                className="px-3 py-1 text-xs bg-gray-800/50 hover:bg-emerald-500/20 border border-gray-700 hover:border-emerald-500/50 rounded transition-colors"
              >
                ${amount.toLocaleString()}
              </button>
            ))}
          </div>
        </div>

        {/* Investment Period */}
        <div>
          <label className="block text-sm text-gray-400 mb-2">Investment Period</label>
          <input
            type="range"
            value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            min={1}
            max={10}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1 year</span>
            <span className="text-emerald-400 font-bold">{years} years</span>
            <span>10 years</span>
          </div>
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
          <div className="bg-emerald-500/10 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              <span className="text-xs text-gray-400">Expected Return</span>
            </div>
            <p className="text-2xl font-bold text-emerald-400">
              ${metrics.expectedReturn.toLocaleString()}
            </p>
            <p className="text-xs text-gray-500 mt-1">{metrics.irr.toFixed(1)}% IRR</p>
          </div>

          <div className="bg-blue-500/10 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <Leaf className="h-4 w-4 text-blue-400" />
              <span className="text-xs text-gray-400">CO₂ Saved</span>
            </div>
            <p className="text-2xl font-bold text-blue-400">
              {metrics.co2Saved.toLocaleString()}
            </p>
            <p className="text-xs text-gray-500 mt-1">tons</p>
          </div>

          <div className="bg-purple-500/10 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <Users className="h-4 w-4 text-purple-400" />
              <span className="text-xs text-gray-400">Homes Powered</span>
            </div>
            <p className="text-2xl font-bold text-purple-400">
              {metrics.homesPowered.toLocaleString()}
            </p>
            <p className="text-xs text-gray-500 mt-1">households</p>
          </div>

          <div className="bg-orange-500/10 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <Users className="h-4 w-4 text-orange-400" />
              <span className="text-xs text-gray-400">Community Share</span>
            </div>
            <p className="text-2xl font-bold text-orange-400">
              ${metrics.communityShare.toLocaleString()}
            </p>
            <p className="text-xs text-gray-500 mt-1">after {years}y</p>
          </div>
        </div>

        {/* Advanced Settings Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-4 py-2 bg-gray-800/30 border border-gray-700 rounded-lg hover:border-emerald-500/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-400">Advanced Settings</span>
          </div>
          <svg
            className={`h-4 w-4 text-gray-400 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Advanced Settings Panel */}
        {showAdvanced && (
          <div className="space-y-4 p-4 bg-gray-800/20 rounded-lg border border-gray-700">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Inflation Rate (%)</label>
                <input
                  type="number"
                  value={advancedSettings.inflationRate}
                  onChange={(e) => setAdvancedSettings(prev => ({ ...prev, inflationRate: Number(e.target.value) }))}
                  min={0}
                  max={10}
                  step={0.1}
                  className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded text-white text-sm focus:border-emerald-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Tax Rate (%)</label>
                <input
                  type="number"
                  value={advancedSettings.taxRate}
                  onChange={(e) => setAdvancedSettings(prev => ({ ...prev, taxRate: Number(e.target.value) }))}
                  min={0}
                  max={40}
                  step={1}
                  className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded text-white text-sm focus:border-emerald-500 focus:outline-none"
                />
              </div>
            </div>
            
            <div className="space-y-3">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={advancedSettings.reinvestDividends}
                  onChange={(e) => setAdvancedSettings(prev => ({ ...prev, reinvestDividends: e.target.checked }))}
                  className="w-4 h-4 rounded border-gray-700 bg-gray-800 text-emerald-500 focus:ring-emerald-500"
                />
                <div>
                  <span className="text-sm text-white">Reinvest Dividends</span>
                  <p className="text-xs text-gray-500">Compound returns by reinvesting annual dividends</p>
                </div>
              </label>
              
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={advancedSettings.includeIncentives}
                  onChange={(e) => setAdvancedSettings(prev => ({ ...prev, includeIncentives: e.target.checked }))}
                  className="w-4 h-4 rounded border-gray-700 bg-gray-800 text-emerald-500 focus:ring-emerald-500"
                />
                <div>
                  <span className="text-sm text-white">Include Government Incentives</span>
                  <p className="text-xs text-gray-500">26% Investment Tax Credit (ITC)</p>
                </div>
              </label>
            </div>

            {/* Additional Metrics */}
            <div className="pt-4 border-t border-gray-700">
              <h4 className="text-sm font-semibold text-gray-400 mb-3">Detailed Projections</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="text-xs">
                  <span className="text-gray-500">Effective IRR:</span>
                  <span className="ml-2 font-bold text-emerald-400">{metrics.effectiveIRR.toFixed(2)}%</span>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500">Total Value:</span>
                  <span className="ml-2 font-bold text-cyan-400">${metrics.totalValue.toLocaleString()}</span>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500">Monthly Income:</span>
                  <span className="ml-2 font-bold text-purple-400">${metrics.monthlyIncome.toLocaleString()}</span>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500">Community Bonus:</span>
                  <span className="ml-2 font-bold text-orange-400">${metrics.communityBonus.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Investment Button */}
        <button className="w-full bg-gradient-to-r from-emerald-500 to-cyan-500 text-white font-bold py-3 rounded-lg hover:shadow-lg hover:shadow-emerald-500/20 transition-all">
          Invest ${investmentAmount.toLocaleString()}
        </button>

        {/* Enhanced Info Section */}
        <div className="space-y-3">
          <div className="flex items-start gap-2 p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <Info className="h-4 w-4 text-emerald-400 mt-0.5" />
            <div className="text-xs text-gray-400">
              <span className="font-semibold text-emerald-400">Community Transfer™:</span> After 7 years, projects transition to local community ownership while you retain returns.
            </div>
          </div>
          
          {advancedSettings.includeIncentives && (
            <div className="flex items-start gap-2 p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
              <Sparkles className="h-4 w-4 text-purple-400 mt-0.5" />
              <div className="text-xs text-gray-400">
                <span className="font-semibold text-purple-400">Tax Benefits:</span> You'll receive ${Math.round(investmentAmount * 0.26).toLocaleString()} in tax credits through the Investment Tax Credit program.
              </div>
            </div>
          )}
          
          <p className="text-xs text-gray-500 text-center">
            Projected returns based on historical performance. Not financial advice.
          </p>
        </div>
      </div>
    </div>
  )
}