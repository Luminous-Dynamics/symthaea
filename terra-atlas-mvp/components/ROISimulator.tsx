'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, Activity, DollarSign, Users, Globe, AlertCircle, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { Line, Area } from 'recharts';
import {
  LineChart,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

interface SimulationParams {
  initialInvestment: number;
  projectType: string;
  baseIRR: number;
  communityTransferYear: number;
  inflationRate: number;
  reinvestmentRate: number;
  carbonPriceGrowth: number;
  riskProfile: 'conservative' | 'moderate' | 'aggressive';
}

interface YearlyProjection {
  year: number;
  value: number;
  distributions: number;
  cumulative: number;
  ownership: number;
  communityValue: number;
  carbonRevenue: number;
  vs_sp500: number;
  vs_bonds: number;
}

interface MonteCarloResult {
  median: number;
  percentile_25: number;
  percentile_75: number;
  percentile_95: number;
  probability_loss: number;
  expected_return: number;
}

export default function ROISimulator() {
  const [params, setParams] = useState<SimulationParams>({
    initialInvestment: 100000,
    projectType: 'solar',
    baseIRR: 12,
    communityTransferYear: 7,
    inflationRate: 2.5,
    reinvestmentRate: 50,
    carbonPriceGrowth: 5,
    riskProfile: 'moderate'
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [projections, setProjections] = useState<YearlyProjection[]>([]);
  const [monteCarloResults, setMonteCarloResults] = useState<MonteCarloResult | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<'value' | 'distributions' | 'comparison'>('value');

  // Risk adjustment factors by profile
  const riskFactors = {
    conservative: { volatility: 0.08, discount: 0.02 },
    moderate: { volatility: 0.15, discount: 0 },
    aggressive: { volatility: 0.25, discount: -0.02 }
  };

  // Project type characteristics
  const projectCharacteristics = {
    solar: { 
      degradation: 0.005, 
      opex_growth: 0.02, 
      revenue_stability: 0.95,
      carbon_intensity: 0.85 
    },
    wind: { 
      degradation: 0.008, 
      opex_growth: 0.025, 
      revenue_stability: 0.90,
      carbon_intensity: 0.95 
    },
    hydro: { 
      degradation: 0.002, 
      opex_growth: 0.015, 
      revenue_stability: 0.98,
      carbon_intensity: 0.90 
    },
    nuclear: { 
      degradation: 0.001, 
      opex_growth: 0.03, 
      revenue_stability: 0.99,
      carbon_intensity: 1.0 
    },
    storage: { 
      degradation: 0.02, 
      opex_growth: 0.02, 
      revenue_stability: 0.85,
      carbon_intensity: 0.5 
    }
  };

  useEffect(() => {
    generateProjections();
    runMonteCarloSimulation();
  }, [params]);

  const generateProjections = () => {
    const years = [];
    const characteristics = projectCharacteristics[params.projectType] || projectCharacteristics.solar;
    const riskAdjustment = riskFactors[params.riskProfile];
    
    let currentValue = params.initialInvestment;
    let cumulative = params.initialInvestment;
    let sp500Value = params.initialInvestment;
    let bondValue = params.initialInvestment;

    for (let year = 0; year <= 20; year++) {
      // Calculate adjusted IRR with degradation and risk
      const degradationFactor = Math.pow(1 - characteristics.degradation, year);
      const adjustedIRR = (params.baseIRR / 100 + riskAdjustment.discount) * degradationFactor;
      
      // Annual distribution
      const distribution = year === 0 ? 0 : currentValue * adjustedIRR;
      
      // Reinvestment logic
      const reinvested = distribution * (params.reinvestmentRate / 100);
      const cashDistribution = distribution - reinvested;
      
      // Update value with reinvestment
      currentValue = currentValue * (1 + adjustedIRR) - cashDistribution;
      cumulative += cashDistribution;

      // Community ownership transfer
      const ownership = year <= params.communityTransferYear ? 100 : Math.max(5, 100 - (year - params.communityTransferYear) * 13.5);
      const communityValue = year > params.communityTransferYear ? currentValue * (1 - ownership / 100) : 0;

      // Carbon credit revenue (growing over time)
      const carbonPrice = 50 * Math.pow(1 + params.carbonPriceGrowth / 100, year);
      const carbonRevenue = (params.initialInvestment / 1000000) * carbonPrice * 1000 * characteristics.carbon_intensity;

      // Comparison investments
      sp500Value *= 1.10; // Historical S&P 500 average
      bondValue *= 1.04; // Treasury bond average

      years.push({
        year,
        value: currentValue,
        distributions: cashDistribution,
        cumulative,
        ownership,
        communityValue,
        carbonRevenue,
        vs_sp500: (currentValue - sp500Value) / sp500Value * 100,
        vs_bonds: (currentValue - bondValue) / bondValue * 100
      });
    }

    setProjections(years);
  };

  const runMonteCarloSimulation = () => {
    const simulations = 10000;
    const results: number[] = [];
    const characteristics = projectCharacteristics[params.projectType] || projectCharacteristics.solar;
    const riskAdjustment = riskFactors[params.riskProfile];
    
    for (let sim = 0; sim < simulations; sim++) {
      let value = params.initialInvestment;
      
      for (let year = 1; year <= 20; year++) {
        // Generate random return with normal distribution
        const randomFactor = gaussianRandom();
        const volatility = riskAdjustment.volatility;
        const expectedReturn = params.baseIRR / 100;
        const actualReturn = expectedReturn + volatility * randomFactor;
        
        value *= (1 + actualReturn);
      }
      
      results.push(value);
    }

    // Sort results for percentile calculation
    results.sort((a, b) => a - b);

    setMonteCarloResults({
      median: results[Math.floor(simulations * 0.5)],
      percentile_25: results[Math.floor(simulations * 0.25)],
      percentile_75: results[Math.floor(simulations * 0.75)],
      percentile_95: results[Math.floor(simulations * 0.95)],
      probability_loss: results.filter(r => r < params.initialInvestment).length / simulations,
      expected_return: results.reduce((a, b) => a + b, 0) / simulations
    });
  };

  // Box-Muller transform for gaussian random numbers
  const gaussianRandom = (): number => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const formatCurrency = (num: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(num);
  };

  const getChartData = () => {
    if (selectedMetric === 'value') {
      return projections.map(p => ({
        year: p.year,
        'Portfolio Value': p.value,
        'Community Value': p.communityValue,
        'Total Value': p.value + p.communityValue
      }));
    } else if (selectedMetric === 'distributions') {
      return projections.map(p => ({
        year: p.year,
        'Annual Distributions': p.distributions,
        'Carbon Revenue': p.carbonRevenue,
        'Cumulative': p.cumulative
      }));
    } else {
      return projections.map(p => ({
        year: p.year,
        'Clean Energy': p.value,
        'S&P 500': params.initialInvestment * Math.pow(1.10, p.year),
        'Bonds': params.initialInvestment * Math.pow(1.04, p.year)
      }));
    }
  };

  const finalProjection = projections[projections.length - 1];
  const totalReturn = finalProjection ? ((finalProjection.value - params.initialInvestment) / params.initialInvestment * 100) : 0;
  const annualizedReturn = finalProjection ? (Math.pow(finalProjection.value / params.initialInvestment, 1/20) - 1) * 100 : 0;

  return (
    <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-8 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">20-Year Investment Projection</h2>
          <p className="text-gray-400">Advanced Monte Carlo simulation with community ownership modeling</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setSelectedMetric('value')}
            className={`px-4 py-2 rounded-lg transition ${
              selectedMetric === 'value' 
                ? 'bg-gradient-to-r from-green-500 to-blue-500 text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Portfolio Value
          </button>
          <button
            onClick={() => setSelectedMetric('distributions')}
            className={`px-4 py-2 rounded-lg transition ${
              selectedMetric === 'distributions'
                ? 'bg-gradient-to-r from-green-500 to-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Cash Flow
          </button>
          <button
            onClick={() => setSelectedMetric('comparison')}
            className={`px-4 py-2 rounded-lg transition ${
              selectedMetric === 'comparison'
                ? 'bg-gradient-to-r from-green-500 to-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Compare
          </button>
        </div>
      </div>

      {/* Input Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 mb-8">
        <div>
          <label className="block text-gray-400 text-sm mb-2">Initial Investment</label>
          <div className="relative">
            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400">$</span>
            <input
              type="number"
              value={params.initialInvestment}
              onChange={(e) => setParams({...params, initialInvestment: Math.max(1000, parseInt(e.target.value) || 0)})}
              className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-green-500"
              min="1000"
              step="10000"
            />
          </div>
        </div>

        <div>
          <label className="block text-gray-400 text-sm mb-2">Project Type</label>
          <select
            value={params.projectType}
            onChange={(e) => setParams({...params, projectType: e.target.value})}
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-green-500"
          >
            <option value="solar">Solar</option>
            <option value="wind">Wind</option>
            <option value="hydro">Hydro</option>
            <option value="nuclear">Nuclear SMR</option>
            <option value="storage">Battery Storage</option>
          </select>
        </div>

        <div>
          <label className="block text-gray-400 text-sm mb-2">Risk Profile</label>
          <select
            value={params.riskProfile}
            onChange={(e) => setParams({...params, riskProfile: e.target.value as any})}
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-green-500"
          >
            <option value="conservative">Conservative (8% volatility)</option>
            <option value="moderate">Moderate (15% volatility)</option>
            <option value="aggressive">Aggressive (25% volatility)</option>
          </select>
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="mb-8">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-gray-400 hover:text-gray-300 mb-4"
        >
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          Advanced Parameters
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-gray-800/50 rounded-lg">
            <div>
              <label className="block text-gray-400 text-xs mb-1">Base IRR: {params.baseIRR}%</label>
              <input
                type="range"
                value={params.baseIRR}
                onChange={(e) => setParams({...params, baseIRR: parseInt(e.target.value)})}
                className="w-full"
                min="8"
                max="20"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-xs mb-1">Community Transfer Year: {params.communityTransferYear}</label>
              <input
                type="range"
                value={params.communityTransferYear}
                onChange={(e) => setParams({...params, communityTransferYear: parseInt(e.target.value)})}
                className="w-full"
                min="5"
                max="15"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-xs mb-1">Reinvestment Rate: {params.reinvestmentRate}%</label>
              <input
                type="range"
                value={params.reinvestmentRate}
                onChange={(e) => setParams({...params, reinvestmentRate: parseInt(e.target.value)})}
                className="w-full"
                min="0"
                max="100"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-xs mb-1">Carbon Price Growth: {params.carbonPriceGrowth}%</label>
              <input
                type="range"
                value={params.carbonPriceGrowth}
                onChange={(e) => setParams({...params, carbonPriceGrowth: parseInt(e.target.value)})}
                className="w-full"
                min="0"
                max="15"
              />
            </div>
          </div>
        )}
      </div>

      {/* Monte Carlo Results */}
      {monteCarloResults && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4 mb-8">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Expected Value</p>
            <p className="text-2xl font-bold text-green-400">{formatCurrency(monteCarloResults.expected_return)}</p>
            <p className="text-xs text-gray-500 mt-1">{((monteCarloResults.expected_return / params.initialInvestment - 1) * 100).toFixed(1)}% return</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Median Outcome</p>
            <p className="text-2xl font-bold text-blue-400">{formatCurrency(monteCarloResults.median)}</p>
            <p className="text-xs text-gray-500 mt-1">50% probability</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Conservative (25th)</p>
            <p className="text-2xl font-bold text-yellow-400">{formatCurrency(monteCarloResults.percentile_25)}</p>
            <p className="text-xs text-gray-500 mt-1">75% exceed this</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Optimistic (75th)</p>
            <p className="text-2xl font-bold text-purple-400">{formatCurrency(monteCarloResults.percentile_75)}</p>
            <p className="text-xs text-gray-500 mt-1">25% exceed this</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Risk of Loss</p>
            <p className="text-2xl font-bold text-red-400">{(monteCarloResults.probability_loss * 100).toFixed(1)}%</p>
            <p className="text-xs text-gray-500 mt-1">After 20 years</p>
          </div>
        </div>
      )}

      {/* Main Chart */}
      <div className="bg-gray-800/30 rounded-xl p-6 mb-8">
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={getChartData()}>
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
              </linearGradient>
              <linearGradient id="colorCommunity" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
              </linearGradient>
              <linearGradient id="colorSP500" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="year" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" tickFormatter={(value) => `$${(value/1000).toFixed(0)}K`} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value: number) => formatCurrency(value)}
            />
            <Legend />
            {selectedMetric === 'value' && (
              <>
                <Area type="monotone" dataKey="Portfolio Value" stroke="#10b981" fillOpacity={1} fill="url(#colorValue)" />
                <Area type="monotone" dataKey="Community Value" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorCommunity)" />
                <ReferenceLine x={params.communityTransferYear} stroke="#f59e0b" strokeDasharray="3 3" />
              </>
            )}
            {selectedMetric === 'distributions' && (
              <>
                <Area type="monotone" dataKey="Annual Distributions" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                <Area type="monotone" dataKey="Carbon Revenue" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
              </>
            )}
            {selectedMetric === 'comparison' && (
              <>
                <Area type="monotone" dataKey="Clean Energy" stroke="#10b981" fillOpacity={1} fill="url(#colorValue)" />
                <Area type="monotone" dataKey="S&P 500" stroke="#ef4444" fillOpacity={1} fill="url(#colorSP500)" />
                <Area type="monotone" dataKey="Bonds" stroke="#6b7280" fill="#6b7280" fillOpacity={0.3} />
              </>
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Key Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-gradient-to-r from-green-600/20 to-blue-600/20 rounded-xl p-6 border border-green-500/30">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Financial Performance
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Return</span>
              <span className="text-green-400 font-bold">{totalReturn.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Annualized Return</span>
              <span className="text-blue-400 font-bold">{annualizedReturn.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Final Value</span>
              <span className="text-white font-bold">{formatCurrency(finalProjection?.value || 0)}</span>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-xl p-6 border border-purple-500/30">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <Users className="w-5 h-5 text-purple-400" />
            Community Impact
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Transfer Year</span>
              <span className="text-purple-400 font-bold">Year {params.communityTransferYear}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Community Value</span>
              <span className="text-pink-400 font-bold">{formatCurrency(finalProjection?.communityValue || 0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Final Ownership</span>
              <span className="text-white font-bold">{(100 - (finalProjection?.ownership || 100)).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-yellow-600/20 to-orange-600/20 rounded-xl p-6 border border-yellow-500/30">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-yellow-400" />
            Risk Analysis
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Risk Profile</span>
              <span className="text-yellow-400 font-bold capitalize">{params.riskProfile}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Loss Probability</span>
              <span className={`font-bold ${(monteCarloResults?.probability_loss || 0) > 0.1 ? 'text-red-400' : 'text-green-400'}`}>
                {((monteCarloResults?.probability_loss || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Confidence Range</span>
              <span className="text-white font-bold text-sm">
                {formatCurrency(monteCarloResults?.percentile_25 || 0)} - {formatCurrency(monteCarloResults?.percentile_75 || 0)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-8 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-gray-400">
            <p className="font-semibold text-yellow-400 mb-1">Demo Mode - Educational Projections Only</p>
            <p>
              These projections are for educational purposes only and do not constitute investment advice. 
              Actual returns will vary based on project performance, market conditions, and regulatory changes. 
              Past performance does not guarantee future results. Community ownership transfer timelines and 
              percentages are illustrative goals subject to legal and regulatory approval.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}