'use client';

import { useState, useEffect } from 'react';
import { Calculator, Leaf, Home, Car, Trees, DollarSign, TrendingUp, Users, Zap } from 'lucide-react';
import { useAnalytics } from '@/lib/analytics';

interface CalculatorProps {
  projectType?: string;
  capacityMW?: number;
}

export default function ImpactCalculator({ projectType = 'solar', capacityMW = 100 }: CalculatorProps) {
  const analytics = useAnalytics();
  const [investment, setInvestment] = useState(10000);
  const [years, setYears] = useState(20);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [hasCalculated, setHasCalculated] = useState(false);
  
  // Advanced inputs
  const [irr, setIrr] = useState(12);
  const [communityTransferYear, setCommunityTransferYear] = useState(7);
  const [carbonPrice, setCarbonPrice] = useState(50);

  // Calculated values
  const [impact, setImpact] = useState({
    totalReturn: 0,
    annualReturn: 0,
    co2Avoided: 0,
    homesPowered: 0,
    carsOffRoad: 0,
    treesEquivalent: 0,
    jobsCreated: 0,
    communityValue: 0,
    energyGenerated: 0
  });

  useEffect(() => {
    calculateImpact();
  }, [investment, years, irr, projectType, capacityMW]);

  const calculateImpact = () => {
    // Financial calculations
    const totalReturn = investment * Math.pow(1 + irr / 100, years);
    const annualReturn = (totalReturn - investment) / years;

    // Environmental impact (based on investment proportion)
    const projectValue = capacityMW * 1.2 * 1000000; // $1.2M per MW average
    const ownershipPercent = investment / projectValue;
    const personalCapacity = capacityMW * ownershipPercent;

    // CO2 calculations (varies by project type)
    const co2FactorsByType: Record<string, number> = {
      solar: 0.85,
      wind: 0.95,
      hydro: 0.90,
      nuclear: 1.0,
      storage: 0.5
    };
    const co2Factor = co2FactorsByType[projectType] || 0.85;
    const annualCO2Avoided = personalCapacity * 5000 * co2Factor; // tons per MW per year
    const totalCO2Avoided = annualCO2Avoided * years;

    // Other environmental metrics
    const homesPowered = Math.round(personalCapacity * 750); // homes per MW
    const carsOffRoad = Math.round(annualCO2Avoided / 4.6); // avg car emits 4.6 tons/year
    const treesEquivalent = Math.round(annualCO2Avoided * 50); // 1 ton CO2 = 50 trees
    
    // Social impact
    const jobsCreated = personalCapacity * 3; // 3 jobs per MW average
    
    // Community value after transfer
    const communityValue = years > communityTransferYear 
      ? (totalReturn * 0.95) // 95% goes to community eventually
      : 0;

    // Energy generated (MWh)
    const capacityFactor = projectType === 'solar' ? 0.25 : projectType === 'wind' ? 0.35 : 0.5;
    const annualEnergyGenerated = personalCapacity * 8760 * capacityFactor; // MWh per year
    const totalEnergyGenerated = annualEnergyGenerated * years;

    setImpact({
      totalReturn,
      annualReturn,
      co2Avoided: totalCO2Avoided,
      homesPowered,
      carsOffRoad,
      treesEquivalent,
      jobsCreated,
      communityValue,
      energyGenerated: totalEnergyGenerated
    });

    // Track calculator usage (only once per session to avoid spam)
    if (!hasCalculated && analytics) {
      analytics.trackCalculatorUsage(
        'impact_calculator',
        {
          investment,
          years,
          irr,
          projectType,
          capacityMW,
          showAdvanced
        },
        {
          totalReturn: totalReturn.toFixed(0),
          annualReturn: annualReturn.toFixed(0),
          co2Avoided: totalCO2Avoided.toFixed(0),
          homesPowered: homesPowered.toFixed(0)
        }
      );
      setHasCalculated(true);
    }
  };

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toFixed(0);
  };

  const formatCurrency = (num: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(num);
  };

  return (
    <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-8 border border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-gradient-to-r from-green-500 to-blue-500 rounded-xl">
          <Calculator className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-white">Impact Calculator</h2>
          <p className="text-gray-400">See your personal impact from investing</p>
        </div>
      </div>

      {/* Input Section */}
      <div className="space-y-6 mb-8">
        <div>
          <label className="block text-gray-400 text-sm mb-2">Investment Amount</label>
          <div className="relative">
            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400">$</span>
            <input
              type="number"
              value={investment}
              onChange={(e) => setInvestment(Math.max(10, parseInt(e.target.value) || 0))}
              className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-green-500"
              min="10"
              step="1000"
            />
          </div>
          <div className="flex gap-2 mt-2">
            {[1000, 10000, 50000, 100000].map(amount => (
              <button
                key={amount}
                onClick={() => setInvestment(amount)}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition"
              >
                {formatCurrency(amount)}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-gray-400 text-sm mb-2">
            Investment Period: {years} years
          </label>
          <input
            type="range"
            value={years}
            onChange={(e) => setYears(parseInt(e.target.value))}
            className="w-full"
            min="1"
            max="30"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1 year</span>
            <span>15 years</span>
            <span>30 years</span>
          </div>
        </div>

        {showAdvanced && (
          <div className="space-y-4 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-white font-semibold">Advanced Settings</h3>
            <div>
              <label className="block text-gray-400 text-sm mb-1">Expected IRR: {irr}%</label>
              <input
                type="range"
                value={irr}
                onChange={(e) => setIrr(parseInt(e.target.value))}
                className="w-full"
                min="8"
                max="20"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-1">
                Community Transfer Year: {communityTransferYear}
              </label>
              <input
                type="range"
                value={communityTransferYear}
                onChange={(e) => setCommunityTransferYear(parseInt(e.target.value))}
                className="w-full"
                min="5"
                max="20"
              />
            </div>
          </div>
        )}

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-gray-400 hover:text-gray-300"
        >
          {showAdvanced ? '− Hide' : '+ Show'} Advanced Settings
        </button>
      </div>

      {/* Results Section */}
      <div className="space-y-6">
        {/* Financial Impact */}
        <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-xl p-6 border border-green-500/30">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-green-400" />
            Financial Returns
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-400 text-sm">Total Value</p>
              <p className="text-2xl font-bold text-green-400">{formatCurrency(impact.totalReturn)}</p>
              <p className="text-xs text-gray-500 mt-1">
                {((impact.totalReturn / investment - 1) * 100).toFixed(1)}% total return
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-sm">Annual Return</p>
              <p className="text-2xl font-bold text-blue-400">{formatCurrency(impact.annualReturn)}</p>
              <p className="text-xs text-gray-500 mt-1">{irr}% IRR</p>
            </div>
          </div>
          {impact.communityValue > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-700">
              <p className="text-gray-400 text-sm">Community Value (after year {communityTransferYear})</p>
              <p className="text-xl font-bold text-purple-400">{formatCurrency(impact.communityValue)}</p>
              <p className="text-xs text-gray-500 mt-1">95% ownership transfers to local community</p>
            </div>
          )}
        </div>

        {/* Environmental Impact */}
        <div className="bg-gradient-to-r from-green-600/20 to-teal-600/20 rounded-xl p-6 border border-green-500/30">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Leaf className="w-5 h-5 text-green-400" />
            Environmental Impact
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="bg-gray-800 rounded-lg p-3 mb-2">
                <Leaf className="w-8 h-8 text-green-400 mx-auto" />
              </div>
              <p className="text-xl font-bold text-white">{formatNumber(impact.co2Avoided)}</p>
              <p className="text-xs text-gray-400">Tons CO₂ Avoided</p>
            </div>
            <div className="text-center">
              <div className="bg-gray-800 rounded-lg p-3 mb-2">
                <Home className="w-8 h-8 text-blue-400 mx-auto" />
              </div>
              <p className="text-xl font-bold text-white">{formatNumber(impact.homesPowered)}</p>
              <p className="text-xs text-gray-400">Homes Powered</p>
            </div>
            <div className="text-center">
              <div className="bg-gray-800 rounded-lg p-3 mb-2">
                <Car className="w-8 h-8 text-yellow-400 mx-auto" />
              </div>
              <p className="text-xl font-bold text-white">{formatNumber(impact.carsOffRoad)}</p>
              <p className="text-xs text-gray-400">Cars Off Road</p>
            </div>
            <div className="text-center">
              <div className="bg-gray-800 rounded-lg p-3 mb-2">
                <Trees className="w-8 h-8 text-green-400 mx-auto" />
              </div>
              <p className="text-xl font-bold text-white">{formatNumber(impact.treesEquivalent)}</p>
              <p className="text-xs text-gray-400">Trees Equivalent</p>
            </div>
          </div>
        </div>

        {/* Social Impact */}
        <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-xl p-6 border border-purple-500/30">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-purple-400" />
            Social Impact
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-400 text-sm">Jobs Supported</p>
              <p className="text-2xl font-bold text-purple-400">{impact.jobsCreated.toFixed(1)}</p>
              <p className="text-xs text-gray-500 mt-1">Direct & indirect employment</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm">Energy Generated</p>
              <p className="text-2xl font-bold text-blue-400">{formatNumber(impact.energyGenerated)}</p>
              <p className="text-xs text-gray-500 mt-1">MWh over {years} years</p>
            </div>
          </div>
        </div>

        {/* Comparison Text */}
        <div className="bg-gray-800/50 rounded-lg p-4 text-center">
          <p className="text-gray-300">
            Your {formatCurrency(investment)} investment would generate returns equivalent to{' '}
            <span className="text-green-400 font-semibold">
              {((impact.totalReturn - investment) / (investment * 0.07)).toFixed(1)}x
            </span>{' '}
            a traditional 7% index fund, while creating{' '}
            <span className="text-blue-400 font-semibold">
              {formatNumber(impact.co2Avoided)} tons
            </span>{' '}
            of CO₂ reduction and supporting{' '}
            <span className="text-purple-400 font-semibold">
              {impact.jobsCreated.toFixed(1)} jobs
            </span>.
          </p>
        </div>
      </div>
    </div>
  );
}