'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Calculator, TrendingUp, Users, Building2, ChevronRight, Info, DollarSign, PiggyBank, Sparkles } from 'lucide-react';

interface TaxComparison {
  traditional: {
    gross: number;
    corporateTax: number;
    investorDistributions: number;
    operationalCosts: number;
    toCommunity: number;
    effectiveRate: number;
  };
  chimera: {
    gross: number;
    foundationTax: number;
    investorDistributions: number;
    operationalCosts: number;
    toCommunity: number;
    effectiveRate: number;
  };
  difference: {
    absolute: number;
    percentage: number;
    multiplier: number;
  };
}

export default function TaxEfficiencyCalculator() {
  const [investmentAmount, setInvestmentAmount] = useState(10000000); // $10M default
  const [selectedCountry, setSelectedCountry] = useState<'us' | 'eu' | 'global'>('us');
  const [comparison, setComparison] = useState<TaxComparison | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [animatedValues, setAnimatedValues] = useState({
    traditional: 0,
    chimera: 0,
    savings: 0
  });

  // Tax rates by jurisdiction
  const taxRates = {
    us: {
      corporate: 0.21, // 21% federal
      stateAvg: 0.05, // 5% state average
      foundation: 0.025, // 2.5% Swiss foundation
      investorRate: 0.15 // Capital gains
    },
    eu: {
      corporate: 0.23, // 23% average EU
      stateAvg: 0.02, // Additional local
      foundation: 0.025, // Swiss foundation
      investorRate: 0.20 // Capital gains avg
    },
    global: {
      corporate: 0.25, // Global average
      stateAvg: 0.03, // Additional taxes
      foundation: 0.025, // Swiss foundation
      investorRate: 0.18 // Blended rate
    }
  };

  useEffect(() => {
    calculateComparison();
  }, [investmentAmount, selectedCountry]);

  useEffect(() => {
    if (comparison) {
      // Animate the values
      const duration = 1500;
      const steps = 60;
      const interval = duration / steps;
      let step = 0;

      const timer = setInterval(() => {
        step++;
        const progress = step / steps;
        const easing = 1 - Math.pow(1 - progress, 3); // Cubic ease-out

        setAnimatedValues({
          traditional: comparison.traditional.toCommunity * easing,
          chimera: comparison.chimera.toCommunity * easing,
          savings: comparison.difference.absolute * easing
        });

        if (step >= steps) {
          clearInterval(timer);
        }
      }, interval);

      return () => clearInterval(timer);
    }
  }, [comparison]);

  const calculateComparison = () => {
    const rates = taxRates[selectedCountry];
    const operationalRate = 0.15; // 15% operational costs
    const investorShareTraditional = 0.35; // 35% to investors in traditional
    const investorShareChimera = 0.25; // 25% to investors in Chimera (community-aligned)

    // Traditional C-Corp structure
    const traditionalGross = investmentAmount;
    const traditionalCorpTax = traditionalGross * (rates.corporate + rates.stateAvg);
    const traditionalAfterTax = traditionalGross - traditionalCorpTax;
    const traditionalOps = traditionalGross * operationalRate;
    const traditionalInvestors = traditionalAfterTax * investorShareTraditional;
    const traditionalToCommunity = traditionalAfterTax - traditionalOps - traditionalInvestors;

    // Luminous Chimera Model
    const chimeraGross = investmentAmount;
    const chimeraFoundationTax = chimeraGross * rates.foundation;
    const chimeraAfterTax = chimeraGross - chimeraFoundationTax;
    const chimeraOps = chimeraGross * (operationalRate * 0.85); // 15% more efficient
    const chimeraInvestors = chimeraAfterTax * investorShareChimera;
    const chimeraToCommunity = chimeraAfterTax - chimeraOps - chimeraInvestors;

    // Calculate difference
    const difference = chimeraToCommunity - traditionalToCommunity;
    const percentageIncrease = (difference / traditionalToCommunity) * 100;
    const multiplier = chimeraToCommunity / traditionalToCommunity;

    setComparison({
      traditional: {
        gross: traditionalGross,
        corporateTax: traditionalCorpTax,
        investorDistributions: traditionalInvestors,
        operationalCosts: traditionalOps,
        toCommunity: traditionalToCommunity,
        effectiveRate: (traditionalCorpTax / traditionalGross) * 100
      },
      chimera: {
        gross: chimeraGross,
        foundationTax: chimeraFoundationTax,
        investorDistributions: chimeraInvestors,
        operationalCosts: chimeraOps,
        toCommunity: chimeraToCommunity,
        effectiveRate: (chimeraFoundationTax / chimeraGross) * 100
      },
      difference: {
        absolute: difference,
        percentage: percentageIncrease,
        multiplier: multiplier
      }
    });
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-white/95 to-white/90 backdrop-blur-xl rounded-3xl border border-white/20 shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-600 to-teal-600 p-8 text-white">
          <div className="flex items-center gap-3 mb-4">
            <Calculator className="w-8 h-8" />
            <h2 className="text-3xl font-bold">Tax Efficiency Calculator</h2>
          </div>
          <p className="text-emerald-50 text-lg">
            See how the Luminous Chimera Model delivers 30-40% more capital to communities
          </p>
        </div>

        {/* Input Controls */}
        <div className="p-8 space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Investment Amount
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1000000"
                max="100000000"
                step="1000000"
                value={investmentAmount}
                onChange={(e) => setInvestmentAmount(Number(e.target.value))}
                className="flex-1 accent-emerald-600"
              />
              <div className="min-w-[150px] text-right">
                <span className="text-2xl font-bold text-emerald-600">
                  {formatCurrency(investmentAmount)}
                </span>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Tax Jurisdiction
            </label>
            <div className="flex gap-3">
              {[
                { id: 'us', label: 'United States', flag: '🇺🇸' },
                { id: 'eu', label: 'European Union', flag: '🇪🇺' },
                { id: 'global', label: 'Global Average', flag: '🌍' }
              ].map((jurisdiction) => (
                <button
                  key={jurisdiction.id}
                  onClick={() => setSelectedCountry(jurisdiction.id as 'us' | 'eu' | 'global')}
                  className={`
                    flex-1 px-4 py-3 rounded-xl border-2 transition-all
                    ${selectedCountry === jurisdiction.id
                      ? 'border-emerald-500 bg-emerald-50 text-emerald-700'
                      : 'border-gray-200 hover:border-gray-300 text-gray-600'
                    }
                  `}
                >
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-xl">{jurisdiction.flag}</span>
                    <span className="font-semibold">{jurisdiction.label}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Comparison Display */}
        {comparison && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="px-8 pb-8"
          >
            {/* Main Comparison Cards */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {/* Traditional Model */}
              <motion.div
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-50 rounded-2xl p-6 border border-gray-200"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-800">Traditional C-Corp</h3>
                  <Building2 className="w-6 h-6 text-gray-400" />
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Gross Investment</span>
                    <span className="font-medium">{formatCurrency(comparison.traditional.gross)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Corporate Tax ({formatPercentage(comparison.traditional.effectiveRate)})</span>
                    <span className="font-medium text-red-600">-{formatCurrency(comparison.traditional.corporateTax)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Operational Costs</span>
                    <span className="font-medium text-orange-600">-{formatCurrency(comparison.traditional.operationalCosts)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Investor Returns</span>
                    <span className="font-medium text-blue-600">-{formatCurrency(comparison.traditional.investorDistributions)}</span>
                  </div>
                  <div className="pt-3 border-t border-gray-300">
                    <div className="flex justify-between">
                      <span className="font-semibold text-gray-800">To Community</span>
                      <span className="text-xl font-bold text-gray-900">
                        {formatCurrency(animatedValues.traditional)}
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Chimera Model */}
              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-2xl p-6 border-2 border-emerald-300 relative overflow-hidden"
              >
                <div className="absolute top-2 right-2">
                  <span className="px-3 py-1 bg-gradient-to-r from-emerald-500 to-teal-500 text-white text-xs font-bold rounded-full">
                    OPTIMIZED
                  </span>
                </div>
                
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-emerald-800">Luminous Chimera</h3>
                  <Sparkles className="w-6 h-6 text-emerald-500" />
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-emerald-700">Gross Investment</span>
                    <span className="font-medium">{formatCurrency(comparison.chimera.gross)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-emerald-700">Foundation Tax ({formatPercentage(comparison.chimera.effectiveRate)})</span>
                    <span className="font-medium text-emerald-600">-{formatCurrency(comparison.chimera.foundationTax)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-emerald-700">Operational Costs</span>
                    <span className="font-medium text-emerald-600">-{formatCurrency(comparison.chimera.operationalCosts)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-emerald-700">Aligned Returns</span>
                    <span className="font-medium text-emerald-600">-{formatCurrency(comparison.chimera.investorDistributions)}</span>
                  </div>
                  <div className="pt-3 border-t border-emerald-300">
                    <div className="flex justify-between">
                      <span className="font-semibold text-emerald-800">To Community</span>
                      <span className="text-xl font-bold text-emerald-900">
                        {formatCurrency(animatedValues.chimera)}
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Savings Display */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-2xl p-6 text-white"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-2xl font-bold">Community Benefit</h3>
                <PiggyBank className="w-8 h-8 text-emerald-200" />
              </div>
              
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold mb-1">
                    +{formatCurrency(animatedValues.savings)}
                  </div>
                  <div className="text-emerald-200">Additional Capital</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold mb-1">
                    +{formatPercentage(comparison.difference.percentage)}
                  </div>
                  <div className="text-emerald-200">More Efficient</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold mb-1">
                    {comparison.difference.multiplier.toFixed(1)}x
                  </div>
                  <div className="text-emerald-200">Capital Multiplier</div>
                </div>
              </div>
            </motion.div>

            {/* Detailed Breakdown Toggle */}
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="mt-6 w-full py-3 px-6 bg-white border border-gray-200 rounded-xl flex items-center justify-center gap-2 hover:bg-gray-50 transition-colors"
            >
              <Info className="w-4 h-4" />
              <span className="font-semibold">
                {showDetails ? 'Hide' : 'Show'} Detailed Breakdown
              </span>
              <motion.div
                animate={{ rotate: showDetails ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                <ChevronRight className="w-4 h-4" />
              </motion.div>
            </button>

            {/* Detailed Breakdown */}
            <AnimatePresence>
              {showDetails && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-6 overflow-hidden"
                >
                  <div className="bg-gray-50 rounded-xl p-6 space-y-4">
                    <h4 className="font-bold text-lg text-gray-800 mb-4">
                      How the Luminous Chimera Model Works
                    </h4>
                    
                    <div className="space-y-3">
                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-emerald-600 font-bold">1</span>
                        </div>
                        <div>
                          <h5 className="font-semibold text-gray-800">Swiss Foundation Structure</h5>
                          <p className="text-sm text-gray-600">
                            Legal entity in Zug, Switzerland with 2.5% tax rate vs 21-28% corporate tax
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-emerald-600 font-bold">2</span>
                        </div>
                        <div>
                          <h5 className="font-semibold text-gray-800">Mission Immutability</h5>
                          <p className="text-sm text-gray-600">
                            Swiss law prevents purpose changes, ensuring permanent commitment to communities
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-emerald-600 font-bold">3</span>
                        </div>
                        <div>
                          <h5 className="font-semibold text-gray-800">Aligned Investor Returns</h5>
                          <p className="text-sm text-gray-600">
                            Community-aligned investors accept 25% returns vs 35%, knowing more goes to impact
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-emerald-600 font-bold">4</span>
                        </div>
                        <div>
                          <h5 className="font-semibold text-gray-800">Operational Efficiency</h5>
                          <p className="text-sm text-gray-600">
                            15% lower operational costs through decentralized governance and automation
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-emerald-600 font-bold">5</span>
                        </div>
                        <div>
                          <h5 className="font-semibold text-gray-800">Community Ownership Transition</h5>
                          <p className="text-sm text-gray-600">
                            7-year path to full community sovereignty with built-in capacity building
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="mt-6 p-4 bg-emerald-50 rounded-lg border border-emerald-200">
                      <div className="flex gap-2">
                        <TrendingUp className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                        <div className="text-sm text-emerald-800">
                          <strong>Result:</strong> For every ${formatCurrency(1000000)} invested, 
                          communities receive <strong>{formatCurrency(comparison.difference.absolute / (investmentAmount / 1000000))}</strong> more
                          than traditional models, accelerating energy independence by years.
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}