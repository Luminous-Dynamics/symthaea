// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  ArrowLeft, MapPin, Clock, Users, Shield, Award, TrendingUp,
  DollarSign, Battery, Atom, AlertCircle, CheckCircle, Info, 
  ChevronRight, Building, Calendar, Target, BarChart3, Activity, 
  Sparkles, Lock, Zap, Globe, Building2, FileText, Gauge,
  ShieldCheck, AlertTriangle, TrendingDown
} from 'lucide-react'
import { useAuth } from '@/app/contexts/AuthContext'
import PaymentForm from '@/components/PaymentForm'

interface SMRProject {
  id: number
  name: string
  developer: string
  type: string
  subtype: string
  capacity_mw: number
  modules: number
  module_capacity_mw: number
  location: string
  state: string
  country: string
  lat: number
  lon: number
  status: string
  construction_start: string
  commercial_operation: string
  total_investment: number
  investment_raised: number
  investment_needed: number
  levelized_cost_mwh: number
  power_purchase_agreement: string
  capacity_factor: number
  design_life_years: number
  regulatory_status: string
  description: string
  key_features: string[]
  investors: string[]
  risk_factors: string[]
  roi_percentage: number
  min_investment: number
  investment_type: string
  website: string
}

export default function SMRInvestmentPage() {
  const params = useParams()
  const router = useRouter()
  const { user } = useAuth()
  const [project, setProject] = useState<SMRProject | null>(null)
  const [loading, setLoading] = useState(true)
  const [investmentAmount, setInvestmentAmount] = useState('')
  const [investmentTerm, setInvestmentTerm] = useState(60) // SMR default 5 years
  const [showConfirmation, setShowConfirmation] = useState(false)
  const [showPaymentForm, setShowPaymentForm] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [showRiskDisclosure, setShowRiskDisclosure] = useState(false)
  const cacheRef = useRef<Record<string, SMRProject>>({})
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const id = params.id?.toString()
    if (!id) return

    const cached = cacheRef.current[id]
    if (cached) {
      setProject(cached)
      setInvestmentAmount(cached.min_investment.toString())
      setLoading(false)
      return
    }

    const controller = new AbortController()
    controllerRef.current?.abort()
    controllerRef.current = controller
    fetchSMRProject(id, controller)

    return () => controller.abort()
  }, [params.id])

  const fetchSMRProject = async (id: string, controller: AbortController) => {
    try {
      // Fetch specific SMR project
      const res = await fetch('/api/smr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ id: parseInt(id) })
      })
      
      if (!res.ok) {
        throw new Error('Project not found')
      }
      
      const data = await res.json()
      if (!controller.signal.aborted) {
        setProject(data)
        cacheRef.current[id] = data
        // Set minimum investment amount as default
        setInvestmentAmount(data.min_investment.toString())
      }
    } catch (error) {
      if (controller.signal.aborted) return
      console.error('Error fetching SMR project:', error)
    }
    if (!controller.signal.aborted) {
      setLoading(false)
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000000) return `$${(num / 1000000000).toFixed(2)}B`
    if (num >= 1000000) return `$${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `$${(num / 1000).toFixed(0)}K`
    return `$${num.toFixed(0)}`
  }

  const calculateReturns = () => {
    const amount = parseFloat(investmentAmount) || 0
    const annualReturn = amount * (project?.roi_percentage || 0) / 100
    const totalReturn = annualReturn * (investmentTerm / 12)
    return {
      annual: annualReturn,
      total: totalReturn,
      final: amount + totalReturn
    }
  }

  const getRiskLevel = (factors: string[]) => {
    if (factors.some(f => f.toLowerCase().includes('first'))) {
      return { level: 'Medium-High', color: 'text-yellow-400' }
    }
    return { level: 'Medium', color: 'text-green-400' }
  }

  const handleInvestmentSubmit = async () => {
    if (!user) {
      router.push('/auth/login')
      return
    }

    const amount = parseFloat(investmentAmount)
    if (amount < (project?.min_investment || 100000)) {
      setError(`Minimum investment for SMR projects is ${formatNumber(project?.min_investment || 100000)}`)
      return
    }

    // Show payment form for SMR investments
    setShowConfirmation(false)
    setShowPaymentForm(true)
  }

  const handlePaymentSuccess = (paymentIntentId: string) => {
    setSuccess(true)
    setShowPaymentForm(false)
    // Redirect to portfolio after 3 seconds
    setTimeout(() => {
      router.push('/portfolio')
    }, 3000)
  }

  const handlePaymentError = (error: string) => {
    setError(error)
    setShowPaymentForm(false)
  }

  const handlePaymentCancel = () => {
    setShowPaymentForm(false)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <Atom className="h-12 w-12 text-emerald-400 mx-auto mb-4 animate-pulse" />
          <p className="text-gray-400">Loading SMR project details...</p>
        </div>
      </div>
    )
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">SMR Project not found</h2>
          <Link href="/smr" className="text-emerald-400 hover:text-emerald-300">
            Back to SMR projects
          </Link>
        </div>
      </div>
    )
  }

  const returns = calculateReturns()
  const riskInfo = getRiskLevel(project.risk_factors)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/smr" className="flex items-center gap-2 text-gray-400 hover:text-white transition mr-4">
                <ArrowLeft className="h-5 w-5" />
                SMR Projects
              </Link>
              <Link href="/" className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Terra Atlas
              </Link>
            </div>
            <div className="flex items-center space-x-8">
              {user ? (
                <Link href="/portfolio" className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold">
                  My Portfolio
                </Link>
              ) : (
                <Link href="/auth/login" className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-semibold">
                  Login to Invest
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Success Modal */}
      {success && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-emerald-500/50 rounded-xl p-8 max-w-md w-full text-center"
          >
            <CheckCircle className="h-16 w-16 text-emerald-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-white mb-2">SMR Investment Successful!</h2>
            <p className="text-gray-400 mb-6">
              Your investment of {formatNumber(parseFloat(investmentAmount))} in {project.name} has been submitted.
            </p>
            <p className="text-sm text-gray-500 mb-6">
              You will receive detailed investment documents via email within 24 hours.
            </p>
            <Link
              href="/portfolio"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold"
            >
              View Portfolio
              <ChevronRight className="h-4 w-4" />
            </Link>
          </motion.div>
        </div>
      )}

      {/* Risk Disclosure Modal */}
      {showRiskDisclosure && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
          >
            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="h-6 w-6 text-yellow-400" />
              SMR Investment Risk Disclosure
            </h3>
            
            <div className="space-y-4 text-sm text-gray-300">
              <div>
                <h4 className="font-semibold text-white mb-2">Technology Risks</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li>First-of-a-kind deployment may face technical challenges</li>
                  <li>Regulatory approval processes may cause delays</li>
                  <li>Construction cost overruns are possible</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-white mb-2">Market Risks</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li>Electricity price fluctuations affect returns</li>
                  <li>Competition from other energy sources</li>
                  <li>Changes in energy policy and subsidies</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-white mb-2">Investment Terms</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li>Minimum 5-year investment commitment</li>
                  <li>Limited liquidity during construction phase</li>
                  <li>Accredited investor status required</li>
                </ul>
              </div>
              
              <div className="pt-4 border-t border-gray-800">
                <label className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    className="mt-1"
                    onChange={(e) => {
                      if (e.target.checked) {
                        setShowRiskDisclosure(false)
                        setShowConfirmation(true)
                      }
                    }}
                  />
                  <span className="text-xs text-gray-400">
                    I understand and accept the risks associated with SMR investments. 
                    I confirm that I am an accredited investor and can afford the potential loss of this investment.
                  </span>
                </label>
              </div>
            </div>
            
            <div className="mt-6 flex gap-3">
              <button
                onClick={() => setShowRiskDisclosure(false)}
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg font-medium text-gray-300 hover:bg-gray-700 transition"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Investment Confirmation Modal */}
      {showConfirmation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full"
          >
            <h3 className="text-xl font-bold text-white mb-4">Confirm SMR Investment</h3>
            
            <div className="space-y-3 mb-6">
              <div className="flex justify-between">
                <span className="text-gray-400">Project:</span>
                <span className="text-white font-semibold">{project.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Developer:</span>
                <span className="text-white font-semibold">{project.developer}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Investment Amount:</span>
                <span className="text-white font-semibold">{formatNumber(parseFloat(investmentAmount))}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Term:</span>
                <span className="text-white font-semibold">{investmentTerm / 12} years</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Expected Annual Return:</span>
                <span className="text-emerald-400 font-semibold">{project.roi_percentage}%</span>
              </div>
              <div className="flex justify-between border-t border-gray-800 pt-3">
                <span className="text-gray-400">Total Expected Return:</span>
                <span className="text-emerald-400 font-bold">{formatNumber(returns.total)}</span>
              </div>
            </div>

            {error && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => setShowConfirmation(false)}
                disabled={processing}
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg font-medium text-gray-300 hover:bg-gray-700 transition disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleInvestmentSubmit}
                disabled={processing}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {processing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <Lock className="h-4 w-4" />
                    Proceed to Payment
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Payment Form Modal */}
      {showPaymentForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full"
          >
            <h3 className="text-xl font-bold text-white mb-6">Complete SMR Investment</h3>
            
            <PaymentForm
              amount={parseFloat(investmentAmount)}
              projectId={project.id}
              projectName={project.name}
              projectType="SMR"
              investmentTermMonths={investmentTerm}
              onSuccess={handlePaymentSuccess}
              onError={handlePaymentError}
              onCancel={handlePaymentCancel}
            />
          </motion.div>
        </div>
      )}

      {/* Main Content */}
      <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Project Details */}
          <div className="lg:col-span-2 space-y-6">
            {/* Project Header */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <Atom className="h-8 w-8 text-emerald-400" />
                    <span className="px-3 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-xs font-semibold text-emerald-400">
                      Small Modular Reactor
                    </span>
                  </div>
                  <h1 className="text-3xl font-bold text-white mb-2">{project.name}</h1>
                  <div className="flex items-center gap-4 text-gray-400">
                    <div className="flex items-center gap-1">
                      <MapPin className="h-4 w-4" />
                      {project.location}, {project.country}
                    </div>
                    <div className="flex items-center gap-1">
                      <Building className="h-4 w-4" />
                      {project.developer}
                    </div>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  project.status === 'Construction' ? 'text-green-400 bg-green-500/20' :
                  project.status === 'Development' ? 'text-blue-400 bg-blue-500/20' :
                  'text-yellow-400 bg-yellow-500/20'
                }`}>
                  {project.status}
                </span>
              </div>

              {/* Key Technical Specs */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-emerald-400">{project.capacity_mw} MW</div>
                  <div className="text-xs text-gray-500">Total Capacity</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-cyan-400">{project.modules}</div>
                  <div className="text-xs text-gray-500">Modules</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-purple-400">{project.design_life_years}yr</div>
                  <div className="text-xs text-gray-500">Design Life</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-yellow-400">{project.capacity_factor}%</div>
                  <div className="text-xs text-gray-500">Capacity Factor</div>
                </div>
              </div>
            </div>

            {/* Technology Details */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Zap className="h-5 w-5 text-cyan-400" />
                Technology Overview
              </h2>
              <p className="text-gray-300 leading-relaxed mb-4">{project.description}</p>
              
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Reactor Type</div>
                  <div className="text-white font-medium">{project.subtype}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Module Capacity</div>
                  <div className="text-white font-medium">{project.module_capacity_mw} MW each</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">LCOE</div>
                  <div className="text-white font-medium">${project.levelized_cost_mwh}/MWh</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Power Agreement</div>
                  <div className="text-white font-medium">{project.power_purchase_agreement}</div>
                </div>
              </div>
            </div>

            {/* Key Features */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-yellow-400" />
                Key Features
              </h2>
              <div className="space-y-3">
                {project.key_features.map((feature, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                    <span className="text-gray-300">{feature}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Timeline */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Calendar className="h-5 w-5 text-purple-400" />
                Project Timeline
              </h2>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-purple-500/20 rounded-full flex items-center justify-center">
                    <Target className="h-6 w-6 text-purple-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium">Construction Start</div>
                    <div className="text-sm text-gray-400">{project.construction_start}</div>
                  </div>
                </div>
                <div className="border-l-2 border-gray-800 ml-6 h-8"></div>
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-emerald-500/20 rounded-full flex items-center justify-center">
                    <CheckCircle className="h-6 w-6 text-emerald-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium">Commercial Operation</div>
                    <div className="text-sm text-gray-400">{project.commercial_operation}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Current Investors */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Users className="h-5 w-5 text-cyan-400" />
                Strategic Investors
              </h2>
              <div className="flex flex-wrap gap-2">
                {project.investors.map((investor, idx) => (
                  <span key={idx} className="px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-sm text-cyan-400">
                    {investor}
                  </span>
                ))}
              </div>
            </div>

            {/* Risk Factors */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-yellow-400" />
                Risk Assessment
              </h2>
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm text-gray-400">Overall Risk Level:</span>
                  <span className={`font-semibold ${riskInfo.color}`}>{riskInfo.level}</span>
                </div>
              </div>
              <div className="space-y-3">
                {project.risk_factors.map((risk, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <Info className="h-5 w-5 text-yellow-400 mt-0.5" />
                    <span className="text-gray-300">{risk}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Investment Widget */}
          <div className="lg:sticky lg:top-24 h-fit space-y-6">
            {/* Investment Form */}
            <div className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 backdrop-blur-md border border-emerald-500/30 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Atom className="h-6 w-6 text-emerald-400" />
                <h3 className="text-xl font-semibold text-white">SMR Investment</h3>
              </div>

              {/* Investment Progress */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Capital Raised</span>
                  <span className="text-white font-semibold">
                    {((project.investment_raised / project.total_investment) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${(project.investment_raised / project.total_investment) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>{formatNumber(project.investment_raised)} raised</span>
                  <span>{formatNumber(project.investment_needed)} needed</span>
                </div>
              </div>

              {/* Financial Highlights */}
              <div className="grid grid-cols-2 gap-3 mb-6">
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-lg font-bold text-emerald-400">{project.roi_percentage}%</div>
                  <div className="text-xs text-gray-500">Annual ROI</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-lg font-bold text-cyan-400">{formatNumber(project.min_investment)}</div>
                  <div className="text-xs text-gray-500">Min. Investment</div>
                </div>
              </div>

              {/* Investment Amount Input */}
              <div className="mb-4">
                <label className="text-sm font-medium text-gray-300 mb-2 block">
                  Investment Amount (USD)
                </label>
                <div className="relative">
                  <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
                  <input
                    type="number"
                    value={investmentAmount}
                    onChange={(e) => setInvestmentAmount(e.target.value)}
                    placeholder={`Min: ${formatNumber(project.min_investment)}`}
                    min={project.min_investment}
                    step="10000"
                    className="w-full bg-black/30 border border-gray-700 rounded-lg py-3 pl-10 pr-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Minimum: {formatNumber(project.min_investment)} • Accredited investors only
                </p>
              </div>

              {/* Investment Term */}
              <div className="mb-6">
                <label className="text-sm font-medium text-gray-300 mb-2 block">
                  Investment Term
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[60, 84, 120].map(term => (
                    <button
                      key={term}
                      onClick={() => setInvestmentTerm(term)}
                      className={`py-2 px-3 rounded-lg font-medium transition text-sm ${
                        investmentTerm === term
                          ? 'bg-emerald-500/20 border border-emerald-500 text-emerald-400'
                          : 'bg-black/30 border border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {term / 12} yr
                    </button>
                  ))}
                </div>
              </div>

              {/* Returns Calculator */}
              {investmentAmount && parseFloat(investmentAmount) >= project.min_investment && (
                <div className="mb-6 p-4 bg-black/30 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-400 mb-3">Projected Returns</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Annual Return ({project.roi_percentage}%):</span>
                      <span className="text-sm font-semibold text-white">{formatNumber(returns.annual)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Total Return ({investmentTerm / 12} years):</span>
                      <span className="text-sm font-semibold text-emerald-400">{formatNumber(returns.total)}</span>
                    </div>
                    <div className="flex justify-between pt-2 border-t border-gray-800">
                      <span className="text-sm text-gray-400">Final Value:</span>
                      <span className="text-lg font-bold text-emerald-400">{formatNumber(returns.final)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* CTA Button */}
              <button
                onClick={() => {
                  if (!user) {
                    router.push('/auth/login')
                  } else if (parseFloat(investmentAmount) >= project.min_investment) {
                    setShowRiskDisclosure(true)
                  } else {
                    setError(`Minimum investment is ${formatNumber(project.min_investment)}`)
                  }
                }}
                disabled={!investmentAmount || parseFloat(investmentAmount) < project.min_investment}
                className="w-full py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {user ? (
                  <>
                    Review Investment
                    <ChevronRight className="h-5 w-5" />
                  </>
                ) : (
                  <>
                    Login to Invest
                    <Lock className="h-5 w-5" />
                  </>
                )}
              </button>

              {error && (
                <p className="text-sm text-red-400 mt-2">{error}</p>
              )}

              {/* Investment Type Badge */}
              <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-purple-400" />
                  <span className="text-sm font-medium text-purple-400">{project.investment_type}</span>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Accredited investors only • Wire transfer required
                </p>
              </div>
            </div>

            {/* Regulatory Status */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Regulatory Status</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 text-emerald-400" />
                  <div>
                    <div className="text-sm font-medium text-white">{project.regulatory_status}</div>
                    <div className="text-xs text-gray-500">Current regulatory phase</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Shield className="h-5 w-5 text-cyan-400" />
                  <div>
                    <div className="text-sm font-medium text-white">NRC Oversight</div>
                    <div className="text-xs text-gray-500">Full regulatory compliance</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Award className="h-5 w-5 text-purple-400" />
                  <div>
                    <div className="text-sm font-medium text-white">DOE Support</div>
                    <div className="text-xs text-gray-500">Federal backing & grants</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Learn More */}
            <div className="text-center">
              <a
                href={project.website}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 transition"
              >
                <Globe className="h-4 w-4" />
                Visit Developer Website
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
