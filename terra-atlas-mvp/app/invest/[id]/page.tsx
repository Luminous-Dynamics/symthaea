'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  ArrowLeft, MapPin, Clock, Users, Shield, Award, TrendingUp,
  DollarSign, Battery, Sun, Wind, Droplets, Zap, AlertCircle,
  CheckCircle, Info, ChevronRight, Building, Calendar, Target,
  BarChart3, PieChart, Activity, Sparkles, Lock
} from 'lucide-react'
import { useAuth } from '@/app/contexts/AuthContext'
import PaymentForm from '@/components/PaymentForm'

interface ProjectDetails {
  id: number
  name: string
  type: string
  capacity_mw: number
  location: string
  state: string
  country: string
  status: string
  developer: string
  description: string
  roi_percentage: number
  min_investment: number
  max_investment: number
  total_raised: number
  target_amount: number
  investors_count: number
  completion_date: string
  payback_period: string
  risk_level: 'Low' | 'Medium' | 'High'
  highlights: string[]
  risks: string[]
  documents: { name: string; type: string; size: string }[]
}

export default function InvestProjectPage() {
  const params = useParams()
  const router = useRouter()
  const { user } = useAuth()
  const [project, setProject] = useState<ProjectDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [investmentAmount, setInvestmentAmount] = useState('')
  const [investmentTerm, setInvestmentTerm] = useState(12)
  const [showConfirmation, setShowConfirmation] = useState(false)
  const [showPaymentForm, setShowPaymentForm] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [cache, setCache] = useState<Record<string, ProjectDetails>>({})
  const [controller, setController] = useState<AbortController | null>(null)

  useEffect(() => {
    const id = params.id?.toString()
    if (!id) return

    const cached = cache[id]
    if (cached) {
      setProject(cached)
      setLoading(false)
      return
    }

    const abortController = new AbortController()
    controller?.abort()
    setController(abortController)

    fetchProjectDetails(id, abortController)

    return () => abortController.abort()
  }, [params.id])

  const fetchProjectDetails = async (id: string, abortController: AbortController) => {
    try {
      const res = await fetch(`/api/projects/${id}`, { signal: abortController.signal })
      const data = await res.json()
      
      // Enrich with investment details
      const enrichedProject: ProjectDetails = {
        ...data,
        roi_percentage: getROIByType(data.type),
        min_investment: 10,
        max_investment: 1000000,
        total_raised: Math.floor(Math.random() * 5000000) + 1000000,
        target_amount: Math.floor(Math.random() * 10000000) + 5000000,
        investors_count: Math.floor(Math.random() * 500) + 50,
        completion_date: getCompletionDate(data.status),
        payback_period: '5-7 years',
        risk_level: getRiskLevel(data.type, data.status),
        description: generateDetailedDescription(data),
        highlights: generateHighlights(data),
        risks: generateRisks(data),
        documents: [
          { name: 'Investment Prospectus', type: 'PDF', size: '2.4 MB' },
          { name: 'Financial Projections', type: 'XLSX', size: '1.1 MB' },
          { name: 'Environmental Impact Report', type: 'PDF', size: '3.7 MB' },
          { name: 'Technical Specifications', type: 'PDF', size: '5.2 MB' }
        ]
      }
      
      if (!abortController.signal.aborted) {
        setProject(enrichedProject)
        setCache((prev) => ({ ...prev, [id]: enrichedProject }))
      }
    } catch (error) {
      if (abortController.signal.aborted) return
      console.error('Error fetching project:', error)
    }
    if (!abortController.signal.aborted) {
      setLoading(false)
    }
  }

  const getROIByType = (type: string): number => {
    const rois: { [key: string]: number } = {
      'Solar': 14,
      'Wind': 16,
      'Battery': 15,
      'Hydro': 11,
      'Nuclear': 12
    }
    return rois[type] || 13
  }

  const getRiskLevel = (type: string, status: string): 'Low' | 'Medium' | 'High' => {
    if (status === 'Operational') return 'Low'
    if (status === 'Construction') return 'Medium'
    return 'Low'
  }

  const getCompletionDate = (status: string): string => {
    const dates: { [key: string]: string } = {
      'Operational': 'Operating Now',
      'Construction': 'Q3 2025',
      'Development': 'Q2 2026',
      'Planned': 'Q1 2027'
    }
    return dates[status] || 'Q4 2026'
  }

  const generateDetailedDescription = (project: any): string => {
    return `This ${project.capacity_mw}MW ${project.type} project represents a significant opportunity 
    in the renewable energy sector. Located in ${project.location || project.state}, ${project.country}, 
    the facility will provide clean energy to thousands of homes while generating attractive returns for investors.`
  }

  const generateHighlights = (project: any): string[] => {
    return [
      `${project.capacity_mw}MW installed capacity`,
      'Long-term power purchase agreements secured',
      'Experienced development team with 20+ years track record',
      'Tax-advantaged investment structure',
      'Quarterly dividend distributions',
      'Full insurance coverage and performance guarantees'
    ]
  }

  const generateRisks = (project: any): string[] => {
    return [
      'Market price fluctuations for electricity',
      'Regulatory and policy changes',
      'Weather-related performance variations',
      'Technology obsolescence over project lifetime'
    ]
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Solar': return <Sun className="h-6 w-6" />
      case 'Wind': return <Wind className="h-6 w-6" />
      case 'Battery': return <Battery className="h-6 w-6" />
      case 'Hydro': return <Droplets className="h-6 w-6" />
      case 'Nuclear': return <Zap className="h-6 w-6" />
      default: return <Zap className="h-6 w-6" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'Solar': return 'from-yellow-500 to-orange-500'
      case 'Wind': return 'from-blue-500 to-cyan-500'
      case 'Battery': return 'from-purple-500 to-pink-500'
      case 'Hydro': return 'from-cyan-500 to-teal-500'
      case 'Nuclear': return 'from-green-500 to-emerald-500'
      default: return 'from-gray-500 to-gray-600'
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `$${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `$${(num / 1000).toFixed(0)}K`
    return `$${num}`
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

  const handleInvestmentSubmit = async () => {
    if (!user) {
      router.push('/auth/login')
      return
    }

    const amount = parseFloat(investmentAmount)
    if (amount < 10) {
      setError('Minimum investment is $10')
      return
    }

    // Show payment form instead of processing immediately
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
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400"></div>
          <p className="mt-4 text-gray-400">Loading project details...</p>
        </div>
      </div>
    )
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">Project not found</h2>
          <Link href="/invest" className="text-emerald-400 hover:text-emerald-300">
            Back to investments
          </Link>
        </div>
      </div>
    )
  }

  const returns = calculateReturns()

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/invest" className="flex items-center gap-2 text-gray-400 hover:text-white transition mr-4">
                <ArrowLeft className="h-5 w-5" />
                Back
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
            <h2 className="text-2xl font-bold text-white mb-2">Investment Successful!</h2>
            <p className="text-gray-400 mb-6">
              Your investment of {formatNumber(parseFloat(investmentAmount))} in {project.name} has been submitted.
            </p>
            <p className="text-sm text-gray-500 mb-6">
              Redirecting to your portfolio...
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

      {/* Confirmation Modal */}
      {showConfirmation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full"
          >
            <h3 className="text-xl font-bold text-white mb-4">Confirm Investment</h3>
            
            <div className="space-y-3 mb-6">
              <div className="flex justify-between">
                <span className="text-gray-400">Project:</span>
                <span className="text-white font-semibold">{project.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Investment Amount:</span>
                <span className="text-white font-semibold">{formatNumber(parseFloat(investmentAmount))}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Term:</span>
                <span className="text-white font-semibold">{investmentTerm} months</span>
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
            <h3 className="text-xl font-bold text-white mb-6">Complete Payment</h3>
            
            <PaymentForm
              amount={parseFloat(investmentAmount)}
              projectId={project.id}
              projectName={project.name}
              projectType={project.type}
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
                  <h1 className="text-3xl font-bold text-white mb-2">{project.name}</h1>
                  <div className="flex items-center gap-4 text-gray-400">
                    <div className="flex items-center gap-1">
                      <MapPin className="h-4 w-4" />
                      {project.location || project.state}, {project.country}
                    </div>
                    <div className="flex items-center gap-1">
                      <Building className="h-4 w-4" />
                      {project.developer}
                    </div>
                  </div>
                </div>
                <div className={`px-4 py-2 bg-gradient-to-r ${getTypeColor(project.type)} rounded-lg flex items-center gap-2`}>
                  {getTypeIcon(project.type)}
                  <span className="font-semibold">{project.type}</span>
                </div>
              </div>

              {/* Key Stats */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-emerald-400">{project.roi_percentage}%</div>
                  <div className="text-xs text-gray-500">Annual Return</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-white">{project.capacity_mw} MW</div>
                  <div className="text-xs text-gray-500">Capacity</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-cyan-400">{project.investors_count}</div>
                  <div className="text-xs text-gray-500">Investors</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-2xl font-bold text-purple-400">{project.completion_date}</div>
                  <div className="text-xs text-gray-500">Completion</div>
                </div>
              </div>
            </div>

            {/* Description */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Project Overview</h2>
              <p className="text-gray-300 leading-relaxed">{project.description}</p>
            </div>

            {/* Investment Highlights */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-yellow-400" />
                Investment Highlights
              </h2>
              <ul className="space-y-3">
                {project.highlights.map((highlight, idx) => (
                  <li key={idx} className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                    <span className="text-gray-300">{highlight}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Risks */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-yellow-400" />
                Risk Factors
              </h2>
              <ul className="space-y-3">
                {project.risks.map((risk, idx) => (
                  <li key={idx} className="flex items-start gap-3">
                    <Info className="h-5 w-5 text-yellow-400 mt-0.5" />
                    <span className="text-gray-300">{risk}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Documents */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Project Documents</h2>
              <div className="space-y-3">
                {project.documents.map((doc, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gray-800 rounded-lg flex items-center justify-center">
                        <span className="text-xs font-bold text-gray-400">{doc.type}</span>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-white">{doc.name}</div>
                        <div className="text-xs text-gray-500">{doc.size}</div>
                      </div>
                    </div>
                    <button className="text-emerald-400 hover:text-emerald-300 text-sm font-medium">
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Investment Widget */}
          <div className="lg:sticky lg:top-24 h-fit space-y-6">
            {/* Investment Form */}
            <div className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 backdrop-blur-md border border-emerald-500/30 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Make an Investment</h3>

              {/* Funding Progress */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Funding Progress</span>
                  <span className="text-white font-semibold">
                    {((project.total_raised / project.target_amount) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${(project.total_raised / project.target_amount) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>{formatNumber(project.total_raised)} raised</span>
                  <span>{formatNumber(project.target_amount)} goal</span>
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
                    placeholder="Enter amount"
                    min="10"
                    className="w-full bg-black/30 border border-gray-700 rounded-lg py-3 pl-10 pr-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Minimum: $10 • Maximum: {formatNumber(project.max_investment)}
                </p>
              </div>

              {/* Investment Term */}
              <div className="mb-6">
                <label className="text-sm font-medium text-gray-300 mb-2 block">
                  Investment Term
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[12, 24, 36].map(term => (
                    <button
                      key={term}
                      onClick={() => setInvestmentTerm(term)}
                      className={`py-2 px-3 rounded-lg font-medium transition ${
                        investmentTerm === term
                          ? 'bg-emerald-500/20 border border-emerald-500 text-emerald-400'
                          : 'bg-black/30 border border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {term} mo
                    </button>
                  ))}
                </div>
              </div>

              {/* Returns Calculator */}
              {investmentAmount && parseFloat(investmentAmount) >= 10 && (
                <div className="mb-6 p-4 bg-black/30 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-400 mb-3">Expected Returns</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Annual Return ({project.roi_percentage}%):</span>
                      <span className="text-sm font-semibold text-white">{formatNumber(returns.annual)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Total Return ({investmentTerm} months):</span>
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
                  } else if (parseFloat(investmentAmount) >= 10) {
                    setShowConfirmation(true)
                  } else {
                    setError('Minimum investment is $10')
                  }
                }}
                disabled={!investmentAmount || parseFloat(investmentAmount) < 10}
                className="w-full py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {user ? (
                  <>
                    Invest Now
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
            </div>

            {/* Trust Badges */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Shield className="h-5 w-5 text-emerald-400" />
                  <div>
                    <div className="text-sm font-medium text-white">Secure Investment</div>
                    <div className="text-xs text-gray-500">256-bit encrypted transactions</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Award className="h-5 w-5 text-cyan-400" />
                  <div>
                    <div className="text-sm font-medium text-white">Regulated Platform</div>
                    <div className="text-xs text-gray-500">SEC-registered & licensed</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Users className="h-5 w-5 text-purple-400" />
                  <div>
                    <div className="text-sm font-medium text-white">{project.investors_count}+ Investors</div>
                    <div className="text-xs text-gray-500">Join a growing community</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
