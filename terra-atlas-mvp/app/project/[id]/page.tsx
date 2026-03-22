// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import { ArrowLeft, MapPin, Zap, DollarSign, Users, Leaf, TrendingUp, Battery, Globe, Wind, Sun, Droplets, Atom, Calendar, Building, AlertCircle, CheckCircle, Clock, Share2, Heart, Download } from 'lucide-react'
import InvestmentFlow from '@/components/InvestmentFlow'
import { useAuth } from '@/app/contexts/AuthContext'
import { AIScoreBadge, AIAnalysisCard } from '@/components/AIScoreBadge'
import { PledgeForm } from '@/components/PledgeForm'
import { TrustScore } from '@/components/TrustScore'
import { ImpactDashboard } from '@/components/ImpactDashboard'

interface AIAnalysis {
  overallScore: number
  viabilityRating: string
  analyses: Record<string, { score: number; recommendations: string[] }>
  investmentReadiness: {
    score: number
    stage: string
    estimatedTimeToRevenue: string
  }
}

// Dynamically import map for SSR safety
const ProjectMap = dynamic(() => import('@/components/ProjectMap'), {
  loading: () => <div className="w-full h-full bg-gray-900 animate-pulse rounded-xl" />,
  ssr: false
})

interface Project {
  id: number
  name: string
  type: string
  developer: string
  status: string
  country: string
  state: string
  capacity_mw: number
  latitude: number
  longitude: number
  description?: string
  investment_needed?: number
  investment_raised?: number
  min_investment?: number
  irr?: number
  roi_percentage?: number
  completion_date?: string
  power_offtaker?: string
  total_homes_powered?: number
  co2_saved_annual?: number
  location?: string
}

export default function ProjectDetailPage() {
  const params = useParams()
  const router = useRouter()
  const { user } = useAuth()
  const [project, setProject] = useState<Project | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('overview')
  const [liked, setLiked] = useState(false)
  const [showInvestModal, setShowInvestModal] = useState(false)
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null)
  const [showPledgeForm, setShowPledgeForm] = useState(false)

  // Fetch AI analysis when project loads
  useEffect(() => {
    if (!project) return

    const fetchAnalysis = async () => {
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project: {
              id: project.id.toString(),
              name: project.name,
              type: project.type === 'battery' ? 'storage' :
                    project.type === 'smr' ? 'nuclear' :
                    project.type === 'dams' ? 'hydro' :
                    project.type,
              location: {
                latitude: project.latitude,
                longitude: project.longitude,
                state: project.state,
              },
              capacityMw: project.capacity_mw,
              status: project.status,
            },
          }),
        })

        if (response.ok) {
          const data = await response.json()
          setAiAnalysis(data.analysis)
        }
      } catch (error) {
        console.error('Failed to fetch AI analysis:', error)
      }
    }

    fetchAnalysis()
  }, [project])

  useEffect(() => {
    fetchProjectDetails()
  }, [params.id])

  const fetchProjectDetails = async () => {
    try {
      const response = await fetch(`/api/projects/${params.id}`)
      if (response.ok) {
        const data = await response.json()
        // Enhance with calculated fields
        const enhanced = {
          ...data,
          investment_needed: data.capacity_mw * 1000000, // $1M per MW
          investment_raised: data.capacity_mw * 1000000 * 0.35, // 35% raised
          min_investment: 100,
          irr: data.type === 'Solar' ? 12 : 11,
          roi_percentage: data.type === 'Solar' ? 12 : 11, // Same as IRR for now
          completion_date: '2027-Q3',
          power_offtaker: 'Regional Utility Co.',
          total_homes_powered: Math.round(data.capacity_mw * 750),
          co2_saved_annual: Math.round(data.capacity_mw * 500),
          location: `${data.state}, ${data.country}`
        }
        setProject(enhanced)
      }
    } catch (error) {
      console.error('Error fetching project:', error)
    }
    setLoading(false)
  }

  const getTypeIcon = (type: string) => {
    switch(type?.toLowerCase()) {
      case 'solar': return <Sun className="h-6 w-6" />
      case 'wind': return <Wind className="h-6 w-6" />
      case 'hydro': return <Droplets className="h-6 w-6" />
      case 'battery': 
      case 'storage': return <Battery className="h-6 w-6" />
      case 'nuclear':
      case 'smr': return <Atom className="h-6 w-6" />
      default: return <Zap className="h-6 w-6" />
    }
  }

  const getTypeColor = (type: string) => {
    switch(type?.toLowerCase()) {
      case 'solar': return 'from-yellow-500/20 to-orange-500/20 border-yellow-500/30'
      case 'wind': return 'from-blue-500/20 to-cyan-500/20 border-blue-500/30'
      case 'hydro': return 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30'
      case 'battery':
      case 'storage': return 'from-purple-500/20 to-pink-500/20 border-purple-500/30'
      case 'nuclear':
      case 'smr': return 'from-green-500/20 to-emerald-500/20 border-green-500/30'
      default: return 'from-emerald-500/20 to-cyan-500/20 border-emerald-500/30'
    }
  }

  const getStatusColor = (status: string) => {
    switch(status?.toLowerCase()) {
      case 'operational': return 'bg-green-500/20 text-green-400 border-green-500/50'
      case 'construction': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
      case 'development': return 'bg-blue-500/20 text-blue-400 border-blue-500/50'
      case 'planned': return 'bg-purple-500/20 text-purple-400 border-purple-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-white">Loading project details...</div>
      </div>
    )
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Project Not Found</h2>
          <Link href="/explore" className="text-emerald-400 hover:underline">
            Return to Explorer
          </Link>
        </div>
      </div>
    )
  }

  const fundingProgress = ((project.investment_raised || 0) / (project.investment_needed || 1)) * 100

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Terra Atlas
              </Link>
              <span className="ml-2 text-xs text-gray-400">/ Project / {project.id}</span>
            </div>
            <div className="flex space-x-8">
              <Link href="/" className="text-white/70 hover:text-white transition">Home</Link>
              <Link href="/explore" className="text-white/70 hover:text-white transition">Explore</Link>
              <Link href="/horizon" className="text-white/70 hover:text-white transition">Horizon</Link>
              <Link href="/invest" className="text-white/70 hover:text-white transition">Invest</Link>
              <Link href="/api" className="text-white/70 hover:text-white transition">API</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="pt-16">
        <div className={`relative h-96 bg-gradient-to-br ${getTypeColor(project.type)} overflow-hidden`}>
          {/* Background Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute inset-0" style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, currentColor 1px, transparent 1px)`,
              backgroundSize: '32px 32px'
            }}></div>
          </div>

          {/* Content */}
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-full flex flex-col justify-between py-8">
            {/* Top Actions */}
            <div className="flex justify-between items-start">
              <button
                onClick={() => router.back()}
                className="flex items-center gap-2 px-4 py-2 bg-black/30 backdrop-blur-sm rounded-lg border border-white/20 text-white hover:bg-black/50 transition"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </button>
              
              <div className="flex gap-2">
                <button
                  onClick={() => setLiked(!liked)}
                  className={`p-2 rounded-lg border transition ${
                    liked 
                      ? 'bg-red-500/20 border-red-500/50 text-red-400' 
                      : 'bg-black/30 border-white/20 text-white hover:bg-black/50'
                  }`}
                >
                  <Heart className={`h-5 w-5 ${liked ? 'fill-current' : ''}`} />
                </button>
                <button className="p-2 bg-black/30 backdrop-blur-sm rounded-lg border border-white/20 text-white hover:bg-black/50 transition">
                  <Share2 className="h-5 w-5" />
                </button>
                <button className="p-2 bg-black/30 backdrop-blur-sm rounded-lg border border-white/20 text-white hover:bg-black/50 transition">
                  <Download className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Project Info */}
            <div>
              <div className="flex items-center gap-3 mb-3">
                <div className="p-3 bg-white/10 backdrop-blur-sm rounded-xl">
                  {getTypeIcon(project.type)}
                </div>
                <span className={`px-3 py-1 text-sm font-semibold rounded-full border ${getStatusColor(project.status)}`}>
                  {project.status}
                </span>
              </div>
              <h1 className="text-4xl font-bold text-white mb-2">{project.name}</h1>
              <div className="mb-2">
                <TrustScore phiScore={null} harmonyAlignment={null} size="md" />
              </div>
              <div className="flex items-center gap-4 text-white/80">
                <span className="flex items-center gap-1">
                  <MapPin className="h-4 w-4" />
                  {project.state}, {project.country}
                </span>
                <span className="flex items-center gap-1">
                  <Building className="h-4 w-4" />
                  {project.developer}
                </span>
                <span className="flex items-center gap-1">
                  <Zap className="h-4 w-4" />
                  {project.capacity_mw} MW
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics Bar */}
      <div className="bg-gray-950 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <p className="text-xs text-gray-400 mb-1">Capacity</p>
              <p className="text-2xl font-bold text-emerald-400">{project.capacity_mw} MW</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Investment Needed</p>
              <p className="text-2xl font-bold text-cyan-400">${(project.investment_needed / 1000000).toFixed(0)}M</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Min Investment</p>
              <p className="text-2xl font-bold text-purple-400">${project.min_investment}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Expected IRR</p>
              <p className="text-2xl font-bold text-yellow-400">{project.irr}%</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Homes Powered</p>
              <p className="text-2xl font-bold text-blue-400">{project.total_homes_powered?.toLocaleString()}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Funding Progress */}
      <div className="bg-gradient-to-r from-emerald-950/50 to-cyan-950/50 border-b border-emerald-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-sm text-gray-400">Funding Progress</p>
              <p className="text-2xl font-bold text-white">
                ${((project.investment_raised || 0) / 1000000).toFixed(1)}M 
                <span className="text-gray-400 text-lg font-normal"> of ${((project.investment_needed || 0) / 1000000).toFixed(0)}M</span>
              </p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-emerald-400">{fundingProgress.toFixed(1)}%</p>
              <p className="text-sm text-gray-400">Funded</p>
            </div>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-4 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all duration-500"
              style={{ width: `${fundingProgress}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-3">
            <p className="text-sm text-gray-400">
              <span className="text-white font-semibold">1,247</span> investors
            </p>
            <p className="text-sm text-gray-400">
              <Clock className="h-3 w-3 inline mr-1" />
              <span className="text-white font-semibold">142</span> days remaining
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-gray-950 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {['overview', 'financials', 'impact', 'documents', 'updates'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 border-b-2 font-medium text-sm capitalize transition ${
                  activeTab === tab
                    ? 'border-emerald-400 text-emerald-400'
                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-600'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {activeTab === 'overview' && (
              <>
                {/* Description */}
                <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                  <h3 className="text-xl font-bold mb-4 text-white">Project Overview</h3>
                  <p className="text-gray-300 leading-relaxed">
                    {project.description || `This ${project.capacity_mw} MW ${project.type} project represents a significant advancement in renewable energy infrastructure for ${project.state}. Located strategically to maximize resource efficiency, this facility will provide clean, sustainable power to approximately ${project.total_homes_powered?.toLocaleString()} homes while avoiding ${project.co2_saved_annual?.toLocaleString()} tons of CO₂ emissions annually.`}
                  </p>
                  <p className="text-gray-300 leading-relaxed mt-4">
                    The project is being developed by {project.developer}, a leader in renewable energy development with a proven track record of successful project delivery. With power purchase agreements already secured with {project.power_offtaker}, this investment offers stable, long-term returns backed by contracted revenue streams.
                  </p>
                </div>

                {/* Key Features */}
                <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                  <h3 className="text-xl font-bold mb-4 text-white">Key Features</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <div>
                        <p className="font-semibold text-white">Long-term PPA</p>
                        <p className="text-sm text-gray-400">25-year power purchase agreement secured</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <div>
                        <p className="font-semibold text-white">Proven Technology</p>
                        <p className="text-sm text-gray-400">Tier 1 equipment with performance guarantees</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <div>
                        <p className="font-semibold text-white">Insurance Coverage</p>
                        <p className="text-sm text-gray-400">Comprehensive insurance including weather events</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <div>
                        <p className="font-semibold text-white">Community Benefits</p>
                        <p className="text-sm text-gray-400">Local jobs and community investment programs</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Location Map */}
                <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                  <h3 className="text-xl font-bold mb-4 text-white">Location</h3>
                  <div className="h-64 rounded-lg overflow-hidden">
                    <ProjectMap 
                      latitude={project.latitude || 39.8283} 
                      longitude={project.longitude || -98.5795}
                      name={project.name}
                    />
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Coordinates</p>
                      <p className="text-white font-mono">
                        {(project.latitude || 39.8283).toFixed(4)}°N, {Math.abs(project.longitude || -98.5795).toFixed(4)}°W
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400">Grid Connection</p>
                      <p className="text-white">Regional Transmission Network</p>
                    </div>
                  </div>
                </div>
              </>
            )}

            {activeTab === 'financials' && (
              <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                <h3 className="text-xl font-bold mb-4 text-white">Financial Details</h3>
                <div className="space-y-4">
                  <div className="flex justify-between py-3 border-b border-gray-800">
                    <span className="text-gray-400">Total Project Cost</span>
                    <span className="text-white font-semibold">${(project.investment_needed / 1000000).toFixed(1)}M</span>
                  </div>
                  <div className="flex justify-between py-3 border-b border-gray-800">
                    <span className="text-gray-400">Expected IRR</span>
                    <span className="text-emerald-400 font-semibold">{project.irr}%</span>
                  </div>
                  <div className="flex justify-between py-3 border-b border-gray-800">
                    <span className="text-gray-400">Payback Period</span>
                    <span className="text-white font-semibold">7-9 years</span>
                  </div>
                  <div className="flex justify-between py-3 border-b border-gray-800">
                    <span className="text-gray-400">Revenue Model</span>
                    <span className="text-white font-semibold">Fixed PPA + REC Sales</span>
                  </div>
                  <div className="flex justify-between py-3">
                    <span className="text-gray-400">Project Lifetime</span>
                    <span className="text-white font-semibold">25+ years</span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'impact' && (
              <div className="space-y-6">
                <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                  <h3 className="text-xl font-bold mb-4 text-white">Environmental Impact</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Leaf className="h-5 w-5 text-green-400" />
                        <p className="text-gray-400">CO₂ Avoided Annually</p>
                      </div>
                      <p className="text-3xl font-bold text-green-400">{project.co2_saved_annual?.toLocaleString()}</p>
                      <p className="text-sm text-gray-500">tons per year</p>
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Users className="h-5 w-5 text-blue-400" />
                        <p className="text-gray-400">Homes Powered</p>
                      </div>
                      <p className="text-3xl font-bold text-blue-400">{project.total_homes_powered?.toLocaleString()}</p>
                      <p className="text-sm text-gray-500">households</p>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                  <h3 className="text-xl font-bold mb-4 text-white">Community Benefits</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <span className="text-gray-300">200+ construction jobs, 15 permanent positions</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <span className="text-gray-300">$2M annual local tax revenue</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <span className="text-gray-300">Community ownership program after year 7</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <span className="text-gray-300">Educational partnerships with local schools</span>
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar - Investment Flow */}
          <div className="lg:col-span-1">
            <div className="sticky top-24">
              <InvestmentFlow 
                project={project} 
                user={user ? { id: user.id, email: user.email } : undefined}
              />
              
              {/* QOL Impact Dashboard */}
              <ImpactDashboard projectId={params.id as string} className="mt-6" />

              {/* Quick Stats */}
              <div className="mt-6 bg-gray-900/50 backdrop-blur-md rounded-xl p-6 border border-gray-800">
                <h4 className="font-bold text-white mb-4">Investment Highlights</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Risk Level</span>
                    <span className="text-yellow-400 font-semibold">Moderate</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Investment Type</span>
                    <span className="text-white font-semibold">Equity</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Liquidity</span>
                    <span className="text-white font-semibold">After 3 years</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Target Close</span>
                    <span className="text-white font-semibold">{project.completion_date}</span>
                  </div>
                </div>
              </div>

              {/* Risk Disclaimer */}
              <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <div className="flex gap-2">
                  <AlertCircle className="h-5 w-5 text-yellow-400 flex-shrink-0" />
                  <div>
                    <p className="text-sm text-yellow-400 font-semibold mb-1">Investment Risk</p>
                    <p className="text-xs text-gray-400">
                      All investments carry risk. Past performance does not guarantee future results. Please review all documentation before investing.
                    </p>
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