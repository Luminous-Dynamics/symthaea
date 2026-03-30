// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  TrendingUp, DollarSign, PieChart, Activity, Calendar, Clock,
  Download, Filter, Search, ChevronRight, AlertCircle, CheckCircle,
  Award, Target, Zap, Sun, Wind, Battery, Droplets, MapPin,
  ArrowUp, ArrowDown, Minus, BarChart3, Users, Building
} from 'lucide-react'
import { useAuth } from '@/app/contexts/AuthContext'
import { supabase } from '@/lib/supabase'

interface Investment {
  id: string
  project_id: number
  project_name: string
  project_type: string
  amount: number
  status: string
  expected_return: number
  share_percentage: number
  investment_term_months: number
  created_at: string
  confirmed_at?: string
  completed_at?: string
}

interface PortfolioSummary {
  total_investments: number
  unique_projects: number
  total_invested: number
  total_expected_return: number
  avg_return_percentage: number
  active_pledges: number
  total_pledged: number
}

interface Transaction {
  id: string
  type: 'deposit' | 'withdrawal' | 'dividend' | 'fee' | 'refund'
  amount: number
  description: string
  created_at: string
  balance_after: number
}

export default function PortfolioPage() {
  const [investments, setInvestments] = useState<Investment[]>([])
  const [summary, setSummary] = useState<PortfolioSummary | null>(null)
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'investments' | 'transactions' | 'performance'>('overview')
  const [timeRange, setTimeRange] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')
  const { user, session, loading: authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/auth/login')
    } else if (user) {
      fetchPortfolioData()
    }
  }, [user, authLoading, router])

  const fetchPortfolioData = async () => {
    if (!user) return

    try {
      let token = session?.access_token
      if (!token) {
        const { data } = await supabase.auth.getSession()
        token = data.session?.access_token || null
      }

      if (!token) {
        throw new Error('Session expired')
      }
      
      // Fetch investments
      const investmentsRes = await fetch('/api/investments', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      const investmentsData = await investmentsRes.json()
      
      setInvestments(investmentsData.investments || [])
      setSummary(investmentsData.summary || {
        total_investments: 0,
        unique_projects: 0,
        total_invested: 0,
        total_expected_return: 0,
        avg_return_percentage: 0,
        active_pledges: 0,
        total_pledged: 0
      })

      // Generate mock transactions (in production, fetch from API)
      const mockTransactions: Transaction[] = investmentsData.investments?.map((inv: Investment) => ({
        id: `trans-${inv.id}`,
        type: 'deposit' as const,
        amount: inv.amount,
        description: `Investment in ${inv.project_name}`,
        created_at: inv.created_at,
        balance_after: inv.amount
      })) || []
      
      setTransactions(mockTransactions)
    } catch (error) {
      console.error('Error fetching portfolio:', error)
    }
    
    setLoading(false)
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`
    if (num >= 1000) return `$${(num / 1000).toFixed(1)}K`
    return `$${num.toFixed(2)}`
  }

  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Solar': return <Sun className="h-4 w-4" />
      case 'Wind': return <Wind className="h-4 w-4" />
      case 'Battery': return <Battery className="h-4 w-4" />
      case 'Hydro': return <Droplets className="h-4 w-4" />
      default: return <Zap className="h-4 w-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 bg-green-500/20'
      case 'confirmed': return 'text-blue-400 bg-blue-500/20'
      case 'processing': return 'text-yellow-400 bg-yellow-500/20'
      case 'pending': return 'text-gray-400 bg-gray-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case 'deposit': return <ArrowDown className="h-4 w-4 text-green-400" />
      case 'withdrawal': return <ArrowUp className="h-4 w-4 text-red-400" />
      case 'dividend': return <DollarSign className="h-4 w-4 text-emerald-400" />
      case 'fee': return <Minus className="h-4 w-4 text-gray-400" />
      case 'refund': return <ArrowDown className="h-4 w-4 text-blue-400" />
      default: return <Minus className="h-4 w-4 text-gray-400" />
    }
  }

  // Calculate portfolio metrics
  const portfolioGrowth = summary ? ((summary.total_expected_return / summary.total_invested) * 100) : 0
  const totalValue = summary ? (summary.total_invested + summary.total_expected_return) : 0

  // Filter investments based on search
  const filteredInvestments = investments.filter(inv =>
    inv.project_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    inv.project_type.toLowerCase().includes(searchTerm.toLowerCase())
  )

  if (loading || authLoading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400"></div>
          <p className="mt-4 text-gray-400">Loading portfolio...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Terra Atlas
              </Link>
              <span className="ml-2 text-xs text-gray-400">/ Portfolio</span>
            </div>
            <div className="flex items-center space-x-8">
              <Link href="/explore" className="text-white/70 hover:text-white transition">Explore</Link>
              <Link href="/dashboard" className="text-white/70 hover:text-white transition">Dashboard</Link>
              <Link href="/invest" className="text-white/70 hover:text-white transition">Invest</Link>
              <Link href="/portfolio" className="text-white transition">Portfolio</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Header */}
      <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            Investment Portfolio
          </h1>
          <p className="text-gray-400">Track your energy investments and returns</p>
        </div>

        {/* Portfolio Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-br from-emerald-950/30 to-transparent border border-emerald-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <DollarSign className="h-8 w-8 text-emerald-400" />
              {portfolioGrowth > 0 && (
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-full">
                  +{portfolioGrowth.toFixed(1)}%
                </span>
              )}
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {formatNumber(totalValue)}
            </div>
            <div className="text-sm text-gray-400">Total Portfolio Value</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-br from-cyan-950/30 to-transparent border border-cyan-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <TrendingUp className="h-8 w-8 text-cyan-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {formatNumber(summary?.total_expected_return || 0)}
            </div>
            <div className="text-sm text-gray-400">Expected Returns</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-br from-purple-950/30 to-transparent border border-purple-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <PieChart className="h-8 w-8 text-purple-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {summary?.unique_projects || 0}
            </div>
            <div className="text-sm text-gray-400">Active Projects</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-gradient-to-br from-yellow-950/30 to-transparent border border-yellow-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <Award className="h-8 w-8 text-yellow-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {summary?.avg_return_percentage?.toFixed(1) || 0}%
            </div>
            <div className="text-sm text-gray-400">Avg. Return Rate</div>
          </motion.div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mb-6 bg-gray-900/50 p-1 rounded-lg w-fit">
          {['overview', 'investments', 'transactions', 'performance'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as any)}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                activeTab === tab
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Investment Distribution */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Portfolio Distribution</h3>
              {investments.length > 0 ? (
                <div className="space-y-4">
                  {Object.entries(
                    investments.reduce((acc, inv) => {
                      acc[inv.project_type] = (acc[inv.project_type] || 0) + inv.amount
                      return acc
                    }, {} as Record<string, number>)
                  ).map(([type, amount]) => (
                    <div key={type}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getTypeIcon(type)}
                          <span className="text-sm text-gray-300">{type}</span>
                        </div>
                        <span className="text-sm font-semibold text-white">
                          {formatNumber(amount)} ({((amount / (summary?.total_invested || 1)) * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                          style={{ width: `${(amount / (summary?.total_invested || 1)) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <AlertCircle className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">No investments yet</p>
                  <Link href="/invest" className="inline-flex items-center gap-2 mt-4 text-emerald-400 hover:text-emerald-300">
                    Start Investing <ChevronRight className="h-4 w-4" />
                  </Link>
                </div>
              )}
            </div>

            {/* Recent Activity */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
              {transactions.length > 0 ? (
                <div className="space-y-3">
                  {transactions.slice(0, 5).map((transaction) => (
                    <div key={transaction.id} className="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                      <div className="flex items-center gap-3">
                        {getTransactionIcon(transaction.type)}
                        <div>
                          <p className="text-sm font-medium text-white">{transaction.description}</p>
                          <p className="text-xs text-gray-500">{formatDate(transaction.created_at)}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-sm font-semibold ${
                          transaction.type === 'deposit' || transaction.type === 'dividend' ? 'text-green-400' : 'text-gray-400'
                        }`}>
                          {transaction.type === 'withdrawal' || transaction.type === 'fee' ? '-' : '+'}
                          {formatNumber(transaction.amount)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Activity className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">No activity yet</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'investments' && (
          <div className="space-y-6">
            {/* Search and Filter */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-4">
              <div className="flex items-center gap-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search investments..."
                    className="w-full bg-black/30 border border-gray-700 rounded-lg py-2 pl-10 pr-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                  />
                </div>
                <button className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white transition flex items-center gap-2">
                  <Filter className="h-4 w-4" />
                  Filter
                </button>
                <button className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white transition flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Export
                </button>
              </div>
            </div>

            {/* Investments Table */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl overflow-hidden">
              <table className="w-full">
                <thead className="bg-black/30">
                  <tr>
                    <th className="text-left text-xs font-semibold text-gray-400 p-4">Project</th>
                    <th className="text-left text-xs font-semibold text-gray-400 p-4">Type</th>
                    <th className="text-right text-xs font-semibold text-gray-400 p-4">Amount</th>
                    <th className="text-right text-xs font-semibold text-gray-400 p-4">Expected Return</th>
                    <th className="text-center text-xs font-semibold text-gray-400 p-4">Status</th>
                    <th className="text-left text-xs font-semibold text-gray-400 p-4">Date</th>
                    <th className="text-center text-xs font-semibold text-gray-400 p-4">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredInvestments.map((investment) => (
                    <tr key={investment.id} className="border-t border-gray-800 hover:bg-black/20">
                      <td className="p-4">
                        <div className="font-medium text-white">{investment.project_name}</div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          {getTypeIcon(investment.project_type)}
                          <span className="text-sm text-gray-300">{investment.project_type}</span>
                        </div>
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-sm font-semibold text-white">{formatNumber(investment.amount)}</span>
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-sm font-semibold text-emerald-400">
                          {formatNumber(investment.expected_return)}
                        </span>
                      </td>
                      <td className="p-4 text-center">
                        <span className={`inline-flex px-2 py-1 rounded-full text-xs ${getStatusColor(investment.status)}`}>
                          {investment.status}
                        </span>
                      </td>
                      <td className="p-4">
                        <span className="text-sm text-gray-400">{formatDate(investment.created_at)}</span>
                      </td>
                      <td className="p-4 text-center">
                        <Link
                          href={`/project/${investment.project_id}`}
                          className="text-sm text-emerald-400 hover:text-emerald-300"
                        >
                          View
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              {filteredInvestments.length === 0 && (
                <div className="text-center py-12">
                  <AlertCircle className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">No investments found</p>
                  <Link href="/invest" className="inline-flex items-center gap-2 mt-4 text-emerald-400 hover:text-emerald-300">
                    Browse Projects <ChevronRight className="h-4 w-4" />
                  </Link>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'transactions' && (
          <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Transaction History</h3>
            {transactions.length > 0 ? (
              <div className="space-y-3">
                {transactions.map((transaction) => (
                  <div key={transaction.id} className="flex items-center justify-between p-4 bg-black/30 rounded-lg hover:bg-black/40 transition">
                    <div className="flex items-center gap-4">
                      {getTransactionIcon(transaction.type)}
                      <div>
                        <p className="font-medium text-white">{transaction.description}</p>
                        <p className="text-sm text-gray-500">{formatDate(transaction.created_at)}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`font-semibold ${
                        transaction.type === 'deposit' || transaction.type === 'dividend' ? 'text-green-400' : 'text-gray-400'
                      }`}>
                        {transaction.type === 'withdrawal' || transaction.type === 'fee' ? '-' : '+'}
                        {formatNumber(transaction.amount)}
                      </p>
                      <p className="text-xs text-gray-500">Balance: {formatNumber(transaction.balance_after)}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <Activity className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">No transactions yet</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Performance Chart Placeholder */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Portfolio Performance</h3>
              <div className="h-64 flex items-center justify-center">
                <BarChart3 className="h-12 w-12 text-gray-500" />
                <p className="ml-4 text-gray-400">Performance chart coming soon</p>
              </div>
            </div>

            {/* Metrics */}
            <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Key Metrics</h3>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Invested</span>
                  <span className="font-semibold text-white">{formatNumber(summary?.total_invested || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Expected Returns</span>
                  <span className="font-semibold text-emerald-400">{formatNumber(summary?.total_expected_return || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Average ROI</span>
                  <span className="font-semibold text-cyan-400">{summary?.avg_return_percentage?.toFixed(1) || 0}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Active Projects</span>
                  <span className="font-semibold text-purple-400">{summary?.unique_projects || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Investments</span>
                  <span className="font-semibold text-yellow-400">{summary?.total_investments || 0}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/invest"
            className="p-4 bg-gradient-to-r from-emerald-950/20 to-transparent border border-emerald-500/20 rounded-xl hover:border-emerald-500/40 transition group"
          >
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-white mb-1">New Investment</h4>
                <p className="text-sm text-gray-400">Browse opportunities</p>
              </div>
              <TrendingUp className="h-8 w-8 text-emerald-400 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>

          <button className="p-4 bg-gradient-to-r from-cyan-950/20 to-transparent border border-cyan-500/20 rounded-xl hover:border-cyan-500/40 transition group text-left">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-white mb-1">Withdraw Funds</h4>
                <p className="text-sm text-gray-400">Cash out returns</p>
              </div>
              <DollarSign className="h-8 w-8 text-cyan-400 group-hover:scale-110 transition-transform" />
            </div>
          </button>

          <button className="p-4 bg-gradient-to-r from-purple-950/20 to-transparent border border-purple-500/20 rounded-xl hover:border-purple-500/40 transition group text-left">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-white mb-1">Download Report</h4>
                <p className="text-sm text-gray-400">Export portfolio data</p>
              </div>
              <Download className="h-8 w-8 text-purple-400 group-hover:translate-y-1 transition-transform" />
            </div>
          </button>
        </div>
      </div>
    </div>
  )
}
