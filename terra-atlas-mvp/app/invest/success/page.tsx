// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useEffect, useState, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  CheckCircle, Download, ArrowRight, Home, Briefcase, 
  TrendingUp, Calendar, Mail, Phone, Shield, Award
} from 'lucide-react'

interface SessionData {
  status: string
  customerEmail: string
  amountTotal: number
  metadata: {
    projectId: string
    userId?: string
    investmentType?: string
  }
  testMode: boolean
}

export default function InvestmentSuccessPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-gray-950 flex items-center justify-center text-white">Loading...</div>}>
      <InvestmentSuccessContent />
    </Suspense>
  )
}

function InvestmentSuccessContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [loading, setLoading] = useState(true)
  
  const sessionId = searchParams.get('session_id')

  useEffect(() => {
    if (sessionId) {
      fetchSessionData()
    } else {
      setLoading(false)
    }
  }, [sessionId])

  const fetchSessionData = async () => {
    try {
      const response = await fetch(`/api/stripe/checkout?session_id=${sessionId}`)
      const data = await response.json()
      
      if (response.ok) {
        setSessionData(data)
      }
    } catch (error) {
      console.error('Error fetching session data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400"></div>
      </div>
    )
  }

  const amount = sessionData?.amountTotal ? sessionData.amountTotal / 100 : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Success Animation Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-radial from-emerald-500/10 via-transparent to-transparent animate-pulse-slow" />
        <div className="absolute inset-0 bg-gradient-radial from-cyan-500/10 via-transparent to-transparent animate-pulse-slower" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 py-12">
        {/* Success Header */}
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, type: "spring" }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-emerald-500 to-cyan-500 mb-6">
            <CheckCircle className="h-12 w-12 text-white" />
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            Investment Successful!
          </h1>
          
          <p className="text-xl text-gray-400">
            {sessionData?.testMode && (
              <span className="inline-block px-3 py-1 bg-yellow-500/20 text-yellow-400 text-sm rounded-full mb-2">
                TEST MODE - No real payment processed
              </span>
            )}
          </p>
        </motion.div>

        {/* Investment Details */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-8 mb-8"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Investment Details</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-400 mb-1">Amount Invested</p>
                <p className="text-3xl font-bold text-emerald-400">${amount.toFixed(2)}</p>
              </div>
              
              <div>
                <p className="text-sm text-gray-400 mb-1">Investor Email</p>
                <p className="text-lg text-white">{sessionData?.customerEmail || 'Not provided'}</p>
              </div>
              
              <div>
                <p className="text-sm text-gray-400 mb-1">Project ID</p>
                <p className="text-lg text-white">#{sessionData?.metadata?.projectId || 'N/A'}</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-400 mb-1">Payment Status</p>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <p className="text-lg text-green-400">Confirmed</p>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-gray-400 mb-1">Transaction ID</p>
                <p className="text-sm font-mono text-gray-500">{sessionId?.substring(0, 20)}...</p>
              </div>
              
              <div>
                <p className="text-sm text-gray-400 mb-1">Investment Type</p>
                <p className="text-lg text-white">{sessionData?.metadata?.investmentType || 'Energy Project'}</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Next Steps */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 border border-emerald-500/30 rounded-xl p-8 mb-8"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">What Happens Next?</h2>
          
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-emerald-400">1</span>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-1">Email Confirmation</h3>
                <p className="text-sm text-gray-400">
                  You'll receive a detailed confirmation email with your investment certificate within 24 hours
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-cyan-400">2</span>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-1">Portfolio Update</h3>
                <p className="text-sm text-gray-400">
                  Your investment will appear in your portfolio dashboard within minutes
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-purple-400">3</span>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-1">Quarterly Returns</h3>
                <p className="text-sm text-gray-400">
                  You'll start receiving quarterly distributions after the project becomes operational
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Security & Trust Badges */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12"
        >
          <div className="text-center">
            <Shield className="h-8 w-8 text-emerald-400 mx-auto mb-2" />
            <p className="text-xs text-gray-400">Secured Investment</p>
          </div>
          <div className="text-center">
            <Award className="h-8 w-8 text-cyan-400 mx-auto mb-2" />
            <p className="text-xs text-gray-400">SEC Compliant</p>
          </div>
          <div className="text-center">
            <TrendingUp className="h-8 w-8 text-purple-400 mx-auto mb-2" />
            <p className="text-xs text-gray-400">Guaranteed Returns</p>
          </div>
          <div className="text-center">
            <Calendar className="h-8 w-8 text-yellow-400 mx-auto mb-2" />
            <p className="text-xs text-gray-400">Quarterly Payouts</p>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.9 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <Link
            href="/portfolio"
            className="px-8 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all flex items-center justify-center gap-2"
          >
            <Briefcase className="h-5 w-5" />
            View Portfolio
            <ArrowRight className="h-4 w-4" />
          </Link>
          
          <Link
            href="/invest"
            className="px-8 py-3 bg-gray-800 border border-gray-700 rounded-lg font-semibold text-gray-300 hover:bg-gray-700 transition-all flex items-center justify-center gap-2"
          >
            Browse More Projects
          </Link>
          
          <Link
            href="/"
            className="px-8 py-3 bg-gray-800 border border-gray-700 rounded-lg font-semibold text-gray-300 hover:bg-gray-700 transition-all flex items-center justify-center gap-2"
          >
            <Home className="h-5 w-5" />
            Go Home
          </Link>
        </motion.div>

        {/* Support Section */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>Have questions? Contact our support team</p>
          <div className="flex items-center justify-center gap-4 mt-2">
            <a href="mailto:support@terraatlas.io" className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1">
              <Mail className="h-4 w-4" />
              support@terraatlas.io
            </a>
            <span className="text-gray-700">•</span>
            <a href="tel:1-800-TERRA" className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1">
              <Phone className="h-4 w-4" />
              1-800-TERRA
            </a>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.1; }
          50% { opacity: 0.3; }
        }
        
        @keyframes pulse-slower {
          0%, 100% { opacity: 0.05; }
          50% { opacity: 0.2; }
        }
        
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
        
        .animate-pulse-slower {
          animation: pulse-slower 6s ease-in-out infinite;
        }
        
        .bg-gradient-radial {
          background: radial-gradient(circle at center, var(--tw-gradient-from), var(--tw-gradient-via), var(--tw-gradient-to));
        }
      `}</style>
    </div>
  )
}