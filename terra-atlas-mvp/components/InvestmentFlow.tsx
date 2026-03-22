'use client'

import { useState } from 'react'
import { loadStripe } from '@stripe/stripe-js'
import { motion } from 'framer-motion'
import { 
  DollarSign, CreditCard, Shield, Info, TrendingUp, 
  AlertCircle, CheckCircle, Loader, ChevronRight 
} from 'lucide-react'

// Stripe publishable key from environment
const stripePublishableKey = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;
if (!stripePublishableKey) {
  console.warn('WARNING: NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY not configured. Payment functionality disabled.');
}
const stripePromise = stripePublishableKey ? loadStripe(stripePublishableKey) : null;

interface InvestmentFlowProps {
  project: {
    id: number
    name: string
    type: string
    capacity_mw: number
    roi_percentage: number
    min_investment: number
    location?: string
    country?: string
  }
  user?: {
    email?: string
    id?: string
  }
}

export default function InvestmentFlow({ project, user }: InvestmentFlowProps) {
  const [amount, setAmount] = useState<number>(project.min_investment || 10)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [customAmount, setCustomAmount] = useState(false)

  const presetAmounts = [10, 50, 100, 500, 1000, 5000]

  const handleCheckout = async () => {
    try {
      setLoading(true)
      setError(null)

      // Validate amount
      if (amount < project.min_investment) {
        setError(`Minimum investment is $${project.min_investment}`)
        setLoading(false)
        return
      }

      // Create checkout session
      const response = await fetch('/api/stripe/checkout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          projectId: project.id,
          projectName: project.name,
          amount: amount,
          userId: user?.id,
          userEmail: user?.email,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create checkout session')
      }

      // Handle demo mode or real Stripe checkout
      if (data.demoMode) {
        // Demo mode - redirect to demo checkout page
        window.location.href = data.checkoutUrl
      } else {
        // Real Stripe mode
        const stripe = await stripePromise
        if (!stripe) {
          throw new Error('Stripe failed to load')
        }

        const { error: stripeError } = await stripe.redirectToCheckout({
          sessionId: data.sessionId,
        })

        if (stripeError) {
          throw stripeError
        }
      }

    } catch (err: any) {
      console.error('Checkout error:', err)
      setError(err.message || 'An error occurred during checkout')
      setLoading(false)
    }
  }

  const calculateReturn = () => {
    const expectedReturn = amount * (1 + project.roi_percentage / 100)
    return expectedReturn.toFixed(2)
  }

  const calculateMonthlyReturn = () => {
    const annualReturn = amount * (project.roi_percentage / 100)
    return (annualReturn / 12).toFixed(2)
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Test Mode Banner */}
      <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
        <div className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-yellow-400" />
          <div>
            <p className="text-sm font-semibold text-yellow-400">Test Mode Active</p>
            <p className="text-xs text-yellow-400/70">Use card 4242 4242 4242 4242 with any future date</p>
          </div>
        </div>
      </div>

      {/* Project Summary */}
      <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6 mb-6">
        <h3 className="text-lg font-semibold text-white mb-4">Investment Summary</h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Project</span>
            <span className="text-white font-medium">{project.name}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Type</span>
            <span className="text-white">{project.type}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Capacity</span>
            <span className="text-white">{project.capacity_mw} MW</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Annual ROI</span>
            <span className="text-emerald-400 font-semibold">{project.roi_percentage}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Location</span>
            <span className="text-white">{project.location || project.country || 'USA'}</span>
          </div>
        </div>
      </div>

      {/* Amount Selection */}
      <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6 mb-6">
        <h3 className="text-lg font-semibold text-white mb-4">Select Investment Amount</h3>
        
        {/* Preset Amounts */}
        {!customAmount && (
          <div className="grid grid-cols-3 gap-3 mb-4">
            {presetAmounts.map((preset) => (
              <button
                key={preset}
                onClick={() => setAmount(preset)}
                disabled={preset < project.min_investment}
                className={`
                  px-4 py-3 rounded-lg font-semibold transition-all
                  ${amount === preset
                    ? 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white'
                    : preset < project.min_investment
                    ? 'bg-gray-800/50 text-gray-600 cursor-not-allowed'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }
                `}
              >
                ${preset.toLocaleString()}
              </button>
            ))}
          </div>
        )}

        {/* Custom Amount */}
        <div className="space-y-3">
          <button
            onClick={() => setCustomAmount(!customAmount)}
            className="text-sm text-cyan-400 hover:text-cyan-300 transition"
          >
            {customAmount ? '← Back to presets' : 'Enter custom amount'}
          </button>

          {customAmount && (
            <div className="relative">
              <div className="absolute left-3 top-1/2 -translate-y-1/2">
                <DollarSign className="h-5 w-5 text-gray-500" />
              </div>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(Number(e.target.value))}
                min={project.min_investment}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-3 pl-10 pr-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                placeholder={`Minimum: $${project.min_investment}`}
              />
            </div>
          )}

          {amount < project.min_investment && (
            <p className="text-sm text-red-400">
              Minimum investment is ${project.min_investment}
            </p>
          )}
        </div>
      </div>

      {/* Expected Returns */}
      <div className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 border border-emerald-500/30 rounded-xl p-6 mb-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-emerald-400" />
          Expected Returns
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-400 mb-1">After 1 Year</p>
            <p className="text-2xl font-bold text-emerald-400">${calculateReturn()}</p>
            <p className="text-xs text-emerald-400/70">+${(Number(calculateReturn()) - amount).toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400 mb-1">Monthly Income</p>
            <p className="text-2xl font-bold text-cyan-400">${calculateMonthlyReturn()}</p>
            <p className="text-xs text-cyan-400/70">Quarterly distributions</p>
          </div>
        </div>
      </div>

      {/* Security Badges */}
      <div className="flex items-center justify-center gap-6 mb-6">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Shield className="h-4 w-4" />
          <span>Bank-grade security</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <CreditCard className="h-4 w-4" />
          <span>Secure payments</span>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-400" />
            <p className="text-sm text-red-400">{error}</p>
          </div>
        </div>
      )}

      {/* Checkout Button */}
      <button
        onClick={handleCheckout}
        disabled={loading || amount < project.min_investment}
        className={`
          w-full py-4 rounded-lg font-semibold text-lg transition-all flex items-center justify-center gap-3
          ${loading || amount < project.min_investment
            ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
            : 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white hover:shadow-lg hover:shadow-emerald-500/25 transform hover:scale-[1.02]'
          }
        `}
      >
        {loading ? (
          <>
            <Loader className="h-5 w-5 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <CreditCard className="h-5 w-5" />
            Invest ${amount.toLocaleString()} Now
            <ChevronRight className="h-5 w-5" />
          </>
        )}
      </button>

      {/* Terms Notice */}
      <p className="text-xs text-gray-500 text-center mt-4">
        By continuing, you agree to our{' '}
        <a href="/terms" className="text-cyan-400 hover:text-cyan-300">
          Terms of Service
        </a>{' '}
        and{' '}
        <a href="/privacy" className="text-cyan-400 hover:text-cyan-300">
          Privacy Policy
        </a>
      </p>
    </div>
  )
}