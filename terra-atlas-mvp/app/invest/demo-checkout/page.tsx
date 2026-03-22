'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { CreditCard, Lock, CheckCircle, Loader } from 'lucide-react'

export default function DemoCheckoutPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-gray-950 flex items-center justify-center"><Loader className="animate-spin text-white" /></div>}>
      <DemoCheckoutContent />
    </Suspense>
  )
}

function DemoCheckoutContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [processing, setProcessing] = useState(false)
  const [formData, setFormData] = useState({
    cardNumber: '4242 4242 4242 4242',
    expiry: '12/25',
    cvv: '123',
    name: 'Demo User'
  })

  const sessionId = searchParams.get('session')
  const amount = searchParams.get('amount')
  const projectName = searchParams.get('project')
  const projectId = searchParams.get('projectId')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setProcessing(true)

    // Simulate payment processing
    await new Promise(resolve => setTimeout(resolve, 2000))

    // Redirect to success page with demo session
    router.push(`/invest/success?session_id=${sessionId}&demo=true`)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Header */}
      <div className="bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Terra Atlas Demo Checkout
              </h1>
              <p className="text-xs text-gray-400 mt-1">Test mode - No real charges</p>
            </div>
            <div className="flex items-center gap-2 text-green-400">
              <Lock className="h-5 w-5" />
              <span className="text-sm font-semibold">Secure Demo</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Order Summary */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-900/50 backdrop-blur-md border border-gray-800 rounded-xl p-6"
          >
            <h2 className="text-lg font-semibold mb-4">Order Summary</h2>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Project</span>
                <span className="text-white font-medium">{projectName}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Type</span>
                <span className="text-white">Energy Investment</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Investment Amount</span>
                <span className="text-white font-semibold">${amount}</span>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-gray-800">
              <div className="flex justify-between text-lg">
                <span className="font-semibold">Total</span>
                <span className="font-bold text-emerald-400">${amount}.00 USD</span>
              </div>
            </div>

            {/* Demo Notice */}
            <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <p className="text-sm text-yellow-400">
                <strong>Demo Mode:</strong> This is a simulated checkout. No real payment will be processed.
              </p>
            </div>
          </motion.div>

          {/* Payment Form */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-900/50 backdrop-blur-md border border-gray-800 rounded-xl p-6"
          >
            <h2 className="text-lg font-semibold mb-4">Payment Information</h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Card Number */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Card Number</label>
                <div className="relative">
                  <input
                    type="text"
                    value={formData.cardNumber}
                    onChange={(e) => setFormData({...formData, cardNumber: e.target.value})}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg py-3 px-4 pr-12 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                    placeholder="4242 4242 4242 4242"
                    readOnly
                  />
                  <CreditCard className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
                </div>
                <p className="text-xs text-gray-500 mt-1">Demo card pre-filled</p>
              </div>

              {/* Expiry and CVV */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Expiry Date</label>
                  <input
                    type="text"
                    value={formData.expiry}
                    onChange={(e) => setFormData({...formData, expiry: e.target.value})}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg py-3 px-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                    placeholder="MM/YY"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">CVV</label>
                  <input
                    type="text"
                    value={formData.cvv}
                    onChange={(e) => setFormData({...formData, cvv: e.target.value})}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg py-3 px-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                    placeholder="123"
                    readOnly
                  />
                </div>
              </div>

              {/* Name */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Cardholder Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({...formData, name: e.target.value})}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg py-3 px-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                  placeholder="John Doe"
                />
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={processing}
                className={`
                  w-full py-4 rounded-lg font-semibold text-lg transition-all flex items-center justify-center gap-3
                  ${processing
                    ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white hover:shadow-lg hover:shadow-emerald-500/25'
                  }
                `}
              >
                {processing ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin" />
                    Processing Demo Payment...
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-5 w-5" />
                    Complete Demo Investment
                  </>
                )}
              </button>

              {/* Cancel Link */}
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => router.push('/invest/cancel')}
                  className="text-sm text-gray-400 hover:text-white transition"
                >
                  Cancel and return
                </button>
              </div>
            </form>

            {/* Security Badges */}
            <div className="mt-6 pt-6 border-t border-gray-800">
              <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Lock className="h-3 w-3" />
                  Demo SSL
                </span>
                <span>•</span>
                <span>PCI Demo Compliant</span>
                <span>•</span>
                <span>256-bit Demo Encryption</span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}