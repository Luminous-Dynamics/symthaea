// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { XCircle, ArrowLeft, RefreshCw, HelpCircle } from 'lucide-react'

export default function InvestmentCancelPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Background Animation */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-radial from-red-500/5 via-transparent to-transparent" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 py-12">
        {/* Cancel Header */}
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, type: "spring" }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-red-500/20 to-orange-500/20 mb-6">
            <XCircle className="h-12 w-12 text-red-400" />
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Investment Cancelled
          </h1>
          
          <p className="text-xl text-gray-400">
            Your investment process was cancelled. No charges were made.
          </p>
        </motion.div>

        {/* Information Box */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-8 mb-8"
        >
          <h2 className="text-2xl font-semibold text-white mb-4">What happened?</h2>
          <p className="text-gray-300 mb-4">
            You cancelled the checkout process. This could happen for several reasons:
          </p>
          <ul className="space-y-3 text-gray-400">
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 mt-0.5">•</span>
              <span>You decided to review the project details further</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 mt-0.5">•</span>
              <span>You want to explore other investment opportunities</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 mt-0.5">•</span>
              <span>You need time to consider your investment strategy</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 mt-0.5">•</span>
              <span>Technical issues during the checkout process</span>
            </li>
          </ul>
        </motion.div>

        {/* Next Steps */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 border border-emerald-500/30 rounded-xl p-8 mb-8"
        >
          <h2 className="text-2xl font-semibold text-white mb-4">Ready to Try Again?</h2>
          <p className="text-gray-300 mb-6">
            The project is still available for investment. You can return anytime to complete your investment when you&rsquo;re ready.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <button
              onClick={() => window.history.back()}
              className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all flex items-center justify-center gap-2"
            >
              <RefreshCw className="h-5 w-5" />
              Try Again
            </button>
            <Link
              href="/invest"
              className="px-6 py-3 bg-gray-800 border border-gray-700 rounded-lg font-semibold text-gray-300 hover:bg-gray-700 transition-all flex items-center justify-center gap-2"
            >
              Browse Other Projects
            </Link>
          </div>
        </motion.div>

        {/* Support Section */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="text-center"
        >
          <div className="inline-flex items-center gap-2 text-gray-400 mb-4">
            <HelpCircle className="h-5 w-5" />
            <span>Need Help?</span>
          </div>
          <p className="text-gray-400 mb-6">
            If you experienced any issues during checkout or have questions about investing, our support team is here to help.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="mailto:support@terraatlas.io"
              className="text-cyan-400 hover:text-cyan-300 transition"
            >
              support@terraatlas.io
            </a>
            <span className="text-gray-700">•</span>
            <a
              href="/faq"
              className="text-cyan-400 hover:text-cyan-300 transition"
            >
              Investment FAQ
            </a>
            <span className="text-gray-700">•</span>
            <Link
              href="/"
              className="text-cyan-400 hover:text-cyan-300 transition flex items-center gap-1"
            >
              <ArrowLeft className="h-4 w-4" />
              Go Home
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  )
}