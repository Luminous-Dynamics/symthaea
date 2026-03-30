// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import Link from 'next/link'

export default function CTASection() {
  return (
    <section className="py-20 sm:py-24 md:py-32 px-4 sm:px-6 bg-gradient-to-t from-emerald-950/20 to-black">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-extralight mb-4 sm:mb-6 px-2">
          <span className="text-white/80">Ready to</span>
          <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent ml-2 sm:ml-3">
            Make an Impact?
          </span>
        </h2>
        <p className="text-white/50 text-sm sm:text-base md:text-lg mb-10 sm:mb-12 max-w-2xl mx-auto px-2">
          Join thousands of investors building the clean energy infrastructure of tomorrow.
          Start with as little as $10.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 justify-center px-2">
          <Link
            href="/explore"
            prefetch={true}
            className="px-8 sm:px-10 py-3.5 sm:py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full text-white font-medium hover:shadow-lg hover:shadow-emerald-500/25 transition-all transform hover:scale-105 text-center min-h-[48px] flex items-center justify-center text-sm sm:text-base"
          >
            Browse Live Projects
          </Link>
          <Link
            href="/api"
            prefetch={true}
            className="px-8 sm:px-10 py-3.5 sm:py-4 bg-white/5 backdrop-blur border border-white/20 rounded-full text-white/80 hover:bg-white/10 hover:border-white/30 transition-all text-center min-h-[48px] flex items-center justify-center text-sm sm:text-base"
          >
            Developer API Access
          </Link>
        </div>

        <p className="text-white/30 text-xs sm:text-sm mt-10 sm:mt-12 px-4">
          No account required to explore • SEC-compliant investment platform • Your data stays private
        </p>
      </div>
    </section>
  )
}
