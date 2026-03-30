// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

interface Stats {
  total_projects: number
  active: number
  total_capacity_mw: number
  countries: number
}

export default function LandingPage() {
  const [stats, setStats] = useState<Stats>({
    total_projects: 106000,
    active: 31000,
    total_capacity_mw: 2704000,
    countries: 150
  })

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-b from-black/80 via-black/40 to-transparent backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 sm:h-20">
            <Link href="/" className="flex items-center gap-2 sm:gap-3">
              <div className="relative">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-emerald-400 via-cyan-400 to-purple-400 opacity-80" />
                <div className="absolute inset-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-emerald-400 via-cyan-400 to-purple-400 animate-pulse-slow blur-md" />
              </div>
              <h1 className="text-lg sm:text-xl font-light tracking-wider">
                <span className="text-white/90 font-thin">Terra</span>
                <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent ml-1 font-normal">Atlas</span>
              </h1>
            </Link>

            <div className="hidden md:flex items-center gap-6">
              <Link href="/" className="text-white/60 hover:text-white transition text-sm">Home</Link>
              <Link href="/explore" className="text-white/60 hover:text-white transition text-sm">Explore</Link>
              <Link href="/horizon" className="text-white/60 hover:text-white transition text-sm">Horizon</Link>
              <Link href="/invest" className="px-5 py-2.5 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full text-white text-sm font-medium hover:shadow-lg transition">
                Start Investing
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero: The Hook */}
      <section className="relative pt-32 pb-20 px-4 sm:px-6 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-radial from-amber-500/10 via-transparent to-transparent" />

        <div className="max-w-7xl mx-auto relative">
          <div className="text-center mb-16">
            <div className="mb-6">
              <span className="inline-block px-5 py-2 bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-md border border-amber-400/30 rounded-full text-xs font-light text-amber-200 tracking-wider">
                ⚡ The Energy Revolution
              </span>
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extralight mb-8 leading-tight">
              <span className="block text-white/90 mb-3">What if energy was</span>
              <span className="block bg-gradient-to-r from-amber-300 via-orange-300 to-amber-300 bg-clip-text text-transparent font-light animate-shimmer bg-size-200" style={{ backgroundSize: '200% auto' }}>
                essentially free?
              </span>
            </h1>

            <p className="text-xl sm:text-2xl md:text-3xl text-white/60 max-w-4xl mx-auto mb-8 leading-relaxed">
              In 30+ locations worldwide, renewable energy costs less than $0.02/kWh.
              <span className="block mt-3 text-emerald-300 text-lg sm:text-xl md:text-2xl">
                Now you can invest in it from $10.
              </span>
            </p>
          </div>
        </div>
      </section>

      {/* Proof: Show Don't Tell */}
      <section className="py-20 px-4 sm:px-6 bg-gradient-to-b from-black via-slate-950/50 to-black">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extralight mb-4 text-white/95">
              Not Theory.
              <span className="block mt-2 text-amber-300">Operating Reality.</span>
            </h2>
            <p className="text-lg text-white/60 max-w-2xl mx-auto">
              These locations already prove energy abundance creates prosperity
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-12">
            {/* Iceland */}
            <div className="bg-gradient-to-br from-amber-500/10 to-transparent border border-amber-400/20 rounded-3xl p-8 hover:border-amber-400/40 transition-all duration-500">
              <div className="text-5xl mb-4">🇮🇸</div>
              <h3 className="text-2xl font-light text-white/95 mb-3">Iceland</h3>
              <div className="text-4xl font-bold text-amber-400 mb-2">$0.01/kWh</div>
              <p className="text-white/60 mb-4">
                Geothermal abundance powers aluminum exports worth <span className="text-amber-300 font-semibold">$2B/year</span>.
                They import the bauxite—they export the energy arbitrage.
              </p>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-xs text-emerald-300">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Operating since 1969
              </div>
            </div>

            {/* Chile */}
            <div className="bg-gradient-to-br from-orange-500/10 to-transparent border border-orange-400/20 rounded-3xl p-8 hover:border-orange-400/40 transition-all duration-500">
              <div className="text-5xl mb-4">🇨🇱</div>
              <h3 className="text-2xl font-light text-white/95 mb-3">Chile</h3>
              <div className="text-4xl font-bold text-orange-400 mb-2">$0.013/kWh</div>
              <p className="text-white/60 mb-4">
                Atacama Desert solar radiation is so intense that new farms produce power
                <span className="text-orange-300 font-semibold"> cheaper than coal</span>.
                20GW expansion planned.
              </p>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-xs text-emerald-300">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                2.8GW operational
              </div>
            </div>

            {/* Quebec */}
            <div className="bg-gradient-to-br from-cyan-500/10 to-transparent border border-cyan-400/20 rounded-3xl p-8 hover:border-cyan-400/40 transition-all duration-500">
              <div className="text-5xl mb-4">🇨🇦</div>
              <h3 className="text-2xl font-light text-white/95 mb-3">Quebec</h3>
              <div className="text-4xl font-bold text-cyan-400 mb-2">$0.02/kWh</div>
              <p className="text-white/60 mb-4">
                Hydroelectric surplus attracts <span className="text-cyan-300 font-semibold">$9B in data center investment</span>.
                Tech giants flock here for the cheapest computing on Earth.
              </p>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-xs text-emerald-300">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                37GW capacity
              </div>
            </div>
          </div>

          {/* Pattern Statement */}
          <div className="text-center max-w-4xl mx-auto">
            <div className="bg-gradient-to-br from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-xl border border-amber-400/30 rounded-3xl p-8">
              <p className="text-xl sm:text-2xl text-white/80 leading-relaxed">
                <span className="text-amber-300 font-semibold">The pattern is clear:</span> Where energy is abundant,
                <span className="text-emerald-300"> new economies emerge</span>.
              </p>
              <p className="text-lg text-white/60 mt-4">
                We've identified 30+ locations with similar potential worldwide.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* The Opportunity */}
      <section className="py-20 px-4 sm:px-6 bg-gradient-to-b from-black via-emerald-950/10 to-black">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extralight mb-4 text-white/95">
              The <span className="text-emerald-300">$10 Trillion</span> Opportunity
            </h2>
            <p className="text-lg text-white/60 max-w-3xl mx-auto">
              Global energy transition needs massive investment. Most projects fail. Terra Atlas changes that.
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid md:grid-cols-4 gap-6 mb-16">
            <div className="bg-gradient-to-br from-emerald-500/5 to-transparent border border-emerald-400/20 rounded-2xl p-6 text-center">
              <div className="text-4xl font-bold text-emerald-400 mb-2">
                {(stats.total_projects / 1000).toFixed(0)}K+
              </div>
              <div className="text-white/60 text-sm">Projects Worldwide</div>
            </div>

            <div className="bg-gradient-to-br from-cyan-500/5 to-transparent border border-cyan-400/20 rounded-2xl p-6 text-center">
              <div className="text-4xl font-bold text-cyan-400 mb-2">
                {stats.countries}+
              </div>
              <div className="text-white/60 text-sm">Countries</div>
            </div>

            <div className="bg-gradient-to-br from-purple-500/5 to-transparent border border-purple-400/20 rounded-2xl p-6 text-center">
              <div className="text-4xl font-bold text-purple-400 mb-2">
                {(stats.total_capacity_mw / 1000000).toFixed(1)}TW
              </div>
              <div className="text-white/60 text-sm">Clean Capacity</div>
            </div>

            <div className="bg-gradient-to-br from-amber-500/5 to-transparent border border-amber-400/20 rounded-2xl p-6 text-center">
              <div className="text-4xl font-bold text-amber-400 mb-2">
                $10
              </div>
              <div className="text-white/60 text-sm">Minimum Investment</div>
            </div>
          </div>

          {/* What Makes Us Different */}
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gradient-to-br from-red-500/5 to-transparent border border-red-400/20 rounded-3xl p-8">
              <h3 className="text-2xl font-light text-red-400 mb-6">❌ Traditional Problems</h3>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">•</span>
                  <span>72% of clean energy projects fail or withdraw</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">•</span>
                  <span>$100K+ minimum investment requirements</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">•</span>
                  <span>Accredited investors only</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">•</span>
                  <span>Opaque pricing and hidden fees</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">•</span>
                  <span>Exit = IPO or acquisition (extractive)</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-emerald-500/5 to-transparent border border-emerald-400/20 rounded-3xl p-8">
              <h3 className="text-2xl font-light text-emerald-400 mb-6">✅ Terra Atlas Solution</h3>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">•</span>
                  <span>Focus on energy-abundant locations (&lt;$0.02/kWh)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">•</span>
                  <span>$10 minimum - accessible to everyone</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">•</span>
                  <span>No accreditation required</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">•</span>
                  <span>Transparent pricing, live data, real-time tracking</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">•</span>
                  <span>Dynamic Regenerative Exit → community ownership</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extralight mb-4 text-white/95">
              Simple to Start,
              <span className="block mt-2 text-emerald-300">Powerful in Impact</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="relative">
              <div className="bg-gradient-to-br from-emerald-500/5 to-transparent border border-emerald-400/20 rounded-3xl p-8 h-full hover:border-emerald-400/40 transition-all duration-500">
                <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-emerald-400/20 to-cyan-400/20 border-2 border-emerald-400/40 rounded-2xl mb-6">
                  <span className="text-2xl font-light text-emerald-300">1</span>
                </div>
                <h3 className="text-2xl font-light text-white/95 mb-4">Explore Projects</h3>
                <p className="text-white/60 leading-relaxed mb-4">
                  Browse 106K+ verified energy projects worldwide on our interactive 3D globe.
                  Filter by technology, location, returns, and energy cost.
                </p>
                <Link href="/explore" className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300 font-medium">
                  Explore the Globe
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </Link>
              </div>
            </div>

            {/* Step 2 */}
            <div className="relative">
              <div className="bg-gradient-to-br from-cyan-500/5 to-transparent border border-cyan-400/20 rounded-3xl p-8 h-full hover:border-cyan-400/40 transition-all duration-500">
                <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-cyan-400/20 to-blue-400/20 border-2 border-cyan-400/40 rounded-2xl mb-6">
                  <span className="text-2xl font-light text-cyan-300">2</span>
                </div>
                <h3 className="text-2xl font-light text-white/95 mb-4">Invest From $10</h3>
                <p className="text-white/60 leading-relaxed mb-4">
                  Choose projects that align with your values. Fractional ownership means you can
                  diversify across multiple technologies and regions.
                </p>
                <div className="inline-flex items-center gap-2 text-cyan-400 font-medium">
                  11-14% average returns
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="relative">
              <div className="bg-gradient-to-br from-purple-500/5 to-transparent border border-purple-400/20 rounded-3xl p-8 h-full hover:border-purple-400/40 transition-all duration-500">
                <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-purple-400/20 to-pink-400/20 border-2 border-purple-400/40 rounded-2xl mb-6">
                  <span className="text-2xl font-light text-purple-300">3</span>
                </div>
                <h3 className="text-2xl font-light text-white/95 mb-4">Track & Earn</h3>
                <p className="text-white/60 leading-relaxed mb-4">
                  Monitor your portfolio in real-time. Receive quarterly distributions.
                  Watch your environmental impact grow as communities gain ownership.
                </p>
                <div className="inline-flex items-center gap-2 text-purple-400 font-medium">
                  2.4 tons CO₂ saved per $100
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* The Vision: Living Cities */}
      <section className="py-20 px-4 sm:px-6 bg-gradient-to-b from-black via-amber-950/10 to-black">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extralight mb-4 text-white/95">
              Building
              <span className="block mt-2 bg-gradient-to-r from-amber-300 via-orange-300 to-amber-300 bg-clip-text text-transparent animate-shimmer bg-size-200" style={{ backgroundSize: '200% auto' }}>
                Living Cities
              </span>
            </h2>
            <p className="text-lg text-white/60 max-w-3xl mx-auto">
              Terra Atlas isn't just funding projects—we're enabling 30+ energy-abundant cities where prosperity flows from renewable resources
            </p>
          </div>

          <div className="bg-gradient-to-br from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-xl border border-amber-400/30 rounded-3xl p-8 sm:p-12 mb-12">
            <div className="grid md:grid-cols-3 gap-8 mb-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-amber-400 mb-2">$3B</div>
                <div className="text-white/70 text-sm">builds 300MW infrastructure</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-orange-400 mb-2">$150M</div>
                <div className="text-white/70 text-sm">annual revenue at &lt;$0.02/kWh</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-emerald-400 mb-2">$30B</div>
                <div className="text-white/70 text-sm">city value at maturity</div>
              </div>
            </div>

            <div className="text-center">
              <Link href="/horizon" className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-500 rounded-full text-white font-medium hover:shadow-xl hover:shadow-amber-500/30 transition-all hover:scale-105">
                Explore the Vision
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 sm:px-6 bg-gradient-to-t from-emerald-950/20 to-black">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-extralight mb-6 text-white/95">
            Ready to
            <span className="block mt-2 text-emerald-300">Make an Impact?</span>
          </h2>
          <p className="text-lg text-white/60 mb-10 max-w-2xl mx-auto">
            Join thousands of investors building clean energy infrastructure while earning 11-14% returns.
            Start with as little as $10.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/"
              className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full text-white font-medium text-lg hover:shadow-2xl hover:shadow-emerald-500/30 transition-all hover:scale-105"
            >
              Explore Projects
            </Link>
            <Link
              href="/api"
              className="px-8 py-4 bg-white/5 border border-white/20 rounded-full text-white/90 font-medium text-lg hover:bg-white/10 hover:border-white/30 transition-all"
            >
              Developer API
            </Link>
          </div>

          <p className="text-white/30 text-sm mt-10">
            No account required to explore • SEC-compliant platform • Data privacy guaranteed
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-4 border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="text-center md:text-left">
              <p className="text-white/40 text-sm">© 2025 Terra Atlas</p>
              <p className="text-white/30 text-xs mt-1">Building energy abundance for all</p>
            </div>
            <div className="flex flex-wrap justify-center gap-4 sm:gap-6">
              <Link href="/privacy" className="text-white/40 hover:text-white/60 text-xs sm:text-sm transition">Privacy</Link>
              <Link href="/terms" className="text-white/40 hover:text-white/60 text-xs sm:text-sm transition">Terms</Link>
              <Link href="/contact" className="text-white/40 hover:text-white/60 text-xs sm:text-sm transition">Contact</Link>
              <Link href="/api" className="text-white/40 hover:text-white/60 text-xs sm:text-sm transition">API</Link>
            </div>
          </div>
        </div>
      </footer>

      <style jsx>{`
        @keyframes shimmer {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        .animate-shimmer {
          animation: shimmer 3s ease-in-out infinite;
        }

        .animate-pulse-slow {
          animation: pulse 6s ease-in-out infinite;
        }

        .bg-gradient-radial {
          background: radial-gradient(circle at center, var(--tw-gradient-from), var(--tw-gradient-via), var(--tw-gradient-to));
        }

        .bg-size-200 {
          background-size: 200% 200%;
        }
      `}</style>
    </div>
  )
}
