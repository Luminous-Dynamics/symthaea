// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { ComponentType, useMemo, useState, useEffect } from 'react'
import Link from 'next/link'
import EnhancedStatCard from '@/components/EnhancedStatCard'
import type { StatsResponse } from '@/lib/schemas/stats'

const navLinks = [
  { href: '/explore', label: 'Explore', prefetch: true },
  { href: '/horizon', label: 'Horizon', prefetch: false },
  { href: '/api', label: 'API', prefetch: true },
]

interface TerraGlobeProps {
  minimal?: boolean
  showLegend?: boolean
  showStats?: boolean
}

interface HeroSectionProps {
  onScrollToProof: () => void
  TerraGlobeComponent: ComponentType<TerraGlobeProps>
  stats?: StatsResponse | null
  statsLoading?: boolean
}

const formatCompactNumber = (value: number, maximumFractionDigits = 1) =>
  Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits }).format(value)

export default function HeroSection({
  onScrollToProof,
  TerraGlobeComponent,
  stats,
  statsLoading,
}: HeroSectionProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  // Prevent body scroll when menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [isMobileMenuOpen])

  const heroStats = useMemo(() => {
    const overview = stats?.overview
    const entries = [
      {
        label: 'In queue',
        icon: '🌍',
        fallback: '79K',
        value:
          overview && typeof overview.total_projects === 'number'
            ? `${formatCompactNumber(overview.total_projects)}`
            : null,
      },
      {
        label: 'Capacity',
        icon: '⚡',
        fallback: '3 TW',
        value:
          overview && typeof overview.total_capacity_gw === 'number'
            ? `${formatCompactNumber(overview.total_capacity_gw, 0)} GW`
            : null,
      },
      {
        label: 'Est. jobs',
        icon: '👷',
        fallback: '1.4M',
        value:
          overview && typeof overview.estimated_jobs === 'number'
            ? `${formatCompactNumber(overview.estimated_jobs)}`
            : null,
      },
      {
        label: 'Est. homes',
        icon: '🏡',
        fallback: '2.2B',
        value:
          overview && typeof overview.estimated_homes_powered === 'number'
            ? `${formatCompactNumber(overview.estimated_homes_powered)}`
            : null,
      },
    ]

    return entries.map((entry) => ({
      ...entry,
      display: entry.value ?? entry.fallback,
      isLoading: statsLoading,
    }))
  }, [stats, statsLoading])

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-b from-black/80 via-black/40 to-transparent backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 sm:h-20">
            <Link href="/" className="flex items-center gap-2 sm:gap-3 min-h-[44px] min-w-[44px]">
              <div className="relative">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-emerald-400 via-cyan-400 to-purple-400 opacity-80" />
                <div className="absolute inset-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-emerald-400 via-cyan-400 to-purple-400 animate-pulse-slow blur-md" />
              </div>
              <h1 className="text-lg sm:text-xl font-light tracking-wider">
                <span className="text-white/90 font-thin">Terra</span>
                <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent ml-1 sm:ml-1.5 font-normal">Atlas</span>
              </h1>
            </Link>

            <div className="hidden md:flex items-center gap-6 lg:gap-8">
              <Link href="/explore" prefetch className="relative text-white/60 hover:text-white transition-all text-sm tracking-wide group min-h-[44px] flex items-center">
                Explore
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-emerald-400 to-cyan-400 group-hover:w-full transition-all duration-300" />
              </Link>
              <Link href="/horizon" prefetch={false} className="relative text-white/60 hover:text-white transition-all text-sm tracking-wide group min-h-[44px] flex items-center">
                Horizon
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-emerald-400 to-cyan-400 group-hover:w-full transition-all duration-300" />
              </Link>
              <Link href="/api" prefetch className="relative text-white/60 hover:text-white transition-all text-sm tracking-wide group min-h-[44px] flex items-center">
                API
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-emerald-400 to-cyan-400 group-hover:w-full transition-all duration-300" />
              </Link>
              <Link href="/invest" prefetch={false} className="relative overflow-hidden px-5 py-2.5 rounded-full text-white/90 text-sm tracking-wide group">
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/20 via-cyan-500/20 to-purple-500/20 group-hover:from-emerald-500/30 group-hover:via-cyan-500/30 group-hover:to-purple-500/30 transition-all" />
                <div className="absolute inset-0 border border-white/20 rounded-full group-hover:border-white/30 transition-all" />
                <span className="relative">Start Investing</span>
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden relative w-12 h-12 flex items-center justify-center"
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={isMobileMenuOpen}
            >
              <div className="relative w-6 h-5 flex flex-col justify-between">
                <span className={`block h-0.5 bg-white transition-all duration-300 origin-center ${
                  isMobileMenuOpen ? 'rotate-45 translate-y-2' : ''
                }`} />
                <span className={`block h-0.5 bg-white transition-all duration-300 ${
                  isMobileMenuOpen ? 'opacity-0' : ''
                }`} />
                <span className={`block h-0.5 bg-white transition-all duration-300 origin-center ${
                  isMobileMenuOpen ? '-rotate-45 -translate-y-2' : ''
                }`} />
              </div>
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Mobile Menu Panel */}
      <div className={`fixed top-16 sm:top-20 left-0 right-0 bottom-0 bg-black/95 backdrop-blur-xl z-40 md:hidden transform transition-transform duration-300 ease-out ${
        isMobileMenuOpen ? 'translate-x-0' : 'translate-x-full'
      }`}>
        <div className="flex flex-col h-full p-6">
          <div className="flex-1 space-y-2">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                prefetch={link.prefetch}
                onClick={() => setIsMobileMenuOpen(false)}
                className="block px-4 py-4 rounded-xl text-lg font-light text-white/70 hover:text-white hover:bg-white/5 transition-all"
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Mobile CTA */}
          <div className="pt-6 border-t border-white/10">
            <Link
              href="/invest"
              prefetch={false}
              className="block w-full px-6 py-4 rounded-xl text-center text-white font-medium bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 transition-all"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Start Investing
            </Link>
          </div>
        </div>
      </div>

      <section className="relative h-screen overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-950 via-slate-950 to-emerald-950 opacity-60" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/10 via-transparent to-transparent animate-pulse-slower" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,_var(--tw-gradient-stops))] from-emerald-900/10 via-transparent to-transparent animate-pulse-slow" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-900/5 via-transparent to-transparent" />
        </div>

        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative w-full h-full">
            <TerraGlobeComponent minimal={true} />
          </div>
        </div>

        <div className="absolute inset-0 bg-gradient-radial from-transparent via-transparent to-black/40 pointer-events-none" />
        <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/80 pointer-events-none" />

        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-radial from-emerald-500/5 via-transparent to-transparent animate-pulse-slow" />
          <div className="absolute inset-0 bg-gradient-radial from-cyan-500/5 via-transparent to-transparent animate-pulse-slower" />
        </div>

        <div className="relative z-10 h-full flex flex-col pointer-events-none">
          <div className="flex-1 flex flex-col justify-start pt-16 sm:pt-20 px-4 sm:px-6">
            <div className="text-center max-w-5xl mx-auto">
              <div className="mb-6 sm:mb-8 md:mb-10 animate-fade-in-up">
                <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-extralight text-white/95 mb-3 px-2 leading-tight drop-shadow-[0_0_30px_rgba(255,255,255,0.3)]">
                  What if energy was
                  <span className="block sm:inline sm:ml-3 mt-2 sm:mt-0 bg-gradient-to-r from-amber-300 via-orange-300 to-amber-300 bg-clip-text text-transparent font-light animate-shimmer bg-size-200 drop-shadow-[0_0_60px_rgba(251,191,36,0.6)]" style={{ backgroundSize: '200% auto' }}>
                    essentially free?
                  </span>
                </h2>
              </div>

              <div className="mb-6 sm:mb-8 animate-fade-in-up animation-delay-50">
                <span className="inline-block px-5 sm:px-7 py-2 sm:py-2.5 bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-md border border-amber-400/30 rounded-full text-xs sm:text-sm font-light text-amber-200 tracking-wider shadow-lg shadow-amber-500/10 uppercase">
                  Proof: Investors are already doing it.
                </span>
              </div>

              <div className="flex flex-wrap justify-center gap-2 sm:gap-3 mb-8 sm:mb-10 animate-fade-in-up animation-delay-200 px-2">
                {heroStats.map((stat, index) => (
                  <EnhancedStatCard
                    key={stat.label}
                    value={stat.display}
                    label={stat.label}
                    icon={stat.icon}
                    index={index}
                    isLoading={stat.isLoading}
                  />
                ))}
              </div>

              <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center items-stretch sm:items-center animate-fade-in-up animation-delay-500 px-2">
                <Link
                  href="/explore"
                  prefetch
                  className="group relative overflow-visible px-8 sm:px-12 py-4 sm:py-5 rounded-2xl text-white font-light text-base sm:text-lg text-center min-h-[56px] flex items-center justify-center transition-all duration-500 hover:-translate-y-2"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 rounded-2xl blur-xl opacity-40 group-hover:opacity-60 transition-opacity duration-500" />
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 rounded-2xl blur-md opacity-50 group-hover:opacity-70 transition-opacity duration-500" />
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 rounded-2xl" />
                  <div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-700 translate-x-[-200%] group-hover:translate-x-[200%]"
                    style={{ transition: 'transform 1s ease-out, opacity 0.7s' }}
                  />
                  <span className="relative flex items-center gap-2.5 drop-shadow-lg">
                    <span className="text-xl opacity-90">🌱</span>
                    <span className="font-medium">Explore Energy Projects</span>
                    <svg className="w-5 h-5 group-hover:translate-x-1.5 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                  </span>
                </Link>

                <Link
                  href="/api"
                  prefetch
                  className="group relative overflow-hidden px-8 sm:px-12 py-4 sm:py-5 rounded-2xl text-white/90 font-light text-base sm:text-lg text-center min-h-[56px] flex items-center justify-center transition-all duration-500 hover:-translate-y-1"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/20 via-cyan-400/20 to-blue-400/20 rounded-2xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-white/5 to-white/10 backdrop-blur-xl rounded-2xl border border-white/20 group-hover:border-white/40 transition-all duration-300" />
                  <div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 translate-x-[-200%] group-hover:translate-x-[200%]"
                    style={{ transition: 'transform 1.2s ease-out' }}
                  />
                  <span className="relative flex items-center gap-2.5">
                    <span className="text-xl opacity-80">🔌</span>
                    Integrate with API
                    <svg className="w-5 h-5 opacity-70 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </span>
                </Link>
              </div>

              <div className="mt-10 sm:mt-12 flex flex-wrap justify-center gap-3 sm:gap-4 animate-fade-in-up animation-delay-600 px-2">
                <span className="group relative flex items-center gap-2.5 px-4 sm:px-5 py-3 bg-gradient-to-br from-emerald-500/5 via-emerald-500/3 to-transparent backdrop-blur-xl rounded-full text-xs sm:text-sm text-emerald-200/90 font-light whitespace-nowrap transition-all duration-300 hover:scale-105 border border-emerald-400/20 hover:border-emerald-400/40 cursor-default">
                  <span className="relative w-2 h-2">
                    <span className="absolute inset-0 bg-emerald-400 rounded-full animate-pulse" />
                    <span className="absolute inset-0 bg-emerald-400 rounded-full animate-ping opacity-75" />
                  </span>
                  <span className="relative">Live data streams</span>
                </span>
                <span className="group relative flex items-center gap-2.5 px-4 sm:px-5 py-3 bg-gradient-to-br from-cyan-500/5 via-cyan-500/3 to-transparent backdrop-blur-xl rounded-full text-xs sm:text-sm text-cyan-200/90 font-light whitespace-nowrap transition-all duration-300 hover:scale-105 border border-cyan-400/20 hover:border-cyan-400/40 cursor-default">
                  <svg className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                  <span className="hidden xs:inline">Bank-level security</span>
                  <span className="xs:hidden">Secure</span>
                </span>
                <span className="group relative flex items-center gap-2.5 px-4 sm:px-5 py-3 bg-gradient-to-br from-blue-500/5 via-blue-500/3 to-transparent backdrop-blur-xl rounded-full text-xs sm:text-sm text-blue-200/90 font-light whitespace-nowrap transition-all duration-300 hover:scale-105 border border-blue-400/20 hover:border-blue-400/40 cursor-default">
                  <svg className="w-4 h-4 text-blue-400 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="relative">Regulated & compliant</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={onScrollToProof}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 group cursor-pointer hover:scale-110 transition-transform duration-300 pointer-events-auto"
          aria-label="Scroll to proof section"
        >
          <div className="flex flex-col items-center gap-2">
            <span className="text-white/60 text-xs sm:text-sm tracking-wider uppercase font-light group-hover:text-white/80 transition-colors">
              Scroll to see proof
            </span>
            <div className="w-6 h-10 border border-white/30 rounded-full p-1 group-hover:border-amber-400/50 transition-colors">
              <div className="w-1 h-3 bg-white/50 rounded-full mx-auto animate-scroll-down group-hover:bg-amber-400/80"></div>
            </div>
          </div>
        </button>
      </section>
    </>
  )
}
