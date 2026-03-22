'use client'

import { useState, useEffect, Suspense, useCallback } from 'react'
import dynamic from 'next/dynamic'
import HeroSection from './homepage/sections/HeroSection'
import ProofSection from './homepage/sections/ProofSection'
import LazySection from '@/components/LazySection'
import { StatsResponseSchema, type StatsResponse } from '@/lib/schemas/stats'
import logger from '@/lib/logger'
import { useScrollReveal } from '@/lib/hooks/useScrollReveal'

// Dynamic import for Terra Atlas Globe with 100 real energy sites
const TerraGlobe = dynamic(
  () => import('../components/TerraGlobeWithSites'),
  {
    ssr: false,
    loading: () => (
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-950 via-slate-950 to-emerald-950">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative">
            {/* Breathing Globe Silhouette */}
            <div className="relative w-64 h-64 flex items-center justify-center">
              {/* Outer glow rings - pulsing */}
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-emerald-400/20 via-cyan-400/20 to-blue-400/20 blur-3xl animate-pulse-slow" />
              <div className="absolute inset-4 rounded-full bg-gradient-to-r from-cyan-400/10 via-blue-400/10 to-emerald-400/10 blur-2xl animate-pulse-slower" />

              {/* Main globe circle - breathing effect */}
              <div className="absolute inset-16 rounded-full border-2 border-emerald-400/30 animate-breathing" />
              <div className="absolute inset-20 rounded-full border border-cyan-400/20 animate-breathing animation-delay-200" />
              <div className="absolute inset-24 rounded-full border border-blue-400/10 animate-breathing animation-delay-400" />

              {/* Center dot - gentle pulse */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400 animate-pulse shadow-lg shadow-emerald-400/50" />
              </div>
            </div>

            {/* Elegant loading text */}
            <div className="mt-8 text-center">
              <p className="text-white/70 text-sm font-light tracking-wider mb-2 animate-fade-in">
                Awakening the Planet
              </p>
              <div className="flex items-center justify-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400/60 animate-pulse" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 animate-pulse" style={{ animationDelay: '200ms' }} />
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400/60 animate-pulse" style={{ animationDelay: '400ms' }} />
              </div>
            </div>
          </div>
        </div>

        {/* Add breathing animation */}
        <style jsx>{`
          @keyframes breathing {
            0%, 100% {
              transform: scale(1);
              opacity: 0.3;
            }
            50% {
              transform: scale(1.05);
              opacity: 0.6;
            }
          }
          .animate-breathing {
            animation: breathing 3s ease-in-out infinite;
          }
        `}</style>
      </div>
    )
  }
)

const SectionSkeleton = ({ title }: { title: string }) => (
  <section className="py-20 sm:py-24 md:py-32 px-4 sm:px-6 animate-pulse">
    <div className="max-w-6xl mx-auto border border-white/5 rounded-3xl bg-white/5 backdrop-blur-sm p-10 sm:p-16">
      <div className="h-6 sm:h-8 w-48 sm:w-64 bg-white/10 rounded-full mx-auto mb-6" />
      <p className="h-4 sm:h-5 w-2/3 bg-white/5 rounded-full mx-auto" aria-label={`Loading ${title}`} />
    </div>
  </section>
)

const ImpactCalculatorSection = dynamic(
  () => import('./homepage/sections/ImpactCalculatorSection'),
  {
    loading: () => <SectionSkeleton title="Impact Calculator" />,
  }
)

const PhilosophySection = dynamic(
  () => import('./homepage/sections/PhilosophySection'),
  {
    loading: () => <SectionSkeleton title="Principles" />,
  }
)

const SuccessStoriesSection = dynamic(
  () => import('./homepage/sections/SuccessStoriesSection'),
  {
    loading: () => <SectionSkeleton title="Success Stories" />,
  }
)

const CTASection = dynamic(
  () => import('./homepage/sections/CTASection'),
  {
    loading: () => <SectionSkeleton title="Call to Action" />,
  }
)

const FooterSection = dynamic(
  () => import('./homepage/sections/FooterSection'),
  {
    loading: () => <section className="py-10 sm:py-12 px-4 sm:px-6 border-t border-white/5 animate-pulse" />,
  }
)

// Dynamic import for test panel (only in development)
const IntelligenceTestPanel = dynamic(
  () => import('../components/IntelligenceTestPanel'),
  {
    ssr: false
  }
)

// Prefetch important routes for better navigation performance
if (typeof window !== 'undefined') {
  // Prefetch explore page after initial load
  setTimeout(() => {
    const link = document.createElement('link')
    link.rel = 'prefetch'
    link.href = '/explore'
    document.head.appendChild(link)
  }, 2000)
}

export default function Homepage() {
  const [hoveredPrinciple, setHoveredPrinciple] = useState<number | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [showTestPanel, setShowTestPanel] = useState(false)
  const [showBackToTop, setShowBackToTop] = useState(false)

  useScrollReveal()

  const fetchStats = useCallback(async (signal?: AbortSignal) => {
    try {
      setLoading(true)
      const response = await fetch('/api/stats', {
        signal,
        headers: {
          Accept: 'application/json',
        },
        cache: 'no-store',
      })

      if (!response.ok) {
        logger.error({
          message: 'Stats API error',
          context: { status: response.status },
        })
        setStats(null)
        return
      }

      const data = await response.json()
      const parsed = StatsResponseSchema.safeParse(data)

      if (parsed.success) {
        setStats(parsed.data)
      } else {
        logger.error({
          message: 'Stats schema mismatch',
          context: { issues: parsed.error.issues },
        })
        setStats(null)
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      logger.error('Error fetching stats', error)
      setStats(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchStats(controller.signal)

    if (process.env.NODE_ENV === 'development') {
      setShowTestPanel(true)
    }

    return () => controller.abort()
  }, [fetchStats])

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 800)
    }

    window.addEventListener('scroll', handleScroll)
    handleScroll()

    return () => {
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  // Smooth scroll functions
  const scrollToProof = () => {
    const proofSection = document.getElementById('proof-section')
    if (proofSection) {
      proofSection.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Removed mounted guard - React hydration handles this automatically
  // The page now renders immediately with beautiful content

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      <HeroSection
        onScrollToProof={scrollToProof}
        TerraGlobeComponent={TerraGlobe}
        stats={stats}
        statsLoading={loading}
      />

      <ProofSection />

      <LazySection fallback={<SectionSkeleton title="Impact Calculator" />} rootMargin="300px">
        <Suspense fallback={<SectionSkeleton title="Impact Calculator" />}>
          <ImpactCalculatorSection />
        </Suspense>
      </LazySection>

      <LazySection fallback={<SectionSkeleton title="Seven Principles" />} rootMargin="200px">
        <Suspense fallback={<SectionSkeleton title="Seven Principles" />}>
          <PhilosophySection
            hoveredPrinciple={hoveredPrinciple}
            setHoveredPrinciple={setHoveredPrinciple}
          />
        </Suspense>
      </LazySection>

      <LazySection fallback={<SectionSkeleton title="Success Stories" />} rootMargin="200px">
        <Suspense fallback={<SectionSkeleton title="Success Stories" />}>
          <SuccessStoriesSection />
        </Suspense>
      </LazySection>

      <LazySection fallback={<SectionSkeleton title="Join Terra Atlas" />} rootMargin="150px">
        <Suspense fallback={<SectionSkeleton title="Join Terra Atlas" />}>
          <CTASection />
        </Suspense>
      </LazySection>

      <LazySection
        fallback={<section className="py-10 sm:py-12 px-4 sm:px-6 border-t border-white/5 animate-pulse" />}
        rootMargin="150px"
      >
        <Suspense fallback={<section className="py-10 sm:py-12 px-4 sm:px-6 border-t border-white/5 animate-pulse" />}>
          <FooterSection />
        </Suspense>
      </LazySection>

      {/* Back to Top Button - Floating */}
      {showBackToTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-8 right-8 z-40 group"
          aria-label="Back to top"
        >
          <div className="relative">
            {/* Glow effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full blur-xl opacity-40 group-hover:opacity-60 transition-opacity"></div>

            {/* Button */}
            <div className="relative flex items-center justify-center w-14 h-14 bg-gradient-to-br from-emerald-500/90 to-cyan-500/90 backdrop-blur-lg rounded-full border border-white/20 shadow-2xl shadow-emerald-500/30 hover:scale-110 hover:shadow-emerald-500/50 transition-all duration-300">
              <svg
                className="w-6 h-6 text-white group-hover:-translate-y-1 transition-transform"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </div>
          </div>

          {/* Tooltip */}
          <div className="absolute bottom-full right-0 mb-2 px-3 py-1.5 bg-black/80 backdrop-blur-sm border border-white/20 rounded-lg text-white text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            Back to top
          </div>
        </button>
      )}

      <style jsx>{`
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes scroll-down {
          0%, 100% { transform: translateY(0); opacity: 0.5; }
          50% { transform: translateY(8px); opacity: 1; }
        }

        @keyframes pulse-slow {
          0%, 100% { opacity: 0.05; }
          50% { opacity: 0.15; }
        }

        @keyframes pulse-slower {
          0%, 100% { opacity: 0.03; }
          50% { opacity: 0.1; }
        }

        @keyframes gradient-flow {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }

        @keyframes glow {
          0%, 100% {
            filter: brightness(1) drop-shadow(0 0 20px rgba(16, 185, 129, 0.3));
          }
          50% {
            filter: brightness(1.1) drop-shadow(0 0 30px rgba(16, 185, 129, 0.5));
          }
        }

        /* Scroll Reveal Animation */
        .scroll-reveal {
          opacity: 0;
          transform: translateY(30px);
          transition: opacity 0.6s ease-out, transform 0.6s ease-out;
        }

        .scroll-revealed {
          opacity: 1;
          transform: translateY(0);
        }

        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }

        .animate-float {
          animation: float 6s ease-in-out infinite;
        }

        .animate-glow {
          animation: glow 4s ease-in-out infinite;
        }

        .animate-fade-in-up {
          animation: fade-in-up 1s ease-out;
        }

        .animate-scroll-down {
          animation: scroll-down 2s ease-in-out infinite;
        }

        .animate-pulse-slow {
          animation: pulse-slow 6s ease-in-out infinite;
        }

        .animate-pulse-slower {
          animation: pulse-slower 8s ease-in-out infinite;
        }

        .animate-gradient-flow {
          animation: gradient-flow 5s ease infinite;
          background-size: 200% 200%;
        }

        .animation-delay-50 {
          animation-delay: 50ms;
        }

        .animation-delay-100 {
          animation-delay: 100ms;
        }

        .animation-delay-200 {
          animation-delay: 200ms;
        }

        .animation-delay-300 {
          animation-delay: 300ms;
        }

        .animation-delay-400 {
          animation-delay: 400ms;
        }

        .animation-delay-500 {
          animation-delay: 500ms;
        }

        .animation-delay-600 {
          animation-delay: 600ms;
        }

        .animation-delay-800 {
          animation-delay: 800ms;
        }

        @keyframes shimmer {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        .animate-shimmer {
          animation: shimmer 3s ease-in-out infinite;
        }

        .animate-glow-pulse {
          animation: glow 3s ease-in-out infinite;
        }

        .bg-gradient-radial {
          background: radial-gradient(circle at center, var(--tw-gradient-from), var(--tw-gradient-via), var(--tw-gradient-to));
        }

        .bg-size-200 {
          background-size: 200% 200%;
        }
      `}</style>

      {/* Intelligence Test Panel (Development Only) */}
      {showTestPanel && <IntelligenceTestPanel />}
    </div>
  )
}
