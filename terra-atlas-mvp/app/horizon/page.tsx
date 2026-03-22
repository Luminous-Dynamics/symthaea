'use client'

import { useState } from 'react'
import Link from 'next/link'

interface LivingCity {
  name: string
  country: string
  flag: string
  energyType: string
  cost: string
  capacity: string
  status: 'proven' | 'identified' | 'proposed'
  population: string
  description: string
  proof?: string
}

export default function HorizonPage() {
  const [selectedRegion, setSelectedRegion] = useState('all')

  const livingCities: LivingCity[] = [
    // Proven Examples
    {
      name: 'Reykjavik Region',
      country: 'Iceland',
      flag: '🇮🇸',
      energyType: 'Geothermal',
      cost: '$0.01/kWh',
      capacity: '2.6 GW',
      status: 'proven',
      population: '240,000',
      description: 'Geothermal abundance powers aluminum smelting and data centers. $2B annual exports from energy arbitrage.',
      proof: 'Operating since 1969'
    },
    {
      name: 'Atacama Solar Belt',
      country: 'Chile',
      flag: '🇨🇱',
      energyType: 'Solar',
      cost: '$0.013/kWh',
      capacity: '2.8 GW (20 GW planned)',
      status: 'proven',
      population: 'Regional hub',
      description: 'World\'s highest solar radiation. Cheaper than coal, powering mining operations and hydrogen production.',
      proof: 'Online since 2014'
    },
    {
      name: 'Quebec Corridor',
      country: 'Canada',
      flag: '🇨🇦',
      energyType: 'Hydroelectric',
      cost: '$0.02/kWh',
      capacity: '37 GW',
      status: 'proven',
      population: '8.6M',
      description: 'Massive hydroelectric resources attract $9B in data center investment. Cheapest computing on Earth.',
      proof: 'Operating for 60+ years'
    },

    // Identified Opportunities
    {
      name: 'Menengai Crater',
      country: 'Kenya',
      flag: '🇰🇪',
      energyType: 'Geothermal',
      cost: '<$0.02/kWh (est)',
      capacity: '1.6 GW potential',
      status: 'identified',
      population: 'New development',
      description: 'Rift Valley geothermal. Potential to power regional manufacturing and agriculture processing.'
    },
    {
      name: 'Moomba Region',
      country: 'Australia',
      flag: '🇦🇺',
      energyType: 'Solar + Geothermal',
      cost: '<$0.015/kWh (est)',
      capacity: '10+ GW potential',
      status: 'identified',
      population: 'New development',
      description: 'Outback solar abundance plus geothermal. Ideal for energy-intensive manufacturing and green hydrogen.'
    },
    {
      name: 'Northern Cape',
      country: 'South Africa',
      flag: '🇿🇦',
      energyType: 'Solar',
      cost: '<$0.02/kWh (est)',
      capacity: '5+ GW potential',
      status: 'identified',
      population: 'Regional hub',
      description: 'Exceptional solar resources. Opportunity for industrial development and energy export.'
    },
    {
      name: 'Patagonia Corridor',
      country: 'Argentina',
      flag: '🇦🇷',
      energyType: 'Wind + Hydro',
      cost: '<$0.02/kWh (est)',
      capacity: '3+ GW potential',
      status: 'identified',
      population: 'New development',
      description: 'Consistent high winds plus hydroelectric. Perfect for green aluminum and data processing.'
    },
    {
      name: 'Norwegian Fjords',
      country: 'Norway',
      flag: '🇳🇴',
      energyType: 'Hydroelectric',
      cost: '<$0.02/kWh',
      capacity: '32 GW existing',
      status: 'proven',
      population: '5.5M',
      description: 'Massive hydroelectric surplus. Already powering aluminum and tech industries.'
    },
    {
      name: 'Java Geothermal Belt',
      country: 'Indonesia',
      flag: '🇮🇩',
      energyType: 'Geothermal',
      cost: '<$0.03/kWh (est)',
      capacity: '29 GW potential',
      status: 'identified',
      population: 'Regional hubs',
      description: 'Ring of Fire location. Enormous untapped geothermal potential for development.'
    }
  ]

  const filteredCities = selectedRegion === 'all'
    ? livingCities
    : livingCities.filter(city => {
        if (selectedRegion === 'proven') return city.status === 'proven'
        if (selectedRegion === 'identified') return city.status === 'identified'
        return true
      })

  const getStatusBadge = (status: string) => {
    switch(status) {
      case 'proven':
        return { text: 'Proven Model', color: 'from-emerald-500 to-green-500', border: 'border-emerald-500/30' }
      case 'identified':
        return { text: 'Identified Opportunity', color: 'from-amber-500 to-orange-500', border: 'border-amber-500/30' }
      case 'proposed':
        return { text: 'Under Analysis', color: 'from-cyan-500 to-blue-500', border: 'border-cyan-500/30' }
      default:
        return { text: 'Unknown', color: 'from-gray-500 to-gray-600', border: 'border-gray-500/30' }
    }
  }

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
                <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent ml-1 sm:ml-1.5 font-normal">Atlas</span>
              </h1>
            </Link>

            <div className="hidden md:flex items-center gap-6 lg:gap-8">
              <Link href="/" className="text-white/60 hover:text-white transition text-sm">Home</Link>
              <Link href="/explore" className="text-white/60 hover:text-white transition text-sm">Explore</Link>
              <Link href="/horizon" className="text-white transition text-sm">Horizon</Link>
              <Link href="/api" className="text-white/60 hover:text-white transition text-sm">API</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-24 sm:pt-32 pb-16 sm:pb-20 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          {/* Background Glow */}
          <div className="absolute inset-0 bg-gradient-radial from-amber-500/10 via-transparent to-transparent pointer-events-none" />

          <div className="relative text-center mb-12 sm:mb-16">
            <div className="inline-block px-5 py-2 bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-md border border-amber-400/30 rounded-full text-xs font-light text-amber-200 tracking-widest uppercase mb-6 shadow-lg shadow-amber-500/5">
              🏙️ The Terra Lumina Vision
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extralight mb-6 text-white/95 leading-tight">
              Living Cities:
              <span className="block mt-2 bg-gradient-to-r from-amber-300 via-orange-300 to-amber-300 bg-clip-text text-transparent font-light animate-shimmer bg-size-200" style={{ backgroundSize: '200% auto' }}>
                Where Energy is Abundant
              </span>
            </h1>

            <p className="text-lg sm:text-xl md:text-2xl text-white/60 max-w-4xl mx-auto leading-relaxed font-light px-4">
              In 30+ locations worldwide, renewable energy costs less than $0.02/kWh—essentially free.
              <span className="block mt-3 text-emerald-300">We&rsquo;re building cities there.</span>
            </p>
          </div>

          {/* The Model */}
          <div className="max-w-5xl mx-auto mb-16 sm:mb-20">
            <div className="bg-gradient-to-br from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-xl border border-amber-400/30 rounded-3xl p-8 sm:p-12">
              <h2 className="text-2xl sm:text-3xl font-light text-amber-300 mb-8 text-center">The Living Cities Model</h2>

              {/* Flow Diagram */}
              <div className="grid md:grid-cols-4 gap-4 md:gap-6 mb-8">
                <div className="bg-black/40 border border-amber-400/20 rounded-2xl p-6 text-center">
                  <div className="text-3xl sm:text-4xl font-bold text-amber-400 mb-2">$3B</div>
                  <div className="text-sm text-white/70">Initial Investment</div>
                  <div className="text-xs text-white/50 mt-2">Build renewable infrastructure</div>
                </div>

                <div className="hidden md:flex items-center justify-center">
                  <div className="text-3xl text-amber-400/40">→</div>
                </div>

                <div className="bg-black/40 border border-orange-400/20 rounded-2xl p-6 text-center">
                  <div className="text-3xl sm:text-4xl font-bold text-orange-400 mb-2">300MW</div>
                  <div className="text-sm text-white/70">Power Generation</div>
                  <div className="text-xs text-white/50 mt-2">At &lt;$0.02/kWh cost</div>
                </div>

                <div className="hidden md:flex items-center justify-center">
                  <div className="text-3xl text-orange-400/40">→</div>
                </div>

                <div className="bg-black/40 border border-emerald-400/20 rounded-2xl p-6 text-center">
                  <div className="text-3xl sm:text-4xl font-bold text-emerald-400 mb-2">$30B</div>
                  <div className="text-sm text-white/70">City Value</div>
                  <div className="text-xs text-white/50 mt-2">At maturity (20 years)</div>
                </div>
              </div>

              {/* Value Proposition */}
              <div className="grid md:grid-cols-2 gap-6 pt-8 border-t border-amber-400/20">
                <div>
                  <h3 className="text-xl text-amber-300 mb-4">For Investors</h3>
                  <ul className="space-y-2 text-sm text-white/70">
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400 mt-0.5">✓</span>
                      <span>15-25% IRR from energy sales</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400 mt-0.5">✓</span>
                      <span>Real estate appreciation</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400 mt-0.5">✓</span>
                      <span>Community ownership transition</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400 mt-0.5">✓</span>
                      <span>Tax-optimized structure</span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-xl text-emerald-300 mb-4">For Communities</h3>
                  <ul className="space-y-2 text-sm text-white/70">
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-0.5">✓</span>
                      <span>Energy independence</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-0.5">✓</span>
                      <span>Local job creation (1000+ jobs per city)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-0.5">✓</span>
                      <span>Eventual community ownership</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-0.5">✓</span>
                      <span>Sustainable prosperity</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Filter Section */}
      <section className="px-4 sm:px-6 mb-12">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-wrap justify-center gap-3 sm:gap-4">
            <button
              onClick={() => setSelectedRegion('all')}
              className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all ${
                selectedRegion === 'all'
                  ? 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white shadow-lg shadow-emerald-500/25'
                  : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white border border-white/10'
              }`}
            >
              All Locations ({livingCities.length})
            </button>
            <button
              onClick={() => setSelectedRegion('proven')}
              className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all ${
                selectedRegion === 'proven'
                  ? 'bg-gradient-to-r from-emerald-500 to-green-500 text-white shadow-lg shadow-emerald-500/25'
                  : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white border border-white/10'
              }`}
            >
              ✓ Proven Models
            </button>
            <button
              onClick={() => setSelectedRegion('identified')}
              className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all ${
                selectedRegion === 'identified'
                  ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg shadow-amber-500/25'
                  : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white border border-white/10'
              }`}
            >
              🔍 Identified Opportunities
            </button>
          </div>
        </div>
      </section>

      {/* Living Cities Grid */}
      <section className="px-4 sm:px-6 pb-20 sm:pb-32">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
            {filteredCities.map((city, index) => {
              const badge = getStatusBadge(city.status)
              return (
                <div
                  key={index}
                  className="group relative bg-gradient-to-br from-white/5 to-white/[0.02] backdrop-blur-lg border border-white/10 rounded-3xl p-6 sm:p-8 hover:border-white/20 hover:shadow-2xl hover:shadow-white/5 transition-all duration-500 hover:-translate-y-1"
                >
                  {/* Status Badge */}
                  <div className={`absolute top-4 right-4 px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${badge.color} border ${badge.border}`}>
                    {badge.text}
                  </div>

                  {/* Flag & Title */}
                  <div className="text-5xl mb-4">{city.flag}</div>
                  <h3 className="text-2xl font-light text-white/95 mb-1">{city.name}</h3>
                  <div className="text-sm text-white/50 mb-4">{city.country}</div>

                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-3 mb-6">
                    <div className="bg-black/40 rounded-xl p-3">
                      <div className="text-xs text-white/50 mb-1">Energy Cost</div>
                      <div className="text-lg font-semibold text-amber-400">{city.cost}</div>
                    </div>
                    <div className="bg-black/40 rounded-xl p-3">
                      <div className="text-xs text-white/50 mb-1">Capacity</div>
                      <div className="text-lg font-semibold text-emerald-400 truncate">{city.capacity}</div>
                    </div>
                    <div className="bg-black/40 rounded-xl p-3">
                      <div className="text-xs text-white/50 mb-1">Type</div>
                      <div className="text-sm font-medium text-cyan-400">{city.energyType}</div>
                    </div>
                    <div className="bg-black/40 rounded-xl p-3">
                      <div className="text-xs text-white/50 mb-1">Population</div>
                      <div className="text-sm font-medium text-purple-400">{city.population}</div>
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-sm text-white/60 leading-relaxed mb-4">
                    {city.description}
                  </p>

                  {/* Proof Badge (for proven cities) */}
                  {city.proof && (
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-xs text-emerald-300">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      {city.proof}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-4 sm:px-6 pb-20 sm:pb-32">
        <div className="max-w-5xl mx-auto">
          <div className="bg-gradient-to-br from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-xl border border-amber-400/30 rounded-3xl p-8 sm:p-12 text-center">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-light mb-6 text-white/95">
              Ready to Build the Future?
            </h2>
            <p className="text-lg sm:text-xl text-white/60 mb-8 max-w-3xl mx-auto leading-relaxed">
              Terra Atlas makes investing in Living Cities accessible from $10. Fund the infrastructure, earn returns, and help create energy-abundant communities worldwide.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/explore"
                className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full text-white font-medium text-lg hover:shadow-2xl hover:shadow-emerald-500/30 transition-all hover:scale-105"
              >
                Explore Projects
              </Link>
              <Link
                href="/"
                className="px-8 py-4 bg-white/5 border border-white/20 rounded-full text-white/90 font-medium text-lg hover:bg-white/10 hover:border-white/30 transition-all"
              >
                Learn How It Works
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 sm:py-12 px-4 sm:px-6 border-t border-white/5">
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
