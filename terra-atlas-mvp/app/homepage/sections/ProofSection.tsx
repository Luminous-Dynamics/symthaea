// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export default function ProofSection() {
  return (
    <section id="proof-section" className="py-20 sm:py-24 md:py-32 px-4 sm:px-6 bg-gradient-to-b from-black via-slate-950/50 to-black">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12 sm:mb-16 md:mb-20">
          <div className="inline-block px-5 py-2 bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-md border border-amber-400/20 rounded-full text-xs font-light text-amber-200/90 tracking-widest uppercase mb-6 shadow-lg shadow-amber-500/5">
            ⚡ Not Theory. Operating Reality.
          </div>
          <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-extralight mb-5 text-white/95 leading-tight px-2">
            Energy Abundance
            <span className="block mt-2 bg-gradient-to-r from-amber-300 via-orange-300 to-amber-300 bg-clip-text text-transparent font-light animate-shimmer bg-size-200" style={{ backgroundSize: '200% auto' }}>
              Already Exists
            </span>
          </h2>
          <p className="text-base sm:text-lg md:text-xl text-white/50 max-w-3xl mx-auto leading-relaxed font-light px-4">
            In these locations, renewable energy costs less than $0.02/kWh—essentially free. The results speak for themselves.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 sm:gap-8 mb-12 sm:mb-16">
          <div className="group relative scroll-reveal bg-gradient-to-br from-amber-500/5 to-transparent backdrop-blur-lg border border-amber-400/20 rounded-3xl p-6 sm:p-8 hover:from-amber-500/10 hover:border-amber-400/40 hover:shadow-xl hover:shadow-amber-500/10 transition-all duration-500">
            <div className="text-5xl mb-4">🇮🇸</div>
            <h3 className="text-2xl font-light text-white/95 mb-3">Iceland&apos;s Aluminum Economy</h3>
            <p className="text-white/60 text-sm sm:text-base leading-relaxed mb-4">
              Iceland exports aluminum—not because they have bauxite (they import it), but because smelting requires massive electricity and theirs costs <span className="text-amber-300 font-semibold">$0.01/kWh</span>.
            </p>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-amber-400">$2B</span>
              <span className="text-sm text-white/50">annual revenue from energy arbitrage</span>
            </div>
          </div>

          <div className="group relative scroll-reveal animation-delay-200 bg-gradient-to-br from-orange-500/5 to-transparent backdrop-blur-lg border border-orange-400/20 rounded-3xl p-6 sm:p-8 hover:from-orange-500/10 hover:border-orange-400/40 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-500">
            <div className="text-5xl mb-4">🇨🇱</div>
            <h3 className="text-2xl font-light text-white/95 mb-3">Chile&apos;s Solar Revolution</h3>
            <p className="text-white/60 text-sm sm:text-base leading-relaxed mb-4">
              The Atacama Desert has the world&apos;s highest solar radiation. New solar farms produce at <span className="text-orange-300 font-semibold">$0.013/kWh</span>—cheaper than coal.
            </p>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-orange-400">2.8GW</span>
                <span className="text-sm text-white/50">installed capacity</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-orange-300">20GW</span>
                <span className="text-xs text-white/40">planned expansion</span>
              </div>
            </div>
          </div>

          <div className="group relative scroll-reveal animation-delay-400 bg-gradient-to-br from-cyan-500/5 to-transparent backdrop-blur-lg border border-cyan-400/20 rounded-3xl p-6 sm:p-8 hover:from-cyan-500/10 hover:border-cyan-400/40 hover:shadow-xl hover:shadow-cyan-500/10 transition-all duration-500">
            <div className="text-5xl mb-4">🇨🇦</div>
            <h3 className="text-2xl font-light text-white/95 mb-3">Quebec&apos;s Data Center Boom</h3>
            <p className="text-white/60 text-sm sm:text-base leading-relaxed mb-4">
              Tech giants flock to Quebec for one reason: hydroelectric power at <span className="text-cyan-300 font-semibold">$0.02/kWh</span>. It&apos;s the cheapest computing on Earth.
            </p>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-cyan-400">$9B</span>
              <span className="text-sm text-white/50">in new data center investment</span>
            </div>
          </div>
        </div>

        <div className="text-center max-w-4xl mx-auto">
          <div className="bg-gradient-to-br from-amber-500/10 via-orange-500/10 to-amber-500/10 backdrop-blur-xl border border-amber-400/30 rounded-3xl p-6 sm:p-8 md:p-10">
            <p className="text-lg sm:text-xl md:text-2xl text-white/80 leading-relaxed mb-4 font-light">
              <span className="text-amber-300 font-semibold">The pattern is clear:</span> Where energy is abundant, prosperity follows.
            </p>
            <p className="text-base sm:text-lg md:text-xl text-white/60 leading-relaxed">
              We&apos;ve identified <span className="text-amber-300 font-semibold">30+ locations worldwide</span> with similar potential.
              <span className="block mt-3 text-emerald-300">Now you can invest in them from $10.</span>
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
