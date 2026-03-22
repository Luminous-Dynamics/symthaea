'use client'

export default function ImpactCalculatorSection() {
  return (
    <section className="py-20 sm:py-24 md:py-32 px-4 sm:px-6 relative">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12 sm:mb-16 md:mb-20">
          <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-extralight mb-4 sm:mb-6 px-2">
            <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
              Your Impact Calculator
            </span>
          </h2>
          <p className="text-white/50 text-sm sm:text-base md:text-lg max-w-3xl mx-auto px-2">
            See the real-world impact of your investment in clean energy
          </p>
        </div>

        <div className="bg-gradient-to-br from-emerald-950/20 via-black to-cyan-950/20 backdrop-blur-xl border border-emerald-400/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-12">
          <div className="grid md:grid-cols-3 gap-6 sm:gap-8 text-center">
            <div>
              <p className="text-white/60 text-sm sm:text-base mb-3 sm:mb-4">If you invest</p>
              <div className="text-3xl sm:text-4xl md:text-5xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">$100</div>
            </div>
            <div className="hidden md:flex items-center justify-center">
              <div className="text-white/30 text-2xl">→</div>
            </div>
            <div className="md:hidden">
              <div className="text-white/30 text-2xl">↓</div>
            </div>
            <div>
              <p className="text-white/60 text-sm sm:text-base mb-3 sm:mb-4">Annual impact</p>
              <div className="space-y-2.5 sm:space-y-3">
                <div className="flex items-center justify-center gap-2 sm:gap-3">
                  <span className="text-xl sm:text-2xl">🌱</span>
                  <span className="text-emerald-400 font-semibold text-sm sm:text-base">2.4 tons CO₂ saved</span>
                </div>
                <div className="flex items-center justify-center gap-2 sm:gap-3">
                  <span className="text-xl sm:text-2xl">⚡</span>
                  <span className="text-cyan-400 font-semibold text-sm sm:text-base">3 homes powered</span>
                </div>
                <div className="flex items-center justify-center gap-2 sm:gap-3">
                  <span className="text-xl sm:text-2xl">💰</span>
                  <span className="text-amber-400 font-semibold text-sm sm:text-base">$11-14 returns</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 sm:mt-10 md:mt-12 text-center">
          <p className="text-white/40 text-xs sm:text-sm px-4">
            Impact calculations based on average performance across 31 operational projects
          </p>
        </div>
      </div>
    </section>
  )
}
