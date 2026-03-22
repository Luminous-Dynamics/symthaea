export default function SuccessStoriesSection() {
  return (
    <section className="py-20 sm:py-24 md:py-32 px-4 sm:px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12 sm:mb-16 md:mb-20">
          <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-extralight mb-4 sm:mb-6 px-2">
            <span className="text-white/80">Success</span>
            <span className="bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-transparent ml-2 sm:ml-3">
              Stories
            </span>
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-10 sm:gap-12">
          <div className="space-y-6 sm:space-y-8">
            <h3 className="text-xl sm:text-2xl font-light text-amber-400">For Investors</h3>
            <div className="space-y-3 sm:space-y-4">
              {[
                'Average 13.7% IRR across 31 operational projects',
                '$47.6B saved through transmission corridor sharing',
                'Tax-optimized structure with renewable energy credits',
                'Quarterly distributions with full transparency',
              ].map((item) => (
                <div className="flex items-start gap-3" key={item}>
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 flex-shrink-0"></div>
                  <p className="text-white/70 text-sm sm:text-base">{item}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6 sm:space-y-8">
            <h3 className="text-xl sm:text-2xl font-light text-cyan-400">For Communities</h3>
            <div className="space-y-3 sm:space-y-4">
              {[
                '138,000 green jobs created across 60 countries',
                '5 projects already transitioned to community ownership',
                '$658B in local economic development generated',
                'Energy independence for 385,000+ homes',
              ].map((item) => (
                <div className="flex items-start gap-3" key={item}>
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-2 flex-shrink-0"></div>
                  <p className="text-white/70 text-sm sm:text-base">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
