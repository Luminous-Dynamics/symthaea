'use client'

interface PhilosophySectionProps {
  hoveredPrinciple: number | null
  setHoveredPrinciple: (value: number | null) => void
}

const principles = [
  { icon: '🌊', name: 'Transparency', desc: 'Open data, clear terms' },
  { icon: '🌱', name: 'Regeneration', desc: 'Healing Earth & economy' },
  { icon: '⚡', name: 'Efficiency', desc: '74% cost reduction' },
  { icon: '🛡️', name: 'Resilience', desc: 'Built for generations' },
  { icon: '🤲', name: 'Access', desc: 'Everyone can invest' },
  { icon: '🔄', name: 'Transition', desc: 'Path to public good' },
  { icon: '✨', name: 'Hope', desc: 'Action over anxiety' },
]

export default function PhilosophySection({ hoveredPrinciple, setHoveredPrinciple }: PhilosophySectionProps) {
  return (
    <section className="py-20 sm:py-24 md:py-32 px-4 sm:px-6 bg-gradient-to-b from-black via-gray-950/50 to-black">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12 sm:mb-16 md:mb-20">
          <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-extralight mb-4 sm:mb-6 px-2">
            <span className="text-white/80">Built on</span>
            <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent ml-2 sm:ml-3">
              Seven Principles
            </span>
          </h2>
          <p className="text-white/50 text-sm sm:text-base md:text-lg max-w-3xl mx-auto px-2">
            Our investment philosophy balances returns with responsibility, creating value for all stakeholders
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 sm:gap-6">
          {principles.map((principle, i) => (
            <div
              key={principle.name}
              className="group text-center cursor-pointer"
              onMouseEnter={() => setHoveredPrinciple(i)}
              onMouseLeave={() => setHoveredPrinciple(null)}
            >
              <div
                className={`
                  w-16 h-16 sm:w-20 sm:h-20 mx-auto mb-3 sm:mb-4 rounded-xl sm:rounded-2xl
                  flex items-center justify-center text-2xl sm:text-3xl
                  transition-all transform
                  ${hoveredPrinciple === i
                    ? 'bg-gradient-to-br from-white/20 to-white/10 scale-110 rotate-3'
                    : 'bg-white/5 border border-white/10'
                  }
                `}
              >
                {principle.icon}
              </div>
              <h3 className="text-white/80 font-light text-sm sm:text-base mb-1">{principle.name}</h3>
              <p className="text-xs text-white/40">{principle.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
