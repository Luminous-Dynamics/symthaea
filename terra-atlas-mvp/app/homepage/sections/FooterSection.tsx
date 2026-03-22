'use client'

import Link from 'next/link'

export default function FooterSection() {
  return (
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
  )
}
