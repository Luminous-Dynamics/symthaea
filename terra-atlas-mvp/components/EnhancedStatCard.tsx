// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useEffect, useState, useRef } from 'react'

interface EnhancedStatCardProps {
  value: string
  label: string
  icon: string
  index: number
  isLoading?: boolean
}

export default function EnhancedStatCard({ value, label, icon, index, isLoading = false }: EnhancedStatCardProps) {
  const [count, setCount] = useState(0)
  const [isVisible, setIsVisible] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)

  // Extract number from value string - handles K, M, t suffixes
  const parseValue = (val: string): number => {
    const match = val.match(/([\d.]+)([KMt+])?/)
    if (!match) return 0

    const num = parseFloat(match[1])
    const suffix = match[2]

    if (suffix === 'K') return num * 1000
    if (suffix === 'M') return num * 1000000
    if (suffix === 't') return num  // Already in the right unit for display
    return num
  }

  const targetNumber = parseValue(value)
  const hasPercentage = value.includes('%')
  const hasDollarSign = value.includes('$')
  const hasSuffix = value.match(/[KMt+]/)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
        }
      },
      { threshold: 0.1 }
    )

    if (cardRef.current) {
      observer.observe(cardRef.current)
    }

    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!isVisible || isNaN(targetNumber)) return

    const duration = 2000 // 2 seconds
    const steps = 60
    const increment = targetNumber / steps
    let currentStep = 0

    const timer = setInterval(() => {
      currentStep++
      if (currentStep >= steps) {
        setCount(targetNumber)
        clearInterval(timer)
      } else {
        setCount(Math.floor(increment * currentStep))
      }
    }, duration / steps)

    return () => clearInterval(timer)
  }, [isVisible, targetNumber])

  const formatCount = (num: number) => {
    // Preserve original format while animating
    if (value.includes('K+')) {
      const k = (num / 1000).toFixed(0)
      return `${k}K+`
    }
    if (value.includes('t')) {
      return `${num.toFixed(1)}t`
    }
    if (hasDollarSign) {
      return `$${num}`
    }
    if (hasPercentage) {
      return `${num.toFixed(1)}%`
    }
    return num.toLocaleString()
  }

  const displayValue = isVisible && !isNaN(targetNumber) ? formatCount(count) : value

  return (
    <div
      ref={cardRef}
      className="group inline-flex items-center gap-1.5 bg-gradient-to-br from-white/5 to-white/[0.02] backdrop-blur-xl rounded-full px-3 py-1.5
                 border border-white/10 hover:border-emerald-400/30
                 transition-all duration-300 hover:scale-105 hover:bg-white/10"
      style={{
        animationDelay: `${index * 150}ms`,
        animationFillMode: 'backwards'
      }}
    >
      {/* Icon - tiny */}
      <span className="text-sm">{icon}</span>

      {/* Value - Option B Enhanced - Larger & More Dramatic */}
      {isLoading ? (
        <span className="inline-block w-12 h-5 bg-gradient-to-r from-white/10 via-white/20 to-white/10 rounded animate-pulse" />
      ) : (
        <span className="text-lg font-bold bg-gradient-to-r from-white via-emerald-300 to-cyan-300
                        bg-clip-text text-transparent tabular-nums drop-shadow-[0_0_15px_rgba(16,185,129,0.4)]
                        transition-all duration-300 group-hover:scale-110">
          {displayValue}
        </span>
      )}

      {/* Label - tiny */}
      <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 transition-colors whitespace-nowrap">
        {label}
      </span>
    </div>
  )
}