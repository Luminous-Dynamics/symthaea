'use client'

import { useEffect, useRef, useState, ReactNode } from 'react'

interface LazySectionProps {
  children: ReactNode
  fallback: ReactNode
  rootMargin?: string
  once?: boolean
}

export default function LazySection({
  children,
  fallback,
  rootMargin = '200px',
  once = true,
}: LazySectionProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [isActive, setIsActive] = useState(false)

  useEffect(() => {
    if (isActive && once) return

    const target = containerRef.current
    if (!target) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsActive(true)
          if (once) {
            observer.disconnect()
          }
        } else if (!once) {
          setIsActive(false)
        }
      },
      { rootMargin }
    )

    observer.observe(target)

    return () => observer.disconnect()
  }, [isActive, once, rootMargin])

  return (
    <div ref={containerRef}>
      {isActive ? children : fallback}
    </div>
  )
}
