'use client'

import { useEffect, useRef } from 'react'
import { MapPin } from 'lucide-react'

interface ProjectMapProps {
  latitude: number
  longitude: number
  name: string
}

export default function ProjectMap({ latitude, longitude, name }: ProjectMapProps) {
  const mapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // In production, this would integrate with Mapbox or Google Maps
    // For now, show a static map visualization
    if (mapRef.current) {
      // Create a simple canvas-based map visualization
      const canvas = document.createElement('canvas')
      canvas.width = mapRef.current.clientWidth
      canvas.height = mapRef.current.clientHeight
      canvas.style.width = '100%'
      canvas.style.height = '100%'
      
      const ctx = canvas.getContext('2d')
      if (ctx) {
        // Draw gradient background
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
        gradient.addColorStop(0, '#1a1a2e')
        gradient.addColorStop(1, '#0f0f23')
        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'
        ctx.lineWidth = 1
        for (let i = 0; i < canvas.width; i += 40) {
          ctx.beginPath()
          ctx.moveTo(i, 0)
          ctx.lineTo(i, canvas.height)
          ctx.stroke()
        }
        for (let i = 0; i < canvas.height; i += 40) {
          ctx.beginPath()
          ctx.moveTo(0, i)
          ctx.lineTo(canvas.width, i)
          ctx.stroke()
        }
        
        // Draw location marker
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        
        // Outer glow
        const glowGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 50)
        glowGradient.addColorStop(0, 'rgba(16, 185, 129, 0.3)')
        glowGradient.addColorStop(1, 'rgba(16, 185, 129, 0)')
        ctx.fillStyle = glowGradient
        ctx.beginPath()
        ctx.arc(centerX, centerY, 50, 0, Math.PI * 2)
        ctx.fill()
        
        // Inner marker
        ctx.fillStyle = '#10b981'
        ctx.beginPath()
        ctx.arc(centerX, centerY, 8, 0, Math.PI * 2)
        ctx.fill()
        
        // Marker ring
        ctx.strokeStyle = '#10b981'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centerX, centerY, 15, 0, Math.PI * 2)
        ctx.stroke()
        
        // Pulsing ring animation
        let radius = 20
        let opacity = 1
        const animate = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          
          // Redraw background
          ctx.fillStyle = gradient
          ctx.fillRect(0, 0, canvas.width, canvas.height)
          
          // Redraw grid
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'
          ctx.lineWidth = 1
          for (let i = 0; i < canvas.width; i += 40) {
            ctx.beginPath()
            ctx.moveTo(i, 0)
            ctx.lineTo(i, canvas.height)
            ctx.stroke()
          }
          for (let i = 0; i < canvas.height; i += 40) {
            ctx.beginPath()
            ctx.moveTo(0, i)
            ctx.lineTo(canvas.width, i)
            ctx.stroke()
          }
          
          // Draw animated ring
          ctx.strokeStyle = `rgba(16, 185, 129, ${opacity})`
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
          ctx.stroke()
          
          // Draw static elements
          ctx.fillStyle = glowGradient
          ctx.beginPath()
          ctx.arc(centerX, centerY, 50, 0, Math.PI * 2)
          ctx.fill()
          
          ctx.fillStyle = '#10b981'
          ctx.beginPath()
          ctx.arc(centerX, centerY, 8, 0, Math.PI * 2)
          ctx.fill()
          
          ctx.strokeStyle = '#10b981'
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(centerX, centerY, 15, 0, Math.PI * 2)
          ctx.stroke()
          
          // Update animation
          radius += 1
          opacity -= 0.02
          if (radius > 60) {
            radius = 20
            opacity = 1
          }
          
          requestAnimationFrame(animate)
        }
        animate()
      }
      
      mapRef.current.innerHTML = ''
      mapRef.current.appendChild(canvas)
    }
  }, [latitude, longitude])

  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <div ref={mapRef} className="w-full h-full" />
      
      {/* Overlay with location info */}
      <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-emerald-400" />
          <div>
            <p className="text-xs text-gray-400">Project Location</p>
            <p className="text-sm font-semibold text-white">{name}</p>
            <p className="text-xs text-gray-400 font-mono">
              {latitude.toFixed(4)}°, {longitude.toFixed(4)}°
            </p>
          </div>
        </div>
      </div>
      
      {/* Map controls (visual only for now) */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button className="p-2 bg-black/50 backdrop-blur-sm rounded-lg border border-white/10 hover:bg-black/70 transition">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
        </button>
        <button className="p-2 bg-black/50 backdrop-blur-sm rounded-lg border border-white/10 hover:bg-black/70 transition">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
        </button>
      </div>
      
      {/* Attribution */}
      <div className="absolute bottom-4 right-4 text-xs text-gray-500">
        Terra Atlas Maps
      </div>
    </div>
  )
}