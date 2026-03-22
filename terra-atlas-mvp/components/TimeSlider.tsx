'use client'

import { useState, useEffect } from 'react'
import { format } from 'date-fns'
import { Calendar, Clock } from 'lucide-react'

interface TimeSliderProps {
  startDate: Date
  endDate: Date
  onChange: (range: { start: Date; end: Date }) => void
}

export default function TimeSlider({ startDate, endDate, onChange }: TimeSliderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedRange, setSelectedRange] = useState(7) // Days

  const ranges = [
    { label: '24h', days: 1 },
    { label: '3d', days: 3 },
    { label: '7d', days: 7 },
    { label: '14d', days: 14 },
    { label: '30d', days: 30 }
  ]

  const handleRangeChange = (days: number) => {
    setSelectedRange(days)
    const end = new Date()
    const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000)
    onChange({ start, end })
  }

  return (
    <div className="bg-gray-900/90 backdrop-blur-lg rounded-lg p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-400">Time Range</span>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-blue-400" />
          <span className="text-xs text-gray-400">
            {format(startDate, 'MMM d')} - {format(endDate, 'MMM d, yyyy')}
          </span>
        </div>
      </div>

      {/* Quick Range Buttons */}
      <div className="flex gap-2 mb-3">
        {ranges.map((range) => (
          <button
            key={range.days}
            onClick={() => handleRangeChange(range.days)}
            className={`px-3 py-1 text-xs rounded-md transition-all ${
              selectedRange === range.days
                ? 'bg-blue-500 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {range.label}
          </button>
        ))}
      </div>

      {/* Visual Timeline */}
      <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-blue-500" />
        
        {/* Animated pulse */}
        <div className="absolute right-0 top-0 h-full w-1 bg-blue-400 animate-pulse" />
        
        {/* Date markers */}
        <div className="absolute inset-0 flex justify-between px-1">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="w-0.5 h-full bg-gray-700"
              style={{ opacity: 0.3 + (i * 0.15) }}
            />
          ))}
        </div>
      </div>

      {/* Live Update Indicator */}
      <div className="mt-3 flex items-center justify-center gap-2">
        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
        <span className="text-xs text-gray-400">Live data updates every 3 hours</span>
      </div>
    </div>
  )
}