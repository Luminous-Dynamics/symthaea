'use client'

import { Flame, Activity, Cloud, Factory, AlertTriangle, Globe, Wind, Mountain, Sun } from 'lucide-react'

interface LayerToggleProps {
  activeLayer: string
  onLayerChange: (layer: string) => void
}

const layers = [
  { id: 'fires', label: 'Fires', icon: Flame, color: 'text-red-500' },
  { id: 'earthquakes', label: 'Earthquakes', icon: Activity, color: 'text-orange-500' },
  { id: 'weather', label: 'Weather', icon: Cloud, color: 'text-blue-500' },
  { id: 'emissions', label: 'Emissions', icon: Factory, color: 'text-gray-400' },
  { id: 'noaa-alerts', label: 'Alerts', icon: AlertTriangle, color: 'text-yellow-500' },
  { id: 'nasa-eonet', label: 'Natural Events', icon: Globe, color: 'text-purple-500' },
  { id: 'air-quality', label: 'Air Quality', icon: Wind, color: 'text-green-500' },
  { id: 'volcanoes', label: 'Volcanoes', icon: Mountain, color: 'text-red-600' },
  { id: 'solar-flares', label: 'Solar', icon: Sun, color: 'text-yellow-400' }
]

export default function LayerToggle({ activeLayer, onLayerChange }: LayerToggleProps) {
  return (
    <div className="flex gap-2 bg-gray-900/90 backdrop-blur-lg rounded-lg p-2 border border-gray-800">
      {layers.map((layer) => {
        const Icon = layer.icon
        const isActive = activeLayer === layer.id
        
        return (
          <button
            key={layer.id}
            onClick={() => onLayerChange(layer.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
              isActive
                ? 'bg-gray-800 text-white shadow-lg'
                : 'text-gray-400 hover:bg-gray-800/50 hover:text-white'
            }`}
          >
            <Icon className={`w-4 h-4 ${isActive ? layer.color : ''}`} />
            <span className="text-sm font-medium">{layer.label}</span>
          </button>
        )
      })}
    </div>
  )
}