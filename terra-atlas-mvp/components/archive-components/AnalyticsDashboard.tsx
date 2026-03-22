import React from 'react';
import { useState, useEffect } from 'react'
import {
  getTotalStorageCapacity,
  getHydrogenCapacity,
  getOffshoreWindStats,
  getNuclearStats,
  pumpedHydroStorage,
  batteryStorageMegaprojects,
  greenHydrogenProjects,
  offshoreWindFarms,
  nuclearFleet
} from '../data/global-energy-storage'
import {
  getInfrastructureStats,
  globalTransmissionLines,
  globalEnergyHubs,
  renewableClusters
} from '../data/global-infrastructure-complete'

export default function AnalyticsDashboard({ sites = [], filters = {} }) {
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedRegion, setSelectedRegion] = useState('global')
  
  // Calculate statistics
  const storageStats = getTotalStorageCapacity()
  const hydrogenStats = getHydrogenCapacity()
  const windStats = getOffshoreWindStats()
  const nuclearStats = getNuclearStats()
  const infraStats = getInfrastructureStats()
  
  // Regional breakdown
  const getRegionalStats = (region) => {
    const regionBounds = {
      'north-america': { minLat: 25, maxLat: 75, minLng: -170, maxLng: -50 },
      'europe': { minLat: 35, maxLat: 75, minLng: -25, maxLng: 50 },
      'asia': { minLat: -10, maxLat: 75, minLng: 50, maxLng: 180 },
      'africa': { minLat: -35, maxLat: 40, minLng: -20, maxLng: 55 },
      'south-america': { minLat: -60, maxLat: 15, minLng: -85, maxLng: -30 },
      'oceania': { minLat: -50, maxLat: 0, minLng: 110, maxLng: 180 }
    }
    
    if (region === 'global') {
      return {
        solarCapacity: sites.filter(s => s.technology === 'solar').reduce((sum, s) => sum + (s.capacity_mw || 0), 0),
        windCapacity: sites.filter(s => s.technology === 'wind').reduce((sum, s) => sum + (s.capacity_mw || 0), 0),
        hydroCapacity: sites.filter(s => s.technology === 'hydro').reduce((sum, s) => sum + (s.capacity_mw || 0), 0),
        nuclearCapacity: nuclearFleet.reduce((sum, n) => sum + n.capacity_mw, 0),
        storageCapacity: storageStats.total_mw,
        hydrogenCapacity: hydrogenStats.electrolysis_gw * 1000,
        totalInvestment: sites.reduce((sum, s) => sum + (s.investment || 0), 0) / 1000000000,
        projectCount: sites.length
      }
    }
    
    const bounds = regionBounds[region]
    if (!bounds) return null
    
    const regionalSites = sites.filter(s => 
      s.lat >= bounds.minLat && s.lat <= bounds.maxLat &&
      s.lng >= bounds.minLng && s.lng <= bounds.maxLng
    )
    
    return {
      solarCapacity: regionalSites.filter(s => s.technology === 'solar').reduce((sum, s) => sum + (s.capacity_mw || 0), 0),
      windCapacity: regionalSites.filter(s => s.technology === 'wind').reduce((sum, s) => sum + (s.capacity_mw || 0), 0),
      hydroCapacity: regionalSites.filter(s => s.technology === 'hydro').reduce((sum, s) => sum + (s.capacity_mw || 0), 0),
      nuclearCapacity: nuclearFleet.filter(n => 
        n.coordinates[0] >= bounds.minLat && n.coordinates[0] <= bounds.maxLat &&
        n.coordinates[1] >= bounds.minLng && n.coordinates[1] <= bounds.maxLng
      ).reduce((sum, n) => sum + n.capacity_mw, 0),
      storageCapacity: pumpedHydroStorage.filter(p =>
        p.coordinates[0] >= bounds.minLat && p.coordinates[0] <= bounds.maxLat &&
        p.coordinates[1] >= bounds.minLng && p.coordinates[1] <= bounds.maxLng
      ).reduce((sum, p) => sum + p.capacity_mw, 0),
      hydrogenCapacity: greenHydrogenProjects.filter(h =>
        h.coordinates[0] >= bounds.minLat && h.coordinates[0] <= bounds.maxLat &&
        h.coordinates[1] >= bounds.minLng && h.coordinates[1] <= bounds.maxLng
      ).reduce((sum, h) => sum + (h.capacity_gw_electrolysis || 0) * 1000, 0),
      totalInvestment: regionalSites.reduce((sum, s) => sum + (s.investment || 0), 0) / 1000000000,
      projectCount: regionalSites.length
    }
  }
  
  const regionalStats = getRegionalStats(selectedRegion)
  
  // Calculate growth projections
  const getGrowthProjections = () => {
    const currentYear = new Date().getFullYear()
    const projections = []
    
    for (let year = currentYear; year <= currentYear + 10; year++) {
      const yearFactor = (year - currentYear) / 10
      projections.push({
        year,
        solar: Math.round(regionalStats.solarCapacity * (1 + yearFactor * 2.5)),
        wind: Math.round(regionalStats.windCapacity * (1 + yearFactor * 2.2)),
        storage: Math.round(regionalStats.storageCapacity * (1 + yearFactor * 3)),
        hydrogen: Math.round(regionalStats.hydrogenCapacity * (1 + yearFactor * 4))
      })
    }
    
    return projections
  }
  
  const formatNumber = (num) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toFixed(0)
  }
  
  const formatCurrency = (num) => {
    if (num >= 1000) return `$${(num / 1000).toFixed(1)}T`
    if (num >= 1) return `$${num.toFixed(1)}B`
    return `$${(num * 1000).toFixed(0)}M`
  }
  
  return (
    <div className="absolute top-20 left-4 bg-white/95 backdrop-blur-lg rounded-xl shadow-2xl z-[999] w-96 max-h-[80vh] overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-4">
        <h2 className="text-xl font-bold mb-2">Global Energy Analytics</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              activeTab === 'overview' ? 'bg-white/20' : 'hover:bg-white/10'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('regional')}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              activeTab === 'regional' ? 'bg-white/20' : 'hover:bg-white/10'
            }`}
          >
            Regional
          </button>
          <button
            onClick={() => setActiveTab('projections')}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              activeTab === 'projections' ? 'bg-white/20' : 'hover:bg-white/10'
            }`}
          >
            Projections
          </button>
          <button
            onClick={() => setActiveTab('investment')}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              activeTab === 'investment' ? 'bg-white/20' : 'hover:bg-white/10'
            }`}
          >
            Investment
          </button>
        </div>
      </div>
      
      {/* Content */}
      <div className="p-4 overflow-y-auto max-h-[calc(80vh-120px)]">
        {activeTab === 'overview' && (
          <div className="space-y-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-3 rounded-lg">
                <div className="text-3xl mb-1">☀️</div>
                <div className="text-2xl font-bold text-amber-900">
                  {formatNumber(regionalStats.solarCapacity)} MW
                </div>
                <div className="text-sm text-amber-700">Solar Capacity</div>
              </div>
              
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-3 rounded-lg">
                <div className="text-3xl mb-1">💨</div>
                <div className="text-2xl font-bold text-green-900">
                  {formatNumber(regionalStats.windCapacity)} MW
                </div>
                <div className="text-sm text-green-700">Wind Capacity</div>
              </div>
              
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-3 rounded-lg">
                <div className="text-3xl mb-1">🔋</div>
                <div className="text-2xl font-bold text-blue-900">
                  {formatNumber(storageStats.total_mw)} MW
                </div>
                <div className="text-sm text-blue-700">Storage Capacity</div>
              </div>
              
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-3 rounded-lg">
                <div className="text-3xl mb-1">⚛️</div>
                <div className="text-2xl font-bold text-purple-900">
                  {formatNumber(nuclearStats.total_capacity_mw)} MW
                </div>
                <div className="text-sm text-purple-700">Nuclear Capacity</div>
              </div>
            </div>
            
            {/* Infrastructure Stats */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-3">Infrastructure Network</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Transmission Lines</span>
                  <span className="font-medium">{infraStats.transmissionLines.total}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Length</span>
                  <span className="font-medium">{formatNumber(infraStats.transmissionLines.totalLength)} km</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Capacity</span>
                  <span className="font-medium">{formatNumber(infraStats.transmissionLines.totalCapacity)} MW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Energy Hubs</span>
                  <span className="font-medium">{infraStats.energyHubs.total}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Renewable Zones</span>
                  <span className="font-medium">{infraStats.renewableClusters.total}</span>
                </div>
              </div>
            </div>
            
            {/* Storage Breakdown */}
            <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 rounded-lg">
              <h3 className="font-semibold text-indigo-900 mb-3">Energy Storage Details</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-indigo-700">Pumped Hydro</span>
                  <span className="font-medium text-indigo-900">{formatNumber(storageStats.pumped_mw)} MW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-indigo-700">Battery Storage</span>
                  <span className="font-medium text-indigo-900">{formatNumber(storageStats.battery_mw)} MW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-indigo-700">Total Energy</span>
                  <span className="font-medium text-indigo-900">
                    {(storageStats.pumped_gwh + storageStats.battery_gwh).toFixed(1)} GWh
                  </span>
                </div>
              </div>
            </div>
            
            {/* Hydrogen Economy */}
            <div className="bg-gradient-to-br from-green-50 to-emerald-100 p-4 rounded-lg">
              <h3 className="font-semibold text-emerald-900 mb-3">Green Hydrogen Economy</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-emerald-700">Electrolysis Capacity</span>
                  <span className="font-medium text-emerald-900">{hydrogenStats.electrolysis_gw.toFixed(1)} GW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-emerald-700">H₂ Production</span>
                  <span className="font-medium text-emerald-900">{hydrogenStats.h2_production_mtpa.toFixed(1)} MTPA</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-emerald-700">Investment Pipeline</span>
                  <span className="font-medium text-emerald-900">${hydrogenStats.investment_billion.toFixed(0)}B</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'regional' && (
          <div className="space-y-4">
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="global">🌍 Global</option>
              <option value="north-america">🌎 North America</option>
              <option value="europe">🌍 Europe</option>
              <option value="asia">🌏 Asia-Pacific</option>
              <option value="africa">🌍 Africa</option>
              <option value="south-america">🌎 South America</option>
              <option value="oceania">🌏 Oceania</option>
            </select>
            
            {regionalStats && (
              <>
                <div className="bg-gradient-to-r from-blue-50 to-green-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-3">
                    {selectedRegion === 'global' ? 'Global' : selectedRegion.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')} Overview
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Projects</span>
                      <span className="font-bold text-lg">{regionalStats.projectCount}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Investment</span>
                      <span className="font-bold text-lg">{formatCurrency(regionalStats.totalInvestment)}</span>
                    </div>
                  </div>
                </div>
                
                {/* Technology Mix */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-3">Technology Mix</h3>
                  <div className="space-y-3">
                    {[
                      { name: 'Solar', value: regionalStats.solarCapacity, color: 'amber', icon: '☀️' },
                      { name: 'Wind', value: regionalStats.windCapacity, color: 'green', icon: '💨' },
                      { name: 'Hydro', value: regionalStats.hydroCapacity, color: 'blue', icon: '💧' },
                      { name: 'Nuclear', value: regionalStats.nuclearCapacity, color: 'purple', icon: '⚛️' },
                      { name: 'Storage', value: regionalStats.storageCapacity, color: 'indigo', icon: '🔋' },
                      { name: 'Hydrogen', value: regionalStats.hydrogenCapacity, color: 'emerald', icon: '💚' }
                    ].map(tech => {
                      const total = regionalStats.solarCapacity + regionalStats.windCapacity + 
                                   regionalStats.hydroCapacity + regionalStats.nuclearCapacity + 
                                   regionalStats.storageCapacity + regionalStats.hydrogenCapacity
                      const percentage = total > 0 ? (tech.value / total * 100).toFixed(1) : 0
                      
                      return (
                        <div key={tech.name} className="relative">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm font-medium text-gray-700">
                              {tech.icon} {tech.name}
                            </span>
                            <span className="text-sm font-bold text-gray-900">
                              {formatNumber(tech.value)} MW ({percentage}%)
                            </span>
                          </div>
                          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div 
                              className={`h-full bg-gradient-to-r from-${tech.color}-400 to-${tech.color}-600 rounded-full transition-all duration-500`}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
        
        {activeTab === 'projections' && (
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-900 mb-3">10-Year Growth Projections</h3>
              <p className="text-sm text-purple-700 mb-4">
                Based on current trends and announced projects
              </p>
              
              <div className="space-y-3">
                {getGrowthProjections().filter((_, i) => i % 2 === 0).map(proj => (
                  <div key={proj.year} className="bg-white/80 p-3 rounded-lg">
                    <div className="font-semibold text-gray-900 mb-2">{proj.year}</div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-amber-600">☀️ Solar:</span>
                        <span className="font-medium ml-1">{formatNumber(proj.solar)} MW</span>
                      </div>
                      <div>
                        <span className="text-green-600">💨 Wind:</span>
                        <span className="font-medium ml-1">{formatNumber(proj.wind)} MW</span>
                      </div>
                      <div>
                        <span className="text-blue-600">🔋 Storage:</span>
                        <span className="font-medium ml-1">{formatNumber(proj.storage)} MW</span>
                      </div>
                      <div>
                        <span className="text-emerald-600">💚 H₂:</span>
                        <span className="font-medium ml-1">{formatNumber(proj.hydrogen)} MW</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-4 p-3 bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg">
                <div className="text-sm font-semibold text-green-900 mb-2">Key Growth Drivers</div>
                <ul className="text-xs text-green-800 space-y-1">
                  <li>• Solar: 25% CAGR driven by cost reductions</li>
                  <li>• Wind: 22% CAGR from offshore expansion</li>
                  <li>• Storage: 30% CAGR for grid stability</li>
                  <li>• Hydrogen: 40% CAGR for industrial decarbonization</li>
                </ul>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'investment' && (
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-lg">
              <h3 className="font-semibold text-orange-900 mb-3">Investment Opportunities</h3>
              
              {/* Top Projects by Investment */}
              <div className="space-y-3">
                <div className="text-sm font-semibold text-gray-700 mb-2">Mega Projects (&gt;$10B)</div>
                {[
                  { name: 'Western Green Energy Hub', investment: 100, type: 'Hydrogen', location: 'Australia' },
                  { name: 'NEOM Green Hydrogen', investment: 8.4, type: 'Hydrogen', location: 'Saudi Arabia' },
                  { name: 'Project AMAN', investment: 40, type: 'Hydrogen', location: 'Mauritania' },
                  { name: 'Hinkley Point C', investment: 42, type: 'Nuclear', location: 'UK' },
                  { name: 'Vogtle 3&4', investment: 30, type: 'Nuclear', location: 'USA' }
                ].map(project => (
                  <div key={project.name} className="bg-white p-3 rounded-lg border border-orange-200">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-semibold text-gray-900">{project.name}</div>
                        <div className="text-xs text-gray-600">{project.type} • {project.location}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-orange-600">${project.investment}B</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Investment by Technology */}
              <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200">
                <div className="text-sm font-semibold text-gray-700 mb-2">Investment by Technology</div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Green Hydrogen</span>
                    <span className="font-medium">${hydrogenStats.investment_billion.toFixed(0)}B</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Offshore Wind</span>
                    <span className="font-medium">$180B</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Nuclear</span>
                    <span className="font-medium">$150B</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Energy Storage</span>
                    <span className="font-medium">$120B</span>
                  </div>
                </div>
              </div>
              
              {/* ROI Indicators */}
              <div className="p-3 bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg">
                <div className="text-sm font-semibold text-green-900 mb-2">Highest ROI Sectors</div>
                <ul className="text-xs text-green-800 space-y-1">
                  <li>🥇 Battery Storage: 18-25% IRR</li>
                  <li>🥈 Solar + Storage: 15-20% IRR</li>
                  <li>🥉 Offshore Wind: 12-18% IRR</li>
                  <li>🏅 Green Hydrogen: 10-15% IRR (2030+)</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}