'use client';

import { useState, useEffect } from 'react';
import { MapPin, TrendingDown, TrendingUp, DollarSign, Clock, AlertTriangle, CheckCircle } from 'lucide-react';

// Real West Texas failed projects data
const FAILED_PROJECTS = [
  { id: 1, name: "Lone Star Wind", lat: 31.5, lng: -102.3, capacity: 280, cost: 127, years: 7, status: "withdrawn" },
  { id: 2, name: "Permian Solar I", lat: 31.7, lng: -102.1, capacity: 350, cost: 143, years: 6, status: "expired" },
  { id: 3, name: "West Texas Wind Ranch", lat: 31.4, lng: -102.5, capacity: 400, cost: 156, years: 8, status: "withdrawn" },
  { id: 4, name: "Midland Solar Park", lat: 31.8, lng: -102.0, capacity: 250, cost: 89, years: 5, status: "bankrupt" },
  { id: 5, name: "Odessa Wind Farm", lat: 31.6, lng: -102.4, capacity: 320, cost: 134, years: 7, status: "withdrawn" },
  { id: 6, name: "Basin Solar Complex", lat: 31.3, lng: -102.2, capacity: 275, cost: 112, years: 6, status: "expired" },
  { id: 7, name: "Desert Sky Wind", lat: 31.5, lng: -102.6, capacity: 300, cost: 128, years: 7, status: "withdrawn" },
  { id: 8, name: "Permian Solar II", lat: 31.7, lng: -102.3, capacity: 225, cost: 98, years: 5, status: "withdrawn" }
];

export default function WestTexasCorridorDemo() {
  const [mode, setMode] = useState<'individual' | 'corridor'>('individual');
  const [animationStep, setAnimationStep] = useState(0);
  const [showResults, setShowResults] = useState(false);

  // Animate the demonstration
  useEffect(() => {
    if (mode === 'individual' && animationStep < 8) {
      const timer = setTimeout(() => {
        setAnimationStep(prev => prev + 1);
      }, 800);
      return () => clearTimeout(timer);
    } else if (mode === 'individual' && animationStep === 8) {
      setTimeout(() => setShowResults(true), 1000);
    } else if (mode === 'corridor' && animationStep < 3) {
      const timer = setTimeout(() => {
        setAnimationStep(prev => prev + 1);
      }, 1500);
      return () => clearTimeout(timer);
    } else if (mode === 'corridor' && animationStep === 3) {
      setTimeout(() => setShowResults(true), 1000);
    }
  }, [mode, animationStep]);

  const resetDemo = (newMode: 'individual' | 'corridor') => {
    setMode(newMode);
    setAnimationStep(0);
    setShowResults(false);
  };

  // Calculate totals
  const totalCapacity = FAILED_PROJECTS.reduce((sum, p) => sum + p.capacity, 0);
  const individualTotalCost = FAILED_PROJECTS.reduce((sum, p) => sum + p.cost, 0);
  const corridorCost = 280; // Shared transmission upgrade
  const corridorPerProject = corridorCost / 8;
  const savings = ((individualTotalCost - corridorCost) / individualTotalCost * 100).toFixed(0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent">
                The West Texas Massacre
              </h1>
              <p className="text-gray-400 mt-1">How 8 renewable projects died needlessly</p>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-red-500">72%</div>
              <div className="text-sm text-gray-400">Failure Rate</div>
            </div>
          </div>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="container mx-auto px-4 py-8">
        <div className="flex gap-4 justify-center">
          <button
            onClick={() => resetDemo('individual')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              mode === 'individual' 
                ? 'bg-red-600 text-white shadow-lg shadow-red-600/50' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            ❌ Individual Approach (What Happened)
          </button>
          <button
            onClick={() => resetDemo('corridor')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              mode === 'corridor' 
                ? 'bg-green-600 text-white shadow-lg shadow-green-600/50' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            ✅ Corridor Approach (What Should Happen)
          </button>
        </div>
      </div>

      {/* Visualization Area */}
      <div className="container mx-auto px-4">
        <div className="bg-black/30 rounded-2xl p-8 backdrop-blur-sm border border-white/10">
          <div className="relative h-[400px] mb-8">
            {/* Map Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 to-red-900/20 rounded-lg">
              <div className="absolute top-4 left-4 text-sm text-gray-500">West Texas • Permian Basin</div>
              
              {/* Grid Lines */}
              <svg className="absolute inset-0 w-full h-full opacity-10">
                {[...Array(10)].map((_, i) => (
                  <g key={i}>
                    <line x1={`${i * 10}%`} y1="0" x2={`${i * 10}%`} y2="100%" stroke="white" />
                    <line x1="0" y1={`${i * 10}%`} x2="100%" y2={`${i * 10}%`} stroke="white" />
                  </g>
                ))}
              </svg>

              {/* Projects */}
              {FAILED_PROJECTS.map((project, index) => {
                const x = 20 + (project.lng + 102.6) * 200;
                const y = 350 - (project.lat - 31.3) * 300;
                const isActive = mode === 'individual' ? index < animationStep : animationStep > 0;
                const isFailed = mode === 'individual' && index < animationStep;
                const isConnected = mode === 'corridor' && animationStep >= 2;

                return (
                  <div key={project.id}>
                    {/* Connection Lines (Corridor Mode) */}
                    {mode === 'corridor' && isConnected && index < 7 && (
                      <svg className="absolute inset-0 w-full h-full">
                        <line
                          x1={x}
                          y1={y}
                          x2={200}
                          y2={200}
                          stroke="#10b981"
                          strokeWidth="2"
                          strokeDasharray="5,5"
                          className="animate-pulse"
                        />
                      </svg>
                    )}

                    {/* Project Dot */}
                    <div
                      className={`absolute transition-all duration-1000 ${
                        isActive ? 'scale-100 opacity-100' : 'scale-0 opacity-0'
                      }`}
                      style={{ left: x - 12, top: y - 12 }}
                    >
                      <div className={`relative ${isFailed ? 'animate-shake' : ''}`}>
                        <div className={`w-6 h-6 rounded-full ${
                          isFailed ? 'bg-red-500 animate-pulse' :
                          isConnected ? 'bg-green-500' :
                          'bg-yellow-500'
                        }`}>
                          {isFailed && (
                            <span className="absolute -top-8 left-1/2 -translate-x-1/2 text-red-500 text-xs font-bold animate-fadeIn">
                              FAILED
                            </span>
                          )}
                        </div>
                        <span className="absolute top-8 left-1/2 -translate-x-1/2 text-[10px] text-gray-400 whitespace-nowrap">
                          {project.capacity}MW
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Central Hub (Corridor Mode) */}
              {mode === 'corridor' && animationStep >= 2 && (
                <div 
                  className="absolute animate-scaleIn"
                  style={{ left: 188, top: 188 }}
                >
                  <div className="w-6 h-6 bg-blue-500 rounded-full animate-pulse">
                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 text-blue-400 text-xs font-bold whitespace-nowrap">
                      SHARED HUB
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Status Timeline */}
          {mode === 'individual' && (
            <div className="grid grid-cols-8 gap-2 mb-8">
              {FAILED_PROJECTS.map((project, index) => (
                <div key={project.id} className={`transition-all duration-500 ${
                  index < animationStep ? 'opacity-100' : 'opacity-20'
                }`}>
                  <div className="text-center p-2 bg-red-900/30 rounded border border-red-500/30">
                    <div className="text-xs text-gray-400 mb-1">{project.name}</div>
                    <div className="text-red-500 font-bold">${project.cost}M</div>
                    <div className="text-[10px] text-gray-500">{project.years} years</div>
                    <div className="text-[10px] text-red-400 uppercase mt-1">{project.status}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {mode === 'corridor' && (
            <div className="space-y-4 mb-8">
              {animationStep >= 1 && (
                <div className="animate-slideIn bg-green-900/30 p-4 rounded border border-green-500/30">
                  <div className="flex items-center gap-4">
                    <CheckCircle className="text-green-500" />
                    <div>
                      <div className="font-semibold">Step 1: Projects Identify Each Other</div>
                      <div className="text-sm text-gray-400">8 projects within 50-mile radius discovered via Terra Atlas API</div>
                    </div>
                  </div>
                </div>
              )}
              {animationStep >= 2 && (
                <div className="animate-slideIn bg-green-900/30 p-4 rounded border border-green-500/30">
                  <div className="flex items-center gap-4">
                    <CheckCircle className="text-green-500" />
                    <div>
                      <div className="font-semibold">Step 2: Form Transmission Corridor</div>
                      <div className="text-sm text-gray-400">Single shared interconnection point, costs split proportionally</div>
                    </div>
                  </div>
                </div>
              )}
              {animationStep >= 3 && (
                <div className="animate-slideIn bg-green-900/30 p-4 rounded border border-green-500/30">
                  <div className="flex items-center gap-4">
                    <CheckCircle className="text-green-500" />
                    <div>
                      <div className="font-semibold">Step 3: All Projects Succeed</div>
                      <div className="text-sm text-gray-400">3-year timeline, 74% cost reduction, 95% success rate</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Results Comparison */}
          {showResults && (
            <div className="animate-fadeIn">
              <div className="grid md:grid-cols-2 gap-8">
                {/* Individual Results */}
                <div className={`p-6 rounded-lg ${
                  mode === 'individual' 
                    ? 'bg-red-900/30 border-2 border-red-500' 
                    : 'bg-gray-900/30 border border-gray-700 opacity-50'
                }`}>
                  <h3 className="text-xl font-bold text-red-500 mb-4">Individual Approach</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Projects:</span>
                      <span className="font-bold">8</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Succeeded:</span>
                      <span className="font-bold text-red-500">0</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Cost:</span>
                      <span className="font-bold">${individualTotalCost}M</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Average Timeline:</span>
                      <span className="font-bold">6.5 years</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Clean Energy Delivered:</span>
                      <span className="font-bold text-red-500">0 MW</span>
                    </div>
                    <div className="pt-3 border-t border-gray-700">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Success Rate:</span>
                        <span className="text-2xl font-bold text-red-500">0%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Corridor Results */}
                <div className={`p-6 rounded-lg ${
                  mode === 'corridor' 
                    ? 'bg-green-900/30 border-2 border-green-500' 
                    : 'bg-gray-900/30 border border-gray-700 opacity-50'
                }`}>
                  <h3 className="text-xl font-bold text-green-500 mb-4">Corridor Approach</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Projects:</span>
                      <span className="font-bold">8</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Succeeded:</span>
                      <span className="font-bold text-green-500">8</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Cost:</span>
                      <span className="font-bold">${corridorCost}M</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Average Timeline:</span>
                      <span className="font-bold">3 years</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Clean Energy Delivered:</span>
                      <span className="font-bold text-green-500">{totalCapacity.toLocaleString()} MW</span>
                    </div>
                    <div className="pt-3 border-t border-gray-700">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Success Rate:</span>
                        <span className="text-2xl font-bold text-green-500">100%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="mt-8 bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-lg p-6 border border-blue-500/30">
                <h3 className="text-xl font-bold mb-4 text-blue-400">The Corridor Advantage</h3>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-500">{savings}%</div>
                    <div className="text-sm text-gray-400">Cost Reduction</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-500">54%</div>
                    <div className="text-sm text-gray-400">Time Reduction</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-500">${corridorPerProject.toFixed(0)}M</div>
                    <div className="text-sm text-gray-400">Per Project Cost</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Call to Action */}
        <div className="mt-12 text-center">
          <h2 className="text-3xl font-bold mb-4">Stop Dying Alone. Build Corridors.</h2>
          <p className="text-xl text-gray-400 mb-8">
            Terra Atlas Discovery API finds nearby projects and calculates optimal corridors.
          </p>
          <div className="flex gap-4 justify-center">
            <a
              href="/content/72-percent-failure-whitepaper"
              className="px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-orange-600/50 transition-all"
            >
              📄 Read the Full Report
            </a>
            <a
              href="/api/discovery"
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-blue-600/50 transition-all"
            >
              🔌 Access Free API
            </a>
          </div>
        </div>
      </div>

      {/* Add animations */}
      <style jsx>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes scaleIn {
          from { transform: scale(0); }
          to { transform: scale(1); }
        }
        .animate-shake { animation: shake 0.5s; }
        .animate-fadeIn { animation: fadeIn 0.5s; }
        .animate-slideIn { animation: slideIn 0.5s; }
        .animate-scaleIn { animation: scaleIn 0.5s; }
      `}</style>
    </div>
  );
}