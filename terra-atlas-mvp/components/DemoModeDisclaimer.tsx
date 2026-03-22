'use client';

import { useState, useEffect } from 'react';
import { X } from 'lucide-react';

export default function DemoModeDisclaimer() {
  const [isVisible, setIsVisible] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);

  useEffect(() => {
    // Check if user has acknowledged the disclaimer this session
    const acknowledged = sessionStorage.getItem('demo-disclaimer-acknowledged');
    if (acknowledged) {
      setIsMinimized(true);
    }
  }, []);

  const handleMinimize = () => {
    setIsMinimized(true);
    sessionStorage.setItem('demo-disclaimer-acknowledged', 'true');
  };

  if (!isVisible) return null;

  if (isMinimized) {
    return (
      <div 
        className="fixed top-4 right-4 z-50 cursor-pointer"
        onClick={() => setIsMinimized(false)}
      >
        <div className="bg-yellow-500/20 backdrop-blur-lg border border-yellow-500/50 rounded-lg px-3 py-1 hover:bg-yellow-500/30 transition">
          <span className="text-yellow-300 text-xs font-medium">
            📋 DEMO MODE
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-yellow-900/95 to-orange-900/95 backdrop-blur-lg border-b border-yellow-500/50 shadow-2xl">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-2xl animate-pulse">⚠️</span>
              <div>
                <h3 className="text-yellow-100 font-bold text-sm">
                  DEMONSTRATION PLATFORM - Not Accepting Real Investments
                </h3>
                <p className="text-yellow-200/80 text-xs mt-0.5">
                  Explore our vision for democratized clean energy investing. Real investment capabilities coming soon.
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => window.location.href = '/about/demo'}
              className="px-3 py-1 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-100 text-xs font-medium rounded-lg transition"
            >
              Learn More
            </button>
            <button
              onClick={handleMinimize}
              className="p-1 hover:bg-yellow-500/20 rounded-lg transition"
              aria-label="Minimize disclaimer"
            >
              <X className="w-4 h-4 text-yellow-200" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Alternative subtle version for pages where we want less intrusion
export function DemoModeBadge() {
  return (
    <div className="inline-flex items-center px-2 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-full">
      <span className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-2"></span>
      <span className="text-yellow-400 text-xs font-medium">Demo Mode</span>
    </div>
  );
}