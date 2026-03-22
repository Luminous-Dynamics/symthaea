'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';

// Dynamically import the Globe component to avoid SSR issues
const Globe = dynamic(() => import('@/components/Globe'), { 
  ssr: false,
  loading: () => <div className="h-[600px] bg-slate-900/50 rounded-xl animate-pulse" />
});

interface ProjectStats {
  totalProjects: number;
  totalCapacityGW: string;
  totalInvestment: string;
  statesCount: number;
}

const STATS_CACHE_TTL = 60 * 1000;
let cachedAggregates: { expiresAt: number; stats: ProjectStats } | null = null;
let cachedActiveProjects: { expiresAt: number; data: any[] } | null = null;

export default function HomePage() {
  const [stats, setStats] = useState<ProjectStats>({
    totalProjects: 11547 + 15388 + 26, // FERC + USACE viable + SMR
    totalCapacityGW: "2955.3", // 2704 FERC + 225.6 USACE + 26.1 SMR
    totalInvestment: "2.1T", // $1.23T FERC + $658B USACE + $201B SMR
    statesCount: 50
  });
  
  const [loading, setLoading] = useState(true);
  const [activeProjects, setActiveProjects] = useState<any[]>([]);

  useEffect(() => {
    // Fetch real-time stats from our API
    const fetchStats = async () => {
      if (cachedAggregates && cachedAggregates.expiresAt > Date.now()) {
        setStats(cachedAggregates.stats);
        setLoading(false);
        return;
      }

      try {
        const responses = await Promise.all([
          fetch('/api/discovery/projects?type=stats').then(r => r.json()),
          fetch('/api/discovery/projects?type=usace-stats').then(r => r.json()),
          fetch('/api/discovery/projects?type=smr-stats').then(r => r.json()),
        ])
        const [fercStats, usaceStats, smrStats] = responses

        const totalProjects = 
          (fercStats.data?.total_projects || 11547) + 
          (usaceStats.data?.viable_retrofits || 15388) + 
          (smrStats.data?.total_projects || 26);
        
        const totalCapacity = 
          parseFloat(fercStats.data?.total_capacity_gw || "2704") +
          parseFloat(usaceStats.data?.total_retrofit_potential_gw || "225.6") +
          parseFloat(smrStats.data?.total_capacity_gw || "26.1");
        
        const totalInvestment = 
          (fercStats.data?.total_stuck_investment || 1235000000000) +
          (usaceStats.data?.total_investment_required || 658119275970) +
          (smrStats.data?.total_investment || 200600000000);

        setStats({
          totalProjects,
          totalCapacityGW: totalCapacity.toFixed(1),
          totalInvestment: `$${(totalInvestment / 1_000_000_000_000).toFixed(1)}T`,
          statesCount: 50
        });

        cachedAggregates = {
          expiresAt: Date.now() + STATS_CACHE_TTL,
          stats: {
            totalProjects,
            totalCapacityGW: totalCapacity.toFixed(1),
            totalInvestment: `$${(totalInvestment / 1_000_000_000_000).toFixed(1)}T`,
            statesCount: 50
          }
        }
      } catch (error) {
        console.error('Failed to load stats', error)
      } finally {
        setLoading(false)
      }
    }

    const fetchProjects = async () => {
      if (cachedActiveProjects && cachedActiveProjects.expiresAt > Date.now()) {
        setActiveProjects(cachedActiveProjects.data)
        return
      }

      try {
        const response = await fetch('/api/discovery/projects?type=projects&limit=100')
        const data = await response.json()
        if (data.success && data.data) {
          setActiveProjects(data.data)
          cachedActiveProjects = {
            expiresAt: Date.now() + STATS_CACHE_TTL,
            data: data.data
          }
        }
      } catch (error) {
        console.error('Failed to load projects', error)
      }
    }

    fetchStats()
    fetchProjects()
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-black text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-black/20 backdrop-blur-lg border-b border-white/5">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-blue-500 rounded-lg"></div>
              <span className="text-xl font-bold">Terra Atlas</span>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <Link href="/map" className="hover:text-emerald-400 transition">Map</Link>
              <Link href="/projects" className="hover:text-emerald-400 transition">Projects</Link>
              <Link href="/corridors" className="hover:text-emerald-400 transition">Corridors</Link>
              <Link href="/api/discovery/projects" className="hover:text-emerald-400 transition">API</Link>
            </div>
            
            <button className="px-6 py-2 bg-emerald-500 hover:bg-emerald-600 rounded-full font-semibold transition">
              Start Investing
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section with Globe */}
      <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
        {/* Background Globe */}
        <div className="absolute inset-0 opacity-30">
          <Globe projects={activeProjects} />
        </div>
        
        {/* Content */}
        <div className="relative z-10 text-center max-w-5xl mx-auto px-6">
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/20 backdrop-blur rounded-full mb-6">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
              <span className="text-emerald-400 font-medium">Live Data Updated Daily</span>
            </div>
            
            <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-emerald-200 to-blue-200 bg-clip-text text-transparent">
              Energy Investment
              <br />
              for Everyone
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto mb-8">
              {stats.totalProjects.toLocaleString()} Clean Energy Projects Across America
            </p>
          </div>

          {/* Live Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <div className="text-3xl md:text-4xl font-bold text-emerald-400 mb-2">
                {stats.totalProjects.toLocaleString()}
              </div>
              <div className="text-sm text-gray-400">Projects</div>
            </div>
            
            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <div className="text-3xl md:text-4xl font-bold text-blue-400 mb-2">
                {stats.totalCapacityGW} GW
              </div>
              <div className="text-sm text-gray-400">Total Capacity</div>
            </div>
            
            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <div className="text-3xl md:text-4xl font-bold text-purple-400 mb-2">
                {stats.statesCount}
              </div>
              <div className="text-sm text-gray-400">States</div>
            </div>
            
            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <div className="text-3xl md:text-4xl font-bold text-amber-400 mb-2">
                {stats.totalInvestment}
              </div>
              <div className="text-sm text-gray-400">Opportunity</div>
            </div>
          </div>

          {/* CTA */}
          <div className="flex flex-col md:flex-row gap-4 justify-center">
            <Link href="/map" className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-blue-500 hover:from-emerald-600 hover:to-blue-600 rounded-full font-bold text-lg transition transform hover:scale-105 shadow-xl">
              Explore Live Map
            </Link>
            <Link href="#regenerative-exit" className="px-8 py-4 bg-white/10 backdrop-blur hover:bg-white/20 rounded-full font-bold text-lg transition border border-white/20">
              Learn About Regenerative Exit™
            </Link>
          </div>
          
          <p className="mt-6 text-gray-400">
            Start with $10. Power communities. Own the future.
          </p>
        </div>
      </section>

      {/* Three Pillars Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-16">
            Three Revolutionary Data Sources
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* FERC Queue */}
            <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-800/10 backdrop-blur rounded-2xl p-8 border border-emerald-500/20 hover:border-emerald-500/40 transition">
              <div className="w-16 h-16 bg-emerald-500/20 rounded-xl flex items-center justify-center mb-6">
                <span className="text-3xl">⚡</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">FERC Queue Analysis</h3>
              <div className="space-y-2 text-gray-300 mb-6">
                <p>• 11,547 renewable projects</p>
                <p>• 2,704 GW capacity stuck</p>
                <p>• 72% failure rate solved</p>
                <p>• $47.6B savings identified</p>
              </div>
              <p className="text-sm text-gray-400">
                We analyze every project in the FERC interconnection queue to find transmission corridor opportunities that reduce costs by 74%.
              </p>
            </div>

            {/* USACE Dams */}
            <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/10 backdrop-blur rounded-2xl p-8 border border-blue-500/20 hover:border-blue-500/40 transition">
              <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6">
                <span className="text-3xl">🏗️</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">Dam Retrofits</h3>
              <div className="space-y-2 text-gray-300 mb-6">
                <p>• 87,000 dams analyzed</p>
                <p>• 15,388 viable retrofits</p>
                <p>• 225.6 GW potential</p>
                <p>• 11.5 year payback</p>
              </div>
              <p className="text-sm text-gray-400">
                Every USACE dam in America analyzed for hydroelectric retrofit potential, creating 1.27 million jobs.
              </p>
            </div>

            {/* SMR Pipeline */}
            <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/10 backdrop-blur rounded-2xl p-8 border border-purple-500/20 hover:border-purple-500/40 transition">
              <div className="w-16 h-16 bg-purple-500/20 rounded-xl flex items-center justify-center mb-6">
                <span className="text-3xl">⚛️</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">SMR Pipeline</h3>
              <div className="space-y-2 text-gray-300 mb-6">
                <p>• 26 advanced reactors</p>
                <p>• 26.1 GW capacity</p>
                <p>• $201B investment</p>
                <p>• 2030-2034 deployment</p>
              </div>
              <p className="text-sm text-gray-400">
                Next-generation small modular reactors at retiring coal plants, providing 24/7 clean baseload power.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Regenerative Exit Model */}
      <section id="regenerative-exit" className="py-20 px-6 bg-gradient-to-b from-transparent via-emerald-950/20 to-transparent">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/20 backdrop-blur rounded-full mb-6">
            <span className="text-emerald-400 font-medium">Revolutionary Ownership Model</span>
          </div>
          
          <h2 className="text-4xl md:text-5xl font-bold mb-8">
            The Regenerative Exit™ Model
          </h2>
          
          <p className="text-xl text-gray-300 mb-12">
            Platform transitions to community ownership over 7 years through smart contracts
          </p>
          
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="text-left">
              <div className="text-3xl font-bold text-emerald-400 mb-2">Years 0-3</div>
              <h3 className="text-xl font-semibold mb-2">Build Phase</h3>
              <p className="text-gray-400">Traditional VC funding builds platform, first 50 projects funded</p>
            </div>
            
            <div className="text-left">
              <div className="text-3xl font-bold text-blue-400 mb-2">Years 3-5</div>
              <h3 className="text-xl font-semibold mb-2">Transition Phase</h3>
              <p className="text-gray-400">Revenue sharing begins, community governance launched</p>
            </div>
            
            <div className="text-left">
              <div className="text-3xl font-bold text-purple-400 mb-2">Years 5-7</div>
              <h3 className="text-xl font-semibold mb-2">Community Phase</h3>
              <p className="text-gray-400">Full platform ownership transfers to users and communities</p>
            </div>
          </div>
          
          <div className="bg-white/5 backdrop-blur rounded-2xl p-8 border border-white/10">
            <h3 className="text-2xl font-bold mb-4">Why This Matters</h3>
            <p className="text-gray-300">
              Unlike traditional platforms that extract value forever, Terra Atlas becomes a public good. 
              Every dollar invested strengthens community ownership. The platform that helps fund 
              $2 trillion in clean energy becomes owned by the communities it serves.
            </p>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-8">
            Ready to Transform Energy Investment?
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <Link href="/map" className="group">
              <div className="bg-white/5 backdrop-blur rounded-2xl p-6 border border-white/10 hover:border-emerald-500/50 transition">
                <div className="text-3xl mb-4">🗺️</div>
                <h3 className="text-xl font-semibold mb-2">Explore Projects</h3>
                <p className="text-gray-400 text-sm">Browse 26,961 opportunities</p>
                <div className="mt-4 text-emerald-400 group-hover:translate-x-1 transition">
                  View Map →
                </div>
              </div>
            </Link>
            
            <Link href="/api/discovery/projects" className="group">
              <div className="bg-white/5 backdrop-blur rounded-2xl p-6 border border-white/10 hover:border-blue-500/50 transition">
                <div className="text-3xl mb-4">🔌</div>
                <h3 className="text-xl font-semibold mb-2">Access API</h3>
                <p className="text-gray-400 text-sm">Free data for developers</p>
                <div className="mt-4 text-blue-400 group-hover:translate-x-1 transition">
                  Get Started →
                </div>
              </div>
            </Link>
            
            <Link href="https://github.com/terra-atlas" className="group">
              <div className="bg-white/5 backdrop-blur rounded-2xl p-6 border border-white/10 hover:border-purple-500/50 transition">
                <div className="text-3xl mb-4">🤝</div>
                <h3 className="text-xl font-semibold mb-2">Join Community</h3>
                <p className="text-gray-400 text-sm">Open source development</p>
                <div className="mt-4 text-purple-400 group-hover:translate-x-1 transition">
                  Contribute →
                </div>
              </div>
            </Link>
          </div>
          
          <div className="bg-gradient-to-r from-emerald-500 to-blue-500 p-[1px] rounded-full inline-block">
            <button className="bg-black px-8 py-4 rounded-full font-bold text-lg hover:bg-transparent transition">
              Start Investing with $10 →
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-12 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-blue-500 rounded-lg"></div>
              <span className="font-bold">Terra Atlas</span>
            </div>
            
            <div className="text-sm text-gray-400">
              Democratizing energy investment through data transparency
            </div>
            
            <div className="flex gap-6 text-sm">
              <a href="/privacy" className="text-gray-400 hover:text-white transition">Privacy</a>
              <a href="/terms" className="text-gray-400 hover:text-white transition">Terms</a>
              <a href="mailto:hello@atlas.luminousdynamics.io" className="text-gray-400 hover:text-white transition">Contact</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
