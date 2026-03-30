// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useMemo } from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { MyceliumGraph, GraphNode, GraphLink, NodeType } from '@/components/graph/MyceliumGraph';
import {
  Network,
  Users,
  Music2,
  Heart,
  Sparkles,
  TrendingUp,
  Globe,
  Zap,
  Layers,
  Eye,
} from 'lucide-react';

// Generate mock network data
function generateMockData(): { nodes: GraphNode[]; links: GraphLink[] } {
  const artists: GraphNode[] = [
    { id: 'a1', type: 'artist', name: 'Aurora Rising', size: 20, connections: 45 },
    { id: 'a2', type: 'artist', name: 'Cosmic Drift', size: 18, connections: 38 },
    { id: 'a3', type: 'artist', name: 'Neon Pulse', size: 15, connections: 32 },
    { id: 'a4', type: 'artist', name: 'Ethereal Waves', size: 12, connections: 28 },
    { id: 'a5', type: 'artist', name: 'Midnight Echo', size: 14, connections: 25 },
    { id: 'a6', type: 'artist', name: 'Solar Flare', size: 10, connections: 22 },
    { id: 'a7', type: 'artist', name: 'Deep Current', size: 11, connections: 20 },
    { id: 'a8', type: 'artist', name: 'Velvet Storm', size: 9, connections: 18 },
  ];

  const listeners: GraphNode[] = [
    { id: 'l1', type: 'listener', name: 'Alex', size: 8, connections: 12 },
    { id: 'l2', type: 'listener', name: 'Jordan', size: 7, connections: 10 },
    { id: 'l3', type: 'listener', name: 'Morgan', size: 9, connections: 14 },
    { id: 'l4', type: 'listener', name: 'Casey', size: 6, connections: 8 },
    { id: 'l5', type: 'listener', name: 'Riley', size: 8, connections: 11 },
    { id: 'l6', type: 'listener', name: 'Taylor', size: 7, connections: 9 },
    { id: 'l7', type: 'listener', name: 'Quinn', size: 5, connections: 7 },
    { id: 'l8', type: 'listener', name: 'Sage', size: 6, connections: 8 },
    { id: 'l9', type: 'listener', name: 'Drew', size: 7, connections: 10 },
    { id: 'l10', type: 'listener', name: 'Blake', size: 5, connections: 6 },
  ];

  const genres: GraphNode[] = [
    { id: 'g1', type: 'genre', name: 'Ambient', size: 16, connections: 30 },
    { id: 'g2', type: 'genre', name: 'Synthwave', size: 14, connections: 25 },
    { id: 'g3', type: 'genre', name: 'Lo-Fi', size: 12, connections: 22 },
    { id: 'g4', type: 'genre', name: 'Electronic', size: 18, connections: 35 },
  ];

  const songs: GraphNode[] = [
    { id: 's1', type: 'song', name: 'Stellar Dreams', size: 8, connections: 15 },
    { id: 's2', type: 'song', name: 'Midnight Hour', size: 7, connections: 12 },
    { id: 's3', type: 'song', name: 'Ocean Waves', size: 6, connections: 10 },
    { id: 's4', type: 'song', name: 'Neon Lights', size: 9, connections: 18 },
    { id: 's5', type: 'song', name: 'Electric Soul', size: 7, connections: 14 },
  ];

  const events: GraphNode[] = [
    { id: 'e1', type: 'event', name: 'Summer Solstice', size: 12, connections: 20 },
    { id: 'e2', type: 'event', name: 'Album Launch', size: 10, connections: 15 },
  ];

  const nodes = [...artists, ...listeners, ...genres, ...songs, ...events];

  // Generate links
  const links: GraphLink[] = [];

  // Artist collaborations
  links.push({ source: 'a1', target: 'a2', type: 'collaborate', strength: 0.8 });
  links.push({ source: 'a1', target: 'a3', type: 'collaborate', strength: 0.6 });
  links.push({ source: 'a2', target: 'a4', type: 'influence', strength: 0.5 });
  links.push({ source: 'a3', target: 'a5', type: 'similar', strength: 0.7 });
  links.push({ source: 'a4', target: 'a6', type: 'collaborate', strength: 0.4 });
  links.push({ source: 'a5', target: 'a7', type: 'influence', strength: 0.6 });
  links.push({ source: 'a6', target: 'a8', type: 'similar', strength: 0.5 });

  // Listener-Artist connections (patronage)
  links.push({ source: 'l1', target: 'a1', type: 'patron', strength: 0.9 });
  links.push({ source: 'l1', target: 'a2', type: 'listen', strength: 0.7 });
  links.push({ source: 'l2', target: 'a2', type: 'patron', strength: 0.8 });
  links.push({ source: 'l2', target: 'a3', type: 'listen', strength: 0.6 });
  links.push({ source: 'l3', target: 'a1', type: 'patron', strength: 0.85 });
  links.push({ source: 'l3', target: 'a4', type: 'listen', strength: 0.5 });
  links.push({ source: 'l4', target: 'a3', type: 'listen', strength: 0.7 });
  links.push({ source: 'l5', target: 'a5', type: 'patron', strength: 0.75 });
  links.push({ source: 'l6', target: 'a1', type: 'listen', strength: 0.6 });
  links.push({ source: 'l7', target: 'a6', type: 'listen', strength: 0.5 });
  links.push({ source: 'l8', target: 'a7', type: 'listen', strength: 0.55 });
  links.push({ source: 'l9', target: 'a2', type: 'patron', strength: 0.7 });
  links.push({ source: 'l10', target: 'a8', type: 'listen', strength: 0.45 });

  // Artist-Genre connections
  links.push({ source: 'a1', target: 'g1', type: 'similar', strength: 0.8 });
  links.push({ source: 'a1', target: 'g4', type: 'similar', strength: 0.6 });
  links.push({ source: 'a2', target: 'g2', type: 'similar', strength: 0.9 });
  links.push({ source: 'a3', target: 'g4', type: 'similar', strength: 0.7 });
  links.push({ source: 'a4', target: 'g1', type: 'similar', strength: 0.85 });
  links.push({ source: 'a5', target: 'g3', type: 'similar', strength: 0.8 });
  links.push({ source: 'a6', target: 'g2', type: 'similar', strength: 0.65 });
  links.push({ source: 'a7', target: 'g1', type: 'similar', strength: 0.7 });
  links.push({ source: 'a8', target: 'g4', type: 'similar', strength: 0.6 });

  // Song connections
  links.push({ source: 's1', target: 'a1', type: 'similar', strength: 1 });
  links.push({ source: 's2', target: 'a2', type: 'similar', strength: 1 });
  links.push({ source: 's3', target: 'a4', type: 'similar', strength: 1 });
  links.push({ source: 's4', target: 'a3', type: 'similar', strength: 1 });
  links.push({ source: 's5', target: 'a5', type: 'similar', strength: 1 });

  // Event connections
  links.push({ source: 'e1', target: 'a1', type: 'similar', strength: 0.9 });
  links.push({ source: 'e1', target: 'a2', type: 'similar', strength: 0.8 });
  links.push({ source: 'e1', target: 'l1', type: 'similar', strength: 0.7 });
  links.push({ source: 'e1', target: 'l3', type: 'similar', strength: 0.7 });
  links.push({ source: 'e2', target: 'a3', type: 'similar', strength: 0.9 });
  links.push({ source: 'e2', target: 'l2', type: 'similar', strength: 0.6 });

  return { nodes, links };
}

export default function MyceliumPage() {
  const { nodes, links } = useMemo(() => generateMockData(), []);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [viewMode, setViewMode] = useState<'network' | 'clusters' | 'timeline'>('network');

  // Calculate network stats
  const stats = useMemo(() => {
    const artistCount = nodes.filter(n => n.type === 'artist').length;
    const listenerCount = nodes.filter(n => n.type === 'listener').length;
    const totalConnections = links.length;
    const avgConnectionStrength = links.reduce((sum, l) => sum + l.strength, 0) / links.length;

    return { artistCount, listenerCount, totalConnections, avgConnectionStrength };
  }, [nodes, links]);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Hero */}
          <div className="relative rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-green-600 via-emerald-600 to-teal-600">
            <div className="absolute inset-0 bg-black/20" />

            {/* Network lines background */}
            <svg className="absolute inset-0 w-full h-full opacity-20">
              {[...Array(20)].map((_, i) => (
                <line
                  key={i}
                  x1={`${Math.random() * 100}%`}
                  y1={`${Math.random() * 100}%`}
                  x2={`${Math.random() * 100}%`}
                  y2={`${Math.random() * 100}%`}
                  stroke="white"
                  strokeWidth="1"
                />
              ))}
              {[...Array(15)].map((_, i) => (
                <circle
                  key={i}
                  cx={`${Math.random() * 100}%`}
                  cy={`${Math.random() * 100}%`}
                  r={3 + Math.random() * 5}
                  fill="white"
                />
              ))}
            </svg>

            <div className="relative p-8 md:p-12">
              <div className="flex items-center gap-3 mb-4">
                <Network className="w-8 h-8" />
                <span className="text-sm font-medium uppercase tracking-wider opacity-80">
                  Mycelium Network
                </span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                The Living Web of Music
              </h1>
              <p className="text-lg opacity-80 max-w-xl mb-6">
                Explore the interconnected network of artists, listeners, and musical moments.
                Like mycelium in a forest, every connection nourishes the whole.
              </p>

              {/* Quick Stats */}
              <div className="flex items-center gap-8">
                <div>
                  <p className="text-3xl font-bold">{stats.artistCount}</p>
                  <p className="text-sm opacity-70">Artists</p>
                </div>
                <div className="w-px h-10 bg-white/20" />
                <div>
                  <p className="text-3xl font-bold">{stats.listenerCount}</p>
                  <p className="text-sm opacity-70">Listeners</p>
                </div>
                <div className="w-px h-10 bg-white/20" />
                <div>
                  <p className="text-3xl font-bold">{stats.totalConnections}</p>
                  <p className="text-sm opacity-70">Connections</p>
                </div>
                <div className="w-px h-10 bg-white/20" />
                <div>
                  <p className="text-3xl font-bold">{(stats.avgConnectionStrength * 100).toFixed(0)}%</p>
                  <p className="text-sm opacity-70">Avg Strength</p>
                </div>
              </div>
            </div>
          </div>

          {/* View Mode Tabs */}
          <div className="flex items-center gap-4 mb-6">
            <button
              onClick={() => setViewMode('network')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                viewMode === 'network'
                  ? 'bg-green-500 text-white'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <Network className="w-4 h-4" />
              Network View
            </button>
            <button
              onClick={() => setViewMode('clusters')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                viewMode === 'clusters'
                  ? 'bg-green-500 text-white'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <Layers className="w-4 h-4" />
              Clusters
            </button>
            <button
              onClick={() => setViewMode('timeline')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                viewMode === 'timeline'
                  ? 'bg-green-500 text-white'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              Growth Timeline
            </button>
          </div>

          {/* Main Graph */}
          <section className="mb-10">
            <MyceliumGraph
              nodes={nodes}
              links={links}
              width={1200}
              height={700}
              onNodeClick={setSelectedNode}
              highlightedNodeId={selectedNode?.id}
            />
          </section>

          {/* Network Insights */}
          <section className="mb-10">
            <h2 className="text-2xl font-bold mb-6">Network Insights</h2>
            <div className="grid grid-cols-4 gap-4">
              <InsightCard
                icon={Heart}
                title="Strongest Bond"
                value="Aurora Rising ↔ Alex"
                description="Deepest patron relationship"
                color="from-pink-500 to-rose-500"
              />
              <InsightCard
                icon={Users}
                title="Most Connected"
                value="Aurora Rising"
                description="45 connections across the network"
                color="from-purple-500 to-indigo-500"
              />
              <InsightCard
                icon={Zap}
                title="Rising Star"
                value="Velvet Storm"
                description="Fastest growing connection rate"
                color="from-yellow-500 to-orange-500"
              />
              <InsightCard
                icon={Globe}
                title="Bridge Node"
                value="Electronic"
                description="Connects 4 distinct communities"
                color="from-green-500 to-emerald-500"
              />
            </div>
          </section>

          {/* Connection Types */}
          <section className="mb-10">
            <h2 className="text-2xl font-bold mb-6">Connection Types</h2>
            <div className="grid grid-cols-5 gap-4">
              <ConnectionTypeCard
                type="listen"
                color="#10B981"
                count={links.filter(l => l.type === 'listen').length}
                description="Listener discoveries"
              />
              <ConnectionTypeCard
                type="patron"
                color="#F59E0B"
                count={links.filter(l => l.type === 'patron').length}
                description="Patronage bonds"
              />
              <ConnectionTypeCard
                type="collaborate"
                color="#8B5CF6"
                count={links.filter(l => l.type === 'collaborate').length}
                description="Creative collaborations"
              />
              <ConnectionTypeCard
                type="influence"
                color="#EC4899"
                count={links.filter(l => l.type === 'influence').length}
                description="Artistic influence"
              />
              <ConnectionTypeCard
                type="similar"
                color="#6B7280"
                count={links.filter(l => l.type === 'similar').length}
                description="Style similarities"
              />
            </div>
          </section>

          {/* Philosophy */}
          <section className="mb-10">
            <div className="p-8 rounded-2xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-white/10">
              <div className="flex items-start gap-6">
                <Network className="w-12 h-12 text-green-400 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold mb-3">Why Mycelium?</h3>
                  <p className="text-muted-foreground mb-4">
                    In nature, mycelium networks connect trees in a forest, allowing them to share
                    nutrients and communicate. The Mycelix network works the same way:
                  </p>
                  <ul className="space-y-2 text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-green-400">•</span>
                      <span><strong className="text-white">Nutrient flow</strong> — Value flows to where it's needed most</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400">•</span>
                      <span><strong className="text-white">No central hub</strong> — The network is decentralized and resilient</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400">•</span>
                      <span><strong className="text-white">Emergent intelligence</strong> — The network learns and adapts</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400">•</span>
                      <span><strong className="text-white">Symbiosis</strong> — Artists and listeners grow together</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>

      <Player />
    </div>
  );
}

interface InsightCardProps {
  icon: typeof Heart;
  title: string;
  value: string;
  description: string;
  color: string;
}

function InsightCard({ icon: Icon, title, value, description, color }: InsightCardProps) {
  return (
    <div className="bg-white/5 rounded-xl p-4 hover:bg-white/10 transition-colors">
      <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${color} flex items-center justify-center mb-3`}>
        <Icon className="w-5 h-5" />
      </div>
      <p className="text-sm text-muted-foreground mb-1">{title}</p>
      <p className="font-semibold mb-1">{value}</p>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

interface ConnectionTypeCardProps {
  type: string;
  color: string;
  count: number;
  description: string;
}

function ConnectionTypeCard({ type, color, count, description }: ConnectionTypeCardProps) {
  return (
    <div className="bg-white/5 rounded-xl p-4 text-center">
      <div
        className="w-4 h-4 rounded-full mx-auto mb-2"
        style={{ backgroundColor: color }}
      />
      <p className="font-semibold capitalize mb-1">{type}</p>
      <p className="text-2xl font-bold mb-1">{count}</p>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  );
}
