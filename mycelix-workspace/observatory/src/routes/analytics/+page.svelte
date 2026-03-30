<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';

  // ============================================================================
  // Types
  // ============================================================================

  interface TrustMetric {
    label: string;
    value: number;
    change: number;
    trend: 'up' | 'down' | 'stable';
  }

  interface HappMetrics {
    name: string;
    avgTrust: number;
    transactions: number;
    growth: number;
    epistemicAvg: string;
  }

  interface TimeSeriesPoint {
    timestamp: number;
    trust: number;
    transactions: number;
    byzantineEvents: number;
  }

  interface AnalyticsStats {
    networkTrust: number;
    trustChange24h: number;
    totalTransactions: number;
    transactionsChange24h: number;
    byzantineEvents: number;
    epistemicCoverage: number;
  }

  // ============================================================================
  // Stores
  // ============================================================================

  const stats = writable<AnalyticsStats>({
    networkTrust: 0.847,
    trustChange24h: 0.023,
    totalTransactions: 2345678,
    transactionsChange24h: 12456,
    byzantineEvents: 3,
    epistemicCoverage: 0.78,
  });

  const trustMetrics = writable<TrustMetric[]>([
    { label: 'Quality Score (PoGQ)', value: 0.89, change: 0.02, trend: 'up' },
    { label: 'Consistency Score', value: 0.85, change: 0.01, trend: 'up' },
    { label: 'Reputation Score', value: 0.82, change: -0.01, trend: 'down' },
    { label: 'Composite MATL', value: 0.847, change: 0.023, trend: 'up' },
  ]);

  const happMetrics = writable<HappMetrics[]>([
    { name: 'Core', avgTrust: 0.92, transactions: 456789, growth: 12.5, epistemicAvg: 'E3-N2-M2' },
    { name: 'Marketplace', avgTrust: 0.89, transactions: 234567, growth: 8.3, epistemicAvg: 'E2-N2-M2' },
    { name: 'Identity', avgTrust: 0.94, transactions: 123456, growth: 15.2, epistemicAvg: 'E4-N3-M3' },
    { name: 'Mail', avgTrust: 0.86, transactions: 89012, growth: 5.7, epistemicAvg: 'E2-N1-M1' },
    { name: 'Governance', avgTrust: 0.91, transactions: 45678, growth: 22.1, epistemicAvg: 'E3-N3-M3' },
    { name: 'EduNet', avgTrust: 0.88, transactions: 34567, growth: 18.9, epistemicAvg: 'E3-N2-M2' },
    { name: 'Fabrication', avgTrust: 0.85, transactions: 23456, growth: 9.4, epistemicAvg: 'E3-N2-M2' },
    { name: 'Supply Chain', avgTrust: 0.87, transactions: 12345, growth: 14.2, epistemicAvg: 'E3-N2-M2' },
  ]);

  const timeSeries = writable<TimeSeriesPoint[]>([
    { timestamp: Date.now() - 86400000 * 7, trust: 0.82, transactions: 45000, byzantineEvents: 5 },
    { timestamp: Date.now() - 86400000 * 6, trust: 0.83, transactions: 47000, byzantineEvents: 4 },
    { timestamp: Date.now() - 86400000 * 5, trust: 0.835, transactions: 48500, byzantineEvents: 3 },
    { timestamp: Date.now() - 86400000 * 4, trust: 0.84, transactions: 51000, byzantineEvents: 4 },
    { timestamp: Date.now() - 86400000 * 3, trust: 0.842, transactions: 52000, byzantineEvents: 2 },
    { timestamp: Date.now() - 86400000 * 2, trust: 0.845, transactions: 54000, byzantineEvents: 3 },
    { timestamp: Date.now() - 86400000 * 1, trust: 0.847, transactions: 56000, byzantineEvents: 3 },
  ]);

  // ============================================================================
  // Helpers
  // ============================================================================

  let currentTime = new Date().toLocaleTimeString();
  let interval: ReturnType<typeof setInterval>;

  onMount(() => {
    interval = setInterval(() => {
      currentTime = new Date().toLocaleTimeString();
      simulateAnalytics();
    }, 5000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  function simulateAnalytics() {
    stats.update(s => ({
      ...s,
      totalTransactions: s.totalTransactions + Math.floor(Math.random() * 100),
      networkTrust: Math.min(1, Math.max(0.7, s.networkTrust + (Math.random() - 0.5) * 0.005)),
    }));
  }

  function getTrendIcon(trend: string): string {
    switch (trend) {
      case 'up': return '↑';
      case 'down': return '↓';
      default: return '→';
    }
  }

  function getTrendColor(trend: string): string {
    switch (trend) {
      case 'up': return 'text-green-400';
      case 'down': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }

  function formatNumber(n: number): string {
    if (n >= 1000000) return (n / 1000000).toFixed(2) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
  }

  function formatPercent(n: number): string {
    return (n * 100).toFixed(1) + '%';
  }

  function formatChange(n: number): string {
    const sign = n >= 0 ? '+' : '';
    return sign + (n * 100).toFixed(2) + '%';
  }

  function formatDate(ts: number): string {
    return new Date(ts).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  }

  // Simple bar chart rendering
  function getBarWidth(value: number, max: number): string {
    return ((value / max) * 100) + '%';
  }
</script>

<svelte:head>
  <title>Analytics | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <!-- Page Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">📊</span>
        <div>
          <h1 class="text-lg font-bold">Analytics Dashboard</h1>
          <p class="text-xs text-gray-400">MATL Metrics & Trust Trends</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <div class="text-right">
          <p class="text-xs text-gray-400">Network Trust</p>
          <p class="text-lg font-bold text-green-400">{formatPercent($stats.networkTrust)}</p>
        </div>
        <span class="text-gray-400 font-mono text-sm">{currentTime}</span>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Key Metrics -->
    <div class="grid grid-cols-2 md:grid-cols-6 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Network Trust</h3>
        <p class="text-2xl font-bold mt-1 text-green-400">{formatPercent($stats.networkTrust)}</p>
        <p class="text-xs text-green-400">{formatChange($stats.trustChange24h)} 24h</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Total Transactions</h3>
        <p class="text-2xl font-bold mt-1">{formatNumber($stats.totalTransactions)}</p>
        <p class="text-xs text-green-400">+{formatNumber($stats.transactionsChange24h)} 24h</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Byzantine Events</h3>
        <p class="text-2xl font-bold mt-1 {$stats.byzantineEvents > 5 ? 'text-yellow-400' : 'text-green-400'}">
          {$stats.byzantineEvents}
        </p>
        <p class="text-xs text-gray-400">Last 24h</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Epistemic Coverage</h3>
        <p class="text-2xl font-bold mt-1 text-purple-400">{formatPercent($stats.epistemicCoverage)}</p>
        <p class="text-xs text-gray-400">E3+ Claims</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">BFT Margin</h3>
        <p class="text-2xl font-bold mt-1 text-blue-400">+2.2%</p>
        <p class="text-xs text-gray-400">Above 34% threshold</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Active hApps</h3>
        <p class="text-2xl font-bold mt-1 text-cyan-400">18</p>
        <p class="text-xs text-gray-400">8 production</p>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- MATL Components -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>🔐</span> MATL Components
          </h2>
        </div>
        <div class="p-4 space-y-4">
          {#each $trustMetrics as metric}
            <div class="p-3 bg-gray-700/50 rounded-lg">
              <div class="flex justify-between items-center">
                <span class="text-sm">{metric.label}</span>
                <div class="flex items-center gap-2">
                  <span class="font-bold text-lg">{formatPercent(metric.value)}</span>
                  <span class={`text-sm ${getTrendColor(metric.trend)}`}>
                    {getTrendIcon(metric.trend)} {formatChange(metric.change)}
                  </span>
                </div>
              </div>
              <div class="mt-2 w-full bg-gray-600 rounded-full h-2">
                <div
                  class={`h-2 rounded-full ${metric.value >= 0.85 ? 'bg-green-500' : metric.value >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style="width: {metric.value * 100}%"
                ></div>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- hApp Performance -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 lg:col-span-2">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>📦</span> hApp Performance
          </h2>
        </div>
        <div class="p-4">
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-gray-400 text-xs uppercase">
                  <th class="text-left pb-3">hApp</th>
                  <th class="text-right pb-3">Avg Trust</th>
                  <th class="text-right pb-3">Transactions</th>
                  <th class="text-right pb-3">Growth</th>
                  <th class="text-right pb-3">Epistemic</th>
                </tr>
              </thead>
              <tbody>
                {#each $happMetrics as happ}
                  <tr class="border-t border-gray-700">
                    <td class="py-3 font-medium">{happ.name}</td>
                    <td class="py-3 text-right">
                      <span class={happ.avgTrust >= 0.9 ? 'text-green-400' : happ.avgTrust >= 0.85 ? 'text-yellow-400' : 'text-red-400'}>
                        {formatPercent(happ.avgTrust)}
                      </span>
                    </td>
                    <td class="py-3 text-right">{formatNumber(happ.transactions)}</td>
                    <td class="py-3 text-right">
                      <span class="text-green-400">+{happ.growth.toFixed(1)}%</span>
                    </td>
                    <td class="py-3 text-right">
                      <span class="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">
                        {happ.epistemicAvg}
                      </span>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Trend Charts -->
    <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Trust Trend -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>📈</span> Trust Trend (7 Days)
          </h2>
        </div>
        <div class="p-4">
          <div class="flex items-end gap-2 h-40">
            {#each $timeSeries as point, i}
              <div class="flex-1 flex flex-col items-center">
                <div
                  class="w-full bg-green-500/50 rounded-t"
                  style="height: {(point.trust - 0.8) * 500}%"
                ></div>
                <p class="text-xs text-gray-400 mt-2">{formatDate(point.timestamp).split(' ')[0]}</p>
              </div>
            {/each}
          </div>
          <div class="mt-2 flex justify-between text-xs text-gray-400">
            <span>Min: 82.0%</span>
            <span>Max: 84.7%</span>
            <span>Avg: 83.4%</span>
          </div>
        </div>
      </div>

      <!-- Transaction Volume -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>📊</span> Transaction Volume (7 Days)
          </h2>
        </div>
        <div class="p-4">
          <div class="flex items-end gap-2 h-40">
            {#each $timeSeries as point, i}
              <div class="flex-1 flex flex-col items-center">
                <div
                  class="w-full bg-blue-500/50 rounded-t"
                  style="height: {(point.transactions / 60000) * 100}%"
                ></div>
                <p class="text-xs text-gray-400 mt-2">{formatDate(point.timestamp).split(' ')[0]}</p>
              </div>
            {/each}
          </div>
          <div class="mt-2 flex justify-between text-xs text-gray-400">
            <span>Min: 45K</span>
            <span>Max: 56K</span>
            <span>Total: 353K</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Analysis Panels -->
    <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>🎯</span> MATL Formula
        </h3>
        <div class="bg-gray-900 rounded p-3 font-mono text-xs">
          <p class="text-green-400">Score = 0.4×Q + 0.3×C + 0.3×R</p>
        </div>
        <div class="mt-3 space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Q (Quality)</span>
            <span>40% weight</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">C (Consistency)</span>
            <span>30% weight</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">R (Reputation)</span>
            <span>30% weight</span>
          </div>
        </div>
      </div>

      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>📜</span> Epistemic Distribution
        </h3>
        <div class="space-y-2">
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">E4 (Public Reproduce)</span>
              <span>12%</span>
            </div>
            <div class="w-full bg-gray-600 rounded-full h-2">
              <div class="bg-purple-500 h-2 rounded-full" style="width: 12%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">E3 (Cryptographic)</span>
              <span>45%</span>
            </div>
            <div class="w-full bg-gray-600 rounded-full h-2">
              <div class="bg-green-500 h-2 rounded-full" style="width: 45%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">E2 (Private Verify)</span>
              <span>28%</span>
            </div>
            <div class="w-full bg-gray-600 rounded-full h-2">
              <div class="bg-yellow-500 h-2 rounded-full" style="width: 28%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">E1/E0 (Lower)</span>
              <span>15%</span>
            </div>
            <div class="w-full bg-gray-600 rounded-full h-2">
              <div class="bg-gray-500 h-2 rounded-full" style="width: 15%"></div>
            </div>
          </div>
        </div>
      </div>

      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>🛡️</span> Byzantine Analysis
        </h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Tolerance</span>
            <span class="text-green-400">34% (validated)</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Current</span>
            <span class="text-green-400">36.2%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Safety Margin</span>
            <span class="text-blue-400">+2.2%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Cartel Risk</span>
            <span class="text-green-400">Low</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">FL Model Integrity</span>
            <span class="text-green-400">99.7%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Analytics v0.1.0 • Data refreshed every 5s</p>
      <p class="mt-1">Trust Metrics for the Civilizational OS</p>
    </footer>
  </main>
</div>
