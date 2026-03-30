<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    fetchStats,
    fetchLeaderboard,
    fetchTopDependencies,
    fetchUnderSupported,
    fetchEcosystems,
    getIsLive,
    type AttributionStats,
    type LeaderboardEntry,
    type TopDependency,
    type EcosystemStat,
  } from '$lib/attribution-client';

  let stats: AttributionStats = {
    totalDependencies: 0,
    totalEcosystems: 0,
    verifiedCount: 0,
    totalUsage: 0,
    activeAttestations: 0,
    totalPledges: 0,
  };
  let leaderboard: LeaderboardEntry[] = [];
  let topDeps: TopDependency[] = [];
  let underSupported: LeaderboardEntry[] = [];
  let ecosystems: EcosystemStat[] = [];
  let isLive = false;
  let loading = true;
  let interval: ReturnType<typeof setInterval> | null = null;

  async function loadData() {
    const [s, lb, td, us, eco] = await Promise.all([
      fetchStats(),
      fetchLeaderboard(10),
      fetchTopDependencies(5),
      fetchUnderSupported(5),
      fetchEcosystems(),
    ]);
    stats = s;
    leaderboard = lb;
    topDeps = td;
    underSupported = us;
    ecosystems = eco;
    isLive = getIsLive();
    loading = false;
  }

  onMount(async () => {
    await loadData();
    interval = setInterval(loadData, 10000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  function ecosystemColor(eco: string): string {
    const colors: Record<string, string> = {
      RustCrate: 'text-orange-400',
      NpmPackage: 'text-red-400',
      PythonPackage: 'text-blue-400',
      GoModule: 'text-cyan-400',
      NixFlake: 'text-purple-400',
      RubyGem: 'text-pink-400',
      MavenPackage: 'text-yellow-400',
    };
    return colors[eco] || 'text-gray-400';
  }

  function formatScore(score: number): string {
    return score.toFixed(1);
  }

  function maxScore(): number {
    if (leaderboard.length === 0) return 1;
    return leaderboard[0].weighted_score || 1;
  }
</script>

<div class="min-h-screen bg-gray-900 text-white">
  <!-- Header -->
  <header class="bg-gray-800 border-b border-gray-700 px-6 py-4">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <h1 class="text-xl font-bold">Attribution Registry</h1>
        <span class="text-sm text-gray-400">Open-Source Stewardship Dashboard</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-block w-2 h-2 rounded-full {isLive ? 'bg-green-500' : 'bg-yellow-500'}"></span>
        <span class="text-sm text-gray-400">{isLive ? 'Live' : 'Demo'} Mode</span>
      </div>
    </div>
  </header>

  <main class="p-6 max-w-7xl mx-auto">
    {#if loading}
      <div class="flex items-center justify-center h-64">
        <div class="text-gray-400">Loading attribution data...</div>
      </div>
    {:else}
      <!-- Stats Grid -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div class="text-sm text-gray-400">Dependencies</div>
          <div class="text-2xl font-bold">{stats.totalDependencies.toLocaleString()}</div>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div class="text-sm text-gray-400">Ecosystems</div>
          <div class="text-2xl font-bold">{stats.totalEcosystems}</div>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div class="text-sm text-gray-400">Verified</div>
          <div class="text-2xl font-bold text-green-400">{stats.verifiedCount.toLocaleString()}</div>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div class="text-sm text-gray-400">Total Usage</div>
          <div class="text-2xl font-bold">{stats.totalUsage.toLocaleString()}</div>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div class="text-sm text-gray-400">Attestations</div>
          <div class="text-2xl font-bold text-blue-400">{stats.activeAttestations}</div>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div class="text-sm text-gray-400">Pledges</div>
          <div class="text-2xl font-bold text-purple-400">{stats.totalPledges}</div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Stewardship Leaderboard (2/3 width) -->
        <div class="lg:col-span-2 bg-gray-800 rounded-lg border border-gray-700">
          <div class="p-4 border-b border-gray-700">
            <h2 class="text-lg font-semibold">Stewardship Leaderboard</h2>
            <p class="text-sm text-gray-400 mt-1">Dependencies ranked by weighted stewardship score</p>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead>
                <tr class="text-sm text-gray-400 border-b border-gray-700">
                  <th class="text-left py-3 px-4">#</th>
                  <th class="text-left py-3 px-4">Dependency</th>
                  <th class="text-right py-3 px-4">Usage</th>
                  <th class="text-right py-3 px-4">Pledges</th>
                  <th class="text-right py-3 px-4">Score</th>
                  <th class="py-3 px-4 w-32"></th>
                </tr>
              </thead>
              <tbody>
                {#each leaderboard as entry, i}
                  <tr class="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td class="py-3 px-4 text-gray-400">{i + 1}</td>
                    <td class="py-3 px-4 font-mono text-sm">{entry.dependency_id}</td>
                    <td class="py-3 px-4 text-right">{entry.usage_count}</td>
                    <td class="py-3 px-4 text-right">{entry.pledge_count}</td>
                    <td class="py-3 px-4 text-right font-semibold text-green-400">
                      {formatScore(entry.weighted_score)}
                    </td>
                    <td class="py-3 px-4">
                      <div class="w-full bg-gray-700 rounded-full h-2">
                        <div
                          class="bg-green-500 h-2 rounded-full"
                          style="width: {(entry.weighted_score / maxScore()) * 100}%"
                        ></div>
                      </div>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Sidebar (1/3 width) -->
        <div class="space-y-6">
          <!-- Top Dependencies -->
          <div class="bg-gray-800 rounded-lg border border-gray-700">
            <div class="p-4 border-b border-gray-700">
              <h2 class="text-lg font-semibold">Top Dependencies</h2>
              <p class="text-sm text-gray-400">By usage count</p>
            </div>
            <div class="p-4 space-y-3">
              {#each topDeps as dep, i}
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="text-gray-500 text-sm w-4">{i + 1}</span>
                    <span class="font-mono text-sm truncate max-w-[180px]">{dep.dependency_id}</span>
                  </div>
                  <span class="text-sm text-gray-400">{dep.usage_count}</span>
                </div>
              {/each}
            </div>
          </div>

          <!-- Under-Supported -->
          <div class="bg-gray-800 rounded-lg border border-gray-700">
            <div class="p-4 border-b border-gray-700">
              <h2 class="text-lg font-semibold text-yellow-400">Under-Supported</h2>
              <p class="text-sm text-gray-400">High usage, low reciprocity</p>
            </div>
            <div class="p-4 space-y-3">
              {#each underSupported as dep}
                <div class="flex items-center justify-between">
                  <span class="font-mono text-sm truncate max-w-[180px]">{dep.dependency_id}</span>
                  <div class="text-right">
                    <span class="text-sm text-gray-400">{dep.usage_count} uses</span>
                    <span class="text-sm text-red-400 ml-2">{dep.pledge_count} pledges</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>

          <!-- Ecosystem Breakdown -->
          <div class="bg-gray-800 rounded-lg border border-gray-700">
            <div class="p-4 border-b border-gray-700">
              <h2 class="text-lg font-semibold">Ecosystems</h2>
            </div>
            <div class="p-4 space-y-3">
              {#each ecosystems as eco}
                <div class="flex items-center justify-between">
                  <span class="text-sm {ecosystemColor(eco.ecosystem)}">{eco.ecosystem}</span>
                  <div class="flex items-center gap-2">
                    <div class="w-24 bg-gray-700 rounded-full h-2">
                      <div
                        class="bg-blue-500 h-2 rounded-full"
                        style="width: {Math.min(100, (eco.count / (ecosystems[0]?.count || 1)) * 100)}%"
                      ></div>
                    </div>
                    <span class="text-sm text-gray-400 w-8 text-right">{eco.count}</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </div>
      </div>
    {/if}
  </main>
</div>
