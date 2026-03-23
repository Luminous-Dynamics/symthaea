<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  GateMetricsPanel — Consciousness gate pass/fail rates and tier distribution.

  Reads from BridgeMetricsSnapshot (gate_pass, gate_fail, tier_counts[5],
  credential_source_counts[4]) returned by get_bridge_metrics() zome call.

  Can be embedded in any page or used standalone in /analytics/gates.
-->
<script lang="ts">
  export let gatePass: number = 0;
  export let gateFail: number = 0;
  export let tierCounts: number[] = [0, 0, 0, 0, 0];
  export let credentialSourceCounts: number[] = [0, 0, 0, 0];
  export let totalSuccess: number = 0;
  export let totalErrors: number = 0;
  export let totalCrossCluster: number = 0;
  export let rateLimitHits: number = 0;

  const tierNames = ['Observer', 'Participant', 'Citizen', 'Steward', 'Guardian'];
  const tierColors = ['text-gray-400', 'text-blue-400', 'text-green-400', 'text-yellow-400', 'text-purple-400'];
  const sourceNames = ['Fresh', 'Refresh', 'Cache', 'Bootstrap'];

  $: gateTotal = gatePass + gateFail;
  $: passRate = gateTotal > 0 ? gatePass / gateTotal : 1.0;
  $: tierTotal = tierCounts.reduce((a, b) => a + b, 0);
  $: sourceTotal = credentialSourceCounts.reduce((a, b) => a + b, 0);
</script>

<!-- Gate Health Summary -->
<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
  <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
    <p class="text-xs text-gray-500 uppercase tracking-wider">Gate Pass Rate</p>
    <p class="text-2xl font-bold mt-1 {passRate >= 0.9 ? 'text-green-400' : passRate >= 0.7 ? 'text-yellow-400' : 'text-red-400'}">
      {gateTotal > 0 ? (passRate * 100).toFixed(1) : '--'}%
    </p>
    <p class="text-xs text-gray-500 mt-1">{gatePass} pass / {gateFail} fail</p>
  </div>

  <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
    <p class="text-xs text-gray-500 uppercase tracking-wider">Total Dispatches</p>
    <p class="text-2xl font-bold mt-1 text-blue-400">{totalSuccess + totalErrors}</p>
    <p class="text-xs text-gray-500 mt-1">{totalSuccess} ok / {totalErrors} err</p>
  </div>

  <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
    <p class="text-xs text-gray-500 uppercase tracking-wider">Cross-Cluster</p>
    <p class="text-2xl font-bold mt-1 text-cyan-400">{totalCrossCluster}</p>
    <p class="text-xs text-gray-500 mt-1">inter-DNA calls</p>
  </div>

  <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
    <p class="text-xs text-gray-500 uppercase tracking-wider">Rate Limit Hits</p>
    <p class="text-2xl font-bold mt-1 {rateLimitHits > 0 ? 'text-red-400' : 'text-green-400'}">{rateLimitHits}</p>
    <p class="text-xs text-gray-500 mt-1">quota rejections</p>
  </div>
</div>

<!-- Tier Distribution -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
  <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
    <h3 class="text-sm font-medium text-gray-400 mb-3">Consciousness Tier Distribution</h3>
    {#if tierTotal > 0}
      <div class="space-y-2">
        {#each tierNames as name, i}
          {@const pct = tierTotal > 0 ? (tierCounts[i] / tierTotal) * 100 : 0}
          <div class="flex items-center gap-3">
            <span class="w-24 text-xs {tierColors[i]}">{name}</span>
            <div class="flex-1 bg-gray-700 rounded-full h-2">
              <div
                class="h-2 rounded-full transition-all duration-500 {
                  i === 0 ? 'bg-gray-500' :
                  i === 1 ? 'bg-blue-500' :
                  i === 2 ? 'bg-green-500' :
                  i === 3 ? 'bg-yellow-500' : 'bg-purple-500'
                }"
                style="width: {pct}%"
              ></div>
            </div>
            <span class="w-16 text-right text-xs text-gray-500">{tierCounts[i]}</span>
          </div>
        {/each}
      </div>
    {:else}
      <p class="text-xs text-gray-600 italic">No gate checks recorded yet</p>
    {/if}
  </div>

  <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
    <h3 class="text-sm font-medium text-gray-400 mb-3">Credential Source</h3>
    {#if sourceTotal > 0}
      <div class="space-y-2">
        {#each sourceNames as name, i}
          {@const pct = sourceTotal > 0 ? (credentialSourceCounts[i] / sourceTotal) * 100 : 0}
          <div class="flex items-center gap-3">
            <span class="w-24 text-xs text-gray-400">{name}</span>
            <div class="flex-1 bg-gray-700 rounded-full h-2">
              <div class="h-2 rounded-full bg-teal-500 transition-all duration-500" style="width: {pct}%"></div>
            </div>
            <span class="w-16 text-right text-xs text-gray-500">{credentialSourceCounts[i]}</span>
          </div>
        {/each}
      </div>
    {:else}
      <p class="text-xs text-gray-600 italic">No credentials issued yet</p>
    {/if}
  </div>
</div>
