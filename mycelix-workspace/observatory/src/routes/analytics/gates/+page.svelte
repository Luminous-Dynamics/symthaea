<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  /analytics/gates — Consciousness Gate Metrics Dashboard

  Displays real-time gate pass/fail rates, tier distribution, credential
  source breakdown, and dispatch health from BridgeMetricsSnapshot.

  In demo mode (no conductor), shows simulated data.
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import GateMetricsPanel from '$lib/components/GateMetricsPanel.svelte';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';
  import { callZome, isConnected } from '$lib/conductor';
  import { conductorStatus$ } from '$lib/conductor';

  // ============================================================================
  // Types (mirrors BridgeMetricsSnapshot from Rust)
  // ============================================================================

  interface BridgeMetricsSnapshot {
    total_success: number;
    total_errors: number;
    total_cross_cluster: number;
    rate_limit_hits: number;
    gate_pass: number;
    gate_fail: number;
    tier_counts: number[];
    credential_source_counts: number[];
    call_counts: Array<{ key: string; success_count: number; error_count: number }>;
    error_counts: Array<{ code: string; count: number }>;
    latency: { p50_us: number; p95_us: number; p99_us: number } | null;
  }

  // ============================================================================
  // State
  // ============================================================================

  let metrics: BridgeMetricsSnapshot = {
    total_success: 0,
    total_errors: 0,
    total_cross_cluster: 0,
    rate_limit_hits: 0,
    gate_pass: 0,
    gate_fail: 0,
    tier_counts: [0, 0, 0, 0, 0],
    credential_source_counts: [0, 0, 0, 0],
    call_counts: [],
    error_counts: [],
    latency: null,
  };

  const lastUpdated = writable(0);
  const loadError = writable('');
  const refreshing = writable(false);

  let pollInterval: ReturnType<typeof setInterval> | null = null;

  // Demo data for when conductor is not connected
  function demoMetrics(): BridgeMetricsSnapshot {
    return {
      total_success: 12847 + Math.floor(Math.random() * 100),
      total_errors: 234 + Math.floor(Math.random() * 10),
      total_cross_cluster: 3891 + Math.floor(Math.random() * 50),
      rate_limit_hits: 7,
      gate_pass: 9823 + Math.floor(Math.random() * 50),
      gate_fail: 412 + Math.floor(Math.random() * 10),
      tier_counts: [
        1204 + Math.floor(Math.random() * 20),  // Observer
        3456 + Math.floor(Math.random() * 30),   // Participant
        4012 + Math.floor(Math.random() * 25),   // Citizen
        1891 + Math.floor(Math.random() * 15),   // Steward
        672 + Math.floor(Math.random() * 10),    // Guardian
      ],
      credential_source_counts: [
        5123, // Fresh
        2341, // Refresh
        3456, // Cache
        315,  // Bootstrap
      ],
      call_counts: [
        { key: 'commons_bridge::dispatch_call', success_count: 4521, error_count: 89 },
        { key: 'civic_bridge::dispatch_call', success_count: 2103, error_count: 45 },
        { key: 'governance_bridge::verify_consciousness_gate', success_count: 3891, error_count: 67 },
        { key: 'identity::get_consciousness_credential', success_count: 2332, error_count: 33 },
      ],
      error_counts: [
        { code: 'BRG-001', count: 7 },
        { code: 'BRG-007', count: 89 },
        { code: 'BRG-010', count: 45 },
      ],
      latency: { p50_us: 1250, p95_us: 4800, p99_us: 12300 },
    };
  }

  // Bridges to query (each bridge has its own metrics)
  const BRIDGE_ZOMES = [
    { role: 'commons', zome: 'commons_bridge' },
    { role: 'civic', zome: 'civic_bridge' },
    { role: 'governance', zome: 'governance_bridge' },
  ];

  async function fetchMetrics() {
    $refreshing = true;
    $loadError = '';

    try {
      if (!isConnected()) {
        // Demo mode
        metrics = demoMetrics();
        $lastUpdated = Date.now();
        return;
      }

      // Query the first available bridge (they all share bridge-common metrics)
      const raw = await callZome({
        role_name: 'governance',
        zome_name: 'governance_bridge',
        fn_name: 'get_bridge_metrics',
        payload: null,
      });

      // get_bridge_metrics returns a JSON string
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      metrics = {
        total_success: parsed.total_success ?? 0,
        total_errors: parsed.total_errors ?? 0,
        total_cross_cluster: parsed.total_cross_cluster ?? 0,
        rate_limit_hits: parsed.rate_limit_hits ?? 0,
        gate_pass: parsed.gate_pass ?? 0,
        gate_fail: parsed.gate_fail ?? 0,
        tier_counts: parsed.tier_counts ?? [0, 0, 0, 0, 0],
        credential_source_counts: parsed.credential_source_counts ?? [0, 0, 0, 0],
        call_counts: parsed.call_counts ?? [],
        error_counts: parsed.error_counts ?? [],
        latency: parsed.latency ?? null,
      };
      $lastUpdated = Date.now();
    } catch (e) {
      $loadError = `Failed to fetch gate metrics: ${e}`;
    } finally {
      $refreshing = false;
    }
  }

  onMount(() => {
    fetchMetrics();
    pollInterval = setInterval(fetchMetrics, 15_000); // Refresh every 15s
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
  });
</script>

<svelte:head>
  <title>Consciousness Gate Metrics | Observatory</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-6xl mx-auto">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-2xl font-bold">Consciousness Gate Metrics</h1>
        <p class="text-sm text-gray-500 mt-1">
          Real-time pass/fail rates, tier distribution, and bridge dispatch health
        </p>
      </div>
      {#if $conductorStatus$ === 'demo'}
        <span class="px-3 py-1 bg-yellow-900/50 text-yellow-400 rounded-full text-xs font-medium">
          Demo Mode
        </span>
      {/if}
    </div>

    <FreshnessBar {lastUpdated} {loadError} {refreshing} refresh={fetchMetrics} />

    <!-- Gate Metrics Panel -->
    <GateMetricsPanel
      gatePass={metrics.gate_pass}
      gateFail={metrics.gate_fail}
      tierCounts={metrics.tier_counts}
      credentialSourceCounts={metrics.credential_source_counts}
      totalSuccess={metrics.total_success}
      totalErrors={metrics.total_errors}
      totalCrossCluster={metrics.total_cross_cluster}
      rateLimitHits={metrics.rate_limit_hits}
    />

    <!-- Detailed Tables -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
      <!-- Top Zome Calls -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-sm font-medium text-gray-400 mb-3">Top Dispatch Calls</h3>
        {#if metrics.call_counts.length > 0}
          <div class="space-y-1">
            {#each metrics.call_counts.slice(0, 10) as cc}
              <div class="flex items-center justify-between text-xs py-1 border-b border-gray-700/50">
                <span class="text-gray-300 font-mono truncate max-w-[60%]">{cc.key}</span>
                <div class="flex gap-3">
                  <span class="text-green-400">{cc.success_count}</span>
                  {#if cc.error_count > 0}
                    <span class="text-red-400">{cc.error_count}</span>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <p class="text-xs text-gray-600 italic">No dispatch data</p>
        {/if}
      </div>

      <!-- Latency & Errors -->
      <div class="space-y-4">
        {#if metrics.latency}
          <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 class="text-sm font-medium text-gray-400 mb-3">Dispatch Latency</h3>
            <div class="grid grid-cols-3 gap-4 text-center">
              <div>
                <p class="text-xs text-gray-500">p50</p>
                <p class="text-lg font-bold text-green-400">{(metrics.latency.p50_us / 1000).toFixed(1)}ms</p>
              </div>
              <div>
                <p class="text-xs text-gray-500">p95</p>
                <p class="text-lg font-bold text-yellow-400">{(metrics.latency.p95_us / 1000).toFixed(1)}ms</p>
              </div>
              <div>
                <p class="text-xs text-gray-500">p99</p>
                <p class="text-lg font-bold text-red-400">{(metrics.latency.p99_us / 1000).toFixed(1)}ms</p>
              </div>
            </div>
          </div>
        {/if}

        {#if metrics.error_counts.length > 0}
          <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 class="text-sm font-medium text-gray-400 mb-3">Error Codes</h3>
            <div class="space-y-1">
              {#each metrics.error_counts as ec}
                <div class="flex items-center justify-between text-xs py-1">
                  <span class="font-mono text-red-300">{ec.code}</span>
                  <span class="text-gray-500">{ec.count}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
