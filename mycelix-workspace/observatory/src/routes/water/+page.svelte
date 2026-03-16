<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    getAllWaterSystems,
    getActiveWaterAlerts,
    registerWaterSystem,
    submitWaterReading,
    type WaterSystem,
    type WaterReading,
    type ContaminationAlert,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';

  let systems: WaterSystem[] = [];
  let alerts: ContaminationAlert[] = [];
  let loading = true;
  let error = '';

  // Register System form
  let showSystemForm = false;
  let sysName = '';
  let sysType: 'RoofRainwater' | 'GroundCatchment' | 'FogCollection' | 'DewCollection' | 'Snowmelt' = 'RoofRainwater';
  let sysCapacity = 1000;
  let sysCatchment = 50;

  // Report Reading form
  let showReadingForm = false;
  let rdSourceId = '';
  let rdParameter = 'pH';
  let rdValue = 7.0;
  let rdUnit = 'pH';
  let rdLocation = '';

  let submitting = false;

  const systemTypes = [
    { value: 'RoofRainwater', label: 'Roof Rainwater' },
    { value: 'GroundCatchment', label: 'Ground Catchment' },
    { value: 'FogCollection', label: 'Fog Collection' },
    { value: 'DewCollection', label: 'Dew Collection' },
    { value: 'Snowmelt', label: 'Snowmelt' },
  ] as const;

  // Freshness — 60s polling (water safety is critical)
  async function fetchData() {
    [systems, alerts] = await Promise.all([
      getAllWaterSystems(),
      getActiveWaterAlerts(),
    ]);
  }

  const freshness = createFreshness(fetchData, 60_000);
  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } = freshness;

  onMount(async () => {
    await refresh();
    loading = false;
    startPolling();
  });

  onDestroy(() => stopPolling());

  async function handleRegisterSystem() {
    if (!sysName || sysCapacity <= 0) return;
    submitting = true;
    try {
      const s = await registerWaterSystem({
        name: sysName,
        system_type: sysType,
        capacity_liters: sysCapacity,
        catchment_area_sqm: sysCatchment > 0 ? sysCatchment : undefined,
        efficiency_percent: 0,
        owner_did: '',
        location_lat: 0,
        location_lon: 0,
        installed_at: Date.now(),
      });
      systems = [...systems, s];
      sysName = '';
      sysCapacity = 1000;
      sysCatchment = 50;
      sysType = 'RoofRainwater';
      showSystemForm = false;
      toasts.success('System registered');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to register system');
    } finally {
      submitting = false;
    }
  }

  async function handleSubmitReading() {
    if (!rdSourceId) return;
    submitting = true;
    try {
      await submitWaterReading({
        source_id: rdSourceId,
        parameter: rdParameter,
        value: rdValue,
        unit: rdUnit,
        location: rdLocation,
        recorded_at: Date.now(),
        recorder_did: '',
      });
      rdSourceId = '';
      rdParameter = 'pH';
      rdValue = 7.0;
      rdUnit = 'pH';
      rdLocation = '';
      showReadingForm = false;
      toasts.success('Reading submitted');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to submit reading');
    } finally {
      submitting = false;
    }
  }

  function totalCapacity(systems: WaterSystem[]): number {
    return systems.reduce((sum, s) => sum + s.capacity_liters, 0);
  }

  function severityColor(severity: string): string {
    switch (severity) {
      case 'Emergency': return 'bg-red-900/50 border-red-500 text-red-200';
      case 'Warning': return 'bg-yellow-900/50 border-yellow-500 text-yellow-200';
      default: return 'bg-blue-900/50 border-blue-500 text-blue-200';
    }
  }
</script>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <div class="container mx-auto max-w-6xl">
    <h1 class="text-2xl font-bold mb-1">Water Safety</h1>
    <p class="text-gray-400 mb-4">Community water harvesting systems, storage levels, and quality monitoring.</p>
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />

    {#if loading}
      <div class="text-gray-400">Loading water data...</div>
    {:else if error}
      <div class="bg-red-900/30 border border-red-500 rounded p-4 text-red-200">{error}</div>
    {:else}

      <!-- Active Alerts -->
      {#if alerts.length > 0}
        <div class="mb-6">
          <h2 class="text-lg font-semibold mb-3 text-red-400">Active Alerts</h2>
          {#each alerts as alert}
            <div class="border rounded-lg p-4 mb-2 {severityColor(alert.severity)}">
              <div class="flex items-center justify-between">
                <div>
                  <span class="font-semibold uppercase text-sm">{alert.severity}</span>
                  <span class="mx-2">—</span>
                  <span>{alert.contaminant} detected: {alert.measured_value} (threshold: {alert.threshold_value})</span>
                </div>
                <span class="text-sm opacity-75">{new Date(alert.reported_at).toLocaleDateString()}</span>
              </div>
              <p class="text-sm mt-1 opacity-75">Source: {alert.source_id} | Reported by: {alert.reported_by}</p>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Summary Stats -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-cyan-400">{systems.length}</div>
          <div class="text-sm text-gray-400">Harvest Systems</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-cyan-400">{(totalCapacity(systems) / 1000).toFixed(1)}k L</div>
          <div class="text-sm text-gray-400">Total Capacity</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold {alerts.length > 0 ? 'text-yellow-400' : 'text-green-400'}">{alerts.length}</div>
          <div class="text-sm text-gray-400">Active Alerts</div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="flex gap-3 mb-6">
        <button on:click={() => { showSystemForm = !showSystemForm; showReadingForm = false; }}
          class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium transition-colors">
          + Register System
        </button>
        <button on:click={() => { showReadingForm = !showReadingForm; showSystemForm = false; }}
          class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium transition-colors"
          disabled={systems.length === 0}>
          + Report Reading
        </button>
      </div>

      <!-- Register System Form -->
      {#if showSystemForm}
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
          <h2 class="text-sm font-semibold text-gray-300 mb-4">Register New Water System</h2>
          <form on:submit|preventDefault={handleRegisterSystem} class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label for="sname" class="text-xs text-gray-400">System Name</label>
              <input id="sname" bind:value={sysName} placeholder="Rooftop catchment A"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500" />
            </div>
            <div>
              <label for="stype" class="text-xs text-gray-400">System Type</label>
              <select id="stype" bind:value={sysType}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500">
                {#each systemTypes as t}
                  <option value={t.value}>{t.label}</option>
                {/each}
              </select>
            </div>
            <div>
              <label for="scap" class="text-xs text-gray-400">Capacity (liters)</label>
              <input id="scap" type="number" bind:value={sysCapacity} min="1" max="1000000"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500" />
            </div>
            <div>
              <label for="scatch" class="text-xs text-gray-400">Catchment Area (m&sup2;)</label>
              <input id="scatch" type="number" bind:value={sysCatchment} min="0" max="100000"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500" />
            </div>
            <div class="md:col-span-2">
              <button type="submit" disabled={submitting || !sysName || sysCapacity <= 0}
                class="w-full bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
                {submitting ? 'Registering...' : 'Register System'}
              </button>
            </div>
          </form>
        </div>
      {/if}

      <!-- Report Reading Form -->
      {#if showReadingForm}
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
          <h2 class="text-sm font-semibold text-gray-300 mb-4">Report Water Quality Reading</h2>
          <form on:submit|preventDefault={handleSubmitReading} class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label for="rdsrc" class="text-xs text-gray-400">System</label>
              <select id="rdsrc" bind:value={rdSourceId}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500">
                <option value="">Select system...</option>
                {#each systems as sys}
                  <option value={sys.id}>{sys.name}</option>
                {/each}
              </select>
            </div>
            <div>
              <label for="rdparam" class="text-xs text-gray-400">Parameter</label>
              <select id="rdparam" bind:value={rdParameter}
                on:change={() => { rdUnit = rdParameter === 'pH' ? 'pH' : rdParameter === 'Turbidity' ? 'NTU' : 'ppm'; }}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500">
                <option value="pH">pH</option>
                <option value="Turbidity">Turbidity</option>
                <option value="TDS">Total Dissolved Solids</option>
                <option value="Chlorine">Chlorine</option>
                <option value="E.coli">E. coli</option>
              </select>
            </div>
            <div>
              <label for="rdval" class="text-xs text-gray-400">Value ({rdUnit})</label>
              <input id="rdval" type="number" bind:value={rdValue} min="0" step="0.1"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500" />
            </div>
            <div>
              <label for="rdloc" class="text-xs text-gray-400">Sample Location</label>
              <input id="rdloc" type="text" bind:value={rdLocation} placeholder="e.g. Tank inlet"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500" />
            </div>
            <div class="md:col-span-2">
              <button type="submit" disabled={submitting || !rdSourceId}
                class="w-full bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
                {submitting ? 'Submitting...' : 'Submit Reading'}
              </button>
            </div>
          </form>
        </div>
      {/if}

      <!-- Systems List -->
      <h2 class="text-lg font-semibold mb-3">Harvest Systems</h2>
      <div class="space-y-3">
        {#each systems as system}
          <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div class="flex items-start justify-between">
              <div>
                <h3 class="font-semibold text-white">{system.name}</h3>
                <p class="text-sm text-gray-400 mt-1">
                  {system.system_type.replace(/([A-Z])/g, ' $1').trim()} | {system.capacity_liters.toLocaleString()} L capacity
                  {#if system.catchment_area_sqm}
                    | {system.catchment_area_sqm} m² catchment
                  {/if}
                </p>
              </div>
              <div class="text-right">
                <div class="text-sm text-gray-400">Efficiency</div>
                <div class="text-lg font-bold text-cyan-400">{system.efficiency_percent}%</div>
              </div>
            </div>
            <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
              <span>Owner: {system.owner_did}</span>
              <span>Installed: {new Date(system.installed_at).toLocaleDateString()}</span>
              <span>Location: {system.location_lat.toFixed(3)}, {system.location_lon.toFixed(3)}</span>
            </div>
          </div>
        {/each}
      </div>

      {#if systems.length === 0}
        <div class="text-center text-gray-500 py-12">
          <p class="text-lg">No water systems registered yet.</p>
          <p class="text-sm mt-1">Register a rainwater harvesting system to start tracking community water supply.</p>
        </div>
      {/if}
    {/if}
  </div>
</div>
