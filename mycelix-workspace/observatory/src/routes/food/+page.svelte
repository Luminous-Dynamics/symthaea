<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import {
    getAllPlots,
    registerPlot,
    recordHarvest,
    logResourceInput,
    getCommunityInputs,
    getNutrientSummary,
    RESOURCE_TYPES,
    RESOURCE_TYPE_LABELS,
    type FoodPlot,
    type HarvestRecord,
    type ResourceInput,
    type ResourceType,
    type NutrientSummary,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';
  import { exportHarvestsCsv, exportPlotsCsv } from '$lib/data-export';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';

  function resourceLabel(type: string): string {
    return (RESOURCE_TYPE_LABELS as Record<string, string>)[type] ?? type;
  }

  // ============================================================================
  // Stores
  // ============================================================================

  const plots = writable<FoodPlot[]>([]);
  const harvests = writable<HarvestRecord[]>([]);
  const resourceInputs = writable<ResourceInput[]>([]);
  const nutrientSummary = writable<NutrientSummary | null>(null);

  // Plot form
  let plotName = '';
  let plotLocation = '';
  let plotArea = 20;
  let plotType = 'Raised beds';
  let showPlotForm = false;

  // Harvest form
  let harvestPlotId = '';
  let harvestCrop = '';
  let harvestKg = 1;
  let harvestNotes = '';
  let showHarvestForm = false;

  // Resource input form
  let riType: ResourceType = 'KitchenWaste';
  let riKg = 5;
  let riPlotId = '';
  let riNotes = '';
  let showResourceForm = false;

  let submitting = false;
  let loading = true;

  const plotTypes = ['Raised beds', 'Open field', 'Greenhouse tunnel', 'Container garden', 'Vertical garden', 'Rooftop'];

  // ============================================================================
  // Freshness — 2min polling
  // ============================================================================

  async function fetchData() {
    const [p, ri, ns] = await Promise.all([
      getAllPlots(),
      getCommunityInputs(),
      getNutrientSummary(),
    ]);
    plots.set(p);
    resourceInputs.set(ri);
    nutrientSummary.set(ns);
  }

  const freshness = createFreshness(fetchData, 120_000);
  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } = freshness;

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    await refresh();
    loading = false;
    startPolling();
  });

  onDestroy(() => stopPolling());

  async function handleRegisterPlot() {
    if (!plotName || !plotLocation || plotArea <= 0) return;
    submitting = true;
    try {
      const p = await registerPlot(plotName, plotLocation, plotArea, plotType);
      plots.update(list => [...list, p]);
      plotName = '';
      plotLocation = '';
      plotArea = 20;
      showPlotForm = false;
      toasts.success('Plot registered');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to register plot');
    } finally {
      submitting = false;
    }
  }

  async function handleRecordHarvest() {
    if (!harvestPlotId || !harvestCrop || harvestKg <= 0) return;
    submitting = true;
    try {
      const h = await recordHarvest(harvestPlotId, harvestCrop, harvestKg, harvestNotes);
      harvests.update(list => [...list, h]);
      harvestCrop = '';
      harvestKg = 1;
      harvestNotes = '';
      showHarvestForm = false;
      toasts.success('Harvest recorded');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to record harvest');
    } finally {
      submitting = false;
    }
  }

  async function handleLogResource() {
    if (riKg <= 0) return;
    submitting = true;
    try {
      const ri = await logResourceInput(riType, riKg, riPlotId || null, riNotes);
      resourceInputs.update(list => [ri, ...list]);
      // Refresh summary
      const ns = await getNutrientSummary();
      nutrientSummary.set(ns);
      riType = 'KitchenWaste';
      riKg = 5;
      riPlotId = '';
      riNotes = '';
      showResourceForm = false;
      toasts.success('Input logged');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to log input');
    } finally {
      submitting = false;
    }
  }

  function totalArea(plots: FoodPlot[]): number {
    return plots.reduce((sum, p) => sum + p.area_sqm, 0);
  }

  function totalHarvest(harvests: HarvestRecord[]): number {
    return harvests.reduce((sum, h) => sum + h.quantity_kg, 0);
  }

  function totalResourceKg(inputs: ResourceInput[]): number {
    return inputs.reduce((sum, ri) => sum + ri.quantity_kg, 0);
  }

  function timeAgo(ts: number): string {
    const diff = Date.now() - ts;
    const days = Math.floor(diff / 86400000);
    if (days > 0) return `${days}d ago`;
    const hours = Math.floor(diff / 3600000);
    if (hours > 0) return `${hours}h ago`;
    return 'just now';
  }
</script>

<svelte:head>
  <title>Food Production | Mycelix Observatory</title>
</svelte:head>

{#if loading}
  <div class="text-white p-8 text-center text-gray-400">Loading food production data...</div>
{:else}
<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl" aria-hidden="true">&#x1F331;</span>
        <div>
          <h1 class="text-lg font-bold">Food Production</h1>
          <p class="text-xs text-gray-400">Community growing, harvest tracking</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <div class="text-right">
          <p class="text-xs text-gray-400">Total Growing Area</p>
          <p class="text-lg font-bold text-green-400">{totalArea($plots)} m&sup2;</p>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />
    <!-- Stats -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Active Plots</h3>
        <p class="text-2xl font-bold mt-1 text-green-400">{$plots.length}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Total Area</h3>
        <p class="text-2xl font-bold mt-1">{totalArea($plots)} m&sup2;</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Harvests Logged</h3>
        <p class="text-2xl font-bold mt-1 text-yellow-400">{$harvests.length}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Total Yield</h3>
        <p class="text-2xl font-bold mt-1 text-orange-400">{totalHarvest($harvests).toFixed(1)} kg</p>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex gap-3 mb-6">
      <button on:click={() => { showPlotForm = !showPlotForm; showHarvestForm = false; showResourceForm = false; }}
        class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors">
        + Register Plot
      </button>
      <button on:click={() => { showHarvestForm = !showHarvestForm; showPlotForm = false; showResourceForm = false; }}
        class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm font-medium transition-colors"
        disabled={$plots.length === 0}>
        + Record Harvest
      </button>
      <button on:click={() => { showResourceForm = !showResourceForm; showPlotForm = false; showHarvestForm = false; }}
        class="px-4 py-2 bg-amber-600 hover:bg-amber-700 rounded text-sm font-medium transition-colors">
        + Log Contribution
      </button>
    </div>

    <!-- Register Plot Form -->
    {#if showPlotForm}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-sm font-semibold text-gray-300 mb-4">Register New Plot</h2>
        <form on:submit|preventDefault={handleRegisterPlot} class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="pname" class="text-xs text-gray-400">Plot Name</label>
            <input id="pname" bind:value={plotName} placeholder="My backyard garden"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
          </div>
          <div>
            <label for="ploc" class="text-xs text-gray-400">Location</label>
            <input id="ploc" bind:value={plotLocation} placeholder="e.g. 12 Main St, Suburb"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
          </div>
          <div>
            <label for="parea" class="text-xs text-gray-400">Area (m&sup2;)</label>
            <input id="parea" type="number" bind:value={plotArea} min="1" max="10000"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
          </div>
          <div>
            <label for="ptype" class="text-xs text-gray-400">Plot Type</label>
            <select id="ptype" bind:value={plotType}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500">
              {#each plotTypes as t}
                <option value={t}>{t}</option>
              {/each}
            </select>
          </div>
          <div class="md:col-span-2">
            <button type="submit" disabled={submitting || !plotName || !plotLocation}
              class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
              {submitting ? 'Registering...' : 'Register Plot'}
            </button>
          </div>
        </form>
      </div>
    {/if}

    <!-- Record Harvest Form -->
    {#if showHarvestForm}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-sm font-semibold text-gray-300 mb-4">Record Harvest</h2>
        <form on:submit|preventDefault={handleRecordHarvest} class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="hplot" class="text-xs text-gray-400">Plot</label>
            <select id="hplot" bind:value={harvestPlotId}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-yellow-500">
              <option value="">Select plot...</option>
              {#each $plots as plot}
                <option value={plot.id}>{plot.name}</option>
              {/each}
            </select>
          </div>
          <div>
            <label for="hcrop" class="text-xs text-gray-400">Crop</label>
            <input id="hcrop" bind:value={harvestCrop} placeholder="Spinach, tomatoes, etc."
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-yellow-500" />
          </div>
          <div>
            <label for="hkg" class="text-xs text-gray-400">Quantity (kg)</label>
            <input id="hkg" type="number" bind:value={harvestKg} min="0.1" step="0.1"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-yellow-500" />
          </div>
          <div>
            <label for="hnotes" class="text-xs text-gray-400">Notes</label>
            <input id="hnotes" bind:value={harvestNotes} placeholder="Quality, weather, etc."
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-yellow-500" />
          </div>
          <div class="md:col-span-2">
            <button type="submit" disabled={submitting || !harvestPlotId || !harvestCrop}
              class="w-full bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
              {submitting ? 'Recording...' : 'Record Harvest'}
            </button>
          </div>
        </form>
      </div>
    {/if}

    <!-- Log Resource Contribution Form -->
    {#if showResourceForm}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-sm font-semibold text-gray-300 mb-4">Log a Contribution</h2>
        <form on:submit|preventDefault={handleLogResource} class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="ritype" class="text-xs text-gray-400">Resource Type</label>
            <select id="ritype" bind:value={riType}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-amber-500">
              {#each RESOURCE_TYPES as t}
                <option value={t}>{RESOURCE_TYPE_LABELS[t]}</option>
              {/each}
            </select>
          </div>
          <div>
            <label for="rikg" class="text-xs text-gray-400">Quantity (kg)</label>
            <input id="rikg" type="number" bind:value={riKg} min="0.1" step="0.1" max="100000"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-amber-500" />
          </div>
          <div>
            <label for="riplot" class="text-xs text-gray-400">Plot (optional)</label>
            <select id="riplot" bind:value={riPlotId}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-amber-500">
              <option value="">No specific plot</option>
              {#each $plots as plot}
                <option value={plot.id}>{plot.name}</option>
              {/each}
            </select>
          </div>
          <div>
            <label for="rinotes" class="text-xs text-gray-400">Notes</label>
            <input id="rinotes" bind:value={riNotes} placeholder="Source, quality, etc."
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-amber-500" />
          </div>
          <div class="md:col-span-2">
            <button type="submit" disabled={submitting || riKg <= 0}
              class="w-full bg-amber-600 hover:bg-amber-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
              {submitting ? 'Logging...' : 'Log Contribution'}
            </button>
          </div>
        </form>
      </div>
    {/if}

    <!-- Nutrient Summary Card -->
    {#if $nutrientSummary}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
          <span aria-hidden="true">&#x1F33E;</span> Community Nutrient Summary
        </h2>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
          <div class="bg-gray-700/50 rounded p-3">
            <p class="text-xs text-gray-400">Contributions</p>
            <p class="text-xl font-bold text-amber-400">{$nutrientSummary.total_contributions}</p>
          </div>
          <div class="bg-gray-700/50 rounded p-3">
            <p class="text-xs text-gray-400">Total Input</p>
            <p class="text-xl font-bold">{totalResourceKg($resourceInputs).toFixed(1)} kg</p>
          </div>
          <div class="bg-gray-700/50 rounded p-3">
            <p class="text-xs text-gray-400">Est. Nitrogen</p>
            <p class="text-xl font-bold text-green-400">{$nutrientSummary.total_nitrogen_kg.toFixed(2)} kg</p>
          </div>
          <div class="bg-gray-700/50 rounded p-3">
            <p class="text-xs text-gray-400">Est. Phosphorus</p>
            <p class="text-xl font-bold text-blue-400">{$nutrientSummary.total_phosphorus_kg.toFixed(2)} kg</p>
          </div>
          <div class="bg-gray-700/50 rounded p-3">
            <p class="text-xs text-gray-400">Est. Potassium</p>
            <p class="text-xl font-bold text-purple-400">{$nutrientSummary.total_potassium_kg.toFixed(2)} kg</p>
          </div>
        </div>
        {#if $nutrientSummary.total_kg_by_type.length > 0}
          <div class="space-y-2">
            <p class="text-xs text-gray-400">By Resource Type</p>
            {#each $nutrientSummary.total_kg_by_type as [type, kg]}
              <div class="flex justify-between items-center text-sm">
                <span class="text-gray-300">{resourceLabel(type)}</span>
                <span class="text-gray-400">{kg.toFixed(1)} kg</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Recent Contributions -->
    {#if $resourceInputs.length > 0}
      <div class="bg-gray-800 rounded-lg border border-gray-700 mb-6">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span aria-hidden="true">&#x267B;</span> Recent Contributions
          </h2>
        </div>
        <div class="p-4 space-y-2">
          {#each $resourceInputs.slice(0, 10) as ri}
            <div class="flex justify-between items-center p-3 bg-gray-700/50 rounded text-sm">
              <div>
                <span class="font-medium text-amber-300">{RESOURCE_TYPE_LABELS[ri.input_type]}</span>
                <span class="text-gray-400 ml-2">{ri.quantity_kg} kg</span>
                {#if ri.notes}
                  <span class="text-gray-500 ml-2">&mdash; {ri.notes}</span>
                {/if}
              </div>
              <div class="text-xs text-gray-500">
                {ri.contributor_did} &middot; {timeAgo(ri.contributed_at)}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Plot List -->
    <div class="bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700">
        <h2 class="text-lg font-semibold flex items-center gap-2">
          <span aria-hidden="true">&#x1F33F;</span> Community Plots
        </h2>
      </div>
      <div class="p-4 space-y-3">
        {#each $plots as plot}
          <div class="p-4 bg-gray-700/50 rounded-lg">
            <div class="flex justify-between items-start">
              <div>
                <p class="font-medium">{plot.name}</p>
                <p class="text-xs text-gray-400 mt-1">{plot.location}</p>
              </div>
              <div class="text-right">
                <span class="text-sm font-bold text-green-400">{plot.area_sqm} m&sup2;</span>
                <p class="text-xs text-gray-400 mt-1">{plot.plot_type}</p>
              </div>
            </div>
            <div class="flex justify-between text-xs text-gray-500 mt-2">
              <span>Owner: {plot.owner_did}</span>
              <span>Since {new Date(plot.created_at).toLocaleDateString()}</span>
            </div>
          </div>
        {:else}
          <p class="text-gray-500 text-center py-8">No plots registered yet. Be the first!</p>
        {/each}
      </div>
    </div>

    <div class="mt-6 flex justify-end gap-2">
      <button on:click={() => exportPlotsCsv($plots)}
        class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
        Export Plots CSV
      </button>
      <button on:click={() => exportHarvestsCsv($harvests)}
        class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
        Export Harvests CSV
      </button>
    </div>

    <footer class="mt-4 text-center text-gray-500 text-sm">
      <p>Food Production Tracker &middot; Mycelix Commons</p>
    </footer>
  </main>
</div>
{/if}
