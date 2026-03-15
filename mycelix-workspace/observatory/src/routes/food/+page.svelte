<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import {
    getAllPlots,
    registerPlot,
    recordHarvest,
    type FoodPlot,
    type HarvestRecord,
  } from '$lib/resilience-client';

  // ============================================================================
  // Stores
  // ============================================================================

  const plots = writable<FoodPlot[]>([]);
  const harvests = writable<HarvestRecord[]>([]);

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

  let submitting = false;

  const plotTypes = ['Raised beds', 'Open field', 'Greenhouse tunnel', 'Container garden', 'Vertical garden', 'Rooftop'];

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    try {
      const p = await getAllPlots();
      plots.set(p);
    } catch (e) {
      console.warn('[Food] Failed to load plots, using defaults:', e);
    }
  });

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
</script>

<svelte:head>
  <title>Food Production | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">&#x1F331;</span>
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
      <button on:click={() => { showPlotForm = !showPlotForm; showHarvestForm = false; }}
        class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors">
        + Register Plot
      </button>
      <button on:click={() => { showHarvestForm = !showHarvestForm; showPlotForm = false; }}
        class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm font-medium transition-colors"
        disabled={$plots.length === 0}>
        + Record Harvest
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
            <input id="ploc" bind:value={plotLocation} placeholder="Florida, Roodepoort"
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

    <!-- Plot List -->
    <div class="bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700">
        <h2 class="text-lg font-semibold flex items-center gap-2">
          <span>&#x1F33F;</span> Community Plots
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

    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Food Production Tracker &middot; Mycelix Commons</p>
    </footer>
  </main>
</div>
