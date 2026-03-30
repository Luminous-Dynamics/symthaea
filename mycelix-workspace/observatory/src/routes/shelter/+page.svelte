<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    getAvailableUnits,
    type HousingUnit,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';

  let units: HousingUnit[] = [];
  let loading = true;
  let error = '';
  let filterType = '';
  let filterAccessible = false;

  // Placement request form state
  let selectedUnitId: string | null = null;
  let formName = '';
  let formPhone = '';
  let formPeopleCount = 1;
  let formAccessibilityNeeds = '';
  let formSubmitted = false;
  let formSubmitting = false;

  onMount(async () => {
    try {
      units = await getAvailableUnits();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load shelter data';
    } finally {
      loading = false;
    }
  });

  $: filteredUnits = units.filter(u => {
    if (filterType && u.unit_type !== filterType) return false;
    if (filterAccessible && u.accessibility_features.length === 0) return false;
    return true;
  });

  $: selectedUnit = units.find(u => u.unit_number === selectedUnitId) ?? null;

  function unitTypeLabel(t: string): string {
    const labels: Record<string, string> = {
      Studio: 'Studio',
      OneBedroom: '1 Bed',
      TwoBedroom: '2 Bed',
      ThreeBedroom: '3 Bed',
      FourPlus: '4+ Bed',
      Accessible: 'Accessible',
      Family: 'Family',
    };
    return labels[t] ?? t;
  }

  function accessLabel(feature: string): string {
    const labels: Record<string, string> = {
      WheelchairAccessible: 'Wheelchair',
      Elevator: 'Elevator',
      GrabBars: 'Grab Bars',
      WideDoorways: 'Wide Doors',
      LowCounters: 'Low Counters',
      VisualAlerts: 'Visual Alerts',
      HearingLoop: 'Hearing Loop',
    };
    return labels[feature] ?? feature;
  }

  function openRequestForm(unitNumber: string) {
    selectedUnitId = unitNumber;
    formName = '';
    formPhone = '';
    formPeopleCount = 1;
    formAccessibilityNeeds = '';
    formSubmitted = false;
    formSubmitting = false;
  }

  function closeRequestForm() {
    selectedUnitId = null;
    formSubmitted = false;
  }

  async function submitRequest() {
    if (!formName.trim() || !formPhone.trim()) return;
    formSubmitting = true;
    try {
      // Demo mode: simulate a brief delay then confirm
      await new Promise(resolve => setTimeout(resolve, 600));
      formSubmitting = false;
      formSubmitted = true;
      toasts.success('Placement requested');
    } catch (e) {
      formSubmitting = false;
      toasts.error(e instanceof Error ? e.message : 'Failed to request placement');
    }
  }

  const UNIT_TYPES = ['Studio', 'OneBedroom', 'TwoBedroom', 'ThreeBedroom', 'FourPlus', 'Accessible', 'Family'];
</script>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <div class="container mx-auto max-w-6xl">
    <h1 class="text-2xl font-bold mb-1">Emergency Shelter</h1>
    <p class="text-gray-400 mb-6">Available housing units for emergency placement. Filter by size, accessibility, and location.</p>

    {#if loading}
      <div class="text-gray-400">Loading shelter data...</div>
    {:else if error}
      <div class="bg-red-900/30 border border-red-500 rounded p-4 text-red-200">{error}</div>
    {:else}

      <!-- Summary -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-emerald-400">{units.length}</div>
          <div class="text-sm text-gray-400">Available Units</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-emerald-400">{units.reduce((sum, u) => sum + u.bedrooms, 0)}</div>
          <div class="text-sm text-gray-400">Total Bedrooms</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-emerald-400">{units.filter(u => u.accessibility_features.length > 0).length}</div>
          <div class="text-sm text-gray-400">Accessible Units</div>
        </div>
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-3 mb-6">
        <div class="flex flex-wrap gap-2">
          <button
            class="px-3 py-1 rounded-full text-sm transition-colors {!filterType ? 'bg-emerald-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
            on:click={() => filterType = ''}
          >All Sizes</button>
          {#each UNIT_TYPES as ut}
            <button
              class="px-3 py-1 rounded-full text-sm transition-colors {filterType === ut ? 'bg-emerald-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
              on:click={() => filterType = ut}
            >{unitTypeLabel(ut)}</button>
          {/each}
        </div>
        <label class="flex items-center gap-2 text-sm text-gray-400 cursor-pointer ml-2">
          <input type="checkbox" bind:checked={filterAccessible} class="rounded" />
          Accessible only
        </label>
      </div>

      <!-- Units List -->
      {#if filteredUnits.length > 0}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          {#each filteredUnits as unit}
            <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
              <div class="flex items-start justify-between">
                <div>
                  <h3 class="font-semibold text-white">Unit {unit.unit_number}</h3>
                  <p class="text-sm text-gray-400">{unitTypeLabel(unit.unit_type)} | {unit.square_meters} m²</p>
                </div>
                <span class="text-xs px-2 py-1 rounded-full bg-emerald-900/50 text-emerald-300">{unit.status}</span>
              </div>

              <div class="grid grid-cols-3 gap-2 mt-3 text-center">
                <div class="bg-gray-800/50 rounded p-2">
                  <div class="text-lg font-bold text-white">{unit.bedrooms}</div>
                  <div class="text-xs text-gray-400">Beds</div>
                </div>
                <div class="bg-gray-800/50 rounded p-2">
                  <div class="text-lg font-bold text-white">{unit.bathrooms}</div>
                  <div class="text-xs text-gray-400">Baths</div>
                </div>
                <div class="bg-gray-800/50 rounded p-2">
                  <div class="text-lg font-bold text-white">{unit.floor === 0 ? 'G' : unit.floor}</div>
                  <div class="text-xs text-gray-400">Floor</div>
                </div>
              </div>

              {#if unit.accessibility_features.length > 0}
                <div class="flex flex-wrap gap-1 mt-3">
                  {#each unit.accessibility_features as feature}
                    <span class="px-2 py-0.5 rounded text-xs bg-blue-900/50 text-blue-300">{accessLabel(feature)}</span>
                  {/each}
                </div>
              {/if}

              <button
                class="mt-3 w-full py-2 rounded text-sm font-medium transition-colors bg-emerald-700 hover:bg-emerald-600 text-white"
                on:click={() => openRequestForm(unit.unit_number)}
              >Request Placement</button>
            </div>
          {/each}
        </div>
      {:else}
        <div class="text-center text-gray-500 py-12">
          <p class="text-lg">No units match your filters.</p>
          <p class="text-sm mt-1">Try adjusting the size or accessibility filters.</p>
        </div>
      {/if}
    {/if}
  </div>
</div>

<!-- Placement Request Form Overlay -->
{#if selectedUnitId !== null}
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <div
    class="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
    on:click|self={closeRequestForm}
    on:keydown={e => { if (e.key === 'Escape') closeRequestForm(); }}
    role="dialog"
    tabindex="-1"
    aria-modal="true"
    aria-label="Request placement form"
  >
    <div class="bg-gray-800 rounded-lg border border-gray-700 w-full max-w-md p-4 sm:p-6">
      {#if formSubmitted}
        <div class="text-center py-4">
          <div class="text-emerald-400 text-4xl mb-3" aria-hidden="true">&#10003;</div>
          <h2 class="text-xl font-bold text-white mb-2">Request Submitted</h2>
          <p class="text-gray-400 text-sm mb-1">
            Your placement request for <span class="text-emerald-300 font-medium">Unit {selectedUnitId}</span> has been received.
          </p>
          <p class="text-gray-500 text-xs mb-4">A shelter coordinator will contact you to discuss next steps.</p>
          <button
            class="px-6 py-2 rounded text-sm font-medium bg-emerald-700 hover:bg-emerald-600 text-white transition-colors"
            on:click={closeRequestForm}
          >Close</button>
        </div>
      {:else}
        <div class="flex items-start justify-between mb-4">
          <div>
            <h2 class="text-lg font-bold text-white">Request Placement</h2>
            {#if selectedUnit}
              <p class="text-sm text-gray-400">Unit {selectedUnit.unit_number} &mdash; {unitTypeLabel(selectedUnit.unit_type)}, {selectedUnit.bedrooms} bed / {selectedUnit.bathrooms} bath</p>
            {/if}
          </div>
          <button
            class="text-gray-500 hover:text-gray-300 transition-colors text-xl leading-none"
            on:click={closeRequestForm}
            aria-label="Close form"
          >&times;</button>
        </div>

        <form on:submit|preventDefault={submitRequest} class="space-y-4">
          <div>
            <label for="req-name" class="block text-sm text-gray-400 mb-1">Name <span class="text-red-400">*</span></label>
            <input
              id="req-name"
              type="text"
              bind:value={formName}
              required
              class="w-full rounded bg-gray-900 border border-gray-700 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              placeholder="Full name"
            />
          </div>

          <div>
            <label for="req-phone" class="block text-sm text-gray-400 mb-1">Phone <span class="text-red-400">*</span></label>
            <input
              id="req-phone"
              type="tel"
              bind:value={formPhone}
              required
              class="w-full rounded bg-gray-900 border border-gray-700 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              placeholder="Contact number"
            />
          </div>

          <div>
            <label for="req-people" class="block text-sm text-gray-400 mb-1">Number of People</label>
            <input
              id="req-people"
              type="number"
              min="1"
              max="20"
              bind:value={formPeopleCount}
              class="w-full rounded bg-gray-900 border border-gray-700 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>

          <div>
            <label for="req-access" class="block text-sm text-gray-400 mb-1">Accessibility Needs</label>
            <textarea
              id="req-access"
              bind:value={formAccessibilityNeeds}
              rows="3"
              class="w-full rounded bg-gray-900 border border-gray-700 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 resize-none"
              placeholder="Wheelchair access, ground floor, medical equipment space, etc."
            ></textarea>
          </div>

          <button
            type="submit"
            disabled={formSubmitting || !formName.trim() || !formPhone.trim()}
            class="w-full py-2 rounded text-sm font-medium transition-colors
              {formSubmitting || !formName.trim() || !formPhone.trim()
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-emerald-700 hover:bg-emerald-600 text-white'}"
          >
            {#if formSubmitting}
              Submitting...
            {:else}
              Submit Request
            {/if}
          </button>

          <p class="text-xs text-gray-500 text-center">A coordinator will review your request and follow up within 24 hours.</p>
        </form>
      {/if}
    </div>
  </div>
{/if}
