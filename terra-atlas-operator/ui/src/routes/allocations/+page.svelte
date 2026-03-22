<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
--><script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';

  interface ProjectSummary {
    id: string;
    name: string;
    project_type: string;
    capacity_mw: number;
    status: string;
    funded_percentage: number;
    phi_score: number | null;
    harmony_alignment: number | null;
  }

  const HARMONY_OPTIONS = [
    'Resonant Coherence',
    'Pan-Sentient Flourishing',
    'Ecological Reciprocity',
    'Epistemic Humility',
    'Radical Translucency',
    'Compassionate Action',
    'Temporal Stewardship',
    'Sacred Stillness',
  ];

  let projects: ProjectSummary[] = $state([]);
  let selectedProject: ProjectSummary | null = $state(null);
  let pledgeAmount = $state('');
  let currency = $state('TEND');
  let harmonyIntent = $state('Ecological Reciprocity');
  let submitting = $state(false);
  let submitResult: string | null = $state(null);
  let submitError: string | null = $state(null);

  // Trust profile (would come from conductor in production)
  let trustScore = $state(0.55);
  let trustTier = $state('Citizen');
  const MIN_TRUST = 0.2;

  onMount(async () => {
    try {
      projects = await invoke<ProjectSummary[]>('get_projects');
    } catch (e) {
      console.warn('Using demo data:', e);
    }
  });

  async function submitPledge() {
    if (!selectedProject || !pledgeAmount) return;
    submitting = true;
    submitResult = null;
    submitError = null;

    try {
      const result = await invoke<string>('submit_pledge', {
        projectId: selectedProject.id,
        amount: parseInt(pledgeAmount),
        currency,
        harmonyIntent,
        trustScore,
        trustTier,
        pledgerDid: 'did:mycelix:local-agent', // Would come from conductor agent key
      });
      submitResult = result;
      pledgeAmount = '';
    } catch (e) {
      submitError = String(e);
    } finally {
      submitting = false;
    }
  }

  function trustColor(score: number): string {
    if (score >= 0.8) return 'text-violet-400';
    if (score >= 0.6) return 'text-blue-400';
    if (score >= 0.4) return 'text-cyan-400';
    return 'text-amber-400';
  }
</script>

<main class="min-h-screen bg-gray-950 text-white p-8">
  <header class="mb-8">
    <a href="/" class="text-gray-400 hover:text-white text-sm mb-2 inline-block">&larr; Dashboard</a>
    <h1 class="text-2xl font-bold">Allocations</h1>
    <p class="text-gray-400 text-sm mt-1">Pledge resources toward regenerative projects via trust-weighted matching</p>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Project List -->
    <div class="lg:col-span-2 space-y-3">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Available Projects</h2>

      {#each projects as project}
        <button
          onclick={() => selectedProject = project}
          class="w-full text-left rounded-lg p-4 border transition-all
            {selectedProject?.id === project.id
              ? 'bg-violet-900/20 border-violet-500/50'
              : 'bg-gray-900 border-gray-800 hover:border-gray-700'}"
        >
          <div class="flex items-center justify-between">
            <div>
              <span class="font-medium">{project.name}</span>
              <div class="text-xs text-gray-400 mt-0.5">
                {project.project_type} &middot; {project.capacity_mw} MW &middot; {project.status}
              </div>
            </div>
            <div class="text-right">
              {#if project.phi_score !== null}
                <span class="font-mono text-sm {trustColor(project.phi_score)}">
                  &#934; {(project.phi_score * 100).toFixed(0)}
                </span>
              {:else}
                <span class="text-gray-600 text-xs">Unscored</span>
              {/if}
              <div class="text-xs text-gray-500 mt-0.5">{project.funded_percentage}% funded</div>
            </div>
          </div>
        </button>
      {/each}

      {#if projects.length === 0}
        <p class="text-gray-500 text-sm">No projects available. Connect to conductor first.</p>
      {/if}
    </div>

    <!-- Pledge Panel -->
    <div class="space-y-4">
      <!-- Trust Profile -->
      <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
        <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Your Trust Profile</h3>
        <div class="flex items-center justify-between mb-2">
          <span class="font-mono text-lg {trustColor(trustScore)}">&#934; {(trustScore * 100).toFixed(0)}</span>
          <span class="text-xs bg-gray-800 px-2 py-1 rounded {trustColor(trustScore)}">{trustTier}</span>
        </div>
        {#if trustScore >= MIN_TRUST}
          <span class="text-xs text-emerald-400">Eligible to pledge</span>
        {:else}
          <span class="text-xs text-red-400">Requires Participant tier (&#934; &ge; {MIN_TRUST * 100})</span>
        {/if}
        <p class="text-gray-600 text-xs mt-2">Loaded from your Holochain source chain</p>
      </div>

      <!-- Pledge Form -->
      {#if selectedProject}
        <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
          <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Pledge to</h3>
          <p class="text-sm text-white font-medium mb-4">{selectedProject.name}</p>

          <div class="space-y-3">
            <div>
              <label class="text-xs text-gray-400 block mb-1">Amount</label>
              <div class="flex gap-2">
                <input
                  type="number" min="1" max="40"
                  bind:value={pledgeAmount}
                  placeholder="0"
                  disabled={trustScore < MIN_TRUST}
                  class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono focus:border-violet-500 focus:outline-none disabled:opacity-50"
                />
                <select bind:value={currency}
                  class="bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-sm focus:border-violet-500 focus:outline-none">
                  <option>TEND</option>
                  <option>SAP</option>
                </select>
              </div>
              <p class="text-xs text-gray-600 mt-1">TEND balance cap: &pm;40</p>
            </div>

            <div>
              <label class="text-xs text-gray-400 block mb-1">Harmony Intent</label>
              <select bind:value={harmonyIntent}
                disabled={trustScore < MIN_TRUST}
                class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:border-violet-500 focus:outline-none disabled:opacity-50">
                {#each HARMONY_OPTIONS as h}
                  <option>{h}</option>
                {/each}
              </select>
            </div>

            <button
              onclick={submitPledge}
              disabled={trustScore < MIN_TRUST || !pledgeAmount || submitting}
              class="w-full py-2.5 rounded-lg font-medium text-sm transition-all
                bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500
                text-white shadow-lg shadow-violet-500/20
                disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {submitting ? 'Submitting to DHT...' : `Pledge ${pledgeAmount || '...'} ${currency}`}
            </button>

            {#if submitResult}
              <p class="text-emerald-400 text-sm text-center">{submitResult}</p>
            {/if}
            {#if submitError}
              <p class="text-red-400 text-sm text-center">{submitError}</p>
            {/if}

            <p class="text-xs text-gray-600 text-center">
              Pledges expire in 24h. Stored on your source chain. Demurrage applies.
            </p>
          </div>
        </div>
      {:else}
        <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
          <p class="text-gray-500 text-sm">Select a project to pledge resources.</p>
        </div>
      {/if}

      <!-- Anti-Speculation Notice -->
      <div class="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
        <h4 class="text-xs font-semibold text-gray-500 mb-2">Anti-Speculation</h4>
        <ul class="text-xs text-gray-500 space-y-1">
          <li>5%/month demurrage on matched pledges</li>
          <li>No secondary market for matches</li>
          <li>No leverage or margin</li>
          <li>Trust score re-checked at confirmation</li>
          <li>All data sovereign on your source chain</li>
        </ul>
      </div>
    </div>
  </div>
</main>
