<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { allTags, selectedTags } from '../stores/thoughts';

  export let showLabels = true;
  export let clusterByDomain = false;
  export let linkStrength = 0.5;

  const dispatch = createEventDispatcher();

  function handleReset() {
    dispatch('reset');
  }

  function handleZoomIn() {
    dispatch('zoom', { direction: 'in' });
  }

  function handleZoomOut() {
    dispatch('zoom', { direction: 'out' });
  }
</script>

<div class="graph-controls">
  <div class="control-group">
    <label class="toggle">
      <input type="checkbox" bind:checked={showLabels} />
      <span>Show labels</span>
    </label>

    <label class="toggle">
      <input type="checkbox" bind:checked={clusterByDomain} />
      <span>Cluster by domain</span>
    </label>
  </div>

  <div class="control-group">
    <label class="slider">
      <span>Link strength</span>
      <input type="range" min="0.1" max="1" step="0.1" bind:value={linkStrength} />
    </label>
  </div>

  <div class="control-group">
    <button class="btn" on:click={handleZoomIn} title="Zoom in">+</button>
    <button class="btn" on:click={handleZoomOut} title="Zoom out">−</button>
    <button class="btn" on:click={handleReset} title="Reset view">⟲</button>
  </div>

  {#if $allTags.length > 0}
    <div class="tag-filter">
      <span class="label">Filter by tag:</span>
      <div class="tags">
        {#each $allTags.slice(0, 8) as tag}
          <button
            class="tag"
            class:active={$selectedTags.includes(tag)}
            on:click={() => {
              if ($selectedTags.includes(tag)) {
                $selectedTags = $selectedTags.filter((t) => t !== tag);
              } else {
                $selectedTags = [...$selectedTags, tag];
              }
            }}
          >
            {tag}
          </button>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .graph-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    padding: 12px;
    background: #1a1a2e;
    border-radius: 8px;
    border: 1px solid #2a2a4e;
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: #888;
    cursor: pointer;
  }

  .toggle input {
    accent-color: #7c3aed;
  }

  .slider {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    color: #888;
  }

  .slider input {
    width: 80px;
    accent-color: #7c3aed;
  }

  .btn {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 6px;
    color: #888;
    font-size: 1.1rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn:hover {
    background: #3a3a5e;
    color: #fff;
  }

  .tag-filter {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
  }

  .tag-filter .label {
    font-size: 0.75rem;
    color: #666;
  }

  .tags {
    display: flex;
    gap: 4px;
  }

  .tag {
    padding: 2px 8px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    color: #888;
    font-size: 0.7rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tag:hover {
    border-color: #5a5a8e;
  }

  .tag.active {
    background: #7c3aed;
    border-color: #7c3aed;
    color: white;
  }
</style>
