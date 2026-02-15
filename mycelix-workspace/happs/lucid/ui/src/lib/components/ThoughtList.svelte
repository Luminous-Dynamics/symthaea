<script lang="ts">
  import ThoughtCard from './ThoughtCard.svelte';
  import { filteredThoughts, isLoading, error, searchQuery, allTags, selectedTags, minConfidence } from '../stores/thoughts';

  export let compact = false;
</script>

<div class="thought-list" class:compact>
  <!-- Search & Filter Bar -->
  <div class="filters">
    <input
      type="search"
      placeholder="Search thoughts..."
      bind:value={$searchQuery}
      class="search-input"
    />

    <div class="filter-row">
      <label class="confidence-filter">
        <span>Min confidence:</span>
        <input type="range" min="0" max="1" step="0.1" bind:value={$minConfidence} />
        <span class="value">{Math.round($minConfidence * 100)}%</span>
      </label>
    </div>

    {#if $allTags.length > 0}
      <div class="tag-filters">
        {#each $allTags.slice(0, 10) as tag}
          <button
            class="tag-btn"
            class:active={$selectedTags.includes(tag)}
            on:click={() => {
              if ($selectedTags.includes(tag)) {
                $selectedTags = $selectedTags.filter((t) => t !== tag);
              } else {
                $selectedTags = [...$selectedTags, tag];
              }
            }}
          >
            #{tag}
          </button>
        {/each}
        {#if $selectedTags.length > 0}
          <button class="clear-btn" on:click={() => ($selectedTags = [])}>
            Clear
          </button>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Error Display -->
  {#if $error}
    <div class="error">
      <span>⚠</span> {$error}
    </div>
  {/if}

  <!-- Loading State -->
  {#if $isLoading}
    <div class="loading">
      <div class="spinner"></div>
      <span>Loading thoughts...</span>
    </div>
  {/if}

  <!-- Thoughts Grid -->
  {#if !$isLoading}
    {#if $filteredThoughts.length === 0}
      <div class="empty">
        {#if $searchQuery || $selectedTags.length > 0}
          <p>No thoughts match your filters.</p>
          <button on:click={() => { $searchQuery = ''; $selectedTags = []; $minConfidence = 0; }}>
            Clear filters
          </button>
        {:else}
          <p>No thoughts yet. Create your first thought above!</p>
        {/if}
      </div>
    {:else}
      <div class="grid">
        {#each $filteredThoughts as thought (thought.id)}
          <ThoughtCard {thought} />
        {/each}
      </div>
      <div class="count">
        Showing {$filteredThoughts.length} thought{$filteredThoughts.length !== 1 ? 's' : ''}
      </div>
    {/if}
  {/if}
</div>

<style>
  .thought-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .filters {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    background: #1a1a2e;
    border-radius: 8px;
  }

  .search-input {
    width: 100%;
    padding: 10px 14px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    color: #e5e5e5;
    font-size: 0.95rem;
  }

  .search-input:focus {
    outline: none;
    border-color: #7c3aed;
  }

  .search-input::placeholder {
    color: #666;
  }

  .filter-row {
    display: flex;
    gap: 16px;
  }

  .confidence-filter {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    color: #888;
  }

  .confidence-filter .value {
    font-family: monospace;
    min-width: 35px;
  }

  .tag-filters {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tag-btn {
    padding: 4px 10px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    color: #888;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tag-btn:hover {
    border-color: #5a5a8e;
    color: #aaa;
  }

  .tag-btn.active {
    background: #7c3aed;
    border-color: #7c3aed;
    color: white;
  }

  .clear-btn {
    padding: 4px 10px;
    background: none;
    border: none;
    color: #666;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .clear-btn:hover {
    color: #ef4444;
  }

  .error {
    padding: 12px 16px;
    background: #7f1d1d;
    border: 1px solid #991b1b;
    border-radius: 8px;
    color: #fecaca;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 40px;
    color: #888;
  }

  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #333;
    border-top-color: #7c3aed;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .empty {
    text-align: center;
    padding: 40px;
    color: #666;
  }

  .empty button {
    margin-top: 12px;
    padding: 8px 16px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 6px;
    color: #888;
    cursor: pointer;
  }

  .empty button:hover {
    border-color: #5a5a8e;
    color: #aaa;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }

  .compact .grid {
    grid-template-columns: 1fr;
  }

  .count {
    text-align: center;
    font-size: 0.8rem;
    color: #555;
    padding: 8px;
  }
</style>
