<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';

  interface Claim {
    id: string;
    epistemic_tier: string;
    category: string;
    description: string;
    created_at: string;
  }

  let claims: Claim[] = [];
  let searchQuery = '';
  let selectedTier = '';

  // Mock data for demonstration
  const mockClaims: Claim[] = [
    {
      id: '1',
      epistemic_tier: 'E4',
      category: 'longevity',
      description: 'Reproducible study on NAD+ supplementation effects',
      created_at: '2025-11-01T10:00:00Z'
    },
    {
      id: '2',
      epistemic_tier: 'E3',
      category: 'genomics',
      description: 'CRISPR-Cas9 gene editing methodology for cancer research',
      created_at: '2025-11-10T14:30:00Z'
    },
    {
      id: '3',
      epistemic_tier: 'E2',
      category: 'climate',
      description: 'Multi-source verification of ocean temperature datasets',
      created_at: '2025-11-12T09:15:00Z'
    }
  ];

  onMount(() => {
    // In production, this would fetch from the API
    claims = mockClaims;
  });

  $: filteredClaims = claims.filter(claim => {
    const matchesSearch = searchQuery === '' ||
      claim.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      claim.category.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesTier = selectedTier === '' || claim.epistemic_tier === selectedTier;

    return matchesSearch && matchesTier;
  });

  function getTierColor(tier: string): string {
    const colors: Record<string, string> = {
      'E0': 'bg-gray-500',
      'E1': 'bg-blue-500',
      'E2': 'bg-green-500',
      'E3': 'bg-yellow-500',
      'E4': 'bg-purple-500'
    };
    return colors[tier] || 'bg-gray-500';
  }
</script>

<main>
  <header>
    <h1>Mycelix-DeSci</h1>
    <p class="tagline">Verifiable, Privacy-Preserving Infrastructure for Decentralized Science</p>
  </header>

  <div class="search-section">
    <input
      type="text"
      placeholder="Search claims..."
      bind:value={searchQuery}
      class="search-input"
    />

    <select bind:value={selectedTier} class="tier-select">
      <option value="">All Tiers</option>
      <option value="E0">E0 - Unverified</option>
      <option value="E1">E1 - Single Source</option>
      <option value="E2">E2 - Multi Source</option>
      <option value="E3">E3 - Reproducible</option>
      <option value="E4">E4 - Peer Reviewed</option>
    </select>
  </div>

  <div class="claims-grid">
    {#each filteredClaims as claim (claim.id)}
      <div class="claim-card">
        <div class="claim-header">
          <span class="tier-badge {getTierColor(claim.epistemic_tier)}">
            {claim.epistemic_tier}
          </span>
          <span class="category-badge">{claim.category}</span>
        </div>
        <p class="claim-description">{claim.description}</p>
        <div class="claim-footer">
          <span class="claim-date">
            {new Date(claim.created_at).toLocaleDateString()}
          </span>
          <button class="view-button">View Details</button>
        </div>
      </div>
    {/each}
  </div>

  {#if filteredClaims.length === 0}
    <p class="no-results">No claims found matching your criteria.</p>
  {/if}
</main>

<style>
  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    text-align: center;
    margin-bottom: 3rem;
  }

  h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(to right, #646cff, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .tagline {
    font-size: 1.1rem;
    opacity: 0.8;
  }

  .search-section {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .search-input {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid #444;
    border-radius: 8px;
    background: #1a1a1a;
    color: inherit;
    font-size: 1rem;
  }

  .tier-select {
    padding: 0.75rem;
    border: 1px solid #444;
    border-radius: 8px;
    background: #1a1a1a;
    color: inherit;
    font-size: 1rem;
    cursor: pointer;
  }

  .claims-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  .claim-card {
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 1.5rem;
    transition: transform 0.2s, border-color 0.2s;
  }

  .claim-card:hover {
    transform: translateY(-4px);
    border-color: #646cff;
  }

  .claim-header {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  .tier-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 600;
    color: white;
  }

  .category-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.875rem;
    background: #2a2a2a;
    color: #aaa;
  }

  .claim-description {
    margin-bottom: 1rem;
    line-height: 1.6;
  }

  .claim-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .claim-date {
    font-size: 0.875rem;
    opacity: 0.7;
  }

  .view-button {
    padding: 0.5rem 1rem;
    background: #646cff;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
  }

  .view-button:hover {
    background: #535bf2;
  }

  .no-results {
    text-align: center;
    padding: 3rem;
    opacity: 0.6;
  }

  .bg-gray-500 { background-color: #6b7280; }
  .bg-blue-500 { background-color: #3b82f6; }
  .bg-green-500 { background-color: #22c55e; }
  .bg-yellow-500 { background-color: #eab308; }
  .bg-purple-500 { background-color: #a855f7; }
</style>
