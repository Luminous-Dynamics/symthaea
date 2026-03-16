<script lang="ts">
  import { onMount } from 'svelte';
  import {
    searchClaimsByTag,
    getGraphStats,
    submitClaim,
    type KnowledgeClaim,
    type GraphStats,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';

  let claims: KnowledgeClaim[] = [];
  let stats: GraphStats | null = null;
  let loading = true;
  let error = '';
  let searchTag = '';
  let activeTag = '';

  let showForm = false;
  let submitting = false;
  let formContent = '';
  let formTags = '';
  let formConfidence = 0.7;

  const RESILIENCE_TAGS = [
    'food-production', 'water-safety', 'first-aid', 'energy',
    'emergency', 'maintenance', 'permaculture', 'cost-saving',
    'communications', 'health', 'baking', 'infrastructure',
  ];

  onMount(async () => {
    try {
      stats = await getGraphStats();
      // Load all claims initially via common resilience tag
      claims = await searchClaimsByTag('food-production');
      activeTag = 'food-production';
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load knowledge data';
    } finally {
      loading = false;
    }
  });

  async function searchByTag(tag: string) {
    activeTag = tag;
    searchTag = '';
    loading = true;
    try {
      claims = await searchClaimsByTag(tag);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Search failed';
    } finally {
      loading = false;
    }
  }

  async function handleSearch() {
    if (!searchTag.trim()) return;
    await searchByTag(searchTag.trim().toLowerCase());
  }

  async function handleSubmitClaim() {
    if (!formContent.trim()) return;
    submitting = true;
    error = '';
    try {
      const tags = formTags
        .split(',')
        .map((t) => t.trim().toLowerCase())
        .filter((t) => t.length > 0);
      const result = await submitClaim(formContent.trim(), tags, formConfidence);
      claims = [result, ...claims];
      showForm = false;
      formContent = '';
      formTags = '';
      formConfidence = 0.7;
      toasts.success('Claim submitted');
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to submit claim';
      toasts.error(error);
    } finally {
      submitting = false;
    }
  }

  function confidenceColor(c: number): string {
    if (c >= 0.9) return 'text-green-400';
    if (c >= 0.7) return 'text-cyan-400';
    if (c >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  }

  function epistemicBar(score: number): string {
    return `${Math.round(score * 100)}%`;
  }
</script>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <div class="container mx-auto max-w-6xl">
    <h1 class="text-2xl font-bold mb-1">Community Knowledge</h1>
    <p class="text-gray-400 mb-6">Shared expertise, practical know-how, and fact-checked claims for community resilience.</p>

    {#if error}
      <div class="bg-red-900/30 border border-red-500 rounded p-4 text-red-200 mb-4">{error}</div>
    {/if}

    <!-- Stats -->
    {#if stats}
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-2xl font-bold text-purple-400">{stats.total_claims}</div>
          <div class="text-xs text-gray-400">Claims</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-2xl font-bold text-purple-400">{stats.total_relationships}</div>
          <div class="text-xs text-gray-400">Relationships</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-2xl font-bold text-purple-400">{stats.total_ontologies}</div>
          <div class="text-xs text-gray-400">Ontologies</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-2xl font-bold text-purple-400">{stats.total_concepts}</div>
          <div class="text-xs text-gray-400">Concepts</div>
        </div>
      </div>
    {/if}

    <!-- Share Knowledge Toggle -->
    <div class="mb-4">
      <button
        class="px-4 py-2 rounded-lg text-white transition-colors {showForm ? 'bg-gray-700 hover:bg-gray-600' : 'bg-purple-700 hover:bg-purple-600'}"
        on:click={() => (showForm = !showForm)}
      >
        {showForm ? 'Cancel' : '+ Share Knowledge'}
      </button>
    </div>

    <!-- Submit Claim Form -->
    {#if showForm}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-lg font-semibold text-purple-300 mb-4">Share Knowledge</h2>
        <form on:submit|preventDefault={handleSubmitClaim} class="space-y-4">
          <div>
            <label for="claim-content" class="block text-sm text-gray-300 mb-1">Content</label>
            <textarea
              id="claim-content"
              bind:value={formContent}
              placeholder="What practical knowledge can you share?"
              rows="4"
              class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none resize-y"
              disabled={submitting}
            ></textarea>
          </div>

          <div>
            <label for="claim-tags" class="block text-sm text-gray-300 mb-1">Tags</label>
            <input
              id="claim-tags"
              type="text"
              bind:value={formTags}
              placeholder="water-safety, first-aid, permaculture"
              class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none"
              disabled={submitting}
            />
          </div>

          <div>
            <label for="claim-confidence" class="block text-sm text-gray-300 mb-1">
              Confidence: <span class="text-purple-400 font-semibold">{Math.round(formConfidence * 100)}%</span>
            </label>
            <input
              id="claim-confidence"
              type="range"
              bind:value={formConfidence}
              min="0.1"
              max="1.0"
              step="0.05"
              class="w-full accent-purple-500"
              disabled={submitting}
            />
          </div>

          <button
            type="submit"
            class="px-6 py-2 bg-purple-700 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
            disabled={submitting || !formContent.trim()}
          >
            {submitting ? 'Submitting...' : 'Submit Claim'}
          </button>
        </form>
      </div>
    {/if}

    <!-- Search -->
    <div class="mb-4">
      <form on:submit|preventDefault={handleSearch} class="flex gap-2">
        <input
          type="text"
          bind:value={searchTag}
          placeholder="Search by topic (e.g., water-safety, first-aid)..."
          aria-label="Search knowledge by topic"
          class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none"
        />
        <button type="submit" class="px-4 py-2 bg-purple-700 hover:bg-purple-600 rounded-lg text-white transition-colors">Search</button>
      </form>
    </div>

    <!-- Tag Cloud -->
    <div class="flex flex-wrap gap-2 mb-6">
      {#each RESILIENCE_TAGS as tag}
        <button
          class="px-3 py-1 rounded-full text-sm transition-colors {activeTag === tag ? 'bg-purple-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'}"
          on:click={() => searchByTag(tag)}
        >
          {tag}
        </button>
      {/each}
    </div>

    <!-- Results -->
    {#if loading}
      <div class="text-gray-400">Searching...</div>
    {:else if claims.length === 0}
      <div class="text-center text-gray-500 py-12">
        <p class="text-lg">No claims found for "{activeTag}"</p>
        <p class="text-sm mt-1">Be the first to share knowledge on this topic.</p>
      </div>
    {:else}
      <div class="space-y-3">
        {#each claims as claim}
          <div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
            <p class="text-white mb-2">{claim.content}</p>

            <div class="flex items-center gap-4 text-xs">
              <!-- Confidence -->
              <span class="font-semibold {confidenceColor(claim.confidence)}">
                {Math.round(claim.confidence * 100)}% confidence
              </span>

              <!-- E-N-M scores -->
              <div class="flex items-center gap-2 text-gray-500">
                <span title="Empirical evidence">E:{epistemicBar(claim.e_score)}</span>
                <span title="Normative endorsement">N:{epistemicBar(claim.n_score)}</span>
                <span title="Materiality / impact">M:{epistemicBar(claim.m_score)}</span>
              </div>

              <span class="text-gray-600">|</span>
              <span class="text-gray-500">{claim.author_did}</span>
              <span class="text-gray-500">{new Date(claim.created_at).toLocaleDateString()}</span>
            </div>

            <!-- Tags -->
            <div class="flex flex-wrap gap-1 mt-2">
              {#each claim.tags as tag}
                <button
                  class="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 transition-colors"
                  on:click={() => searchByTag(tag)}
                >
                  {tag}
                </button>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>
