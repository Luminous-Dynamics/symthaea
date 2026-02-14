<script lang="ts">
  import type { Thought } from '@mycelix/lucid-client';
  import {
    generateAugmentationReport,
    generateSteelman,
    detectBiases,
    detectBlindSpots,
    simulatePerspective,
    getAvailablePerspectives,
    type AugmentationReport,
    type SteelmanArgument,
    type CognitiveBias,
    type BlindSpot,
    type PerspectiveSimulation,
  } from '../services/cognitive-augmentation';

  export let thought: Thought;
  export let recentThoughts: Thought[] = [];
  export let useLLM: boolean = false;

  let activeTab: 'overview' | 'steelman' | 'biases' | 'blindspots' | 'perspectives' = 'overview';
  let report: AugmentationReport | null = null;
  let loading = false;
  let selectedPerspective: string = '';

  async function analyzeThought() {
    loading = true;
    try {
      report = await generateAugmentationReport(thought, recentThoughts, useLLM);
    } catch (error) {
      console.error('Augmentation failed:', error);
    }
    loading = false;
  }

  function getBiasIcon(biasType: string): string {
    const icons: Record<string, string> = {
      confirmation: '🔄',
      anchoring: '⚓',
      availability: '📰',
      bandwagon: '🚃',
      authority: '👔',
      framing: '🖼️',
      sunk_cost: '💸',
      hindsight: '🔮',
    };
    return icons[biasType] || '🧠';
  }

  function getSeverityColor(severity: string): string {
    switch (severity) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      default: return 'text-blue-400';
    }
  }

  function getScoreColor(score: number): string {
    if (score >= 0.7) return 'text-green-400';
    if (score >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  }

  function formatBiasName(name: string): string {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  async function addPerspective() {
    if (!selectedPerspective || !report) return;
    const sim = simulatePerspective(thought, selectedPerspective);
    report.perspectives = [...report.perspectives, sim];
    selectedPerspective = '';
  }
</script>

<div class="cognitive-augmentation bg-gray-900 rounded-lg p-4 border border-gray-700">
  <div class="header flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-purple-300">Cognitive Augmentation</h3>
    <button
      on:click={analyzeThought}
      disabled={loading}
      class="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 rounded text-sm transition-colors"
    >
      {loading ? 'Analyzing...' : 'Analyze Thought'}
    </button>
  </div>

  {#if !report}
    <div class="empty-state text-center py-8 text-gray-500">
      <div class="text-4xl mb-2">🧠</div>
      <p>Click "Analyze Thought" to receive cognitive augmentation</p>
      <p class="text-sm mt-1">Steelman arguments, bias detection, blind spots, and perspectives</p>
    </div>
  {:else}
    <!-- Tabs -->
    <div class="tabs flex gap-1 mb-4 border-b border-gray-700 pb-2">
      {#each ['overview', 'steelman', 'biases', 'blindspots', 'perspectives'] as tab (tab)}
        <button
          on:click={() => activeTab = tab as typeof activeTab}
          class="px-3 py-1 rounded-t text-sm transition-colors {activeTab === tab
            ? 'bg-gray-700 text-white'
            : 'text-gray-400 hover:text-gray-200'}"
        >
          {tab.charAt(0).toUpperCase() + tab.slice(1)}
          {#if tab === 'biases' && report.biases.length > 0}
            <span class="ml-1 px-1.5 py-0.5 bg-yellow-600 rounded-full text-xs">{report.biases.length}</span>
          {/if}
          {#if tab === 'blindspots' && report.blindSpots.length > 0}
            <span class="ml-1 px-1.5 py-0.5 bg-blue-600 rounded-full text-xs">{report.blindSpots.length}</span>
          {/if}
        </button>
      {/each}
    </div>

    <!-- Tab Content -->
    <div class="tab-content">
      {#if activeTab === 'overview'}
        <div class="overview space-y-4">
          <!-- Score -->
          <div class="score-card bg-gray-800 rounded p-4">
            <div class="flex items-center justify-between">
              <span class="text-gray-400">Cognitive Quality Score</span>
              <span class="text-2xl font-bold {getScoreColor(report.overallScore)}">
                {Math.round(report.overallScore * 100)}%
              </span>
            </div>
            <div class="mt-2 h-2 bg-gray-700 rounded overflow-hidden">
              <div
                class="h-full transition-all duration-500 {report.overallScore >= 0.7 ? 'bg-green-500' : report.overallScore >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}"
                style="width: {report.overallScore * 100}%"
              ></div>
            </div>
          </div>

          <!-- Quick Summary -->
          <div class="grid grid-cols-3 gap-3">
            <div class="stat bg-gray-800 rounded p-3 text-center">
              <div class="text-2xl">{report.biases.length}</div>
              <div class="text-xs text-gray-400">Potential Biases</div>
            </div>
            <div class="stat bg-gray-800 rounded p-3 text-center">
              <div class="text-2xl">{report.blindSpots.length}</div>
              <div class="text-xs text-gray-400">Blind Spots</div>
            </div>
            <div class="stat bg-gray-800 rounded p-3 text-center">
              <div class="text-2xl">{report.perspectives.length}</div>
              <div class="text-xs text-gray-400">Perspectives</div>
            </div>
          </div>

          <!-- Recommendations -->
          <div class="recommendations bg-gray-800 rounded p-4">
            <h4 class="text-sm font-semibold text-gray-300 mb-2">Recommendations</h4>
            <ul class="space-y-2">
              {#each report.recommendations as rec}
                <li class="flex items-start gap-2 text-sm">
                  <span class="text-purple-400">→</span>
                  <span class="text-gray-300">{rec}</span>
                </li>
              {/each}
            </ul>
          </div>
        </div>

      {:else if activeTab === 'steelman'}
        <div class="steelman space-y-4">
          {#if report.steelman}
            <div class="original bg-gray-800 rounded p-4">
              <h4 class="text-sm font-semibold text-gray-400 mb-2">Your Position</h4>
              <p class="text-gray-300">{report.steelman.originalPosition}</p>
            </div>

            <div class="steelman-version bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded p-4 border border-purple-700/50">
              <h4 class="text-sm font-semibold text-purple-300 mb-2">
                Strongest Counter-Position
                <span class="ml-2 text-xs text-gray-400">
                  Strength: {Math.round(report.steelman.strengthScore * 100)}%
                </span>
              </h4>
              <p class="text-gray-200">{report.steelman.steelmanVersion}</p>
            </div>

            <div class="counter-arguments">
              <h4 class="text-sm font-semibold text-gray-400 mb-2">Counter-Arguments</h4>
              <div class="space-y-2">
                {#each report.steelman.counterArguments as counter}
                  <div class="counter bg-gray-800 rounded p-3">
                    <div class="flex items-center justify-between mb-1">
                      <span class="text-xs px-2 py-0.5 bg-gray-700 rounded text-gray-300">
                        {counter.type}
                      </span>
                      <span class="text-xs text-gray-500">
                        {counter.sourceFramework}
                      </span>
                    </div>
                    <p class="text-sm text-gray-300">{counter.argument}</p>
                  </div>
                {/each}
              </div>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-4">No steelman generated</p>
          {/if}
        </div>

      {:else if activeTab === 'biases'}
        <div class="biases space-y-3">
          {#if report.biases.length === 0}
            <div class="empty text-center py-8 text-gray-500">
              <div class="text-4xl mb-2">✨</div>
              <p>No significant cognitive biases detected</p>
              <p class="text-sm mt-1">Your reasoning appears balanced</p>
            </div>
          {:else}
            {#each report.biases as bias}
              <div class="bias bg-gray-800 rounded p-4 border-l-4 border-yellow-500">
                <div class="flex items-center gap-2 mb-2">
                  <span class="text-xl">{getBiasIcon(bias.biasType)}</span>
                  <span class="font-semibold text-yellow-300">{formatBiasName(bias.biasType)}</span>
                  <span class="ml-auto text-xs text-gray-500">
                    {Math.round(bias.confidence * 100)}% confidence
                  </span>
                </div>
                <p class="text-sm text-gray-300 mb-2">{bias.description}</p>
                <div class="trigger text-xs text-gray-500 mb-2">
                  Triggered by: "{bias.triggeredBy}..."
                </div>
                <div class="debiasing bg-gray-900 rounded p-2 text-sm">
                  <span class="text-green-400">Debiasing:</span>
                  <span class="text-gray-300 ml-1">{bias.debiasingSuggestion}</span>
                </div>
              </div>
            {/each}
          {/if}
        </div>

      {:else if activeTab === 'blindspots'}
        <div class="blindspots space-y-3">
          {#if report.blindSpots.length === 0}
            <div class="empty text-center py-8 text-gray-500">
              <div class="text-4xl mb-2">👁️</div>
              <p>No significant blind spots detected</p>
              <p class="text-sm mt-1">Your thinking shows good coverage</p>
            </div>
          {:else}
            {#each report.blindSpots as spot}
              <div class="spot bg-gray-800 rounded p-4">
                <div class="flex items-center gap-2 mb-2">
                  <span class="font-semibold capitalize {getSeverityColor(spot.severity)}">
                    {spot.domain}
                  </span>
                  <span class="ml-auto text-xs px-2 py-0.5 rounded {
                    spot.severity === 'high' ? 'bg-red-900 text-red-300' :
                    spot.severity === 'medium' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-blue-900 text-blue-300'
                  }">
                    {spot.severity}
                  </span>
                </div>
                <p class="text-sm text-gray-300 mb-3">{spot.description}</p>
                <div class="suggestions">
                  <span class="text-xs text-gray-500">Questions to explore:</span>
                  <ul class="mt-1 space-y-1">
                    {#each spot.suggestedExplorations as suggestion}
                      <li class="text-sm text-gray-400 flex items-start gap-2">
                        <span class="text-purple-400">?</span>
                        {suggestion}
                      </li>
                    {/each}
                  </ul>
                </div>
              </div>
            {/each}
          {/if}
        </div>

      {:else if activeTab === 'perspectives'}
        <div class="perspectives space-y-4">
          <!-- Add perspective -->
          <div class="add-perspective flex gap-2">
            <select
              bind:value={selectedPerspective}
              class="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm"
            >
              <option value="">Add a perspective...</option>
              {#each getAvailablePerspectives() as p}
                {#if !report.perspectives.some(rp => rp.perspectiveName.toLowerCase() === p)}
                  <option value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
                {/if}
              {/each}
            </select>
            <button
              on:click={addPerspective}
              disabled={!selectedPerspective}
              class="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 rounded text-sm"
            >
              Add
            </button>
          </div>

          <!-- Perspective cards -->
          {#each report.perspectives as perspective}
            <div class="perspective bg-gray-800 rounded p-4">
              <div class="flex items-center gap-2 mb-3">
                <span class="text-lg">🎭</span>
                <span class="font-semibold text-purple-300 capitalize">{perspective.perspectiveName}</span>
                <span class="text-xs text-gray-500">{perspective.perspectiveType}</span>
              </div>

              <!-- Worldview summary -->
              {#if perspective.worldview.coreValues.length > 0}
                <div class="worldview mb-3 text-xs">
                  <span class="text-gray-500">Values:</span>
                  <div class="flex flex-wrap gap-1 mt-1">
                    {#each perspective.worldview.coreValues as value}
                      <span class="px-2 py-0.5 bg-gray-700 rounded">{value}</span>
                    {/each}
                  </div>
                </div>
              {/if}

              <!-- Response -->
              <div class="response bg-gray-900 rounded p-3 mb-3">
                <p class="text-sm text-gray-300 italic">"{perspective.response}"</p>
              </div>

              <!-- Key differences -->
              {#if perspective.keyDifferences.length > 0}
                <div class="differences">
                  <span class="text-xs text-gray-500">Key differences from your view:</span>
                  <ul class="mt-1 space-y-1">
                    {#each perspective.keyDifferences as diff}
                      <li class="text-sm text-gray-400 flex items-start gap-2">
                        <span class="text-orange-400">↔</span>
                        {diff}
                      </li>
                    {/each}
                  </ul>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .cognitive-augmentation {
    max-height: 600px;
    overflow-y: auto;
  }

  .tab-content {
    min-height: 200px;
  }

  select option {
    background-color: #1f2937;
  }
</style>
