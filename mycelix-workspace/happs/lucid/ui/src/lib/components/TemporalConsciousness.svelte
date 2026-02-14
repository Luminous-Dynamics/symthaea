<script lang="ts">
  import type { Thought } from '@mycelix/lucid-client';
  import {
    createTemporalSnapshot,
    groupThoughtsByPeriod,
    excavateWorldview,
    predictAllTrajectories,
    initiatePastSelfDialogue,
    generatePastSelfResponse,
    extractDialogueInsights,
    compareSnapshots,
    type TemporalSnapshot,
    type ArchaeologyReport,
    type BeliefTrajectory,
    type PastSelfDialogue,
    type WorldviewLayer,
  } from '../services/temporal-consciousness';

  export let thoughts: Thought[] = [];

  let activeTab: 'timeline' | 'archaeology' | 'trajectories' | 'dialogue' = 'timeline';
  let periodType: 'week' | 'month' | 'quarter' | 'year' = 'month';

  // Timeline state
  let snapshots: Map<string, TemporalSnapshot> = new Map();
  let selectedPeriod: string | null = null;

  // Archaeology state
  let archaeologyReport: ArchaeologyReport | null = null;

  // Trajectory state
  let trajectories: BeliefTrajectory[] = [];

  // Dialogue state
  let pastDialogue: PastSelfDialogue | null = null;
  let dialogueInput = '';
  let selectedPastPeriod: string | null = null;

  $: {
    // Rebuild snapshots when thoughts or period type changes
    const periods = groupThoughtsByPeriod(thoughts, periodType);
    snapshots = new Map();
    for (const [period, periodThoughts] of periods) {
      snapshots.set(period, createTemporalSnapshot(periodThoughts));
    }
  }

  function runArchaeology() {
    const periods = groupThoughtsByPeriod(thoughts, periodType);
    archaeologyReport = excavateWorldview(periods);
  }

  function computeTrajectories() {
    if (thoughts.length < 5) return;
    const currentSnapshot = createTemporalSnapshot(thoughts);
    trajectories = predictAllTrajectories(currentSnapshot.worldviewProfile);
  }

  function startDialogue(period: string) {
    const pastSnapshot = snapshots.get(period);
    const currentSnapshot = createTemporalSnapshot(thoughts);

    if (pastSnapshot && currentSnapshot) {
      pastDialogue = initiatePastSelfDialogue(currentSnapshot, pastSnapshot);
      selectedPastPeriod = period;
    }
  }

  function sendMessage() {
    if (!pastDialogue || !dialogueInput.trim()) return;

    generatePastSelfResponse(pastDialogue, dialogueInput);
    pastDialogue = pastDialogue; // Trigger reactivity
    dialogueInput = '';
  }

  function endDialogue() {
    if (pastDialogue) {
      extractDialogueInsights(pastDialogue);
      pastDialogue = pastDialogue; // Trigger reactivity
    }
  }

  function formatDate(timestamp: number): string {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  }

  function getTrajectoryColor(direction: string): string {
    switch (direction) {
      case 'strengthen': return 'text-green-400';
      case 'weaken': return 'text-red-400';
      case 'transform': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  }

  function getTrajectoryArrow(direction: string): string {
    switch (direction) {
      case 'strengthen': return '↗';
      case 'weaken': return '↘';
      case 'transform': return '↯';
      default: return '→';
    }
  }
</script>

<div class="temporal-consciousness bg-gray-900 rounded-lg p-4 border border-gray-700">
  <div class="header flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-cyan-300">Temporal Consciousness</h3>
    <select
      bind:value={periodType}
      class="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm"
    >
      <option value="week">Weekly</option>
      <option value="month">Monthly</option>
      <option value="quarter">Quarterly</option>
      <option value="year">Yearly</option>
    </select>
  </div>

  <!-- Tabs -->
  <div class="tabs flex gap-1 mb-4 border-b border-gray-700 pb-2">
    {#each ['timeline', 'archaeology', 'trajectories', 'dialogue'] as tab (tab)}
      <button
        on:click={() => activeTab = tab as 'timeline' | 'archaeology' | 'trajectories' | 'dialogue'}
        class="px-3 py-1 rounded-t text-sm transition-colors {activeTab === tab
          ? 'bg-gray-700 text-white'
          : 'text-gray-400 hover:text-gray-200'}"
      >
        {tab.charAt(0).toUpperCase() + tab.slice(1)}
      </button>
    {/each}
  </div>

  <!-- Tab Content -->
  <div class="tab-content min-h-[300px]">
    {#if activeTab === 'timeline'}
      <div class="timeline">
        {#if snapshots.size === 0}
          <div class="empty text-center py-8 text-gray-500">
            <div class="text-4xl mb-2">📅</div>
            <p>No temporal data available</p>
            <p class="text-sm mt-1">Add thoughts to see your timeline</p>
          </div>
        {:else}
          <div class="timeline-view space-y-3">
            {#each [...snapshots.entries()].sort((a, b) => b[1].timestamp - a[1].timestamp) as [period, snapshot]}
              <div
                role="button"
                tabindex="0"
                on:click={() => selectedPeriod = selectedPeriod === period ? null : period}
                on:keydown={(e) => e.key === 'Enter' && (selectedPeriod = selectedPeriod === period ? null : period)}
                class="w-full text-left period-card bg-gray-800 rounded p-3 hover:bg-gray-750 transition-colors border-l-4 cursor-pointer {
                  selectedPeriod === period ? 'border-cyan-500' : 'border-gray-600'
                }"
              >
                <div class="flex items-center justify-between mb-2">
                  <span class="font-semibold text-cyan-300">{period}</span>
                  <span class="text-xs text-gray-500">{snapshot.thoughts.length} thoughts</span>
                </div>

                <div class="themes flex flex-wrap gap-1 mb-2">
                  {#each snapshot.dominantThemes.slice(0, 4) as theme}
                    <span class="px-2 py-0.5 bg-gray-700 rounded text-xs">{theme}</span>
                  {/each}
                </div>

                {#if selectedPeriod === period}
                  <div class="details mt-3 pt-3 border-t border-gray-700">
                    <!-- Epistemic Balance -->
                    <div class="balance grid grid-cols-3 gap-2 text-xs mb-3">
                      <div class="text-center">
                        <div class="text-cyan-400">{Math.round(snapshot.epistemicBalance.empirical * 100)}%</div>
                        <div class="text-gray-500">Empirical</div>
                      </div>
                      <div class="text-center">
                        <div class="text-purple-400">{Math.round(snapshot.epistemicBalance.normative * 100)}%</div>
                        <div class="text-gray-500">Normative</div>
                      </div>
                      <div class="text-center">
                        <div class="text-green-400">{Math.round(snapshot.epistemicBalance.material * 100)}%</div>
                        <div class="text-gray-500">Material</div>
                      </div>
                    </div>

                    <!-- Core Beliefs -->
                    <div class="beliefs">
                      <div class="text-xs text-gray-500 mb-1">Core Beliefs:</div>
                      {#each snapshot.worldviewProfile.coreBeliefs.slice(0, 3) as belief}
                        <div class="belief text-sm text-gray-300 mb-1">
                          <span class="text-cyan-400">{belief.category}:</span>
                          {belief.content.slice(0, 60)}...
                        </div>
                      {/each}
                    </div>

                    <!-- Dialogue button -->
                    <button
                      on:click|stopPropagation={() => { startDialogue(period); activeTab = 'dialogue'; }}
                      class="mt-3 w-full px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 rounded text-sm"
                    >
                      Talk to Past Self
                    </button>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

    {:else if activeTab === 'archaeology'}
      <div class="archaeology">
        <button
          on:click={runArchaeology}
          class="w-full mb-4 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded"
        >
          Excavate Worldview
        </button>

        {#if archaeologyReport}
          <div class="report space-y-4">
            <!-- Layers -->
            <div class="layers">
              <h4 class="text-sm font-semibold text-gray-400 mb-2">Worldview Layers</h4>
              <div class="layer-stack relative">
                {#each archaeologyReport.layers as layer, i}
                  <div
                    class="layer bg-gray-800 rounded p-3 mb-2 border-l-4"
                    style="border-color: hsl({i * 40}, 60%, 50%)"
                  >
                    <div class="flex justify-between items-center mb-1">
                      <span class="font-semibold">{layer.label}</span>
                      <span class="text-xs text-gray-500">{layer.overallTone}</span>
                    </div>

                    {#if layer.significantShifts.length > 0}
                      <div class="shifts text-xs text-gray-400 mt-2">
                        <span class="text-yellow-400">Shifts:</span>
                        {#each layer.significantShifts.slice(0, 2) as shift}
                          <div class="shift ml-2">
                            {shift.domain}: {shift.from} → {shift.to}
                          </div>
                        {/each}
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            </div>

            <!-- Persistent Beliefs -->
            {#if archaeologyReport.persistentBeliefs.length > 0}
              <div class="persistent">
                <h4 class="text-sm font-semibold text-green-400 mb-2">Persistent Beliefs</h4>
                {#each archaeologyReport.persistentBeliefs as belief}
                  <div class="bg-gray-800 rounded p-2 mb-1 text-sm">
                    <span class="text-green-400">{belief.category}:</span>
                    {belief.content.slice(0, 80)}
                  </div>
                {/each}
              </div>
            {/if}

            <!-- Abandoned Beliefs -->
            {#if archaeologyReport.abandonedBeliefs.length > 0}
              <div class="abandoned">
                <h4 class="text-sm font-semibold text-red-400 mb-2">Abandoned Beliefs</h4>
                {#each archaeologyReport.abandonedBeliefs as belief}
                  <div class="bg-gray-800 rounded p-2 mb-1 text-sm opacity-70">
                    <span class="text-red-400">{belief.category}:</span>
                    {belief.content.slice(0, 80)}
                  </div>
                {/each}
              </div>
            {/if}

            <!-- Emergent Patterns -->
            {#if archaeologyReport.emergentPatterns.length > 0}
              <div class="patterns">
                <h4 class="text-sm font-semibold text-purple-400 mb-2">Emergent Patterns</h4>
                <ul class="space-y-1">
                  {#each archaeologyReport.emergentPatterns as pattern}
                    <li class="text-sm text-gray-300 flex items-start gap-2">
                      <span class="text-purple-400">◆</span>
                      {pattern}
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}
          </div>
        {:else}
          <div class="empty text-center py-8 text-gray-500">
            <div class="text-4xl mb-2">🏛️</div>
            <p>Click "Excavate Worldview" to analyze your belief evolution</p>
          </div>
        {/if}
      </div>

    {:else if activeTab === 'trajectories'}
      <div class="trajectories">
        <button
          on:click={computeTrajectories}
          class="w-full mb-4 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded"
        >
          Predict Trajectories
        </button>

        {#if trajectories.length > 0}
          <div class="trajectory-list space-y-3">
            {#each trajectories.slice(0, 6) as trajectory}
              <div class="trajectory bg-gray-800 rounded p-3">
                <div class="flex items-center justify-between mb-2">
                  <span class="font-semibold text-gray-300">{trajectory.belief.category}</span>
                  <span class="text-xs text-gray-500">
                    Confidence: {Math.round(trajectory.confidenceInPrediction * 100)}%
                  </span>
                </div>

                <!-- Current state -->
                <div class="current flex items-center gap-2 mb-2 text-sm">
                  <span class="text-gray-500">Now:</span>
                  <div class="flex-1 h-2 bg-gray-700 rounded overflow-hidden">
                    <div
                      class="h-full bg-cyan-500"
                      style="width: {trajectory.currentPosition * 100}%"
                    ></div>
                  </div>
                  <span class="text-cyan-400">{Math.round(trajectory.currentPosition * 100)}%</span>
                </div>

                <!-- Velocity -->
                <div class="velocity text-xs text-gray-400 mb-2">
                  Velocity: <span class="{trajectory.velocity > 0 ? 'text-green-400' : trajectory.velocity < 0 ? 'text-red-400' : 'text-gray-400'}">
                    {trajectory.velocity > 0 ? '+' : ''}{(trajectory.velocity * 100).toFixed(1)}%/period
                  </span>
                </div>

                <!-- Predictions -->
                <div class="predictions">
                  <div class="text-xs text-gray-500 mb-1">Predicted trajectory:</div>
                  <div class="flex items-center gap-1">
                    {#each trajectory.predictedFuture as prediction, i}
                      <div class="prediction flex-1 text-center">
                        <div class="text-lg {getTrajectoryColor(prediction.likelyDirection)}">
                          {getTrajectoryArrow(prediction.likelyDirection)}
                        </div>
                        <div class="text-xs text-gray-500">
                          {Math.round(prediction.predictedConfidence * 100)}%
                        </div>
                      </div>
                      {#if i < trajectory.predictedFuture.length - 1}
                        <div class="text-gray-600">→</div>
                      {/if}
                    {/each}
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <div class="empty text-center py-8 text-gray-500">
            <div class="text-4xl mb-2">🔮</div>
            <p>Click "Predict Trajectories" to see where your beliefs are heading</p>
            <p class="text-sm mt-1">Requires at least 5 thoughts</p>
          </div>
        {/if}
      </div>

    {:else if activeTab === 'dialogue'}
      <div class="dialogue">
        {#if !pastDialogue}
          <div class="select-period">
            <p class="text-gray-400 mb-3">Select a time period to dialogue with your past self:</p>
            <div class="grid grid-cols-2 gap-2">
              {#each [...snapshots.entries()].slice(0, 6) as [period, snapshot]}
                <button
                  on:click={() => startDialogue(period)}
                  class="p-3 bg-gray-800 hover:bg-gray-750 rounded text-left"
                >
                  <div class="font-semibold text-cyan-300">{period}</div>
                  <div class="text-xs text-gray-500">{snapshot.thoughts.length} thoughts</div>
                </button>
              {/each}
            </div>
          </div>
        {:else}
          <div class="dialogue-active">
            <div class="dialogue-header flex items-center justify-between mb-4">
              <div>
                <span class="text-cyan-300">Conversing with</span>
                <span class="font-semibold ml-1">{selectedPastPeriod} self</span>
              </div>
              <button
                on:click={() => { pastDialogue = null; selectedPastPeriod = null; }}
                class="text-xs text-gray-500 hover:text-gray-300"
              >
                End Dialogue
              </button>
            </div>

            <!-- Messages -->
            <div class="messages space-y-3 max-h-[250px] overflow-y-auto mb-4">
              {#each pastDialogue.messages as message}
                <div class="message {message.speaker === 'present' ? 'text-right' : 'text-left'}">
                  <div class="inline-block max-w-[80%] p-3 rounded-lg {
                    message.speaker === 'present'
                      ? 'bg-cyan-900/50 text-cyan-100'
                      : 'bg-gray-800 text-gray-300'
                  }">
                    <div class="text-xs text-gray-500 mb-1">
                      {message.speaker === 'present' ? 'You (now)' : `You (${selectedPastPeriod})`}
                      {#if message.emotionalTone}
                        <span class="ml-1 italic">feeling {message.emotionalTone}</span>
                      {/if}
                    </div>
                    <p class="text-sm">{message.content}</p>
                  </div>
                </div>
              {/each}
            </div>

            <!-- Input -->
            <div class="input-area flex gap-2">
              <input
                type="text"
                bind:value={dialogueInput}
                on:keydown={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Ask your past self something..."
                class="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
              <button
                on:click={sendMessage}
                disabled={!dialogueInput.trim()}
                class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 rounded text-sm"
              >
                Send
              </button>
            </div>

            <!-- Insights -->
            {#if pastDialogue.insightsGained.length > 0}
              <div class="insights mt-4 p-3 bg-gray-800 rounded">
                <h4 class="text-sm font-semibold text-purple-300 mb-2">Insights Gained</h4>
                <ul class="space-y-1">
                  {#each pastDialogue.insightsGained as insight}
                    <li class="text-sm text-gray-300 flex items-start gap-2">
                      <span class="text-purple-400">★</span>
                      {insight}
                    </li>
                  {/each}
                </ul>
              </div>
            {:else}
              <button
                on:click={endDialogue}
                class="mt-4 w-full px-3 py-2 bg-purple-600 hover:bg-purple-500 rounded text-sm"
              >
                Extract Insights
              </button>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .temporal-consciousness {
    max-height: 700px;
    overflow-y: auto;
  }

  .bg-gray-750 {
    background-color: #374151;
  }

  .messages {
    scrollbar-width: thin;
    scrollbar-color: #4b5563 #1f2937;
  }
</style>
