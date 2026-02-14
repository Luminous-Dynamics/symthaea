<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createThought } from '../stores/thoughts';
  import {
    ambientCapture,
    type AmbientState,
    type ThoughtCandidate,
  } from '../services/ambient-capture';

  let state: AmbientState;
  let isExpanded = false;
  let showSettings = false;

  onMount(() => {
    ambientCapture.initialize();
    state = ambientCapture.getState();

    ambientCapture.onStateChange((newState) => {
      state = newState;
    });

    ambientCapture.onCandidate((candidate) => {
      // Play subtle notification sound or visual pulse
      pulseNotification();
    });
  });

  function toggleListening() {
    if (state.isListening) {
      ambientCapture.stopListening();
    } else {
      ambientCapture.startListening();
    }
  }

  async function approveCandidate(index: number) {
    const candidate = ambientCapture.approveCandidate(index);
    if (candidate) {
      const input = ambientCapture.candidateToThoughtInput(candidate);
      await createThought(input as any);
      ambientCapture.dismissCandidate(index);
    }
  }

  function dismissCandidate(index: number) {
    ambientCapture.dismissCandidate(index);
  }

  function dismissAll() {
    ambientCapture.clearCandidates();
  }

  let pulseActive = false;
  function pulseNotification() {
    pulseActive = true;
    setTimeout(() => (pulseActive = false), 1000);
  }

  function getTypeColor(type: string): string {
    const colors: Record<string, string> = {
      Claim: '#3b82f6',
      Question: '#8b5cf6',
      Observation: '#10b981',
      Belief: '#f59e0b',
      Hypothesis: '#ec4899',
      Goal: '#22c55e',
      Plan: '#0ea5e9',
      Memory: '#f97316',
      Intuition: '#a855f7',
      Reflection: '#64748b',
    };
    return colors[type] || '#6b7280';
  }

  function formatTime(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
</script>

<div class="ambient-capture" class:expanded={isExpanded} class:pulse={pulseActive}>
  <!-- Floating Button -->
  <button
    class="capture-btn"
    class:listening={state?.isListening}
    on:click={() => (isExpanded = !isExpanded)}
    title={state?.isListening ? 'Listening for insights...' : 'Ambient Capture'}
  >
    {#if state?.isListening}
      <div class="listening-indicator">
        <span class="wave"></span>
        <span class="wave"></span>
        <span class="wave"></span>
      </div>
    {:else}
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
        <line x1="8" y1="23" x2="16" y2="23" />
      </svg>
    {/if}

    {#if state?.candidates?.length > 0}
      <span class="badge">{state.candidates.length}</span>
    {/if}
  </button>

  <!-- Expanded Panel -->
  {#if isExpanded}
    <div class="panel">
      <header>
        <h3>Ambient Capture</h3>
        <button class="settings-btn" on:click={() => (showSettings = !showSettings)}>⚙️</button>
        <button class="close-btn" on:click={() => (isExpanded = false)}>×</button>
      </header>

      {#if showSettings}
        <div class="settings">
          <label>
            <span>Insight Threshold</span>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.1"
              value={state?.insightThreshold || 0.3}
              on:input={(e) => ambientCapture.setThreshold(parseFloat(e.currentTarget.value))}
            />
            <span class="value">{Math.round((state?.insightThreshold || 0.3) * 100)}%</span>
          </label>
          <p class="hint">Higher = fewer but more significant captures</p>
        </div>
      {/if}

      <div class="controls">
        {#if ambientCapture.voiceSupported}
          <button
            class="listen-btn"
            class:active={state?.isListening}
            on:click={toggleListening}
          >
            {state?.isListening ? '⏹ Stop Listening' : '🎤 Start Listening'}
          </button>
        {:else}
          <p class="unsupported">Voice capture not supported in this browser</p>
        {/if}
      </div>

      {#if state?.isListening && state?.currentTranscript}
        <div class="transcript">
          <span class="label">Hearing:</span>
          <span class="text">{state.currentTranscript}</span>
        </div>
      {/if}

      <div class="candidates">
        {#if state?.candidates?.length > 0}
          <div class="candidates-header">
            <span>Captured Insights ({state.candidates.length})</span>
            <button class="dismiss-all" on:click={dismissAll}>Clear All</button>
          </div>

          {#each state.candidates as candidate, i}
            <div class="candidate">
              <div class="candidate-header">
                <span class="type" style="background: {getTypeColor(candidate.suggestedType)}">
                  {candidate.suggestedType}
                </span>
                <span class="score">{Math.round(candidate.insightScore * 100)}%</span>
                <span class="time">{formatTime(candidate.context.timestamp)}</span>
              </div>

              <p class="content">{candidate.content}</p>

              {#if candidate.tags.length > 0}
                <div class="tags">
                  {#each candidate.tags as tag}
                    <span class="tag">#{tag}</span>
                  {/each}
                </div>
              {/if}

              <div class="actions">
                <button class="approve" on:click={() => approveCandidate(i)}>
                  ✓ Save
                </button>
                <button class="dismiss" on:click={() => dismissCandidate(i)}>
                  ✗ Dismiss
                </button>
              </div>
            </div>
          {/each}
        {:else}
          <div class="empty">
            {#if state?.isListening}
              <p>Listening for insights...</p>
              <p class="hint">Speak naturally. I'll capture meaningful moments.</p>
            {:else}
              <p>No captured insights yet</p>
              <p class="hint">Start listening or select text on any page</p>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .ambient-capture {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 1000;
  }

  .capture-btn {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border: none;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    transition: all 0.3s ease;
    position: relative;
  }

  .capture-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 30px rgba(124, 58, 237, 0.5);
  }

  .capture-btn.listening {
    animation: glow 2s ease-in-out infinite;
  }

  @keyframes glow {
    0%, 100% { box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4); }
    50% { box-shadow: 0 4px 40px rgba(124, 58, 237, 0.8); }
  }

  .ambient-capture.pulse .capture-btn {
    animation: pulse 0.5s ease-out;
  }

  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
  }

  .listening-indicator {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 20px;
  }

  .wave {
    width: 4px;
    height: 100%;
    background: white;
    border-radius: 2px;
    animation: wave 1s ease-in-out infinite;
  }

  .wave:nth-child(2) { animation-delay: 0.2s; }
  .wave:nth-child(3) { animation-delay: 0.4s; }

  @keyframes wave {
    0%, 100% { height: 8px; }
    50% { height: 20px; }
  }

  .badge {
    position: absolute;
    top: -4px;
    right: -4px;
    min-width: 20px;
    height: 20px;
    padding: 0 6px;
    background: #ef4444;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .panel {
    position: absolute;
    bottom: 70px;
    right: 0;
    width: 360px;
    max-height: 500px;
    background: #1e1e2e;
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  header {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #2a2a4e;
  }

  header h3 {
    margin: 0;
    font-size: 0.95rem;
    color: #fff;
    flex: 1;
  }

  .settings-btn,
  .close-btn {
    background: none;
    border: none;
    color: #888;
    font-size: 1rem;
    cursor: pointer;
    padding: 4px;
  }

  .settings-btn:hover,
  .close-btn:hover {
    color: #fff;
  }

  .settings {
    padding: 12px 16px;
    background: #252540;
    border-bottom: 1px solid #2a2a4e;
  }

  .settings label {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.85rem;
    color: #888;
  }

  .settings input[type="range"] {
    flex: 1;
    accent-color: #7c3aed;
  }

  .settings .value {
    min-width: 36px;
    font-family: monospace;
  }

  .settings .hint {
    margin: 8px 0 0;
    font-size: 0.75rem;
    color: #666;
  }

  .controls {
    padding: 12px 16px;
    border-bottom: 1px solid #2a2a4e;
  }

  .listen-btn {
    width: 100%;
    padding: 12px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    color: #e5e5e5;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .listen-btn:hover {
    border-color: #5a5a8e;
  }

  .listen-btn.active {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(59, 130, 246, 0.2));
    border-color: #7c3aed;
  }

  .unsupported {
    color: #f59e0b;
    font-size: 0.85rem;
    text-align: center;
  }

  .transcript {
    padding: 8px 16px;
    background: rgba(124, 58, 237, 0.1);
    border-bottom: 1px solid #2a2a4e;
    font-size: 0.85rem;
  }

  .transcript .label {
    color: #7c3aed;
    margin-right: 8px;
  }

  .transcript .text {
    color: #a5a5c5;
    font-style: italic;
  }

  .candidates {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }

  .candidates-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    font-size: 0.8rem;
    color: #888;
  }

  .dismiss-all {
    background: none;
    border: none;
    color: #666;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .dismiss-all:hover {
    color: #ef4444;
  }

  .candidate {
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
  }

  .candidate-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .type {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    color: white;
  }

  .score {
    font-size: 0.75rem;
    color: #10b981;
    font-family: monospace;
  }

  .time {
    margin-left: auto;
    font-size: 0.7rem;
    color: #666;
  }

  .content {
    margin: 0 0 8px;
    font-size: 0.85rem;
    color: #e5e5e5;
    line-height: 1.4;
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
  }

  .tag {
    padding: 1px 6px;
    background: #1a1a2e;
    border-radius: 3px;
    font-size: 0.7rem;
    color: #888;
  }

  .actions {
    display: flex;
    gap: 8px;
  }

  .actions button {
    flex: 1;
    padding: 6px;
    border-radius: 4px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .approve {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #10b981;
  }

  .approve:hover {
    background: rgba(16, 185, 129, 0.3);
  }

  .dismiss {
    background: none;
    border: 1px solid #3a3a5e;
    color: #888;
  }

  .dismiss:hover {
    border-color: #ef4444;
    color: #ef4444;
  }

  .empty {
    text-align: center;
    padding: 24px;
    color: #888;
  }

  .empty .hint {
    font-size: 0.8rem;
    color: #666;
    margin-top: 8px;
  }
</style>
