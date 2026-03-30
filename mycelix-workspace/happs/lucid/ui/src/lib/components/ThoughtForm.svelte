<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { createThought, coherenceStatus } from '../stores/thoughts';
  import type { ThoughtCoherenceStatus } from '../stores/thoughts';
  import { EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel, ThoughtType } from '@mycelix/lucid-client';
  import { createEventDispatcher, onMount } from 'svelte';
  import { autoClassify, isOllamaAvailable, isSymthaeaClassificationAvailable } from '../services/ai-classifier';

  const dispatch = createEventDispatcher();

  let content = '';
  let thoughtType: ThoughtType = ThoughtType.Claim;
  let confidence = 0.7;
  let tagsInput = '';
  let domain = '';

  // E/N/M/H values
  let empirical: EmpiricalLevel = EmpiricalLevel.E2;
  let normative: NormativeLevel = NormativeLevel.N0;
  let materiality: MaterialityLevel = MaterialityLevel.M2;
  let harmonic: HarmonicLevel = HarmonicLevel.H2;

  let isSubmitting = false;
  let expanded = false;

  // AI classification
  let isClassifying = false;
  let hasOllama = false;
  let hasSymthaea = false;
  let useLLM = false;
  let aiReasoning = '';
  let classificationPhi: number | undefined = undefined;
  let classificationCoherence: number | undefined = undefined;

  // Coherence toast
  let coherenceToast: { message: string; type: 'success' | 'warning' | 'info'; phi?: number } | null = null;
  let toastTimeout: ReturnType<typeof setTimeout> | null = null;

  onMount(async () => {
    hasOllama = await isOllamaAvailable();
    hasSymthaea = isSymthaeaClassificationAvailable();
  });

  async function handleAutoClassify() {
    if (!content.trim()) return;

    isClassifying = true;
    aiReasoning = '';

    try {
      const suggestion = await autoClassify(content, useLLM && hasOllama);

      // Apply suggestions
      thoughtType = suggestion.thoughtType as ThoughtType;
      empirical = suggestion.epistemic.empirical;
      normative = suggestion.epistemic.normative;
      materiality = suggestion.epistemic.materiality;
      harmonic = suggestion.epistemic.harmonic;
      confidence = suggestion.confidence;

      if (suggestion.tags.length > 0) {
        const existingTags = tagsInput.split(',').map(t => t.trim()).filter(t => t);
        const newTags = [...new Set([...existingTags, ...suggestion.tags])];
        tagsInput = newTags.join(', ');
      }

      aiReasoning = suggestion.reasoning || '';
      classificationPhi = suggestion.phi;
      classificationCoherence = suggestion.coherence;
      expanded = true;
    } catch (err) {
      console.error('Classification failed:', err);
    } finally {
      isClassifying = false;
    }
  }

  const thoughtTypes = Object.values(ThoughtType);
  const empiricalLevels = Object.values(EmpiricalLevel);
  const normativeLevels = Object.values(NormativeLevel);
  const materialityLevels = Object.values(MaterialityLevel);
  const harmonicLevels = Object.values(HarmonicLevel);

  async function handleSubmit() {
    if (!content.trim()) return;

    isSubmitting = true;

    const tags = tagsInput
      .split(',')
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    const thought = await createThought({
      content: content.trim(),
      thought_type: thoughtType,
      confidence,
      tags,
      domain: domain.trim() || undefined,
      epistemic: {
        empirical,
        normative,
        materiality,
        harmonic,
      },
    });

    if (thought) {
      const createdThoughtId = thought.id;
      content = '';
      tagsInput = '';
      domain = '';
      expanded = false;
      classificationPhi = undefined;
      classificationCoherence = undefined;
      aiReasoning = '';
      dispatch('created', thought);

      // Watch coherenceStatus for the new thought's results (async from checkCoherenceForNewThought)
      const unsubscribe = coherenceStatus.subscribe(($status) => {
        const result: ThoughtCoherenceStatus | undefined = $status.get(createdThoughtId);
        if (result) {
          unsubscribe();
          showCoherenceToast(result);
        }
      });
      // Auto-unsubscribe after 10s if no result arrives
      setTimeout(() => { try { unsubscribe(); } catch (_) {} }, 10000);
    }

    isSubmitting = false;
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  }

  function showCoherenceToast(result: ThoughtCoherenceStatus) {
    if (toastTimeout) clearTimeout(toastTimeout);

    if (result.contradictions.length > 0) {
      const first = result.contradictions[0];
      coherenceToast = {
        message: `Potential contradiction with: "${first.conflictingContentPreview}"`,
        type: 'warning',
        phi: result.phiScore,
      };
    } else if (result.isCoherent) {
      coherenceToast = {
        message: `Coherent with your worldview`,
        type: 'success',
        phi: result.phiScore,
      };
    } else {
      coherenceToast = {
        message: `Coherence check complete`,
        type: 'info',
        phi: result.phiScore,
      };
    }

    toastTimeout = setTimeout(() => { coherenceToast = null; }, 6000);
  }

  function phiColor(phi: number): string {
    if (phi >= 0.6) return '#10b981';
    if (phi >= 0.3) return '#f59e0b';
    return '#ef4444';
  }
</script>

<form class="thought-form" on:submit|preventDefault={handleSubmit} aria-label="Create new thought">
  <div class="main-input">
    <label class="sr-only" for="thought-content">Thought content</label>
    <textarea
      id="thought-content"
      bind:value={content}
      placeholder="What's on your mind? Enter a thought, question, or observation..."
      rows="3"
      on:keydown={handleKeydown}
      disabled={isSubmitting}
      aria-describedby="submit-hint"
    ></textarea>
    <button
      type="submit"
      disabled={!content.trim() || isSubmitting}
      aria-label={isSubmitting ? 'Submitting thought' : 'Add thought'}
    >
      {isSubmitting ? '...' : '+'}
    </button>
    <span id="submit-hint" class="sr-only">Press Ctrl+Enter to submit</span>
  </div>

  <div class="form-actions">
    <button type="button" class="expand-btn" on:click={() => (expanded = !expanded)}>
      {expanded ? '▼ Less options' : '▶ More options'}
    </button>

    <button
      type="button"
      class="ai-btn"
      on:click={handleAutoClassify}
      disabled={!content.trim() || isClassifying}
      title={hasSymthaea ? 'Classify with Symthaea HDC engine' : hasOllama ? 'Classify with LLM' : 'Classify with keyword patterns'}
    >
      {#if isClassifying}
        <span class="spinner"></span>
      {:else if hasSymthaea}
        <span class="backend-icon symthaea"></span>
      {:else}
        ✨
      {/if}
      Auto-classify
    </button>

    {#if hasSymthaea}
      <span class="backend-label symthaea">Symthaea</span>
    {:else if hasOllama}
      <label class="llm-toggle">
        <input type="checkbox" bind:checked={useLLM} />
        <span>Use LLM</span>
      </label>
    {/if}
  </div>

  {#if aiReasoning}
    <div class="ai-reasoning">
      <span class="label">AI:</span> {aiReasoning}
      {#if classificationPhi !== undefined}
        <span class="phi-badge" style="background: {phiColor(classificationPhi)}" title="Integration score: how well this thought connects with your worldview ({classificationPhi.toFixed(2)})">
          &Phi; {classificationPhi.toFixed(2)}
        </span>
      {/if}
    </div>
  {/if}

  {#if coherenceToast}
    <div class="coherence-toast toast-{coherenceToast.type}" role="status" aria-live="polite">
      <span class="toast-icon">
        {#if coherenceToast.type === 'success'}&#10003;{:else if coherenceToast.type === 'warning'}&#9888;{:else}&#8505;{/if}
      </span>
      <span class="toast-message">{coherenceToast.message}</span>
      {#if coherenceToast.phi !== undefined}
        <span class="toast-phi" style="color: {phiColor(coherenceToast.phi)}">&Phi; {coherenceToast.phi.toFixed(2)}</span>
      {/if}
      <button class="toast-dismiss" on:click={() => coherenceToast = null} aria-label="Dismiss">&times;</button>
    </div>
  {/if}

  {#if expanded}
    <div class="options">
      <div class="row">
        <label>
          <span>Type</span>
          <select bind:value={thoughtType}>
            {#each thoughtTypes as t}
              <option value={t}>{t}</option>
            {/each}
          </select>
        </label>

        <label>
          <span id="confidence-label">Confidence</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            bind:value={confidence}
            aria-labelledby="confidence-label"
            aria-valuenow={Math.round(confidence * 100)}
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuetext="{Math.round(confidence * 100)}%"
          />
          <span class="value" aria-hidden="true">{Math.round(confidence * 100)}%</span>
        </label>
      </div>

      <div class="row">
        <label>
          <span>Tags (comma-separated)</span>
          <input type="text" bind:value={tagsInput} placeholder="philosophy, ethics, personal" />
        </label>

        <label>
          <span>Domain</span>
          <input type="text" bind:value={domain} placeholder="optional domain" />
        </label>
      </div>

      <div class="epistemic-section">
        <h4>Epistemic Classification</h4>
        <div class="epistemic-grid">
          <label>
            <span title="E0=Subjective, E1=Anecdotal, E2=Observational, E3=Peer-reviewed, E4=Reproducible">Empirical (E)</span>
            <select bind:value={empirical}>
              {#each empiricalLevels as level}
                <option value={level}>{level}</option>
              {/each}
            </select>
          </label>

          <label>
            <span title="N0=Personal, N1=Community, N2=Universal, N3=Axiomatic">Normative (N)</span>
            <select bind:value={normative}>
              {#each normativeLevels as level}
                <option value={level}>{level}</option>
              {/each}
            </select>
          </label>

          <label>
            <span title="M0=Ephemeral, M1=Situational, M2=Structural, M3=Foundational">Materiality (M)</span>
            <select bind:value={materiality}>
              {#each materialityLevels as level}
                <option value={level}>{level}</option>
              {/each}
            </select>
          </label>

          <label>
            <span title="H0=Dissonant, H1=Uncertain, H2=Partial, H3=Resonant, H4=Harmonic">Harmonic (H)</span>
            <select bind:value={harmonic}>
              {#each harmonicLevels as level}
                <option value={level}>{level}</option>
              {/each}
            </select>
          </label>
        </div>
      </div>
    </div>
  {/if}
</form>

<style>
  .thought-form {
    background: #1e1e2e;
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 16px;
  }

  .main-input {
    display: flex;
    gap: 12px;
  }

  textarea {
    flex: 1;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    padding: 12px;
    color: #e5e5e5;
    font-size: 0.95rem;
    resize: vertical;
    font-family: inherit;
  }

  textarea:focus {
    outline: none;
    border-color: #7c3aed;
  }

  textarea::placeholder {
    color: #666;
  }

  button[type='submit'] {
    width: 48px;
    height: 48px;
    background: #7c3aed;
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    transition: background 0.2s;
  }

  button[type='submit']:hover:not(:disabled) {
    background: #6d28d9;
  }

  button[type='submit']:disabled {
    background: #4a4a6e;
    cursor: not-allowed;
  }

  .form-actions {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 8px;
  }

  .expand-btn {
    background: none;
    border: none;
    color: #888;
    font-size: 0.8rem;
    cursor: pointer;
    padding: 8px 0;
    text-align: left;
  }

  .expand-btn:hover {
    color: #aaa;
  }

  .ai-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 0.8rem;
    cursor: pointer;
    transition: opacity 0.2s;
  }

  .ai-btn:hover:not(:disabled) {
    opacity: 0.9;
  }

  .ai-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .ai-btn .spinner {
    width: 12px;
    height: 12px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .backend-icon.symthaea {
    display: inline-block;
    width: 12px;
    height: 12px;
    background: radial-gradient(circle, #10b981, #059669);
    border-radius: 50%;
    box-shadow: 0 0 4px rgba(16, 185, 129, 0.5);
  }

  .backend-label.symthaea {
    font-size: 0.7rem;
    color: #10b981;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  .llm-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #888;
    cursor: pointer;
  }

  .llm-toggle input {
    accent-color: #7c3aed;
  }

  .ai-reasoning {
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(124, 58, 237, 0.1);
    border-left: 3px solid #7c3aed;
    border-radius: 0 6px 6px 0;
    font-size: 0.8rem;
    color: #a5a5c5;
  }

  .ai-reasoning .label {
    color: #7c3aed;
    font-weight: 600;
  }

  .options {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding-top: 8px;
    border-top: 1px solid #2a2a4e;
    margin-top: 8px;
  }

  .row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  label span {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
  }

  label .value {
    font-family: monospace;
    min-width: 40px;
    text-align: right;
  }

  select,
  input[type='text'] {
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 6px;
    padding: 8px 10px;
    color: #e5e5e5;
    font-size: 0.9rem;
  }

  select:focus,
  input:focus {
    outline: none;
    border-color: #7c3aed;
  }

  input[type='range'] {
    flex: 1;
  }

  .epistemic-section {
    padding-top: 8px;
  }

  .epistemic-section h4 {
    font-size: 0.8rem;
    color: #888;
    margin: 0 0 12px;
    text-transform: uppercase;
  }

  .epistemic-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .phi-badge {
    display: inline-block;
    margin-left: 8px;
    padding: 2px 8px;
    border-radius: 10px;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: monospace;
    vertical-align: middle;
  }

  .coherence-toast {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.85rem;
    animation: slideIn 0.3s ease-out;
  }

  .toast-success {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
  }

  .toast-warning {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
  }

  .toast-info {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #93c5fd;
  }

  .toast-icon {
    font-size: 1rem;
    flex-shrink: 0;
  }

  .toast-message {
    flex: 1;
  }

  .toast-phi {
    font-family: monospace;
    font-size: 0.8rem;
    font-weight: 600;
  }

  .toast-dismiss {
    background: none;
    border: none;
    color: inherit;
    opacity: 0.6;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }

  .toast-dismiss:hover {
    opacity: 1;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 600px) {
    .row,
    .epistemic-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
