<script lang="ts">
  import type { EpistemicClassification } from '@mycelix/lucid-client';

  /**
   * EpistemicBadge - Enhanced E/N/M/H classification display
   *
   * Features:
   * - Phi (consciousness integration) display on hover
   * - Color-coded H level with semantic meaning
   * - Domain expertise indicator
   * - Confidence interval visualization
   * - Interactive tooltip with full explanations
   */

  // ============================================================================
  // PROPS
  // ============================================================================

  export let epistemic: EpistemicClassification;
  export let compact: boolean = false;

  /** Optional phi value from Symthaea (0.0-1.0) */
  export let phi: number | undefined = undefined;

  /** Optional coherence value (0.0-1.0) */
  export let coherence: number | undefined = undefined;

  /** Optional confidence value (0.0-1.0) */
  export let confidence: number | undefined = undefined;

  /** Optional domain expertise indicator */
  export let domain: string | undefined = undefined;

  /** Show the confidence interval visualization */
  export let showConfidence: boolean = false;

  /** Enable interactive tooltip on hover */
  export let interactive: boolean = true;

  // ============================================================================
  // DERIVED STATE
  // ============================================================================

  // Extract numeric values from enum strings
  $: e = parseInt(epistemic.empirical.replace('E', ''));
  $: n = parseInt(epistemic.normative.replace('N', ''));
  $: m = parseInt(epistemic.materiality.replace('M', ''));
  $: h = parseInt(epistemic.harmonic.replace('H', ''));

  // Calculate overall strength (0-1)
  $: strength = (0.4 * (e / 4) + 0.25 * (n / 3) + 0.2 * (m / 3) + 0.15 * (h / 4));

  // Color based on strength
  $: strengthColor = strength > 0.7 ? 'high' : strength > 0.4 ? 'medium' : 'low';

  // H-level specific colors (unique to harmonic dimension)
  $: hColor = getHarmonicColor(h);

  // Confidence interval (for visualization)
  $: confidenceWidth = confidence !== undefined ? confidence * 100 : 70;
  $: confidenceLow = Math.max(0, (confidence ?? 0.7) - 0.15) * 100;
  $: confidenceHigh = Math.min(100, (confidence ?? 0.7) + 0.15) * 100;

  // Tooltip state
  let showTooltip = false;

  // ============================================================================
  // HELPER DATA
  // ============================================================================

  // Detailed tooltips for each dimension
  const dimensionInfo = {
    E: {
      name: 'Empirical',
      levels: [
        { level: 0, name: 'Unsubstantiated', desc: 'No evidence provided' },
        { level: 1, name: 'Anecdotal', desc: 'Personal experience' },
        { level: 2, name: 'Observational', desc: 'Multiple observations' },
        { level: 3, name: 'Empirical', desc: 'Systematic evidence' },
        { level: 4, name: 'Verified', desc: 'Independently verified' },
      ],
    },
    N: {
      name: 'Normative',
      levels: [
        { level: 0, name: 'Personal', desc: 'Individual preference' },
        { level: 1, name: 'Contested', desc: 'Disputed norms' },
        { level: 2, name: 'Emerging', desc: 'Growing acceptance' },
        { level: 3, name: 'Axiomatic', desc: 'Foundational principle' },
      ],
    },
    M: {
      name: 'Materiality',
      levels: [
        { level: 0, name: 'Abstract', desc: 'Theoretical only' },
        { level: 1, name: 'Potential', desc: 'Possible implications' },
        { level: 2, name: 'Practical', desc: 'Real applications' },
        { level: 3, name: 'Foundational', desc: 'Core infrastructure' },
      ],
    },
    H: {
      name: 'Harmonic',
      levels: [
        { level: 0, name: 'Discordant', desc: 'Conflicts with worldview' },
        { level: 1, name: 'Neutral', desc: 'No particular alignment' },
        { level: 2, name: 'Resonant', desc: 'Moderate coherence' },
        { level: 3, name: 'Harmonic', desc: 'High coherence' },
        { level: 4, name: 'Transcendent', desc: 'Perfect alignment' },
      ],
    },
  };

  // ============================================================================
  // HELPERS
  // ============================================================================

  function getHarmonicColor(hLevel: number): string {
    switch (hLevel) {
      case 0: return '#ef4444'; // Red - discordant
      case 1: return '#94a3b8'; // Gray - neutral
      case 2: return '#3b82f6'; // Blue - resonant
      case 3: return '#8b5cf6'; // Purple - harmonic
      case 4: return '#22c55e'; // Green - transcendent
      default: return '#94a3b8';
    }
  }

  function getPhiDescription(phiValue: number): string {
    if (phiValue >= 0.8) return 'Highly Integrated';
    if (phiValue >= 0.6) return 'Well Integrated';
    if (phiValue >= 0.4) return 'Moderately Integrated';
    if (phiValue >= 0.2) return 'Weakly Integrated';
    return 'Fragmented';
  }

  function getDomainColor(domain: string): string {
    const hash = domain.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    const hue = hash % 360;
    return `hsl(${hue}, 50%, 60%)`;
  }
</script>

{#if compact}
  <!-- Compact mode: single line badge -->
  <span
    class="badge compact {strengthColor}"
    class:interactive
    title="E{e}N{n}M{m}H{h}"
    role="status"
    on:mouseenter={() => interactive && (showTooltip = true)}
    on:mouseleave={() => showTooltip = false}
  >
    <span class="code">E{e}N{n}M{m}H{h}</span>
    {#if phi !== undefined}
      <span class="phi-indicator" style="background: {getHarmonicColor(h)}">
        φ{(phi * 100).toFixed(0)}
      </span>
    {/if}
    {#if domain}
      <span class="domain-dot" style="background: {getDomainColor(domain)}" title={domain}></span>
    {/if}
  </span>

  <!-- Tooltip popup for compact mode -->
  {#if showTooltip && interactive}
    <div class="tooltip">
      <div class="tooltip-header">
        <span class="epistemic-code">E{e}N{n}M{m}H{h}</span>
        {#if phi !== undefined}
          <span class="phi-badge">φ = {phi.toFixed(2)}</span>
        {/if}
      </div>
      <div class="tooltip-grid">
        <div class="tooltip-dim">
          <span class="dim-label" style="color: {strength > 0.5 ? '#34d399' : '#f87171'}">E{e}</span>
          <span class="dim-name">{dimensionInfo.E.levels[e].name}</span>
        </div>
        <div class="tooltip-dim">
          <span class="dim-label">N{n}</span>
          <span class="dim-name">{dimensionInfo.N.levels[n].name}</span>
        </div>
        <div class="tooltip-dim">
          <span class="dim-label">M{m}</span>
          <span class="dim-name">{dimensionInfo.M.levels[m].name}</span>
        </div>
        <div class="tooltip-dim">
          <span class="dim-label" style="color: {hColor}">H{h}</span>
          <span class="dim-name">{dimensionInfo.H.levels[h].name}</span>
        </div>
      </div>
      {#if phi !== undefined}
        <div class="tooltip-phi">
          <span class="phi-label">Integration: {getPhiDescription(phi)}</span>
        </div>
      {/if}
      {#if domain}
        <div class="tooltip-domain">
          <span class="domain-label" style="color: {getDomainColor(domain)}">Domain: {domain}</span>
        </div>
      {/if}
    </div>
  {/if}
{:else}
  <!-- Full mode: detailed display -->
  <div
    class="epistemic-badge {strengthColor}"
    class:interactive
    role="status"
    on:mouseenter={() => interactive && (showTooltip = true)}
    on:mouseleave={() => showTooltip = false}
  >
    <!-- Header with phi and domain -->
    {#if phi !== undefined || domain}
      <div class="badge-header">
        {#if phi !== undefined}
          <div class="phi-display">
            <span class="phi-symbol">φ</span>
            <span class="phi-value">{phi.toFixed(2)}</span>
            <span class="phi-desc">{getPhiDescription(phi)}</span>
          </div>
        {/if}
        {#if domain}
          <div class="domain-badge" style="border-color: {getDomainColor(domain)}">
            <span class="domain-icon" style="background: {getDomainColor(domain)}"></span>
            <span class="domain-name">{domain}</span>
          </div>
        {/if}
      </div>
    {/if}

    <!-- E/N/M/H Dimensions -->
    <div class="dimensions">
      <div class="dimension" title={dimensionInfo.E.levels[e].desc}>
        <span class="label">E</span>
        <span class="value">{e}</span>
        <div class="bar"><div class="fill" style="width: {(e / 4) * 100}%"></div></div>
        <span class="level-name">{dimensionInfo.E.levels[e].name}</span>
      </div>
      <div class="dimension" title={dimensionInfo.N.levels[n].desc}>
        <span class="label">N</span>
        <span class="value">{n}</span>
        <div class="bar"><div class="fill" style="width: {(n / 3) * 100}%"></div></div>
        <span class="level-name">{dimensionInfo.N.levels[n].name}</span>
      </div>
      <div class="dimension" title={dimensionInfo.M.levels[m].desc}>
        <span class="label">M</span>
        <span class="value">{m}</span>
        <div class="bar"><div class="fill" style="width: {(m / 3) * 100}%"></div></div>
        <span class="level-name">{dimensionInfo.M.levels[m].name}</span>
      </div>
      <div class="dimension harmonic" title={dimensionInfo.H.levels[h].desc}>
        <span class="label" style="color: {hColor}">H</span>
        <span class="value" style="color: {hColor}">{h}</span>
        <div class="bar h-bar">
          <div class="fill h-fill" style="width: {(h / 4) * 100}%; background: {hColor}"></div>
        </div>
        <span class="level-name" style="color: {hColor}">{dimensionInfo.H.levels[h].name}</span>
      </div>
    </div>

    <!-- Confidence interval visualization -->
    {#if showConfidence && confidence !== undefined}
      <div class="confidence-section">
        <span class="confidence-label">Confidence: {(confidence * 100).toFixed(0)}%</span>
        <div class="confidence-bar">
          <div class="confidence-range" style="left: {confidenceLow}%; width: {confidenceHigh - confidenceLow}%"></div>
          <div class="confidence-point" style="left: {confidenceWidth}%"></div>
        </div>
        <div class="confidence-labels">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    {/if}

    <!-- Coherence indicator -->
    {#if coherence !== undefined}
      <div class="coherence-indicator">
        <span class="coherence-label">Coherence</span>
        <div class="coherence-dots">
          {#each Array(5) as _, i}
            <span
              class="coherence-dot"
              class:active={coherence >= (i + 1) / 5}
            ></span>
          {/each}
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  /* Compact Badge */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.75rem;
    font-weight: 600;
    position: relative;
  }

  .badge.interactive {
    cursor: help;
  }

  .badge.high { background: #065f46; color: #d1fae5; }
  .badge.medium { background: #92400e; color: #fef3c7; }
  .badge.low { background: #7f1d1d; color: #fecaca; }

  .phi-indicator {
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 0.65rem;
    color: white;
  }

  .domain-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  /* Tooltip */
  .tooltip {
    position: absolute;
    top: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
    min-width: 200px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .tooltip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #334155;
  }

  .epistemic-code {
    font-family: monospace;
    font-weight: 700;
    font-size: 1rem;
  }

  .phi-badge {
    background: #3b82f6;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
  }

  .tooltip-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .tooltip-dim {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .dim-label {
    font-family: monospace;
    font-weight: 700;
    font-size: 0.85rem;
  }

  .dim-name {
    font-size: 0.7rem;
    color: #94a3b8;
  }

  .tooltip-phi,
  .tooltip-domain {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #334155;
    font-size: 0.75rem;
  }

  /* Full Badge */
  .epistemic-badge {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px 16px;
    background: #1a1a2e;
    border-radius: 8px;
    border: 1px solid #2a2a4e;
  }

  .badge-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a4e;
  }

  .phi-display {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .phi-symbol {
    font-size: 1.2rem;
    font-weight: 700;
    color: #3b82f6;
  }

  .phi-value {
    font-size: 1.1rem;
    font-weight: 600;
    font-family: monospace;
  }

  .phi-desc {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-left: 4px;
  }

  .domain-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border: 1px solid;
    border-radius: 12px;
    font-size: 0.7rem;
  }

  .domain-icon {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .dimensions {
    display: flex;
    gap: 12px;
  }

  .dimension {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    min-width: 50px;
  }

  .label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
  }

  .value {
    font-size: 1.1rem;
    font-weight: 700;
    font-family: monospace;
  }

  .high .value { color: #34d399; }
  .medium .value { color: #fbbf24; }
  .low .value { color: #f87171; }

  .level-name {
    font-size: 0.6rem;
    color: #666;
    text-align: center;
  }

  .bar {
    width: 100%;
    height: 4px;
    background: #333;
    border-radius: 2px;
    overflow: hidden;
  }

  .fill {
    height: 100%;
    transition: width 0.3s ease;
  }

  .high .fill { background: #34d399; }
  .medium .fill { background: #fbbf24; }
  .low .fill { background: #f87171; }

  .h-bar {
    background: #222;
  }

  /* Confidence Section */
  .confidence-section {
    padding-top: 8px;
    border-top: 1px solid #2a2a4e;
  }

  .confidence-label {
    font-size: 0.7rem;
    color: #888;
    display: block;
    margin-bottom: 6px;
  }

  .confidence-bar {
    position: relative;
    height: 8px;
    background: #333;
    border-radius: 4px;
  }

  .confidence-range {
    position: absolute;
    height: 100%;
    background: rgba(59, 130, 246, 0.3);
    border-radius: 4px;
  }

  .confidence-point {
    position: absolute;
    top: 50%;
    width: 12px;
    height: 12px;
    background: #3b82f6;
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }

  .confidence-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #666;
    margin-top: 4px;
  }

  /* Coherence Indicator */
  .coherence-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-top: 8px;
    border-top: 1px solid #2a2a4e;
  }

  .coherence-label {
    font-size: 0.7rem;
    color: #888;
  }

  .coherence-dots {
    display: flex;
    gap: 4px;
  }

  .coherence-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #333;
  }

  .coherence-dot.active {
    background: #8b5cf6;
  }
</style>
