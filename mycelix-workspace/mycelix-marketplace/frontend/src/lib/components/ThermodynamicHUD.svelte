<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
-->
<script lang="ts">
  /**
   * Thermodynamic HUD Component
   *
   * Visualizes the 'Cost of Reality' for a product by aggregating
   * E4 Thermodynamic Claims from the supply chain.
   */
  export let total_joules: number = 0;
  export let logistics_joules: number = 0;
  export let production_joules: number = 0;
  export let verified: boolean = false;

  $: entropy_score = (total_joules / 1000000).toFixed(2); // Convert to MegaJoules for readability
</script>

<section class="thermo-hud">
  <header class="hud-header">
    <div class="title-group">
      <p class="eyebrow">Thermodynamic Ledger</p>
      <h4>Cost of Reality</h4>
    </div>
    {#if verified}
      <span class="status verified">VERIFIED E4</span>
    {:else}
      <span class="status pending">ESTIMATED</span>
    {/if}
  </header>

  <div class="metrics-grid">
    <div class="metric">
      <span class="label">Total Entropy</span>
      <span class="value">{entropy_score} MJ</span>
    </div>
    <div class="metric">
      <span class="label">Production</span>
      <span class="value">{ (production_joules / 1000).toFixed(1) } kJ</span>
    </div>
    <div class="metric">
      <span class="label">Logistics</span>
      <span class="value">{ (logistics_joules / 1000).toFixed(1) } kJ</span>
    </div>
  </div>

  <div class="visualization">
    <div class="progress-bar">
      <div class="fill production" style="width: { (production_joules / total_joules) * 100 }%"></div>
      <div class="fill logistics" style="width: { (logistics_joules / total_joules) * 100 }%"></div>
    </div>
    <div class="legend">
      <span><i class="dot production"></i> Production</span>
      <span><i class="dot logistics"></i> Logistics</span>
    </div>
  </div>

  <p class="explanation">
    This product cost <strong>{entropy_score} MegaJoules</strong> of planetary energy to manifest. 
    Staked SAP ensures this ledger is empirically accurate.
  </p>
</section>

<style>
  .thermo-hud {
    background: #0f172a;
    color: white;
    border-radius: 1rem;
    padding: 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .hud-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.25rem;
  }

  .eyebrow {
    color: #fbbf24;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem;
    font-weight: 800;
    margin: 0;
  }

  h4 {
    margin: 0.2rem 0 0;
    font-size: 1.1rem;
    font-weight: 600;
  }

  .status {
    font-size: 0.65rem;
    font-weight: 900;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
  }

  .status.verified {
    background: #059669;
    color: white;
  }

  .status.pending {
    background: #475569;
    color: #cbd5e1;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .metric {
    display: flex;
    flex-direction: column;
  }

  .label {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-bottom: 0.25rem;
  }

  .value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
  }

  .visualization {
    margin-bottom: 1rem;
  }

  .progress-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 99px;
    display: flex;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  .fill {
    height: 100%;
  }

  .fill.production { background: #6366f1; }
  .fill.logistics { background: #fbbf24; }

  .legend {
    display: flex;
    gap: 1rem;
    font-size: 0.7rem;
    color: #94a3b8;
  }

  .dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 4px;
  }

  .dot.production { background: #6366f1; }
  .dot.logistics { background: #fbbf24; }

  .explanation {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.5;
    margin: 0;
  }

  strong {
    color: white;
  }
</style>
