<script lang="ts">
  import type { RiskSignal } from '$types';

  export let risk: RiskSignal | null = null;
  export let expanded: boolean = false;

  $: hasRisk = risk && risk.score > 0.5;
</script>

{#if hasRisk}
  <div class="risk-chip">
    <div class="risk-header">
      <span class="label">Review suggested</span>
      <span class="score">Risk {(risk?.score || 0).toFixed(2)}</span>
    </div>
    <p class="reason">{risk?.flags?.[0] || 'Potential anomaly detected'}</p>
    {#if expanded && risk?.flags?.length}
      <ul class="flag-list">
        {#each risk.flags as flag}
          <li>{flag}</li>
        {/each}
      </ul>
    {/if}
  </div>
{/if}

<style>
  .risk-chip {
    background: #fff7ed;
    color: #9a3412;
    border: 1px solid #fed7aa;
    border-radius: 0.75rem;
    padding: 0.6rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .risk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
  }

  .label {
    font-weight: 700;
  }

  .score {
    font-size: 0.85rem;
    font-weight: 700;
    color: #b45309;
  }

  .reason {
    margin: 0;
    font-size: 0.9rem;
  }

  .flag-list {
    margin: 0.25rem 0 0;
    padding-left: 1.2rem;
    color: #7c2d12;
    font-size: 0.85rem;
  }
</style>
