<script lang="ts">
  import type { RiskSignal } from '$types';

  export let risk: RiskSignal | null = null;
  export let proof: string | null = null;
  export let open = false;
  let simulateProof = false;

  $: hasRisk = risk && (risk.score || 0) > 0.5;
  $: primaryFlag = risk?.flags?.[0];
  $: simulatedScore = simulateProof ? Math.max(0, (risk?.score || 0) - 0.25) : risk?.score || 0;
</script>

{#if hasRisk}
  <details class="insight" open={open}>
    <summary>
      <span>Why this risk?</span>
      <span class="score">Risk {simulatedScore.toFixed(2)}</span>
    </summary>
    <div class="insight-body">
      {#if proof}
        <p class="proof-line">
          Proof status: <span class={`proof-pill small ${proof}`}>{proof}</span>
        </p>
      {/if}
      <p class="lead">{primaryFlag || 'Potential anomaly detected'}</p>
      {#if risk?.flags?.length}
        <ul>
          {#each risk.flags as flag}
            <li>{flag}</li>
          {/each}
        </ul>
      {/if}
      <label class="simulate">
        <input type="checkbox" bind:checked={simulateProof} />
        Simulate proof fulfilled (see projected risk)
      </label>
    </div>
  </details>
{/if}

<style>
  .insight {
    border: 1px solid #fed7aa;
    background: #fff7ed;
    border-radius: 0.5rem;
    padding: 0.75rem 0.9rem;
  }

  summary {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    font-weight: 700;
    color: #9a3412;
  }

  .score {
    font-size: 0.8rem;
    color: #b45309;
  }

  .insight-body {
    margin-top: 0.6rem;
    color: #7c2d12;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .lead {
    margin: 0;
    font-weight: 700;
  }

  ul {
    margin: 0.25rem 0 0;
    padding-left: 1.1rem;
  }

  li {
    margin-bottom: 0.15rem;
  }

  .proof-line {
    margin: 0;
    font-size: 0.9rem;
    color: #9a3412;
  }

  .simulate {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.9rem;
    color: #9a3412;
    font-weight: 700;
  }
</style>
