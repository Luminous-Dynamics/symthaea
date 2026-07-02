<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { holochain } from '$lib/stores';
  import { initHolochainClient } from '$lib/holochain';
  import { notifications } from '$lib/stores';

  export let showWhenConnecting = false;
  export let tone: 'info' | 'warn' = 'warn';

  const toneColors: Record<typeof tone, { bg: string; text: string; accent: string }> = {
    info: { bg: '#ebf8ff', text: '#2a4365', accent: '#3182ce' },
    warn: { bg: '#fffaf0', text: '#7b341e', accent: '#dd6b20' },
  };

  async function retry() {
    try {
      await initHolochainClient($holochain.url);
      notifications.info('Reconnected', `Connected to Holochain at ${$holochain.url}`);
    } catch (e: any) {
      notifications.error('Reconnect failed', e?.message || 'Unable to reach Holochain conductor');
    }
  }

  $: visible =
    $holochain.status === 'error' ||
    $holochain.status === 'disconnected' ||
    (showWhenConnecting && $holochain.status === 'connecting');
  $: colors = toneColors[tone];
</script>

{#if visible}
  <div
    class="hc-inline"
    role="status"
    aria-live="polite"
    style={`background:${colors.bg};color:${colors.text};border-color:${colors.accent};`}
  >
    <div class="hc-copy">
      <div class="hc-title">
        {#if $holochain.status === 'error'}
          Holochain connection failed
        {:else if $holochain.status === 'connecting'}
          Connecting to Holochain…
        {:else}
          Not connected to Holochain
        {/if}
      </div>
      <div class="hc-detail">
        {#if $holochain.status === 'connecting'}
          Establishing conductor session. This may take a moment.
        {:else if $holochain.status === 'error'}
          Check the conductor URL and try again.
        {:else}
          Start your conductor or update the WebSocket URL, then retry.
        {/if}
        {#if $holochain.error}
          <span class="hc-error">{$holochain.error}</span>
        {/if}
      </div>
      <div class="hc-url">Target: {$holochain.url}</div>
    </div>
    <button
      class="hc-retry"
      on:click={retry}
      disabled={$holochain.status === 'connecting'}
      style={`background:${colors.accent};color:white;`}
    >
      {$holochain.status === 'connecting' ? 'Connecting…' : 'Retry'}
    </button>
  </div>
{/if}

<style>
  .hc-inline {
    border: 1px solid;
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1rem;
  }

  .hc-copy {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    flex: 1;
  }

  .hc-title {
    font-weight: 700;
  }

  .hc-detail {
    font-size: 0.95rem;
    line-height: 1.4;
  }

  .hc-error {
    display: block;
    margin-top: 0.25rem;
    color: #c53030;
  }

  .hc-url {
    font-size: 0.85rem;
    opacity: 0.8;
  }

  .hc-retry {
    border: none;
    padding: 0.55rem 0.9rem;
    border-radius: 0.5rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.15s ease, transform 0.15s ease;
  }

  .hc-retry:hover:not(:disabled) {
    transform: translateY(-1px);
  }

  .hc-retry:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  @media (max-width: 640px) {
    .hc-inline {
      flex-direction: column;
      align-items: flex-start;
    }

    .hc-retry {
      width: 100%;
      text-align: center;
    }
  }
</style>
