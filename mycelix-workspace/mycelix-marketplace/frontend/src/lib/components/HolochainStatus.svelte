<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { holochain } from '$lib/stores';
  import { initHolochainClient } from '$lib/holochain';
  import { notifications } from '$lib/stores';

  let reconnecting = false;

  const statusCopy: Record<string, { title: string; detail: string }> = {
    disconnected: {
      title: 'Not connected to Holochain',
      detail: 'Connect to sync listings, purchases, and disputes.',
    },
    connecting: {
      title: 'Connecting to Holochain…',
      detail: 'Establishing a secure conductor session.',
    },
    error: {
      title: 'Holochain connection failed',
      detail: 'Check the conductor URL and restart if needed.',
    },
  };

  async function retry() {
    if (reconnecting || $holochain.status === 'connecting') return;
    reconnecting = true;

    try {
      await initHolochainClient($holochain.url);
      notifications.info('Reconnected', `Holochain at ${$holochain.url}`);
    } catch (e: any) {
      console.error('Holochain reconnect failed:', e);
      notifications.error('Reconnect failed', e?.message || 'Unable to reach Holochain conductor');
    } finally {
      reconnecting = false;
    }
  }

  $: shouldShow = $holochain.status !== 'connected';
  $: copy = statusCopy[$holochain.status] || statusCopy.disconnected;
</script>

{#if shouldShow}
  <div class="hc-banner" role="status" aria-live="polite">
    <div class="hc-text">
      <div class="hc-title">{copy.title}</div>
      <div class="hc-detail">
        {copy.detail}
        {#if $holochain.error}
          <span class="hc-error">{$holochain.error}</span>
        {/if}
      </div>
      <div class="hc-url">Target: {$holochain.url}</div>
    </div>
    <div class="hc-actions">
      <button
        class="hc-button"
        on:click={retry}
        disabled={reconnecting || $holochain.status === 'connecting'}
      >
        {#if reconnecting || $holochain.status === 'connecting'}
          Trying…
        {:else}
          Retry connection
        {/if}
      </button>
    </div>
  </div>
{/if}

<style>
  .hc-banner {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    z-index: 40;
    display: flex;
    gap: 1rem;
    align-items: center;
    background: #1a202c;
    color: #f7fafc;
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
    max-width: 420px;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .hc-text {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.95rem;
  }

  .hc-title {
    font-weight: 700;
    font-size: 1rem;
  }

  .hc-detail {
    color: #e2e8f0;
    line-height: 1.4;
  }

  .hc-error {
    display: block;
    color: #fbb6ce;
    margin-top: 0.25rem;
    word-break: break-word;
  }

  .hc-url {
    font-size: 0.85rem;
    color: #a0aec0;
  }

  .hc-actions {
    display: flex;
    align-items: center;
  }

  .hc-button {
    background: #48bb78;
    color: #0b2617;
    border: none;
    padding: 0.5rem 0.85rem;
    border-radius: 0.5rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
  }

  .hc-button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(72, 187, 120, 0.35);
  }

  .hc-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  @media (max-width: 640px) {
    .hc-banner {
      left: 1rem;
      right: 1rem;
      flex-direction: column;
      align-items: flex-start;
    }

    .hc-actions {
      width: 100%;
    }

    .hc-button {
      width: 100%;
      text-align: center;
    }
  }
</style>
