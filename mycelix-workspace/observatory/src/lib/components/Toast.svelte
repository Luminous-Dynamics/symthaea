<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { toasts } from '$lib/toast';

  const typeStyles: Record<string, string> = {
    success: 'bg-green-800 border-green-600 text-green-100',
    error: 'bg-red-800 border-red-600 text-red-100',
    info: 'bg-blue-800 border-blue-600 text-blue-100',
  };
</script>

{#if $toasts.length > 0}
  <div class="fixed top-4 right-4 z-[60] space-y-2 max-w-sm" aria-live="polite" aria-label="Notifications">
    {#each $toasts as toast (toast.id)}
      <div
        class="px-4 py-3 rounded-lg border shadow-lg text-sm flex items-start gap-2 animate-slide-in {typeStyles[toast.type] ?? typeStyles.info}"
        role="status"
      >
        <span class="flex-1">{toast.message}</span>
        <button
          on:click={() => toasts.dismiss(toast.id)}
          class="text-current opacity-60 hover:opacity-100 ml-2 shrink-0"
          aria-label="Dismiss notification"
        >
          &#x2715;
        </button>
      </div>
    {/each}
  </div>
{/if}

<style>
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  :global(.animate-slide-in) {
    animation: slideIn 0.2s ease-out;
  }
</style>
