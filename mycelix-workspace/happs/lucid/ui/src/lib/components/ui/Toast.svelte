<script lang="ts" context="module">
  import { writable } from 'svelte/store';

  export type ToastType = 'success' | 'error' | 'warning' | 'info';

  export interface Toast {
    id: string;
    type: ToastType;
    message: string;
    duration?: number;
  }

  function createToastStore() {
    const { subscribe, update } = writable<Toast[]>([]);

    function add(type: ToastType, message: string, duration = 5000): string {
      const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`;

      update((toasts) => [...toasts, { id, type, message, duration }]);

      if (duration > 0) {
        setTimeout(() => remove(id), duration);
      }

      return id;
    }

    function remove(id: string): void {
      update((toasts) => toasts.filter((t) => t.id !== id));
    }

    function clear(): void {
      update(() => []);
    }

    return {
      subscribe,
      success: (message: string, duration?: number) => add('success', message, duration),
      error: (message: string, duration?: number) => add('error', message, duration ?? 8000),
      warning: (message: string, duration?: number) => add('warning', message, duration),
      info: (message: string, duration?: number) => add('info', message, duration),
      remove,
      clear,
    };
  }

  export const toasts = createToastStore();
</script>

<script lang="ts">
  import { fly, fade } from 'svelte/transition';
  import { flip } from 'svelte/animate';

  function getIcon(type: ToastType): string {
    switch (type) {
      case 'success':
        return 'check';
      case 'error':
        return 'x';
      case 'warning':
        return '!';
      case 'info':
        return 'i';
    }
  }
</script>

<div class="toast-container" aria-live="polite">
  {#each $toasts as toast (toast.id)}
    <div
      class="toast toast-{toast.type}"
      role="alert"
      in:fly={{ x: 300, duration: 300 }}
      out:fade={{ duration: 200 }}
      animate:flip={{ duration: 300 }}
    >
      <div class="toast-icon">
        <span class="icon-text">{getIcon(toast.type)}</span>
      </div>
      <div class="toast-content">
        <p class="toast-message">{toast.message}</p>
      </div>
      <button
        class="toast-close"
        on:click={() => toasts.remove(toast.id)}
        aria-label="Dismiss notification"
      >
        <span class="close-icon">x</span>
      </button>
    </div>
  {/each}
</div>

<style>
  .toast-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 400px;
  }

  .toast {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    border-radius: 10px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
  }

  .toast-success {
    border-left: 4px solid #22c55e;
  }

  .toast-error {
    border-left: 4px solid #ef4444;
  }

  .toast-warning {
    border-left: 4px solid #f59e0b;
  }

  .toast-info {
    border-left: 4px solid #3b82f6;
  }

  .toast-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .toast-success .toast-icon {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
  }

  .toast-error .toast-icon {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
  }

  .toast-warning .toast-icon {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
  }

  .toast-info .toast-icon {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
  }

  .icon-text {
    font-size: 12px;
    font-weight: 700;
  }

  .toast-content {
    flex: 1;
  }

  .toast-message {
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-primary, #f8fafc);
  }

  .toast-close {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: none;
    background: transparent;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    flex-shrink: 0;
  }

  .toast-close:hover {
    background: var(--bg-tertiary, #0f172a);
    color: var(--text-primary, #f8fafc);
  }

  .close-icon {
    font-size: 14px;
    font-weight: 500;
  }

  @media (max-width: 480px) {
    .toast-container {
      left: 10px;
      right: 10px;
      bottom: 10px;
      max-width: none;
    }
  }
</style>
