import { writable } from 'svelte/store';

export type ToastType = 'success' | 'error' | 'info';

export interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

let nextId = 0;

function createToastStore() {
  const { subscribe, update } = writable<Toast[]>([]);

  function add(message: string, type: ToastType = 'success', durationMs = 4000) {
    const id = nextId++;
    update((toasts) => [...toasts, { id, message, type }]);
    setTimeout(() => dismiss(id), durationMs);
  }

  function dismiss(id: number) {
    update((toasts) => toasts.filter((t) => t.id !== id));
  }

  return {
    subscribe,
    success: (msg: string) => add(msg, 'success'),
    error: (msg: string) => add(msg, 'error', 6000),
    info: (msg: string) => add(msg, 'info'),
    dismiss,
  };
}

export const toasts = createToastStore();
