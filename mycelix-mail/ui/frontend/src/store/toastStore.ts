import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
  action?: ToastAction;
}

interface ToastState {
  toasts: Toast[];
  addToast: (message: string, type: ToastType, duration?: number, action?: ToastAction) => void;
  removeToast: (id: string) => void;
  clearAllToasts: () => void;
}

const generateId = () => Math.random().toString(36).substring(2, 9);
const MAX_TOASTS = 3; // Maximum number of toasts to display at once

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],

  addToast: (message: string, type: ToastType, duration = 5000, action?: ToastAction) => {
    const id = generateId();
    const toast: Toast = { id, message, type, duration, action };

    set((state) => {
      const newToasts = [...state.toasts, toast];

      // If we exceed the limit, remove oldest toasts
      if (newToasts.length > MAX_TOASTS) {
        return { toasts: newToasts.slice(newToasts.length - MAX_TOASTS) };
      }

      return { toasts: newToasts };
    });

    // Auto-remove toast after duration
    if (duration > 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      }, duration);
    }
  },

  removeToast: (id: string) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },

  clearAllToasts: () => {
    set({ toasts: [] });
  },
}));

// Convenience helper functions
export const toast = {
  success: (message: string, duration?: number, action?: ToastAction) =>
    useToastStore.getState().addToast(message, 'success', duration, action),
  error: (message: string, duration?: number, action?: ToastAction) =>
    useToastStore.getState().addToast(message, 'error', duration, action),
  warning: (message: string, duration?: number, action?: ToastAction) =>
    useToastStore.getState().addToast(message, 'warning', duration, action),
  info: (message: string, duration?: number, action?: ToastAction) =>
    useToastStore.getState().addToast(message, 'info', duration, action),
};
