/**
 * Notifications Store
 *
 * Centralized state management for toast notifications and real-time updates
 */

import { writable, derived } from 'svelte/store';

/**
 * Notification type
 */
export type NotificationType = 'info' | 'success' | 'warning' | 'error';

/**
 * Notification data structure
 */
export interface Notification {
  /** Unique notification ID */
  id: string;

  /** Notification type (affects styling) */
  type: NotificationType;

  /** Message to display */
  message: string;

  /** Optional title */
  title?: string;

  /** Duration in milliseconds (0 = manual dismiss only) */
  duration?: number;

  /** Creation timestamp */
  createdAt: number;

  /** Whether notification is dismissible */
  dismissible?: boolean;

  /** Optional action button */
  action?: {
    label: string;
    callback: () => void;
  };
}

/**
 * Default notification duration (5 seconds)
 */
const DEFAULT_DURATION = 5000;

/**
 * Generate unique notification ID
 */
function generateId(): string {
  return `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Create notifications store
 */
function createNotificationsStore() {
  const { subscribe, update } = writable<Notification[]>([]);

  // Track timers for auto-dismiss
  const timers = new Map<string, NodeJS.Timeout>();

  return {
    subscribe,

    /**
     * Add a new notification
     */
    add: (notification: Omit<Notification, 'id' | 'createdAt'>) => {
      const id = generateId();
      const duration = notification.duration ?? DEFAULT_DURATION;

      const newNotification: Notification = {
        ...notification,
        id,
        createdAt: Date.now(),
        dismissible: notification.dismissible ?? true,
      };

      update((notifications) => [...notifications, newNotification]);

      // Auto-dismiss after duration
      if (duration > 0) {
        const timer = setTimeout(() => {
          notificationsStore.dismiss(id);
        }, duration);
        timers.set(id, timer);
      }

      return id;
    },

    /**
     * Dismiss a notification
     */
    dismiss: (id: string) => {
      // Clear timer if exists
      const timer = timers.get(id);
      if (timer) {
        clearTimeout(timer);
        timers.delete(id);
      }

      update((notifications) => notifications.filter((n) => n.id !== id));
    },

    /**
     * Clear all notifications
     */
    clear: () => {
      // Clear all timers
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();

      update(() => []);
    },

    /**
     * Convenience methods for common notification types
     */
    info: (message: string, title?: string, duration?: number) => {
      return notificationsStore.add({ type: 'info', message, title, duration });
    },

    success: (message: string, title?: string, duration?: number) => {
      return notificationsStore.add({ type: 'success', message, title, duration });
    },

    warning: (message: string, title?: string, duration?: number) => {
      return notificationsStore.add({ type: 'warning', message, title, duration });
    },

    error: (message: string, title?: string, duration?: number) => {
      return notificationsStore.add({
        type: 'error',
        message,
        title,
        duration: duration ?? 0, // Errors don't auto-dismiss by default
      });
    },
  };
}

/**
 * Notifications store
 */
export const notificationsStore = createNotificationsStore();

/**
 * Derived store: count of active notifications
 */
export const notificationCount = derived(
  notificationsStore,
  ($notifications) => $notifications.length
);

/**
 * Derived store: has any notifications
 */
export const hasNotifications = derived(notificationCount, ($count) => $count > 0);

/**
 * Derived store: notifications by type
 */
export const notificationsByType = derived(notificationsStore, ($notifications) => {
  const byType: Record<NotificationType, Notification[]> = {
    info: [],
    success: [],
    warning: [],
    error: [],
  };

  $notifications.forEach((notification) => {
    byType[notification.type].push(notification);
  });

  return byType;
});

/**
 * Export default as 'notifications' for cleaner imports
 */
export const notifications = notificationsStore;
