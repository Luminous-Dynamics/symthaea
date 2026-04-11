// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Notifications Component
 *
 * Real-time notifications for trust-related events:
 * - New trust attestations received
 * - Introduction requests
 * - Trust score changes
 * - Credential verifications
 * - Quarantine alerts
 *
 * Displays as a notification panel with actions.
 */

import { useState, useEffect, useCallback } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Notification types
export type TrustNotificationType =
  | 'attestation_received'
  | 'attestation_sent'
  | 'introduction_request'
  | 'introduction_accepted'
  | 'trust_score_change'
  | 'credential_verified'
  | 'credential_expired'
  | 'quarantine_alert'
  | 'network_update';

export interface TrustNotification {
  id: string;
  type: TrustNotificationType;
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionable: boolean;
  data?: {
    contactDid?: string;
    contactEmail?: string;
    contactName?: string;
    trustScore?: number;
    previousScore?: number;
    credentialType?: string;
    introductionId?: string;
  };
}

// Notification store
interface TrustNotificationStore {
  notifications: TrustNotification[];
  unreadCount: number;
  addNotification: (notification: Omit<TrustNotification, 'id' | 'timestamp' | 'read'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
}

export const useTrustNotificationStore = create<TrustNotificationStore>()(
  persist(
    (set, get) => ({
      notifications: [],
      unreadCount: 0,

      addNotification: (notification) => {
        const newNotification: TrustNotification = {
          ...notification,
          id: `notif-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          timestamp: new Date().toISOString(),
          read: false,
        };

        set((state) => ({
          notifications: [newNotification, ...state.notifications].slice(0, 50), // Keep last 50
          unreadCount: state.unreadCount + 1,
        }));
      },

      markAsRead: (id) => {
        set((state) => ({
          notifications: state.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          ),
          unreadCount: Math.max(0, state.unreadCount - (state.notifications.find((n) => n.id === id && !n.read) ? 1 : 0)),
        }));
      },

      markAllAsRead: () => {
        set((state) => ({
          notifications: state.notifications.map((n) => ({ ...n, read: true })),
          unreadCount: 0,
        }));
      },

      removeNotification: (id) => {
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
          unreadCount: Math.max(0, state.unreadCount - (state.notifications.find((n) => n.id === id && !n.read) ? 1 : 0)),
        }));
      },

      clearAll: () => {
        set({ notifications: [], unreadCount: 0 });
      },
    }),
    {
      name: 'trust-notifications',
    }
  )
);

// Notification type config
const notificationConfig: Record<TrustNotificationType, {
  icon: string;
  color: string;
  bgColor: string;
}> = {
  attestation_received: {
    icon: '🤝',
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
  },
  attestation_sent: {
    icon: '✅',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
  },
  introduction_request: {
    icon: '👋',
    color: 'text-amber-600',
    bgColor: 'bg-amber-50 dark:bg-amber-900/20',
  },
  introduction_accepted: {
    icon: '🎉',
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
  },
  trust_score_change: {
    icon: '📊',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/20',
  },
  credential_verified: {
    icon: '🔐',
    color: 'text-teal-600',
    bgColor: 'bg-teal-50 dark:bg-teal-900/20',
  },
  credential_expired: {
    icon: '⚠️',
    color: 'text-red-600',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
  },
  quarantine_alert: {
    icon: '🛡️',
    color: 'text-rose-600',
    bgColor: 'bg-rose-50 dark:bg-rose-900/20',
  },
  network_update: {
    icon: '🌐',
    color: 'text-gray-600',
    bgColor: 'bg-gray-50 dark:bg-gray-800',
  },
};

// Format relative time
function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const date = new Date(timestamp);
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

// Single notification item
function NotificationItem({
  notification,
  onAction,
  onDismiss,
}: {
  notification: TrustNotification;
  onAction?: (notification: TrustNotification) => void;
  onDismiss: (id: string) => void;
}) {
  const config = notificationConfig[notification.type];
  const { markAsRead } = useTrustNotificationStore();

  const handleClick = () => {
    if (!notification.read) {
      markAsRead(notification.id);
    }
    if (notification.actionable && onAction) {
      onAction(notification);
    }
  };

  return (
    <div
      className={`relative p-4 border-b border-gray-100 dark:border-gray-800 ${
        !notification.read ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''
      } hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors`}
    >
      {/* Unread indicator */}
      {!notification.read && (
        <div className="absolute left-2 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-500" />
      )}

      <div className="flex items-start gap-3 pl-4">
        {/* Icon */}
        <div className={`w-10 h-10 rounded-lg ${config.bgColor} flex items-center justify-center flex-shrink-0`}>
          <span className="text-lg">{config.icon}</span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <p className={`font-medium text-sm ${config.color}`}>
              {notification.title}
            </p>
            <span className="text-xs text-gray-400 dark:text-gray-500 flex-shrink-0">
              {formatRelativeTime(notification.timestamp)}
            </span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-0.5 line-clamp-2">
            {notification.message}
          </p>

          {/* Actions */}
          {notification.actionable && (
            <div className="flex items-center gap-2 mt-2">
              <button
                onClick={handleClick}
                className="px-3 py-1 text-xs font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900/30 rounded transition-colors"
              >
                View Details
              </button>
              {notification.type === 'introduction_request' && (
                <>
                  <button className="px-3 py-1 text-xs font-medium text-emerald-600 dark:text-emerald-400 hover:bg-emerald-100 dark:hover:bg-emerald-900/30 rounded transition-colors">
                    Accept
                  </button>
                  <button className="px-3 py-1 text-xs font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded transition-colors">
                    Decline
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Dismiss button */}
        <button
          onClick={() => onDismiss(notification.id)}
          className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors flex-shrink-0"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}

// Notification bell button
export function NotificationBell({
  onClick,
}: {
  onClick: () => void;
}) {
  const { unreadCount } = useTrustNotificationStore();

  return (
    <button
      onClick={onClick}
      className="relative p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      title="Trust Notifications"
    >
      <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
      </svg>
      {unreadCount > 0 && (
        <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
          {unreadCount > 9 ? '9+' : unreadCount}
        </span>
      )}
    </button>
  );
}

// Main notification panel
interface TrustNotificationsProps {
  isOpen: boolean;
  onClose: () => void;
  onAction?: (notification: TrustNotification) => void;
}

export default function TrustNotifications({
  isOpen,
  onClose,
  onAction,
}: TrustNotificationsProps) {
  const {
    notifications,
    unreadCount,
    markAllAsRead,
    removeNotification,
    clearAll,
  } = useTrustNotificationStore();

  // Close on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed right-4 top-16 w-96 max-h-[80vh] bg-white dark:bg-gray-900 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 z-50 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">
              Trust Notifications
            </h3>
            {unreadCount > 0 && (
              <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 text-xs font-medium rounded-full">
                {unreadCount} new
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {unreadCount > 0 && (
              <button
                onClick={markAllAsRead}
                className="p-1.5 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
              >
                Mark all read
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Notifications list */}
        <div className="flex-1 overflow-y-auto">
          {notifications.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 px-4">
              <div className="w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-4">
                <span className="text-3xl">🔔</span>
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                No notifications yet
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-1">
                You'll see trust updates here
              </p>
            </div>
          ) : (
            notifications.map((notification) => (
              <NotificationItem
                key={notification.id}
                notification={notification}
                onAction={onAction}
                onDismiss={removeNotification}
              />
            ))
          )}
        </div>

        {/* Footer */}
        {notifications.length > 0 && (
          <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <button
              onClick={clearAll}
              className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
            >
              Clear all
            </button>
            <button className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300">
              View all activity
            </button>
          </div>
        )}
      </div>
    </>
  );
}

// Helper hook to simulate notifications (for demo)
export function useSimulateTrustNotifications() {
  const { addNotification } = useTrustNotificationStore();

  const simulate = useCallback(() => {
    const types: TrustNotificationType[] = [
      'attestation_received',
      'introduction_request',
      'trust_score_change',
      'credential_verified',
    ];

    const randomType = types[Math.floor(Math.random() * types.length)];

    const messages: Record<TrustNotificationType, { title: string; message: string }> = {
      attestation_received: {
        title: 'New Trust Attestation',
        message: 'Alice has vouched for your trustworthiness',
      },
      attestation_sent: {
        title: 'Attestation Confirmed',
        message: 'Your trust attestation for Bob has been recorded',
      },
      introduction_request: {
        title: 'Introduction Request',
        message: 'Charlie wants to introduce you to Dave',
      },
      introduction_accepted: {
        title: 'Introduction Accepted',
        message: 'Eve has accepted your introduction to Frank',
      },
      trust_score_change: {
        title: 'Trust Score Updated',
        message: 'Your trust score with Grace increased to 85%',
      },
      credential_verified: {
        title: 'Credential Verified',
        message: 'Your Gitcoin Passport has been verified',
      },
      credential_expired: {
        title: 'Credential Expiring',
        message: 'Your GitHub verification expires in 7 days',
      },
      quarantine_alert: {
        title: 'Email Quarantined',
        message: 'An email from unknown sender was held for review',
      },
      network_update: {
        title: 'Network Update',
        message: 'Your trust network has grown to 15 connections',
      },
    };

    addNotification({
      type: randomType,
      title: messages[randomType].title,
      message: messages[randomType].message,
      actionable: ['introduction_request', 'attestation_received', 'quarantine_alert'].includes(randomType),
    });
  }, [addNotification]);

  return simulate;
}

// Export store hook for external use
export { useTrustNotificationStore as useTrustNotifications };
