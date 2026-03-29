// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Notifications Hook
 *
 * Real-time notification system for social features
 * Supports in-app, push, and email notifications
 */

import { useState, useCallback, useEffect, useRef } from 'react';

export type NotificationType =
  | 'follow'
  | 'like'
  | 'comment'
  | 'mention'
  | 'share'
  | 'broadcast_live'
  | 'party_invite'
  | 'new_release'
  | 'milestone'
  | 'tip'
  | 'playlist_add'
  | 'collab_request'
  | 'system';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: number;
  read: boolean;
  actionUrl?: string;
  imageUrl?: string;
  sender?: {
    id: string;
    name: string;
    avatar?: string;
  };
  metadata?: Record<string, any>;
}

export interface NotificationPreferences {
  inApp: boolean;
  push: boolean;
  email: boolean;
  types: Record<NotificationType, boolean>;
}

const DEFAULT_PREFERENCES: NotificationPreferences = {
  inApp: true,
  push: true,
  email: false,
  types: {
    follow: true,
    like: true,
    comment: true,
    mention: true,
    share: true,
    broadcast_live: true,
    party_invite: true,
    new_release: true,
    milestone: true,
    tip: true,
    playlist_add: true,
    collab_request: true,
    system: true,
  },
};

const NOTIFICATION_ICONS: Record<NotificationType, string> = {
  follow: '👤',
  like: '❤️',
  comment: '💬',
  mention: '@',
  share: '🔗',
  broadcast_live: '📡',
  party_invite: '🎉',
  new_release: '🎵',
  milestone: '🏆',
  tip: '💰',
  playlist_add: '📋',
  collab_request: '🤝',
  system: '⚙️',
};

export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [preferences, setPreferences] = useState<NotificationPreferences>(DEFAULT_PREFERENCES);
  const [isConnected, setIsConnected] = useState(false);
  const [permission, setPermission] = useState<NotificationPermission>('default');

  const wsRef = useRef<WebSocket | null>(null);

  // Request push notification permission
  const requestPermission = useCallback(async () => {
    if (!('Notification' in window)) {
      return 'denied';
    }

    const result = await Notification.requestPermission();
    setPermission(result);
    return result;
  }, []);

  // Check current permission
  useEffect(() => {
    if ('Notification' in window) {
      setPermission(Notification.permission);
    }
  }, []);

  // Connect to notification WebSocket
  const connect = useCallback((userId: string, token: string) => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.mycelix.local';
    const ws = new WebSocket(`${wsUrl}/notifications?user=${userId}&token=${token}`);

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const notification = JSON.parse(event.data) as Notification;
        addNotification(notification);
      } catch (e) {
        console.error('Failed to parse notification:', e);
      }
    };

    ws.onerror = () => {
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Reconnect after 5 seconds
      setTimeout(() => {
        if (wsRef.current === ws) {
          connect(userId, token);
        }
      }, 5000);
    };

    wsRef.current = ws;
  }, []);

  // Disconnect
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // Add notification
  const addNotification = useCallback((notification: Notification) => {
    // Check preferences
    if (!preferences.types[notification.type]) {
      return;
    }

    setNotifications(prev => [notification, ...prev.slice(0, 99)]);
    setUnreadCount(prev => prev + 1);

    // Show browser notification if permitted
    if (preferences.push && permission === 'granted') {
      new Notification(notification.title, {
        body: notification.message,
        icon: notification.imageUrl || '/icons/icon-192.png',
        tag: notification.id,
      });
    }

    // Play notification sound
    if (preferences.inApp) {
      playNotificationSound();
    }
  }, [preferences, permission]);

  // Play notification sound
  const playNotificationSound = useCallback(() => {
    const audio = new Audio('/sounds/notification.mp3');
    audio.volume = 0.3;
    audio.play().catch(() => {});
  }, []);

  // Mark as read
  const markAsRead = useCallback((id: string) => {
    setNotifications(prev =>
      prev.map(n => (n.id === id ? { ...n, read: true } : n))
    );
    setUnreadCount(prev => Math.max(0, prev - 1));
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    setUnreadCount(0);
  }, []);

  // Delete notification
  const deleteNotification = useCallback((id: string) => {
    setNotifications(prev => {
      const notification = prev.find(n => n.id === id);
      if (notification && !notification.read) {
        setUnreadCount(c => Math.max(0, c - 1));
      }
      return prev.filter(n => n.id !== id);
    });
  }, []);

  // Clear all notifications
  const clearAll = useCallback(() => {
    setNotifications([]);
    setUnreadCount(0);
  }, []);

  // Update preferences
  const updatePreferences = useCallback((updates: Partial<NotificationPreferences>) => {
    setPreferences(prev => ({ ...prev, ...updates }));
    // Persist to storage/server
    localStorage.setItem('notification_preferences', JSON.stringify({ ...preferences, ...updates }));
  }, [preferences]);

  // Load preferences from storage
  useEffect(() => {
    const stored = localStorage.getItem('notification_preferences');
    if (stored) {
      try {
        setPreferences(JSON.parse(stored));
      } catch (e) {}
    }
  }, []);

  // Mock notifications for development
  const addMockNotification = useCallback(() => {
    const types: NotificationType[] = ['follow', 'like', 'comment', 'new_release', 'tip'];
    const type = types[Math.floor(Math.random() * types.length)];

    const mockNotification: Notification = {
      id: `notif-${Date.now()}`,
      type,
      title: getMockTitle(type),
      message: getMockMessage(type),
      timestamp: Date.now(),
      read: false,
      sender: {
        id: 'user-123',
        name: 'DJ Aurora',
      },
    };

    addNotification(mockNotification);
  }, [addNotification]);

  function getMockTitle(type: NotificationType): string {
    const titles: Record<NotificationType, string> = {
      follow: 'New Follower',
      like: 'New Like',
      comment: 'New Comment',
      mention: 'You were mentioned',
      share: 'Track Shared',
      broadcast_live: 'DJ is Live',
      party_invite: 'Party Invitation',
      new_release: 'New Release',
      milestone: 'Milestone Reached',
      tip: 'You received a tip',
      playlist_add: 'Added to Playlist',
      collab_request: 'Collaboration Request',
      system: 'System Update',
    };
    return titles[type];
  }

  function getMockMessage(type: NotificationType): string {
    const messages: Record<NotificationType, string> = {
      follow: 'DJ Aurora started following you',
      like: 'DJ Aurora liked your track "Summer Vibes"',
      comment: 'DJ Aurora commented on your track',
      mention: 'DJ Aurora mentioned you in a comment',
      share: 'DJ Aurora shared your track',
      broadcast_live: 'DJ Aurora just went live!',
      party_invite: 'DJ Aurora invited you to a listening party',
      new_release: 'DJ Aurora released a new track',
      milestone: 'Congratulations! You reached 1,000 streams',
      tip: 'DJ Aurora sent you a $5.00 tip',
      playlist_add: 'Your track was added to "Chill Vibes"',
      collab_request: 'DJ Aurora wants to collaborate with you',
      system: 'Your account settings have been updated',
    };
    return messages[type];
  }

  // Cleanup
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    // State
    notifications,
    unreadCount,
    preferences,
    isConnected,
    permission,

    // Actions
    connect,
    disconnect,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    clearAll,
    updatePreferences,
    requestPermission,

    // Dev helpers
    addMockNotification,

    // Constants
    NOTIFICATION_ICONS,
  };
}
