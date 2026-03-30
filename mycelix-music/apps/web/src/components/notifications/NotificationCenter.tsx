// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useNotifications, Notification, NotificationType } from '@/hooks/useNotifications';
import { Bell, X, Check, CheckCheck, Settings, Trash2 } from 'lucide-react';

interface NotificationCenterProps {
  className?: string;
}

export function NotificationCenter({ className = '' }: NotificationCenterProps) {
  const {
    notifications,
    unreadCount,
    preferences,
    permission,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    clearAll,
    updatePreferences,
    requestPermission,
    NOTIFICATION_ICONS,
  } = useNotifications();

  const [isOpen, setIsOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setShowSettings(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Format time ago
  const formatTimeAgo = (timestamp: number): string => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);

    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  // Group notifications by date
  const groupedNotifications = notifications.reduce((groups, notification) => {
    const date = new Date(notification.timestamp).toDateString();
    if (!groups[date]) groups[date] = [];
    groups[date].push(notification);
    return groups;
  }, {} as Record<string, Notification[]>);

  return (
    <div className={`relative ${className}`} ref={panelRef}>
      {/* Bell Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 rounded-lg hover:bg-gray-800 transition-colors"
        aria-label={`Notifications ${unreadCount > 0 ? `(${unreadCount} unread)` : ''}`}
      >
        <Bell className="w-5 h-5 text-gray-400" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {/* Notification Panel */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-96 max-h-[80vh] bg-gray-900 border border-gray-800 rounded-xl shadow-2xl overflow-hidden z-50">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-800">
            <h3 className="font-semibold text-white">Notifications</h3>
            <div className="flex items-center gap-2">
              {unreadCount > 0 && (
                <button
                  onClick={markAllAsRead}
                  className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
                  title="Mark all as read"
                >
                  <CheckCheck className="w-4 h-4 text-gray-400" />
                </button>
              )}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-2 rounded-lg transition-colors ${
                  showSettings ? 'bg-gray-800 text-white' : 'hover:bg-gray-800 text-gray-400'
                }`}
                title="Settings"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="p-4 border-b border-gray-800 bg-gray-800/50">
              <h4 className="text-sm font-medium text-white mb-3">Notification Settings</h4>

              {/* Push Permission */}
              {permission !== 'granted' && (
                <button
                  onClick={requestPermission}
                  className="w-full mb-3 px-4 py-2 bg-purple-500 text-white text-sm font-medium rounded-lg hover:bg-purple-600 transition-colors"
                >
                  Enable Push Notifications
                </button>
              )}

              {/* Toggles */}
              <div className="space-y-2">
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-700/50">
                  <span className="text-sm text-gray-300">In-app notifications</span>
                  <input
                    type="checkbox"
                    checked={preferences.inApp}
                    onChange={(e) => updatePreferences({ inApp: e.target.checked })}
                    className="w-4 h-4 accent-purple-500"
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-700/50">
                  <span className="text-sm text-gray-300">Push notifications</span>
                  <input
                    type="checkbox"
                    checked={preferences.push}
                    onChange={(e) => updatePreferences({ push: e.target.checked })}
                    className="w-4 h-4 accent-purple-500"
                    disabled={permission !== 'granted'}
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-700/50">
                  <span className="text-sm text-gray-300">Email notifications</span>
                  <input
                    type="checkbox"
                    checked={preferences.email}
                    onChange={(e) => updatePreferences({ email: e.target.checked })}
                    className="w-4 h-4 accent-purple-500"
                  />
                </label>
              </div>

              {notifications.length > 0 && (
                <button
                  onClick={clearAll}
                  className="w-full mt-3 px-4 py-2 border border-red-500/50 text-red-400 text-sm font-medium rounded-lg hover:bg-red-500/10 transition-colors"
                >
                  Clear All Notifications
                </button>
              )}
            </div>
          )}

          {/* Notifications List */}
          <div className="overflow-y-auto max-h-96">
            {notifications.length === 0 ? (
              <div className="p-8 text-center">
                <Bell className="w-12 h-12 text-gray-700 mx-auto mb-3" />
                <p className="text-gray-500">No notifications yet</p>
              </div>
            ) : (
              Object.entries(groupedNotifications).map(([date, items]) => (
                <div key={date}>
                  <div className="px-4 py-2 bg-gray-800/50 text-xs font-medium text-gray-500">
                    {date === new Date().toDateString() ? 'Today' : date}
                  </div>
                  {items.map((notification) => (
                    <NotificationItem
                      key={notification.id}
                      notification={notification}
                      icon={NOTIFICATION_ICONS[notification.type]}
                      onRead={() => markAsRead(notification.id)}
                      onDelete={() => deleteNotification(notification.id)}
                      formatTimeAgo={formatTimeAgo}
                    />
                  ))}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Individual notification item
interface NotificationItemProps {
  notification: Notification;
  icon: string;
  onRead: () => void;
  onDelete: () => void;
  formatTimeAgo: (timestamp: number) => string;
}

function NotificationItem({
  notification,
  icon,
  onRead,
  onDelete,
  formatTimeAgo,
}: NotificationItemProps) {
  return (
    <div
      className={`relative group flex gap-3 p-4 hover:bg-gray-800/50 transition-colors cursor-pointer ${
        !notification.read ? 'bg-purple-500/5' : ''
      }`}
      onClick={onRead}
    >
      {/* Unread indicator */}
      {!notification.read && (
        <div className="absolute left-2 top-1/2 -translate-y-1/2 w-2 h-2 bg-purple-500 rounded-full" />
      )}

      {/* Icon */}
      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center text-lg">
        {notification.sender?.avatar ? (
          <img
            src={notification.sender.avatar}
            alt=""
            className="w-full h-full rounded-full object-cover"
          />
        ) : (
          icon
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-sm text-white font-medium">{notification.title}</p>
        <p className="text-sm text-gray-400 truncate">{notification.message}</p>
        <p className="text-xs text-gray-500 mt-1">{formatTimeAgo(notification.timestamp)}</p>
      </div>

      {/* Actions */}
      <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="p-1 rounded hover:bg-gray-700"
          title="Delete"
        >
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>
    </div>
  );
}

// Compact notification bell for header
export function NotificationBell({ className = '' }: { className?: string }) {
  return <NotificationCenter className={className} />;
}

export default NotificationCenter;
