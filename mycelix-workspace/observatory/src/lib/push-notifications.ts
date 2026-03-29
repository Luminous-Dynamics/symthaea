// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Push Notifications for Mycelix Observatory
 *
 * Uses the browser Notification API (not Push API) to show desktop notifications
 * for Flash and Immediate priority emergency messages when the tab is in the background.
 */

import { writable } from 'svelte/store';

// ---------------------------------------------------------------------------
// Store — persisted to localStorage
// ---------------------------------------------------------------------------

const STORAGE_KEY = 'mycelix-push-enabled';

function readStored(): boolean {
  if (typeof window === 'undefined') return false;
  return localStorage.getItem(STORAGE_KEY) === 'true';
}

export const pushEnabled = writable<boolean>(readStored());

pushEnabled.subscribe((v) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, String(v));
  }
});

// ---------------------------------------------------------------------------
// Capability checks
// ---------------------------------------------------------------------------

export function isSupported(): boolean {
  return typeof window !== 'undefined' && 'Notification' in window;
}

export function isEnabled(): boolean {
  return isSupported() && Notification.permission === 'granted';
}

// ---------------------------------------------------------------------------
// Permission request
// ---------------------------------------------------------------------------

export async function requestPermission(): Promise<boolean> {
  if (!isSupported()) return false;
  const result = await Notification.requestPermission();
  const granted = result === 'granted';
  pushEnabled.set(granted);
  return granted;
}

// ---------------------------------------------------------------------------
// Show a generic notification
// ---------------------------------------------------------------------------

export function showNotification(
  title: string,
  body: string,
  options?: NotificationOptions & { tag?: string },
): Notification | null {
  if (!isEnabled()) return null;
  return new Notification(title, {
    body,
    icon: '/icon-192.png',
    badge: '/favicon.png',
    ...options,
  });
}

// ---------------------------------------------------------------------------
// Emergency-specific notification
// ---------------------------------------------------------------------------

interface EmergencyPayload {
  content: string;
  priority: string;
  sender_did: string;
}

export function notifyEmergency(message: EmergencyPayload): Notification | null {
  if (message.priority !== 'Flash' && message.priority !== 'Immediate') {
    return null;
  }

  const title =
    message.priority === 'Flash' ? '[FLASH] Emergency' : '[IMMEDIATE] Alert';

  const notification = showNotification(title, message.content, {
    tag: 'emergency-' + Date.now(),
    requireInteraction: true,
  });

  if ('vibrate' in navigator) {
    navigator.vibrate(message.priority === 'Flash' ? [200, 100, 200, 100, 400] : [200, 100, 200]);
  }

  return notification;
}
