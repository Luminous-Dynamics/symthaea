// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * PWA Module - Service Worker Registration and Management
 */

export interface PWAConfig {
  swUrl?: string;
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onOffline?: () => void;
  onOnline?: () => void;
}

export class PWAManager {
  private registration: ServiceWorkerRegistration | null = null;
  private config: PWAConfig;

  constructor(config: PWAConfig = {}) {
    this.config = {
      swUrl: '/sw.js',
      ...config,
    };
  }

  async register(): Promise<ServiceWorkerRegistration | null> {
    if (!('serviceWorker' in navigator)) {
      console.warn('[PWA] Service workers not supported');
      return null;
    }

    try {
      this.registration = await navigator.serviceWorker.register(
        this.config.swUrl!,
        { scope: '/' }
      );

      // Check for updates
      this.registration.addEventListener('updatefound', () => {
        const newWorker = this.registration!.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              this.config.onUpdate?.(this.registration!);
            } else if (newWorker.state === 'activated') {
              this.config.onSuccess?.(this.registration!);
            }
          });
        }
      });

      // Listen for online/offline
      window.addEventListener('online', () => this.config.onOnline?.());
      window.addEventListener('offline', () => this.config.onOffline?.());

      console.log('[PWA] Service worker registered');
      return this.registration;
    } catch (error) {
      console.error('[PWA] Registration failed:', error);
      return null;
    }
  }

  async unregister(): Promise<boolean> {
    if (this.registration) {
      return this.registration.unregister();
    }
    return false;
  }

  async update(): Promise<void> {
    if (this.registration) {
      await this.registration.update();
    }
  }

  skipWaiting(): void {
    if (this.registration?.waiting) {
      this.registration.waiting.postMessage({ type: 'SKIP_WAITING' });
    }
  }

  async requestNotificationPermission(): Promise<NotificationPermission> {
    if (!('Notification' in window)) {
      return 'denied';
    }
    return Notification.requestPermission();
  }

  async subscribeToPush(vapidPublicKey: string): Promise<PushSubscription | null> {
    if (!this.registration) return null;

    try {
      const subscription = await this.registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: this.urlBase64ToUint8Array(vapidPublicKey),
      });
      return subscription;
    } catch (error) {
      console.error('[PWA] Push subscription failed:', error);
      return null;
    }
  }

  async registerBackgroundSync(tag: string): Promise<boolean> {
    if (!this.registration || !('sync' in this.registration)) {
      return false;
    }

    try {
      await (this.registration as any).sync.register(tag);
      return true;
    } catch {
      return false;
    }
  }

  async registerPeriodicSync(tag: string, minInterval: number): Promise<boolean> {
    if (!this.registration) return false;

    try {
      const status = await navigator.permissions.query({
        name: 'periodic-background-sync' as PermissionName,
      });

      if (status.state === 'granted') {
        await (this.registration as any).periodicSync.register(tag, { minInterval });
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  async getCacheStatus(): Promise<Record<string, number>> {
    return new Promise((resolve) => {
      if (!navigator.serviceWorker.controller) {
        resolve({});
        return;
      }

      const channel = new MessageChannel();
      channel.port1.onmessage = (event) => resolve(event.data);

      navigator.serviceWorker.controller.postMessage(
        { type: 'GET_CACHE_STATUS' },
        [channel.port2]
      );
    });
  }

  async clearCache(): Promise<void> {
    navigator.serviceWorker.controller?.postMessage({ type: 'CLEAR_CACHE' });
  }

  async queueEmail(email: any): Promise<void> {
    navigator.serviceWorker.controller?.postMessage({
      type: 'QUEUE_EMAIL',
      payload: email,
    });
  }

  isOnline(): boolean {
    return navigator.onLine;
  }

  private urlBase64ToUint8Array(base64String: string): Uint8Array {
    const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }
}

export const pwaManager = new PWAManager();
export default PWAManager;
