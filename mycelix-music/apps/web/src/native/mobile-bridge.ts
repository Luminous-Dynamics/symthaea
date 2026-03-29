// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile Native Bridge
 *
 * Bridges between web app and native capabilities:
 * - Native audio engine
 * - Media controls
 * - Background playback
 * - Offline storage
 * - Push notifications
 * - Haptic feedback
 * - Share sheet
 * - File system
 */

import { Capacitor } from '@capacitor/core';
import { Preferences } from '@capacitor/preferences';
import { Haptics, ImpactStyle, NotificationType } from '@capacitor/haptics';
import { Share } from '@capacitor/share';
import { Filesystem, Directory, Encoding } from '@capacitor/filesystem';
import { LocalNotifications } from '@capacitor/local-notifications';
import { PushNotifications } from '@capacitor/push-notifications';
import { App } from '@capacitor/app';
import { StatusBar, Style } from '@capacitor/status-bar';
import { Keyboard } from '@capacitor/keyboard';
import { Network } from '@capacitor/network';

// ==================== Types ====================

export interface AudioTrackInfo {
  id: string;
  title: string;
  artist: string;
  album?: string;
  artworkUrl?: string;
  duration: number;
}

export interface DownloadProgress {
  trackId: string;
  bytesDownloaded: number;
  totalBytes: number;
  progress: number;
}

export interface NotificationAction {
  id: string;
  title: string;
  icon?: string;
}

export interface NetworkStatus {
  connected: boolean;
  connectionType: 'wifi' | 'cellular' | 'none' | 'unknown';
}

// ==================== Platform Detection ====================

export const isNative = Capacitor.isNativePlatform();
export const platform = Capacitor.getPlatform();
export const isIOS = platform === 'ios';
export const isAndroid = platform === 'android';
export const isWeb = platform === 'web';

// ==================== Native Audio Engine ====================

class NativeAudioEngine {
  private audioContext: AudioContext | null = null;
  private currentTrack: AudioTrackInfo | null = null;
  private mediaSession: MediaSession | null = null;

  async initialize(): Promise<void> {
    if (isWeb) {
      this.audioContext = new AudioContext();
    }

    // Setup media session for lock screen controls
    if ('mediaSession' in navigator) {
      this.mediaSession = navigator.mediaSession;
      this.setupMediaSession();
    }
  }

  private setupMediaSession(): void {
    if (!this.mediaSession) return;

    this.mediaSession.setActionHandler('play', () => {
      window.dispatchEvent(new CustomEvent('native-audio:play'));
    });

    this.mediaSession.setActionHandler('pause', () => {
      window.dispatchEvent(new CustomEvent('native-audio:pause'));
    });

    this.mediaSession.setActionHandler('previoustrack', () => {
      window.dispatchEvent(new CustomEvent('native-audio:previous'));
    });

    this.mediaSession.setActionHandler('nexttrack', () => {
      window.dispatchEvent(new CustomEvent('native-audio:next'));
    });

    this.mediaSession.setActionHandler('seekto', (details) => {
      window.dispatchEvent(new CustomEvent('native-audio:seek', {
        detail: { position: details.seekTime },
      }));
    });
  }

  updateNowPlaying(track: AudioTrackInfo): void {
    this.currentTrack = track;

    if (this.mediaSession) {
      this.mediaSession.metadata = new MediaMetadata({
        title: track.title,
        artist: track.artist,
        album: track.album || '',
        artwork: track.artworkUrl ? [
          { src: track.artworkUrl, sizes: '96x96', type: 'image/png' },
          { src: track.artworkUrl, sizes: '256x256', type: 'image/png' },
          { src: track.artworkUrl, sizes: '512x512', type: 'image/png' },
        ] : [],
      });
    }
  }

  updatePlaybackState(state: 'playing' | 'paused' | 'none'): void {
    if (this.mediaSession) {
      this.mediaSession.playbackState = state;
    }
  }

  updatePosition(position: number, duration: number): void {
    if (this.mediaSession) {
      this.mediaSession.setPositionState({
        duration,
        playbackRate: 1,
        position,
      });
    }
  }
}

export const nativeAudio = new NativeAudioEngine();

// ==================== Haptic Feedback ====================

export const haptics = {
  async impact(style: 'light' | 'medium' | 'heavy' = 'medium'): Promise<void> {
    if (!isNative) return;

    const styleMap = {
      light: ImpactStyle.Light,
      medium: ImpactStyle.Medium,
      heavy: ImpactStyle.Heavy,
    };

    await Haptics.impact({ style: styleMap[style] });
  },

  async notification(type: 'success' | 'warning' | 'error' = 'success'): Promise<void> {
    if (!isNative) return;

    const typeMap = {
      success: NotificationType.Success,
      warning: NotificationType.Warning,
      error: NotificationType.Error,
    };

    await Haptics.notification({ type: typeMap[type] });
  },

  async vibrate(): Promise<void> {
    if (!isNative) return;
    await Haptics.vibrate();
  },

  async selectionStart(): Promise<void> {
    if (!isNative) return;
    await Haptics.selectionStart();
  },

  async selectionChanged(): Promise<void> {
    if (!isNative) return;
    await Haptics.selectionChanged();
  },

  async selectionEnd(): Promise<void> {
    if (!isNative) return;
    await Haptics.selectionEnd();
  },
};

// ==================== Storage ====================

export const storage = {
  async get<T>(key: string): Promise<T | null> {
    if (isNative) {
      const { value } = await Preferences.get({ key });
      return value ? JSON.parse(value) : null;
    }
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : null;
  },

  async set<T>(key: string, value: T): Promise<void> {
    const stringValue = JSON.stringify(value);
    if (isNative) {
      await Preferences.set({ key, value: stringValue });
    } else {
      localStorage.setItem(key, stringValue);
    }
  },

  async remove(key: string): Promise<void> {
    if (isNative) {
      await Preferences.remove({ key });
    } else {
      localStorage.removeItem(key);
    }
  },

  async clear(): Promise<void> {
    if (isNative) {
      await Preferences.clear();
    } else {
      localStorage.clear();
    }
  },

  async keys(): Promise<string[]> {
    if (isNative) {
      const { keys } = await Preferences.keys();
      return keys;
    }
    return Object.keys(localStorage);
  },
};

// ==================== Offline Storage (Files) ====================

export const offlineStorage = {
  async downloadTrack(
    trackId: string,
    audioUrl: string,
    onProgress?: (progress: DownloadProgress) => void
  ): Promise<string> {
    const fileName = `track_${trackId}.mp3`;

    if (isNative) {
      // Download using native file system
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      const base64 = await blobToBase64(blob);

      await Filesystem.writeFile({
        path: `downloads/${fileName}`,
        data: base64,
        directory: Directory.Data,
      });

      const fileInfo = await Filesystem.stat({
        path: `downloads/${fileName}`,
        directory: Directory.Data,
      });

      return fileInfo.uri;
    } else {
      // Use IndexedDB for web
      const response = await fetch(audioUrl);
      const blob = await response.blob();

      const db = await openOfflineDB();
      await storeInIndexedDB(db, 'tracks', trackId, blob);

      return URL.createObjectURL(blob);
    }
  },

  async getTrack(trackId: string): Promise<Blob | null> {
    if (isNative) {
      try {
        const result = await Filesystem.readFile({
          path: `downloads/track_${trackId}.mp3`,
          directory: Directory.Data,
        });

        return base64ToBlob(result.data as string, 'audio/mpeg');
      } catch {
        return null;
      }
    } else {
      const db = await openOfflineDB();
      return getFromIndexedDB(db, 'tracks', trackId);
    }
  },

  async deleteTrack(trackId: string): Promise<void> {
    if (isNative) {
      await Filesystem.deleteFile({
        path: `downloads/track_${trackId}.mp3`,
        directory: Directory.Data,
      });
    } else {
      const db = await openOfflineDB();
      await deleteFromIndexedDB(db, 'tracks', trackId);
    }
  },

  async getDownloadedTrackIds(): Promise<string[]> {
    if (isNative) {
      try {
        const result = await Filesystem.readdir({
          path: 'downloads',
          directory: Directory.Data,
        });

        return result.files
          .filter(f => f.name.startsWith('track_') && f.name.endsWith('.mp3'))
          .map(f => f.name.replace('track_', '').replace('.mp3', ''));
      } catch {
        return [];
      }
    } else {
      const db = await openOfflineDB();
      return getAllKeysFromIndexedDB(db, 'tracks');
    }
  },

  async getStorageUsed(): Promise<number> {
    if (isNative) {
      try {
        const result = await Filesystem.readdir({
          path: 'downloads',
          directory: Directory.Data,
        });

        let total = 0;
        for (const file of result.files) {
          const stat = await Filesystem.stat({
            path: `downloads/${file.name}`,
            directory: Directory.Data,
          });
          total += stat.size;
        }
        return total;
      } catch {
        return 0;
      }
    } else {
      const estimate = await navigator.storage?.estimate?.();
      return estimate?.usage || 0;
    }
  },
};

// ==================== IndexedDB Helpers ====================

async function openOfflineDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-offline', 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('tracks')) {
        db.createObjectStore('tracks');
      }
    };
  });
}

async function storeInIndexedDB(
  db: IDBDatabase,
  store: string,
  key: string,
  value: Blob
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const objectStore = tx.objectStore(store);
    const request = objectStore.put(value, key);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

async function getFromIndexedDB(
  db: IDBDatabase,
  store: string,
  key: string
): Promise<Blob | null> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const objectStore = tx.objectStore(store);
    const request = objectStore.get(key);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || null);
  });
}

async function deleteFromIndexedDB(
  db: IDBDatabase,
  store: string,
  key: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const objectStore = tx.objectStore(store);
    const request = objectStore.delete(key);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

async function getAllKeysFromIndexedDB(
  db: IDBDatabase,
  store: string
): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const objectStore = tx.objectStore(store);
    const request = objectStore.getAllKeys();
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result as string[]);
  });
}

// ==================== Blob Helpers ====================

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = (reader.result as string).split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function base64ToBlob(base64: string, mimeType: string): Blob {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

// ==================== Share ====================

export const share = {
  async shareTrack(track: AudioTrackInfo): Promise<void> {
    const shareData = {
      title: track.title,
      text: `Listen to ${track.title} by ${track.artist} on Mycelix Music`,
      url: `https://mycelix.music/track/${track.id}`,
    };

    if (isNative) {
      await Share.share(shareData);
    } else if (navigator.share) {
      await navigator.share(shareData);
    }
  },

  async sharePlaylist(playlistId: string, playlistName: string): Promise<void> {
    await Share.share({
      title: playlistName,
      text: `Check out this playlist on Mycelix Music`,
      url: `https://mycelix.music/playlist/${playlistId}`,
    });
  },
};

// ==================== Notifications ====================

export const notifications = {
  async requestPermission(): Promise<boolean> {
    if (isNative) {
      const result = await PushNotifications.requestPermissions();
      return result.receive === 'granted';
    }
    const result = await Notification.requestPermission();
    return result === 'granted';
  },

  async showLocal(options: {
    title: string;
    body: string;
    id?: number;
    actions?: NotificationAction[];
  }): Promise<void> {
    if (isNative) {
      await LocalNotifications.schedule({
        notifications: [{
          id: options.id || Date.now(),
          title: options.title,
          body: options.body,
          schedule: { at: new Date(Date.now()) },
          actionTypeId: options.actions ? 'MEDIA_CONTROLS' : undefined,
        }],
      });
    } else {
      new Notification(options.title, { body: options.body });
    }
  },

  async registerPush(): Promise<string | null> {
    if (!isNative) return null;

    await PushNotifications.register();

    return new Promise((resolve) => {
      PushNotifications.addListener('registration', (token) => {
        resolve(token.value);
      });

      PushNotifications.addListener('registrationError', () => {
        resolve(null);
      });
    });
  },

  onPushReceived(callback: (notification: { title: string; body: string; data: unknown }) => void): void {
    if (!isNative) return;

    PushNotifications.addListener('pushNotificationReceived', (notification) => {
      callback({
        title: notification.title || '',
        body: notification.body || '',
        data: notification.data,
      });
    });
  },
};

// ==================== Network ====================

export const network = {
  async getStatus(): Promise<NetworkStatus> {
    if (isNative) {
      const status = await Network.getStatus();
      return {
        connected: status.connected,
        connectionType: status.connectionType as NetworkStatus['connectionType'],
      };
    }
    return {
      connected: navigator.onLine,
      connectionType: 'unknown',
    };
  },

  onStatusChange(callback: (status: NetworkStatus) => void): () => void {
    if (isNative) {
      const handler = Network.addListener('networkStatusChange', (status) => {
        callback({
          connected: status.connected,
          connectionType: status.connectionType as NetworkStatus['connectionType'],
        });
      });
      return () => handler.remove();
    } else {
      const onlineHandler = () => callback({ connected: true, connectionType: 'unknown' });
      const offlineHandler = () => callback({ connected: false, connectionType: 'none' });

      window.addEventListener('online', onlineHandler);
      window.addEventListener('offline', offlineHandler);

      return () => {
        window.removeEventListener('online', onlineHandler);
        window.removeEventListener('offline', offlineHandler);
      };
    }
  },
};

// ==================== App Lifecycle ====================

export const appLifecycle = {
  onStateChange(callback: (state: 'active' | 'inactive' | 'background') => void): () => void {
    if (isNative) {
      const handler = App.addListener('appStateChange', ({ isActive }) => {
        callback(isActive ? 'active' : 'background');
      });
      return () => handler.remove();
    } else {
      const handler = () => {
        callback(document.visibilityState === 'visible' ? 'active' : 'inactive');
      };
      document.addEventListener('visibilitychange', handler);
      return () => document.removeEventListener('visibilitychange', handler);
    }
  },

  async exitApp(): Promise<void> {
    if (isNative) {
      await App.exitApp();
    }
  },

  async getInfo(): Promise<{ name: string; version: string; build: string }> {
    if (isNative) {
      const info = await App.getInfo();
      return { name: info.name, version: info.version, build: info.build };
    }
    return { name: 'Mycelix Music', version: '1.0.0', build: '1' };
  },
};

// ==================== Status Bar ====================

export const statusBar = {
  async setStyle(style: 'dark' | 'light'): Promise<void> {
    if (!isNative) return;
    await StatusBar.setStyle({ style: style === 'dark' ? Style.Dark : Style.Light });
  },

  async setBackgroundColor(color: string): Promise<void> {
    if (!isNative) return;
    await StatusBar.setBackgroundColor({ color });
  },

  async show(): Promise<void> {
    if (!isNative) return;
    await StatusBar.show();
  },

  async hide(): Promise<void> {
    if (!isNative) return;
    await StatusBar.hide();
  },
};

// ==================== Keyboard ====================

export const keyboard = {
  onShow(callback: (height: number) => void): () => void {
    if (!isNative) return () => {};

    const handler = Keyboard.addListener('keyboardWillShow', (info) => {
      callback(info.keyboardHeight);
    });
    return () => handler.remove();
  },

  onHide(callback: () => void): () => void {
    if (!isNative) return () => {};

    const handler = Keyboard.addListener('keyboardWillHide', callback);
    return () => handler.remove();
  },

  async hide(): Promise<void> {
    if (!isNative) return;
    await Keyboard.hide();
  },
};

// ==================== Initialize ====================

export async function initializeNativeBridge(): Promise<void> {
  if (!isNative) return;

  await nativeAudio.initialize();

  // Set status bar style
  await statusBar.setStyle('dark');
  await statusBar.setBackgroundColor('#0F0F0F');

  // Request notification permission
  await notifications.requestPermission();

  console.log(`Native bridge initialized for ${platform}`);
}

export default {
  isNative,
  platform,
  isIOS,
  isAndroid,
  isWeb,
  nativeAudio,
  haptics,
  storage,
  offlineStorage,
  share,
  notifications,
  network,
  appLifecycle,
  statusBar,
  keyboard,
  initializeNativeBridge,
};
