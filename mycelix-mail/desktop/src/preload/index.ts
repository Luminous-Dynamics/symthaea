/**
 * Mycelix Mail Desktop - Preload Script
 *
 * Exposes safe APIs to the renderer process
 */

import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Store
  store: {
    get: (key: string) => ipcRenderer.invoke('store:get', key),
    set: (key: string, value: unknown) => ipcRenderer.invoke('store:set', key, value),
  },

  // Theme
  theme: {
    get: () => ipcRenderer.invoke('theme:get'),
    set: (theme: 'system' | 'light' | 'dark') => ipcRenderer.invoke('theme:set', theme),
  },

  // Window controls
  window: {
    minimize: () => ipcRenderer.send('window:minimize'),
    maximize: () => ipcRenderer.send('window:maximize'),
    close: () => ipcRenderer.send('window:close'),
  },

  // Notifications
  notification: {
    show: (options: { title: string; body: string; data?: unknown }) =>
      ipcRenderer.send('notification:show', options),
  },

  // Badge (macOS dock)
  badge: {
    set: (count: number) => ipcRenderer.send('badge:set', count),
  },

  // Shell
  shell: {
    openExternal: (url: string) => ipcRenderer.send('shell:openExternal', url),
  },

  // Updates
  updater: {
    check: () => ipcRenderer.invoke('updater:check'),
    install: () => ipcRenderer.send('updater:install'),
  },

  // Event listeners
  on: (channel: string, callback: (...args: unknown[]) => void) => {
    const allowedChannels = [
      'navigate',
      'check-emails',
      'update-available',
      'update-downloaded',
      'update-error',
    ];

    if (allowedChannels.includes(channel)) {
      const subscription = (_event: Electron.IpcRendererEvent, ...args: unknown[]) =>
        callback(...args);
      ipcRenderer.on(channel, subscription);

      return () => {
        ipcRenderer.removeListener(channel, subscription);
      };
    }

    return () => {};
  },

  // Platform info
  platform: process.platform,
  isElectron: true,
});

// Type declaration for renderer
declare global {
  interface Window {
    electronAPI: {
      store: {
        get: (key: string) => Promise<unknown>;
        set: (key: string, value: unknown) => Promise<void>;
      };
      theme: {
        get: () => Promise<'system' | 'light' | 'dark'>;
        set: (theme: 'system' | 'light' | 'dark') => Promise<void>;
      };
      window: {
        minimize: () => void;
        maximize: () => void;
        close: () => void;
      };
      notification: {
        show: (options: { title: string; body: string; data?: unknown }) => void;
      };
      badge: {
        set: (count: number) => void;
      };
      shell: {
        openExternal: (url: string) => void;
      };
      updater: {
        check: () => Promise<unknown>;
        install: () => void;
      };
      on: (channel: string, callback: (...args: unknown[]) => void) => () => void;
      platform: NodeJS.Platform;
      isElectron: boolean;
    };
  }
}
