// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Desktop Native Bridge (Tauri)
 *
 * Native desktop capabilities:
 * - System audio routing (ASIO/CoreAudio)
 * - Multi-window workspaces
 * - File system integration
 * - System tray
 * - Global shortcuts
 * - Native menus
 * - Auto-updater
 */

import { invoke } from '@tauri-apps/api/tauri';
import { listen, emit, Event } from '@tauri-apps/api/event';
import { open, save, message, ask, confirm } from '@tauri-apps/api/dialog';
import { readBinaryFile, writeBinaryFile, readDir, createDir, removeFile, exists } from '@tauri-apps/api/fs';
import { appDataDir, audioDir, downloadDir, homeDir } from '@tauri-apps/api/path';
import { WebviewWindow, appWindow, LogicalSize, PhysicalPosition } from '@tauri-apps/api/window';
import { register, unregister, isRegistered } from '@tauri-apps/api/globalShortcut';
import { sendNotification, requestPermission, isPermissionGranted } from '@tauri-apps/api/notification';
import { checkUpdate, installUpdate } from '@tauri-apps/api/updater';
import { exit, relaunch } from '@tauri-apps/api/process';
import { writeText, readText } from '@tauri-apps/api/clipboard';
import { platform, arch, version } from '@tauri-apps/api/os';

// ==================== Types ====================

export interface AudioDevice {
  id: string;
  name: string;
  type: 'input' | 'output';
  sampleRates: number[];
  bufferSizes: number[];
  channels: number;
  isDefault: boolean;
}

export interface AudioConfig {
  inputDevice?: string;
  outputDevice?: string;
  sampleRate: number;
  bufferSize: number;
  bitDepth: 16 | 24 | 32;
}

export interface WindowConfig {
  label: string;
  title: string;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  resizable?: boolean;
  decorations?: boolean;
  alwaysOnTop?: boolean;
  transparent?: boolean;
}

export interface UpdateInfo {
  version: string;
  date: string;
  body: string;
}

export interface SystemInfo {
  platform: string;
  arch: string;
  osVersion: string;
  appVersion: string;
}

// ==================== Platform Detection ====================

export const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

// ==================== System Audio ====================

export const systemAudio = {
  async getDevices(): Promise<AudioDevice[]> {
    if (!isTauri) return [];
    return invoke<AudioDevice[]>('get_audio_devices');
  },

  async getDefaultInput(): Promise<AudioDevice | null> {
    if (!isTauri) return null;
    return invoke<AudioDevice | null>('get_default_input_device');
  },

  async getDefaultOutput(): Promise<AudioDevice | null> {
    if (!isTauri) return null;
    return invoke<AudioDevice | null>('get_default_output_device');
  },

  async setConfig(config: AudioConfig): Promise<void> {
    if (!isTauri) return;
    await invoke('set_audio_config', { config });
  },

  async getConfig(): Promise<AudioConfig | null> {
    if (!isTauri) return null;
    return invoke<AudioConfig>('get_audio_config');
  },

  async startEngine(): Promise<void> {
    if (!isTauri) return;
    await invoke('start_audio_engine');
  },

  async stopEngine(): Promise<void> {
    if (!isTauri) return;
    await invoke('stop_audio_engine');
  },

  async getLatency(): Promise<number> {
    if (!isTauri) return 0;
    return invoke<number>('get_audio_latency');
  },

  onDeviceChange(callback: (devices: AudioDevice[]) => void): () => void {
    if (!isTauri) return () => {};

    const unlisten = listen<AudioDevice[]>('audio-devices-changed', (event) => {
      callback(event.payload);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  },
};

// ==================== Multi-Window ====================

export const windows = {
  async create(config: WindowConfig): Promise<WebviewWindow> {
    const webview = new WebviewWindow(config.label, {
      title: config.title,
      width: config.width || 800,
      height: config.height || 600,
      x: config.x,
      y: config.y,
      resizable: config.resizable ?? true,
      decorations: config.decorations ?? true,
      alwaysOnTop: config.alwaysOnTop ?? false,
      transparent: config.transparent ?? false,
      center: config.x === undefined && config.y === undefined,
    });

    return webview;
  },

  async getAll(): Promise<WebviewWindow[]> {
    if (!isTauri) return [];
    const windows = await WebviewWindow.getAll();
    return windows;
  },

  async getCurrent(): Promise<WebviewWindow> {
    return appWindow;
  },

  async setSize(width: number, height: number): Promise<void> {
    if (!isTauri) return;
    await appWindow.setSize(new LogicalSize(width, height));
  },

  async setPosition(x: number, y: number): Promise<void> {
    if (!isTauri) return;
    await appWindow.setPosition(new PhysicalPosition(x, y));
  },

  async setTitle(title: string): Promise<void> {
    if (!isTauri) return;
    await appWindow.setTitle(title);
  },

  async setFullscreen(fullscreen: boolean): Promise<void> {
    if (!isTauri) return;
    await appWindow.setFullscreen(fullscreen);
  },

  async isFullscreen(): Promise<boolean> {
    if (!isTauri) return false;
    return appWindow.isFullscreen();
  },

  async minimize(): Promise<void> {
    if (!isTauri) return;
    await appWindow.minimize();
  },

  async maximize(): Promise<void> {
    if (!isTauri) return;
    await appWindow.maximize();
  },

  async unmaximize(): Promise<void> {
    if (!isTauri) return;
    await appWindow.unmaximize();
  },

  async close(): Promise<void> {
    if (!isTauri) return;
    await appWindow.close();
  },

  async hide(): Promise<void> {
    if (!isTauri) return;
    await appWindow.hide();
  },

  async show(): Promise<void> {
    if (!isTauri) return;
    await appWindow.show();
  },

  async setAlwaysOnTop(alwaysOnTop: boolean): Promise<void> {
    if (!isTauri) return;
    await appWindow.setAlwaysOnTop(alwaysOnTop);
  },

  onCloseRequested(callback: () => boolean | Promise<boolean>): () => void {
    if (!isTauri) return () => {};

    const unlisten = appWindow.onCloseRequested(async (event) => {
      const shouldClose = await callback();
      if (!shouldClose) {
        event.preventDefault();
      }
    });

    return () => {
      unlisten.then(fn => fn());
    };
  },
};

// ==================== File System ====================

export const fileSystem = {
  async openFileDialog(options?: {
    filters?: { name: string; extensions: string[] }[];
    multiple?: boolean;
    directory?: boolean;
    defaultPath?: string;
  }): Promise<string | string[] | null> {
    if (!isTauri) return null;

    return open({
      filters: options?.filters,
      multiple: options?.multiple,
      directory: options?.directory,
      defaultPath: options?.defaultPath,
    });
  },

  async saveFileDialog(options?: {
    filters?: { name: string; extensions: string[] }[];
    defaultPath?: string;
  }): Promise<string | null> {
    if (!isTauri) return null;

    return save({
      filters: options?.filters,
      defaultPath: options?.defaultPath,
    });
  },

  async readFile(path: string): Promise<Uint8Array> {
    if (!isTauri) throw new Error('Not running in Tauri');
    return readBinaryFile(path);
  },

  async writeFile(path: string, data: Uint8Array): Promise<void> {
    if (!isTauri) return;
    await writeBinaryFile(path, data);
  },

  async readDirectory(path: string): Promise<{ name: string; isDir: boolean }[]> {
    if (!isTauri) return [];
    const entries = await readDir(path);
    return entries.map(e => ({
      name: e.name || '',
      isDir: e.children !== undefined,
    }));
  },

  async createDirectory(path: string): Promise<void> {
    if (!isTauri) return;
    await createDir(path, { recursive: true });
  },

  async deleteFile(path: string): Promise<void> {
    if (!isTauri) return;
    await removeFile(path);
  },

  async exists(path: string): Promise<boolean> {
    if (!isTauri) return false;
    return exists(path);
  },

  async getAppDataDir(): Promise<string> {
    if (!isTauri) return '';
    return appDataDir();
  },

  async getAudioDir(): Promise<string> {
    if (!isTauri) return '';
    return audioDir();
  },

  async getDownloadDir(): Promise<string> {
    if (!isTauri) return '';
    return downloadDir();
  },

  async getHomeDir(): Promise<string> {
    if (!isTauri) return '';
    return homeDir();
  },
};

// ==================== Global Shortcuts ====================

export const shortcuts = {
  async register(shortcut: string, callback: () => void): Promise<void> {
    if (!isTauri) return;
    await register(shortcut, callback);
  },

  async unregister(shortcut: string): Promise<void> {
    if (!isTauri) return;
    await unregister(shortcut);
  },

  async isRegistered(shortcut: string): Promise<boolean> {
    if (!isTauri) return false;
    return isRegistered(shortcut);
  },

  async registerMediaKeys(callbacks: {
    onPlayPause?: () => void;
    onNext?: () => void;
    onPrevious?: () => void;
    onStop?: () => void;
  }): Promise<void> {
    if (!isTauri) return;

    if (callbacks.onPlayPause) {
      await register('MediaPlayPause', callbacks.onPlayPause);
    }
    if (callbacks.onNext) {
      await register('MediaNextTrack', callbacks.onNext);
    }
    if (callbacks.onPrevious) {
      await register('MediaPreviousTrack', callbacks.onPrevious);
    }
    if (callbacks.onStop) {
      await register('MediaStop', callbacks.onStop);
    }
  },
};

// ==================== Dialogs ====================

export const dialogs = {
  async message(msg: string, options?: { title?: string; type?: 'info' | 'warning' | 'error' }): Promise<void> {
    if (!isTauri) {
      alert(msg);
      return;
    }
    await message(msg, { title: options?.title, type: options?.type });
  },

  async ask(msg: string, options?: { title?: string; type?: 'info' | 'warning' | 'error' }): Promise<boolean> {
    if (!isTauri) {
      return window.confirm(msg);
    }
    return ask(msg, { title: options?.title, type: options?.type });
  },

  async confirm(msg: string, options?: { title?: string; type?: 'info' | 'warning' | 'error' }): Promise<boolean> {
    if (!isTauri) {
      return window.confirm(msg);
    }
    return confirm(msg, { title: options?.title, type: options?.type });
  },
};

// ==================== Notifications ====================

export const notifications = {
  async requestPermission(): Promise<boolean> {
    if (!isTauri) {
      const result = await Notification.requestPermission();
      return result === 'granted';
    }
    return requestPermission();
  },

  async isPermissionGranted(): Promise<boolean> {
    if (!isTauri) return Notification.permission === 'granted';
    return isPermissionGranted();
  },

  async send(title: string, body?: string): Promise<void> {
    if (!isTauri) {
      new Notification(title, { body });
      return;
    }
    sendNotification({ title, body });
  },
};

// ==================== Clipboard ====================

export const clipboard = {
  async writeText(text: string): Promise<void> {
    if (!isTauri) {
      await navigator.clipboard.writeText(text);
      return;
    }
    await writeText(text);
  },

  async readText(): Promise<string> {
    if (!isTauri) {
      return navigator.clipboard.readText();
    }
    return readText();
  },
};

// ==================== Updates ====================

export const updater = {
  async check(): Promise<UpdateInfo | null> {
    if (!isTauri) return null;

    try {
      const { shouldUpdate, manifest } = await checkUpdate();
      if (shouldUpdate && manifest) {
        return {
          version: manifest.version,
          date: manifest.date,
          body: manifest.body,
        };
      }
      return null;
    } catch {
      return null;
    }
  },

  async install(): Promise<void> {
    if (!isTauri) return;
    await installUpdate();
  },

  onUpdateAvailable(callback: (info: UpdateInfo) => void): () => void {
    if (!isTauri) return () => {};

    const unlisten = listen<UpdateInfo>('tauri://update-available', (event) => {
      callback(event.payload);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  },
};

// ==================== System Info ====================

export const system = {
  async getInfo(): Promise<SystemInfo> {
    if (!isTauri) {
      return {
        platform: 'web',
        arch: 'unknown',
        osVersion: navigator.userAgent,
        appVersion: '1.0.0',
      };
    }

    const [platformName, archName, osVersion] = await Promise.all([
      platform(),
      arch(),
      version(),
    ]);

    return {
      platform: platformName,
      arch: archName,
      osVersion,
      appVersion: '1.0.0', // Would come from package
    };
  },

  async exit(code?: number): Promise<void> {
    if (!isTauri) return;
    await exit(code || 0);
  },

  async relaunch(): Promise<void> {
    if (!isTauri) {
      location.reload();
      return;
    }
    await relaunch();
  },
};

// ==================== IPC Events ====================

export const ipc = {
  async emit<T>(event: string, payload?: T): Promise<void> {
    if (!isTauri) return;
    await emit(event, payload);
  },

  listen<T>(event: string, callback: (payload: T) => void): () => void {
    if (!isTauri) return () => {};

    const unlisten = listen<T>(event, (e: Event<T>) => {
      callback(e.payload);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  },

  async invoke<T, R>(command: string, args?: T): Promise<R> {
    if (!isTauri) throw new Error('Not running in Tauri');
    return invoke<R>(command, args as Record<string, unknown>);
  },
};

// ==================== Initialize ====================

export async function initializeDesktopBridge(): Promise<void> {
  if (!isTauri) return;

  // Register media keys
  await shortcuts.registerMediaKeys({
    onPlayPause: () => window.dispatchEvent(new CustomEvent('desktop:play-pause')),
    onNext: () => window.dispatchEvent(new CustomEvent('desktop:next')),
    onPrevious: () => window.dispatchEvent(new CustomEvent('desktop:previous')),
    onStop: () => window.dispatchEvent(new CustomEvent('desktop:stop')),
  });

  // Handle window close
  windows.onCloseRequested(async () => {
    // Check for unsaved changes
    const hasUnsaved = window.dispatchEvent(new CustomEvent('desktop:check-unsaved'));
    if (hasUnsaved) {
      const shouldClose = await dialogs.confirm(
        'You have unsaved changes. Are you sure you want to quit?',
        { title: 'Unsaved Changes', type: 'warning' }
      );
      return shouldClose;
    }
    return true;
  });

  // Check for updates
  const update = await updater.check();
  if (update) {
    const shouldUpdate = await dialogs.ask(
      `Version ${update.version} is available. Would you like to update now?`,
      { title: 'Update Available' }
    );
    if (shouldUpdate) {
      await updater.install();
    }
  }

  console.log('Desktop bridge initialized');
}

export default {
  isTauri,
  systemAudio,
  windows,
  fileSystem,
  shortcuts,
  dialogs,
  notifications,
  clipboard,
  updater,
  system,
  ipc,
  initializeDesktopBridge,
};
