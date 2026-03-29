// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Plugin SDK
 *
 * TypeScript SDK for building frontend plugins
 */

// ============================================================================
// Plugin Types
// ============================================================================

export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  description: string;
  author: string;
  license: string;
  homepage?: string;
  repository?: string;
  capabilities: PluginCapability[];
  permissions: PluginPermission[];
  settingsSchema?: JSONSchema;
  dependencies?: PluginDependency[];
  entrypoints: {
    main?: string;
    settings?: string;
    sidebar?: string;
  };
}

export type PluginCapability =
  | 'email-processing'
  | 'integration'
  | 'trust'
  | 'ui'
  | 'storage'
  | 'notifications'
  | 'scheduling';

export type PluginPermission =
  | 'read:emails'
  | 'write:emails'
  | 'read:contacts'
  | 'write:contacts'
  | 'read:calendar'
  | 'write:calendar'
  | 'read:trust'
  | 'write:trust'
  | 'network'
  | 'storage'
  | 'notifications';

export interface PluginDependency {
  pluginId: string;
  versionRequirement: string;
}

export interface JSONSchema {
  type: string;
  properties?: Record<string, JSONSchema>;
  required?: string[];
  [key: string]: unknown;
}

// ============================================================================
// Plugin Context
// ============================================================================

export interface PluginContext {
  /** Plugin manifest */
  manifest: PluginManifest;

  /** Plugin settings */
  settings: Record<string, unknown>;

  /** API client */
  api: PluginAPI;

  /** Storage interface */
  storage: PluginStorage;

  /** Event emitter */
  events: PluginEvents;

  /** UI utilities */
  ui: PluginUI;

  /** Logger */
  logger: PluginLogger;
}

export interface PluginAPI {
  /** Make authenticated API request */
  fetch(path: string, options?: RequestInit): Promise<Response>;

  /** Get current user */
  getCurrentUser(): Promise<User>;

  /** Get emails */
  getEmails(options?: EmailQueryOptions): Promise<Email[]>;

  /** Get contacts */
  getContacts(options?: ContactQueryOptions): Promise<Contact[]>;

  /** Get trust score */
  getTrustScore(userId: string): Promise<TrustScore>;
}

export interface PluginStorage {
  /** Get value */
  get<T>(key: string): Promise<T | null>;

  /** Set value */
  set<T>(key: string, value: T): Promise<void>;

  /** Remove value */
  remove(key: string): Promise<void>;

  /** List keys */
  keys(): Promise<string[]>;

  /** Clear all storage */
  clear(): Promise<void>;
}

export interface PluginEvents {
  /** Subscribe to event */
  on<T>(event: PluginEvent, handler: (data: T) => void): () => void;

  /** Emit event */
  emit<T>(event: PluginEvent, data: T): void;
}

export type PluginEvent =
  | 'email:received'
  | 'email:sent'
  | 'email:opened'
  | 'contact:added'
  | 'contact:updated'
  | 'trust:updated'
  | 'settings:changed';

export interface PluginUI {
  /** Show notification */
  showNotification(options: NotificationOptions): void;

  /** Show modal */
  showModal(options: ModalOptions): Promise<unknown>;

  /** Show toast */
  showToast(message: string, type?: 'success' | 'error' | 'info' | 'warning'): void;

  /** Register sidebar item */
  registerSidebarItem(item: SidebarItem): void;

  /** Register toolbar action */
  registerToolbarAction(action: ToolbarAction): void;

  /** Register email action */
  registerEmailAction(action: EmailAction): void;

  /** Register compose action */
  registerComposeAction(action: ComposeAction): void;
}

export interface PluginLogger {
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

// ============================================================================
// Data Types
// ============================================================================

export interface User {
  id: string;
  email: string;
  name?: string;
  avatar?: string;
}

export interface Email {
  id: string;
  threadId: string;
  from: EmailAddress;
  to: EmailAddress[];
  cc: EmailAddress[];
  subject: string;
  bodyText: string;
  bodyHtml?: string;
  receivedAt: string;
  labels: string[];
  read: boolean;
  starred: boolean;
  attachments: Attachment[];
}

export interface EmailAddress {
  email: string;
  name?: string;
}

export interface Attachment {
  id: string;
  filename: string;
  contentType: string;
  size: number;
}

export interface Contact {
  id: string;
  email: string;
  name?: string;
  trustScore: number;
  tags: string[];
}

export interface TrustScore {
  score: number;
  attestations: Attestation[];
  sources: string[];
}

export interface Attestation {
  from: string;
  level: number;
  context: string;
  createdAt: string;
}

export interface EmailQueryOptions {
  folder?: string;
  labels?: string[];
  from?: string;
  to?: string;
  subject?: string;
  after?: string;
  before?: string;
  limit?: number;
  offset?: number;
}

export interface ContactQueryOptions {
  search?: string;
  tags?: string[];
  minTrust?: number;
  limit?: number;
  offset?: number;
}

// ============================================================================
// UI Types
// ============================================================================

export interface NotificationOptions {
  title: string;
  body: string;
  icon?: string;
  actions?: NotificationAction[];
  persistent?: boolean;
}

export interface NotificationAction {
  label: string;
  action: string;
}

export interface ModalOptions {
  title: string;
  content: React.ReactNode;
  size?: 'small' | 'medium' | 'large';
  actions?: ModalAction[];
}

export interface ModalAction {
  label: string;
  variant?: 'primary' | 'secondary' | 'danger';
  onClick: () => void | Promise<void>;
}

export interface SidebarItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  route: string;
  badge?: string | number;
  order?: number;
}

export interface ToolbarAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  tooltip?: string;
}

export interface EmailAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  onClick: (email: Email) => void;
  shouldShow?: (email: Email) => boolean;
}

export interface ComposeAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  onClick: (draft: Draft) => void;
}

export interface Draft {
  to: EmailAddress[];
  cc: EmailAddress[];
  bcc: EmailAddress[];
  subject: string;
  bodyText: string;
  bodyHtml?: string;
  attachments: File[];
}

// ============================================================================
// Plugin Base Class
// ============================================================================

export abstract class Plugin {
  protected context!: PluginContext;

  /** Called when plugin is loaded */
  abstract onLoad(context: PluginContext): Promise<void>;

  /** Called when plugin is unloaded */
  async onUnload(): Promise<void> {}

  /** Called when settings change */
  async onSettingsChange(settings: Record<string, unknown>): Promise<void> {}

  /** Get plugin settings schema */
  getSettingsSchema(): JSONSchema | null {
    return null;
  }

  /** Render settings UI */
  renderSettings?(): React.ReactNode;

  /** Render sidebar content */
  renderSidebar?(): React.ReactNode;

  /** Render main view */
  renderMain?(): React.ReactNode;
}

// ============================================================================
// Plugin Registry
// ============================================================================

class PluginRegistry {
  private plugins: Map<string, Plugin> = new Map();
  private contexts: Map<string, PluginContext> = new Map();

  async register(manifest: PluginManifest, PluginClass: new () => Plugin): Promise<void> {
    const plugin = new PluginClass();
    const context = this.createContext(manifest);

    this.plugins.set(manifest.id, plugin);
    this.contexts.set(manifest.id, context);

    await plugin.onLoad(context);
  }

  async unregister(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (plugin) {
      await plugin.onUnload();
      this.plugins.delete(pluginId);
      this.contexts.delete(pluginId);
    }
  }

  getPlugin(pluginId: string): Plugin | undefined {
    return this.plugins.get(pluginId);
  }

  listPlugins(): string[] {
    return Array.from(this.plugins.keys());
  }

  private createContext(manifest: PluginManifest): PluginContext {
    return {
      manifest,
      settings: {},
      api: this.createAPI(manifest),
      storage: this.createStorage(manifest.id),
      events: this.createEvents(),
      ui: this.createUI(manifest.id),
      logger: this.createLogger(manifest.id),
    };
  }

  private createAPI(manifest: PluginManifest): PluginAPI {
    const checkPermission = (required: PluginPermission) => {
      if (!manifest.permissions.includes(required)) {
        throw new Error(`Plugin lacks permission: ${required}`);
      }
    };

    return {
      async fetch(path: string, options?: RequestInit): Promise<Response> {
        return fetch(`/api${path}`, {
          ...options,
          headers: {
            ...options?.headers,
            'X-Plugin-Id': manifest.id,
          },
        });
      },

      async getCurrentUser(): Promise<User> {
        const res = await fetch('/api/me');
        return res.json();
      },

      async getEmails(options?: EmailQueryOptions): Promise<Email[]> {
        checkPermission('read:emails');
        const params = new URLSearchParams(options as Record<string, string>);
        const res = await fetch(`/api/emails?${params}`);
        return res.json();
      },

      async getContacts(options?: ContactQueryOptions): Promise<Contact[]> {
        checkPermission('read:contacts');
        const params = new URLSearchParams(options as Record<string, string>);
        const res = await fetch(`/api/contacts?${params}`);
        return res.json();
      },

      async getTrustScore(userId: string): Promise<TrustScore> {
        checkPermission('read:trust');
        const res = await fetch(`/api/trust/${userId}`);
        return res.json();
      },
    };
  }

  private createStorage(pluginId: string): PluginStorage {
    const prefix = `plugin:${pluginId}:`;

    return {
      async get<T>(key: string): Promise<T | null> {
        const value = localStorage.getItem(prefix + key);
        return value ? JSON.parse(value) : null;
      },

      async set<T>(key: string, value: T): Promise<void> {
        localStorage.setItem(prefix + key, JSON.stringify(value));
      },

      async remove(key: string): Promise<void> {
        localStorage.removeItem(prefix + key);
      },

      async keys(): Promise<string[]> {
        return Object.keys(localStorage)
          .filter((k) => k.startsWith(prefix))
          .map((k) => k.slice(prefix.length));
      },

      async clear(): Promise<void> {
        const keys = await this.keys();
        keys.forEach((k) => localStorage.removeItem(prefix + k));
      },
    };
  }

  private createEvents(): PluginEvents {
    const handlers: Map<string, Set<Function>> = new Map();

    return {
      on<T>(event: PluginEvent, handler: (data: T) => void): () => void {
        if (!handlers.has(event)) {
          handlers.set(event, new Set());
        }
        handlers.get(event)!.add(handler);
        return () => handlers.get(event)!.delete(handler);
      },

      emit<T>(event: PluginEvent, data: T): void {
        handlers.get(event)?.forEach((h) => h(data));
      },
    };
  }

  private createUI(pluginId: string): PluginUI {
    return {
      showNotification(options: NotificationOptions): void {
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification(options.title, { body: options.body, icon: options.icon });
        }
      },

      async showModal(options: ModalOptions): Promise<unknown> {
        // Would integrate with app's modal system
        return new Promise((resolve) => {
          console.log('Modal:', options);
          resolve(undefined);
        });
      },

      showToast(message: string, type = 'info'): void {
        // Would integrate with app's toast system
        console.log(`Toast (${type}):`, message);
      },

      registerSidebarItem(item: SidebarItem): void {
        // Would register with app's sidebar
        console.log('Sidebar item:', item);
      },

      registerToolbarAction(action: ToolbarAction): void {
        console.log('Toolbar action:', action);
      },

      registerEmailAction(action: EmailAction): void {
        console.log('Email action:', action);
      },

      registerComposeAction(action: ComposeAction): void {
        console.log('Compose action:', action);
      },
    };
  }

  private createLogger(pluginId: string): PluginLogger {
    const prefix = `[Plugin:${pluginId}]`;
    return {
      debug: (msg, ...args) => console.debug(prefix, msg, ...args),
      info: (msg, ...args) => console.info(prefix, msg, ...args),
      warn: (msg, ...args) => console.warn(prefix, msg, ...args),
      error: (msg, ...args) => console.error(prefix, msg, ...args),
    };
  }
}

export const pluginRegistry = new PluginRegistry();

// ============================================================================
// Helper Functions
// ============================================================================

export function definePlugin(manifest: PluginManifest, PluginClass: new () => Plugin) {
  return { manifest, PluginClass };
}

export function usePluginContext(): PluginContext {
  // Would use React context in actual implementation
  throw new Error('usePluginContext must be used within a plugin');
}
