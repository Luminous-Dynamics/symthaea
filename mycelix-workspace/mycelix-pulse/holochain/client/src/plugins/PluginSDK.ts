// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Plugin SDK - Extensible Plugin Architecture
 *
 * Features:
 * - Plugin lifecycle management
 * - Hooks for email processing
 * - UI extension points
 * - Sandboxed execution
 * - Inter-plugin communication
 */

import { EventEmitter } from 'events';

export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  author?: string;
  description?: string;
  homepage?: string;
  repository?: string;
  permissions: PluginPermission[];
  hooks: PluginHook[];
  uiExtensions?: UIExtension[];
  settings?: PluginSetting[];
}

export type PluginPermission =
  | 'emails:read'
  | 'emails:write'
  | 'emails:send'
  | 'contacts:read'
  | 'contacts:write'
  | 'trust:read'
  | 'trust:write'
  | 'storage:read'
  | 'storage:write'
  | 'network:fetch'
  | 'notifications:show'
  | 'ui:toolbar'
  | 'ui:compose'
  | 'ui:sidebar';

export type PluginHook =
  | 'email:beforeSend'
  | 'email:afterSend'
  | 'email:beforeReceive'
  | 'email:afterReceive'
  | 'email:beforeDisplay'
  | 'compose:beforeOpen'
  | 'compose:beforeSend'
  | 'contact:beforeSave'
  | 'trust:beforeAttest'
  | 'search:beforeQuery'
  | 'search:afterQuery';

export interface UIExtension {
  id: string;
  type: 'toolbar-button' | 'sidebar-panel' | 'compose-action' | 'email-action' | 'settings-section';
  label: string;
  icon?: string;
  component?: string;
}

export interface PluginSetting {
  id: string;
  type: 'text' | 'number' | 'boolean' | 'select' | 'color';
  label: string;
  description?: string;
  default?: any;
  options?: { label: string; value: any }[];
}

export interface PluginContext {
  manifest: PluginManifest;
  settings: Record<string, any>;
  storage: PluginStorage;
  api: PluginAPI;
  ui: PluginUI;
}

export interface PluginStorage {
  get: (key: string) => Promise<any>;
  set: (key: string, value: any) => Promise<void>;
  delete: (key: string) => Promise<void>;
  list: () => Promise<string[]>;
  clear: () => Promise<void>;
}

export interface PluginAPI {
  emails: {
    list: (options?: any) => Promise<any[]>;
    get: (hash: string) => Promise<any>;
    send: (email: any) => Promise<string>;
    archive: (hash: string) => Promise<void>;
    trash: (hash: string) => Promise<void>;
  };
  contacts: {
    list: () => Promise<any[]>;
    get: (id: string) => Promise<any>;
    create: (contact: any) => Promise<string>;
    update: (id: string, data: any) => Promise<void>;
    delete: (id: string) => Promise<void>;
  };
  trust: {
    getScore: (agent: string) => Promise<number>;
    getNetwork: () => Promise<any[]>;
    attest: (agent: string, level: number) => Promise<string>;
  };
  fetch: (url: string, options?: RequestInit) => Promise<Response>;
  notify: (title: string, options?: NotificationOptions) => Promise<void>;
}

export interface PluginUI {
  showModal: (options: ModalOptions) => Promise<any>;
  showToast: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
  registerComponent: (id: string, component: any) => void;
  getTheme: () => any;
}

interface ModalOptions {
  title: string;
  content: string | any;
  actions?: { label: string; action: string; primary?: boolean }[];
}

export interface Plugin {
  manifest: PluginManifest;
  activate: (context: PluginContext) => Promise<void>;
  deactivate: () => Promise<void>;
  onHook?: (hook: PluginHook, data: any) => Promise<any>;
}

export class PluginManager extends EventEmitter {
  private plugins: Map<string, Plugin> = new Map();
  private activePlugins: Set<string> = new Set();
  private contexts: Map<string, PluginContext> = new Map();
  private hooks: Map<PluginHook, Set<string>> = new Map();
  private services: any;

  constructor(services: any) {
    super();
    this.services = services;
  }

  /**
   * Register a plugin
   */
  async register(plugin: Plugin): Promise<void> {
    const { id } = plugin.manifest;

    if (this.plugins.has(id)) {
      throw new Error(`Plugin ${id} is already registered`);
    }

    // Validate manifest
    this.validateManifest(plugin.manifest);

    this.plugins.set(id, plugin);
    this.emit('plugin-registered', plugin.manifest);
  }

  /**
   * Unregister a plugin
   */
  async unregister(pluginId: string): Promise<void> {
    if (this.activePlugins.has(pluginId)) {
      await this.deactivate(pluginId);
    }

    this.plugins.delete(pluginId);
    this.emit('plugin-unregistered', pluginId);
  }

  /**
   * Activate a plugin
   */
  async activate(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      throw new Error(`Plugin ${pluginId} not found`);
    }

    if (this.activePlugins.has(pluginId)) {
      return;
    }

    // Check permissions
    const granted = await this.requestPermissions(plugin.manifest.permissions);
    if (!granted) {
      throw new Error('Required permissions not granted');
    }

    // Create context
    const context = this.createContext(plugin.manifest);
    this.contexts.set(pluginId, context);

    // Register hooks
    for (const hook of plugin.manifest.hooks) {
      if (!this.hooks.has(hook)) {
        this.hooks.set(hook, new Set());
      }
      this.hooks.get(hook)!.add(pluginId);
    }

    // Activate plugin
    await plugin.activate(context);
    this.activePlugins.add(pluginId);

    this.emit('plugin-activated', pluginId);
  }

  /**
   * Deactivate a plugin
   */
  async deactivate(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin || !this.activePlugins.has(pluginId)) {
      return;
    }

    // Unregister hooks
    for (const [hook, plugins] of this.hooks) {
      plugins.delete(pluginId);
    }

    // Deactivate plugin
    await plugin.deactivate();
    this.activePlugins.delete(pluginId);
    this.contexts.delete(pluginId);

    this.emit('plugin-deactivated', pluginId);
  }

  /**
   * Execute a hook
   */
  async executeHook(hook: PluginHook, data: any): Promise<any> {
    const pluginIds = this.hooks.get(hook);
    if (!pluginIds || pluginIds.size === 0) {
      return data;
    }

    let result = data;

    for (const pluginId of pluginIds) {
      const plugin = this.plugins.get(pluginId);
      if (plugin?.onHook) {
        try {
          result = await plugin.onHook(hook, result);
        } catch (error) {
          console.error(`Plugin ${pluginId} hook error:`, error);
          this.emit('plugin-error', { pluginId, hook, error });
        }
      }
    }

    return result;
  }

  /**
   * Get all plugins
   */
  getPlugins(): PluginManifest[] {
    return Array.from(this.plugins.values()).map((p) => p.manifest);
  }

  /**
   * Get active plugins
   */
  getActivePlugins(): PluginManifest[] {
    return Array.from(this.activePlugins)
      .map((id) => this.plugins.get(id)?.manifest)
      .filter((m): m is PluginManifest => !!m);
  }

  /**
   * Check if plugin is active
   */
  isActive(pluginId: string): boolean {
    return this.activePlugins.has(pluginId);
  }

  /**
   * Get plugin settings
   */
  getSettings(pluginId: string): Record<string, any> {
    return this.contexts.get(pluginId)?.settings || {};
  }

  /**
   * Update plugin settings
   */
  async updateSettings(pluginId: string, settings: Record<string, any>): Promise<void> {
    const context = this.contexts.get(pluginId);
    if (context) {
      context.settings = { ...context.settings, ...settings };
      await this.savePluginSettings(pluginId, context.settings);
      this.emit('plugin-settings-updated', { pluginId, settings: context.settings });
    }
  }

  // Private methods

  private validateManifest(manifest: PluginManifest): void {
    if (!manifest.id || !manifest.name || !manifest.version) {
      throw new Error('Invalid plugin manifest: missing required fields');
    }

    // Validate permissions
    const validPermissions = new Set<PluginPermission>([
      'emails:read', 'emails:write', 'emails:send',
      'contacts:read', 'contacts:write',
      'trust:read', 'trust:write',
      'storage:read', 'storage:write',
      'network:fetch', 'notifications:show',
      'ui:toolbar', 'ui:compose', 'ui:sidebar',
    ]);

    for (const permission of manifest.permissions) {
      if (!validPermissions.has(permission)) {
        throw new Error(`Invalid permission: ${permission}`);
      }
    }
  }

  private async requestPermissions(permissions: PluginPermission[]): Promise<boolean> {
    // In a real implementation, this would show a permission dialog
    // For now, auto-grant
    return true;
  }

  private createContext(manifest: PluginManifest): PluginContext {
    const storage = this.createStorage(manifest.id);
    const api = this.createAPI(manifest);
    const ui = this.createUI(manifest.id);

    return {
      manifest,
      settings: this.loadPluginSettings(manifest),
      storage,
      api,
      ui,
    };
  }

  private createStorage(pluginId: string): PluginStorage {
    const prefix = `plugin:${pluginId}:`;

    return {
      get: async (key) => {
        const data = localStorage.getItem(prefix + key);
        return data ? JSON.parse(data) : null;
      },
      set: async (key, value) => {
        localStorage.setItem(prefix + key, JSON.stringify(value));
      },
      delete: async (key) => {
        localStorage.removeItem(prefix + key);
      },
      list: async () => {
        const keys: string[] = [];
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key?.startsWith(prefix)) {
            keys.push(key.slice(prefix.length));
          }
        }
        return keys;
      },
      clear: async () => {
        const keys = await this.createStorage(pluginId).list();
        for (const key of keys) {
          localStorage.removeItem(prefix + key);
        }
      },
    };
  }

  private createAPI(manifest: PluginManifest): PluginAPI {
    const hasPermission = (perm: PluginPermission) =>
      manifest.permissions.includes(perm);

    return {
      emails: {
        list: async (options) => {
          if (!hasPermission('emails:read')) throw new Error('Permission denied');
          return this.services.messages.getMessages(options);
        },
        get: async (hash) => {
          if (!hasPermission('emails:read')) throw new Error('Permission denied');
          return this.services.messages.getMessage(hash);
        },
        send: async (email) => {
          if (!hasPermission('emails:send')) throw new Error('Permission denied');
          return this.services.messages.send(email);
        },
        archive: async (hash) => {
          if (!hasPermission('emails:write')) throw new Error('Permission denied');
          return this.services.messages.archive(hash);
        },
        trash: async (hash) => {
          if (!hasPermission('emails:write')) throw new Error('Permission denied');
          return this.services.messages.trash(hash);
        },
      },
      contacts: {
        list: async () => {
          if (!hasPermission('contacts:read')) throw new Error('Permission denied');
          return this.services.contacts.getAllContacts();
        },
        get: async (id) => {
          if (!hasPermission('contacts:read')) throw new Error('Permission denied');
          return this.services.contacts.getContact(id);
        },
        create: async (contact) => {
          if (!hasPermission('contacts:write')) throw new Error('Permission denied');
          return this.services.contacts.createContact(contact);
        },
        update: async (id, data) => {
          if (!hasPermission('contacts:write')) throw new Error('Permission denied');
          return this.services.contacts.updateContact(id, data);
        },
        delete: async (id) => {
          if (!hasPermission('contacts:write')) throw new Error('Permission denied');
          return this.services.contacts.deleteContact(id);
        },
      },
      trust: {
        getScore: async (agent) => {
          if (!hasPermission('trust:read')) throw new Error('Permission denied');
          const score = await this.services.trust.getTrustScore(agent);
          return score?.combinedTrust || 0;
        },
        getNetwork: async () => {
          if (!hasPermission('trust:read')) throw new Error('Permission denied');
          return this.services.trust.getMyTrustNetwork();
        },
        attest: async (agent, level) => {
          if (!hasPermission('trust:write')) throw new Error('Permission denied');
          return this.services.trust.createAttestation({ toAgent: agent, trustLevel: level });
        },
      },
      fetch: async (url, options) => {
        if (!hasPermission('network:fetch')) throw new Error('Permission denied');
        return fetch(url, options);
      },
      notify: async (title, options) => {
        if (!hasPermission('notifications:show')) throw new Error('Permission denied');
        if (Notification.permission === 'granted') {
          new Notification(title, options);
        }
      },
    };
  }

  private createUI(pluginId: string): PluginUI {
    return {
      showModal: async (options) => {
        this.emit('show-modal', { pluginId, ...options });
        return new Promise((resolve) => {
          this.once(`modal-result:${pluginId}`, resolve);
        });
      },
      showToast: (message, type = 'info') => {
        this.emit('show-toast', { pluginId, message, type });
      },
      registerComponent: (id, component) => {
        this.emit('register-component', { pluginId, componentId: id, component });
      },
      getTheme: () => {
        // Return current theme
        return {};
      },
    };
  }

  private loadPluginSettings(manifest: PluginManifest): Record<string, any> {
    const saved = localStorage.getItem(`plugin-settings:${manifest.id}`);
    const defaults: Record<string, any> = {};

    for (const setting of manifest.settings || []) {
      defaults[setting.id] = setting.default;
    }

    return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
  }

  private async savePluginSettings(pluginId: string, settings: Record<string, any>): Promise<void> {
    localStorage.setItem(`plugin-settings:${pluginId}`, JSON.stringify(settings));
  }
}

export default PluginManager;
