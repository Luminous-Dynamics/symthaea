// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Platform Manager
 *
 * Central orchestration for all Mycelix subsystems:
 * - Lifecycle management
 * - Service registry
 * - Cross-module coordination
 * - State synchronization
 */

import { EventBus, getEventBus, createLoggingMiddleware } from './event-bus';

// ==================== Types ====================

export interface PlatformConfig {
  apiBaseUrl: string;
  wsBaseUrl: string;
  cdnBaseUrl: string;
  features: FeatureFlags;
  environment: 'development' | 'staging' | 'production';
  debug: boolean;
}

export interface FeatureFlags {
  collaboration: boolean;
  ai: boolean;
  streaming: boolean;
  xr: boolean;
  marketplace: boolean;
  spatial: boolean;
  offline: boolean;
}

export interface ServiceStatus {
  name: string;
  status: 'initializing' | 'ready' | 'error' | 'disabled';
  error?: string;
  startTime?: number;
  readyTime?: number;
}

export type ServiceName =
  | 'audio'
  | 'collaboration'
  | 'social'
  | 'creator'
  | 'ai'
  | 'streaming'
  | 'marketplace'
  | 'spatial'
  | 'xr'
  | 'codecs'
  | 'pwa';

export interface PlatformState {
  initialized: boolean;
  user: UserState | null;
  playback: PlaybackState;
  network: NetworkState;
  services: Map<ServiceName, ServiceStatus>;
}

export interface UserState {
  id: string;
  name: string;
  avatar: string;
  email: string;
  subscription: 'free' | 'premium' | 'creator';
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  audioQuality: 'low' | 'medium' | 'high' | 'lossless';
  spatialAudio: boolean;
  notifications: boolean;
  autoplay: boolean;
  crossfade: number;
  normalization: boolean;
}

export interface PlaybackState {
  currentTrack: string | null;
  queue: string[];
  position: number;
  duration: number;
  volume: number;
  muted: boolean;
  playing: boolean;
  repeat: 'none' | 'one' | 'all';
  shuffle: boolean;
}

export interface NetworkState {
  online: boolean;
  connectionType: string;
  effectiveType: string;
  downlink: number;
  rtt: number;
}

// ==================== Service Registry ====================

class ServiceRegistry {
  private services: Map<string, any> = new Map();
  private factories: Map<string, () => Promise<any>> = new Map();
  private status: Map<string, ServiceStatus> = new Map();

  register<T>(name: string, factory: () => Promise<T>): void {
    this.factories.set(name, factory);
    this.status.set(name, { name, status: 'initializing' });
  }

  async get<T>(name: string): Promise<T> {
    if (this.services.has(name)) {
      return this.services.get(name);
    }

    const factory = this.factories.get(name);
    if (!factory) {
      throw new Error(`Service not registered: ${name}`);
    }

    const status = this.status.get(name)!;
    status.startTime = Date.now();

    try {
      const service = await factory();
      this.services.set(name, service);
      status.status = 'ready';
      status.readyTime = Date.now();
      return service;
    } catch (error) {
      status.status = 'error';
      status.error = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    }
  }

  has(name: string): boolean {
    return this.factories.has(name);
  }

  isReady(name: string): boolean {
    return this.status.get(name)?.status === 'ready';
  }

  getStatus(): Map<string, ServiceStatus> {
    return new Map(this.status);
  }

  disable(name: string): void {
    const status = this.status.get(name);
    if (status) {
      status.status = 'disabled';
    }
  }
}

// ==================== Platform Manager ====================

export class PlatformManager {
  private config: PlatformConfig;
  private eventBus: EventBus;
  private registry: ServiceRegistry;
  private state: PlatformState;
  private stateListeners: Set<(state: PlatformState) => void> = new Set();

  constructor(config: Partial<PlatformConfig> = {}) {
    this.config = {
      apiBaseUrl: config.apiBaseUrl || '/api',
      wsBaseUrl: config.wsBaseUrl || 'wss://api.mycelix.io',
      cdnBaseUrl: config.cdnBaseUrl || 'https://cdn.mycelix.io',
      environment: config.environment || 'production',
      debug: config.debug || false,
      features: {
        collaboration: true,
        ai: true,
        streaming: true,
        xr: true,
        marketplace: true,
        spatial: true,
        offline: true,
        ...config.features,
      },
    };

    this.eventBus = getEventBus();
    this.registry = new ServiceRegistry();

    this.state = {
      initialized: false,
      user: null,
      playback: {
        currentTrack: null,
        queue: [],
        position: 0,
        duration: 0,
        volume: 1,
        muted: false,
        playing: false,
        repeat: 'none',
        shuffle: false,
      },
      network: {
        online: typeof navigator !== 'undefined' ? navigator.onLine : true,
        connectionType: 'unknown',
        effectiveType: '4g',
        downlink: 10,
        rtt: 50,
      },
      services: new Map(),
    };

    this.setupEventHandlers();
    this.setupNetworkMonitoring();
  }

  async initialize(): Promise<void> {
    if (this.state.initialized) return;

    console.log('[Platform] Initializing Mycelix...');

    // Add logging middleware in development
    if (this.config.debug) {
      this.eventBus.use(createLoggingMiddleware({ verbose: true }));
    }

    // Register core services
    await this.registerServices();

    // Initialize enabled services
    await this.initializeServices();

    this.state.initialized = true;
    this.notifyStateChange();

    console.log('[Platform] Mycelix initialized');
  }

  private async registerServices(): Promise<void> {
    const { features } = this.config;

    // Always register audio service
    this.registry.register('audio', async () => {
      const { getAudioEngine } = await import('../audio-engine');
      return getAudioEngine();
    });

    // Conditional service registration
    if (features.collaboration) {
      this.registry.register('collaboration', async () => {
        const { getCollaborationManager } = await import('../collaboration');
        return getCollaborationManager();
      });
    }

    if (features.ai) {
      this.registry.register('ai', async () => {
        const { getAICreativeManager } = await import('../ai-creative');
        return getAICreativeManager(this.config.apiBaseUrl);
      });
    }

    if (features.streaming) {
      this.registry.register('streaming', async () => {
        const { getLiveStreamingManager } = await import('../live-streaming');
        return getLiveStreamingManager();
      });
    }

    if (features.xr) {
      this.registry.register('xr', async () => {
        const { getXRStudioManager } = await import('../webxr');
        return getXRStudioManager();
      });
    }

    if (features.marketplace) {
      this.registry.register('marketplace', async () => {
        const { getMarketplaceManager } = await import('../marketplace');
        return getMarketplaceManager();
      });
    }

    if (features.spatial) {
      this.registry.register('spatial', async () => {
        const { getSpatialAudioManager } = await import('../spatial-audio');
        return getSpatialAudioManager();
      });
    }

    if (features.offline) {
      this.registry.register('pwa', async () => {
        // Register service worker
        if ('serviceWorker' in navigator) {
          await navigator.serviceWorker.register('/service-worker.js');
        }
        return { registered: true };
      });
    }

    this.registry.register('codecs', async () => {
      const { getCodecManager } = await import('../codecs');
      const manager = getCodecManager();
      await manager.initialize();
      return manager;
    });

    this.registry.register('social', async () => {
      const { getSocialManager } = await import('../social');
      return getSocialManager();
    });

    this.registry.register('creator', async () => {
      const { getCreatorEconomyManager } = await import('../creator-economy');
      return getCreatorEconomyManager();
    });
  }

  private async initializeServices(): Promise<void> {
    const priorityOrder: ServiceName[] = [
      'audio',
      'codecs',
      'pwa',
      'social',
      'creator',
      'collaboration',
      'ai',
      'streaming',
      'spatial',
      'xr',
      'marketplace',
    ];

    for (const serviceName of priorityOrder) {
      if (this.registry.has(serviceName)) {
        try {
          await this.registry.get(serviceName);
          console.log(`[Platform] ${serviceName} ready`);
        } catch (error) {
          console.error(`[Platform] ${serviceName} failed:`, error);
        }
      }
    }

    this.state.services = this.registry.getStatus();
  }

  private setupEventHandlers(): void {
    // Playback state sync
    this.eventBus.on('audio:play', ({ trackId, position }) => {
      this.updateState({
        playback: {
          ...this.state.playback,
          currentTrack: trackId,
          playing: true,
          position: position || 0,
        },
      });
    });

    this.eventBus.on('audio:pause', () => {
      this.updateState({
        playback: { ...this.state.playback, playing: false },
      });
    });

    this.eventBus.on('audio:seek', ({ position }) => {
      this.updateState({
        playback: { ...this.state.playback, position },
      });
    });

    this.eventBus.on('audio:volume', ({ volume }) => {
      this.updateState({
        playback: { ...this.state.playback, volume },
      });
    });

    // User state sync
    this.eventBus.on('user:login', ({ userId }) => {
      this.loadUser(userId);
    });

    this.eventBus.on('user:logout', () => {
      this.updateState({ user: null });
    });
  }

  private setupNetworkMonitoring(): void {
    if (typeof window === 'undefined') return;

    window.addEventListener('online', () => {
      this.updateState({
        network: { ...this.state.network, online: true },
      });
      this.eventBus.emitSync('system:online', {});
    });

    window.addEventListener('offline', () => {
      this.updateState({
        network: { ...this.state.network, online: false },
      });
      this.eventBus.emitSync('system:offline', {});
    });

    // Network Information API
    const connection = (navigator as any).connection;
    if (connection) {
      const updateNetworkInfo = () => {
        this.updateState({
          network: {
            ...this.state.network,
            connectionType: connection.type || 'unknown',
            effectiveType: connection.effectiveType || '4g',
            downlink: connection.downlink || 10,
            rtt: connection.rtt || 50,
          },
        });
      };

      connection.addEventListener('change', updateNetworkInfo);
      updateNetworkInfo();
    }
  }

  private async loadUser(userId: string): Promise<void> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/users/${userId}`);
      const user = await response.json();
      this.updateState({ user });
    } catch (error) {
      console.error('[Platform] Failed to load user:', error);
    }
  }

  private updateState(partial: Partial<PlatformState>): void {
    this.state = { ...this.state, ...partial };
    this.notifyStateChange();
  }

  private notifyStateChange(): void {
    for (const listener of this.stateListeners) {
      listener(this.state);
    }
  }

  // ==================== Public API ====================

  getConfig(): PlatformConfig {
    return { ...this.config };
  }

  getState(): PlatformState {
    return { ...this.state };
  }

  subscribe(listener: (state: PlatformState) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  getEventBus(): EventBus {
    return this.eventBus;
  }

  async getService<T>(name: ServiceName): Promise<T> {
    return this.registry.get<T>(name);
  }

  isServiceReady(name: ServiceName): boolean {
    return this.registry.isReady(name);
  }

  isFeatureEnabled(feature: keyof FeatureFlags): boolean {
    return this.config.features[feature];
  }

  async updateUserPreferences(preferences: Partial<UserPreferences>): Promise<void> {
    if (!this.state.user) return;

    const newPreferences = { ...this.state.user.preferences, ...preferences };

    await fetch(`${this.config.apiBaseUrl}/users/${this.state.user.id}/preferences`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newPreferences),
    });

    this.updateState({
      user: { ...this.state.user, preferences: newPreferences },
    });

    this.eventBus.emitSync('user:preferences', {
      key: Object.keys(preferences)[0],
      value: Object.values(preferences)[0],
    });
  }

  // Playback controls
  async play(trackId: string, position?: number): Promise<void> {
    this.eventBus.emitSync('audio:play', { trackId, position });
  }

  async pause(): Promise<void> {
    if (this.state.playback.currentTrack) {
      this.eventBus.emitSync('audio:pause', {
        trackId: this.state.playback.currentTrack,
      });
    }
  }

  async seek(position: number): Promise<void> {
    this.eventBus.emitSync('audio:seek', { position });
  }

  async setVolume(volume: number): Promise<void> {
    this.eventBus.emitSync('audio:volume', { volume });
  }

  // Queue controls
  async addToQueue(trackId: string, position?: number): Promise<void> {
    this.eventBus.emitSync('queue:add', { trackId, position });
    this.updateState({
      playback: {
        ...this.state.playback,
        queue: position !== undefined
          ? [...this.state.playback.queue.slice(0, position), trackId, ...this.state.playback.queue.slice(position)]
          : [...this.state.playback.queue, trackId],
      },
    });
  }

  async removeFromQueue(trackId: string): Promise<void> {
    this.eventBus.emitSync('queue:remove', { trackId });
    this.updateState({
      playback: {
        ...this.state.playback,
        queue: this.state.playback.queue.filter(id => id !== trackId),
      },
    });
  }

  async clearQueue(): Promise<void> {
    this.eventBus.emitSync('queue:clear', {});
    this.updateState({
      playback: { ...this.state.playback, queue: [] },
    });
  }

  // Cleanup
  async dispose(): Promise<void> {
    this.eventBus.clear();
    this.stateListeners.clear();
  }
}

// ==================== Singleton ====================

let platformManager: PlatformManager | null = null;

export function getPlatformManager(config?: Partial<PlatformConfig>): PlatformManager {
  if (!platformManager) {
    platformManager = new PlatformManager(config);
  }
  return platformManager;
}

export async function initializePlatform(config?: Partial<PlatformConfig>): Promise<PlatformManager> {
  const manager = getPlatformManager(config);
  await manager.initialize();
  return manager;
}

export default {
  PlatformManager,
  getPlatformManager,
  initializePlatform,
};
