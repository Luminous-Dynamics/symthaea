// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Configuration System
 *
 * Centralized configuration management:
 * - Feature flags with remote sync
 * - Environment-based config
 * - A/B testing support
 * - Runtime configuration updates
 * - Type-safe config access
 */

// ==================== Types ====================

export interface AppConfig {
  app: AppSettings;
  api: APISettings;
  features: FeatureFlags;
  limits: LimitSettings;
  ui: UISettings;
  audio: AudioSettings;
  analytics: AnalyticsSettings;
  experiments: ExperimentSettings;
}

export interface AppSettings {
  name: string;
  version: string;
  environment: Environment;
  debug: boolean;
  logLevel: LogLevel;
}

export type Environment = 'development' | 'staging' | 'production';
export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'none';

export interface APISettings {
  baseUrl: string;
  wsUrl: string;
  cdnUrl: string;
  timeout: number;
  retries: number;
  rateLimit: number;
}

export interface FeatureFlags {
  // Core features
  collaboration: boolean;
  aiCreative: boolean;
  liveStreaming: boolean;
  vrStudio: boolean;
  marketplace: boolean;
  spatialAudio: boolean;
  offlineMode: boolean;
  wasmCodecs: boolean;

  // UI features
  darkMode: boolean;
  animations: boolean;
  haptics: boolean;
  experimentalUI: boolean;

  // Social features
  comments: boolean;
  directMessages: boolean;
  stories: boolean;
  liveChat: boolean;

  // Creator features
  monetization: boolean;
  analytics: boolean;
  subscriptions: boolean;
  tipping: boolean;
  nfts: boolean;

  // Playback features
  crossfade: boolean;
  gapless: boolean;
  loudnessNormalization: boolean;
  replayGain: boolean;
}

export interface LimitSettings {
  maxUploadSize: number;
  maxTrackLength: number;
  maxPlaylistSize: number;
  maxQueueSize: number;
  maxCollaborators: number;
  maxStreamViewers: number;
  freeStreamMinutes: number;
  freeStorageBytes: number;
}

export interface UISettings {
  theme: 'light' | 'dark' | 'system';
  accentColor: string;
  fontSize: 'small' | 'medium' | 'large';
  reducedMotion: boolean;
  highContrast: boolean;
  compactMode: boolean;
}

export interface AudioSettings {
  defaultQuality: 'low' | 'medium' | 'high' | 'lossless';
  streamingQuality: 'auto' | 'low' | 'medium' | 'high';
  downloadQuality: 'medium' | 'high' | 'lossless';
  bufferSize: number;
  crossfadeDuration: number;
  normalizationTarget: number;
  spatialEnabled: boolean;
}

export interface AnalyticsSettings {
  enabled: boolean;
  anonymize: boolean;
  sampleRate: number;
  events: string[];
}

export interface ExperimentSettings {
  enabled: boolean;
  userId?: string;
  experiments: Experiment[];
}

export interface Experiment {
  id: string;
  name: string;
  enabled: boolean;
  variants: ExperimentVariant[];
  allocation: number;
  startDate?: Date;
  endDate?: Date;
}

export interface ExperimentVariant {
  id: string;
  name: string;
  weight: number;
  config: Record<string, any>;
}

// ==================== Default Configuration ====================

const defaultConfig: AppConfig = {
  app: {
    name: 'Mycelix',
    version: '1.0.0',
    environment: 'production',
    debug: false,
    logLevel: 'warn',
  },
  api: {
    baseUrl: '/api',
    wsUrl: 'wss://api.mycelix.io',
    cdnUrl: 'https://cdn.mycelix.io',
    timeout: 30000,
    retries: 3,
    rateLimit: 100,
  },
  features: {
    collaboration: true,
    aiCreative: true,
    liveStreaming: true,
    vrStudio: true,
    marketplace: true,
    spatialAudio: true,
    offlineMode: true,
    wasmCodecs: true,
    darkMode: true,
    animations: true,
    haptics: true,
    experimentalUI: false,
    comments: true,
    directMessages: true,
    stories: true,
    liveChat: true,
    monetization: true,
    analytics: true,
    subscriptions: true,
    tipping: true,
    nfts: false,
    crossfade: true,
    gapless: true,
    loudnessNormalization: true,
    replayGain: false,
  },
  limits: {
    maxUploadSize: 500 * 1024 * 1024, // 500MB
    maxTrackLength: 60 * 60, // 1 hour
    maxPlaylistSize: 10000,
    maxQueueSize: 1000,
    maxCollaborators: 10,
    maxStreamViewers: 10000,
    freeStreamMinutes: 60,
    freeStorageBytes: 5 * 1024 * 1024 * 1024, // 5GB
  },
  ui: {
    theme: 'system',
    accentColor: '#6366f1',
    fontSize: 'medium',
    reducedMotion: false,
    highContrast: false,
    compactMode: false,
  },
  audio: {
    defaultQuality: 'high',
    streamingQuality: 'auto',
    downloadQuality: 'high',
    bufferSize: 4096,
    crossfadeDuration: 5,
    normalizationTarget: -14,
    spatialEnabled: false,
  },
  analytics: {
    enabled: true,
    anonymize: false,
    sampleRate: 1.0,
    events: ['play', 'like', 'share', 'follow', 'subscribe'],
  },
  experiments: {
    enabled: true,
    experiments: [],
  },
};

// ==================== Environment Configs ====================

const environmentConfigs: Record<Environment, Partial<AppConfig>> = {
  development: {
    app: {
      ...defaultConfig.app,
      environment: 'development',
      debug: true,
      logLevel: 'debug',
    },
    api: {
      ...defaultConfig.api,
      baseUrl: 'http://localhost:3000/api',
      wsUrl: 'ws://localhost:3000',
    },
    features: {
      ...defaultConfig.features,
      experimentalUI: true,
      nfts: true,
    },
  },
  staging: {
    app: {
      ...defaultConfig.app,
      environment: 'staging',
      debug: true,
      logLevel: 'info',
    },
    api: {
      ...defaultConfig.api,
      baseUrl: 'https://staging-api.mycelix.io',
      wsUrl: 'wss://staging-api.mycelix.io',
    },
  },
  production: {
    app: {
      ...defaultConfig.app,
      environment: 'production',
      debug: false,
      logLevel: 'warn',
    },
  },
};

// ==================== Configuration Manager ====================

export class ConfigManager {
  private config: AppConfig;
  private remoteConfig: Partial<AppConfig> | null = null;
  private userOverrides: Partial<AppConfig> = {};
  private listeners: Set<(config: AppConfig) => void> = new Set();
  private experimentAssignments: Map<string, string> = new Map();
  private syncInterval: NodeJS.Timeout | null = null;

  constructor(environment: Environment = 'production') {
    this.config = this.mergeConfigs(
      defaultConfig,
      environmentConfigs[environment] || {}
    );

    this.loadUserOverrides();
    this.loadExperimentAssignments();
  }

  /**
   * Initialize with remote configuration
   */
  async initialize(remoteUrl?: string): Promise<void> {
    if (remoteUrl) {
      await this.syncRemoteConfig(remoteUrl);
      this.startAutoSync(remoteUrl);
    }
  }

  /**
   * Get the current configuration
   */
  getConfig(): AppConfig {
    return this.mergeConfigs(
      this.config,
      this.remoteConfig || {},
      this.userOverrides
    );
  }

  /**
   * Get a specific config value by path
   */
  get<T>(path: string): T {
    const config = this.getConfig();
    return path.split('.').reduce((obj: any, key) => obj?.[key], config) as T;
  }

  /**
   * Check if a feature is enabled
   */
  isFeatureEnabled(feature: keyof FeatureFlags): boolean {
    return this.get<boolean>(`features.${feature}`);
  }

  /**
   * Set a user override
   */
  set<K extends keyof AppConfig>(
    section: K,
    key: keyof AppConfig[K],
    value: any
  ): void {
    if (!this.userOverrides[section]) {
      this.userOverrides[section] = {} as any;
    }
    (this.userOverrides[section] as any)[key] = value;
    this.saveUserOverrides();
    this.notifyListeners();
  }

  /**
   * Set a feature flag override
   */
  setFeature(feature: keyof FeatureFlags, enabled: boolean): void {
    if (!this.userOverrides.features) {
      this.userOverrides.features = {} as FeatureFlags;
    }
    (this.userOverrides.features as any)[feature] = enabled;
    this.saveUserOverrides();
    this.notifyListeners();
  }

  /**
   * Reset user overrides
   */
  resetOverrides(): void {
    this.userOverrides = {};
    this.saveUserOverrides();
    this.notifyListeners();
  }

  /**
   * Subscribe to config changes
   */
  subscribe(listener: (config: AppConfig) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Get experiment variant for user
   */
  getExperimentVariant(experimentId: string): ExperimentVariant | null {
    const experiment = this.config.experiments.experiments.find(
      e => e.id === experimentId
    );

    if (!experiment || !experiment.enabled) return null;

    // Check date constraints
    const now = new Date();
    if (experiment.startDate && now < experiment.startDate) return null;
    if (experiment.endDate && now > experiment.endDate) return null;

    // Get or assign variant
    let variantId = this.experimentAssignments.get(experimentId);

    if (!variantId) {
      variantId = this.assignVariant(experiment);
      this.experimentAssignments.set(experimentId, variantId);
      this.saveExperimentAssignments();
    }

    return experiment.variants.find(v => v.id === variantId) || null;
  }

  /**
   * Check if user is in experiment variant
   */
  isInVariant(experimentId: string, variantId: string): boolean {
    const variant = this.getExperimentVariant(experimentId);
    return variant?.id === variantId;
  }

  private assignVariant(experiment: Experiment): string {
    // Use user ID or random for consistent assignment
    const userId = this.config.experiments.userId || crypto.randomUUID();
    const hash = this.hashString(`${experiment.id}:${userId}`);
    const normalized = hash / 0xFFFFFFFF;

    // Check if user is in experiment at all
    if (normalized > experiment.allocation) {
      return 'control';
    }

    // Assign to variant based on weights
    let cumulative = 0;
    const variantRoll = Math.random();

    for (const variant of experiment.variants) {
      cumulative += variant.weight;
      if (variantRoll <= cumulative) {
        return variant.id;
      }
    }

    return experiment.variants[0]?.id || 'control';
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  private async syncRemoteConfig(url: string): Promise<void> {
    try {
      const response = await fetch(url, {
        headers: { 'Cache-Control': 'no-cache' },
      });

      if (response.ok) {
        this.remoteConfig = await response.json();
        this.notifyListeners();
      }
    } catch (error) {
      console.warn('[Config] Failed to sync remote config:', error);
    }
  }

  private startAutoSync(url: string): void {
    // Sync every 5 minutes
    this.syncInterval = setInterval(() => {
      this.syncRemoteConfig(url);
    }, 5 * 60 * 1000);
  }

  private loadUserOverrides(): void {
    try {
      const stored = localStorage.getItem('mycelix:config:overrides');
      if (stored) {
        this.userOverrides = JSON.parse(stored);
      }
    } catch {
      // Ignore
    }
  }

  private saveUserOverrides(): void {
    try {
      localStorage.setItem(
        'mycelix:config:overrides',
        JSON.stringify(this.userOverrides)
      );
    } catch {
      // Ignore
    }
  }

  private loadExperimentAssignments(): void {
    try {
      const stored = localStorage.getItem('mycelix:experiments');
      if (stored) {
        const assignments = JSON.parse(stored);
        this.experimentAssignments = new Map(Object.entries(assignments));
      }
    } catch {
      // Ignore
    }
  }

  private saveExperimentAssignments(): void {
    try {
      const assignments = Object.fromEntries(this.experimentAssignments);
      localStorage.setItem('mycelix:experiments', JSON.stringify(assignments));
    } catch {
      // Ignore
    }
  }

  private mergeConfigs(...configs: Partial<AppConfig>[]): AppConfig {
    const result = { ...defaultConfig };

    for (const config of configs) {
      for (const [section, values] of Object.entries(config)) {
        if (values && typeof values === 'object') {
          (result as any)[section] = {
            ...(result as any)[section],
            ...values,
          };
        }
      }
    }

    return result;
  }

  private notifyListeners(): void {
    const config = this.getConfig();
    for (const listener of this.listeners) {
      listener(config);
    }
  }

  dispose(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }
    this.listeners.clear();
  }
}

// ==================== React Hook (if using React) ====================

export function createConfigHook(manager: ConfigManager) {
  return function useConfig<T>(path?: string): T {
    // This would use React's useState and useEffect
    // Simplified version for reference
    if (path) {
      return manager.get<T>(path);
    }
    return manager.getConfig() as unknown as T;
  };
}

export function createFeatureFlagHook(manager: ConfigManager) {
  return function useFeatureFlag(feature: keyof FeatureFlags): boolean {
    return manager.isFeatureEnabled(feature);
  };
}

export function createExperimentHook(manager: ConfigManager) {
  return function useExperiment(experimentId: string): ExperimentVariant | null {
    return manager.getExperimentVariant(experimentId);
  };
}

// ==================== Singleton ====================

let configManager: ConfigManager | null = null;

export function getConfigManager(environment?: Environment): ConfigManager {
  if (!configManager) {
    configManager = new ConfigManager(environment);
  }
  return configManager;
}

export default {
  ConfigManager,
  getConfigManager,
  createConfigHook,
  createFeatureFlagHook,
  createExperimentHook,
};
