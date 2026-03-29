// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Configuration Module
 *
 * Centralized configuration management for the Mycelix SDK.
 * Supports environment-based configuration, validation, and type-safe access.
 */

import { validate, ValidationError } from '../errors.js';

// ============================================================================
// Core Configuration Types
// ============================================================================

/**
 * MATL (Trust Layer) configuration
 */
export interface MATLConfig {
  /** Default trust threshold for Byzantine detection (0-1) */
  defaultTrustThreshold: number;
  /** Maximum Byzantine tolerance (0-0.34, 34% validated) */
  maxByzantineTolerance: number;
  /** Default window size for adaptive thresholds */
  adaptiveWindowSize: number;
  /** Sigma multiplier for anomaly detection */
  sigmaMultiplier: number;
  /** Laplace smoothing prior for reputation */
  laplacePrior: number;
}

/**
 * Federated Learning configuration
 */
export interface FLConfig {
  /** Minimum participants required for aggregation */
  minParticipants: number;
  /** Maximum participants per round */
  maxParticipants: number;
  /** Round timeout in milliseconds */
  roundTimeout: number;
  /** Default Byzantine tolerance for FL */
  byzantineTolerance: number;
  /** Default aggregation method */
  defaultAggregationMethod: 'fedavg' | 'trimmed_mean' | 'median' | 'krum' | 'trust_weighted';
  /** Trust threshold for participant inclusion */
  trustThreshold: number;
}

/**
 * Security configuration
 */
export interface SecurityConfig {
  /** Default hash algorithm */
  hashAlgorithm: 'SHA-256' | 'SHA-384' | 'SHA-512';
  /** Default rate limit max requests */
  rateLimitMaxRequests: number;
  /** Default rate limit window in ms */
  rateLimitWindowMs: number;
  /** Enable constant-time operations by default */
  enableConstantTime: boolean;
  /** Default secret TTL in ms (0 = no expiration) */
  defaultSecretTTL: number;
  /** Max audit log size */
  maxAuditLogSize: number;
  /** Max JSON nesting depth for sanitization */
  maxJsonDepth: number;
  /** Max JSON size in bytes */
  maxJsonSize: number;
}

/**
 * Bridge configuration
 */
export interface BridgeConfig {
  /** Default message TTL in ms */
  defaultMessageTTL: number;
  /** Max message queue size */
  maxQueueSize: number;
  /** Enable cross-hApp reputation */
  enableCrossHappReputation: boolean;
  /** Default weight for reputation aggregation */
  defaultReputationWeight: number;
}

/**
 * Client configuration
 */
export interface ClientConfig {
  /** Connection timeout in ms */
  connectionTimeout: number;
  /** Request timeout in ms */
  requestTimeout: number;
  /** Enable automatic reconnection */
  autoReconnect: boolean;
  /** Max reconnection attempts */
  maxReconnectAttempts: number;
  /** Reconnection delay in ms */
  reconnectDelay: number;
  /** Enable request batching */
  enableBatching: boolean;
  /** Batch size limit */
  maxBatchSize: number;
  /** Batch timeout in ms */
  batchTimeout: number;
}

/**
 * Logging configuration
 */
export interface LogConfig {
  /** Log level */
  level: 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'silent';
  /** Enable structured logging */
  structured: boolean;
  /** Include timestamps */
  timestamps: boolean;
  /** Custom logger function */
  customLogger?: (level: string, message: string, context?: unknown) => void;
}

/**
 * Complete SDK configuration
 */
export interface MycelixConfig {
  matl: MATLConfig;
  fl: FLConfig;
  security: SecurityConfig;
  bridge: BridgeConfig;
  client: ClientConfig;
  logging: LogConfig;
}

// ============================================================================
// Default Configurations
// ============================================================================

export const DEFAULT_MATL_CONFIG: MATLConfig = {
  defaultTrustThreshold: 0.5,
  maxByzantineTolerance: 0.34,
  adaptiveWindowSize: 100,
  sigmaMultiplier: 2.0,
  laplacePrior: 1,
};

export const DEFAULT_FL_CONFIG: FLConfig = {
  minParticipants: 3,
  maxParticipants: 100,
  roundTimeout: 60000,
  byzantineTolerance: 0.33,
  defaultAggregationMethod: 'trust_weighted',
  trustThreshold: 0.5,
};

export const DEFAULT_SECURITY_CONFIG: SecurityConfig = {
  hashAlgorithm: 'SHA-256',
  rateLimitMaxRequests: 100,
  rateLimitWindowMs: 60000,
  enableConstantTime: true,
  defaultSecretTTL: 0,
  maxAuditLogSize: 10000,
  maxJsonDepth: 10,
  maxJsonSize: 1048576, // 1MB
};

export const DEFAULT_BRIDGE_CONFIG: BridgeConfig = {
  defaultMessageTTL: 30000,
  maxQueueSize: 1000,
  enableCrossHappReputation: true,
  defaultReputationWeight: 1.0,
};

export const DEFAULT_CLIENT_CONFIG: ClientConfig = {
  connectionTimeout: 30000,
  requestTimeout: 10000,
  autoReconnect: true,
  maxReconnectAttempts: 5,
  reconnectDelay: 1000,
  enableBatching: false,
  maxBatchSize: 10,
  batchTimeout: 50,
};

export const DEFAULT_LOG_CONFIG: LogConfig = {
  level: 'info',
  structured: false,
  timestamps: true,
};

export const DEFAULT_CONFIG: MycelixConfig = {
  matl: DEFAULT_MATL_CONFIG,
  fl: DEFAULT_FL_CONFIG,
  security: DEFAULT_SECURITY_CONFIG,
  bridge: DEFAULT_BRIDGE_CONFIG,
  client: DEFAULT_CLIENT_CONFIG,
  logging: DEFAULT_LOG_CONFIG,
};

// ============================================================================
// Configuration Validation
// ============================================================================

/**
 * Validate MATL configuration
 */
export function validateMATLConfig(config: Partial<MATLConfig>): void {
  const v = validate();

  if (config.defaultTrustThreshold !== undefined) {
    v.inRange('defaultTrustThreshold', config.defaultTrustThreshold, 0, 1);
  }
  if (config.maxByzantineTolerance !== undefined) {
    v.inRange('maxByzantineTolerance', config.maxByzantineTolerance, 0, 0.34);
  }
  if (config.adaptiveWindowSize !== undefined) {
    v.positive('adaptiveWindowSize', config.adaptiveWindowSize);
  }
  if (config.sigmaMultiplier !== undefined) {
    v.positive('sigmaMultiplier', config.sigmaMultiplier);
  }
  if (config.laplacePrior !== undefined) {
    v.positive('laplacePrior', config.laplacePrior);
  }

  v.throwIfInvalid();
}

/**
 * Validate FL configuration
 */
export function validateFLConfig(config: Partial<FLConfig>): void {
  const v = validate();

  if (config.minParticipants !== undefined) {
    v.positive('minParticipants', config.minParticipants);
  }
  if (config.maxParticipants !== undefined) {
    v.positive('maxParticipants', config.maxParticipants);
  }
  if (config.roundTimeout !== undefined) {
    v.positive('roundTimeout', config.roundTimeout);
  }
  if (config.byzantineTolerance !== undefined) {
    v.inRange('byzantineTolerance', config.byzantineTolerance, 0, 0.34);
  }
  if (config.trustThreshold !== undefined) {
    v.inRange('trustThreshold', config.trustThreshold, 0, 1);
  }

  v.throwIfInvalid();

  // Cross-field validation
  if (
    config.minParticipants !== undefined &&
    config.maxParticipants !== undefined &&
    config.minParticipants > config.maxParticipants
  ) {
    throw new ValidationError(
      'minParticipants',
      'must not exceed maxParticipants',
      config.minParticipants
    );
  }
}

/**
 * Validate Security configuration
 */
export function validateSecurityConfig(config: Partial<SecurityConfig>): void {
  const v = validate();

  if (config.rateLimitMaxRequests !== undefined) {
    v.positive('rateLimitMaxRequests', config.rateLimitMaxRequests);
  }
  if (config.rateLimitWindowMs !== undefined) {
    v.positive('rateLimitWindowMs', config.rateLimitWindowMs);
  }
  if (config.maxAuditLogSize !== undefined) {
    v.positive('maxAuditLogSize', config.maxAuditLogSize);
  }
  if (config.maxJsonDepth !== undefined) {
    v.positive('maxJsonDepth', config.maxJsonDepth);
  }
  if (config.maxJsonSize !== undefined) {
    v.positive('maxJsonSize', config.maxJsonSize);
  }

  v.throwIfInvalid();
}

/**
 * Validate complete configuration
 */
export function validateConfig(config: Partial<MycelixConfig>): void {
  if (config.matl) validateMATLConfig(config.matl);
  if (config.fl) validateFLConfig(config.fl);
  if (config.security) validateSecurityConfig(config.security);
}

// ============================================================================
// Configuration Manager
// ============================================================================

/**
 * Configuration manager for the Mycelix SDK
 */
export class ConfigManager {
  private config: MycelixConfig;
  private frozen: boolean = false;
  private changeListeners: Array<(config: MycelixConfig) => void> = [];

  constructor(initialConfig: Partial<MycelixConfig> = {}) {
    this.config = this.mergeConfig(DEFAULT_CONFIG, initialConfig);
    validateConfig(this.config);
  }

  /**
   * Get current configuration
   */
  get(): MycelixConfig {
    return this.frozen ? Object.freeze({ ...this.config }) : { ...this.config };
  }

  /**
   * Get a specific module's configuration
   */
  getModule<K extends keyof MycelixConfig>(module: K): MycelixConfig[K] {
    return { ...this.config[module] };
  }

  /**
   * Update configuration
   */
  set(updates: Partial<MycelixConfig>): void {
    if (this.frozen) {
      throw new Error('Configuration is frozen and cannot be modified');
    }

    const newConfig = this.mergeConfig(this.config, updates);
    validateConfig(newConfig);
    this.config = newConfig;
    this.notifyListeners();
  }

  /**
   * Update a specific module's configuration
   */
  setModule<K extends keyof MycelixConfig>(
    module: K,
    updates: Partial<MycelixConfig[K]>
  ): void {
    this.set({ [module]: { ...this.config[module], ...updates } } as Partial<MycelixConfig>);
  }

  /**
   * Reset configuration to defaults
   */
  reset(): void {
    if (this.frozen) {
      throw new Error('Configuration is frozen and cannot be modified');
    }

    this.config = { ...DEFAULT_CONFIG };
    this.notifyListeners();
  }

  /**
   * Reset a specific module to defaults
   */
  resetModule<K extends keyof MycelixConfig>(module: K): void {
    this.setModule(module, DEFAULT_CONFIG[module]);
  }

  /**
   * Freeze configuration (prevents further changes)
   */
  freeze(): void {
    this.frozen = true;
  }

  /**
   * Check if configuration is frozen
   */
  isFrozen(): boolean {
    return this.frozen;
  }

  /**
   * Register a change listener
   */
  onChange(listener: (config: MycelixConfig) => void): () => void {
    this.changeListeners.push(listener);
    return () => {
      const index = this.changeListeners.indexOf(listener);
      if (index >= 0) {
        this.changeListeners.splice(index, 1);
      }
    };
  }

  /**
   * Load configuration from environment variables
   */
  loadFromEnv(prefix: string = 'MYCELIX_'): void {
    if (this.frozen) {
      throw new Error('Configuration is frozen and cannot be modified');
    }

    const env = typeof process !== 'undefined' ? process.env : {};
    const updates: Partial<MycelixConfig> = {};

    // MATL config
    if (env[`${prefix}MATL_TRUST_THRESHOLD`]) {
      updates.matl = {
        ...this.config.matl,
        defaultTrustThreshold: parseFloat(env[`${prefix}MATL_TRUST_THRESHOLD`]!),
      };
    }
    if (env[`${prefix}MATL_BYZANTINE_TOLERANCE`]) {
      updates.matl = {
        ...updates.matl,
        ...this.config.matl,
        maxByzantineTolerance: parseFloat(env[`${prefix}MATL_BYZANTINE_TOLERANCE`]!),
      };
    }

    // FL config
    if (env[`${prefix}FL_MIN_PARTICIPANTS`]) {
      updates.fl = {
        ...this.config.fl,
        minParticipants: parseInt(env[`${prefix}FL_MIN_PARTICIPANTS`]!, 10),
      };
    }
    if (env[`${prefix}FL_MAX_PARTICIPANTS`]) {
      updates.fl = {
        ...updates.fl,
        ...this.config.fl,
        maxParticipants: parseInt(env[`${prefix}FL_MAX_PARTICIPANTS`]!, 10),
      };
    }
    if (env[`${prefix}FL_ROUND_TIMEOUT`]) {
      updates.fl = {
        ...updates.fl,
        ...this.config.fl,
        roundTimeout: parseInt(env[`${prefix}FL_ROUND_TIMEOUT`]!, 10),
      };
    }

    // Security config
    if (env[`${prefix}SECURITY_RATE_LIMIT_MAX`]) {
      updates.security = {
        ...this.config.security,
        rateLimitMaxRequests: parseInt(env[`${prefix}SECURITY_RATE_LIMIT_MAX`]!, 10),
      };
    }
    if (env[`${prefix}SECURITY_RATE_LIMIT_WINDOW`]) {
      updates.security = {
        ...updates.security,
        ...this.config.security,
        rateLimitWindowMs: parseInt(env[`${prefix}SECURITY_RATE_LIMIT_WINDOW`]!, 10),
      };
    }

    // Logging config
    if (env[`${prefix}LOG_LEVEL`]) {
      const level = env[`${prefix}LOG_LEVEL`]! as LogConfig['level'];
      if (['debug', 'info', 'warn', 'error', 'silent'].includes(level)) {
        updates.logging = {
          ...this.config.logging,
          level,
        };
      }
    }

    if (Object.keys(updates).length > 0) {
      this.set(updates);
    }
  }

  /**
   * Export configuration as JSON
   */
  toJSON(): string {
    return JSON.stringify(this.config, null, 2);
  }

  /**
   * Import configuration from JSON
   */
  fromJSON(json: string): void {
    const parsed = JSON.parse(json) as Partial<MycelixConfig>;
    this.set(parsed);
  }

  /**
   * Deep merge configurations
   */
  private mergeConfig(
    base: MycelixConfig,
    overrides: Partial<MycelixConfig>
  ): MycelixConfig {
    return {
      matl: { ...base.matl, ...overrides.matl },
      fl: { ...base.fl, ...overrides.fl },
      security: { ...base.security, ...overrides.security },
      bridge: { ...base.bridge, ...overrides.bridge },
      client: { ...base.client, ...overrides.client },
      logging: { ...base.logging, ...overrides.logging },
    };
  }

  /**
   * Notify change listeners
   */
  private notifyListeners(): void {
    const config = this.get();
    for (const listener of this.changeListeners) {
      try {
        listener(config);
      } catch (e) {
        // Ignore listener errors
      }
    }
  }
}

// ============================================================================
// Global Configuration Instance
// ============================================================================

let globalConfig: ConfigManager | null = null;

/**
 * Get or create the global configuration manager
 */
export function getConfig(): ConfigManager {
  if (!globalConfig) {
    globalConfig = new ConfigManager();
  }
  return globalConfig;
}

/**
 * Initialize global configuration with custom settings
 */
export function initConfig(config: Partial<MycelixConfig>): ConfigManager {
  globalConfig = new ConfigManager(config);
  return globalConfig;
}

/**
 * Reset global configuration
 */
export function resetGlobalConfig(): void {
  globalConfig = null;
}

// ============================================================================
// Configuration Presets
// ============================================================================

/**
 * Development configuration preset
 */
export const DEV_CONFIG: Partial<MycelixConfig> = {
  fl: {
    ...DEFAULT_FL_CONFIG,
    minParticipants: 2,
    roundTimeout: 120000,
  },
  security: {
    ...DEFAULT_SECURITY_CONFIG,
    rateLimitMaxRequests: 1000,
    rateLimitWindowMs: 10000,
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'debug',
  },
};

/**
 * Production configuration preset
 */
export const PROD_CONFIG: Partial<MycelixConfig> = {
  fl: {
    ...DEFAULT_FL_CONFIG,
    minParticipants: 5,
    roundTimeout: 30000,
  },
  security: {
    ...DEFAULT_SECURITY_CONFIG,
    rateLimitMaxRequests: 50,
    rateLimitWindowMs: 60000,
    enableConstantTime: true,
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'warn',
    structured: true,
  },
};

/**
 * Test configuration preset
 */
export const TEST_CONFIG: Partial<MycelixConfig> = {
  fl: {
    ...DEFAULT_FL_CONFIG,
    minParticipants: 2,
    maxParticipants: 10,
    roundTimeout: 5000,
  },
  security: {
    ...DEFAULT_SECURITY_CONFIG,
    rateLimitMaxRequests: 10000,
    rateLimitWindowMs: 1000,
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 5000,
    requestTimeout: 2000,
    autoReconnect: false,
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'silent',
  },
};

/**
 * High-security configuration preset
 */
export const HIGH_SECURITY_CONFIG: Partial<MycelixConfig> = {
  matl: {
    ...DEFAULT_MATL_CONFIG,
    defaultTrustThreshold: 0.7,
    maxByzantineTolerance: 0.33,
  },
  fl: {
    ...DEFAULT_FL_CONFIG,
    byzantineTolerance: 0.25,
    trustThreshold: 0.7,
    defaultAggregationMethod: 'krum',
  },
  security: {
    ...DEFAULT_SECURITY_CONFIG,
    hashAlgorithm: 'SHA-512',
    rateLimitMaxRequests: 20,
    rateLimitWindowMs: 60000,
    enableConstantTime: true,
    defaultSecretTTL: 300000, // 5 minutes
    maxJsonDepth: 5,
  },
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create a configuration preset
 */
export function createPreset(
  name: string,
  config: Partial<MycelixConfig>
): { name: string; config: Partial<MycelixConfig>; apply: () => ConfigManager } {
  return {
    name,
    config,
    apply: () => initConfig(config),
  };
}

/**
 * Merge multiple configuration presets
 */
export function mergePresets(
  ...presets: Array<Partial<MycelixConfig>>
): Partial<MycelixConfig> {
  return presets.reduce((acc, preset) => {
    return {
      matl: { ...acc.matl, ...preset.matl },
      fl: { ...acc.fl, ...preset.fl },
      security: { ...acc.security, ...preset.security },
      bridge: { ...acc.bridge, ...preset.bridge },
      client: { ...acc.client, ...preset.client },
      logging: { ...acc.logging, ...preset.logging },
    } as Partial<MycelixConfig>;
  }, {} as Partial<MycelixConfig>);
}

// ============================================================================
// Preset Validation
// ============================================================================

/**
 * Validation warning for preset configuration
 */
export interface PresetValidationWarning {
  /** Warning code */
  code: string;
  /** Human-readable message */
  message: string;
  /** Suggested fix */
  suggestion?: string;
  /** Severity level */
  severity: 'info' | 'warning' | 'error';
}

/**
 * Result of preset validation
 */
export interface PresetValidationResult {
  /** Whether the preset is valid for intended use */
  valid: boolean;
  /** Warnings found during validation */
  warnings: PresetValidationWarning[];
  /** Preset type detected */
  detectedType: 'dev' | 'prod' | 'test' | 'high-security' | 'custom';
}

/**
 * Validate a development configuration preset
 */
export function validateDevPreset(config: Partial<MycelixConfig>): PresetValidationWarning[] {
  const warnings: PresetValidationWarning[] = [];

  // Dev should have higher rate limits
  if ((config.security?.rateLimitMaxRequests ?? 1000) < 100) {
    warnings.push({
      code: 'DEV_LOW_RATE_LIMIT',
      message: 'Rate limit seems low for development',
      suggestion: 'Consider increasing rateLimitMaxRequests for easier testing',
      severity: 'info',
    });
  }

  // Dev should have debug logging
  if (config.logging?.level !== 'debug' && config.logging?.level !== 'trace') {
    warnings.push({
      code: 'DEV_NO_DEBUG_LOGGING',
      message: 'Debug logging not enabled for development',
      suggestion: 'Set logging.level to "debug" for better debugging',
      severity: 'info',
    });
  }

  return warnings;
}

/**
 * Validate a production configuration preset
 */
export function validateProdPreset(config: Partial<MycelixConfig>): PresetValidationWarning[] {
  const warnings: PresetValidationWarning[] = [];

  // Production should have enough participants
  if ((config.fl?.minParticipants ?? 3) < 5) {
    warnings.push({
      code: 'PROD_LOW_PARTICIPANTS',
      message: `FL requires only ${config.fl?.minParticipants ?? 3} participants`,
      suggestion: 'Consider requiring at least 5 participants for Byzantine resilience',
      severity: 'warning',
    });
  }

  // Production should have reasonable rate limits
  if ((config.security?.rateLimitMaxRequests ?? 100) > 500) {
    warnings.push({
      code: 'PROD_HIGH_RATE_LIMIT',
      message: 'Rate limit seems high for production',
      suggestion: 'Consider lowering rateLimitMaxRequests to prevent abuse',
      severity: 'warning',
    });
  }

  // Production should use constant time
  if (config.security?.enableConstantTime === false) {
    warnings.push({
      code: 'PROD_NO_CONSTANT_TIME',
      message: 'Constant-time operations disabled',
      suggestion: 'Enable constant-time operations to prevent timing attacks',
      severity: 'error',
    });
  }

  // Production should not use debug logging
  if (config.logging?.level === 'debug' || config.logging?.level === 'trace') {
    warnings.push({
      code: 'PROD_DEBUG_LOGGING',
      message: 'Debug logging enabled in production',
      suggestion: 'Use "info" or "warn" level in production for performance',
      severity: 'warning',
    });
  }

  return warnings;
}

/**
 * Validate a test configuration preset
 */
export function validateTestPreset(config: Partial<MycelixConfig>): PresetValidationWarning[] {
  const warnings: PresetValidationWarning[] = [];

  // Test should have short timeouts
  if ((config.fl?.roundTimeout ?? 30000) > 30000) {
    warnings.push({
      code: 'TEST_LONG_TIMEOUT',
      message: 'Round timeout is long for testing',
      suggestion: 'Consider shorter roundTimeout for faster tests',
      severity: 'info',
    });
  }

  return warnings;
}

/**
 * Validate a high-security configuration preset
 */
export function validateHighSecurityPreset(config: Partial<MycelixConfig>): PresetValidationWarning[] {
  const warnings: PresetValidationWarning[] = [];

  // High security should use strong hash
  if (config.security?.hashAlgorithm !== 'SHA-512') {
    warnings.push({
      code: 'SEC_WEAK_HASH',
      message: `Using ${config.security?.hashAlgorithm ?? 'SHA-256'} instead of SHA-512`,
      suggestion: 'Use SHA-512 for high-security scenarios',
      severity: 'warning',
    });
  }

  // High security should have short secret TTL
  if ((config.security?.defaultSecretTTL ?? 3600000) > 600000) {
    warnings.push({
      code: 'SEC_LONG_SECRET_TTL',
      message: 'Secret TTL is long for high-security',
      suggestion: 'Consider shorter defaultSecretTTL (≤10 minutes)',
      severity: 'warning',
    });
  }

  // High security should have low Byzantine tolerance
  if ((config.fl?.byzantineTolerance ?? 0.33) > 0.33) {
    warnings.push({
      code: 'SEC_HIGH_BYZANTINE_TOLERANCE',
      message: 'Byzantine tolerance is high for high-security',
      suggestion: 'Consider ≤33% byzantineTolerance',
      severity: 'warning',
    });
  }

  return warnings;
}

/**
 * Detect the type of a configuration preset
 */
export function detectPresetType(config: Partial<MycelixConfig>): PresetValidationResult['detectedType'] {
  // Check for high-security indicators
  if (
    config.security?.hashAlgorithm === 'SHA-512' ||
    (config.fl?.byzantineTolerance ?? 0.33) <= 0.25 ||
    (config.security?.rateLimitMaxRequests ?? 100) <= 30
  ) {
    return 'high-security';
  }

  // Check for test indicators
  if (
    config.logging?.level === 'silent' ||
    config.client?.autoReconnect === false ||
    (config.security?.rateLimitMaxRequests ?? 100) >= 5000
  ) {
    return 'test';
  }

  // Check for dev indicators
  if (
    config.logging?.level === 'debug' ||
    (config.security?.rateLimitMaxRequests ?? 100) >= 500 ||
    (config.fl?.minParticipants ?? 3) <= 2
  ) {
    return 'dev';
  }

  // Check for prod indicators
  if (
    (config.logging?.level === 'warn' || config.logging?.level === 'error') ||
    config.security?.enableConstantTime === true ||
    (config.fl?.minParticipants ?? 3) >= 5
  ) {
    return 'prod';
  }

  return 'custom';
}

/**
 * Validate a configuration preset and get warnings
 */
export function validatePreset(
  config: Partial<MycelixConfig>,
  intendedType?: 'dev' | 'prod' | 'test' | 'high-security'
): PresetValidationResult {
  const detectedType = detectPresetType(config);
  const type = intendedType ?? detectedType;

  let warnings: PresetValidationWarning[] = [];

  switch (type) {
    case 'dev':
      warnings = validateDevPreset(config);
      break;
    case 'prod':
      warnings = validateProdPreset(config);
      break;
    case 'test':
      warnings = validateTestPreset(config);
      break;
    case 'high-security':
      warnings = validateHighSecurityPreset(config);
      break;
  }

  // Check if intended type matches detected type
  if (intendedType && intendedType !== detectedType) {
    warnings.unshift({
      code: 'TYPE_MISMATCH',
      message: `Config appears to be ${detectedType} but intended as ${intendedType}`,
      suggestion: `Review configuration to match ${intendedType} requirements`,
      severity: 'warning',
    });
  }

  const hasErrors = warnings.some((w) => w.severity === 'error');

  return {
    valid: !hasErrors,
    warnings,
    detectedType,
  };
}

/**
 * Select a validated preset by name
 */
export function selectPreset(
  name: 'dev' | 'prod' | 'test' | 'high-security'
): { config: Partial<MycelixConfig>; validation: PresetValidationResult } {
  const presets: Record<string, Partial<MycelixConfig>> = {
    dev: DEV_CONFIG,
    prod: PROD_CONFIG,
    test: TEST_CONFIG,
    'high-security': HIGH_SECURITY_CONFIG,
  };

  const config = presets[name];
  const validation = validatePreset(config, name);

  return { config, validation };
}

// ============================================================================
// Civic Configuration Exports (GovOS)
// ============================================================================

export * from './civic.js';
