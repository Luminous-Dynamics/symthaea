// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Configuration Module Tests
 *
 * Tests for configuration management, validation, and presets.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as config from '../src/config/index.js';

describe('Configuration Module', () => {
  beforeEach(() => {
    config.resetGlobalConfig();
  });

  afterEach(() => {
    config.resetGlobalConfig();
  });

  // ============================================================================
  // Default Configuration Tests
  // ============================================================================

  describe('Default Configuration', () => {
    it('should have valid default MATL config', () => {
      const matl = config.DEFAULT_MATL_CONFIG;

      expect(matl.defaultTrustThreshold).toBe(0.5);
      expect(matl.maxByzantineTolerance).toBe(0.34);
      expect(matl.adaptiveWindowSize).toBe(100);
      expect(matl.sigmaMultiplier).toBe(2.0);
      expect(matl.laplacePrior).toBe(1);
    });

    it('should have valid default FL config', () => {
      const fl = config.DEFAULT_FL_CONFIG;

      expect(fl.minParticipants).toBe(3);
      expect(fl.maxParticipants).toBe(100);
      expect(fl.roundTimeout).toBe(60000);
      expect(fl.byzantineTolerance).toBe(0.33);
      expect(fl.defaultAggregationMethod).toBe('trust_weighted');
      expect(fl.trustThreshold).toBe(0.5);
    });

    it('should have valid default Security config', () => {
      const security = config.DEFAULT_SECURITY_CONFIG;

      expect(security.hashAlgorithm).toBe('SHA-256');
      expect(security.rateLimitMaxRequests).toBe(100);
      expect(security.rateLimitWindowMs).toBe(60000);
      expect(security.enableConstantTime).toBe(true);
      expect(security.maxAuditLogSize).toBe(10000);
    });

    it('should have valid default Bridge config', () => {
      const bridge = config.DEFAULT_BRIDGE_CONFIG;

      expect(bridge.defaultMessageTTL).toBe(30000);
      expect(bridge.maxQueueSize).toBe(1000);
      expect(bridge.enableCrossHappReputation).toBe(true);
    });

    it('should have valid default Client config', () => {
      const client = config.DEFAULT_CLIENT_CONFIG;

      expect(client.connectionTimeout).toBe(30000);
      expect(client.requestTimeout).toBe(10000);
      expect(client.autoReconnect).toBe(true);
      expect(client.maxReconnectAttempts).toBe(5);
    });

    it('should have valid default Log config', () => {
      const logging = config.DEFAULT_LOG_CONFIG;

      expect(logging.level).toBe('info');
      expect(logging.structured).toBe(false);
      expect(logging.timestamps).toBe(true);
    });
  });

  // ============================================================================
  // ConfigManager Tests
  // ============================================================================

  describe('ConfigManager', () => {
    it('should create with default configuration', () => {
      const manager = new config.ConfigManager();
      const current = manager.get();

      expect(current.matl).toEqual(config.DEFAULT_MATL_CONFIG);
      expect(current.fl).toEqual(config.DEFAULT_FL_CONFIG);
      expect(current.security).toEqual(config.DEFAULT_SECURITY_CONFIG);
    });

    it('should accept initial configuration', () => {
      const manager = new config.ConfigManager({
        matl: { ...config.DEFAULT_MATL_CONFIG, defaultTrustThreshold: 0.7 },
      });

      expect(manager.get().matl.defaultTrustThreshold).toBe(0.7);
    });

    it('should update configuration', () => {
      const manager = new config.ConfigManager();

      manager.set({
        fl: { ...config.DEFAULT_FL_CONFIG, minParticipants: 5 },
      });

      expect(manager.get().fl.minParticipants).toBe(5);
    });

    it('should update specific module configuration', () => {
      const manager = new config.ConfigManager();

      manager.setModule('security', { rateLimitMaxRequests: 200 });

      expect(manager.get().security.rateLimitMaxRequests).toBe(200);
    });

    it('should get specific module configuration', () => {
      const manager = new config.ConfigManager();
      const matl = manager.getModule('matl');

      expect(matl).toEqual(config.DEFAULT_MATL_CONFIG);
    });

    it('should reset configuration to defaults', () => {
      const manager = new config.ConfigManager();

      manager.set({
        matl: { ...config.DEFAULT_MATL_CONFIG, defaultTrustThreshold: 0.9 },
      });

      manager.reset();

      expect(manager.get().matl.defaultTrustThreshold).toBe(0.5);
    });

    it('should reset specific module to defaults', () => {
      const manager = new config.ConfigManager();

      manager.setModule('fl', { minParticipants: 10 });
      manager.resetModule('fl');

      expect(manager.get().fl.minParticipants).toBe(3);
    });

    it('should freeze configuration', () => {
      const manager = new config.ConfigManager();

      manager.freeze();

      expect(manager.isFrozen()).toBe(true);
      expect(() => manager.set({ matl: { ...config.DEFAULT_MATL_CONFIG } })).toThrow(
        'Configuration is frozen'
      );
    });

    it('should call change listeners', () => {
      const manager = new config.ConfigManager();
      const listener = vi.fn();

      manager.onChange(listener);
      manager.set({
        logging: { ...config.DEFAULT_LOG_CONFIG, level: 'debug' },
      });

      expect(listener).toHaveBeenCalledTimes(1);
      expect(listener.mock.calls[0][0].logging.level).toBe('debug');
    });

    it('should unsubscribe change listeners', () => {
      const manager = new config.ConfigManager();
      const listener = vi.fn();

      const unsubscribe = manager.onChange(listener);
      unsubscribe();

      manager.set({
        logging: { ...config.DEFAULT_LOG_CONFIG, level: 'debug' },
      });

      expect(listener).not.toHaveBeenCalled();
    });

    it('should export and import JSON', () => {
      const manager = new config.ConfigManager();

      manager.setModule('matl', { defaultTrustThreshold: 0.8 });

      const json = manager.toJSON();
      const parsed = JSON.parse(json);

      expect(parsed.matl.defaultTrustThreshold).toBe(0.8);

      const newManager = new config.ConfigManager();
      newManager.fromJSON(json);

      expect(newManager.get().matl.defaultTrustThreshold).toBe(0.8);
    });
  });

  // ============================================================================
  // Validation Tests
  // ============================================================================

  describe('Configuration Validation', () => {
    it('should validate MATL trust threshold range', () => {
      expect(() =>
        config.validateMATLConfig({ defaultTrustThreshold: 1.5 })
      ).toThrow();
      expect(() =>
        config.validateMATLConfig({ defaultTrustThreshold: -0.1 })
      ).toThrow();
      expect(() =>
        config.validateMATLConfig({ defaultTrustThreshold: 0.5 })
      ).not.toThrow();
    });

    it('should validate MATL Byzantine tolerance range', () => {
      expect(() =>
        config.validateMATLConfig({ maxByzantineTolerance: 0.5 })
      ).toThrow();
      expect(() =>
        config.validateMATLConfig({ maxByzantineTolerance: 0.34 })
      ).not.toThrow();
    });

    it('should validate FL participant counts', () => {
      expect(() => config.validateFLConfig({ minParticipants: 0 })).toThrow();
      expect(() => config.validateFLConfig({ maxParticipants: 0 })).toThrow();
      expect(() =>
        config.validateFLConfig({ minParticipants: 5, maxParticipants: 3 })
      ).toThrow();
    });

    it('should validate Security rate limit config', () => {
      expect(() =>
        config.validateSecurityConfig({ rateLimitMaxRequests: 0 })
      ).toThrow();
      expect(() =>
        config.validateSecurityConfig({ rateLimitWindowMs: -1 })
      ).toThrow();
    });

    it('should validate complete configuration', () => {
      expect(() =>
        config.validateConfig({
          matl: { defaultTrustThreshold: 2.0 },
        })
      ).toThrow();

      expect(() =>
        config.validateConfig({
          matl: { defaultTrustThreshold: 0.5 },
          fl: { minParticipants: 3 },
        })
      ).not.toThrow();
    });

    it('should reject invalid config in ConfigManager', () => {
      expect(
        () =>
          new config.ConfigManager({
            matl: { ...config.DEFAULT_MATL_CONFIG, defaultTrustThreshold: 2.0 },
          })
      ).toThrow();
    });
  });

  // ============================================================================
  // Global Configuration Tests
  // ============================================================================

  describe('Global Configuration', () => {
    it('should get global config singleton', () => {
      const conf1 = config.getConfig();
      const conf2 = config.getConfig();

      expect(conf1).toBe(conf2);
    });

    it('should initialize global config with custom settings', () => {
      const manager = config.initConfig({
        matl: { ...config.DEFAULT_MATL_CONFIG, defaultTrustThreshold: 0.6 },
      });

      expect(manager.get().matl.defaultTrustThreshold).toBe(0.6);
      expect(config.getConfig().get().matl.defaultTrustThreshold).toBe(0.6);
    });

    it('should reset global config', () => {
      config.initConfig({
        matl: { ...config.DEFAULT_MATL_CONFIG, defaultTrustThreshold: 0.9 },
      });

      config.resetGlobalConfig();

      // After reset, getConfig should create a new instance
      const newConfig = config.getConfig();
      expect(newConfig.get().matl.defaultTrustThreshold).toBe(0.5);
    });
  });

  // ============================================================================
  // Configuration Presets Tests
  // ============================================================================

  describe('Configuration Presets', () => {
    it('should have valid DEV_CONFIG preset', () => {
      expect(config.DEV_CONFIG.fl?.minParticipants).toBe(2);
      expect(config.DEV_CONFIG.logging?.level).toBe('debug');
      expect(config.DEV_CONFIG.security?.rateLimitMaxRequests).toBe(1000);
    });

    it('should have valid PROD_CONFIG preset', () => {
      expect(config.PROD_CONFIG.fl?.minParticipants).toBe(5);
      expect(config.PROD_CONFIG.logging?.level).toBe('warn');
      expect(config.PROD_CONFIG.security?.rateLimitMaxRequests).toBe(50);
    });

    it('should have valid TEST_CONFIG preset', () => {
      expect(config.TEST_CONFIG.fl?.roundTimeout).toBe(5000);
      expect(config.TEST_CONFIG.logging?.level).toBe('silent');
      expect(config.TEST_CONFIG.client?.autoReconnect).toBe(false);
    });

    it('should have valid HIGH_SECURITY_CONFIG preset', () => {
      expect(config.HIGH_SECURITY_CONFIG.matl?.defaultTrustThreshold).toBe(0.7);
      expect(config.HIGH_SECURITY_CONFIG.fl?.defaultAggregationMethod).toBe('krum');
      expect(config.HIGH_SECURITY_CONFIG.security?.hashAlgorithm).toBe('SHA-512');
    });

    it('should create custom preset', () => {
      const myPreset = config.createPreset('my-preset', {
        matl: { ...config.DEFAULT_MATL_CONFIG, defaultTrustThreshold: 0.8 },
      });

      expect(myPreset.name).toBe('my-preset');

      const manager = myPreset.apply();
      expect(manager.get().matl.defaultTrustThreshold).toBe(0.8);
    });

    it('should merge multiple presets', () => {
      const merged = config.mergePresets(config.DEV_CONFIG, {
        security: { ...config.DEFAULT_SECURITY_CONFIG, rateLimitMaxRequests: 500 },
      });

      expect(merged.fl?.minParticipants).toBe(2); // From DEV_CONFIG
      expect(merged.security?.rateLimitMaxRequests).toBe(500); // From override
    });

    it('should validate DEV_CONFIG preset', () => {
      const result = config.validatePreset(config.DEV_CONFIG, 'dev');
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('dev');
    });

    it('should validate PROD_CONFIG preset', () => {
      const result = config.validatePreset(config.PROD_CONFIG, 'prod');
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('prod');
    });

    it('should validate TEST_CONFIG preset', () => {
      const result = config.validatePreset(config.TEST_CONFIG, 'test');
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('test');
    });

    it('should validate HIGH_SECURITY_CONFIG preset', () => {
      const result = config.validatePreset(config.HIGH_SECURITY_CONFIG, 'high-security');
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('high-security');
    });

    it('should detect config type from settings', () => {
      expect(config.detectPresetType({ logging: { level: 'debug' } })).toBe('dev');
      expect(config.detectPresetType({ logging: { level: 'silent' } })).toBe('test');
      expect(config.detectPresetType({ security: { hashAlgorithm: 'SHA-512' } })).toBe('high-security');
      expect(config.detectPresetType({ logging: { level: 'warn' } })).toBe('prod');
    });

    it('should warn when prod config has debug logging', () => {
      const badProdConfig = {
        logging: { level: 'debug' as const },
      };
      const result = config.validatePreset(badProdConfig, 'prod');
      expect(result.warnings.some((w) => w.code === 'PROD_DEBUG_LOGGING')).toBe(true);
    });

    it('should warn when high-security config uses weak hash', () => {
      const weakSecurityConfig = {
        security: { hashAlgorithm: 'SHA-256' as const },
      };
      const result = config.validatePreset(weakSecurityConfig, 'high-security');
      expect(result.warnings.some((w) => w.code === 'SEC_WEAK_HASH')).toBe(true);
    });

    it('should select preset by name with validation', () => {
      const { config: devConfig, validation } = config.selectPreset('dev');
      expect(devConfig).toBe(config.DEV_CONFIG);
      expect(validation.valid).toBe(true);
    });

    it('should detect type mismatch', () => {
      // A config that looks like dev but claimed as prod
      const mismatchedConfig = {
        logging: { level: 'debug' as const },
        security: { rateLimitMaxRequests: 1000 },
      };
      const result = config.validatePreset(mismatchedConfig, 'prod');
      expect(result.warnings.some((w) => w.code === 'TYPE_MISMATCH')).toBe(true);
    });
  });

  // ============================================================================
  // Environment Loading Tests
  // ============================================================================

  describe('Environment Loading', () => {
    it('should load config from environment variables', () => {
      const originalEnv = process.env;

      process.env = {
        ...originalEnv,
        MYCELIX_MATL_TRUST_THRESHOLD: '0.75',
        MYCELIX_FL_MIN_PARTICIPANTS: '5',
        MYCELIX_LOG_LEVEL: 'debug',
      };

      const manager = new config.ConfigManager();
      manager.loadFromEnv();

      expect(manager.get().matl.defaultTrustThreshold).toBe(0.75);
      expect(manager.get().fl.minParticipants).toBe(5);
      expect(manager.get().logging.level).toBe('debug');

      process.env = originalEnv;
    });

    it('should use custom prefix for env loading', () => {
      const originalEnv = process.env;

      process.env = {
        ...originalEnv,
        CUSTOM_MATL_TRUST_THRESHOLD: '0.9',
      };

      const manager = new config.ConfigManager();
      manager.loadFromEnv('CUSTOM_');

      expect(manager.get().matl.defaultTrustThreshold).toBe(0.9);

      process.env = originalEnv;
    });

    it('should not load env when frozen', () => {
      const manager = new config.ConfigManager();
      manager.freeze();

      expect(() => manager.loadFromEnv()).toThrow('Configuration is frozen');
    });
  });

  // ============================================================================
  // Integration Tests
  // ============================================================================

  describe('Configuration Integration', () => {
    it('should work with typical development workflow', () => {
      // Initialize with dev config
      const manager = config.initConfig(config.DEV_CONFIG);

      // Override specific settings
      manager.setModule('security', { rateLimitMaxRequests: 500 });

      // Verify merged settings
      const current = manager.get();
      expect(current.fl.minParticipants).toBe(2); // From DEV_CONFIG
      expect(current.security.rateLimitMaxRequests).toBe(500); // Override
      expect(current.logging.level).toBe('debug'); // From DEV_CONFIG
    });

    it('should work with production freeze pattern', () => {
      // Initialize production config
      const manager = config.initConfig(config.PROD_CONFIG);

      // Freeze before application starts
      manager.freeze();

      // Attempts to modify should fail
      expect(() =>
        manager.setModule('security', { rateLimitMaxRequests: 1000 })
      ).toThrow();

      // Reading should still work
      expect(manager.get().security.rateLimitMaxRequests).toBe(50);
    });

    it('should support dynamic config updates with listeners', () => {
      const manager = new config.ConfigManager();
      const updates: string[] = [];

      manager.onChange((cfg) => {
        updates.push(cfg.logging.level);
      });

      manager.setModule('logging', { level: 'debug' });
      manager.setModule('logging', { level: 'error' });
      manager.setModule('logging', { level: 'warn' });

      expect(updates).toEqual(['debug', 'error', 'warn']);
    });
  });
});
