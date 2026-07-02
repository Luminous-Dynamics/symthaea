// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Feature Flags System
 *
 * Flexible feature flag management with support for:
 * - Boolean flags
 * - Percentage rollouts
 * - User targeting
 * - A/B testing
 * - Time-based flags
 */

import { Request } from 'express';
import { createHash } from 'crypto';

/**
 * Feature flag types
 */
export type FlagType = 'boolean' | 'percentage' | 'targeted' | 'scheduled';

/**
 * Feature flag definition
 */
export interface FeatureFlag {
  name: string;
  description: string;
  type: FlagType;
  enabled: boolean;
  /** Percentage rollout (0-100) */
  percentage?: number;
  /** Target specific wallet addresses */
  targetAddresses?: string[];
  /** Environment-specific overrides */
  environments?: Record<string, boolean>;
  /** Schedule for automatic enable/disable */
  schedule?: {
    enableAt?: Date;
    disableAt?: Date;
  };
  /** Metadata */
  metadata?: Record<string, unknown>;
  /** Created timestamp */
  createdAt: Date;
  /** Updated timestamp */
  updatedAt: Date;
}

/**
 * Feature evaluation context
 */
export interface EvaluationContext {
  walletAddress?: string;
  userId?: string;
  environment?: string;
  attributes?: Record<string, unknown>;
}

/**
 * Feature flags configuration
 */
const defaultFlags: Record<string, Omit<FeatureFlag, 'createdAt' | 'updatedAt'>> = {
  // API Features
  'api.v3.enabled': {
    name: 'api.v3.enabled',
    description: 'Enable API v3 endpoints',
    type: 'boolean',
    enabled: false,
  },
  'api.rate_limit.enhanced': {
    name: 'api.rate_limit.enhanced',
    description: 'Enhanced rate limiting with burst support',
    type: 'percentage',
    enabled: true,
    percentage: 50,
  },

  // Music Features
  'music.streaming.hls': {
    name: 'music.streaming.hls',
    description: 'HLS streaming support',
    type: 'boolean',
    enabled: true,
  },
  'music.lyrics.sync': {
    name: 'music.lyrics.sync',
    description: 'Synchronized lyrics display',
    type: 'percentage',
    enabled: true,
    percentage: 25,
  },
  'music.recommendations.ai': {
    name: 'music.recommendations.ai',
    description: 'AI-powered music recommendations',
    type: 'targeted',
    enabled: true,
    targetAddresses: [],
  },

  // Wallet Features
  'wallet.multi_chain': {
    name: 'wallet.multi_chain',
    description: 'Multi-chain wallet support',
    type: 'boolean',
    enabled: false,
  },
  'wallet.gasless': {
    name: 'wallet.gasless',
    description: 'Gasless transactions',
    type: 'percentage',
    enabled: true,
    percentage: 10,
  },

  // Analytics Features
  'analytics.realtime': {
    name: 'analytics.realtime',
    description: 'Real-time analytics dashboard',
    type: 'boolean',
    enabled: true,
  },
  'analytics.export': {
    name: 'analytics.export',
    description: 'Analytics data export',
    type: 'targeted',
    enabled: true,
    targetAddresses: [],
  },

  // Experimental Features
  'experimental.websocket.v2': {
    name: 'experimental.websocket.v2',
    description: 'WebSocket protocol v2',
    type: 'percentage',
    enabled: false,
    percentage: 5,
  },
  'experimental.graphql': {
    name: 'experimental.graphql',
    description: 'GraphQL API',
    type: 'boolean',
    enabled: false,
  },
};

/**
 * Feature Flags Manager
 */
export class FeatureFlags {
  private flags: Map<string, FeatureFlag> = new Map();
  private overrides: Map<string, boolean> = new Map();
  private environment: string;

  constructor(environment = 'development') {
    this.environment = environment;
    this.loadDefaultFlags();
    this.loadEnvironmentOverrides();
  }

  /**
   * Load default flags
   */
  private loadDefaultFlags(): void {
    const now = new Date();

    for (const [name, config] of Object.entries(defaultFlags)) {
      this.flags.set(name, {
        ...config,
        createdAt: now,
        updatedAt: now,
      });
    }
  }

  /**
   * Load environment variable overrides
   */
  private loadEnvironmentOverrides(): void {
    // Load from environment variables
    // Format: FEATURE_FLAG_<name>=true/false
    for (const [key, value] of Object.entries(process.env)) {
      if (key.startsWith('FEATURE_FLAG_')) {
        const flagName = key
          .replace('FEATURE_FLAG_', '')
          .toLowerCase()
          .replace(/_/g, '.');

        this.overrides.set(flagName, value === 'true' || value === '1');
      }
    }
  }

  /**
   * Check if a feature is enabled
   */
  isEnabled(flagName: string, context: EvaluationContext = {}): boolean {
    // Check overrides first
    if (this.overrides.has(flagName)) {
      return this.overrides.get(flagName)!;
    }

    const flag = this.flags.get(flagName);
    if (!flag) {
      return false;
    }

    // Check environment-specific settings
    if (flag.environments && flag.environments[this.environment] !== undefined) {
      return flag.environments[this.environment];
    }

    // Check schedule
    if (flag.schedule) {
      const now = new Date();
      if (flag.schedule.enableAt && now < flag.schedule.enableAt) {
        return false;
      }
      if (flag.schedule.disableAt && now >= flag.schedule.disableAt) {
        return false;
      }
    }

    // Not enabled globally
    if (!flag.enabled) {
      return false;
    }

    // Evaluate based on type
    switch (flag.type) {
      case 'boolean':
        return flag.enabled;

      case 'percentage':
        return this.evaluatePercentage(flagName, flag.percentage || 0, context);

      case 'targeted':
        return this.evaluateTargeted(flag, context);

      case 'scheduled':
        return flag.enabled;

      default:
        return false;
    }
  }

  /**
   * Evaluate percentage rollout
   */
  private evaluatePercentage(
    flagName: string,
    percentage: number,
    context: EvaluationContext
  ): boolean {
    // Use wallet address or generate random for anonymous users
    const identifier = context.walletAddress || context.userId || Math.random().toString();

    // Create consistent hash
    const hash = createHash('sha256')
      .update(`${flagName}:${identifier}`)
      .digest('hex');

    // Convert first 8 chars to number (0-100)
    const hashValue = parseInt(hash.slice(0, 8), 16) % 100;

    return hashValue < percentage;
  }

  /**
   * Evaluate targeted flag
   */
  private evaluateTargeted(flag: FeatureFlag, context: EvaluationContext): boolean {
    if (!flag.targetAddresses || flag.targetAddresses.length === 0) {
      return flag.enabled;
    }

    if (!context.walletAddress) {
      return false;
    }

    const normalizedAddress = context.walletAddress.toLowerCase();
    return flag.targetAddresses.some(
      addr => addr.toLowerCase() === normalizedAddress
    );
  }

  /**
   * Get flag configuration
   */
  getFlag(flagName: string): FeatureFlag | undefined {
    return this.flags.get(flagName);
  }

  /**
   * Get all flags
   */
  getAllFlags(): FeatureFlag[] {
    return Array.from(this.flags.values());
  }

  /**
   * Update a flag
   */
  updateFlag(flagName: string, updates: Partial<FeatureFlag>): void {
    const existing = this.flags.get(flagName);
    if (!existing) {
      throw new Error(`Flag not found: ${flagName}`);
    }

    this.flags.set(flagName, {
      ...existing,
      ...updates,
      name: flagName, // Prevent name changes
      updatedAt: new Date(),
    });
  }

  /**
   * Create a new flag
   */
  createFlag(config: Omit<FeatureFlag, 'createdAt' | 'updatedAt'>): void {
    if (this.flags.has(config.name)) {
      throw new Error(`Flag already exists: ${config.name}`);
    }

    const now = new Date();
    this.flags.set(config.name, {
      ...config,
      createdAt: now,
      updatedAt: now,
    });
  }

  /**
   * Delete a flag
   */
  deleteFlag(flagName: string): void {
    this.flags.delete(flagName);
    this.overrides.delete(flagName);
  }

  /**
   * Set runtime override
   */
  setOverride(flagName: string, enabled: boolean): void {
    this.overrides.set(flagName, enabled);
  }

  /**
   * Clear runtime override
   */
  clearOverride(flagName: string): void {
    this.overrides.delete(flagName);
  }

  /**
   * Clear all overrides
   */
  clearAllOverrides(): void {
    this.overrides.clear();
  }

  /**
   * Add target address to a flag
   */
  addTarget(flagName: string, walletAddress: string): void {
    const flag = this.flags.get(flagName);
    if (!flag) {
      throw new Error(`Flag not found: ${flagName}`);
    }

    if (flag.type !== 'targeted') {
      throw new Error(`Flag is not targeted type: ${flagName}`);
    }

    const targets = flag.targetAddresses || [];
    const normalizedAddress = walletAddress.toLowerCase();

    if (!targets.includes(normalizedAddress)) {
      targets.push(normalizedAddress);
      this.updateFlag(flagName, { targetAddresses: targets });
    }
  }

  /**
   * Remove target address from a flag
   */
  removeTarget(flagName: string, walletAddress: string): void {
    const flag = this.flags.get(flagName);
    if (!flag) return;

    const targets = flag.targetAddresses || [];
    const normalizedAddress = walletAddress.toLowerCase();
    const newTargets = targets.filter(addr => addr !== normalizedAddress);

    this.updateFlag(flagName, { targetAddresses: newTargets });
  }

  /**
   * Get flags status summary
   */
  getSummary(): {
    total: number;
    enabled: number;
    disabled: number;
    byType: Record<FlagType, number>;
  } {
    const flags = this.getAllFlags();
    const byType: Record<FlagType, number> = {
      boolean: 0,
      percentage: 0,
      targeted: 0,
      scheduled: 0,
    };

    for (const flag of flags) {
      byType[flag.type]++;
    }

    return {
      total: flags.length,
      enabled: flags.filter(f => f.enabled).length,
      disabled: flags.filter(f => !f.enabled).length,
      byType,
    };
  }
}

/**
 * Singleton instance
 */
let instance: FeatureFlags | null = null;

export function getFeatureFlags(): FeatureFlags {
  if (!instance) {
    instance = new FeatureFlags(process.env.NODE_ENV || 'development');
  }
  return instance;
}

export function resetFeatureFlags(): void {
  instance = null;
}

/**
 * Express middleware for feature flags
 */
export function featureFlagMiddleware() {
  return (req: Request, res: any, next: any): void => {
    const flags = getFeatureFlags();

    // Build context from request
    const context: EvaluationContext = {
      walletAddress: req.auth?.address || (req as any).context?.walletAddress,
      environment: process.env.NODE_ENV,
      attributes: {
        path: req.path,
        method: req.method,
        userAgent: req.headers['user-agent'],
      },
    };

    // Attach helper to request
    (req as any).isFeatureEnabled = (flagName: string) => {
      return flags.isEnabled(flagName, context);
    };

    next();
  };
}

/**
 * Feature flag guard for routes
 */
export function requireFeature(flagName: string, fallback?: any) {
  return (req: Request, res: any, next: any): void => {
    const isEnabled = (req as any).isFeatureEnabled?.(flagName) ??
      getFeatureFlags().isEnabled(flagName, {
        walletAddress: req.auth?.address,
      });

    if (!isEnabled) {
      if (fallback) {
        return fallback(req, res, next);
      }
      res.status(404).json({
        success: false,
        error: {
          code: 'FEATURE_DISABLED',
          message: 'This feature is not available',
        },
      });
      return;
    }

    next();
  };
}

/**
 * Check feature in code
 */
export function isFeatureEnabled(
  flagName: string,
  context?: EvaluationContext
): boolean {
  return getFeatureFlags().isEnabled(flagName, context);
}

export default {
  getFeatureFlags,
  resetFeatureFlags,
  featureFlagMiddleware,
  requireFeature,
  isFeatureEnabled,
};
