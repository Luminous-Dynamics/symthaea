// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Type-Safe Configuration Management
 *
 * Provides validated, typed configuration from environment variables.
 * Fails fast on startup if configuration is invalid.
 */

import { z } from 'zod';

/**
 * Environment schema
 */
const envSchema = z.object({
  // Application
  NODE_ENV: z.enum(['development', 'staging', 'production', 'test']).default('development'),
  PORT: z.string().transform(Number).pipe(z.number().min(1).max(65535)).default('3100'),
  // Secure-by-default: bind to localhost unless explicitly configured otherwise.
  HOST: z.string().default('127.0.0.1'),
  API_VERSION: z.string().default('v2'),

  // Database
  DATABASE_URL: z.string().url().or(z.string().regex(/^postgresql:\/\//)),
  DB_POOL_MIN: z.string().transform(Number).pipe(z.number().min(0)).default('2'),
  DB_POOL_MAX: z.string().transform(Number).pipe(z.number().min(1)).default('20'),
  DB_IDLE_TIMEOUT: z.string().transform(Number).pipe(z.number().min(0)).default('30000'),
  DB_CONNECTION_TIMEOUT: z.string().transform(Number).pipe(z.number().min(0)).default('5000'),

  // Redis (optional)
  REDIS_URL: z.string().url().optional(),
  REDIS_KEY_PREFIX: z.string().default('mycelix:'),

  // Security
  // Secure-by-default: empty means "localhost only" (enforced in middleware).
  // Set "*" explicitly if you really want permissive CORS.
  CORS_ORIGIN: z.string().default(''),
  RATE_LIMIT_WINDOW_MS: z.string().transform(Number).pipe(z.number().min(1000)).default('60000'),
  RATE_LIMIT_MAX: z.string().transform(Number).pipe(z.number().min(1)).default('100'),

  // Logging
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  LOG_FORMAT: z.enum(['json', 'pretty']).default('json'),

  // IPFS
  IPFS_GATEWAY_URL: z.string().url().default('https://ipfs.io/ipfs'),

  // Blockchain
  RPC_URL: z.string().url().optional(),
  CHAIN_ID: z.string().transform(Number).pipe(z.number().positive()).optional(),

  // Monitoring
  SENTRY_DSN: z.string().url().optional(),
  OTEL_EXPORTER_OTLP_ENDPOINT: z.string().url().optional(),

  // Feature flags
  ENABLE_SWAGGER: z.string().transform(v => v === 'true').default('true'),
  ENABLE_METRICS: z.string().transform(v => v === 'true').default('true'),
  ENABLE_AUDIT_LOG: z.string().transform(v => v === 'true').default('true'),
});

/**
 * Parsed environment type
 */
export type Env = z.infer<typeof envSchema>;

/**
 * Structured configuration
 */
export interface Config {
  env: 'development' | 'staging' | 'production' | 'test';
  isDev: boolean;
  isProd: boolean;
  isTest: boolean;

  server: {
    port: number;
    host: string;
    apiVersion: string;
  };

  database: {
    url: string;
    pool: {
      min: number;
      max: number;
      idleTimeout: number;
      connectionTimeout: number;
    };
  };

  redis: {
    url: string | undefined;
    keyPrefix: string;
    enabled: boolean;
  };

  security: {
    corsOrigin: string | string[];
    rateLimit: {
      windowMs: number;
      max: number;
    };
  };

  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'pretty';
  };

  ipfs: {
    gatewayUrl: string;
  };

  blockchain: {
    rpcUrl: string | undefined;
    chainId: number | undefined;
  };

  monitoring: {
    sentryDsn: string | undefined;
    otelEndpoint: string | undefined;
  };

  features: {
    swagger: boolean;
    metrics: boolean;
    auditLog: boolean;
  };
}

/**
 * Parse and validate environment variables
 */
function parseEnv(): Env {
  const result = envSchema.safeParse(process.env);

  if (!result.success) {
    console.error('Invalid environment configuration:');
    for (const error of result.error.errors) {
      console.error(`  ${error.path.join('.')}: ${error.message}`);
    }
    throw new Error('Configuration validation failed');
  }

  return result.data;
}

/**
 * Build configuration object from environment
 */
function buildConfig(env: Env): Config {
  return {
    env: env.NODE_ENV,
    isDev: env.NODE_ENV === 'development',
    isProd: env.NODE_ENV === 'production',
    isTest: env.NODE_ENV === 'test',

    server: {
      port: env.PORT,
      host: env.HOST,
      apiVersion: env.API_VERSION,
    },

    database: {
      url: env.DATABASE_URL,
      pool: {
        min: env.DB_POOL_MIN,
        max: env.DB_POOL_MAX,
        idleTimeout: env.DB_IDLE_TIMEOUT,
        connectionTimeout: env.DB_CONNECTION_TIMEOUT,
      },
    },

    redis: {
      url: env.REDIS_URL,
      keyPrefix: env.REDIS_KEY_PREFIX,
      enabled: !!env.REDIS_URL,
    },

    security: {
      corsOrigin: env.CORS_ORIGIN.includes(',')
        ? env.CORS_ORIGIN.split(',').map(s => s.trim())
        : env.CORS_ORIGIN,
      rateLimit: {
        windowMs: env.RATE_LIMIT_WINDOW_MS,
        max: env.RATE_LIMIT_MAX,
      },
    },

    logging: {
      level: env.LOG_LEVEL,
      format: env.LOG_FORMAT,
    },

    ipfs: {
      gatewayUrl: env.IPFS_GATEWAY_URL,
    },

    blockchain: {
      rpcUrl: env.RPC_URL,
      chainId: env.CHAIN_ID,
    },

    monitoring: {
      sentryDsn: env.SENTRY_DSN,
      otelEndpoint: env.OTEL_EXPORTER_OTLP_ENDPOINT,
    },

    features: {
      swagger: env.ENABLE_SWAGGER,
      metrics: env.ENABLE_METRICS,
      auditLog: env.ENABLE_AUDIT_LOG,
    },
  };
}

// Singleton configuration instance
let _config: Config | null = null;

/**
 * Get the application configuration
 * Parses and validates on first call, returns cached instance thereafter
 */
export function getConfig(): Config {
  if (!_config) {
    const env = parseEnv();
    _config = buildConfig(env);

    // Log configuration summary in development
    if (_config.isDev) {
      console.log('Configuration loaded:', {
        env: _config.env,
        port: _config.server.port,
        database: '***',
        redis: _config.redis.enabled ? 'enabled' : 'disabled',
        features: _config.features,
      });
    }
  }

  return _config;
}

/**
 * Reset configuration (useful for testing)
 */
export function resetConfig(): void {
  _config = null;
}

/**
 * Validate configuration without storing
 */
export function validateConfig(): { valid: boolean; errors: string[] } {
  const result = envSchema.safeParse(process.env);

  if (result.success) {
    return { valid: true, errors: [] };
  }

  return {
    valid: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Get required environment variable (throws if missing)
 */
export function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Required environment variable ${name} is not set`);
  }
  return value;
}

/**
 * Get optional environment variable with default
 */
export function getEnv(name: string, defaultValue: string): string {
  return process.env[name] || defaultValue;
}

export default getConfig;
