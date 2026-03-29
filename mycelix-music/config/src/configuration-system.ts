// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Platform - Configuration & Secrets Management
 *
 * Production-grade configuration system with:
 * - Strongly typed configuration schemas with Zod validation
 * - Multi-source configuration loading (env, files, remote)
 * - Secrets management with Vault, AWS Secrets Manager, Azure Key Vault
 * - Hot reload capabilities with change notifications
 * - Multi-tenant configuration overrides
 * - Environment-aware defaults and validation
 */

import { EventEmitter } from 'events';
import * as crypto from 'crypto';

// ============================================================================
// CONFIGURATION SCHEMA DEFINITIONS
// ============================================================================

/**
 * Zod-like schema validation system
 */
type SchemaType = 'string' | 'number' | 'boolean' | 'array' | 'object' | 'enum';

interface SchemaDefinition {
  type: SchemaType;
  required?: boolean;
  default?: any;
  description?: string;
  sensitive?: boolean;
  enum?: string[];
  items?: SchemaDefinition;
  properties?: Record<string, SchemaDefinition>;
  min?: number;
  max?: number;
  pattern?: string;
  transform?: (value: any) => any;
  validate?: (value: any) => boolean | string;
}

interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

interface ValidationError {
  path: string;
  message: string;
  value?: any;
}

interface ValidationWarning {
  path: string;
  message: string;
}

/**
 * Configuration schema builder with fluent API
 */
class ConfigSchema {
  private schema: Record<string, SchemaDefinition> = {};

  string(name: string, options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'string', ...options };
    return this;
  }

  number(name: string, options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'number', ...options };
    return this;
  }

  boolean(name: string, options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'boolean', ...options };
    return this;
  }

  enum(name: string, values: string[], options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'enum', enum: values, ...options };
    return this;
  }

  array(name: string, items: SchemaDefinition, options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'array', items, ...options };
    return this;
  }

  object(name: string, properties: Record<string, SchemaDefinition>, options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'object', properties, ...options };
    return this;
  }

  sensitive(name: string, options: Partial<SchemaDefinition> = {}): this {
    this.schema[name] = { type: 'string', sensitive: true, ...options };
    return this;
  }

  getSchema(): Record<string, SchemaDefinition> {
    return this.schema;
  }

  validate(config: Record<string, any>): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    for (const [key, def] of Object.entries(this.schema)) {
      const value = config[key];
      const path = key;

      // Check required
      if (def.required && (value === undefined || value === null)) {
        errors.push({ path, message: `Required configuration '${key}' is missing` });
        continue;
      }

      if (value === undefined || value === null) {
        continue;
      }

      // Type validation
      const typeError = this.validateType(value, def, path);
      if (typeError) {
        errors.push(typeError);
        continue;
      }

      // Custom validation
      if (def.validate) {
        const result = def.validate(value);
        if (result !== true) {
          errors.push({
            path,
            message: typeof result === 'string' ? result : `Validation failed for '${key}'`,
            value
          });
        }
      }

      // Pattern validation for strings
      if (def.type === 'string' && def.pattern) {
        const regex = new RegExp(def.pattern);
        if (!regex.test(value)) {
          errors.push({ path, message: `Value doesn't match pattern ${def.pattern}`, value });
        }
      }

      // Range validation for numbers
      if (def.type === 'number') {
        if (def.min !== undefined && value < def.min) {
          errors.push({ path, message: `Value ${value} is less than minimum ${def.min}`, value });
        }
        if (def.max !== undefined && value > def.max) {
          errors.push({ path, message: `Value ${value} is greater than maximum ${def.max}`, value });
        }
      }
    }

    // Check for unknown keys
    for (const key of Object.keys(config)) {
      if (!this.schema[key]) {
        warnings.push({ path: key, message: `Unknown configuration key '${key}'` });
      }
    }

    return { valid: errors.length === 0, errors, warnings };
  }

  private validateType(value: any, def: SchemaDefinition, path: string): ValidationError | null {
    switch (def.type) {
      case 'string':
        if (typeof value !== 'string') {
          return { path, message: `Expected string, got ${typeof value}`, value };
        }
        break;
      case 'number':
        if (typeof value !== 'number' || isNaN(value)) {
          return { path, message: `Expected number, got ${typeof value}`, value };
        }
        break;
      case 'boolean':
        if (typeof value !== 'boolean') {
          return { path, message: `Expected boolean, got ${typeof value}`, value };
        }
        break;
      case 'enum':
        if (!def.enum?.includes(value)) {
          return { path, message: `Value must be one of: ${def.enum?.join(', ')}`, value };
        }
        break;
      case 'array':
        if (!Array.isArray(value)) {
          return { path, message: `Expected array, got ${typeof value}`, value };
        }
        break;
      case 'object':
        if (typeof value !== 'object' || Array.isArray(value)) {
          return { path, message: `Expected object, got ${typeof value}`, value };
        }
        break;
    }
    return null;
  }

  applyDefaults(config: Record<string, any>): Record<string, any> {
    const result = { ...config };
    for (const [key, def] of Object.entries(this.schema)) {
      if (result[key] === undefined && def.default !== undefined) {
        result[key] = typeof def.default === 'function' ? def.default() : def.default;
      }
    }
    return result;
  }

  applyTransforms(config: Record<string, any>): Record<string, any> {
    const result = { ...config };
    for (const [key, def] of Object.entries(this.schema)) {
      if (result[key] !== undefined && def.transform) {
        result[key] = def.transform(result[key]);
      }
    }
    return result;
  }
}

// ============================================================================
// MYCELIX CONFIGURATION SCHEMA
// ============================================================================

/**
 * Complete Mycelix platform configuration interface
 */
interface MycelixConfig {
  // Environment
  environment: 'development' | 'staging' | 'production';
  serviceName: string;
  serviceVersion: string;
  instanceId: string;

  // Server
  server: {
    host: string;
    port: number;
    trustProxy: boolean;
    corsOrigins: string[];
    rateLimitWindowMs: number;
    rateLimitMaxRequests: number;
  };

  // Database
  database: {
    host: string;
    port: number;
    name: string;
    username: string;
    password: string;
    poolMin: number;
    poolMax: number;
    ssl: boolean;
    replicaHosts?: string[];
  };

  // Redis
  redis: {
    host: string;
    port: number;
    password?: string;
    db: number;
    cluster: boolean;
    sentinelHosts?: string[];
    sentinelName?: string;
  };

  // Authentication
  auth: {
    jwtSecret: string;
    jwtExpiresIn: string;
    refreshTokenExpiresIn: string;
    bcryptRounds: number;
    mfaIssuer: string;
    oauth: {
      google?: { clientId: string; clientSecret: string };
      apple?: { clientId: string; teamId: string; keyId: string; privateKey: string };
      spotify?: { clientId: string; clientSecret: string };
    };
  };

  // Storage
  storage: {
    provider: 'local' | 's3' | 'gcs' | 'azure';
    bucket: string;
    region?: string;
    accessKeyId?: string;
    secretAccessKey?: string;
    cdnUrl?: string;
    maxFileSize: number;
  };

  // Streaming
  streaming: {
    hlsSegmentDuration: number;
    dashSegmentDuration: number;
    maxBitrate: number;
    transcodingConcurrency: number;
    cacheEnabled: boolean;
    cacheTtl: number;
  };

  // Search
  search: {
    provider: 'elasticsearch' | 'meilisearch' | 'typesense';
    hosts: string[];
    apiKey?: string;
    indexPrefix: string;
  };

  // Messaging
  messaging: {
    provider: 'kafka' | 'rabbitmq' | 'sqs';
    brokers: string[];
    username?: string;
    password?: string;
    groupId: string;
  };

  // Observability
  observability: {
    logging: {
      level: 'debug' | 'info' | 'warn' | 'error';
      format: 'json' | 'pretty';
      outputs: string[];
    };
    metrics: {
      enabled: boolean;
      port: number;
      path: string;
    };
    tracing: {
      enabled: boolean;
      samplingRate: number;
      exporterEndpoint?: string;
    };
  };

  // Feature Flags
  featureFlags: {
    provider: 'local' | 'launchdarkly' | 'unleash' | 'flagsmith';
    apiKey?: string;
    refreshIntervalMs: number;
    defaults: Record<string, boolean>;
  };

  // External Services
  external: {
    stripe?: { secretKey: string; webhookSecret: string };
    sendgrid?: { apiKey: string; fromEmail: string };
    twilio?: { accountSid: string; authToken: string; fromNumber: string };
    openai?: { apiKey: string; model: string };
    anthropic?: { apiKey: string; model: string };
  };

  // Multi-tenant
  multiTenant: {
    enabled: boolean;
    isolationLevel: 'schema' | 'row' | 'database';
    defaultTenant: string;
  };
}

/**
 * Build the Mycelix configuration schema
 */
function buildMycelixSchema(): ConfigSchema {
  const schema = new ConfigSchema();

  // Environment
  schema
    .enum('NODE_ENV', ['development', 'staging', 'production'], {
      default: 'development',
      description: 'Runtime environment'
    })
    .string('SERVICE_NAME', {
      default: 'mycelix-music',
      description: 'Service identifier'
    })
    .string('SERVICE_VERSION', {
      default: '1.0.0',
      description: 'Service version'
    })
    .string('INSTANCE_ID', {
      default: () => crypto.randomUUID(),
      description: 'Unique instance identifier'
    });

  // Server
  schema
    .string('SERVER_HOST', { default: '0.0.0.0' })
    .number('SERVER_PORT', { default: 3000, min: 1, max: 65535 })
    .boolean('SERVER_TRUST_PROXY', { default: false })
    .string('CORS_ORIGINS', { default: '*' })
    .number('RATE_LIMIT_WINDOW_MS', { default: 60000 })
    .number('RATE_LIMIT_MAX_REQUESTS', { default: 100 });

  // Database
  schema
    .string('DATABASE_HOST', { required: true })
    .number('DATABASE_PORT', { default: 5432 })
    .string('DATABASE_NAME', { required: true })
    .sensitive('DATABASE_USERNAME', { required: true })
    .sensitive('DATABASE_PASSWORD', { required: true })
    .number('DATABASE_POOL_MIN', { default: 2, min: 1 })
    .number('DATABASE_POOL_MAX', { default: 10, min: 1 })
    .boolean('DATABASE_SSL', { default: true })
    .string('DATABASE_REPLICA_HOSTS', { default: '' });

  // Redis
  schema
    .string('REDIS_HOST', { default: 'localhost' })
    .number('REDIS_PORT', { default: 6379 })
    .sensitive('REDIS_PASSWORD', {})
    .number('REDIS_DB', { default: 0 })
    .boolean('REDIS_CLUSTER', { default: false })
    .string('REDIS_SENTINEL_HOSTS', { default: '' })
    .string('REDIS_SENTINEL_NAME', { default: 'mymaster' });

  // Authentication
  schema
    .sensitive('JWT_SECRET', { required: true, min: 32 })
    .string('JWT_EXPIRES_IN', { default: '1h' })
    .string('REFRESH_TOKEN_EXPIRES_IN', { default: '7d' })
    .number('BCRYPT_ROUNDS', { default: 12, min: 10, max: 15 })
    .string('MFA_ISSUER', { default: 'Mycelix Music' })
    .sensitive('OAUTH_GOOGLE_CLIENT_ID', {})
    .sensitive('OAUTH_GOOGLE_CLIENT_SECRET', {})
    .sensitive('OAUTH_APPLE_CLIENT_ID', {})
    .sensitive('OAUTH_APPLE_TEAM_ID', {})
    .sensitive('OAUTH_APPLE_KEY_ID', {})
    .sensitive('OAUTH_APPLE_PRIVATE_KEY', {})
    .sensitive('OAUTH_SPOTIFY_CLIENT_ID', {})
    .sensitive('OAUTH_SPOTIFY_CLIENT_SECRET', {});

  // Storage
  schema
    .enum('STORAGE_PROVIDER', ['local', 's3', 'gcs', 'azure'], { default: 'local' })
    .string('STORAGE_BUCKET', { default: 'mycelix-media' })
    .string('STORAGE_REGION', { default: 'us-east-1' })
    .sensitive('STORAGE_ACCESS_KEY_ID', {})
    .sensitive('STORAGE_SECRET_ACCESS_KEY', {})
    .string('STORAGE_CDN_URL', {})
    .number('STORAGE_MAX_FILE_SIZE', { default: 500 * 1024 * 1024 }); // 500MB

  // Streaming
  schema
    .number('STREAMING_HLS_SEGMENT_DURATION', { default: 6 })
    .number('STREAMING_DASH_SEGMENT_DURATION', { default: 4 })
    .number('STREAMING_MAX_BITRATE', { default: 320000 })
    .number('STREAMING_TRANSCODING_CONCURRENCY', { default: 4 })
    .boolean('STREAMING_CACHE_ENABLED', { default: true })
    .number('STREAMING_CACHE_TTL', { default: 3600 });

  // Search
  schema
    .enum('SEARCH_PROVIDER', ['elasticsearch', 'meilisearch', 'typesense'], { default: 'elasticsearch' })
    .string('SEARCH_HOSTS', { default: 'http://localhost:9200' })
    .sensitive('SEARCH_API_KEY', {})
    .string('SEARCH_INDEX_PREFIX', { default: 'mycelix_' });

  // Messaging
  schema
    .enum('MESSAGING_PROVIDER', ['kafka', 'rabbitmq', 'sqs'], { default: 'kafka' })
    .string('MESSAGING_BROKERS', { default: 'localhost:9092' })
    .sensitive('MESSAGING_USERNAME', {})
    .sensitive('MESSAGING_PASSWORD', {})
    .string('MESSAGING_GROUP_ID', { default: 'mycelix-consumers' });

  // Observability
  schema
    .enum('LOG_LEVEL', ['debug', 'info', 'warn', 'error'], { default: 'info' })
    .enum('LOG_FORMAT', ['json', 'pretty'], { default: 'json' })
    .string('LOG_OUTPUTS', { default: 'stdout' })
    .boolean('METRICS_ENABLED', { default: true })
    .number('METRICS_PORT', { default: 9090 })
    .string('METRICS_PATH', { default: '/metrics' })
    .boolean('TRACING_ENABLED', { default: true })
    .number('TRACING_SAMPLING_RATE', { default: 0.1, min: 0, max: 1 })
    .string('TRACING_EXPORTER_ENDPOINT', {});

  // Feature Flags
  schema
    .enum('FEATURE_FLAGS_PROVIDER', ['local', 'launchdarkly', 'unleash', 'flagsmith'], { default: 'local' })
    .sensitive('FEATURE_FLAGS_API_KEY', {})
    .number('FEATURE_FLAGS_REFRESH_INTERVAL_MS', { default: 30000 });

  // External Services
  schema
    .sensitive('STRIPE_SECRET_KEY', {})
    .sensitive('STRIPE_WEBHOOK_SECRET', {})
    .sensitive('SENDGRID_API_KEY', {})
    .string('SENDGRID_FROM_EMAIL', { default: 'noreply@mycelix.music' })
    .sensitive('TWILIO_ACCOUNT_SID', {})
    .sensitive('TWILIO_AUTH_TOKEN', {})
    .string('TWILIO_FROM_NUMBER', {})
    .sensitive('OPENAI_API_KEY', {})
    .string('OPENAI_MODEL', { default: 'gpt-4' })
    .sensitive('ANTHROPIC_API_KEY', {})
    .string('ANTHROPIC_MODEL', { default: 'claude-3-opus-20240229' });

  // Multi-tenant
  schema
    .boolean('MULTI_TENANT_ENABLED', { default: false })
    .enum('MULTI_TENANT_ISOLATION_LEVEL', ['schema', 'row', 'database'], { default: 'row' })
    .string('MULTI_TENANT_DEFAULT_TENANT', { default: 'default' });

  return schema;
}

// ============================================================================
// CONFIGURATION SOURCES
// ============================================================================

/**
 * Configuration source interface
 */
interface ConfigSource {
  name: string;
  priority: number;
  load(): Promise<Record<string, any>>;
  watch?(callback: (changes: Record<string, any>) => void): void;
}

/**
 * Environment variables configuration source
 */
class EnvConfigSource implements ConfigSource {
  name = 'environment';
  priority = 100;

  constructor(private prefix: string = '') {}

  async load(): Promise<Record<string, any>> {
    const config: Record<string, any> = {};

    for (const [key, value] of Object.entries(process.env)) {
      if (this.prefix && !key.startsWith(this.prefix)) {
        continue;
      }

      const configKey = this.prefix ? key.slice(this.prefix.length) : key;
      config[configKey] = this.parseValue(value);
    }

    return config;
  }

  private parseValue(value: string | undefined): any {
    if (value === undefined) return undefined;
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (/^\d+$/.test(value)) return parseInt(value, 10);
    if (/^\d+\.\d+$/.test(value)) return parseFloat(value);
    return value;
  }
}

/**
 * JSON file configuration source
 */
class JsonFileConfigSource implements ConfigSource {
  name: string;
  priority: number;
  private watchCallback?: (changes: Record<string, any>) => void;

  constructor(
    private filePath: string,
    priority: number = 50
  ) {
    this.name = `file:${filePath}`;
    this.priority = priority;
  }

  async load(): Promise<Record<string, any>> {
    try {
      const fs = await import('fs/promises');
      const content = await fs.readFile(this.filePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return {};
      }
      throw error;
    }
  }

  async watch(callback: (changes: Record<string, any>) => void): Promise<void> {
    this.watchCallback = callback;
    const fs = await import('fs');

    fs.watch(this.filePath, async (eventType) => {
      if (eventType === 'change' && this.watchCallback) {
        const newConfig = await this.load();
        this.watchCallback(newConfig);
      }
    });
  }
}

/**
 * YAML file configuration source
 */
class YamlFileConfigSource implements ConfigSource {
  name: string;
  priority: number;

  constructor(
    private filePath: string,
    priority: number = 50
  ) {
    this.name = `yaml:${filePath}`;
    this.priority = priority;
  }

  async load(): Promise<Record<string, any>> {
    try {
      const fs = await import('fs/promises');
      const content = await fs.readFile(this.filePath, 'utf-8');
      return this.parseYaml(content);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return {};
      }
      throw error;
    }
  }

  private parseYaml(content: string): Record<string, any> {
    // Simple YAML parser for configuration files
    const result: Record<string, any> = {};
    const lines = content.split('\n');
    const stack: { indent: number; obj: any; key?: string }[] = [{ indent: -1, obj: result }];

    for (const line of lines) {
      if (!line.trim() || line.trim().startsWith('#')) continue;

      const indent = line.search(/\S/);
      const trimmed = line.trim();

      // Pop stack to correct level
      while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
        stack.pop();
      }

      const parent = stack[stack.length - 1].obj;

      if (trimmed.includes(':')) {
        const [key, ...valueParts] = trimmed.split(':');
        const value = valueParts.join(':').trim();

        if (value) {
          // Key-value pair
          parent[key.trim()] = this.parseValue(value);
        } else {
          // Nested object
          parent[key.trim()] = {};
          stack.push({ indent, obj: parent[key.trim()], key: key.trim() });
        }
      } else if (trimmed.startsWith('- ')) {
        // Array item
        const key = stack[stack.length - 1].key;
        if (key && !Array.isArray(parent[key])) {
          const parentParent = stack[stack.length - 2]?.obj;
          if (parentParent) {
            parentParent[key] = [];
            stack[stack.length - 1].obj = parentParent[key];
          }
        }
        if (Array.isArray(stack[stack.length - 1].obj)) {
          stack[stack.length - 1].obj.push(this.parseValue(trimmed.slice(2)));
        }
      }
    }

    return result;
  }

  private parseValue(value: string): any {
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (value === 'null') return null;
    if (/^\d+$/.test(value)) return parseInt(value, 10);
    if (/^\d+\.\d+$/.test(value)) return parseFloat(value);
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
      return value.slice(1, -1);
    }
    return value;
  }
}

// ============================================================================
// SECRETS MANAGEMENT
// ============================================================================

/**
 * Secrets provider interface
 */
interface SecretsProvider {
  name: string;
  initialize(): Promise<void>;
  getSecret(key: string): Promise<string | undefined>;
  setSecret(key: string, value: string): Promise<void>;
  deleteSecret(key: string): Promise<void>;
  listSecrets(): Promise<string[]>;
  rotateSecret?(key: string): Promise<string>;
}

/**
 * HashiCorp Vault secrets provider
 */
class VaultSecretsProvider implements SecretsProvider {
  name = 'vault';
  private token: string = '';
  private cache = new Map<string, { value: string; expiresAt: number }>();

  constructor(
    private address: string,
    private mountPath: string = 'secret',
    private cacheTtlMs: number = 300000 // 5 minutes
  ) {}

  async initialize(): Promise<void> {
    // Try multiple auth methods
    this.token = await this.authenticate();
  }

  private async authenticate(): Promise<string> {
    // Try token from environment
    if (process.env.VAULT_TOKEN) {
      return process.env.VAULT_TOKEN;
    }

    // Try Kubernetes auth
    if (process.env.KUBERNETES_SERVICE_HOST) {
      return this.authenticateKubernetes();
    }

    // Try AppRole auth
    if (process.env.VAULT_ROLE_ID && process.env.VAULT_SECRET_ID) {
      return this.authenticateAppRole();
    }

    throw new Error('No Vault authentication method available');
  }

  private async authenticateKubernetes(): Promise<string> {
    const fs = await import('fs/promises');
    const jwt = await fs.readFile('/var/run/secrets/kubernetes.io/serviceaccount/token', 'utf-8');
    const role = process.env.VAULT_ROLE || 'mycelix';

    const response = await fetch(`${this.address}/v1/auth/kubernetes/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, jwt }),
    });

    const data = await response.json();
    return data.auth.client_token;
  }

  private async authenticateAppRole(): Promise<string> {
    const response = await fetch(`${this.address}/v1/auth/approle/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role_id: process.env.VAULT_ROLE_ID,
        secret_id: process.env.VAULT_SECRET_ID,
      }),
    });

    const data = await response.json();
    return data.auth.client_token;
  }

  async getSecret(key: string): Promise<string | undefined> {
    // Check cache
    const cached = this.cache.get(key);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }

    const response = await fetch(
      `${this.address}/v1/${this.mountPath}/data/${key}`,
      {
        headers: { 'X-Vault-Token': this.token },
      }
    );

    if (!response.ok) {
      if (response.status === 404) return undefined;
      throw new Error(`Vault error: ${response.statusText}`);
    }

    const data = await response.json();
    const value = data.data?.data?.value;

    if (value) {
      this.cache.set(key, {
        value,
        expiresAt: Date.now() + this.cacheTtlMs,
      });
    }

    return value;
  }

  async setSecret(key: string, value: string): Promise<void> {
    await fetch(`${this.address}/v1/${this.mountPath}/data/${key}`, {
      method: 'POST',
      headers: {
        'X-Vault-Token': this.token,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data: { value } }),
    });

    this.cache.delete(key);
  }

  async deleteSecret(key: string): Promise<void> {
    await fetch(`${this.address}/v1/${this.mountPath}/metadata/${key}`, {
      method: 'DELETE',
      headers: { 'X-Vault-Token': this.token },
    });

    this.cache.delete(key);
  }

  async listSecrets(): Promise<string[]> {
    const response = await fetch(
      `${this.address}/v1/${this.mountPath}/metadata?list=true`,
      {
        headers: { 'X-Vault-Token': this.token },
      }
    );

    const data = await response.json();
    return data.data?.keys || [];
  }

  async rotateSecret(key: string): Promise<string> {
    const newValue = crypto.randomBytes(32).toString('base64');
    await this.setSecret(key, newValue);
    return newValue;
  }
}

/**
 * AWS Secrets Manager provider
 */
class AWSSecretsProvider implements SecretsProvider {
  name = 'aws-secrets-manager';
  private cache = new Map<string, { value: string; expiresAt: number }>();

  constructor(
    private region: string = 'us-east-1',
    private prefix: string = 'mycelix/',
    private cacheTtlMs: number = 300000
  ) {}

  async initialize(): Promise<void> {
    // AWS SDK handles credential initialization
  }

  async getSecret(key: string): Promise<string | undefined> {
    const cached = this.cache.get(key);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }

    // Using AWS SDK v3 pattern
    const secretId = `${this.prefix}${key}`;

    try {
      // Simulated AWS API call structure
      const response = await this.callAWSAPI('secretsmanager', 'GetSecretValue', {
        SecretId: secretId,
      });

      const value = response.SecretString;
      if (value) {
        this.cache.set(key, {
          value,
          expiresAt: Date.now() + this.cacheTtlMs,
        });
      }

      return value;
    } catch (error: any) {
      if (error.name === 'ResourceNotFoundException') {
        return undefined;
      }
      throw error;
    }
  }

  async setSecret(key: string, value: string): Promise<void> {
    const secretId = `${this.prefix}${key}`;

    try {
      await this.callAWSAPI('secretsmanager', 'UpdateSecret', {
        SecretId: secretId,
        SecretString: value,
      });
    } catch (error: any) {
      if (error.name === 'ResourceNotFoundException') {
        await this.callAWSAPI('secretsmanager', 'CreateSecret', {
          Name: secretId,
          SecretString: value,
        });
      } else {
        throw error;
      }
    }

    this.cache.delete(key);
  }

  async deleteSecret(key: string): Promise<void> {
    const secretId = `${this.prefix}${key}`;

    await this.callAWSAPI('secretsmanager', 'DeleteSecret', {
      SecretId: secretId,
      ForceDeleteWithoutRecovery: true,
    });

    this.cache.delete(key);
  }

  async listSecrets(): Promise<string[]> {
    const response = await this.callAWSAPI('secretsmanager', 'ListSecrets', {
      Filters: [{ Key: 'name', Values: [this.prefix] }],
    });

    return response.SecretList?.map((s: any) =>
      s.Name.replace(this.prefix, '')
    ) || [];
  }

  private async callAWSAPI(service: string, action: string, params: any): Promise<any> {
    // This would use the actual AWS SDK in production
    // Simplified for demonstration
    const endpoint = `https://${service}.${this.region}.amazonaws.com`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-amz-json-1.1',
        'X-Amz-Target': `${service}.${action}`,
      },
      body: JSON.stringify(params),
    });

    return response.json();
  }
}

/**
 * Azure Key Vault provider
 */
class AzureKeyVaultProvider implements SecretsProvider {
  name = 'azure-key-vault';
  private accessToken: string = '';
  private cache = new Map<string, { value: string; expiresAt: number }>();

  constructor(
    private vaultUrl: string,
    private cacheTtlMs: number = 300000
  ) {}

  async initialize(): Promise<void> {
    this.accessToken = await this.getAccessToken();
  }

  private async getAccessToken(): Promise<string> {
    // Try managed identity first
    if (process.env.IDENTITY_ENDPOINT) {
      return this.getManagedIdentityToken();
    }

    // Fall back to service principal
    return this.getServicePrincipalToken();
  }

  private async getManagedIdentityToken(): Promise<string> {
    const response = await fetch(
      `${process.env.IDENTITY_ENDPOINT}?resource=https://vault.azure.net&api-version=2019-08-01`,
      {
        headers: { 'X-IDENTITY-HEADER': process.env.IDENTITY_HEADER || '' },
      }
    );

    const data = await response.json();
    return data.access_token;
  }

  private async getServicePrincipalToken(): Promise<string> {
    const tenantId = process.env.AZURE_TENANT_ID;
    const clientId = process.env.AZURE_CLIENT_ID;
    const clientSecret = process.env.AZURE_CLIENT_SECRET;

    const response = await fetch(
      `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          grant_type: 'client_credentials',
          client_id: clientId || '',
          client_secret: clientSecret || '',
          scope: 'https://vault.azure.net/.default',
        }),
      }
    );

    const data = await response.json();
    return data.access_token;
  }

  async getSecret(key: string): Promise<string | undefined> {
    const cached = this.cache.get(key);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }

    const response = await fetch(
      `${this.vaultUrl}/secrets/${key}?api-version=7.4`,
      {
        headers: { Authorization: `Bearer ${this.accessToken}` },
      }
    );

    if (!response.ok) {
      if (response.status === 404) return undefined;
      throw new Error(`Azure Key Vault error: ${response.statusText}`);
    }

    const data = await response.json();
    const value = data.value;

    if (value) {
      this.cache.set(key, {
        value,
        expiresAt: Date.now() + this.cacheTtlMs,
      });
    }

    return value;
  }

  async setSecret(key: string, value: string): Promise<void> {
    await fetch(`${this.vaultUrl}/secrets/${key}?api-version=7.4`, {
      method: 'PUT',
      headers: {
        Authorization: `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ value }),
    });

    this.cache.delete(key);
  }

  async deleteSecret(key: string): Promise<void> {
    await fetch(`${this.vaultUrl}/secrets/${key}?api-version=7.4`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${this.accessToken}` },
    });

    this.cache.delete(key);
  }

  async listSecrets(): Promise<string[]> {
    const response = await fetch(
      `${this.vaultUrl}/secrets?api-version=7.4`,
      {
        headers: { Authorization: `Bearer ${this.accessToken}` },
      }
    );

    const data = await response.json();
    return data.value?.map((s: any) => s.id.split('/').pop()) || [];
  }
}

/**
 * Local encrypted secrets provider (for development)
 */
class LocalSecretsProvider implements SecretsProvider {
  name = 'local';
  private secrets = new Map<string, string>();
  private encryptionKey: Buffer;

  constructor(encryptionKey?: string) {
    this.encryptionKey = encryptionKey
      ? Buffer.from(encryptionKey, 'base64')
      : crypto.randomBytes(32);
  }

  async initialize(): Promise<void> {
    // Load from encrypted file if exists
    try {
      const fs = await import('fs/promises');
      const encrypted = await fs.readFile('.secrets.enc', 'utf-8');
      const decrypted = this.decrypt(encrypted);
      const data = JSON.parse(decrypted);

      for (const [key, value] of Object.entries(data)) {
        this.secrets.set(key, value as string);
      }
    } catch {
      // No existing secrets file
    }
  }

  async getSecret(key: string): Promise<string | undefined> {
    return this.secrets.get(key);
  }

  async setSecret(key: string, value: string): Promise<void> {
    this.secrets.set(key, value);
    await this.persist();
  }

  async deleteSecret(key: string): Promise<void> {
    this.secrets.delete(key);
    await this.persist();
  }

  async listSecrets(): Promise<string[]> {
    return Array.from(this.secrets.keys());
  }

  private async persist(): Promise<void> {
    const fs = await import('fs/promises');
    const data = Object.fromEntries(this.secrets);
    const encrypted = this.encrypt(JSON.stringify(data));
    await fs.writeFile('.secrets.enc', encrypted);
  }

  private encrypt(text: string): string {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-gcm', this.encryptionKey, iv);

    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const authTag = cipher.getAuthTag();

    return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
  }

  private decrypt(data: string): string {
    const [ivHex, authTagHex, encrypted] = data.split(':');
    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');

    const decipher = crypto.createDecipheriv('aes-256-gcm', this.encryptionKey, iv);
    decipher.setAuthTag(authTag);

    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  }
}

// ============================================================================
// CONFIGURATION MANAGER
// ============================================================================

/**
 * Main configuration manager with hot reload and multi-tenant support
 */
class ConfigurationManager extends EventEmitter {
  private config: Record<string, any> = {};
  private sources: ConfigSource[] = [];
  private secretsProvider?: SecretsProvider;
  private schema: ConfigSchema;
  private tenantConfigs = new Map<string, Record<string, any>>();
  private initialized = false;

  constructor() {
    super();
    this.schema = buildMycelixSchema();
  }

  /**
   * Add a configuration source
   */
  addSource(source: ConfigSource): this {
    this.sources.push(source);
    this.sources.sort((a, b) => a.priority - b.priority);
    return this;
  }

  /**
   * Set the secrets provider
   */
  setSecretsProvider(provider: SecretsProvider): this {
    this.secretsProvider = provider;
    return this;
  }

  /**
   * Initialize configuration from all sources
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      throw new Error('Configuration already initialized');
    }

    // Initialize secrets provider
    if (this.secretsProvider) {
      await this.secretsProvider.initialize();
    }

    // Load from all sources (lower priority first, higher priority overrides)
    let merged: Record<string, any> = {};

    for (const source of this.sources) {
      const sourceConfig = await source.load();
      merged = this.deepMerge(merged, sourceConfig);

      // Setup watch if supported
      if (source.watch) {
        source.watch((changes) => this.handleConfigChange(source.name, changes));
      }
    }

    // Load secrets
    merged = await this.loadSecrets(merged);

    // Apply defaults and transforms
    merged = this.schema.applyDefaults(merged);
    merged = this.schema.applyTransforms(merged);

    // Validate
    const validation = this.schema.validate(merged);
    if (!validation.valid) {
      const errorMsg = validation.errors
        .map(e => `  - ${e.path}: ${e.message}`)
        .join('\n');
      throw new Error(`Configuration validation failed:\n${errorMsg}`);
    }

    // Log warnings
    for (const warning of validation.warnings) {
      console.warn(`Config warning: ${warning.path} - ${warning.message}`);
    }

    this.config = merged;
    this.initialized = true;

    this.emit('initialized', this.getConfig());
  }

  /**
   * Load secrets from the secrets provider
   */
  private async loadSecrets(config: Record<string, any>): Promise<Record<string, any>> {
    if (!this.secretsProvider) {
      return config;
    }

    const schema = this.schema.getSchema();
    const result = { ...config };

    for (const [key, def] of Object.entries(schema)) {
      if (def.sensitive && !result[key]) {
        const secretValue = await this.secretsProvider.getSecret(key);
        if (secretValue) {
          result[key] = secretValue;
        }
      }
    }

    return result;
  }

  /**
   * Handle configuration changes from watched sources
   */
  private async handleConfigChange(sourceName: string, changes: Record<string, any>): Promise<void> {
    const oldConfig = { ...this.config };

    // Reload all sources to get merged config
    let merged: Record<string, any> = {};
    for (const source of this.sources) {
      const sourceConfig = await source.load();
      merged = this.deepMerge(merged, sourceConfig);
    }

    merged = await this.loadSecrets(merged);
    merged = this.schema.applyDefaults(merged);
    merged = this.schema.applyTransforms(merged);

    const validation = this.schema.validate(merged);
    if (!validation.valid) {
      console.error('Config change validation failed, keeping previous config');
      return;
    }

    this.config = merged;

    // Emit change events for modified keys
    const changedKeys = this.findChangedKeys(oldConfig, this.config);
    for (const key of changedKeys) {
      this.emit(`change:${key}`, this.config[key], oldConfig[key]);
    }

    this.emit('change', { source: sourceName, changes: changedKeys });
  }

  /**
   * Find keys that changed between two configs
   */
  private findChangedKeys(oldConfig: Record<string, any>, newConfig: Record<string, any>): string[] {
    const changed: string[] = [];
    const allKeys = new Set([...Object.keys(oldConfig), ...Object.keys(newConfig)]);

    for (const key of allKeys) {
      if (JSON.stringify(oldConfig[key]) !== JSON.stringify(newConfig[key])) {
        changed.push(key);
      }
    }

    return changed;
  }

  /**
   * Deep merge two objects
   */
  private deepMerge(target: Record<string, any>, source: Record<string, any>): Record<string, any> {
    const result = { ...target };

    for (const [key, value] of Object.entries(source)) {
      if (value !== undefined) {
        if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
          result[key] = this.deepMerge(result[key] || {}, value);
        } else {
          result[key] = value;
        }
      }
    }

    return result;
  }

  /**
   * Get a configuration value
   */
  get<T = any>(key: string, defaultValue?: T): T {
    const value = key.split('.').reduce((obj, k) => obj?.[k], this.config);
    return (value !== undefined ? value : defaultValue) as T;
  }

  /**
   * Get typed configuration sections
   */
  getConfig(): MycelixConfig {
    return this.buildTypedConfig(this.config);
  }

  /**
   * Get tenant-specific configuration
   */
  getTenantConfig(tenantId: string): MycelixConfig {
    const baseConfig = this.getConfig();
    const tenantOverrides = this.tenantConfigs.get(tenantId) || {};

    return this.deepMerge(baseConfig, tenantOverrides) as MycelixConfig;
  }

  /**
   * Set tenant-specific configuration overrides
   */
  setTenantConfig(tenantId: string, overrides: Partial<MycelixConfig>): void {
    this.tenantConfigs.set(tenantId, overrides);
  }

  /**
   * Build typed configuration from raw config
   */
  private buildTypedConfig(raw: Record<string, any>): MycelixConfig {
    return {
      environment: raw.NODE_ENV || 'development',
      serviceName: raw.SERVICE_NAME || 'mycelix-music',
      serviceVersion: raw.SERVICE_VERSION || '1.0.0',
      instanceId: raw.INSTANCE_ID || crypto.randomUUID(),

      server: {
        host: raw.SERVER_HOST || '0.0.0.0',
        port: raw.SERVER_PORT || 3000,
        trustProxy: raw.SERVER_TRUST_PROXY || false,
        corsOrigins: (raw.CORS_ORIGINS || '*').split(','),
        rateLimitWindowMs: raw.RATE_LIMIT_WINDOW_MS || 60000,
        rateLimitMaxRequests: raw.RATE_LIMIT_MAX_REQUESTS || 100,
      },

      database: {
        host: raw.DATABASE_HOST,
        port: raw.DATABASE_PORT || 5432,
        name: raw.DATABASE_NAME,
        username: raw.DATABASE_USERNAME,
        password: raw.DATABASE_PASSWORD,
        poolMin: raw.DATABASE_POOL_MIN || 2,
        poolMax: raw.DATABASE_POOL_MAX || 10,
        ssl: raw.DATABASE_SSL !== false,
        replicaHosts: raw.DATABASE_REPLICA_HOSTS?.split(',').filter(Boolean),
      },

      redis: {
        host: raw.REDIS_HOST || 'localhost',
        port: raw.REDIS_PORT || 6379,
        password: raw.REDIS_PASSWORD,
        db: raw.REDIS_DB || 0,
        cluster: raw.REDIS_CLUSTER || false,
        sentinelHosts: raw.REDIS_SENTINEL_HOSTS?.split(',').filter(Boolean),
        sentinelName: raw.REDIS_SENTINEL_NAME || 'mymaster',
      },

      auth: {
        jwtSecret: raw.JWT_SECRET,
        jwtExpiresIn: raw.JWT_EXPIRES_IN || '1h',
        refreshTokenExpiresIn: raw.REFRESH_TOKEN_EXPIRES_IN || '7d',
        bcryptRounds: raw.BCRYPT_ROUNDS || 12,
        mfaIssuer: raw.MFA_ISSUER || 'Mycelix Music',
        oauth: {
          google: raw.OAUTH_GOOGLE_CLIENT_ID ? {
            clientId: raw.OAUTH_GOOGLE_CLIENT_ID,
            clientSecret: raw.OAUTH_GOOGLE_CLIENT_SECRET,
          } : undefined,
          apple: raw.OAUTH_APPLE_CLIENT_ID ? {
            clientId: raw.OAUTH_APPLE_CLIENT_ID,
            teamId: raw.OAUTH_APPLE_TEAM_ID,
            keyId: raw.OAUTH_APPLE_KEY_ID,
            privateKey: raw.OAUTH_APPLE_PRIVATE_KEY,
          } : undefined,
          spotify: raw.OAUTH_SPOTIFY_CLIENT_ID ? {
            clientId: raw.OAUTH_SPOTIFY_CLIENT_ID,
            clientSecret: raw.OAUTH_SPOTIFY_CLIENT_SECRET,
          } : undefined,
        },
      },

      storage: {
        provider: raw.STORAGE_PROVIDER || 'local',
        bucket: raw.STORAGE_BUCKET || 'mycelix-media',
        region: raw.STORAGE_REGION,
        accessKeyId: raw.STORAGE_ACCESS_KEY_ID,
        secretAccessKey: raw.STORAGE_SECRET_ACCESS_KEY,
        cdnUrl: raw.STORAGE_CDN_URL,
        maxFileSize: raw.STORAGE_MAX_FILE_SIZE || 500 * 1024 * 1024,
      },

      streaming: {
        hlsSegmentDuration: raw.STREAMING_HLS_SEGMENT_DURATION || 6,
        dashSegmentDuration: raw.STREAMING_DASH_SEGMENT_DURATION || 4,
        maxBitrate: raw.STREAMING_MAX_BITRATE || 320000,
        transcodingConcurrency: raw.STREAMING_TRANSCODING_CONCURRENCY || 4,
        cacheEnabled: raw.STREAMING_CACHE_ENABLED !== false,
        cacheTtl: raw.STREAMING_CACHE_TTL || 3600,
      },

      search: {
        provider: raw.SEARCH_PROVIDER || 'elasticsearch',
        hosts: (raw.SEARCH_HOSTS || 'http://localhost:9200').split(','),
        apiKey: raw.SEARCH_API_KEY,
        indexPrefix: raw.SEARCH_INDEX_PREFIX || 'mycelix_',
      },

      messaging: {
        provider: raw.MESSAGING_PROVIDER || 'kafka',
        brokers: (raw.MESSAGING_BROKERS || 'localhost:9092').split(','),
        username: raw.MESSAGING_USERNAME,
        password: raw.MESSAGING_PASSWORD,
        groupId: raw.MESSAGING_GROUP_ID || 'mycelix-consumers',
      },

      observability: {
        logging: {
          level: raw.LOG_LEVEL || 'info',
          format: raw.LOG_FORMAT || 'json',
          outputs: (raw.LOG_OUTPUTS || 'stdout').split(','),
        },
        metrics: {
          enabled: raw.METRICS_ENABLED !== false,
          port: raw.METRICS_PORT || 9090,
          path: raw.METRICS_PATH || '/metrics',
        },
        tracing: {
          enabled: raw.TRACING_ENABLED !== false,
          samplingRate: raw.TRACING_SAMPLING_RATE || 0.1,
          exporterEndpoint: raw.TRACING_EXPORTER_ENDPOINT,
        },
      },

      featureFlags: {
        provider: raw.FEATURE_FLAGS_PROVIDER || 'local',
        apiKey: raw.FEATURE_FLAGS_API_KEY,
        refreshIntervalMs: raw.FEATURE_FLAGS_REFRESH_INTERVAL_MS || 30000,
        defaults: {},
      },

      external: {
        stripe: raw.STRIPE_SECRET_KEY ? {
          secretKey: raw.STRIPE_SECRET_KEY,
          webhookSecret: raw.STRIPE_WEBHOOK_SECRET,
        } : undefined,
        sendgrid: raw.SENDGRID_API_KEY ? {
          apiKey: raw.SENDGRID_API_KEY,
          fromEmail: raw.SENDGRID_FROM_EMAIL || 'noreply@mycelix.music',
        } : undefined,
        twilio: raw.TWILIO_ACCOUNT_SID ? {
          accountSid: raw.TWILIO_ACCOUNT_SID,
          authToken: raw.TWILIO_AUTH_TOKEN,
          fromNumber: raw.TWILIO_FROM_NUMBER,
        } : undefined,
        openai: raw.OPENAI_API_KEY ? {
          apiKey: raw.OPENAI_API_KEY,
          model: raw.OPENAI_MODEL || 'gpt-4',
        } : undefined,
        anthropic: raw.ANTHROPIC_API_KEY ? {
          apiKey: raw.ANTHROPIC_API_KEY,
          model: raw.ANTHROPIC_MODEL || 'claude-3-opus-20240229',
        } : undefined,
      },

      multiTenant: {
        enabled: raw.MULTI_TENANT_ENABLED || false,
        isolationLevel: raw.MULTI_TENANT_ISOLATION_LEVEL || 'row',
        defaultTenant: raw.MULTI_TENANT_DEFAULT_TENANT || 'default',
      },
    };
  }

  /**
   * Export configuration (with secrets masked)
   */
  export(includeSensitive: boolean = false): Record<string, any> {
    const schema = this.schema.getSchema();
    const result: Record<string, any> = {};

    for (const [key, value] of Object.entries(this.config)) {
      const def = schema[key];
      if (def?.sensitive && !includeSensitive) {
        result[key] = '***REDACTED***';
      } else {
        result[key] = value;
      }
    }

    return result;
  }
}

// ============================================================================
// ENVIRONMENT-SPECIFIC CONFIGURATION
// ============================================================================

/**
 * Environment-specific configuration factory
 */
class EnvironmentConfigFactory {
  static createForEnvironment(env: string): ConfigurationManager {
    const manager = new ConfigurationManager();

    // Base configuration file
    manager.addSource(new JsonFileConfigSource('config/default.json', 10));

    // Environment-specific configuration
    manager.addSource(new JsonFileConfigSource(`config/${env}.json`, 20));

    // Local overrides (not committed to git)
    manager.addSource(new JsonFileConfigSource('config/local.json', 30));

    // Environment variables (highest priority)
    manager.addSource(new EnvConfigSource());

    // Setup secrets provider based on environment
    const secretsProvider = this.createSecretsProvider(env);
    if (secretsProvider) {
      manager.setSecretsProvider(secretsProvider);
    }

    return manager;
  }

  private static createSecretsProvider(env: string): SecretsProvider | undefined {
    // Production uses real secrets management
    if (env === 'production') {
      if (process.env.VAULT_ADDR) {
        return new VaultSecretsProvider(process.env.VAULT_ADDR);
      }
      if (process.env.AWS_REGION) {
        return new AWSSecretsProvider(process.env.AWS_REGION);
      }
      if (process.env.AZURE_KEY_VAULT_URL) {
        return new AzureKeyVaultProvider(process.env.AZURE_KEY_VAULT_URL);
      }
    }

    // Development uses local encrypted secrets
    return new LocalSecretsProvider(process.env.SECRETS_ENCRYPTION_KEY);
  }
}

// ============================================================================
// CONFIGURATION DECORATORS (for dependency injection)
// ============================================================================

/**
 * Configuration injection metadata
 */
const CONFIG_METADATA_KEY = Symbol('config');

/**
 * Decorator to inject configuration values
 */
function Config(key: string) {
  return function (target: any, propertyKey: string) {
    Reflect.defineMetadata(CONFIG_METADATA_KEY, key, target, propertyKey);
  };
}

/**
 * Decorator to inject entire configuration section
 */
function ConfigSection(section: keyof MycelixConfig) {
  return function (target: any, propertyKey: string) {
    Reflect.defineMetadata(CONFIG_METADATA_KEY, `section:${section}`, target, propertyKey);
  };
}

/**
 * Resolve configuration dependencies on an object
 */
function resolveConfigDependencies(target: any, configManager: ConfigurationManager): void {
  const prototype = Object.getPrototypeOf(target);
  const properties = Object.getOwnPropertyNames(prototype);

  for (const property of properties) {
    const configKey = Reflect.getMetadata(CONFIG_METADATA_KEY, prototype, property);
    if (configKey) {
      if (configKey.startsWith('section:')) {
        const section = configKey.replace('section:', '') as keyof MycelixConfig;
        const config = configManager.getConfig();
        target[property] = config[section];
      } else {
        target[property] = configManager.get(configKey);
      }
    }
  }
}

// ============================================================================
// CONFIGURATION UTILITIES
// ============================================================================

/**
 * Configuration utilities for common operations
 */
class ConfigUtils {
  /**
   * Generate a sample configuration file
   */
  static generateSampleConfig(format: 'json' | 'yaml' | 'env'): string {
    const schema = buildMycelixSchema().getSchema();

    switch (format) {
      case 'json':
        return this.generateJsonSample(schema);
      case 'yaml':
        return this.generateYamlSample(schema);
      case 'env':
        return this.generateEnvSample(schema);
    }
  }

  private static generateJsonSample(schema: Record<string, SchemaDefinition>): string {
    const config: Record<string, any> = {};

    for (const [key, def] of Object.entries(schema)) {
      if (def.sensitive) {
        config[key] = '<SECRET>';
      } else if (def.default !== undefined) {
        config[key] = typeof def.default === 'function' ? def.default() : def.default;
      } else if (def.enum) {
        config[key] = def.enum[0];
      } else {
        config[key] = `<${def.type.toUpperCase()}>`;
      }
    }

    return JSON.stringify(config, null, 2);
  }

  private static generateYamlSample(schema: Record<string, SchemaDefinition>): string {
    const lines: string[] = ['# Mycelix Music Platform Configuration', ''];

    for (const [key, def] of Object.entries(schema)) {
      if (def.description) {
        lines.push(`# ${def.description}`);
      }

      let value: string;
      if (def.sensitive) {
        value = '<SECRET>';
      } else if (def.default !== undefined) {
        const defaultVal = typeof def.default === 'function' ? def.default() : def.default;
        value = typeof defaultVal === 'string' ? `"${defaultVal}"` : String(defaultVal);
      } else if (def.enum) {
        value = def.enum[0];
      } else {
        value = `<${def.type.toUpperCase()}>`;
      }

      lines.push(`${key}: ${value}`);
      lines.push('');
    }

    return lines.join('\n');
  }

  private static generateEnvSample(schema: Record<string, SchemaDefinition>): string {
    const lines: string[] = ['# Mycelix Music Platform Environment Variables', ''];

    for (const [key, def] of Object.entries(schema)) {
      if (def.description) {
        lines.push(`# ${def.description}`);
      }

      let value: string;
      if (def.sensitive) {
        value = '';
      } else if (def.default !== undefined) {
        const defaultVal = typeof def.default === 'function' ? def.default() : def.default;
        value = String(defaultVal);
      } else if (def.enum) {
        lines.push(`# Options: ${def.enum.join(', ')}`);
        value = def.enum[0];
      } else {
        value = '';
      }

      const required = def.required ? ' # REQUIRED' : '';
      lines.push(`${key}=${value}${required}`);
      lines.push('');
    }

    return lines.join('\n');
  }

  /**
   * Validate a configuration file
   */
  static async validateConfigFile(filePath: string): Promise<ValidationResult> {
    const fs = await import('fs/promises');
    const content = await fs.readFile(filePath, 'utf-8');

    let config: Record<string, any>;
    if (filePath.endsWith('.json')) {
      config = JSON.parse(content);
    } else if (filePath.endsWith('.yaml') || filePath.endsWith('.yml')) {
      const yamlSource = new YamlFileConfigSource(filePath);
      config = await yamlSource.load();
    } else {
      throw new Error('Unsupported config file format');
    }

    return buildMycelixSchema().validate(config);
  }

  /**
   * Compare two configurations and return differences
   */
  static compareConfigs(
    config1: Record<string, any>,
    config2: Record<string, any>
  ): { added: string[]; removed: string[]; changed: string[] } {
    const added: string[] = [];
    const removed: string[] = [];
    const changed: string[] = [];

    const allKeys = new Set([...Object.keys(config1), ...Object.keys(config2)]);

    for (const key of allKeys) {
      if (!(key in config1)) {
        added.push(key);
      } else if (!(key in config2)) {
        removed.push(key);
      } else if (JSON.stringify(config1[key]) !== JSON.stringify(config2[key])) {
        changed.push(key);
      }
    }

    return { added, removed, changed };
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

let configInstance: ConfigurationManager | null = null;

/**
 * Get or create the configuration manager singleton
 */
async function getConfig(): Promise<ConfigurationManager> {
  if (!configInstance) {
    const env = process.env.NODE_ENV || 'development';
    configInstance = EnvironmentConfigFactory.createForEnvironment(env);
    await configInstance.initialize();
  }
  return configInstance;
}

/**
 * Reset configuration (for testing)
 */
function resetConfig(): void {
  configInstance = null;
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  // Main classes
  ConfigurationManager,
  ConfigSchema,

  // Configuration sources
  ConfigSource,
  EnvConfigSource,
  JsonFileConfigSource,
  YamlFileConfigSource,

  // Secrets providers
  SecretsProvider,
  VaultSecretsProvider,
  AWSSecretsProvider,
  AzureKeyVaultProvider,
  LocalSecretsProvider,

  // Factory and utilities
  EnvironmentConfigFactory,
  ConfigUtils,

  // Decorators
  Config,
  ConfigSection,
  resolveConfigDependencies,

  // Types
  MycelixConfig,
  ValidationResult,
  ValidationError,
  ValidationWarning,
  SchemaDefinition,

  // Singleton access
  getConfig,
  resetConfig,
};

/**
 * Create the configuration system
 */
export async function createConfigurationSystem(): Promise<{
  manager: ConfigurationManager;
  config: MycelixConfig;
  utils: typeof ConfigUtils;
}> {
  const manager = await getConfig();

  return {
    manager,
    config: manager.getConfig(),
    utils: ConfigUtils,
  };
}
