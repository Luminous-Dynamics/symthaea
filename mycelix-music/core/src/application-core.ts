// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Platform - Application Core & Integration Layer
 *
 * Production-grade application orchestration with:
 * - Dependency injection container with lifecycle management
 * - Service registry and discovery
 * - Cross-cutting concerns (logging, errors, auth)
 * - Event bus for internal communication
 * - Health checks and graceful shutdown
 * - Request context propagation
 */

import { EventEmitter } from 'events';
import * as crypto from 'crypto';

// ============================================================================
// DEPENDENCY INJECTION CONTAINER
// ============================================================================

type Constructor<T = any> = new (...args: any[]) => T;
type Factory<T = any> = (container: Container) => T | Promise<T>;

interface ServiceDescriptor<T = any> {
  id: string;
  implementation?: Constructor<T>;
  factory?: Factory<T>;
  instance?: T;
  lifecycle: 'singleton' | 'transient' | 'scoped';
  dependencies: string[];
  tags: string[];
  initialized: boolean;
}

interface ScopeContext {
  id: string;
  instances: Map<string, any>;
  parent?: ScopeContext;
}

/**
 * Dependency injection container with lifecycle management
 */
class Container {
  private services = new Map<string, ServiceDescriptor>();
  private aliases = new Map<string, string>();
  private scopes = new Map<string, ScopeContext>();
  private initOrder: string[] = [];
  private shutdownOrder: string[] = [];

  /**
   * Register a service with its constructor
   */
  register<T>(
    id: string,
    implementation: Constructor<T>,
    options: {
      lifecycle?: 'singleton' | 'transient' | 'scoped';
      dependencies?: string[];
      tags?: string[];
    } = {}
  ): this {
    this.services.set(id, {
      id,
      implementation,
      lifecycle: options.lifecycle || 'singleton',
      dependencies: options.dependencies || [],
      tags: options.tags || [],
      initialized: false,
    });
    return this;
  }

  /**
   * Register a service with a factory function
   */
  registerFactory<T>(
    id: string,
    factory: Factory<T>,
    options: {
      lifecycle?: 'singleton' | 'transient' | 'scoped';
      dependencies?: string[];
      tags?: string[];
    } = {}
  ): this {
    this.services.set(id, {
      id,
      factory,
      lifecycle: options.lifecycle || 'singleton',
      dependencies: options.dependencies || [],
      tags: options.tags || [],
      initialized: false,
    });
    return this;
  }

  /**
   * Register a pre-created instance
   */
  registerInstance<T>(id: string, instance: T, tags: string[] = []): this {
    this.services.set(id, {
      id,
      instance,
      lifecycle: 'singleton',
      dependencies: [],
      tags,
      initialized: true,
    });
    return this;
  }

  /**
   * Create an alias for a service
   */
  alias(alias: string, target: string): this {
    this.aliases.set(alias, target);
    return this;
  }

  /**
   * Resolve a service by id
   */
  async resolve<T>(id: string, scopeId?: string): Promise<T> {
    // Resolve alias
    const resolvedId = this.aliases.get(id) || id;
    const descriptor = this.services.get(resolvedId);

    if (!descriptor) {
      throw new Error(`Service '${id}' not registered`);
    }

    // Handle scoped services
    if (descriptor.lifecycle === 'scoped' && scopeId) {
      const scope = this.scopes.get(scopeId);
      if (scope?.instances.has(resolvedId)) {
        return scope.instances.get(resolvedId);
      }
    }

    // Return existing singleton instance
    if (descriptor.lifecycle === 'singleton' && descriptor.instance) {
      return descriptor.instance;
    }

    // Create new instance
    const instance = await this.createInstance<T>(descriptor);

    // Store based on lifecycle
    if (descriptor.lifecycle === 'singleton') {
      descriptor.instance = instance;
      descriptor.initialized = true;
      this.initOrder.push(resolvedId);
    } else if (descriptor.lifecycle === 'scoped' && scopeId) {
      const scope = this.scopes.get(scopeId);
      scope?.instances.set(resolvedId, instance);
    }

    return instance;
  }

  /**
   * Create instance from descriptor
   */
  private async createInstance<T>(descriptor: ServiceDescriptor): Promise<T> {
    // Resolve dependencies first
    const deps = await Promise.all(
      descriptor.dependencies.map(dep => this.resolve(dep))
    );

    if (descriptor.factory) {
      return descriptor.factory(this);
    }

    if (descriptor.implementation) {
      return new descriptor.implementation(...deps);
    }

    throw new Error(`No implementation for service '${descriptor.id}'`);
  }

  /**
   * Get all services with a specific tag
   */
  async resolveByTag<T>(tag: string): Promise<T[]> {
    const services: T[] = [];

    for (const [id, descriptor] of this.services) {
      if (descriptor.tags.includes(tag)) {
        services.push(await this.resolve<T>(id));
      }
    }

    return services;
  }

  /**
   * Create a new scope for scoped services
   */
  createScope(parentId?: string): string {
    const scopeId = crypto.randomUUID();
    const parent = parentId ? this.scopes.get(parentId) : undefined;

    this.scopes.set(scopeId, {
      id: scopeId,
      instances: new Map(),
      parent,
    });

    return scopeId;
  }

  /**
   * Dispose a scope and its instances
   */
  async disposeScope(scopeId: string): Promise<void> {
    const scope = this.scopes.get(scopeId);
    if (!scope) return;

    // Dispose instances in reverse order
    const instances = Array.from(scope.instances.values()).reverse();
    for (const instance of instances) {
      if (typeof instance.dispose === 'function') {
        await instance.dispose();
      }
    }

    this.scopes.delete(scopeId);
  }

  /**
   * Initialize all singleton services
   */
  async initializeAll(): Promise<void> {
    for (const [id, descriptor] of this.services) {
      if (descriptor.lifecycle === 'singleton' && !descriptor.initialized) {
        await this.resolve(id);
      }
    }
  }

  /**
   * Shutdown all services in reverse initialization order
   */
  async shutdown(): Promise<void> {
    const order = [...this.initOrder].reverse();

    for (const id of order) {
      const descriptor = this.services.get(id);
      if (descriptor?.instance && typeof descriptor.instance.shutdown === 'function') {
        try {
          await descriptor.instance.shutdown();
        } catch (error) {
          console.error(`Error shutting down ${id}:`, error);
        }
      }
    }

    // Clear all scopes
    for (const scopeId of this.scopes.keys()) {
      await this.disposeScope(scopeId);
    }
  }

  /**
   * Check if a service is registered
   */
  has(id: string): boolean {
    const resolvedId = this.aliases.get(id) || id;
    return this.services.has(resolvedId);
  }

  /**
   * Get service metadata
   */
  getMetadata(id: string): ServiceDescriptor | undefined {
    const resolvedId = this.aliases.get(id) || id;
    return this.services.get(resolvedId);
  }
}

// ============================================================================
// SERVICE REGISTRY
// ============================================================================

interface ServiceInfo {
  id: string;
  name: string;
  version: string;
  status: 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
  health: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  metadata: Record<string, any>;
  endpoints: ServiceEndpoint[];
  dependencies: string[];
  startedAt?: Date;
  lastHealthCheck?: Date;
}

interface ServiceEndpoint {
  protocol: 'http' | 'grpc' | 'websocket' | 'tcp';
  host: string;
  port: number;
  path?: string;
}

interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  message?: string;
  details?: Record<string, any>;
  duration: number;
}

type HealthCheckFn = () => Promise<HealthCheckResult>;

/**
 * Service registry for service discovery and health monitoring
 */
class ServiceRegistry extends EventEmitter {
  private services = new Map<string, ServiceInfo>();
  private healthChecks = new Map<string, HealthCheckFn>();
  private healthCheckInterval?: NodeJS.Timeout;

  constructor(private checkIntervalMs: number = 30000) {
    super();
  }

  /**
   * Register a service
   */
  register(info: Omit<ServiceInfo, 'status' | 'health'>): void {
    this.services.set(info.id, {
      ...info,
      status: 'stopped',
      health: 'unknown',
    });

    this.emit('registered', info.id);
  }

  /**
   * Unregister a service
   */
  unregister(serviceId: string): void {
    this.services.delete(serviceId);
    this.healthChecks.delete(serviceId);
    this.emit('unregistered', serviceId);
  }

  /**
   * Update service status
   */
  updateStatus(serviceId: string, status: ServiceInfo['status']): void {
    const service = this.services.get(serviceId);
    if (service) {
      const oldStatus = service.status;
      service.status = status;

      if (status === 'running') {
        service.startedAt = new Date();
      }

      this.emit('statusChanged', { serviceId, oldStatus, newStatus: status });
    }
  }

  /**
   * Register a health check function
   */
  registerHealthCheck(serviceId: string, check: HealthCheckFn): void {
    this.healthChecks.set(serviceId, check);
  }

  /**
   * Run health check for a service
   */
  async checkHealth(serviceId: string): Promise<HealthCheckResult> {
    const check = this.healthChecks.get(serviceId);
    const service = this.services.get(serviceId);

    if (!check || !service) {
      return { status: 'unknown', duration: 0 };
    }

    const start = Date.now();
    try {
      const result = await Promise.race([
        check(),
        new Promise<HealthCheckResult>((_, reject) =>
          setTimeout(() => reject(new Error('Health check timeout')), 10000)
        ),
      ]);

      result.duration = Date.now() - start;
      service.health = result.status;
      service.lastHealthCheck = new Date();

      this.emit('healthCheck', { serviceId, result });
      return result;
    } catch (error) {
      const result: HealthCheckResult = {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Unknown error',
        duration: Date.now() - start,
      };

      service.health = 'unhealthy';
      service.lastHealthCheck = new Date();

      this.emit('healthCheck', { serviceId, result });
      return result;
    }
  }

  /**
   * Run health checks for all services
   */
  async checkAllHealth(): Promise<Map<string, HealthCheckResult>> {
    const results = new Map<string, HealthCheckResult>();

    await Promise.all(
      Array.from(this.services.keys()).map(async (serviceId) => {
        const result = await this.checkHealth(serviceId);
        results.set(serviceId, result);
      })
    );

    return results;
  }

  /**
   * Start periodic health checks
   */
  startHealthChecks(): void {
    if (this.healthCheckInterval) return;

    this.healthCheckInterval = setInterval(async () => {
      await this.checkAllHealth();
    }, this.checkIntervalMs);
  }

  /**
   * Stop periodic health checks
   */
  stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = undefined;
    }
  }

  /**
   * Get service info
   */
  get(serviceId: string): ServiceInfo | undefined {
    return this.services.get(serviceId);
  }

  /**
   * Get all services
   */
  getAll(): ServiceInfo[] {
    return Array.from(this.services.values());
  }

  /**
   * Find services by criteria
   */
  find(predicate: (service: ServiceInfo) => boolean): ServiceInfo[] {
    return this.getAll().filter(predicate);
  }

  /**
   * Get aggregated health status
   */
  getAggregatedHealth(): { status: 'healthy' | 'degraded' | 'unhealthy'; services: Record<string, string> } {
    const services: Record<string, string> = {};
    let hasUnhealthy = false;
    let hasDegraded = false;

    for (const [id, info] of this.services) {
      services[id] = info.health;
      if (info.health === 'unhealthy') hasUnhealthy = true;
      if (info.health === 'degraded') hasDegraded = true;
    }

    return {
      status: hasUnhealthy ? 'unhealthy' : hasDegraded ? 'degraded' : 'healthy',
      services,
    };
  }
}

// ============================================================================
// EVENT BUS
// ============================================================================

interface DomainEvent {
  id: string;
  type: string;
  aggregateId?: string;
  aggregateType?: string;
  payload: any;
  metadata: {
    timestamp: Date;
    correlationId?: string;
    causationId?: string;
    userId?: string;
    tenantId?: string;
    version: number;
  };
}

type EventHandler<T = any> = (event: DomainEvent & { payload: T }) => Promise<void>;

interface EventSubscription {
  id: string;
  eventType: string;
  handler: EventHandler;
  options: {
    async: boolean;
    retries: number;
    retryDelayMs: number;
  };
}

/**
 * Internal event bus for domain events
 */
class EventBus extends EventEmitter {
  private subscriptions = new Map<string, EventSubscription[]>();
  private pendingEvents: DomainEvent[] = [];
  private processing = false;
  private eventStore?: EventStore;

  /**
   * Set event store for persistence
   */
  setEventStore(store: EventStore): void {
    this.eventStore = store;
  }

  /**
   * Subscribe to an event type
   */
  subscribe<T>(
    eventType: string,
    handler: EventHandler<T>,
    options: Partial<EventSubscription['options']> = {}
  ): string {
    const subscription: EventSubscription = {
      id: crypto.randomUUID(),
      eventType,
      handler: handler as EventHandler,
      options: {
        async: options.async ?? true,
        retries: options.retries ?? 3,
        retryDelayMs: options.retryDelayMs ?? 1000,
      },
    };

    const subs = this.subscriptions.get(eventType) || [];
    subs.push(subscription);
    this.subscriptions.set(eventType, subs);

    return subscription.id;
  }

  /**
   * Unsubscribe from events
   */
  unsubscribe(subscriptionId: string): void {
    for (const [eventType, subs] of this.subscriptions) {
      const filtered = subs.filter(s => s.id !== subscriptionId);
      if (filtered.length !== subs.length) {
        this.subscriptions.set(eventType, filtered);
        break;
      }
    }
  }

  /**
   * Publish an event
   */
  async publish<T>(
    type: string,
    payload: T,
    options: Partial<DomainEvent['metadata']> = {}
  ): Promise<DomainEvent> {
    const event: DomainEvent = {
      id: crypto.randomUUID(),
      type,
      payload,
      metadata: {
        timestamp: new Date(),
        version: 1,
        ...options,
      },
    };

    // Persist event if store is configured
    if (this.eventStore) {
      await this.eventStore.append(event);
    }

    // Queue for processing
    this.pendingEvents.push(event);

    // Process if not already processing
    if (!this.processing) {
      await this.processEvents();
    }

    return event;
  }

  /**
   * Process pending events
   */
  private async processEvents(): Promise<void> {
    this.processing = true;

    while (this.pendingEvents.length > 0) {
      const event = this.pendingEvents.shift()!;
      await this.dispatchEvent(event);
    }

    this.processing = false;
  }

  /**
   * Dispatch event to subscribers
   */
  private async dispatchEvent(event: DomainEvent): Promise<void> {
    const subscriptions = this.subscriptions.get(event.type) || [];

    // Also dispatch to wildcard subscribers
    const wildcardSubs = this.subscriptions.get('*') || [];
    const allSubs = [...subscriptions, ...wildcardSubs];

    const promises = allSubs.map(async (sub) => {
      if (sub.options.async) {
        // Fire and forget with retry
        this.executeWithRetry(sub, event).catch(err => {
          this.emit('error', { subscription: sub.id, event, error: err });
        });
      } else {
        // Synchronous execution
        await this.executeWithRetry(sub, event);
      }
    });

    // Wait for sync handlers
    const syncPromises = allSubs
      .filter(s => !s.options.async)
      .map((_, i) => promises[i]);

    await Promise.all(syncPromises);
  }

  /**
   * Execute handler with retry logic
   */
  private async executeWithRetry(
    subscription: EventSubscription,
    event: DomainEvent
  ): Promise<void> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= subscription.options.retries; attempt++) {
      try {
        await subscription.handler(event);
        return;
      } catch (error) {
        lastError = error as Error;

        if (attempt < subscription.options.retries) {
          await this.delay(subscription.options.retryDelayMs * Math.pow(2, attempt));
        }
      }
    }

    throw lastError;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Event store interface for event persistence
 */
interface EventStore {
  append(event: DomainEvent): Promise<void>;
  getEvents(aggregateId: string, fromVersion?: number): Promise<DomainEvent[]>;
  getAllEvents(fromTimestamp?: Date): Promise<DomainEvent[]>;
}

/**
 * In-memory event store (for development)
 */
class InMemoryEventStore implements EventStore {
  private events: DomainEvent[] = [];

  async append(event: DomainEvent): Promise<void> {
    this.events.push(event);
  }

  async getEvents(aggregateId: string, fromVersion = 0): Promise<DomainEvent[]> {
    return this.events.filter(
      e => e.aggregateId === aggregateId && e.metadata.version > fromVersion
    );
  }

  async getAllEvents(fromTimestamp?: Date): Promise<DomainEvent[]> {
    if (!fromTimestamp) return [...this.events];
    return this.events.filter(e => e.metadata.timestamp >= fromTimestamp);
  }
}

// ============================================================================
// REQUEST CONTEXT
// ============================================================================

interface RequestContext {
  id: string;
  correlationId: string;
  userId?: string;
  tenantId?: string;
  sessionId?: string;
  ipAddress?: string;
  userAgent?: string;
  startTime: Date;
  attributes: Map<string, any>;
}

/**
 * Async local storage for request context propagation
 */
class ContextManager {
  private static storage = new Map<string, RequestContext>();
  private static currentContextId: string | null = null;

  /**
   * Create a new request context
   */
  static create(initial: Partial<RequestContext> = {}): RequestContext {
    const context: RequestContext = {
      id: crypto.randomUUID(),
      correlationId: initial.correlationId || crypto.randomUUID(),
      userId: initial.userId,
      tenantId: initial.tenantId,
      sessionId: initial.sessionId,
      ipAddress: initial.ipAddress,
      userAgent: initial.userAgent,
      startTime: new Date(),
      attributes: new Map(Object.entries(initial.attributes || {})),
    };

    this.storage.set(context.id, context);
    return context;
  }

  /**
   * Run function with context
   */
  static async run<T>(context: RequestContext, fn: () => Promise<T>): Promise<T> {
    const previousId = this.currentContextId;
    this.currentContextId = context.id;

    try {
      return await fn();
    } finally {
      this.currentContextId = previousId;
    }
  }

  /**
   * Get current context
   */
  static current(): RequestContext | undefined {
    if (!this.currentContextId) return undefined;
    return this.storage.get(this.currentContextId);
  }

  /**
   * Get or create context
   */
  static getOrCreate(): RequestContext {
    return this.current() || this.create();
  }

  /**
   * Set attribute on current context
   */
  static setAttribute(key: string, value: any): void {
    const context = this.current();
    if (context) {
      context.attributes.set(key, value);
    }
  }

  /**
   * Get attribute from current context
   */
  static getAttribute<T>(key: string): T | undefined {
    return this.current()?.attributes.get(key);
  }

  /**
   * Clear context
   */
  static clear(contextId: string): void {
    this.storage.delete(contextId);
  }
}

// ============================================================================
// CROSS-CUTTING CONCERNS
// ============================================================================

/**
 * Unified logger with context awareness
 */
interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: Date;
  context?: Partial<RequestContext>;
  data?: Record<string, any>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
}

class Logger {
  private outputs: LogOutput[] = [];

  constructor(
    private serviceName: string,
    private minLevel: LogEntry['level'] = 'info'
  ) {}

  addOutput(output: LogOutput): void {
    this.outputs.push(output);
  }

  private shouldLog(level: LogEntry['level']): boolean {
    const levels = ['debug', 'info', 'warn', 'error'];
    return levels.indexOf(level) >= levels.indexOf(this.minLevel);
  }

  private log(level: LogEntry['level'], message: string, data?: Record<string, any>, error?: Error): void {
    if (!this.shouldLog(level)) return;

    const context = ContextManager.current();
    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date(),
      context: context ? {
        id: context.id,
        correlationId: context.correlationId,
        userId: context.userId,
        tenantId: context.tenantId,
      } : undefined,
      data: { service: this.serviceName, ...data },
      error: error ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
      } : undefined,
    };

    for (const output of this.outputs) {
      output.write(entry);
    }
  }

  debug(message: string, data?: Record<string, any>): void {
    this.log('debug', message, data);
  }

  info(message: string, data?: Record<string, any>): void {
    this.log('info', message, data);
  }

  warn(message: string, data?: Record<string, any>): void {
    this.log('warn', message, data);
  }

  error(message: string, error?: Error, data?: Record<string, any>): void {
    this.log('error', message, data, error);
  }

  child(additionalData: Record<string, any>): Logger {
    const child = new Logger(this.serviceName, this.minLevel);
    child.outputs = this.outputs;
    return child;
  }
}

interface LogOutput {
  write(entry: LogEntry): void;
}

class ConsoleLogOutput implements LogOutput {
  constructor(private format: 'json' | 'pretty' = 'json') {}

  write(entry: LogEntry): void {
    if (this.format === 'json') {
      console.log(JSON.stringify(entry));
    } else {
      const timestamp = entry.timestamp.toISOString();
      const level = entry.level.toUpperCase().padEnd(5);
      const contextInfo = entry.context?.correlationId
        ? `[${entry.context.correlationId.slice(0, 8)}]`
        : '';

      console.log(`${timestamp} ${level} ${contextInfo} ${entry.message}`);

      if (entry.data) {
        console.log('  Data:', JSON.stringify(entry.data, null, 2));
      }

      if (entry.error) {
        console.log('  Error:', entry.error.message);
        if (entry.error.stack) {
          console.log(entry.error.stack);
        }
      }
    }
  }
}

/**
 * Application error types
 */
class AppError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500,
    public details?: Record<string, any>
  ) {
    super(message);
    this.name = 'AppError';
  }

  toJSON() {
    return {
      error: {
        code: this.code,
        message: this.message,
        details: this.details,
      },
    };
  }
}

class ValidationError extends AppError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'VALIDATION_ERROR', 400, details);
    this.name = 'ValidationError';
  }
}

class NotFoundError extends AppError {
  constructor(resource: string, id: string) {
    super(`${resource} not found: ${id}`, 'NOT_FOUND', 404, { resource, id });
    this.name = 'NotFoundError';
  }
}

class UnauthorizedError extends AppError {
  constructor(message: string = 'Unauthorized') {
    super(message, 'UNAUTHORIZED', 401);
    this.name = 'UnauthorizedError';
  }
}

class ForbiddenError extends AppError {
  constructor(message: string = 'Forbidden') {
    super(message, 'FORBIDDEN', 403);
    this.name = 'ForbiddenError';
  }
}

class ConflictError extends AppError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'CONFLICT', 409, details);
    this.name = 'ConflictError';
  }
}

class RateLimitError extends AppError {
  constructor(retryAfter: number) {
    super('Rate limit exceeded', 'RATE_LIMIT_EXCEEDED', 429, { retryAfter });
    this.name = 'RateLimitError';
  }
}

/**
 * Error handler middleware
 */
class ErrorHandler {
  constructor(private logger: Logger) {}

  handle(error: Error): { statusCode: number; body: any } {
    if (error instanceof AppError) {
      this.logger.warn(`Application error: ${error.message}`, { code: error.code });
      return {
        statusCode: error.statusCode,
        body: error.toJSON(),
      };
    }

    // Unexpected error
    this.logger.error('Unexpected error', error);

    return {
      statusCode: 500,
      body: {
        error: {
          code: 'INTERNAL_ERROR',
          message: 'An unexpected error occurred',
        },
      },
    };
  }
}

// ============================================================================
// MIDDLEWARE CHAIN
// ============================================================================

type Middleware<TContext = any> = (
  context: TContext,
  next: () => Promise<void>
) => Promise<void>;

/**
 * Middleware pipeline builder
 */
class MiddlewarePipeline<TContext = any> {
  private middlewares: Middleware<TContext>[] = [];

  use(middleware: Middleware<TContext>): this {
    this.middlewares.push(middleware);
    return this;
  }

  async execute(context: TContext): Promise<void> {
    let index = 0;

    const next = async (): Promise<void> => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        await middleware(context, next);
      }
    };

    await next();
  }
}

/**
 * Common middleware factories
 */
const Middlewares = {
  /**
   * Request timing middleware
   */
  timing(): Middleware<any> {
    return async (ctx, next) => {
      const start = Date.now();
      try {
        await next();
      } finally {
        const duration = Date.now() - start;
        ctx.timing = duration;
      }
    };
  },

  /**
   * Error handling middleware
   */
  errorHandler(handler: ErrorHandler): Middleware<any> {
    return async (ctx, next) => {
      try {
        await next();
      } catch (error) {
        const result = handler.handle(error as Error);
        ctx.statusCode = result.statusCode;
        ctx.body = result.body;
      }
    };
  },

  /**
   * Request context middleware
   */
  requestContext(): Middleware<any> {
    return async (ctx, next) => {
      const context = ContextManager.create({
        correlationId: ctx.headers?.['x-correlation-id'],
        userId: ctx.user?.id,
        tenantId: ctx.headers?.['x-tenant-id'],
        ipAddress: ctx.ip,
        userAgent: ctx.headers?.['user-agent'],
      });

      await ContextManager.run(context, async () => {
        await next();
      });

      ContextManager.clear(context.id);
    };
  },

  /**
   * Rate limiting middleware
   */
  rateLimit(options: { windowMs: number; max: number }): Middleware<any> {
    const requests = new Map<string, { count: number; resetAt: number }>();

    return async (ctx, next) => {
      const key = ctx.user?.id || ctx.ip || 'anonymous';
      const now = Date.now();

      let record = requests.get(key);
      if (!record || record.resetAt < now) {
        record = { count: 0, resetAt: now + options.windowMs };
        requests.set(key, record);
      }

      record.count++;

      if (record.count > options.max) {
        throw new RateLimitError(Math.ceil((record.resetAt - now) / 1000));
      }

      await next();
    };
  },

  /**
   * Logging middleware
   */
  logging(logger: Logger): Middleware<any> {
    return async (ctx, next) => {
      logger.info(`Request: ${ctx.method} ${ctx.path}`, {
        query: ctx.query,
        userAgent: ctx.headers?.['user-agent'],
      });

      await next();

      logger.info(`Response: ${ctx.statusCode}`, {
        duration: ctx.timing,
        contentLength: ctx.responseSize,
      });
    };
  },
};

// ============================================================================
// APPLICATION LIFECYCLE
// ============================================================================

interface LifecycleHook {
  name: string;
  priority: number;
  handler: () => Promise<void>;
}

/**
 * Application lifecycle manager
 */
class LifecycleManager extends EventEmitter {
  private startupHooks: LifecycleHook[] = [];
  private shutdownHooks: LifecycleHook[] = [];
  private state: 'stopped' | 'starting' | 'running' | 'stopping' = 'stopped';
  private shutdownSignals = ['SIGTERM', 'SIGINT', 'SIGUSR2'];

  /**
   * Register startup hook
   */
  onStartup(name: string, handler: () => Promise<void>, priority: number = 100): void {
    this.startupHooks.push({ name, handler, priority });
    this.startupHooks.sort((a, b) => a.priority - b.priority);
  }

  /**
   * Register shutdown hook
   */
  onShutdown(name: string, handler: () => Promise<void>, priority: number = 100): void {
    this.shutdownHooks.push({ name, handler, priority });
    this.shutdownHooks.sort((a, b) => a.priority - b.priority);
  }

  /**
   * Start the application
   */
  async start(): Promise<void> {
    if (this.state !== 'stopped') {
      throw new Error(`Cannot start from state: ${this.state}`);
    }

    this.state = 'starting';
    this.emit('starting');

    for (const hook of this.startupHooks) {
      this.emit('hookStarting', hook.name);
      try {
        await hook.handler();
        this.emit('hookCompleted', hook.name);
      } catch (error) {
        this.emit('hookFailed', { name: hook.name, error });
        throw error;
      }
    }

    this.state = 'running';
    this.emit('started');

    // Setup shutdown signal handlers
    this.setupSignalHandlers();
  }

  /**
   * Stop the application
   */
  async stop(): Promise<void> {
    if (this.state !== 'running') {
      return;
    }

    this.state = 'stopping';
    this.emit('stopping');

    // Run shutdown hooks in reverse priority order
    const hooks = [...this.shutdownHooks].reverse();

    for (const hook of hooks) {
      this.emit('hookStarting', hook.name);
      try {
        await Promise.race([
          hook.handler(),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Shutdown hook timeout')), 30000)
          ),
        ]);
        this.emit('hookCompleted', hook.name);
      } catch (error) {
        this.emit('hookFailed', { name: hook.name, error });
        // Continue with other hooks even if one fails
      }
    }

    this.state = 'stopped';
    this.emit('stopped');
  }

  /**
   * Setup signal handlers for graceful shutdown
   */
  private setupSignalHandlers(): void {
    const handler = async (signal: string) => {
      console.log(`\nReceived ${signal}, shutting down gracefully...`);
      await this.stop();
      process.exit(0);
    };

    for (const signal of this.shutdownSignals) {
      process.once(signal, () => handler(signal));
    }
  }

  /**
   * Get current state
   */
  getState(): string {
    return this.state;
  }
}

// ============================================================================
// APPLICATION BUILDER
// ============================================================================

interface ApplicationConfig {
  name: string;
  version: string;
  environment: string;
}

/**
 * Main application class that orchestrates everything
 */
class Application {
  readonly container: Container;
  readonly registry: ServiceRegistry;
  readonly eventBus: EventBus;
  readonly lifecycle: LifecycleManager;
  readonly logger: Logger;

  private middlewarePipeline: MiddlewarePipeline<any>;
  private errorHandler: ErrorHandler;

  constructor(private config: ApplicationConfig) {
    this.container = new Container();
    this.registry = new ServiceRegistry();
    this.eventBus = new EventBus();
    this.lifecycle = new LifecycleManager();
    this.logger = new Logger(config.name);
    this.middlewarePipeline = new MiddlewarePipeline();
    this.errorHandler = new ErrorHandler(this.logger);

    // Setup default log output
    this.logger.addOutput(
      new ConsoleLogOutput(config.environment === 'development' ? 'pretty' : 'json')
    );

    // Register core services
    this.container.registerInstance('config', this.config);
    this.container.registerInstance('logger', this.logger);
    this.container.registerInstance('eventBus', this.eventBus);
    this.container.registerInstance('registry', this.registry);

    // Setup default middleware
    this.middlewarePipeline
      .use(Middlewares.timing())
      .use(Middlewares.requestContext())
      .use(Middlewares.errorHandler(this.errorHandler))
      .use(Middlewares.logging(this.logger));
  }

  /**
   * Use middleware
   */
  use(middleware: Middleware<any>): this {
    this.middlewarePipeline.use(middleware);
    return this;
  }

  /**
   * Register a service
   */
  service<T>(
    id: string,
    implementation: Constructor<T>,
    options: {
      lifecycle?: 'singleton' | 'transient' | 'scoped';
      dependencies?: string[];
      tags?: string[];
    } = {}
  ): this {
    this.container.register(id, implementation, options);
    return this;
  }

  /**
   * Register a factory
   */
  factory<T>(
    id: string,
    factory: Factory<T>,
    options: {
      lifecycle?: 'singleton' | 'transient' | 'scoped';
      dependencies?: string[];
      tags?: string[];
    } = {}
  ): this {
    this.container.registerFactory(id, factory, options);
    return this;
  }

  /**
   * Add startup hook
   */
  onStartup(name: string, handler: () => Promise<void>, priority?: number): this {
    this.lifecycle.onStartup(name, handler, priority);
    return this;
  }

  /**
   * Add shutdown hook
   */
  onShutdown(name: string, handler: () => Promise<void>, priority?: number): this {
    this.lifecycle.onShutdown(name, handler, priority);
    return this;
  }

  /**
   * Subscribe to events
   */
  on<T>(eventType: string, handler: EventHandler<T>): this {
    this.eventBus.subscribe(eventType, handler);
    return this;
  }

  /**
   * Start the application
   */
  async start(): Promise<void> {
    this.logger.info(`Starting ${this.config.name} v${this.config.version}`, {
      environment: this.config.environment,
      nodeVersion: process.version,
    });

    // Initialize all services
    this.lifecycle.onStartup('container', async () => {
      await this.container.initializeAll();
    }, 10);

    // Start health checks
    this.lifecycle.onStartup('healthChecks', async () => {
      this.registry.startHealthChecks();
    }, 90);

    // Lifecycle events
    this.lifecycle.on('hookStarting', (name) => {
      this.logger.debug(`Starting: ${name}`);
    });

    this.lifecycle.on('hookCompleted', (name) => {
      this.logger.debug(`Completed: ${name}`);
    });

    this.lifecycle.on('hookFailed', ({ name, error }) => {
      this.logger.error(`Failed: ${name}`, error as Error);
    });

    // Emit started event
    this.lifecycle.on('started', () => {
      this.eventBus.publish('application.started', {
        name: this.config.name,
        version: this.config.version,
      });
    });

    await this.lifecycle.start();

    this.logger.info(`${this.config.name} started successfully`);
  }

  /**
   * Stop the application
   */
  async stop(): Promise<void> {
    this.logger.info(`Stopping ${this.config.name}...`);

    // Stop health checks
    this.lifecycle.onShutdown('healthChecks', async () => {
      this.registry.stopHealthChecks();
    }, 10);

    // Shutdown container
    this.lifecycle.onShutdown('container', async () => {
      await this.container.shutdown();
    }, 90);

    await this.lifecycle.stop();

    this.logger.info(`${this.config.name} stopped`);
  }

  /**
   * Handle a request through the middleware pipeline
   */
  async handleRequest(context: any): Promise<any> {
    await this.middlewarePipeline.execute(context);
    return context;
  }
}

/**
 * Application builder with fluent API
 */
class ApplicationBuilder {
  private config: ApplicationConfig = {
    name: 'mycelix-music',
    version: '1.0.0',
    environment: process.env.NODE_ENV || 'development',
  };

  private serviceRegistrations: Array<() => void> = [];
  private startupHooks: Array<{ name: string; handler: () => Promise<void>; priority?: number }> = [];
  private shutdownHooks: Array<{ name: string; handler: () => Promise<void>; priority?: number }> = [];

  /**
   * Set application name
   */
  name(name: string): this {
    this.config.name = name;
    return this;
  }

  /**
   * Set application version
   */
  version(version: string): this {
    this.config.version = version;
    return this;
  }

  /**
   * Set environment
   */
  environment(env: string): this {
    this.config.environment = env;
    return this;
  }

  /**
   * Add startup hook
   */
  onStartup(name: string, handler: () => Promise<void>, priority?: number): this {
    this.startupHooks.push({ name, handler, priority });
    return this;
  }

  /**
   * Add shutdown hook
   */
  onShutdown(name: string, handler: () => Promise<void>, priority?: number): this {
    this.shutdownHooks.push({ name, handler, priority });
    return this;
  }

  /**
   * Build the application
   */
  build(): Application {
    const app = new Application(this.config);

    for (const hook of this.startupHooks) {
      app.onStartup(hook.name, hook.handler, hook.priority);
    }

    for (const hook of this.shutdownHooks) {
      app.onShutdown(hook.name, hook.handler, hook.priority);
    }

    return app;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  // Container
  Container,
  Constructor,
  Factory,
  ServiceDescriptor,

  // Registry
  ServiceRegistry,
  ServiceInfo,
  ServiceEndpoint,
  HealthCheckResult,
  HealthCheckFn,

  // Events
  EventBus,
  DomainEvent,
  EventHandler,
  EventStore,
  InMemoryEventStore,

  // Context
  ContextManager,
  RequestContext,

  // Logging
  Logger,
  LogEntry,
  LogOutput,
  ConsoleLogOutput,

  // Errors
  AppError,
  ValidationError,
  NotFoundError,
  UnauthorizedError,
  ForbiddenError,
  ConflictError,
  RateLimitError,
  ErrorHandler,

  // Middleware
  Middleware,
  MiddlewarePipeline,
  Middlewares,

  // Lifecycle
  LifecycleManager,
  LifecycleHook,

  // Application
  Application,
  ApplicationBuilder,
  ApplicationConfig,
};

/**
 * Create the application core
 */
export function createApplication(config?: Partial<ApplicationConfig>): Application {
  const builder = new ApplicationBuilder();

  if (config?.name) builder.name(config.name);
  if (config?.version) builder.version(config.version);
  if (config?.environment) builder.environment(config.environment);

  return builder.build();
}

/**
 * Create the application builder
 */
export function createApplicationBuilder(): ApplicationBuilder {
  return new ApplicationBuilder();
}
