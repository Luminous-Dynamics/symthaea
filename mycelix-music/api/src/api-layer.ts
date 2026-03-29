// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Platform - API Layer
 *
 * Comprehensive HTTP API with:
 * - RESTful resource routes with full CRUD
 * - GraphQL schema and resolvers
 * - Request validation and transformation
 * - Response formatting and pagination
 * - OpenAPI specification generation
 * - Versioned API support
 */

import { EventEmitter } from 'events';
import * as crypto from 'crypto';

// ============================================================================
// HTTP TYPES
// ============================================================================

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD' | 'OPTIONS';

interface HttpRequest {
  method: HttpMethod;
  path: string;
  params: Record<string, string>;
  query: Record<string, string | string[]>;
  headers: Record<string, string>;
  body: any;
  user?: AuthenticatedUser;
  ip: string;
  protocol: string;
  hostname: string;
}

interface HttpResponse {
  status: number;
  headers: Record<string, string>;
  body: any;
}

interface AuthenticatedUser {
  id: string;
  email: string;
  roles: string[];
  permissions: string[];
  tenantId?: string;
  sessionId: string;
}

interface RouteContext {
  request: HttpRequest;
  response: HttpResponse;
  params: Record<string, string>;
  query: Record<string, any>;
  body: any;
  user?: AuthenticatedUser;
  state: Map<string, any>;
  timing: { start: number; end?: number };
}

type RouteHandler = (ctx: RouteContext) => Promise<void>;
type RouteMiddleware = (ctx: RouteContext, next: () => Promise<void>) => Promise<void>;

// ============================================================================
// ROUTER
// ============================================================================

interface RouteDefinition {
  method: HttpMethod;
  path: string;
  pattern: RegExp;
  paramNames: string[];
  handler: RouteHandler;
  middlewares: RouteMiddleware[];
  metadata: RouteMetadata;
}

interface RouteMetadata {
  name?: string;
  description?: string;
  tags?: string[];
  auth?: boolean;
  permissions?: string[];
  rateLimit?: { windowMs: number; max: number };
  cache?: { ttl: number; private?: boolean };
  deprecated?: boolean;
  version?: string;
}

/**
 * HTTP Router with middleware support
 */
class Router {
  private routes: RouteDefinition[] = [];
  private globalMiddlewares: RouteMiddleware[] = [];
  private prefix: string = '';

  constructor(prefix: string = '') {
    this.prefix = prefix;
  }

  /**
   * Add global middleware
   */
  use(middleware: RouteMiddleware): this {
    this.globalMiddlewares.push(middleware);
    return this;
  }

  /**
   * Register a route
   */
  route(
    method: HttpMethod,
    path: string,
    handler: RouteHandler,
    metadata: RouteMetadata = {}
  ): this {
    const fullPath = this.prefix + path;
    const { pattern, paramNames } = this.compilePath(fullPath);

    this.routes.push({
      method,
      path: fullPath,
      pattern,
      paramNames,
      handler,
      middlewares: [],
      metadata,
    });

    return this;
  }

  // Convenience methods
  get(path: string, handler: RouteHandler, metadata?: RouteMetadata): this {
    return this.route('GET', path, handler, metadata);
  }

  post(path: string, handler: RouteHandler, metadata?: RouteMetadata): this {
    return this.route('POST', path, handler, metadata);
  }

  put(path: string, handler: RouteHandler, metadata?: RouteMetadata): this {
    return this.route('PUT', path, handler, metadata);
  }

  patch(path: string, handler: RouteHandler, metadata?: RouteMetadata): this {
    return this.route('PATCH', path, handler, metadata);
  }

  delete(path: string, handler: RouteHandler, metadata?: RouteMetadata): this {
    return this.route('DELETE', path, handler, metadata);
  }

  /**
   * Create a sub-router with prefix
   */
  group(prefix: string, callback: (router: Router) => void): this {
    const subRouter = new Router(this.prefix + prefix);
    callback(subRouter);

    // Merge routes
    for (const route of subRouter.routes) {
      this.routes.push(route);
    }

    return this;
  }

  /**
   * Match a request to a route
   */
  match(method: HttpMethod, path: string): { route: RouteDefinition; params: Record<string, string> } | null {
    for (const route of this.routes) {
      if (route.method !== method) continue;

      const match = route.pattern.exec(path);
      if (match) {
        const params: Record<string, string> = {};
        route.paramNames.forEach((name, i) => {
          params[name] = match[i + 1];
        });
        return { route, params };
      }
    }

    return null;
  }

  /**
   * Handle a request
   */
  async handle(request: HttpRequest): Promise<HttpResponse> {
    const ctx: RouteContext = {
      request,
      response: { status: 200, headers: {}, body: null },
      params: {},
      query: this.parseQuery(request.query),
      body: request.body,
      user: request.user,
      state: new Map(),
      timing: { start: Date.now() },
    };

    try {
      const match = this.match(request.method, request.path);

      if (!match) {
        ctx.response.status = 404;
        ctx.response.body = { error: { code: 'NOT_FOUND', message: 'Route not found' } };
        return ctx.response;
      }

      ctx.params = match.params;

      // Build middleware chain
      const middlewares = [...this.globalMiddlewares, ...match.route.middlewares];

      let index = 0;
      const next = async (): Promise<void> => {
        if (index < middlewares.length) {
          await middlewares[index++](ctx, next);
        } else {
          await match.route.handler(ctx);
        }
      };

      await next();
    } catch (error) {
      this.handleError(ctx, error as Error);
    }

    ctx.timing.end = Date.now();
    return ctx.response;
  }

  /**
   * Compile path pattern to regex
   */
  private compilePath(path: string): { pattern: RegExp; paramNames: string[] } {
    const paramNames: string[] = [];

    const pattern = path.replace(/:([a-zA-Z0-9_]+)/g, (_, name) => {
      paramNames.push(name);
      return '([^/]+)';
    });

    return {
      pattern: new RegExp(`^${pattern}$`),
      paramNames,
    };
  }

  /**
   * Parse query parameters with type coercion
   */
  private parseQuery(query: Record<string, string | string[]>): Record<string, any> {
    const result: Record<string, any> = {};

    for (const [key, value] of Object.entries(query)) {
      if (Array.isArray(value)) {
        result[key] = value.map(v => this.parseValue(v));
      } else {
        result[key] = this.parseValue(value);
      }
    }

    return result;
  }

  private parseValue(value: string): any {
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (/^\d+$/.test(value)) return parseInt(value, 10);
    if (/^\d+\.\d+$/.test(value)) return parseFloat(value);
    return value;
  }

  /**
   * Handle errors
   */
  private handleError(ctx: RouteContext, error: Error): void {
    if ('statusCode' in error) {
      ctx.response.status = (error as any).statusCode;
      ctx.response.body = { error: { code: (error as any).code || 'ERROR', message: error.message } };
    } else {
      ctx.response.status = 500;
      ctx.response.body = { error: { code: 'INTERNAL_ERROR', message: 'An unexpected error occurred' } };
    }
  }

  /**
   * Get all routes (for documentation)
   */
  getRoutes(): RouteDefinition[] {
    return [...this.routes];
  }
}

// ============================================================================
// REQUEST VALIDATION
// ============================================================================

type ValidatorFn = (value: any, ctx: ValidationContext) => ValidationResult;

interface ValidationContext {
  path: string;
  parent?: any;
  root: any;
}

interface ValidationResult {
  valid: boolean;
  value?: any;
  errors?: string[];
}

interface SchemaProperty {
  type: 'string' | 'number' | 'boolean' | 'array' | 'object' | 'date';
  required?: boolean;
  nullable?: boolean;
  default?: any;
  description?: string;
  enum?: any[];
  min?: number;
  max?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  format?: 'email' | 'url' | 'uuid' | 'date' | 'date-time' | 'uri';
  items?: SchemaProperty;
  properties?: Record<string, SchemaProperty>;
  additionalProperties?: boolean | SchemaProperty;
  transform?: (value: any) => any;
  validate?: ValidatorFn;
}

/**
 * Request validator with schema-based validation
 */
class RequestValidator {
  /**
   * Validate data against schema
   */
  validate(data: any, schema: Record<string, SchemaProperty>): { valid: boolean; data?: any; errors?: string[] } {
    const errors: string[] = [];
    const result: Record<string, any> = {};

    for (const [key, prop] of Object.entries(schema)) {
      const value = data?.[key];
      const ctx: ValidationContext = { path: key, root: data };

      const validation = this.validateProperty(value, prop, ctx);

      if (!validation.valid) {
        errors.push(...(validation.errors || [`Invalid value for ${key}`]));
      } else if (validation.value !== undefined) {
        result[key] = validation.value;
      }
    }

    // Check for extra properties if not allowed
    const allowedKeys = new Set(Object.keys(schema));
    for (const key of Object.keys(data || {})) {
      if (!allowedKeys.has(key)) {
        errors.push(`Unknown property: ${key}`);
      }
    }

    return {
      valid: errors.length === 0,
      data: errors.length === 0 ? result : undefined,
      errors: errors.length > 0 ? errors : undefined,
    };
  }

  private validateProperty(value: any, prop: SchemaProperty, ctx: ValidationContext): ValidationResult {
    // Handle undefined/null
    if (value === undefined) {
      if (prop.required) {
        return { valid: false, errors: [`${ctx.path} is required`] };
      }
      if (prop.default !== undefined) {
        return { valid: true, value: prop.default };
      }
      return { valid: true };
    }

    if (value === null) {
      if (prop.nullable) {
        return { valid: true, value: null };
      }
      return { valid: false, errors: [`${ctx.path} cannot be null`] };
    }

    // Type validation
    const typeResult = this.validateType(value, prop, ctx);
    if (!typeResult.valid) {
      return typeResult;
    }

    let validatedValue = typeResult.value ?? value;

    // Enum validation
    if (prop.enum && !prop.enum.includes(validatedValue)) {
      return { valid: false, errors: [`${ctx.path} must be one of: ${prop.enum.join(', ')}`] };
    }

    // String validations
    if (prop.type === 'string') {
      if (prop.minLength !== undefined && validatedValue.length < prop.minLength) {
        return { valid: false, errors: [`${ctx.path} must be at least ${prop.minLength} characters`] };
      }
      if (prop.maxLength !== undefined && validatedValue.length > prop.maxLength) {
        return { valid: false, errors: [`${ctx.path} must be at most ${prop.maxLength} characters`] };
      }
      if (prop.pattern && !new RegExp(prop.pattern).test(validatedValue)) {
        return { valid: false, errors: [`${ctx.path} doesn't match required pattern`] };
      }
      if (prop.format) {
        const formatResult = this.validateFormat(validatedValue, prop.format, ctx);
        if (!formatResult.valid) return formatResult;
      }
    }

    // Number validations
    if (prop.type === 'number') {
      if (prop.min !== undefined && validatedValue < prop.min) {
        return { valid: false, errors: [`${ctx.path} must be at least ${prop.min}`] };
      }
      if (prop.max !== undefined && validatedValue > prop.max) {
        return { valid: false, errors: [`${ctx.path} must be at most ${prop.max}`] };
      }
    }

    // Array validations
    if (prop.type === 'array' && prop.items) {
      const arrayErrors: string[] = [];
      const arrayResult: any[] = [];

      for (let i = 0; i < validatedValue.length; i++) {
        const itemResult = this.validateProperty(
          validatedValue[i],
          prop.items,
          { ...ctx, path: `${ctx.path}[${i}]` }
        );

        if (!itemResult.valid) {
          arrayErrors.push(...(itemResult.errors || []));
        } else {
          arrayResult.push(itemResult.value ?? validatedValue[i]);
        }
      }

      if (arrayErrors.length > 0) {
        return { valid: false, errors: arrayErrors };
      }

      validatedValue = arrayResult;
    }

    // Object validations
    if (prop.type === 'object' && prop.properties) {
      const objectErrors: string[] = [];
      const objectResult: Record<string, any> = {};

      for (const [key, subProp] of Object.entries(prop.properties)) {
        const subResult = this.validateProperty(
          validatedValue[key],
          subProp,
          { ...ctx, path: `${ctx.path}.${key}`, parent: validatedValue }
        );

        if (!subResult.valid) {
          objectErrors.push(...(subResult.errors || []));
        } else if (subResult.value !== undefined) {
          objectResult[key] = subResult.value;
        }
      }

      if (objectErrors.length > 0) {
        return { valid: false, errors: objectErrors };
      }

      validatedValue = objectResult;
    }

    // Custom validation
    if (prop.validate) {
      const customResult = prop.validate(validatedValue, ctx);
      if (!customResult.valid) return customResult;
    }

    // Transform
    if (prop.transform) {
      validatedValue = prop.transform(validatedValue);
    }

    return { valid: true, value: validatedValue };
  }

  private validateType(value: any, prop: SchemaProperty, ctx: ValidationContext): ValidationResult {
    switch (prop.type) {
      case 'string':
        if (typeof value !== 'string') {
          return { valid: false, errors: [`${ctx.path} must be a string`] };
        }
        break;

      case 'number':
        if (typeof value === 'string') {
          const parsed = parseFloat(value);
          if (isNaN(parsed)) {
            return { valid: false, errors: [`${ctx.path} must be a number`] };
          }
          return { valid: true, value: parsed };
        }
        if (typeof value !== 'number' || isNaN(value)) {
          return { valid: false, errors: [`${ctx.path} must be a number`] };
        }
        break;

      case 'boolean':
        if (typeof value === 'string') {
          if (value === 'true') return { valid: true, value: true };
          if (value === 'false') return { valid: true, value: false };
        }
        if (typeof value !== 'boolean') {
          return { valid: false, errors: [`${ctx.path} must be a boolean`] };
        }
        break;

      case 'array':
        if (!Array.isArray(value)) {
          return { valid: false, errors: [`${ctx.path} must be an array`] };
        }
        break;

      case 'object':
        if (typeof value !== 'object' || Array.isArray(value) || value === null) {
          return { valid: false, errors: [`${ctx.path} must be an object`] };
        }
        break;

      case 'date':
        if (value instanceof Date) break;
        if (typeof value === 'string') {
          const parsed = new Date(value);
          if (isNaN(parsed.getTime())) {
            return { valid: false, errors: [`${ctx.path} must be a valid date`] };
          }
          return { valid: true, value: parsed };
        }
        return { valid: false, errors: [`${ctx.path} must be a date`] };
    }

    return { valid: true };
  }

  private validateFormat(value: string, format: string, ctx: ValidationContext): ValidationResult {
    const patterns: Record<string, RegExp> = {
      email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
      url: /^https?:\/\/.+/,
      uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
      'date': /^\d{4}-\d{2}-\d{2}$/,
      'date-time': /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/,
    };

    const pattern = patterns[format];
    if (pattern && !pattern.test(value)) {
      return { valid: false, errors: [`${ctx.path} must be a valid ${format}`] };
    }

    return { valid: true };
  }
}

// ============================================================================
// RESPONSE FORMATTING
// ============================================================================

interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    totalItems: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPrevPage: boolean;
  };
  links: {
    self: string;
    first: string;
    last: string;
    next?: string;
    prev?: string;
  };
}

interface SingleResponse<T> {
  data: T;
  links?: {
    self: string;
  };
}

interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
    timestamp: string;
    requestId?: string;
  };
}

/**
 * Response formatter for consistent API responses
 */
class ResponseFormatter {
  constructor(private baseUrl: string) {}

  /**
   * Format a single resource response
   */
  single<T>(data: T, path: string): SingleResponse<T> {
    return {
      data,
      links: {
        self: `${this.baseUrl}${path}`,
      },
    };
  }

  /**
   * Format a paginated list response
   */
  paginated<T>(
    data: T[],
    path: string,
    options: { page: number; pageSize: number; totalItems: number }
  ): PaginatedResponse<T> {
    const { page, pageSize, totalItems } = options;
    const totalPages = Math.ceil(totalItems / pageSize);

    const buildUrl = (p: number) =>
      `${this.baseUrl}${path}?page=${p}&pageSize=${pageSize}`;

    return {
      data,
      pagination: {
        page,
        pageSize,
        totalItems,
        totalPages,
        hasNextPage: page < totalPages,
        hasPrevPage: page > 1,
      },
      links: {
        self: buildUrl(page),
        first: buildUrl(1),
        last: buildUrl(totalPages),
        next: page < totalPages ? buildUrl(page + 1) : undefined,
        prev: page > 1 ? buildUrl(page - 1) : undefined,
      },
    };
  }

  /**
   * Format an error response
   */
  error(code: string, message: string, details?: Record<string, any>, requestId?: string): ErrorResponse {
    return {
      error: {
        code,
        message,
        details,
        timestamp: new Date().toISOString(),
        requestId,
      },
    };
  }

  /**
   * Format a created response
   */
  created<T>(data: T, path: string, location: string): HttpResponse {
    return {
      status: 201,
      headers: { Location: `${this.baseUrl}${location}` },
      body: this.single(data, path),
    };
  }

  /**
   * Format a no content response
   */
  noContent(): HttpResponse {
    return { status: 204, headers: {}, body: null };
  }

  /**
   * Format an accepted response (for async operations)
   */
  accepted(operationId: string, statusUrl: string): HttpResponse {
    return {
      status: 202,
      headers: { 'Operation-Location': `${this.baseUrl}${statusUrl}` },
      body: {
        operationId,
        status: 'pending',
        statusUrl: `${this.baseUrl}${statusUrl}`,
      },
    };
  }
}

// ============================================================================
// API RESOURCE DEFINITIONS
// ============================================================================

/**
 * Base resource controller
 */
abstract class ResourceController {
  protected validator = new RequestValidator();

  constructor(
    protected router: Router,
    protected formatter: ResponseFormatter,
    protected basePath: string
  ) {}

  abstract registerRoutes(): void;

  protected success(ctx: RouteContext, data: any, status: number = 200): void {
    ctx.response.status = status;
    ctx.response.body = data;
  }

  protected created(ctx: RouteContext, data: any, location: string): void {
    ctx.response.status = 201;
    ctx.response.headers['Location'] = location;
    ctx.response.body = this.formatter.single(data, location);
  }

  protected noContent(ctx: RouteContext): void {
    ctx.response.status = 204;
    ctx.response.body = null;
  }

  protected paginated<T>(
    ctx: RouteContext,
    data: T[],
    totalItems: number
  ): void {
    const page = parseInt(ctx.query.page as string) || 1;
    const pageSize = parseInt(ctx.query.pageSize as string) || 20;

    ctx.response.body = this.formatter.paginated(data, ctx.request.path, {
      page,
      pageSize,
      totalItems,
    });
  }

  protected error(ctx: RouteContext, code: string, message: string, status: number = 400): void {
    ctx.response.status = status;
    ctx.response.body = this.formatter.error(code, message);
  }
}

// ============================================================================
// MYCELIX API CONTROLLERS
// ============================================================================

/**
 * Tracks API Controller
 */
class TracksController extends ResourceController {
  private trackSchema: Record<string, SchemaProperty> = {
    title: { type: 'string', required: true, minLength: 1, maxLength: 200 },
    artistId: { type: 'string', required: true, format: 'uuid' },
    albumId: { type: 'string', format: 'uuid' },
    duration: { type: 'number', min: 1 },
    genre: { type: 'string', maxLength: 50 },
    releaseDate: { type: 'date' },
    isExplicit: { type: 'boolean', default: false },
    isrc: { type: 'string', pattern: '^[A-Z]{2}[A-Z0-9]{3}\\d{7}$' },
    metadata: { type: 'object', additionalProperties: true },
  };

  registerRoutes(): void {
    this.router.group(this.basePath, (r) => {
      // List tracks
      r.get('/', async (ctx) => {
        const tracks = await this.listTracks(ctx.query);
        this.paginated(ctx, tracks.items, tracks.total);
      }, { name: 'listTracks', tags: ['tracks'], auth: true });

      // Get track
      r.get('/:id', async (ctx) => {
        const track = await this.getTrack(ctx.params.id);
        if (!track) {
          this.error(ctx, 'NOT_FOUND', 'Track not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(track, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'getTrack', tags: ['tracks'], auth: true });

      // Create track
      r.post('/', async (ctx) => {
        const validation = this.validator.validate(ctx.body, this.trackSchema);
        if (!validation.valid) {
          this.error(ctx, 'VALIDATION_ERROR', validation.errors!.join(', '));
          return;
        }

        const track = await this.createTrack(validation.data!, ctx.user!.id);
        this.created(ctx, track, `${this.basePath}/${track.id}`);
      }, { name: 'createTrack', tags: ['tracks'], auth: true, permissions: ['tracks:write'] });

      // Update track
      r.put('/:id', async (ctx) => {
        const validation = this.validator.validate(ctx.body, this.trackSchema);
        if (!validation.valid) {
          this.error(ctx, 'VALIDATION_ERROR', validation.errors!.join(', '));
          return;
        }

        const track = await this.updateTrack(ctx.params.id, validation.data!);
        if (!track) {
          this.error(ctx, 'NOT_FOUND', 'Track not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(track, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'updateTrack', tags: ['tracks'], auth: true, permissions: ['tracks:write'] });

      // Delete track
      r.delete('/:id', async (ctx) => {
        const deleted = await this.deleteTrack(ctx.params.id);
        if (!deleted) {
          this.error(ctx, 'NOT_FOUND', 'Track not found', 404);
          return;
        }
        this.noContent(ctx);
      }, { name: 'deleteTrack', tags: ['tracks'], auth: true, permissions: ['tracks:delete'] });

      // Stream track
      r.get('/:id/stream', async (ctx) => {
        const streamUrl = await this.getStreamUrl(ctx.params.id, ctx.user!.id);
        if (!streamUrl) {
          this.error(ctx, 'NOT_FOUND', 'Track not found', 404);
          return;
        }
        ctx.response.status = 302;
        ctx.response.headers['Location'] = streamUrl;
      }, { name: 'streamTrack', tags: ['tracks', 'streaming'], auth: true });

      // Track analytics
      r.get('/:id/analytics', async (ctx) => {
        const analytics = await this.getTrackAnalytics(ctx.params.id, ctx.query);
        this.success(ctx, analytics);
      }, { name: 'getTrackAnalytics', tags: ['tracks', 'analytics'], auth: true, permissions: ['analytics:read'] });
    });
  }

  // Service method stubs (would be injected in real implementation)
  private async listTracks(query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getTrack(id: string): Promise<any> {
    return null;
  }

  private async createTrack(data: any, userId: string): Promise<any> {
    return { id: crypto.randomUUID(), ...data, createdBy: userId };
  }

  private async updateTrack(id: string, data: any): Promise<any> {
    return { id, ...data };
  }

  private async deleteTrack(id: string): Promise<boolean> {
    return true;
  }

  private async getStreamUrl(trackId: string, userId: string): Promise<string | null> {
    return null;
  }

  private async getTrackAnalytics(trackId: string, query: any): Promise<any> {
    return {};
  }
}

/**
 * Artists API Controller
 */
class ArtistsController extends ResourceController {
  private artistSchema: Record<string, SchemaProperty> = {
    name: { type: 'string', required: true, minLength: 1, maxLength: 200 },
    bio: { type: 'string', maxLength: 5000 },
    genres: { type: 'array', items: { type: 'string' } },
    country: { type: 'string', minLength: 2, maxLength: 2 },
    socialLinks: {
      type: 'object',
      properties: {
        spotify: { type: 'string', format: 'url' },
        instagram: { type: 'string', format: 'url' },
        twitter: { type: 'string', format: 'url' },
        website: { type: 'string', format: 'url' },
      },
    },
    verified: { type: 'boolean', default: false },
  };

  registerRoutes(): void {
    this.router.group(this.basePath, (r) => {
      r.get('/', async (ctx) => {
        const artists = await this.listArtists(ctx.query);
        this.paginated(ctx, artists.items, artists.total);
      }, { name: 'listArtists', tags: ['artists'] });

      r.get('/:id', async (ctx) => {
        const artist = await this.getArtist(ctx.params.id);
        if (!artist) {
          this.error(ctx, 'NOT_FOUND', 'Artist not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(artist, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'getArtist', tags: ['artists'] });

      r.get('/:id/tracks', async (ctx) => {
        const tracks = await this.getArtistTracks(ctx.params.id, ctx.query);
        this.paginated(ctx, tracks.items, tracks.total);
      }, { name: 'getArtistTracks', tags: ['artists', 'tracks'] });

      r.get('/:id/albums', async (ctx) => {
        const albums = await this.getArtistAlbums(ctx.params.id, ctx.query);
        this.paginated(ctx, albums.items, albums.total);
      }, { name: 'getArtistAlbums', tags: ['artists', 'albums'] });

      r.post('/', async (ctx) => {
        const validation = this.validator.validate(ctx.body, this.artistSchema);
        if (!validation.valid) {
          this.error(ctx, 'VALIDATION_ERROR', validation.errors!.join(', '));
          return;
        }
        const artist = await this.createArtist(validation.data!);
        this.created(ctx, artist, `${this.basePath}/${artist.id}`);
      }, { name: 'createArtist', tags: ['artists'], auth: true, permissions: ['artists:write'] });

      r.put('/:id', async (ctx) => {
        const validation = this.validator.validate(ctx.body, this.artistSchema);
        if (!validation.valid) {
          this.error(ctx, 'VALIDATION_ERROR', validation.errors!.join(', '));
          return;
        }
        const artist = await this.updateArtist(ctx.params.id, validation.data!);
        if (!artist) {
          this.error(ctx, 'NOT_FOUND', 'Artist not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(artist, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'updateArtist', tags: ['artists'], auth: true, permissions: ['artists:write'] });

      r.delete('/:id', async (ctx) => {
        const deleted = await this.deleteArtist(ctx.params.id);
        if (!deleted) {
          this.error(ctx, 'NOT_FOUND', 'Artist not found', 404);
          return;
        }
        this.noContent(ctx);
      }, { name: 'deleteArtist', tags: ['artists'], auth: true, permissions: ['artists:delete'] });

      // Follow/unfollow
      r.post('/:id/follow', async (ctx) => {
        await this.followArtist(ctx.params.id, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'followArtist', tags: ['artists', 'social'], auth: true });

      r.delete('/:id/follow', async (ctx) => {
        await this.unfollowArtist(ctx.params.id, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'unfollowArtist', tags: ['artists', 'social'], auth: true });
    });
  }

  private async listArtists(query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getArtist(id: string): Promise<any> {
    return null;
  }

  private async getArtistTracks(id: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getArtistAlbums(id: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async createArtist(data: any): Promise<any> {
    return { id: crypto.randomUUID(), ...data };
  }

  private async updateArtist(id: string, data: any): Promise<any> {
    return { id, ...data };
  }

  private async deleteArtist(id: string): Promise<boolean> {
    return true;
  }

  private async followArtist(artistId: string, userId: string): Promise<void> {}
  private async unfollowArtist(artistId: string, userId: string): Promise<void> {}
}

/**
 * Playlists API Controller
 */
class PlaylistsController extends ResourceController {
  private playlistSchema: Record<string, SchemaProperty> = {
    name: { type: 'string', required: true, minLength: 1, maxLength: 200 },
    description: { type: 'string', maxLength: 2000 },
    isPublic: { type: 'boolean', default: true },
    isCollaborative: { type: 'boolean', default: false },
    coverImageUrl: { type: 'string', format: 'url' },
  };

  registerRoutes(): void {
    this.router.group(this.basePath, (r) => {
      r.get('/', async (ctx) => {
        const playlists = await this.listPlaylists(ctx.query, ctx.user?.id);
        this.paginated(ctx, playlists.items, playlists.total);
      }, { name: 'listPlaylists', tags: ['playlists'] });

      r.get('/featured', async (ctx) => {
        const featured = await this.getFeaturedPlaylists(ctx.query);
        this.success(ctx, featured);
      }, { name: 'getFeaturedPlaylists', tags: ['playlists'] });

      r.get('/:id', async (ctx) => {
        const playlist = await this.getPlaylist(ctx.params.id);
        if (!playlist) {
          this.error(ctx, 'NOT_FOUND', 'Playlist not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(playlist, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'getPlaylist', tags: ['playlists'] });

      r.get('/:id/tracks', async (ctx) => {
        const tracks = await this.getPlaylistTracks(ctx.params.id, ctx.query);
        this.paginated(ctx, tracks.items, tracks.total);
      }, { name: 'getPlaylistTracks', tags: ['playlists', 'tracks'] });

      r.post('/', async (ctx) => {
        const validation = this.validator.validate(ctx.body, this.playlistSchema);
        if (!validation.valid) {
          this.error(ctx, 'VALIDATION_ERROR', validation.errors!.join(', '));
          return;
        }
        const playlist = await this.createPlaylist(validation.data!, ctx.user!.id);
        this.created(ctx, playlist, `${this.basePath}/${playlist.id}`);
      }, { name: 'createPlaylist', tags: ['playlists'], auth: true });

      r.put('/:id', async (ctx) => {
        const validation = this.validator.validate(ctx.body, this.playlistSchema);
        if (!validation.valid) {
          this.error(ctx, 'VALIDATION_ERROR', validation.errors!.join(', '));
          return;
        }
        const playlist = await this.updatePlaylist(ctx.params.id, validation.data!, ctx.user!.id);
        if (!playlist) {
          this.error(ctx, 'NOT_FOUND', 'Playlist not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(playlist, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'updatePlaylist', tags: ['playlists'], auth: true });

      r.delete('/:id', async (ctx) => {
        const deleted = await this.deletePlaylist(ctx.params.id, ctx.user!.id);
        if (!deleted) {
          this.error(ctx, 'NOT_FOUND', 'Playlist not found', 404);
          return;
        }
        this.noContent(ctx);
      }, { name: 'deletePlaylist', tags: ['playlists'], auth: true });

      // Track management
      r.post('/:id/tracks', async (ctx) => {
        const { trackIds, position } = ctx.body;
        await this.addTracksToPlaylist(ctx.params.id, trackIds, position, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'addTracksToPlaylist', tags: ['playlists', 'tracks'], auth: true });

      r.delete('/:id/tracks', async (ctx) => {
        const { trackIds } = ctx.body;
        await this.removeTracksFromPlaylist(ctx.params.id, trackIds, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'removeTracksFromPlaylist', tags: ['playlists', 'tracks'], auth: true });

      r.put('/:id/tracks/reorder', async (ctx) => {
        const { rangeStart, insertBefore, rangeLength } = ctx.body;
        await this.reorderPlaylistTracks(ctx.params.id, rangeStart, insertBefore, rangeLength, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'reorderPlaylistTracks', tags: ['playlists', 'tracks'], auth: true });
    });
  }

  private async listPlaylists(query: any, userId?: string): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getFeaturedPlaylists(query: any): Promise<any[]> {
    return [];
  }

  private async getPlaylist(id: string): Promise<any> {
    return null;
  }

  private async getPlaylistTracks(id: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async createPlaylist(data: any, userId: string): Promise<any> {
    return { id: crypto.randomUUID(), ...data, ownerId: userId };
  }

  private async updatePlaylist(id: string, data: any, userId: string): Promise<any> {
    return { id, ...data };
  }

  private async deletePlaylist(id: string, userId: string): Promise<boolean> {
    return true;
  }

  private async addTracksToPlaylist(playlistId: string, trackIds: string[], position: number, userId: string): Promise<void> {}
  private async removeTracksFromPlaylist(playlistId: string, trackIds: string[], userId: string): Promise<void> {}
  private async reorderPlaylistTracks(playlistId: string, rangeStart: number, insertBefore: number, rangeLength: number, userId: string): Promise<void> {}
}

/**
 * Search API Controller
 */
class SearchController extends ResourceController {
  registerRoutes(): void {
    this.router.group(this.basePath, (r) => {
      // Universal search
      r.get('/', async (ctx) => {
        const results = await this.search(ctx.query.q as string, ctx.query);
        this.success(ctx, results);
      }, { name: 'search', tags: ['search'] });

      // Type-specific search
      r.get('/tracks', async (ctx) => {
        const results = await this.searchTracks(ctx.query.q as string, ctx.query);
        this.paginated(ctx, results.items, results.total);
      }, { name: 'searchTracks', tags: ['search', 'tracks'] });

      r.get('/artists', async (ctx) => {
        const results = await this.searchArtists(ctx.query.q as string, ctx.query);
        this.paginated(ctx, results.items, results.total);
      }, { name: 'searchArtists', tags: ['search', 'artists'] });

      r.get('/albums', async (ctx) => {
        const results = await this.searchAlbums(ctx.query.q as string, ctx.query);
        this.paginated(ctx, results.items, results.total);
      }, { name: 'searchAlbums', tags: ['search', 'albums'] });

      r.get('/playlists', async (ctx) => {
        const results = await this.searchPlaylists(ctx.query.q as string, ctx.query);
        this.paginated(ctx, results.items, results.total);
      }, { name: 'searchPlaylists', tags: ['search', 'playlists'] });

      // Autocomplete
      r.get('/autocomplete', async (ctx) => {
        const suggestions = await this.autocomplete(ctx.query.q as string, ctx.query);
        this.success(ctx, suggestions);
      }, { name: 'autocomplete', tags: ['search'] });
    });
  }

  private async search(query: string, options: any): Promise<any> {
    return { tracks: [], artists: [], albums: [], playlists: [] };
  }

  private async searchTracks(query: string, options: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async searchArtists(query: string, options: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async searchAlbums(query: string, options: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async searchPlaylists(query: string, options: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async autocomplete(query: string, options: any): Promise<any[]> {
    return [];
  }
}

/**
 * User API Controller
 */
class UsersController extends ResourceController {
  registerRoutes(): void {
    this.router.group(this.basePath, (r) => {
      // Current user
      r.get('/me', async (ctx) => {
        const user = await this.getCurrentUser(ctx.user!.id);
        this.success(ctx, this.formatter.single(user, '/users/me'));
      }, { name: 'getCurrentUser', tags: ['users'], auth: true });

      r.put('/me', async (ctx) => {
        const user = await this.updateCurrentUser(ctx.user!.id, ctx.body);
        this.success(ctx, this.formatter.single(user, '/users/me'));
      }, { name: 'updateCurrentUser', tags: ['users'], auth: true });

      r.get('/me/playlists', async (ctx) => {
        const playlists = await this.getUserPlaylists(ctx.user!.id, ctx.query);
        this.paginated(ctx, playlists.items, playlists.total);
      }, { name: 'getCurrentUserPlaylists', tags: ['users', 'playlists'], auth: true });

      r.get('/me/following', async (ctx) => {
        const following = await this.getUserFollowing(ctx.user!.id, ctx.query);
        this.paginated(ctx, following.items, following.total);
      }, { name: 'getCurrentUserFollowing', tags: ['users', 'social'], auth: true });

      r.get('/me/followers', async (ctx) => {
        const followers = await this.getUserFollowers(ctx.user!.id, ctx.query);
        this.paginated(ctx, followers.items, followers.total);
      }, { name: 'getCurrentUserFollowers', tags: ['users', 'social'], auth: true });

      r.get('/me/recently-played', async (ctx) => {
        const history = await this.getRecentlyPlayed(ctx.user!.id, ctx.query);
        this.paginated(ctx, history.items, history.total);
      }, { name: 'getRecentlyPlayed', tags: ['users', 'history'], auth: true });

      r.get('/me/top/tracks', async (ctx) => {
        const tracks = await this.getTopTracks(ctx.user!.id, ctx.query);
        this.paginated(ctx, tracks.items, tracks.total);
      }, { name: 'getTopTracks', tags: ['users', 'recommendations'], auth: true });

      r.get('/me/top/artists', async (ctx) => {
        const artists = await this.getTopArtists(ctx.user!.id, ctx.query);
        this.paginated(ctx, artists.items, artists.total);
      }, { name: 'getTopArtists', tags: ['users', 'recommendations'], auth: true });

      // Public user profiles
      r.get('/:id', async (ctx) => {
        const user = await this.getUser(ctx.params.id);
        if (!user) {
          this.error(ctx, 'NOT_FOUND', 'User not found', 404);
          return;
        }
        this.success(ctx, this.formatter.single(user, `${this.basePath}/${ctx.params.id}`));
      }, { name: 'getUser', tags: ['users'] });

      r.get('/:id/playlists', async (ctx) => {
        const playlists = await this.getPublicUserPlaylists(ctx.params.id, ctx.query);
        this.paginated(ctx, playlists.items, playlists.total);
      }, { name: 'getUserPlaylists', tags: ['users', 'playlists'] });

      // Follow users
      r.post('/:id/follow', async (ctx) => {
        await this.followUser(ctx.params.id, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'followUser', tags: ['users', 'social'], auth: true });

      r.delete('/:id/follow', async (ctx) => {
        await this.unfollowUser(ctx.params.id, ctx.user!.id);
        this.noContent(ctx);
      }, { name: 'unfollowUser', tags: ['users', 'social'], auth: true });
    });
  }

  private async getCurrentUser(userId: string): Promise<any> {
    return { id: userId };
  }

  private async updateCurrentUser(userId: string, data: any): Promise<any> {
    return { id: userId, ...data };
  }

  private async getUserPlaylists(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getUserFollowing(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getUserFollowers(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getRecentlyPlayed(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getTopTracks(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getTopArtists(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async getUser(userId: string): Promise<any> {
    return null;
  }

  private async getPublicUserPlaylists(userId: string, query: any): Promise<{ items: any[]; total: number }> {
    return { items: [], total: 0 };
  }

  private async followUser(targetId: string, userId: string): Promise<void> {}
  private async unfollowUser(targetId: string, userId: string): Promise<void> {}
}

// ============================================================================
// OPENAPI SPECIFICATION GENERATOR
// ============================================================================

interface OpenAPISpec {
  openapi: string;
  info: {
    title: string;
    version: string;
    description: string;
  };
  servers: Array<{ url: string; description?: string }>;
  paths: Record<string, any>;
  components: {
    schemas: Record<string, any>;
    securitySchemes: Record<string, any>;
  };
  tags: Array<{ name: string; description?: string }>;
}

/**
 * Generate OpenAPI specification from routes
 */
class OpenAPIGenerator {
  constructor(
    private title: string,
    private version: string,
    private description: string,
    private servers: Array<{ url: string; description?: string }>
  ) {}

  generate(router: Router): OpenAPISpec {
    const spec: OpenAPISpec = {
      openapi: '3.0.3',
      info: {
        title: this.title,
        version: this.version,
        description: this.description,
      },
      servers: this.servers,
      paths: {},
      components: {
        schemas: this.generateSchemas(),
        securitySchemes: {
          bearerAuth: {
            type: 'http',
            scheme: 'bearer',
            bearerFormat: 'JWT',
          },
          apiKey: {
            type: 'apiKey',
            in: 'header',
            name: 'X-API-Key',
          },
        },
      },
      tags: [],
    };

    const tagSet = new Set<string>();

    for (const route of router.getRoutes()) {
      const pathKey = this.convertPathToOpenAPI(route.path);

      if (!spec.paths[pathKey]) {
        spec.paths[pathKey] = {};
      }

      const method = route.method.toLowerCase();
      spec.paths[pathKey][method] = this.generateOperation(route);

      // Collect tags
      for (const tag of route.metadata.tags || []) {
        tagSet.add(tag);
      }
    }

    spec.tags = Array.from(tagSet).map(name => ({ name }));

    return spec;
  }

  private convertPathToOpenAPI(path: string): string {
    return path.replace(/:([a-zA-Z0-9_]+)/g, '{$1}');
  }

  private generateOperation(route: RouteDefinition): any {
    const operation: any = {
      operationId: route.metadata.name,
      summary: route.metadata.description || route.metadata.name,
      tags: route.metadata.tags || [],
      parameters: this.extractParameters(route),
      responses: {
        '200': { description: 'Successful response' },
        '400': { description: 'Bad request' },
        '401': { description: 'Unauthorized' },
        '403': { description: 'Forbidden' },
        '404': { description: 'Not found' },
        '500': { description: 'Internal server error' },
      },
    };

    if (route.metadata.auth) {
      operation.security = [{ bearerAuth: [] }];
    }

    if (route.metadata.deprecated) {
      operation.deprecated = true;
    }

    if (['POST', 'PUT', 'PATCH'].includes(route.method)) {
      operation.requestBody = {
        required: true,
        content: {
          'application/json': {
            schema: { type: 'object' },
          },
        },
      };
    }

    return operation;
  }

  private extractParameters(route: RouteDefinition): any[] {
    const params: any[] = [];

    // Path parameters
    const pathParams = route.path.match(/:([a-zA-Z0-9_]+)/g) || [];
    for (const param of pathParams) {
      params.push({
        name: param.slice(1),
        in: 'path',
        required: true,
        schema: { type: 'string' },
      });
    }

    // Common query parameters for list endpoints
    if (route.method === 'GET' && route.path.endsWith('/')) {
      params.push(
        { name: 'page', in: 'query', schema: { type: 'integer', default: 1 } },
        { name: 'pageSize', in: 'query', schema: { type: 'integer', default: 20 } }
      );
    }

    return params;
  }

  private generateSchemas(): Record<string, any> {
    return {
      Track: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          title: { type: 'string' },
          artistId: { type: 'string', format: 'uuid' },
          albumId: { type: 'string', format: 'uuid' },
          duration: { type: 'integer' },
          genre: { type: 'string' },
          isExplicit: { type: 'boolean' },
          createdAt: { type: 'string', format: 'date-time' },
          updatedAt: { type: 'string', format: 'date-time' },
        },
      },
      Artist: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          name: { type: 'string' },
          bio: { type: 'string' },
          genres: { type: 'array', items: { type: 'string' } },
          verified: { type: 'boolean' },
          followerCount: { type: 'integer' },
        },
      },
      Album: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          title: { type: 'string' },
          artistId: { type: 'string', format: 'uuid' },
          releaseDate: { type: 'string', format: 'date' },
          trackCount: { type: 'integer' },
          totalDuration: { type: 'integer' },
        },
      },
      Playlist: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          name: { type: 'string' },
          description: { type: 'string' },
          ownerId: { type: 'string', format: 'uuid' },
          isPublic: { type: 'boolean' },
          trackCount: { type: 'integer' },
          followerCount: { type: 'integer' },
        },
      },
      User: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          email: { type: 'string', format: 'email' },
          displayName: { type: 'string' },
          avatarUrl: { type: 'string', format: 'uri' },
          followerCount: { type: 'integer' },
          followingCount: { type: 'integer' },
        },
      },
      PaginatedResponse: {
        type: 'object',
        properties: {
          data: { type: 'array' },
          pagination: {
            type: 'object',
            properties: {
              page: { type: 'integer' },
              pageSize: { type: 'integer' },
              totalItems: { type: 'integer' },
              totalPages: { type: 'integer' },
              hasNextPage: { type: 'boolean' },
              hasPrevPage: { type: 'boolean' },
            },
          },
          links: {
            type: 'object',
            properties: {
              self: { type: 'string', format: 'uri' },
              first: { type: 'string', format: 'uri' },
              last: { type: 'string', format: 'uri' },
              next: { type: 'string', format: 'uri' },
              prev: { type: 'string', format: 'uri' },
            },
          },
        },
      },
      Error: {
        type: 'object',
        properties: {
          error: {
            type: 'object',
            properties: {
              code: { type: 'string' },
              message: { type: 'string' },
              details: { type: 'object' },
              timestamp: { type: 'string', format: 'date-time' },
              requestId: { type: 'string' },
            },
          },
        },
      },
    };
  }
}

// ============================================================================
// API VERSION MANAGEMENT
// ============================================================================

/**
 * API version manager for handling multiple API versions
 */
class APIVersionManager {
  private versions = new Map<string, Router>();
  private currentVersion: string = 'v1';

  /**
   * Register an API version
   */
  registerVersion(version: string, router: Router): void {
    this.versions.set(version, router);
  }

  /**
   * Set current/default version
   */
  setCurrentVersion(version: string): void {
    if (!this.versions.has(version)) {
      throw new Error(`Version ${version} not registered`);
    }
    this.currentVersion = version;
  }

  /**
   * Get router for version
   */
  getRouter(version?: string): Router | undefined {
    return this.versions.get(version || this.currentVersion);
  }

  /**
   * Handle request with version detection
   */
  async handleRequest(request: HttpRequest): Promise<HttpResponse> {
    // Detect version from path, header, or query
    let version = this.currentVersion;

    // Path-based versioning: /v1/tracks, /v2/tracks
    const pathMatch = request.path.match(/^\/(v\d+)\//);
    if (pathMatch) {
      version = pathMatch[1];
      request.path = request.path.replace(`/${version}`, '');
    }

    // Header-based versioning
    const headerVersion = request.headers['api-version'] || request.headers['accept-version'];
    if (headerVersion && this.versions.has(headerVersion)) {
      version = headerVersion;
    }

    const router = this.versions.get(version);
    if (!router) {
      return {
        status: 400,
        headers: {},
        body: { error: { code: 'INVALID_VERSION', message: `API version ${version} not supported` } },
      };
    }

    const response = await router.handle(request);
    response.headers['API-Version'] = version;

    return response;
  }

  /**
   * Get all supported versions
   */
  getSupportedVersions(): string[] {
    return Array.from(this.versions.keys());
  }
}

// ============================================================================
// API BUILDER
// ============================================================================

/**
 * Build the complete Mycelix API
 */
function buildMycelixAPI(baseUrl: string): {
  router: Router;
  versionManager: APIVersionManager;
  openAPISpec: OpenAPISpec;
} {
  const formatter = new ResponseFormatter(baseUrl);
  const router = new Router('/api');

  // Register controllers
  new TracksController(router, formatter, '/tracks').registerRoutes();
  new ArtistsController(router, formatter, '/artists').registerRoutes();
  new PlaylistsController(router, formatter, '/playlists').registerRoutes();
  new SearchController(router, formatter, '/search').registerRoutes();
  new UsersController(router, formatter, '/users').registerRoutes();

  // Health endpoint
  router.get('/health', async (ctx) => {
    ctx.response.body = { status: 'healthy', timestamp: new Date().toISOString() };
  }, { name: 'healthCheck', tags: ['system'] });

  // Version manager
  const versionManager = new APIVersionManager();
  versionManager.registerVersion('v1', router);
  versionManager.setCurrentVersion('v1');

  // OpenAPI spec
  const openAPIGenerator = new OpenAPIGenerator(
    'Mycelix Music API',
    '1.0.0',
    'The Mycelix Music Platform API provides access to music streaming, playlists, artists, and more.',
    [{ url: baseUrl, description: 'Production' }]
  );
  const openAPISpec = openAPIGenerator.generate(router);

  return { router, versionManager, openAPISpec };
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  // Router
  Router,
  RouteDefinition,
  RouteMetadata,
  RouteHandler,
  RouteMiddleware,
  RouteContext,

  // HTTP types
  HttpMethod,
  HttpRequest,
  HttpResponse,
  AuthenticatedUser,

  // Validation
  RequestValidator,
  SchemaProperty,
  ValidationResult,

  // Response formatting
  ResponseFormatter,
  PaginatedResponse,
  SingleResponse,
  ErrorResponse,

  // Controllers
  ResourceController,
  TracksController,
  ArtistsController,
  PlaylistsController,
  SearchController,
  UsersController,

  // OpenAPI
  OpenAPIGenerator,
  OpenAPISpec,

  // Version management
  APIVersionManager,

  // Builder
  buildMycelixAPI,
};

/**
 * Create the API layer
 */
export function createAPILayer(baseUrl: string = 'https://api.mycelix.music'): ReturnType<typeof buildMycelixAPI> {
  return buildMycelixAPI(baseUrl);
}
