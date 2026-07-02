// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Security Layer
 *
 * Client-side security utilities:
 * - CSRF token management
 * - Request signing
 * - Input validation and sanitization
 * - Rate limiting
 * - Content Security Policy
 * - Secure storage
 */

// ==================== Types ====================

export interface SecurityConfig {
  csrfTokenHeader: string;
  csrfCookieName: string;
  rateLimitWindow: number;
  rateLimitMax: number;
  signatureHeader: string;
  encryptionKey?: string;
}

export interface ValidationRule<T = any> {
  type: 'string' | 'number' | 'boolean' | 'array' | 'object' | 'email' | 'url' | 'uuid';
  required?: boolean;
  min?: number;
  max?: number;
  pattern?: RegExp;
  enum?: T[];
  custom?: (value: T) => boolean | string;
}

export interface ValidationSchema {
  [key: string]: ValidationRule;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
  value?: any;
}

export interface RateLimitState {
  count: number;
  resetAt: number;
  remaining: number;
  blocked: boolean;
}

// ==================== CSRF Protection ====================

export class CSRFManager {
  private token: string | null = null;
  private config: Pick<SecurityConfig, 'csrfTokenHeader' | 'csrfCookieName'>;

  constructor(config: Partial<SecurityConfig> = {}) {
    this.config = {
      csrfTokenHeader: config.csrfTokenHeader || 'X-CSRF-Token',
      csrfCookieName: config.csrfCookieName || 'csrf_token',
    };

    this.loadToken();
  }

  private loadToken(): void {
    if (typeof document === 'undefined') return;

    // Try to get from meta tag
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) {
      this.token = meta.getAttribute('content');
      return;
    }

    // Try to get from cookie
    const cookies = document.cookie.split(';');
    for (const cookie of cookies) {
      const [name, value] = cookie.trim().split('=');
      if (name === this.config.csrfCookieName) {
        this.token = decodeURIComponent(value);
        return;
      }
    }
  }

  getToken(): string | null {
    return this.token;
  }

  setToken(token: string): void {
    this.token = token;
  }

  getHeaders(): Record<string, string> {
    if (!this.token) return {};
    return { [this.config.csrfTokenHeader]: this.token };
  }

  async refreshToken(): Promise<string> {
    const response = await fetch('/api/csrf-token');
    const { token } = await response.json();
    this.token = token;
    return token;
  }

  /**
   * Wrap fetch with CSRF protection
   */
  wrapFetch(originalFetch: typeof fetch): typeof fetch {
    return async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method?.toUpperCase() || 'GET';

      // Only add CSRF token for mutating requests
      if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
        const headers = new Headers(init?.headers);
        if (this.token) {
          headers.set(this.config.csrfTokenHeader, this.token);
        }

        return originalFetch(input, {
          ...init,
          headers,
        });
      }

      return originalFetch(input, init);
    };
  }
}

// ==================== Request Signing ====================

export class RequestSigner {
  private secretKey: string;
  private algorithm: AlgorithmIdentifier = 'SHA-256';

  constructor(secretKey: string) {
    this.secretKey = secretKey;
  }

  async sign(payload: string | object, timestamp?: number): Promise<string> {
    const ts = timestamp || Date.now();
    const data = typeof payload === 'string' ? payload : JSON.stringify(payload);
    const message = `${ts}.${data}`;

    const encoder = new TextEncoder();
    const keyData = encoder.encode(this.secretKey);
    const messageData = encoder.encode(message);

    const key = await crypto.subtle.importKey(
      'raw',
      keyData,
      { name: 'HMAC', hash: this.algorithm },
      false,
      ['sign']
    );

    const signature = await crypto.subtle.sign('HMAC', key, messageData);
    const signatureArray = Array.from(new Uint8Array(signature));
    const signatureHex = signatureArray.map(b => b.toString(16).padStart(2, '0')).join('');

    return `t=${ts},v1=${signatureHex}`;
  }

  async verify(
    payload: string | object,
    signature: string,
    maxAge: number = 300000 // 5 minutes
  ): Promise<boolean> {
    const parts = signature.split(',');
    const timestampPart = parts.find(p => p.startsWith('t='));
    const signaturePart = parts.find(p => p.startsWith('v1='));

    if (!timestampPart || !signaturePart) return false;

    const timestamp = parseInt(timestampPart.slice(2));
    const providedSignature = signaturePart.slice(3);

    // Check timestamp freshness
    if (Date.now() - timestamp > maxAge) return false;

    // Verify signature
    const expectedSignature = await this.sign(payload, timestamp);
    const expectedPart = expectedSignature.split(',').find(p => p.startsWith('v1='));
    if (!expectedPart) return false;

    return providedSignature === expectedPart.slice(3);
  }
}

// ==================== Input Validation ====================

export class Validator {
  private static emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  private static urlRegex = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;
  private static uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

  static validate(data: Record<string, any>, schema: ValidationSchema): ValidationResult {
    const errors: ValidationError[] = [];

    for (const [field, rules] of Object.entries(schema)) {
      const value = data[field];
      const fieldErrors = this.validateField(field, value, rules);
      errors.push(...fieldErrors);
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  private static validateField(
    field: string,
    value: any,
    rules: ValidationRule
  ): ValidationError[] {
    const errors: ValidationError[] = [];

    // Required check
    if (rules.required && (value === undefined || value === null || value === '')) {
      errors.push({ field, message: `${field} is required`, value });
      return errors;
    }

    // Skip further validation if not required and empty
    if (value === undefined || value === null) {
      return errors;
    }

    // Type checks
    switch (rules.type) {
      case 'string':
        if (typeof value !== 'string') {
          errors.push({ field, message: `${field} must be a string`, value });
        } else {
          if (rules.min !== undefined && value.length < rules.min) {
            errors.push({ field, message: `${field} must be at least ${rules.min} characters`, value });
          }
          if (rules.max !== undefined && value.length > rules.max) {
            errors.push({ field, message: `${field} must be at most ${rules.max} characters`, value });
          }
          if (rules.pattern && !rules.pattern.test(value)) {
            errors.push({ field, message: `${field} has invalid format`, value });
          }
        }
        break;

      case 'number':
        if (typeof value !== 'number' || isNaN(value)) {
          errors.push({ field, message: `${field} must be a number`, value });
        } else {
          if (rules.min !== undefined && value < rules.min) {
            errors.push({ field, message: `${field} must be at least ${rules.min}`, value });
          }
          if (rules.max !== undefined && value > rules.max) {
            errors.push({ field, message: `${field} must be at most ${rules.max}`, value });
          }
        }
        break;

      case 'boolean':
        if (typeof value !== 'boolean') {
          errors.push({ field, message: `${field} must be a boolean`, value });
        }
        break;

      case 'array':
        if (!Array.isArray(value)) {
          errors.push({ field, message: `${field} must be an array`, value });
        } else {
          if (rules.min !== undefined && value.length < rules.min) {
            errors.push({ field, message: `${field} must have at least ${rules.min} items`, value });
          }
          if (rules.max !== undefined && value.length > rules.max) {
            errors.push({ field, message: `${field} must have at most ${rules.max} items`, value });
          }
        }
        break;

      case 'object':
        if (typeof value !== 'object' || value === null || Array.isArray(value)) {
          errors.push({ field, message: `${field} must be an object`, value });
        }
        break;

      case 'email':
        if (typeof value !== 'string' || !this.emailRegex.test(value)) {
          errors.push({ field, message: `${field} must be a valid email`, value });
        }
        break;

      case 'url':
        if (typeof value !== 'string' || !this.urlRegex.test(value)) {
          errors.push({ field, message: `${field} must be a valid URL`, value });
        }
        break;

      case 'uuid':
        if (typeof value !== 'string' || !this.uuidRegex.test(value)) {
          errors.push({ field, message: `${field} must be a valid UUID`, value });
        }
        break;
    }

    // Enum check
    if (rules.enum && !rules.enum.includes(value)) {
      errors.push({
        field,
        message: `${field} must be one of: ${rules.enum.join(', ')}`,
        value,
      });
    }

    // Custom validation
    if (rules.custom) {
      const result = rules.custom(value);
      if (result !== true) {
        errors.push({
          field,
          message: typeof result === 'string' ? result : `${field} is invalid`,
          value,
        });
      }
    }

    return errors;
  }
}

// ==================== Input Sanitization ====================

export class Sanitizer {
  /**
   * Sanitize HTML to prevent XSS
   */
  static sanitizeHTML(html: string): string {
    const div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
  }

  /**
   * Escape HTML entities
   */
  static escapeHTML(str: string): string {
    const escapeMap: Record<string, string> = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#x27;',
      '/': '&#x2F;',
    };

    return str.replace(/[&<>"'/]/g, char => escapeMap[char]);
  }

  /**
   * Remove all HTML tags
   */
  static stripHTML(html: string): string {
    return html.replace(/<[^>]*>/g, '');
  }

  /**
   * Sanitize for SQL (basic - use parameterized queries!)
   */
  static sanitizeSQL(str: string): string {
    return str.replace(/['";\\]/g, '');
  }

  /**
   * Sanitize filename
   */
  static sanitizeFilename(filename: string): string {
    return filename
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .replace(/\.{2,}/g, '.')
      .slice(0, 255);
  }

  /**
   * Sanitize URL
   */
  static sanitizeURL(url: string): string {
    try {
      const parsed = new URL(url);
      // Only allow http and https
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        return '';
      }
      return parsed.href;
    } catch {
      return '';
    }
  }

  /**
   * Remove null bytes
   */
  static removeNullBytes(str: string): string {
    return str.replace(/\0/g, '');
  }

  /**
   * Normalize unicode
   */
  static normalizeUnicode(str: string): string {
    return str.normalize('NFC');
  }

  /**
   * Full sanitization pipeline
   */
  static sanitize(str: string): string {
    return this.normalizeUnicode(
      this.removeNullBytes(
        this.escapeHTML(str.trim())
      )
    );
  }
}

// ==================== Rate Limiter ====================

export class RateLimiter {
  private limits: Map<string, { count: number; resetAt: number }> = new Map();
  private config: Pick<SecurityConfig, 'rateLimitWindow' | 'rateLimitMax'>;

  constructor(config: Partial<SecurityConfig> = {}) {
    this.config = {
      rateLimitWindow: config.rateLimitWindow || 60000, // 1 minute
      rateLimitMax: config.rateLimitMax || 100,
    };
  }

  check(key: string): RateLimitState {
    const now = Date.now();
    let limit = this.limits.get(key);

    // Reset if window expired
    if (!limit || limit.resetAt <= now) {
      limit = {
        count: 0,
        resetAt: now + this.config.rateLimitWindow,
      };
      this.limits.set(key, limit);
    }

    const remaining = Math.max(0, this.config.rateLimitMax - limit.count);
    const blocked = limit.count >= this.config.rateLimitMax;

    return {
      count: limit.count,
      resetAt: limit.resetAt,
      remaining,
      blocked,
    };
  }

  consume(key: string, amount: number = 1): RateLimitState {
    const state = this.check(key);

    if (!state.blocked) {
      const limit = this.limits.get(key)!;
      limit.count += amount;
    }

    return this.check(key);
  }

  reset(key: string): void {
    this.limits.delete(key);
  }

  /**
   * Create a rate-limited function
   */
  wrap<T extends (...args: any[]) => Promise<any>>(
    key: string,
    fn: T
  ): (...args: Parameters<T>) => Promise<ReturnType<T>> {
    return async (...args: Parameters<T>): Promise<ReturnType<T>> => {
      const state = this.consume(key);
      if (state.blocked) {
        throw new RateLimitError(
          `Rate limit exceeded. Try again in ${Math.ceil((state.resetAt - Date.now()) / 1000)}s`
        );
      }
      return fn(...args);
    };
  }
}

export class RateLimitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RateLimitError';
  }
}

// ==================== Secure Storage ====================

export class SecureStorage {
  private prefix: string;
  private encryptionKey?: CryptoKey;

  constructor(prefix: string = 'mycelix') {
    this.prefix = prefix;
  }

  async initializeEncryption(password: string): Promise<void> {
    const encoder = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      encoder.encode(password),
      'PBKDF2',
      false,
      ['deriveBits', 'deriveKey']
    );

    this.encryptionKey = await crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: encoder.encode(this.prefix),
        iterations: 100000,
        hash: 'SHA-256',
      },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
  }

  async setItem(key: string, value: any): Promise<void> {
    const fullKey = `${this.prefix}:${key}`;
    const data = JSON.stringify(value);

    if (this.encryptionKey) {
      const encrypted = await this.encrypt(data);
      localStorage.setItem(fullKey, encrypted);
    } else {
      localStorage.setItem(fullKey, data);
    }
  }

  async getItem<T>(key: string): Promise<T | null> {
    const fullKey = `${this.prefix}:${key}`;
    const data = localStorage.getItem(fullKey);

    if (!data) return null;

    try {
      if (this.encryptionKey) {
        const decrypted = await this.decrypt(data);
        return JSON.parse(decrypted);
      }
      return JSON.parse(data);
    } catch {
      return null;
    }
  }

  removeItem(key: string): void {
    localStorage.removeItem(`${this.prefix}:${key}`);
  }

  clear(): void {
    const keysToRemove: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(this.prefix)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
  }

  private async encrypt(data: string): Promise<string> {
    if (!this.encryptionKey) throw new Error('Encryption not initialized');

    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoder = new TextEncoder();

    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      encoder.encode(data)
    );

    const combined = new Uint8Array(iv.length + encrypted.byteLength);
    combined.set(iv);
    combined.set(new Uint8Array(encrypted), iv.length);

    return btoa(String.fromCharCode(...combined));
  }

  private async decrypt(data: string): Promise<string> {
    if (!this.encryptionKey) throw new Error('Encryption not initialized');

    const combined = new Uint8Array(
      atob(data).split('').map(c => c.charCodeAt(0))
    );

    const iv = combined.slice(0, 12);
    const encrypted = combined.slice(12);

    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      encrypted
    );

    return new TextDecoder().decode(decrypted);
  }
}

// ==================== Content Security Policy ====================

export class CSPBuilder {
  private directives: Map<string, Set<string>> = new Map();

  addDirective(directive: string, ...sources: string[]): this {
    if (!this.directives.has(directive)) {
      this.directives.set(directive, new Set());
    }
    sources.forEach(s => this.directives.get(directive)!.add(s));
    return this;
  }

  defaultSrc(...sources: string[]): this {
    return this.addDirective('default-src', ...sources);
  }

  scriptSrc(...sources: string[]): this {
    return this.addDirective('script-src', ...sources);
  }

  styleSrc(...sources: string[]): this {
    return this.addDirective('style-src', ...sources);
  }

  imgSrc(...sources: string[]): this {
    return this.addDirective('img-src', ...sources);
  }

  fontSrc(...sources: string[]): this {
    return this.addDirective('font-src', ...sources);
  }

  connectSrc(...sources: string[]): this {
    return this.addDirective('connect-src', ...sources);
  }

  mediaSrc(...sources: string[]): this {
    return this.addDirective('media-src', ...sources);
  }

  frameSrc(...sources: string[]): this {
    return this.addDirective('frame-src', ...sources);
  }

  workerSrc(...sources: string[]): this {
    return this.addDirective('worker-src', ...sources);
  }

  build(): string {
    const parts: string[] = [];
    for (const [directive, sources] of this.directives) {
      parts.push(`${directive} ${Array.from(sources).join(' ')}`);
    }
    return parts.join('; ');
  }

  static recommended(): CSPBuilder {
    return new CSPBuilder()
      .defaultSrc("'self'")
      .scriptSrc("'self'", "'unsafe-inline'", "'unsafe-eval'")
      .styleSrc("'self'", "'unsafe-inline'")
      .imgSrc("'self'", 'data:', 'https:')
      .fontSrc("'self'", 'https:', 'data:')
      .connectSrc("'self'", 'https:', 'wss:')
      .mediaSrc("'self'", 'https:', 'blob:')
      .workerSrc("'self'", 'blob:');
  }
}

// ==================== Security Manager ====================

export class SecurityManager {
  public readonly csrf: CSRFManager;
  public readonly rateLimiter: RateLimiter;
  public readonly storage: SecureStorage;
  private signer?: RequestSigner;

  constructor(config: Partial<SecurityConfig> = {}) {
    this.csrf = new CSRFManager(config);
    this.rateLimiter = new RateLimiter(config);
    this.storage = new SecureStorage();

    if (config.encryptionKey) {
      this.signer = new RequestSigner(config.encryptionKey);
    }
  }

  async initializeEncryption(password: string): Promise<void> {
    await this.storage.initializeEncryption(password);
  }

  validate(data: Record<string, any>, schema: ValidationSchema): ValidationResult {
    return Validator.validate(data, schema);
  }

  sanitize(input: string): string {
    return Sanitizer.sanitize(input);
  }

  async signRequest(payload: any): Promise<string | null> {
    if (!this.signer) return null;
    return this.signer.sign(payload);
  }

  createSecureFetch(): typeof fetch {
    return this.csrf.wrapFetch(fetch);
  }
}

// ==================== Singleton ====================

let securityManager: SecurityManager | null = null;

export function getSecurityManager(config?: Partial<SecurityConfig>): SecurityManager {
  if (!securityManager) {
    securityManager = new SecurityManager(config);
  }
  return securityManager;
}

export default {
  SecurityManager,
  getSecurityManager,
  CSRFManager,
  RequestSigner,
  Validator,
  Sanitizer,
  RateLimiter,
  SecureStorage,
  CSPBuilder,
};
