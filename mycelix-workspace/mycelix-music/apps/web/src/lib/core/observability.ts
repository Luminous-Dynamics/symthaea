// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Observability System
 *
 * Comprehensive monitoring and tracking:
 * - Error tracking and reporting
 * - Performance monitoring (Core Web Vitals)
 * - User analytics
 * - Session recording
 * - Custom metrics
 * - Distributed tracing
 */

// ==================== Types ====================

export interface ErrorReport {
  id: string;
  message: string;
  stack?: string;
  type: ErrorType;
  severity: ErrorSeverity;
  context: ErrorContext;
  timestamp: number;
  fingerprint: string;
  handled: boolean;
}

export type ErrorType =
  | 'runtime'
  | 'network'
  | 'promise'
  | 'resource'
  | 'audio'
  | 'webgl'
  | 'webrtc'
  | 'wasm';

export type ErrorSeverity = 'debug' | 'info' | 'warning' | 'error' | 'fatal';

export interface ErrorContext {
  url: string;
  userAgent: string;
  userId?: string;
  sessionId: string;
  releaseVersion: string;
  tags: Record<string, string>;
  extra: Record<string, any>;
  breadcrumbs: Breadcrumb[];
}

export interface Breadcrumb {
  type: 'navigation' | 'http' | 'user' | 'console' | 'error';
  category: string;
  message: string;
  data?: Record<string, any>;
  timestamp: number;
}

export interface PerformanceMetric {
  name: string;
  value: number;
  unit: 'ms' | 'bytes' | 'count' | 'percent';
  timestamp: number;
  tags: Record<string, string>;
}

export interface WebVitals {
  lcp: number | null;  // Largest Contentful Paint
  fid: number | null;  // First Input Delay
  cls: number | null;  // Cumulative Layout Shift
  fcp: number | null;  // First Contentful Paint
  ttfb: number | null; // Time to First Byte
  inp: number | null;  // Interaction to Next Paint
}

export interface AnalyticsEvent {
  name: string;
  properties: Record<string, any>;
  timestamp: number;
  sessionId: string;
  userId?: string;
}

export interface Span {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  status: 'ok' | 'error';
  attributes: Record<string, any>;
}

export interface UserSession {
  id: string;
  userId?: string;
  startTime: number;
  lastActivity: number;
  pageViews: number;
  events: number;
  errors: number;
  device: DeviceInfo;
}

export interface DeviceInfo {
  type: 'desktop' | 'tablet' | 'mobile';
  os: string;
  browser: string;
  screenWidth: number;
  screenHeight: number;
  language: string;
  timezone: string;
}

// ==================== Error Tracker ====================

export class ErrorTracker {
  private errors: ErrorReport[] = [];
  private breadcrumbs: Breadcrumb[] = [];
  private maxBreadcrumbs = 100;
  private maxErrors = 100;
  private sessionId: string;
  private userId?: string;
  private releaseVersion: string;
  private tags: Record<string, string> = {};
  private beforeSend?: (error: ErrorReport) => ErrorReport | null;
  private onError?: (error: ErrorReport) => void;
  private endpoint?: string;

  constructor(config: {
    releaseVersion: string;
    endpoint?: string;
    beforeSend?: (error: ErrorReport) => ErrorReport | null;
    onError?: (error: ErrorReport) => void;
  }) {
    this.sessionId = crypto.randomUUID();
    this.releaseVersion = config.releaseVersion;
    this.endpoint = config.endpoint;
    this.beforeSend = config.beforeSend;
    this.onError = config.onError;

    this.setupGlobalHandlers();
  }

  private setupGlobalHandlers(): void {
    if (typeof window === 'undefined') return;

    // Uncaught errors
    window.onerror = (message, source, lineno, colno, error) => {
      this.captureError(error || new Error(String(message)), {
        type: 'runtime',
        handled: false,
      });
    };

    // Unhandled promise rejections
    window.onunhandledrejection = (event) => {
      this.captureError(event.reason, {
        type: 'promise',
        handled: false,
      });
    };

    // Resource errors
    window.addEventListener('error', (event) => {
      if (event.target !== window) {
        this.captureMessage(`Resource failed to load: ${(event.target as any)?.src || (event.target as any)?.href}`, {
          type: 'resource',
          severity: 'warning',
        });
      }
    }, true);

    // Console errors
    const originalConsoleError = console.error;
    console.error = (...args) => {
      this.addBreadcrumb({
        type: 'console',
        category: 'console',
        message: args.map(a => String(a)).join(' '),
      });
      originalConsoleError.apply(console, args);
    };
  }

  captureError(error: Error | unknown, options: {
    type?: ErrorType;
    severity?: ErrorSeverity;
    handled?: boolean;
    tags?: Record<string, string>;
    extra?: Record<string, any>;
  } = {}): string {
    const err = error instanceof Error ? error : new Error(String(error));

    const report: ErrorReport = {
      id: crypto.randomUUID(),
      message: err.message,
      stack: err.stack,
      type: options.type || 'runtime',
      severity: options.severity || 'error',
      handled: options.handled ?? true,
      fingerprint: this.generateFingerprint(err),
      timestamp: Date.now(),
      context: {
        url: typeof window !== 'undefined' ? window.location.href : '',
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
        userId: this.userId,
        sessionId: this.sessionId,
        releaseVersion: this.releaseVersion,
        tags: { ...this.tags, ...options.tags },
        extra: options.extra || {},
        breadcrumbs: [...this.breadcrumbs],
      },
    };

    // Apply beforeSend hook
    const finalReport = this.beforeSend ? this.beforeSend(report) : report;
    if (!finalReport) return report.id;

    // Store locally
    this.errors.push(finalReport);
    if (this.errors.length > this.maxErrors) {
      this.errors.shift();
    }

    // Notify listeners
    this.onError?.(finalReport);

    // Send to endpoint
    this.sendError(finalReport);

    return report.id;
  }

  captureMessage(message: string, options: {
    type?: ErrorType;
    severity?: ErrorSeverity;
    tags?: Record<string, string>;
    extra?: Record<string, any>;
  } = {}): string {
    return this.captureError(new Error(message), {
      ...options,
      handled: true,
    });
  }

  addBreadcrumb(breadcrumb: Omit<Breadcrumb, 'timestamp'>): void {
    this.breadcrumbs.push({
      ...breadcrumb,
      timestamp: Date.now(),
    });

    if (this.breadcrumbs.length > this.maxBreadcrumbs) {
      this.breadcrumbs.shift();
    }
  }

  setUser(userId: string | undefined): void {
    this.userId = userId;
  }

  setTag(key: string, value: string): void {
    this.tags[key] = value;
  }

  setTags(tags: Record<string, string>): void {
    Object.assign(this.tags, tags);
  }

  getErrors(): ErrorReport[] {
    return [...this.errors];
  }

  clearErrors(): void {
    this.errors = [];
  }

  private generateFingerprint(error: Error): string {
    const parts = [
      error.name,
      error.message.split('\n')[0],
      error.stack?.split('\n')[1] || '',
    ];
    return this.hashString(parts.join('|'));
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  private async sendError(report: ErrorReport): Promise<void> {
    if (!this.endpoint) return;

    try {
      await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(report),
      });
    } catch {
      // Silently fail
    }
  }
}

// ==================== Performance Monitor ====================

export class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private webVitals: WebVitals = {
    lcp: null,
    fid: null,
    cls: null,
    fcp: null,
    ttfb: null,
    inp: null,
  };
  private spans: Map<string, Span> = new Map();
  private observers: PerformanceObserver[] = [];
  private onMetric?: (metric: PerformanceMetric) => void;
  private endpoint?: string;

  constructor(config: {
    endpoint?: string;
    onMetric?: (metric: PerformanceMetric) => void;
  } = {}) {
    this.endpoint = config.endpoint;
    this.onMetric = config.onMetric;

    this.setupObservers();
    this.measureWebVitals();
  }

  private setupObservers(): void {
    if (typeof PerformanceObserver === 'undefined') return;

    // Long tasks
    try {
      const longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.recordMetric('long_task', entry.duration, 'ms', {
            attribution: (entry as any).attribution?.[0]?.name || 'unknown',
          });
        }
      });
      longTaskObserver.observe({ entryTypes: ['longtask'] });
      this.observers.push(longTaskObserver);
    } catch {
      // Not supported
    }

    // Resources
    try {
      const resourceObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as PerformanceResourceTiming[]) {
          this.recordMetric('resource_load', entry.duration, 'ms', {
            type: entry.initiatorType,
            name: entry.name.split('/').pop() || entry.name,
          });
        }
      });
      resourceObserver.observe({ entryTypes: ['resource'] });
      this.observers.push(resourceObserver);
    } catch {
      // Not supported
    }

    // Navigation
    try {
      const navigationObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as PerformanceNavigationTiming[]) {
          this.recordMetric('dom_interactive', entry.domInteractive, 'ms');
          this.recordMetric('dom_complete', entry.domComplete, 'ms');
          this.recordMetric('load_complete', entry.loadEventEnd, 'ms');
        }
      });
      navigationObserver.observe({ entryTypes: ['navigation'] });
      this.observers.push(navigationObserver);
    } catch {
      // Not supported
    }
  }

  private measureWebVitals(): void {
    if (typeof PerformanceObserver === 'undefined') return;

    // LCP
    try {
      new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        this.webVitals.lcp = lastEntry.startTime;
        this.recordMetric('web_vital_lcp', lastEntry.startTime, 'ms');
      }).observe({ entryTypes: ['largest-contentful-paint'] });
    } catch {}

    // FID
    try {
      new PerformanceObserver((list) => {
        const entry = list.getEntries()[0] as PerformanceEventTiming;
        this.webVitals.fid = entry.processingStart - entry.startTime;
        this.recordMetric('web_vital_fid', this.webVitals.fid, 'ms');
      }).observe({ entryTypes: ['first-input'] });
    } catch {}

    // CLS
    try {
      let clsValue = 0;
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as any[]) {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        }
        this.webVitals.cls = clsValue;
        this.recordMetric('web_vital_cls', clsValue, 'count');
      }).observe({ entryTypes: ['layout-shift'] });
    } catch {}

    // FCP
    try {
      new PerformanceObserver((list) => {
        const entry = list.getEntries().find(e => e.name === 'first-contentful-paint');
        if (entry) {
          this.webVitals.fcp = entry.startTime;
          this.recordMetric('web_vital_fcp', entry.startTime, 'ms');
        }
      }).observe({ entryTypes: ['paint'] });
    } catch {}

    // TTFB
    if (typeof performance !== 'undefined' && performance.timing) {
      const ttfb = performance.timing.responseStart - performance.timing.requestStart;
      this.webVitals.ttfb = ttfb;
      this.recordMetric('web_vital_ttfb', ttfb, 'ms');
    }
  }

  recordMetric(
    name: string,
    value: number,
    unit: PerformanceMetric['unit'],
    tags: Record<string, string> = {}
  ): void {
    const metric: PerformanceMetric = {
      name,
      value,
      unit,
      timestamp: Date.now(),
      tags,
    };

    this.metrics.push(metric);
    this.onMetric?.(metric);

    // Send to endpoint periodically
    if (this.metrics.length >= 50) {
      this.flush();
    }
  }

  startSpan(name: string, attributes: Record<string, any> = {}): Span {
    const span: Span = {
      traceId: crypto.randomUUID(),
      spanId: crypto.randomUUID(),
      name,
      startTime: performance.now(),
      status: 'ok',
      attributes,
    };

    this.spans.set(span.spanId, span);
    return span;
  }

  endSpan(spanId: string, status: 'ok' | 'error' = 'ok'): void {
    const span = this.spans.get(spanId);
    if (span) {
      span.endTime = performance.now();
      span.duration = span.endTime - span.startTime;
      span.status = status;

      this.recordMetric(`span_${span.name}`, span.duration, 'ms', {
        status,
        ...Object.fromEntries(
          Object.entries(span.attributes).map(([k, v]) => [k, String(v)])
        ),
      });
    }
  }

  measure<T>(name: string, fn: () => T): T {
    const span = this.startSpan(name);
    try {
      const result = fn();
      this.endSpan(span.spanId, 'ok');
      return result;
    } catch (error) {
      this.endSpan(span.spanId, 'error');
      throw error;
    }
  }

  async measureAsync<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const span = this.startSpan(name);
    try {
      const result = await fn();
      this.endSpan(span.spanId, 'ok');
      return result;
    } catch (error) {
      this.endSpan(span.spanId, 'error');
      throw error;
    }
  }

  getWebVitals(): WebVitals {
    return { ...this.webVitals };
  }

  getMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  async flush(): Promise<void> {
    if (!this.endpoint || this.metrics.length === 0) return;

    const metricsToSend = [...this.metrics];
    this.metrics = [];

    try {
      await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics: metricsToSend }),
      });
    } catch {
      // Put metrics back
      this.metrics = [...metricsToSend, ...this.metrics];
    }
  }

  dispose(): void {
    for (const observer of this.observers) {
      observer.disconnect();
    }
    this.observers = [];
  }
}

// ==================== Analytics ====================

export class Analytics {
  private events: AnalyticsEvent[] = [];
  private session: UserSession;
  private maxEvents = 1000;
  private flushInterval: NodeJS.Timeout | null = null;
  private endpoint?: string;
  private anonymize: boolean;

  constructor(config: {
    endpoint?: string;
    anonymize?: boolean;
    flushIntervalMs?: number;
  } = {}) {
    this.endpoint = config.endpoint;
    this.anonymize = config.anonymize ?? false;

    this.session = this.createSession();
    this.setupFlushInterval(config.flushIntervalMs || 30000);
    this.setupActivityTracking();
  }

  private createSession(): UserSession {
    return {
      id: crypto.randomUUID(),
      startTime: Date.now(),
      lastActivity: Date.now(),
      pageViews: 0,
      events: 0,
      errors: 0,
      device: this.detectDevice(),
    };
  }

  private detectDevice(): DeviceInfo {
    if (typeof window === 'undefined') {
      return {
        type: 'desktop',
        os: 'unknown',
        browser: 'unknown',
        screenWidth: 0,
        screenHeight: 0,
        language: 'en',
        timezone: 'UTC',
      };
    }

    const ua = navigator.userAgent;
    let type: DeviceInfo['type'] = 'desktop';

    if (/Mobile|Android|iPhone|iPad/.test(ua)) {
      type = /iPad|Tablet/.test(ua) ? 'tablet' : 'mobile';
    }

    let os = 'unknown';
    if (/Windows/.test(ua)) os = 'Windows';
    else if (/Mac/.test(ua)) os = 'macOS';
    else if (/Linux/.test(ua)) os = 'Linux';
    else if (/Android/.test(ua)) os = 'Android';
    else if (/iOS|iPhone|iPad/.test(ua)) os = 'iOS';

    let browser = 'unknown';
    if (/Chrome/.test(ua)) browser = 'Chrome';
    else if (/Firefox/.test(ua)) browser = 'Firefox';
    else if (/Safari/.test(ua)) browser = 'Safari';
    else if (/Edge/.test(ua)) browser = 'Edge';

    return {
      type,
      os,
      browser,
      screenWidth: window.screen.width,
      screenHeight: window.screen.height,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    };
  }

  private setupFlushInterval(ms: number): void {
    this.flushInterval = setInterval(() => {
      this.flush();
    }, ms);
  }

  private setupActivityTracking(): void {
    if (typeof window === 'undefined') return;

    // Track page visibility
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.flush();
      }
    });

    // Track page unload
    window.addEventListener('beforeunload', () => {
      this.flush();
    });
  }

  track(name: string, properties: Record<string, any> = {}): void {
    const event: AnalyticsEvent = {
      name,
      properties: this.anonymize ? this.anonymizeProperties(properties) : properties,
      timestamp: Date.now(),
      sessionId: this.session.id,
      userId: this.session.userId,
    };

    this.events.push(event);
    this.session.events++;
    this.session.lastActivity = Date.now();

    if (this.events.length >= this.maxEvents) {
      this.flush();
    }
  }

  page(name: string, properties: Record<string, any> = {}): void {
    this.session.pageViews++;
    this.track('page_view', {
      page_name: name,
      page_url: typeof window !== 'undefined' ? window.location.href : '',
      referrer: typeof document !== 'undefined' ? document.referrer : '',
      ...properties,
    });
  }

  identify(userId: string, traits: Record<string, any> = {}): void {
    this.session.userId = userId;
    this.track('identify', { user_id: userId, ...traits });
  }

  error(errorId: string): void {
    this.session.errors++;
    this.track('error', { error_id: errorId });
  }

  getSession(): UserSession {
    return { ...this.session };
  }

  getEvents(): AnalyticsEvent[] {
    return [...this.events];
  }

  private anonymizeProperties(properties: Record<string, any>): Record<string, any> {
    const sensitiveKeys = ['email', 'name', 'phone', 'address', 'ip'];
    const anonymized: Record<string, any> = {};

    for (const [key, value] of Object.entries(properties)) {
      if (sensitiveKeys.some(k => key.toLowerCase().includes(k))) {
        anonymized[key] = '[REDACTED]';
      } else {
        anonymized[key] = value;
      }
    }

    return anonymized;
  }

  async flush(): Promise<void> {
    if (!this.endpoint || this.events.length === 0) return;

    const eventsToSend = [...this.events];
    this.events = [];

    try {
      const body = JSON.stringify({
        session: this.session,
        events: eventsToSend,
      });

      // Use sendBeacon for reliability during page unload
      if (navigator.sendBeacon) {
        navigator.sendBeacon(this.endpoint, body);
      } else {
        await fetch(this.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
          keepalive: true,
        });
      }
    } catch {
      // Put events back
      this.events = [...eventsToSend, ...this.events];
    }
  }

  dispose(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    this.flush();
  }
}

// ==================== Observability Manager ====================

export class ObservabilityManager {
  public readonly errors: ErrorTracker;
  public readonly performance: PerformanceMonitor;
  public readonly analytics: Analytics;

  constructor(config: {
    releaseVersion: string;
    errorEndpoint?: string;
    metricsEndpoint?: string;
    analyticsEndpoint?: string;
    anonymizeAnalytics?: boolean;
  }) {
    this.errors = new ErrorTracker({
      releaseVersion: config.releaseVersion,
      endpoint: config.errorEndpoint,
    });

    this.performance = new PerformanceMonitor({
      endpoint: config.metricsEndpoint,
    });

    this.analytics = new Analytics({
      endpoint: config.analyticsEndpoint,
      anonymize: config.anonymizeAnalytics,
    });
  }

  dispose(): void {
    this.performance.dispose();
    this.analytics.dispose();
  }
}

// ==================== Singleton ====================

let observabilityManager: ObservabilityManager | null = null;

export function getObservabilityManager(config?: {
  releaseVersion: string;
  errorEndpoint?: string;
  metricsEndpoint?: string;
  analyticsEndpoint?: string;
}): ObservabilityManager {
  if (!observabilityManager && config) {
    observabilityManager = new ObservabilityManager(config);
  }
  return observabilityManager!;
}

export default {
  ObservabilityManager,
  getObservabilityManager,
  ErrorTracker,
  PerformanceMonitor,
  Analytics,
};
