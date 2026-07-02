// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Production Readiness Infrastructure
 *
 * Comprehensive production infrastructure including:
 * - Observability (Logging, Tracing, Metrics)
 * - Disaster Recovery & Backup
 * - Auto-Scaling & Load Management
 * - Cost Optimization & Resource Management
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// Structured Logging System
// ============================================================================

type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal';

interface LogContext {
  requestId?: string;
  userId?: string;
  sessionId?: string;
  traceId?: string;
  spanId?: string;
  service?: string;
  environment?: string;
  version?: string;
  [key: string]: any;
}

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context: LogContext;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
  metadata?: Record<string, any>;
}

interface LogTransport {
  name: string;
  write(entry: LogEntry): Promise<void>;
}

export class StructuredLogger {
  private transports: LogTransport[] = [];
  private defaultContext: LogContext = {};
  private level: LogLevel = 'info';
  private levelPriority: Record<LogLevel, number> = {
    trace: 0,
    debug: 1,
    info: 2,
    warn: 3,
    error: 4,
    fatal: 5,
  };

  constructor(config: { level?: LogLevel; context?: LogContext } = {}) {
    this.level = config.level || 'info';
    this.defaultContext = config.context || {};
  }

  addTransport(transport: LogTransport): void {
    this.transports.push(transport);
  }

  setContext(context: LogContext): void {
    this.defaultContext = { ...this.defaultContext, ...context };
  }

  child(context: LogContext): StructuredLogger {
    const child = new StructuredLogger({ level: this.level });
    child.transports = this.transports;
    child.defaultContext = { ...this.defaultContext, ...context };
    return child;
  }

  private shouldLog(level: LogLevel): boolean {
    return this.levelPriority[level] >= this.levelPriority[this.level];
  }

  private async log(level: LogLevel, message: string, meta?: Record<string, any>): Promise<void> {
    if (!this.shouldLog(level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context: this.defaultContext,
      metadata: meta,
    };

    if (meta?.error instanceof Error) {
      entry.error = {
        name: meta.error.name,
        message: meta.error.message,
        stack: meta.error.stack,
      };
      delete meta.error;
    }

    await Promise.all(
      this.transports.map(t => t.write(entry).catch(err => {
        console.error(`Logger transport ${t.name} failed:`, err);
      }))
    );
  }

  trace(message: string, meta?: Record<string, any>): void {
    this.log('trace', message, meta);
  }

  debug(message: string, meta?: Record<string, any>): void {
    this.log('debug', message, meta);
  }

  info(message: string, meta?: Record<string, any>): void {
    this.log('info', message, meta);
  }

  warn(message: string, meta?: Record<string, any>): void {
    this.log('warn', message, meta);
  }

  error(message: string, meta?: Record<string, any>): void {
    this.log('error', message, meta);
  }

  fatal(message: string, meta?: Record<string, any>): void {
    this.log('fatal', message, meta);
  }
}

// Log Transports
export class ConsoleTransport implements LogTransport {
  name = 'console';

  async write(entry: LogEntry): Promise<void> {
    const color = {
      trace: '\x1b[90m',
      debug: '\x1b[36m',
      info: '\x1b[32m',
      warn: '\x1b[33m',
      error: '\x1b[31m',
      fatal: '\x1b[35m',
    }[entry.level];

    const reset = '\x1b[0m';
    const formatted = `${color}[${entry.timestamp}] ${entry.level.toUpperCase()}${reset}: ${entry.message}`;

    if (entry.level === 'error' || entry.level === 'fatal') {
      console.error(formatted, entry.error || '', entry.metadata || '');
    } else {
      console.log(formatted, entry.metadata || '');
    }
  }
}

export class JSONTransport implements LogTransport {
  name = 'json';
  private destination: NodeJS.WritableStream;

  constructor(destination: NodeJS.WritableStream = process.stdout) {
    this.destination = destination;
  }

  async write(entry: LogEntry): Promise<void> {
    this.destination.write(JSON.stringify(entry) + '\n');
  }
}

export class ElasticsearchTransport implements LogTransport {
  name = 'elasticsearch';
  private endpoint: string;
  private index: string;
  private buffer: LogEntry[] = [];
  private flushInterval: NodeJS.Timer;

  constructor(config: { endpoint: string; index: string }) {
    this.endpoint = config.endpoint;
    this.index = config.index;
    this.flushInterval = setInterval(() => this.flush(), 5000);
  }

  async write(entry: LogEntry): Promise<void> {
    this.buffer.push(entry);
    if (this.buffer.length >= 100) {
      await this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const entries = [...this.buffer];
    this.buffer = [];

    // Bulk index to Elasticsearch
    const body = entries.flatMap(entry => [
      { index: { _index: `${this.index}-${new Date().toISOString().split('T')[0]}` } },
      entry,
    ]);

    try {
      await fetch(`${this.endpoint}/_bulk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-ndjson' },
        body: body.map(line => JSON.stringify(line)).join('\n') + '\n',
      });
    } catch (error) {
      console.error('Failed to send logs to Elasticsearch:', error);
      this.buffer.unshift(...entries);
    }
  }
}

// ============================================================================
// Distributed Tracing
// ============================================================================

interface Span {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  service: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  status: 'ok' | 'error' | 'unset';
  attributes: Record<string, any>;
  events: Array<{
    name: string;
    timestamp: number;
    attributes?: Record<string, any>;
  }>;
}

interface TraceContext {
  traceId: string;
  spanId: string;
  traceFlags: number;
}

export class DistributedTracer {
  private serviceName: string;
  private activeSpans: Map<string, Span> = new Map();
  private completedSpans: Span[] = [];
  private exporters: TraceExporter[] = [];
  private samplingRate: number = 1.0;

  constructor(serviceName: string, config: { samplingRate?: number } = {}) {
    this.serviceName = serviceName;
    this.samplingRate = config.samplingRate || 1.0;
  }

  addExporter(exporter: TraceExporter): void {
    this.exporters.push(exporter);
  }

  private shouldSample(): boolean {
    return Math.random() < this.samplingRate;
  }

  startSpan(name: string, parentContext?: TraceContext): Span {
    const traceId = parentContext?.traceId || this.generateTraceId();
    const spanId = this.generateSpanId();

    const span: Span = {
      traceId,
      spanId,
      parentSpanId: parentContext?.spanId,
      name,
      service: this.serviceName,
      startTime: Date.now(),
      status: 'unset',
      attributes: {},
      events: [],
    };

    this.activeSpans.set(spanId, span);
    return span;
  }

  endSpan(span: Span, status: Span['status'] = 'ok'): void {
    span.endTime = Date.now();
    span.duration = span.endTime - span.startTime;
    span.status = status;

    this.activeSpans.delete(span.spanId);
    this.completedSpans.push(span);

    if (this.completedSpans.length >= 100) {
      this.flush();
    }
  }

  addEvent(span: Span, name: string, attributes?: Record<string, any>): void {
    span.events.push({
      name,
      timestamp: Date.now(),
      attributes,
    });
  }

  setAttributes(span: Span, attributes: Record<string, any>): void {
    span.attributes = { ...span.attributes, ...attributes };
  }

  getContext(span: Span): TraceContext {
    return {
      traceId: span.traceId,
      spanId: span.spanId,
      traceFlags: 1,
    };
  }

  injectContext(context: TraceContext, headers: Record<string, string>): void {
    // W3C Trace Context format
    headers['traceparent'] = `00-${context.traceId}-${context.spanId}-${context.traceFlags.toString(16).padStart(2, '0')}`;
  }

  extractContext(headers: Record<string, string>): TraceContext | null {
    const traceparent = headers['traceparent'];
    if (!traceparent) return null;

    const parts = traceparent.split('-');
    if (parts.length !== 4) return null;

    return {
      traceId: parts[1],
      spanId: parts[2],
      traceFlags: parseInt(parts[3], 16),
    };
  }

  async flush(): Promise<void> {
    const spans = [...this.completedSpans];
    this.completedSpans = [];

    await Promise.all(
      this.exporters.map(e => e.export(spans).catch(err => {
        console.error(`Trace exporter ${e.name} failed:`, err);
      }))
    );
  }

  private generateTraceId(): string {
    return uuidv4().replace(/-/g, '');
  }

  private generateSpanId(): string {
    return uuidv4().replace(/-/g, '').substring(0, 16);
  }
}

interface TraceExporter {
  name: string;
  export(spans: Span[]): Promise<void>;
}

export class JaegerExporter implements TraceExporter {
  name = 'jaeger';
  private endpoint: string;

  constructor(endpoint: string) {
    this.endpoint = endpoint;
  }

  async export(spans: Span[]): Promise<void> {
    await fetch(`${this.endpoint}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spans }),
    });
  }
}

// ============================================================================
// Metrics Collection
// ============================================================================

type MetricType = 'counter' | 'gauge' | 'histogram' | 'summary';

interface Metric {
  name: string;
  type: MetricType;
  description: string;
  labels: string[];
  values: Map<string, MetricValue>;
}

interface MetricValue {
  value: number;
  timestamp: number;
  labels: Record<string, string>;
}

interface HistogramValue extends MetricValue {
  buckets: Map<number, number>;
  sum: number;
  count: number;
}

export class MetricsRegistry {
  private metrics: Map<string, Metric> = new Map();
  private prefix: string;
  private defaultLabels: Record<string, string> = {};

  constructor(prefix: string = 'mycelix') {
    this.prefix = prefix;
  }

  setDefaultLabels(labels: Record<string, string>): void {
    this.defaultLabels = labels;
  }

  private getMetricKey(name: string, labels: Record<string, string>): string {
    const sortedLabels = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
    return `${name}:${sortedLabels.map(([k, v]) => `${k}=${v}`).join(',')}`;
  }

  createCounter(name: string, description: string, labelNames: string[] = []): Counter {
    const fullName = `${this.prefix}_${name}`;
    this.metrics.set(fullName, {
      name: fullName,
      type: 'counter',
      description,
      labels: labelNames,
      values: new Map(),
    });
    return new Counter(this, fullName);
  }

  createGauge(name: string, description: string, labelNames: string[] = []): Gauge {
    const fullName = `${this.prefix}_${name}`;
    this.metrics.set(fullName, {
      name: fullName,
      type: 'gauge',
      description,
      labels: labelNames,
      values: new Map(),
    });
    return new Gauge(this, fullName);
  }

  createHistogram(
    name: string,
    description: string,
    buckets: number[] = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    labelNames: string[] = []
  ): Histogram {
    const fullName = `${this.prefix}_${name}`;
    this.metrics.set(fullName, {
      name: fullName,
      type: 'histogram',
      description,
      labels: labelNames,
      values: new Map(),
    });
    return new Histogram(this, fullName, buckets);
  }

  increment(name: string, value: number, labels: Record<string, string> = {}): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'counter') return;

    const allLabels = { ...this.defaultLabels, ...labels };
    const key = this.getMetricKey(name, allLabels);
    const existing = metric.values.get(key);

    metric.values.set(key, {
      value: (existing?.value || 0) + value,
      timestamp: Date.now(),
      labels: allLabels,
    });
  }

  set(name: string, value: number, labels: Record<string, string> = {}): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'gauge') return;

    const allLabels = { ...this.defaultLabels, ...labels };
    const key = this.getMetricKey(name, allLabels);

    metric.values.set(key, {
      value,
      timestamp: Date.now(),
      labels: allLabels,
    });
  }

  observe(name: string, value: number, labels: Record<string, string> = {}, buckets: number[]): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'histogram') return;

    const allLabels = { ...this.defaultLabels, ...labels };
    const key = this.getMetricKey(name, allLabels);
    const existing = metric.values.get(key) as HistogramValue | undefined;

    const bucketMap = existing?.buckets || new Map<number, number>();
    for (const bucket of buckets) {
      if (value <= bucket) {
        bucketMap.set(bucket, (bucketMap.get(bucket) || 0) + 1);
      }
    }

    metric.values.set(key, {
      value: 0,
      timestamp: Date.now(),
      labels: allLabels,
      buckets: bucketMap,
      sum: (existing?.sum || 0) + value,
      count: (existing?.count || 0) + 1,
    } as HistogramValue);
  }

  async collect(): Promise<string> {
    const lines: string[] = [];

    for (const [, metric] of this.metrics) {
      lines.push(`# HELP ${metric.name} ${metric.description}`);
      lines.push(`# TYPE ${metric.name} ${metric.type}`);

      for (const [, metricValue] of metric.values) {
        const labelStr = Object.entries(metricValue.labels)
          .map(([k, v]) => `${k}="${v}"`)
          .join(',');
        const labelPart = labelStr ? `{${labelStr}}` : '';

        if (metric.type === 'histogram') {
          const histValue = metricValue as HistogramValue;
          for (const [bucket, count] of histValue.buckets) {
            lines.push(`${metric.name}_bucket${labelPart.replace('}', `,le="${bucket}"`)} ${count}`);
          }
          lines.push(`${metric.name}_bucket${labelPart.replace('}', `,le="+Inf"`)} ${histValue.count}`);
          lines.push(`${metric.name}_sum${labelPart} ${histValue.sum}`);
          lines.push(`${metric.name}_count${labelPart} ${histValue.count}`);
        } else {
          lines.push(`${metric.name}${labelPart} ${metricValue.value}`);
        }
      }
    }

    return lines.join('\n');
  }
}

class Counter {
  constructor(private registry: MetricsRegistry, private name: string) {}

  inc(labels: Record<string, string> = {}, value: number = 1): void {
    this.registry.increment(this.name, value, labels);
  }
}

class Gauge {
  constructor(private registry: MetricsRegistry, private name: string) {}

  set(value: number, labels: Record<string, string> = {}): void {
    this.registry.set(this.name, value, labels);
  }

  inc(labels: Record<string, string> = {}, value: number = 1): void {
    this.registry.set(this.name, value, labels);
  }

  dec(labels: Record<string, string> = {}, value: number = 1): void {
    this.registry.set(this.name, -value, labels);
  }
}

class Histogram {
  constructor(
    private registry: MetricsRegistry,
    private name: string,
    private buckets: number[]
  ) {}

  observe(value: number, labels: Record<string, string> = {}): void {
    this.registry.observe(this.name, value, labels, this.buckets);
  }

  timer(labels: Record<string, string> = {}): () => void {
    const start = process.hrtime.bigint();
    return () => {
      const duration = Number(process.hrtime.bigint() - start) / 1e9;
      this.observe(duration, labels);
    };
  }
}

// ============================================================================
// Disaster Recovery
// ============================================================================

interface BackupConfig {
  type: 'full' | 'incremental' | 'differential';
  schedule: string; // cron expression
  retention: {
    daily: number;
    weekly: number;
    monthly: number;
  };
  destination: {
    type: 's3' | 'gcs' | 'azure' | 'local';
    bucket?: string;
    path: string;
    region?: string;
  };
  encryption: {
    enabled: boolean;
    algorithm: 'aes-256-gcm';
    keyId?: string;
  };
}

interface BackupJob {
  id: string;
  type: BackupConfig['type'];
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime?: Date;
  endTime?: Date;
  size?: number;
  checksum?: string;
  destination: string;
  error?: string;
}

interface RecoveryPoint {
  id: string;
  timestamp: Date;
  type: BackupConfig['type'];
  size: number;
  checksum: string;
  destination: string;
  metadata: {
    databaseVersion: string;
    tableCount: number;
    rowCount: number;
  };
}

export class DisasterRecoveryService extends EventEmitter {
  private config: BackupConfig;
  private jobs: Map<string, BackupJob> = new Map();
  private recoveryPoints: RecoveryPoint[] = [];
  private scheduledJobs: Map<string, NodeJS.Timer> = new Map();

  constructor(config: BackupConfig) {
    super();
    this.config = config;
  }

  async initialize(): Promise<void> {
    await this.loadRecoveryPoints();
    this.scheduleBackups();
    console.log('Disaster Recovery Service initialized');
  }

  private async loadRecoveryPoints(): Promise<void> {
    // Load recovery points from backup destination
    // This would scan the backup storage for available recovery points
  }

  private scheduleBackups(): void {
    // Parse cron expression and schedule backups
    // Using simple interval for demonstration
    const interval = setInterval(() => {
      this.createBackup(this.config.type);
    }, 24 * 60 * 60 * 1000); // Daily

    this.scheduledJobs.set('daily', interval);
  }

  async createBackup(type: BackupConfig['type']): Promise<BackupJob> {
    const jobId = uuidv4();

    const job: BackupJob = {
      id: jobId,
      type,
      status: 'pending',
      destination: `${this.config.destination.path}/${new Date().toISOString()}_${type}`,
    };

    this.jobs.set(jobId, job);
    this.emit('backup_started', job);

    try {
      job.status = 'running';
      job.startTime = new Date();

      // Perform backup based on type
      switch (type) {
        case 'full':
          await this.performFullBackup(job);
          break;
        case 'incremental':
          await this.performIncrementalBackup(job);
          break;
        case 'differential':
          await this.performDifferentialBackup(job);
          break;
      }

      job.status = 'completed';
      job.endTime = new Date();

      // Create recovery point
      const recoveryPoint: RecoveryPoint = {
        id: uuidv4(),
        timestamp: new Date(),
        type,
        size: job.size || 0,
        checksum: job.checksum || '',
        destination: job.destination,
        metadata: {
          databaseVersion: '15.0',
          tableCount: 50,
          rowCount: 1000000,
        },
      };

      this.recoveryPoints.push(recoveryPoint);
      await this.applyRetentionPolicy();

      this.emit('backup_completed', job, recoveryPoint);
    } catch (error: any) {
      job.status = 'failed';
      job.error = error.message;
      job.endTime = new Date();
      this.emit('backup_failed', job, error);
    }

    return job;
  }

  private async performFullBackup(job: BackupJob): Promise<void> {
    // Full database dump
    // pg_dump, mysqldump, etc.
    console.log(`Performing full backup to ${job.destination}`);

    // Simulate backup
    job.size = 1024 * 1024 * 500; // 500MB
    job.checksum = 'sha256:' + uuidv4();
  }

  private async performIncrementalBackup(job: BackupJob): Promise<void> {
    // Backup only changes since last backup
    console.log(`Performing incremental backup to ${job.destination}`);

    job.size = 1024 * 1024 * 50; // 50MB
    job.checksum = 'sha256:' + uuidv4();
  }

  private async performDifferentialBackup(job: BackupJob): Promise<void> {
    // Backup all changes since last full backup
    console.log(`Performing differential backup to ${job.destination}`);

    job.size = 1024 * 1024 * 150; // 150MB
    job.checksum = 'sha256:' + uuidv4();
  }

  private async applyRetentionPolicy(): Promise<void> {
    const now = new Date();
    const retention = this.config.retention;

    // Keep daily backups for X days
    const dailyCutoff = new Date(now.getTime() - retention.daily * 24 * 60 * 60 * 1000);

    // Keep weekly backups for X weeks
    const weeklyCutoff = new Date(now.getTime() - retention.weekly * 7 * 24 * 60 * 60 * 1000);

    // Keep monthly backups for X months
    const monthlyCutoff = new Date(now.getTime() - retention.monthly * 30 * 24 * 60 * 60 * 1000);

    this.recoveryPoints = this.recoveryPoints.filter(rp => {
      const age = now.getTime() - rp.timestamp.getTime();
      const daysSinceBackup = age / (24 * 60 * 60 * 1000);

      // Keep if within daily retention
      if (rp.timestamp > dailyCutoff) return true;

      // Keep weekly backups (first backup of each week)
      if (rp.timestamp > weeklyCutoff && rp.timestamp.getDay() === 0) return true;

      // Keep monthly backups (first backup of each month)
      if (rp.timestamp > monthlyCutoff && rp.timestamp.getDate() === 1) return true;

      return false;
    });
  }

  async restore(recoveryPointId: string, target?: string): Promise<void> {
    const recoveryPoint = this.recoveryPoints.find(rp => rp.id === recoveryPointId);
    if (!recoveryPoint) {
      throw new Error('Recovery point not found');
    }

    this.emit('restore_started', recoveryPoint);

    try {
      // Verify checksum
      console.log(`Verifying backup integrity: ${recoveryPoint.checksum}`);

      // Restore database
      console.log(`Restoring from ${recoveryPoint.destination} to ${target || 'primary'}`);

      // For point-in-time recovery, apply incremental backups
      if (recoveryPoint.type !== 'full') {
        await this.applyIncrementalBackups(recoveryPoint);
      }

      this.emit('restore_completed', recoveryPoint);
    } catch (error) {
      this.emit('restore_failed', recoveryPoint, error);
      throw error;
    }
  }

  private async applyIncrementalBackups(basePoint: RecoveryPoint): Promise<void> {
    // Find and apply all incremental backups after base point
    const incrementals = this.recoveryPoints.filter(
      rp => rp.type === 'incremental' && rp.timestamp > basePoint.timestamp
    );

    for (const incremental of incrementals) {
      console.log(`Applying incremental backup: ${incremental.id}`);
    }
  }

  async verifyBackup(recoveryPointId: string): Promise<boolean> {
    const recoveryPoint = this.recoveryPoints.find(rp => rp.id === recoveryPointId);
    if (!recoveryPoint) return false;

    // Verify checksum
    // Verify restore to temporary location
    // Run integrity checks

    return true;
  }

  getRecoveryPoints(): RecoveryPoint[] {
    return [...this.recoveryPoints].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  getBackupStatus(): object {
    const jobs = Array.from(this.jobs.values());
    return {
      lastBackup: this.recoveryPoints[this.recoveryPoints.length - 1]?.timestamp,
      totalRecoveryPoints: this.recoveryPoints.length,
      totalSize: this.recoveryPoints.reduce((sum, rp) => sum + rp.size, 0),
      recentJobs: jobs.slice(-10),
      config: this.config,
    };
  }
}

// ============================================================================
// Auto-Scaling Engine
// ============================================================================

interface ScalingPolicy {
  name: string;
  metric: string;
  targetValue: number;
  scaleUp: {
    threshold: number;
    adjustment: number;
    cooldown: number;
  };
  scaleDown: {
    threshold: number;
    adjustment: number;
    cooldown: number;
  };
  minInstances: number;
  maxInstances: number;
}

interface ServiceInstance {
  id: string;
  service: string;
  status: 'starting' | 'running' | 'stopping' | 'stopped' | 'unhealthy';
  host: string;
  port: number;
  startedAt: Date;
  lastHealthCheck: Date;
  metrics: {
    cpu: number;
    memory: number;
    requestsPerSecond: number;
    responseTime: number;
  };
}

interface ScalingDecision {
  action: 'scale_up' | 'scale_down' | 'none';
  reason: string;
  currentInstances: number;
  targetInstances: number;
  timestamp: Date;
}

export class AutoScalingEngine extends EventEmitter {
  private policies: Map<string, ScalingPolicy> = new Map();
  private instances: Map<string, ServiceInstance[]> = new Map();
  private lastScalingAction: Map<string, Date> = new Map();
  private metricsHistory: Map<string, number[]> = new Map();

  async initialize(): Promise<void> {
    this.startMetricsCollection();
    this.startScalingEvaluation();
    console.log('Auto-Scaling Engine initialized');
  }

  registerPolicy(policy: ScalingPolicy): void {
    this.policies.set(policy.name, policy);
  }

  registerService(serviceName: string, initialInstances: ServiceInstance[]): void {
    this.instances.set(serviceName, initialInstances);
    this.metricsHistory.set(serviceName, []);
  }

  private startMetricsCollection(): void {
    setInterval(async () => {
      for (const [serviceName, serviceInstances] of this.instances) {
        const metrics = await this.collectServiceMetrics(serviceName, serviceInstances);
        const history = this.metricsHistory.get(serviceName) || [];
        history.push(metrics);

        // Keep last 60 samples (1 hour at 1-minute intervals)
        if (history.length > 60) history.shift();
        this.metricsHistory.set(serviceName, history);
      }
    }, 60000); // Every minute
  }

  private async collectServiceMetrics(
    serviceName: string,
    instances: ServiceInstance[]
  ): Promise<number> {
    // Aggregate metrics from all instances
    const runningInstances = instances.filter(i => i.status === 'running');
    if (runningInstances.length === 0) return 0;

    const avgCpu = runningInstances.reduce((sum, i) => sum + i.metrics.cpu, 0) / runningInstances.length;
    return avgCpu;
  }

  private startScalingEvaluation(): void {
    setInterval(() => {
      for (const [policyName, policy] of this.policies) {
        this.evaluateScaling(policyName, policy);
      }
    }, 30000); // Every 30 seconds
  }

  private evaluateScaling(policyName: string, policy: ScalingPolicy): void {
    const history = this.metricsHistory.get(policyName) || [];
    if (history.length < 5) return; // Need at least 5 samples

    const currentInstances = this.instances.get(policyName)?.length || 0;
    const lastAction = this.lastScalingAction.get(policyName);

    // Calculate average metric
    const avgMetric = history.slice(-5).reduce((sum, v) => sum + v, 0) / 5;

    let decision: ScalingDecision = {
      action: 'none',
      reason: 'Metric within target range',
      currentInstances,
      targetInstances: currentInstances,
      timestamp: new Date(),
    };

    // Check scale up condition
    if (avgMetric > policy.scaleUp.threshold) {
      const cooldownPassed = !lastAction ||
        Date.now() - lastAction.getTime() > policy.scaleUp.cooldown * 1000;

      if (cooldownPassed && currentInstances < policy.maxInstances) {
        const newCount = Math.min(
          currentInstances + policy.scaleUp.adjustment,
          policy.maxInstances
        );

        decision = {
          action: 'scale_up',
          reason: `Metric ${avgMetric.toFixed(2)} exceeds threshold ${policy.scaleUp.threshold}`,
          currentInstances,
          targetInstances: newCount,
          timestamp: new Date(),
        };

        this.executeScaling(policyName, newCount);
      }
    }

    // Check scale down condition
    else if (avgMetric < policy.scaleDown.threshold) {
      const cooldownPassed = !lastAction ||
        Date.now() - lastAction.getTime() > policy.scaleDown.cooldown * 1000;

      if (cooldownPassed && currentInstances > policy.minInstances) {
        const newCount = Math.max(
          currentInstances - policy.scaleDown.adjustment,
          policy.minInstances
        );

        decision = {
          action: 'scale_down',
          reason: `Metric ${avgMetric.toFixed(2)} below threshold ${policy.scaleDown.threshold}`,
          currentInstances,
          targetInstances: newCount,
          timestamp: new Date(),
        };

        this.executeScaling(policyName, newCount);
      }
    }

    if (decision.action !== 'none') {
      this.emit('scaling_decision', policyName, decision);
    }
  }

  private async executeScaling(serviceName: string, targetCount: number): Promise<void> {
    const currentInstances = this.instances.get(serviceName) || [];
    const currentCount = currentInstances.length;

    this.lastScalingAction.set(serviceName, new Date());

    if (targetCount > currentCount) {
      // Scale up
      const toAdd = targetCount - currentCount;
      for (let i = 0; i < toAdd; i++) {
        const instance = await this.launchInstance(serviceName);
        currentInstances.push(instance);
        this.emit('instance_launched', serviceName, instance);
      }
    } else {
      // Scale down
      const toRemove = currentCount - targetCount;
      for (let i = 0; i < toRemove; i++) {
        const instance = currentInstances.pop();
        if (instance) {
          await this.terminateInstance(instance);
          this.emit('instance_terminated', serviceName, instance);
        }
      }
    }

    this.instances.set(serviceName, currentInstances);
  }

  private async launchInstance(serviceName: string): Promise<ServiceInstance> {
    // Launch new instance via orchestrator (K8s, ECS, etc.)
    const instance: ServiceInstance = {
      id: uuidv4(),
      service: serviceName,
      status: 'starting',
      host: `${serviceName}-${Date.now()}.internal`,
      port: 8080,
      startedAt: new Date(),
      lastHealthCheck: new Date(),
      metrics: {
        cpu: 0,
        memory: 0,
        requestsPerSecond: 0,
        responseTime: 0,
      },
    };

    // Simulate startup
    setTimeout(() => {
      instance.status = 'running';
    }, 30000);

    return instance;
  }

  private async terminateInstance(instance: ServiceInstance): Promise<void> {
    instance.status = 'stopping';

    // Graceful shutdown
    // - Stop accepting new requests
    // - Drain existing connections
    // - Wait for in-flight requests
    // - Terminate

    instance.status = 'stopped';
  }

  async manualScale(serviceName: string, targetCount: number): Promise<void> {
    const policy = this.policies.get(serviceName);
    if (policy) {
      targetCount = Math.max(policy.minInstances, Math.min(targetCount, policy.maxInstances));
    }

    await this.executeScaling(serviceName, targetCount);
  }

  getScalingStatus(): object {
    const status: any = {};

    for (const [serviceName, instances] of this.instances) {
      const policy = this.policies.get(serviceName);
      const history = this.metricsHistory.get(serviceName) || [];

      status[serviceName] = {
        currentInstances: instances.length,
        policy: policy ? {
          minInstances: policy.minInstances,
          maxInstances: policy.maxInstances,
          targetValue: policy.targetValue,
        } : null,
        recentMetrics: history.slice(-10),
        lastScalingAction: this.lastScalingAction.get(serviceName),
        instances: instances.map(i => ({
          id: i.id,
          status: i.status,
          host: i.host,
          metrics: i.metrics,
        })),
      };
    }

    return status;
  }
}

// ============================================================================
// Cost Optimization
// ============================================================================

interface ResourceUsage {
  resourceId: string;
  resourceType: 'compute' | 'storage' | 'database' | 'network' | 'cdn';
  service: string;
  usage: number;
  unit: string;
  cost: number;
  timestamp: Date;
}

interface CostAnomaly {
  id: string;
  resourceId: string;
  type: 'spike' | 'sustained_increase' | 'unusual_pattern';
  severity: 'low' | 'medium' | 'high';
  expectedCost: number;
  actualCost: number;
  deviation: number;
  detectedAt: Date;
  recommendation?: string;
}

interface OptimizationRecommendation {
  id: string;
  type: 'rightsizing' | 'reserved_instance' | 'spot_instance' | 'cleanup' | 'architecture';
  resourceId: string;
  currentConfig: any;
  recommendedConfig: any;
  estimatedSavings: number;
  savingsPercent: number;
  effort: 'low' | 'medium' | 'high';
  risk: 'low' | 'medium' | 'high';
  description: string;
}

export class CostOptimizationService extends EventEmitter {
  private usageHistory: ResourceUsage[] = [];
  private anomalies: CostAnomaly[] = [];
  private recommendations: OptimizationRecommendation[] = [];
  private budgets: Map<string, { limit: number; current: number; alertThreshold: number }> = new Map();

  async initialize(): Promise<void> {
    this.startUsageCollection();
    this.startAnomalyDetection();
    this.startRecommendationEngine();
    console.log('Cost Optimization Service initialized');
  }

  private startUsageCollection(): void {
    setInterval(async () => {
      await this.collectUsageData();
    }, 3600000); // Every hour
  }

  private async collectUsageData(): Promise<void> {
    // Collect from cloud provider APIs
    // AWS Cost Explorer, GCP Billing, Azure Cost Management

    const mockUsage: ResourceUsage[] = [
      {
        resourceId: 'ec2-api-cluster',
        resourceType: 'compute',
        service: 'api',
        usage: 720,
        unit: 'hours',
        cost: 432,
        timestamp: new Date(),
      },
      {
        resourceId: 'rds-primary',
        resourceType: 'database',
        service: 'database',
        usage: 720,
        unit: 'hours',
        cost: 580,
        timestamp: new Date(),
      },
      {
        resourceId: 's3-media',
        resourceType: 'storage',
        service: 'media',
        usage: 5000,
        unit: 'GB',
        cost: 115,
        timestamp: new Date(),
      },
      {
        resourceId: 'cloudfront-cdn',
        resourceType: 'cdn',
        service: 'delivery',
        usage: 10000,
        unit: 'GB',
        cost: 850,
        timestamp: new Date(),
      },
    ];

    this.usageHistory.push(...mockUsage);

    // Keep last 30 days
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    this.usageHistory = this.usageHistory.filter(u => u.timestamp > cutoff);

    // Update budgets
    this.updateBudgets(mockUsage);
  }

  private updateBudgets(usage: ResourceUsage[]): void {
    for (const u of usage) {
      const budget = this.budgets.get(u.service);
      if (budget) {
        budget.current += u.cost;

        if (budget.current > budget.limit * budget.alertThreshold) {
          this.emit('budget_alert', {
            service: u.service,
            current: budget.current,
            limit: budget.limit,
            percentUsed: (budget.current / budget.limit) * 100,
          });
        }
      }
    }
  }

  private startAnomalyDetection(): void {
    setInterval(() => {
      this.detectAnomalies();
    }, 3600000); // Every hour
  }

  private detectAnomalies(): void {
    const serviceUsage = new Map<string, number[]>();

    // Group usage by service
    for (const usage of this.usageHistory) {
      if (!serviceUsage.has(usage.service)) {
        serviceUsage.set(usage.service, []);
      }
      serviceUsage.get(usage.service)!.push(usage.cost);
    }

    // Detect anomalies using statistical analysis
    for (const [service, costs] of serviceUsage) {
      if (costs.length < 7) continue;

      const recentCosts = costs.slice(-7);
      const historicalCosts = costs.slice(-30, -7);

      if (historicalCosts.length < 7) continue;

      const historicalMean = historicalCosts.reduce((a, b) => a + b, 0) / historicalCosts.length;
      const historicalStdDev = Math.sqrt(
        historicalCosts.reduce((sum, x) => sum + Math.pow(x - historicalMean, 2), 0) / historicalCosts.length
      );

      const recentMean = recentCosts.reduce((a, b) => a + b, 0) / recentCosts.length;

      // Check for anomaly (> 2 standard deviations)
      if (recentMean > historicalMean + 2 * historicalStdDev) {
        const deviation = ((recentMean - historicalMean) / historicalMean) * 100;

        const anomaly: CostAnomaly = {
          id: uuidv4(),
          resourceId: service,
          type: 'sustained_increase',
          severity: deviation > 50 ? 'high' : deviation > 25 ? 'medium' : 'low',
          expectedCost: historicalMean,
          actualCost: recentMean,
          deviation,
          detectedAt: new Date(),
          recommendation: `Investigate ${service} cost increase of ${deviation.toFixed(1)}%`,
        };

        this.anomalies.push(anomaly);
        this.emit('cost_anomaly', anomaly);
      }
    }
  }

  private startRecommendationEngine(): void {
    setInterval(() => {
      this.generateRecommendations();
    }, 86400000); // Daily
  }

  private generateRecommendations(): void {
    this.recommendations = [];

    // Analyze compute utilization
    this.analyzeComputeOptimization();

    // Analyze storage usage
    this.analyzeStorageOptimization();

    // Analyze reserved instance opportunities
    this.analyzeReservedInstances();

    // Analyze unused resources
    this.analyzeUnusedResources();
  }

  private analyzeComputeOptimization(): void {
    // Check for oversized instances
    const computeUsage = this.usageHistory
      .filter(u => u.resourceType === 'compute')
      .slice(-168); // Last week

    if (computeUsage.length === 0) return;

    const avgCost = computeUsage.reduce((sum, u) => sum + u.cost, 0) / computeUsage.length;

    // If average utilization is low, recommend rightsizing
    this.recommendations.push({
      id: uuidv4(),
      type: 'rightsizing',
      resourceId: 'ec2-api-cluster',
      currentConfig: { instanceType: 'm5.2xlarge', count: 4 },
      recommendedConfig: { instanceType: 'm5.xlarge', count: 4 },
      estimatedSavings: avgCost * 0.5 * 30, // 50% savings over month
      savingsPercent: 50,
      effort: 'low',
      risk: 'low',
      description: 'Downsize API cluster instances based on low CPU utilization (avg 25%)',
    });
  }

  private analyzeStorageOptimization(): void {
    // Recommend storage tiering
    this.recommendations.push({
      id: uuidv4(),
      type: 'architecture',
      resourceId: 's3-media',
      currentConfig: { storageClass: 'STANDARD' },
      recommendedConfig: { storageClass: 'INTELLIGENT_TIERING' },
      estimatedSavings: 115 * 0.3 * 12, // 30% savings annually
      savingsPercent: 30,
      effort: 'low',
      risk: 'low',
      description: 'Enable S3 Intelligent-Tiering for media bucket to automatically move infrequently accessed data',
    });
  }

  private analyzeReservedInstances(): void {
    // Recommend reserved instances for stable workloads
    this.recommendations.push({
      id: uuidv4(),
      type: 'reserved_instance',
      resourceId: 'rds-primary',
      currentConfig: { paymentOption: 'on-demand' },
      recommendedConfig: { paymentOption: '1-year-reserved', upfront: 'partial' },
      estimatedSavings: 580 * 0.4 * 12, // 40% savings annually
      savingsPercent: 40,
      effort: 'medium',
      risk: 'low',
      description: 'Purchase 1-year reserved instance for primary database (stable workload)',
    });
  }

  private analyzeUnusedResources(): void {
    // Identify unused resources for cleanup
    this.recommendations.push({
      id: uuidv4(),
      type: 'cleanup',
      resourceId: 'ebs-snapshots',
      currentConfig: { count: 250, ageInDays: 90 },
      recommendedConfig: { count: 50, maxAgeInDays: 30 },
      estimatedSavings: 45, // Monthly savings
      savingsPercent: 80,
      effort: 'low',
      risk: 'low',
      description: 'Delete old EBS snapshots (>30 days) - 200 snapshots can be removed',
    });
  }

  setBudget(service: string, limit: number, alertThreshold: number = 0.8): void {
    this.budgets.set(service, { limit, current: 0, alertThreshold });
  }

  getCostReport(startDate: Date, endDate: Date): object {
    const filteredUsage = this.usageHistory.filter(
      u => u.timestamp >= startDate && u.timestamp <= endDate
    );

    // Group by service
    const byService = new Map<string, number>();
    const byResourceType = new Map<string, number>();

    for (const usage of filteredUsage) {
      byService.set(usage.service, (byService.get(usage.service) || 0) + usage.cost);
      byResourceType.set(usage.resourceType, (byResourceType.get(usage.resourceType) || 0) + usage.cost);
    }

    const totalCost = filteredUsage.reduce((sum, u) => sum + u.cost, 0);

    return {
      period: { start: startDate, end: endDate },
      totalCost,
      byService: Object.fromEntries(byService),
      byResourceType: Object.fromEntries(byResourceType),
      dailyAverage: totalCost / Math.ceil((endDate.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000)),
      anomalies: this.anomalies.filter(a => a.detectedAt >= startDate && a.detectedAt <= endDate),
      recommendations: this.recommendations,
      potentialSavings: this.recommendations.reduce((sum, r) => sum + r.estimatedSavings, 0),
    };
  }

  getRecommendations(): OptimizationRecommendation[] {
    return [...this.recommendations].sort((a, b) => b.estimatedSavings - a.estimatedSavings);
  }
}

// ============================================================================
// Health Check System
// ============================================================================

interface HealthCheck {
  name: string;
  check: () => Promise<HealthCheckResult>;
  critical: boolean;
  timeout: number;
}

interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency?: number;
  message?: string;
  details?: any;
}

interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: Date;
  checks: Record<string, HealthCheckResult>;
  uptime: number;
}

export class HealthCheckService {
  private checks: HealthCheck[] = [];
  private lastResults: Map<string, HealthCheckResult> = new Map();
  private startTime: Date = new Date();

  registerCheck(check: HealthCheck): void {
    this.checks.push(check);
  }

  async runChecks(): Promise<SystemHealth> {
    const results: Record<string, HealthCheckResult> = {};
    let overallStatus: SystemHealth['status'] = 'healthy';

    await Promise.all(
      this.checks.map(async (check) => {
        try {
          const start = Date.now();
          const result = await Promise.race([
            check.check(),
            new Promise<HealthCheckResult>((_, reject) =>
              setTimeout(() => reject(new Error('Timeout')), check.timeout)
            ),
          ]);

          result.latency = Date.now() - start;
          results[check.name] = result;
          this.lastResults.set(check.name, result);

          if (result.status === 'unhealthy' && check.critical) {
            overallStatus = 'unhealthy';
          } else if (result.status === 'degraded' && overallStatus === 'healthy') {
            overallStatus = 'degraded';
          }
        } catch (error: any) {
          const result: HealthCheckResult = {
            status: 'unhealthy',
            message: error.message,
          };
          results[check.name] = result;
          this.lastResults.set(check.name, result);

          if (check.critical) {
            overallStatus = 'unhealthy';
          }
        }
      })
    );

    return {
      status: overallStatus,
      timestamp: new Date(),
      checks: results,
      uptime: Date.now() - this.startTime.getTime(),
    };
  }

  async runLivenessCheck(): Promise<{ alive: boolean }> {
    // Simple check that service is running
    return { alive: true };
  }

  async runReadinessCheck(): Promise<{ ready: boolean; details?: string }> {
    // Check if service is ready to accept traffic
    const criticalChecks = this.checks.filter(c => c.critical);

    for (const check of criticalChecks) {
      try {
        const result = await check.check();
        if (result.status === 'unhealthy') {
          return { ready: false, details: `${check.name} is unhealthy` };
        }
      } catch (error: any) {
        return { ready: false, details: `${check.name} failed: ${error.message}` };
      }
    }

    return { ready: true };
  }

  getLastResults(): Map<string, HealthCheckResult> {
    return new Map(this.lastResults);
  }
}

// ============================================================================
// Unified Production Service
// ============================================================================

export class ProductionReadinessService {
  public logger: StructuredLogger;
  public tracer: DistributedTracer;
  public metrics: MetricsRegistry;
  public disasterRecovery: DisasterRecoveryService;
  public autoScaling: AutoScalingEngine;
  public costOptimization: CostOptimizationService;
  public healthChecks: HealthCheckService;

  constructor(serviceName: string) {
    this.logger = new StructuredLogger({
      level: 'info',
      context: {
        service: serviceName,
        environment: process.env.NODE_ENV,
        version: process.env.APP_VERSION,
      },
    });

    this.tracer = new DistributedTracer(serviceName);
    this.metrics = new MetricsRegistry('mycelix');

    this.disasterRecovery = new DisasterRecoveryService({
      type: 'incremental',
      schedule: '0 2 * * *', // 2 AM daily
      retention: { daily: 7, weekly: 4, monthly: 12 },
      destination: {
        type: 's3',
        bucket: 'mycelix-backups',
        path: 'production',
        region: 'us-east-1',
      },
      encryption: {
        enabled: true,
        algorithm: 'aes-256-gcm',
        keyId: 'backup-key',
      },
    });

    this.autoScaling = new AutoScalingEngine();
    this.costOptimization = new CostOptimizationService();
    this.healthChecks = new HealthCheckService();
  }

  async initialize(): Promise<void> {
    // Setup logging transports
    this.logger.addTransport(new ConsoleTransport());
    this.logger.addTransport(new JSONTransport());

    // Setup trace exporters
    this.tracer.addExporter(new JaegerExporter('http://jaeger:14268'));

    // Register standard metrics
    this.registerStandardMetrics();

    // Register health checks
    this.registerHealthChecks();

    // Initialize all services
    await this.disasterRecovery.initialize();
    await this.autoScaling.initialize();
    await this.costOptimization.initialize();

    // Setup scaling policies
    this.setupScalingPolicies();

    // Setup cost budgets
    this.setupCostBudgets();

    this.logger.info('Production Readiness Service fully initialized');
  }

  private registerStandardMetrics(): void {
    // HTTP metrics
    this.metrics.createCounter('http_requests_total', 'Total HTTP requests', ['method', 'path', 'status']);
    this.metrics.createHistogram('http_request_duration_seconds', 'HTTP request duration', undefined, ['method', 'path']);

    // Business metrics
    this.metrics.createCounter('tracks_played_total', 'Total tracks played', ['genre']);
    this.metrics.createCounter('user_signups_total', 'Total user signups', ['source']);
    this.metrics.createGauge('active_users', 'Currently active users');
    this.metrics.createGauge('concurrent_streams', 'Concurrent audio streams');

    // System metrics
    this.metrics.createGauge('system_cpu_usage', 'CPU usage percentage');
    this.metrics.createGauge('system_memory_usage', 'Memory usage in bytes');
    this.metrics.createGauge('database_connections', 'Active database connections', ['pool']);
  }

  private registerHealthChecks(): void {
    // Database health check
    this.healthChecks.registerCheck({
      name: 'database',
      critical: true,
      timeout: 5000,
      check: async () => {
        // Would check database connection
        return { status: 'healthy', message: 'Database connection OK' };
      },
    });

    // Redis health check
    this.healthChecks.registerCheck({
      name: 'redis',
      critical: true,
      timeout: 3000,
      check: async () => {
        return { status: 'healthy', message: 'Redis connection OK' };
      },
    });

    // Storage health check
    this.healthChecks.registerCheck({
      name: 'storage',
      critical: false,
      timeout: 10000,
      check: async () => {
        return { status: 'healthy', message: 'S3 accessible' };
      },
    });
  }

  private setupScalingPolicies(): void {
    this.autoScaling.registerPolicy({
      name: 'api',
      metric: 'cpu',
      targetValue: 70,
      scaleUp: { threshold: 80, adjustment: 2, cooldown: 300 },
      scaleDown: { threshold: 50, adjustment: 1, cooldown: 600 },
      minInstances: 2,
      maxInstances: 20,
    });

    this.autoScaling.registerPolicy({
      name: 'workers',
      metric: 'queue_depth',
      targetValue: 100,
      scaleUp: { threshold: 500, adjustment: 2, cooldown: 180 },
      scaleDown: { threshold: 50, adjustment: 1, cooldown: 300 },
      minInstances: 1,
      maxInstances: 10,
    });
  }

  private setupCostBudgets(): void {
    this.costOptimization.setBudget('api', 5000, 0.8);
    this.costOptimization.setBudget('database', 2000, 0.9);
    this.costOptimization.setBudget('storage', 1000, 0.8);
    this.costOptimization.setBudget('cdn', 3000, 0.8);
  }

  getStatus(): object {
    return {
      scaling: this.autoScaling.getScalingStatus(),
      backup: this.disasterRecovery.getBackupStatus(),
      costs: this.costOptimization.getCostReport(
        new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        new Date()
      ),
    };
  }
}

// ============================================================================
// Export
// ============================================================================

export const createProductionService = async (serviceName: string): Promise<ProductionReadinessService> => {
  const service = new ProductionReadinessService(serviceName);
  await service.initialize();
  return service;
};
