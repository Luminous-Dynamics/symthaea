// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Dashboard Metrics System
 *
 * Provides visual dashboard components for monitoring agent trust metrics,
 * network health, and security alerts.
 *
 * @packageDocumentation
 * @module agentic/dashboard
 */

// =============================================================================
// Configuration
// =============================================================================

/**
 * Dashboard configuration
 */
export interface DashboardConfig {
  /** Maximum time series points to retain */
  maxTimeSeriesPoints: number;
  /** Alert retention duration in milliseconds */
  alertRetentionMs: number;
  /** Metrics aggregation window in milliseconds */
  aggregationWindowMs: number;
}

/**
 * Create default dashboard configuration
 */
export function createDefaultDashboardConfig(): DashboardConfig {
  return {
    maxTimeSeriesPoints: 1000,
    alertRetentionMs: 24 * 60 * 60 * 1000, // 24 hours
    aggregationWindowMs: 60 * 1000, // 1 minute
  };
}

// =============================================================================
// Live Metrics
// =============================================================================

/**
 * Live dashboard metrics
 */
export interface LiveMetrics {
  /** Timestamp of metrics snapshot */
  timestamp: number;
  /** Total active agents */
  activeAgents: number;
  /** Total trust score (aggregate) */
  totalTrust: number;
  /** Average trust score */
  averageTrust: number;
  /** Network health (0.0-1.0) */
  networkHealth: number;
  /** Transactions per second */
  tps: number;
  /** Active alerts by severity */
  alerts: AlertCounts;
  /** Collective Phi coherence */
  collectivePhi: number;
  /** Byzantine threat level (0.0-1.0) */
  byzantineThreat: number;
}

/**
 * Alert counts by severity
 */
export interface AlertCounts {
  critical: number;
  high: number;
  medium: number;
  low: number;
  info: number;
}

/**
 * Create empty alert counts
 */
export function createEmptyAlertCounts(): AlertCounts {
  return { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
}

// =============================================================================
// Time Series
// =============================================================================

/**
 * Time series data point
 */
export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
}

/**
 * Time series tracker
 */
export class TimeSeries {
  private points: TimeSeriesPoint[] = [];
  private maxPoints: number;

  constructor(maxPoints = 1000) {
    this.maxPoints = maxPoints;
  }

  /**
   * Add a data point
   */
  add(value: number, timestamp = Date.now()): void {
    this.points.push({ timestamp, value });

    // Trim if needed
    if (this.points.length > this.maxPoints) {
      this.points = this.points.slice(-this.maxPoints);
    }
  }

  /**
   * Get points in time range
   */
  range(startTime: number, endTime = Date.now()): TimeSeriesPoint[] {
    return this.points.filter(
      (p) => p.timestamp >= startTime && p.timestamp <= endTime
    );
  }

  /**
   * Get latest N points
   */
  latest(n: number): TimeSeriesPoint[] {
    return this.points.slice(-n);
  }

  /**
   * Compute moving average
   */
  movingAverage(windowSize: number): number {
    const recent = this.latest(windowSize);
    if (recent.length === 0) return 0;
    return recent.reduce((sum, p) => sum + p.value, 0) / recent.length;
  }

  /**
   * Get all points
   */
  getPoints(): TimeSeriesPoint[] {
    return [...this.points];
  }
}

// =============================================================================
// Alert System
// =============================================================================

/**
 * Alert severity levels
 */
export enum AlertSeverity {
  Critical = 'critical',
  High = 'high',
  Medium = 'medium',
  Low = 'low',
  Info = 'info',
}

/**
 * Alert status
 */
export enum AlertStatus {
  Active = 'active',
  Acknowledged = 'acknowledged',
  Resolved = 'resolved',
}

/**
 * Alert definition
 */
export interface Alert {
  /** Unique alert ID */
  id: string;
  /** Alert severity */
  severity: AlertSeverity;
  /** Alert title */
  title: string;
  /** Alert description */
  description: string;
  /** Source component */
  source: string;
  /** Current status */
  status: AlertStatus;
  /** Creation timestamp */
  createdAt: number;
  /** Last update timestamp */
  updatedAt: number;
  /** Resolution timestamp (if resolved) */
  resolvedAt?: number;
}

/**
 * Alert panel manager
 */
export class AlertPanel {
  private alerts: Map<string, Alert> = new Map();
  private retentionMs: number;
  private idCounter = 0;

  constructor(retentionMs = 24 * 60 * 60 * 1000) {
    this.retentionMs = retentionMs;
  }

  /**
   * Create a new alert
   */
  createAlert(
    severity: AlertSeverity,
    title: string,
    description: string,
    source: string
  ): Alert {
    const now = Date.now();
    const id = `alert-${now}-${++this.idCounter}`;

    const alert: Alert = {
      id,
      severity,
      title,
      description,
      source,
      status: AlertStatus.Active,
      createdAt: now,
      updatedAt: now,
    };

    this.alerts.set(id, alert);
    this.cleanup();

    return alert;
  }

  /**
   * Acknowledge an alert
   */
  acknowledge(alertId: string): boolean {
    const alert = this.alerts.get(alertId);
    if (!alert) return false;

    alert.status = AlertStatus.Acknowledged;
    alert.updatedAt = Date.now();
    return true;
  }

  /**
   * Resolve an alert
   */
  resolve(alertId: string): boolean {
    const alert = this.alerts.get(alertId);
    if (!alert) return false;

    alert.status = AlertStatus.Resolved;
    alert.updatedAt = Date.now();
    alert.resolvedAt = Date.now();
    return true;
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): Alert[] {
    return Array.from(this.alerts.values()).filter(
      (a) => a.status === AlertStatus.Active
    );
  }

  /**
   * Get alerts by severity
   */
  getAlertsBySeverity(severity: AlertSeverity): Alert[] {
    return Array.from(this.alerts.values()).filter(
      (a) => a.severity === severity
    );
  }

  /**
   * Count alerts by status
   */
  countBySeverity(): AlertCounts {
    const counts = createEmptyAlertCounts();
    for (const alert of this.alerts.values()) {
      if (alert.status !== AlertStatus.Resolved) {
        counts[alert.severity]++;
      }
    }
    return counts;
  }

  /**
   * Clean up old resolved alerts
   */
  private cleanup(): void {
    const cutoff = Date.now() - this.retentionMs;
    for (const [id, alert] of this.alerts) {
      if (alert.status === AlertStatus.Resolved && alert.resolvedAt! < cutoff) {
        this.alerts.delete(id);
      }
    }
  }
}

// =============================================================================
// Metrics Input
// =============================================================================

/**
 * Input data for metrics update
 */
export interface MetricsInput {
  /** Trust scores from agents */
  trustScores: number[];
  /** Transaction count this period */
  transactionCount: number;
  /** New alerts to create */
  alerts: Array<{
    severity: AlertSeverity;
    title: string;
    description: string;
    source: string;
  }>;
  /** Phi values from agents */
  phiValues: number[];
  /** Threat levels detected */
  threats: number[];
}

// =============================================================================
// Dashboard
// =============================================================================

/**
 * Main dashboard component
 */
export class Dashboard {
  private config: DashboardConfig;
  readonly alerts: AlertPanel;
  private trustSeries: TimeSeries;
  private tpsSeries: TimeSeries;
  private phiSeries: TimeSeries;
  private threatSeries: TimeSeries;
  private lastUpdateTime = 0;

  constructor(config: DashboardConfig = createDefaultDashboardConfig()) {
    this.config = config;
    void this.config; // Suppress unused warning
    this.alerts = new AlertPanel(config.alertRetentionMs);
    this.trustSeries = new TimeSeries(config.maxTimeSeriesPoints);
    this.tpsSeries = new TimeSeries(config.maxTimeSeriesPoints);
    this.phiSeries = new TimeSeries(config.maxTimeSeriesPoints);
    this.threatSeries = new TimeSeries(config.maxTimeSeriesPoints);
  }

  /**
   * Update dashboard with new metrics
   */
  update(input: MetricsInput, timestamp = Date.now()): LiveMetrics {
    // Calculate time delta for TPS
    const timeDelta = timestamp - this.lastUpdateTime;
    this.lastUpdateTime = timestamp;

    // Compute aggregate metrics
    const totalTrust = input.trustScores.reduce((a, b) => a + b, 0);
    const averageTrust =
      input.trustScores.length > 0
        ? totalTrust / input.trustScores.length
        : 0;

    const collectivePhi =
      input.phiValues.length > 0
        ? input.phiValues.reduce((a, b) => a + b, 0) / input.phiValues.length
        : 0;

    const byzantineThreat =
      input.threats.length > 0
        ? Math.max(...input.threats)
        : 0;

    // Compute TPS
    const tps =
      timeDelta > 0 ? (input.transactionCount * 1000) / timeDelta : 0;

    // Create alerts
    for (const alertDef of input.alerts) {
      this.alerts.createAlert(
        alertDef.severity,
        alertDef.title,
        alertDef.description,
        alertDef.source
      );
    }

    // Update time series
    this.trustSeries.add(averageTrust, timestamp);
    this.tpsSeries.add(tps, timestamp);
    this.phiSeries.add(collectivePhi, timestamp);
    this.threatSeries.add(byzantineThreat, timestamp);

    // Compute network health (composite metric)
    const networkHealth = this.computeNetworkHealth(
      averageTrust,
      collectivePhi,
      byzantineThreat
    );

    const metrics: LiveMetrics = {
      timestamp,
      activeAgents: input.trustScores.length,
      totalTrust,
      averageTrust,
      networkHealth,
      tps,
      alerts: this.alerts.countBySeverity(),
      collectivePhi,
      byzantineThreat,
    };

    return metrics;
  }

  /**
   * Compute network health score
   */
  private computeNetworkHealth(
    avgTrust: number,
    phi: number,
    threat: number
  ): number {
    // Weighted combination: trust (0.4), phi (0.3), inverse threat (0.3)
    return (
      0.4 * Math.max(0, Math.min(1, avgTrust)) +
      0.3 * Math.max(0, Math.min(1, phi)) +
      0.3 * (1 - Math.max(0, Math.min(1, threat)))
    );
  }

  /**
   * Get trust time series
   */
  getTrustHistory(windowMs?: number): TimeSeriesPoint[] {
    if (windowMs) {
      return this.trustSeries.range(Date.now() - windowMs);
    }
    return this.trustSeries.getPoints();
  }

  /**
   * Get TPS time series
   */
  getTpsHistory(windowMs?: number): TimeSeriesPoint[] {
    if (windowMs) {
      return this.tpsSeries.range(Date.now() - windowMs);
    }
    return this.tpsSeries.getPoints();
  }
}
