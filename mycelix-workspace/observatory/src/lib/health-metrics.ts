// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Observatory Health Metrics
 *
 * Real-time health monitoring for the Mycelix ecosystem.
 * Provides dashboard-ready metrics for all services.
 */

import { writable, derived, type Readable, type Writable } from 'svelte/store';

// ============================================================================
// Types
// ============================================================================

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

export interface ServiceHealth {
  id: string;
  name: string;
  status: HealthStatus;
  latencyMs: number;
  lastCheck: number;
  uptime: number; // seconds
  errorCount: number;
  metadata?: Record<string, unknown>;
}

export interface HappHealth {
  happId: string;
  name: string;
  status: HealthStatus;
  zomeCount: number;
  entryCount: number;
  linkCount: number;
  agentCount: number;
  lastActivity: number;
  metrics: {
    callsPerSecond: number;
    avgLatencyMs: number;
    errorRate: number;
  };
}

export interface ConductorHealth {
  id: string;
  url: string;
  status: HealthStatus;
  connectedAgents: number;
  installedApps: number;
  memoryUsageMb: number;
  cpuPercent: number;
  uptime: number;
}

export interface NetworkHealth {
  peerCount: number;
  bootstrapStatus: HealthStatus;
  signalServerStatus: HealthStatus;
  avgLatencyMs: number;
  messageRate: number;
}

export interface MATLHealth {
  byzantineTolerance: number;
  avgTrustScore: number;
  reputationUpdatesPerSecond: number;
  byzantineDetections: number;
}

export interface FLHealth {
  activeRounds: number;
  participantCount: number;
  gradientRate: number;
  avgAggregationTimeMs: number;
  byzantineRejections: number;
}

export interface EcosystemHealth {
  overall: HealthStatus;
  services: ServiceHealth[];
  happs: HappHealth[];
  conductors: ConductorHealth[];
  network: NetworkHealth;
  matl: MATLHealth;
  fl: FLHealth;
  timestamp: number;
}

export interface HealthAlert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  service?: string;
  timestamp: number;
  acknowledged: boolean;
}

// ============================================================================
// Default Values
// ============================================================================

const defaultNetworkHealth: NetworkHealth = {
  peerCount: 0,
  bootstrapStatus: 'unknown',
  signalServerStatus: 'unknown',
  avgLatencyMs: 0,
  messageRate: 0,
};

const defaultMATLHealth: MATLHealth = {
  byzantineTolerance: 0.34,
  avgTrustScore: 0,
  reputationUpdatesPerSecond: 0,
  byzantineDetections: 0,
};

const defaultFLHealth: FLHealth = {
  activeRounds: 0,
  participantCount: 0,
  gradientRate: 0,
  avgAggregationTimeMs: 0,
  byzantineRejections: 0,
};

const defaultEcosystemHealth: EcosystemHealth = {
  overall: 'unknown',
  services: [],
  happs: [],
  conductors: [],
  network: defaultNetworkHealth,
  matl: defaultMATLHealth,
  fl: defaultFLHealth,
  timestamp: Date.now(),
};

// ============================================================================
// Stores
// ============================================================================

/** Ecosystem health state */
export const ecosystemHealth: Writable<EcosystemHealth> = writable(defaultEcosystemHealth);

/** Active alerts */
export const healthAlerts: Writable<HealthAlert[]> = writable([]);

/** Health check interval (ms) */
export const healthCheckInterval: Writable<number> = writable(5000);

/** Is health monitoring active */
export const isMonitoring: Writable<boolean> = writable(false);

// ============================================================================
// Derived Stores
// ============================================================================

/** Overall system status */
export const overallStatus: Readable<HealthStatus> = derived(
  ecosystemHealth,
  ($health) => $health.overall
);

/** Count of unhealthy services */
export const unhealthyCount: Readable<number> = derived(
  ecosystemHealth,
  ($health) =>
    $health.services.filter((s) => s.status === 'unhealthy').length +
    $health.conductors.filter((c) => c.status === 'unhealthy').length +
    $health.happs.filter((h) => h.status === 'unhealthy').length
);

/** Active (unacknowledged) alerts */
export const activeAlerts: Readable<HealthAlert[]> = derived(
  healthAlerts,
  ($alerts) => $alerts.filter((a) => !a.acknowledged)
);

/** Critical alerts count */
export const criticalAlertCount: Readable<number> = derived(
  activeAlerts,
  ($alerts) => $alerts.filter((a) => a.severity === 'critical' || a.severity === 'error').length
);

/** Average latency across services */
export const avgLatency: Readable<number> = derived(ecosystemHealth, ($health) => {
  const latencies = $health.services.map((s) => s.latencyMs);
  if (latencies.length === 0) return 0;
  return latencies.reduce((a, b) => a + b, 0) / latencies.length;
});

// ============================================================================
// Actions
// ============================================================================

/**
 * Update ecosystem health
 */
export function updateHealth(health: Partial<EcosystemHealth>): void {
  ecosystemHealth.update((current) => ({
    ...current,
    ...health,
    overall: calculateOverallStatus({ ...current, ...health }),
    timestamp: Date.now(),
  }));
}

/**
 * Update a specific service's health
 */
export function updateServiceHealth(serviceId: string, health: Partial<ServiceHealth>): void {
  ecosystemHealth.update((current) => {
    const services = current.services.map((s) =>
      s.id === serviceId ? { ...s, ...health } : s
    );

    // Add new service if not exists
    if (!services.find((s) => s.id === serviceId)) {
      services.push({
        id: serviceId,
        name: health.name ?? serviceId,
        status: health.status ?? 'unknown',
        latencyMs: health.latencyMs ?? 0,
        lastCheck: Date.now(),
        uptime: health.uptime ?? 0,
        errorCount: health.errorCount ?? 0,
        ...health,
      });
    }

    return {
      ...current,
      services,
      overall: calculateOverallStatus({ ...current, services }),
      timestamp: Date.now(),
    };
  });
}

/**
 * Update a specific hApp's health
 */
export function updateHappHealth(happId: string, health: Partial<HappHealth>): void {
  ecosystemHealth.update((current) => {
    const happs = current.happs.map((h) =>
      h.happId === happId ? { ...h, ...health } : h
    );

    if (!happs.find((h) => h.happId === happId)) {
      happs.push({
        happId,
        name: health.name ?? happId,
        status: health.status ?? 'unknown',
        zomeCount: health.zomeCount ?? 0,
        entryCount: health.entryCount ?? 0,
        linkCount: health.linkCount ?? 0,
        agentCount: health.agentCount ?? 0,
        lastActivity: Date.now(),
        metrics: health.metrics ?? {
          callsPerSecond: 0,
          avgLatencyMs: 0,
          errorRate: 0,
        },
        ...health,
      });
    }

    return {
      ...current,
      happs,
      overall: calculateOverallStatus({ ...current, happs }),
      timestamp: Date.now(),
    };
  });
}

/**
 * Add a health alert
 */
export function addAlert(alert: Omit<HealthAlert, 'id' | 'timestamp' | 'acknowledged'>): void {
  const newAlert: HealthAlert = {
    ...alert,
    id: `alert_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
    timestamp: Date.now(),
    acknowledged: false,
  };

  healthAlerts.update((alerts) => [newAlert, ...alerts.slice(0, 99)]); // Keep last 100
}

/**
 * Acknowledge an alert
 */
export function acknowledgeAlert(alertId: string): void {
  healthAlerts.update((alerts) =>
    alerts.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a))
  );
}

/**
 * Clear all acknowledged alerts
 */
export function clearAcknowledgedAlerts(): void {
  healthAlerts.update((alerts) => alerts.filter((a) => !a.acknowledged));
}

// ============================================================================
// Health Check Functions
// ============================================================================

/**
 * Perform a health check on a URL
 */
async function checkUrl(url: string, timeoutMs: number = 5000): Promise<{ healthy: boolean; latencyMs: number }> {
  const start = Date.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    return {
      healthy: response.ok,
      latencyMs: Date.now() - start,
    };
  } catch {
    return {
      healthy: false,
      latencyMs: Date.now() - start,
    };
  }
}

/**
 * Check conductor health via admin API
 */
export async function checkConductorHealth(
  adminUrl: string
): Promise<ConductorHealth> {
  const { healthy, latencyMs } = await checkUrl(adminUrl);

  return {
    id: adminUrl,
    url: adminUrl,
    status: healthy ? 'healthy' : 'unhealthy',
    connectedAgents: 0, // Would need actual conductor query
    installedApps: 0,
    memoryUsageMb: 0,
    cpuPercent: 0,
    uptime: 0,
  };
}

/**
 * Calculate overall status from components
 */
function calculateOverallStatus(health: EcosystemHealth): HealthStatus {
  const allStatuses: HealthStatus[] = [
    ...health.services.map((s) => s.status),
    ...health.conductors.map((c) => c.status),
    ...health.happs.map((h) => h.status),
    health.network.bootstrapStatus,
    health.network.signalServerStatus,
  ];

  if (allStatuses.length === 0) return 'unknown';

  const unhealthyCount = allStatuses.filter((s) => s === 'unhealthy').length;
  const degradedCount = allStatuses.filter((s) => s === 'degraded').length;
  const unknownCount = allStatuses.filter((s) => s === 'unknown').length;

  if (unhealthyCount > allStatuses.length * 0.5) return 'unhealthy';
  if (unhealthyCount > 0 || degradedCount > allStatuses.length * 0.3) return 'degraded';
  if (unknownCount === allStatuses.length) return 'unknown';

  return 'healthy';
}

// ============================================================================
// Monitoring Control
// ============================================================================

let monitoringInterval: ReturnType<typeof setInterval> | null = null;

/**
 * Start health monitoring
 */
export function startMonitoring(options: {
  conductorUrls?: string[];
  checkIntervalMs?: number;
} = {}): void {
  stopMonitoring();

  const intervalMs = options.checkIntervalMs ?? 5000;
  healthCheckInterval.set(intervalMs);
  isMonitoring.set(true);

  // Initial check
  performHealthChecks(options.conductorUrls ?? []);

  // Periodic checks
  monitoringInterval = setInterval(() => {
    performHealthChecks(options.conductorUrls ?? []);
  }, intervalMs);
}

/**
 * Stop health monitoring
 */
export function stopMonitoring(): void {
  if (monitoringInterval) {
    clearInterval(monitoringInterval);
    monitoringInterval = null;
  }
  isMonitoring.set(false);
}

/**
 * Perform all health checks
 */
async function performHealthChecks(conductorUrls: string[]): Promise<void> {
  const conductorHealthPromises = conductorUrls.map(checkConductorHealth);

  try {
    const conductors = await Promise.all(conductorHealthPromises);

    updateHealth({ conductors });

    // Generate alerts for unhealthy services
    for (const conductor of conductors) {
      if (conductor.status === 'unhealthy') {
        addAlert({
          severity: 'error',
          title: 'Conductor Unhealthy',
          message: `Conductor at ${conductor.url} is not responding`,
          service: conductor.id,
        });
      }
    }
  } catch (error) {
    console.error('Health check failed:', error);
  }
}

// ============================================================================
// Export Summary Stats
// ============================================================================

/**
 * Get a summary of ecosystem health for display
 */
export function getHealthSummary(): Readable<{
  status: HealthStatus;
  healthyServices: number;
  totalServices: number;
  activeAlerts: number;
  avgLatencyMs: number;
  uptime: string;
}> {
  return derived(
    [ecosystemHealth, activeAlerts, avgLatency],
    ([$health, $alerts, $latency]) => {
      const allItems = [
        ...$health.services,
        ...$health.conductors,
        ...$health.happs,
      ];
      const healthyCount = allItems.filter((i) => i.status === 'healthy').length;

      // Calculate uptime from oldest service
      const oldestUptime = Math.max(...$health.services.map((s) => s.uptime), 0);
      const days = Math.floor(oldestUptime / 86400);
      const hours = Math.floor((oldestUptime % 86400) / 3600);
      const uptime = days > 0 ? `${days}d ${hours}h` : `${hours}h`;

      return {
        status: $health.overall,
        healthyServices: healthyCount,
        totalServices: allItems.length,
        activeAlerts: $alerts.length,
        avgLatencyMs: Math.round($latency),
        uptime,
      };
    }
  );
}
