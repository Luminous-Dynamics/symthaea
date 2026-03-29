// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Chaos Engineering Framework
 *
 * Systematic resilience testing through controlled failure injection.
 * Based on principles from Netflix Chaos Monkey and Gremlin.
 */

import { EventEmitter } from 'events';

// ============================================================================
// Core Types
// ============================================================================

export interface ChaosExperiment {
  id: string;
  name: string;
  description: string;
  targetService: string;
  faultType: FaultType;
  config: FaultConfig;
  duration: number; // seconds
  cooldown: number; // seconds between runs
  enabled: boolean;
  schedule?: CronSchedule;
  rollback: RollbackStrategy;
  steadyStateHypothesis: SteadyStateHypothesis;
}

export type FaultType =
  | 'latency'
  | 'error'
  | 'resource_exhaustion'
  | 'network_partition'
  | 'dns_failure'
  | 'certificate_expiry'
  | 'disk_full'
  | 'memory_pressure'
  | 'cpu_stress'
  | 'process_kill'
  | 'clock_skew'
  | 'dependency_unavailable';

export interface FaultConfig {
  // Latency injection
  latencyMs?: number;
  latencyJitterMs?: number;

  // Error injection
  errorRate?: number; // 0-1
  errorCode?: number;
  errorMessage?: string;

  // Resource exhaustion
  memoryMb?: number;
  cpuPercent?: number;
  diskFillPercent?: number;
  fileDescriptors?: number;

  // Network
  packetLossPercent?: number;
  bandwidthKbps?: number;
  partitionedServices?: string[];

  // Process
  targetProcess?: string;
  signalType?: 'SIGTERM' | 'SIGKILL' | 'SIGSTOP';

  // Clock
  clockSkewSeconds?: number;

  // Targeting
  targetPercentage?: number; // % of requests/instances affected
  targetPods?: string[];
  targetEndpoints?: string[];
}

export interface SteadyStateHypothesis {
  metrics: MetricCheck[];
  timeout: number;
}

export interface MetricCheck {
  name: string;
  query: string; // Prometheus query
  operator: 'lt' | 'gt' | 'eq' | 'lte' | 'gte';
  threshold: number;
  tolerancePercent?: number;
}

export interface RollbackStrategy {
  automatic: boolean;
  triggerConditions: TriggerCondition[];
  actions: RollbackAction[];
}

export interface TriggerCondition {
  metric: string;
  threshold: number;
  duration: number; // seconds
}

export interface RollbackAction {
  type: 'stop_experiment' | 'scale_up' | 'restart_pods' | 'switch_region' | 'notify';
  config: Record<string, unknown>;
}

export interface CronSchedule {
  expression: string;
  timezone: string;
}

export interface ExperimentResult {
  experimentId: string;
  startTime: Date;
  endTime: Date;
  status: 'success' | 'failure' | 'aborted' | 'rollback';
  steadyStateBefore: MetricSnapshot[];
  steadyStateAfter: MetricSnapshot[];
  impactMetrics: ImpactMetrics;
  logs: ExperimentLog[];
  findings: Finding[];
}

export interface MetricSnapshot {
  name: string;
  value: number;
  timestamp: Date;
}

export interface ImpactMetrics {
  errorRateIncrease: number;
  latencyP50Increase: number;
  latencyP99Increase: number;
  availabilityDrop: number;
  affectedUsers: number;
  recoveryTimeSeconds: number;
}

export interface ExperimentLog {
  timestamp: Date;
  level: 'info' | 'warn' | 'error';
  message: string;
  metadata?: Record<string, unknown>;
}

export interface Finding {
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  recommendation: string;
  relatedMetrics: string[];
}

// ============================================================================
// Chaos Controller
// ============================================================================

export class ChaosController extends EventEmitter {
  private experiments: Map<string, ChaosExperiment> = new Map();
  private runningExperiments: Map<string, AbortController> = new Map();
  private results: ExperimentResult[] = [];
  private prometheusUrl: string;
  private kubernetesClient: KubernetesClient;

  constructor(config: {
    prometheusUrl: string;
    kubeConfig?: string;
  }) {
    super();
    this.prometheusUrl = config.prometheusUrl;
    this.kubernetesClient = new KubernetesClient(config.kubeConfig);
  }

  // Register an experiment
  registerExperiment(experiment: ChaosExperiment): void {
    this.experiments.set(experiment.id, experiment);
    this.emit('experiment:registered', experiment);
  }

  // Run an experiment
  async runExperiment(experimentId: string): Promise<ExperimentResult> {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }

    if (!experiment.enabled) {
      throw new Error(`Experiment ${experimentId} is disabled`);
    }

    if (this.runningExperiments.has(experimentId)) {
      throw new Error(`Experiment ${experimentId} is already running`);
    }

    const abortController = new AbortController();
    this.runningExperiments.set(experimentId, abortController);

    const result: ExperimentResult = {
      experimentId,
      startTime: new Date(),
      endTime: new Date(),
      status: 'success',
      steadyStateBefore: [],
      steadyStateAfter: [],
      impactMetrics: {
        errorRateIncrease: 0,
        latencyP50Increase: 0,
        latencyP99Increase: 0,
        availabilityDrop: 0,
        affectedUsers: 0,
        recoveryTimeSeconds: 0,
      },
      logs: [],
      findings: [],
    };

    try {
      // 1. Verify steady state before
      this.log(result, 'info', 'Verifying steady state hypothesis before experiment');
      result.steadyStateBefore = await this.verifySteadyState(experiment.steadyStateHypothesis);

      if (!this.isSteadyStateValid(result.steadyStateBefore, experiment.steadyStateHypothesis)) {
        throw new Error('System not in steady state before experiment');
      }

      // 2. Inject fault
      this.log(result, 'info', `Injecting fault: ${experiment.faultType}`);
      this.emit('experiment:started', experiment);

      await this.injectFault(experiment, abortController.signal);

      // 3. Wait for duration
      this.log(result, 'info', `Fault active for ${experiment.duration}s`);
      await this.waitWithMonitoring(experiment, result, abortController.signal);

      // 4. Remove fault
      this.log(result, 'info', 'Removing fault');
      await this.removeFault(experiment);

      // 5. Verify steady state after (with recovery time)
      this.log(result, 'info', 'Waiting for recovery and verifying steady state');
      const recoveryStart = Date.now();

      await this.waitForRecovery(experiment, result, abortController.signal);

      result.impactMetrics.recoveryTimeSeconds = (Date.now() - recoveryStart) / 1000;
      result.steadyStateAfter = await this.verifySteadyState(experiment.steadyStateHypothesis);

      // 6. Analyze results
      result.findings = this.analyzeResults(experiment, result);
      result.status = result.findings.some(f => f.severity === 'critical') ? 'failure' : 'success';

    } catch (error) {
      result.status = 'aborted';
      this.log(result, 'error', `Experiment aborted: ${error}`);

      // Ensure fault is removed
      await this.removeFault(experiment).catch(e =>
        this.log(result, 'error', `Failed to remove fault during abort: ${e}`)
      );

      // Execute rollback if needed
      if (experiment.rollback.automatic) {
        await this.executeRollback(experiment, result);
        result.status = 'rollback';
      }
    } finally {
      result.endTime = new Date();
      this.runningExperiments.delete(experimentId);
      this.results.push(result);
      this.emit('experiment:completed', result);
    }

    return result;
  }

  // Stop a running experiment
  async stopExperiment(experimentId: string): Promise<void> {
    const controller = this.runningExperiments.get(experimentId);
    if (controller) {
      controller.abort();
    }
  }

  // Inject various fault types
  private async injectFault(experiment: ChaosExperiment, signal: AbortSignal): Promise<void> {
    const { faultType, config, targetService } = experiment;

    switch (faultType) {
      case 'latency':
        await this.injectLatency(targetService, config, signal);
        break;
      case 'error':
        await this.injectErrors(targetService, config, signal);
        break;
      case 'network_partition':
        await this.injectNetworkPartition(targetService, config, signal);
        break;
      case 'resource_exhaustion':
        await this.injectResourceExhaustion(targetService, config, signal);
        break;
      case 'process_kill':
        await this.killProcess(targetService, config, signal);
        break;
      case 'dns_failure':
        await this.injectDnsFailure(targetService, config, signal);
        break;
      case 'memory_pressure':
        await this.injectMemoryPressure(targetService, config, signal);
        break;
      case 'cpu_stress':
        await this.injectCpuStress(targetService, config, signal);
        break;
      case 'disk_full':
        await this.injectDiskFull(targetService, config, signal);
        break;
      case 'clock_skew':
        await this.injectClockSkew(targetService, config, signal);
        break;
      case 'dependency_unavailable':
        await this.makeDependencyUnavailable(targetService, config, signal);
        break;
      default:
        throw new Error(`Unknown fault type: ${faultType}`);
    }
  }

  private async injectLatency(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Use Istio VirtualService for latency injection
    const virtualService = {
      apiVersion: 'networking.istio.io/v1beta1',
      kind: 'VirtualService',
      metadata: {
        name: `chaos-latency-${service}`,
        namespace: 'mycelix',
        labels: { 'chaos-experiment': 'true' },
      },
      spec: {
        hosts: [service],
        http: [{
          fault: {
            delay: {
              percentage: { value: config.targetPercentage || 100 },
              fixedDelay: `${config.latencyMs}ms`,
            },
          },
          route: [{ destination: { host: service } }],
        }],
      },
    };

    await this.kubernetesClient.apply(virtualService);
  }

  private async injectErrors(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Use Istio VirtualService for error injection
    const virtualService = {
      apiVersion: 'networking.istio.io/v1beta1',
      kind: 'VirtualService',
      metadata: {
        name: `chaos-error-${service}`,
        namespace: 'mycelix',
        labels: { 'chaos-experiment': 'true' },
      },
      spec: {
        hosts: [service],
        http: [{
          fault: {
            abort: {
              percentage: { value: (config.errorRate || 0.5) * 100 },
              httpStatus: config.errorCode || 500,
            },
          },
          route: [{ destination: { host: service } }],
        }],
      },
    };

    await this.kubernetesClient.apply(virtualService);
  }

  private async injectNetworkPartition(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Use NetworkPolicy to partition services
    const networkPolicy = {
      apiVersion: 'networking.k8s.io/v1',
      kind: 'NetworkPolicy',
      metadata: {
        name: `chaos-partition-${service}`,
        namespace: 'mycelix',
        labels: { 'chaos-experiment': 'true' },
      },
      spec: {
        podSelector: {
          matchLabels: { app: service },
        },
        policyTypes: ['Egress'],
        egress: [{
          to: [{
            podSelector: {
              matchExpressions: [{
                key: 'app',
                operator: 'NotIn',
                values: config.partitionedServices || [],
              }],
            },
          }],
        }],
      },
    };

    await this.kubernetesClient.apply(networkPolicy);
  }

  private async injectResourceExhaustion(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Deploy stress container as sidecar
    const stressJob = {
      apiVersion: 'batch/v1',
      kind: 'Job',
      metadata: {
        name: `chaos-stress-${service}-${Date.now()}`,
        namespace: 'mycelix',
        labels: { 'chaos-experiment': 'true' },
      },
      spec: {
        template: {
          spec: {
            containers: [{
              name: 'stress',
              image: 'progrium/stress',
              args: [
                '--cpu', String(config.cpuPercent || 80),
                '--vm', '1',
                '--vm-bytes', `${config.memoryMb || 256}M`,
              ],
            }],
            restartPolicy: 'Never',
            nodeSelector: {
              'mycelix.mail/service': service,
            },
          },
        },
      },
    };

    await this.kubernetesClient.apply(stressJob);
  }

  private async killProcess(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Kill random pod
    const pods = await this.kubernetesClient.getPods('mycelix', { app: service });
    const targetPods = config.targetPods || pods.slice(0, Math.ceil(pods.length * (config.targetPercentage || 50) / 100));

    for (const pod of targetPods) {
      await this.kubernetesClient.deletePod('mycelix', pod);
    }
  }

  private async injectDnsFailure(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Modify CoreDNS config to fail lookups
    const configMap = await this.kubernetesClient.getConfigMap('kube-system', 'coredns');
    const chaosConfig = `
    chaos-${service}.mycelix.svc.cluster.local {
        erratic {
            drop 100
        }
    }
    `;

    configMap.data['Corefile'] = configMap.data['Corefile'] + chaosConfig;
    await this.kubernetesClient.apply(configMap);
  }

  private async injectMemoryPressure(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Use stress-ng for memory pressure
    await this.kubernetesClient.exec('mycelix', service, [
      'stress-ng', '--vm', '1', '--vm-bytes', `${config.memoryMb || 512}M`, '--vm-keep'
    ]);
  }

  private async injectCpuStress(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    await this.kubernetesClient.exec('mycelix', service, [
      'stress-ng', '--cpu', String(config.cpuPercent || 80), '--cpu-load', String(config.cpuPercent || 80)
    ]);
  }

  private async injectDiskFull(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    const fillPercent = config.diskFillPercent || 90;
    await this.kubernetesClient.exec('mycelix', service, [
      'dd', 'if=/dev/zero', `of=/tmp/chaos-fill-${Date.now()}`, 'bs=1M', `count=${fillPercent * 10}`
    ]);
  }

  private async injectClockSkew(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    const skewSeconds = config.clockSkewSeconds || 300;
    await this.kubernetesClient.exec('mycelix', service, [
      'date', '-s', `+${skewSeconds} seconds`
    ]);
  }

  private async makeDependencyUnavailable(service: string, config: FaultConfig, signal: AbortSignal): Promise<void> {
    // Scale dependency to 0
    for (const dep of config.partitionedServices || []) {
      await this.kubernetesClient.scale('mycelix', dep, 0);
    }
  }

  // Remove injected faults
  private async removeFault(experiment: ChaosExperiment): Promise<void> {
    // Delete chaos resources by label
    await this.kubernetesClient.deleteByLabel('mycelix', { 'chaos-experiment': 'true' });

    // Restore any scaled down services
    if (experiment.faultType === 'dependency_unavailable') {
      for (const dep of experiment.config.partitionedServices || []) {
        await this.kubernetesClient.scale('mycelix', dep, 3); // Default replicas
      }
    }
  }

  // Monitoring and verification
  private async verifySteadyState(hypothesis: SteadyStateHypothesis): Promise<MetricSnapshot[]> {
    const snapshots: MetricSnapshot[] = [];

    for (const check of hypothesis.metrics) {
      const value = await this.queryPrometheus(check.query);
      snapshots.push({
        name: check.name,
        value,
        timestamp: new Date(),
      });
    }

    return snapshots;
  }

  private isSteadyStateValid(snapshots: MetricSnapshot[], hypothesis: SteadyStateHypothesis): boolean {
    for (const check of hypothesis.metrics) {
      const snapshot = snapshots.find(s => s.name === check.name);
      if (!snapshot) return false;

      const tolerance = check.tolerancePercent ? check.threshold * (check.tolerancePercent / 100) : 0;

      switch (check.operator) {
        case 'lt': if (!(snapshot.value < check.threshold + tolerance)) return false; break;
        case 'gt': if (!(snapshot.value > check.threshold - tolerance)) return false; break;
        case 'lte': if (!(snapshot.value <= check.threshold + tolerance)) return false; break;
        case 'gte': if (!(snapshot.value >= check.threshold - tolerance)) return false; break;
        case 'eq': if (Math.abs(snapshot.value - check.threshold) > tolerance) return false; break;
      }
    }

    return true;
  }

  private async waitWithMonitoring(
    experiment: ChaosExperiment,
    result: ExperimentResult,
    signal: AbortSignal
  ): Promise<void> {
    const startTime = Date.now();
    const endTime = startTime + experiment.duration * 1000;

    while (Date.now() < endTime && !signal.aborted) {
      // Check rollback conditions
      for (const condition of experiment.rollback.triggerConditions) {
        const value = await this.queryPrometheus(condition.metric);
        if (value > condition.threshold) {
          this.log(result, 'warn', `Rollback condition triggered: ${condition.metric} = ${value}`);
          if (experiment.rollback.automatic) {
            throw new Error(`Rollback triggered: ${condition.metric} exceeded threshold`);
          }
        }
      }

      // Collect impact metrics
      const [errorRate, latencyP50, latencyP99] = await Promise.all([
        this.queryPrometheus(`rate(http_requests_total{service="${experiment.targetService}",status=~"5.."}[1m])`),
        this.queryPrometheus(`histogram_quantile(0.5, rate(http_request_duration_seconds_bucket{service="${experiment.targetService}"}[1m]))`),
        this.queryPrometheus(`histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{service="${experiment.targetService}"}[1m]))`),
      ]);

      result.impactMetrics.errorRateIncrease = Math.max(result.impactMetrics.errorRateIncrease, errorRate);
      result.impactMetrics.latencyP50Increase = Math.max(result.impactMetrics.latencyP50Increase, latencyP50);
      result.impactMetrics.latencyP99Increase = Math.max(result.impactMetrics.latencyP99Increase, latencyP99);

      await this.sleep(5000); // Check every 5 seconds
    }
  }

  private async waitForRecovery(
    experiment: ChaosExperiment,
    result: ExperimentResult,
    signal: AbortSignal
  ): Promise<void> {
    const maxWait = experiment.steadyStateHypothesis.timeout * 1000;
    const startTime = Date.now();

    while (Date.now() - startTime < maxWait && !signal.aborted) {
      const snapshots = await this.verifySteadyState(experiment.steadyStateHypothesis);
      if (this.isSteadyStateValid(snapshots, experiment.steadyStateHypothesis)) {
        this.log(result, 'info', 'System recovered to steady state');
        return;
      }
      await this.sleep(5000);
    }

    this.log(result, 'warn', 'System did not fully recover within timeout');
  }

  private async executeRollback(experiment: ChaosExperiment, result: ExperimentResult): Promise<void> {
    this.log(result, 'info', 'Executing rollback actions');

    for (const action of experiment.rollback.actions) {
      try {
        switch (action.type) {
          case 'stop_experiment':
            // Already done by abort
            break;
          case 'scale_up':
            await this.kubernetesClient.scale('mycelix', experiment.targetService, action.config.replicas as number);
            break;
          case 'restart_pods':
            await this.kubernetesClient.rolloutRestart('mycelix', experiment.targetService);
            break;
          case 'notify':
            await this.sendNotification(action.config as NotificationConfig);
            break;
        }
      } catch (error) {
        this.log(result, 'error', `Rollback action failed: ${action.type} - ${error}`);
      }
    }
  }

  // Analysis
  private analyzeResults(experiment: ChaosExperiment, result: ExperimentResult): Finding[] {
    const findings: Finding[] = [];

    // Check recovery time
    if (result.impactMetrics.recoveryTimeSeconds > 60) {
      findings.push({
        severity: result.impactMetrics.recoveryTimeSeconds > 300 ? 'critical' : 'high',
        title: 'Slow Recovery Time',
        description: `Service took ${result.impactMetrics.recoveryTimeSeconds}s to recover`,
        recommendation: 'Consider implementing faster health checks and auto-scaling',
        relatedMetrics: ['recovery_time'],
      });
    }

    // Check error rate impact
    if (result.impactMetrics.errorRateIncrease > 0.1) {
      findings.push({
        severity: result.impactMetrics.errorRateIncrease > 0.5 ? 'critical' : 'high',
        title: 'High Error Rate During Fault',
        description: `Error rate increased to ${(result.impactMetrics.errorRateIncrease * 100).toFixed(1)}%`,
        recommendation: 'Implement circuit breakers and graceful degradation',
        relatedMetrics: ['error_rate'],
      });
    }

    // Check latency impact
    if (result.impactMetrics.latencyP99Increase > 5) {
      findings.push({
        severity: 'medium',
        title: 'Significant Latency Increase',
        description: `P99 latency increased to ${result.impactMetrics.latencyP99Increase.toFixed(2)}s`,
        recommendation: 'Consider adding timeouts and async processing',
        relatedMetrics: ['latency_p99'],
      });
    }

    // Check if steady state was restored
    if (!this.isSteadyStateValid(result.steadyStateAfter, experiment.steadyStateHypothesis)) {
      findings.push({
        severity: 'critical',
        title: 'Steady State Not Restored',
        description: 'System did not return to steady state after fault removal',
        recommendation: 'Review recovery procedures and health check configurations',
        relatedMetrics: experiment.steadyStateHypothesis.metrics.map(m => m.name),
      });
    }

    return findings;
  }

  // Helpers
  private async queryPrometheus(query: string): Promise<number> {
    const response = await fetch(`${this.prometheusUrl}/api/v1/query?query=${encodeURIComponent(query)}`);
    const data = await response.json();
    return parseFloat(data.data?.result?.[0]?.value?.[1] || '0');
  }

  private async sendNotification(config: NotificationConfig): Promise<void> {
    // Implement notification (Slack, PagerDuty, etc.)
  }

  private log(result: ExperimentResult, level: 'info' | 'warn' | 'error', message: string): void {
    const log: ExperimentLog = { timestamp: new Date(), level, message };
    result.logs.push(log);
    this.emit('log', log);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Get results
  getResults(): ExperimentResult[] {
    return [...this.results];
  }

  getExperiment(id: string): ChaosExperiment | undefined {
    return this.experiments.get(id);
  }

  listExperiments(): ChaosExperiment[] {
    return Array.from(this.experiments.values());
  }
}

// ============================================================================
// Kubernetes Client (simplified)
// ============================================================================

interface NotificationConfig {
  channel: string;
  message: string;
}

class KubernetesClient {
  constructor(kubeConfig?: string) {
    // Initialize k8s client
  }

  async apply(resource: unknown): Promise<void> {
    // Apply resource to cluster
  }

  async getPods(namespace: string, labels: Record<string, string>): Promise<string[]> {
    return [];
  }

  async deletePod(namespace: string, name: string): Promise<void> {}

  async deleteByLabel(namespace: string, labels: Record<string, string>): Promise<void> {}

  async scale(namespace: string, deployment: string, replicas: number): Promise<void> {}

  async rolloutRestart(namespace: string, deployment: string): Promise<void> {}

  async getConfigMap(namespace: string, name: string): Promise<{ data: Record<string, string> }> {
    return { data: {} };
  }

  async exec(namespace: string, pod: string, command: string[]): Promise<void> {}
}

// ============================================================================
// Pre-defined Experiments
// ============================================================================

export const predefinedExperiments: Omit<ChaosExperiment, 'id'>[] = [
  {
    name: 'API Server Latency',
    description: 'Inject 500ms latency into API server responses',
    targetService: 'mycelix-api',
    faultType: 'latency',
    config: {
      latencyMs: 500,
      latencyJitterMs: 100,
      targetPercentage: 50,
    },
    duration: 300,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        { metric: 'http_requests_total{status="500"}', threshold: 100, duration: 60 },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        { type: 'notify', config: { channel: '#chaos-alerts', message: 'Latency experiment triggered rollback' } },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        { name: 'error_rate', query: 'rate(http_requests_total{status=~"5.."}[5m])', operator: 'lt', threshold: 0.01 },
        { name: 'latency_p99', query: 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))', operator: 'lt', threshold: 1 },
      ],
      timeout: 300,
    },
  },
  {
    name: 'Database Connection Loss',
    description: 'Simulate database becoming unavailable',
    targetService: 'mycelix-api',
    faultType: 'dependency_unavailable',
    config: {
      partitionedServices: ['postgres', 'redis'],
    },
    duration: 120,
    cooldown: 7200,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        { metric: 'http_requests_total{status="503"}', threshold: 500, duration: 30 },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        { name: 'availability', query: 'avg(up{job="mycelix-api"})', operator: 'eq', threshold: 1 },
      ],
      timeout: 180,
    },
  },
  {
    name: 'Pod Failure Recovery',
    description: 'Kill 50% of API pods and measure recovery',
    targetService: 'mycelix-api',
    faultType: 'process_kill',
    config: {
      targetPercentage: 50,
      signalType: 'SIGKILL',
    },
    duration: 60,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: false,
      triggerConditions: [],
      actions: [],
    },
    steadyStateHypothesis: {
      metrics: [
        { name: 'pod_count', query: 'count(up{job="mycelix-api"})', operator: 'gte', threshold: 3 },
        { name: 'ready_pods', query: 'sum(kube_pod_status_ready{pod=~"mycelix-api.*"})', operator: 'gte', threshold: 3 },
      ],
      timeout: 300,
    },
  },
  {
    name: 'Memory Pressure',
    description: 'Induce memory pressure to test OOM handling',
    targetService: 'mycelix-worker',
    faultType: 'memory_pressure',
    config: {
      memoryMb: 1024,
    },
    duration: 180,
    cooldown: 7200,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        { metric: 'container_memory_usage_bytes{pod=~"mycelix-worker.*"}', threshold: 2147483648, duration: 60 },
      ],
      actions: [
        { type: 'restart_pods', config: {} },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        { name: 'memory_usage', query: 'container_memory_usage_bytes{pod=~"mycelix-worker.*"} / container_spec_memory_limit_bytes{pod=~"mycelix-worker.*"}', operator: 'lt', threshold: 0.8 },
      ],
      timeout: 300,
    },
  },
  {
    name: 'Network Partition - Trust Service',
    description: 'Isolate trust service from other services',
    targetService: 'mycelix-trust',
    faultType: 'network_partition',
    config: {
      partitionedServices: ['mycelix-api', 'mycelix-worker'],
    },
    duration: 120,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        { metric: 'trust_attestation_errors_total', threshold: 100, duration: 30 },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        { name: 'trust_health', query: 'up{job="mycelix-trust"}', operator: 'eq', threshold: 1 },
        { name: 'trust_latency', query: 'histogram_quantile(0.99, rate(trust_operation_duration_seconds_bucket[5m]))', operator: 'lt', threshold: 2 },
      ],
      timeout: 180,
    },
  },
];

export default ChaosController;
