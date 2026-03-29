// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Network Chaos Experiments
 *
 * Tests resilience of the web-of-trust system under failure conditions.
 */

import { ChaosExperiment } from '../chaos-framework';

export const trustNetworkExperiments: ChaosExperiment[] = [
  {
    id: 'trust-001',
    name: 'Trust Graph Database Latency',
    description: 'Simulate slow graph database queries affecting trust calculations',
    targetService: 'neo4j',
    faultType: 'latency',
    config: {
      latencyMs: 3000,
      latencyJitterMs: 1000,
      targetPercentage: 40,
    },
    duration: 300,
    cooldown: 1800,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'trust_calculation_timeout_total',
          threshold: 50,
          duration: 60,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'trust_calc_latency',
          query: 'histogram_quantile(0.99, rate(trust_calculation_duration_seconds_bucket[5m]))',
          operator: 'lt',
          threshold: 2,
        },
        {
          name: 'trust_cache_hit_rate',
          query: 'rate(trust_cache_hits_total[5m]) / (rate(trust_cache_hits_total[5m]) + rate(trust_cache_misses_total[5m]))',
          operator: 'gt',
          threshold: 0.8,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'trust-002',
    name: 'Attestation Service Partition',
    description: 'Network partition preventing attestation propagation',
    targetService: 'mycelix-attestation',
    faultType: 'network_partition',
    config: {
      partitionedServices: ['mycelix-trust', 'mycelix-api'],
    },
    duration: 180,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'attestation_sync_lag_seconds',
          threshold: 300,
          duration: 60,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        {
          type: 'notify',
          config: {
            channel: '#trust-ops',
            message: 'Attestation partition experiment - sync lag detected',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'attestation_sync',
          query: 'attestation_sync_lag_seconds',
          operator: 'lt',
          threshold: 30,
        },
        {
          name: 'attestation_service_up',
          query: 'up{job="mycelix-attestation"}',
          operator: 'eq',
          threshold: 1,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'trust-003',
    name: 'Trust Cache Invalidation Storm',
    description: 'Simulate cache invalidation causing thundering herd',
    targetService: 'redis-trust-cache',
    faultType: 'process_kill',
    config: {
      targetPercentage: 100,
      signalType: 'SIGTERM',
    },
    duration: 30,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'neo4j_active_connections',
          threshold: 1000,
          duration: 30,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        { type: 'scale_up', config: { replicas: 3 } },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'cache_available',
          query: 'up{job="redis-trust-cache"}',
          operator: 'eq',
          threshold: 1,
        },
        {
          name: 'graph_db_connections',
          query: 'neo4j_active_connections',
          operator: 'lt',
          threshold: 100,
        },
      ],
      timeout: 180,
    },
  },

  {
    id: 'trust-004',
    name: 'Key Server Unavailable',
    description: 'Simulate public key server being unavailable',
    targetService: 'keyserver',
    faultType: 'dependency_unavailable',
    config: {
      partitionedServices: ['keyserver', 'hkp-proxy'],
    },
    duration: 300,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: false,
      triggerConditions: [],
      actions: [],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'key_lookup_success',
          query: 'rate(key_lookup_success_total[5m]) / rate(key_lookup_attempts_total[5m])',
          operator: 'gt',
          threshold: 0.9,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'trust-005',
    name: 'Trust Propagation Delay',
    description: 'Simulate delays in trust score propagation across the network',
    targetService: 'mycelix-trust-propagation',
    faultType: 'latency',
    config: {
      latencyMs: 30000,
      targetPercentage: 80,
    },
    duration: 600,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'trust_score_staleness_seconds',
          threshold: 600,
          duration: 300,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'propagation_lag',
          query: 'trust_propagation_lag_seconds',
          operator: 'lt',
          threshold: 60,
        },
        {
          name: 'eventual_consistency',
          query: 'trust_score_consistency_ratio',
          operator: 'gt',
          threshold: 0.95,
        },
      ],
      timeout: 900,
    },
  },

  {
    id: 'trust-006',
    name: 'Revocation Processing Failure',
    description: 'Simulate failures in processing trust revocations',
    targetService: 'mycelix-revocation',
    faultType: 'error',
    config: {
      errorRate: 0.7,
      errorCode: 500,
      targetEndpoints: ['/revoke', '/revocation/process'],
    },
    duration: 180,
    cooldown: 1800,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'revocation_queue_length',
          threshold: 100,
          duration: 60,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        {
          type: 'notify',
          config: {
            channel: '#security-ops',
            message: 'CRITICAL: Revocation processing failure experiment triggered',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'revocation_success',
          query: 'rate(revocation_success_total[5m]) / rate(revocation_attempts_total[5m])',
          operator: 'gt',
          threshold: 0.99,
        },
        {
          name: 'revocation_latency',
          query: 'histogram_quantile(0.99, rate(revocation_processing_duration_seconds_bucket[5m]))',
          operator: 'lt',
          threshold: 5,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'trust-007',
    name: 'Byzantine Trust Node',
    description: 'Simulate a node returning incorrect trust scores',
    targetService: 'mycelix-trust',
    faultType: 'error',
    config: {
      errorRate: 0.2,
      errorCode: 200, // Returns 200 but with wrong data
      targetEndpoints: ['/trust/score', '/trust/path'],
    },
    duration: 300,
    cooldown: 7200,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'trust_score_divergence',
          threshold: 0.1,
          duration: 60,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        {
          type: 'notify',
          config: {
            channel: '#security-ops',
            message: 'Byzantine node simulation detected score divergence',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'trust_consistency',
          query: 'trust_score_consistency_check_success_ratio',
          operator: 'gt',
          threshold: 0.99,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'trust-008',
    name: 'Graph Traversal Memory Pressure',
    description: 'Deep trust path calculations causing memory pressure',
    targetService: 'mycelix-trust',
    faultType: 'memory_pressure',
    config: {
      memoryMb: 2048,
    },
    duration: 180,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'container_memory_usage_bytes{pod=~"mycelix-trust.*"}',
          threshold: 3221225472, // 3GB
          duration: 60,
        },
      ],
      actions: [
        { type: 'restart_pods', config: {} },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'memory_usage',
          query: 'container_memory_usage_bytes{pod=~"mycelix-trust.*"} / container_spec_memory_limit_bytes{pod=~"mycelix-trust.*"}',
          operator: 'lt',
          threshold: 0.8,
        },
        {
          name: 'gc_pause',
          query: 'rate(go_gc_pause_seconds_total{job="mycelix-trust"}[5m])',
          operator: 'lt',
          threshold: 0.1,
        },
      ],
      timeout: 300,
    },
  },
];

export default trustNetworkExperiments;
