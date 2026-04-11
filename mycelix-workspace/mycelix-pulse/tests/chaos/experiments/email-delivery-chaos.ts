// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Delivery Chaos Experiments
 *
 * Tests resilience of the email delivery pipeline under various failure conditions.
 */

import { ChaosExperiment } from '../chaos-framework';

export const emailDeliveryExperiments: ChaosExperiment[] = [
  {
    id: 'email-001',
    name: 'SMTP Relay Latency',
    description: 'Simulate slow SMTP relay responses to test queue handling',
    targetService: 'mycelix-smtp',
    faultType: 'latency',
    config: {
      latencyMs: 5000,
      latencyJitterMs: 2000,
      targetPercentage: 30,
      targetEndpoints: ['/smtp/send', '/smtp/relay'],
    },
    duration: 600,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'email_queue_length',
          threshold: 10000,
          duration: 120,
        },
        {
          metric: 'email_delivery_failures_total',
          threshold: 500,
          duration: 60,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        { type: 'scale_up', config: { replicas: 5 } },
        {
          type: 'notify',
          config: {
            channel: '#email-ops',
            message: 'SMTP latency experiment triggered rollback - queue backing up',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'queue_length',
          query: 'email_queue_length{queue="outbound"}',
          operator: 'lt',
          threshold: 1000,
        },
        {
          name: 'delivery_rate',
          query: 'rate(email_delivered_total[5m])',
          operator: 'gt',
          threshold: 10,
        },
        {
          name: 'error_rate',
          query: 'rate(email_delivery_failures_total[5m]) / rate(email_delivery_attempts_total[5m])',
          operator: 'lt',
          threshold: 0.01,
        },
      ],
      timeout: 600,
    },
  },

  {
    id: 'email-002',
    name: 'Attachment Processing Failure',
    description: 'Simulate failures in attachment scanning/processing service',
    targetService: 'mycelix-attachments',
    faultType: 'error',
    config: {
      errorRate: 0.5,
      errorCode: 503,
      errorMessage: 'Attachment processing service unavailable',
      targetEndpoints: ['/attachments/scan', '/attachments/process'],
    },
    duration: 300,
    cooldown: 1800,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'attachment_processing_errors_total',
          threshold: 100,
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
          name: 'attachment_processing_success',
          query: 'rate(attachment_processing_success_total[5m]) / rate(attachment_processing_attempts_total[5m])',
          operator: 'gt',
          threshold: 0.95,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'email-003',
    name: 'Email Search Index Unavailable',
    description: 'Simulate Elasticsearch/search index being unavailable',
    targetService: 'elasticsearch',
    faultType: 'dependency_unavailable',
    config: {
      partitionedServices: ['elasticsearch'],
    },
    duration: 180,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: false,
      triggerConditions: [],
      actions: [
        {
          type: 'notify',
          config: {
            channel: '#search-ops',
            message: 'Search index unavailability test in progress',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'search_available',
          query: 'up{job="elasticsearch"}',
          operator: 'eq',
          threshold: 1,
        },
        {
          name: 'search_latency',
          query: 'histogram_quantile(0.99, rate(elasticsearch_search_duration_seconds_bucket[5m]))',
          operator: 'lt',
          threshold: 1,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'email-004',
    name: 'Spam Filter Overload',
    description: 'Simulate spam filter service under extreme load',
    targetService: 'mycelix-spam-filter',
    faultType: 'cpu_stress',
    config: {
      cpuPercent: 95,
    },
    duration: 300,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'spam_filter_queue_length',
          threshold: 5000,
          duration: 120,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        { type: 'scale_up', config: { replicas: 4 } },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'spam_filter_latency',
          query: 'histogram_quantile(0.99, rate(spam_filter_duration_seconds_bucket[5m]))',
          operator: 'lt',
          threshold: 0.5,
        },
        {
          name: 'spam_filter_queue',
          query: 'spam_filter_queue_length',
          operator: 'lt',
          threshold: 500,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'email-005',
    name: 'DKIM/SPF Verification Timeout',
    description: 'Simulate DNS lookups for DKIM/SPF taking too long',
    targetService: 'mycelix-verification',
    faultType: 'latency',
    config: {
      latencyMs: 10000,
      targetPercentage: 50,
      targetEndpoints: ['/verify/dkim', '/verify/spf', '/verify/dmarc'],
    },
    duration: 300,
    cooldown: 1800,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'email_verification_timeout_total',
          threshold: 200,
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
          name: 'verification_success_rate',
          query: 'rate(email_verification_success_total[5m]) / rate(email_verification_attempts_total[5m])',
          operator: 'gt',
          threshold: 0.9,
        },
      ],
      timeout: 300,
    },
  },

  {
    id: 'email-006',
    name: 'Storage Backend Partition',
    description: 'Network partition between API and email storage',
    targetService: 'mycelix-api',
    faultType: 'network_partition',
    config: {
      partitionedServices: ['minio', 's3-storage'],
    },
    duration: 120,
    cooldown: 3600,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'storage_errors_total',
          threshold: 50,
          duration: 30,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        {
          type: 'notify',
          config: {
            channel: '#storage-ops',
            message: 'Storage partition experiment triggered rollback',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'storage_available',
          query: 'up{job="minio"}',
          operator: 'eq',
          threshold: 1,
        },
        {
          name: 'storage_error_rate',
          query: 'rate(storage_errors_total[5m])',
          operator: 'lt',
          threshold: 1,
        },
      ],
      timeout: 180,
    },
  },

  {
    id: 'email-007',
    name: 'Encryption Service Crash',
    description: 'Simulate crash of PGP/encryption service',
    targetService: 'mycelix-crypto',
    faultType: 'process_kill',
    config: {
      targetPercentage: 100,
      signalType: 'SIGKILL',
    },
    duration: 60,
    cooldown: 1800,
    enabled: true,
    rollback: {
      automatic: false,
      triggerConditions: [],
      actions: [],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'crypto_service_up',
          query: 'up{job="mycelix-crypto"}',
          operator: 'eq',
          threshold: 1,
        },
        {
          name: 'crypto_operations_available',
          query: 'rate(crypto_operations_success_total[5m])',
          operator: 'gt',
          threshold: 0,
        },
      ],
      timeout: 120,
    },
  },

  {
    id: 'email-008',
    name: 'Queue Database Disk Full',
    description: 'Simulate queue database running out of disk space',
    targetService: 'redis-queue',
    faultType: 'disk_full',
    config: {
      diskFillPercent: 95,
    },
    duration: 120,
    cooldown: 7200,
    enabled: true,
    rollback: {
      automatic: true,
      triggerConditions: [
        {
          metric: 'redis_rejected_connections_total',
          threshold: 10,
          duration: 30,
        },
      ],
      actions: [
        { type: 'stop_experiment', config: {} },
        {
          type: 'notify',
          config: {
            channel: '#queue-ops',
            message: 'CRITICAL: Queue disk full experiment triggered rollback',
          },
        },
      ],
    },
    steadyStateHypothesis: {
      metrics: [
        {
          name: 'redis_available',
          query: 'up{job="redis-queue"}',
          operator: 'eq',
          threshold: 1,
        },
        {
          name: 'redis_memory_ok',
          query: 'redis_memory_used_bytes / redis_memory_max_bytes',
          operator: 'lt',
          threshold: 0.9,
        },
      ],
      timeout: 180,
    },
  },
];

export default emailDeliveryExperiments;
