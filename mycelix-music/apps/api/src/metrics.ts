// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import client from 'prom-client';

const register = new client.Registry();
client.collectDefaultMetrics({ register });

const apiRequestHistogram = new client.Histogram({
  name: 'api_request_duration_seconds',
  help: 'Duration of API requests in seconds',
  labelNames: ['method', 'path', 'status'],
  buckets: [0.05, 0.1, 0.25, 0.5, 1, 2, 5],
});

const indexerLagGauge = new client.Gauge({
  name: 'indexer_lag_blocks',
  help: 'Current indexer lag in blocks',
});
const uploadQuotaBlocks = new client.Counter({
  name: 'upload_quota_blocks',
  help: 'Number of upload requests blocked by quota',
  labelNames: ['kind'],
});
const poisonCounter = new client.Counter({
  name: 'indexer_poison_total',
  help: 'Total poison events recorded by indexer',
});
const retryQueueGauge = new client.Gauge({
  name: 'indexer_retry_queue_length',
  help: 'Current retry queue length',
});

register.registerMetric(apiRequestHistogram);
register.registerMetric(indexerLagGauge);
register.registerMetric(uploadQuotaBlocks);
register.registerMetric(poisonCounter);
register.registerMetric(retryQueueGauge);

export function recordRequest(labels: { method: string; path: string; status: number }, durationMs: number) {
  apiRequestHistogram.labels(labels.method, labels.path, String(labels.status)).observe(durationMs / 1000);
}

export function setIndexerLag(blocks: number) {
  indexerLagGauge.set(blocks);
}

export function metricsHandler(_req: any, res: any) {
  res.setHeader('Content-Type', register.contentType);
  register.metrics().then((data: string) => res.end(data));
}

export function recordUploadBlock(kind: string) {
  uploadQuotaBlocks.labels(kind).inc();
}

export function recordPoison() {
  poisonCounter.inc();
}

export function setRetryQueueLength(n: number) {
  retryQueueGauge.set(n);
}
