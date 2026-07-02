// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * K6 Load Test - Normal Production Load
 *
 * Purpose: Simulate normal production traffic
 * Duration: 10 minutes
 * VUs: 0 → 100 → 0
 * Target: 50-100 req/s sustained
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const eventsCreated = new Counter('events_created');
const claimsVerified = new Counter('claims_verified');
const lineageQueries = new Counter('lineage_queries');
const errorRate = new Rate('errors');
const eventCreationTime = new Trend('event_creation_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 VUs
    { duration: '6m', target: 100 },   // Stay at 100 VUs
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<100', 'p(99)<200'],  // 95% < 100ms, 99% < 200ms
    'http_req_failed': ['rate<0.01'],                 // < 1% error rate
    'errors': ['rate<0.01'],
    'event_creation_time': ['p(95)<150'],             // Event creation < 150ms
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

// Event types distribution (realistic mix)
const EVENT_TYPES = [
  { type: 'PRODUCED', weight: 0.3 },
  { type: 'SHIPPED', weight: 0.25 },
  { type: 'RECEIVED', weight: 0.25 },
  { type: 'TRANSFORMED', weight: 0.15 },
  { type: 'CERTIFIED', weight: 0.05 },
];

function selectEventType() {
  const rand = Math.random();
  let cumulative = 0;
  for (const evt of EVENT_TYPES) {
    cumulative += evt.weight;
    if (rand < cumulative) return evt.type;
  }
  return 'PRODUCED';
}

const createEvent = (eventType, batchId, prevBatchIds = null) => {
  const event = {
    '@context': ['https://www.w3.org/2018/credentials/v1'],
    'type': ['VerifiableCredential'],
    'issuer': `did:mycelix:org:load-test-vu${__VU}`,
    'issuanceDate': new Date().toISOString(),
    'credentialSubject': {
      'eventType': eventType,
      'productId': `PRODUCT-${batchId}`,
      'batchId': batchId,
      'quantity': Math.random() * 1000 + 100,
      'unit': 'kg',
      'facility': {
        'id': `FACILITY-VU${__VU}`,
        'name': `Load Test Facility ${__VU}`
      },
      'timestamp': new Date().toISOString()
    }
  };

  // Add previous batch IDs for transformation events
  if (eventType === 'TRANSFORMED' && prevBatchIds) {
    event.credentialSubject.prevBatchIds = prevBatchIds;
  }

  return event;
};

export default function () {
  const iteration = __ITER;
  const vu = __VU;

  // Scenario: Create events with realistic workflow

  // 70% of requests: Create new events
  if (Math.random() < 0.7) {
    const eventType = selectEventType();
    const batchId = `LOAD-BATCH-VU${vu}-${iteration}`;

    const start = Date.now();
    const res = http.post(
      `${BASE_URL}/v1/events`,
      JSON.stringify(createEvent(eventType, batchId)),
      {
        headers: { 'Content-Type': 'application/json' },
        tags: { operation: 'create_event' },
      }
    );

    const success = check(res, {
      'create event is 201': (r) => r.status === 201,
      'has claim_id': (r) => r.json('claim_id') !== undefined,
      'has lineage_hash': (r) => r.json('lineage_hash') !== undefined,
    });

    if (success) {
      eventsCreated.add(1);
      eventCreationTime.add(Date.now() - start);
    } else {
      errorRate.add(1);
      console.error(`Event creation failed: ${res.status}`);
    }
  }

  // 20% of requests: Get existing claims
  else if (Math.random() < 0.9) {
    // Generate a claim ID from a previous iteration
    const oldIter = Math.max(0, iteration - Math.floor(Math.random() * 10));
    const claimBatch = `LOAD-BATCH-VU${vu}-${oldIter}`;

    const res = http.get(
      `${BASE_URL}/v1/batches/${claimBatch}/lineage`,
      {
        tags: { operation: 'get_lineage' },
      }
    );

    if (check(res, {
      'get lineage is 200 or 404': (r) => r.status === 200 || r.status === 404,
    })) {
      if (res.status === 200) {
        lineageQueries.add(1);
      }
    } else {
      errorRate.add(1);
    }
  }

  // 10% of requests: Health check
  else {
    const res = http.get(`${BASE_URL}/health`, {
      tags: { operation: 'health_check' },
    });

    check(res, {
      'health check is 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  // Think time: 0.5-2 seconds between requests
  sleep(Math.random() * 1.5 + 0.5);
}

export function handleSummary(data) {
  const summary = {
    stdout: generateTextSummary(data),
  };

  // Optionally save to file
  if (__ENV.SAVE_RESULTS) {
    summary['results.json'] = JSON.stringify(data);
  }

  return summary;
}

function generateTextSummary(data) {
  const metrics = data.metrics;

  let text = '\n';
  text += '╔═══════════════════════════════════════════════════════╗\n';
  text += '║         LOAD TEST RESULTS                             ║\n';
  text += '╠═══════════════════════════════════════════════════════╣\n';
  text += `║ Test Duration:     ${(data.state.testRunDurationMs / 1000).toFixed(0)}s                                  ║\n`;
  text += `║ Total Requests:    ${metrics.http_reqs.values.count.toFixed(0).padEnd(6)}                              ║\n`;
  text += `║ Requests/sec:      ${metrics.http_reqs.values.rate.toFixed(2).padEnd(6)}                              ║\n`;
  text += '╠═══════════════════════════════════════════════════════╣\n';
  text += `║ Events Created:    ${(metrics.events_created?.values.count || 0).toFixed(0).padEnd(6)}                              ║\n`;
  text += `║ Lineage Queries:   ${(metrics.lineage_queries?.values.count || 0).toFixed(0).padEnd(6)}                              ║\n`;
  text += `║ Claims Verified:   ${(metrics.claims_verified?.values.count || 0).toFixed(0).padEnd(6)}                              ║\n`;
  text += '╠═══════════════════════════════════════════════════════╣\n';
  text += `║ p50 Latency:       ${metrics.http_req_duration.values['p(50)'].toFixed(2).padEnd(6)}ms                            ║\n`;
  text += `║ p95 Latency:       ${metrics.http_req_duration.values['p(95)'].toFixed(2).padEnd(6)}ms                            ║\n`;
  text += `║ p99 Latency:       ${metrics.http_req_duration.values['p(99)'].toFixed(2).padEnd(6)}ms                            ║\n`;
  text += `║ Max Latency:       ${metrics.http_req_duration.values.max.toFixed(2).padEnd(6)}ms                            ║\n`;
  text += '╠═══════════════════════════════════════════════════════╣\n';

  const errorPct = (metrics.http_req_failed.values.rate * 100).toFixed(2);
  const errorStatus = errorPct < 1 ? '✅' : '⚠️ ';
  text += `║ Error Rate:        ${errorStatus} ${errorPct.padEnd(5)}%                              ║\n`;

  // Check if thresholds passed
  let allPassed = true;
  if (metrics.http_req_duration.values['p(95)'] >= 100) allPassed = false;
  if (metrics.http_req_duration.values['p(99)'] >= 200) allPassed = false;
  if (metrics.http_req_failed.values.rate >= 0.01) allPassed = false;

  text += '╠═══════════════════════════════════════════════════════╣\n';
  if (allPassed) {
    text += '║ Overall:           ✅ ALL THRESHOLDS PASSED             ║\n';
  } else {
    text += '║ Overall:           ⚠️  SOME THRESHOLDS FAILED           ║\n';
  }
  text += '╚═══════════════════════════════════════════════════════╝\n';
  text += '\n';

  return text;
}
