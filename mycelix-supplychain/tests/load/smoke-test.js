// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * K6 Smoke Test - Quick Sanity Check
 *
 * Purpose: Verify the system works with minimal load
 * Duration: 30 seconds
 * VUs: 1-5
 * Target: Confirm basic functionality before larger tests
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '10s', target: 2 },  // Ramp up to 2 VUs
    { duration: '15s', target: 2 },  // Stay at 2 VUs
    { duration: '5s', target: 0 },   // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500'],  // 95% of requests under 500ms
    'http_req_failed': ['rate<0.05'],    // Error rate under 5%
    'errors': ['rate<0.05'],             // Custom error rate under 5%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

// Sample event data
const createEvent = (batchId) => ({
  '@context': ['https://www.w3.org/2018/credentials/v1'],
  'type': ['VerifiableCredential'],
  'issuer': 'did:mycelix:org:smoke-test',
  'issuanceDate': new Date().toISOString(),
  'credentialSubject': {
    'eventType': 'PRODUCED',
    'productId': `TEST-PRODUCT-${batchId}`,
    'batchId': batchId,
    'quantity': 100.0,
    'unit': 'kg',
    'facility': {
      'id': 'TEST-FACILITY',
      'name': 'Test Facility'
    },
    'timestamp': new Date().toISOString()
  }
});

export default function () {
  const batchId = `SMOKE-BATCH-${__VU}-${__ITER}`;

  // Test 1: Health Check
  let res = http.get(`${BASE_URL}/health`);
  check(res, {
    'health check is 200': (r) => r.status === 200,
    'health check has status': (r) => JSON.parse(r.body).status !== undefined,
  }) || errorRate.add(1);

  sleep(0.5);

  // Test 2: Create Event
  res = http.post(
    `${BASE_URL}/v1/events`,
    JSON.stringify(createEvent(batchId)),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );

  const created = check(res, {
    'event create is 201': (r) => r.status === 201,
    'event has claim_id': (r) => JSON.parse(r.body).claim_id !== undefined,
    'event has lineage_hash': (r) => JSON.parse(r.body).lineage_hash !== undefined,
  });

  if (!created) {
    console.error(`Failed to create event: ${res.status} ${res.body}`);
    errorRate.add(1);
    return;
  }

  const claimId = JSON.parse(res.body).claim_id;

  sleep(0.5);

  // Test 3: Get Claim
  res = http.get(`${BASE_URL}/v1/claims/${claimId}`);
  check(res, {
    'get claim is 200': (r) => r.status === 200,
    'claim has id': (r) => JSON.parse(r.body).claim.id === claimId,
  }) || errorRate.add(1);

  sleep(1);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options?.indent || '';
  const enableColors = options?.enableColors || false;

  let summary = '\n';
  summary += `${indent}=== Smoke Test Results ===\n`;
  summary += `${indent}Duration: ${data.state.testRunDurationMs}ms\n`;
  summary += `${indent}Requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += `${indent}Errors: ${data.metrics.errors ? data.metrics.errors.values.rate * 100 : 0}%\n`;
  summary += `${indent}p95 latency: ${data.metrics.http_req_duration.values['p(95)']}ms\n`;
  summary += `${indent}p99 latency: ${data.metrics.http_req_duration.values['p(99)']}ms\n`;

  if (data.metrics.http_req_failed.values.rate > 0) {
    summary += `${indent}⚠️  WARNING: ${data.metrics.http_req_failed.values.rate * 100}% failed requests\n`;
  } else {
    summary += `${indent}✅ All checks passed!\n`;
  }

  return summary;
}
