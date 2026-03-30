// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * k6 Load Testing Script for Mycelix Music API
 *
 * Installation:
 *   brew install k6  (macOS)
 *   apt install k6   (Ubuntu)
 *
 * Usage:
 *   k6 run load-tests/api-load.js
 *   k6 run --vus 50 --duration 5m load-tests/api-load.js
 *   k6 run --out json=results.json load-tests/api-load.js
 *
 * Environment Variables:
 *   API_URL - Base URL for API (default: http://localhost:3100)
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const songListDuration = new Trend('song_list_duration');
const songDetailDuration = new Trend('song_detail_duration');
const analyticsDuration = new Trend('analytics_duration');
const requestsTotal = new Counter('requests_total');

// Configuration
const API_URL = __ENV.API_URL || 'http://localhost:3100';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - verify system works
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '30s',
      startTime: '0s',
      tags: { test_type: 'smoke' },
    },
    // Load test - normal traffic
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },  // Ramp up
        { duration: '3m', target: 20 },  // Stay at 20 users
        { duration: '1m', target: 0 },   // Ramp down
      ],
      startTime: '30s',
      tags: { test_type: 'load' },
    },
    // Stress test - find breaking point
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },  // Ramp to 50
        { duration: '3m', target: 50 },  // Hold
        { duration: '2m', target: 100 }, // Ramp to 100
        { duration: '3m', target: 100 }, // Hold
        { duration: '2m', target: 0 },   // Ramp down
      ],
      startTime: '6m',
      tags: { test_type: 'stress' },
    },
    // Spike test - sudden traffic burst
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 100 }, // Instant spike
        { duration: '1m', target: 100 },  // Hold
        { duration: '10s', target: 0 },   // Drop
      ],
      startTime: '18m',
      tags: { test_type: 'spike' },
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.05'],     // Less than 5% errors
    errors: ['rate<0.1'],               // Custom error rate under 10%
  },
};

// Setup - runs once before the test
export function setup() {
  console.log(`Testing API at: ${API_URL}`);

  // Verify API is healthy
  const healthRes = http.get(`${API_URL}/health`);
  if (healthRes.status !== 200) {
    throw new Error(`API health check failed: ${healthRes.status}`);
  }

  return { baseUrl: API_URL };
}

// Default function - runs for each virtual user
export default function (data) {
  const baseUrl = data.baseUrl;

  group('Health Checks', () => {
    const res = http.get(`${baseUrl}/health`);
    requestsTotal.add(1);

    check(res, {
      'health status 200': (r) => r.status === 200,
      'health response valid': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'ok';
        } catch {
          return false;
        }
      },
    }) || errorRate.add(1);
  });

  sleep(0.5);

  group('Song Listing', () => {
    // List songs with pagination
    const listRes = http.get(`${baseUrl}/api/songs?limit=20&offset=0`);
    requestsTotal.add(1);
    songListDuration.add(listRes.timings.duration);

    const listOk = check(listRes, {
      'song list status 200': (r) => r.status === 200,
      'song list is array': (r) => {
        try {
          return Array.isArray(JSON.parse(r.body));
        } catch {
          return false;
        }
      },
    });

    if (!listOk) errorRate.add(1);

    // Parse songs for detail requests
    let songs = [];
    try {
      songs = JSON.parse(listRes.body);
    } catch {}

    // Get random song detail
    if (songs.length > 0) {
      const randomSong = songs[Math.floor(Math.random() * songs.length)];
      if (randomSong && randomSong.id) {
        const detailRes = http.get(`${baseUrl}/api/songs/${randomSong.id}`);
        requestsTotal.add(1);
        songDetailDuration.add(detailRes.timings.duration);

        check(detailRes, {
          'song detail status 200 or 404': (r) => [200, 404].includes(r.status),
        }) || errorRate.add(1);
      }
    }
  });

  sleep(0.5);

  group('Search', () => {
    const queries = ['electronic', 'rock', 'ambient', 'jazz', 'hip hop'];
    const query = queries[Math.floor(Math.random() * queries.length)];

    const searchRes = http.get(`${baseUrl}/api/songs?q=${encodeURIComponent(query)}&limit=10`);
    requestsTotal.add(1);

    check(searchRes, {
      'search status 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  });

  sleep(0.5);

  group('Analytics', () => {
    // Top songs
    const topRes = http.get(`${baseUrl}/api/analytics/top-songs?limit=10`);
    requestsTotal.add(1);
    analyticsDuration.add(topRes.timings.duration);

    check(topRes, {
      'top songs status 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  });

  sleep(1);
}

// Teardown - runs once after the test
export function teardown(data) {
  console.log('Load test complete');
}

// Handle test summary
export function handleSummary(data) {
  return {
    'load-tests/summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: '  ', enableColors: true }),
  };
}

// Simple text summary helper
function textSummary(data, options = {}) {
  const { indent = '', enableColors = false } = options;

  const lines = [
    '',
    '=== LOAD TEST SUMMARY ===',
    '',
    `Total Requests: ${data.metrics.requests_total?.values?.count || 'N/A'}`,
    `Error Rate: ${((data.metrics.errors?.values?.rate || 0) * 100).toFixed(2)}%`,
    '',
    'Response Times:',
    `  p50: ${(data.metrics.http_req_duration?.values?.['p(50)'] || 0).toFixed(2)}ms`,
    `  p90: ${(data.metrics.http_req_duration?.values?.['p(90)'] || 0).toFixed(2)}ms`,
    `  p95: ${(data.metrics.http_req_duration?.values?.['p(95)'] || 0).toFixed(2)}ms`,
    `  p99: ${(data.metrics.http_req_duration?.values?.['p(99)'] || 0).toFixed(2)}ms`,
    '',
    'Custom Metrics:',
    `  Song List p95: ${(data.metrics.song_list_duration?.values?.['p(95)'] || 0).toFixed(2)}ms`,
    `  Song Detail p95: ${(data.metrics.song_detail_duration?.values?.['p(95)'] || 0).toFixed(2)}ms`,
    `  Analytics p95: ${(data.metrics.analytics_duration?.values?.['p(95)'] || 0).toFixed(2)}ms`,
    '',
  ];

  return lines.map(l => indent + l).join('\n');
}
