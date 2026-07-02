// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * k6 Soak Test for Mycelix Music API
 *
 * Purpose: Test system stability over extended period
 * Detects: Memory leaks, connection exhaustion, degradation over time
 *
 * Usage:
 *   k6 run load-tests/soak-test.js
 *   k6 run --duration 2h load-tests/soak-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const responseTrend = new Trend('response_time_trend');

const API_URL = __ENV.API_URL || 'http://localhost:3100';

export const options = {
  stages: [
    { duration: '5m', target: 20 },   // Ramp up
    { duration: '30m', target: 20 },  // Sustained load (adjust for longer soak)
    { duration: '5m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<3000'], // 99% under 3s for soak
    http_req_failed: ['rate<0.01'],    // Less than 1% errors
    errors: ['rate<0.01'],
  },
};

export default function () {
  // Mix of typical user actions
  const actions = [
    () => {
      // Health check
      const res = http.get(`${API_URL}/health`);
      check(res, { 'health ok': (r) => r.status === 200 }) || errorRate.add(1);
      responseTrend.add(res.timings.duration);
    },
    () => {
      // Browse songs
      const res = http.get(`${API_URL}/api/songs?limit=20`);
      check(res, { 'songs ok': (r) => r.status === 200 }) || errorRate.add(1);
      responseTrend.add(res.timings.duration);
    },
    () => {
      // Search
      const res = http.get(`${API_URL}/api/songs?q=test&limit=10`);
      check(res, { 'search ok': (r) => r.status === 200 }) || errorRate.add(1);
      responseTrend.add(res.timings.duration);
    },
    () => {
      // Analytics
      const res = http.get(`${API_URL}/api/analytics/top-songs`);
      check(res, { 'analytics ok': (r) => r.status === 200 }) || errorRate.add(1);
      responseTrend.add(res.timings.duration);
    },
  ];

  // Execute random action
  const action = actions[Math.floor(Math.random() * actions.length)];
  action();

  // Think time between actions
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}
