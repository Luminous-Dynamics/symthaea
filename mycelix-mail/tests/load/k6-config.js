// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * K6 Load Testing Configuration for Mycelix Mail
 *
 * Performance benchmarks and load testing scenarios
 * Run with: k6 run k6-config.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

// Custom metrics
const emailSendDuration = new Trend('email_send_duration');
const emailListDuration = new Trend('email_list_duration');
const searchDuration = new Trend('search_duration');
const authDuration = new Trend('auth_duration');
const errorRate = new Rate('errors');
const emailsSent = new Counter('emails_sent');
const emailsRead = new Counter('emails_read');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_URL = `${BASE_URL}/api/v1`;

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - quick sanity check
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      tags: { test_type: 'smoke' },
      exec: 'smokeTest',
    },

    // Load test - normal expected load
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp to 100
        { duration: '5m', target: 100 },  // Stay at 100
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'load' },
      exec: 'loadTest',
      startTime: '1m', // Start after smoke test
    },

    // Stress test - find breaking point
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 300 },
        { duration: '5m', target: 300 },
        { duration: '2m', target: 400 },
        { duration: '5m', target: 400 },
        { duration: '5m', target: 0 },
      ],
      tags: { test_type: 'stress' },
      exec: 'stressTest',
      startTime: '20m',
    },

    // Spike test - sudden surge
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 50 },
        { duration: '30s', target: 500 },  // Spike!
        { duration: '1m', target: 500 },
        { duration: '30s', target: 50 },   // Back down
        { duration: '2m', target: 50 },
        { duration: '1m', target: 0 },
      ],
      tags: { test_type: 'spike' },
      exec: 'spikeTest',
      startTime: '50m',
    },

    // Soak test - extended duration
    soak: {
      executor: 'constant-vus',
      vus: 100,
      duration: '2h',
      tags: { test_type: 'soak' },
      exec: 'soakTest',
      startTime: '60m',
    },
  },

  thresholds: {
    // Response time thresholds
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    'http_req_duration{endpoint:login}': ['p(95)<300'],
    'http_req_duration{endpoint:list_emails}': ['p(95)<400'],
    'http_req_duration{endpoint:send_email}': ['p(95)<600'],
    'http_req_duration{endpoint:search}': ['p(95)<800'],

    // Custom metric thresholds
    email_send_duration: ['p(95)<600'],
    email_list_duration: ['p(95)<400'],
    search_duration: ['p(95)<800'],
    auth_duration: ['p(95)<300'],

    // Error rate threshold
    errors: ['rate<0.01'], // Less than 1% errors

    // Throughput
    http_reqs: ['rate>100'], // At least 100 req/s
  },
};

// Test data generators
function generateEmail() {
  return {
    to: [`recipient-${randomString(8)}@example.com`],
    subject: `Test Email ${randomString(16)}`,
    body: `This is a test email body with random content: ${randomString(200)}`,
  };
}

function generateSearchQuery() {
  const queries = [
    'from:john@example.com',
    'subject:meeting',
    'has:attachment',
    'is:unread',
    'newer_than:7d',
    `${randomString(8)}`,
  ];
  return queries[randomIntBetween(0, queries.length - 1)];
}

// Authentication helper
function authenticate() {
  const loginPayload = JSON.stringify({
    email: `loadtest-${__VU}@mycelix.test`,
    password: 'loadtest-password-123',
  });

  const loginStart = Date.now();
  const loginRes = http.post(`${API_URL}/auth/login`, loginPayload, {
    headers: { 'Content-Type': 'application/json' },
    tags: { endpoint: 'login' },
  });
  authDuration.add(Date.now() - loginStart);

  check(loginRes, {
    'login successful': (r) => r.status === 200,
    'has access token': (r) => r.json('access_token') !== undefined,
  }) || errorRate.add(1);

  return loginRes.json('access_token');
}

// Get auth headers
function authHeaders(token) {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };
}

// ============================================================================
// Test Functions
// ============================================================================

export function smokeTest() {
  const token = authenticate();
  if (!token) return;

  group('Smoke Test - Basic Operations', () => {
    // List emails
    const listRes = http.get(`${API_URL}/emails?folder=inbox&limit=10`, {
      headers: authHeaders(token),
      tags: { endpoint: 'list_emails' },
    });
    check(listRes, { 'list emails ok': (r) => r.status === 200 });

    // Get folders
    const foldersRes = http.get(`${API_URL}/folders`, {
      headers: authHeaders(token),
      tags: { endpoint: 'folders' },
    });
    check(foldersRes, { 'get folders ok': (r) => r.status === 200 });

    sleep(1);
  });
}

export function loadTest() {
  const token = authenticate();
  if (!token) return;

  group('Load Test - Typical User Journey', () => {
    // 1. Check inbox
    const listStart = Date.now();
    const listRes = http.get(`${API_URL}/emails?folder=inbox&limit=50`, {
      headers: authHeaders(token),
      tags: { endpoint: 'list_emails' },
    });
    emailListDuration.add(Date.now() - listStart);

    check(listRes, { 'list emails ok': (r) => r.status === 200 }) || errorRate.add(1);
    emailsRead.add(1);

    sleep(randomIntBetween(1, 3));

    // 2. Read a specific email (if any)
    if (listRes.status === 200) {
      const emails = listRes.json('emails') || listRes.json() || [];
      if (emails.length > 0) {
        const emailId = emails[0].id;
        const readRes = http.get(`${API_URL}/emails/${emailId}`, {
          headers: authHeaders(token),
          tags: { endpoint: 'read_email' },
        });
        check(readRes, { 'read email ok': (r) => r.status === 200 });
        emailsRead.add(1);
      }
    }

    sleep(randomIntBetween(2, 5));

    // 3. Send an email (30% of users)
    if (Math.random() < 0.3) {
      const email = generateEmail();
      const sendStart = Date.now();
      const sendRes = http.post(`${API_URL}/emails/send`, JSON.stringify(email), {
        headers: authHeaders(token),
        tags: { endpoint: 'send_email' },
      });
      emailSendDuration.add(Date.now() - sendStart);

      check(sendRes, { 'send email ok': (r) => r.status === 200 || r.status === 201 }) || errorRate.add(1);
      emailsSent.add(1);
    }

    sleep(randomIntBetween(1, 3));

    // 4. Search (20% of users)
    if (Math.random() < 0.2) {
      const query = generateSearchQuery();
      const searchStart = Date.now();
      const searchRes = http.get(`${API_URL}/emails/search?q=${encodeURIComponent(query)}`, {
        headers: authHeaders(token),
        tags: { endpoint: 'search' },
      });
      searchDuration.add(Date.now() - searchStart);

      check(searchRes, { 'search ok': (r) => r.status === 200 }) || errorRate.add(1);
    }

    sleep(randomIntBetween(2, 5));
  });
}

export function stressTest() {
  const token = authenticate();
  if (!token) return;

  group('Stress Test - Heavy Operations', () => {
    // Rapid email listing
    for (let i = 0; i < 5; i++) {
      const listRes = http.get(`${API_URL}/emails?folder=inbox&limit=100&offset=${i * 100}`, {
        headers: authHeaders(token),
        tags: { endpoint: 'list_emails' },
      });
      check(listRes, { 'list ok': (r) => r.status === 200 }) || errorRate.add(1);
    }

    // Multiple searches
    for (let i = 0; i < 3; i++) {
      const query = generateSearchQuery();
      const searchRes = http.get(`${API_URL}/emails/search?q=${encodeURIComponent(query)}`, {
        headers: authHeaders(token),
        tags: { endpoint: 'search' },
      });
      check(searchRes, { 'search ok': (r) => r.status === 200 }) || errorRate.add(1);
    }

    // Batch email send
    const emails = [];
    for (let i = 0; i < 5; i++) {
      emails.push(generateEmail());
    }

    for (const email of emails) {
      const sendRes = http.post(`${API_URL}/emails/send`, JSON.stringify(email), {
        headers: authHeaders(token),
        tags: { endpoint: 'send_email' },
      });
      check(sendRes, { 'send ok': (r) => r.status === 200 || r.status === 201 }) || errorRate.add(1);
      emailsSent.add(1);
    }

    sleep(1);
  });
}

export function spikeTest() {
  // Same as load test but we're testing system behavior under spike
  loadTest();
}

export function soakTest() {
  // Extended load test for memory leaks and stability
  loadTest();
  sleep(randomIntBetween(5, 10));
}

// ============================================================================
// Setup and Teardown
// ============================================================================

export function setup() {
  console.log('Setting up load test...');
  console.log(`Target: ${BASE_URL}`);

  // Verify API is accessible
  const healthRes = http.get(`${BASE_URL}/health`);
  if (healthRes.status !== 200) {
    throw new Error(`API health check failed: ${healthRes.status}`);
  }

  // Create test users if needed
  const setupUsers = [];
  for (let i = 1; i <= 500; i++) {
    setupUsers.push({
      email: `loadtest-${i}@mycelix.test`,
      password: 'loadtest-password-123',
      name: `Load Test User ${i}`,
    });
  }

  return { users: setupUsers };
}

export function teardown(data) {
  console.log('Load test complete.');
  console.log(`Total emails sent: ${emailsSent}`);
  console.log(`Total emails read: ${emailsRead}`);
}

// Default function for standalone runs
export default function () {
  loadTest();
}
