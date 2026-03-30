// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - k6 Load Tests
 *
 * Run with: k6 run k6-load-test.js
 *
 * Options:
 *   --env BASE_URL=http://localhost:8080
 *   --env API_KEY=your-api-key
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const emailsSent = new Counter('emails_sent');
const emailsReceived = new Counter('emails_received');
const searchQueries = new Counter('search_queries');
const errorRate = new Rate('errors');
const emailSendDuration = new Trend('email_send_duration');
const searchDuration = new Trend('search_duration');
const inboxLoadDuration = new Trend('inbox_load_duration');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - basic functionality
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      tags: { scenario: 'smoke' },
      env: { SCENARIO: 'smoke' },
    },

    // Load test - normal traffic
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 50 },   // Steady state
        { duration: '2m', target: 100 },  // Peak
        { duration: '5m', target: 100 },  // Sustained peak
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { scenario: 'load' },
      startTime: '1m',
    },

    // Stress test - beyond normal capacity
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '3m', target: 200 },
        { duration: '3m', target: 300 },
        { duration: '2m', target: 0 },
      ],
      tags: { scenario: 'stress' },
      startTime: '17m',
    },

    // Spike test - sudden traffic surge
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 200 },  // Instant spike
        { duration: '1m', target: 200 },   // Maintain
        { duration: '10s', target: 0 },    // Quick recovery
      ],
      tags: { scenario: 'spike' },
      startTime: '28m',
    },

    // Soak test - extended duration
    soak: {
      executor: 'constant-vus',
      vus: 30,
      duration: '30m',
      tags: { scenario: 'soak' },
      startTime: '30m',
    },
  },

  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% under 500ms
    http_req_failed: ['rate<0.01'],                   // Error rate under 1%
    errors: ['rate<0.05'],                            // Custom error rate under 5%
    email_send_duration: ['p(95)<2000'],              // Email send under 2s
    search_duration: ['p(95)<300'],                   // Search under 300ms
    inbox_load_duration: ['p(95)<200'],               // Inbox load under 200ms
  },
};

// Helper functions
function getHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
    'X-Request-ID': randomString(16),
  };
}

function handleResponse(res, operation) {
  const success = res.status >= 200 && res.status < 300;
  if (!success) {
    errorRate.add(1);
    console.error(`${operation} failed: ${res.status} - ${res.body}`);
  }
  return success;
}

// Test user pool simulation
function getTestUser() {
  const userIndex = randomIntBetween(1, 100);
  return {
    id: `user-${userIndex}`,
    email: `user${userIndex}@test.mycelix.local`,
  };
}

// Main test function
export default function () {
  const user = getTestUser();

  group('Authentication', () => {
    const loginRes = http.post(
      `${BASE_URL}/api/v1/auth/login`,
      JSON.stringify({
        email: user.email,
        password: 'test-password',
      }),
      { headers: getHeaders() }
    );

    check(loginRes, {
      'login successful': (r) => r.status === 200,
      'has access token': (r) => JSON.parse(r.body).access_token !== undefined,
    });
  });

  group('Inbox Operations', () => {
    // Load inbox
    const startTime = Date.now();
    const inboxRes = http.get(
      `${BASE_URL}/api/v1/emails?folder=inbox&limit=50`,
      { headers: getHeaders() }
    );
    inboxLoadDuration.add(Date.now() - startTime);

    const inboxSuccess = check(inboxRes, {
      'inbox loaded': (r) => r.status === 200,
      'has emails': (r) => {
        const body = JSON.parse(r.body);
        return body.emails && body.emails.length >= 0;
      },
    });

    if (!inboxSuccess) {
      errorRate.add(1);
    }

    // Load email detail
    if (inboxRes.status === 200) {
      const inbox = JSON.parse(inboxRes.body);
      if (inbox.emails && inbox.emails.length > 0) {
        const emailId = inbox.emails[0].id;
        const detailRes = http.get(
          `${BASE_URL}/api/v1/emails/${emailId}`,
          { headers: getHeaders() }
        );

        check(detailRes, {
          'email detail loaded': (r) => r.status === 200,
        });

        emailsReceived.add(1);
      }
    }
  });

  group('Send Email', () => {
    const startTime = Date.now();
    const sendRes = http.post(
      `${BASE_URL}/api/v1/emails/send`,
      JSON.stringify({
        to: [`recipient${randomIntBetween(1, 100)}@test.mycelix.local`],
        subject: `Test email ${randomString(8)}`,
        body_text: `This is a load test email. Random content: ${randomString(100)}`,
        body_html: `<p>This is a <strong>load test</strong> email.</p>`,
      }),
      { headers: getHeaders() }
    );
    emailSendDuration.add(Date.now() - startTime);

    const sendSuccess = check(sendRes, {
      'email sent': (r) => r.status === 200 || r.status === 202,
      'has message id': (r) => {
        if (r.status !== 200 && r.status !== 202) return false;
        const body = JSON.parse(r.body);
        return body.message_id !== undefined;
      },
    });

    if (sendSuccess) {
      emailsSent.add(1);
    } else {
      handleResponse(sendRes, 'send email');
    }
  });

  group('Search', () => {
    const searchTerms = ['meeting', 'report', 'urgent', 'project', 'update'];
    const term = searchTerms[randomIntBetween(0, searchTerms.length - 1)];

    const startTime = Date.now();
    const searchRes = http.get(
      `${BASE_URL}/api/v1/search?q=${encodeURIComponent(term)}&limit=20`,
      { headers: getHeaders() }
    );
    searchDuration.add(Date.now() - startTime);

    const searchSuccess = check(searchRes, {
      'search completed': (r) => r.status === 200,
      'has results': (r) => {
        const body = JSON.parse(r.body);
        return body.results !== undefined;
      },
    });

    if (searchSuccess) {
      searchQueries.add(1);
    } else {
      handleResponse(searchRes, 'search');
    }
  });

  group('Contacts', () => {
    // List contacts
    const listRes = http.get(
      `${BASE_URL}/api/v1/contacts?limit=50`,
      { headers: getHeaders() }
    );

    check(listRes, {
      'contacts loaded': (r) => r.status === 200,
    });

    // Create contact (occasionally)
    if (randomIntBetween(1, 10) === 1) {
      const createRes = http.post(
        `${BASE_URL}/api/v1/contacts`,
        JSON.stringify({
          email: `new-contact-${randomString(8)}@example.com`,
          name: `Test Contact ${randomString(5)}`,
        }),
        { headers: getHeaders() }
      );

      check(createRes, {
        'contact created': (r) => r.status === 201,
      });
    }
  });

  group('Trust Network', () => {
    // Get trust score
    const trustRes = http.get(
      `${BASE_URL}/api/v1/trust/score?entity=contact-${randomIntBetween(1, 50)}`,
      { headers: getHeaders() }
    );

    check(trustRes, {
      'trust score retrieved': (r) => r.status === 200 || r.status === 404,
    });

    // Create attestation (occasionally)
    if (randomIntBetween(1, 20) === 1) {
      const attestRes = http.post(
        `${BASE_URL}/api/v1/attestations`,
        JSON.stringify({
          subject_id: `user-${randomIntBetween(1, 100)}`,
          context: 'email_verified',
          claims: { verified: true },
        }),
        { headers: getHeaders() }
      );

      check(attestRes, {
        'attestation created': (r) => r.status === 201,
      });
    }
  });

  group('Folders & Labels', () => {
    const foldersRes = http.get(
      `${BASE_URL}/api/v1/folders`,
      { headers: getHeaders() }
    );

    check(foldersRes, {
      'folders loaded': (r) => r.status === 200,
    });

    const labelsRes = http.get(
      `${BASE_URL}/api/v1/labels`,
      { headers: getHeaders() }
    );

    check(labelsRes, {
      'labels loaded': (r) => r.status === 200,
    });
  });

  // Realistic user think time
  sleep(randomIntBetween(1, 3));
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Total emails sent: ${emailsSent.values.count}`);
  console.log(`Total emails received: ${emailsReceived.values.count}`);
  console.log(`Total searches: ${searchQueries.values.count}`);
}

// Handle summary
export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, opts) {
  // Simple text summary
  let summary = '\n========== LOAD TEST SUMMARY ==========\n\n';

  summary += `Duration: ${(data.state.testRunDurationMs / 1000).toFixed(1)}s\n`;
  summary += `VUs Max: ${data.metrics.vus_max?.values?.max || 0}\n\n`;

  summary += '--- HTTP Metrics ---\n';
  const httpDuration = data.metrics.http_req_duration;
  if (httpDuration) {
    summary += `  Requests: ${httpDuration.values.count}\n`;
    summary += `  Avg: ${httpDuration.values.avg.toFixed(2)}ms\n`;
    summary += `  P95: ${httpDuration.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  P99: ${httpDuration.values['p(99)'].toFixed(2)}ms\n`;
  }

  const httpFailed = data.metrics.http_req_failed;
  if (httpFailed) {
    summary += `  Failed: ${(httpFailed.values.rate * 100).toFixed(2)}%\n`;
  }

  summary += '\n--- Custom Metrics ---\n';
  if (data.metrics.email_send_duration) {
    summary += `  Email Send P95: ${data.metrics.email_send_duration.values['p(95)'].toFixed(2)}ms\n`;
  }
  if (data.metrics.search_duration) {
    summary += `  Search P95: ${data.metrics.search_duration.values['p(95)'].toFixed(2)}ms\n`;
  }
  if (data.metrics.inbox_load_duration) {
    summary += `  Inbox Load P95: ${data.metrics.inbox_load_duration.values['p(95)'].toFixed(2)}ms\n`;
  }

  summary += '\n--- Thresholds ---\n';
  for (const [name, threshold] of Object.entries(data.thresholds || {})) {
    const status = threshold.ok ? '✓' : '✗';
    summary += `  ${status} ${name}\n`;
  }

  summary += '\n========================================\n';

  return summary;
}
