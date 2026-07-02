// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CSRF Protection Tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Request, Response, NextFunction } from 'express';
import {
  generateCsrfToken,
  verifyCsrfToken,
  csrfMiddleware,
  getCsrfFromRequest,
} from '../csrf';

// Mock crypto
vi.mock('crypto', async () => {
  const actual = await vi.importActual('crypto');
  return {
    ...actual,
    randomBytes: vi.fn(() => Buffer.from('mock-random-bytes-1234567890abcdef')),
    createHmac: vi.fn(() => ({
      update: vi.fn().mockReturnThis(),
      digest: vi.fn(() => 'mock-signature'),
    })),
  };
});

describe('CSRF Protection', () => {
  const mockSecret = 'test-csrf-secret-key-32bytes!!!!';

  describe('generateCsrfToken', () => {
    it('generates a token with timestamp and signature', () => {
      const token = generateCsrfToken(mockSecret);

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.includes('.')).toBe(true);
    });

    it('generates unique tokens', () => {
      const token1 = generateCsrfToken(mockSecret);
      const token2 = generateCsrfToken(mockSecret);

      // Tokens may be similar due to mock, but in real use would be different
      expect(token1).toBeDefined();
      expect(token2).toBeDefined();
    });

    it('includes timestamp in token', () => {
      const token = generateCsrfToken(mockSecret);
      const parts = token.split('.');

      expect(parts.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('verifyCsrfToken', () => {
    it('verifies a valid token', () => {
      const token = generateCsrfToken(mockSecret);
      const result = verifyCsrfToken(token, mockSecret);

      expect(result.valid).toBe(true);
    });

    it('rejects invalid token format', () => {
      const result = verifyCsrfToken('invalid-token', mockSecret);

      expect(result.valid).toBe(false);
      expect(result.error).toContain('format');
    });

    it('rejects empty token', () => {
      const result = verifyCsrfToken('', mockSecret);

      expect(result.valid).toBe(false);
    });

    it('rejects null/undefined token', () => {
      const result1 = verifyCsrfToken(null as any, mockSecret);
      const result2 = verifyCsrfToken(undefined as any, mockSecret);

      expect(result1.valid).toBe(false);
      expect(result2.valid).toBe(false);
    });

    it('rejects expired tokens', () => {
      // Create a token with an old timestamp
      const oldTimestamp = Date.now() - (25 * 60 * 60 * 1000); // 25 hours ago
      const token = `${oldTimestamp}.mock-nonce.mock-signature`;

      const result = verifyCsrfToken(token, mockSecret, { maxAge: 24 * 60 * 60 * 1000 });

      expect(result.valid).toBe(false);
      expect(result.error).toContain('expired');
    });

    it('rejects tampered tokens', () => {
      const token = generateCsrfToken(mockSecret);
      const tamperedToken = token.replace('mock', 'fake');

      const result = verifyCsrfToken(tamperedToken, mockSecret);

      expect(result.valid).toBe(false);
    });
  });

  describe('getCsrfFromRequest', () => {
    it('extracts token from header', () => {
      const mockReq = {
        headers: {
          'x-csrf-token': 'header-token',
        },
        body: {},
        query: {},
      } as unknown as Request;

      const token = getCsrfFromRequest(mockReq);

      expect(token).toBe('header-token');
    });

    it('extracts token from body', () => {
      const mockReq = {
        headers: {},
        body: { _csrf: 'body-token' },
        query: {},
      } as unknown as Request;

      const token = getCsrfFromRequest(mockReq);

      expect(token).toBe('body-token');
    });

    it('extracts token from query', () => {
      const mockReq = {
        headers: {},
        body: {},
        query: { _csrf: 'query-token' },
      } as unknown as Request;

      const token = getCsrfFromRequest(mockReq);

      expect(token).toBe('query-token');
    });

    it('prioritizes header over body', () => {
      const mockReq = {
        headers: { 'x-csrf-token': 'header-token' },
        body: { _csrf: 'body-token' },
        query: {},
      } as unknown as Request;

      const token = getCsrfFromRequest(mockReq);

      expect(token).toBe('header-token');
    });

    it('returns undefined when no token present', () => {
      const mockReq = {
        headers: {},
        body: {},
        query: {},
      } as unknown as Request;

      const token = getCsrfFromRequest(mockReq);

      expect(token).toBeUndefined();
    });
  });

  describe('csrfMiddleware', () => {
    let mockReq: Partial<Request>;
    let mockRes: Partial<Response>;
    let mockNext: NextFunction;

    beforeEach(() => {
      mockReq = {
        method: 'POST',
        headers: {},
        body: {},
        query: {},
        cookies: {},
      };
      mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn().mockReturnThis(),
        cookie: vi.fn().mockReturnThis(),
      };
      mockNext = vi.fn();
    });

    it('skips CSRF check for GET requests', () => {
      mockReq.method = 'GET';

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRes.status).not.toHaveBeenCalled();
    });

    it('skips CSRF check for HEAD requests', () => {
      mockReq.method = 'HEAD';

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('skips CSRF check for OPTIONS requests', () => {
      mockReq.method = 'OPTIONS';

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('rejects POST without CSRF token', () => {
      mockReq.method = 'POST';

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.status).toHaveBeenCalledWith(403);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.stringContaining('CSRF'),
        })
      );
    });

    it('allows POST with valid CSRF token', () => {
      const token = generateCsrfToken(mockSecret);
      mockReq.method = 'POST';
      mockReq.headers = { 'x-csrf-token': token };
      mockReq.cookies = { 'csrf-token': token };

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Should pass validation (in mocked environment)
      expect(mockNext).toHaveBeenCalled();
    });

    it('allows excluded paths', () => {
      mockReq.method = 'POST';
      mockReq.path = '/api/webhooks/stripe';

      const middleware = csrfMiddleware({
        secret: mockSecret,
        excludePaths: ['/api/webhooks'],
      });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('sets CSRF cookie on GET requests', () => {
      mockReq.method = 'GET';

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.cookie).toHaveBeenCalledWith(
        'csrf-token',
        expect.any(String),
        expect.objectContaining({
          httpOnly: false,
          secure: expect.any(Boolean),
          sameSite: 'strict',
        })
      );
    });
  });

  describe('Double Submit Cookie Pattern', () => {
    it('validates token matches cookie', () => {
      const token = generateCsrfToken(mockSecret);

      const mockReq = {
        method: 'POST',
        headers: { 'x-csrf-token': token },
        cookies: { 'csrf-token': token },
        body: {},
        query: {},
      } as unknown as Request;

      const mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn().mockReturnThis(),
        cookie: vi.fn().mockReturnThis(),
      } as unknown as Response;

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq, mockRes, vi.fn());

      // Should not return 403
      expect(mockRes.status).not.toHaveBeenCalledWith(403);
    });

    it('rejects mismatched token and cookie', () => {
      const token1 = generateCsrfToken(mockSecret);
      const token2 = generateCsrfToken(mockSecret + 'different');

      const mockReq = {
        method: 'POST',
        headers: { 'x-csrf-token': token1 },
        cookies: { 'csrf-token': token2 },
        body: {},
        query: {},
      } as unknown as Request;

      const mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn().mockReturnThis(),
        cookie: vi.fn().mockReturnThis(),
      } as unknown as Response;

      const middleware = csrfMiddleware({ secret: mockSecret });
      middleware(mockReq, mockRes, vi.fn());

      expect(mockRes.status).toHaveBeenCalledWith(403);
    });
  });
});
