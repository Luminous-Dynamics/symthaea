// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Validation Middleware Tests
 */

import { Request, Response } from 'express';
import { z } from 'zod';
import { validateBody, validateQuery, validateParams } from '../../middleware/validate';

// Mock response
const createMockResponse = () => {
  const res: Partial<Response> = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
  };
  return res as Response;
};

// Mock request
const createMockRequest = (overrides: Partial<Request> = {}) => {
  return {
    body: {},
    query: {},
    params: {},
    validated: {},
    ...overrides,
  } as Request;
};

describe('Validation Middleware', () => {
  describe('validateBody', () => {
    const schema = z.object({
      name: z.string().min(1),
      age: z.number().positive(),
      email: z.string().email().optional(),
    });

    it('should pass valid body to next', async () => {
      const req = createMockRequest({
        body: { name: 'John', age: 25 },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateBody(schema);
      await middleware(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.body).toEqual({ name: 'John', age: 25 });
    });

    it('should transform and validate data', async () => {
      const transformSchema = z.object({
        name: z.string().transform(s => s.toUpperCase()),
      });

      const req = createMockRequest({
        body: { name: 'john' },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateBody(transformSchema);
      await middleware(req, res, next);

      expect(req.body.name).toBe('JOHN');
    });

    it('should return 422 for invalid body', async () => {
      const req = createMockRequest({
        body: { name: '', age: -5 },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateBody(schema);
      await middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(422);
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'VALIDATION_ERROR',
          }),
        })
      );
      expect(next).not.toHaveBeenCalled();
    });

    it('should include field-specific errors', async () => {
      const req = createMockRequest({
        body: { name: '', age: 'not a number' },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateBody(schema);
      await middleware(req, res, next);

      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            details: expect.objectContaining({
              errors: expect.objectContaining({
                name: expect.any(Array),
                age: expect.any(Array),
              }),
            }),
          }),
        })
      );
    });
  });

  describe('validateQuery', () => {
    const schema = z.object({
      page: z.string().transform(Number).pipe(z.number().positive()).optional(),
      limit: z.string().transform(Number).pipe(z.number().min(1).max(100)).optional(),
      search: z.string().optional(),
    });

    it('should validate query parameters', async () => {
      const req = createMockRequest({
        query: { page: '1', limit: '20', search: 'test' },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateQuery(schema);
      await middleware(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.query).toEqual({ page: 1, limit: 20, search: 'test' });
    });

    it('should handle missing optional params', async () => {
      const req = createMockRequest({
        query: {},
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateQuery(schema);
      await middleware(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should reject invalid query params', async () => {
      const req = createMockRequest({
        query: { limit: '500' }, // Over max
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateQuery(schema);
      await middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(422);
      expect(next).not.toHaveBeenCalled();
    });
  });

  describe('validateParams', () => {
    const schema = z.object({
      id: z.string().uuid(),
    });

    it('should validate URL parameters', async () => {
      const validUuid = '123e4567-e89b-12d3-a456-426614174000';
      const req = createMockRequest({
        params: { id: validUuid },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateParams(schema);
      await middleware(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.params.id).toBe(validUuid);
    });

    it('should reject invalid UUID', async () => {
      const req = createMockRequest({
        params: { id: 'not-a-uuid' },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateParams(schema);
      await middleware(req, res, next);

      expect(res.status).toHaveBeenCalledWith(422);
      expect(next).not.toHaveBeenCalled();
    });
  });

  describe('Custom schemas', () => {
    it('should validate wallet address format', async () => {
      const walletSchema = z.object({
        address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
      });

      const validReq = createMockRequest({
        body: { address: '0x1234567890abcdef1234567890abcdef12345678' },
      });
      const invalidReq = createMockRequest({
        body: { address: '0xinvalid' },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateBody(walletSchema);

      await middleware(validReq, res, next);
      expect(next).toHaveBeenCalled();

      next.mockClear();
      await middleware(invalidReq, res, next);
      expect(next).not.toHaveBeenCalled();
    });

    it('should validate IPFS hash format', async () => {
      const ipfsSchema = z.object({
        hash: z.string().regex(/^Qm[1-9A-HJ-NP-Za-km-z]{44}$/),
      });

      const validReq = createMockRequest({
        body: { hash: 'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG' },
      });
      const res = createMockResponse();
      const next = jest.fn();

      const middleware = validateBody(ipfsSchema);
      await middleware(validReq, res, next);

      expect(next).toHaveBeenCalled();
    });
  });
});
