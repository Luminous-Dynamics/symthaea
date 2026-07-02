// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Validation Middleware
 *
 * Express middleware for validating requests using Zod schemas.
 * Provides type-safe request handling with automatic error responses.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import { z, ZodError, ZodSchema } from 'zod';
import { errors } from '../utils/response';

/**
 * Validation target - which part of the request to validate
 */
export type ValidationTarget = 'body' | 'query' | 'params';

/**
 * Validation options
 */
export interface ValidateOptions {
  /** Strip unknown keys from the validated object */
  stripUnknown?: boolean;
  /** Custom error message prefix */
  errorPrefix?: string;
}

/**
 * Format Zod errors into a user-friendly structure
 */
function formatZodErrors(error: ZodError): Record<string, string[]> {
  const formatted: Record<string, string[]> = {};

  for (const issue of error.issues) {
    const path = issue.path.join('.') || '_root';
    if (!formatted[path]) {
      formatted[path] = [];
    }
    formatted[path].push(issue.message);
  }

  return formatted;
}

/**
 * Create validation middleware for a specific target
 */
export function validate<T extends ZodSchema>(
  schema: T,
  target: ValidationTarget = 'body',
  options: ValidateOptions = {}
): RequestHandler {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const dataToValidate = req[target];

      // Parse and validate
      const parseOptions = options.stripUnknown !== false
        ? { stripUnknown: true }
        : undefined;

      const validated = await schema.parseAsync(dataToValidate);

      // Replace request data with validated (and potentially transformed) data
      (req as any)[target] = validated;

      // Also attach typed data for convenience
      if (!req.validated) {
        req.validated = {};
      }
      req.validated[target] = validated;

      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const formatted = formatZodErrors(error);
        const prefix = options.errorPrefix || `Invalid ${target}`;

        errors.validationError(res, {
          message: prefix,
          errors: formatted,
        });
        return;
      }

      // Re-throw unexpected errors
      next(error);
    }
  };
}

/**
 * Validate request body
 */
export function validateBody<T extends ZodSchema>(
  schema: T,
  options?: ValidateOptions
): RequestHandler {
  return validate(schema, 'body', options);
}

/**
 * Validate query parameters
 */
export function validateQuery<T extends ZodSchema>(
  schema: T,
  options?: ValidateOptions
): RequestHandler {
  return validate(schema, 'query', options);
}

/**
 * Validate URL parameters
 */
export function validateParams<T extends ZodSchema>(
  schema: T,
  options?: ValidateOptions
): RequestHandler {
  return validate(schema, 'params', options);
}

/**
 * Combined validation for multiple targets
 */
export interface ValidationSchemas {
  body?: ZodSchema;
  query?: ZodSchema;
  params?: ZodSchema;
}

export function validateRequest(schemas: ValidationSchemas): RequestHandler[] {
  const middlewares: RequestHandler[] = [];

  if (schemas.params) {
    middlewares.push(validateParams(schemas.params));
  }
  if (schemas.query) {
    middlewares.push(validateQuery(schemas.query));
  }
  if (schemas.body) {
    middlewares.push(validateBody(schemas.body));
  }

  return middlewares;
}

/**
 * Type helper for extracting validated types
 */
export type ValidatedBody<T extends ZodSchema> = z.infer<T>;
export type ValidatedQuery<T extends ZodSchema> = z.infer<T>;
export type ValidatedParams<T extends ZodSchema> = z.infer<T>;

/**
 * Extend Express Request to include validated data
 */
declare global {
  namespace Express {
    interface Request {
      validated?: {
        body?: unknown;
        query?: unknown;
        params?: unknown;
      };
    }
  }
}

/**
 * Type-safe request handler that includes validated data
 */
export type TypedRequestHandler<
  TBody = unknown,
  TQuery = unknown,
  TParams = unknown,
> = (
  req: Request & {
    body: TBody;
    query: TQuery;
    params: TParams;
  },
  res: Response,
  next: NextFunction
) => void | Promise<void>;

/**
 * Helper to create a typed handler with validation
 */
export function typedHandler<
  TBody extends ZodSchema,
  TQuery extends ZodSchema,
  TParams extends ZodSchema,
>(
  schemas: {
    body?: TBody;
    query?: TQuery;
    params?: TParams;
  },
  handler: TypedRequestHandler<
    TBody extends ZodSchema ? z.infer<TBody> : unknown,
    TQuery extends ZodSchema ? z.infer<TQuery> : unknown,
    TParams extends ZodSchema ? z.infer<TParams> : unknown
  >
): RequestHandler[] {
  return [
    ...validateRequest(schemas),
    handler as RequestHandler,
  ];
}
