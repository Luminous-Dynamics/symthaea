// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Async Handler Utility
 * Wraps async route handlers to properly catch and forward errors
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';

/**
 * Wrap an async route handler to catch errors and pass them to Express error handler
 */
export function asyncHandler<T extends Request = Request>(
  fn: (req: T, res: Response, next: NextFunction) => Promise<any>
): RequestHandler {
  return (req, res, next) => {
    Promise.resolve(fn(req as T, res, next)).catch(next);
  };
}

/**
 * Wrap multiple async handlers
 */
export function asyncHandlers<T extends Request = Request>(
  ...handlers: Array<(req: T, res: Response, next: NextFunction) => Promise<any>>
): RequestHandler[] {
  return handlers.map(handler => asyncHandler(handler));
}

/**
 * Create a typed async handler with custom request type
 */
export function createAsyncHandler<T extends Request>() {
  return (fn: (req: T, res: Response, next: NextFunction) => Promise<any>): RequestHandler => {
    return asyncHandler(fn);
  };
}

export default asyncHandler;
