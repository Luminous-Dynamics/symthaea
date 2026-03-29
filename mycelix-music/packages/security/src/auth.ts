// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication Middleware
 *
 * JWT verification and user authentication.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import jwt from 'jsonwebtoken';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    address: string;
    role: 'user' | 'artist' | 'admin' | 'moderator';
    subscription?: 'free' | 'basic' | 'premium' | 'artist';
  };
}

export interface AuthOptions {
  secret: string;
  optional?: boolean; // Allow unauthenticated requests
  roles?: string[]; // Required roles
}

/**
 * Create authentication middleware
 */
export function authMiddleware(options: AuthOptions): RequestHandler {
  const { secret, optional = false, roles = [] } = options;

  return async (
    req: AuthenticatedRequest,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    const authHeader = req.headers.authorization;
    const token = authHeader?.startsWith('Bearer ')
      ? authHeader.slice(7)
      : req.query.token as string;

    if (!token) {
      if (optional) {
        return next();
      }
      res.status(401).json({
        error: 'Unauthorized',
        message: 'Authentication token required',
      });
      return;
    }

    try {
      const decoded = jwt.verify(token, secret) as {
        sub?: string;
        address?: string;
        role?: string;
        subscription?: string;
        iat: number;
        exp: number;
      };

      // Set user on request
      req.user = {
        id: decoded.sub || decoded.address || '',
        address: decoded.address || decoded.sub || '',
        role: (decoded.role as any) || 'user',
        subscription: decoded.subscription as any,
      };

      // Check role requirements
      if (roles.length > 0 && !roles.includes(req.user.role)) {
        res.status(403).json({
          error: 'Forbidden',
          message: 'Insufficient permissions',
        });
        return;
      }

      next();
    } catch (error) {
      if (error instanceof jwt.TokenExpiredError) {
        res.status(401).json({
          error: 'Unauthorized',
          message: 'Token expired',
          code: 'TOKEN_EXPIRED',
        });
        return;
      }

      if (error instanceof jwt.JsonWebTokenError) {
        res.status(401).json({
          error: 'Unauthorized',
          message: 'Invalid token',
          code: 'INVALID_TOKEN',
        });
        return;
      }

      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Authentication failed',
      });
    }
  };
}

/**
 * Create a role-based access control middleware
 */
export function requireRole(...roles: string[]): RequestHandler {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        error: 'Unauthorized',
        message: 'Authentication required',
      });
      return;
    }

    if (!roles.includes(req.user.role)) {
      res.status(403).json({
        error: 'Forbidden',
        message: `Required role: ${roles.join(' or ')}`,
      });
      return;
    }

    next();
  };
}

/**
 * Create a subscription-based access control middleware
 */
export function requireSubscription(...subscriptions: string[]): RequestHandler {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        error: 'Unauthorized',
        message: 'Authentication required',
      });
      return;
    }

    const userSub = req.user.subscription || 'free';
    if (!subscriptions.includes(userSub)) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'Subscription upgrade required',
        required: subscriptions,
        current: userSub,
      });
      return;
    }

    next();
  };
}

/**
 * Verify wallet signature for authentication
 */
export function verifyWalletSignature(
  message: string,
  signature: string,
  expectedAddress: string
): boolean {
  try {
    // In production, use ethers.js or viem to verify
    // const recoveredAddress = ethers.verifyMessage(message, signature);
    // return recoveredAddress.toLowerCase() === expectedAddress.toLowerCase();

    // Placeholder for signature verification
    return signature.length > 0 && expectedAddress.length > 0;
  } catch {
    return false;
  }
}
