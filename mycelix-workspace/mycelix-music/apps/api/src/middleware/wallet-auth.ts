// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Wallet Authentication Middleware
 *
 * Verifies Ethereum wallet signatures for authenticated endpoints.
 * Supports EIP-191 personal signatures and EIP-712 typed data.
 */

import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import { verifyMessage } from 'ethers';
import { errors } from '../utils/response';

/**
 * Signature verification result
 */
export interface VerificationResult {
  valid: boolean;
  address?: string;
  error?: string;
}

/**
 * Auth payload structure
 */
export interface AuthPayload {
  address: string;
  timestamp: number;
  nonce: string;
  action?: string;
}

/**
 * Extend Express Request with auth info
 */
declare global {
  namespace Express {
    interface Request {
      auth?: {
        address: string;
        payload: AuthPayload;
      };
    }
  }
}

/**
 * Verify Ethereum signature using ethers.js ECDSA recovery
 *
 * This performs proper cryptographic verification:
 * 1. Validates signature format (65 bytes: r, s, v)
 * 2. Recovers the signer address from the signature
 * 3. Compares recovered address to expected address (case-insensitive)
 */
function verifySimpleSignature(
  message: string,
  signature: string,
  expectedAddress: string
): boolean {
  // Validate signature format
  if (!signature.startsWith('0x') || signature.length !== 132) {
    return false;
  }

  // In development mode with explicit flag, accept well-formed signatures
  // This should ONLY be enabled for local testing
  if (process.env.NODE_ENV === 'development' && process.env.SKIP_AUTH_VERIFY === 'true') {
    console.warn('WARNING: Signature verification disabled - development mode only');
    return true;
  }

  try {
    // Use ethers.js to recover the signer address from the signature
    // verifyMessage handles EIP-191 personal_sign format
    const recoveredAddress = verifyMessage(message, signature);

    // Compare addresses (case-insensitive as Ethereum addresses are case-insensitive)
    const isValid = recoveredAddress.toLowerCase() === expectedAddress.toLowerCase();

    if (!isValid) {
      console.warn(
        `Signature verification failed: expected ${expectedAddress}, recovered ${recoveredAddress}`
      );
    }

    return isValid;
  } catch (error) {
    console.error('Signature verification error:', error);
    return false;
  }
}

/**
 * Verify Ethereum signature using ethers.js pattern
 */
export async function verifySignature(
  message: string,
  signature: string,
  expectedAddress: string
): Promise<VerificationResult> {
  try {
    // Validate inputs
    if (!message || !signature || !expectedAddress) {
      return { valid: false, error: 'Missing required parameters' };
    }

    // Validate address format
    if (!/^0x[a-fA-F0-9]{40}$/.test(expectedAddress)) {
      return { valid: false, error: 'Invalid address format' };
    }

    // Validate signature format
    if (!/^0x[a-fA-F0-9]{130}$/.test(signature)) {
      return { valid: false, error: 'Invalid signature format' };
    }

    const valid = verifySimpleSignature(message, signature, expectedAddress);

    if (valid) {
      return { valid: true, address: expectedAddress.toLowerCase() };
    }

    return { valid: false, error: 'Signature verification failed' };
  } catch (error) {
    return { valid: false, error: (error as Error).message };
  }
}

/**
 * Create message to sign
 */
export function createSignMessage(payload: AuthPayload): string {
  return [
    'Mycelix Music Authentication',
    '',
    `Address: ${payload.address}`,
    `Timestamp: ${payload.timestamp}`,
    `Nonce: ${payload.nonce}`,
    payload.action ? `Action: ${payload.action}` : '',
  ].filter(Boolean).join('\n');
}

/**
 * Generate a nonce for authentication
 */
export function generateNonce(): string {
  return createHash('sha256')
    .update(Date.now().toString() + Math.random().toString())
    .digest('hex')
    .slice(0, 32);
}

/**
 * Validate timestamp is within acceptable range
 */
function isTimestampValid(timestamp: number, maxAgeMs = 5 * 60 * 1000): boolean {
  const now = Date.now();
  const age = now - timestamp;
  return age >= 0 && age <= maxAgeMs;
}

/**
 * Wallet authentication middleware options
 */
export interface WalletAuthOptions {
  /** Maximum age of signature in milliseconds */
  maxAge?: number;
  /** Skip verification in development */
  skipInDev?: boolean;
  /** Required action for this endpoint */
  requiredAction?: string;
}

/**
 * Create wallet authentication middleware
 */
export function walletAuth(options: WalletAuthOptions = {}) {
  const maxAge = options.maxAge || 5 * 60 * 1000; // 5 minutes default

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Skip in development if configured
    if (options.skipInDev && process.env.NODE_ENV === 'development') {
      // Check for dev address header
      const devAddress = req.headers['x-dev-address'] as string;
      if (devAddress && /^0x[a-fA-F0-9]{40}$/.test(devAddress)) {
        req.auth = {
          address: devAddress.toLowerCase(),
          payload: {
            address: devAddress.toLowerCase(),
            timestamp: Date.now(),
            nonce: 'dev-nonce',
          },
        };
        req.context.walletAddress = devAddress.toLowerCase();
        return next();
      }
    }

    // Get signature from header
    const signature = req.headers['x-wallet-signature'] as string;
    if (!signature) {
      errors.unauthorized(res, 'Wallet signature required');
      return;
    }

    // Get payload from header (base64 encoded JSON)
    const payloadHeader = req.headers['x-wallet-payload'] as string;
    if (!payloadHeader) {
      errors.unauthorized(res, 'Auth payload required');
      return;
    }

    // Decode payload
    let payload: AuthPayload;
    try {
      const decoded = Buffer.from(payloadHeader, 'base64').toString('utf8');
      payload = JSON.parse(decoded);
    } catch {
      errors.badRequest(res, 'Invalid auth payload');
      return;
    }

    // Validate payload structure
    if (!payload.address || !payload.timestamp || !payload.nonce) {
      errors.badRequest(res, 'Incomplete auth payload');
      return;
    }

    // Validate timestamp
    if (!isTimestampValid(payload.timestamp, maxAge)) {
      errors.unauthorized(res, 'Signature expired');
      return;
    }

    // Validate required action
    if (options.requiredAction && payload.action !== options.requiredAction) {
      errors.unauthorized(res, 'Invalid action in signature');
      return;
    }

    // Create message and verify signature
    const message = createSignMessage(payload);
    const result = await verifySignature(message, signature, payload.address);

    if (!result.valid) {
      errors.unauthorized(res, result.error || 'Invalid signature');
      return;
    }

    // Attach auth info to request
    req.auth = {
      address: result.address!,
      payload,
    };

    // Also attach to context
    if (req.context) {
      req.context.walletAddress = result.address;
    }

    next();
  };
}

/**
 * Optional wallet auth - doesn't fail if not provided
 */
export function optionalWalletAuth(options: WalletAuthOptions = {}) {
  const authMiddleware = walletAuth(options);

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Check if auth headers are present
    const hasAuth = req.headers['x-wallet-signature'] && req.headers['x-wallet-payload'];

    if (!hasAuth) {
      // No auth provided, continue without
      return next();
    }

    // Auth provided, validate it
    await authMiddleware(req, res, next);
  };
}

/**
 * Require specific wallet address (owner check)
 */
export function requireOwner(getOwnerId: (req: Request) => string | Promise<string>) {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    if (!req.auth) {
      errors.unauthorized(res, 'Authentication required');
      return;
    }

    const ownerId = await getOwnerId(req);

    if (req.auth.address.toLowerCase() !== ownerId.toLowerCase()) {
      errors.forbidden(res, 'Not authorized to access this resource');
      return;
    }

    next();
  };
}

/**
 * Generate auth challenge for client
 */
export function generateAuthChallenge(address: string, action?: string): {
  payload: AuthPayload;
  message: string;
  payloadEncoded: string;
} {
  const payload: AuthPayload = {
    address: address.toLowerCase(),
    timestamp: Date.now(),
    nonce: generateNonce(),
    action,
  };

  const message = createSignMessage(payload);
  const payloadEncoded = Buffer.from(JSON.stringify(payload)).toString('base64');

  return { payload, message, payloadEncoded };
}
