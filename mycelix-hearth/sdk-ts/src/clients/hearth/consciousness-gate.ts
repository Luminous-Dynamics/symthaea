// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Consciousness Gate Error Handling for Hearth SDK
 *
 * Structured error types for consciousness gate rejections and
 * retry-on-expired middleware for Hearth domain clients.
 *
 * @module @mycelix/hearth-sdk/consciousness-gate
 */

import { HearthError, classifyError } from './errors';
import type { ConsciousnessCredential, ConsciousnessTier } from './types';

// ============================================================================
// Types
// ============================================================================

/** Structured rejection details from a consciousness gate */
export interface ConsciousnessGateRejection {
  action: string;
  zome: string;
  actualTier: string;
  requiredTier: string;
  weightBp: number;
  reasons: string[];
  expired: boolean;
}

// ============================================================================
// ConsciousnessGateError
// ============================================================================

/**
 * Specialized error for consciousness gate rejections in Hearth zomes.
 *
 * Parses the structured error message from Rust `require_consciousness()`
 * and provides typed access to rejection details.
 */
export class ConsciousnessGateError extends HearthError {
  public readonly rejection: ConsciousnessGateRejection;

  constructor(message: string, rejection: ConsciousnessGateRejection) {
    super({
      code: 'UNAUTHORIZED',
      message,
      zome: rejection.zome,
      fnName: rejection.action,
    });
    this.name = 'ConsciousnessGateError';
    this.rejection = rejection;
  }

  /**
   * Attempt to parse a WASM/zome error into a ConsciousnessGateError.
   *
   * Returns null if the error is not a consciousness gate rejection.
   */
  static fromWasmError(error: unknown): ConsciousnessGateError | null {
    const msg = error instanceof Error ? error.message : String(error);

    // Match consciousness gate rejection pattern
    const gateMatch = msg.match(
      /Consciousness gate:\s*tier\s+(\w+)\s+insufficient\.\s*Reasons:\s*(.*)/i
    );

    if (gateMatch) {
      const actualTier = gateMatch[1];
      const reasonsStr = gateMatch[2];
      const reasons = reasonsStr
        .split(',')
        .map((r) => r.trim())
        .filter(Boolean);
      const expired = reasons.some((r) => r.toLowerCase().includes('expired'));

      const tierMatch = reasonsStr.match(/below required (\w+)/);
      const requiredTier = tierMatch?.[1] ?? 'Unknown';

      return new ConsciousnessGateError(msg, {
        action: 'unknown',
        zome: 'unknown',
        actualTier,
        requiredTier,
        weightBp: 0,
        reasons,
        expired,
      });
    }

    // Match expired credential pattern
    const expiredMatch = msg.match(/Credential expired at (\d+)/i);
    if (expiredMatch) {
      return new ConsciousnessGateError(msg, {
        action: 'unknown',
        zome: 'unknown',
        actualTier: 'Observer',
        requiredTier: 'Participant',
        weightBp: 0,
        reasons: [msg],
        expired: true,
      });
    }

    return null;
  }
}

// ============================================================================
// Retry-on-expired middleware
// ============================================================================

/**
 * Wrap a Hearth zome call with automatic retry on expired credential.
 *
 * If the call fails with a consciousness gate error where the credential
 * is expired, triggers a credential refresh via the bridge and retries once.
 */
export async function withGateRetry<T>(
  call: () => Promise<T>,
  refreshFn: () => Promise<void>
): Promise<T> {
  try {
    return await call();
  } catch (e) {
    const gateErr = ConsciousnessGateError.fromWasmError(e);
    if (gateErr?.rejection.expired) {
      await refreshFn();
      return await call(); // Single retry after refresh
    }
    throw e;
  }
}
