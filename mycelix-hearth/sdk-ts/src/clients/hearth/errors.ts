// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Typed error class for Hearth SDK zome call failures.
 *
 * Wraps raw Holochain/WASM errors with structured context:
 * the zome and function that failed, plus the original error.
 */
export class HearthError extends Error {
  readonly code: HearthErrorCode;
  readonly zome: string;
  readonly fnName: string;
  readonly cause: unknown;

  constructor(opts: {
    code: HearthErrorCode;
    message: string;
    zome: string;
    fnName: string;
    cause?: unknown;
  }) {
    super(opts.message);
    this.name = 'HearthError';
    this.code = opts.code;
    this.zome = opts.zome;
    this.fnName = opts.fnName;
    this.cause = opts.cause;
  }
}

export type HearthErrorCode =
  | 'ZOME_CALL_FAILED'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'VALIDATION_FAILED'
  | 'NETWORK_ERROR';

/**
 * Classify a raw Holochain error into a HearthErrorCode.
 */
export function classifyError(err: unknown): HearthErrorCode {
  const msg = String(err).toLowerCase();
  if (msg.includes('not found') || msg.includes('no entry')) return 'NOT_FOUND';
  if (msg.includes('unauthorized') || msg.includes('not a member') || msg.includes('not authorized'))
    return 'UNAUTHORIZED';
  if (msg.includes('invalid') || msg.includes('validation')) return 'VALIDATION_FAILED';
  if (msg.includes('network') || msg.includes('timeout') || msg.includes('connection'))
    return 'NETWORK_ERROR';
  return 'ZOME_CALL_FAILED';
}
