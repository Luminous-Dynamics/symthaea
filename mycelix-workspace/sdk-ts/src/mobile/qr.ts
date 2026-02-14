/**
 * QR Code Utilities
 *
 * Utilities for encoding and decoding QR code payloads
 * for mobile wallet interactions.
 */

import type {
  QRPayloadType,
  QRPayload,
  WalletConnectPayload,
  Permission,
  PresentationRequest,
  RequestedCredential,
  CredentialType,
} from './types.js';
import type { AgentId, HappId } from '../utils/index.js';

// =============================================================================
// QR Payload Builder
// =============================================================================

/**
 * Builder for creating QR code payloads
 */
export class QRPayloadBuilder<T = unknown> {
  private type: QRPayloadType;
  private data: T;
  private signature?: Uint8Array;
  private expiresAt?: number;

  constructor(type: QRPayloadType, data: T) {
    this.type = type;
    this.data = data;
  }

  /**
   * Add signature to payload
   */
  withSignature(signature: Uint8Array): this {
    this.signature = signature;
    return this;
  }

  /**
   * Set expiration time
   */
  expiresIn(ms: number): this {
    this.expiresAt = Date.now() + ms;
    return this;
  }

  /**
   * Set absolute expiration
   */
  expiresAtTime(timestamp: number): this {
    this.expiresAt = timestamp;
    return this;
  }

  /**
   * Build the payload
   */
  build(): QRPayload<T> {
    return {
      type: this.type,
      version: 1,
      data: this.data,
      signature: this.signature,
      expiresAt: this.expiresAt,
    };
  }
}

// =============================================================================
// Encoding & Decoding
// =============================================================================

/**
 * Encode a QR payload to a string for display
 */
export function encodeQRPayload<T>(payload: QRPayload<T>): string {
  // Create a compact representation
  const compact = {
    t: payload.type,
    v: payload.version,
    d: payload.data,
    s: payload.signature ? uint8ArrayToBase64(payload.signature) : undefined,
    e: payload.expiresAt,
  };

  // Encode to base64 JSON
  const json = JSON.stringify(compact);
  return `mycelix://${btoa(json)}`;
}

/**
 * Decode a QR payload from a string
 */
export function decodeQRPayload<T = unknown>(encoded: string): QRPayload<T> | null {
  try {
    // Remove protocol prefix
    const base64 = encoded.replace(/^mycelix:\/\//, '');
    const json = atob(base64);
    const compact = JSON.parse(json);

    return {
      type: compact.t as QRPayloadType,
      version: compact.v,
      data: compact.d as T,
      signature: compact.s ? base64ToUint8Array(compact.s) : undefined,
      expiresAt: compact.e,
    };
  } catch {
    return null;
  }
}

/**
 * Check if a QR payload is expired
 */
export function isPayloadExpired(payload: QRPayload): boolean {
  if (!payload.expiresAt) return false;
  return Date.now() > payload.expiresAt;
}

/**
 * Validate a QR payload structure
 */
export function validateQRPayload(payload: QRPayload): ValidationResult {
  const errors: string[] = [];

  if (!payload.type) {
    errors.push('Missing payload type');
  }

  if (typeof payload.version !== 'number') {
    errors.push('Invalid version');
  }

  if (payload.data === undefined || payload.data === null) {
    errors.push('Missing payload data');
  }

  if (payload.expiresAt && typeof payload.expiresAt !== 'number') {
    errors.push('Invalid expiration timestamp');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// =============================================================================
// Payload Builders
// =============================================================================

/**
 * Create a wallet connect payload
 */
export function createWalletConnectPayload(
  happId: HappId,
  happName: string,
  permissions: Permission[],
  callbackUrl?: string
): QRPayload<WalletConnectPayload> {
  const sessionId = generateSessionId();
  return new QRPayloadBuilder<WalletConnectPayload>('wallet_connect', {
    sessionId,
    happId,
    happName,
    requiredPermissions: permissions,
    callbackUrl,
  })
    .expiresIn(300000) // 5 minutes
    .build();
}

/**
 * Create a credential request payload
 */
export function createCredentialRequestPayload(
  verifier: AgentId,
  verifierName: string,
  requestedCredentials: RequestedCredential[],
  expiresInMs: number = 600000 // 10 minutes
): QRPayload<PresentationRequest> {
  const request: PresentationRequest = {
    requestId: generateRequestId(),
    verifier,
    verifierName,
    requestedCredentials,
    challenge: generateChallenge(),
    createdAt: Date.now(),
    expiresAt: Date.now() + expiresInMs,
  };

  return new QRPayloadBuilder<PresentationRequest>('credential_request', request)
    .expiresIn(expiresInMs)
    .build();
}

/**
 * Create a contact exchange payload
 */
export function createContactExchangePayload(
  agentId: AgentId,
  displayName: string,
  publicKey: Uint8Array,
  metadata?: Record<string, string>
): QRPayload<ContactExchangeData> {
  return new QRPayloadBuilder<ContactExchangeData>('contact_exchange', {
    agentId,
    displayName,
    publicKey: uint8ArrayToBase64(publicKey),
    metadata,
    timestamp: Date.now(),
  })
    .expiresIn(300000) // 5 minutes
    .build();
}

/**
 * Create a payment request payload
 */
export function createPaymentRequestPayload(
  recipient: AgentId,
  amount: number,
  currency: string,
  description?: string,
  reference?: string
): QRPayload<PaymentRequestData> {
  return new QRPayloadBuilder<PaymentRequestData>('payment_request', {
    requestId: generateRequestId(),
    recipient,
    amount,
    currency,
    description,
    reference,
    createdAt: Date.now(),
  })
    .expiresIn(900000) // 15 minutes
    .build();
}

/**
 * Create a deep link payload
 */
export function createDeepLinkPayload(
  path: string,
  params?: Record<string, string>
): QRPayload<DeepLinkData> {
  return new QRPayloadBuilder<DeepLinkData>('deep_link', {
    path,
    params,
  }).build();
}

// =============================================================================
// Credential Request Helpers
// =============================================================================

/**
 * Build a credential request with predicates
 */
export class CredentialRequestBuilder {
  private credentials: RequestedCredential[] = [];

  /**
   * Request a credential type
   */
  request(type: CredentialType, required: boolean = true): this {
    this.credentials.push({
      type,
      required,
    });
    return this;
  }

  /**
   * Request specific claims from a credential
   */
  requestClaims(
    type: CredentialType,
    claims: string[],
    required: boolean = true
  ): this {
    this.credentials.push({
      type,
      required,
      claimsToReveal: claims,
    });
    return this;
  }

  /**
   * Request with predicate (zero-knowledge proof)
   */
  requestWithPredicate(
    type: CredentialType,
    predicates: Array<{
      claim: string;
      operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
      value: unknown;
    }>,
    required: boolean = true
  ): this {
    this.credentials.push({
      type,
      required,
      predicates,
    });
    return this;
  }

  /**
   * Build the list of requested credentials
   */
  build(): RequestedCredential[] {
    return [...this.credentials];
  }
}

// =============================================================================
// Types
// =============================================================================

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface ContactExchangeData {
  agentId: AgentId;
  displayName: string;
  publicKey: string; // Base64 encoded
  metadata?: Record<string, string>;
  timestamp: number;
}

export interface PaymentRequestData {
  requestId: string;
  recipient: AgentId;
  amount: number;
  currency: string;
  description?: string;
  reference?: string;
  createdAt: number;
}

export interface DeepLinkData {
  path: string;
  params?: Record<string, string>;
}

// =============================================================================
// Helper Functions
// =============================================================================

function generateSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function generateRequestId(): string {
  return `req-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function generateChallenge(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return uint8ArrayToBase64(bytes);
}

function uint8ArrayToBase64(arr: Uint8Array): string {
  return btoa(String.fromCharCode(...arr));
}

function base64ToUint8Array(str: string): Uint8Array {
  return new Uint8Array(atob(str).split('').map((c) => c.charCodeAt(0)));
}

// =============================================================================
// QR Scanner Integration
// =============================================================================

/**
 * Result from QR code scanning
 */
export interface QRScanResult<T = unknown> {
  success: boolean;
  payload?: QRPayload<T>;
  rawData?: string;
  error?: string;
}

/**
 * Process a scanned QR code
 */
export function processScannedQR<T = unknown>(rawData: string): QRScanResult<T> {
  // Check if it's a Mycelix QR code
  if (!rawData.startsWith('mycelix://')) {
    return {
      success: false,
      rawData,
      error: 'Not a valid Mycelix QR code',
    };
  }

  const payload = decodeQRPayload<T>(rawData);
  if (!payload) {
    return {
      success: false,
      rawData,
      error: 'Failed to decode QR payload',
    };
  }

  // Validate structure
  const validation = validateQRPayload(payload);
  if (!validation.valid) {
    return {
      success: false,
      rawData,
      payload,
      error: validation.errors.join(', '),
    };
  }

  // Check expiration
  if (isPayloadExpired(payload)) {
    return {
      success: false,
      rawData,
      payload,
      error: 'QR code has expired',
    };
  }

  return {
    success: true,
    payload,
    rawData,
  };
}

/**
 * Get appropriate action for a QR payload type
 */
export function getQRPayloadAction(type: QRPayloadType): string {
  switch (type) {
    case 'wallet_connect':
      return 'Connect to application';
    case 'credential_request':
      return 'Present credential';
    case 'credential_offer':
      return 'Accept credential';
    case 'payment_request':
      return 'Confirm payment';
    case 'contact_exchange':
      return 'Add contact';
    case 'deep_link':
      return 'Open in app';
    default:
      return 'Process QR code';
  }
}
