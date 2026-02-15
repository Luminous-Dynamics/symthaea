/**
 * Mobile Wallet SDK Types
 *
 * Type definitions for building mobile applications
 * that interact with the Mycelix ecosystem.
 */

import type { AgentId, HappId, TrustScore } from '../utils/index.js';

// =============================================================================
// Core Types
// =============================================================================

/**
 * Wallet identifier
 */
export type WalletId = string;

/**
 * Account identifier within a wallet
 */
export type AccountId = string;

/**
 * Key identifier
 */
export type KeyId = string;

/**
 * Supported key types
 */
export type KeyType = 'ed25519' | 'secp256k1' | 'x25519';

/**
 * Wallet status
 */
export type WalletStatus = 'locked' | 'unlocked' | 'uninitialized';

/**
 * Connection status
 */
export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error';

// =============================================================================
// Wallet Types
// =============================================================================

/**
 * Wallet metadata
 */
export interface WalletMetadata {
  walletId: WalletId;
  name: string;
  createdAt: number;
  lastAccessedAt: number;
  accountCount: number;
  backupRequired: boolean;
  version: number;
}

/**
 * Account within a wallet
 */
export interface WalletAccount {
  accountId: AccountId;
  walletId: WalletId;
  name: string;
  agentId: AgentId;
  keyId: KeyId;
  createdAt: number;
  isDefault: boolean;
  trustScore?: TrustScore;
  connectedHapps: HappId[];
}

/**
 * Key material (encrypted when stored)
 */
export interface KeyMaterial {
  keyId: KeyId;
  keyType: KeyType;
  publicKey: Uint8Array;
  encryptedPrivateKey?: Uint8Array;
  derivationPath?: string;
  createdAt: number;
}

/**
 * Wallet configuration
 */
export interface WalletConfig {
  name: string;
  enableBiometrics: boolean;
  autoLockTimeout: number; // ms, 0 = never
  requirePinOnTransaction: boolean;
  backupReminder: boolean;
}

// =============================================================================
// Credential Types
// =============================================================================

/**
 * Stored credential
 */
export interface StoredCredential {
  credentialId: string;
  type: CredentialType;
  issuer: AgentId;
  subject: AgentId;
  issuedAt: number;
  expiresAt?: number;
  claims: Record<string, unknown>;
  signature: Uint8Array;
  revocationStatus: RevocationStatus;
}

/**
 * Credential types
 */
export type CredentialType =
  | 'VerifiableCredential'
  | 'IdentityCredential'
  | 'EducationCredential'
  | 'EmploymentCredential'
  | 'MembershipCredential'
  | 'AchievementCredential'
  | 'ReputationCredential'
  | 'Custom';

/**
 * Revocation status
 */
export type RevocationStatus = 'valid' | 'revoked' | 'suspended' | 'unknown';

/**
 * Credential presentation request
 */
export interface PresentationRequest {
  requestId: string;
  verifier: AgentId;
  verifierName?: string;
  requestedCredentials: RequestedCredential[];
  challenge: string;
  createdAt: number;
  expiresAt: number;
}

/**
 * Specific credential request
 */
export interface RequestedCredential {
  type: CredentialType;
  required: boolean;
  claimsToReveal?: string[];
  predicates?: CredentialPredicate[];
}

/**
 * Predicate for selective disclosure
 */
export interface CredentialPredicate {
  claim: string;
  operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
  value: unknown;
}

// =============================================================================
// Transaction Types
// =============================================================================

/**
 * Transaction to sign
 */
export interface Transaction {
  transactionId: string;
  type: TransactionType;
  from: AgentId;
  to?: AgentId;
  happId: HappId;
  zomeName: string;
  functionName: string;
  payload: unknown;
  createdAt: number;
  status: TransactionStatus;
}

/**
 * Transaction type
 */
export type TransactionType =
  | 'zome_call'
  | 'credential_issue'
  | 'credential_present'
  | 'reputation_update'
  | 'fl_contribution'
  | 'governance_vote';

/**
 * Transaction status
 */
export type TransactionStatus =
  | 'pending'
  | 'signing'
  | 'submitted'
  | 'confirmed'
  | 'failed'
  | 'rejected';

/**
 * Signed transaction
 */
export interface SignedTransaction {
  transaction: Transaction;
  signature: Uint8Array;
  signedAt: number;
  signedBy: AccountId;
}

// =============================================================================
// QR Code Types
// =============================================================================

/**
 * QR code payload types
 */
export type QRPayloadType =
  | 'wallet_connect'
  | 'credential_request'
  | 'credential_offer'
  | 'payment_request'
  | 'contact_exchange'
  | 'deep_link';

/**
 * QR code payload
 */
export interface QRPayload<T = unknown> {
  type: QRPayloadType;
  version: number;
  data: T;
  signature?: Uint8Array;
  expiresAt?: number;
}

/**
 * Wallet connect payload
 */
export interface WalletConnectPayload {
  sessionId: string;
  happId: HappId;
  happName: string;
  requiredPermissions: Permission[];
  callbackUrl?: string;
}

/**
 * Permission types
 */
export type Permission =
  | 'read_profile'
  | 'read_credentials'
  | 'sign_transactions'
  | 'issue_credentials'
  | 'read_reputation'
  | 'participate_fl';

// =============================================================================
// Biometric Types
// =============================================================================

/**
 * Supported biometric types
 */
export type BiometricType = 'fingerprint' | 'face' | 'iris' | 'voice' | 'none';

/**
 * Biometric capability
 */
export interface BiometricCapability {
  available: boolean;
  enrolled: boolean;
  types: BiometricType[];
  securityLevel: 'weak' | 'strong';
}

/**
 * Biometric authentication result
 */
export interface BiometricResult {
  success: boolean;
  biometricType?: BiometricType;
  errorCode?: BiometricErrorCode;
  errorMessage?: string;
}

/**
 * Biometric error codes
 */
export type BiometricErrorCode =
  | 'user_cancelled'
  | 'lockout'
  | 'not_enrolled'
  | 'not_available'
  | 'hardware_error'
  | 'timeout'
  | 'unknown';

// =============================================================================
// Notification Types
// =============================================================================

/**
 * Wallet notification
 */
export interface WalletNotification {
  notificationId: string;
  type: NotificationType;
  title: string;
  body: string;
  data?: Record<string, unknown>;
  createdAt: number;
  readAt?: number;
  actionRequired: boolean;
}

/**
 * Notification types
 */
export type NotificationType =
  | 'credential_received'
  | 'credential_request'
  | 'transaction_pending'
  | 'transaction_confirmed'
  | 'reputation_updated'
  | 'backup_reminder'
  | 'security_alert';

// =============================================================================
// Storage Types
// =============================================================================

/**
 * Secure storage options
 */
export interface SecureStorageOptions {
  useKeychain: boolean;
  useBiometricProtection: boolean;
  accessibleWhenUnlocked: boolean;
}

/**
 * Storage provider interface
 */
export interface StorageProvider {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
  clear(): Promise<void>;
  getAllKeys(): Promise<string[]>;
}

// =============================================================================
// SDK Configuration
// =============================================================================

/**
 * Mobile SDK configuration
 */
export interface MobileSDKConfig {
  conductorUrl?: string;
  storageProvider?: StorageProvider;
  enableBiometrics: boolean;
  defaultAutoLockTimeout: number;
  enablePushNotifications: boolean;
  networkTimeout: number;
  maxRetries: number;
}

/**
 * Default SDK configuration
 */
export const DEFAULT_MOBILE_CONFIG: MobileSDKConfig = {
  enableBiometrics: true,
  defaultAutoLockTimeout: 300000, // 5 minutes
  enablePushNotifications: true,
  networkTimeout: 30000,
  maxRetries: 3,
};
