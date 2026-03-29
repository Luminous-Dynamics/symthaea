// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mobile Wallet SDK
 *
 * A comprehensive SDK for building mobile applications that interact
 * with the Mycelix ecosystem. Includes wallet management, biometric
 * authentication, credential storage, and QR code utilities.
 *
 * @example Basic Wallet Setup
 * ```typescript
 * import { mobile } from '@mycelix/sdk';
 *
 * // Create wallet manager
 * const wallet = mobile.createWalletManager();
 *
 * // Create a new wallet
 * const metadata = await wallet.createWallet('My Wallet', '123456', {
 *   enableBiometrics: true,
 *   autoLockTimeout: 300000, // 5 minutes
 * });
 *
 * console.log(`Wallet created: ${metadata.walletId}`);
 *
 * // Get default account
 * const account = wallet.getDefaultAccount();
 * console.log(`Agent ID: ${account.agentId}`);
 * ```
 *
 * @example Biometric Authentication
 * ```typescript
 * import { mobile } from '@mycelix/sdk';
 *
 * const biometrics = mobile.createBiometricManager();
 *
 * // Check if available
 * const available = await biometrics.isAvailable();
 *
 * if (available) {
 *   const result = await biometrics.authenticate('Confirm transaction');
 *   if (result.success) {
 *     console.log(`Authenticated with: ${result.biometricType}`);
 *   }
 * }
 * ```
 *
 * @example QR Code Operations
 * ```typescript
 * import { mobile } from '@mycelix/sdk';
 *
 * // Create a wallet connect QR code
 * const payload = mobile.createWalletConnectPayload(
 *   'happ-id',
 *   'My Application',
 *   ['read_profile', 'sign_transactions'],
 *   'https://app.example.com/callback'
 * );
 *
 * const qrData = mobile.encodeQRPayload(payload);
 * console.log('Show QR code with:', qrData);
 *
 * // Process scanned QR
 * const scanResult = mobile.processScannedQR(scannedData);
 * if (scanResult.success) {
 *   const action = mobile.getQRPayloadAction(scanResult.payload.type);
 *   console.log(`Action: ${action}`);
 * }
 * ```
 *
 * @example Credential Request
 * ```typescript
 * import { mobile } from '@mycelix/sdk';
 *
 * // Build a credential request
 * const credentials = new mobile.CredentialRequestBuilder()
 *   .request('IdentityCredential', true)
 *   .requestClaims('EducationCredential', ['degree', 'institution'], true)
 *   .requestWithPredicate('ReputationCredential', [
 *     { claim: 'trustScore', operator: '>=', value: 0.7 }
 *   ], false)
 *   .build();
 *
 * const request = mobile.createCredentialRequestPayload(
 *   'verifier-agent-id',
 *   'Employer Portal',
 *   credentials
 * );
 *
 * const qrData = mobile.encodeQRPayload(request);
 * ```
 *
 * @example Signing Transactions
 * ```typescript
 * import { mobile } from '@mycelix/sdk';
 *
 * const wallet = mobile.createWalletManager();
 * await wallet.unlock('123456');
 *
 * const account = wallet.getDefaultAccount();
 * const data = new TextEncoder().encode('transaction payload');
 *
 * // Sign with account's key
 * const signature = await wallet.sign(account.accountId, data);
 *
 * // Verify
 * const publicKey = wallet.getPublicKey(account.accountId);
 * const valid = await wallet.verify(publicKey!, data, signature);
 * console.log(`Signature valid: ${valid}`);
 * ```
 *
 * @module mobile
 */

// Types
export {
  type WalletId,
  type AccountId,
  type KeyId,
  type KeyType,
  type WalletStatus,
  type ConnectionStatus,
  type WalletMetadata,
  type WalletAccount,
  type KeyMaterial,
  type WalletConfig,
  type StoredCredential,
  type CredentialType,
  type RevocationStatus,
  type PresentationRequest,
  type RequestedCredential,
  type CredentialPredicate,
  type Transaction,
  type TransactionType,
  type TransactionStatus,
  type SignedTransaction,
  type QRPayloadType,
  type QRPayload,
  type WalletConnectPayload,
  type Permission,
  type BiometricType,
  type BiometricCapability,
  type BiometricResult,
  type BiometricErrorCode,
  type WalletNotification,
  type NotificationType,
  type SecureStorageOptions,
  type StorageProvider,
  type MobileSDKConfig,
  DEFAULT_MOBILE_CONFIG,
} from './types.js';

// Wallet Manager
export {
  WalletManager,
  createWalletManager,
  type WalletManagerConfig,
} from './wallet.js';

// Biometric Authentication
export {
  BiometricManager,
  SimulatedBiometricProvider,
  ReactNativeBiometricProvider,
  ExpoBiometricProvider,
  createBiometricManager,
  createSimulatedBiometricProvider,
  biometricErrorToMessage,
  type BiometricProvider,
  type BiometricManagerConfig,
} from './biometrics.js';

// QR Code Utilities
export {
  QRPayloadBuilder,
  CredentialRequestBuilder,
  encodeQRPayload,
  decodeQRPayload,
  isPayloadExpired,
  validateQRPayload,
  createWalletConnectPayload,
  createCredentialRequestPayload,
  createContactExchangePayload,
  createPaymentRequestPayload,
  createDeepLinkPayload,
  processScannedQR,
  getQRPayloadAction,
  type ValidationResult,
  type ContactExchangeData,
  type PaymentRequestData,
  type DeepLinkData,
  type QRScanResult,
} from './qr.js';
