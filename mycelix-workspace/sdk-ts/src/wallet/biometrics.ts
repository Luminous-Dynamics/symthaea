// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Platform Biometric Providers - Fingerprint, Face ID, WebAuthn
 *
 * Abstracts native biometric authentication across platforms:
 * - iOS: Face ID, Touch ID via LocalAuthentication
 * - Android: BiometricPrompt API
 * - Web: WebAuthn / FIDO2
 * - Desktop: OS keychain integration
 *
 * @example
 * ```typescript
 * // Auto-detect platform
 * const biometrics = await BiometricManager.create();
 *
 * // Check availability
 * if (biometrics.isAvailable) {
 *   const result = await biometrics.authenticate('Unlock your wallet');
 *   if (result.success) {
 *     // User verified!
 *   }
 * }
 * ```
 */

import { BehaviorSubject, type Subscription } from '../reactive/index.js';

// =============================================================================
// Types
// =============================================================================

/** Supported biometric types */
export type BiometricType =
  | 'face' // Face ID, Face Unlock
  | 'fingerprint' // Touch ID, fingerprint
  | 'iris' // Iris scanning
  | 'voice' // Voice recognition
  | 'pin' // PIN fallback
  | 'password' // Password fallback
  | 'webauthn' // WebAuthn/FIDO2
  | 'unknown';

/** Platform detection result */
export type Platform = 'ios' | 'android' | 'web' | 'desktop' | 'unknown';

/** Biometric availability status */
export interface BiometricCapability {
  /** Whether biometrics are available */
  isAvailable: boolean;
  /** Available biometric types */
  types: BiometricType[];
  /** Whether device has secure enclave/TEE */
  hasSecureHardware: boolean;
  /** Whether biometrics are enrolled */
  isEnrolled: boolean;
  /** Platform-specific details */
  platform: Platform;
  /** Reason if unavailable */
  unavailableReason?: BiometricUnavailableReason;
}

/** Reasons biometrics might be unavailable */
export type BiometricUnavailableReason =
  | 'not_supported' // Device doesn't support biometrics
  | 'not_enrolled' // No biometrics enrolled
  | 'locked_out' // Too many failed attempts
  | 'passcode_not_set' // Device passcode required first
  | 'permission_denied' // User denied permission
  | 'unknown';

/** Authentication result */
export interface BiometricAuthResult {
  /** Whether authentication succeeded */
  success: boolean;
  /** The biometric type used */
  type?: BiometricType;
  /** Error if failed */
  error?: BiometricError;
  /** Credential data (for WebAuthn) */
  credential?: PublicKeyCredential;
}

/** Biometric error types */
export interface BiometricError {
  code: BiometricErrorCode;
  message: string;
}

export type BiometricErrorCode =
  | 'user_cancelled' // User cancelled authentication
  | 'failed' // Authentication failed
  | 'locked_out' // Too many attempts
  | 'not_available' // Biometrics not available
  | 'not_enrolled' // No biometrics enrolled
  | 'hardware_unavailable' // Hardware busy or unavailable
  | 'timeout' // Authentication timed out
  | 'unknown';

/** Configuration for biometric authentication */
export interface BiometricConfig {
  /** Prompt text shown to user */
  reason?: string;
  /** Title for the dialog */
  title?: string;
  /** Subtitle for Android */
  subtitle?: string;
  /** Allow device PIN/password as fallback? */
  allowFallback?: boolean;
  /** Timeout in milliseconds */
  timeoutMs?: number;
  /** Cancel button text */
  cancelText?: string;
  /** Confirmation required (Android) */
  confirmationRequired?: boolean;
}

/** Biometric state (reactive) */
export interface BiometricState {
  /** Current capability */
  capability: BiometricCapability;
  /** Whether authentication is in progress */
  isAuthenticating: boolean;
  /** Last authentication result */
  lastResult?: BiometricAuthResult;
}

/** Provider interface for platform-specific implementations */
export interface BiometricProvider {
  /** Get platform capabilities */
  getCapability(): Promise<BiometricCapability>;
  /** Perform authentication */
  authenticate(config?: BiometricConfig): Promise<BiometricAuthResult>;
  /** Cancel ongoing authentication */
  cancel(): void;
}

// =============================================================================
// Platform Detection
// =============================================================================

/**
 * Detect the current platform
 */
export function detectPlatform(): Platform {
  if (typeof window === 'undefined') {
    return 'unknown';
  }

  const userAgent = navigator.userAgent.toLowerCase();

  // Check for iOS
  if (/iphone|ipad|ipod/.test(userAgent)) {
    return 'ios';
  }

  // Check for Android
  if (/android/.test(userAgent)) {
    return 'android';
  }

  // Check for Electron/Tauri (desktop)
  if (
    typeof (window as unknown as { __TAURI__?: unknown }).__TAURI__ !== 'undefined' ||
    typeof (window as unknown as { electron?: unknown }).electron !== 'undefined'
  ) {
    return 'desktop';
  }

  // Default to web
  return 'web';
}

// =============================================================================
// Web Provider (WebAuthn)
// =============================================================================

/**
 * WebAuthn-based biometric provider for web browsers
 */
export class WebAuthnProvider implements BiometricProvider {
  private authenticating = false;
  private abortController?: AbortController;

  async getCapability(): Promise<BiometricCapability> {
    // Check if WebAuthn is available
    if (typeof window === 'undefined' || !window.PublicKeyCredential) {
      return {
        isAvailable: false,
        types: [],
        hasSecureHardware: false,
        isEnrolled: false,
        platform: 'web',
        unavailableReason: 'not_supported',
      };
    }

    try {
      // Check platform authenticator availability
      const available = await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable();

      // Check conditional mediation (passkeys)
      let hasPasskeys = false;
      if (typeof PublicKeyCredential.isConditionalMediationAvailable === 'function') {
        hasPasskeys = await PublicKeyCredential.isConditionalMediationAvailable();
      }

      // If passkeys are available, add that as an additional type
      const types: BiometricType[] = available
        ? hasPasskeys
          ? ['webauthn', 'fingerprint'] // Passkeys often indicate Touch ID/fingerprint available
          : ['webauthn']
        : [];

      return {
        isAvailable: available,
        types,
        hasSecureHardware: available,
        isEnrolled: available, // Can't easily check enrollment
        platform: 'web',
        unavailableReason: available ? undefined : 'not_supported',
      };
    } catch {
      return {
        isAvailable: false,
        types: [],
        hasSecureHardware: false,
        isEnrolled: false,
        platform: 'web',
        unavailableReason: 'unknown',
      };
    }
  }

  async authenticate(config?: BiometricConfig): Promise<BiometricAuthResult> {
    if (this.authenticating) {
      return {
        success: false,
        error: { code: 'hardware_unavailable', message: 'Authentication already in progress' },
      };
    }

    this.authenticating = true;
    this.abortController = new AbortController();

    try {
      // Generate a challenge (in real app, this comes from server)
      const challenge = new Uint8Array(32);
      crypto.getRandomValues(challenge);

      const options: PublicKeyCredentialRequestOptions = {
        challenge,
        timeout: config?.timeoutMs ?? 60000,
        userVerification: 'required',
        rpId: typeof window !== 'undefined' ? window.location.hostname : 'localhost',
      };

      const credential = (await navigator.credentials.get({
        publicKey: options,
        signal: this.abortController.signal,
      })) as PublicKeyCredential | null;

      if (credential) {
        return {
          success: true,
          type: 'webauthn',
          credential,
        };
      } else {
        return {
          success: false,
          error: { code: 'failed', message: 'No credential returned' },
        };
      }
    } catch (error) {
      const errorCode = mapWebAuthnError(error);
      return {
        success: false,
        error: {
          code: errorCode,
          message: error instanceof Error ? error.message : 'Authentication failed',
        },
      };
    } finally {
      this.authenticating = false;
      this.abortController = undefined;
    }
  }

  cancel(): void {
    if (this.abortController) {
      this.abortController.abort();
    }
  }
}

function mapWebAuthnError(error: unknown): BiometricErrorCode {
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'AbortError':
        return 'user_cancelled';
      case 'NotAllowedError':
        return 'user_cancelled';
      case 'InvalidStateError':
        return 'not_enrolled';
      case 'SecurityError':
        return 'not_available';
      default:
        return 'unknown';
    }
  }
  return 'unknown';
}

// =============================================================================
// Native Provider Bridge (for React Native / Capacitor)
// =============================================================================

/** Bridge interface for native biometric modules */
export interface NativeBiometricBridge {
  checkAvailability(): Promise<{
    available: boolean;
    biometryType: string;
    hasEnrolledBiometry: boolean;
    reason?: string;
  }>;
  authenticate(options: {
    reason: string;
    title?: string;
    subtitle?: string;
    fallbackEnabled?: boolean;
    cancelTitle?: string;
  }): Promise<{ success: boolean; error?: string }>;
  cancel(): void;
}

/**
 * Provider that bridges to native biometric modules
 * Works with react-native-biometrics, @capacitor/biometrics, etc.
 */
export class NativeBridgeProvider implements BiometricProvider {
  private bridge: NativeBiometricBridge;
  private platform: Platform;

  constructor(bridge: NativeBiometricBridge, platform: Platform) {
    this.bridge = bridge;
    this.platform = platform;
  }

  async getCapability(): Promise<BiometricCapability> {
    try {
      const result = await this.bridge.checkAvailability();

      const types: BiometricType[] = [];
      if (result.biometryType) {
        const type = mapNativeBiometryType(result.biometryType);
        if (type) types.push(type);
      }

      return {
        isAvailable: result.available,
        types,
        hasSecureHardware: result.available, // Native assumes secure hardware
        isEnrolled: result.hasEnrolledBiometry,
        platform: this.platform,
        unavailableReason: result.reason as BiometricUnavailableReason | undefined,
      };
    } catch {
      return {
        isAvailable: false,
        types: [],
        hasSecureHardware: false,
        isEnrolled: false,
        platform: this.platform,
        unavailableReason: 'unknown',
      };
    }
  }

  async authenticate(config?: BiometricConfig): Promise<BiometricAuthResult> {
    try {
      const result = await this.bridge.authenticate({
        reason: config?.reason ?? 'Authenticate to continue',
        title: config?.title,
        subtitle: config?.subtitle,
        fallbackEnabled: config?.allowFallback ?? false,
        cancelTitle: config?.cancelText,
      });

      if (result.success) {
        return { success: true, type: 'fingerprint' }; // Default, could be enhanced
      } else {
        return {
          success: false,
          error: {
            code: mapNativeError(result.error),
            message: result.error ?? 'Authentication failed',
          },
        };
      }
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'unknown',
          message: error instanceof Error ? error.message : 'Authentication failed',
        },
      };
    }
  }

  cancel(): void {
    this.bridge.cancel();
  }
}

function mapNativeBiometryType(type: string): BiometricType | null {
  const lower = type.toLowerCase();
  if (lower.includes('face')) return 'face';
  if (lower.includes('touch') || lower.includes('fingerprint')) return 'fingerprint';
  if (lower.includes('iris')) return 'iris';
  return null;
}

function mapNativeError(error?: string): BiometricErrorCode {
  if (!error) return 'unknown';
  const lower = error.toLowerCase();
  if (lower.includes('cancel')) return 'user_cancelled';
  if (lower.includes('lockout')) return 'locked_out';
  if (lower.includes('not enrolled')) return 'not_enrolled';
  if (lower.includes('not available')) return 'not_available';
  if (lower.includes('timeout')) return 'timeout';
  return 'failed';
}

// =============================================================================
// Mock Provider (for testing)
// =============================================================================

/**
 * Mock provider for testing and development
 */
export class MockBiometricProvider implements BiometricProvider {
  private shouldSucceed: boolean;
  private delay: number;
  private availableTypes: BiometricType[];

  constructor(options?: {
    shouldSucceed?: boolean;
    delay?: number;
    availableTypes?: BiometricType[];
  }) {
    this.shouldSucceed = options?.shouldSucceed ?? true;
    this.delay = options?.delay ?? 500;
    this.availableTypes = options?.availableTypes ?? ['fingerprint'];
  }

  async getCapability(): Promise<BiometricCapability> {
    return {
      isAvailable: this.availableTypes.length > 0,
      types: this.availableTypes,
      hasSecureHardware: true,
      isEnrolled: true,
      platform: 'unknown',
    };
  }

  async authenticate(): Promise<BiometricAuthResult> {
    await new Promise((resolve) => setTimeout(resolve, this.delay));

    if (this.shouldSucceed) {
      return {
        success: true,
        type: this.availableTypes[0] ?? 'fingerprint',
      };
    } else {
      return {
        success: false,
        error: { code: 'failed', message: 'Mock authentication failed' },
      };
    }
  }

  cancel(): void {
    // No-op for mock
  }

  /** Configure mock behavior */
  configure(options: { shouldSucceed?: boolean; delay?: number }): void {
    if (options.shouldSucceed !== undefined) this.shouldSucceed = options.shouldSucceed;
    if (options.delay !== undefined) this.delay = options.delay;
  }
}

// =============================================================================
// Biometric Manager
// =============================================================================

/**
 * Unified biometric authentication manager
 */
export class BiometricManager {
  private provider: BiometricProvider;
  private _state$: BehaviorSubject<BiometricState>;

  private constructor(provider: BiometricProvider, initialCapability: BiometricCapability) {
    this.provider = provider;
    this._state$ = new BehaviorSubject<BiometricState>({
      capability: initialCapability,
      isAuthenticating: false,
    });
  }

  // ===========================================================================
  // Factory Methods
  // ===========================================================================

  /**
   * Create a BiometricManager with auto-detected platform
   */
  static async create(): Promise<BiometricManager> {
    const platform = detectPlatform();
    let provider: BiometricProvider;

    switch (platform) {
      case 'web':
        provider = new WebAuthnProvider();
        break;
      // For native platforms, user needs to provide the bridge
      default:
        provider = new WebAuthnProvider(); // Fallback to WebAuthn
    }

    const capability = await provider.getCapability();
    return new BiometricManager(provider, capability);
  }

  /**
   * Create with a specific provider
   */
  static async createWithProvider(provider: BiometricProvider): Promise<BiometricManager> {
    const capability = await provider.getCapability();
    return new BiometricManager(provider, capability);
  }

  /**
   * Create with native bridge (for React Native, Capacitor, etc.)
   */
  static async createWithNativeBridge(
    bridge: NativeBiometricBridge,
    platform?: Platform
  ): Promise<BiometricManager> {
    const detectedPlatform = platform ?? detectPlatform();
    const provider = new NativeBridgeProvider(bridge, detectedPlatform);
    const capability = await provider.getCapability();
    return new BiometricManager(provider, capability);
  }

  /**
   * Create mock for testing
   */
  static async createMock(options?: {
    shouldSucceed?: boolean;
    delay?: number;
    availableTypes?: BiometricType[];
  }): Promise<BiometricManager> {
    const provider = new MockBiometricProvider(options);
    const capability = await provider.getCapability();
    return new BiometricManager(provider, capability);
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable state */
  get state$(): BehaviorSubject<BiometricState> {
    return this._state$;
  }

  /** Current state */
  get state(): BiometricState {
    return this._state$.value;
  }

  /** Whether biometrics are available */
  get isAvailable(): boolean {
    return this._state$.value.capability.isAvailable;
  }

  /** Available biometric types */
  get availableTypes(): BiometricType[] {
    return this._state$.value.capability.types;
  }

  /** Whether authentication is in progress */
  get isAuthenticating(): boolean {
    return this._state$.value.isAuthenticating;
  }

  /** Subscribe to state changes */
  subscribe(observer: (state: BiometricState) => void): Subscription {
    return this._state$.subscribe(observer);
  }

  // ===========================================================================
  // Core Methods
  // ===========================================================================

  /**
   * Refresh capability information
   */
  async refreshCapability(): Promise<BiometricCapability> {
    const capability = await this.provider.getCapability();
    this._state$.next({
      ...this._state$.value,
      capability,
    });
    return capability;
  }

  /**
   * Authenticate the user
   */
  async authenticate(reasonOrConfig?: string | BiometricConfig): Promise<BiometricAuthResult> {
    if (this._state$.value.isAuthenticating) {
      return {
        success: false,
        error: { code: 'hardware_unavailable', message: 'Authentication already in progress' },
      };
    }

    if (!this._state$.value.capability.isAvailable) {
      return {
        success: false,
        error: {
          code: 'not_available',
          message:
            this._state$.value.capability.unavailableReason ?? 'Biometrics not available',
        },
      };
    }

    const config: BiometricConfig =
      typeof reasonOrConfig === 'string' ? { reason: reasonOrConfig } : reasonOrConfig ?? {};

    this._state$.next({
      ...this._state$.value,
      isAuthenticating: true,
    });

    try {
      const result = await this.provider.authenticate(config);

      this._state$.next({
        ...this._state$.value,
        isAuthenticating: false,
        lastResult: result,
      });

      return result;
    } catch (error) {
      const result: BiometricAuthResult = {
        success: false,
        error: {
          code: 'unknown',
          message: error instanceof Error ? error.message : 'Authentication failed',
        },
      };

      this._state$.next({
        ...this._state$.value,
        isAuthenticating: false,
        lastResult: result,
      });

      return result;
    }
  }

  /**
   * Cancel ongoing authentication
   */
  cancel(): void {
    this.provider.cancel();
    this._state$.next({
      ...this._state$.value,
      isAuthenticating: false,
    });
  }

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  /**
   * Check if a specific biometric type is available
   */
  hasType(type: BiometricType): boolean {
    return this._state$.value.capability.types.includes(type);
  }

  /**
   * Get the primary (preferred) biometric type
   */
  get primaryType(): BiometricType | undefined {
    const types = this._state$.value.capability.types;
    // Priority: face > fingerprint > other
    if (types.includes('face')) return 'face';
    if (types.includes('fingerprint')) return 'fingerprint';
    return types[0];
  }

  /**
   * Get a human-readable description of available biometrics
   */
  getDescription(): string {
    const types = this._state$.value.capability.types;
    if (types.length === 0) return 'No biometrics available';

    const names = types.map((t) => {
      switch (t) {
        case 'face':
          return 'Face ID';
        case 'fingerprint':
          return 'Fingerprint';
        case 'iris':
          return 'Iris';
        case 'webauthn':
          return 'Security Key';
        default:
          return t;
      }
    });

    return names.join(' or ');
  }
}

// =============================================================================
// Utilities
// =============================================================================

/**
 * Get icon name for a biometric type
 */
export function getBiometricIcon(type: BiometricType): string {
  switch (type) {
    case 'face':
      return 'face-id';
    case 'fingerprint':
      return 'fingerprint';
    case 'iris':
      return 'eye';
    case 'voice':
      return 'microphone';
    case 'webauthn':
      return 'key';
    case 'pin':
      return 'keypad';
    case 'password':
      return 'lock';
    default:
      return 'shield';
  }
}

/**
 * Get display name for a biometric type
 */
export function getBiometricName(type: BiometricType): string {
  switch (type) {
    case 'face':
      return 'Face ID';
    case 'fingerprint':
      return 'Fingerprint';
    case 'iris':
      return 'Iris Scan';
    case 'voice':
      return 'Voice Recognition';
    case 'webauthn':
      return 'Security Key';
    case 'pin':
      return 'PIN';
    case 'password':
      return 'Password';
    default:
      return 'Biometric';
  }
}

// =============================================================================
// Factory
// =============================================================================

/**
 * Create a biometric manager (convenience function)
 */
export async function createBiometricManager(): Promise<BiometricManager> {
  return BiometricManager.create();
}
