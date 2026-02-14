/**
 * Biometric Authentication
 *
 * Cross-platform abstraction for biometric authentication
 * on mobile devices.
 */

import type {
  BiometricType,
  BiometricCapability,
  BiometricResult,
  BiometricErrorCode,
} from './types.js';

// =============================================================================
// Biometric Provider Interface
// =============================================================================

/**
 * Interface for platform-specific biometric implementations
 */
export interface BiometricProvider {
  checkCapability(): Promise<BiometricCapability>;
  authenticate(reason: string): Promise<BiometricResult>;
  cancelAuthentication(): void;
}

// =============================================================================
// Biometric Manager
// =============================================================================

export interface BiometricManagerConfig {
  provider?: BiometricProvider;
  fallbackToPin: boolean;
  maxAttempts: number;
  lockoutDuration: number; // ms
}

const DEFAULT_BIOMETRIC_CONFIG: BiometricManagerConfig = {
  fallbackToPin: true,
  maxAttempts: 3,
  lockoutDuration: 30000,
};

/**
 * Manages biometric authentication across platforms
 */
export class BiometricManager {
  private config: BiometricManagerConfig;
  private provider: BiometricProvider;
  private attemptCount: number = 0;
  private lockoutUntil: number = 0;

  constructor(config?: Partial<BiometricManagerConfig>) {
    this.config = { ...DEFAULT_BIOMETRIC_CONFIG, ...config };
    this.provider = config?.provider ?? new SimulatedBiometricProvider();
  }

  /**
   * Check biometric capability
   */
  async checkCapability(): Promise<BiometricCapability> {
    return this.provider.checkCapability();
  }

  /**
   * Check if biometrics are available and enrolled
   */
  async isAvailable(): Promise<boolean> {
    const capability = await this.checkCapability();
    return capability.available && capability.enrolled;
  }

  /**
   * Authenticate with biometrics
   */
  async authenticate(reason: string = 'Confirm your identity'): Promise<BiometricResult> {
    // Check lockout
    if (Date.now() < this.lockoutUntil) {
      const remainingMs = this.lockoutUntil - Date.now();
      return {
        success: false,
        errorCode: 'lockout',
        errorMessage: `Too many attempts. Try again in ${Math.ceil(remainingMs / 1000)} seconds.`,
      };
    }

    // Check availability
    const capability = await this.checkCapability();
    if (!capability.available) {
      return {
        success: false,
        errorCode: 'not_available',
        errorMessage: 'Biometric authentication is not available on this device',
      };
    }

    if (!capability.enrolled) {
      return {
        success: false,
        errorCode: 'not_enrolled',
        errorMessage: 'No biometrics enrolled. Please set up biometrics in device settings.',
      };
    }

    try {
      const result = await this.provider.authenticate(reason);

      if (result.success) {
        this.attemptCount = 0;
      } else {
        this.attemptCount++;

        if (this.attemptCount >= this.config.maxAttempts) {
          this.lockoutUntil = Date.now() + this.config.lockoutDuration;
          this.attemptCount = 0;
          return {
            success: false,
            errorCode: 'lockout',
            errorMessage: `Too many failed attempts. Try again in ${this.config.lockoutDuration / 1000} seconds.`,
          };
        }
      }

      return result;
    } catch (error) {
      return {
        success: false,
        errorCode: 'unknown',
        errorMessage: error instanceof Error ? error.message : 'Authentication failed',
      };
    }
  }

  /**
   * Authenticate with retry logic
   */
  async authenticateWithRetry(
    reason: string,
    maxRetries: number = 2
  ): Promise<BiometricResult> {
    let lastResult: BiometricResult | null = null;

    for (let i = 0; i <= maxRetries; i++) {
      const result = await this.authenticate(reason);
      lastResult = result;

      if (result.success) {
        return result;
      }

      // Don't retry on certain errors
      if (
        result.errorCode === 'user_cancelled' ||
        result.errorCode === 'lockout' ||
        result.errorCode === 'not_available' ||
        result.errorCode === 'not_enrolled'
      ) {
        break;
      }
    }

    return lastResult ?? {
      success: false,
      errorCode: 'unknown',
      errorMessage: 'Authentication failed after retries',
    };
  }

  /**
   * Cancel ongoing authentication
   */
  cancel(): void {
    this.provider.cancelAuthentication();
  }

  /**
   * Reset lockout (should be protected)
   */
  resetLockout(): void {
    this.lockoutUntil = 0;
    this.attemptCount = 0;
  }

  /**
   * Check if currently locked out
   */
  isLockedOut(): boolean {
    return Date.now() < this.lockoutUntil;
  }

  /**
   * Get remaining lockout time in ms
   */
  getLockoutRemaining(): number {
    return Math.max(0, this.lockoutUntil - Date.now());
  }

  /**
   * Get supported biometric types
   */
  async getSupportedTypes(): Promise<BiometricType[]> {
    const capability = await this.checkCapability();
    return capability.types;
  }
}

// =============================================================================
// Simulated Provider (for development/testing)
// =============================================================================

/**
 * Simulated biometric provider for development and testing
 */
export class SimulatedBiometricProvider implements BiometricProvider {
  private simulatedCapability: BiometricCapability = {
    available: true,
    enrolled: true,
    types: ['fingerprint', 'face'],
    securityLevel: 'strong',
  };

  private shouldSucceed: boolean = true;
  private simulatedType: BiometricType = 'fingerprint';

  /**
   * Configure simulation behavior
   */
  configure(options: {
    available?: boolean;
    enrolled?: boolean;
    types?: BiometricType[];
    shouldSucceed?: boolean;
    simulatedType?: BiometricType;
  }): void {
    if (options.available !== undefined) {
      this.simulatedCapability.available = options.available;
    }
    if (options.enrolled !== undefined) {
      this.simulatedCapability.enrolled = options.enrolled;
    }
    if (options.types) {
      this.simulatedCapability.types = options.types;
    }
    if (options.shouldSucceed !== undefined) {
      this.shouldSucceed = options.shouldSucceed;
    }
    if (options.simulatedType) {
      this.simulatedType = options.simulatedType;
    }
  }

  async checkCapability(): Promise<BiometricCapability> {
    // Simulate async check
    await new Promise((resolve) => setTimeout(resolve, 50));
    return { ...this.simulatedCapability };
  }

  async authenticate(_reason: string): Promise<BiometricResult> {
    // Simulate authentication delay
    await new Promise((resolve) => setTimeout(resolve, 500));

    if (this.shouldSucceed) {
      return {
        success: true,
        biometricType: this.simulatedType,
      };
    }

    return {
      success: false,
      errorCode: 'user_cancelled',
      errorMessage: 'User cancelled authentication',
    };
  }

  cancelAuthentication(): void {
    // No-op for simulation
  }
}

// =============================================================================
// React Native Provider (stub for real implementation)
// =============================================================================

/**
 * React Native biometric provider using react-native-biometrics
 * This is a stub - actual implementation would use the native module
 */
export class ReactNativeBiometricProvider implements BiometricProvider {
  async checkCapability(): Promise<BiometricCapability> {
    // In real implementation:
    // const biometrics = new ReactNativeBiometrics();
    // const { available, biometryType } = await biometrics.isSensorAvailable();
    return {
      available: true,
      enrolled: true,
      types: ['fingerprint'],
      securityLevel: 'strong',
    };
  }

  async authenticate(reason: string): Promise<BiometricResult> {
    // In real implementation:
    // const biometrics = new ReactNativeBiometrics();
    // const { success } = await biometrics.simplePrompt({ promptMessage: reason });
    console.log(`Authenticating with reason: ${reason}`);
    return {
      success: true,
      biometricType: 'fingerprint',
    };
  }

  cancelAuthentication(): void {
    // In real implementation:
    // Cancel ongoing prompt
  }
}

// =============================================================================
// Expo Provider (stub for real implementation)
// =============================================================================

/**
 * Expo biometric provider using expo-local-authentication
 * This is a stub - actual implementation would use the Expo module
 */
export class ExpoBiometricProvider implements BiometricProvider {
  async checkCapability(): Promise<BiometricCapability> {
    // In real implementation:
    // import * as LocalAuthentication from 'expo-local-authentication';
    // const hasHardware = await LocalAuthentication.hasHardwareAsync();
    // const isEnrolled = await LocalAuthentication.isEnrolledAsync();
    // const supportedTypes = await LocalAuthentication.supportedAuthenticationTypesAsync();
    return {
      available: true,
      enrolled: true,
      types: ['fingerprint', 'face'],
      securityLevel: 'strong',
    };
  }

  async authenticate(reason: string): Promise<BiometricResult> {
    // In real implementation:
    // import * as LocalAuthentication from 'expo-local-authentication';
    // const result = await LocalAuthentication.authenticateAsync({
    //   promptMessage: reason,
    //   fallbackLabel: 'Use PIN',
    // });
    console.log(`Authenticating with reason: ${reason}`);
    return {
      success: true,
      biometricType: 'fingerprint',
    };
  }

  cancelAuthentication(): void {
    // In real implementation:
    // Cancel ongoing prompt
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a biometric manager
 */
export function createBiometricManager(
  config?: Partial<BiometricManagerConfig>
): BiometricManager {
  return new BiometricManager(config);
}

/**
 * Create a simulated biometric provider for testing
 */
export function createSimulatedBiometricProvider(): SimulatedBiometricProvider {
  return new SimulatedBiometricProvider();
}

/**
 * Helper to convert error code to user-friendly message
 */
export function biometricErrorToMessage(errorCode: BiometricErrorCode): string {
  switch (errorCode) {
    case 'user_cancelled':
      return 'Authentication was cancelled';
    case 'lockout':
      return 'Too many failed attempts. Please try again later.';
    case 'not_enrolled':
      return 'No biometrics enrolled. Please set up biometrics in your device settings.';
    case 'not_available':
      return 'Biometric authentication is not available on this device.';
    case 'hardware_error':
      return 'There was a problem with the biometric sensor. Please try again.';
    case 'timeout':
      return 'Authentication timed out. Please try again.';
    default:
      return 'Authentication failed. Please try again.';
  }
}
