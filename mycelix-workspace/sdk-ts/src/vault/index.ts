/**
 * MycelixVault - Secure Key Management Layer
 *
 * The Vault is the "Pen" in the Glass-Top architecture:
 * - Holds cryptographic keys securely
 * - Handles biometric/PIN authentication
 * - Signs transactions when requested
 * - Never exposes raw private keys
 *
 * Users interact with the unified Wallet facade above.
 * The Vault works invisibly below the glass.
 *
 * @example
 * ```typescript
 * // Create vault (happens once during onboarding)
 * const vault = await MycelixVault.create({
 *   name: 'My Vault',
 *   biometricEnabled: true,
 * });
 *
 * // Unlock with biometrics (invisible to user via FaceID/TouchID)
 * await vault.unlock({ biometric: true });
 *
 * // Sign a transaction (the Wallet facade calls this)
 * const signature = await vault.sign(transactionBytes);
 * ```
 */

import { BehaviorSubject, type Subscription } from '../reactive/index.js';

// =============================================================================
// Types
// =============================================================================

/** Unique identifier for a vault */
export type VaultId = string & { readonly __brand: 'VaultId' };

/** Unique identifier for an account within a vault */
export type AccountId = string & { readonly __brand: 'AccountId' };

/** Unique identifier for a key */
export type KeyId = string & { readonly __brand: 'KeyId' };

/** Vault lifecycle states */
export type VaultStatus =
  | 'uninitialized' // No vault exists
  | 'locked' // Vault exists but is locked
  | 'unlocking' // Authentication in progress
  | 'unlocked' // Ready to sign
  | 'error'; // Something went wrong

/** Authentication methods */
export type AuthMethod = 'pin' | 'biometric' | 'recovery';

/** Key types supported.
 *
 * Aligned with Rust `AlgorithmId` enum in `mycelix-crypto`.
 * PQC types use FIPS 204/205 standard names.
 */
export type KeyType =
  | 'ed25519'                       // Classical Ed25519
  | 'secp256k1'                     // Bitcoin/Ethereum compatibility
  | 'bls12381'                      // BLS signatures
  | 'ml-dsa-65'                     // FIPS 204 ML-DSA-65 (post-quantum)
  | 'ml-dsa-87'                     // FIPS 204 ML-DSA-87 (post-quantum)
  | 'slh-dsa-sha2-128s'             // FIPS 205 SLH-DSA (hash-based PQ)
  | 'hybrid-ed25519-ml-dsa-65'      // Hybrid transition key
  | 'ml-kem-768'                    // FIPS 203 KEM (PQ key agreement)
  | 'ml-kem-1024';                  // FIPS 203 KEM (PQ key agreement)

/** Vault metadata (safe to store unencrypted) */
export interface VaultMetadata {
  vaultId: VaultId;
  name: string;
  createdAt: number;
  lastAccessedAt: number;
  accountCount: number;
  biometricEnabled: boolean;
  autoLockTimeout: number; // milliseconds, 0 = never
  backupRequired: boolean; // true if not backed up yet
}

/** Account within a vault */
export interface VaultAccount {
  accountId: AccountId;
  name: string;
  agentId: string; // Holochain agent ID (public key)
  keyId: KeyId;
  keyType: KeyType;
  isDefault: boolean;
  createdAt: number;
  /** Profile info (hydrated from Identity hApp) */
  profile?: {
    nickname?: string;
    avatar?: string;
    did?: string;
  };
}

/** Key material (encrypted at rest) */
interface KeyMaterial {
  keyId: KeyId;
  keyType: KeyType;
  publicKey: Uint8Array;
  encryptedPrivateKey: Uint8Array; // AES-GCM encrypted
  salt: Uint8Array;
  iv: Uint8Array;
}

/** Vault state (reactive) */
export interface VaultState {
  status: VaultStatus;
  metadata: VaultMetadata | null;
  accounts: VaultAccount[];
  currentAccountId: AccountId | null;
  error: string | null;
}

/** Configuration for creating a vault */
export interface CreateVaultConfig {
  name: string;
  pin?: string; // Required if biometric not enabled
  biometricEnabled?: boolean;
  autoLockTimeout?: number; // Default: 5 minutes
}

/** Options for unlocking the vault */
export interface UnlockOptions {
  pin?: string;
  biometric?: boolean;
  recoveryPhrase?: string[];
}

/** Storage provider interface (platform-specific) */
export interface VaultStorageProvider {
  // Metadata (unencrypted)
  getMetadata(): Promise<VaultMetadata | null>;
  setMetadata(metadata: VaultMetadata): Promise<void>;
  deleteMetadata(): Promise<void>;

  // Encrypted data
  getEncryptedData(key: string): Promise<Uint8Array | null>;
  setEncryptedData(key: string, data: Uint8Array): Promise<void>;
  deleteEncryptedData(key: string): Promise<void>;

  // Biometric
  isBiometricAvailable(): Promise<boolean>;
  authenticateBiometric(reason: string): Promise<boolean>;
  storeBiometricKey(key: Uint8Array): Promise<void>;
  retrieveBiometricKey(): Promise<Uint8Array | null>;
}

/** Biometric provider interface */
export interface BiometricProvider {
  isAvailable(): Promise<boolean>;
  authenticate(reason: string): Promise<boolean>;
  storeSecret(key: string, secret: Uint8Array): Promise<void>;
  retrieveSecret(key: string): Promise<Uint8Array | null>;
}

// =============================================================================
// Default In-Memory Storage (for testing/development)
// =============================================================================

class InMemoryVaultStorage implements VaultStorageProvider {
  private metadata: VaultMetadata | null = null;
  private encryptedData: Map<string, Uint8Array> = new Map();
  private biometricKey: Uint8Array | null = null;

  async getMetadata(): Promise<VaultMetadata | null> {
    return this.metadata;
  }

  async setMetadata(metadata: VaultMetadata): Promise<void> {
    this.metadata = metadata;
  }

  async deleteMetadata(): Promise<void> {
    this.metadata = null;
  }

  async getEncryptedData(key: string): Promise<Uint8Array | null> {
    return this.encryptedData.get(key) ?? null;
  }

  async setEncryptedData(key: string, data: Uint8Array): Promise<void> {
    this.encryptedData.set(key, data);
  }

  async deleteEncryptedData(key: string): Promise<void> {
    this.encryptedData.delete(key);
  }

  async isBiometricAvailable(): Promise<boolean> {
    return false; // In-memory doesn't support biometrics
  }

  async authenticateBiometric(_reason: string): Promise<boolean> {
    return false;
  }

  async storeBiometricKey(key: Uint8Array): Promise<void> {
    this.biometricKey = key;
  }

  async retrieveBiometricKey(): Promise<Uint8Array | null> {
    return this.biometricKey;
  }
}

// =============================================================================
// Cryptographic Utilities
// =============================================================================

/**
 * Derive encryption key from PIN using PBKDF2
 */
async function deriveKeyFromPin(pin: string, salt: Uint8Array): Promise<CryptoKey> {
  const encoder = new TextEncoder();
  const pinBytes = encoder.encode(pin);

  const keyMaterial = await crypto.subtle.importKey('raw', pinBytes, 'PBKDF2', false, [
    'deriveBits',
    'deriveKey',
  ]);

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt as BufferSource,
      iterations: 100000,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Encrypt data with AES-GCM
 */
async function encryptData(
  data: Uint8Array,
  key: CryptoKey
): Promise<{ encrypted: Uint8Array; iv: Uint8Array }> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: iv as BufferSource }, key, data as BufferSource);
  return { encrypted: new Uint8Array(encrypted), iv };
}

/**
 * Decrypt data with AES-GCM
 */
async function decryptData(encrypted: Uint8Array, key: CryptoKey, iv: Uint8Array): Promise<Uint8Array> {
  const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: iv as BufferSource }, key, encrypted as BufferSource);
  return new Uint8Array(decrypted);
}

/**
 * Generate a new Ed25519 keypair
 */
async function generateEd25519Keypair(): Promise<{ publicKey: Uint8Array; privateKey: Uint8Array }> {
  const keypair = await crypto.subtle.generateKey(
    {
      name: 'Ed25519',
    },
    true,
    ['sign', 'verify']
  );

  const publicKey = new Uint8Array(await crypto.subtle.exportKey('raw', keypair.publicKey));
  const privateKey = new Uint8Array(await crypto.subtle.exportKey('pkcs8', keypair.privateKey));

  return { publicKey, privateKey };
}

/**
 * Sign data with Ed25519
 */
async function signEd25519(data: Uint8Array, privateKeyPkcs8: Uint8Array): Promise<Uint8Array> {
  const privateKey = await crypto.subtle.importKey('pkcs8', privateKeyPkcs8 as BufferSource, { name: 'Ed25519' }, false, [
    'sign',
  ]);

  const signature = await crypto.subtle.sign('Ed25519', privateKey, data as BufferSource);
  return new Uint8Array(signature);
}

/**
 * Verify Ed25519 signature
 */
async function verifyEd25519(
  data: Uint8Array,
  signature: Uint8Array,
  publicKey: Uint8Array
): Promise<boolean> {
  const key = await crypto.subtle.importKey('raw', publicKey as BufferSource, { name: 'Ed25519' }, false, ['verify']);
  return crypto.subtle.verify('Ed25519', key, signature as BufferSource, data as BufferSource);
}

/**
 * Generate a BIP39-style recovery phrase (simplified)
 */
function generateRecoveryPhrase(): string[] {
  // In production, use a proper BIP39 wordlist
  const words = [
    'abandon', 'ability', 'able', 'about', 'above', 'absent', 'absorb', 'abstract',
    'absurd', 'abuse', 'access', 'accident', 'account', 'accuse', 'achieve', 'acid',
    'acoustic', 'acquire', 'across', 'act', 'action', 'actor', 'actress', 'actual',
    // ... truncated for brevity, would have 2048 words
  ];

  const phrase: string[] = [];
  const entropy = crypto.getRandomValues(new Uint8Array(16)); // 128 bits = 12 words

  for (let i = 0; i < 12; i++) {
    const index = (entropy[i] * 256 + (entropy[(i + 1) % 16] ?? 0)) % words.length;
    phrase.push(words[index]);
  }

  return phrase;
}

/**
 * Generate unique IDs
 */
function generateVaultId(): VaultId {
  return `vault-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` as VaultId;
}

function generateAccountId(): AccountId {
  return `account-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` as AccountId;
}

function generateKeyId(): KeyId {
  return `key-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` as KeyId;
}

// =============================================================================
// MycelixVault Class
// =============================================================================

/**
 * MycelixVault - Secure key management for the Mycelix ecosystem.
 *
 * This is the "pen" that signs transactions. It holds your keys securely
 * and only releases signatures when you authenticate (PIN/biometric).
 *
 * The Vault is designed to be invisible in normal use:
 * - FaceID/TouchID unlocks automatically
 * - Auto-locks after timeout
 * - Never exposes raw private keys
 */
export class MycelixVault {
  private storage: VaultStorageProvider;
  private encryptionKey: CryptoKey | null = null;
  private keys: Map<KeyId, KeyMaterial> = new Map();
  private autoLockTimer: ReturnType<typeof setTimeout> | null = null;

  // Reactive state
  private _state$: BehaviorSubject<VaultState>;

  private constructor(storage: VaultStorageProvider) {
    this.storage = storage;
    this._state$ = new BehaviorSubject<VaultState>({
      status: 'uninitialized',
      metadata: null,
      accounts: [],
      currentAccountId: null,
      error: null,
    });
  }

  // ===========================================================================
  // Static Factory Methods
  // ===========================================================================

  /**
   * Create a new vault
   */
  static async create(
    config: CreateVaultConfig,
    storage?: VaultStorageProvider
  ): Promise<{ vault: MycelixVault; recoveryPhrase: string[] }> {
    const vault = new MycelixVault(storage ?? new InMemoryVaultStorage());

    // Check if vault already exists
    const existing = await vault.storage.getMetadata();
    if (existing) {
      throw new Error('Vault already exists. Use MycelixVault.load() or delete first.');
    }

    // Validate config
    if (!config.biometricEnabled && !config.pin) {
      throw new Error('Either PIN or biometric must be enabled');
    }

    if (config.pin && (config.pin.length < 4 || config.pin.length > 8)) {
      throw new Error('PIN must be 4-8 digits');
    }

    // Generate master key from PIN
    const salt = crypto.getRandomValues(new Uint8Array(32));
    const pin = config.pin ?? crypto.getRandomValues(new Uint8Array(6)).join('').slice(0, 6);
    const encryptionKey = await deriveKeyFromPin(pin, salt);

    // Generate recovery phrase
    const recoveryPhrase = generateRecoveryPhrase();

    // Create vault metadata
    const metadata: VaultMetadata = {
      vaultId: generateVaultId(),
      name: config.name,
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      accountCount: 0,
      biometricEnabled: config.biometricEnabled ?? false,
      autoLockTimeout: config.autoLockTimeout ?? 5 * 60 * 1000, // 5 minutes
      backupRequired: true,
    };

    // Store metadata and salt
    await vault.storage.setMetadata(metadata);
    await vault.storage.setEncryptedData('salt', salt);

    // Encrypt and store recovery phrase
    const phraseBytes = new TextEncoder().encode(recoveryPhrase.join(' '));
    const { encrypted: encryptedPhrase, iv: phraseIv } = await encryptData(phraseBytes, encryptionKey);
    await vault.storage.setEncryptedData('recovery_phrase', encryptedPhrase);
    await vault.storage.setEncryptedData('recovery_phrase_iv', phraseIv);

    // Store biometric key if enabled
    if (config.biometricEnabled && (await vault.storage.isBiometricAvailable())) {
      const keyBytes = await crypto.subtle.exportKey('raw', encryptionKey);
      await vault.storage.storeBiometricKey(new Uint8Array(keyBytes));
    }

    // Set vault state
    vault.encryptionKey = encryptionKey;
    vault._state$.next({
      status: 'unlocked',
      metadata,
      accounts: [],
      currentAccountId: null,
      error: null,
    });

    // Create default account
    await vault.createAccount({ name: 'Main Account', isDefault: true });

    // Start auto-lock timer
    vault.resetAutoLockTimer();

    return { vault, recoveryPhrase };
  }

  /**
   * Load an existing vault (starts in locked state)
   */
  static async load(storage?: VaultStorageProvider): Promise<MycelixVault> {
    const vault = new MycelixVault(storage ?? new InMemoryVaultStorage());

    const metadata = await vault.storage.getMetadata();
    if (!metadata) {
      vault._state$.next({
        status: 'uninitialized',
        metadata: null,
        accounts: [],
        currentAccountId: null,
        error: null,
      });
      return vault;
    }

    vault._state$.next({
      status: 'locked',
      metadata,
      accounts: [], // Accounts loaded after unlock
      currentAccountId: null,
      error: null,
    });

    return vault;
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable of vault state changes */
  get state$(): BehaviorSubject<VaultState> {
    return this._state$;
  }

  /** Current state (synchronous) */
  get state(): VaultState {
    return this._state$.value;
  }

  /** Current status */
  get status(): VaultStatus {
    return this._state$.value.status;
  }

  /** Whether vault is unlocked and ready */
  get isUnlocked(): boolean {
    return this._state$.value.status === 'unlocked';
  }

  /** Subscribe to state changes */
  subscribe(observer: (state: VaultState) => void): Subscription {
    return this._state$.subscribe(observer);
  }

  // ===========================================================================
  // Authentication
  // ===========================================================================

  /**
   * Unlock the vault with PIN, biometric, or recovery phrase
   */
  async unlock(options: UnlockOptions): Promise<void> {
    const currentState = this._state$.value;

    if (currentState.status === 'unlocked') {
      return; // Already unlocked
    }

    if (currentState.status === 'uninitialized') {
      throw new Error('No vault exists. Create one first.');
    }

    this._state$.next({ ...currentState, status: 'unlocking', error: null });

    try {
      let encryptionKey: CryptoKey;
      const salt = await this.storage.getEncryptedData('salt');
      if (!salt) {
        throw new Error('Vault data corrupted: missing salt');
      }

      if (options.biometric) {
        // Biometric authentication
        if (!(await this.storage.isBiometricAvailable())) {
          throw new Error('Biometric authentication not available');
        }

        const authenticated = await this.storage.authenticateBiometric('Unlock your Mycelix Vault');
        if (!authenticated) {
          throw new Error('Biometric authentication failed');
        }

        const keyBytes = await this.storage.retrieveBiometricKey();
        if (!keyBytes) {
          throw new Error('Biometric key not found. Use PIN to unlock.');
        }

        encryptionKey = await crypto.subtle.importKey('raw', keyBytes as BufferSource, { name: 'AES-GCM', length: 256 }, false, [
          'encrypt',
          'decrypt',
        ]);
      } else if (options.pin) {
        // PIN authentication
        encryptionKey = await deriveKeyFromPin(options.pin, salt);

        // Verify PIN by trying to decrypt something
        const testData = await this.storage.getEncryptedData('recovery_phrase');
        const testIv = await this.storage.getEncryptedData('recovery_phrase_iv');
        if (testData && testIv) {
          try {
            await decryptData(testData, encryptionKey, testIv);
          } catch {
            throw new Error('Invalid PIN');
          }
        }
      } else if (options.recoveryPhrase) {
        // Recovery phrase - regenerate keys
        // In a real implementation, this would derive the master key from the phrase
        throw new Error('Recovery phrase unlock not yet implemented');
      } else {
        throw new Error('No authentication method provided');
      }

      // Load accounts
      const accountsData = await this.storage.getEncryptedData('accounts');
      const accountsIv = await this.storage.getEncryptedData('accounts_iv');
      let accounts: VaultAccount[] = [];

      if (accountsData && accountsIv) {
        const decrypted = await decryptData(accountsData, encryptionKey, accountsIv);
        accounts = JSON.parse(new TextDecoder().decode(decrypted));
      }

      // Load keys
      const keysData = await this.storage.getEncryptedData('keys');
      const keysIv = await this.storage.getEncryptedData('keys_iv');

      if (keysData && keysIv) {
        const decrypted = await decryptData(keysData, encryptionKey, keysIv);
        const keysArray = JSON.parse(new TextDecoder().decode(decrypted));
        this.keys = new Map(
          keysArray.map((k: KeyMaterial & { publicKey: number[]; encryptedPrivateKey: number[]; salt: number[]; iv: number[] }) => [
            k.keyId,
            {
              ...k,
              publicKey: new Uint8Array(k.publicKey),
              encryptedPrivateKey: new Uint8Array(k.encryptedPrivateKey),
              salt: new Uint8Array(k.salt),
              iv: new Uint8Array(k.iv),
            },
          ])
        );
      }

      // Update state
      this.encryptionKey = encryptionKey;
      const defaultAccount = accounts.find((a) => a.isDefault);

      // Update last accessed
      const metadata = { ...currentState.metadata!, lastAccessedAt: Date.now() };
      await this.storage.setMetadata(metadata);

      this._state$.next({
        status: 'unlocked',
        metadata,
        accounts,
        currentAccountId: defaultAccount?.accountId ?? null,
        error: null,
      });

      // Start auto-lock timer
      this.resetAutoLockTimer();
    } catch (error) {
      this._state$.next({
        ...currentState,
        status: 'locked',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Lock the vault (clears keys from memory)
   */
  lock(): void {
    this.encryptionKey = null;
    this.keys.clear();

    if (this.autoLockTimer) {
      clearTimeout(this.autoLockTimer);
      this.autoLockTimer = null;
    }

    const currentState = this._state$.value;
    this._state$.next({
      ...currentState,
      status: 'locked',
      accounts: [],
      currentAccountId: null,
      error: null,
    });
  }

  /**
   * Reset the auto-lock timer (called on activity)
   */
  private resetAutoLockTimer(): void {
    if (this.autoLockTimer) {
      clearTimeout(this.autoLockTimer);
    }

    const timeout = this._state$.value.metadata?.autoLockTimeout ?? 0;
    if (timeout > 0) {
      this.autoLockTimer = setTimeout(() => this.lock(), timeout);
    }
  }

  // ===========================================================================
  // Account Management
  // ===========================================================================

  /**
   * Create a new account in the vault
   */
  async createAccount(options: { name: string; isDefault?: boolean }): Promise<VaultAccount> {
    this.requireUnlocked();

    // Generate keypair
    const { publicKey, privateKey } = await generateEd25519Keypair();
    const keyId = generateKeyId();

    // Encrypt private key
    const keySalt = crypto.getRandomValues(new Uint8Array(32));
    const keyKey = await deriveKeyFromPin(keyId, keySalt); // Use keyId as additional entropy
    const { encrypted: encryptedPrivateKey, iv: keyIv } = await encryptData(privateKey, keyKey);

    // Store key material
    const keyMaterial: KeyMaterial = {
      keyId,
      keyType: 'ed25519',
      publicKey,
      encryptedPrivateKey,
      salt: keySalt,
      iv: keyIv,
    };
    this.keys.set(keyId, keyMaterial);

    // Create account
    const accountId = generateAccountId();
    const account: VaultAccount = {
      accountId,
      name: options.name,
      agentId: Buffer.from(publicKey).toString('hex'),
      keyId,
      keyType: 'ed25519',
      isDefault: options.isDefault ?? false,
      createdAt: Date.now(),
    };

    // Update state
    const currentState = this._state$.value;
    let accounts = [...currentState.accounts, account];

    // If this is default, unset others
    if (options.isDefault) {
      accounts = accounts.map((a) => (a.accountId === accountId ? a : { ...a, isDefault: false }));
    }

    // Save to storage
    await this.saveAccountsAndKeys(accounts);

    // Update metadata
    const metadata = { ...currentState.metadata!, accountCount: accounts.length };
    await this.storage.setMetadata(metadata);

    this._state$.next({
      ...currentState,
      metadata,
      accounts,
      currentAccountId: options.isDefault ? accountId : currentState.currentAccountId ?? accountId,
    });

    this.resetAutoLockTimer();
    return account;
  }

  /**
   * Get all accounts
   */
  getAccounts(): VaultAccount[] {
    return this._state$.value.accounts;
  }

  /**
   * Get current account
   */
  getCurrentAccount(): VaultAccount | null {
    const state = this._state$.value;
    return state.accounts.find((a) => a.accountId === state.currentAccountId) ?? null;
  }

  /**
   * Switch to a different account
   */
  switchAccount(accountId: AccountId): void {
    this.requireUnlocked();

    const state = this._state$.value;
    const account = state.accounts.find((a) => a.accountId === accountId);
    if (!account) {
      throw new Error(`Account ${accountId} not found`);
    }

    this._state$.next({
      ...state,
      currentAccountId: accountId,
    });

    this.resetAutoLockTimer();
  }

  // ===========================================================================
  // Signing Operations
  // ===========================================================================

  /**
   * Sign data with the current account's key
   */
  async sign(data: Uint8Array, accountId?: AccountId): Promise<Uint8Array> {
    this.requireUnlocked();

    const targetAccountId = accountId ?? this._state$.value.currentAccountId;
    if (!targetAccountId) {
      throw new Error('No account selected');
    }

    const account = this._state$.value.accounts.find((a) => a.accountId === targetAccountId);
    if (!account) {
      throw new Error(`Account ${targetAccountId} not found`);
    }

    const keyMaterial = this.keys.get(account.keyId);
    if (!keyMaterial) {
      throw new Error('Key not found');
    }

    // Decrypt private key
    const keyKey = await deriveKeyFromPin(account.keyId, keyMaterial.salt);
    const privateKey = await decryptData(keyMaterial.encryptedPrivateKey, keyKey, keyMaterial.iv);

    // Sign
    const signature = await signEd25519(data, privateKey);

    this.resetAutoLockTimer();
    return signature;
  }

  /**
   * Verify a signature
   */
  async verify(data: Uint8Array, signature: Uint8Array, accountId?: AccountId): Promise<boolean> {
    const targetAccountId = accountId ?? this._state$.value.currentAccountId;
    if (!targetAccountId) {
      throw new Error('No account selected');
    }

    const account = this._state$.value.accounts.find((a) => a.accountId === targetAccountId);
    if (!account) {
      throw new Error(`Account ${targetAccountId} not found`);
    }

    const keyMaterial = this.keys.get(account.keyId);
    if (!keyMaterial) {
      throw new Error('Key not found');
    }

    return verifyEd25519(data, signature, keyMaterial.publicKey);
  }

  /**
   * Get public key for an account
   */
  getPublicKey(accountId?: AccountId): Uint8Array {
    const targetAccountId = accountId ?? this._state$.value.currentAccountId;
    if (!targetAccountId) {
      throw new Error('No account selected');
    }

    const account = this._state$.value.accounts.find((a) => a.accountId === targetAccountId);
    if (!account) {
      throw new Error(`Account ${targetAccountId} not found`);
    }

    const keyMaterial = this.keys.get(account.keyId);
    if (!keyMaterial) {
      throw new Error('Key not found');
    }

    return keyMaterial.publicKey;
  }

  // ===========================================================================
  // Backup & Recovery
  // ===========================================================================

  /**
   * Get recovery phrase (requires unlock)
   */
  async getRecoveryPhrase(): Promise<string[]> {
    this.requireUnlocked();

    const phraseData = await this.storage.getEncryptedData('recovery_phrase');
    const phraseIv = await this.storage.getEncryptedData('recovery_phrase_iv');

    if (!phraseData || !phraseIv || !this.encryptionKey) {
      throw new Error('Recovery phrase not found');
    }

    const decrypted = await decryptData(phraseData, this.encryptionKey, phraseIv);
    return new TextDecoder().decode(decrypted).split(' ');
  }

  /**
   * Mark backup as completed
   */
  async confirmBackup(): Promise<void> {
    this.requireUnlocked();

    const state = this._state$.value;
    if (!state.metadata) return;

    const metadata = { ...state.metadata, backupRequired: false };
    await this.storage.setMetadata(metadata);

    this._state$.next({ ...state, metadata });
  }

  // ===========================================================================
  // Utility Methods
  // ===========================================================================

  /**
   * Delete the vault completely
   */
  async deleteVault(): Promise<void> {
    this.lock();
    await this.storage.deleteMetadata();
    await this.storage.deleteEncryptedData('salt');
    await this.storage.deleteEncryptedData('accounts');
    await this.storage.deleteEncryptedData('accounts_iv');
    await this.storage.deleteEncryptedData('keys');
    await this.storage.deleteEncryptedData('keys_iv');
    await this.storage.deleteEncryptedData('recovery_phrase');
    await this.storage.deleteEncryptedData('recovery_phrase_iv');

    this._state$.next({
      status: 'uninitialized',
      metadata: null,
      accounts: [],
      currentAccountId: null,
      error: null,
    });
  }

  /**
   * Change PIN
   */
  async changePin(currentPin: string, newPin: string): Promise<void> {
    // Re-authenticate with current PIN
    const salt = await this.storage.getEncryptedData('salt');
    if (!salt) throw new Error('Vault corrupted');

    const currentKey = await deriveKeyFromPin(currentPin, salt);

    // Verify current PIN
    const testData = await this.storage.getEncryptedData('recovery_phrase');
    const testIv = await this.storage.getEncryptedData('recovery_phrase_iv');
    if (testData && testIv) {
      try {
        await decryptData(testData, currentKey, testIv);
      } catch {
        throw new Error('Current PIN is incorrect');
      }
    }

    // Generate new salt and key
    const newSalt = crypto.getRandomValues(new Uint8Array(32));
    const newKey = await deriveKeyFromPin(newPin, newSalt);

    // Re-encrypt recovery phrase
    if (testData && testIv) {
      const phraseBytes = await decryptData(testData, currentKey, testIv);
      const { encrypted, iv } = await encryptData(phraseBytes, newKey);
      await this.storage.setEncryptedData('recovery_phrase', encrypted);
      await this.storage.setEncryptedData('recovery_phrase_iv', iv);
    }

    // Update salt
    await this.storage.setEncryptedData('salt', newSalt);

    // Re-save accounts and keys with new encryption
    this.encryptionKey = newKey;
    await this.saveAccountsAndKeys(this._state$.value.accounts);

    // Update biometric if enabled
    const state = this._state$.value;
    if (state.metadata?.biometricEnabled && (await this.storage.isBiometricAvailable())) {
      const keyBytes = await crypto.subtle.exportKey('raw', newKey);
      await this.storage.storeBiometricKey(new Uint8Array(keyBytes));
    }
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private requireUnlocked(): void {
    if (this._state$.value.status !== 'unlocked') {
      throw new Error('Vault is locked. Unlock first.');
    }
    if (!this.encryptionKey) {
      throw new Error('Encryption key not available');
    }
  }

  private async saveAccountsAndKeys(accounts: VaultAccount[]): Promise<void> {
    if (!this.encryptionKey) throw new Error('No encryption key');

    // Save accounts
    const accountsBytes = new TextEncoder().encode(JSON.stringify(accounts));
    const { encrypted: accountsEncrypted, iv: accountsIv } = await encryptData(
      accountsBytes,
      this.encryptionKey
    );
    await this.storage.setEncryptedData('accounts', accountsEncrypted);
    await this.storage.setEncryptedData('accounts_iv', accountsIv);

    // Save keys (convert Uint8Array to array for JSON)
    const keysArray = Array.from(this.keys.values()).map((k) => ({
      ...k,
      publicKey: Array.from(k.publicKey),
      encryptedPrivateKey: Array.from(k.encryptedPrivateKey),
      salt: Array.from(k.salt),
      iv: Array.from(k.iv),
    }));
    const keysBytes = new TextEncoder().encode(JSON.stringify(keysArray));
    const { encrypted: keysEncrypted, iv: keysIv } = await encryptData(keysBytes, this.encryptionKey);
    await this.storage.setEncryptedData('keys', keysEncrypted);
    await this.storage.setEncryptedData('keys_iv', keysIv);
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new vault
 */
export async function createVault(
  config: CreateVaultConfig,
  storage?: VaultStorageProvider
): Promise<{ vault: MycelixVault; recoveryPhrase: string[] }> {
  return MycelixVault.create(config, storage);
}

/**
 * Load an existing vault
 */
export async function loadVault(storage?: VaultStorageProvider): Promise<MycelixVault> {
  return MycelixVault.load(storage);
}

// =============================================================================
// Re-exports
// =============================================================================

// VaultStorageProvider and BiometricProvider are already exported at their declarations above
