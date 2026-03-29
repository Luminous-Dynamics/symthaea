// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile Wallet Manager
 *
 * Manages wallet lifecycle, accounts, and key operations
 * for mobile applications.
 */

import type {
  WalletId,
  AccountId,
  KeyId,
  KeyType,
  WalletStatus,
  WalletMetadata,
  WalletAccount,
  KeyMaterial,
  WalletConfig,
  StorageProvider,
} from './types.js';
import type { AgentId, HappId } from '../utils/index.js';

// =============================================================================
// Wallet Manager
// =============================================================================

export interface WalletManagerConfig {
  storageProvider?: StorageProvider;
  encryptionKey?: Uint8Array;
}

/**
 * Manages wallet operations for mobile applications
 */
export class WalletManager {
  private storage: StorageProvider;
  private encryptionKey: Uint8Array | null = null;
  private currentWallet: WalletMetadata | null = null;
  private status: WalletStatus = 'uninitialized';
  private accounts: Map<AccountId, WalletAccount> = new Map();
  private keys: Map<KeyId, KeyMaterial> = new Map();
  private autoLockTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(config?: WalletManagerConfig) {
    this.storage = config?.storageProvider ?? this.createInMemoryStorage();
    this.encryptionKey = config?.encryptionKey ?? null;
  }

  // ===========================================================================
  // Wallet Lifecycle
  // ===========================================================================

  /**
   * Create a new wallet
   */
  async createWallet(
    name: string,
    pin: string,
    config?: Partial<WalletConfig>
  ): Promise<WalletMetadata> {
    const walletId = this.generateWalletId();

    // Derive encryption key from PIN
    this.encryptionKey = await this.deriveKey(pin, walletId);

    const metadata: WalletMetadata = {
      walletId,
      name,
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      accountCount: 0,
      backupRequired: true,
      version: 1,
    };

    const walletConfig: WalletConfig = {
      name,
      enableBiometrics: config?.enableBiometrics ?? false,
      autoLockTimeout: config?.autoLockTimeout ?? 300000,
      requirePinOnTransaction: config?.requirePinOnTransaction ?? true,
      backupReminder: config?.backupReminder ?? true,
    };

    // Store wallet data
    await this.storage.set(`wallet:${walletId}:metadata`, JSON.stringify(metadata));
    await this.storage.set(`wallet:${walletId}:config`, JSON.stringify(walletConfig));
    await this.storage.set('wallet:current', walletId);

    this.currentWallet = metadata;
    this.status = 'unlocked';

    // Create default account
    await this.createAccount('Main Account', true);

    return metadata;
  }

  /**
   * Import wallet from recovery phrase
   */
  async importWallet(
    name: string,
    recoveryPhrase: string,
    pin: string
  ): Promise<WalletMetadata> {
    // Validate recovery phrase (BIP39)
    if (!this.validateRecoveryPhrase(recoveryPhrase)) {
      throw new Error('Invalid recovery phrase');
    }

    const walletId = this.generateWalletId();
    this.encryptionKey = await this.deriveKey(pin, walletId);

    // Derive master key from recovery phrase
    const masterKey = await this.deriveMasterKey(recoveryPhrase);

    const metadata: WalletMetadata = {
      walletId,
      name,
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      accountCount: 0,
      backupRequired: false,
      version: 1,
    };

    // Store encrypted master key
    const encryptedMasterKey = await this.encrypt(masterKey, this.encryptionKey);
    await this.storage.set(`wallet:${walletId}:master`, this.uint8ArrayToBase64(encryptedMasterKey));
    await this.storage.set(`wallet:${walletId}:metadata`, JSON.stringify(metadata));
    await this.storage.set('wallet:current', walletId);

    this.currentWallet = metadata;
    this.status = 'unlocked';

    // Create default account
    await this.createAccount('Main Account', true);

    return metadata;
  }

  /**
   * Unlock wallet with PIN
   */
  async unlock(pin: string): Promise<boolean> {
    if (!this.currentWallet) {
      await this.loadCurrentWallet();
    }

    if (!this.currentWallet) {
      throw new Error('No wallet found');
    }

    try {
      this.encryptionKey = await this.deriveKey(pin, this.currentWallet.walletId);

      // Verify by trying to decrypt something
      const testData = await this.storage.get(`wallet:${this.currentWallet.walletId}:test`);
      if (testData) {
        await this.decrypt(this.base64ToUint8Array(testData), this.encryptionKey);
      }

      this.status = 'unlocked';
      this.currentWallet.lastAccessedAt = Date.now();
      await this.storage.set(
        `wallet:${this.currentWallet.walletId}:metadata`,
        JSON.stringify(this.currentWallet)
      );

      // Load accounts
      await this.loadAccounts();

      // Start auto-lock timer
      this.startAutoLockTimer();

      return true;
    } catch {
      this.encryptionKey = null;
      return false;
    }
  }

  /**
   * Lock wallet
   */
  lock(): void {
    this.encryptionKey = null;
    this.status = 'locked';
    this.accounts.clear();
    this.keys.clear();
    this.stopAutoLockTimer();
  }

  /**
   * Get wallet status
   */
  getStatus(): WalletStatus {
    return this.status;
  }

  /**
   * Get current wallet metadata
   */
  getCurrentWallet(): WalletMetadata | null {
    return this.currentWallet;
  }

  // ===========================================================================
  // Account Management
  // ===========================================================================

  /**
   * Create a new account
   */
  async createAccount(
    name: string,
    isDefault: boolean = false
  ): Promise<WalletAccount> {
    this.requireUnlocked();

    const accountId = this.generateAccountId();
    const keyId = this.generateKeyId();

    // Generate key pair
    const keyPair = await this.generateKeyPair('ed25519');

    const keyMaterial: KeyMaterial = {
      keyId,
      keyType: 'ed25519',
      publicKey: keyPair.publicKey,
      encryptedPrivateKey: await this.encrypt(keyPair.privateKey, this.encryptionKey!),
      createdAt: Date.now(),
    };

    const account: WalletAccount = {
      accountId,
      walletId: this.currentWallet!.walletId,
      name,
      agentId: this.publicKeyToAgentId(keyPair.publicKey),
      keyId,
      createdAt: Date.now(),
      isDefault,
      connectedHapps: [],
    };

    // If this is default, unset others
    if (isDefault) {
      for (const existing of this.accounts.values()) {
        if (existing.isDefault) {
          existing.isDefault = false;
          await this.saveAccount(existing);
        }
      }
    }

    this.accounts.set(accountId, account);
    this.keys.set(keyId, keyMaterial);

    await this.saveAccount(account);
    await this.saveKey(keyMaterial);

    // Update wallet metadata
    this.currentWallet!.accountCount++;
    await this.storage.set(
      `wallet:${this.currentWallet!.walletId}:metadata`,
      JSON.stringify(this.currentWallet)
    );

    return account;
  }

  /**
   * Get account by ID
   */
  getAccount(accountId: AccountId): WalletAccount | null {
    return this.accounts.get(accountId) ?? null;
  }

  /**
   * Get default account
   */
  getDefaultAccount(): WalletAccount | null {
    for (const account of this.accounts.values()) {
      if (account.isDefault) return account;
    }
    return this.accounts.values().next().value ?? null;
  }

  /**
   * List all accounts
   */
  listAccounts(): WalletAccount[] {
    return Array.from(this.accounts.values());
  }

  /**
   * Set default account
   */
  async setDefaultAccount(accountId: AccountId): Promise<boolean> {
    this.requireUnlocked();

    const account = this.accounts.get(accountId);
    if (!account) return false;

    for (const existing of this.accounts.values()) {
      if (existing.isDefault && existing.accountId !== accountId) {
        existing.isDefault = false;
        await this.saveAccount(existing);
      }
    }

    account.isDefault = true;
    await this.saveAccount(account);
    return true;
  }

  /**
   * Rename account
   */
  async renameAccount(accountId: AccountId, newName: string): Promise<boolean> {
    this.requireUnlocked();

    const account = this.accounts.get(accountId);
    if (!account) return false;

    account.name = newName;
    await this.saveAccount(account);
    return true;
  }

  /**
   * Connect account to a hApp
   */
  async connectToHapp(accountId: AccountId, happId: HappId): Promise<boolean> {
    this.requireUnlocked();

    const account = this.accounts.get(accountId);
    if (!account) return false;

    if (!account.connectedHapps.includes(happId)) {
      account.connectedHapps.push(happId);
      await this.saveAccount(account);
    }
    return true;
  }

  /**
   * Disconnect account from a hApp
   */
  async disconnectFromHapp(accountId: AccountId, happId: HappId): Promise<boolean> {
    this.requireUnlocked();

    const account = this.accounts.get(accountId);
    if (!account) return false;

    const idx = account.connectedHapps.indexOf(happId);
    if (idx !== -1) {
      account.connectedHapps.splice(idx, 1);
      await this.saveAccount(account);
    }
    return true;
  }

  // ===========================================================================
  // Signing Operations
  // ===========================================================================

  /**
   * Sign data with account's private key
   */
  async sign(accountId: AccountId, data: Uint8Array): Promise<Uint8Array> {
    this.requireUnlocked();

    const account = this.accounts.get(accountId);
    if (!account) {
      throw new Error(`Account ${accountId} not found`);
    }

    const keyMaterial = this.keys.get(account.keyId);
    if (!keyMaterial || !keyMaterial.encryptedPrivateKey) {
      throw new Error('Key material not found');
    }

    // Decrypt private key
    const privateKey = await this.decrypt(
      keyMaterial.encryptedPrivateKey,
      this.encryptionKey!
    );

    // Sign (using Web Crypto API simulation)
    return this.signWithKey(privateKey, data);
  }

  /**
   * Verify signature
   */
  async verify(
    publicKey: Uint8Array,
    data: Uint8Array,
    signature: Uint8Array
  ): Promise<boolean> {
    return this.verifyWithKey(publicKey, data, signature);
  }

  /**
   * Get public key for account
   */
  getPublicKey(accountId: AccountId): Uint8Array | null {
    const account = this.accounts.get(accountId);
    if (!account) return null;

    const keyMaterial = this.keys.get(account.keyId);
    return keyMaterial?.publicKey ?? null;
  }

  // ===========================================================================
  // Backup & Recovery
  // ===========================================================================

  /**
   * Generate recovery phrase
   */
  async generateRecoveryPhrase(): Promise<string> {
    // Generate 12 random words from BIP39 wordlist
    const entropy = crypto.getRandomValues(new Uint8Array(16));
    return this.entropyToMnemonic(entropy);
  }

  /**
   * Export encrypted backup
   */
  async exportBackup(password: string): Promise<Uint8Array> {
    this.requireUnlocked();

    const backup = {
      version: 1,
      createdAt: Date.now(),
      wallet: this.currentWallet,
      accounts: Array.from(this.accounts.values()),
      keys: Array.from(this.keys.values()).map((k) => ({
        ...k,
        encryptedPrivateKey: k.encryptedPrivateKey
          ? this.uint8ArrayToBase64(k.encryptedPrivateKey)
          : undefined,
        publicKey: this.uint8ArrayToBase64(k.publicKey),
      })),
    };

    const backupKey = await this.deriveKey(password, 'backup');
    const encrypted = await this.encrypt(
      new TextEncoder().encode(JSON.stringify(backup)),
      backupKey
    );

    return encrypted;
  }

  /**
   * Mark backup as complete
   */
  async markBackupComplete(): Promise<void> {
    this.requireUnlocked();

    if (this.currentWallet) {
      this.currentWallet.backupRequired = false;
      await this.storage.set(
        `wallet:${this.currentWallet.walletId}:metadata`,
        JSON.stringify(this.currentWallet)
      );
    }
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private requireUnlocked(): void {
    if (this.status !== 'unlocked' || !this.encryptionKey) {
      throw new Error('Wallet is locked');
    }
  }

  private async loadCurrentWallet(): Promise<void> {
    const walletId = await this.storage.get('wallet:current');
    if (!walletId) return;

    const metadataStr = await this.storage.get(`wallet:${walletId}:metadata`);
    if (metadataStr) {
      this.currentWallet = JSON.parse(metadataStr);
      this.status = 'locked';
    }
  }

  private async loadAccounts(): Promise<void> {
    if (!this.currentWallet) return;

    const keys = await this.storage.getAllKeys();
    const accountKeys = keys.filter((k) =>
      k.startsWith(`wallet:${this.currentWallet!.walletId}:account:`)
    );

    for (const key of accountKeys) {
      const data = await this.storage.get(key);
      if (data) {
        const account: WalletAccount = JSON.parse(data);
        this.accounts.set(account.accountId, account);

        // Load associated key
        const keyData = await this.storage.get(
          `wallet:${this.currentWallet.walletId}:key:${account.keyId}`
        );
        if (keyData) {
          const keyMaterial = JSON.parse(keyData) as KeyMaterial & {
            publicKey: string;
            encryptedPrivateKey?: string;
          };
          this.keys.set(account.keyId, {
            ...keyMaterial,
            publicKey: this.base64ToUint8Array(keyMaterial.publicKey as unknown as string),
            encryptedPrivateKey: keyMaterial.encryptedPrivateKey
              ? this.base64ToUint8Array(keyMaterial.encryptedPrivateKey as unknown as string)
              : undefined,
          });
        }
      }
    }
  }

  private async saveAccount(account: WalletAccount): Promise<void> {
    await this.storage.set(
      `wallet:${this.currentWallet!.walletId}:account:${account.accountId}`,
      JSON.stringify(account)
    );
  }

  private async saveKey(key: KeyMaterial): Promise<void> {
    const serialized = {
      ...key,
      publicKey: this.uint8ArrayToBase64(key.publicKey),
      encryptedPrivateKey: key.encryptedPrivateKey
        ? this.uint8ArrayToBase64(key.encryptedPrivateKey)
        : undefined,
    };
    await this.storage.set(
      `wallet:${this.currentWallet!.walletId}:key:${key.keyId}`,
      JSON.stringify(serialized)
    );
  }

  private startAutoLockTimer(): void {
    this.stopAutoLockTimer();
    // Default 5 minutes
    this.autoLockTimer = setTimeout(() => this.lock(), 300000);
  }

  private stopAutoLockTimer(): void {
    if (this.autoLockTimer) {
      clearTimeout(this.autoLockTimer);
      this.autoLockTimer = null;
    }
  }

  // Crypto helpers
  private async deriveKey(password: string, salt: string): Promise<Uint8Array> {
    const encoder = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      encoder.encode(password),
      'PBKDF2',
      false,
      ['deriveBits']
    );

    const bits = await crypto.subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: encoder.encode(salt),
        iterations: 100000,
        hash: 'SHA-256',
      },
      keyMaterial,
      256
    );

    return new Uint8Array(bits);
  }

  private async encrypt(data: Uint8Array, key: Uint8Array): Promise<Uint8Array> {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const cryptoKey = await crypto.subtle.importKey(
      'raw',
      key.buffer as ArrayBuffer,
      { name: 'AES-GCM' },
      false,
      ['encrypt']
    );

    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      cryptoKey,
      data.buffer as ArrayBuffer
    );

    // Prepend IV to ciphertext
    const result = new Uint8Array(iv.length + encrypted.byteLength);
    result.set(iv);
    result.set(new Uint8Array(encrypted), iv.length);
    return result;
  }

  private async decrypt(data: Uint8Array, key: Uint8Array): Promise<Uint8Array> {
    const iv = data.slice(0, 12);
    const ciphertext = data.slice(12);

    const cryptoKey = await crypto.subtle.importKey(
      'raw',
      key.buffer as ArrayBuffer,
      { name: 'AES-GCM' },
      false,
      ['decrypt']
    );

    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: iv.buffer },
      cryptoKey,
      ciphertext.buffer
    );

    return new Uint8Array(decrypted);
  }

  private async generateKeyPair(
    _keyType: KeyType
  ): Promise<{ publicKey: Uint8Array; privateKey: Uint8Array }> {
    // Simplified - in production use proper Ed25519 library
    const keyPair = await crypto.subtle.generateKey(
      { name: 'ECDSA', namedCurve: 'P-256' },
      true,
      ['sign', 'verify']
    );

    const publicKey = await crypto.subtle.exportKey('raw', keyPair.publicKey);
    const privateKey = await crypto.subtle.exportKey('pkcs8', keyPair.privateKey);

    return {
      publicKey: new Uint8Array(publicKey),
      privateKey: new Uint8Array(privateKey),
    };
  }

  private async signWithKey(privateKey: Uint8Array, data: Uint8Array): Promise<Uint8Array> {
    const key = await crypto.subtle.importKey(
      'pkcs8',
      privateKey.buffer as ArrayBuffer,
      { name: 'ECDSA', namedCurve: 'P-256' },
      false,
      ['sign']
    );

    const signature = await crypto.subtle.sign(
      { name: 'ECDSA', hash: 'SHA-256' },
      key,
      data.buffer as ArrayBuffer
    );

    return new Uint8Array(signature);
  }

  private async verifyWithKey(
    publicKey: Uint8Array,
    data: Uint8Array,
    signature: Uint8Array
  ): Promise<boolean> {
    try {
      const key = await crypto.subtle.importKey(
        'raw',
        publicKey.buffer as ArrayBuffer,
        { name: 'ECDSA', namedCurve: 'P-256' },
        false,
        ['verify']
      );

      return await crypto.subtle.verify(
        { name: 'ECDSA', hash: 'SHA-256' },
        key,
        signature.buffer as ArrayBuffer,
        data.buffer as ArrayBuffer
      );
    } catch {
      return false;
    }
  }

  private publicKeyToAgentId(publicKey: Uint8Array): AgentId {
    // Simplified - in production use proper Holochain agent ID encoding
    return `uhCAk${this.uint8ArrayToBase64(publicKey).slice(0, 40)}`;
  }

  private async deriveMasterKey(recoveryPhrase: string): Promise<Uint8Array> {
    // Simplified - in production use BIP39/BIP32
    return this.deriveKey(recoveryPhrase, 'master');
  }

  private validateRecoveryPhrase(phrase: string): boolean {
    const words = phrase.trim().split(/\s+/);
    return words.length === 12 || words.length === 24;
  }

  private entropyToMnemonic(_entropy: Uint8Array): string {
    // Simplified - in production use BIP39 library
    const words = [
      'abandon', 'ability', 'able', 'about', 'above', 'absent',
      'absorb', 'abstract', 'absurd', 'abuse', 'access', 'accident',
    ];
    return words.join(' ');
  }

  private uint8ArrayToBase64(arr: Uint8Array): string {
    return btoa(String.fromCharCode(...arr));
  }

  private base64ToUint8Array(str: string): Uint8Array {
    return new Uint8Array(atob(str).split('').map((c) => c.charCodeAt(0)));
  }

  private generateWalletId(): WalletId {
    return `wallet-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private generateAccountId(): AccountId {
    return `account-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private generateKeyId(): KeyId {
    return `key-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private createInMemoryStorage(): StorageProvider {
    const store = new Map<string, string>();
    return {
      get: async (key) => store.get(key) ?? null,
      set: async (key, value) => {
        store.set(key, value);
      },
      remove: async (key) => {
        store.delete(key);
      },
      clear: async () => {
        store.clear();
      },
      getAllKeys: async () => Array.from(store.keys()),
    };
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a wallet manager
 */
export function createWalletManager(config?: WalletManagerConfig): WalletManager {
  return new WalletManager(config);
}
