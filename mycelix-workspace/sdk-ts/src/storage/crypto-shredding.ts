// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Cryptographic Shredding
 *
 * GDPR-compliant erasure via encryption key destruction.
 * @see docs/architecture/uess/UESS-09-GDPR-COMPLIANCE.md
 */

import { hash, secureRandomBytes } from '../security/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Encryption key entry
 */
export interface KeyEntry {
  /** Unique key identifier */
  keyId: string;

  /** Encryption key (to be shredded) */
  key: Uint8Array;

  /** Algorithm used */
  algorithm: 'AES-256-GCM' | 'ChaCha20-Poly1305';

  /** When the key was created */
  createdAt: number;

  /** Data subjects this key protects */
  dataSubjects: string[];

  /** Resources encrypted with this key */
  resources: string[];

  /** Whether key has been shredded */
  shredded: boolean;

  /** When the key was shredded (if applicable) */
  shreddedAt?: number;
}

/**
 * Shredding audit entry
 */
export interface ShredAuditEntry {
  /** Unique audit ID */
  id: string;

  /** Key ID that was shredded */
  keyId: string;

  /** Data subject requesting erasure */
  dataSubject: string;

  /** Timestamp of shredding */
  shreddedAt: number;

  /** Reason for shredding */
  reason: ShredReason;

  /** Resources affected */
  affectedResources: string[];

  /** Who performed the shredding */
  performedBy: string;

  /** Hash of the key before shredding (for verification) */
  keyHash: string;
}

/**
 * Reason for cryptographic shredding
 */
export enum ShredReason {
  GDPR_ERASURE_REQUEST = 'gdpr_erasure_request',
  DATA_RETENTION_EXPIRED = 'data_retention_expired',
  USER_ACCOUNT_DELETION = 'user_account_deletion',
  LEGAL_REQUIREMENT = 'legal_requirement',
  SECURITY_INCIDENT = 'security_incident',
  MANUAL_ADMIN = 'manual_admin',
}

/**
 * Shredding result
 */
export interface ShredResult {
  success: boolean;
  keyId: string;
  affectedResources: number;
  auditEntry: ShredAuditEntry;
  error?: string;
}

// =============================================================================
// Crypto Shredding Manager
// =============================================================================

/**
 * Configuration for crypto shredding manager
 */
export interface CryptoShreddingConfig {
  /** Key derivation salt */
  salt: Uint8Array;

  /** Default algorithm */
  algorithm: 'AES-256-GCM' | 'ChaCha20-Poly1305';

  /** Audit log retention in days */
  auditRetentionDays: number;

  /** Enable secure key overwrite */
  secureOverwrite: boolean;

  /** Number of overwrite passes */
  overwritePasses: number;
}

const DEFAULT_CONFIG: CryptoShreddingConfig = {
  salt: new Uint8Array(32),
  algorithm: 'AES-256-GCM',
  auditRetentionDays: 365 * 7, // 7 years for GDPR
  secureOverwrite: true,
  overwritePasses: 3,
};

/**
 * Cryptographic Shredding Manager
 *
 * Provides GDPR-compliant data erasure through encryption key destruction.
 * When a key is shredded, all data encrypted with it becomes permanently
 * unrecoverable without actually deleting the ciphertext.
 */
export class CryptoShreddingManager {
  private readonly config: CryptoShreddingConfig;
  private readonly keys: Map<string, KeyEntry> = new Map();
  private readonly subjectKeys: Map<string, Set<string>> = new Map();
  private readonly auditLog: ShredAuditEntry[] = [];

  constructor(config?: Partial<CryptoShreddingConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ===========================================================================
  // Key Management
  // ===========================================================================

  /**
   * Generate a new encryption key for a data subject
   */
  async generateKey(dataSubject: string): Promise<string> {
    const keyId = this.generateKeyId();
    const key = secureRandomBytes(32);

    const entry: KeyEntry = {
      keyId,
      key,
      algorithm: this.config.algorithm,
      createdAt: Date.now(),
      dataSubjects: [dataSubject],
      resources: [],
      shredded: false,
    };

    this.keys.set(keyId, entry);

    // Track subject → key mapping
    if (!this.subjectKeys.has(dataSubject)) {
      this.subjectKeys.set(dataSubject, new Set());
    }
    this.subjectKeys.get(dataSubject)!.add(keyId);

    return keyId;
  }

  /**
   * Get encryption key by ID (returns null if shredded)
   */
  getKey(keyId: string): Uint8Array | null {
    const entry = this.keys.get(keyId);
    if (!entry || entry.shredded) {
      return null;
    }
    return entry.key;
  }

  /**
   * Get key for a data subject (creates new if none exists)
   */
  async getOrCreateKeyForSubject(dataSubject: string): Promise<string> {
    const subjectKeyIds = this.subjectKeys.get(dataSubject);
    if (subjectKeyIds && subjectKeyIds.size > 0) {
      // Return first non-shredded key
      for (const keyId of subjectKeyIds) {
        const entry = this.keys.get(keyId);
        if (entry && !entry.shredded) {
          return keyId;
        }
      }
    }

    // No active key, generate new one
    return this.generateKey(dataSubject);
  }

  /**
   * Associate a resource with a key
   */
  registerResource(keyId: string, resource: string): void {
    const entry = this.keys.get(keyId);
    if (!entry) {
      throw new Error(`Key not found: ${keyId}`);
    }
    if (entry.shredded) {
      throw new Error(`Cannot register resource with shredded key: ${keyId}`);
    }
    if (!entry.resources.includes(resource)) {
      entry.resources.push(resource);
    }
  }

  /**
   * Associate additional data subject with a key
   */
  addDataSubject(keyId: string, dataSubject: string): void {
    const entry = this.keys.get(keyId);
    if (!entry) {
      throw new Error(`Key not found: ${keyId}`);
    }
    if (!entry.dataSubjects.includes(dataSubject)) {
      entry.dataSubjects.push(dataSubject);
    }

    if (!this.subjectKeys.has(dataSubject)) {
      this.subjectKeys.set(dataSubject, new Set());
    }
    this.subjectKeys.get(dataSubject)!.add(keyId);
  }

  // ===========================================================================
  // Cryptographic Shredding
  // ===========================================================================

  /**
   * Shred a specific key
   */
  async shredKey(
    keyId: string,
    reason: ShredReason,
    performedBy: string,
    dataSubject?: string
  ): Promise<ShredResult> {
    const entry = this.keys.get(keyId);
    if (!entry) {
      return {
        success: false,
        keyId,
        affectedResources: 0,
        auditEntry: this.createEmptyAuditEntry(keyId, reason, performedBy),
        error: 'Key not found',
      };
    }

    if (entry.shredded) {
      return {
        success: false,
        keyId,
        affectedResources: 0,
        auditEntry: this.createEmptyAuditEntry(keyId, reason, performedBy),
        error: 'Key already shredded',
      };
    }

    // Hash the key before shredding for audit verification
    const keyHash = await this.hashKey(entry.key);

    // Perform secure overwrite if enabled
    if (this.config.secureOverwrite) {
      this.securelyOverwriteKey(entry);
    }

    // Mark as shredded
    entry.shredded = true;
    entry.shreddedAt = Date.now();

    // Create audit entry
    const auditEntry: ShredAuditEntry = {
      id: this.generateAuditId(),
      keyId,
      dataSubject: dataSubject ?? entry.dataSubjects[0] ?? 'unknown',
      shreddedAt: Date.now(),
      reason,
      affectedResources: [...entry.resources],
      performedBy,
      keyHash,
    };

    this.auditLog.push(auditEntry);

    return {
      success: true,
      keyId,
      affectedResources: entry.resources.length,
      auditEntry,
    };
  }

  /**
   * Shred all keys for a data subject (GDPR erasure)
   */
  async shredAllForSubject(
    dataSubject: string,
    reason: ShredReason,
    performedBy: string
  ): Promise<ShredResult[]> {
    const keyIds = this.subjectKeys.get(dataSubject);
    if (!keyIds || keyIds.size === 0) {
      return [];
    }

    const results: ShredResult[] = [];
    for (const keyId of keyIds) {
      const result = await this.shredKey(keyId, reason, performedBy, dataSubject);
      results.push(result);
    }

    return results;
  }

  /**
   * Check if data for a subject is effectively erased
   */
  isSubjectErased(dataSubject: string): boolean {
    const keyIds = this.subjectKeys.get(dataSubject);
    if (!keyIds || keyIds.size === 0) {
      return true; // No keys means no data
    }

    for (const keyId of keyIds) {
      const entry = this.keys.get(keyId);
      if (entry && !entry.shredded) {
        return false;
      }
    }

    return true;
  }

  // ===========================================================================
  // Encryption/Decryption
  // ===========================================================================

  /**
   * Encrypt data with a subject's key
   */
  async encrypt(
    keyId: string,
    plaintext: Uint8Array
  ): Promise<{ ciphertext: Uint8Array; iv: Uint8Array }> {
    const key = this.getKey(keyId);
    if (!key) {
      throw new Error(`Key not available: ${keyId}`);
    }

    const iv = secureRandomBytes(12); // 96-bit IV for AES-GCM

    // Use Web Crypto API - create a copy to ensure ArrayBuffer compatibility
    const keyBuffer = new Uint8Array(key).buffer;
    const ivBuffer = new Uint8Array(iv).buffer;
    const plaintextBuffer = new Uint8Array(plaintext).buffer;

    const cryptoKey = await crypto.subtle.importKey(
      'raw',
      keyBuffer,
      { name: 'AES-GCM' },
      false,
      ['encrypt']
    );

    const ciphertext = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: new Uint8Array(ivBuffer) },
      cryptoKey,
      plaintextBuffer
    );

    return {
      ciphertext: new Uint8Array(ciphertext),
      iv: new Uint8Array(iv),
    };
  }

  /**
   * Decrypt data with a subject's key
   */
  async decrypt(
    keyId: string,
    ciphertext: Uint8Array,
    iv: Uint8Array
  ): Promise<Uint8Array> {
    const key = this.getKey(keyId);
    if (!key) {
      throw new Error(`Key not available (possibly shredded): ${keyId}`);
    }

    // Use Web Crypto API - create a copy to ensure ArrayBuffer compatibility
    const keyBuffer = new Uint8Array(key).buffer;
    const ivBuffer = new Uint8Array(iv).buffer;
    const ciphertextBuffer = new Uint8Array(ciphertext).buffer;

    const cryptoKey = await crypto.subtle.importKey(
      'raw',
      keyBuffer,
      { name: 'AES-GCM' },
      false,
      ['decrypt']
    );

    const plaintext = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: new Uint8Array(ivBuffer) },
      cryptoKey,
      ciphertextBuffer
    );

    return new Uint8Array(plaintext);
  }

  // ===========================================================================
  // Audit Log
  // ===========================================================================

  /**
   * Get audit log entries for a data subject
   */
  getAuditLogForSubject(dataSubject: string): ShredAuditEntry[] {
    return this.auditLog.filter(entry => entry.dataSubject === dataSubject);
  }

  /**
   * Get all audit log entries
   */
  getFullAuditLog(): ShredAuditEntry[] {
    return [...this.auditLog];
  }

  /**
   * Verify a shredding operation occurred
   */
  verifyShredding(keyId: string): ShredAuditEntry | null {
    return this.auditLog.find(entry => entry.keyId === keyId) ?? null;
  }

  /**
   * Get GDPR compliance report for a data subject
   */
  getComplianceReport(dataSubject: string): {
    dataSubject: string;
    isErased: boolean;
    totalKeys: number;
    shreddedKeys: number;
    activeKeys: number;
    affectedResources: string[];
    auditTrail: ShredAuditEntry[];
  } {
    const keyIds = this.subjectKeys.get(dataSubject) ?? new Set();
    let shreddedCount = 0;
    let activeCount = 0;
    const allResources: string[] = [];

    for (const keyId of keyIds) {
      const entry = this.keys.get(keyId);
      if (entry) {
        if (entry.shredded) {
          shreddedCount++;
        } else {
          activeCount++;
        }
        allResources.push(...entry.resources);
      }
    }

    return {
      dataSubject,
      isErased: this.isSubjectErased(dataSubject),
      totalKeys: keyIds.size,
      shreddedKeys: shreddedCount,
      activeKeys: activeCount,
      affectedResources: [...new Set(allResources)],
      auditTrail: this.getAuditLogForSubject(dataSubject),
    };
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private generateKeyId(): string {
    const bytes = secureRandomBytes(16);
    return `key_${Buffer.from(bytes).toString('hex')}`;
  }

  private generateAuditId(): string {
    const bytes = secureRandomBytes(16);
    return `aud_${Buffer.from(bytes).toString('hex')}`;
  }

  private async hashKey(key: Uint8Array): Promise<string> {
    const hashBytes = await hash(key, 'SHA-256');
    return Buffer.from(hashBytes).toString('hex');
  }

  private securelyOverwriteKey(entry: KeyEntry): void {
    const keyLength = entry.key.length;

    for (let pass = 0; pass < this.config.overwritePasses; pass++) {
      // Overwrite with random data
      const randomData = secureRandomBytes(keyLength);
      for (let i = 0; i < keyLength; i++) {
        entry.key[i] = randomData[i];
      }
    }

    // Final overwrite with zeros
    entry.key.fill(0);
  }

  private createEmptyAuditEntry(
    keyId: string,
    reason: ShredReason,
    performedBy: string
  ): ShredAuditEntry {
    return {
      id: 'error',
      keyId,
      dataSubject: 'unknown',
      shreddedAt: Date.now(),
      reason,
      affectedResources: [],
      performedBy,
      keyHash: '',
    };
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new crypto shredding manager
 */
export function createCryptoShreddingManager(
  config?: Partial<CryptoShreddingConfig>
): CryptoShreddingManager {
  const salt = config?.salt ?? secureRandomBytes(32);
  return new CryptoShreddingManager({ ...config, salt });
}

/**
 * Execute a GDPR erasure request
 */
export async function executeGDPRErasure(
  manager: CryptoShreddingManager,
  dataSubject: string,
  performedBy: string
): Promise<{
  success: boolean;
  results: ShredResult[];
  complianceReport: ReturnType<CryptoShreddingManager['getComplianceReport']>;
}> {
  const results = await manager.shredAllForSubject(
    dataSubject,
    ShredReason.GDPR_ERASURE_REQUEST,
    performedBy
  );

  const allSuccess = results.every(r => r.success);
  const complianceReport = manager.getComplianceReport(dataSubject);

  return {
    success: allSuccess,
    results,
    complianceReport,
  };
}
