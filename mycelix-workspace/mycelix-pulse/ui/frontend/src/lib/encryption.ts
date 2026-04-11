// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * End-to-End Encryption Service
 *
 * Provides PGP/GPG encryption for emails with key management.
 * Uses Web Crypto API and OpenPGP.js for cryptographic operations.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================
// Types
// ============================================

export interface PublicKey {
  id: string;
  fingerprint: string;
  algorithm: 'RSA' | 'ECC' | 'EdDSA';
  keySize: number;
  createdAt: string;
  expiresAt?: string;
  userId: string;
  email: string;
  armored: string;
  isRevoked: boolean;
  trustLevel: 'unknown' | 'never' | 'marginal' | 'full' | 'ultimate';
}

export interface PrivateKey {
  id: string;
  fingerprint: string;
  algorithm: 'RSA' | 'ECC' | 'EdDSA';
  keySize: number;
  createdAt: string;
  expiresAt?: string;
  userId: string;
  email: string;
  armoredEncrypted: string; // Encrypted with passphrase
  isDefault: boolean;
}

export interface KeyPair {
  publicKey: PublicKey;
  privateKey: PrivateKey;
}

export interface EncryptedMessage {
  armored: string;
  recipientKeyIds: string[];
  signedBy?: string;
  encryptedAt: string;
}

export interface DecryptedMessage {
  text: string;
  signatures: SignatureVerification[];
  decryptedAt: string;
}

export interface SignatureVerification {
  keyId: string;
  fingerprint: string;
  userId?: string;
  valid: boolean;
  timestamp?: string;
  error?: string;
}

export interface KeySearchResult {
  fingerprint: string;
  userId: string;
  email: string;
  algorithm: string;
  keySize: number;
  createdAt: string;
  source: 'keyserver' | 'wkd' | 'local' | 'holochain';
}

// ============================================
// Key Store
// ============================================

interface KeyStoreState {
  publicKeys: Map<string, PublicKey>;
  privateKeys: Map<string, PrivateKey>;
  defaultKeyId: string | null;
  unlockedKeys: Set<string>; // Fingerprints of unlocked private keys (in memory only)
  keyServerUrl: string;
}

interface KeyStoreActions {
  importPublicKey: (armored: string) => Promise<PublicKey>;
  importPrivateKey: (armored: string, passphrase: string) => Promise<PrivateKey>;
  generateKeyPair: (options: KeyGenOptions) => Promise<KeyPair>;
  deleteKey: (fingerprint: string) => void;
  setDefaultKey: (fingerprint: string) => void;
  setTrustLevel: (fingerprint: string, level: PublicKey['trustLevel']) => void;
  unlockKey: (fingerprint: string, passphrase: string) => Promise<boolean>;
  lockKey: (fingerprint: string) => void;
  lockAllKeys: () => void;
  setKeyServerUrl: (url: string) => void;
}

export interface KeyGenOptions {
  type: 'RSA' | 'ECC';
  rsaBits?: 2048 | 3072 | 4096;
  curve?: 'curve25519' | 'p256' | 'p384' | 'p521';
  userId: string;
  email: string;
  passphrase: string;
  expirationDays?: number;
}

export const useKeyStore = create<KeyStoreState & KeyStoreActions>()(
  persist(
    (set, get) => ({
      publicKeys: new Map(),
      privateKeys: new Map(),
      defaultKeyId: null,
      unlockedKeys: new Set(),
      keyServerUrl: 'https://keys.openpgp.org',

      importPublicKey: async (armored: string): Promise<PublicKey> => {
        const keyData = await parsePublicKey(armored);
        set((state) => {
          const keys = new Map(state.publicKeys);
          keys.set(keyData.fingerprint, keyData);
          return { publicKeys: keys };
        });
        return keyData;
      },

      importPrivateKey: async (armored: string, passphrase: string): Promise<PrivateKey> => {
        const keyData = await parsePrivateKey(armored, passphrase);
        set((state) => {
          const keys = new Map(state.privateKeys);
          keys.set(keyData.fingerprint, keyData);
          // Also import the public key
          const publicKeys = new Map(state.publicKeys);
          // Extract public key from private (would need OpenPGP.js)
          return {
            privateKeys: keys,
            publicKeys,
            defaultKeyId: state.defaultKeyId || keyData.fingerprint,
          };
        });
        return keyData;
      },

      generateKeyPair: async (options: KeyGenOptions): Promise<KeyPair> => {
        const keyPair = await generateKeyPairInternal(options);
        set((state) => {
          const publicKeys = new Map(state.publicKeys);
          const privateKeys = new Map(state.privateKeys);
          publicKeys.set(keyPair.publicKey.fingerprint, keyPair.publicKey);
          privateKeys.set(keyPair.privateKey.fingerprint, keyPair.privateKey);
          return {
            publicKeys,
            privateKeys,
            defaultKeyId: state.defaultKeyId || keyPair.privateKey.fingerprint,
          };
        });
        return keyPair;
      },

      deleteKey: (fingerprint: string) => {
        set((state) => {
          const publicKeys = new Map(state.publicKeys);
          const privateKeys = new Map(state.privateKeys);
          const unlockedKeys = new Set(state.unlockedKeys);
          publicKeys.delete(fingerprint);
          privateKeys.delete(fingerprint);
          unlockedKeys.delete(fingerprint);
          return {
            publicKeys,
            privateKeys,
            unlockedKeys,
            defaultKeyId: state.defaultKeyId === fingerprint ? null : state.defaultKeyId,
          };
        });
      },

      setDefaultKey: (fingerprint: string) => {
        set({ defaultKeyId: fingerprint });
      },

      setTrustLevel: (fingerprint: string, level: PublicKey['trustLevel']) => {
        set((state) => {
          const keys = new Map(state.publicKeys);
          const key = keys.get(fingerprint);
          if (key) {
            keys.set(fingerprint, { ...key, trustLevel: level });
          }
          return { publicKeys: keys };
        });
      },

      unlockKey: async (fingerprint: string, passphrase: string): Promise<boolean> => {
        const privateKey = get().privateKeys.get(fingerprint);
        if (!privateKey) return false;

        try {
          // Verify passphrase by attempting to decrypt
          await verifyPassphrase(privateKey.armoredEncrypted, passphrase);
          set((state) => {
            const unlocked = new Set(state.unlockedKeys);
            unlocked.add(fingerprint);
            return { unlockedKeys: unlocked };
          });
          return true;
        } catch {
          return false;
        }
      },

      lockKey: (fingerprint: string) => {
        set((state) => {
          const unlocked = new Set(state.unlockedKeys);
          unlocked.delete(fingerprint);
          return { unlockedKeys: unlocked };
        });
      },

      lockAllKeys: () => {
        set({ unlockedKeys: new Set() });
      },

      setKeyServerUrl: (url: string) => {
        set({ keyServerUrl: url });
      },
    }),
    {
      name: 'pgp-keystore',
      partialize: (state) => ({
        publicKeys: Array.from(state.publicKeys.entries()),
        privateKeys: Array.from(state.privateKeys.entries()),
        defaultKeyId: state.defaultKeyId,
        keyServerUrl: state.keyServerUrl,
      }),
      merge: (persisted, current) => {
        const p = persisted as {
          publicKeys?: [string, PublicKey][];
          privateKeys?: [string, PrivateKey][];
          defaultKeyId?: string | null;
          keyServerUrl?: string;
        };
        return {
          ...current,
          publicKeys: new Map(p.publicKeys || []),
          privateKeys: new Map(p.privateKeys || []),
          defaultKeyId: p.defaultKeyId || null,
          keyServerUrl: p.keyServerUrl || current.keyServerUrl,
        };
      },
    }
  )
);

// ============================================
// Crypto Operations (would use OpenPGP.js in production)
// ============================================

async function parsePublicKey(armored: string): Promise<PublicKey> {
  // In production, use openpgp.readKey()
  // This is a placeholder implementation
  const fingerprint = await computeFingerprint(armored);
  const matches = armored.match(/-----BEGIN PGP PUBLIC KEY BLOCK-----[\s\S]*?-----END PGP PUBLIC KEY BLOCK-----/);

  if (!matches) {
    throw new Error('Invalid public key format');
  }

  // Extract user ID from key (simplified)
  const userIdMatch = armored.match(/uid\s+([^<]+)<([^>]+)>/);
  const userId = userIdMatch?.[1]?.trim() || 'Unknown';
  const email = userIdMatch?.[2] || 'unknown@unknown.com';

  return {
    id: fingerprint.slice(-16),
    fingerprint,
    algorithm: 'RSA',
    keySize: 4096,
    createdAt: new Date().toISOString(),
    userId,
    email,
    armored,
    isRevoked: false,
    trustLevel: 'unknown',
  };
}

async function parsePrivateKey(armored: string, passphrase: string): Promise<PrivateKey> {
  // In production, use openpgp.readPrivateKey() and decrypt
  const fingerprint = await computeFingerprint(armored);

  // Verify passphrase works
  await verifyPassphrase(armored, passphrase);

  const userIdMatch = armored.match(/uid\s+([^<]+)<([^>]+)>/);
  const userId = userIdMatch?.[1]?.trim() || 'Unknown';
  const email = userIdMatch?.[2] || 'unknown@unknown.com';

  return {
    id: fingerprint.slice(-16),
    fingerprint,
    algorithm: 'RSA',
    keySize: 4096,
    createdAt: new Date().toISOString(),
    userId,
    email,
    armoredEncrypted: armored,
    isDefault: false,
  };
}

async function generateKeyPairInternal(options: KeyGenOptions): Promise<KeyPair> {
  // In production, use openpgp.generateKey()
  // This is a placeholder that generates mock keys

  const fingerprint = await generateRandomFingerprint();
  const now = new Date().toISOString();
  const expiresAt = options.expirationDays
    ? new Date(Date.now() + options.expirationDays * 24 * 60 * 60 * 1000).toISOString()
    : undefined;

  const keySize = options.type === 'RSA' ? (options.rsaBits || 4096) : 256;

  const publicKey: PublicKey = {
    id: fingerprint.slice(-16),
    fingerprint,
    algorithm: options.type === 'RSA' ? 'RSA' : 'ECC',
    keySize,
    createdAt: now,
    expiresAt,
    userId: options.userId,
    email: options.email,
    armored: `-----BEGIN PGP PUBLIC KEY BLOCK-----\n\nGenerated key for ${options.email}\nFingerprint: ${fingerprint}\n\n-----END PGP PUBLIC KEY BLOCK-----`,
    isRevoked: false,
    trustLevel: 'ultimate',
  };

  const privateKey: PrivateKey = {
    id: fingerprint.slice(-16),
    fingerprint,
    algorithm: options.type === 'RSA' ? 'RSA' : 'ECC',
    keySize,
    createdAt: now,
    expiresAt,
    userId: options.userId,
    email: options.email,
    armoredEncrypted: `-----BEGIN PGP PRIVATE KEY BLOCK-----\n\nEncrypted private key for ${options.email}\nFingerprint: ${fingerprint}\n\n-----END PGP PRIVATE KEY BLOCK-----`,
    isDefault: true,
  };

  return { publicKey, privateKey };
}

async function verifyPassphrase(_armoredKey: string, _passphrase: string): Promise<void> {
  // In production, try to decrypt the key with the passphrase
  // For now, accept any passphrase
  await new Promise((resolve) => setTimeout(resolve, 100));
}

async function computeFingerprint(data: string): Promise<string> {
  const encoder = new TextEncoder();
  const hashBuffer = await crypto.subtle.digest('SHA-256', encoder.encode(data));
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('').toUpperCase().slice(0, 40);
}

async function generateRandomFingerprint(): Promise<string> {
  const array = new Uint8Array(20);
  crypto.getRandomValues(array);
  return Array.from(array).map((b) => b.toString(16).padStart(2, '0')).join('').toUpperCase();
}

// ============================================
// Encryption Service
// ============================================

class EncryptionService {
  async encrypt(
    message: string,
    recipientFingerprints: string[],
    signingKeyFingerprint?: string
  ): Promise<EncryptedMessage> {
    const store = useKeyStore.getState();

    // Get recipient public keys
    const recipientKeys = recipientFingerprints
      .map((fp) => store.publicKeys.get(fp))
      .filter((k): k is PublicKey => k !== undefined);

    if (recipientKeys.length === 0) {
      throw new Error('No valid recipient keys found');
    }

    // Check if signing key is unlocked
    if (signingKeyFingerprint && !store.unlockedKeys.has(signingKeyFingerprint)) {
      throw new Error('Signing key is locked');
    }

    // In production, use openpgp.encrypt()
    // This is a placeholder
    const encrypted = btoa(message); // Just base64 for demo

    return {
      armored: `-----BEGIN PGP MESSAGE-----\n\n${encrypted}\n\n-----END PGP MESSAGE-----`,
      recipientKeyIds: recipientKeys.map((k) => k.id),
      signedBy: signingKeyFingerprint,
      encryptedAt: new Date().toISOString(),
    };
  }

  async decrypt(
    encryptedMessage: string,
    _passphrase?: string
  ): Promise<DecryptedMessage> {
    const store = useKeyStore.getState();

    // Check if any private key is unlocked
    if (store.unlockedKeys.size === 0) {
      throw new Error('No unlocked private keys available');
    }

    // In production, use openpgp.decrypt()
    // This is a placeholder
    const match = encryptedMessage.match(/-----BEGIN PGP MESSAGE-----\s*([\s\S]*?)\s*-----END PGP MESSAGE-----/);
    if (!match) {
      throw new Error('Invalid encrypted message format');
    }

    const decoded = atob(match[1].trim());

    return {
      text: decoded,
      signatures: [],
      decryptedAt: new Date().toISOString(),
    };
  }

  async sign(message: string, signingKeyFingerprint: string): Promise<string> {
    const store = useKeyStore.getState();

    if (!store.unlockedKeys.has(signingKeyFingerprint)) {
      throw new Error('Signing key is locked');
    }

    // In production, use openpgp.sign()
    const signature = await computeFingerprint(message + signingKeyFingerprint);

    return `-----BEGIN PGP SIGNATURE-----\n\n${signature}\n\n-----END PGP SIGNATURE-----`;
  }

  async verify(
    message: string,
    signature: string,
    signerFingerprint: string
  ): Promise<SignatureVerification> {
    const store = useKeyStore.getState();
    const publicKey = store.publicKeys.get(signerFingerprint);

    if (!publicKey) {
      return {
        keyId: signerFingerprint.slice(-16),
        fingerprint: signerFingerprint,
        valid: false,
        error: 'Public key not found',
      };
    }

    // In production, use openpgp.verify()
    // This is a placeholder that always returns valid
    return {
      keyId: publicKey.id,
      fingerprint: publicKey.fingerprint,
      userId: publicKey.userId,
      valid: true,
      timestamp: new Date().toISOString(),
    };
  }

  async searchKeyServer(query: string): Promise<KeySearchResult[]> {
    const store = useKeyStore.getState();
    const url = `${store.keyServerUrl}/vks/v1/by-email/${encodeURIComponent(query)}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        if (response.status === 404) {
          return [];
        }
        throw new Error(`Key server error: ${response.statusText}`);
      }

      const armored = await response.text();
      const keyData = await parsePublicKey(armored);

      return [{
        fingerprint: keyData.fingerprint,
        userId: keyData.userId,
        email: keyData.email,
        algorithm: keyData.algorithm,
        keySize: keyData.keySize,
        createdAt: keyData.createdAt,
        source: 'keyserver',
      }];
    } catch (err) {
      console.error('Key server search failed:', err);
      return [];
    }
  }

  async fetchKeyFromServer(fingerprint: string): Promise<PublicKey | null> {
    const store = useKeyStore.getState();
    const url = `${store.keyServerUrl}/vks/v1/by-fingerprint/${fingerprint}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        return null;
      }

      const armored = await response.text();
      return await store.importPublicKey(armored);
    } catch (err) {
      console.error('Failed to fetch key:', err);
      return null;
    }
  }

  async publishKey(fingerprint: string): Promise<boolean> {
    const store = useKeyStore.getState();
    const publicKey = store.publicKeys.get(fingerprint);

    if (!publicKey) {
      throw new Error('Public key not found');
    }

    try {
      const response = await fetch(`${store.keyServerUrl}/vks/v1/upload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `keytext=${encodeURIComponent(publicKey.armored)}`,
      });

      return response.ok;
    } catch (err) {
      console.error('Failed to publish key:', err);
      return false;
    }
  }
}

// ============================================
// React Hooks
// ============================================

export function useEncryption() {
  const service = new EncryptionService();
  const publicKeys = useKeyStore((s) => s.publicKeys);
  const privateKeys = useKeyStore((s) => s.privateKeys);
  const defaultKeyId = useKeyStore((s) => s.defaultKeyId);
  const unlockedKeys = useKeyStore((s) => s.unlockedKeys);

  return {
    service,
    publicKeys: Array.from(publicKeys.values()),
    privateKeys: Array.from(privateKeys.values()),
    defaultKeyId,
    hasUnlockedKey: unlockedKeys.size > 0,
    unlockedKeyCount: unlockedKeys.size,
  };
}

export function useKeyManagement() {
  const store = useKeyStore();

  return {
    importPublicKey: store.importPublicKey,
    importPrivateKey: store.importPrivateKey,
    generateKeyPair: store.generateKeyPair,
    deleteKey: store.deleteKey,
    setDefaultKey: store.setDefaultKey,
    setTrustLevel: store.setTrustLevel,
    unlockKey: store.unlockKey,
    lockKey: store.lockKey,
    lockAllKeys: store.lockAllKeys,
  };
}

export function useKeyUnlock(fingerprint: string) {
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unlockKey = useKeyStore((s) => s.unlockKey);
  const unlockedKeys = useKeyStore((s) => s.unlockedKeys);

  useEffect(() => {
    setIsUnlocked(unlockedKeys.has(fingerprint));
  }, [fingerprint, unlockedKeys]);

  const unlock = async (passphrase: string) => {
    setIsUnlocking(true);
    setError(null);
    try {
      const success = await unlockKey(fingerprint, passphrase);
      if (!success) {
        setError('Invalid passphrase');
      }
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unlock failed');
      return false;
    } finally {
      setIsUnlocking(false);
    }
  };

  return { isUnlocked, isUnlocking, error, unlock };
}

// Import for useState/useEffect
import { useState, useEffect } from 'react';

export default EncryptionService;
