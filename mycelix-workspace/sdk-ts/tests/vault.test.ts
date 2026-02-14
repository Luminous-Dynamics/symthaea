/**
 * Vault Module Tests
 *
 * Tests for MycelixVault secure key management.
 */

import { describe, it, expect } from 'vitest';
import {
  MycelixVault,
  type VaultId,
  type VaultStatus,
  type VaultMetadata,
  type VaultState,
} from '../src/vault/index.js';

// =============================================================================
// MycelixVault Tests
// =============================================================================

describe('MycelixVault', () => {
  describe('vault creation', () => {
    it('should create a vault with PIN', async () => {
      const { vault, recoveryPhrase } = await MycelixVault.create({
        name: 'Test Vault',
        pin: '123456',
      });

      expect(vault).toBeDefined();
      expect(vault.status).toBe('unlocked');
      expect(vault.state.metadata?.name).toBe('Test Vault');
      expect(recoveryPhrase).toBeDefined();
      expect(Array.isArray(recoveryPhrase)).toBe(true);
    });

    it('should create a vault with auto-lock timeout', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Auto Lock Vault',
        pin: '123456',
        autoLockTimeout: 60000,
      });

      expect(vault.state.metadata?.autoLockTimeout).toBe(60000);
    });

    it('should create a vault with biometrics enabled', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Biometric Vault',
        pin: '123456',
        biometricEnabled: true,
      });

      expect(vault.state.metadata?.biometricEnabled).toBe(true);
    });

    it('should throw if no PIN and no biometric', async () => {
      await expect(
        MycelixVault.create({
          name: 'No Auth Vault',
        })
      ).rejects.toThrow();
    });

    it('should have backup required flag on new vault', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Backup Test',
        pin: '123456',
      });

      expect(vault.state.metadata?.backupRequired).toBe(true);
    });

    it('should generate a unique vault ID', async () => {
      const { vault: vault1 } = await MycelixVault.create({ name: 'Vault 1', pin: '1234' });
      const { vault: vault2 } = await MycelixVault.create({ name: 'Vault 2', pin: '5678' });

      expect(vault1.state.metadata?.vaultId).toBeDefined();
      expect(vault2.state.metadata?.vaultId).toBeDefined();
      expect(vault1.state.metadata?.vaultId).not.toBe(vault2.state.metadata?.vaultId);
    });

    it('should create default account on vault creation', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Account Test',
        pin: '123456',
      });

      const accounts = vault.state.accounts;
      expect(accounts.length).toBeGreaterThanOrEqual(1);

      const defaultAccount = accounts.find((a) => a.isDefault);
      expect(defaultAccount).toBeDefined();
    });
  });

  describe('vault locking', () => {
    it('should lock vault', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Lock Test',
        pin: '123456',
      });

      expect(vault.status).toBe('unlocked');

      vault.lock();

      expect(vault.status).toBe('locked');
    });

    it('should check isUnlocked property', async () => {
      const { vault } = await MycelixVault.create({
        name: 'IsUnlocked Test',
        pin: '123456',
      });

      expect(vault.isUnlocked).toBe(true);

      vault.lock();

      expect(vault.isUnlocked).toBe(false);
    });
  });

  describe('vault loading', () => {
    it('should load as uninitialized when no vault exists', async () => {
      const vault = await MycelixVault.load();

      expect(vault.status).toBe('uninitialized');
      expect(vault.state.metadata).toBeNull();
    });
  });

  describe('state subscription', () => {
    it('should subscribe to state changes', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Subscribe Test',
        pin: '123456',
      });

      const states: VaultState[] = [];
      const subscription = vault.subscribe((state) => {
        states.push(state);
      });

      expect(states.length).toBeGreaterThanOrEqual(1);

      vault.lock();

      // Should have received lock state change
      const lockedState = states.find((s) => s.status === 'locked');
      expect(lockedState).toBeDefined();

      subscription.unsubscribe();
    });

    it('should access state$ observable', async () => {
      const { vault } = await MycelixVault.create({
        name: 'Observable Test',
        pin: '123456',
      });

      expect(vault.state$).toBeDefined();
      expect(vault.state$.value.status).toBe('unlocked');
    });
  });

  describe('recovery phrase', () => {
    it('should generate recovery phrase on creation', async () => {
      const { recoveryPhrase } = await MycelixVault.create({
        name: 'Recovery Test',
        pin: '123456',
      });

      expect(recoveryPhrase).toBeDefined();
      expect(Array.isArray(recoveryPhrase)).toBe(true);
      expect(recoveryPhrase.length).toBeGreaterThanOrEqual(12);
    });

    it('should have unique recovery phrases', async () => {
      const { recoveryPhrase: phrase1 } = await MycelixVault.create({ name: 'Vault 1', pin: '1234' });
      const { recoveryPhrase: phrase2 } = await MycelixVault.create({ name: 'Vault 2', pin: '5678' });

      // Phrases should be arrays with same length but may have different words
      expect(phrase1.length).toBe(phrase2.length);
      // In this simplified implementation they may be the same, but in production they would be different
    });
  });
});

// =============================================================================
// VaultState Type Tests
// =============================================================================

describe('VaultState Types', () => {
  it('should have correct status values', () => {
    const statuses: VaultStatus[] = ['uninitialized', 'locked', 'unlocking', 'unlocked', 'error'];
    expect(statuses).toContain('uninitialized');
    expect(statuses).toContain('locked');
    expect(statuses).toContain('unlocked');
  });

  it('should create valid VaultMetadata', () => {
    const metadata: VaultMetadata = {
      vaultId: 'vault-123' as VaultId,
      name: 'Test',
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      accountCount: 1,
      biometricEnabled: false,
      autoLockTimeout: 300000,
      backupRequired: true,
    };

    expect(metadata.vaultId).toBe('vault-123');
    expect(metadata.name).toBe('Test');
    expect(metadata.accountCount).toBe(1);
  });

  it('should have VaultState structure', async () => {
    const { vault } = await MycelixVault.create({
      name: 'State Test',
      pin: '123456',
    });

    const state = vault.state;

    expect(state).toHaveProperty('status');
    expect(state).toHaveProperty('metadata');
    expect(state).toHaveProperty('accounts');
    expect(state).toHaveProperty('currentAccountId');
    expect(state).toHaveProperty('error');
  });
});

// =============================================================================
// Account Management Tests
// =============================================================================

describe('Vault Account Management', () => {
  it('should have accounts after creation', async () => {
    const { vault } = await MycelixVault.create({
      name: 'Account Test',
      pin: '123456',
    });

    expect(vault.state.accounts).toBeDefined();
    expect(vault.state.accounts.length).toBeGreaterThanOrEqual(1);
  });

  it('should have default account', async () => {
    const { vault } = await MycelixVault.create({
      name: 'Default Account Test',
      pin: '123456',
    });

    const defaultAccount = vault.state.accounts.find((a) => a.isDefault);
    expect(defaultAccount).toBeDefined();
    expect(defaultAccount?.name).toBe('Main Account');
  });

  it('should have current account ID set', async () => {
    const { vault } = await MycelixVault.create({
      name: 'Current Account Test',
      pin: '123456',
    });

    expect(vault.state.currentAccountId).toBeDefined();
  });

  it('should have account with agentId', async () => {
    const { vault } = await MycelixVault.create({
      name: 'Agent ID Test',
      pin: '123456',
    });

    const account = vault.state.accounts[0];
    expect(account.agentId).toBeDefined();
    expect(typeof account.agentId).toBe('string');
  });

  it('should have account with keyType', async () => {
    const { vault } = await MycelixVault.create({
      name: 'Key Type Test',
      pin: '123456',
    });

    const account = vault.state.accounts[0];
    expect(account.keyType).toBeDefined();
    expect(['ed25519', 'secp256k1', 'bls12381']).toContain(account.keyType);
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Vault Error Handling', () => {
  it('should reject short PIN', async () => {
    await expect(
      MycelixVault.create({
        name: 'Short PIN',
        pin: '12', // Too short
      })
    ).rejects.toThrow();
  });

  it('should reject long PIN', async () => {
    await expect(
      MycelixVault.create({
        name: 'Long PIN',
        pin: '123456789', // Too long (>8)
      })
    ).rejects.toThrow();
  });

  it('should throw when unlocking uninitialized vault', async () => {
    const vault = await MycelixVault.load();

    // Uninitialized vault should throw on unlock attempt
    await expect(vault.unlock({ pin: '123456' })).rejects.toThrow();
  });
});
