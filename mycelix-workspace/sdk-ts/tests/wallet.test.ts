// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Unified Wallet Tests
 *
 * Tests the Glass-Top architecture:
 * - MycelixVault (key management)
 * - Unified Wallet (facade)
 * - Reactive state
 * - Optimistic UI
 * - Identity resolution
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  MycelixVault,
  createVault,
  loadVault,
  type VaultState,
} from '../src/vault/index.js';
import {
  Wallet,
  createWallet,
  type WalletState,
  type Transaction,
} from '../src/wallet/index.js';
import {
  BehaviorSubject,
  Subject,
  Observable,
  map,
  filter,
  distinctUntilChanged,
} from '../src/reactive/index.js';
import {
  IdentityCache,
  formatIdentity,
  truncateAgentId,
  generateAvatarUrl,
} from '../src/wallet/identity-resolver.js';
import {
  OptimisticStateManager,
  createOptimisticTransaction,
  formatCurrency,
  formatRelativeTime,
  eventToToast,
  applyOptimisticBalanceChange,
  getContextAction,
  formatReputationScore,
  getReputationBadge,
  formatRiskLevel,
} from '../src/wallet/react.js';
import {
  getBiometricIcon,
  getBiometricName,
  MockBiometricProvider,
  BiometricManager,
  type BiometricType,
} from '../src/wallet/biometrics.js';

// =============================================================================
// Reactive State Tests
// =============================================================================

describe('Reactive State (Observable)', () => {
  describe('BehaviorSubject', () => {
    it('should emit current value to new subscribers', () => {
      const subject = new BehaviorSubject(42);
      const values: number[] = [];

      subject.subscribe((v) => values.push(v));

      expect(values).toEqual([42]);
    });

    it('should emit new values to all subscribers', () => {
      const subject = new BehaviorSubject(0);
      const values1: number[] = [];
      const values2: number[] = [];

      subject.subscribe((v) => values1.push(v));
      subject.next(1);
      subject.subscribe((v) => values2.push(v));
      subject.next(2);

      expect(values1).toEqual([0, 1, 2]);
      expect(values2).toEqual([1, 2]); // Gets current value (1) on subscribe, then 2
    });

    it('should allow synchronous value access', () => {
      const subject = new BehaviorSubject({ count: 0 });
      expect(subject.value.count).toBe(0);

      subject.next({ count: 5 });
      expect(subject.value.count).toBe(5);
    });

    it('should support unsubscribe', () => {
      const subject = new BehaviorSubject(0);
      const values: number[] = [];

      const sub = subject.subscribe((v) => values.push(v));
      subject.next(1);
      sub.unsubscribe();
      subject.next(2);

      expect(values).toEqual([0, 1]);
    });

    it('should handle complete', () => {
      const subject = new BehaviorSubject(0);
      let completed = false;

      subject.subscribe({
        next: () => {},
        complete: () => {
          completed = true;
        },
      });

      subject.complete();
      expect(completed).toBe(true);
      expect(subject.closed).toBe(true);
    });
  });

  describe('Subject', () => {
    it('should not emit to new subscribers before next()', () => {
      const subject = new Subject<number>();
      const values: number[] = [];

      subject.subscribe((v) => values.push(v));
      expect(values).toEqual([]);

      subject.next(1);
      expect(values).toEqual([1]);
    });
  });

  describe('Operators', () => {
    it('map should transform values', () => {
      const subject = new BehaviorSubject(5);
      const doubled: number[] = [];

      subject.asObservable().pipe(map((x) => x * 2)).subscribe((v) => doubled.push(v));

      subject.next(10);

      expect(doubled).toEqual([10, 20]);
    });

    it('filter should only pass matching values', () => {
      const subject = new BehaviorSubject(0);
      const evens: number[] = [];

      subject.asObservable().pipe(filter((x) => x % 2 === 0)).subscribe((v) => evens.push(v));

      subject.next(1);
      subject.next(2);
      subject.next(3);
      subject.next(4);

      expect(evens).toEqual([0, 2, 4]);
    });

    it('distinctUntilChanged should dedupe consecutive values', () => {
      const subject = new BehaviorSubject('a');
      const values: string[] = [];

      subject.asObservable().pipe(distinctUntilChanged()).subscribe((v) => values.push(v));

      subject.next('a'); // duplicate
      subject.next('b');
      subject.next('b'); // duplicate
      subject.next('a');

      expect(values).toEqual(['a', 'b', 'a']);
    });
  });
});

// =============================================================================
// MycelixVault Tests
// =============================================================================

describe('MycelixVault', () => {
  let vault: MycelixVault;
  let recoveryPhrase: string[];

  beforeEach(async () => {
    const result = await createVault({
      name: 'Test Vault',
      pin: '1234',
    });
    vault = result.vault;
    recoveryPhrase = result.recoveryPhrase;
  });

  afterEach(() => {
    vault.lock();
  });

  describe('Creation', () => {
    it('should create a vault with recovery phrase', () => {
      expect(vault).toBeDefined();
      expect(recoveryPhrase).toHaveLength(12);
      expect(vault.isUnlocked).toBe(true);
    });

    it('should create default account on creation', () => {
      const accounts = vault.getAccounts();
      expect(accounts.length).toBe(1);
      expect(accounts[0].isDefault).toBe(true);
      expect(accounts[0].name).toBe('Main Account');
    });

    it('should reject short PIN', async () => {
      await expect(
        createVault({ name: 'Bad', pin: '12' })
      ).rejects.toThrow('PIN must be 4-8 digits');
    });

    it('should reject vault without auth method', async () => {
      await expect(
        createVault({ name: 'Bad' })
      ).rejects.toThrow('Either PIN or biometric must be enabled');
    });
  });

  describe('Locking/Unlocking', () => {
    it('should lock and unlock with PIN', async () => {
      expect(vault.isUnlocked).toBe(true);

      vault.lock();
      expect(vault.isUnlocked).toBe(false);
      expect(vault.status).toBe('locked');

      await vault.unlock({ pin: '1234' });
      expect(vault.isUnlocked).toBe(true);
      expect(vault.status).toBe('unlocked');
    });

    it('should reject wrong PIN', async () => {
      vault.lock();

      await expect(vault.unlock({ pin: '0000' })).rejects.toThrow('Invalid PIN');
    });

    it('should emit state changes', async () => {
      const states: VaultState[] = [];
      vault.subscribe((state) => states.push({ ...state }));

      vault.lock();
      await vault.unlock({ pin: '1234' });

      expect(states.some((s) => s.status === 'locked')).toBe(true);
      expect(states.some((s) => s.status === 'unlocked')).toBe(true);
    });
  });

  describe('Account Management', () => {
    it('should create additional accounts', async () => {
      await vault.createAccount({ name: 'Savings' });
      const accounts = vault.getAccounts();

      expect(accounts.length).toBe(2);
      expect(accounts[1].name).toBe('Savings');
    });

    it('should switch accounts', async () => {
      const savings = await vault.createAccount({ name: 'Savings' });
      vault.switchAccount(savings.accountId);

      expect(vault.getCurrentAccount()?.name).toBe('Savings');
    });

    it('should reject operations when locked', async () => {
      vault.lock();

      await expect(vault.createAccount({ name: 'Fail' })).rejects.toThrow('Vault is locked');
    });
  });

  describe('Signing', () => {
    it('should sign data', async () => {
      const data = new TextEncoder().encode('Hello, World!');
      const signature = await vault.sign(data);

      expect(signature).toBeInstanceOf(Uint8Array);
      expect(signature.length).toBeGreaterThan(0);
    });

    it('should verify signatures', async () => {
      const data = new TextEncoder().encode('Hello, World!');
      const signature = await vault.sign(data);

      const valid = await vault.verify(data, signature);
      expect(valid).toBe(true);
    });

    it('should reject tampered data', async () => {
      const data = new TextEncoder().encode('Hello, World!');
      const signature = await vault.sign(data);

      const tampered = new TextEncoder().encode('Hello, Hacker!');
      const valid = await vault.verify(tampered, signature);
      expect(valid).toBe(false);
    });
  });

  describe('Recovery', () => {
    it('should return recovery phrase', async () => {
      const phrase = await vault.getRecoveryPhrase();
      expect(phrase).toEqual(recoveryPhrase);
    });

    it('should mark backup as complete', async () => {
      expect(vault.state.metadata?.backupRequired).toBe(true);

      await vault.confirmBackup();

      expect(vault.state.metadata?.backupRequired).toBe(false);
    });
  });
});

// =============================================================================
// Unified Wallet Tests
// =============================================================================

describe('Unified Wallet', () => {
  let wallet: Wallet;

  beforeEach(async () => {
    const result = await createWallet({
      name: 'Test Wallet',
      pin: '1234',
    });
    wallet = result.wallet;
  });

  afterEach(() => {
    wallet.lock();
  });

  describe('Creation and State', () => {
    it('should create wallet and be ready', () => {
      expect(wallet.ready).toBe(true);
      expect(wallet.locked).toBe(false);
    });

    it('should have initial balance', () => {
      const balance = wallet.balance('MYC');
      expect(balance).toBe(1000); // Mock provider gives 1000 MYC
    });

    it('should have identity', () => {
      expect(wallet.identity).toBeDefined();
      expect(wallet.identity?.agentId).toBeDefined();
    });
  });

  describe('Reactive State', () => {
    it('should emit state changes', () => {
      const states: WalletState[] = [];
      wallet.subscribe((state) => states.push(state));

      expect(states.length).toBeGreaterThan(0);
      expect(states[states.length - 1].ready).toBe(true);
    });

    it('should emit balance updates', (done) => {
      wallet.balances$.subscribe((balances) => {
        const myc = balances.get('MYC');
        if (myc) {
          expect(myc.available).toBeDefined();
          done();
        }
      });
    });
  });

  describe('Send Operation', () => {
    it('should send funds with optimistic UI', async () => {
      const initialBalance = wallet.balance('MYC');

      // Create a mock recipient
      const tx = await wallet.send('mock-recipient-id', 100, 'MYC');

      expect(tx.status).toBe('confirmed');
      expect(tx.amount).toBe(100);
      expect(wallet.balance('MYC')).toBe(initialBalance - 100);
    });

    it('should reject insufficient funds', async () => {
      await expect(wallet.send('recipient', 10000, 'MYC')).rejects.toThrow('Insufficient funds');
    });

    it('should add transaction to history', async () => {
      await wallet.send('recipient', 50, 'MYC');

      const transactions = wallet.transactions;
      expect(transactions.length).toBeGreaterThan(0);
      expect(transactions[0].amount).toBe(50);
    });
  });

  describe('Locking', () => {
    it('should lock and clear state', () => {
      wallet.lock();

      expect(wallet.locked).toBe(true);
      expect(wallet.ready).toBe(false);
      expect(wallet.identity).toBeNull();
    });

    it('should unlock with PIN', async () => {
      wallet.lock();
      await wallet.unlockWithPin('1234');

      expect(wallet.locked).toBe(false);
      expect(wallet.ready).toBe(true);
    });
  });

  describe('Identity Resolution', () => {
    it('should resolve identity', async () => {
      const identity = await wallet.resolveIdentity('some-agent-id');

      expect(identity).toBeDefined();
      expect(identity?.agentId).toBe('some-agent-id');
    });
  });

  describe('Payment Request', () => {
    it('should generate payment request', async () => {
      const request = await wallet.requestPayment(100, 'MYC', 'Test payment');

      expect(request.qrData).toBeDefined();
      expect(request.link).toContain('mycelix://pay');
      expect(request.link).toContain('amount=100');
    });
  });
});

// =============================================================================
// Identity Cache Tests
// =============================================================================

describe('IdentityCache', () => {
  let cache: IdentityCache;

  beforeEach(() => {
    cache = new IdentityCache(1000); // 1 second TTL for testing
  });

  it('should cache and retrieve by agentId', () => {
    const identity = { agentId: 'agent1', nickname: '@alice' };
    cache.set(identity);

    expect(cache.get('agent1')).toEqual(identity);
  });

  it('should cache and retrieve by nickname', () => {
    const identity = { agentId: 'agent1', nickname: '@alice' };
    cache.set(identity);

    expect(cache.getByNickname('@alice')).toEqual(identity);
  });

  it('should be case-insensitive for nicknames', () => {
    const identity = { agentId: 'agent1', nickname: '@Alice' };
    cache.set(identity);

    expect(cache.getByNickname('@ALICE')).toEqual(identity);
    expect(cache.getByNickname('@alice')).toEqual(identity);
  });

  it('should expire entries after TTL', async () => {
    const identity = { agentId: 'agent1', nickname: '@alice' };
    cache.set(identity);

    // Wait for TTL
    await new Promise((resolve) => setTimeout(resolve, 1100));

    expect(cache.get('agent1')).toBeNull();
  });

  it('should invalidate entries', () => {
    const identity = { agentId: 'agent1', nickname: '@alice' };
    cache.set(identity);
    cache.invalidate('agent1');

    expect(cache.get('agent1')).toBeNull();
    expect(cache.getByNickname('@alice')).toBeNull();
  });
});

// =============================================================================
// Identity Formatting Tests
// =============================================================================

describe('Identity Formatting', () => {
  it('should format identity with nickname', () => {
    const identity = { agentId: 'agent1', nickname: '@alice', displayName: 'Alice Smith' };

    expect(formatIdentity(identity, 'short')).toBe('@alice');
    expect(formatIdentity(identity, 'full')).toBe('Alice Smith (@alice)');
  });

  it('should fallback to truncated agentId', () => {
    const identity = { agentId: 'uhC0k1234567890abcdef' };

    expect(formatIdentity(identity, 'short')).toBe('uhC0k1...cdef');
  });

  it('should truncate long agent IDs', () => {
    expect(truncateAgentId('uhC0k1234567890abcdef')).toBe('uhC0k1...cdef');
    expect(truncateAgentId('short')).toBe('short');
  });

  it('should generate deterministic avatar URLs', () => {
    const url1 = generateAvatarUrl('agent1');
    const url2 = generateAvatarUrl('agent1');
    const url3 = generateAvatarUrl('agent2');

    expect(url1).toBe(url2);
    expect(url1).not.toBe(url3);
    expect(url1).toContain('dicebear.com');
  });
});

// =============================================================================
// Optimistic State Manager Tests
// =============================================================================

describe('OptimisticStateManager', () => {
  it('should apply optimistic updates', () => {
    const manager = new OptimisticStateManager({ balance: 100 });

    manager.applyOptimistic(
      'tx1',
      (state) => ({ balance: state.balance - 50 }),
      (state) => ({ balance: state.balance + 50 })
    );

    expect(manager.state.balance).toBe(50);
  });

  it('should rollback on failure', () => {
    const manager = new OptimisticStateManager({ balance: 100 });

    const rollback = manager.applyOptimistic(
      'tx1',
      (state) => ({ balance: state.balance - 50 }),
      (state) => ({ balance: state.balance + 50 })
    );

    expect(manager.state.balance).toBe(50);

    rollback();

    expect(manager.state.balance).toBe(100);
  });

  it('should commit confirmed updates', () => {
    const manager = new OptimisticStateManager({ balance: 100 });

    manager.applyOptimistic(
      'tx1',
      (state) => ({ balance: state.balance - 50 }),
      (state) => ({ balance: state.balance + 50 })
    );

    manager.commit('tx1', { balance: 50 });

    expect(manager.state.balance).toBe(50);
  });

  it('should handle multiple pending updates', () => {
    const manager = new OptimisticStateManager({ balance: 100 });

    manager.applyOptimistic(
      'tx1',
      (state) => ({ balance: state.balance - 30 }),
      (state) => ({ balance: state.balance + 30 })
    );

    manager.applyOptimistic(
      'tx2',
      (state) => ({ balance: state.balance - 20 }),
      (state) => ({ balance: state.balance + 20 })
    );

    expect(manager.state.balance).toBe(50);

    manager.commit('tx1', { balance: 70 });
    expect(manager.state.balance).toBe(50); // Still has tx2 pending

    manager.rollback('tx2');
    expect(manager.state.balance).toBe(70);
  });

  it('should notify subscribers', () => {
    const manager = new OptimisticStateManager({ balance: 100 });
    const values: number[] = [];

    manager.subscribe((state) => values.push(state.balance));

    manager.applyOptimistic(
      'tx1',
      (state) => ({ balance: state.balance - 50 }),
      (state) => ({ balance: state.balance + 50 })
    );

    expect(values).toEqual([100, 50]);
  });
});

// =============================================================================
// UI Utility Tests
// =============================================================================

describe('UI Utilities', () => {
  describe('formatCurrency', () => {
    it('should format with proper decimals', () => {
      expect(formatCurrency(1234.5, 'MYC')).toBe('1,234.50');
      expect(formatCurrency(1000, 'USD')).toBe('1,000.00');
    });
  });

  describe('formatRelativeTime', () => {
    it('should format recent times', () => {
      const now = Date.now();

      expect(formatRelativeTime(now - 30000)).toBe('just now');
      expect(formatRelativeTime(now - 120000)).toBe('2m ago');
      expect(formatRelativeTime(now - 3600000)).toBe('1h ago');
      expect(formatRelativeTime(now - 86400000)).toBe('1d ago');
    });
  });

  describe('eventToToast', () => {
    it('should create toast for pending transaction', () => {
      const toast = eventToToast({
        type: 'transaction_pending',
        transaction: {
          id: 'tx1',
          from: { agentId: 'from' },
          to: { agentId: 'to', nickname: '@bob' },
          amount: 50,
          currency: 'MYC',
          direction: 'outgoing',
          status: 'pending',
          createdAt: Date.now(),
        },
      });

      expect(toast?.type).toBe('info');
      expect(toast?.title).toBe('Sending...');
      expect(toast?.message).toContain('@bob');
    });

    it('should create toast for confirmed transaction', () => {
      const toast = eventToToast({
        type: 'transaction_confirmed',
        transaction: {
          id: 'tx1',
          from: { agentId: 'from' },
          to: { agentId: 'to', nickname: '@alice' },
          amount: 100,
          currency: 'MYC',
          direction: 'outgoing',
          status: 'confirmed',
          createdAt: Date.now(),
        },
      });

      expect(toast?.type).toBe('success');
      expect(toast?.title).toBe('Sent!');
    });

    it('should create toast for failed transaction', () => {
      const toast = eventToToast({
        type: 'transaction_failed',
        transaction: {
          id: 'tx1',
          from: { agentId: 'from' },
          to: { agentId: 'to' },
          amount: 100,
          currency: 'MYC',
          direction: 'outgoing',
          status: 'failed',
          createdAt: Date.now(),
        },
        error: new Error('Insufficient funds'),
      });

      expect(toast?.type).toBe('error');
      expect(toast?.message).toBe('Insufficient funds');
    });

    it('should return null for silent events', () => {
      expect(eventToToast({ type: 'refresh_complete' })).toBeNull();
    });
  });

  describe('createOptimisticTransaction', () => {
    it('should create transaction with pending status', () => {
      const tx = createOptimisticTransaction(
        { agentId: 'from', nickname: '@alice' },
        { agentId: 'to', nickname: '@bob' },
        100,
        'MYC',
        'Test'
      );

      expect(tx.status).toBe('pending');
      expect(tx.direction).toBe('outgoing');
      expect(tx.from.nickname).toBe('@alice');
      expect(tx.to.nickname).toBe('@bob');
      expect(tx.id).toContain('optimistic-');
    });
  });
});

// =============================================================================
// Contacts & Favorites Tests
// =============================================================================

import {
  ContactsManager,
  createContactsManager,
  formatContactName,
  truncateAddress,
  getContactInitials,
  getContactColor,
  type Contact,
  type ContactId,
} from '../src/wallet/contacts.js';

describe('ContactsManager', () => {
  let contacts: ContactsManager;

  beforeEach(async () => {
    contacts = createContactsManager();
    await contacts.initialize();
  });

  describe('Add Contact', () => {
    it('should add a new contact', async () => {
      const contact = await contacts.add({
        address: '@alice',
        nickname: 'Alice from Work',
      });

      expect(contact.address).toBe('@alice');
      expect(contact.nickname).toBe('Alice from Work');
      expect(contact.id).toBeDefined();
    });

    it('should reject duplicate addresses', async () => {
      await contacts.add({ address: '@alice' });

      await expect(contacts.add({ address: '@alice' })).rejects.toThrow(
        'Contact with address @alice already exists'
      );
    });

    it('should handle case-insensitive duplicate check', async () => {
      await contacts.add({ address: '@Alice' });

      await expect(contacts.add({ address: '@ALICE' })).rejects.toThrow();
    });
  });

  describe('Favorites', () => {
    it('should mark contact as favorite', async () => {
      const contact = await contacts.add({
        address: '@bob',
        favorite: true,
      });

      expect(contact.favorite).toBe(true);
      expect(contacts.favorites).toContainEqual(expect.objectContaining({ address: '@bob' }));
    });

    it('should toggle favorite status', async () => {
      const contact = await contacts.add({ address: '@carol' });
      expect(contact.favorite).toBe(false);

      const toggled = await contacts.toggleFavorite(contact.id);
      expect(toggled.favorite).toBe(true);

      const toggledBack = await contacts.toggleFavorite(contact.id);
      expect(toggledBack.favorite).toBe(false);
    });
  });

  describe('Search', () => {
    beforeEach(async () => {
      await contacts.add({ address: '@alice', nickname: 'Alice Wonder', tags: ['work'] });
      await contacts.add({ address: '@bob', nickname: 'Bob Builder', tags: ['personal'] });
      await contacts.add({ address: '@charlie', notes: 'Met at conference' });
    });

    it('should search by nickname', () => {
      const results = contacts.search('alice');
      expect(results.length).toBe(1);
      expect(results[0].address).toBe('@alice');
    });

    it('should search by address', () => {
      const results = contacts.search('@bob');
      expect(results.length).toBe(1);
    });

    it('should search by tag', () => {
      const results = contacts.search('work');
      expect(results.length).toBe(1);
    });

    it('should search by notes', () => {
      const results = contacts.search('conference');
      expect(results.length).toBe(1);
      expect(results[0].address).toBe('@charlie');
    });
  });

  describe('Transaction Recording', () => {
    it('should auto-create contact for new recipient', async () => {
      await contacts.recordTransaction('@newperson', 100, 'sent');

      const contact = contacts.getByAddress('@newperson');
      expect(contact).not.toBeNull();
      expect(contact?.transactionCount).toBe(1);
      expect(contact?.totalSent).toBe(100);
    });

    it('should update existing contact stats', async () => {
      await contacts.add({ address: '@frequent' });
      await contacts.recordTransaction('@frequent', 50, 'sent');
      await contacts.recordTransaction('@frequent', 25, 'received');

      const contact = contacts.getByAddress('@frequent');
      expect(contact?.transactionCount).toBe(2);
      expect(contact?.totalSent).toBe(50);
      expect(contact?.totalReceived).toBe(25);
    });

    it('should track recent recipients', async () => {
      await contacts.recordTransaction('@recent1', 10, 'sent');
      await contacts.recordTransaction('@recent2', 20, 'sent');

      expect(contacts.recentRecipients.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Import/Export', () => {
    it('should export contacts', async () => {
      await contacts.add({ address: '@alice', nickname: 'Alice', favorite: true });
      await contacts.add({ address: '@bob', tags: ['friend'] });

      const exported = contacts.export();

      expect(exported.version).toBe(1);
      expect(exported.contacts.length).toBe(2);
    });

    it('should import contacts', async () => {
      const data = {
        version: 1,
        exportedAt: Date.now(),
        contacts: [
          { address: '@imported1', nickname: 'Imported 1', favorite: true, tags: [] },
          { address: '@imported2', nickname: 'Imported 2', favorite: false, tags: ['test'] },
        ],
      };

      const count = await contacts.import(data);

      expect(count).toBe(2);
      expect(contacts.contacts.length).toBe(2);
    });
  });
});

describe('Contact Utilities', () => {
  it('should format contact name with priority', () => {
    // Custom nickname takes priority
    expect(formatContactName({ nickname: 'My Name', address: '@alice' } as Contact)).toBe('My Name');

    // Then identity nickname
    expect(formatContactName({ address: '@alice', identity: { agentId: '1', nickname: '@alice' } } as Contact)).toBe('@alice');

    // Fallback to address
    expect(formatContactName({ address: 'uhC0k12345678' } as Contact)).toBe('uhC0k1...5678');
  });

  it('should truncate long addresses', () => {
    expect(truncateAddress('@alice')).toBe('@alice'); // Keep nicknames
    expect(truncateAddress('short')).toBe('short');
    expect(truncateAddress('uhC0k1234567890abcdef')).toBe('uhC0k1...cdef');
  });

  it('should get initials', () => {
    expect(getContactInitials({ nickname: 'Alice', address: '@alice' } as Contact)).toBe('AL');
    expect(getContactInitials({ address: '@bob' } as Contact)).toBe('BO');
  });

  it('should generate consistent colors', () => {
    const color1 = getContactColor({ address: '@alice' } as Contact);
    const color2 = getContactColor({ address: '@alice' } as Contact);
    const color3 = getContactColor({ address: '@bob' } as Contact);

    expect(color1).toBe(color2);
    expect(color1).toMatch(/^#[A-F0-9]{6}$/i);
  });
});

// =============================================================================
// Offline Queue Tests
// =============================================================================

import {
  OfflineQueue,
  createOfflineQueue,
  type QueueItemId,
  type SendPayload,
} from '../src/wallet/offline-queue.js';

describe('OfflineQueue', () => {
  let queue: OfflineQueue;

  beforeEach(async () => {
    // Disable autoSync so items remain pending during assertions
    queue = createOfflineQueue(undefined, undefined, { autoSync: false });
    await queue.initialize();
  });

  afterEach(() => {
    queue.destroy();
  });

  describe('Enqueue', () => {
    it('should enqueue a send operation', async () => {
      const item = await queue.enqueueSend('@alice', 100, 'MYC', 'Coffee');

      expect(item.id).toBeDefined();
      expect(item.type).toBe('send');
      expect(item.status).toBe('pending');
      expect((item.payload as SendPayload).to).toBe('@alice');
      expect((item.payload as SendPayload).amount).toBe(100);
    });

    it('should track pending count', async () => {
      expect(queue.pendingCount).toBe(0);

      await queue.enqueueSend('@alice', 100, 'MYC');
      expect(queue.pendingCount).toBe(1);

      await queue.enqueueSend('@bob', 50, 'MYC');
      expect(queue.pendingCount).toBe(2);
    });

    it('should dedupe by idempotency key', async () => {
      const item1 = await queue.enqueue('send', { to: '@alice', amount: 100 }, {
        idempotencyKey: 'unique-key',
      });
      const item2 = await queue.enqueue('send', { to: '@alice', amount: 100 }, {
        idempotencyKey: 'unique-key',
      });

      expect(item1.id).toBe(item2.id);
      expect(queue.pendingCount).toBe(1);
    });
  });

  describe('Cancel', () => {
    it('should cancel pending operations', async () => {
      const item = await queue.enqueueSend('@alice', 100, 'MYC');
      expect(queue.pendingCount).toBe(1);

      const cancelled = await queue.cancel(item.id);

      expect(cancelled).toBe(true);
      expect(queue.get(item.id)?.status).toBe('cancelled');
      expect(queue.pendingCount).toBe(0);
    });

    it('should not cancel non-pending items', async () => {
      const cancelled = await queue.cancel('nonexistent' as QueueItemId);
      expect(cancelled).toBe(false);
    });
  });

  describe('Sync', () => {
    it('should process pending items on sync', async () => {
      // Set offline first to prevent auto-sync
      queue['_state$'].next({ ...queue.state, isOnline: false });

      await queue.enqueueSend('@alice', 100, 'MYC');
      await queue.enqueueSend('@bob', 50, 'MYC');

      // Now go online and sync
      queue['_state$'].next({ ...queue.state, isOnline: true });
      const result = await queue.sync();

      expect(result.processed).toBe(2);
      expect(result.failed).toBe(0);
      expect(result.remaining).toBe(0);
    });

    it('should emit sync events', async () => {
      // Register listener FIRST
      const events: string[] = [];
      queue.onSyncEvent((e) => events.push(e.type));

      // Set offline to prevent auto-sync
      queue['_state$'].next({ ...queue.state, isOnline: false });

      await queue.enqueueSend('@alice', 100, 'MYC');

      // Now go online and sync
      queue['_state$'].next({ ...queue.state, isOnline: true });
      await queue.sync();

      expect(events).toContain('sync_started');
      expect(events).toContain('item_synced');
      expect(events).toContain('sync_completed');
    });

    it('should respect offline status', async () => {
      // Simulate offline by manually setting state
      queue['_state$'].next({ ...queue.state, isOnline: false });

      await queue.enqueueSend('@alice', 100, 'MYC');
      const result = await queue.sync();

      expect(result.error).toBe('offline');
      expect(result.processed).toBe(0);
    });
  });

  describe('Reactive State', () => {
    it('should notify subscribers of state changes', async () => {
      const pendingCounts: number[] = [];
      queue.subscribe((state) => pendingCounts.push(state.pendingCount));

      await queue.enqueueSend('@alice', 100, 'MYC');
      await queue.sync();

      expect(pendingCounts).toContain(0);
      expect(pendingCounts).toContain(1);
    });
  });
});

// =============================================================================
// Biometric Tests
// =============================================================================

import {
  BiometricManager,
  MockBiometricProvider,
  detectPlatform,
  getBiometricIcon,
  getBiometricName,
} from '../src/wallet/biometrics.js';

describe('BiometricManager', () => {
  describe('Mock Provider', () => {
    it('should report availability', async () => {
      const manager = await BiometricManager.createMock({
        availableTypes: ['fingerprint', 'face'],
      });

      expect(manager.isAvailable).toBe(true);
      expect(manager.availableTypes).toContain('fingerprint');
      expect(manager.availableTypes).toContain('face');
    });

    it('should authenticate successfully', async () => {
      const manager = await BiometricManager.createMock({
        shouldSucceed: true,
        delay: 100,
      });

      const result = await manager.authenticate('Verify to continue');

      expect(result.success).toBe(true);
      expect(result.type).toBe('fingerprint');
    });

    it('should fail authentication when configured', async () => {
      const manager = await BiometricManager.createMock({
        shouldSucceed: false,
        delay: 100,
      });

      const result = await manager.authenticate();

      expect(result.success).toBe(false);
      expect(result.error?.code).toBe('failed');
    });

    it('should not allow concurrent authentications', async () => {
      const manager = await BiometricManager.createMock({
        shouldSucceed: true,
        delay: 500,
      });

      // Start first auth (don't await)
      const firstAuth = manager.authenticate();

      // Try second auth immediately
      const secondResult = await manager.authenticate();

      expect(secondResult.success).toBe(false);
      expect(secondResult.error?.code).toBe('hardware_unavailable');

      // First auth should still succeed
      const firstResult = await firstAuth;
      expect(firstResult.success).toBe(true);
    });
  });

  describe('Primary Type', () => {
    it('should prefer face over fingerprint', async () => {
      const manager = await BiometricManager.createMock({
        availableTypes: ['fingerprint', 'face'],
      });

      expect(manager.primaryType).toBe('face');
    });

    it('should fallback to fingerprint', async () => {
      const manager = await BiometricManager.createMock({
        availableTypes: ['fingerprint'],
      });

      expect(manager.primaryType).toBe('fingerprint');
    });
  });

  describe('Description', () => {
    it('should describe available types', async () => {
      const manager = await BiometricManager.createMock({
        availableTypes: ['fingerprint'],
      });

      expect(manager.getDescription()).toBe('Fingerprint');
    });

    it('should handle no biometrics', async () => {
      const manager = await BiometricManager.createMock({
        availableTypes: [],
      });

      expect(manager.getDescription()).toBe('No biometrics available');
    });
  });
});

describe('Biometric Utilities', () => {
  it('should return correct icons', () => {
    expect(getBiometricIcon('face')).toBe('face-id');
    expect(getBiometricIcon('fingerprint')).toBe('fingerprint');
    expect(getBiometricIcon('webauthn')).toBe('key');
  });

  it('should return correct names', () => {
    expect(getBiometricName('face')).toBe('Face ID');
    expect(getBiometricName('fingerprint')).toBe('Fingerprint');
    expect(getBiometricName('webauthn')).toBe('Security Key');
  });
});

// =============================================================================
// Animation Tests
// =============================================================================

import {
  AnimatedValue,
  AnimatedCounter,
  createAnimatedBalance,
  formatCurrency as formatAnimCurrency,
  formatChange,
  formatPercentage,
  getChangeColor,
  SPRING_PRESETS,
  EASING,
  tween,
  createSuccessPulse,
  isPulseComplete,
} from '../src/wallet/animations.js';

describe('AnimatedValue', () => {
  it('should initialize with value', () => {
    const animated = new AnimatedValue(100);
    expect(animated.value).toBe(100);
    expect(animated.isAnimating).toBe(false);
  });

  it('should set value immediately', () => {
    const animated = new AnimatedValue(0);
    animated.setValue(50);

    expect(animated.value).toBe(50);
    expect(animated.isAnimating).toBe(false);
  });

  it('should animate with spring physics', async () => {
    const animated = new AnimatedValue(0, SPRING_PRESETS.stiff);
    const values: number[] = [];

    animated.subscribe((state) => values.push(state.value));
    animated.springTo(100);

    // Wait for animation (stiff springs need ~1s to fully settle)
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Should have intermediate values
    expect(values.length).toBeGreaterThan(2);
    // Spring physics may not perfectly reach target — allow ±2
    expect(Math.abs(values[values.length - 1] - 100)).toBeLessThan(2);
  });

  it('should stop animation on demand', () => {
    const animated = new AnimatedValue(0);
    animated.springTo(100);

    expect(animated.isAnimating).toBe(true);

    animated.stop();

    expect(animated.isAnimating).toBe(false);
  });

  it('should finish animation instantly', () => {
    const animated = new AnimatedValue(0);
    animated.springTo(100);
    animated.finish();

    expect(animated.value).toBe(100);
    expect(animated.isAnimating).toBe(false);
  });

  it('should add to current target', () => {
    const animated = new AnimatedValue(50);
    animated.add(25);

    expect(animated.target).toBe(75);
  });
});

describe('AnimatedCounter', () => {
  it('should format values', () => {
    const counter = new AnimatedCounter(1000, (v) => `$${v.toLocaleString()}`);
    expect(counter.formattedValue).toBe('$1,000');
  });

  it('should update formatted value on count', async () => {
    const counter = new AnimatedCounter(0);
    const formatted: string[] = [];

    counter.subscribe((f) => formatted.push(f));
    counter.countTo(100);

    // Wait a bit for some animation frames, then finish
    await new Promise((resolve) => setTimeout(resolve, 100));
    counter.finish(); // Force complete to exact target

    expect(formatted.length).toBeGreaterThan(1);
    expect(counter.value).toBe(100);
  });
});

describe('Currency Formatting', () => {
  it('should format with currency code', () => {
    expect(formatAnimCurrency(1234.56, { currency: 'MYC' })).toBe('M 1,234.56 MYC');
  });

  it('should format compact large numbers', () => {
    expect(formatAnimCurrency(1500000, { currency: 'MYC', compact: true })).toContain('M');
  });

  it('should format change with sign', () => {
    expect(formatChange(100, { currency: 'MYC' })).toContain('+');
    expect(formatChange(-50, { currency: 'MYC' })).toContain('-');
  });

  it('should format percentage', () => {
    expect(formatPercentage(0.156)).toBe('15.6%');
    expect(formatPercentage(0.5, 0)).toBe('50%');
  });
});

describe('Visual Feedback', () => {
  it('should create success pulse', () => {
    const pulse = createSuccessPulse();

    expect(pulse.color).toBe('#22c55e');
    expect(pulse.progress).toBe(0);
  });

  it('should detect pulse completion', async () => {
    const pulse = createSuccessPulse(100);

    expect(isPulseComplete(pulse)).toBe(false);

    await new Promise((resolve) => setTimeout(resolve, 150));

    expect(isPulseComplete(pulse)).toBe(true);
  });

  it('should return correct change colors', () => {
    expect(getChangeColor(100)).toBe('#22c55e'); // Green for positive
    expect(getChangeColor(-50)).toBe('#ef4444'); // Red for negative
    expect(getChangeColor(0)).toBe('#6b7280'); // Gray for zero
  });
});

describe('Easing Functions', () => {
  it('should have correct boundary values', () => {
    for (const [name, fn] of Object.entries(EASING)) {
      expect(fn(0)).toBeCloseTo(0, 5);
      expect(fn(1)).toBeCloseTo(1, 5);
    }
  });

  it('should ease correctly at midpoint', () => {
    expect(EASING.linear(0.5)).toBe(0.5);
    expect(EASING.easeIn(0.5)).toBeLessThan(0.5);
    expect(EASING.easeOut(0.5)).toBeGreaterThan(0.5);
  });
});

describe('Tween', () => {
  it('should interpolate values', () => {
    const t = tween(0, 100, 200, EASING.linear);

    // Before start, should return the 'from' value
    expect(t.getValue()).toBe(0);
    expect(t.getProgress()).toBe(0);

    t.start();

    // Right after start, progress should be small but positive
    expect(t.getValue()).toBeGreaterThanOrEqual(0);
    expect(t.getProgress()).toBeGreaterThanOrEqual(0);
  });

  it('should complete after duration', async () => {
    const t = tween(0, 100, 100, EASING.linear);
    t.start();

    await new Promise((resolve) => setTimeout(resolve, 150));

    expect(t.isComplete()).toBe(true);
    expect(t.getValue()).toBe(100);
  });
});

describe('Animated Balance Helper', () => {
  it('should create animated balance', () => {
    const balance = createAnimatedBalance(1000, 'MYC');

    expect(balance.formatted$.value).toContain('1,000');

    balance.destroy();
  });

  it('should update on add/subtract', () => {
    const balance = createAnimatedBalance(100, 'MYC');
    const values: string[] = [];

    balance.formatted$.subscribe((f) => values.push(f));

    balance.add(50);

    // Target should be 150
    expect(values.length).toBeGreaterThan(0);

    balance.destroy();
  });
});

// =============================================================================
// Template Tests
// =============================================================================

import {
  TemplateManager,
  createTemplateManager,
  getCategoryEmoji,
  getCategoryName,
  formatSchedule,
  type TemplateId,
} from '../src/wallet/templates.js';

describe('TemplateManager', () => {
  let templates: TemplateManager;

  beforeEach(async () => {
    templates = createTemplateManager();
    await templates.initialize();
  });

  afterEach(() => {
    templates.destroy();
  });

  describe('Create Template', () => {
    it('should create a template', async () => {
      const template = await templates.create({
        name: 'Morning Coffee',
        emoji: '\u2615',
        to: '@local-cafe',
        amount: 5.5,
        currency: 'MYC',
      });

      expect(template.id).toBeDefined();
      expect(template.name).toBe('Morning Coffee');
      expect(template.amount).toBe(5.5);
      expect(template.usageCount).toBe(0);
    });

    it('should create with category', async () => {
      const template = await templates.create({
        name: 'Lunch',
        to: '@restaurant',
        amount: 15,
        category: 'food',
      });

      expect(template.category).toBe('food');
    });
  });

  describe('Update Template', () => {
    it('should update template fields', async () => {
      const template = await templates.create({
        name: 'Original',
        to: '@recipient',
        amount: 100,
      });

      const updated = await templates.update(template.id, {
        name: 'Updated Name',
        amount: 150,
      });

      expect(updated.name).toBe('Updated Name');
      expect(updated.amount).toBe(150);
      expect(updated.to).toBe('@recipient'); // Unchanged
    });

    it('should toggle active status', async () => {
      const template = await templates.create({
        name: 'Test',
        to: '@test',
        amount: 10,
      });

      expect(template.isActive).toBe(true);

      const deactivated = await templates.update(template.id, { isActive: false });
      expect(deactivated.isActive).toBe(false);
    });
  });

  describe('Delete Template', () => {
    it('should remove template', async () => {
      const template = await templates.create({
        name: 'ToDelete',
        to: '@delete',
        amount: 1,
      });

      await templates.delete(template.id);

      expect(templates.get(template.id)).toBeNull();
    });
  });

  describe('Search', () => {
    beforeEach(async () => {
      await templates.create({ name: 'Morning Coffee', to: '@cafe', amount: 5, tags: ['daily'] });
      await templates.create({ name: 'Weekly Rent', to: '@landlord', amount: 500 });
      await templates.create({ name: 'Gym Membership', to: '@gym', amount: 50, category: 'bills' });
    });

    it('should search by name', () => {
      const results = templates.search('coffee');
      expect(results.length).toBe(1);
      expect(results[0].name).toBe('Morning Coffee');
    });

    it('should search by recipient', () => {
      const results = templates.search('@landlord');
      expect(results.length).toBe(1);
    });

    it('should search by tag', () => {
      const results = templates.search('daily');
      expect(results.length).toBe(1);
    });
  });

  describe('Execution', () => {
    it('should execute template', async () => {
      const template = await templates.create({
        name: 'Test Payment',
        to: '@recipient',
        amount: 100,
        currency: 'MYC',
        memo: 'Test memo',
      });

      const mockWallet = {
        send: vi.fn().mockResolvedValue({ id: 'tx-123' }),
      };

      const result = await templates.execute(template.id, mockWallet);

      expect(result.success).toBe(true);
      expect(result.transactionId).toBe('tx-123');
      expect(mockWallet.send).toHaveBeenCalledWith('@recipient', 100, 'MYC', { memo: 'Test memo' });
    });

    it('should increment usage count', async () => {
      const template = await templates.create({
        name: 'Usage Test',
        to: '@test',
        amount: 10,
      });

      const mockWallet = { send: vi.fn().mockResolvedValue({ id: 'tx-1' }) };

      await templates.execute(template.id, mockWallet);
      await templates.execute(template.id, mockWallet);

      const updated = templates.get(template.id);
      expect(updated?.usageCount).toBe(2);
      expect(updated?.lastUsedAt).toBeDefined();
    });

    it('should handle execution failure', async () => {
      const template = await templates.create({
        name: 'Fail Test',
        to: '@fail',
        amount: 10,
      });

      const mockWallet = {
        send: vi.fn().mockRejectedValue(new Error('Insufficient funds')),
      };

      const result = await templates.execute(template.id, mockWallet);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Insufficient funds');
    });

    it('should not execute inactive template', async () => {
      const template = await templates.create({
        name: 'Inactive',
        to: '@inactive',
        amount: 10,
      });

      await templates.update(template.id, { isActive: false });

      const mockWallet = { send: vi.fn() };
      const result = await templates.execute(template.id, mockWallet);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Template is inactive');
      expect(mockWallet.send).not.toHaveBeenCalled();
    });
  });

  describe('Preview', () => {
    it('should preview template', async () => {
      const template = await templates.create({
        name: 'Preview Test',
        to: '@preview',
        amount: 99,
        currency: 'MYC',
        memo: 'Preview memo',
      });

      const preview = templates.preview(template.id);

      expect(preview).toEqual({
        to: '@preview',
        amount: 99,
        currency: 'MYC',
        memo: 'Preview memo',
      });
    });

    it('should return null for missing template', () => {
      expect(templates.preview('missing' as TemplateId)).toBeNull();
    });
  });

  describe('Import/Export', () => {
    it('should export templates', async () => {
      await templates.create({ name: 'Export 1', to: '@e1', amount: 10 });
      await templates.create({ name: 'Export 2', to: '@e2', amount: 20 });

      const exported = templates.export();

      expect(exported.version).toBe(1);
      expect(exported.templates.length).toBe(2);
    });

    it('should import templates', async () => {
      const data = {
        version: 1,
        exportedAt: Date.now(),
        templates: [
          { name: 'Imported 1', to: '@i1', amount: 100, tags: [] },
          { name: 'Imported 2', to: '@i2', amount: 200, category: 'food' as const, tags: [] },
        ],
      };

      const count = await templates.import(data);

      expect(count).toBe(2);
      expect(templates.templates.length).toBe(2);
    });
  });
});

describe('Template Utilities', () => {
  it('should return category emojis', () => {
    expect(getCategoryEmoji('food')).toBe('\uD83C\uDF55');
    expect(getCategoryEmoji('transport')).toBe('\uD83D\uDE97');
    expect(getCategoryEmoji('bills')).toBe('\uD83D\uDCDD');
  });

  it('should return category names', () => {
    expect(getCategoryName('food')).toBe('Food & Drinks');
    expect(getCategoryName('bills')).toBe('Bills & Utilities');
    expect(getCategoryName('savings')).toBe('Savings');
  });

  it('should format schedules', () => {
    expect(formatSchedule({ type: 'daily', time: '09:00', enabled: true })).toBe('Daily at 09:00');
    expect(formatSchedule({ type: 'weekly', dayOfWeek: 1, time: '10:00', enabled: true })).toBe('Every Monday at 10:00');
    expect(formatSchedule({ type: 'monthly', dayOfMonth: 15, time: '08:00', enabled: true })).toBe('Monthly on day 15 at 08:00');
    expect(formatSchedule({ type: 'daily', enabled: false })).toBe('Paused');
  });
});

// =============================================================================
// React Utility Functions Tests
// =============================================================================

describe('React Utility Functions', () => {
  describe('applyOptimisticBalanceChange', () => {
    it('should apply positive balance change', () => {
      const balances = new Map<string, { currency: string; available: number; pending: number; total: number }>();
      balances.set('MYC', { currency: 'MYC', available: 100, pending: 50, total: 150 });

      const result = applyOptimisticBalanceChange(balances, 'MYC', 30);

      expect(result.get('MYC')?.available).toBe(130);
      expect(result.get('MYC')?.pending).toBe(20);
    });

    it('should apply negative balance change', () => {
      const balances = new Map<string, { currency: string; available: number; pending: number; total: number }>();
      balances.set('MYC', { currency: 'MYC', available: 100, pending: 0, total: 100 });

      const result = applyOptimisticBalanceChange(balances, 'MYC', -30);

      expect(result.get('MYC')?.available).toBe(70);
      expect(result.get('MYC')?.pending).toBe(30);
    });

    it('should create new balance if currency not found', () => {
      const balances = new Map();

      const result = applyOptimisticBalanceChange(balances, 'USD', 100);

      expect(result.get('USD')?.available).toBe(100);
      expect(result.get('USD')?.pending).toBe(-100);
    });

    it('should not mutate original map', () => {
      const balances = new Map<string, { currency: string; available: number; pending: number; total: number }>();
      balances.set('MYC', { currency: 'MYC', available: 100, pending: 0, total: 100 });

      const result = applyOptimisticBalanceChange(balances, 'MYC', 50);

      expect(balances.get('MYC')?.available).toBe(100);
      expect(result).not.toBe(balances);
    });
  });

  describe('getContextAction', () => {
    it('should return scan action for home context', () => {
      const onScan = vi.fn();
      const result = getContextAction('home', { onScan });

      expect(result?.label).toBe('Scan');
      expect(result?.icon).toBe('qr-code');

      result?.action();
      expect(onScan).toHaveBeenCalled();
    });

    it('should return share action for receive context', () => {
      const onShare = vi.fn();
      const result = getContextAction('receive', { onShare });

      expect(result?.label).toBe('Share');
      expect(result?.icon).toBe('share');
    });

    it('should return scan action for send context', () => {
      const onScan = vi.fn();
      const result = getContextAction('send', { onScan });

      expect(result?.label).toBe('Scan');
      expect(result?.icon).toBe('qr-code');
    });

    it('should return edit action for settings context', () => {
      const onEdit = vi.fn();
      const result = getContextAction('settings', { onEdit });

      expect(result?.label).toBe('Edit');
      expect(result?.icon).toBe('pencil');
    });

    it('should return edit action for profile context', () => {
      const onEdit = vi.fn();
      const result = getContextAction('profile', { onEdit });

      expect(result?.label).toBe('Edit');
      expect(result?.icon).toBe('pencil');
    });

    it('should return null if handler not provided', () => {
      const result = getContextAction('home', {});

      expect(result).toBeNull();
    });
  });

  describe('formatReputationScore', () => {
    it('should format score as percentage', () => {
      expect(formatReputationScore(0.95)).toBe('95%');
      expect(formatReputationScore(0.5)).toBe('50%');
      expect(formatReputationScore(0)).toBe('0%');
      expect(formatReputationScore(1)).toBe('100%');
    });

    it('should round decimal percentages', () => {
      expect(formatReputationScore(0.333)).toBe('33%');
      expect(formatReputationScore(0.666)).toBe('67%');
    });
  });

  describe('getReputationBadge', () => {
    it('should return excellent badge for 90%+', () => {
      const badge = getReputationBadge(0.95);
      expect(badge.label).toBe('Excellent');
      expect(badge.color).toBe('green');
      expect(badge.icon).toBe('star');
    });

    it('should return good badge for 70-89%', () => {
      const badge = getReputationBadge(0.75);
      expect(badge.label).toBe('Good');
      expect(badge.color).toBe('green');
      expect(badge.icon).toBe('check-circle');
    });

    it('should return fair badge for 50-69%', () => {
      const badge = getReputationBadge(0.55);
      expect(badge.label).toBe('Fair');
      expect(badge.color).toBe('yellow');
      expect(badge.icon).toBe('minus-circle');
    });

    it('should return low badge for 30-49%', () => {
      const badge = getReputationBadge(0.35);
      expect(badge.label).toBe('Low');
      expect(badge.color).toBe('orange');
      expect(badge.icon).toBe('alert-circle');
    });

    it('should return poor badge for 1-29%', () => {
      const badge = getReputationBadge(0.15);
      expect(badge.label).toBe('Poor');
      expect(badge.color).toBe('red');
      expect(badge.icon).toBe('x-circle');
    });

    it('should return unknown badge for 0', () => {
      const badge = getReputationBadge(0);
      expect(badge.label).toBe('Unknown');
      expect(badge.color).toBe('gray');
      expect(badge.icon).toBe('help-circle');
    });
  });

  describe('formatRiskLevel', () => {
    it('should format low risk', () => {
      const result = formatRiskLevel('low');
      expect(result.label).toBe('Low Risk');
      expect(result.color).toBe('green');
      expect(result.description).toBe('This transaction appears safe.');
    });

    it('should format medium risk', () => {
      const result = formatRiskLevel('medium');
      expect(result.label).toBe('Medium Risk');
      expect(result.color).toBe('yellow');
      expect(result.description).toBe('Proceed with normal caution.');
    });

    it('should format high risk', () => {
      const result = formatRiskLevel('high');
      expect(result.label).toBe('High Risk');
      expect(result.color).toBe('orange');
      expect(result.description).toBe('Consider verifying identity first.');
    });

    it('should format very high risk', () => {
      const result = formatRiskLevel('very_high');
      expect(result.label).toBe('Very High Risk');
      expect(result.color).toBe('red');
      expect(result.description).toBe('Verification strongly recommended.');
    });
  });
});

// =============================================================================
// Biometric Tests
// =============================================================================

describe('Biometric Utilities', () => {
  describe('getBiometricIcon', () => {
    it('should return correct icon for face', () => {
      expect(getBiometricIcon('face')).toBe('face-id');
    });

    it('should return correct icon for fingerprint', () => {
      expect(getBiometricIcon('fingerprint')).toBe('fingerprint');
    });

    it('should return correct icon for iris', () => {
      expect(getBiometricIcon('iris')).toBe('eye');
    });

    it('should return correct icon for voice', () => {
      expect(getBiometricIcon('voice')).toBe('microphone');
    });

    it('should return correct icon for webauthn', () => {
      expect(getBiometricIcon('webauthn')).toBe('key');
    });

    it('should return correct icon for pin', () => {
      expect(getBiometricIcon('pin')).toBe('keypad');
    });

    it('should return correct icon for password', () => {
      expect(getBiometricIcon('password')).toBe('lock');
    });

    it('should return shield for unknown type', () => {
      expect(getBiometricIcon('unknown')).toBe('shield');
    });
  });

  describe('getBiometricName', () => {
    it('should return correct name for face', () => {
      expect(getBiometricName('face')).toBe('Face ID');
    });

    it('should return correct name for fingerprint', () => {
      expect(getBiometricName('fingerprint')).toBe('Fingerprint');
    });

    it('should return correct name for iris', () => {
      expect(getBiometricName('iris')).toBe('Iris Scan');
    });

    it('should return correct name for voice', () => {
      expect(getBiometricName('voice')).toBe('Voice Recognition');
    });

    it('should return correct name for webauthn', () => {
      expect(getBiometricName('webauthn')).toBe('Security Key');
    });

    it('should return correct name for pin', () => {
      expect(getBiometricName('pin')).toBe('PIN');
    });

    it('should return correct name for password', () => {
      expect(getBiometricName('password')).toBe('Password');
    });

    it('should return Biometric for unknown type', () => {
      expect(getBiometricName('unknown')).toBe('Biometric');
    });
  });

  describe('MockBiometricProvider', () => {
    it('should return available capability', async () => {
      const provider = new MockBiometricProvider();
      const capability = await provider.getCapability();

      expect(capability.isAvailable).toBe(true);
      expect(capability.types).toContain('fingerprint');
      expect(capability.hasSecureHardware).toBe(true);
      expect(capability.isEnrolled).toBe(true);
    });

    it('should return unavailable when no types configured', async () => {
      const provider = new MockBiometricProvider({ availableTypes: [] });
      const capability = await provider.getCapability();

      expect(capability.isAvailable).toBe(false);
      expect(capability.types).toEqual([]);
    });

    it('should authenticate successfully by default', async () => {
      const provider = new MockBiometricProvider({ delay: 10 });
      const result = await provider.authenticate();

      expect(result.success).toBe(true);
      expect(result.type).toBe('fingerprint');
    });

    it('should fail authentication when configured to fail', async () => {
      const provider = new MockBiometricProvider({ shouldSucceed: false, delay: 10 });
      const result = await provider.authenticate();

      expect(result.success).toBe(false);
      expect(result.error?.code).toBe('failed');
      expect(result.error?.message).toBe('Mock authentication failed');
    });

    it('should allow reconfiguration', async () => {
      const provider = new MockBiometricProvider({ delay: 10 });

      // Initially succeeds
      let result = await provider.authenticate();
      expect(result.success).toBe(true);

      // Reconfigure to fail
      provider.configure({ shouldSucceed: false });
      result = await provider.authenticate();
      expect(result.success).toBe(false);

      // Reconfigure to succeed again
      provider.configure({ shouldSucceed: true });
      result = await provider.authenticate();
      expect(result.success).toBe(true);
    });

    it('should use custom biometric types', async () => {
      const provider = new MockBiometricProvider({
        availableTypes: ['face', 'iris'],
        delay: 10,
      });

      const capability = await provider.getCapability();
      expect(capability.types).toEqual(['face', 'iris']);

      const result = await provider.authenticate();
      expect(result.type).toBe('face');
    });

    it('should handle cancel call gracefully', () => {
      const provider = new MockBiometricProvider();
      expect(() => provider.cancel()).not.toThrow();
    });
  });

  describe('BiometricManager', () => {
    it('should create with mock provider', async () => {
      const mockProvider = new MockBiometricProvider({ delay: 10 });
      const manager = await BiometricManager.createWithProvider(mockProvider);

      expect(manager).toBeDefined();
      expect(manager.isAvailable).toBe(true); // MockBiometricProvider is available by default
    });

    it('should check capability and update state', async () => {
      const mockProvider = new MockBiometricProvider({
        availableTypes: ['fingerprint', 'face'],
        delay: 10,
      });
      const manager = await BiometricManager.createWithProvider(mockProvider);

      await manager.refreshCapability();

      expect(manager.isAvailable).toBe(true);
      expect(manager.availableTypes).toContain('fingerprint');
      expect(manager.availableTypes).toContain('face');
    });

    it('should authenticate successfully', async () => {
      const mockProvider = new MockBiometricProvider({ delay: 10 });
      const manager = await BiometricManager.createWithProvider(mockProvider);

      const result = await manager.authenticate('Test prompt');

      expect(result.success).toBe(true);
    });

    it('should track authenticating state', async () => {
      const mockProvider = new MockBiometricProvider({ delay: 100 });
      const manager = await BiometricManager.createWithProvider(mockProvider);

      expect(manager.isAuthenticating).toBe(false);

      const authPromise = manager.authenticate('Test');

      // Give it a moment to start
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(manager.isAuthenticating).toBe(true);

      await authPromise;
      expect(manager.isAuthenticating).toBe(false);
    });

    it('should emit state changes', async () => {
      const mockProvider = new MockBiometricProvider({ delay: 10 });
      const manager = await BiometricManager.createWithProvider(mockProvider);

      const states: boolean[] = [];
      const sub = manager.state$.subscribe((state) => {
        states.push(state.capability.isAvailable);
      });

      await manager.refreshCapability();

      sub.unsubscribe();

      // Should have recorded state change
      expect(states.length).toBeGreaterThan(0);
    });
  });
});
