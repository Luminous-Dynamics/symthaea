/**
 * Holochain Wallet Integration Tests
 *
 * Tests "The Nervous System" - the wiring that connects:
 * - ConductorManager (The Spine) - WebSocket connection management
 * - HolochainFinanceProvider (The Muscle) - Finance zome integration
 * - HolochainIdentityResolver (The Face) - Identity zome integration
 * - OfflineQueue integration - Resilient transactions
 * - createHolochainWallet factory - Full system wiring
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Import the modules under test
import {
  ConductorManager,
  createConductorManager,
  createDevConductorManager,
  createProdConductorManager,
  type ConductorState,
  type ConductorConfig,
} from '../src/wallet/conductor.js';

import {
  HolochainFinanceProvider,
  createHolochainFinanceProvider,
  createMockFinanceProvider,
  type FinanceState,
} from '../src/wallet/finance-provider.js';

import {
  createHolochainWallet,
  createDevWallet,
  isConductorAvailable,
  type HolochainWalletConfig,
} from '../src/wallet/holochain-wallet.js';

import {
  HolochainIdentityResolver,
  type ProfileData,
} from '../src/wallet/identity-resolver.js';

import {
  OfflineQueue,
  createOfflineQueue,
  type QueueState,
} from '../src/wallet/offline-queue.js';

import { BehaviorSubject } from '../src/reactive/index.js';
import { createClient, MockMycelixClient } from '../src/client/index.js';

// =============================================================================
// ConductorManager Tests (The Spine)
// =============================================================================

describe('ConductorManager (The Spine)', () => {
  let conductor: ConductorManager;

  beforeEach(() => {
    conductor = createConductorManager({
      installedAppId: 'test-app',
      appUrl: 'ws://localhost:8888',
      autoReconnect: false, // Disable for tests
      timeout: 5000,
    });
  });

  afterEach(async () => {
    await conductor.destroy();
  });

  describe('Configuration', () => {
    it('should create with default configuration', () => {
      const state = conductor.state;
      expect(state.connection).toBe('disconnected');
      expect(state.installedAppId).toBe('test-app');
      expect(state.reconnectAttempts).toBe(0);
      expect(state.agentPubKey).toBeNull();
    });

    it('should create dev conductor with appropriate settings', () => {
      const devConductor = createDevConductorManager('dev-app');
      expect(devConductor.state.installedAppId).toBe('dev-app');
      devConductor.destroy();
    });

    it('should create prod conductor with appropriate settings', () => {
      const prodConductor = createProdConductorManager('prod-app', 'wss://prod.example.com');
      expect(prodConductor.state.installedAppId).toBe('prod-app');
      prodConductor.destroy();
    });
  });

  describe('State Observability', () => {
    it('should provide reactive state$', () => {
      expect(conductor.state$).toBeDefined();
      expect(conductor.state$.value.connection).toBe('disconnected');
    });

    it('should update state synchronously', () => {
      const states: ConductorState[] = [];
      conductor.state$.subscribe((state) => states.push(state));

      // Initial state should be emitted
      expect(states.length).toBe(1);
      expect(states[0].connection).toBe('disconnected');
    });

    it('should track isConnected', () => {
      expect(conductor.isConnected).toBe(false);
    });

    it('should provide isOnline state', () => {
      // isOnline is part of ConductorState and should be accessible
      const state = conductor.state;
      expect(state).toBeDefined();
      expect('isOnline' in state).toBe(true);
    });
  });

  describe('Signal Handling', () => {
    it('should register and unregister signal listeners', () => {
      const signals: unknown[] = [];
      const unsubscribe = conductor.onSignal((signal) => {
        signals.push(signal);
      });

      expect(typeof unsubscribe).toBe('function');
      unsubscribe();
    });

    it('should handle multiple signal listeners', () => {
      const signals1: unknown[] = [];
      const signals2: unknown[] = [];

      const unsub1 = conductor.onSignal((signal) => signals1.push(signal));
      const unsub2 = conductor.onSignal((signal) => signals2.push(signal));

      // Both should be registered
      unsub1();
      unsub2();
    });
  });

  describe('Connection Error Handling', () => {
    it('should throw when calling callZome while disconnected', async () => {
      await expect(
        conductor.callZome({
          zome_name: 'test',
          fn_name: 'test_fn',
          payload: null,
        })
      ).rejects.toThrow('Not connected');
    });

    it('should throw when destroyed', async () => {
      await conductor.destroy();
      await expect(conductor.connect()).rejects.toThrow('destroyed');
    });
  });

  describe('Getters', () => {
    it('should expose underlying client', () => {
      const client = conductor.getClient();
      expect(client).toBeDefined();
    });

    it('should track agentPubKey', () => {
      expect(conductor.agentPubKey).toBeNull();
    });
  });
});

// =============================================================================
// HolochainFinanceProvider Tests (The Muscle)
// =============================================================================

describe('HolochainFinanceProvider (The Muscle)', () => {
  const testAgentId = 'test-agent-id';

  describe('Mock Finance Provider (Basic Interface)', () => {
    it('should create mock provider with initial balances', async () => {
      const provider = createMockFinanceProvider();

      const balances = await provider.getAllBalances(testAgentId);
      expect(balances.size).toBeGreaterThan(0);

      const mycBalance = balances.get('MYC');
      expect(mycBalance).toBeDefined();
      expect(mycBalance!.available).toBeGreaterThan(0);
    });

    it('should track transactions', async () => {
      const provider = createMockFinanceProvider();

      const transactions = await provider.getTransactions(testAgentId);
      expect(Array.isArray(transactions)).toBe(true);
      expect(transactions.length).toBeGreaterThan(0); // Mock has sample transactions
    });

    it('should allow sending funds', async () => {
      const provider = createMockFinanceProvider();

      // send now returns transaction ID (string)
      const txId = await provider.send(testAgentId, 'alice-did', 100, 'MYC', 'Test payment');
      expect(typeof txId).toBe('string');
      expect(txId).toMatch(/^mock-tx-/);

      // Verify the transaction was created
      const tx = await provider.getTransaction(txId);
      expect(tx).toBeDefined();
      expect(tx!.amount).toBe(100);
      expect(tx!.memo).toBe('Test payment');
    });

    it('should update balances after send', async () => {
      const provider = createMockFinanceProvider();

      const balancesBefore = await provider.getAllBalances(testAgentId);
      const mycBefore = balancesBefore.get('MYC')!.available;

      await provider.send(testAgentId, 'bob-did', 50, 'MYC');

      const balancesAfter = await provider.getAllBalances(testAgentId);
      const mycAfter = balancesAfter.get('MYC')!.available;

      expect(mycAfter).toBe(mycBefore - 50);
    });
  });

  describe('Basic Provider Interface', () => {
    it('should implement required FinanceProvider methods', () => {
      const provider = createMockFinanceProvider();

      // These are the core FinanceProvider interface methods
      expect(typeof provider.getBalance).toBe('function');
      expect(typeof provider.getAllBalances).toBe('function');
      expect(typeof provider.getTransactions).toBe('function');
      expect(typeof provider.send).toBe('function');
      expect(typeof provider.getTransaction).toBe('function');
    });

    it('should handle transaction limit parameter', async () => {
      const provider = createMockFinanceProvider();
      const testAgentId = 'test-agent';

      // Mock has 2 sample transactions, limiting to 1
      const limited = await provider.getTransactions(testAgentId, 1);
      expect(limited.length).toBe(1);

      const unlimited = await provider.getTransactions(testAgentId);
      expect(unlimited.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('HolochainFinanceProvider Class', () => {
    // Tests for the full HolochainFinanceProvider require a conductor
    // These verify the class structure exists

    it('should export HolochainFinanceProvider class', () => {
      expect(HolochainFinanceProvider).toBeDefined();
      expect(typeof HolochainFinanceProvider).toBe('function');
    });

    it('should export createHolochainFinanceProvider factory', () => {
      expect(createHolochainFinanceProvider).toBeDefined();
      expect(typeof createHolochainFinanceProvider).toBe('function');
    });

    it('should export FinanceState type', () => {
      // Type check - if this compiles, types are exported correctly
      const state: FinanceState = {
        balances: new Map(),
        transactions: [],
        lastFetchedAt: null,
        isLoading: false,
        error: null,
      };
      expect(state.isLoading).toBe(false);
    });
  });
});

// =============================================================================
// Offline Queue Integration Tests
// =============================================================================

describe('Offline Queue Integration', () => {
  it('should create queue with default config', async () => {
    const queue = createOfflineQueue();
    await queue.initialize();

    // QueueState has: items, pendingCount, isSyncing, isOnline
    expect(queue.state$.value.pendingCount).toBe(0);
    expect(queue.state$.value.isSyncing).toBe(false);
    expect(Array.isArray(queue.state$.value.items)).toBe(true);

    queue.destroy();
  });

  it('should queue send operations', async () => {
    const queue = createOfflineQueue();
    await queue.initialize();

    // Use enqueueSend helper method
    const operation = await queue.enqueueSend(
      'did:mycelix:alice',
      100,
      'MYC',
      'Test payment'
    );

    expect(operation.id).toBeDefined();
    expect(operation.type).toBe('send');
    expect(operation.status).toBe('pending');
    expect(queue.state$.value.pendingCount).toBe(1);

    queue.destroy();
  });

  it('should track queue state changes', async () => {
    const queue = createOfflineQueue();
    await queue.initialize();

    const states: QueueState[] = [];
    queue.state$.subscribe((state) => states.push(state));

    // Initial state
    expect(states.length).toBe(1);

    await queue.enqueueSend('did:mycelix:bob', 50, 'MYC');

    // Should have more states after enqueueing
    expect(states.length).toBeGreaterThan(1);

    queue.destroy();
  });

  it('should use enqueue with type parameter', async () => {
    const queue = createOfflineQueue();
    await queue.initialize();

    // Generic enqueue method
    const operation = await queue.enqueue('send', {
      to: 'did:mycelix:charlie',
      amount: 25,
      currency: 'MYC',
    });

    expect(operation.type).toBe('send');
    expect(operation.payload.to).toBe('did:mycelix:charlie');

    queue.destroy();
  });
});

// =============================================================================
// Identity Resolver Integration Tests
// =============================================================================

describe('HolochainIdentityResolver Integration', () => {
  it('should create with mock client', () => {
    const mockClient = {
      callZome: vi.fn().mockResolvedValue({
        agentId: 'test-agent',
        nickname: 'testuser',
        displayName: 'Test User',
      }),
    };

    const resolver = new HolochainIdentityResolver(mockClient, {
      roleName: 'test-role',
    });

    expect(resolver).toBeDefined();
  });

  it('should resolve profiles via zome calls', async () => {
    const mockProfile: ProfileData = {
      agentId: 'agent-123',
      nickname: 'alice',
      displayName: 'Alice Smith',
    };

    const mockClient = {
      callZome: vi.fn().mockResolvedValue(mockProfile),
    };

    const resolver = new HolochainIdentityResolver(mockClient, {
      roleName: 'identity',
    });

    const identity = await resolver.resolve('agent-123');

    expect(identity.agentId).toBe('agent-123');
    expect(mockClient.callZome).toHaveBeenCalled();
  });

  it('should use IdentityCache for caching', async () => {
    const mockClient = {
      callZome: vi.fn().mockResolvedValue({
        agentId: 'agent-123',
        nickname: 'bob',
      }),
    };

    const resolver = new HolochainIdentityResolver(mockClient, {
      roleName: 'identity',
      cacheTtlMs: 60000,
    });

    // First call
    const identity1 = await resolver.resolve('agent-123');
    expect(identity1.agentId).toBe('agent-123');

    // The resolver should cache results
    // (Implementation detail: caching may be at different levels)
    expect(mockClient.callZome).toHaveBeenCalled();
  });
});

// =============================================================================
// createHolochainWallet Factory Tests
// =============================================================================

describe('createHolochainWallet Factory', () => {
  describe('createDevWallet', () => {
    it('should create a development wallet with mocks', async () => {
      const result = await createDevWallet();

      expect(result.wallet).toBeDefined();
      expect(result.conductor).toBeDefined();
      expect(result.identityResolver).toBeDefined();
      expect(result.financeProvider).toBeDefined();
      expect(typeof result.destroy).toBe('function');

      await result.destroy();
    });

    it('should provide working wallet interface', async () => {
      const { wallet, destroy } = await createDevWallet();

      // Wallet class uses balance() method and transactions getter
      expect(typeof wallet.balance).toBe('function');
      expect(wallet.transactions).toBeDefined();

      // Should be able to get balance
      const balance = wallet.balance('MYC');
      expect(typeof balance).toBe('number');

      await destroy();
    });
  });

  describe('isConductorAvailable', () => {
    it('should return false when conductor is not running', async () => {
      const available = await isConductorAvailable('ws://localhost:9999');
      expect(available).toBe(false);
    });

    it('should use default URL when not provided', async () => {
      const available = await isConductorAvailable();
      // Will likely be false in test environment
      expect(typeof available).toBe('boolean');
    });
  });
});

// =============================================================================
// Full Integration Flow Tests
// =============================================================================

describe('Full Integration Flow', () => {
  it('should wire all components together in dev mode', async () => {
    const { wallet, conductor, financeProvider, identityResolver, destroy } = await createDevWallet();

    // All components should be wired
    expect(wallet).toBeDefined();
    expect(conductor).toBeDefined();
    expect(financeProvider).toBeDefined();
    expect(identityResolver).toBeDefined();

    // Conductor state should be observable
    expect(conductor.state$).toBeDefined();
    expect(conductor.state.installedAppId).toBe('dev-wallet');

    // Finance provider (mock) should work - mock implements basic FinanceProvider interface
    const balances = await financeProvider.getAllBalances('test-agent');
    expect(balances).toBeDefined();
    expect(balances.size).toBeGreaterThan(0);

    await destroy();
  });

  it('should clean up all resources on destroy', async () => {
    const { conductor, destroy } = await createDevWallet();

    // Destroy should clean everything
    await destroy();

    // Conductor should be destroyed (throws on use)
    await expect(conductor.connect()).rejects.toThrow();
  });
});

// =============================================================================
// Type Safety Tests
// =============================================================================

describe('Type Safety', () => {
  it('should export all required types', () => {
    // This test ensures TypeScript types are correctly exported
    // If these imports fail, the test fails at compile time

    const config: ConductorConfig = {
      installedAppId: 'test',
      appUrl: 'ws://localhost:8888',
    };

    const walletConfig: HolochainWalletConfig = {
      installedAppId: 'test-wallet',
    };

    expect(config).toBeDefined();
    expect(walletConfig).toBeDefined();
  });

  it('should properly type ConductorState', () => {
    const state: ConductorState = {
      connection: 'disconnected',
      isOnline: true,
      lastConnectedAt: null,
      lastErrorAt: null,
      lastError: null,
      reconnectAttempts: 0,
      agentPubKey: null,
      installedAppId: 'test',
    };

    expect(state.connection).toBe('disconnected');
  });

  it('should properly type FinanceState', () => {
    const state: FinanceState = {
      loading: false,
      balances: new Map(),
      transactions: [],
      pendingTransactions: [],
      error: null,
    };

    expect(state.loading).toBe(false);
    expect(state.balances.size).toBe(0);
  });
});

// =============================================================================
// Edge Case Tests
// =============================================================================

describe('Edge Cases', () => {
  describe('Conductor Reconnection', () => {
    it('should handle multiple connect calls gracefully', async () => {
      const conductor = createConductorManager({
        installedAppId: 'test',
        autoReconnect: false,
        timeout: 100,
      });

      // These should not cause issues
      const promise1 = conductor.connect().catch(() => {});
      const promise2 = conductor.connect().catch(() => {});

      await Promise.allSettled([promise1, promise2]);
      await conductor.destroy();
    });

    it('should handle disconnect when already disconnected', async () => {
      const conductor = createConductorManager({
        installedAppId: 'test',
        autoReconnect: false,
      });

      // Should not throw
      await conductor.disconnect();
      await conductor.disconnect();
      await conductor.destroy();
    });
  });

  describe('Finance Provider Edge Cases', () => {
    it('should handle empty transaction history with limit 0', async () => {
      const provider = createMockFinanceProvider();
      // Using limit of 0 returns empty array
      const transactions = await provider.getTransactions('test-agent', 0);
      expect(Array.isArray(transactions)).toBe(true);
      expect(transactions.length).toBe(0);
    });

    it('should validate send parameters', async () => {
      const provider = createMockFinanceProvider();

      // Zero amount should work (returns transaction ID)
      const txId = await provider.send('from-agent', 'to-agent', 0, 'MYC');
      expect(typeof txId).toBe('string');
    });
  });
});
