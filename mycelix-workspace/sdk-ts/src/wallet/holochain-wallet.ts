/**
 * Holochain Wallet Factory - Wiring the Nervous System
 *
 * Factory function that creates a fully-wired wallet connecting:
 * - ConductorManager (WebSocket to conductor)
 * - HolochainIdentityResolver (profiles, DIDs)
 * - HolochainFinanceProvider (balances, payments)
 * - OfflineQueue (resilient transactions)
 * - Wallet facade (unified API)
 *
 * This is the "crossing from Architecture to Reality" moment.
 *
 * @example
 * ```typescript
 * // Create a fully-wired Holochain wallet
 * const wallet = await createHolochainWallet({
 *   installedAppId: 'mycelix-wallet',
 *   appUrl: 'ws://localhost:8888',
 * });
 *
 * // Now everything works against real zomes!
 * const balance = await wallet.getBalance('MYC');
 * await wallet.send('@alice', 100, 'MYC', 'Coffee');
 * ```
 */

import { type WalletBridge, createWalletBridge } from './bridge-integration.js';
import { type ConductorManager, createConductorManager, type ConductorConfig } from './conductor.js';
import {
  type HolochainFinanceProvider,
  createHolochainFinanceProvider,
} from './finance-provider.js';
import { HolochainIdentityResolver } from './identity-resolver.js';
import {
  type OfflineQueue,
  createOfflineQueue,
  createPersistentOfflineQueue,
} from './offline-queue.js';
import { Wallet } from './wallet.js';
import { createVault } from '../vault/index.js';

// =============================================================================
// Types
// =============================================================================

/** Configuration for creating a Holochain-connected wallet */
export interface HolochainWalletConfig {
  /** Holochain conductor WebSocket URL (default: ws://localhost:8888) */
  appUrl?: string;

  /** Installed app ID in the conductor */
  installedAppId: string;

  /** Current user's DID (will be fetched if not provided) */
  myDid?: string;

  /** DAO context for TEND balances (default: 'default-dao') */
  daoDid?: string;

  /** Enable offline queue with IndexedDB persistence (default: true for browsers) */
  enableOfflineQueue?: boolean;

  /** Auto-reconnect to conductor (default: true) */
  autoReconnect?: boolean;

  /** Auto-refresh balances interval in ms (default: 30000, 0 to disable) */
  refreshIntervalMs?: number;

  /** Connection timeout in ms (default: 30000) */
  timeout?: number;
}

/** Result of creating a Holochain wallet */
export interface HolochainWalletResult {
  /** The unified wallet facade */
  wallet: Wallet;

  /** The conductor connection manager */
  conductor: ConductorManager;

  /** The identity resolver */
  identityResolver: HolochainIdentityResolver;

  /** The finance provider */
  financeProvider: HolochainFinanceProvider;

  /** The offline queue (if enabled) */
  offlineQueue?: OfflineQueue;

  /** The bridge integration for cross-hApp reputation */
  bridge: WalletBridge;

  /** Cleanup function */
  destroy: () => Promise<void>;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a fully-wired Holochain wallet.
 *
 * This factory:
 * 1. Creates the conductor connection
 * 2. Wires up identity resolution to the identity hApp
 * 3. Wires up finance operations to the finance hApp
 * 4. Sets up offline queue for resilient transactions
 * 5. Returns a unified Wallet facade
 *
 * @example
 * ```typescript
 * const { wallet, conductor, destroy } = await createHolochainWallet({
 *   installedAppId: 'mycelix-wallet',
 * });
 *
 * // Use the wallet
 * await wallet.unlock('1234');
 * const balance = await wallet.getBalance('MYC');
 *
 * // Clean up when done
 * await destroy();
 * ```
 */
export async function createHolochainWallet(
  config: HolochainWalletConfig
): Promise<HolochainWalletResult> {
  // 1. Create conductor connection
  const conductorConfig: ConductorConfig = {
    appUrl: config.appUrl ?? 'ws://localhost:8888',
    installedAppId: config.installedAppId,
    autoReconnect: config.autoReconnect ?? true,
    timeout: config.timeout ?? 30000,
  };

  const conductor = createConductorManager(conductorConfig);

  // 2. Connect to conductor
  await conductor.connect();

  // 3. Get agent DID (use provided or derive from agent key)
  let myDid = config.myDid;
  if (!myDid && conductor.agentPubKey) {
    // Derive DID from agent public key
    const pubKeyHex = Array.from(conductor.agentPubKey)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
    myDid = `did:mycelix:${pubKeyHex.slice(0, 32)}`;
  }
  myDid = myDid ?? 'did:mycelix:unknown';

  // 4. Create offline queue (if enabled)
  let offlineQueue: OfflineQueue | undefined;
  const enableOffline = config.enableOfflineQueue ?? (typeof window !== 'undefined');

  if (enableOffline) {
    // Use IndexedDB in browsers, in-memory elsewhere
    if (typeof window !== 'undefined' && typeof indexedDB !== 'undefined') {
      offlineQueue = createPersistentOfflineQueue(`mycelix-wallet-${config.installedAppId}`);
    } else {
      offlineQueue = createOfflineQueue();
    }
    await offlineQueue.initialize();
  }

  // 5. Create finance provider
  const financeProvider = createHolochainFinanceProvider(conductor, {
    myDid,
    daoDid: config.daoDid ?? 'default-dao',
    roleName: config.installedAppId,
    offlineQueue,
    refreshIntervalMs: config.refreshIntervalMs ?? 30000,
  });

  // 6. Wire offline queue executor to finance provider
  // Now that we have the setExecutor() method, we can properly wire the executor
  if (offlineQueue) {
    offlineQueue.setExecutor(financeProvider.createQueueExecutor());
  }

  // 7. Create identity resolver
  const identityResolver = new HolochainIdentityResolver(
    {
      callZome: async (params) => {
        return conductor.callZome({
          role_name: params.role_name ?? config.installedAppId,
          zome_name: params.zome_name,
          fn_name: params.fn_name,
          payload: params.payload,
        });
      },
    },
    {
      roleName: config.installedAppId,
      cacheTtlMs: 5 * 60 * 1000, // 5 minute cache
    }
  );

  // 8. Create a minimal vault for the wallet
  const { vault } = await createVault({
    name: `holochain-${config.installedAppId}`,
    pin: '0000', // Default PIN for programmatic use
  });

  // 9. Create the wallet facade by connecting components
  const wallet = await Wallet.connect({
    vault,
    identityResolver,
    financeProvider,
  });

  // 10. Create bridge integration for cross-hApp reputation
  const bridge = createWalletBridge(financeProvider);
  bridge.watchTransactions(); // Auto-update reputation based on transactions

  // 11. Start auto-refresh if configured
  if (config.refreshIntervalMs !== 0) {
    financeProvider.startAutoRefresh();
  }

  // 12. Return everything
  return {
    wallet,
    conductor,
    identityResolver,
    financeProvider,
    offlineQueue,
    bridge,
    destroy: async () => {
      bridge.destroy();
      financeProvider.stopAutoRefresh();
      financeProvider.destroy();
      if (offlineQueue) {
        offlineQueue.destroy();
      }
      await conductor.destroy();
    },
  };
}

/**
 * Create a development wallet with mock conductor.
 * Useful for UI development without a running conductor.
 */
export async function createDevWallet(): Promise<HolochainWalletResult> {
  // Create mock implementations
  const { createMockFinanceProvider } = await import('./finance-provider.js');
  const mockFinance = createMockFinanceProvider();

  // Create a minimal conductor manager (won't actually connect)
  const conductor = createConductorManager({
    installedAppId: 'dev-wallet',
    appUrl: 'ws://localhost:8888',
    autoReconnect: false,
  });

  // Create identity resolver with mock client
  const identityResolver = new HolochainIdentityResolver(
    {
      callZome: async (params) => {
        // Return mock data
        if (params.fn_name === 'get_profile') {
          return {
            agentId: params.payload as string,
            nickname: 'dev-user',
            displayName: 'Development User',
          };
        }
        return null;
      },
    },
    { roleName: 'identity' }
  );

  // Create a minimal vault for the dev wallet
  const { vault } = await createVault({
    name: 'dev-wallet',
    pin: '0000', // Default PIN for dev mode
  });

  // Create wallet with mocks via Wallet.connect
  const wallet = await Wallet.connect({
    vault,
    identityResolver,
    financeProvider: mockFinance,
  });

  // Create bridge (without transaction watching since it's mock)
  const bridge = createWalletBridge();

  return {
    wallet,
    conductor,
    identityResolver,
    financeProvider: mockFinance as HolochainFinanceProvider,
    bridge,
    destroy: async () => {
      bridge.destroy();
      await conductor.destroy();
    },
  };
}

/**
 * Check if the conductor is available at the given URL.
 */
export async function isConductorAvailable(appUrl = 'ws://localhost:8888'): Promise<boolean> {
  try {
    const conductor = createConductorManager({
      installedAppId: 'health-check',
      appUrl,
      autoReconnect: false,
      timeout: 5000,
    });

    await conductor.connect();
    await conductor.disconnect();
    return true;
  } catch {
    return false;
  }
}
