// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Unified Client Bootstrap
 *
 * Single initialization point that wires all 16 services together
 * with proper dependency injection and lifecycle management.
 */

import { AppWebsocket, AppAgentClient } from '@holochain/client';
import { SignalHub } from '../SignalHub';

// Import all services
import {
  CryptoService,
  CacheService,
  HealthService,
  BatchService,
  SpamFilterService,
  ImportExportService,
  NotificationQueueService,
  OfflineQueueService,
  EmailStateMachine,
  MetricsService,
  KeyExchangeService,
  ContactService,
  SchedulerService,
  ThreadService,
  AuditLogService,
} from '../services';

// Import zome clients
import { MessagesClient } from '../zomes/MessagesClient';
import { TrustClient } from '../zomes/TrustClient';
import { ProfilesClient } from '../zomes/ProfilesClient';
import { CapabilitiesClient } from '../zomes/CapabilitiesClient';
import { SyncClient } from '../zomes/SyncClient';
import { FederationClient } from '../zomes/FederationClient';
import { SearchClient } from '../zomes/SearchClient';
import { BackupClient } from '../zomes/BackupClient';
import { ContactsClient } from '../zomes/ContactsClient';
import { KeysClient } from '../zomes/KeysClient';
import { SchedulerClient } from '../zomes/SchedulerClient';
import { AuditClient } from '../zomes/AuditClient';

export interface MycelixClientConfig {
  /** Holochain websocket URL */
  websocketUrl: string;
  /** Installed app ID */
  appId: string;
  /** Cache configuration */
  cache?: {
    maxSize?: number;
    ttlMs?: number;
    persistToIndexedDB?: boolean;
  };
  /** Notification configuration */
  notifications?: {
    enabled?: boolean;
    quietHoursStart?: number;
    quietHoursEnd?: number;
  };
  /** Offline configuration */
  offline?: {
    maxQueueSize?: number;
    syncIntervalMs?: number;
  };
  /** Metrics configuration */
  metrics?: {
    enabled?: boolean;
    flushIntervalMs?: number;
  };
  /** Health check configuration */
  health?: {
    checkIntervalMs?: number;
    alertThresholds?: Record<string, number>;
  };
}

export interface ServiceContainer {
  // Core infrastructure
  signalHub: SignalHub;
  cache: CacheService;
  health: HealthService;
  metrics: MetricsService;

  // Crypto and security
  crypto: CryptoService;
  keyExchange: KeyExchangeService;

  // Email operations
  messages: MessagesClient;
  emailState: EmailStateMachine;
  thread: ThreadService;
  batch: BatchService;
  scheduler: SchedulerService;
  importExport: ImportExportService;

  // Trust and contacts
  trust: TrustClient;
  contacts: ContactService;
  contactsZome: ContactsClient;
  spam: SpamFilterService;

  // Sync and offline
  sync: SyncClient;
  offline: OfflineQueueService;

  // Federation
  federation: FederationClient;

  // Search and backup
  search: SearchClient;
  backup: BackupClient;

  // Notifications and audit
  notifications: NotificationQueueService;
  audit: AuditLogService;

  // Profiles and capabilities
  profiles: ProfilesClient;
  capabilities: CapabilitiesClient;

  // Keys
  keys: KeysClient;
  schedulerZome: SchedulerClient;
  auditZome: AuditClient;
}

export class MycelixClient {
  private client: AppAgentClient | null = null;
  private services: ServiceContainer | null = null;
  private config: MycelixClientConfig;
  private initialized = false;
  private shutdownCallbacks: (() => Promise<void>)[] = [];

  constructor(config: MycelixClientConfig) {
    this.config = config;
  }

  /**
   * Initialize the Mycelix client and all services
   */
  async initialize(): Promise<ServiceContainer> {
    if (this.initialized && this.services) {
      return this.services;
    }

    console.log('[MycelixClient] Initializing...');

    // Connect to Holochain
    const websocket = await AppWebsocket.connect(this.config.websocketUrl);
    this.client = await AppAgentClient.connect(websocket, this.config.appId);

    // Create service container with dependency injection
    this.services = await this.createServices(this.client);

    // Start background processes
    await this.startBackgroundProcesses();

    // Register shutdown handlers
    this.registerShutdownHandlers();

    this.initialized = true;
    console.log('[MycelixClient] Initialization complete');

    return this.services;
  }

  /**
   * Create all services with proper dependency injection
   */
  private async createServices(client: AppAgentClient): Promise<ServiceContainer> {
    const roleName = 'mycelix_mail';
    const zomeName = (name: string) => `mail_${name}`;

    // Phase 1: Core infrastructure (no dependencies)
    console.log('[MycelixClient] Creating core infrastructure...');

    const signalHub = new SignalHub();

    const cache = new CacheService({
      maxSize: this.config.cache?.maxSize ?? 1000,
      ttlMs: this.config.cache?.ttlMs ?? 5 * 60 * 1000,
      persistToIndexedDB: this.config.cache?.persistToIndexedDB ?? true,
    });

    const health = new HealthService({
      checkIntervalMs: this.config.health?.checkIntervalMs ?? 30000,
    });

    const metrics = new MetricsService({
      enabled: this.config.metrics?.enabled ?? true,
      flushIntervalMs: this.config.metrics?.flushIntervalMs ?? 60000,
    });

    // Phase 2: Crypto services
    console.log('[MycelixClient] Creating crypto services...');

    const crypto = new CryptoService();
    await crypto.initialize();

    // Phase 3: Zome clients
    console.log('[MycelixClient] Creating zome clients...');

    const messages = new MessagesClient(client, roleName, zomeName('messages'));
    const trust = new TrustClient(client, roleName, zomeName('trust'));
    const profiles = new ProfilesClient(client, roleName, zomeName('profiles'));
    const capabilities = new CapabilitiesClient(client, roleName, zomeName('capabilities'));
    const sync = new SyncClient(client, roleName, zomeName('sync'));
    const federation = new FederationClient(client, roleName, zomeName('federation'));
    const search = new SearchClient(client, roleName, zomeName('search'));
    const backup = new BackupClient(client, roleName, zomeName('backup'));
    const contactsZome = new ContactsClient(client, roleName, zomeName('contacts'));
    const keys = new KeysClient(client, roleName, zomeName('keys'));
    const schedulerZome = new SchedulerClient(client, roleName, zomeName('scheduler'));
    const auditZome = new AuditClient(client, roleName, zomeName('audit'));

    // Phase 4: Higher-level services with dependencies
    console.log('[MycelixClient] Creating application services...');

    const keyExchange = new KeyExchangeService(keys, crypto);
    await keyExchange.initialize();

    const emailState = new EmailStateMachine(messages);

    const thread = new ThreadService(messages, cache);

    const batch = new BatchService(messages, {
      maxBatchSize: 50,
      parallelism: 4,
    });

    const scheduler = new SchedulerService(schedulerZome, messages);

    const importExport = new ImportExportService(messages, {
      batchSize: 100,
    });

    const contacts = new ContactService(contactsZome, trust, cache);

    const spam = new SpamFilterService(trust, {
      trustThreshold: 0.3,
      autoQuarantine: true,
    });

    const offline = new OfflineQueueService(messages, sync, {
      maxQueueSize: this.config.offline?.maxQueueSize ?? 10000,
      syncIntervalMs: this.config.offline?.syncIntervalMs ?? 30000,
    });
    await offline.initialize();

    const notifications = new NotificationQueueService({
      enabled: this.config.notifications?.enabled ?? true,
      quietHoursStart: this.config.notifications?.quietHoursStart,
      quietHoursEnd: this.config.notifications?.quietHoursEnd,
    });

    const audit = new AuditLogService(auditZome, {
      enabled: true,
      localStorageKey: 'mycelix_audit',
    });

    // Phase 5: Wire up signal handlers
    console.log('[MycelixClient] Wiring signal handlers...');

    this.wireSignalHandlers(signalHub, {
      messages,
      trust,
      sync,
      notifications,
      metrics,
      offline,
      cache,
    });

    // Phase 6: Register health checks
    console.log('[MycelixClient] Registering health checks...');

    health.registerCheck('holochain', async () => {
      try {
        await client.appInfo();
        return { status: 'healthy' };
      } catch (e) {
        return { status: 'unhealthy', error: String(e) };
      }
    });

    health.registerCheck('cache', async () => ({
      status: 'healthy',
      details: { size: cache.size() },
    }));

    health.registerCheck('offline_queue', async () => ({
      status: 'healthy',
      details: { pending: offline.getPendingCount() },
    }));

    health.registerCheck('keys', async () => {
      const status = await keys.needsRefresh();
      return {
        status: status === 'Ok' ? 'healthy' : 'degraded',
        details: { bundleStatus: status },
      };
    });

    return {
      signalHub,
      cache,
      health,
      metrics,
      crypto,
      keyExchange,
      messages,
      emailState,
      thread,
      batch,
      scheduler,
      importExport,
      trust,
      contacts,
      contactsZome,
      spam,
      sync,
      offline,
      federation,
      search,
      backup,
      notifications,
      audit,
      profiles,
      capabilities,
      keys,
      schedulerZome,
      auditZome,
    };
  }

  /**
   * Wire up signal handlers for real-time updates
   */
  private wireSignalHandlers(
    signalHub: SignalHub,
    services: {
      messages: MessagesClient;
      trust: TrustClient;
      sync: SyncClient;
      notifications: NotificationQueueService;
      metrics: MetricsService;
      offline: OfflineQueueService;
      cache: CacheService;
    }
  ): void {
    // New email received
    signalHub.on('email_received', async (signal) => {
      services.metrics.increment('emails.received');
      services.cache.invalidate(`inbox:*`);

      services.notifications.enqueue({
        type: 'email',
        priority: signal.priority ?? 'normal',
        title: `New email from ${signal.from}`,
        body: signal.subject,
        data: { emailHash: signal.hash },
      });
    });

    // Email delivered
    signalHub.on('email_delivered', async (signal) => {
      services.metrics.increment('emails.delivered');
      services.cache.invalidate(`sent:*`);
    });

    // Trust attestation received
    signalHub.on('attestation_received', async (signal) => {
      services.metrics.increment('trust.attestations_received');
      services.cache.invalidate(`trust:${signal.from}`);
    });

    // Sync state changed
    signalHub.on('sync_state_changed', async (signal) => {
      if (signal.state === 'synced') {
        services.offline.onSyncComplete();
      }
    });

    // Key bundle low
    signalHub.on('keys_low', async (signal) => {
      services.notifications.enqueue({
        type: 'system',
        priority: 'high',
        title: 'Pre-key bundle running low',
        body: `Only ${signal.remaining} keys remaining. Consider rotating.`,
      });
    });
  }

  /**
   * Start background processes
   */
  private async startBackgroundProcesses(): Promise<void> {
    if (!this.services) return;

    console.log('[MycelixClient] Starting background processes...');

    // Start health checks
    this.services.health.start();
    this.shutdownCallbacks.push(async () => this.services!.health.stop());

    // Start metrics collection
    this.services.metrics.start();
    this.shutdownCallbacks.push(async () => this.services!.metrics.stop());

    // Start offline sync
    this.services.offline.start();
    this.shutdownCallbacks.push(async () => this.services!.offline.stop());

    // Start scheduler polling
    this.services.scheduler.startPolling(60000);
    this.shutdownCallbacks.push(async () => this.services!.scheduler.stopPolling());

    // Start notification processing
    this.services.notifications.start();
    this.shutdownCallbacks.push(async () => this.services!.notifications.stop());
  }

  /**
   * Register shutdown handlers
   */
  private registerShutdownHandlers(): void {
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => this.shutdown());
    }

    if (typeof process !== 'undefined') {
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());
    }
  }

  /**
   * Get the service container
   */
  getServices(): ServiceContainer {
    if (!this.services) {
      throw new Error('MycelixClient not initialized. Call initialize() first.');
    }
    return this.services;
  }

  /**
   * Get a specific service
   */
  getService<K extends keyof ServiceContainer>(name: K): ServiceContainer[K] {
    return this.getServices()[name];
  }

  /**
   * Check if client is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get client health status
   */
  async getHealth(): Promise<Record<string, any>> {
    if (!this.services) {
      return { status: 'not_initialized' };
    }
    return this.services.health.getStatus();
  }

  /**
   * Get client metrics
   */
  getMetrics(): Record<string, any> {
    if (!this.services) {
      return {};
    }
    return this.services.metrics.getAll();
  }

  /**
   * Shutdown the client gracefully
   */
  async shutdown(): Promise<void> {
    if (!this.initialized) return;

    console.log('[MycelixClient] Shutting down...');

    // Run shutdown callbacks in reverse order
    for (const callback of this.shutdownCallbacks.reverse()) {
      try {
        await callback();
      } catch (e) {
        console.error('[MycelixClient] Shutdown callback error:', e);
      }
    }

    // Clear cache
    if (this.services) {
      this.services.cache.clear();
    }

    this.initialized = false;
    this.services = null;
    this.client = null;

    console.log('[MycelixClient] Shutdown complete');
  }
}

/**
 * Create a singleton instance for convenience
 */
let defaultClient: MycelixClient | null = null;

export function createMycelixClient(config: MycelixClientConfig): MycelixClient {
  defaultClient = new MycelixClient(config);
  return defaultClient;
}

export function getMycelixClient(): MycelixClient {
  if (!defaultClient) {
    throw new Error('MycelixClient not created. Call createMycelixClient() first.');
  }
  return defaultClient;
}

/**
 * React hook for using the Mycelix client
 */
export function useMycelixClient(): ServiceContainer {
  return getMycelixClient().getServices();
}

export default MycelixClient;
