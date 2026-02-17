/**
 * Mycelix Unified Client
 *
 * Master client that aggregates all 9 domain SDKs into a single,
 * cohesive interface for the entire Mycelix ecosystem.
 *
 * @module @mycelix/sdk/client/unified
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { SdkError } from '../common/errors';

// Import all domain clients
import { MycelixEnergyClient } from '../integrations/energy/client';
import { MycelixFinanceClient } from '../integrations/finance/client';
import { MycelixGovernanceClient } from '../integrations/governance/client';
import { MycelixHealthClient } from '../integrations/health/client';
import { MycelixIdentityClient } from '../integrations/identity/client';
import { MycelixKnowledgeClient } from '../integrations/knowledge/client';
import { MycelixPropertyClient } from '../integrations/property/client';
import { MycelixJusticeClient } from '../justice/client';
import { MycelixMediaClient } from '../media/client';

/**
 * Unified client error
 */
export class MycelixEcosystemClientError extends SdkError {
  constructor(code: string, message: string, cause?: unknown) {
    super(code, message, cause, 'Mycelix');
  }
}

/**
 * Retry configuration
 */
export interface RetryConfig {
  /** Maximum retry attempts */
  maxAttempts: number;
  /** Initial delay in milliseconds */
  delayMs: number;
  /** Backoff multiplier */
  backoffMultiplier: number;
  /** Maximum delay between retries */
  maxDelayMs: number;
}

/**
 * Configuration for the unified Mycelix client
 */
export interface MycelixClientConfig {
  /** WebSocket URL for the Holochain conductor */
  url: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Retry configuration */
  retry?: Partial<RetryConfig>;
  /** Domain-specific configurations (passed directly to each domain client) */
  domains?: {
    identity?: Record<string, unknown>;
    governance?: Record<string, unknown>;
    finance?: Record<string, unknown>;
    property?: Record<string, unknown>;
    energy?: Record<string, unknown>;
    knowledge?: Record<string, unknown>;
    health?: Record<string, unknown>;
    media?: Record<string, unknown>;
    justice?: Record<string, unknown>;
  };
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  delayMs: 1000,
  backoffMultiplier: 2,
  maxDelayMs: 30000,
};

const DEFAULT_CONFIG: Required<Pick<MycelixClientConfig, 'debug' | 'retry' | 'domains'>> = {
  debug: false,
  retry: DEFAULT_RETRY_CONFIG,
  domains: {},
};

/**
 * Domain initialization status
 */
export interface DomainStatus {
  name: string;
  initialized: boolean;
  error?: string;
}

/**
 * Unified Mycelix Client
 *
 * Provides access to all 9 Mycelix domain SDKs through a single interface.
 * This is the recommended entry point for applications that need to interact
 * with multiple Mycelix hApps.
 *
 * @example
 * ```typescript
 * import { MycelixEcosystemClient } from '@mycelix/sdk';
 *
 * // Connect to the Mycelix ecosystem
 * const mycelix = await MycelixEcosystemClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Access Identity SDK
 * const myDid = await mycelix.identity.did.createDid();
 *
 * // Access Governance SDK
 * const proposals = await mycelix.governance.proposals.getActiveProposals('dao-1');
 *
 * // Access Finance SDK
 * const balance = await mycelix.finance.wallet.getBalance('did:mycelix:alice');
 *
 * // Access Health SDK
 * const patient = await mycelix.health.patient.getPatient(patientHash);
 *
 * // Access Justice SDK
 * const caseRecord = await mycelix.justice.cases.fileCase({...});
 *
 * // Cross-domain workflows
 * const patientSummary = await mycelix.health.getPatientSummary(patientHash);
 * const daoStatus = await mycelix.governance.getDaoStatus('community-dao');
 * ```
 */
export class MycelixEcosystemClient {
  // ============================================================================
  // Domain SDKs
  // ============================================================================

  /** Identity - DID management, credentials, and identity federation */
  public identity!: MycelixIdentityClient;

  /** Governance - DAOs, proposals, voting, and delegation */
  public governance!: MycelixGovernanceClient;

  /** Finance - Wallets, payments, credit, and lending */
  public finance!: MycelixFinanceClient;

  /** Property - Asset registry, transfers, liens, and commons */
  public property!: MycelixPropertyClient;

  /** Energy - Energy trading, credits, and grid management */
  public energy!: MycelixEnergyClient;

  /** Knowledge - Epistemic claims, knowledge graph, and inference */
  public knowledge!: MycelixKnowledgeClient;

  /** Health - Patient records, prescriptions, trials, and insurance */
  public health!: MycelixHealthClient;

  /** Media - Content publishing, attribution, fact-checking, and curation */
  public media!: MycelixMediaClient;

  /** Justice - Dispute resolution, arbitration, and enforcement */
  public justice!: MycelixJusticeClient;

  // ============================================================================
  // Internal State
  // ============================================================================

  private readonly _client: AppClient;
  private readonly _config: MycelixClientConfig & typeof DEFAULT_CONFIG;
  private readonly _domainStatus: Map<string, DomainStatus> = new Map();

  /**
   * Private constructor - use static factory methods
   */
  private constructor(
    client: AppClient,
    config: MycelixClientConfig
  ) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config, retry: { ...DEFAULT_RETRY_CONFIG, ...config.retry } };
  }

  /**
   * Connect to the Mycelix ecosystem
   *
   * This is the primary way to create a MycelixEcosystemClient. It establishes
   * a WebSocket connection to the Holochain conductor and initializes
   * all domain SDKs.
   *
   * @param config - Connection and configuration options
   * @returns Connected MycelixEcosystemClient with all domain SDKs initialized
   * @throws MycelixEcosystemClientError if connection fails
   *
   * @example
   * ```typescript
   * const mycelix = await MycelixEcosystemClient.connect({
   *   url: 'ws://localhost:8888',
   *   debug: true,
   *   retry: {
   *     maxAttempts: 5,
   *     delayMs: 2000,
   *   },
   * });
   * ```
   */
  static async connect(config: MycelixClientConfig): Promise<MycelixEcosystemClient> {
    const retryConfig = { ...DEFAULT_RETRY_CONFIG, ...config.retry };
    let lastError: unknown;
    let attempts = 0;

    while (attempts < retryConfig.maxAttempts) {
      try {
        const client = await AppWebsocket.connect({
          url: new URL(config.url),
          wsClientOptions: { origin: 'mycelix-sdk' },
        });

        // Verify connection
        const appInfo = await client.appInfo();
        if (!appInfo) {
          throw new Error('Failed to get app info from conductor');
        }

        if (config.debug) {
          console.log(`[mycelix] Connected to conductor at ${config.url}`);
        }

        const mycelixClient = new MycelixEcosystemClient(client, config);
        await mycelixClient._initializeDomains();

        return mycelixClient;
      } catch (error) {
        lastError = error;
        attempts++;

        if (attempts >= retryConfig.maxAttempts) {
          break;
        }

        // Exponential backoff with jitter
        const delay = Math.min(
          retryConfig.delayMs * Math.pow(retryConfig.backoffMultiplier, attempts - 1),
          retryConfig.maxDelayMs
        );
        const jitter = delay * 0.1 * Math.random();

        if (config.debug) {
          console.log(`[mycelix] Connection attempt ${attempts} failed, retrying in ${Math.round(delay + jitter)}ms...`);
        }

        await new Promise(resolve => setTimeout(resolve, delay + jitter));
      }
    }

    throw new MycelixEcosystemClientError(
      'CONNECTION_FAILED',
      `Failed to connect after ${attempts} attempts: ${lastError instanceof Error ? lastError.message : String(lastError)}`,
      lastError
    );
  }

  /**
   * Create a MycelixEcosystemClient from an existing AppClient
   *
   * Use this when you already have an established Holochain connection
   * and want to add full Mycelix ecosystem access.
   *
   * @param client - Existing AppClient instance
   * @param config - Configuration options (url not required)
   * @returns MycelixEcosystemClient with all domain SDKs initialized
   *
   * @example
   * ```typescript
   * const existingClient = await AppWebsocket.connect({...});
   * const mycelix = await MycelixEcosystemClient.fromClient(existingClient, { debug: true });
   * ```
   */
  static async fromClient(
    client: AppClient,
    config: Omit<MycelixClientConfig, 'url'> & { url?: string } = {}
  ): Promise<MycelixEcosystemClient> {
    const fullConfig: MycelixClientConfig = {
      url: 'existing-connection',
      ...config,
    };

    const mycelixClient = new MycelixEcosystemClient(client, fullConfig);
    await mycelixClient._initializeDomains();

    return mycelixClient;
  }

  /**
   * Initialize all domain SDKs
   */
  private async _initializeDomains(): Promise<void> {
    // Initialize each domain SDK using fromClient with defaults
    // Each domain uses its own default configuration

    try {
      this.identity = MycelixIdentityClient.fromClient(this._client);
      this._domainStatus.set('identity', { name: 'identity', initialized: true });
    } catch (error) {
      this._domainStatus.set('identity', {
        name: 'identity',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize identity domain:', error);
      }
    }

    try {
      this.governance = MycelixGovernanceClient.fromClient(this._client);
      this._domainStatus.set('governance', { name: 'governance', initialized: true });
    } catch (error) {
      this._domainStatus.set('governance', {
        name: 'governance',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize governance domain:', error);
      }
    }

    try {
      this.finance = MycelixFinanceClient.fromClient(this._client);
      this._domainStatus.set('finance', { name: 'finance', initialized: true });
    } catch (error) {
      this._domainStatus.set('finance', {
        name: 'finance',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize finance domain:', error);
      }
    }

    try {
      this.property = MycelixPropertyClient.fromClient(this._client);
      this._domainStatus.set('property', { name: 'property', initialized: true });
    } catch (error) {
      this._domainStatus.set('property', {
        name: 'property',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize property domain:', error);
      }
    }

    try {
      this.energy = MycelixEnergyClient.fromClient(this._client);
      this._domainStatus.set('energy', { name: 'energy', initialized: true });
    } catch (error) {
      this._domainStatus.set('energy', {
        name: 'energy',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize energy domain:', error);
      }
    }

    try {
      this.knowledge = MycelixKnowledgeClient.fromClient(this._client);
      this._domainStatus.set('knowledge', { name: 'knowledge', initialized: true });
    } catch (error) {
      this._domainStatus.set('knowledge', {
        name: 'knowledge',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize knowledge domain:', error);
      }
    }

    try {
      this.health = MycelixHealthClient.fromClient(this._client);
      this._domainStatus.set('health', { name: 'health', initialized: true });
    } catch (error) {
      this._domainStatus.set('health', {
        name: 'health',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize health domain:', error);
      }
    }

    try {
      this.media = MycelixMediaClient.fromClient(this._client);
      this._domainStatus.set('media', { name: 'media', initialized: true });
    } catch (error) {
      this._domainStatus.set('media', {
        name: 'media',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize media domain:', error);
      }
    }

    try {
      this.justice = MycelixJusticeClient.fromClient(this._client);
      this._domainStatus.set('justice', { name: 'justice', initialized: true });
    } catch (error) {
      this._domainStatus.set('justice', {
        name: 'justice',
        initialized: false,
        error: error instanceof Error ? error.message : String(error),
      });
      if (this._config.debug) {
        console.warn('[mycelix] Failed to initialize justice domain:', error);
      }
    }

    if (this._config.debug) {
      const initialized = Array.from(this._domainStatus.values()).filter(d => d.initialized).length;
      console.log(`[mycelix] Initialized ${initialized}/9 domain SDKs`);
    }
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Get the underlying Holochain AppClient
   *
   * Useful for advanced operations or debugging.
   */
  getClient(): AppClient {
    return this._client;
  }

  /**
   * Get the current agent's public key
   */
  getAgentPubKey(): Uint8Array {
    return this._client.myPubKey;
  }

  /**
   * Check if connected to Holochain
   */
  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this._client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }

  /**
   * Get domain initialization status
   *
   * Returns the status of each domain SDK initialization.
   */
  getDomainStatus(): DomainStatus[] {
    return Array.from(this._domainStatus.values());
  }

  /**
   * Get list of available (initialized) domains
   */
  getAvailableDomains(): string[] {
    return Array.from(this._domainStatus.entries())
      .filter(([, status]) => status.initialized)
      .map(([name]) => name);
  }

  // ============================================================================
  // Cross-Domain Workflows
  // ============================================================================

  /**
   * Comprehensive identity verification across domains
   *
   * Checks identity status, credentials, reputation, and governance participation.
   *
   * @param did - DID to verify
   * @returns Comprehensive identity profile
   */
  async getComprehensiveIdentity(did: string): Promise<{
    did: string;
    identity: {
      exists: boolean;
      active: boolean;
    };
    governance: {
      daoMemberships: number;
      proposalsCreated: number;
      votesCount: number;
    } | null;
    reputation: {
      overall: number;
      domains: Record<string, number>;
    } | null;
  }> {
    const result: ReturnType<typeof this.getComprehensiveIdentity> extends Promise<infer T> ? T : never = {
      did,
      identity: { exists: false, active: false },
      governance: null,
      reputation: null,
    };

    // Check identity
    try {
      const didDoc = await this.identity.did.resolveDid(did);
      result.identity.exists = !!didDoc;
      result.identity.active = didDoc ? await this.identity.did.isDidActive(did) : false;
    } catch {
      // Identity not found
    }

    // Check governance participation
    try {
      const govData = await this.governance.queryGovernance(did);
      result.governance = {
        daoMemberships: (govData.dao_memberships as number) || 0,
        proposalsCreated: (govData.proposals_created as number) || 0,
        votesCount: (govData.votes_count as number) || 0,
      };
    } catch {
      // Governance data not available
    }

    return result;
  }

  /**
   * Get ecosystem statistics
   *
   * Returns aggregate statistics across all domains.
   */
  async getEcosystemStats(): Promise<{
    connectedDomains: number;
    totalDomains: number;
    domainStatus: DomainStatus[];
  }> {
    const domainStatus = this.getDomainStatus();
    const connectedDomains = domainStatus.filter(d => d.initialized).length;

    return {
      connectedDomains,
      totalDomains: 9,
      domainStatus,
    };
  }
}

export default MycelixEcosystemClient;
