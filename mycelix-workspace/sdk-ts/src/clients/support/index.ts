/**
 * Support hApp Client
 *
 * Complete TypeScript client for the Mycelix Support domain providing access
 * to all 3 support zomes:
 *
 * - **knowledge** - Articles, resolutions, reputation, curation
 * - **tickets** - Support tickets, comments, autonomous actions, preemptive alerts
 * - **diagnostics** - Diagnostic runs, privacy preferences, cognitive updates
 *
 * @module @mycelix/sdk/clients/support
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { KnowledgeClient } from './knowledge';
import { TicketsClient } from './tickets';
import { DiagnosticsClient } from './diagnostics';
import { SupportError } from './types';

// ============================================================================
// Client Configuration
// ============================================================================

export interface SupportClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface SupportConnectionOptions {
  url: string;
  timeout?: number;
  config?: SupportClientConfig;
}

const DEFAULT_CONFIG: Required<SupportClientConfig> = {
  roleName: 'commons',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate Support Client
// ============================================================================

/**
 * Unified Support hApp Client
 *
 * @example
 * ```typescript
 * const support = await SupportClient.connect({ url: 'ws://localhost:8888' });
 *
 * const article = await support.knowledge.createArticle({
 *   title: 'How to restart a node',
 *   content: 'Step-by-step guide...',
 *   category: 'Holochain',
 *   tags: ['node', 'restart'],
 *   author: myPubKey,
 *   source: 'Community',
 *   difficultyLevel: 'Beginner',
 *   upvotes: 0,
 *   verified: false,
 *   deprecated: false,
 *   deprecationReason: null,
 *   version: 1,
 * });
 *
 * const ticketHash = await support.tickets.createTicket({
 *   title: 'Node not syncing',
 *   description: 'My node stopped syncing 2 hours ago',
 *   category: 'Holochain',
 *   priority: 'High',
 *   status: 'Open',
 *   requester: myPubKey,
 *   assignee: null,
 *   autonomyLevel: 'Advisory',
 *   systemInfo: null,
 *   isPreemptive: false,
 *   predictionConfidence: null,
 *   createdAt: Date.now(),
 *   updatedAt: Date.now(),
 * });
 * ```
 */
export class SupportClient {
  public readonly knowledge: KnowledgeClient;
  public readonly tickets: TicketsClient;
  public readonly diagnostics: DiagnosticsClient;

  private readonly _client: AppClient;
  private readonly _config: Required<SupportClientConfig>;

  private constructor(client: AppClient, config: SupportClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.knowledge = new KnowledgeClient(client, baseConfig);
    this.tickets = new TicketsClient(client, baseConfig);
    this.diagnostics = new DiagnosticsClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[support-sdk] Client initialized with 3 zome clients');
    }
  }

  static async connect(options: SupportConnectionOptions): Promise<SupportClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new SupportClient(client, options.config);
    } catch (error) {
      throw new SupportError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  static fromClient(client: AppClient, config: SupportClientConfig = {}): SupportClient {
    return new SupportClient(client, config);
  }

  getClient(): AppClient {
    return this._client;
  }

  getAgentPubKey(): Uint8Array {
    return this._client.myPubKey;
  }

  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this._client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createSupportClient(client: AppClient, config?: SupportClientConfig): SupportClient {
  return SupportClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { KnowledgeClient, type KnowledgeClientConfig } from './knowledge';
export { TicketsClient, type TicketsClientConfig } from './tickets';
export { DiagnosticsClient, type DiagnosticsClientConfig } from './diagnostics';
export * from './types';
export default SupportClient;
