/**
 * Mycelix Fabrication Client
 *
 * Main entry point for the Fabrication hApp SDK.
 * Provides access to all sub-clients for:
 * - Designs (HDC-enhanced parametric designs)
 * - Printers (printer registry and matching)
 * - Prints (job management with PoGF scoring)
 * - Materials (specification and sourcing)
 * - Verification (safety claims with Knowledge bridge)
 * - Bridge (cross-hApp integration)
 * - Symthaea (HDC operations and AI assistance)
 */

import type { AppClient, AppWebsocket, AppInfo } from '@holochain/client';
import { DesignsClient } from './clients/designs';
import { PrintersClient } from './clients/printers';
import { PrintsClient } from './clients/prints';
import { MaterialsClient } from './clients/materials';
import { VerificationClient } from './clients/verification';
import { BridgeClient } from './clients/bridge';
import { SymthaeaClient } from './clients/symthaea';

export interface FabricationConfig {
  /**
   * Holochain app websocket URL
   * Default: 'ws://localhost:8888'
   */
  url?: string;

  /**
   * Installed app ID
   * Default: 'fabrication'
   */
  installedAppId?: string;

  /**
   * Role name for the DNA
   * Default: 'fabrication'
   */
  roleName?: string;

  /**
   * Existing app client (if already connected)
   */
  client?: AppClient;
}

export class FabricationClient {
  private _client: AppClient | null = null;
  private _appInfo: AppInfo | null = null;
  private readonly config: Required<Omit<FabricationConfig, 'client'>> & { client?: AppClient };

  // Sub-clients
  private _designs: DesignsClient | null = null;
  private _printers: PrintersClient | null = null;
  private _prints: PrintsClient | null = null;
  private _materials: MaterialsClient | null = null;
  private _verification: VerificationClient | null = null;
  private _bridge: BridgeClient | null = null;
  private _symthaea: SymthaeaClient | null = null;

  constructor(config: FabricationConfig = {}) {
    this.config = {
      url: config.url || 'ws://localhost:8888',
      installedAppId: config.installedAppId || 'fabrication',
      roleName: config.roleName || 'fabrication',
      client: config.client,
    };
  }

  /**
   * Connect to the Holochain conductor
   */
  async connect(): Promise<void> {
    if (this.config.client) {
      this._client = this.config.client;
    } else {
      const { AppWebsocket } = await import('@holochain/client');
      this._client = await AppWebsocket.connect(this.config.url);
    }

    this._appInfo = await this._client.appInfo();

    // Initialize sub-clients
    this._designs = new DesignsClient(this._client, this.config.roleName, 'designs');
    this._printers = new PrintersClient(this._client, this.config.roleName, 'printers');
    this._prints = new PrintsClient(this._client, this.config.roleName, 'prints');
    this._materials = new MaterialsClient(this._client, this.config.roleName, 'materials');
    this._verification = new VerificationClient(this._client, this.config.roleName, 'verification');
    this._bridge = new BridgeClient(this._client, this.config.roleName, 'bridge');
    this._symthaea = new SymthaeaClient(this._client, this.config.roleName, 'symthaea');
  }

  /**
   * Disconnect from the Holochain conductor
   */
  async disconnect(): Promise<void> {
    if (this._client && 'close' in this._client) {
      await (this._client as AppWebsocket).close();
    }
    this._client = null;
    this._appInfo = null;
    this._designs = null;
    this._printers = null;
    this._prints = null;
    this._materials = null;
    this._verification = null;
    this._bridge = null;
    this._symthaea = null;
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this._client !== null;
  }

  /**
   * Get app info
   */
  get appInfo(): AppInfo | null {
    return this._appInfo;
  }

  /**
   * Get the underlying Holochain client
   */
  get client(): AppClient {
    if (!this._client) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._client;
  }

  // =========================================================================
  // SUB-CLIENT ACCESSORS
  // =========================================================================

  /**
   * Designs client for HDC-enhanced parametric designs
   */
  get designs(): DesignsClient {
    if (!this._designs) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._designs;
  }

  /**
   * Printers client for printer registry and matching
   */
  get printers(): PrintersClient {
    if (!this._printers) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._printers;
  }

  /**
   * Prints client for job management with PoGF scoring
   */
  get prints(): PrintsClient {
    if (!this._prints) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._prints;
  }

  /**
   * Materials client for material specifications
   */
  get materials(): MaterialsClient {
    if (!this._materials) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._materials;
  }

  /**
   * Verification client for safety claims and Knowledge bridge
   */
  get verification(): VerificationClient {
    if (!this._verification) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._verification;
  }

  /**
   * Bridge client for cross-hApp integration
   */
  get bridge(): BridgeClient {
    if (!this._bridge) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._bridge;
  }

  /**
   * Symthaea client for HDC operations and AI assistance
   */
  get symthaea(): SymthaeaClient {
    if (!this._symthaea) {
      throw new Error('Not connected. Call connect() first.');
    }
    return this._symthaea;
  }

  // =========================================================================
  // CONVENIENCE METHODS
  // =========================================================================

  /**
   * Quick search for designs by description
   *
   * Uses Symthaea's semantic search to find designs matching
   * the natural language description.
   */
  async searchDesigns(description: string, limit: number = 10) {
    return this.symthaea.searchByDescription(description, 0.7, limit);
  }

  /**
   * Find local printers that can print a design
   *
   * Combines printer matching with availability checking.
   */
  async findPrintersForDesign(designHash: Uint8Array, location?: { geohash: string; country: string }) {
    const matches = await this.printers.matchDesign(designHash);

    if (location) {
      const nearby = await this.printers.findNearby(location, 100);
      // Filter matches to only include nearby printers
      const nearbyHashes = new Set(nearby.map((r) => r.signed_action.hashed.hash.toString()));
      return matches.filter((m) => nearbyHashes.has(m.printerHash.toString()));
    }

    return matches;
  }

  /**
   * Calculate PoGF score for a print configuration
   *
   * PoGF = (E × 0.3) + (M × 0.3) + (Q × 0.2) + (L × 0.2)
   */
  calculatePogScore(config: {
    energyRenewableFraction: number;
    materialCircularity: number;
    qualityVerified: number;
    localParticipation: number;
  }): number {
    return this.prints.calculatePogScore(
      config.energyRenewableFraction,
      config.materialCircularity,
      config.qualityVerified,
      config.localParticipation
    );
  }

  /**
   * Estimate MYCELIUM reward for a print
   */
  estimateMyceliumReward(pogScore: number, qualityScore: number): number {
    return this.prints.estimateMyceliumReward(pogScore, qualityScore);
  }
}
