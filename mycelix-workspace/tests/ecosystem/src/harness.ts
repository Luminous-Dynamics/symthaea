/**
 * Mycelix Ecosystem Test Harness
 *
 * Provides utilities for testing cross-hApp interactions in the Mycelix ecosystem.
 * Supports multi-agent scenarios with Bridge Protocol validation.
 */

import {
  Conductor,
  createConductor,
  cleanAllConductors,
  AppBundleSource,
  AppWebsocket,
  AgentPubKey,
} from '@holochain/tryorama';
import { decodeHashFromBase64, encodeHashToBase64 } from '@holochain/client';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const WORKSPACE_ROOT = path.resolve(__dirname, '..', '..', '..');

type HappSpec = {
  bundleRelPath?: string;
  searchDirRelPath?: string;
  roleName?: string;
};

// hApp bundle specs (relative to workspace root)
const HAPP_SPECS = {
  // NOTE: This is currently the 0TML ZeroTrustML Credits hApp (role: `credits`),
  // not a canonical “Mycelix-Core” bundle.
  core: {
    bundleRelPath: 'happs/core/0TML/zerotrustml-dna/zerotrustml.happ',
    roleName: 'credits',
  },
  identity: {
    bundleRelPath: 'happs/identity/mycelix-identity.happ',
    roleName: 'identity',
  },
  knowledge: {
    bundleRelPath: 'happs/knowledge/mycelix-knowledge.happ',
    roleName: 'knowledge',
  },
  governance: {
    bundleRelPath: 'happs/governance/mycelix-governance.happ',
    roleName: 'governance',
  },
  marketplace: {
    bundleRelPath: 'happs/marketplace/backend/mycelix_marketplace.happ',
    roleName: 'marketplace',
  },
  edunet: {
    bundleRelPath: 'happs/edunet/happ/mycelix-edunet.happ',
    roleName: 'edunet',
  },
  // Not currently built into a `.happ` bundle in this workspace; keep as “discoverable”.
  mail: {
    searchDirRelPath: 'happs/mail',
    roleName: 'main',
  },
  // Not currently packaged into a `.happ` in this workspace; keep as “discoverable”.
  supplychain: {
    searchDirRelPath: 'happs/supplychain',
  },
  epistemicMarkets: {
    bundleRelPath: 'happs/epistemic-markets/epistemic-markets.happ',
    roleName: 'epistemic_markets',
  },
  fabrication: {
    bundleRelPath: 'happs/fabrication/fabrication.happ',
    roleName: 'fabrication',
  },
  finance: {
    bundleRelPath: 'happs/finance/mycelix-finance.happ',
    roleName: 'finance',
  },
  // Cluster DNAs: commons (2 roles: commons_land + commons_care), civic, hearth
  commons: {
    searchDirRelPath: 'happs/commons',
    roleName: 'commons_land',
  },
  civic: {
    searchDirRelPath: 'happs/civic',
    roleName: 'civic',
  },
  hearth: {
    searchDirRelPath: 'happs/hearth',
    roleName: 'hearth',
  },
  health: {
    searchDirRelPath: 'happs/health',
    roleName: 'health',
  },
  // Not currently packaged into `.happ` bundles in this workspace; keep as "discoverable".
  energy: {
    searchDirRelPath: 'happs/energy',
  },
} as const satisfies Record<string, HappSpec>;

export type HappName = keyof typeof HAPP_SPECS;

function findFirstHappFile(rootDir: string, maxDepth: number = 6): string | null {
  function walk(dir: string, depth: number): string | null {
    if (depth > maxDepth) return null;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return null;
    }

    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith('.happ')) return path.join(dir, entry.name);
    }

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (entry.name === '.git' || entry.name === 'node_modules' || entry.name === 'target') continue;
      const found = walk(path.join(dir, entry.name), depth + 1);
      if (found) return found;
    }

    return null;
  }

  return walk(rootDir, 0);
}

export function resolveHappBundlePath(happName: HappName): string {
  const spec = HAPP_SPECS[happName];

  if (spec.bundleRelPath) {
    return path.resolve(WORKSPACE_ROOT, spec.bundleRelPath);
  }

  if (spec.searchDirRelPath) {
    const searchRoot = path.resolve(WORKSPACE_ROOT, spec.searchDirRelPath);
    const found = findFirstHappFile(searchRoot);
    if (found) return found;
    return path.resolve(searchRoot, `${happName}.happ`);
  }

  return path.resolve(WORKSPACE_ROOT, 'happs', happName, `${happName}.happ`);
}

export function happBundleExists(happName: HappName): boolean {
  return fs.existsSync(resolveHappBundlePath(happName));
}

/**
 * Configuration for a test scenario
 */
export interface ScenarioConfig {
  /** Number of agents in the scenario */
  agentCount: number;
  /** hApps to install for each agent */
  happs: HappName[];
  /** Optional: specific hApps per agent (overrides happs) */
  agentHapps?: Record<number, HappName[]>;
}

/**
 * A test agent with installed hApps
 */
export interface TestAgent {
  /** Agent index in the scenario */
  index: number;
  /** Agent public key */
  pubKey: AgentPubKey;
  /** Installed hApp cells, keyed by hApp name */
  cells: Record<HappName, {
    cellId: [Uint8Array, Uint8Array];
    client: AppWebsocket;
  }>;
}

/**
 * Result from a cross-hApp bridge call
 */
export interface BridgeCallResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  latencyMs: number;
}

export interface ReputationSource {
  source_happ: string;
  score: number;
  interactions: number;
}

export interface AggregatedReputation {
  did: string;
  aggregate_score: number;
  sources: ReputationSource[];
  total_interactions: number;
}

// Backwards-compat alias (older tests used `ReputationScore`).
export type ReputationScore = AggregatedReputation;

/**
 * Ecosystem Test Harness
 *
 * Manages multi-hApp test scenarios with the Mycelix ecosystem.
 */
export class EcosystemTestHarness {
  private conductors: Conductor[] = [];
  private agents: TestAgent[] = [];
  private config: ScenarioConfig;

  constructor(config: ScenarioConfig) {
    this.config = config;
  }

  /**
   * Initialize the test scenario
   */
  async setup(): Promise<void> {
    // Clean up any existing conductors
    await cleanAllConductors();

    // Create a conductor per agent (each conductor has a distinct agent key)
    for (let i = 0; i < this.config.agentCount; i++) {
      const conductor = await createConductor();
      this.conductors.push(conductor);
      const happsToInstall = this.config.agentHapps?.[i] ?? this.config.happs;
      const agent = await this.createAgent(conductor, i, happsToInstall);
      this.agents.push(agent);
    }
  }

  /**
   * Create a test agent with specified hApps
   */
  private async createAgent(conductor: Conductor, index: number, happs: HappName[]): Promise<TestAgent> {
    const firstHapp = happs[0];
    if (!firstHapp) throw new Error('Scenario has no happs configured');

    const agent: TestAgent = {
      index,
      pubKey: new Uint8Array(39), // Will be set after first install
      cells: {} as TestAgent['cells'],
    };

    for (const happName of happs) {
      const happPath = resolveHappBundlePath(happName);
      if (!fs.existsSync(happPath)) {
        throw new Error(
          `hApp bundle for '${happName}' not found at '${happPath}'. ` +
          `Run 'just build-${happName}' (or build the relevant bundle) from '${WORKSPACE_ROOT}'.`,
        );
      }

      const appBundleSource: AppBundleSource = {
        path: happPath,
      };

      // Install the hApp
      const installedAppId = `${happName}-agent-${index}`;
      const appInfo = await conductor.installApp(appBundleSource, {
        installedAppId,
      });

      const preferredRole = HAPP_SPECS[happName]?.roleName;

      const candidates: Array<unknown> = [];
      if (preferredRole && (appInfo as any).cell_info?.[preferredRole]) {
        candidates.push(...(appInfo as any).cell_info[preferredRole]);
      } else if ((appInfo as any).cell_info) {
        for (const cells of Object.values((appInfo as any).cell_info)) {
          if (Array.isArray(cells)) candidates.push(...cells);
        }
      }

      const provisioned = candidates.find((c: any) => c && typeof c === 'object' && 'provisioned' in c) as any;
      if (!provisioned?.provisioned?.cell_id) {
        const roles = Object.keys((appInfo as any).cell_info ?? {});
        throw new Error(
          `Failed to provision cell for '${happName}'. ` +
          `Expected role '${preferredRole ?? '(first role)'}', available roles: ${roles.join(', ') || '(none)'}`,
        );
      }
      const cellId = provisioned.provisioned.cell_id as [Uint8Array, Uint8Array];

      // Create client
      const client = await conductor.connectAppClient(installedAppId);

      agent.cells[happName] = {
        cellId,
        client,
      };

      // Set agent pubKey from first cell
      if (!agent.pubKey.length || agent.pubKey.every(b => b === 0) || happName === firstHapp) {
        agent.pubKey = cellId[1];
      }
    }

    return agent;
  }

  /**
   * Tear down the test scenario
   */
  async teardown(): Promise<void> {
    for (const conductor of this.conductors) {
      await conductor.shutDown();
    }
    this.conductors = [];
    this.agents = [];
    await cleanAllConductors();
  }

  /**
   * Get an agent by index
   */
  getAgent(index: number): TestAgent {
    const agent = this.agents[index];
    if (!agent) {
      throw new Error(`Agent ${index} not found`);
    }
    return agent;
  }

  /**
   * Get all agents
   */
  getAllAgents(): TestAgent[] {
    return [...this.agents];
  }

  /**
   * Call a zome function on an agent's hApp
   */
  async callZome<T = unknown>(
    agentIndex: number,
    happName: HappName,
    zomeName: string,
    fnName: string,
    payload: unknown = null,
  ): Promise<T> {
    const agent = this.getAgent(agentIndex);
    const cell = agent.cells[happName];

    if (!cell) {
      throw new Error(`Agent ${agentIndex} does not have ${happName} installed`);
    }

    const result = await cell.client.callZome({
      cell_id: cell.cellId,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    });

    return result as T;
  }

  /**
   * Cross-hApp routing (canonical router not yet implemented in zomes).
   *
   * Prefer calling explicit bridge zomes directly, e.g.:
   * - Identity: `identity_bridge/query_identity`, `identity_bridge/report_reputation`, `identity_bridge/get_reputation`
   * - Knowledge: `knowledge_bridge/query_knowledge`
   * - Governance: `governance_bridge/*`
   */
  async bridgeCall<T = unknown>(
    sourceAgentIndex: number,
    sourceHapp: HappName,
    targetHapp: HappName,
    queryType: string,
    payload: unknown = null,
  ): Promise<BridgeCallResult<T>> {
    const startTime = Date.now();

    try {
      return {
        success: false,
        error:
          `bridgeCall not implemented (no canonical cross-hApp router). ` +
          `source=${sourceHapp} target=${targetHapp} query=${queryType} payload=${payload ? 'provided' : 'null'}`,
        latencyMs: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        latencyMs: Date.now() - startTime,
      };
    }
  }

  /**
   * Query aggregated reputation via Identity Bridge (`identity_bridge/get_reputation`).
   */
  async queryAggregateReputation(
    agentIndex: number,
    targetAgentPubKey: AgentPubKey,
  ): Promise<AggregatedReputation | null> {
    const agent = this.getAgent(agentIndex);

    if (!agent.cells.identity) {
      throw new Error('Agent does not have identity hApp installed (required for identity_bridge reputation)');
    }

    try {
      const did = `did:mycelix:${encodeHashToBase64(targetAgentPubKey)}`;
      const result = await this.callZome<AggregatedReputation>(
        agentIndex,
        'identity',
        'identity_bridge',
        'get_reputation',
        did,
      );
      return result;
    } catch {
      return null;
    }
  }

  /**
   * Wait for DHT sync between agents
   */
  async waitForSync(timeoutMs: number = 5000): Promise<void> {
    // Simple wait - in production would use signals
    await new Promise(resolve => setTimeout(resolve, timeoutMs));
  }

  /**
   * Generate test DID for an agent
   */
  generateTestDid(agentIndex: number): string {
    const agent = this.getAgent(agentIndex);
    return `did:mycelix:${encodeHashToBase64(agent.pubKey)}`;
  }
}

/**
 * Create a simple scenario with the core integration hApps
 */
export function createCoreScenario(agentCount: number = 2): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'knowledge', 'governance'],
  });
}

/**
 * Create a marketplace scenario
 */
export function createMarketplaceScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'marketplace', 'epistemicMarkets'],
  });
}

/**
 * Create an education scenario
 */
export function createEducationScenario(agentCount: number = 4): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'edunet', 'knowledge'],
  });
}

/**
 * Create a full ecosystem scenario (all happs)
 */
export function createFullEcosystemScenario(agentCount: number = 2): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: [
      'identity',
      'knowledge',
      'governance',
      'marketplace',
      'edunet',
      'epistemicMarkets',
    ],
  });
}

/**
 * Create a supply chain scenario
 */
export function createSupplyChainScenario(agentCount: number = 4): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'marketplace', 'supplychain'],
  });
}

/**
 * Create a mail scenario
 */
export function createMailScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'mail'],
  });
}

/**
 * Create a governance-focused scenario
 */
export function createGovernanceScenario(agentCount: number = 5): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'governance', 'knowledge'],
  });
}

export function createFinanceScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'finance'],
  });
}

export function createJusticeScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'justice'],
  });
}

export function createEnergyScenario(agentCount: number = 4): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'energy', 'governance'],
  });
}

export function createPropertyScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'property'],
  });
}

export function createCommonsScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'commons'],
  });
}

export function createCivicScenario(agentCount: number = 3): EcosystemTestHarness {
  return new EcosystemTestHarness({
    agentCount,
    happs: ['identity', 'civic'],
  });
}

/**
 * Helper to encode agent pub key to string
 */
export function encodeAgentPubKey(pubKey: AgentPubKey): string {
  return encodeHashToBase64(pubKey);
}

/**
 * Helper to decode agent pub key from string
 */
export function decodeAgentPubKey(encoded: string): AgentPubKey {
  return decodeHashFromBase64(encoded) as AgentPubKey;
}
