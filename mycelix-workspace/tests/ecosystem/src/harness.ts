/**
 * Mycelix Ecosystem Test Harness
 *
 * Provides utilities for testing cross-hApp interactions in the Mycelix ecosystem.
 * Supports multi-agent scenarios with Bridge Protocol validation.
 *
 * Uses Tryorama 0.19+ Scenario API for conductor lifecycle management.
 */

import {
  Scenario,
  runScenario,
} from '@holochain/tryorama';
import type {
  PlayerApp,
  AppWithOptions,
} from '@holochain/tryorama';
import type {
  AgentPubKey,
  AppBundleSource,
} from '@holochain/client';
import { encodeHashToBase64, decodeHashFromBase64 } from '@holochain/client';
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
  mail: {
    searchDirRelPath: 'happs/mail',
    roleName: 'main',
  },
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

  if ('bundleRelPath' in spec && spec.bundleRelPath) {
    return path.resolve(WORKSPACE_ROOT, spec.bundleRelPath);
  }

  if ('searchDirRelPath' in spec && spec.searchDirRelPath) {
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
 * Build an AppBundleSource for a given hApp name.
 */
export function appBundleSource(happName: HappName): AppBundleSource {
  return { type: 'path', value: resolveHappBundlePath(happName) };
}

/**
 * Build AppWithOptions for installing a hApp via Tryorama Scenario.
 */
export function appWithOptions(happName: HappName): AppWithOptions {
  return { appBundleSource: appBundleSource(happName) };
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

export type ReputationScore = AggregatedReputation;

/**
 * Call a zome function on a PlayerApp.
 */
export async function callZome<T = unknown>(
  player: PlayerApp,
  roleName: string,
  zomeName: string,
  fnName: string,
  payload: unknown = null,
): Promise<T> {
  const cell = player.namedCells.get(roleName);
  if (!cell) {
    throw new Error(`Role '${roleName}' not found. Available: ${[...player.namedCells.keys()].join(', ')}`);
  }

  return cell.callZome<T>({
    zome_name: zomeName,
    fn_name: fnName,
    payload,
  });
}

/**
 * Generate a test DID from an agent's public key.
 */
export function generateTestDid(agentPubKey: AgentPubKey): string {
  return `did:mycelix:${encodeHashToBase64(agentPubKey)}`;
}

/**
 * Wait for DHT sync between agents.
 */
export async function waitForSync(timeoutMs: number = 5000): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, timeoutMs));
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

// Re-export Tryorama types for convenience
export { Scenario, runScenario };
export type { PlayerApp, AppWithOptions };
