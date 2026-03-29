// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Attribution Registry — TypeScript SDK Client
 *
 * Typed bindings for the attribution hApp's three zome pairs:
 * - registry: DependencyIdentity CRUD
 * - usage: UsageReceipt + UsageAttestation (ZK-STARK)
 * - reciprocity: ReciprocityPledge + StewardshipScore
 */

// ── Types ────────────────────────────────────────────────────────────

export type DependencyEcosystem =
  | 'RustCrate'
  | 'NpmPackage'
  | 'PythonPackage'
  | 'NixFlake'
  | 'GoModule'
  | 'RubyGem'
  | 'MavenPackage'
  | 'Other';

export interface DependencyIdentity {
  id: string;
  name: string;
  ecosystem: DependencyEcosystem;
  maintainer_did: string;
  repository_url: string | null;
  license: string | null;
  description: string;
  version: string | null;
  registered_at: number; // Holochain Timestamp (microseconds)
  verified: boolean;
}

export interface UpdateDependencyInput {
  original_action_hash: Uint8Array;
  dependency: DependencyIdentity;
}

export type UsageType =
  | 'DirectDependency'
  | 'Transitive'
  | 'InternalTooling'
  | 'Production'
  | 'Research';

export type UsageScale = 'Small' | 'Medium' | 'Large' | 'Enterprise';

export interface UsageReceipt {
  id: string;
  dependency_id: string;
  user_did: string;
  organization: string | null;
  usage_type: UsageType;
  scale: UsageScale | null;
  version_range: string | null;
  context: string | null;
  attested_at: number;
}

export interface UsageAttestation {
  id: string;
  dependency_id: string;
  user_did: string;
  witness_commitment: Uint8Array;
  proof_bytes: Uint8Array;
  verified: boolean;
  generated_at: number;
  expires_at: number | null;
  verifier_pubkey: Uint8Array | null;
  verifier_signature: Uint8Array | null;
}

export interface VerifyAttestationInput {
  original_action_hash: Uint8Array;
  verifier_pubkey: Uint8Array;
  verifier_signature: Uint8Array;
}

export type PledgeType =
  | 'Financial'
  | 'Compute'
  | 'Bandwidth'
  | 'DeveloperTime'
  | 'QA'
  | 'Documentation'
  | 'Other';

export type Currency =
  | 'USD'
  | 'EUR'
  | 'GBP'
  | 'BTC'
  | 'ETH'
  | { Other: string };

export interface ReciprocityPledge {
  id: string;
  dependency_id: string;
  contributor_did: string;
  organization: string | null;
  pledge_type: PledgeType;
  amount: number | null;
  currency: Currency | null;
  description: string;
  evidence_url: string | null;
  period: string | null;
  pledged_at: number;
  acknowledged: boolean;
}

export interface StewardshipScore {
  dependency_id: string;
  usage_count: number;
  pledge_count: number;
  ratio: number;
  weighted_score: number;
  pledge_type_counts: [string, number][];
}

// ── Pagination Types ──────────────────────────────────────────────────

export interface PaginationInput {
  offset: number;
  limit: number;
}

export interface PaginatedResult<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

export interface TopDependency {
  dependency_id: string;
  usage_count: number;
}

export interface BulkRegisterResult {
  registered: Record<string, unknown>[];
  skipped: string[];
}

export interface BulkUsageResult {
  recorded: number;
  records: Record<string, unknown>[];
}

export interface LeaderboardEntry {
  dependency_id: string;
  usage_count: number;
  pledge_count: number;
  weighted_score: number;
}

export interface EcosystemStat {
  ecosystem: string;
  count: number;
}

export interface EcosystemStatistics {
  total_dependencies: number;
  ecosystems: EcosystemStat[];
  verified_count: number;
}

// ── Signals ──────────────────────────────────────────────────────────

export interface RegistrySignal {
  type: 'DependencyRegistered' | 'DependencyUpdated' | 'DependencyVerified';
  payload: {
    dependency_id: string;
    name?: string;
    ecosystem?: string;
  };
}

export interface RenewAttestationInput {
  original_action_hash: Uint8Array;
  new_proof_bytes: Uint8Array;
  new_witness_commitment: Uint8Array;
}

export interface UsageSignal {
  type: 'UsageRecorded' | 'AttestationSubmitted' | 'AttestationVerified' | 'AttestationRevoked' | 'AttestationRenewed';
  payload: {
    dependency_id: string;
    user_did?: string;
    attestation_id?: string;
  };
}

export interface ReciprocitySignal {
  type: 'PledgeRecorded' | 'PledgeAcknowledged';
  payload: {
    dependency_id: string;
    contributor_did?: string;
    pledge_type?: string;
    pledge_id?: string;
  };
}

export type AttributionSignal = RegistrySignal | UsageSignal | ReciprocitySignal;

export function isAttributionSignal(signal: unknown): signal is AttributionSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'type' in signal &&
    'payload' in signal
  );
}

// ── Constants ────────────────────────────────────────────────────────

const ATTRIBUTION_ROLE = 'attribution';

// ── Client Interface ─────────────────────────────────────────────────

interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ── Attribution Client ───────────────────────────────────────────────

export class AttributionClient {
  constructor(private client: ZomeCallable) {}

  // ── Registry ─────────────────────────────────────────────────────

  async registerDependency(dep: DependencyIdentity): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'register_dependency',
      payload: dep,
    }) as Promise<Record<string, unknown>>;
  }

  async updateDependency(input: UpdateDependencyInput): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'update_dependency',
      payload: input,
    }) as Promise<Record<string, unknown>>;
  }

  async getDependency(id: string): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'get_dependency',
      payload: id,
    }) as Promise<Record<string, unknown> | null>;
  }

  async getAllDependencies(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'get_all_dependencies',
      payload: null,
    }) as Promise<Record<string, unknown>[]>;
  }

  async getAllDependenciesPaginated(
    pagination: PaginationInput,
  ): Promise<PaginatedResult<Record<string, unknown>>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'get_all_dependencies_paginated',
      payload: pagination,
    }) as Promise<PaginatedResult<Record<string, unknown>>>;
  }

  async bulkRegisterDependencies(
    deps: DependencyIdentity[],
  ): Promise<BulkRegisterResult> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'bulk_register_dependencies',
      payload: deps,
    }) as Promise<BulkRegisterResult>;
  }

  async getDependenciesByEcosystem(ecosystem: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'get_dependencies_by_ecosystem',
      payload: ecosystem,
    }) as Promise<Record<string, unknown>[]>;
  }

  async getMaintainerDependencies(did: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'get_maintainer_dependencies',
      payload: did,
    }) as Promise<Record<string, unknown>[]>;
  }

  async verifyDependency(id: string): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'verify_dependency',
      payload: id,
    }) as Promise<Record<string, unknown>>;
  }

  async getEcosystemStatistics(): Promise<EcosystemStatistics> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'registry',
      fn_name: 'get_ecosystem_statistics',
      payload: null,
    }) as Promise<EcosystemStatistics>;
  }

  // ── Usage ────────────────────────────────────────────────────────

  async recordUsage(receipt: UsageReceipt): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'record_usage',
      payload: receipt,
    }) as Promise<Record<string, unknown>>;
  }

  async bulkRecordUsage(receipts: UsageReceipt[]): Promise<BulkUsageResult> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'bulk_record_usage',
      payload: receipts,
    }) as Promise<BulkUsageResult>;
  }

  async getDependencyUsage(depId: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_dependency_usage',
      payload: depId,
    }) as Promise<Record<string, unknown>[]>;
  }

  async getDependencyUsagePaginated(
    depId: string,
    pagination: PaginationInput,
  ): Promise<PaginatedResult<Record<string, unknown>>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_dependency_usage_paginated',
      payload: { id: depId, pagination },
    }) as Promise<PaginatedResult<Record<string, unknown>>>;
  }

  async getTopDependencies(limit: number): Promise<TopDependency[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_top_dependencies',
      payload: limit,
    }) as Promise<TopDependency[]>;
  }

  async getUserUsage(did: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_user_usage',
      payload: did,
    }) as Promise<Record<string, unknown>[]>;
  }

  async getUsageCount(depId: string): Promise<number> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_usage_count',
      payload: depId,
    }) as Promise<number>;
  }

  async getAttestationCount(): Promise<number> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_attestation_count',
      payload: null,
    }) as Promise<number>;
  }

  async submitUsageAttestation(att: UsageAttestation): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'submit_usage_attestation',
      payload: att,
    }) as Promise<Record<string, unknown>>;
  }

  async verifyUsageAttestation(input: VerifyAttestationInput): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'verify_usage_attestation',
      payload: input,
    }) as Promise<Record<string, unknown>>;
  }

  async getDependencyAttestations(depId: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'get_dependency_attestations',
      payload: depId,
    }) as Promise<Record<string, unknown>[]>;
  }

  async revokeAttestation(actionHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'revoke_attestation',
      payload: actionHash,
    }) as Promise<Record<string, unknown>>;
  }

  async renewAttestation(input: RenewAttestationInput): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'usage',
      fn_name: 'renew_attestation',
      payload: input,
    }) as Promise<Record<string, unknown>>;
  }

  // ── Reciprocity ──────────────────────────────────────────────────

  async recordPledge(pledge: ReciprocityPledge): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'record_pledge',
      payload: pledge,
    }) as Promise<Record<string, unknown>>;
  }

  async acknowledgePledge(id: string): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'acknowledge_pledge',
      payload: id,
    }) as Promise<Record<string, unknown>>;
  }

  async getDependencyPledges(depId: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'get_dependency_pledges',
      payload: depId,
    }) as Promise<Record<string, unknown>[]>;
  }

  async getDependencyPledgesPaginated(
    depId: string,
    pagination: PaginationInput,
  ): Promise<PaginatedResult<Record<string, unknown>>> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'get_dependency_pledges_paginated',
      payload: { id: depId, pagination },
    }) as Promise<PaginatedResult<Record<string, unknown>>>;
  }

  async getContributorPledges(did: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'get_contributor_pledges',
      payload: did,
    }) as Promise<Record<string, unknown>[]>;
  }

  async computeStewardshipScore(depId: string): Promise<StewardshipScore> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'compute_stewardship_score',
      payload: depId,
    }) as Promise<StewardshipScore>;
  }

  async getStewardshipLeaderboard(limit: number): Promise<LeaderboardEntry[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'get_stewardship_leaderboard',
      payload: limit,
    }) as Promise<LeaderboardEntry[]>;
  }

  async getPledgeCount(): Promise<number> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'get_pledge_count',
      payload: null,
    }) as Promise<number>;
  }

  async getUnderSupportedDependencies(limit: number): Promise<LeaderboardEntry[]> {
    return this.client.callZome({
      role_name: ATTRIBUTION_ROLE,
      zome_name: 'reciprocity',
      fn_name: 'get_under_supported_dependencies',
      payload: limit,
    }) as Promise<LeaderboardEntry[]>;
  }
}

// ── Singleton Factory ────────────────────────────────────────────────

let instance: AttributionClient | null = null;

export function createAttributionClient(client: ZomeCallable): AttributionClient {
  if (!instance) instance = new AttributionClient(client);
  return instance;
}

export function resetAttributionClient(): void {
  instance = null;
}
