/**
 * Bridge Zome Client
 *
 * Cross-zome dispatch, cross-cluster communication, governance gating,
 * event broadcasting, and consciousness credential queries for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/bridge
 */

import type {
  DispatchInput,
  DispatchResult,
  ResolveQueryInput,
  EventTypeQuery,
  CrossClusterDispatchInput,
  SeveranceInput,
  HearthSyncInput,
  BridgeHealth,
  GateAuditInput,
  GovernanceAuditFilter,
  GovernanceAuditResult,
  ConsciousnessCredential,
} from './types';
import type { ActionHash, AgentPubKey } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface BridgeClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class BridgeClient {
  private readonly zomeName = 'hearth_bridge';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<BridgeClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Intra-Cluster Dispatch
  // ============================================================================

  async dispatchCall(input: DispatchInput): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'dispatch_call', payload: input });
  }

  async queryHearth(input: DispatchInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'query_hearth', payload: input });
  }

  async resolveQuery(input: ResolveQueryInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'resolve_query', payload: input });
  }

  async broadcastEvent(input: DispatchInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'broadcast_event', payload: input });
  }

  // ============================================================================
  // Cross-Cluster Dispatch
  // ============================================================================

  async dispatchPersonalCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'dispatch_personal_call', payload: input });
  }

  async dispatchIdentityCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'dispatch_identity_call', payload: input });
  }

  async dispatchCommonsCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'dispatch_commons_call', payload: input });
  }

  async dispatchCivicCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'dispatch_civic_call', payload: input });
  }

  // ============================================================================
  // Convenience Cross-Cluster Methods
  // ============================================================================

  async verifyMemberIdentity(agent: AgentPubKey): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'verify_member_identity', payload: agent });
  }

  async escalateEmergency(alertData: string): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'escalate_emergency', payload: alertData });
  }

  async queryTimebankBalance(agent: AgentPubKey): Promise<DispatchResult> {
    return this.client.callZome<DispatchResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'query_timebank_balance', payload: agent });
  }

  // ============================================================================
  // Governance & Consciousness
  // ============================================================================

  async getConsciousnessCredential(did: string): Promise<ConsciousnessCredential> {
    return this.client.callZome<ConsciousnessCredential>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_consciousness_credential', payload: did });
  }

  async logGovernanceGate(input: GateAuditInput): Promise<void> {
    return this.client.callZome<void>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'log_governance_gate', payload: input });
  }

  async getGovernanceAuditTrail(filter: GovernanceAuditFilter): Promise<GovernanceAuditResult> {
    return this.client.callZome<GovernanceAuditResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_governance_audit_trail', payload: filter });
  }

  // ============================================================================
  // Events & Queries
  // ============================================================================

  async getDomainEvents(domain: string) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_domain_events', payload: domain });
  }

  async getAllEvents() {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_all_events', payload: null });
  }

  async getEventsByType(query: EventTypeQuery) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_events_by_type', payload: query });
  }

  async getMyQueries() {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_queries', payload: null });
  }

  // ============================================================================
  // Lifecycle
  // ============================================================================

  async initiateSeverance(input: SeveranceInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'initiate_severance', payload: input });
  }

  async hearthSync(input: HearthSyncInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'hearth_sync', payload: input });
  }

  async getWeeklyDigests(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_weekly_digests', payload: hearthHash });
  }

  async healthCheck(): Promise<BridgeHealth> {
    return this.client.callZome<BridgeHealth>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'health_check', payload: null });
  }
}
