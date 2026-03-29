// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Delegation Client
 *
 * Liquid democracy with domain-specific delegation and cycle detection.
 */

import { type AgentPubKey, type Record as HolochainRecord } from '@holochain/client';

import type { DAOClient } from './client';
import type {
  Delegation,
  DelegationDomain,
  DelegatedPowerResult,
} from './types';

/**
 * Maximum delegation chain depth (prevents infinite loops)
 */
export const MAX_DELEGATION_DEPTH = 5;

/**
 * Maximum power concentration (prevents plutocracy)
 */
export const MAX_POWER_CONCENTRATION = 0.15; // 15% of total voting power

/**
 * Delegation graph node for visualization
 */
export interface DelegationNode {
  agent: AgentPubKey;
  displayName: string;
  directPower: number;
  delegatedPower: number;
  totalPower: number;
  delegatesTo: AgentPubKey[];
  delegatesFrom: AgentPubKey[];
}

/**
 * Client for delegation operations
 */
export class DelegationClient {
  constructor(private client: DAOClient) {}

  private get zome() {
    return this.client.getZomeName('governance');
  }

  // =========================================================================
  // DELEGATION MANAGEMENT
  // =========================================================================

  /**
   * Delegate voting power to another member
   *
   * @example
   * ```typescript
   * // Delegate all voting power
   * await delegation.delegate(expertPubKey, 'All');
   *
   * // Delegate only technical decisions
   * await delegation.delegate(techExpertPubKey, 'Technical');
   * ```
   */
  async delegate(
    delegate: AgentPubKey,
    domain: DelegationDomain = 'All'
  ): Promise<Delegation> {
    // Pre-check for cycles
    const wouldCycle = await this.checkForCycle(delegate, domain);
    if (wouldCycle) {
      throw new Error('Delegation would create a cycle');
    }

    // Check power concentration
    const newPower = await this.calculatePotentialPower(delegate, domain);
    if (newPower.concentration > MAX_POWER_CONCENTRATION) {
      throw new Error(
        `Delegation would exceed power concentration limit of ${MAX_POWER_CONCENTRATION * 100}%`
      );
    }

    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'create_delegation',
      { delegate, domain }
    );

    return this.decodeDelegation(record);
  }

  /**
   * Revoke a delegation
   */
  async revoke(delegate: AgentPubKey, domain?: DelegationDomain): Promise<void> {
    await this.client.callZome(
      this.zome,
      'revoke_delegation',
      { delegate, domain }
    );
  }

  /**
   * Revoke all delegations
   */
  async revokeAll(): Promise<void> {
    await this.client.callZome(
      this.zome,
      'revoke_all_delegations',
      null
    );
  }

  /**
   * Get active delegations made by the current user
   */
  async getMyDelegations(): Promise<Delegation[]> {
    const myPubKey = await this.client.getMyPubKey();
    return this.getDelegationsFrom(myPubKey);
  }

  /**
   * Get delegations from a specific member
   */
  async getDelegationsFrom(agent: AgentPubKey): Promise<Delegation[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_delegations_from',
      agent
    );

    return records.map(r => this.decodeDelegation(r));
  }

  /**
   * Get delegations to a specific member
   */
  async getDelegationsTo(agent: AgentPubKey): Promise<Delegation[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_delegations_to',
      agent
    );

    return records.map(r => this.decodeDelegation(r));
  }

  // =========================================================================
  // POWER CALCULATION
  // =========================================================================

  /**
   * Calculate total voting power for an agent
   *
   * @example
   * ```typescript
   * const power = await delegation.getVotingPower();
   * console.log(`Direct: ${power.direct_power}`);
   * console.log(`Delegated: ${power.transitive_power}`);
   * console.log(`Total: ${power.total_power}`);
   * console.log(`From ${power.delegator_count} delegators`);
   * ```
   */
  async getVotingPower(
    agent?: AgentPubKey,
    domain?: DelegationDomain
  ): Promise<DelegatedPowerResult> {
    const pubKey = agent || await this.client.getMyPubKey();
    return this.client.callZome<DelegatedPowerResult>(
      this.zome,
      'get_delegated_power',
      { agent: pubKey, domain: domain || 'All' }
    );
  }

  /**
   * Get the delegation chain from one agent to another
   */
  async getDelegationChain(
    from: AgentPubKey,
    domain: DelegationDomain = 'All'
  ): Promise<AgentPubKey[]> {
    return this.client.callZome<AgentPubKey[]>(
      this.zome,
      'get_delegation_chain',
      { from, domain }
    );
  }

  /**
   * Check if there's a delegation path between two agents
   */
  async isDelegatingTo(
    from: AgentPubKey,
    to: AgentPubKey,
    domain: DelegationDomain = 'All'
  ): Promise<boolean> {
    const chain = await this.getDelegationChain(from, domain);
    return chain.some(a => a.toString() === to.toString());
  }

  // =========================================================================
  // GRAPH ANALYSIS
  // =========================================================================

  /**
   * Get the delegation graph for visualization
   */
  async getDelegationGraph(domain: DelegationDomain = 'All'): Promise<{
    nodes: DelegationNode[];
    edges: Array<{ from: AgentPubKey; to: AgentPubKey; domain: DelegationDomain }>;
  }> {
    return this.client.callZome(
      this.zome,
      'get_delegation_graph',
      domain
    );
  }

  /**
   * Get power distribution statistics
   */
  async getPowerDistribution(): Promise<{
    gini_coefficient: number;
    top_10_percent_power: number;
    median_power: number;
    power_holders: number;
    total_power: number;
  }> {
    return this.client.callZome(
      this.zome,
      'get_power_distribution',
      null
    );
  }

  /**
   * Identify potential power concentration issues
   */
  async detectConcentrationRisks(): Promise<Array<{
    agent: AgentPubKey;
    power_percentage: number;
    delegator_count: number;
    warning: string;
  }>> {
    return this.client.callZome(
      this.zome,
      'detect_concentration_risks',
      null
    );
  }

  // =========================================================================
  // CYCLE DETECTION
  // =========================================================================

  /**
   * Check if a delegation would create a cycle
   */
  async checkForCycle(
    proposedDelegate: AgentPubKey,
    domain: DelegationDomain = 'All'
  ): Promise<boolean> {
    const myPubKey = await this.client.getMyPubKey();

    // Check if the proposed delegate (or anyone in their chain) delegates back to us
    const chain = await this.getDelegationChain(proposedDelegate, domain);
    return chain.some(a => a.toString() === myPubKey.toString());
  }

  /**
   * Get depth of delegation for an agent
   */
  async getDelegationDepth(
    agent: AgentPubKey,
    domain: DelegationDomain = 'All'
  ): Promise<number> {
    const chain = await this.getDelegationChain(agent, domain);
    return chain.length;
  }

  // =========================================================================
  // ADVANCED FEATURES
  // =========================================================================

  /**
   * Calculate what power would be if a delegation was created
   */
  async calculatePotentialPower(
    proposedDelegate: AgentPubKey,
    domain: DelegationDomain
  ): Promise<{
    new_power: DelegatedPowerResult;
    concentration: number;
  }> {
    const currentPower = await this.getVotingPower(proposedDelegate, domain);
    const myPower = await this.getVotingPower(undefined, domain);

    const newTotalPower = currentPower.total_power + myPower.direct_power;

    // Get total system power for concentration calculation
    const distribution = await this.getPowerDistribution();
    const concentration = newTotalPower / distribution.total_power;

    return {
      new_power: {
        direct_power: currentPower.direct_power,
        transitive_power: currentPower.transitive_power + myPower.direct_power,
        total_power: newTotalPower,
        delegator_count: currentPower.delegator_count + 1,
      },
      concentration,
    };
  }

  /**
   * Find the most trusted delegate for a domain
   */
  async suggestDelegates(
    domain: DelegationDomain,
    limit: number = 5
  ): Promise<Array<{
    agent: AgentPubKey;
    display_name: string;
    reputation: number;
    current_delegators: number;
    expertise_match: number;
  }>> {
    return this.client.callZome(
      this.zome,
      'suggest_delegates',
      { domain, limit }
    );
  }

  /**
   * Get delegation history for audit
   */
  async getDelegationHistory(
    agent?: AgentPubKey
  ): Promise<Array<{
    delegation: Delegation;
    event: 'Created' | 'Revoked';
    timestamp: number;
  }>> {
    const pubKey = agent || await this.client.getMyPubKey();
    return this.client.callZome(
      this.zome,
      'get_delegation_history',
      pubKey
    );
  }

  // =========================================================================
  // DOMAIN HELPERS
  // =========================================================================

  /**
   * Get all available delegation domains
   */
  getDomains(): DelegationDomain[] {
    return ['All', 'Governance', 'Technical', 'Economic', 'Social', 'Cultural'];
  }

  /**
   * Check if a domain is a subset of another
   */
  isDomainSubset(subset: DelegationDomain, superset: DelegationDomain): boolean {
    if (superset === 'All') return true;
    return subset === superset;
  }

  /**
   * Get human-readable domain description
   */
  describeDomain(domain: DelegationDomain): string {
    const descriptions: Record<DelegationDomain, string> = {
      All: 'All governance decisions',
      Governance: 'Process and rule changes',
      Technical: 'Protocol and implementation decisions',
      Economic: 'Treasury and financial decisions',
      Social: 'Community and membership decisions',
      Cultural: 'Values and mission decisions',
    };
    return descriptions[domain];
  }

  private decodeDelegation(record: HolochainRecord): Delegation {
    return (record as any).entry.Present.entry as Delegation;
  }
}
