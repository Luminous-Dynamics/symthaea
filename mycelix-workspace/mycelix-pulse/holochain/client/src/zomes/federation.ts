// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Federation Zome Client
 * Cross-cell email routing and bridge management
 */

import type { AppClient, AgentPubKey, DnaHash } from '@holochain/client';
import type {
  FederatedNetwork,
  NetworkType,
  NetworkCapabilities,
  FederatedEnvelope,
  DeliveryStatus,
  SendFederatedInput,
} from '../types';

export class FederationZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'mycelix_mail',
    private zomeName: string = 'mail_federation'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ==================== NETWORK MANAGEMENT ====================

  /**
   * Register a new federated network
   */
  async registerNetwork(input: {
    network_id: string;
    name: string;
    domain: string;
    network_type: NetworkType;
    capabilities?: NetworkCapabilities;
    bootstrap_nodes?: string[];
  }): Promise<void> {
    await this.callZome('register_network', {
      network_id: input.network_id,
      name: input.name,
      domain: input.domain,
      network_type: input.network_type,
      capabilities: input.capabilities ?? {
        supports_e2e: true,
        supports_pqc: true,
        supports_attachments: true,
        max_attachment_size: 25 * 1024 * 1024, // 25MB
        supports_read_receipts: true,
        supports_threading: true,
        rate_limit_per_hour: null,
      },
      trust_requirements: {
        min_trust_score: 0.3,
        require_admin_attestation: false,
        require_member_attestations: null,
        required_categories: [],
        auto_approve_known_contacts: true,
        quarantine_period: null,
      },
      bootstrap_nodes: input.bootstrap_nodes ?? [],
    });
  }

  /**
   * Get network by ID
   */
  async getNetwork(networkId: string): Promise<FederatedNetwork | null> {
    return this.callZome('get_network', networkId);
  }

  /**
   * Get all known networks
   */
  async getAllNetworks(): Promise<FederatedNetwork[]> {
    return this.callZome('get_all_networks', null);
  }

  /**
   * Deactivate a network
   */
  async deactivateNetwork(networkId: string): Promise<void> {
    await this.callZome('deactivate_network', networkId);
  }

  // ==================== ROUTING ====================

  /**
   * Create a federation route
   */
  async createRoute(input: {
    source_network: string;
    source_pattern: string;
    dest_network: string;
    dest_pattern: string;
    priority?: number;
    route_type?: 'direct' | 'bridge' | 'relay';
  }): Promise<string> {
    return this.callZome('create_route', {
      source_network: input.source_network,
      source_pattern: input.source_pattern,
      dest_network: input.dest_network,
      dest_pattern: input.dest_pattern,
      priority: input.priority ?? 100,
      route_type: input.route_type === 'bridge'
        ? { BridgeAgent: { bridge_agent: null } }
        : input.route_type === 'relay'
        ? { RelayServer: { relay_url: '' } }
        : 'DirectHolochain',
    });
  }

  /**
   * Get routes for a network
   */
  async getRoutes(networkId: string): Promise<Array<{
    route_id: string;
    source_pattern: string;
    dest_network: string;
    dest_pattern: string;
    priority: number;
    is_active: boolean;
    success_rate: number;
  }>> {
    return this.callZome('get_routes', networkId);
  }

  /**
   * Update route priority
   */
  async updateRoutePriority(routeId: string, priority: number): Promise<void> {
    await this.callZome('update_route_priority', { route_id: routeId, priority });
  }

  /**
   * Deactivate route
   */
  async deactivateRoute(routeId: string): Promise<void> {
    await this.callZome('deactivate_route', routeId);
  }

  // ==================== FEDERATED MESSAGING ====================

  /**
   * Send a federated message
   */
  async sendFederated(input: SendFederatedInput): Promise<{
    envelope_id: string;
    route_used: string;
    estimated_delivery: number | null;
  }> {
    return this.callZome('send_federated', {
      dest_network: input.dest_network,
      dest_agent: input.dest_agent,
      payload: input.payload,
      encryption_scheme: 'HybridX25519Kyber',
      priority: input.priority ?? 'Normal',
      ttl: input.ttl ?? 86400,
    });
  }

  /**
   * Get envelope status
   */
  async getEnvelopeStatus(envelopeId: string): Promise<{
    envelope_id: string;
    status: DeliveryStatus;
    hop_count: number;
    last_update: number;
  } | null> {
    return this.callZome('get_envelope_status', envelopeId);
  }

  /**
   * Get pending outbound envelopes
   */
  async getPendingOutbound(): Promise<FederatedEnvelope[]> {
    return this.callZome('get_pending_outbound', null);
  }

  /**
   * Get inbound envelopes
   */
  async getInboundEnvelopes(limit?: number): Promise<FederatedEnvelope[]> {
    return this.callZome('get_inbound_envelopes', { limit: limit ?? 50 });
  }

  /**
   * Retry failed envelope
   */
  async retryEnvelope(envelopeId: string): Promise<void> {
    await this.callZome('retry_envelope', envelopeId);
  }

  // ==================== BRIDGE MANAGEMENT ====================

  /**
   * Register as a bridge agent
   */
  async registerBridge(input: {
    bridge_type: NetworkType;
    connected_networks: string[];
    endpoint?: string;
    capacity?: number;
  }): Promise<void> {
    await this.callZome('register_bridge', {
      bridge_type: input.bridge_type,
      connected_networks: input.connected_networks,
      endpoint: input.endpoint ?? null,
      capacity: input.capacity ?? 1000,
    });
  }

  /**
   * Get bridges for a network
   */
  async getBridges(networkId: string): Promise<Array<{
    agent: AgentPubKey;
    bridge_type: NetworkType;
    is_active: boolean;
    capacity: number;
    current_load: number;
    trust_score: number;
  }>> {
    return this.callZome('get_bridges', networkId);
  }

  /**
   * Send bridge heartbeat (keeps bridge active)
   */
  async bridgeHeartbeat(): Promise<void> {
    await this.callZome('bridge_heartbeat', null);
  }

  /**
   * Deactivate bridge
   */
  async deactivateBridge(): Promise<void> {
    await this.callZome('deactivate_bridge', null);
  }

  // ==================== DOMAIN MANAGEMENT ====================

  /**
   * Register a domain
   */
  async registerDomain(input: {
    domain: string;
    network_id: string;
    dns_verification?: string;
  }): Promise<void> {
    await this.callZome('register_domain', {
      domain: input.domain,
      network_id: input.network_id,
      dns_verification: input.dns_verification ?? null,
      mail_routing: [{ priority: 10, target_network: input.network_id, target_agent: null, weight: 100 }],
    });
  }

  /**
   * Lookup domain
   */
  async lookupDomain(domain: string): Promise<{
    domain: string;
    network_id: string;
    is_verified: boolean;
    mail_routing: Array<{ priority: number; target_network: string }>;
  } | null> {
    return this.callZome('lookup_domain', domain);
  }

  /**
   * Verify domain DNS record
   */
  async verifyDomain(domain: string): Promise<boolean> {
    return this.callZome('verify_domain', domain);
  }

  // ==================== PEER DISCOVERY ====================

  /**
   * Discover a peer network
   */
  async discoverPeer(networkId: string, dnaHash: DnaHash): Promise<void> {
    await this.callZome('discover_peer', { network_id: networkId, dna_hash: dnaHash });
  }

  /**
   * Get known peers
   */
  async getKnownPeers(): Promise<Array<{
    network_id: string;
    last_seen: number;
    latency_ms: number;
    trust_level: string;
    is_blocked: boolean;
  }>> {
    return this.callZome('get_known_peers', null);
  }

  /**
   * Block a peer
   */
  async blockPeer(networkId: string, reason: string): Promise<void> {
    await this.callZome('block_peer', { network_id: networkId, reason });
  }

  /**
   * Unblock a peer
   */
  async unblockPeer(networkId: string): Promise<void> {
    await this.callZome('unblock_peer', networkId);
  }

  // ==================== AUDIT LOGS ====================

  /**
   * Get federation audit logs
   */
  async getAuditLogs(networkId: string, limit?: number): Promise<Array<{
    log_id: string;
    action: string;
    source_network: string;
    dest_network: string | null;
    timestamp: number;
    success: boolean;
  }>> {
    return this.callZome('get_federation_audit_logs', { network_id: networkId, limit: limit ?? 100 });
  }
}
