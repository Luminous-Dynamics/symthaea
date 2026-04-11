// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federation Coordinator Zome
//!
//! Manages cross-cell email federation, routing, and bridge protocols.
//! Enables communication between different Holochain networks and external systems.

use hdk::prelude::*;
use mail_federation_integrity::*;

// ==================== ANCHOR HELPERS ====================

/// Create an anchor hash for a network ID
fn network_anchor(network_id: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("federation_network:{}", network_id));
    path.path_entry_hash()
}

/// Create an anchor hash for a domain
fn domain_anchor(domain: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("federation_domain:{}", domain));
    path.path_entry_hash()
}

// ==================== CONSTANTS ====================

/// Maximum hops for federated messages
const MAX_HOPS: u8 = 10;

/// Default TTL for envelopes (24 hours)
const DEFAULT_TTL: u64 = 86400;

/// Maximum queue size per destination
#[allow(dead_code)]
const MAX_QUEUE_PER_DEST: usize = 1000;

/// Retry delay (seconds)
#[allow(dead_code)]
const RETRY_DELAY: u64 = 60;

/// Maximum retries
#[allow(dead_code)]
const MAX_RETRIES: u8 = 5;

// ==================== SIGNALS ====================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum FederationSignal {
    /// New network discovered
    NetworkDiscovered {
        network_id: String,
        network_type: NetworkType,
    },
    /// Envelope received from federated network
    EnvelopeReceived {
        envelope_id: String,
        from_network: String,
    },
    /// Envelope delivered
    EnvelopeDelivered {
        envelope_id: String,
        to_network: String,
    },
    /// Delivery failed
    DeliveryFailed {
        envelope_id: String,
        error: String,
    },
    /// Bridge status changed
    BridgeStatusChanged {
        bridge_agent: AgentPubKey,
        is_active: bool,
    },
    /// Route changed
    RouteChanged {
        route_id: String,
        is_active: bool,
    },
    /// Peer status changed
    PeerStatusChanged {
        network_id: String,
        is_reachable: bool,
    },
}

// ==================== INPUTS/OUTPUTS ====================

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterNetworkInput {
    pub network_id: String,
    pub name: String,
    pub domain: String,
    pub network_type: NetworkType,
    pub capabilities: NetworkCapabilities,
    pub trust_requirements: TrustRequirements,
    pub bootstrap_nodes: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRouteInput {
    pub source_network: String,
    pub source_pattern: String,
    pub dest_network: String,
    pub dest_pattern: String,
    pub priority: u32,
    pub route_type: RouteType,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SendFederatedInput {
    pub dest_network: String,
    pub dest_agent: String,
    pub payload: Vec<u8>,
    pub encryption_scheme: EncryptionScheme,
    pub priority: EnvelopePriority,
    pub ttl: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SendFederatedOutput {
    pub envelope_id: String,
    pub route_used: String,
    pub estimated_delivery: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterDomainInput {
    pub domain: String,
    pub network_id: String,
    pub dns_verification: Option<String>,
    pub mail_routing: Vec<MailRouting>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterBridgeInput {
    pub bridge_type: NetworkType,
    pub connected_networks: Vec<String>,
    pub endpoint: Option<String>,
    pub capacity: u32,
}

// ==================== NETWORK MANAGEMENT ====================

/// Register a new federated network
#[hdk_extern]
pub fn register_network(input: RegisterNetworkInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let dna_info = dna_info()?;
    let now = sys_time()?;

    // Check for duplicate network ID
    let existing = get_network(input.network_id.clone())?;
    if existing.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network with ID '{}' already exists", input.network_id)
        )));
    }

    let network = FederatedNetwork {
        network_id: input.network_id.clone(),
        dna_hash: dna_info.hash,
        name: input.name,
        domain: input.domain,
        network_type: input.network_type.clone(),
        network_pubkey: vec![], // Would generate network key
        bootstrap_nodes: input.bootstrap_nodes,
        registered_at: now,
        is_active: true,
        capabilities: input.capabilities,
        trust_requirements: input.trust_requirements,
    };

    let hash = create_entry(EntryTypes::FederatedNetwork(network.clone()))?;

    // Link by network ID for lookup
    let network_id_anchor = network_anchor(&input.network_id)?;
    create_link(
        network_id_anchor,
        hash.clone(),
        LinkTypes::NetworkIdToNetwork,
        (),
    )?;

    // Link to owning agent
    create_link(
        agent.clone(),
        hash.clone(),
        LinkTypes::AgentToOwnedNetworks,
        (),
    )?;

    // Audit log
    log_federation_action(FederationAction::NetworkRegistered, &input.network_id, None)?;

    emit_signal(FederationSignal::NetworkDiscovered {
        network_id: input.network_id,
        network_type: input.network_type,
    })?;

    Ok(hash)
}

/// Get network by ID
#[hdk_extern]
pub fn get_network(network_id: String) -> ExternResult<Option<FederatedNetwork>> {
    let network_id_anchor = network_anchor(&network_id)?;

    let links = get_links(LinkQuery::try_new(network_id_anchor, LinkTypes::NetworkIdToNetwork)?, GetStrategy::default())?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let network: FederatedNetwork = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid network"))?;
                return Ok(Some(network));
            }
        }
    }

    Ok(None)
}

/// Deactivate a network
#[hdk_extern]
pub fn deactivate_network(network_id: String) -> ExternResult<()> {
    // In full implementation, would update network entry
    log_federation_action(FederationAction::NetworkDeregistered, &network_id, None)?;
    Ok(())
}

// ==================== ROUTING ====================

/// Check if the calling agent owns a given network
fn is_network_owner(network_id: &str) -> ExternResult<bool> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToOwnedNetworks)?,
        GetStrategy::default(),
    )?;
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(network) = record
                    .entry()
                    .to_app_option::<FederatedNetwork>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if network.network_id == network_id {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

/// Create a federation route
#[hdk_extern]
pub fn create_route(input: CreateRouteInput) -> ExternResult<ActionHash> {
    // Verify caller owns the source network
    if !is_network_owner(&input.source_network)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the network owner can create routes for a network".to_string()
        )));
    }

    let now = sys_time()?;

    let route = FederationRoute {
        route_id: format!(
            "route:{}->{}:{}",
            input.source_network,
            input.dest_network,
            now.as_micros()
        ),
        source_network: input.source_network.clone(),
        source_pattern: input.source_pattern,
        dest_network: input.dest_network.clone(),
        dest_pattern: input.dest_pattern,
        priority: input.priority,
        route_type: input.route_type,
        is_active: true,
        created_at: now,
        last_used: None,
        success_count: 0,
        failure_count: 0,
    };

    let hash = create_entry(EntryTypes::FederationRoute(route.clone()))?;

    // Link to source network
    let source_hash = network_anchor(&input.source_network)?;
    create_link(source_hash, hash.clone(), LinkTypes::NetworkToRoutes, ())?;

    log_federation_action(FederationAction::RouteCreated, &input.source_network, Some(&input.dest_network))?;

    emit_signal(FederationSignal::RouteChanged {
        route_id: route.route_id,
        is_active: true,
    })?;

    Ok(hash)
}

/// Get routes for a network
#[hdk_extern]
pub fn get_routes(network_id: String) -> ExternResult<Vec<FederationRoute>> {
    let network_hash = network_anchor(&network_id)?;

    let links = get_links(LinkQuery::try_new(network_hash, LinkTypes::NetworkToRoutes)?, GetStrategy::default())?;

    let mut routes = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let route: FederationRoute = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid route"))?;
                if route.is_active {
                    routes.push(route);
                }
            }
        }
    }

    // Sort by priority (descending)
    routes.sort_by(|a, b| b.priority.cmp(&a.priority));

    Ok(routes)
}

/// Find best route for destination
fn find_best_route(
    source_network: &str,
    dest_network: &str,
    dest_pattern: &str,
) -> ExternResult<Option<FederationRoute>> {
    let routes = get_routes(source_network.to_string())?;

    for route in routes {
        if route.dest_network == dest_network && route.is_active {
            // Pattern matching would be more sophisticated in real implementation
            if route.dest_pattern == "*" || route.dest_pattern == dest_pattern {
                return Ok(Some(route));
            }
        }
    }

    Ok(None)
}

// ==================== MESSAGE SENDING ====================

/// Send a federated message
#[hdk_extern]
pub fn send_federated(input: SendFederatedInput) -> ExternResult<SendFederatedOutput> {
    let agent = agent_info()?.agent_initial_pubkey;
    let _dna_info = dna_info()?;
    let now = sys_time()?;

    // Get our network ID
    let our_network = get_our_network_id()?;

    // Find route to destination
    let route = find_best_route(&our_network, &input.dest_network, &input.dest_agent)?
        .ok_or(wasm_error!("No route to destination network"))?;

    // Generate envelope ID
    let envelope_id = format!("env:{}:{}:{}", our_network, input.dest_network, now.as_micros());

    // Create encryption metadata
    let encryption_meta = EnvelopeEncryption {
        scheme: input.encryption_scheme,
        ephemeral_pubkey: vec![], // Would generate ephemeral key
        key_exchange: vec![],     // Would perform key exchange
        nonce: vec![0u8; 24],     // Would generate random nonce
    };

    // Sign envelope
    let signature = sign_envelope(&envelope_id, &input.payload)?;

    let envelope = FederatedEnvelope {
        envelope_id: envelope_id.clone(),
        source_network: our_network.clone(),
        source_agent: agent.to_string(),
        dest_network: input.dest_network.clone(),
        dest_agent: input.dest_agent,
        encrypted_payload: input.payload,
        encryption_meta,
        signature,
        timestamp: now,
        ttl: input.ttl.unwrap_or(DEFAULT_TTL),
        hop_count: 0,
        max_hops: MAX_HOPS,
        previous_hops: vec![],
        priority: input.priority,
        status: DeliveryStatus::Queued,
    };

    let _hash = create_entry(EntryTypes::FederatedEnvelope(envelope.clone()))?;

    // Attempt delivery based on route type
    let delivery_result = attempt_delivery(&envelope, &route)?;

    log_federation_action(
        FederationAction::EnvelopeSent,
        &our_network,
        Some(&input.dest_network),
    )?;

    emit_signal(FederationSignal::EnvelopeDelivered {
        envelope_id: envelope_id.clone(),
        to_network: input.dest_network,
    })?;

    Ok(SendFederatedOutput {
        envelope_id,
        route_used: route.route_id,
        estimated_delivery: Some(delivery_result.estimated_ms),
    })
}

/// Sign envelope contents
fn sign_envelope(envelope_id: &str, payload: &[u8]) -> ExternResult<Vec<u8>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let mut data = envelope_id.as_bytes().to_vec();
    data.extend_from_slice(payload);
    let signed = sign(agent, data)?;
    Ok(signed.0.to_vec())
}

/// Attempt to deliver envelope via route
fn attempt_delivery(
    envelope: &FederatedEnvelope,
    route: &FederationRoute,
) -> ExternResult<DeliveryResult> {
    match &route.route_type {
        RouteType::DirectHolochain => {
            // Direct P2P delivery via Holochain remote signal
            deliver_via_holochain(envelope)
        }
        RouteType::BridgeAgent { bridge_agent } => {
            // Send to bridge agent for relay
            deliver_via_bridge(envelope, bridge_agent)
        }
        RouteType::RelayServer { relay_url } => {
            // Send to relay server (would use HTTP)
            deliver_via_relay(envelope, relay_url)
        }
        RouteType::Gateway { gateway_id } => {
            // Send to gateway service
            deliver_via_gateway(envelope, gateway_id)
        }
        RouteType::Multipath { routes } => {
            // Try multiple routes in parallel
            deliver_via_multipath(envelope, routes)
        }
    }
}

#[derive(Debug)]
struct DeliveryResult {
    #[allow(dead_code)]
    success: bool,
    estimated_ms: u64,
}

fn deliver_via_holochain(envelope: &FederatedEnvelope) -> ExternResult<DeliveryResult> {
    // In full implementation, would use call_remote or remote_signal
    // For now, log the action
    log_federation_action(
        FederationAction::EnvelopeSent,
        &envelope.source_network,
        Some(&envelope.dest_network),
    )?;

    Ok(DeliveryResult {
        success: true,
        estimated_ms: 1000,
    })
}

fn deliver_via_bridge(
    envelope: &FederatedEnvelope,
    bridge_agent: &AgentPubKey,
) -> ExternResult<DeliveryResult> {
    // Send to bridge agent via remote signal
    let signal = ExternIO::encode(envelope).map_err(|e| wasm_error!(e))?;
    send_remote_signal(signal, vec![bridge_agent.clone()])?;

    Ok(DeliveryResult {
        success: true,
        estimated_ms: 5000,
    })
}

fn deliver_via_relay(
    _envelope: &FederatedEnvelope,
    _relay_url: &str,
) -> ExternResult<DeliveryResult> {
    // Would use HTTP to send to relay server
    // This requires host capabilities not available in wasm
    Ok(DeliveryResult {
        success: true,
        estimated_ms: 10000,
    })
}

fn deliver_via_gateway(
    _envelope: &FederatedEnvelope,
    _gateway_id: &str,
) -> ExternResult<DeliveryResult> {
    // Would look up gateway and send
    Ok(DeliveryResult {
        success: true,
        estimated_ms: 15000,
    })
}

fn deliver_via_multipath(
    envelope: &FederatedEnvelope,
    _route_ids: &[String],
) -> ExternResult<DeliveryResult> {
    // Try direct first, then fallback
    deliver_via_holochain(envelope)
}

/// Get our network ID
fn get_our_network_id() -> ExternResult<String> {
    let dna_info = dna_info()?;
    // Use DNA hash as network ID
    Ok(dna_info.hash.to_string())
}

// ==================== MESSAGE RECEIVING ====================

/// Handle incoming federated envelope
#[hdk_extern]
pub fn receive_federated_envelope(envelope: FederatedEnvelope) -> ExternResult<bool> {
    let our_network = get_our_network_id()?;

    // Check if this is for us or needs relay
    if envelope.dest_network == our_network {
        // Deliver locally
        deliver_locally(&envelope)?;

        emit_signal(FederationSignal::EnvelopeReceived {
            envelope_id: envelope.envelope_id.clone(),
            from_network: envelope.source_network.clone(),
        })?;

        log_federation_action(
            FederationAction::EnvelopeReceived,
            &envelope.source_network,
            Some(&our_network),
        )?;

        Ok(true)
    } else {
        // Need to relay
        relay_envelope(envelope)?;
        Ok(true)
    }
}

/// Deliver envelope locally
fn deliver_locally(envelope: &FederatedEnvelope) -> ExternResult<()> {
    // Store envelope for recipient
    create_entry(EntryTypes::FederatedEnvelope(envelope.clone()))?;

    // In full implementation, would:
    // 1. Decrypt payload
    // 2. Verify signature
    // 3. Create local email entry
    // 4. Notify recipient

    log_federation_action(
        FederationAction::EnvelopeDelivered,
        &envelope.source_network,
        Some(&envelope.dest_network),
    )?;

    Ok(())
}

/// Relay envelope to next hop
fn relay_envelope(mut envelope: FederatedEnvelope) -> ExternResult<()> {
    // Check hop count
    if envelope.hop_count >= envelope.max_hops {
        log_federation_action(
            FederationAction::DeliveryFailed,
            &envelope.source_network,
            Some(&envelope.dest_network),
        )?;
        return Err(wasm_error!("Maximum hops exceeded"));
    }

    // Add ourselves to hop list
    let our_network = get_our_network_id()?;
    envelope.previous_hops.push(our_network.clone());
    envelope.hop_count += 1;
    envelope.status = DeliveryStatus::InTransit {
        current_hop: our_network.clone(),
    };

    // Find route to destination
    let route = find_best_route(&our_network, &envelope.dest_network, &envelope.dest_agent)?
        .ok_or(wasm_error!("No route to destination"))?;

    // Forward
    attempt_delivery(&envelope, &route)?;

    log_federation_action(
        FederationAction::EnvelopeRelayed,
        &our_network,
        Some(&envelope.dest_network),
    )?;

    Ok(())
}

// ==================== BRIDGE MANAGEMENT ====================

/// Register as a bridge agent
#[hdk_extern]
pub fn register_bridge(input: RegisterBridgeInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let bridge = BridgeAgent {
        agent: agent.clone(),
        bridge_type: input.bridge_type,
        connected_networks: input.connected_networks.clone(),
        endpoint: input.endpoint,
        registered_at: now,
        last_heartbeat: Some(now),
        is_active: true,
        capacity: input.capacity,
        current_load: 0,
        trust_score: 1.0,
    };

    let hash = create_entry(EntryTypes::BridgeAgent(bridge))?;

    create_link(
        agent.clone(),
        hash.clone(),
        LinkTypes::AgentToBridgeRegistrations,
        (),
    )?;

    // Link to each network
    for network_id in input.connected_networks {
        let network_hash = network_anchor(&network_id)?;
        create_link(network_hash, hash.clone(), LinkTypes::NetworkToBridges, ())?;
    }

    log_federation_action(FederationAction::BridgeConnected, "local", None)?;

    emit_signal(FederationSignal::BridgeStatusChanged {
        bridge_agent: agent,
        is_active: true,
    })?;

    Ok(hash)
}

/// Get bridges for a network
#[hdk_extern]
pub fn get_bridges(network_id: String) -> ExternResult<Vec<BridgeAgent>> {
    let network_hash = network_anchor(&network_id)?;

    let links = get_links(LinkQuery::try_new(network_hash, LinkTypes::NetworkToBridges)?, GetStrategy::default())?;

    let mut bridges = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let bridge: BridgeAgent = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid bridge"))?;
                if bridge.is_active {
                    bridges.push(bridge);
                }
            }
        }
    }

    // Sort by trust score and load
    bridges.sort_by(|a, b| {
        let a_score = a.trust_score * (1.0 - a.current_load as f64 / a.capacity as f64);
        let b_score = b.trust_score * (1.0 - b.current_load as f64 / b.capacity as f64);
        b_score.partial_cmp(&a_score).unwrap()
    });

    Ok(bridges)
}

/// Bridge heartbeat (keeps bridge active)
#[hdk_extern]
pub fn bridge_heartbeat(_: ()) -> ExternResult<()> {
    // Would update last_heartbeat on bridge entry
    Ok(())
}

// ==================== DOMAIN REGISTRATION ====================

/// Register a domain for federation
#[hdk_extern]
pub fn register_domain(input: RegisterDomainInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let domain = DomainRegistration {
        domain: input.domain.clone(),
        network_id: input.network_id,
        dns_verification: input.dns_verification,
        admin_agent: agent,
        registered_at: now,
        expires_at: None,
        is_verified: false, // Would verify DNS record
        mail_routing: input.mail_routing,
    };

    let hash = create_entry(EntryTypes::DomainRegistration(domain))?;

    // Link by domain for lookup
    let domain_hash = domain_anchor(&input.domain)?;
    create_link(domain_hash, hash.clone(), LinkTypes::DomainToRegistration, ())?;

    log_federation_action(FederationAction::DomainRegistered, &input.domain, None)?;

    Ok(hash)
}

/// Lookup domain registration
#[hdk_extern]
pub fn lookup_domain(domain: String) -> ExternResult<Option<DomainRegistration>> {
    let domain_hash = domain_anchor(&domain)?;

    let links = get_links(LinkQuery::try_new(domain_hash, LinkTypes::DomainToRegistration)?, GetStrategy::default())?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let registration: DomainRegistration = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid domain"))?;
                return Ok(Some(registration));
            }
        }
    }

    Ok(None)
}

/// Verify domain DNS record
#[hdk_extern]
pub fn verify_domain(domain: String) -> ExternResult<bool> {
    // In full implementation, would check DNS TXT record
    // For now, just log
    log_federation_action(FederationAction::DomainVerified, &domain, None)?;
    Ok(true)
}

// ==================== PEER MANAGEMENT ====================

#[derive(Serialize, Deserialize, Debug)]
pub struct DiscoverPeerInput {
    pub network_id: String,
    pub dna_hash: DnaHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BlockPeerInput {
    pub network_id: String,
    pub reason: String,
}

/// Discover federation peer
#[hdk_extern]
pub fn discover_peer(input: DiscoverPeerInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let peer = FederationPeer {
        network_id: input.network_id.clone(),
        dna_hash: input.dna_hash,
        discovered_via: DiscoveryMethod::Manual,
        last_seen: now,
        latency_ms: 0,
        messages_exchanged: 0,
        trust_level: PeerTrustLevel::Unknown,
        is_blocked: false,
        block_reason: None,
    };

    let hash = create_entry(EntryTypes::FederationPeer(peer))?;

    let network_hash = network_anchor(&input.network_id)?;
    create_link(network_hash, hash.clone(), LinkTypes::NetworkToPeers, ())?;

    log_federation_action(FederationAction::PeerDiscovered, &input.network_id, None)?;

    emit_signal(FederationSignal::PeerStatusChanged {
        network_id: input.network_id,
        is_reachable: true,
    })?;

    Ok(hash)
}

/// Block a peer
#[hdk_extern]
pub fn block_peer(input: BlockPeerInput) -> ExternResult<()> {
    log_federation_action(FederationAction::PeerBlocked, &input.network_id, None)?;

    emit_signal(FederationSignal::PeerStatusChanged {
        network_id: input.network_id,
        is_reachable: false,
    })?;

    Ok(())
}

// ==================== AUDIT LOGGING ====================

fn log_federation_action(
    action: FederationAction,
    source_network: &str,
    dest_network: Option<&str>,
) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let log = FederationAuditLog {
        log_id: format!("log:{}:{}", source_network, now.as_micros()),
        action,
        source_network: source_network.to_string(),
        dest_network: dest_network.map(|s| s.to_string()),
        envelope_id: None,
        actor: Some(agent),
        timestamp: now,
        success: true,
        details: None,
    };

    create_entry(EntryTypes::FederationAuditLog(log))
}

/// Get audit logs for network
#[hdk_extern]
pub fn get_federation_audit_logs(network_id: String) -> ExternResult<Vec<FederationAuditLog>> {
    let network_hash = network_anchor(&network_id)?;

    let links = get_links(LinkQuery::try_new(network_hash, LinkTypes::NetworkToAuditLogs)?, GetStrategy::default())?;

    let mut logs = Vec::new();

    for link in links.into_iter().take(100) {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let log: FederationAuditLog = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid log"))?;
                logs.push(log);
            }
        }
    }

    Ok(logs)
}

// ==================== REMOTE SIGNAL HANDLING ====================

/// Handle incoming remote signal (for federation)
#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    // Try to decode as federated envelope
    if let Ok(envelope) = signal.decode::<FederatedEnvelope>() {
        receive_federated_envelope(envelope)?;
    }

    Ok(())
}
