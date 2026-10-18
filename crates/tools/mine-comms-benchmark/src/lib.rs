// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic routing and receipt semantics for MINE-COMMS-001.
//!
//! This is a software contract harness, not a physical radio/propagation model.
//! Link quality, latency, energy and observation age are explicit supplied data.

use std::collections::{BTreeMap, BTreeSet};

const STATIC_LOCAL_RADIO_PENALTY: u64 = 10_000;
const STATIC_TTE_PENALTY: u64 = 1_000_000;
const ADAPTIVE_ENERGY_WEIGHT: u64 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelKind {
    Backbone,
    LocalRadio,
    ThroughEarth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkState {
    Up,
    Down,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkHealth {
    Healthy,
    Down,
    Unknown,
    Stale,
    ZeroQuality,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Link {
    pub id: String,
    pub from: String,
    pub to: String,
    pub channel: ChannelKind,
    pub state: LinkState,
    /// Quality estimate on a 0..=1000 scale. This is supplied evidence, not a
    /// measured packet-delivery rate unless the scenario says so.
    pub quality_permille: u16,
    pub observed_at_ms: u64,
    pub max_observation_age_ms: u64,
    pub one_way_latency_ms: u64,
    pub energy_millijoules: u64,
    /// Complete set of known shared-cause domains capable of disabling this
    /// link (e.g. cable route, power feed, shaft or inundation zone).
    pub failure_domains: Vec<String>,
    /// True only when the scenario author asserts that the domain list is
    /// complete for this link. Diversity claims require this evidence.
    pub failure_domains_complete: bool,
}

impl Link {
    pub fn health(&self, now_ms: u64) -> LinkHealth {
        match self.state {
            LinkState::Down => LinkHealth::Down,
            LinkState::Unknown => LinkHealth::Unknown,
            LinkState::Up if now_ms.saturating_sub(self.observed_at_ms) > self.max_observation_age_ms => {
                LinkHealth::Stale
            }
            LinkState::Up if self.quality_permille == 0 => LinkHealth::ZeroQuality,
            LinkState::Up => LinkHealth::Healthy,
        }
    }

    fn validate(&self, now_ms: u64) -> Result<(), &'static str> {
        if self.id.trim().is_empty() {
            return Err("link id must not be empty");
        }
        if self.from.trim().is_empty() || self.to.trim().is_empty() {
            return Err("link endpoints must not be empty");
        }
        if self.from == self.to {
            return Err("self-loop links are not accepted by this harness");
        }
        if self.quality_permille > 1000 {
            return Err("link quality must be within 0..=1000 permille");
        }
        if self.observed_at_ms > now_ms {
            return Err("link observation timestamp is in the future");
        }
        let mut domains = BTreeSet::new();
        for domain in &self.failure_domains {
            if domain.trim().is_empty() {
                return Err("failure domain identifiers must not be empty");
            }
            if !domains.insert(domain.as_str()) {
                return Err("failure domain identifiers must be unique per link");
            }
        }
        if self.failure_domains_complete && self.failure_domains.is_empty() {
            return Err("complete failure-domain coverage requires at least one domain");
        }
        Ok(())
    }

    fn route_cost(&self, policy: RoutePolicy) -> u64 {
        match policy {
            RoutePolicy::FixedChannelPreference => {
                let penalty = match self.channel {
                    ChannelKind::Backbone => 0,
                    ChannelKind::LocalRadio => STATIC_LOCAL_RADIO_PENALTY,
                    ChannelKind::ThroughEarth => STATIC_TTE_PENALTY,
                };
                penalty
                    .saturating_add(1_000)
                    .saturating_add(self.one_way_latency_ms)
            }
            RoutePolicy::AdaptiveLatencyEnergy => {
                let q = u64::from(self.quality_permille.max(1));
                self.one_way_latency_ms
                    .saturating_mul(1000)
                    .checked_div(q)
                    .unwrap_or(u64::MAX)
                    .saturating_add(self.energy_millijoules.saturating_mul(ADAPTIVE_ENERGY_WEIGHT))
                    .max(1)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutePolicy {
    /// Prefer the conventional channel hierarchy, then lower latency.
    FixedChannelPreference,
    /// Minimize a declared score using link quality, latency and energy.
    /// This is a benchmark policy, not a learned or safety-certified controller.
    AdaptiveLatencyEnergy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Route {
    pub nodes: Vec<String>,
    pub link_ids: Vec<String>,
    pub link_latencies_ms: Vec<u64>,
    pub latency_ms: u64,
    pub energy_millijoules: u64,
    pub bottleneck_quality_permille: u16,
    pub score: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageClass {
    Distress,
    Evacuation,
    PersonnelStatus,
    Voice,
    Telemetry,
    Bulk,
}

impl MessageClass {
    /// Lower rank means a more urgent message. Used by the harness/reporting;
    /// a production scheduler still needs its own queue and starvation tests.
    pub fn priority_rank(self) -> u8 {
        match self {
            Self::Distress => 0,
            Self::Evacuation => 1,
            Self::PersonnelStatus => 2,
            Self::Voice => 3,
            Self::Telemetry => 4,
            Self::Bulk => 5,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub id: String,
    pub class: MessageClass,
    pub source: String,
    pub destination: String,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    /// Authentication state is reported separately from transport delivery.
    /// This boolean represents an already-verified input fact; it implements
    /// no cryptographic scheme.
    pub origin_authenticated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageEventKind {
    Created,
    Queued,
    Transmitted,
    Relayed,
    Received,
    OriginAuthenticated,
    OriginUnverified,
    Acknowledged,
    AcknowledgementUnavailable,
    NoRoute,
    Expired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageEvent {
    pub at_ms: u64,
    pub kind: MessageEventKind,
    pub node: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReceiptDisposition {
    Acknowledged,
    AcknowledgedUnverifiedOrigin,
    ReceivedUnacknowledged,
    NoRoute,
    ExpiredBeforeDelivery,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageReceipt {
    pub message_id: String,
    pub class: MessageClass,
    pub disposition: ReceiptDisposition,
    pub forward_route: Option<Route>,
    pub acknowledgement_route: Option<Route>,
    pub delivered_at_ms: Option<u64>,
    pub acknowledged_at_ms: Option<u64>,
    pub total_energy_millijoules: u64,
    pub events: Vec<MessageEvent>,
}

/// Find the minimum-cost usable directed route. Invalid graph data is an error;
/// down, unknown, zero-quality and stale links are excluded from pathfinding.
pub fn find_route(
    links: &[Link],
    source: &str,
    destination: &str,
    now_ms: u64,
    policy: RoutePolicy,
) -> Result<Option<Route>, &'static str> {
    if source.trim().is_empty() || destination.trim().is_empty() {
        return Err("route endpoints must not be empty");
    }

    let mut link_ids = BTreeSet::new();
    let mut nodes = BTreeSet::new();
    nodes.insert(source.to_owned());
    nodes.insert(destination.to_owned());

    for link in links {
        link.validate(now_ms)?;
        if !link_ids.insert(link.id.as_str()) {
            return Err("link ids must be unique");
        }
        nodes.insert(link.from.clone());
        nodes.insert(link.to.clone());
    }

    if source == destination {
        return Ok(Some(Route {
            nodes: vec![source.to_owned()],
            link_ids: Vec::new(),
            link_latencies_ms: Vec::new(),
            latency_ms: 0,
            energy_millijoules: 0,
            bottleneck_quality_permille: 1000,
            score: 0,
        }));
    }

    let mut distance: BTreeMap<String, u64> = BTreeMap::new();
    let mut previous: BTreeMap<String, (String, usize)> = BTreeMap::new();
    let mut visited = BTreeSet::new();
    distance.insert(source.to_owned(), 0);

    // Sorted graph nodes and link identifiers make tie-breaking reproducible.
    let mut ordered_links: Vec<usize> = (0..links.len()).collect();
    ordered_links.sort_by(|a, b| links[*a].id.cmp(&links[*b].id));

    for _ in 0..nodes.len() {
        let mut best: Option<(String, u64)> = None;
        for candidate in &nodes {
            if visited.contains(candidate) {
                continue;
            }
            let Some(candidate_cost) = distance.get(candidate).copied() else {
                continue;
            };
            let replace = match &best {
                None => true,
                Some((best_node, best_cost)) => {
                    candidate_cost < *best_cost
                        || (candidate_cost == *best_cost && candidate < best_node)
                }
            };
            if replace {
                best = Some((candidate.clone(), candidate_cost));
            }
        }

        let Some((current, current_cost)) = best else {
            break;
        };
        if current == destination {
            break;
        }
        visited.insert(current.clone());

        for index in &ordered_links {
            let link = &links[*index];
            if link.from != current || link.health(now_ms) != LinkHealth::Healthy {
                continue;
            }
            if visited.contains(&link.to) {
                continue;
            }

            let candidate_cost = current_cost.saturating_add(link.route_cost(policy));
            let improves = match distance.get(&link.to) {
                None => true,
                Some(old_cost) if candidate_cost < *old_cost => true,
                // Equal scores keep the first path discovered from sorted
                // nodes and sorted link IDs, making ties deterministic.
                _ => false,
            };
            if improves {
                distance.insert(link.to.clone(), candidate_cost);
                previous.insert(link.to.clone(), (current.clone(), *index));
            }
        }
    }

    let Some(score) = distance.get(destination).copied() else {
        return Ok(None);
    };

    let mut reverse_path = Vec::new();
    let mut cursor = destination.to_owned();
    while cursor != source {
        let Some((previous_node, link_index)) = previous.get(&cursor).cloned() else {
            return Err("route reconstruction failed for a reachable destination");
        };
        reverse_path.push(link_index);
        cursor = previous_node;
    }
    reverse_path.reverse();

    let mut route_nodes = vec![source.to_owned()];
    let mut route_link_ids = Vec::with_capacity(reverse_path.len());
    let mut route_latencies = Vec::with_capacity(reverse_path.len());
    let mut latency_ms = 0u64;
    let mut energy_millijoules = 0u64;
    let mut bottleneck_quality_permille = 1000u16;

    for index in reverse_path {
        let link = &links[index];
        route_nodes.push(link.to.clone());
        route_link_ids.push(link.id.clone());
        route_latencies.push(link.one_way_latency_ms);
        latency_ms = latency_ms.saturating_add(link.one_way_latency_ms);
        energy_millijoules = energy_millijoules.saturating_add(link.energy_millijoules);
        bottleneck_quality_permille = bottleneck_quality_permille.min(link.quality_permille);
    }

    Ok(Some(Route {
        nodes: route_nodes,
        link_ids: route_link_ids,
        link_latencies_ms: route_latencies,
        latency_ms,
        energy_millijoules,
        bottleneck_quality_permille,
        score,
    }))
}

/// A pair of routes with disjoint declared failure domains.
/// Their shared source/destination are expected endpoints; internal relay-node
/// domains are included automatically in the disjointness check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FailureDomainDiverseRoutes {
    pub primary: Route,
    pub secondary: Route,
    pub combined_score: u64,
}

const MAX_DIVERSITY_CANDIDATE_PATHS: usize = 4096;
const MAX_DIVERSITY_HOPS: usize = 64;

/// Find two simple source-to-destination routes that do not share any declared
/// link failure domain or internal relay node.
///
/// A route is eligible only when every link explicitly marks its failure-domain
/// coverage complete. Missing domain knowledge cannot be interpreted as
/// independence. This bounded exhaustive search is intended for small
/// benchmark topologies, not operational network routing.
pub fn find_failure_domain_diverse_routes(
    links: &[Link],
    source: &str,
    destination: &str,
    now_ms: u64,
    policy: RoutePolicy,
    max_hops: usize,
    max_candidate_paths: usize,
) -> Result<Option<FailureDomainDiverseRoutes>, &'static str> {
    if source.trim().is_empty() || destination.trim().is_empty() {
        return Err("route endpoints must not be empty");
    }
    if source == destination {
        return Ok(None);
    }
    if max_hops == 0 || max_hops > MAX_DIVERSITY_HOPS {
        return Err("max_hops must be within 1..=64");
    }
    if max_candidate_paths == 0 || max_candidate_paths > MAX_DIVERSITY_CANDIDATE_PATHS {
        return Err("max_candidate_paths must be within 1..=4096");
    }

    let mut link_ids = BTreeSet::new();
    let mut node_names = BTreeSet::new();
    node_names.insert(source.to_owned());
    node_names.insert(destination.to_owned());
    for link in links {
        link.validate(now_ms)?;
        if !link_ids.insert(link.id.as_str()) {
            return Err("link ids must be unique");
        }
        node_names.insert(link.from.clone());
        node_names.insert(link.to.clone());
    }

    fn visit_paths(
        current: &str,
        destination: &str,
        links: &[Link],
        ordered_links: &[usize],
        now_ms: u64,
        policy: RoutePolicy,
        max_hops: usize,
        max_candidate_paths: usize,
        visited_nodes: &mut BTreeSet<String>,
        path_indices: &mut Vec<usize>,
        paths: &mut Vec<Vec<usize>>,
    ) -> Result<(), &'static str> {
        if current == destination {
            if paths.len() >= max_candidate_paths {
                return Err("candidate path cap exceeded; narrow the topology or hop bound");
            }
            paths.push(path_indices.clone());
            return Ok(());
        }
        if path_indices.len() >= max_hops {
            return Ok(());
        }

        for index in ordered_links {
            let link = &links[*index];
            if link.from != current
                || link.health(now_ms) != LinkHealth::Healthy
                || !link.failure_domains_complete
                || link.failure_domains.is_empty()
                || visited_nodes.contains(&link.to)
            {
                continue;
            }

            visited_nodes.insert(link.to.clone());
            path_indices.push(*index);
            visit_paths(
                &link.to,
                destination,
                links,
                ordered_links,
                now_ms,
                policy,
                max_hops,
                max_candidate_paths,
                visited_nodes,
                path_indices,
                paths,
            )?;
            path_indices.pop();
            visited_nodes.remove(&link.to);
        }
        Ok(())
    }

    let mut ordered_links: Vec<usize> = (0..links.len()).collect();
    ordered_links.sort_by(|a, b| links[*a].id.cmp(&links[*b].id));
    let mut paths = Vec::new();
    let mut visited_nodes = BTreeSet::new();
    visited_nodes.insert(source.to_owned());
    visit_paths(
        source,
        destination,
        links,
        &ordered_links,
        now_ms,
        policy,
        max_hops,
        max_candidate_paths,
        &mut visited_nodes,
        &mut Vec::new(),
        &mut paths,
    )?;

    struct Candidate {
        route: Route,
        domains: BTreeSet<String>,
    }

    let mut candidates = Vec::with_capacity(paths.len());
    for path in paths {
        let mut route_nodes = vec![source.to_owned()];
        let mut route_link_ids = Vec::with_capacity(path.len());
        let mut route_latencies = Vec::with_capacity(path.len());
        let mut latency_ms = 0u64;
        let mut energy_millijoules = 0u64;
        let mut bottleneck_quality_permille = 1000u16;
        let mut score = 0u64;
        let mut domains = BTreeSet::new();

        for index in &path {
            let link = &links[*index];
            route_nodes.push(link.to.clone());
            route_link_ids.push(link.id.clone());
            route_latencies.push(link.one_way_latency_ms);
            latency_ms = latency_ms.saturating_add(link.one_way_latency_ms);
            energy_millijoules = energy_millijoules.saturating_add(link.energy_millijoules);
            bottleneck_quality_permille = bottleneck_quality_permille.min(link.quality_permille);
            score = score.saturating_add(link.route_cost(policy));
            domains.extend(link.failure_domains.iter().cloned());
        }

        // Two routes sharing an internal relay are not independent even if
        // their physical links have distinct identifiers.
        for relay in route_nodes.iter().skip(1).take(route_nodes.len().saturating_sub(2)) {
            domains.insert(format!("relay-node:{relay}"));
        }

        candidates.push(Candidate {
            route: Route {
                nodes: route_nodes,
                link_ids: route_link_ids,
                link_latencies_ms: route_latencies,
                latency_ms,
                energy_millijoules,
                bottleneck_quality_permille,
                score,
            },
            domains,
        });
    }

    candidates.sort_by(|a, b| {
        a.route
            .score
            .cmp(&b.route.score)
            .then_with(|| a.route.link_ids.cmp(&b.route.link_ids))
    });

    let mut best: Option<(u64, usize, usize)> = None;
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            if !candidates[i].domains.is_disjoint(&candidates[j].domains) {
                continue;
            }
            let combined_score = candidates[i]
                .route
                .score
                .saturating_add(candidates[j].route.score);
            let is_better = match best {
                None => true,
                Some((old_score, old_i, old_j)) => {
                    combined_score < old_score
                        || (combined_score == old_score && (i, j) < (old_i, old_j))
                }
            };
            if is_better {
                best = Some((combined_score, i, j));
            }
        }
    }

    Ok(best.map(|(combined_score, i, j)| FailureDomainDiverseRoutes {
        primary: candidates[i].route.clone(),
        secondary: candidates[j].route.clone(),
        combined_score,
    }))
}

/// Evaluate a message's outbound route and return acknowledgement path. This
/// checks graph/receipt semantics only; it does not simulate packet loss,
/// contention, radio propagation, queuing or hardware.
pub fn evaluate_message(
    links: &[Link],
    message: &Message,
    now_ms: u64,
    policy: RoutePolicy,
) -> Result<MessageReceipt, &'static str> {
    if message.id.trim().is_empty() {
        return Err("message id must not be empty");
    }
    if message.source.trim().is_empty() || message.destination.trim().is_empty() {
        return Err("message endpoints must not be empty");
    }
    if message.created_at_ms > now_ms {
        return Err("message creation timestamp is in the future");
    }
    if message.expires_at_ms < message.created_at_ms {
        return Err("message expiry precedes creation");
    }

    let mut events = vec![
        MessageEvent {
            at_ms: message.created_at_ms,
            kind: MessageEventKind::Created,
            node: message.source.clone(),
        },
        MessageEvent {
            at_ms: message.created_at_ms,
            kind: MessageEventKind::Queued,
            node: message.source.clone(),
        },
    ];

    if message.expires_at_ms < now_ms {
        events.push(MessageEvent {
            at_ms: message.expires_at_ms,
            kind: MessageEventKind::Expired,
            node: message.source.clone(),
        });
        return Ok(MessageReceipt {
            message_id: message.id.clone(),
            class: message.class,
            disposition: ReceiptDisposition::ExpiredBeforeDelivery,
            forward_route: None,
            acknowledgement_route: None,
            delivered_at_ms: None,
            acknowledged_at_ms: None,
            total_energy_millijoules: 0,
            events,
        });
    }

    // This evaluator runs at now_ms. An older queued message cannot be
    // transmitted retroactively at its original creation timestamp.
    let transmission_start_ms = now_ms.max(message.created_at_ms);

    let Some(forward_route) = find_route(
        links,
        &message.source,
        &message.destination,
        now_ms,
        policy,
    )? else {
        events.push(MessageEvent {
            at_ms: transmission_start_ms,
            kind: MessageEventKind::NoRoute,
            node: message.source.clone(),
        });
        return Ok(MessageReceipt {
            message_id: message.id.clone(),
            class: message.class,
            disposition: ReceiptDisposition::NoRoute,
            forward_route: None,
            acknowledgement_route: None,
            delivered_at_ms: None,
            acknowledged_at_ms: None,
            total_energy_millijoules: 0,
            events,
        });
    };

    let delivered_at_ms = transmission_start_ms
        .checked_add(forward_route.latency_ms)
        .ok_or("delivery timestamp overflow")?;
    if delivered_at_ms > message.expires_at_ms {
        events.push(MessageEvent {
            at_ms: message.expires_at_ms,
            kind: MessageEventKind::Expired,
            node: message.source.clone(),
        });
        return Ok(MessageReceipt {
            message_id: message.id.clone(),
            class: message.class,
            disposition: ReceiptDisposition::ExpiredBeforeDelivery,
            forward_route: Some(forward_route),
            acknowledgement_route: None,
            delivered_at_ms: None,
            acknowledged_at_ms: None,
            total_energy_millijoules: 0,
            events,
        });
    }

    if !forward_route.link_ids.is_empty() {
        events.push(MessageEvent {
            at_ms: transmission_start_ms,
            kind: MessageEventKind::Transmitted,
            node: message.source.clone(),
        });
    }

    let mut event_time = transmission_start_ms;
    if forward_route.link_ids.is_empty() {
        events.push(MessageEvent {
            at_ms: delivered_at_ms,
            kind: MessageEventKind::Received,
            node: message.destination.clone(),
        });
    } else {
        for index in 0..forward_route.link_ids.len() {
            event_time = event_time
                .checked_add(forward_route.link_latencies_ms[index])
                .ok_or("relay timestamp overflow")?;
            let final_hop = index + 1 == forward_route.link_ids.len();
            events.push(MessageEvent {
                at_ms: event_time,
                kind: if final_hop {
                    MessageEventKind::Received
                } else {
                    MessageEventKind::Relayed
                },
                node: forward_route.nodes[index + 1].clone(),
            });
        }
    }

    events.push(MessageEvent {
        at_ms: delivered_at_ms,
        kind: if message.origin_authenticated {
            MessageEventKind::OriginAuthenticated
        } else {
            MessageEventKind::OriginUnverified
        },
        node: message.destination.clone(),
    });

    // Re-evaluate the return path at receiver-arrival time. Link evidence
    // can age out while the message is in flight.
    let Some(acknowledgement_route) = find_route(
        links,
        &message.destination,
        &message.source,
        delivered_at_ms,
        policy,
    )? else {
        events.push(MessageEvent {
            at_ms: delivered_at_ms,
            kind: MessageEventKind::AcknowledgementUnavailable,
            node: message.destination.clone(),
        });
        return Ok(MessageReceipt {
            message_id: message.id.clone(),
            class: message.class,
            disposition: ReceiptDisposition::ReceivedUnacknowledged,
            forward_route: Some(forward_route.clone()),
            acknowledgement_route: None,
            delivered_at_ms: Some(delivered_at_ms),
            acknowledged_at_ms: None,
            total_energy_millijoules: forward_route.energy_millijoules,
            events,
        });
    };

    let acknowledged_at_ms = delivered_at_ms
        .checked_add(acknowledgement_route.latency_ms)
        .ok_or("acknowledgement timestamp overflow")?;
    events.push(MessageEvent {
        at_ms: acknowledged_at_ms,
        kind: MessageEventKind::Acknowledged,
        node: message.source.clone(),
    });

    Ok(MessageReceipt {
        message_id: message.id.clone(),
        class: message.class,
        disposition: if message.origin_authenticated {
            ReceiptDisposition::Acknowledged
        } else {
            ReceiptDisposition::AcknowledgedUnverifiedOrigin
        },
        forward_route: Some(forward_route.clone()),
        acknowledgement_route: Some(acknowledgement_route.clone()),
        delivered_at_ms: Some(delivered_at_ms),
        acknowledged_at_ms: Some(acknowledged_at_ms),
        total_energy_millijoules: forward_route
            .energy_millijoules
            .saturating_add(acknowledgement_route.energy_millijoules),
        events,
    })
}

/// A bounded scheduling-window decision. This planner orders whole messages;
/// it does not model payload size, airtime, retransmissions or contention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuePlan {
    pub dispatched: Vec<String>,
    pub deferred: Vec<String>,
    pub expired: Vec<String>,
}

/// Plan a bounded dispatch window with explicit priority, age and expiry.
/// Higher-priority classes dispatch first; ties use creation time then message
/// id. Duplicate ids and malformed timestamps are errors, never silent drops.
pub fn plan_queue(
    messages: &[Message],
    now_ms: u64,
    service_slots: usize,
) -> Result<QueuePlan, &'static str> {
    let mut seen_ids = BTreeSet::new();
    let mut live: Vec<&Message> = Vec::new();
    let mut expired: Vec<&Message> = Vec::new();

    for message in messages {
        if message.id.trim().is_empty() {
            return Err("queued message id must not be empty");
        }
        if message.source.trim().is_empty() || message.destination.trim().is_empty() {
            return Err("queued message endpoints must not be empty");
        }
        if message.created_at_ms > now_ms {
            return Err("queued message creation timestamp is in the future");
        }
        if message.expires_at_ms < message.created_at_ms {
            return Err("queued message expiry precedes creation");
        }
        if !seen_ids.insert(message.id.as_str()) {
            return Err("queued message ids must be unique");
        }

        if message.expires_at_ms < now_ms {
            expired.push(message);
        } else {
            live.push(message);
        }
    }

    live.sort_by(|a, b| {
        a.class
            .priority_rank()
            .cmp(&b.class.priority_rank())
            .then_with(|| a.created_at_ms.cmp(&b.created_at_ms))
            .then_with(|| a.id.cmp(&b.id))
    });
    expired.sort_by(|a, b| a.id.cmp(&b.id));

    let split_at = service_slots.min(live.len());
    let dispatched = live[..split_at]
        .iter()
        .map(|message| message.id.clone())
        .collect();
    let deferred = live[split_at..]
        .iter()
        .map(|message| message.id.clone())
        .collect();
    let expired = expired.iter().map(|message| message.id.clone()).collect();

    Ok(QueuePlan {
        dispatched,
        deferred,
        expired,
    })
}

/// Single serialized channel service profile. Rate and delays are synthetic
/// scenario inputs until calibrated against a physical link.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AirtimeProfile {
    pub id: String,
    pub bitrate_bits_per_second: u64,
    pub framing_overhead_bytes: u64,
    pub propagation_delay_ms: u64,
    /// Maximum number of transmission attempts per message, including the first.
    pub max_attempts: u8,
}

impl AirtimeProfile {
    fn validate(&self) -> Result<(), &'static str> {
        if self.id.trim().is_empty() {
            return Err("airtime profile id must not be empty");
        }
        if self.bitrate_bits_per_second == 0 {
            return Err("channel bitrate must be positive");
        }
        if self.max_attempts == 0 {
            return Err("maximum transmission attempts must be positive");
        }
        Ok(())
    }

    fn airtime_ms(&self, payload_bytes: u64) -> Result<u64, &'static str> {
        let frame_bytes = u128::from(payload_bytes)
            .checked_add(u128::from(self.framing_overhead_bytes))
            .ok_or("frame byte count overflow")?;
        if frame_bytes == 0 {
            return Err("empty payload with no framing overhead is not transmittable");
        }
        let frame_bits = frame_bytes
            .checked_mul(8)
            .ok_or("frame bit count overflow")?;
        let numerator = frame_bits
            .checked_mul(1000)
            .ok_or("airtime numerator overflow")?;
        let rate = u128::from(self.bitrate_bits_per_second);
        let rounded_up = numerator
            .checked_add(rate - 1)
            .ok_or("airtime rounding overflow")?
            / rate;
        u64::try_from(rounded_up).map_err(|_| "airtime does not fit in milliseconds")
    }
}

/// A queued one-hop frame with a deterministic injected receive trace.
/// Each bool says whether that specific attempt is received by the next hop.
/// This is not a packet-loss probability and does not model acknowledgements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedTransmission {
    pub message: Message,
    pub payload_bytes: u64,
    /// Must contain exactly one outcome for each configured attempt.
    pub attempt_succeeds: Vec<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AirtimeDisposition {
    Received,
    ExpiredInQueue,
    ExpiredBeforeTransmission,
    ExpiredAfterFailedAttempts,
    RetryBudgetExhausted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AirtimePacketResult {
    pub message_id: String,
    pub class: MessageClass,
    pub disposition: AirtimeDisposition,
    pub attempts: u8,
    pub failed_attempts: u8,
    pub first_transmission_at_ms: Option<u64>,
    pub received_at_ms: Option<u64>,
    /// Time from function enqueue to first transmission.
    pub queue_wait_ms: Option<u64>,
    /// Sum of airtime consumed by all attempts for this message.
    pub airtime_ms: u64,
    /// Receiver arrival time minus message creation time.
    pub creation_to_receive_ms: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AirtimeSimulation {
    pub profile_id: String,
    pub enqueued_at_ms: u64,
    pub finished_at_ms: u64,
    pub total_airtime_ms: u64,
    pub results: Vec<AirtimePacketResult>,
}

/// Schedule a finite queue on one serialized, half-duplex service channel.
///
/// This deliberately models transmission serialization and supplied per-attempt
/// outcomes only: no stochastic loss, channel contention, acknowledgements,
/// backoff or multi-hop forwarding. Strict priority is non-preemptive: a frame
/// gets its bounded retry budget before the next queued frame is served.
pub fn simulate_airtime_queue(
    profile: &AirtimeProfile,
    queued: &[QueuedTransmission],
    now_ms: u64,
) -> Result<AirtimeSimulation, &'static str> {
    profile.validate()?;

    let mut ids = BTreeSet::new();
    let mut ordered: Vec<&QueuedTransmission> = Vec::with_capacity(queued.len());
    for packet in queued {
        let message = &packet.message;
        if message.id.trim().is_empty() {
            return Err("queued message id must not be empty");
        }
        if message.source.trim().is_empty() || message.destination.trim().is_empty() {
            return Err("queued message endpoints must not be empty");
        }
        if message.created_at_ms > now_ms {
            return Err("queued message creation timestamp is in the future");
        }
        if message.expires_at_ms < message.created_at_ms {
            return Err("queued message expiry precedes creation");
        }
        if packet.attempt_succeeds.len() != usize::from(profile.max_attempts) {
            return Err("attempt trace length must equal configured maximum attempts");
        }
        if !ids.insert(message.id.as_str()) {
            return Err("queued message ids must be unique");
        }
        // Validate the frame duration before simulating any packet.
        profile.airtime_ms(packet.payload_bytes)?;
        ordered.push(packet);
    }

    ordered.sort_by(|a, b| {
        a.message
            .class
            .priority_rank()
            .cmp(&b.message.class.priority_rank())
            .then_with(|| a.message.created_at_ms.cmp(&b.message.created_at_ms))
            .then_with(|| a.message.id.cmp(&b.message.id))
    });

    let mut current_time_ms = now_ms;
    let mut total_airtime_ms = 0u64;
    let mut results = Vec::with_capacity(ordered.len());

    for packet in ordered {
        let message = &packet.message;
        let frame_airtime_ms = profile.airtime_ms(packet.payload_bytes)?;
        let mut attempts = 0u8;
        let mut failed_attempts = 0u8;
        let mut packet_airtime_ms = 0u64;
        let mut first_transmission_at_ms = None;
        let mut received_at_ms = None;
        let mut disposition = AirtimeDisposition::RetryBudgetExhausted;

        if current_time_ms > message.expires_at_ms {
            disposition = AirtimeDisposition::ExpiredInQueue;
        } else {
            for attempt_index in 0..profile.max_attempts {
                let tx_end_ms = current_time_ms
                    .checked_add(frame_airtime_ms)
                    .ok_or("transmission end timestamp overflow")?;
                let arrival_ms = tx_end_ms
                    .checked_add(profile.propagation_delay_ms)
                    .ok_or("receiver arrival timestamp overflow")?;

                // Do not spend channel capacity on a frame that cannot arrive
                // before expiry under the declared profile.
                if arrival_ms > message.expires_at_ms {
                    disposition = if attempts == 0 {
                        AirtimeDisposition::ExpiredBeforeTransmission
                    } else {
                        AirtimeDisposition::ExpiredAfterFailedAttempts
                    };
                    break;
                }

                if first_transmission_at_ms.is_none() {
                    first_transmission_at_ms = Some(current_time_ms);
                }
                attempts = attempts.saturating_add(1);
                packet_airtime_ms = packet_airtime_ms
                    .checked_add(frame_airtime_ms)
                    .ok_or("per-message airtime overflow")?;
                total_airtime_ms = total_airtime_ms
                    .checked_add(frame_airtime_ms)
                    .ok_or("total airtime overflow")?;
                current_time_ms = tx_end_ms;

                if packet.attempt_succeeds[usize::from(attempt_index)] {
                    received_at_ms = Some(arrival_ms);
                    disposition = AirtimeDisposition::Received;
                    break;
                }

                failed_attempts = failed_attempts.saturating_add(1);
            }
        }

        let queue_wait_ms = first_transmission_at_ms
            .map(|first_tx| first_tx.saturating_sub(now_ms));
        let creation_to_receive_ms = received_at_ms
            .map(|arrival| arrival.saturating_sub(message.created_at_ms));

        results.push(AirtimePacketResult {
            message_id: message.id.clone(),
            class: message.class,
            disposition,
            attempts,
            failed_attempts,
            first_transmission_at_ms,
            received_at_ms,
            queue_wait_ms,
            airtime_ms: packet_airtime_ms,
            creation_to_receive_ms,
        });
    }

    Ok(AirtimeSimulation {
        profile_id: profile.id.clone(),
        enqueued_at_ms: now_ms,
        finished_at_ms: current_time_ms,
        total_airtime_ms,
        results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(
        id: &str,
        from: &str,
        to: &str,
        channel: ChannelKind,
        latency: u64,
        quality: u16,
        energy: u64,
    ) -> Link {
        Link {
            id: id.to_owned(),
            from: from.to_owned(),
            to: to.to_owned(),
            channel,
            state: LinkState::Up,
            quality_permille: quality,
            observed_at_ms: 100,
            max_observation_age_ms: 10_000,
            one_way_latency_ms: latency,
            energy_millijoules: energy,
            failure_domains: vec![format!(
                "physical-link:{}",
                id.strip_suffix("-ab").or_else(|| id.strip_suffix("-ba")).unwrap_or(id)
            )],
            failure_domains_complete: true,
        }
    }

    fn bidirectional(
        links: &mut Vec<Link>,
        id: &str,
        a: &str,
        b: &str,
        channel: ChannelKind,
        latency: u64,
        quality: u16,
        energy: u64,
    ) {
        links.push(link(
            &format!("{id}-ab"),
            a,
            b,
            channel,
            latency,
            quality,
            energy,
        ));
        links.push(link(
            &format!("{id}-ba"),
            b,
            a,
            channel,
            latency,
            quality,
            energy,
        ));
    }

    fn demo_links() -> Vec<Link> {
        let mut links = Vec::new();
        bidirectional(&mut links, "backbone", "surface", "junction", ChannelKind::Backbone, 10, 995, 5);
        bidirectional(&mut links, "local-1", "junction", "level-4", ChannelKind::LocalRadio, 15, 990, 3);
        bidirectional(&mut links, "local-2", "level-4", "refuge", ChannelKind::LocalRadio, 10, 990, 2);
        bidirectional(&mut links, "tte", "surface", "level-4", ChannelKind::ThroughEarth, 500, 850, 45);
        links
    }

    fn distress_message() -> Message {
        Message {
            id: "SOS-1".to_owned(),
            class: MessageClass::Distress,
            source: "surface".to_owned(),
            destination: "refuge".to_owned(),
            created_at_ms: 1_000,
            expires_at_ms: 10_000,
            origin_authenticated: true,
        }
    }

    #[test]
    fn static_policy_prefers_established_backbone_when_available() {
        let links = demo_links();
        let route = find_route(
            &links,
            "surface",
            "refuge",
            1_000,
            RoutePolicy::FixedChannelPreference,
        )
        .expect("valid graph")
        .expect("route exists");
        assert_eq!(route.nodes.iter().map(String::as_str).collect::<Vec<_>>(), vec!["surface", "junction", "level-4", "refuge"]);
        assert!(route.link_ids.iter().all(|id| !id.starts_with("tte")));
    }

    #[test]
    fn both_policies_use_tte_when_backbone_is_unavailable() {
        let mut links = demo_links();
        for item in &mut links {
            if item.id.starts_with("backbone") {
                item.state = LinkState::Down;
            }
        }
        for policy in [RoutePolicy::FixedChannelPreference, RoutePolicy::AdaptiveLatencyEnergy] {
            let route = find_route(&links, "surface", "refuge", 1_000, policy)
                .expect("valid graph")
                .expect("TTE fallback exists");
            assert!(route.link_ids.iter().any(|id| id.starts_with("tte")));
        }
    }

    #[test]
    fn adaptive_policy_can_prefer_a_fast_measured_backup() {
        let mut links = Vec::new();
        bidirectional(&mut links, "slow-backbone", "a", "z", ChannelKind::Backbone, 500, 990, 5);
        bidirectional(&mut links, "fast-tte", "a", "z", ChannelKind::ThroughEarth, 30, 990, 1);
        let fixed = find_route(&links, "a", "z", 1_000, RoutePolicy::FixedChannelPreference)
            .expect("valid graph")
            .expect("fixed path");
        let adaptive = find_route(&links, "a", "z", 1_000, RoutePolicy::AdaptiveLatencyEnergy)
            .expect("valid graph")
            .expect("adaptive path");
        assert!(fixed.link_ids[0].starts_with("slow-backbone"));
        assert!(adaptive.link_ids[0].starts_with("fast-tte"));
    }

    #[test]
    fn stale_up_link_is_not_routed_as_healthy() {
        let mut links = vec![link("stale", "a", "b", ChannelKind::Backbone, 1, 1000, 1)];
        links[0].observed_at_ms = 100;
        links[0].max_observation_age_ms = 10;
        assert_eq!(links[0].health(111), LinkHealth::Stale);
        assert!(find_route(&links, "a", "b", 111, RoutePolicy::AdaptiveLatencyEnergy)
            .expect("valid graph")
            .is_none());
    }

    #[test]
    fn unknown_down_and_zero_quality_links_are_not_routable() {
        let mut down = vec![link("x", "a", "b", ChannelKind::Backbone, 1, 1000, 1)];
        down[0].state = LinkState::Down;
        assert_eq!(down[0].health(100), LinkHealth::Down);
        assert!(find_route(&down, "a", "b", 100, RoutePolicy::FixedChannelPreference).unwrap().is_none());

        let mut unknown = down.clone();
        unknown[0].state = LinkState::Unknown;
        assert_eq!(unknown[0].health(100), LinkHealth::Unknown);
        assert!(find_route(&unknown, "a", "b", 100, RoutePolicy::FixedChannelPreference).unwrap().is_none());

        let mut zero = down.clone();
        zero[0].state = LinkState::Up;
        zero[0].quality_permille = 0;
        assert_eq!(zero[0].health(100), LinkHealth::ZeroQuality);
        assert!(find_route(&zero, "a", "b", 100, RoutePolicy::FixedChannelPreference).unwrap().is_none());
    }

    #[test]
    fn delayed_queue_does_not_receive_retroactive_delivery_timestamps() {
        let links = vec![
            link("ab", "a", "b", ChannelKind::Backbone, 10, 1000, 3),
            link("ba", "b", "a", ChannelKind::Backbone, 10, 1000, 3),
        ];
        let mut message = distress_message();
        message.source = "a".to_owned();
        message.destination = "b".to_owned();
        message.created_at_ms = 100;
        message.expires_at_ms = 1_000;
        let receipt = evaluate_message(&links, &message, 500, RoutePolicy::FixedChannelPreference)
            .expect("valid delayed message");
        assert_eq!(receipt.delivered_at_ms, Some(510));
        assert_eq!(receipt.acknowledged_at_ms, Some(520));
        assert!(receipt.events.iter().any(|event| {
            event.kind == MessageEventKind::Transmitted && event.at_ms == 500
        }));
    }

    #[test]
    fn message_already_expired_at_evaluation_is_not_transmitted() {
        let links = vec![
            link("ab", "a", "b", ChannelKind::Backbone, 10, 1000, 3),
            link("ba", "b", "a", ChannelKind::Backbone, 10, 1000, 3),
        ];
        let mut message = distress_message();
        message.source = "a".to_owned();
        message.destination = "b".to_owned();
        message.created_at_ms = 100;
        message.expires_at_ms = 200;
        let receipt = evaluate_message(&links, &message, 201, RoutePolicy::FixedChannelPreference)
            .expect("expired message is a typed outcome");
        assert_eq!(receipt.disposition, ReceiptDisposition::ExpiredBeforeDelivery);
        assert!(receipt.forward_route.is_none());
        assert!(receipt.delivered_at_ms.is_none());
        assert!(!receipt.events.iter().any(|event| event.kind == MessageEventKind::Transmitted));
    }

    #[test]
    fn acknowledgement_path_must_still_be_fresh_when_receiver_gets_message() {
        let mut forward = link("ab", "a", "b", ChannelKind::Backbone, 10, 1000, 3);
        forward.max_observation_age_ms = 100;
        let mut reverse = link("ba", "b", "a", ChannelKind::Backbone, 10, 1000, 3);
        reverse.max_observation_age_ms = 5;
        let links = vec![forward, reverse];
        let mut message = distress_message();
        message.source = "a".to_owned();
        message.destination = "b".to_owned();
        message.created_at_ms = 100;
        message.expires_at_ms = 1_000;
        let receipt = evaluate_message(&links, &message, 100, RoutePolicy::FixedChannelPreference)
            .expect("valid message, return path ages out in flight");
        assert_eq!(receipt.delivered_at_ms, Some(110));
        assert_eq!(receipt.disposition, ReceiptDisposition::ReceivedUnacknowledged);
        assert!(receipt.acknowledgement_route.is_none());
        assert!(receipt.acknowledged_at_ms.is_none());
    }

    #[test]
    fn receiver_must_be_able_to_acknowledge_end_to_end_delivery() {
        let mut links = vec![
            link("forward", "a", "b", ChannelKind::Backbone, 4, 1000, 3),
            link("reverse", "b", "a", ChannelKind::Backbone, 4, 1000, 3),
        ];
        links[1].state = LinkState::Down;
        let mut message = distress_message();
        message.source = "a".to_owned();
        message.destination = "b".to_owned();
        let receipt = evaluate_message(&links, &message, 100, RoutePolicy::FixedChannelPreference)
            .expect("valid message and graph");
        assert_eq!(receipt.disposition, ReceiptDisposition::ReceivedUnacknowledged);
        assert!(receipt.delivered_at_ms.is_some());
        assert!(receipt.acknowledged_at_ms.is_none());
        assert!(receipt.events.iter().any(|event| event.kind == MessageEventKind::AcknowledgementUnavailable));
    }

    #[test]
    fn distress_can_be_received_when_origin_authentication_is_unavailable() {
        let links = vec![
            link("ab", "a", "b", ChannelKind::Backbone, 4, 1000, 3),
            link("ba", "b", "a", ChannelKind::Backbone, 4, 1000, 3),
        ];
        let mut message = distress_message();
        message.source = "a".to_owned();
        message.destination = "b".to_owned();
        message.origin_authenticated = false;
        let receipt = evaluate_message(&links, &message, 100, RoutePolicy::FixedChannelPreference)
            .expect("valid message and graph");
        assert_eq!(receipt.disposition, ReceiptDisposition::AcknowledgedUnverifiedOrigin);
        assert!(receipt.events.iter().any(|event| event.kind == MessageEventKind::OriginUnverified));
        assert!(receipt.events.iter().any(|event| event.kind == MessageEventKind::Acknowledged));
    }

    #[test]
    fn a_message_expiring_before_delivery_is_not_reported_delivered() {
        let links = vec![
            link("ab", "a", "b", ChannelKind::Backbone, 100, 1000, 3),
            link("ba", "b", "a", ChannelKind::Backbone, 100, 1000, 3),
        ];
        let mut message = distress_message();
        message.source = "a".to_owned();
        message.destination = "b".to_owned();
        message.expires_at_ms = message.created_at_ms + 50;
        let receipt = evaluate_message(&links, &message, 1_000, RoutePolicy::FixedChannelPreference)
            .expect("valid message and graph");
        assert_eq!(receipt.disposition, ReceiptDisposition::ExpiredBeforeDelivery);
        assert!(receipt.delivered_at_ms.is_none());
    }

    #[test]
    fn no_route_is_distinct_from_expiry_and_unacknowledged_reception() {
        let receipt = evaluate_message(
            &[],
            &distress_message(),
            1_000,
            RoutePolicy::FixedChannelPreference,
        )
        .expect("valid empty graph");
        assert_eq!(receipt.disposition, ReceiptDisposition::NoRoute);
        assert!(receipt.forward_route.is_none());
    }

    #[test]
    fn equal_cost_paths_have_deterministic_ties() {
        let mut links = Vec::new();
        bidirectional(&mut links, "path-b", "a", "b", ChannelKind::Backbone, 1, 1000, 1);
        bidirectional(&mut links, "path-z", "b", "z", ChannelKind::Backbone, 1, 1000, 1);
        bidirectional(&mut links, "path-c", "a", "c", ChannelKind::Backbone, 1, 1000, 1);
        bidirectional(&mut links, "path-d", "c", "z", ChannelKind::Backbone, 1, 1000, 1);
        let route = find_route(&links, "a", "z", 100, RoutePolicy::FixedChannelPreference)
            .expect("valid graph")
            .expect("route exists");
        assert_eq!(route.nodes.iter().map(String::as_str).collect::<Vec<_>>(), vec!["a", "b", "z"]);
    }

    #[test]
    fn duplicate_link_ids_and_future_observations_are_rejected() {
        let first = link("duplicate", "a", "b", ChannelKind::Backbone, 1, 1000, 1);
        let second = link("duplicate", "b", "a", ChannelKind::Backbone, 1, 1000, 1);
        assert!(find_route(&[first, second], "a", "b", 100, RoutePolicy::FixedChannelPreference).is_err());

        let mut future = link("future", "a", "b", ChannelKind::Backbone, 1, 1000, 1);
        future.observed_at_ms = 200;
        assert!(find_route(&[future], "a", "b", 100, RoutePolicy::FixedChannelPreference).is_err());
    }

    #[test]
    fn nominal_and_fallback_receipts_are_reproducible() {
        let links = demo_links();
        let message = distress_message();
        let first = evaluate_message(&links, &message, 1_000, RoutePolicy::AdaptiveLatencyEnergy).unwrap();
        let second = evaluate_message(&links, &message, 1_000, RoutePolicy::AdaptiveLatencyEnergy).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.disposition, ReceiptDisposition::Acknowledged);

        let mut damaged = links;
        for item in &mut damaged {
            if item.id.starts_with("backbone") {
                item.state = LinkState::Down;
            }
        }
        let fallback = evaluate_message(&damaged, &message, 1_000, RoutePolicy::AdaptiveLatencyEnergy).unwrap();
        assert_eq!(fallback.disposition, ReceiptDisposition::Acknowledged);
        assert!(fallback.forward_route.unwrap().link_ids.iter().any(|id| id.starts_with("tte")));
    }

    fn diversity_network() -> Vec<Link> {
        vec![
            link("sa", "s", "a", ChannelKind::Backbone, 10, 990, 5),
            link("ad", "a", "d", ChannelKind::Backbone, 10, 990, 5),
            link("sb", "s", "b", ChannelKind::LocalRadio, 20, 980, 8),
            link("bd", "b", "d", ChannelKind::LocalRadio, 20, 980, 8),
        ]
    }

    #[test]
    fn two_paths_are_accepted_only_when_declared_failure_domains_are_disjoint() {
        let links = diversity_network();
        let pair = find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 100,
        )
        .expect("valid bounded graph")
        .expect("two independent paths");
        assert!(pair.primary.link_ids.iter().all(|id| pair.secondary.link_ids.iter().all(|other| other != id)));
        assert_ne!(pair.primary.nodes, pair.secondary.nodes);
    }

    #[test]
    fn shared_power_domain_invalidates_apparent_link_diversity() {
        let mut links = diversity_network();
        for item in &mut links {
            if item.id == "ad" || item.id == "bd" {
                item.failure_domains.push("power-feed:shared".to_owned());
            }
        }
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 100,
        )
        .expect("valid bounded graph")
        .is_none());
    }

    #[test]
    fn incomplete_domain_inventory_cannot_claim_independence() {
        let mut links = diversity_network();
        links[2].failure_domains_complete = false;
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 100,
        )
        .expect("valid graph with incomplete evidence")
        .is_none());
    }

    #[test]
    fn stale_links_are_excluded_from_independent_route_claims() {
        let mut links = diversity_network();
        links[2].observed_at_ms = 1;
        links[2].max_observation_age_ms = 1;
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 100,
        )
        .expect("valid graph")
        .is_none());
    }

    #[test]
    fn diverse_route_search_is_bounded_and_validates_domain_manifest() {
        let links = diversity_network();
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 0, 100,
        ).is_err());
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 0,
        ).is_err());
        let mut malformed = links;
        malformed[0].failure_domains_complete = true;
        malformed[0].failure_domains.clear();
        assert!(find_failure_domain_diverse_routes(
            &malformed, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 100,
        ).is_err());
    }

    #[test]
    fn shared_internal_relay_invalidates_apparently_distinct_paths() {
        let links = vec![
            link("sa", "s", "a", ChannelKind::Backbone, 10, 990, 5),
            link("ac", "a", "c", ChannelKind::Backbone, 10, 990, 5),
            link("sb", "s", "b", ChannelKind::LocalRadio, 20, 980, 8),
            link("bc", "b", "c", ChannelKind::LocalRadio, 20, 980, 8),
            link("cd-a", "c", "d", ChannelKind::Backbone, 10, 990, 5),
            link("cd-b", "c", "d", ChannelKind::LocalRadio, 20, 980, 8),
        ];
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 5, 100,
        )
        .expect("valid graph")
        .is_none());
    }

    #[test]
    fn candidate_path_cap_fails_explicitly_instead_of_silently_truncating() {
        let links = diversity_network();
        assert!(find_failure_domain_diverse_routes(
            &links, "s", "d", 100, RoutePolicy::AdaptiveLatencyEnergy, 4, 1,
        )
        .is_err());
    }

    fn airtime_profile(rate: u64, overhead: u64, propagation: u64, attempts: u8) -> AirtimeProfile {
        AirtimeProfile {
            id: "SYNTHETIC-CHANNEL".to_owned(),
            bitrate_bits_per_second: rate,
            framing_overhead_bytes: overhead,
            propagation_delay_ms: propagation,
            max_attempts: attempts,
        }
    }

    fn queued_packet(
        id: &str,
        class: MessageClass,
        payload_bytes: u64,
        expires_at_ms: u64,
        attempt_succeeds: &[bool],
    ) -> QueuedTransmission {
        QueuedTransmission {
            message: Message {
                id: id.to_owned(),
                class,
                source: "surface".to_owned(),
                destination: "next-hop".to_owned(),
                created_at_ms: 1_000,
                expires_at_ms,
                origin_authenticated: true,
            },
            payload_bytes,
            attempt_succeeds: attempt_succeeds.to_vec(),
        }
    }

    #[test]
    fn airtime_queue_accounts_for_framing_serialization_and_propagation() {
        // 12 frame bytes at 8,000 bit/s is exactly 12 ms of airtime.
        let profile = airtime_profile(8_000, 2, 3, 1);
        let packets = vec![
            queued_packet("a", MessageClass::Distress, 10, 2_000, &[true]),
            queued_packet("b", MessageClass::Telemetry, 8, 2_000, &[true]),
        ];
        let sim = simulate_airtime_queue(&profile, &packets, 1_000).expect("valid trace");
        assert_eq!(sim.results.len(), 2);
        assert_eq!(sim.results[0].airtime_ms, 12);
        assert_eq!(sim.results[0].received_at_ms, Some(1_015));
        assert_eq!(sim.results[1].first_transmission_at_ms, Some(1_012));
        assert_eq!(sim.results[1].received_at_ms, Some(1_025));
        assert_eq!(sim.total_airtime_ms, 22);
        assert_eq!(sim.finished_at_ms, 1_022);
    }

    #[test]
    fn airtime_scheduler_delivers_critical_alert_before_bulk_under_alarm_storm() {
        let profile = airtime_profile(8_000, 10, 2, 1);
        let packets = vec![
            queued_packet("bulk", MessageClass::Bulk, 500, 10_000, &[true]),
            queued_packet("telemetry", MessageClass::Telemetry, 100, 10_000, &[true]),
            queued_packet("evac", MessageClass::Evacuation, 40, 10_000, &[true]),
            queued_packet("sos", MessageClass::Distress, 20, 10_000, &[true]),
        ];
        let sim = simulate_airtime_queue(&profile, &packets, 1_000).expect("valid trace");
        let sos = sim.results.iter().find(|r| r.message_id == "sos").unwrap();
        let bulk = sim.results.iter().find(|r| r.message_id == "bulk").unwrap();
        assert!(sos.received_at_ms.unwrap() < bulk.received_at_ms.unwrap());
        assert_eq!(sos.queue_wait_ms, Some(0));
        assert!(bulk.queue_wait_ms.unwrap() > sos.queue_wait_ms.unwrap());
    }

    #[test]
    fn critical_message_uses_bounded_retry_budget_before_lower_priority_traffic() {
        let profile = airtime_profile(8_000, 0, 0, 2);
        let packets = vec![
            queued_packet("bulk", MessageClass::Bulk, 20, 5_000, &[true, true]),
            queued_packet("sos", MessageClass::Distress, 10, 5_000, &[false, true]),
        ];
        let sim = simulate_airtime_queue(&profile, &packets, 1_000).expect("valid trace");
        let sos = sim.results.iter().find(|r| r.message_id == "sos").unwrap();
        let bulk = sim.results.iter().find(|r| r.message_id == "bulk").unwrap();
        assert_eq!(sos.disposition, AirtimeDisposition::Received);
        assert_eq!(sos.attempts, 2);
        assert!(sos.received_at_ms.unwrap() < bulk.received_at_ms.unwrap());
        assert_eq!(sos.failed_attempts, 1);
    }

    #[test]
    fn expired_packet_does_not_consume_airtime_after_longer_critical_frame() {
        let profile = airtime_profile(8_000, 0, 0, 1);
        let packets = vec![
            queued_packet("sos", MessageClass::Distress, 100, 5_000, &[true]),
            queued_packet("bulk-expired", MessageClass::Bulk, 10, 1_010, &[true]),
        ];
        let sim = simulate_airtime_queue(&profile, &packets, 1_000).expect("valid trace");
        let expired = sim.results.iter().find(|r| r.message_id == "bulk-expired").unwrap();
        assert_eq!(expired.disposition, AirtimeDisposition::ExpiredInQueue);
        assert_eq!(expired.attempts, 0);
        assert_eq!(expired.airtime_ms, 0);
    }

    #[test]
    fn airtime_queue_fails_closed_on_invalid_profile_trace_and_duplicate_ids() {
        let packet = queued_packet("a", MessageClass::Distress, 10, 2_000, &[true]);
        assert!(simulate_airtime_queue(&airtime_profile(0, 2, 1, 1), &[packet.clone()], 1_000).is_err());
        assert!(simulate_airtime_queue(&airtime_profile(8_000, 2, 1, 2), &[packet.clone()], 1_000).is_err());
        assert!(simulate_airtime_queue(&airtime_profile(8_000, 2, 1, 1), &[packet.clone(), packet], 1_000).is_err());
    }

    #[test]
    fn alarm_storm_dispatches_distress_before_bulk_and_excludes_expired() {
        let make_message = |id: &str, class: MessageClass, expires_at_ms: u64| Message {
            id: id.to_owned(),
            class,
            source: "surface".to_owned(),
            destination: "refuge".to_owned(),
            created_at_ms: 900,
            expires_at_ms,
            origin_authenticated: true,
        };
        let messages = vec![
            make_message("bulk", MessageClass::Bulk, 5_000),
            make_message("telemetry", MessageClass::Telemetry, 5_000),
            make_message("evacuate", MessageClass::Evacuation, 5_000),
            make_message("sos", MessageClass::Distress, 5_000),
            make_message("expired-sos", MessageClass::Distress, 999),
        ];
        let plan = plan_queue(&messages, 1_000, 2).expect("valid queue");
        assert_eq!(plan.dispatched.iter().map(String::as_str).collect::<Vec<_>>(), vec!["sos", "evacuate"]);
        assert_eq!(plan.deferred.iter().map(String::as_str).collect::<Vec<_>>(), vec!["telemetry", "bulk"]);
        assert_eq!(plan.expired.iter().map(String::as_str).collect::<Vec<_>>(), vec!["expired-sos"]);
    }

    #[test]
    fn queue_rejects_duplicate_ids_instead_of_merging_messages() {
        let first = distress_message();
        let second = first.clone();
        assert!(plan_queue(&[first, second], 1_000, 1).is_err());
    }

    #[test]
    fn priority_order_is_explicit_and_stable() {
        assert!(MessageClass::Distress.priority_rank() < MessageClass::Evacuation.priority_rank());
        assert!(MessageClass::Evacuation.priority_rank() < MessageClass::Telemetry.priority_rank());
        assert!(MessageClass::Telemetry.priority_rank() < MessageClass::Bulk.priority_rank());
    }
}
