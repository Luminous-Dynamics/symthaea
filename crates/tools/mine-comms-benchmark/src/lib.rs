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

    let Some(forward_route) = find_route(
        links,
        &message.source,
        &message.destination,
        now_ms,
        policy,
    )? else {
        events.push(MessageEvent {
            at_ms: message.created_at_ms,
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

    let delivered_at_ms = message
        .created_at_ms
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
            at_ms: message.created_at_ms,
            kind: MessageEventKind::Transmitted,
            node: message.source.clone(),
        });
    }

    let mut event_time = message.created_at_ms;
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

    let Some(acknowledgement_route) = find_route(
        links,
        &message.destination,
        &message.source,
        now_ms,
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
        assert_eq!(plan.dispatched, vec!["sos", "evacuate"]);
        assert_eq!(plan.deferred, vec!["telemetry", "bulk"]);
        assert_eq!(plan.expired, vec!["expired-sos"]);
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
