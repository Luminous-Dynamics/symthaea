// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

use mine_comms_benchmark::{
    evaluate_message, ChannelKind, Link, LinkState, Message, MessageClass, RoutePolicy,
};

fn add_pair(
    links: &mut Vec<Link>,
    id: &str,
    from: &str,
    to: &str,
    channel: ChannelKind,
    latency_ms: u64,
    quality_permille: u16,
    energy_millijoules: u64,
) {
    for (suffix, source, destination) in [("ab", from, to), ("ba", to, from)] {
        links.push(Link {
            id: format!("{id}-{suffix}"),
            from: source.to_owned(),
            to: destination.to_owned(),
            channel,
            state: LinkState::Up,
            quality_permille,
            observed_at_ms: 1_000,
            max_observation_age_ms: 60_000,
            one_way_latency_ms: latency_ms,
            energy_millijoules,
        });
    }
}

fn demo_links() -> Vec<Link> {
    let mut links = Vec::new();
    add_pair(&mut links, "backbone", "surface", "junction", ChannelKind::Backbone, 10, 995, 5);
    add_pair(&mut links, "local-1", "junction", "level-4", ChannelKind::LocalRadio, 15, 990, 3);
    add_pair(&mut links, "local-2", "level-4", "refuge", ChannelKind::LocalRadio, 10, 990, 2);
    add_pair(&mut links, "tte", "surface", "level-4", ChannelKind::ThroughEarth, 500, 850, 45);
    links
}

fn report(case: &str, links: &[Link], message: &Message, policy: RoutePolicy) {
    match evaluate_message(links, message, 1_000, policy) {
        Ok(receipt) => {
            let path = receipt
                .forward_route
                .as_ref()
                .map(|route| route.nodes.join(" -> "))
                .unwrap_or_else(|| "none".to_owned());
            println!(
                "case={case} policy={policy:?} class={:?} priority={} disposition={:?} path={path} delivered_at_ms={:?} acknowledged_at_ms={:?} energy_mJ={}",
                receipt.class,
                receipt.class.priority_rank(),
                receipt.disposition,
                receipt.delivered_at_ms,
                receipt.acknowledged_at_ms,
                receipt.total_energy_millijoules
            );
        }
        Err(error) => println!("case={case} policy={policy:?} error={error}"),
    }
}

fn main() {
    let baseline = demo_links();
    let message = Message {
        id: "SOS-SYNTHETIC-001".to_owned(),
        class: MessageClass::Distress,
        source: "surface".to_owned(),
        destination: "refuge".to_owned(),
        created_at_ms: 1_000,
        expires_at_ms: 10_000,
        origin_authenticated: true,
    };

    println!("MINE-COMMS-001 deterministic route/receipt contract demo");
    println!("synthetic_only=true physical_propagation_model=false");
    for policy in [RoutePolicy::FixedChannelPreference, RoutePolicy::AdaptiveLatencyEnergy] {
        report("nominal", &baseline, &message, policy);
    }

    let mut backbone_severed = baseline;
    for link in &mut backbone_severed {
        if link.id.starts_with("backbone") {
            link.state = LinkState::Down;
        }
    }
    for policy in [RoutePolicy::FixedChannelPreference, RoutePolicy::AdaptiveLatencyEnergy] {
        report("backbone_severed", &backbone_severed, &message, policy);
    }
    println!("claim_ceiling=synthetic routing semantics only; no radio, mine-safety, or field-reliability claim");
}
