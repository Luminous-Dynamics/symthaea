// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

use mine_comms_benchmark::{
    evaluate_message, plan_queue, simulate_airtime_queue, AirtimeProfile, ChannelKind, Link,
    LinkState, Message, MessageClass, QueuedTransmission, RoutePolicy,
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
            failure_domains: vec![format!("physical-link:{id}")],
            failure_domains_complete: true,
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
    let policies = [RoutePolicy::FixedChannelPreference, RoutePolicy::AdaptiveLatencyEnergy];

    println!("MINE-COMMS-001 deterministic route/receipt contract demo");
    println!("synthetic_only=true physical_propagation_model=false");
    for policy in policies {
        report("nominal", &baseline, &message, policy);
    }

    let mut backbone_severed = baseline.clone();
    for link in &mut backbone_severed {
        if link.id.starts_with("backbone") {
            link.state = LinkState::Down;
        }
    }
    for policy in policies {
        report("backbone_severed", &backbone_severed, &message, policy);
    }

    let mut local_relay_severed = baseline.clone();
    for link in &mut local_relay_severed {
        if link.id.starts_with("local-1") {
            link.state = LinkState::Down;
        }
    }
    for policy in policies {
        report("local_relay_severed", &local_relay_severed, &message, policy);
    }

    let mut stale_tte_after_backbone_loss = baseline.clone();
    for link in &mut stale_tte_after_backbone_loss {
        if link.id.starts_with("backbone") {
            link.state = LinkState::Down;
        }
        if link.id.starts_with("tte") {
            link.observed_at_ms = 500;
            link.max_observation_age_ms = 100;
        }
    }
    for policy in policies {
        report("stale_tte_after_backbone_loss", &stale_tte_after_backbone_loss, &message, policy);
    }

    let mut no_return_path = baseline.clone();
    for link in &mut no_return_path {
        if link.id.ends_with("-ba") {
            link.state = LinkState::Down;
        }
    }
    for policy in policies {
        report("forward_path_without_ack_path", &no_return_path, &message, policy);
    }

    let mut short_deadline = message.clone();
    short_deadline.id = "SOS-EXPIRED-SYNTHETIC-001".to_owned();
    short_deadline.expires_at_ms = 1_020;
    for policy in policies {
        report("expires_before_arrival", &baseline, &short_deadline, policy);
    }

    let make_message = |id: &str, class: MessageClass, expires_at_ms: u64| Message {
        id: id.to_owned(),
        class,
        source: "surface".to_owned(),
        destination: "refuge".to_owned(),
        created_at_ms: 900,
        expires_at_ms,
        origin_authenticated: true,
    };
    let alarm_storm = vec![
        make_message("BULK-001", MessageClass::Bulk, 5_000),
        make_message("TELEMETRY-001", MessageClass::Telemetry, 5_000),
        make_message("EVACUATION-001", MessageClass::Evacuation, 5_000),
        make_message("SOS-002", MessageClass::Distress, 5_000),
        make_message("EXPIRED-SOS-001", MessageClass::Distress, 999),
    ];
    match plan_queue(&alarm_storm, 1_000, 2) {
        Ok(plan) => println!(
            "case=alarm_storm service_slots=2 dispatched={} deferred={} expired={}",
            plan.dispatched.join(","),
            plan.deferred.join(","),
            plan.expired.join(",")
        ),
        Err(error) => println!("case=alarm_storm error={error}"),
    }

    let airtime_profile = AirtimeProfile {
        id: "LAB-LOW-RATE-SYNTHETIC".to_owned(),
        bitrate_bits_per_second: 8_000,
        framing_overhead_bytes: 10,
        propagation_delay_ms: 2,
        max_attempts: 2,
    };
    let airtime_queue = vec![
        QueuedTransmission {
            message: Message {
                id: "BULK-AIRTIME".to_owned(),
                class: MessageClass::Bulk,
                source: "surface".to_owned(),
                destination: "level-4".to_owned(),
                created_at_ms: 900,
                expires_at_ms: 10_000,
                origin_authenticated: true,
            },
            payload_bytes: 500,
            attempt_succeeds: vec![true, true],
        },
        QueuedTransmission {
            message: Message {
                id: "SOS-AIRTIME".to_owned(),
                class: MessageClass::Distress,
                source: "surface".to_owned(),
                destination: "level-4".to_owned(),
                created_at_ms: 900,
                expires_at_ms: 10_000,
                origin_authenticated: true,
            },
            payload_bytes: 20,
            attempt_succeeds: vec![false, true],
        },
    ];
    match simulate_airtime_queue(&airtime_profile, &airtime_queue, 1_000) {
        Ok(sim) => {
            println!(
                "case=airtime_alarm_storm channel={} total_airtime_ms={} finished_at_ms={}",
                sim.profile_id, sim.total_airtime_ms, sim.finished_at_ms
            );
            for result in sim.results {
                println!(
                    "airtime_packet={} class={:?} disposition={:?} attempts={} queue_wait_ms={:?} airtime_ms={} received_at_ms={:?}",
                    result.message_id,
                    result.class,
                    result.disposition,
                    result.attempts,
                    result.queue_wait_ms,
                    result.airtime_ms,
                    result.received_at_ms
                );
            }
        }
        Err(error) => println!("case=airtime_alarm_storm error={error}"),
    }

    println!("claim_ceiling=synthetic routing semantics only; no radio, mine-safety, or field-reliability claim");
}
