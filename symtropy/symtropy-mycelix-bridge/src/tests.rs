// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unit tests that exercise the bridge plumbing without a live conductor.
//!
//! Tests that need a real Holochain conductor are gated behind the
//! `live-conductor` feature (not yet in Milestone 1).

use bevy::prelude::*;
use flume::bounded;

use crate::config::MycelixConfig;
use crate::events::{MycelixRequest, MycelixResponse};
use crate::resource::{MycelixClient, MycelixResponseInbox, MycelixSendError};
use crate::systems::pump_responses;

#[test]
fn config_default_hits_local_shared_conductor() {
    let cfg = MycelixConfig::default();
    assert_eq!(cfg.conductor_url, "ws://localhost:8888");
    assert_eq!(cfg.app_id, "mycelix-governance");
    assert_eq!(cfg.role, "governance");
    assert_eq!(cfg.inflight_budget, 128);
    assert_eq!(
        cfg.bridge_binary.to_string_lossy(),
        "mycelix-conductor-bridge"
    );
}

#[test]
fn config_builder_overrides_conductor_url() {
    let cfg = MycelixConfig::default().with_conductor_url("ws://example.com:9000");
    assert_eq!(cfg.conductor_url, "ws://example.com:9000");
}

#[test]
fn config_builder_overrides_bridge_binary() {
    let cfg = MycelixConfig::default().with_bridge_binary("/opt/custom/bridge");
    assert_eq!(cfg.bridge_binary.to_string_lossy(), "/opt/custom/bridge");
}

#[test]
fn response_requester_extracts_entity() {
    let e = Entity::PLACEHOLDER;
    let success = MycelixResponse::ActiveProposals {
        requester: e,
        proposals: vec![],
    };
    let error = MycelixResponse::Error {
        requester: e,
        reason: "boom".to_string(),
    };
    assert_eq!(success.requester(), e);
    assert_eq!(error.requester(), e);
}

#[test]
fn client_send_succeeds_with_room_in_channel() {
    let (tx, rx) = bounded(4);
    let client = MycelixClient::new(tx);
    client
        .send(MycelixRequest::GetActiveProposals {
            requester: Entity::PLACEHOLDER,
        })
        .expect("should succeed with empty channel");
    assert_eq!(rx.len(), 1);
}

#[test]
fn client_send_returns_full_when_at_capacity() {
    let (tx, _rx) = bounded(1);
    let client = MycelixClient::new(tx);
    client
        .send(MycelixRequest::GetActiveProposals {
            requester: Entity::PLACEHOLDER,
        })
        .unwrap();
    match client.send(MycelixRequest::GetActiveProposals {
        requester: Entity::PLACEHOLDER,
    }) {
        Err(MycelixSendError::Full) => {}
        other => panic!("expected Full, got {other:?}"),
    }
}

#[test]
fn client_send_returns_disconnected_when_rx_dropped() {
    let (tx, rx) = bounded::<MycelixRequest>(4);
    drop(rx);
    let client = MycelixClient::new(tx);
    match client.send(MycelixRequest::GetActiveProposals {
        requester: Entity::PLACEHOLDER,
    }) {
        Err(MycelixSendError::Disconnected) => {}
        other => panic!("expected Disconnected, got {other:?}"),
    }
}

#[test]
fn pump_responses_drains_inbox_into_event_stream() {
    let (resp_tx, resp_rx) = bounded(4);
    let mut app = App::new();
    app.add_message::<MycelixResponse>()
        .insert_resource(MycelixResponseInbox { rx: resp_rx })
        .add_systems(Update, pump_responses);

    // Push two responses into the inbox before running a schedule tick.
    resp_tx
        .send(MycelixResponse::ActiveProposals {
            requester: Entity::PLACEHOLDER,
            proposals: vec![],
        })
        .unwrap();
    resp_tx
        .send(MycelixResponse::Error {
            requester: Entity::PLACEHOLDER,
            reason: "test".to_string(),
        })
        .unwrap();

    app.update();

    let events = app.world().resource::<Messages<MycelixResponse>>();
    assert_eq!(events.len(), 2, "both inbox items should be emitted");
}

#[test]
fn pump_responses_is_idempotent_when_inbox_empty() {
    let (_resp_tx, resp_rx) = bounded::<MycelixResponse>(4);
    let mut app = App::new();
    app.add_message::<MycelixResponse>()
        .insert_resource(MycelixResponseInbox { rx: resp_rx })
        .add_systems(Update, pump_responses);

    app.update();

    let events = app.world().resource::<Messages<MycelixResponse>>();
    assert_eq!(events.len(), 0);
}

// ---------------------------------------------------------------------------
// Milestone 2 — new request / response variants + requester() exhaustive
// ---------------------------------------------------------------------------

#[test]
fn request_requester_extracts_entity_from_every_variant() {
    let e = Entity::PLACEHOLDER;
    let variants = [
        MycelixRequest::GetActiveProposals { requester: e },
        MycelixRequest::SubmitProposal {
            requester: e,
            proposal_id: "MIP-001".to_string(),
            title: "T".to_string(),
            description: "D".to_string(),
            author_did: "did:key:z6Mk".to_string(),
        },
        MycelixRequest::CastVote {
            requester: e,
            proposal_id: "MIP-001".to_string(),
            voter_did: "did:key:z6Mk".to_string(),
            approve: true,
            rationale: String::new(),
        },
        MycelixRequest::QueryTendBalance {
            requester: e,
            member_did: "did:key:z6Mk".to_string(),
        },
    ];
    for v in variants {
        assert_eq!(v.requester(), e);
    }
}

#[test]
fn response_requester_extracts_entity_from_every_variant() {
    let e = Entity::PLACEHOLDER;
    let variants = [
        MycelixResponse::ActiveProposals {
            requester: e,
            proposals: vec![],
        },
        MycelixResponse::ProposalSubmitted {
            requester: e,
            action_hash: "hash".to_string(),
        },
        MycelixResponse::VoteCast {
            requester: e,
            proposal_id: "MIP-001".to_string(),
        },
        MycelixResponse::TendBalance {
            requester: e,
            balance: serde_json::json!({ "balance": 0 }),
        },
        MycelixResponse::Error {
            requester: e,
            reason: "boom".to_string(),
        },
    ];
    for v in variants {
        assert_eq!(v.requester(), e);
    }
}

#[test]
fn client_send_accepts_all_variants() {
    let (tx, rx) = bounded(8);
    let client = MycelixClient::new(tx);
    client
        .send(MycelixRequest::GetActiveProposals {
            requester: Entity::PLACEHOLDER,
        })
        .unwrap();
    client
        .send(MycelixRequest::SubmitProposal {
            requester: Entity::PLACEHOLDER,
            proposal_id: "MIP-001".to_string(),
            title: "T".to_string(),
            description: "D".to_string(),
            author_did: "did:key:z6Mk".to_string(),
        })
        .unwrap();
    client
        .send(MycelixRequest::CastVote {
            requester: Entity::PLACEHOLDER,
            proposal_id: "MIP-001".to_string(),
            voter_did: "did:key:z6Mk".to_string(),
            approve: true,
            rationale: "".to_string(),
        })
        .unwrap();
    client
        .send(MycelixRequest::QueryTendBalance {
            requester: Entity::PLACEHOLDER,
            member_did: "did:key:z6Mk".to_string(),
        })
        .unwrap();
    assert_eq!(rx.len(), 4);
}
