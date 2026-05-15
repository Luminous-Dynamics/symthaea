// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Threshold Signing Coordinator Zome
//!
//! Business logic for DKG-based threshold signatures on governance decisions.
//!
//! Workflow:
//! 1. Create committee with threshold and member count
//! 2. Members register with their K-Vector trust scores
//! 3. Members run DKG ceremony off-chain using feldman-dkg
//! 4. Public commitments are submitted to advance ceremony
//! 5. Once complete, members can collectively sign governance decisions
//! 6. Threshold signatures are verified and stored

use hdk::prelude::*;
use mycelix_zome_helpers as _;
use threshold_signing_integrity::*;

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ThresholdSignal {
    CommitteeCreated {
        committee_id: String,
    },
    MemberRegistered {
        committee_id: String,
        member_did: String,
    },
    CeremonyAdvanced {
        committee_id: String,
        stage: u8,
    },
    DecisionSigned {
        decision_hash: ActionHash,
    },
}

// ── Committee Management ─────────────────────────────────────────────────────

/// Input for creating a new signing committee.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCommitteeInput {
    pub committee_id: String,
    pub name: String,
    pub threshold: u32,
    pub member_count: u32,
    pub scope: CommitteeScope,
    pub min_phi: Option<f64>,
    pub signature_algorithm: ThresholdSignatureAlgorithm,
    pub pq_required: bool,
}

#[hdk_extern]
pub fn create_committee(input: CreateCommitteeInput) -> ExternResult<Record> {
    let committee = SigningCommittee {
        id: input.committee_id.clone(),
        name: input.name,
        threshold: input.threshold,
        member_count: input.member_count,
        phase: DkgPhase::Registration,
        public_key: None,
        commitments: Vec::new(),
        scope: input.scope,
        created_at: sys_time()?,
        active: true,
        epoch: 1,
        min_phi: input.min_phi,
        signature_algorithm: input.signature_algorithm,
        pq_required: input.pq_required,
    };

    let action_hash = create_entry(EntryTypes::SigningCommittee(committee))?;

    let _ = emit_signal(ThresholdSignal::CommitteeCreated {
        committee_id: input.committee_id,
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Committee record not found".into()
    )))
}
