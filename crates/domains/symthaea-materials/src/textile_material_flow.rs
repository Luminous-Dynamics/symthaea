// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent lot-lineage and material-conservation contract.
//!
//! A single recovery-route assessment qualifies one physical lot. This module
//! checks the transformations that connect many such lots through sorting,
//! split/merge, repair, disassembly and recycling. Every input lot is consumed
//! at most once; splitting is explicit; outputs have a single producer; the
//! lineage graph is acyclic; and each event accounts for its incoming mass.
//!
//! This is a declared-ledger verifier. It does not authenticate the evidence
//! artifacts, establish chemical purity, or prove that a physical event occurred.

use crate::textile_recovery::{
    AcceptanceOutcome, DownstreamAcceptance, EvidenceKind, EvidenceRef, EvidenceTier,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Integer scale for mass fractions: one million parts represent 100%.
pub const MATERIAL_FLOW_FRACTION_SCALE_PPM: u64 = 1_000_000;

/// Physical or process identity for one immutable material lot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLot {
    /// Stable lot identifier; never reuse an identifier after transformation.
    pub lot_id: String,
    /// Measured lot mass in integer milligrams.
    pub mass_mg: u64,
    /// Declared material family or composition label; not a composition measurement.
    pub material_class: String,
    /// Incoming inventory or the exact event that created this lot.
    pub origin: LotOrigin,
    /// Current declared disposition of this lot.
    pub disposition: LotDisposition,
    /// Required only when a receiver has issued a final accepted/rejected decision.
    pub receiver_acceptance: Option<DownstreamAcceptance>,
}

/// Declares exactly one producer for every lot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LotOrigin {
    /// Material entering the ledger from outside the measured process boundary.
    Incoming {
        /// Source inventory evidence for this exact lot and mass.
        inventory_evidence: EvidenceRef,
    },
    /// Lot created by exactly one transformation event in this ledger.
    EventOutput {
        /// Exact event identifier that created the lot.
        event_id: String,
    },
}

/// Disposition of a lot after its current operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LotDisposition {
    /// Available for exactly one later transformation event.
    InProcess,
    /// Terminal in this ledger but awaiting a receiving-process decision.
    AwaitingReceiver,
    /// Terminal and accepted by the declared independent receiver.
    Accepted,
    /// Terminal and rejected by the declared independent receiver.
    Rejected,
    /// Terminal and quarantined; not counted as recovered output.
    Quarantined,
    /// Terminal residual/waste stream; not counted as recovered output.
    Waste,
}

/// Physical process which transforms one or more complete input lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowProcess {
    /// Physical sorting and routing based on admitted material observations.
    Sorting,
    /// Physical repair or refurbishment of a textile item/lot.
    Repair,
    /// Physical separation of fasteners, trims or reusable components.
    ComponentDisassembly,
    /// Physical opening of textiles into recovered fibre/material.
    MechanicalOpening,
    /// Physical chemical-recycling process.
    ChemicalRecycling,
    /// Physical production of an explicitly lower-grade output stream.
    Downcycling,
}

impl FlowProcess {
    fn expected_evidence_kind(self) -> EvidenceKind {
        match self {
            Self::Sorting => EvidenceKind::SortingTrial,
            Self::Repair => EvidenceKind::RepairTrial,
            Self::ComponentDisassembly => EvidenceKind::ComponentRecoveryTrial,
            Self::MechanicalOpening => EvidenceKind::MechanicalRecyclingTrial,
            Self::ChemicalRecycling => EvidenceKind::ChemicalRecyclingTrial,
            Self::Downcycling => EvidenceKind::DowncyclingTrial,
        }
    }
}

/// One mass-conserving transformation over complete input/output lots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialFlowEvent {
    /// Stable event identifier.
    pub event_id: String,
    /// Declared physical process.
    pub process: FlowProcess,
    /// One or more complete lots consumed by this event.
    pub input_lot_ids: Vec<String>,
    /// One or more newly-created lots emitted by this event.
    pub output_lot_ids: Vec<String>,
    /// Explicit unaccounted residual, in milligrams. Never counted as accepted output.
    pub unaccounted_mass_mg: u64,
    /// Expanded uncertainty for the event's complete mass balance, in milligrams.
    pub uncertainty_mass_mg: u64,
    /// Maximum unaccounted residual as parts per million of event input mass.
    pub max_unaccounted_ppm: u32,
    /// Maximum expanded uncertainty as parts per million of event input mass.
    pub max_uncertainty_ppm: u32,
    /// Physical process evidence. Must identify the exact process type and be PhysicalTrial.
    pub evidence: EvidenceRef,
}

/// Complete directed material-flow ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialFlowLedger {
    /// Versioned ledger identifier.
    pub ledger_id: String,
    /// Immutable lots, including incoming and all event-produced lots.
    pub lots: Vec<MaterialLot>,
    /// Physical transformation events.
    pub events: Vec<MaterialFlowEvent>,
    /// Explicit claim boundary for the resulting report.
    pub claim_ceiling: String,
}

/// Deterministic material-flow validation finding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialFlowFinding {
    /// Stable finding code.
    pub code: String,
    /// Lot or event identifier to which the finding applies.
    pub entity_id: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Totals reported from the ledger. Intermediate masses are never summed as
/// terminal outputs, preventing multi-stage output double counting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialFlowTotals {
    /// Sum of source/incoming lot masses.
    pub incoming_mass_mg: u128,
    /// Sum of terminal lots with independent receiver acceptance.
    pub accepted_mass_mg: u128,
    /// Sum of terminal lots explicitly rejected by a receiver.
    pub rejected_mass_mg: u128,
    /// Sum of terminal lots explicitly declared as waste.
    pub waste_mass_mg: u128,
    /// Sum of terminal lots quarantined and not counted as recovered output.
    pub quarantined_mass_mg: u128,
    /// Sum of terminal lots awaiting a receiver decision.
    pub awaiting_receiver_mass_mg: u128,
    /// Sum of event-declared unaccounted residual mass.
    pub unaccounted_mass_mg: u128,
    /// Sum of declared event mass-measurement uncertainties.
    pub uncertainty_mass_mg: u128,
}

/// Result of validating a material-flow ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialFlowReport {
    /// True only when every structural, evidence, chronology and mass gate passes.
    pub valid: bool,
    /// Ledger identity.
    pub ledger_id: String,
    /// Number of incoming lots.
    pub incoming_lot_count: usize,
    /// Number of transformation events.
    pub event_count: usize,
    /// Number of terminal lots.
    pub terminal_lot_count: usize,
    /// Aggregate terminal/source totals.
    pub totals: MaterialFlowTotals,
    /// Deterministically sorted findings.
    pub findings: Vec<MaterialFlowFinding>,
}

fn add(
    findings: &mut Vec<MaterialFlowFinding>,
    code: &'static str,
    entity_id: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(MaterialFlowFinding {
        code: code.to_owned(),
        entity_id: entity_id.into(),
        message: message.into(),
    });
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_evidence_ref(evidence: &EvidenceRef) -> bool {
    valid_digest(&evidence.sha256)
        && !evidence.evidence_id.trim().is_empty()
        && !evidence.method_ref.trim().is_empty()
        && !evidence.issuer_id.trim().is_empty()
}

fn is_terminal(disposition: LotDisposition) -> bool {
    disposition != LotDisposition::InProcess
}

fn add_total(target: &mut u128, value: u64) -> bool {
    match target.checked_add(u128::from(value)) {
        Some(sum) => {
            *target = sum;
            true
        }
        None => false,
    }
}

/// Validate lot identity, exact producers/consumers, lineage DAG, timestamps,
/// receiver acceptance and per-event mass conservation.
pub fn validate_material_flow(ledger: &MaterialFlowLedger) -> MaterialFlowReport {
    let mut findings = Vec::new();
    let mut totals = MaterialFlowTotals {
        incoming_mass_mg: 0,
        accepted_mass_mg: 0,
        rejected_mass_mg: 0,
        waste_mass_mg: 0,
        quarantined_mass_mg: 0,
        awaiting_receiver_mass_mg: 0,
        unaccounted_mass_mg: 0,
        uncertainty_mass_mg: 0,
    };

    if ledger.ledger_id.trim().is_empty() {
        add(
            &mut findings,
            "LEDGER_ID_MISSING",
            "<ledger>",
            "material-flow ledger must have a stable non-empty identity",
        );
    }
    if ledger.claim_ceiling.trim().is_empty() {
        add(
            &mut findings,
            "CLAIM_CEILING_MISSING",
            &ledger.ledger_id,
            "material-flow ledger must state what its report does and does not qualify",
        );
    }

    let mut events = BTreeMap::<&str, &MaterialFlowEvent>::new();
    for event in &ledger.events {
        match events.entry(event.event_id.as_str()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(event);
            }
            std::collections::btree_map::Entry::Occupied(_) => add(
                &mut findings,
                "DUPLICATE_EVENT_ID",
                &event.event_id,
                "event IDs must be unique",
            ),
        }
    }

    let mut lots = BTreeMap::<&str, &MaterialLot>::new();
    for lot in &ledger.lots {
        if lot.lot_id.trim().is_empty() || lot.mass_mg == 0 || lot.material_class.trim().is_empty() {
            add(
                &mut findings,
                "LOT_IDENTITY_OR_MASS_INVALID",
                &lot.lot_id,
                "lot ID, positive mass and material class are required",
            );
        }
        if lots.insert(lot.lot_id.as_str(), lot).is_some() {
            add(
                &mut findings,
                "DUPLICATE_LOT_ID",
                &lot.lot_id,
                "lot IDs are immutable and unique within one ledger",
            );
        }
    }


    let mut evidence_ids = BTreeSet::<&str>::new();
    let mut evidence_digests = BTreeSet::<&str>::new();
    let mut incoming_lot_count = 0_usize;
    let mut terminal_lot_count = 0_usize;
    let mut consumer_by_lot = BTreeMap::<&str, &str>::new();

    for lot in &ledger.lots {
        match &lot.origin {
            LotOrigin::Incoming { inventory_evidence } => {
                incoming_lot_count += 1;
                if !valid_evidence_ref(inventory_evidence)
                    || inventory_evidence.kind != EvidenceKind::PhysicalInventory
                    || inventory_evidence.tier != EvidenceTier::SensorMeasured
                {
                    add(
                        &mut findings,
                        "INCOMING_INVENTORY_EVIDENCE_INVALID",
                        &lot.lot_id,
                        "incoming lot requires source-bound SensorMeasured PhysicalInventory evidence",
                    );
                }
                if !evidence_ids.insert(inventory_evidence.evidence_id.as_str()) {
                    add(
                        &mut findings,
                        "DUPLICATE_EVIDENCE_ID",
                        &lot.lot_id,
                        "evidence IDs must be unique across inventory, events and receiver decisions",
                    );
                }
                if !evidence_digests.insert(inventory_evidence.sha256.as_str()) {
                    add(
                        &mut findings,
                        "DUPLICATE_EVIDENCE_DIGEST",
                        &lot.lot_id,
                        "distinct evidence records cannot reuse the same source bytes",
                    );
                }
                if !add_total(&mut totals.incoming_mass_mg, lot.mass_mg) {
                    add(
                        &mut findings,
                        "MASS_TOTAL_OVERFLOW",
                        &lot.lot_id,
                        "incoming mass total overflows u128",
                    );
                }
            }
            LotOrigin::EventOutput { event_id } => {
                if event_id.trim().is_empty() {
                    add(
                        &mut findings,
                        "LOT_PRODUCER_ID_MISSING",
                        &lot.lot_id,
                        "event-produced lot must identify its producer event",
                    );
                }
            }
        }

        if is_terminal(lot.disposition) {
            terminal_lot_count += 1;
        }
        match lot.disposition {
            LotDisposition::Accepted | LotDisposition::Rejected => {
                let Some(acceptance) = lot.receiver_acceptance.as_ref() else {
                    add(
                        &mut findings,
                        "RECEIVER_DECISION_MISSING",
                        &lot.lot_id,
                        "accepted/rejected terminal lot requires an independent lot-specific receiver decision",
                    );
                    continue;
                };
                let mut receiver_valid = true;
                if !valid_evidence_ref(&acceptance.evidence)
                    || acceptance.evidence.kind != EvidenceKind::DownstreamAcceptance
                    || acceptance.evidence.tier != EvidenceTier::IndependentAcceptance
                    || acceptance.evidence.verifier_id.as_deref().is_none_or(|id| {
                        id.trim().is_empty() || id == acceptance.evidence.issuer_id
                    })
                    || acceptance.specification_id.trim().is_empty()
                    || acceptance.accepted_output_lot_id != lot.lot_id
                {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_DECISION_INVALID",
                        &lot.lot_id,
                        "receiver evidence must be independent, lot-specific and bound to a versioned receiving specification",
                    );
                }
                let expected_outcome = match lot.disposition {
                    LotDisposition::Accepted => AcceptanceOutcome::Accepted,
                    LotDisposition::Rejected => AcceptanceOutcome::Rejected,
                    _ => unreachable!(),
                };
                if acceptance.outcome != expected_outcome {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_DISPOSITION_MISMATCH",
                        &lot.lot_id,
                        "lot disposition does not match the receiver's exact acceptance outcome",
                    );
                }
                let (basis_id, basis_issuer, basis_digest, basis_time) = match &lot.origin {
                    LotOrigin::Incoming { inventory_evidence } => (
                        inventory_evidence.evidence_id.as_str(),
                        inventory_evidence.issuer_id.as_str(),
                        inventory_evidence.sha256.as_str(),
                        inventory_evidence.observed_at_ns,
                    ),
                    LotOrigin::EventOutput { event_id } => events
                        .get(event_id.as_str())
                        .map(|event| (
                            event.evidence.evidence_id.as_str(),
                            event.evidence.issuer_id.as_str(),
                            event.evidence.sha256.as_str(),
                            event.evidence.observed_at_ns,
                        ))
                        .unwrap_or(("", "", "", u64::MAX)),
                };
                if acceptance.basis_evidence_id != basis_id {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_BASIS_EVIDENCE_MISMATCH",
                        &lot.lot_id,
                        "receiver decision must bind to the exact inventory or process-trial evidence that produced this lot",
                    );
                }
                if acceptance.evidence.issuer_id == basis_issuer {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_NOT_INDEPENDENT_OF_PROCESS",
                        &lot.lot_id,
                        "receiver organization must be independent of the incoming inventory or process-trial issuer",
                    );
                }
                if acceptance.evidence.verifier_id.as_deref() == Some(basis_issuer) {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_VERIFIER_NOT_INDEPENDENT_OF_PROCESS",
                        &lot.lot_id,
                        "independent verifier must also be distinct from the incoming inventory or process-trial issuer",
                    );
                }
                if acceptance.evidence.sha256 == basis_digest {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_EVIDENCE_REUSES_BASIS_BYTES",
                        &lot.lot_id,
                        "receiver acceptance must be a distinct evidence artifact from the process/inventory evidence",
                    );
                }
                if acceptance.evidence.observed_at_ns < basis_time {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "RECEIVER_DECISION_PRECEDES_OUTPUT",
                        &lot.lot_id,
                        "receiver decision cannot precede the exact inventory or process evidence that produced the lot",
                    );
                }
                if !evidence_ids.insert(acceptance.evidence.evidence_id.as_str()) {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "DUPLICATE_EVIDENCE_ID",
                        &lot.lot_id,
                        "receiver acceptance must have its own unique evidence ID",
                    );
                }
                if !evidence_digests.insert(acceptance.evidence.sha256.as_str()) {
                    receiver_valid = false;
                    add(
                        &mut findings,
                        "DUPLICATE_EVIDENCE_DIGEST",
                        &lot.lot_id,
                        "receiver acceptance must reference distinct source bytes",
                    );
                }
                if receiver_valid
                    && expected_outcome == AcceptanceOutcome::Accepted
                    && !add_total(&mut totals.accepted_mass_mg, lot.mass_mg)
                {
                    add(
                        &mut findings,
                        "MASS_TOTAL_OVERFLOW",
                        &lot.lot_id,
                        "accepted mass total overflows u128",
                    );
                }
                if receiver_valid
                    && expected_outcome == AcceptanceOutcome::Rejected
                    && !add_total(&mut totals.rejected_mass_mg, lot.mass_mg)
                {
                    add(
                        &mut findings,
                        "MASS_TOTAL_OVERFLOW",
                        &lot.lot_id,
                        "rejected mass total overflows u128",
                    );
                }
            }
            LotDisposition::AwaitingReceiver => {
                if lot.receiver_acceptance.is_some() {
                    add(
                        &mut findings,
                        "UNRESOLVED_LOT_HAS_RECEIVER_DECISION",
                        &lot.lot_id,
                        "awaiting-receiver lot cannot already contain a final receiver decision",
                    );
                }
                if !add_total(&mut totals.awaiting_receiver_mass_mg, lot.mass_mg) {
                    add(
                        &mut findings,
                        "MASS_TOTAL_OVERFLOW",
                        &lot.lot_id,
                        "awaiting-receiver mass total overflows u128",
                    );
                }
            }
            LotDisposition::Waste => {
                if lot.receiver_acceptance.is_some() {
                    add(
                        &mut findings,
                        "WASTE_LOT_HAS_RECEIVER_DECISION",
                        &lot.lot_id,
                        "waste disposition cannot also be counted as receiver-accepted/rejected product",
                    );
                }
                if !add_total(&mut totals.waste_mass_mg, lot.mass_mg) {
                    add(
                        &mut findings,
                        "MASS_TOTAL_OVERFLOW",
                        &lot.lot_id,
                        "waste mass total overflows u128",
                    );
                }
            }
            LotDisposition::Quarantined => {
                if lot.receiver_acceptance.is_some() {
                    add(
                        &mut findings,
                        "NONTERMINAL_LOT_HAS_RECEIVER_DECISION",
                        &lot.lot_id,
                        "quarantined lots cannot carry final receiver acceptance",
                    );
                }
                if !add_total(&mut totals.quarantined_mass_mg, lot.mass_mg) {
                    add(
                        &mut findings,
                        "MASS_TOTAL_OVERFLOW",
                        &lot.lot_id,
                        "quarantined mass total overflows u128",
                    );
                }
            }
            LotDisposition::InProcess => {
                if lot.receiver_acceptance.is_some() {
                    add(
                        &mut findings,
                        "NONTERMINAL_LOT_HAS_RECEIVER_DECISION",
                        &lot.lot_id,
                        "in-process lots cannot carry final receiver acceptance",
                    );
                }
            }
        }
    }

    for event in &ledger.events {
        if event.event_id.trim().is_empty() {
            add(
                &mut findings,
                "EVENT_ID_MISSING",
                "<event>",
                "event ID must be non-empty",
            );
        }
        if event.input_lot_ids.is_empty() || event.output_lot_ids.is_empty() {
            add(
                &mut findings,
                "EVENT_INPUT_OR_OUTPUT_EMPTY",
                &event.event_id,
                "each transformation must consume at least one complete lot and produce at least one new lot",
            );
        }
        if !valid_evidence_ref(&event.evidence)
            || event.evidence.tier != EvidenceTier::PhysicalTrial
            || event.evidence.kind != event.process.expected_evidence_kind()
        {
            add(
                &mut findings,
                "TRANSFORMATION_EVIDENCE_INVALID",
                &event.event_id,
                "transformation requires PhysicalTrial evidence with the exact process evidence kind",
            );
        }
        if !evidence_ids.insert(event.evidence.evidence_id.as_str()) {
            add(
                &mut findings,
                "DUPLICATE_EVIDENCE_ID",
                &event.event_id,
                "event evidence ID is duplicated within the ledger",
            );
        }
        if !evidence_digests.insert(event.evidence.sha256.as_str()) {
            add(
                &mut findings,
                "DUPLICATE_EVIDENCE_DIGEST",
                &event.event_id,
                "distinct process events cannot reuse the same source bytes",
            );
        }
        if event.max_unaccounted_ppm > MATERIAL_FLOW_FRACTION_SCALE_PPM as u32
            || event.max_uncertainty_ppm > MATERIAL_FLOW_FRACTION_SCALE_PPM as u32
        {
            add(
                &mut findings,
                "EVENT_POLICY_INVALID",
                &event.event_id,
                "unaccounted and uncertainty ppm ceilings must be between zero and one million",
            );
        }

        let mut input_total = 0_u128;
        let mut seen_inputs = BTreeSet::new();
        for input_id in &event.input_lot_ids {
            if !seen_inputs.insert(input_id.as_str()) {
                add(
                    &mut findings,
                    "EVENT_INPUT_DUPLICATE",
                    &event.event_id,
                    format!("input lot {input_id} is listed more than once"),
                );
                continue;
            }
            let Some(lot) = lots.get(input_id.as_str()) else {
                add(
                    &mut findings,
                    "EVENT_INPUT_UNKNOWN",
                    &event.event_id,
                    format!("input lot {input_id} is absent from the ledger"),
                );
                continue;
            };
            if consumer_by_lot.insert(input_id.as_str(), event.event_id.as_str()).is_some() {
                add(
                    &mut findings,
                    "LOT_CONSUMED_MORE_THAN_ONCE",
                    input_id,
                    "a lot must be split into explicit child lots before more than one downstream use",
                );
            }
            if lot.disposition != LotDisposition::InProcess {
                add(
                    &mut findings,
                    "TERMINAL_LOT_CONSUMED",
                    input_id,
                    "terminal, accepted, rejected, waste or quarantined lot cannot be consumed by another event",
                );
            }
            match input_total.checked_add(u128::from(lot.mass_mg)) {
                Some(total) => input_total = total,
                None => add(
                    &mut findings,
                    "EVENT_MASS_OVERFLOW",
                    &event.event_id,
                    "event input mass total overflows u128",
                ),
            }
            if let LotOrigin::Incoming { inventory_evidence } = &lot.origin {
                if inventory_evidence.observed_at_ns > event.evidence.observed_at_ns {
                    add(
                        &mut findings,
                        "EVENT_PRECEDES_INPUT_INVENTORY",
                        &event.event_id,
                        format!("event occurs before incoming inventory for lot {input_id}"),
                    );
                }
            } else if let LotOrigin::EventOutput { event_id } = &lot.origin {
                if let Some(producer) = events.get(event_id.as_str()) {
                    if producer.evidence.observed_at_ns > event.evidence.observed_at_ns {
                        add(
                            &mut findings,
                            "EVENT_LINEAGE_TIME_REVERSAL",
                            &event.event_id,
                            format!("event occurs before producer {event_id} for lot {input_id}"),
                        );
                    }
                }
            }
        }
        let mut output_total = 0_u128;
        let mut seen_outputs = BTreeSet::new();
        for output_id in &event.output_lot_ids {
            if !seen_outputs.insert(output_id.as_str()) {
                add(
                    &mut findings,
                    "EVENT_OUTPUT_DUPLICATE",
                    &event.event_id,
                    format!("output lot {output_id} is listed more than once"),
                );
                continue;
            }
            let Some(lot) = lots.get(output_id.as_str()) else {
                add(
                    &mut findings,
                    "EVENT_OUTPUT_UNKNOWN",
                    &event.event_id,
                    format!("output lot {output_id} is absent from the ledger"),
                );
                continue;
            };
            match &lot.origin {
                LotOrigin::EventOutput { event_id } if event_id == &event.event_id => {}
                _ => add(
                    &mut findings,
                    "EVENT_OUTPUT_PRODUCER_MISMATCH",
                    output_id,
                    "output lot origin must point to exactly this event",
                ),
            }
            if lot.disposition != LotDisposition::InProcess && consumer_by_lot.contains_key(output_id.as_str()) {
                add(
                    &mut findings,
                    "TERMINAL_OUTPUT_CONSUMED",
                    output_id,
                    "terminal event output cannot also be consumed downstream",
                );
            }
            match output_total.checked_add(u128::from(lot.mass_mg)) {
                Some(total) => output_total = total,
                None => add(
                    &mut findings,
                    "EVENT_MASS_OVERFLOW",
                    &event.event_id,
                    "event output mass total overflows u128",
                ),
            }
        }
        let accounted = output_total
            .checked_add(u128::from(event.unaccounted_mass_mg))
            .unwrap_or(u128::MAX);
        let difference = input_total.abs_diff(accounted);
        if difference > u128::from(event.uncertainty_mass_mg) {
            add(
                &mut findings,
                "EVENT_MASS_BALANCE_NOT_CLOSED",
                &event.event_id,
                format!(
                    "input mass {input_total} mg differs from output plus unaccounted {accounted} mg by {difference} mg, above uncertainty {} mg",
                    event.uncertainty_mass_mg
                ),
            );
        }
        if input_total == 0 {
            add(
                &mut findings,
                "EVENT_INPUT_MASS_ZERO",
                &event.event_id,
                "event input mass must be positive",
            );
        } else {
            let unaccounted_ppm = u128::from(event.unaccounted_mass_mg)
                .saturating_mul(u128::from(MATERIAL_FLOW_FRACTION_SCALE_PPM))
                / input_total;
            if unaccounted_ppm > u128::from(event.max_unaccounted_ppm) {
                add(
                    &mut findings,
                    "UNACCOUNTED_MASS_ABOVE_POLICY",
                    &event.event_id,
                    format!(
                        "unaccounted mass fraction {unaccounted_ppm} ppm exceeds declared ceiling {} ppm",
                        event.max_unaccounted_ppm
                    ),
                );
            }
            let uncertainty_ppm = u128::from(event.uncertainty_mass_mg)
                .saturating_mul(u128::from(MATERIAL_FLOW_FRACTION_SCALE_PPM))
                / input_total;
            if uncertainty_ppm > u128::from(event.max_uncertainty_ppm) {
                add(
                    &mut findings,
                    "MASS_UNCERTAINTY_ABOVE_POLICY",
                    &event.event_id,
                    format!(
                        "mass uncertainty {uncertainty_ppm} ppm exceeds declared ceiling {} ppm",
                        event.max_uncertainty_ppm
                    ),
                );
            }
        }
        if !add_total(&mut totals.unaccounted_mass_mg, event.unaccounted_mass_mg) {
            add(
                &mut findings,
                "MASS_TOTAL_OVERFLOW",
                &event.event_id,
                "unaccounted mass total overflows u128",
            );
        }
        if !add_total(&mut totals.uncertainty_mass_mg, event.uncertainty_mass_mg) {
            add(
                &mut findings,
                "MASS_TOTAL_OVERFLOW",
                &event.event_id,
                "mass uncertainty total overflows u128",
            );
        }
    }

    for lot in &ledger.lots {
        match &lot.origin {
            LotOrigin::EventOutput { event_id } => {
                let Some(event) = events.get(event_id.as_str()) else {
                    add(
                        &mut findings,
                        "LOT_PRODUCER_UNKNOWN",
                        &lot.lot_id,
                        format!("producer event {event_id} is absent"),
                    );
                    continue;
                };
                if !event.output_lot_ids.iter().any(|id| id == &lot.lot_id) {
                    add(
                        &mut findings,
                        "LOT_NOT_LISTED_BY_PRODUCER",
                        &lot.lot_id,
                        "lot points to a producer event that does not list it as output",
                    );
                }
            }
            LotOrigin::Incoming { .. } => {}
        }
        let consumed = consumer_by_lot.contains_key(lot.lot_id.as_str());
        if consumed && is_terminal(lot.disposition) {
            add(
                &mut findings,
                "CONSUMED_LOT_MARKED_TERMINAL",
                &lot.lot_id,
                "a lot consumed by another event must remain InProcess in its source record",
            );
        }
        if !consumed && lot.disposition == LotDisposition::InProcess {
            add(
                &mut findings,
                "UNRESOLVED_UNCONSUMED_LOT",
                &lot.lot_id,
                "every unconsumed lot must have an explicit terminal disposition",
            );
        }
    }

    // Build the event dependency graph from produced input lots and prove it is acyclic.
    let mut indegree = BTreeMap::<&str, usize>::new();
    let mut edges = BTreeMap::<&str, BTreeSet<&str>>::new();
    for event in &ledger.events {
        indegree.entry(event.event_id.as_str()).or_insert(0);
    }
    for event in &ledger.events {
        for input_id in &event.input_lot_ids {
            if let Some(lot) = lots.get(input_id.as_str()) {
                if let LotOrigin::EventOutput { event_id: producer_id } = &lot.origin {
                    if events.contains_key(producer_id.as_str())
                        && edges.entry(producer_id.as_str()).or_default().insert(event.event_id.as_str())
                    {
                        *indegree.entry(event.event_id.as_str()).or_default() += 1;
                    }
                }
            }
        }
    }
    let mut queue: VecDeque<&str> = indegree
        .iter()
        .filter_map(|(event_id, degree)| (*degree == 0).then_some(*event_id))
        .collect();
    let mut visited = 0_usize;
    while let Some(event_id) = queue.pop_front() {
        visited += 1;
        if let Some(next) = edges.get(event_id) {
            for next_event in next {
                if let Some(degree) = indegree.get_mut(next_event) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        queue.push_back(next_event);
                    }
                }
            }
        }
    }
    if visited != indegree.len() {
        add(
            &mut findings,
            "EVENT_GRAPH_CYCLE",
            &ledger.ledger_id,
            "transformation lineage contains a cycle; process outputs cannot precede their producers",
        );
    }

    let terminal_mass = totals
        .accepted_mass_mg
        .checked_add(totals.rejected_mass_mg)
        .and_then(|value| value.checked_add(totals.waste_mass_mg))
        .and_then(|value| value.checked_add(totals.quarantined_mass_mg))
        .and_then(|value| value.checked_add(totals.awaiting_receiver_mass_mg));
    let ledger_accounted_mass = terminal_mass
        .and_then(|value| value.checked_add(totals.unaccounted_mass_mg));
    match ledger_accounted_mass {
        Some(accounted) => {
            let difference = totals.incoming_mass_mg.abs_diff(accounted);
            if difference > totals.uncertainty_mass_mg {
                add(
                    &mut findings,
                    "LEDGER_MASS_BALANCE_NOT_CLOSED",
                    &ledger.ledger_id,
                    format!(
                        "incoming mass {} mg differs from terminal outputs plus unaccounted residual {} mg by {} mg, above aggregate uncertainty {} mg",
                        totals.incoming_mass_mg,
                        accounted,
                        difference,
                        totals.uncertainty_mass_mg
                    ),
                );
            }
        }
        None => add(
            &mut findings,
            "LEDGER_MASS_TOTAL_OVERFLOW",
            &ledger.ledger_id,
            "terminal plus unaccounted material-flow mass overflows u128",
        ),
    }

    findings.sort_by(|left, right| {
        (&left.entity_id, left.code, &left.message).cmp(&(
            &right.entity_id,
            right.code,
            &right.message,
        ))
    });

    MaterialFlowReport {
        valid: findings.is_empty(),
        ledger_id: ledger.ledger_id.clone(),
        incoming_lot_count,
        event_count: ledger.events.len(),
        terminal_lot_count,
        totals,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn evidence(
        id: &str,
        kind: EvidenceKind,
        tier: EvidenceTier,
        issuer: &str,
        verifier: Option<&str>,
        sha: char,
        time: u64,
    ) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.to_owned(),
            sha256: digest(sha),
            kind,
            tier,
            method_ref: "textile-flow-method-v1".to_owned(),
            issuer_id: issuer.to_owned(),
            verifier_id: verifier.map(str::to_owned),
            observed_at_ns: time,
        }
    }

    fn incoming_lot(id: &str, mass_mg: u64, evidence_id: &str, sha: char) -> MaterialLot {
        MaterialLot {
            lot_id: id.to_owned(),
            mass_mg,
            material_class: "post-consumer-cotton-blend".to_owned(),
            origin: LotOrigin::Incoming {
                inventory_evidence: evidence(
                    evidence_id,
                    EvidenceKind::PhysicalInventory,
                    EvidenceTier::SensorMeasured,
                    "intake-scale",
                    None,
                    sha,
                    10,
                ),
            },
            disposition: LotDisposition::InProcess,
            receiver_acceptance: None,
        }
    }

    fn output_lot(
        id: &str,
        mass_mg: u64,
        event_id: &str,
        disposition: LotDisposition,
    ) -> MaterialLot {
        MaterialLot {
            lot_id: id.to_owned(),
            mass_mg,
            material_class: "sorted-textile".to_owned(),
            origin: LotOrigin::EventOutput {
                event_id: event_id.to_owned(),
            },
            disposition,
            receiver_acceptance: None,
        }
    }

    fn event(
        id: &str,
        process: FlowProcess,
        inputs: &[&str],
        outputs: &[&str],
        unaccounted: u64,
        uncertainty: u64,
        evidence_id: &str,
        evidence_sha: char,
        observed_at: u64,
    ) -> MaterialFlowEvent {
        MaterialFlowEvent {
            event_id: id.to_owned(),
            process,
            input_lot_ids: inputs.iter().map(|value| (*value).to_owned()).collect(),
            output_lot_ids: outputs.iter().map(|value| (*value).to_owned()).collect(),
            unaccounted_mass_mg: unaccounted,
            uncertainty_mass_mg: uncertainty,
            max_unaccounted_ppm: 20_000,
            max_uncertainty_ppm: 5_000,
            evidence: evidence(
                evidence_id,
                process.expected_evidence_kind(),
                EvidenceTier::PhysicalTrial,
                "process-operator",
                None,
                evidence_sha,
                observed_at,
            ),
        }
    }

    fn sorting_split_ledger() -> MaterialFlowLedger {
        MaterialFlowLedger {
            ledger_id: "ledger-001".to_owned(),
            lots: vec![
                incoming_lot("incoming-001", 10_000, "inventory-001", 'a'),
                output_lot("cotton-stream", 6_000, "sort-001", LotDisposition::AwaitingReceiver),
                output_lot("poly-stream", 3_800, "sort-001", LotDisposition::AwaitingReceiver),
            ],
            events: vec![event(
                "sort-001",
                FlowProcess::Sorting,
                &["incoming-001"],
                &["cotton-stream", "poly-stream"],
                200,
                10,
                "sort-evidence-001",
                'b',
                20,
            )],
            claim_ceiling: "synthetic-ledger-contract-only".to_owned(),
        }
    }

    fn attach_receiver(lot: &mut MaterialLot, outcome: AcceptanceOutcome, id: &str, sha: char) {
        lot.disposition = match outcome {
            AcceptanceOutcome::Accepted => LotDisposition::Accepted,
            AcceptanceOutcome::Rejected => LotDisposition::Rejected,
            AcceptanceOutcome::Conditional | AcceptanceOutcome::Unknown => LotDisposition::Accepted,
        };
        lot.receiver_acceptance = Some(DownstreamAcceptance {
            specification_id: "receiver-spec-v1".to_owned(),
            accepted_output_lot_id: lot.lot_id.clone(),
            basis_evidence_id: "sort-evidence-001".to_owned(),
            outcome,
            evidence: evidence(
                id,
                EvidenceKind::DownstreamAcceptance,
                EvidenceTier::IndependentAcceptance,
                "receiver-lab",
                Some("independent-auditor"),
                sha,
                30,
            ),
        });
    }

    #[test]
    fn physical_split_is_explicit_and_mass_conserves() {
        let ledger = sorting_split_ledger();
        let report = validate_material_flow(&ledger);
        assert!(report.valid, "{:#?}", report.findings);
        assert_eq!(report.totals.incoming_mass_mg, 10_000);
        assert_eq!(report.totals.accepted_mass_mg, 0);
        assert_eq!(report.totals.awaiting_receiver_mass_mg, 3_800);
        assert_eq!(report.totals.unaccounted_mass_mg, 200);
    }

    #[test]
    fn merge_consumes_two_complete_lots_into_one_output() {
        let mut ledger = sorting_split_ledger();
        ledger.lots = vec![
            incoming_lot("cotton-a", 5_000, "inventory-a", 'a'),
            incoming_lot("cotton-b", 6_000, "inventory-b", 'b'),
            output_lot("merged-feedstock", 10_900, "merge-001", LotDisposition::AwaitingReceiver),
        ];
        ledger.events = vec![event(
            "merge-001",
            FlowProcess::MechanicalOpening,
            &["cotton-a", "cotton-b"],
            &["merged-feedstock"],
            100,
            10,
            "merge-evidence",
            'c',
            20,
        )];
        let report = validate_material_flow(&ledger);
        assert!(report.valid, "{:#?}", report.findings);
        assert_eq!(report.totals.incoming_mass_mg, 11_000);
        assert_eq!(report.totals.awaiting_receiver_mass_mg, 10_900);
    }

    #[test]
    fn accepted_lot_requires_exact_independent_receiver_decision() {
        let mut ledger = sorting_split_ledger();
        attach_receiver(&mut ledger.lots[1], AcceptanceOutcome::Accepted, "receiver-001", 'c');
        let report = validate_material_flow(&ledger);
        assert!(report.valid, "{:#?}", report.findings);
        assert_eq!(report.totals.accepted_mass_mg, 6_000);
    }

    #[test]
    fn receiver_rejection_is_not_counted_as_accepted_material() {
        let mut ledger = sorting_split_ledger();
        ledger.lots[1].disposition = LotDisposition::Rejected;
        attach_receiver(&mut ledger.lots[1], AcceptanceOutcome::Rejected, "receiver-001", 'c');
        let report = validate_material_flow(&ledger);
        assert!(report.valid, "{:#?}", report.findings);
        assert_eq!(report.totals.accepted_mass_mg, 0);
        assert_eq!(report.totals.rejected_mass_mg, 6_000);
    }

    #[test]
    fn conditional_receiver_decision_cannot_be_promoted_to_accepted() {
        let mut ledger = sorting_split_ledger();
        attach_receiver(&mut ledger.lots[1], AcceptanceOutcome::Conditional, "receiver-001", 'c');
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "RECEIVER_DISPOSITION_MISMATCH"));
        assert_eq!(report.totals.accepted_mass_mg, 0);
    }

    #[test]
    fn double_consumption_requires_an_explicit_split_lot() {
        let mut ledger = sorting_split_ledger();
        ledger.lots.push(output_lot("reuse-output", 5_000, "repair-001", LotDisposition::AwaitingReceiver));
        ledger.events.push(event(
            "repair-001",
            FlowProcess::Repair,
            &["incoming-001"],
            &["reuse-output"],
            5_000,
            10,
            "repair-evidence",
            'c',
            25,
        ));
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "LOT_CONSUMED_MORE_THAN_ONCE"));
    }

    #[test]
    fn quarantined_terminal_mass_is_reported_separately() {
        let mut ledger = sorting_split_ledger();
        ledger.lots[1].disposition = LotDisposition::Quarantined;
        let report = validate_material_flow(&ledger);
        assert!(report.valid, "{:#?}", report.findings);
        assert_eq!(report.totals.quarantined_mass_mg, 6_000);
        assert_eq!(report.totals.accepted_mass_mg, 0);
    }

    #[test]
    fn whole_ledger_conservation_catches_missing_terminal_mass() {
        let mut ledger = sorting_split_ledger();
        ledger.lots.retain(|lot| lot.lot_id != "poly-stream");
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "LEDGER_MASS_BALANCE_NOT_CLOSED"));
    }

    #[test]
    fn event_mass_mismatch_and_excess_residual_fail_closed() {
        let mut ledger = sorting_split_ledger();
        ledger.events[0].unaccounted_mass_mg = 2_000;
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "EVENT_MASS_BALANCE_NOT_CLOSED"));
        assert!(report.findings.iter().any(|item| item.code == "UNACCOUNTED_MASS_ABOVE_POLICY"));
    }

    #[test]
    fn output_must_be_created_by_the_event_that_declares_it() {
        let mut ledger = sorting_split_ledger();
        if let LotOrigin::EventOutput { event_id } = &mut ledger.lots[1].origin {
            *event_id = "other-event".to_owned();
        }
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "EVENT_OUTPUT_PRODUCER_MISMATCH"));
    }

    #[test]
    fn in_process_lot_cannot_disappear_without_consumption_or_terminal_status() {
        let mut ledger = sorting_split_ledger();
        ledger.lots[2].disposition = LotDisposition::InProcess;
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "UNRESOLVED_UNCONSUMED_LOT"));
    }

    #[test]
    fn event_cannot_precede_its_input_or_producer() {
        let mut ledger = sorting_split_ledger();
        ledger.events[0].evidence.observed_at_ns = 5;
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "EVENT_PRECEDES_INPUT_INVENTORY"));
    }

    #[test]
    fn event_lineage_cycle_is_rejected() {
        let mut ledger = sorting_split_ledger();
        ledger.lots = vec![
            output_lot("lot-a", 10_000, "event-b", LotDisposition::InProcess),
            output_lot("lot-b", 10_000, "event-a", LotDisposition::InProcess),
        ];
        ledger.events = vec![
            event(
                "event-a",
                FlowProcess::MechanicalOpening,
                &["lot-a"],
                &["lot-b"],
                0,
                0,
                "event-a-evidence",
                'c',
                20,
            ),
            event(
                "event-b",
                FlowProcess::MechanicalOpening,
                &["lot-b"],
                &["lot-a"],
                0,
                0,
                "event-b-evidence",
                'd',
                30,
            ),
        ];
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "EVENT_GRAPH_CYCLE"));
    }

    #[test]
    fn event_consuming_its_own_output_is_rejected_as_a_self_cycle() {
        let mut ledger = sorting_split_ledger();
        ledger.lots = vec![output_lot(
            "self-lot",
            10_000,
            "self-event",
            LotDisposition::InProcess,
        )];
        ledger.events = vec![event(
            "self-event",
            FlowProcess::MechanicalOpening,
            &["self-lot"],
            &["self-lot"],
            0,
            0,
            "self-event-evidence",
            'c',
            20,
        )];
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "EVENT_GRAPH_CYCLE"));
    }

    #[test]
    fn synthetic_or_simulated_process_evidence_cannot_qualify_physical_flow() {
        let mut ledger = sorting_split_ledger();
        ledger.events[0].evidence.tier = EvidenceTier::Simulation;
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "TRANSFORMATION_EVIDENCE_INVALID"));
    }

    #[test]
    fn multi_stage_lineage_counts_only_terminal_mass_once() {
        let mut ledger = sorting_split_ledger();
        ledger.lots = vec![
            incoming_lot("incoming-001", 10_000, "inventory-001", 'a'),
            output_lot("cotton-stream", 6_000, "sort-001", LotDisposition::InProcess),
            output_lot("poly-stream", 3_800, "sort-001", LotDisposition::InProcess),
            output_lot("cotton-fibre", 5_900, "open-cotton", LotDisposition::AwaitingReceiver),
            output_lot("poly-downcycle", 3_730, "downcycle-poly", LotDisposition::AwaitingReceiver),
        ];
        ledger.events = vec![
            event(
                "sort-001",
                FlowProcess::Sorting,
                &["incoming-001"],
                &["cotton-stream", "poly-stream"],
                200,
                10,
                "sort-evidence-001",
                'b',
                20,
            ),
            event(
                "open-cotton",
                FlowProcess::MechanicalOpening,
                &["cotton-stream"],
                &["cotton-fibre"],
                100,
                10,
                "open-evidence-001",
                'c',
                30,
            ),
            event(
                "downcycle-poly",
                FlowProcess::Downcycling,
                &["poly-stream"],
                &["poly-downcycle"],
                70,
                10,
                "downcycle-evidence-001",
                'd',
                35,
            ),
        ];
        for item in &mut ledger.events {
            item.max_unaccounted_ppm = 30_000;
        }

        let report = validate_material_flow(&ledger);
        assert!(report.valid, "{:#?}", report.findings);
        assert_eq!(report.totals.incoming_mass_mg, 10_000);
        assert_eq!(report.totals.awaiting_receiver_mass_mg, 9_630);
        assert_eq!(report.totals.unaccounted_mass_mg, 370);
    }

    #[test]
    fn receiver_issuer_and_verifier_must_be_independent_of_producer() {
        let mut issuer_conflict = sorting_split_ledger();
        attach_receiver(
            &mut issuer_conflict.lots[1],
            AcceptanceOutcome::Accepted,
            "receiver-issuer-conflict",
            'c',
        );
        issuer_conflict.lots[1]
            .receiver_acceptance
            .as_mut()
            .unwrap()
            .evidence
            .issuer_id = "process-operator".to_owned();
        assert!(
            validate_material_flow(&issuer_conflict)
                .findings
                .iter()
                .any(|item| item.code == "RECEIVER_NOT_INDEPENDENT_OF_PROCESS")
        );

        let mut verifier_conflict = sorting_split_ledger();
        attach_receiver(
            &mut verifier_conflict.lots[1],
            AcceptanceOutcome::Accepted,
            "receiver-verifier-conflict",
            'c',
        );
        verifier_conflict.lots[1]
            .receiver_acceptance
            .as_mut()
            .unwrap()
            .evidence
            .verifier_id = Some("process-operator".to_owned());
        assert!(
            validate_material_flow(&verifier_conflict)
                .findings
                .iter()
                .any(|item| item.code == "RECEIVER_VERIFIER_NOT_INDEPENDENT_OF_PROCESS")
        );
    }

    #[test]
    fn receiver_decision_cannot_predate_the_producing_trial() {
        let mut ledger = sorting_split_ledger();
        attach_receiver(
            &mut ledger.lots[1],
            AcceptanceOutcome::Accepted,
            "receiver-too-early",
            'c',
        );
        ledger.lots[1]
            .receiver_acceptance
            .as_mut()
            .unwrap()
            .evidence
            .observed_at_ns = 15;
        assert!(
            validate_material_flow(&ledger)
                .findings
                .iter()
                .any(|item| item.code == "RECEIVER_DECISION_PRECEDES_OUTPUT")
        );
    }

    #[test]
    fn unique_receiver_evidence_cannot_reuse_process_bytes() {
        let mut ledger = sorting_split_ledger();
        attach_receiver(&mut ledger.lots[1], AcceptanceOutcome::Accepted, "receiver-001", 'b');
        let report = validate_material_flow(&ledger);
        assert!(!report.valid);
        assert!(report.findings.iter().any(|item| item.code == "RECEIVER_EVIDENCE_REUSES_BASIS_BYTES"));
        assert!(report.findings.iter().any(|item| item.code == "DUPLICATE_EVIDENCE_DIGEST"));
    }
}
