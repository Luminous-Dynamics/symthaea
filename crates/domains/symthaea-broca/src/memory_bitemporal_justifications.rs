// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Transactional composition of the bitemporal source ledger and OR-of-AND
//! truth-maintenance graph.
//!
//! The ledger owns source valid-time and transaction-time semantics. The
//! justification graph owns alternative proof support and derived-fact
//! dependency semantics. This runtime makes source admission/supersession/
//! retraction update both structures atomically and evaluates historical
//! support using the source revisions eligible at an explicit
//! (valid_at, known_at, scope) coordinate.
//!
//! This is an in-memory reference runtime. It is not a SQLite persistence
//! adapter, authorization engine, source-authenticity verifier, or entailment
//! engine. Exact scope partitioning is enforced; principal authorization must
//! still be checked by the caller. Contested source keys are conservatively
//! excluded from derived-proof support because a trusted contradiction
//! resolver is not yet integrated.

use crate::memory_bitemporal::{
    BitemporalSourceLedger, SourceEventOperation, SourceEventRevision,
};
use crate::memory_justifications::{
    AsOfSupportStatus, DerivedConclusionInput, JustificationGraph, RetractionReport,
};
use chrono::{DateTime, NaiveDate, Utc};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Auditable consequence of applying one append-only source event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceEventApplicationReceipt {
    pub event_revision_id: String,
    /// The old source assertion invalidated by Supersede or Retract.
    pub invalidated_revision_id: Option<String>,
    /// Derived conclusions whose active support changed.
    pub retraction_report: Option<RetractionReport>,
    /// True for Assert and Supersede, whose new assertion revision is registered.
    pub registered_new_assertion: bool,
}

/// In-memory reference runtime joining source history and proof maintenance.
#[derive(Debug, Clone, Default)]
pub struct BitemporalJustificationRuntime {
    ledger: BitemporalSourceLedger,
    justifications: JustificationGraph,
    /// Transaction time when each derived conclusion first became part of memory.
    derived_known_at: BTreeMap<String, DateTime<Utc>>,
}

impl BitemporalJustificationRuntime {
    /// Append source events atomically to both the bitemporal ledger and the
    /// alternative-justification graph. All revisions in a batch must satisfy
    /// the ledger's one-transaction-time rule.
    ///
    /// On any validation, ledger, or graph error, neither structure is changed.
    pub fn append_source_events(
        &mut self,
        revisions: impl IntoIterator<Item = SourceEventRevision>,
    ) -> Result<Vec<SourceEventApplicationReceipt>, String> {
        let revisions: Vec<SourceEventRevision> = revisions.into_iter().collect();
        if revisions.is_empty() {
            return Err("an append batch must contain at least one source event".into());
        }

        let transaction_time = revisions[0].recorded_at;
        if self.derived_known_at.values().any(|derived_at| *derived_at >= transaction_time) {
            return Err(
                "source event transaction time must be strictly later than every committed derived conclusion"
                    .into(),
            );
        }

        let mut candidate_ledger = self.ledger.clone();
        candidate_ledger.append_batch(revisions.clone())?;
        let mut candidate_graph = self.justifications.clone();
        let mut receipts = Vec::with_capacity(revisions.len());

        for revision in revisions {
            let mut invalidated_revision_id = None;
            let mut retraction_report = None;
            let registered_new_assertion = match &revision.operation {
                SourceEventOperation::Assert { .. } => {
                    candidate_graph.register_source_event(revision.revision_id.clone())?;
                    true
                }
                SourceEventOperation::Supersede {
                    target_revision_id,
                    ..
                } => {
                    let report = candidate_graph
                        .invalidate_source_event(target_revision_id)?;
                    invalidated_revision_id = Some(target_revision_id.clone());
                    retraction_report = Some(report);
                    candidate_graph.register_source_event(revision.revision_id.clone())?;
                    true
                }
                SourceEventOperation::Retract {
                    target_revision_id,
                    ..
                } => {
                    let report = candidate_graph
                        .invalidate_source_event(target_revision_id)?;
                    invalidated_revision_id = Some(target_revision_id.clone());
                    retraction_report = Some(report);
                    false
                }
            };

            receipts.push(SourceEventApplicationReceipt {
                event_revision_id: revision.revision_id,
                invalidated_revision_id,
                retraction_report,
                registered_new_assertion,
            });
        }

        let candidate = Self {
            ledger: candidate_ledger,
            justifications: candidate_graph,
            derived_known_at: self.derived_known_at.clone(),
        };
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(receipts)
    }

    /// Add one derived conclusion with an explicit transaction/knowledge time.
    ///
    /// A standalone derivation is one transaction. Use derive_batch_at when
    /// multiple conclusions must share a timestamp and become visible atomically.
    pub fn derive_at(
        &mut self,
        conclusion: DerivedConclusionInput,
        recorded_at: DateTime<Utc>,
    ) -> Result<(), String> {
        self.derive_batch_at([conclusion], recorded_at)
    }

    /// Atomically add a topologically ordered batch of derived conclusions at
    /// one transaction/knowledge time.
    ///
    /// A child may depend on a parent earlier in this same batch. A parent that
    /// appears later is rejected. The entire batch commits or none of it does.
    /// Each standalone batch must have a strictly later timestamp than the
    /// previous derived batch; source batches must be strictly later than all
    /// previously committed derived batches.
    pub fn derive_batch_at(
        &mut self,
        conclusions: impl IntoIterator<Item = DerivedConclusionInput>,
        recorded_at: DateTime<Utc>,
    ) -> Result<(), String> {
        let conclusions: Vec<DerivedConclusionInput> = conclusions.into_iter().collect();
        if conclusions.is_empty() {
            return Err("a derived-fact batch must contain at least one conclusion".into());
        }
        self.verify_consistency()?;

        if self.ledger.events().last().is_some_and(|event| event.recorded_at >= recorded_at) {
            return Err(
                "derived-fact batch timestamp must be strictly later than the latest committed source-event batch"
                    .into(),
            );
        }
        if self.derived_known_at.values().any(|known_at| *known_at >= recorded_at) {
            return Err(
                "a new derived-fact batch must have a strictly later timestamp than prior derived batches"
                    .into(),
            );
        }

        let mut candidate_graph = self.justifications.clone();
        let mut candidate_known_at = self.derived_known_at.clone();
        let mut batch_fact_ids = BTreeSet::new();

        for conclusion in conclusions {
            if !batch_fact_ids.insert(conclusion.fact_id.clone()) {
                return Err(format!("duplicate derived fact ID {:?} within batch", conclusion.fact_id));
            }
            if candidate_known_at.contains_key(&conclusion.fact_id) {
                return Err(format!("duplicate derived fact ID {:?}", conclusion.fact_id));
            }

            // Validate each source/parent against the exact transaction-time
            // state visible to this batch. Parents must precede children in the
            // batch; this is the explicit topological insertion order.
            for justification in &conclusion.justifications {
                for dependency in &justification.dependencies {
                    match dependency {
                        crate::memory_justifications::SupportRef::SourceEvent(source_id) => {
                            let source_event = self.ledger.events().iter().find(|event| {
                                event.revision_id == *source_id
                                    && matches!(
                                        &event.operation,
                                        SourceEventOperation::Assert { .. }
                                            | SourceEventOperation::Supersede { .. }
                                    )
                            }).ok_or_else(|| format!("unknown source revision {source_id:?}"))?;
                            if source_event.recorded_at > recorded_at {
                                return Err(format!(
                                    "source revision {source_id:?} was not known at derived-fact recorded_at"
                                ));
                            }
                            if self.ledger.events().iter().any(|event| {
                                event.recorded_at <= recorded_at
                                    && matches!(
                                        &event.operation,
                                        SourceEventOperation::Supersede { target_revision_id, .. }
                                            | SourceEventOperation::Retract { target_revision_id, .. }
                                            if target_revision_id == source_id
                                    )
                            }) {
                                return Err(format!(
                                    "source revision {source_id:?} was already superseded or retracted at derived-fact recorded_at"
                                ));
                            }
                        }
                        crate::memory_justifications::SupportRef::DerivedFact(parent_id) => {
                            let parent_known_at = candidate_known_at.get(parent_id)
                                .ok_or_else(|| format!("unknown derived parent {parent_id:?} or parent follows child in batch"))?;
                            if parent_known_at > &recorded_at {
                                return Err(format!(
                                    "derived parent {parent_id:?} was not known at child recorded_at"
                                ));
                            }
                        }
                    }
                }
            }

            let fact_id = conclusion.fact_id.clone();
            candidate_graph.derive(conclusion)?;
            candidate_known_at.insert(fact_id, recorded_at);
        }

        let mut candidate_runtime = self.clone();
        candidate_runtime.justifications = candidate_graph;
        candidate_runtime.derived_known_at = candidate_known_at;
        candidate_runtime.verify_consistency()?;
        *self = candidate_runtime;
        Ok(())
    }

    /// Evaluate all derived conclusions at a specific world-time date,
    /// transaction-time cutoff, and exact source scope.
    ///
    /// A later source correction does not rewrite the result for an earlier
    /// known_at. A source superseded/retracted by known_at is ineligible.
    /// A source outside valid_at or in another scope is ineligible.
    /// Conflicting active values for the same subject/attribute/scope key are
    /// conservatively withheld from proofs until a contradiction resolver is
    /// integrated. Unsupported is not interpreted as false.
    pub fn support_statuses_at(
        &self,
        valid_at: NaiveDate,
        known_at: DateTime<Utc>,
        scope: &str,
    ) -> Result<BTreeMap<String, AsOfSupportStatus>, String> {
        self.verify_consistency()?;
        let eligible_source_ids = self
            .ledger
            .active_assertion_revision_ids_at(valid_at, known_at, scope)?;
        let eligible_derived_fact_ids: BTreeSet<String> = self
            .derived_known_at
            .iter()
            .filter_map(|(fact_id, derived_at)| {
                (*derived_at <= known_at).then(|| fact_id.clone())
            })
            .collect();
        self.justifications.support_statuses_at_with_evidence(
            &valid_at.format("%Y-%m-%d").to_string(),
            &eligible_source_ids,
            &eligible_derived_fact_ids,
        )
    }

    /// Return one conclusion's support status at the supplied bitemporal
    /// coordinate. Missing IDs are errors, never implicit Unsupported facts.
    pub fn support_status_at(
        &self,
        fact_id: &str,
        valid_at: NaiveDate,
        known_at: DateTime<Utc>,
        scope: &str,
    ) -> Result<AsOfSupportStatus, String> {
        self.support_statuses_at(valid_at, known_at, scope)?
            .remove(fact_id)
            .ok_or_else(|| format!("unknown derived fact {fact_id:?} or not yet known at the requested known_at"))
    }

    /// Check source-ledger / justification-graph source identity parity and
    /// each component's own internal invariants.
    pub fn verify_consistency(&self) -> Result<(), String> {
        self.ledger.verify_consistency()?;
        self.justifications.verify_consistency()?;

        let expected_source_ids: BTreeSet<String> = self
            .ledger
            .events()
            .iter()
            .filter_map(|event| match &event.operation {
                SourceEventOperation::Assert { .. }
                | SourceEventOperation::Supersede { .. } => Some(event.revision_id.clone()),
                SourceEventOperation::Retract { .. } => None,
            })
            .collect();

        if expected_source_ids.len() != self.justifications.source_event_count() {
            return Err(format!(
                "source identity cardinality mismatch: ledger has {} assertion revisions but graph has {}",
                expected_source_ids.len(),
                self.justifications.source_event_count()
            ));
        }
        for revision_id in &expected_source_ids {
            if !self.justifications.has_source_event(revision_id) {
                return Err(format!(
                    "bitemporal assertion revision {revision_id:?} is missing from justification graph"
                ));
            }
        }
        if self.derived_known_at.len() != self.justifications.fact_count() {
            return Err(format!(
                "derived-fact knowledge-time cardinality mismatch: {} timestamps for {} facts",
                self.derived_known_at.len(),
                self.justifications.fact_count()
            ));
        }
        for fact_id in self.derived_known_at.keys() {
            if self.justifications.fact(fact_id).is_none() {
                return Err(format!("knowledge-time index references missing fact {fact_id:?}"));
            }
        }
        Ok(())
    }

    /// Deterministic digest binding both append-only source history and
    /// canonical alternative-justification state. This is an integrity
    /// commitment for parity, not a signature or anti-rollback mechanism.
    pub fn state_sha256(&self) -> Result<String, String> {
        self.verify_consistency()?;
        let source_sha256 = self.ledger.state_sha256()?;
        let justification_sha256 = self.justifications.state_sha256()?;
        let encoded = serde_json::to_vec(&(source_sha256, justification_sha256, &self.derived_known_at))
            .map_err(|error| format!("cannot serialize bitemporal justification state: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }

    pub fn ledger(&self) -> &BitemporalSourceLedger {
        &self.ledger
    }

    pub fn justifications(&self) -> &JustificationGraph {
        &self.justifications
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_bitemporal::{
        BitemporalEvidenceStatus, SourceAssertion, SourceEventOperation,
    };
    use crate::memory_justifications::{DerivedConclusionInput, JustificationInput, SupportRef};
    use chrono::TimeZone;

    fn date(value: &str) -> NaiveDate {
        NaiveDate::parse_from_str(value, "%Y-%m-%d").unwrap()
    }

    fn instant(day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, day, 12, 0, 0).unwrap()
    }

    fn assertion(value: &str, scope: &str, valid_from: &str, valid_to: Option<&str>) -> SourceAssertion {
        SourceAssertion {
            subject: "account-17".into(),
            attribute: "plan".into(),
            value: value.into(),
            valid_from: date(valid_from),
            valid_to: valid_to.map(date),
            source_id: format!("source-{value}"),
            scope: scope.into(),
            payload_sha256: "a".repeat(64),
        }
    }

    fn assert_event(id: &str, day: u32, value: &str) -> SourceEventRevision {
        SourceEventRevision {
            revision_id: id.into(),
            recorded_at: instant(day),
            operation: SourceEventOperation::Assert {
                assertion: assertion(value, "tenant-1", "2026-01-01", None),
            },
        }
    }

    fn conclusion(
        id: &str,
        justifications: impl IntoIterator<Item = JustificationInput>,
    ) -> DerivedConclusionInput {
        DerivedConclusionInput {
            fact_id: id.into(),
            subject: "account-17".into(),
            attribute: "eligible".into(),
            value: "true".into(),
            valid_from: "2026-01-01".into(),
            valid_to: None,
            rule_id: "two-independent-ways".into(),
            rule_version: "v1".into(),
            justifications: justifications.into_iter().collect(),
        }
    }

    fn source(id: &str) -> SupportRef {
        SupportRef::SourceEvent(id.into())
    }

    fn justification(id: &str, dependencies: impl IntoIterator<Item = SupportRef>) -> JustificationInput {
        JustificationInput {
            justification_id: id.into(),
            dependencies: dependencies.into_iter().collect(),
        }
    }

    fn base_runtime() -> BitemporalJustificationRuntime {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([
                assert_event("A", 1, "gold"),
                assert_event("B", 1, "gold"),
            ])
            .unwrap();
        runtime
            .derive_at(conclusion(
                "C",
                [
                    justification("via-A", [source("A")]),
                    justification("via-B", [source("B")]),
                ],
            ), instant(2))
            .unwrap();
        runtime
    }

    #[test]
    fn later_retraction_does_not_rewrite_historical_known_at_support() {
        let mut runtime = base_runtime();
        let before = runtime
            .support_status_at("C", date("2026-01-15"), instant(2), "tenant-1")
            .unwrap();
        assert_eq!(
            before,
            AsOfSupportStatus::Supported {
                active_justification_ids: BTreeSet::from([
                    "via-A".to_owned(),
                    "via-B".to_owned(),
                ]),
            }
        );

        let receipts = runtime
            .append_source_events([SourceEventRevision {
                revision_id: "R-A".into(),
                recorded_at: instant(3),
                operation: SourceEventOperation::Retract {
                    target_revision_id: "A".into(),
                    reason: "source withdrawn".into(),
                },
            }])
            .unwrap();
        assert_eq!(receipts[0].invalidated_revision_id.as_deref(), Some("A"));
        assert_eq!(
            receipts[0]
                .retraction_report
                .as_ref()
                .unwrap()
                .retained_with_alternative_support,
            vec!["C"]
        );

        let historical = runtime
            .support_status_at("C", date("2026-01-15"), instant(2), "tenant-1")
            .unwrap();
        assert_eq!(historical, before);
        let current = runtime
            .support_status_at("C", date("2026-01-15"), instant(3), "tenant-1")
            .unwrap();
        assert_eq!(
            current,
            AsOfSupportStatus::Supported {
                active_justification_ids: BTreeSet::from(["via-B".to_owned()]),
            }
        );
    }

    #[test]
    fn source_supersession_and_final_retraction_propagate_through_alternatives() {
        let mut runtime = base_runtime();
        runtime
            .append_source_events([SourceEventRevision {
                revision_id: "A2".into(),
                recorded_at: instant(3),
                operation: SourceEventOperation::Supersede {
                    target_revision_id: "A".into(),
                    assertion: assertion("platinum", "tenant-1", "2026-01-01", None),
                },
            }])
            .unwrap();

        let after_supersede = runtime
            .support_status_at("C", date("2026-01-15"), instant(3), "tenant-1")
            .unwrap();
        assert_eq!(
            after_supersede,
            AsOfSupportStatus::Supported {
                active_justification_ids: BTreeSet::from(["via-B".to_owned()]),
            }
        );

        runtime
            .append_source_events([SourceEventRevision {
                revision_id: "R-B".into(),
                recorded_at: instant(4),
                operation: SourceEventOperation::Retract {
                    target_revision_id: "B".into(),
                    reason: "second source withdrawn".into(),
                },
            }])
            .unwrap();
        assert_eq!(
            runtime
                .support_status_at("C", date("2026-01-15"), instant(4), "tenant-1")
                .unwrap(),
            AsOfSupportStatus::Unsupported
        );
        assert_eq!(
            runtime
                .support_status_at("C", date("2026-01-15"), instant(3), "tenant-1")
                .unwrap(),
            after_supersede
        );
    }

    #[test]
    fn source_valid_time_and_derived_fact_valid_time_are_both_enforced() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([SourceEventRevision {
                revision_id: "WINDOW".into(),
                recorded_at: instant(1),
                operation: SourceEventOperation::Assert {
                    assertion: assertion(
                        "eligible",
                        "tenant-1",
                        "2026-01-10",
                        Some("2026-01-20"),
                    ),
                },
            }])
            .unwrap();
        runtime
            .derive_at(conclusion(
                "C",
                [justification("via-window", [source("WINDOW")])],
            ), instant(2))
            .unwrap();

        assert!(matches!(
            runtime
                .support_status_at("C", date("2026-01-10"), instant(2), "tenant-1")
                .unwrap(),
            AsOfSupportStatus::Supported { .. }
        ));
        assert_eq!(
            runtime
                .support_status_at("C", date("2026-01-20"), instant(2), "tenant-1")
                .unwrap(),
            AsOfSupportStatus::Unsupported
        );
    }

    #[test]
    fn contested_source_key_is_withheld_from_proof_support() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([
                assert_event("A", 1, "gold"),
                assert_event("B", 1, "silver"),
            ])
            .unwrap();
        runtime
            .derive_at(conclusion(
                "C",
                [justification("via-A", [source("A")])],
            ), instant(2))
            .unwrap();

        let ledger_status = runtime
            .ledger()
            .query(
                "account-17",
                "plan",
                "tenant-1",
                date("2026-01-15"),
                instant(2),
            )
            .unwrap()
            .status;
        assert!(matches!(ledger_status, BitemporalEvidenceStatus::Contested { .. }));
        assert_eq!(
            runtime
                .support_status_at("C", date("2026-01-15"), instant(2), "tenant-1")
                .unwrap(),
            AsOfSupportStatus::Unsupported
        );
    }

    #[test]
    fn exact_scope_prevents_cross_tenant_support_mixing() {
        let mut runtime = BitemporalJustificationRuntime::default();
        let event_a = assert_event("A", 1, "gold");
        let event_b = SourceEventRevision {
            revision_id: "B".into(),
            recorded_at: instant(1),
            operation: SourceEventOperation::Assert {
                assertion: assertion("gold", "tenant-2", "2026-01-01", None),
            },
        };
        runtime.append_source_events([event_a, event_b]).unwrap();
        runtime
            .derive_at(conclusion(
                "C",
                [justification("both-tenants", [source("A"), source("B")])],
            ), instant(2))
            .unwrap();

        assert_eq!(
            runtime
                .support_status_at("C", date("2026-01-15"), instant(2), "tenant-1")
                .unwrap(),
            AsOfSupportStatus::Unsupported
        );
    }

    #[test]
    fn failed_source_batch_leaves_both_ledger_and_graph_unchanged() {
        let mut runtime = base_runtime();
        let before = runtime.state_sha256().unwrap();
        let failed = runtime.append_source_events([
            assert_event("NEW", 3, "bronze"),
            SourceEventRevision {
                revision_id: "R-MISSING".into(),
                recorded_at: instant(3),
                operation: SourceEventOperation::Retract {
                    target_revision_id: "does-not-exist".into(),
                    reason: "invalid target".into(),
                },
            },
        ]);
        assert!(failed.is_err());
        assert_eq!(runtime.state_sha256().unwrap(), before);
        runtime.verify_consistency().unwrap();
    }

    #[test]
    fn derived_conclusion_is_hidden_before_its_own_recorded_at() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([assert_event("A", 1, "gold")])
            .unwrap();
        runtime
            .derive_at(
                conclusion("C", [justification("via-A", [source("A")])]),
                instant(3),
            )
            .unwrap();

        let before_derived = runtime
            .support_statuses_at(date("2026-01-15"), instant(2), "tenant-1")
            .unwrap();
        assert!(!before_derived.contains_key("C"));
        assert!(runtime
            .support_status_at("C", date("2026-01-15"), instant(2), "tenant-1")
            .unwrap_err()
            .contains("not yet known"));

        assert!(matches!(
            runtime
                .support_status_at("C", date("2026-01-15"), instant(3), "tenant-1")
                .unwrap(),
            AsOfSupportStatus::Supported { .. }
        ));
    }

    #[test]
    fn source_event_batch_cannot_be_backdated_before_a_derived_commit() {
        let mut runtime = base_runtime();
        let before = runtime.state_sha256().unwrap();
        let error = runtime
            .append_source_events([assert_event("LATE-BACKDATED", 1, "bronze")])
            .unwrap_err();
        assert!(error.contains("strictly later than every committed derived"));
        assert_eq!(runtime.state_sha256().unwrap(), before);
    }

    #[test]
    fn derived_conclusion_cannot_be_backdated_before_a_source_commit() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([assert_event("A", 2, "gold")])
            .unwrap();
        let before = runtime.state_sha256().unwrap();
        let error = runtime
            .derive_at(
                conclusion("C", [justification("via-A", [source("A")])]),
                instant(1),
            )
            .unwrap_err();
        assert!(error.contains("strictly later than the latest committed source-event batch"));
        assert_eq!(runtime.state_sha256().unwrap(), before);
    }

    #[test]
    fn derivation_rejects_an_unknown_source_revision() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([assert_event("A", 2, "gold")])
            .unwrap();
        let before = runtime.state_sha256().unwrap();
        let result = runtime.derive_at(
            conclusion("C", [justification("via-missing", [source("MISSING")])]),
            instant(3),
        );
        assert!(result.unwrap_err().contains("unknown source revision"));
        assert_eq!(runtime.state_sha256().unwrap(), before);
    }

    #[test]
    fn same_time_parent_and_child_commit_atomically_in_topological_order() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([assert_event("A", 1, "gold")])
            .unwrap();

        let parent = conclusion("P", [justification("via-A", [source("A")])]);
        let child = conclusion(
            "C",
            [justification(
                "via-P",
                [SupportRef::DerivedFact("P".into())],
            )],
        );
        runtime.derive_batch_at([parent, child], instant(2)).unwrap();

        for fact_id in ["P", "C"] {
            assert!(matches!(
                runtime
                    .support_status_at(fact_id, date("2026-01-15"), instant(2), "tenant-1")
                    .unwrap(),
                AsOfSupportStatus::Supported { .. }
            ));
        }
        runtime.verify_consistency().unwrap();
    }

    #[test]
    fn derived_batch_requires_topological_parent_order_and_rolls_back_on_failure() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([assert_event("A", 1, "gold")])
            .unwrap();
        let before = runtime.state_sha256().unwrap();

        let child_first = conclusion(
            "C",
            [justification(
                "via-P",
                [SupportRef::DerivedFact("P".into())],
            )],
        );
        let parent_second = conclusion("P", [justification("via-A", [source("A")])]);
        let error = runtime
            .derive_batch_at([child_first, parent_second], instant(2))
            .unwrap_err();
        assert!(error.contains("unknown derived parent"));
        assert_eq!(runtime.state_sha256().unwrap(), before);

        let valid_parent = conclusion("P", [justification("via-A", [source("A")])]);
        let invalid_child = conclusion(
            "C",
            [justification("via-missing", [source("MISSING")])],
        );
        let error = runtime
            .derive_batch_at([valid_parent, invalid_child], instant(2))
            .unwrap_err();
        assert!(error.contains("unknown source revision"));
        assert_eq!(runtime.state_sha256().unwrap(), before);
    }

    #[test]
    fn separate_derived_batches_require_strictly_later_transaction_times() {
        let mut runtime = BitemporalJustificationRuntime::default();
        runtime
            .append_source_events([assert_event("A", 1, "gold")])
            .unwrap();
        runtime
            .derive_at(
                conclusion("P", [justification("via-A", [source("A")])]),
                instant(2),
            )
            .unwrap();
        let before = runtime.state_sha256().unwrap();
        let error = runtime
            .derive_at(
                conclusion("C", [justification("via-A", [source("A")])]),
                instant(2),
            )
            .unwrap_err();
        assert!(error.contains("strictly later timestamp"));
        assert_eq!(runtime.state_sha256().unwrap(), before);
    }

    #[test]
    fn sixty_four_seed_alternative_support_retraction_sweep_matches_oracle() {
        for seed in 0u64..64 {
            let source_count = 2 + (seed as usize % 6);
            let mut source_ids: Vec<String> =
                (0..source_count).map(|index| format!("S{index}")).collect();
            let mut runtime = BitemporalJustificationRuntime::default();
            let source_events: Vec<SourceEventRevision> = source_ids
                .iter()
                .map(|id| assert_event(id, 1, "gold"))
                .collect();
            runtime.append_source_events(source_events).unwrap();

            let alternatives: Vec<JustificationInput> = source_ids
                .iter()
                .map(|id| justification(&format!("via-{id}"), [source(id)]))
                .collect();
            runtime
                .derive_at(conclusion("C", alternatives), instant(2))
                .unwrap();

            // Deterministic seed-dependent ordering without an RNG dependency.
            let rotation = seed as usize % source_count;
            source_ids.rotate_left(rotation);
            if seed % 2 == 1 {
                source_ids.reverse();
            }

            let mut remaining: BTreeSet<String> = source_ids.iter().cloned().collect();
            for (index, source_id) in source_ids.iter().enumerate() {
                let day = index as u32 + 3;
                let before_retraction = remaining
                    .iter()
                    .map(|id| format!("via-{id}"))
                    .collect::<BTreeSet<_>>();
                if !before_retraction.is_empty() {
                    assert_eq!(
                        runtime
                            .support_status_at(
                                "C",
                                date("2026-01-15"),
                                instant(day - 1),
                                "tenant-1",
                            )
                            .unwrap(),
                        AsOfSupportStatus::Supported {
                            active_justification_ids: before_retraction,
                        },
                        "historical support mismatch for seed {seed} before retracting {source_id}",
                    );
                }

                runtime
                    .append_source_events([SourceEventRevision {
                        revision_id: format!("R-{seed}-{index}"),
                        recorded_at: instant(day),
                        operation: SourceEventOperation::Retract {
                            target_revision_id: source_id.clone(),
                            reason: format!("seed {seed} retraction {index}"),
                        },
                    }])
                    .unwrap();
                remaining.remove(source_id);

                let expected = if remaining.is_empty() {
                    AsOfSupportStatus::Unsupported
                } else {
                    AsOfSupportStatus::Supported {
                        active_justification_ids: remaining
                            .iter()
                            .map(|id| format!("via-{id}"))
                            .collect(),
                    }
                };
                assert_eq!(
                    runtime
                        .support_status_at(
                            "C",
                            date("2026-01-15"),
                            instant(day),
                            "tenant-1",
                        )
                        .unwrap(),
                    expected,
                    "current support mismatch for seed {seed} after retracting {source_id}",
                );
                runtime.verify_consistency().unwrap();
            }

            assert_eq!(
                runtime
                    .support_status_at(
                        "C",
                        date("2026-01-15"),
                        instant(source_count as u32 + 2),
                        "tenant-1",
                    )
                    .unwrap(),
                AsOfSupportStatus::Unsupported,
                "all alternatives should be exhausted for seed {seed}",
            );
        }
    }

    #[test]
    fn ledger_and_graph_source_identity_sets_remain_in_sync() {
        let runtime = base_runtime();
        runtime.verify_consistency().unwrap();
        assert_eq!(runtime.justifications().source_event_count(), 2);
        assert!(runtime.justifications().has_source_event("A"));
        assert!(runtime.justifications().has_source_event("B"));
    }
}
