// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical evidence encoding for deterministic external hashing and signing.
//!
//! The kernel intentionally does not implement a cryptographic hash. It emits
//! schema-tagged canonical bytes that a caller can bind to an approved digest,
//! signature, transparency log, or evidence ledger.

use crate::conflict::{DefeatBasis, LegalStatus, LiteralResolution};
use crate::defeasible::{Derivation, Rule};
use crate::deontic::{DeonticProposition, Modality, StructuredNorm};
use crate::hohfeld::{Jural, JuralRelation};
use crate::lifecycle::{LifecycleAssessment, NormState};
use crate::model::{Literal, QueryId, RulePackId, SemanticProfileId, SourceRef};
use crate::transition::TransitionRecord;

/// Append a stable, schema-controlled representation of one result payload.
pub trait CanonicalEvidence {
    fn append_canonical(&self, output: &mut String);

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        self.append_canonical(&mut output);
        output.into_bytes()
    }
}

/// Metadata binding evidence to the selected rule pack, query, and semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceManifest {
    pub schema_version: u16,
    pub engine_version: String,
    pub semantic_profile: SemanticProfileId,
    pub rule_pack: RulePackId,
    pub query: QueryId,
}

impl EvidenceManifest {
    pub fn v1(semantic_profile: SemanticProfileId, rule_pack: RulePackId, query: QueryId) -> Self {
        Self {
            schema_version: 1,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            semantic_profile,
            rule_pack,
            query,
        }
    }
}

/// A canonical evidence envelope ready for caller-selected cryptographic
/// hashing and signing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceEnvelope<T> {
    pub manifest: EvidenceManifest,
    pub payload: T,
}

impl<T> EvidenceEnvelope<T> {
    pub fn new(manifest: EvidenceManifest, payload: T) -> Self {
        Self { manifest, payload }
    }
}

impl<T: CanonicalEvidence> CanonicalEvidence for EvidenceEnvelope<T> {
    fn append_canonical(&self, output: &mut String) {
        output.push_str("symthaea-legal-evidence\n");
        unsigned(output, "schema", u64::from(self.manifest.schema_version));
        text(output, "engine", &self.manifest.engine_version);
        text(output, "profile", self.manifest.semantic_profile.as_str());
        text(output, "rule-pack", self.manifest.rule_pack.as_str());
        text(output, "query", self.manifest.query.as_str());
        output.push_str("payload-begin\n");
        self.payload.append_canonical(output);
        output.push_str("payload-end\n");
    }
}

impl CanonicalEvidence for Derivation {
    fn append_canonical(&self, output: &mut String) {
        output.push_str("legacy-derivation-v1\n");
        unsigned(output, "initial-count", self.initial_facts.len() as u64);
        for fact in &self.initial_facts {
            text(output, "initial", fact);
        }
        unsigned(output, "fact-count", self.facts.len() as u64);
        for fact in &self.facts {
            text(output, "fact", fact);
        }
        let mut steps = self.steps.clone();
        steps.sort_unstable_by(|left, right| {
            left.stratum
                .cmp(&right.stratum)
                .then_with(|| left.conclusion.cmp(&right.conclusion))
                .then_with(|| left.supporting_rules.cmp(&right.supporting_rules))
        });
        unsigned(output, "step-count", steps.len() as u64);
        for step in steps {
            output.push_str("step\n");
            text(output, "conclusion", &step.conclusion);
            unsigned(output, "stratum", step.stratum as u64);
            let mut supporting_rules = step.supporting_rules;
            supporting_rules.sort_unstable();
            supporting_rules.dedup();
            unsigned(output, "support-count", supporting_rules.len() as u64);
            for rule in &supporting_rules {
                append_legacy_rule(output, rule);
            }
        }
    }
}

impl CanonicalEvidence for LiteralResolution {
    fn append_canonical(&self, output: &mut String) {
        output.push_str("literal-resolution-v1\n");
        append_literal(output, "query", &self.query);
        text(output, "status", legal_status(self.status));
        rule_ids(output, "support", &self.undefeated_support);
        rule_ids(output, "opposition", &self.undefeated_opposition);
        rule_ids(output, "defeater", &self.blocking_defeaters);
        let mut defeats = self.defeats.clone();
        defeats.sort_unstable();
        defeats.dedup();
        unsigned(output, "defeat-count", defeats.len() as u64);
        for defeat in &defeats {
            output.push_str("defeat\n");
            text(output, "winner", defeat.winner.as_str());
            text(output, "loser", defeat.loser.as_str());
            text(
                output,
                "basis",
                match defeat.basis {
                    DefeatBasis::StrictOverDefeasible => "strict-over-defeasible",
                    DefeatBasis::ExplicitPriority => "explicit-priority",
                },
            );
        }
    }
}

impl CanonicalEvidence for LifecycleAssessment {
    fn append_canonical(&self, output: &mut String) {
        output.push_str("norm-lifecycle-v1\n");
        text(output, "state", norm_state(self.state));
        let mut decisive_events = self.decisive_events.clone();
        decisive_events.sort_unstable();
        decisive_events.dedup();
        unsigned(output, "decisive-event-count", decisive_events.len() as u64);
        for event in &decisive_events {
            text(output, "decisive-event", event.as_str());
        }
        match &self.activated_reparation {
            Some(norm) => {
                text(output, "has-reparation", "true");
                append_structured_norm(output, norm);
            }
            None => text(output, "has-reparation", "false"),
        }
    }
}

impl CanonicalEvidence for TransitionRecord {
    fn append_canonical(&self, output: &mut String) {
        output.push_str("legal-transition-v1\n");
        append_relation(output, "power", &self.power);
        let mut retracted = self.retracted.clone();
        retracted.sort_unstable();
        retracted.dedup();
        unsigned(output, "retracted-count", retracted.len() as u64);
        for relation in &retracted {
            append_relation(output, "retracted", relation);
        }
        let mut asserted = self.asserted.clone();
        asserted.sort_unstable();
        asserted.dedup();
        unsigned(output, "asserted-count", asserted.len() as u64);
        for relation in &asserted {
            append_relation(output, "asserted", relation);
        }
    }
}

fn append_legacy_rule(output: &mut String, rule: &Rule) {
    output.push_str("legacy-rule\n");
    let mut conditions = rule.conditions.clone();
    conditions.sort_unstable();
    conditions.dedup();
    let mut exceptions = rule.exceptions.clone();
    exceptions.sort_unstable();
    exceptions.dedup();
    unsigned(output, "condition-count", conditions.len() as u64);
    for condition in conditions {
        text(output, "condition", &condition);
    }
    unsigned(output, "exception-count", exceptions.len() as u64);
    for exception in exceptions {
        text(output, "exception", &exception);
    }
    text(output, "conclusion", &rule.conclusion);
}

fn append_structured_norm(output: &mut String, norm: &StructuredNorm) {
    output.push_str("structured-norm\n");
    text(
        output,
        "modality",
        match norm.modality {
            Modality::Obligatory => "obligatory",
            Modality::Permitted => "permitted",
            Modality::Forbidden => "forbidden",
        },
    );
    append_proposition(output, &norm.proposition);
}

fn append_proposition(output: &mut String, proposition: &DeonticProposition) {
    output.push_str("proposition\n");
    text(output, "bearer", proposition.bearer.as_str());
    text(output, "action", proposition.action.as_str());
    match &proposition.beneficiary {
        Some(beneficiary) => text(output, "beneficiary", beneficiary.as_str()),
        None => text(output, "beneficiary", ""),
    }
}

fn append_relation(output: &mut String, label: &str, relation: &JuralRelation) {
    output.push_str(label);
    output.push('\n');
    text(output, "holder", relation.holder.as_str());
    text(output, "counterparty", relation.counterparty.as_str());
    text(output, "position", jural(relation.position));
    text(output, "action", relation.action.as_str());
    append_source(output, relation.source.as_ref());
}

fn append_source(output: &mut String, source: Option<&SourceRef>) {
    match source {
        Some(source) => {
            text(output, "source-document", source.document.as_str());
            text(output, "source-provision", source.provision.as_str());
        }
        None => {
            text(output, "source-document", "");
            text(output, "source-provision", "");
        }
    }
}

fn append_literal(output: &mut String, label: &str, literal: &Literal) {
    let sign = if literal.is_positive() { "+" } else { "-" };
    text(output, &format!("{label}-sign"), sign);
    text(output, &format!("{label}-atom"), literal.atom().as_str());
}

fn rule_ids(output: &mut String, label: &str, rules: &[crate::model::RuleId]) {
    let mut rules = rules.to_vec();
    rules.sort_unstable();
    rules.dedup();
    unsigned(output, &format!("{label}-count"), rules.len() as u64);
    for rule in &rules {
        text(output, label, rule.as_str());
    }
}

fn legal_status(status: LegalStatus) -> &'static str {
    match status {
        LegalStatus::Supported => "supported",
        LegalStatus::Refuted => "refuted",
        LegalStatus::Both => "both",
        LegalStatus::Undetermined => "undetermined",
    }
}

fn norm_state(state: NormState) -> &'static str {
    match state {
        NormState::NotYetEffective => "not-yet-effective",
        NormState::Active => "active",
        NormState::Fulfilled => "fulfilled",
        NormState::FulfilledLate => "fulfilled-late",
        NormState::Exercised => "exercised",
        NormState::Violated => "violated",
        NormState::TemporallyAmbiguous => "temporally-ambiguous",
        NormState::Waived => "waived",
        NormState::Expired => "expired",
    }
}

fn jural(position: Jural) -> &'static str {
    match position {
        Jural::Right => "right",
        Jural::Duty => "duty",
        Jural::Privilege => "privilege",
        Jural::NoRight => "no-right",
        Jural::Power => "power",
        Jural::Liability => "liability",
        Jural::Immunity => "immunity",
        Jural::Disability => "disability",
    }
}

/// Length-prefixed UTF-8 field encoding prevents delimiter ambiguity.
fn text(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push(' ');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push('\n');
}

fn unsigned(output: &mut String, label: &str, value: u64) {
    text(output, label, &value.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defeasible::{Rule, try_derive_with_trace};
    use crate::facts::FactBase;
    use crate::inference::{InferenceProfile, infer};
    use crate::model::{Atom, QueryId, RuleId, RulePackId, SemanticProfileId};
    use crate::proof::ProofGraph;
    use crate::rules::{FormalRule, RuleKind, RulePack};

    #[test]
    fn canonical_derivation_is_rule_order_invariant() {
        let default = Rule::new(&["resident"], &["exempt"], "register");
        let exception = Rule::new(&["diplomat"], &[], "exempt");
        let left = try_derive_with_trace(
            &[default.clone(), exception.clone()],
            &["resident", "diplomat"],
        )
        .unwrap();
        let right =
            try_derive_with_trace(&[exception, default], &["diplomat", "resident"]).unwrap();

        assert_eq!(left.canonical_bytes(), right.canonical_bytes());
    }

    #[test]
    fn public_vector_order_does_not_change_canonical_resolution() {
        let query = Literal::Positive(Atom::new("enter").unwrap());
        let first = LiteralResolution {
            query: query.clone(),
            status: LegalStatus::Supported,
            undefeated_support: vec![
                crate::model::RuleId::new("b").unwrap(),
                crate::model::RuleId::new("a").unwrap(),
            ],
            undefeated_opposition: Vec::new(),
            blocking_defeaters: Vec::new(),
            defeats: Vec::new(),
        };
        let second = LiteralResolution {
            query,
            status: LegalStatus::Supported,
            undefeated_support: vec![
                crate::model::RuleId::new("a").unwrap(),
                crate::model::RuleId::new("b").unwrap(),
            ],
            undefeated_opposition: Vec::new(),
            blocking_defeaters: Vec::new(),
            defeats: Vec::new(),
        };

        assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    }

    #[test]
    fn length_prefixing_distinguishes_embedded_delimiters() {
        let mut first = String::new();
        text(&mut first, "field", "a\nb");
        let mut second = String::new();
        text(&mut second, "field", "a");
        text(&mut second, "b", "");
        assert_ne!(first, second);
    }

    #[test]
    fn envelope_binds_semantics_pack_query_and_engine_version() {
        let derivation =
            try_derive_with_trace(&[Rule::new(&["adult"], &[], "capacity")], &["adult"]).unwrap();
        let envelope = EvidenceEnvelope::new(
            EvidenceManifest::v1(
                SemanticProfileId::new("legacy-stratified-v1").unwrap(),
                RulePackId::new("capacity-rules-v1").unwrap(),
                QueryId::new("capacity-of-party-7").unwrap(),
            ),
            derivation,
        );
        let text = String::from_utf8(envelope.canonical_bytes()).unwrap();

        assert!(text.contains("symthaea-legal-evidence"));
        assert!(text.contains("legacy-stratified-v1"));
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn grounded_evidence_is_input_order_invariant_and_records_guards() {
        let positive = |value: &str| Literal::Positive(Atom::new(value).unwrap());
        let default = FormalRule::new(
            RuleId::new("resident-default").unwrap(),
            RuleKind::Defeasible,
            [positive("resident")],
            positive("register"),
        )
        .unwrap()
        .with_exceptions([positive("exempt")])
        .unwrap();
        let capacity = FormalRule::new(
            RuleId::new("adult-resident").unwrap(),
            RuleKind::Strict,
            [positive("adult")],
            positive("resident"),
        )
        .unwrap();
        let left_pack = RulePack::new(
            RulePackId::new("registration-v1").unwrap(),
            [default.clone(), capacity.clone()],
            [],
        )
        .unwrap();
        let right_pack = RulePack::new(
            RulePackId::new("registration-v1").unwrap(),
            [capacity, default],
            [],
        )
        .unwrap();
        let left_facts = FactBase::from_literals([positive("adult"), positive("citizen")]);
        let right_facts = FactBase::from_literals([positive("citizen"), positive("adult")]);
        let profile = InferenceProfile::grounded_blocking_v1();

        let left = infer(&left_pack, &left_facts, &profile).unwrap();
        let right = infer(&right_pack, &right_facts, &profile).unwrap();

        assert_eq!(left.canonical_bytes(), right.canonical_bytes());
        let graph_text =
            String::from_utf8(ProofGraph::from_result(&left).canonical_bytes()).unwrap();
        assert!(graph_text.contains("guard\n"));
        assert!(graph_text.contains("exception-atom"));
        assert!(graph_text.contains("exempt"));
    }

    #[test]
    fn literal_encoding_preserves_explicit_negation() {
        let positive = Literal::Positive(Atom::new("liable").unwrap());
        let negative = positive.opposite();
        let mut left = String::new();
        append_literal(&mut left, "query", &positive);
        let mut right = String::new();
        append_literal(&mut right, "query", &negative);
        assert_ne!(left, right);
    }
}
