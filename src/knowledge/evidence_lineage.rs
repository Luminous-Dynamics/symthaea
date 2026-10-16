// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lineage-aware corroboration accounting.
//!
//! Raw citation count is not independent corroboration: a dozen articles may
//! repeat the same primary source. This module groups only on explicit identity,
//! asserted origin, or a well-formed identical content digest. It does not infer
//! missing ancestry from publisher names or semantic similarity. Distinct
//! attributed lineage groups are evidence of distinct recorded origins, not proof
//! that the origins are independent, honest, or true.
//!
//! Callers must keep per-claim witness sets separate. A synthesis derived from
//! multiple origins should be represented as multiple claim-linked witnesses,
//! with explicit upstream lineage, rather than turning one document into several
//! independent votes.

use super::provenance::ContentDigest;
use std::collections::HashMap;

/// One item offered as a witness for a specific proposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceWitness {
    /// Unique ID for this evidence occurrence in the assessment input.
    pub evidence_id: String,
    /// Stable ID of the source record, not the publisher and not the claim.
    pub source_id: String,
    /// Canonical upstream origin ID, when established by explicit provenance.
    ///
    /// Syndicated, quoted, translated, or otherwise derived documents should
    /// share a common origin ID when the lineage evidence establishes it.
    pub origin_id: Option<String>,
    /// Digest of the exact representation being compared. Identical content
    /// groups together even if origin labels or URLs differ.
    pub content_digest: Option<ContentDigest>,
}

/// Policy for assessing how many distinct *attributed origin groups* support a
/// proposition. This is not a probability or an automatic truth verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceLineagePolicy {
    /// Minimum number of distinct attributed groups required for review.
    pub minimum_distinct_origins: usize,
    /// If true, any group without an explicit origin ID blocks eligibility.
    pub require_all_witnesses_attributed: bool,
}

impl EvidenceLineagePolicy {
    pub fn new(
        minimum_distinct_origins: usize,
        require_all_witnesses_attributed: bool,
    ) -> Result<Self, String> {
        if minimum_distinct_origins == 0 {
            return Err("minimum_distinct_origins must be at least 1".to_owned());
        }
        Ok(Self {
            minimum_distinct_origins,
            require_all_witnesses_attributed,
        })
    }

    /// Conservative review gate: at least two distinct recorded origins and
    /// explicit origin attribution for every witness in the set.
    pub fn strict() -> Self {
        Self {
            minimum_distinct_origins: 2,
            require_all_witnesses_attributed: true,
        }
    }
}

impl Default for EvidenceLineagePolicy {
    fn default() -> Self {
        Self::strict()
    }
}

/// Why the lineage assessment cannot be treated as sufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceLineageAbstentionReason {
    NoEvidence,
    MissingEvidenceIdentity,
    InvalidContentDigest,
    ConflictingOriginAssertions,
    UnknownLineage,
    InsufficientDistinctOrigins,
}

/// Gate result. EligibleForReview means the explicit lineage contract passed;
/// it does not certify epistemic independence or truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceLineageDecision {
    EligibleForReview,
    Abstain,
}

/// One deterministic duplicate/lineage component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceLineageGroup {
    pub evidence_ids: Vec<String>,
    pub source_ids: Vec<String>,
    pub origin_ids: Vec<String>,
    /// False means the group has no attributable canonical origin.
    pub origin_attributed: bool,
}

/// Auditable accounting for a set of witnesses attached to one proposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceLineageAssessment {
    pub decision: EvidenceLineageDecision,
    pub reasons: Vec<EvidenceLineageAbstentionReason>,
    pub raw_witness_count: usize,
    pub duplicate_witness_count: usize,
    pub attributed_origin_group_count: usize,
    pub unresolved_lineage_group_count: usize,
    pub groups: Vec<EvidenceLineageGroup>,
}

impl EvidenceLineageAssessment {
    pub fn is_eligible_for_review(&self) -> bool {
        self.decision == EvidenceLineageDecision::EligibleForReview
    }
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, item: usize) -> usize {
        if self.parent[item] != item {
            self.parent[item] = self.find(self.parent[item]);
        }
        self.parent[item]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
    }
}

fn remember_or_union(
    index_by_key: &mut HashMap<String, usize>,
    key: String,
    index: usize,
    sets: &mut UnionFind,
) {
    if let Some(previous) = index_by_key.get(&key).copied() {
        sets.union(previous, index);
    } else {
        index_by_key.insert(key, index);
    }
}

fn normalized_sha256(digest: &ContentDigest) -> Option<String> {
    if !digest.algorithm.eq_ignore_ascii_case("sha256")
        || digest.value.len() != 64
        || !digest.value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return None;
    }
    Some(digest.value.to_ascii_lowercase())
}

fn push_reason_once(
    reasons: &mut Vec<EvidenceLineageAbstentionReason>,
    reason: EvidenceLineageAbstentionReason,
) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

impl EvidenceLineageAssessment {
    /// Group repeated/derived witnesses using explicit lineage signals.
    ///
    /// The algorithm is near-linear in the number of witnesses (inverse-Ackermann
    /// union-find cost, plus sorting the returned audit groups). A malformed
    /// digest is never used as an identity key and blocks eligibility.
    pub fn assess(
        witnesses: &[EvidenceWitness],
        policy: &EvidenceLineagePolicy,
    ) -> Self {
        use EvidenceLineageAbstentionReason as R;

        if witnesses.is_empty() {
            return Self {
                decision: EvidenceLineageDecision::Abstain,
                reasons: vec![R::NoEvidence],
                raw_witness_count: 0,
                duplicate_witness_count: 0,
                attributed_origin_group_count: 0,
                unresolved_lineage_group_count: 0,
                groups: Vec::new(),
            };
        }

        let mut reasons = Vec::new();
        let mut sets = UnionFind::new(witnesses.len());
        let mut evidence_index = HashMap::new();
        let mut source_index = HashMap::new();
        let mut origin_index = HashMap::new();
        let mut digest_index = HashMap::new();

        for (index, witness) in witnesses.iter().enumerate() {
            let evidence_id = witness.evidence_id.trim();
            let source_id = witness.source_id.trim();
            if evidence_id.is_empty() || source_id.is_empty() {
                push_reason_once(&mut reasons, R::MissingEvidenceIdentity);
            } else {
                remember_or_union(
                    &mut evidence_index,
                    evidence_id.to_owned(),
                    index,
                    &mut sets,
                );
                remember_or_union(
                    &mut source_index,
                    source_id.to_owned(),
                    index,
                    &mut sets,
                );
            }

            if let Some(origin_id) = witness.origin_id.as_deref() {
                let origin_id = origin_id.trim();
                if !origin_id.is_empty() {
                    remember_or_union(
                        &mut origin_index,
                        origin_id.to_owned(),
                        index,
                        &mut sets,
                    );
                }
            }

            if let Some(digest) = &witness.content_digest {
                match normalized_sha256(digest) {
                    Some(value) => remember_or_union(
                        &mut digest_index,
                        value,
                        index,
                        &mut sets,
                    ),
                    None => push_reason_once(&mut reasons, R::InvalidContentDigest),
                }
            }
        }

        let mut component_indexes: HashMap<usize, Vec<usize>> = HashMap::new();
        for index in 0..witnesses.len() {
            let root = sets.find(index);
            component_indexes.entry(root).or_default().push(index);
        }

        let mut ordered_components: Vec<Vec<usize>> = component_indexes.into_values().collect();
        ordered_components.sort_by_key(|component| component[0]);

        let mut groups = Vec::with_capacity(ordered_components.len());
        let mut attributed_origin_group_count = 0;
        let mut unresolved_lineage_group_count = 0;

        for component in ordered_components {
            let mut evidence_ids = Vec::new();
            let mut source_ids = Vec::new();
            let mut origin_ids = Vec::new();
            for index in &component {
                let witness = &witnesses[*index];
                if !witness.evidence_id.trim().is_empty() {
                    evidence_ids.push(witness.evidence_id.trim().to_owned());
                }
                if !witness.source_id.trim().is_empty() {
                    source_ids.push(witness.source_id.trim().to_owned());
                }
                if let Some(origin_id) = witness.origin_id.as_deref() {
                    if !origin_id.trim().is_empty() {
                        origin_ids.push(origin_id.trim().to_owned());
                    }
                }
            }
            evidence_ids.sort();
            evidence_ids.dedup();
            source_ids.sort();
            source_ids.dedup();
            origin_ids.sort();
            origin_ids.dedup();

            if origin_ids.len() > 1 {
                push_reason_once(&mut reasons, R::ConflictingOriginAssertions);
            }

            let origin_attributed = !origin_ids.is_empty();
            if origin_attributed {
                attributed_origin_group_count += 1;
            } else {
                unresolved_lineage_group_count += 1;
            }
            groups.push(EvidenceLineageGroup {
                evidence_ids,
                source_ids,
                origin_ids,
                origin_attributed,
            });
        }

        let duplicate_witness_count = witnesses.len().saturating_sub(groups.len());
        if unresolved_lineage_group_count > 0
            && (policy.require_all_witnesses_attributed
                || attributed_origin_group_count < policy.minimum_distinct_origins)
        {
            push_reason_once(&mut reasons, R::UnknownLineage);
        }
        if attributed_origin_group_count < policy.minimum_distinct_origins
            && unresolved_lineage_group_count == 0
        {
            push_reason_once(&mut reasons, R::InsufficientDistinctOrigins);
        }

        let has_hard_error = reasons.contains(&R::MissingEvidenceIdentity)
            || reasons.contains(&R::InvalidContentDigest);
        let decision = if has_hard_error || !reasons.is_empty() {
            EvidenceLineageDecision::Abstain
        } else if attributed_origin_group_count >= policy.minimum_distinct_origins {
            EvidenceLineageDecision::EligibleForReview
        } else {
            push_reason_once(&mut reasons, R::InsufficientDistinctOrigins);
            EvidenceLineageDecision::Abstain
        };

        Self {
            decision,
            reasons,
            raw_witness_count: witnesses.len(),
            duplicate_witness_count,
            attributed_origin_group_count,
            unresolved_lineage_group_count,
            groups,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness(
        evidence_id: &str,
        source_id: &str,
        origin_id: Option<&str>,
        digest: Option<&str>,
    ) -> EvidenceWitness {
        EvidenceWitness {
            evidence_id: evidence_id.to_owned(),
            source_id: source_id.to_owned(),
            origin_id: origin_id.map(str::to_owned),
            content_digest: digest.map(|value| ContentDigest::sha256(value.to_owned())),
        }
    }

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    #[test]
    fn ten_syndicated_citations_count_as_one_attributed_lineage() {
        let origin = "urn:origin:primary-report";
        let witnesses: Vec<_> = (0..10)
            .map(|index| {
                witness(
                    &format!("evidence-{index}"),
                    &format!("source-{index}"),
                    Some(origin),
                    Some(&digest('a')),
                )
            })
            .collect();

        let result = EvidenceLineageAssessment::assess(
            &witnesses,
            &EvidenceLineagePolicy::strict(),
        );
        assert_eq!(result.raw_witness_count, 10);
        assert_eq!(result.duplicate_witness_count, 9);
        assert_eq!(result.attributed_origin_group_count, 1);
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
        assert!(result
            .reasons
            .contains(&EvidenceLineageAbstentionReason::InsufficientDistinctOrigins));
    }

    #[test]
    fn different_origins_with_distinct_content_meet_review_threshold() {
        let witnesses = vec![
            witness("e1", "s1", Some("origin-A"), Some(&digest('a'))),
            witness("e2", "s2", Some("origin-B"), Some(&digest('b'))),
        ];
        let result = EvidenceLineageAssessment::assess(&witnesses, &EvidenceLineagePolicy::strict());
        assert_eq!(result.attributed_origin_group_count, 2);
        assert_eq!(result.unresolved_lineage_group_count, 0);
        assert_eq!(result.decision, EvidenceLineageDecision::EligibleForReview);
    }

    #[test]
    fn identical_content_with_different_origin_labels_is_not_counted_twice() {
        let witnesses = vec![
            witness("e1", "s1", Some("origin-A"), Some(&digest('a'))),
            witness("e2", "s2", Some("origin-B"), Some(&digest('a'))),
        ];
        let result = EvidenceLineageAssessment::assess(&witnesses, &EvidenceLineagePolicy::strict());
        assert_eq!(result.duplicate_witness_count, 1);
        assert_eq!(result.attributed_origin_group_count, 1);
        assert_eq!(result.groups[0].origin_ids, vec!["origin-A", "origin-B"]);
        assert!(result
            .reasons
            .contains(&EvidenceLineageAbstentionReason::ConflictingOriginAssertions));
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
    }

    #[test]
    fn unknown_origins_are_not_invented_from_distinct_urls() {
        let witnesses = vec![
            witness("e1", "url-A", None, Some(&digest('a'))),
            witness("e2", "url-B", None, Some(&digest('b'))),
        ];
        let result = EvidenceLineageAssessment::assess(&witnesses, &EvidenceLineagePolicy::strict());
        assert_eq!(result.attributed_origin_group_count, 0);
        assert_eq!(result.unresolved_lineage_group_count, 2);
        assert!(result.reasons.contains(&EvidenceLineageAbstentionReason::UnknownLineage));
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
    }

    #[test]
    fn identical_unattributed_copies_are_deduplicated_but_not_credited_as_origins() {
        let witnesses = vec![
            witness("e1", "url-A", None, Some(&digest('a'))),
            witness("e2", "url-B", None, Some(&digest('a'))),
            witness("e3", "url-C", None, Some(&digest('a'))),
        ];
        let result = EvidenceLineageAssessment::assess(
            &witnesses,
            &EvidenceLineagePolicy::new(1, false).unwrap(),
        );
        assert_eq!(result.duplicate_witness_count, 2);
        assert_eq!(result.unresolved_lineage_group_count, 1);
        assert_eq!(result.attributed_origin_group_count, 0);
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
    }

    #[test]
    fn malformed_digest_fails_closed_even_with_two_otherwise_distinct_origins() {
        let witnesses = vec![
            witness("e1", "s1", Some("origin-A"), Some(&digest('a'))),
            witness("e2", "s2", Some("origin-B"), Some(&digest('b'))),
            witness("e3", "s3", Some("origin-C"), Some("invalid")),
        ];
        let result = EvidenceLineageAssessment::assess(&witnesses, &EvidenceLineagePolicy::strict());
        assert!(result.reasons.contains(&EvidenceLineageAbstentionReason::InvalidContentDigest));
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
    }

    #[test]
    fn repeated_source_id_cannot_inflate_the_lineage_count() {
        let witnesses = vec![
            witness("e1", "same-source", Some("origin-A"), Some(&digest('a'))),
            witness("e2", "same-source", Some("origin-B"), Some(&digest('b'))),
            witness("e3", "other-source", Some("origin-C"), Some(&digest('c'))),
        ];
        let result = EvidenceLineageAssessment::assess(&witnesses, &EvidenceLineagePolicy::strict());
        assert_eq!(result.duplicate_witness_count, 1);
        assert_eq!(result.attributed_origin_group_count, 2);
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
    }

    #[test]
    fn empty_input_and_missing_identity_are_explicit() {
        let empty = EvidenceLineageAssessment::assess(&[], &EvidenceLineagePolicy::strict());
        assert_eq!(empty.reasons, vec![EvidenceLineageAbstentionReason::NoEvidence]);

        let invalid = vec![witness("", "", Some("origin-A"), Some(&digest('a')))];
        let result = EvidenceLineageAssessment::assess(&invalid, &EvidenceLineagePolicy::strict());
        assert!(result.reasons.contains(&EvidenceLineageAbstentionReason::MissingEvidenceIdentity));
        assert_eq!(result.decision, EvidenceLineageDecision::Abstain);
    }

    #[test]
    fn policy_rejects_zero_required_origins() {
        assert!(EvidenceLineagePolicy::new(0, true).is_err());
    }

    /// Opt-in release-mode benchmark of the complete in-memory grouping path.
    /// It excludes lineage extraction, network/source lookup, and persistence.
    #[test]
    #[ignore = "manual release-mode scalability benchmark; not a CI correctness gate"]
    fn benchmark_lineage_grouping_scaling() {
        use std::time::Instant;

        const REPEATS: usize = 9;
        let policy = EvidenceLineagePolicy::strict();

        for size in [1_000usize, 10_000, 100_000] {
            let witnesses: Vec<_> = (0..size)
                .map(|index| EvidenceWitness {
                    evidence_id: format!("evidence-{index}"),
                    source_id: format!("source-{index}"),
                    origin_id: Some(format!("origin-{index}")),
                    content_digest: Some(ContentDigest::sha256(format!("{index:064x}"))),
                })
                .collect();
            let expected = EvidenceLineageAssessment::assess(&witnesses, &policy);
            assert_eq!(expected.decision, EvidenceLineageDecision::EligibleForReview);
            assert_eq!(expected.attributed_origin_group_count, size);
            let mut durations_ns = Vec::with_capacity(REPEATS);

            for _ in 0..REPEATS {
                let start = Instant::now();
                let assessment = EvidenceLineageAssessment::assess(&witnesses, &policy);
                std::hint::black_box((
                    assessment.decision,
                    assessment.attributed_origin_group_count,
                    assessment.duplicate_witness_count,
                ));
                durations_ns.push(start.elapsed().as_nanos());
            }
            durations_ns.sort_unstable();
            let p50_ns = durations_ns[REPEATS / 2];
            let p95_index = (REPEATS * 95 + 99) / 100 - 1;
            let p95_ns = durations_ns[p95_index];
            println!(
                "n={size} reps={REPEATS} p50_ns={p50_ns} p95_ns={p95_ns} median_ns_per_item={}",
                p50_ns / size as u128
            );
        }
    }

}
