//! Effective-dated regulatory rule applicability with historical knowledge-time replay.
//!
//! The resolver distinguishes:
//! - valid time: the transaction/project date for which a rule is queried;
//! - knowledge time: the timestamp at which the decision maker froze its evidence.
//!
//! A later publication never silently rewrites an earlier decision. Only an
//! explicit, same-scope supersession chain can resolve overlapping revisions.
//! This is a source-bound version resolver, not a legal interpreter: it does
//! not infer rule meaning, participant eligibility or transitional obligations.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RulePublicationState {
    Draft,
    Published,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleRevisionEvidence<'a> {
    pub revision_id: &'a str,
    pub authority_id: &'a str,
    pub jurisdiction_id: &'a str,
    pub instrument_id: &'a str,
    /// Narrow rule/product scope key, e.g. "DSU/instantaneous-reserve".
    pub scope_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub publication_state: RulePublicationState,
    pub published_at_utc: Option<i64>,
    /// When this exact source record became available to the decision workflow.
    pub available_at_utc: i64,
    pub effective_from_utc: i64,
    /// Exclusive end; None means no supplied end date.
    pub effective_until_utc: Option<i64>,
    /// Explicit predecessor revision. No supersession is inferred from labels
    /// or publication order.
    pub supersedes_revision_id: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleApplicabilityQuery<'a> {
    pub authority_id: &'a str,
    pub jurisdiction_id: &'a str,
    pub instrument_id: &'a str,
    pub scope_id: &'a str,
    pub valid_at_utc: i64,
    pub known_at_utc: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApplicableRuleRevision<'a> {
    pub revision_id: &'a str,
    pub authority_id: &'a str,
    pub jurisdiction_id: &'a str,
    pub instrument_id: &'a str,
    pub scope_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub published_at_utc: i64,
    pub available_at_utc: i64,
    pub effective_from_utc: i64,
    pub effective_until_utc: Option<i64>,
    pub superseded_known_revision_count: u32,
    pub future_known_excluded_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleApplicabilityResult<'a> {
    Applicable(ApplicableRuleRevision<'a>),
    NoPublishedRuleForValidTime { draft_revision_count: u32 },
    IndeterminateFutureRevisionNotKnown { excluded_revision_count: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleApplicabilityError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidEffectiveWindow,
    InvalidPublicationChronology,
    DraftHasPublicationTimestamp,
    PublishedRuleMissingPublicationTimestamp,
    AvailableBeforePublication,
    DuplicateRevisionId,
    DuplicateSourceReferenceWithDifferentDigest,
    SelfSupersession,
    MissingSupersededRevision,
    CrossScopeSupersession,
    SupersessionCycle,
    AmbiguousApplicableRevisions,
    ArithmeticOverflow,
    InvalidQuery,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn matches_scope(revision: RuleRevisionEvidence<'_>, query: RuleApplicabilityQuery<'_>) -> bool {
    revision.authority_id == query.authority_id
        && revision.jurisdiction_id == query.jurisdiction_id
        && revision.instrument_id == query.instrument_id
        && revision.scope_id == query.scope_id
}

fn effective_at(revision: RuleRevisionEvidence<'_>, valid_at_utc: i64) -> bool {
    revision.effective_from_utc <= valid_at_utc
        && revision
            .effective_until_utc
            .map(|end| valid_at_utc < end)
            .unwrap_or(true)
}

fn validate_revision(revision: RuleRevisionEvidence<'_>) -> Result<(), RuleApplicabilityError> {
    if !nonempty(revision.revision_id)
        || !nonempty(revision.authority_id)
        || !nonempty(revision.jurisdiction_id)
        || !nonempty(revision.instrument_id)
        || !nonempty(revision.scope_id)
        || !nonempty(revision.source_ref)
    {
        return Err(RuleApplicabilityError::InvalidEvidenceReference);
    }
    if !valid_sha256(revision.source_sha256) {
        return Err(RuleApplicabilityError::InvalidSourceDigest);
    }
    if revision.available_at_utc < 0
        || revision.effective_from_utc < 0
        || revision.effective_until_utc.is_some_and(|end| end < 0)
    {
        return Err(RuleApplicabilityError::InvalidPublicationChronology);
    }
    if revision
        .effective_until_utc
        .is_some_and(|end| end <= revision.effective_from_utc)
    {
        return Err(RuleApplicabilityError::InvalidEffectiveWindow);
    }
    if revision
        .supersedes_revision_id
        .is_some_and(|previous| !nonempty(previous))
    {
        return Err(RuleApplicabilityError::InvalidEvidenceReference);
    }
    match (revision.publication_state, revision.published_at_utc) {
        (RulePublicationState::Draft, Some(_)) => {
            return Err(RuleApplicabilityError::DraftHasPublicationTimestamp);
        }
        (RulePublicationState::Published, None) => {
            return Err(RuleApplicabilityError::PublishedRuleMissingPublicationTimestamp);
        }
        (RulePublicationState::Published, Some(published_at)) => {
            if published_at < 0 {
                return Err(RuleApplicabilityError::InvalidPublicationChronology);
            }
            if revision.available_at_utc < published_at {
                return Err(RuleApplicabilityError::AvailableBeforePublication);
            }
            if revision.effective_from_utc < published_at {
                // Retrospective instruments may exist, but their retroactivity
                // must be separately reviewed; this generic resolver refuses it.
                return Err(RuleApplicabilityError::InvalidPublicationChronology);
            }
        }
        (RulePublicationState::Draft, None) => {}
    }
    Ok(())
}

fn find_revision_index(revisions: &[RuleRevisionEvidence<'_>], revision_id: &str) -> Option<usize> {
    revisions
        .iter()
        .position(|revision| revision.revision_id == revision_id)
}

fn validate_supersession_graph(
    revisions: &[RuleRevisionEvidence<'_>],
) -> Result<(), RuleApplicabilityError> {
    for (index, revision) in revisions.iter().copied().enumerate() {
        if revisions[..index]
            .iter()
            .any(|previous| previous.revision_id == revision.revision_id)
        {
            return Err(RuleApplicabilityError::DuplicateRevisionId);
        }
        for previous in &revisions[..index] {
            if previous.source_ref == revision.source_ref
                && previous.source_sha256 != revision.source_sha256
            {
                return Err(RuleApplicabilityError::DuplicateSourceReferenceWithDifferentDigest);
            }
        }
        if let Some(parent_id) = revision.supersedes_revision_id {
            if parent_id == revision.revision_id {
                return Err(RuleApplicabilityError::SelfSupersession);
            }
            let Some(parent_index) = find_revision_index(revisions, parent_id) else {
                return Err(RuleApplicabilityError::MissingSupersededRevision);
            };
            let parent = revisions[parent_index];
            if parent.authority_id != revision.authority_id
                || parent.jurisdiction_id != revision.jurisdiction_id
                || parent.instrument_id != revision.instrument_id
                || parent.scope_id != revision.scope_id
            {
                return Err(RuleApplicabilityError::CrossScopeSupersession);
            }
            if revision.effective_from_utc < parent.effective_from_utc {
                return Err(RuleApplicabilityError::InvalidEffectiveWindow);
            }
        }
    }

    for revision in revisions.iter().copied() {
        let mut current = revision;
        let mut visited = 0_usize;
        while let Some(parent_id) = current.supersedes_revision_id {
            visited = visited
                .checked_add(1)
                .ok_or(RuleApplicabilityError::ArithmeticOverflow)?;
            if visited > revisions.len() {
                return Err(RuleApplicabilityError::SupersessionCycle);
            }
            let Some(parent_index) = find_revision_index(revisions, parent_id) else {
                return Err(RuleApplicabilityError::MissingSupersededRevision);
            };
            current = revisions[parent_index];
        }
    }
    Ok(())
}

/// Returns true when `descendant_index` supersedes `ancestor_index` directly
/// or through a complete chain already checked for cycles and missing links.
fn is_descendant_of(
    revisions: &[RuleRevisionEvidence<'_>],
    descendant_index: usize,
    ancestor_index: usize,
) -> bool {
    let ancestor_id = revisions[ancestor_index].revision_id;
    let mut current = revisions[descendant_index];
    while let Some(parent_id) = current.supersedes_revision_id {
        if parent_id == ancestor_id {
            return true;
        }
        let Some(parent_index) = find_revision_index(revisions, parent_id) else {
            return false;
        };
        current = revisions[parent_index];
    }
    false
}

/// Resolve exactly one applicable revision for the supplied authority,
/// jurisdiction, instrument and rule scope. Historical replay excludes any
/// rule source unavailable at `known_at_utc`, even when that revision later
/// became effective for `valid_at_utc`.
pub fn resolve_applicable_rule<'a>(
    query: RuleApplicabilityQuery<'_>,
    revisions: &'a [RuleRevisionEvidence<'a>],
) -> Result<RuleApplicabilityResult<'a>, RuleApplicabilityError> {
    if !nonempty(query.authority_id)
        || !nonempty(query.jurisdiction_id)
        || !nonempty(query.instrument_id)
        || !nonempty(query.scope_id)
        || query.known_at_utc < 0
        || query.valid_at_utc < 0
    {
        return Err(RuleApplicabilityError::InvalidQuery);
    }
    for revision in revisions.iter().copied() {
        validate_revision(revision)?;
    }
    validate_supersession_graph(revisions)?;

    let mut known_candidates = Vec::new();
    let mut future_known_count = 0_u32;
    let mut draft_count = 0_u32;
    for (index, revision) in revisions.iter().copied().enumerate() {
        if !matches_scope(revision, query) || !effective_at(revision, query.valid_at_utc) {
            continue;
        }
        match revision.publication_state {
            RulePublicationState::Draft => {
                draft_count = draft_count
                    .checked_add(1)
                    .ok_or(RuleApplicabilityError::ArithmeticOverflow)?;
            }
            RulePublicationState::Published => {
                if revision.available_at_utc > query.known_at_utc {
                    future_known_count = future_known_count
                        .checked_add(1)
                        .ok_or(RuleApplicabilityError::ArithmeticOverflow)?;
                } else {
                    known_candidates.push(index);
                }
            }
        }
    }

    if known_candidates.is_empty() {
        if future_known_count > 0 {
            return Ok(
                RuleApplicabilityResult::IndeterminateFutureRevisionNotKnown {
                    excluded_revision_count: future_known_count,
                },
            );
        }
        return Ok(RuleApplicabilityResult::NoPublishedRuleForValidTime {
            draft_revision_count: draft_count,
        });
    }

    let mut maximal_candidates = Vec::new();
    for candidate in known_candidates.iter().copied() {
        let superseded_by_another = known_candidates
            .iter()
            .copied()
            .any(|other| other != candidate && is_descendant_of(revisions, other, candidate));
        if !superseded_by_another {
            maximal_candidates.push(candidate);
        }
    }

    if maximal_candidates.len() != 1 {
        return Err(RuleApplicabilityError::AmbiguousApplicableRevisions);
    }
    let selected = revisions[maximal_candidates[0]];
    let published_at_utc = selected
        .published_at_utc
        .ok_or(RuleApplicabilityError::PublishedRuleMissingPublicationTimestamp)?;
    let superseded_known_revision_count =
        u32::try_from(known_candidates.len().saturating_sub(1))
            .map_err(|_| RuleApplicabilityError::ArithmeticOverflow)?;

    Ok(RuleApplicabilityResult::Applicable(
        ApplicableRuleRevision {
            revision_id: selected.revision_id,
            authority_id: selected.authority_id,
            jurisdiction_id: selected.jurisdiction_id,
            instrument_id: selected.instrument_id,
            scope_id: selected.scope_id,
            source_ref: selected.source_ref,
            source_sha256: selected.source_sha256,
            published_at_utc,
            available_at_utc: selected.available_at_utc,
            effective_from_utc: selected.effective_from_utc,
            effective_until_utc: selected.effective_until_utc,
            superseded_known_revision_count,
            future_known_excluded_count: future_known_count,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const NEW_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const ALT_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn revision(
        id: &'static str,
        source: &'static str,
        digest: &'static str,
        published: i64,
        available: i64,
        effective_from: i64,
        effective_until: Option<i64>,
        supersedes: Option<&'static str>,
    ) -> RuleRevisionEvidence<'static> {
        RuleRevisionEvidence {
            revision_id: id,
            authority_id: "NERSA",
            jurisdiction_id: "ZA",
            instrument_id: "SAWEM-MARKET-CODE",
            scope_id: "DSU/RESERVE",
            source_ref: source,
            source_sha256: digest,
            publication_state: RulePublicationState::Published,
            published_at_utc: Some(published),
            available_at_utc: available,
            effective_from_utc: effective_from,
            effective_until_utc: effective_until,
            supersedes_revision_id: supersedes,
        }
    }

    fn query(valid_at: i64, known_at: i64) -> RuleApplicabilityQuery<'static> {
        RuleApplicabilityQuery {
            authority_id: "NERSA",
            jurisdiction_id: "ZA",
            instrument_id: "SAWEM-MARKET-CODE",
            scope_id: "DSU/RESERVE",
            valid_at_utc: valid_at,
            known_at_utc: known_at,
        }
    }

    #[test]
    fn historical_decision_uses_only_the_rule_revision_known_then() {
        let rules = [
            revision("v2", "code/rev2", NEW_SHA, 180, 185, 200, None, Some("v1")),
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, None, None),
        ];
        let result = resolve_applicable_rule(query(220, 170), &rules).unwrap();
        let RuleApplicabilityResult::Applicable(selected) = result else {
            panic!("v1 was published, available and effective at the decision frontier");
        };
        assert_eq!(selected.revision_id, "v1");
        assert_eq!(selected.future_known_excluded_count, 1);
        assert_eq!(selected.superseded_known_revision_count, 0);
    }

    #[test]
    fn effective_superseding_revision_wins_only_after_it_is_known() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, None, None),
            revision("v2", "code/rev2", NEW_SHA, 180, 185, 200, None, Some("v1")),
        ];
        let result = resolve_applicable_rule(query(220, 220), &rules).unwrap();
        let RuleApplicabilityResult::Applicable(selected) = result else {
            panic!("v2 is available and effective at both query times");
        };
        assert_eq!(selected.revision_id, "v2");
        assert_eq!(selected.superseded_known_revision_count, 1);
        assert_eq!(selected.future_known_excluded_count, 0);
    }

    #[test]
    fn later_publication_does_not_rewrite_prepublication_replay() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, Some(200), None),
            revision("v2", "code/rev2", NEW_SHA, 210, 215, 200, None, Some("v1")),
        ];
        let result = resolve_applicable_rule(query(220, 205), &rules).unwrap();
        assert_eq!(
            result,
            RuleApplicabilityResult::IndeterminateFutureRevisionNotKnown {
                excluded_revision_count: 1,
            }
        );
    }

    #[test]
    fn overlapping_unrelated_revisions_are_ambiguous_not_arbitrarily_ordered() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, None, None),
            revision(
                "branch-a",
                "code/a",
                NEW_SHA,
                180,
                185,
                200,
                None,
                Some("v1"),
            ),
            revision(
                "branch-b",
                "code/b",
                ALT_SHA,
                181,
                186,
                200,
                None,
                Some("v1"),
            ),
        ];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &rules),
            Err(RuleApplicabilityError::AmbiguousApplicableRevisions)
        );
    }

    #[test]
    fn missing_superseded_revision_is_indeterminate_by_error_not_assumed() {
        let rules = [revision(
            "v2",
            "code/rev2",
            NEW_SHA,
            180,
            185,
            200,
            None,
            Some("v1"),
        )];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &rules),
            Err(RuleApplicabilityError::MissingSupersededRevision)
        );
    }

    #[test]
    fn supersession_cannot_cross_authority_or_rule_scope() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, None, None),
            RuleRevisionEvidence {
                authority_id: "NTCSA",
                ..revision("v2", "code/rev2", NEW_SHA, 180, 185, 200, None, Some("v1"))
            },
        ];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &rules),
            Err(RuleApplicabilityError::CrossScopeSupersession)
        );
    }

    #[test]
    fn supersession_cycle_is_rejected() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, None, Some("v2")),
            revision("v2", "code/rev2", NEW_SHA, 180, 185, 200, None, Some("v1")),
        ];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &rules),
            Err(RuleApplicabilityError::SupersessionCycle)
        );
    }

    #[test]
    fn draft_and_not_yet_effective_rules_do_not_become_applicable() {
        let draft = RuleRevisionEvidence {
            publication_state: RulePublicationState::Draft,
            published_at_utc: None,
            effective_from_utc: 200,
            ..revision("draft", "draft/code", OLD_SHA, 0, 10, 200, None, None)
        };
        let future_published =
            revision("future", "code/future", NEW_SHA, 180, 185, 300, None, None);
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &[draft, future_published]),
            Ok(RuleApplicabilityResult::NoPublishedRuleForValidTime {
                draft_revision_count: 1,
            })
        );
    }

    #[test]
    fn wrong_authority_jurisdiction_instrument_and_scope_do_not_match() {
        let wrong = [
            RuleRevisionEvidence {
                authority_id: "NTCSA",
                ..revision(
                    "wrong-authority",
                    "rule/a",
                    OLD_SHA,
                    90,
                    92,
                    100,
                    None,
                    None,
                )
            },
            RuleRevisionEvidence {
                jurisdiction_id: "NA",
                ..revision(
                    "wrong-jurisdiction",
                    "rule/b",
                    NEW_SHA,
                    90,
                    92,
                    100,
                    None,
                    None,
                )
            },
            RuleRevisionEvidence {
                instrument_id: "GRID-CODE",
                ..revision(
                    "wrong-instrument",
                    "rule/c",
                    ALT_SHA,
                    90,
                    92,
                    100,
                    None,
                    None,
                )
            },
            RuleRevisionEvidence {
                scope_id: "ENERGY/ARBITRAGE",
                ..revision("wrong-scope", "rule/d", OLD_SHA, 90, 92, 100, None, None)
            },
        ];
        assert_eq!(
            resolve_applicable_rule(query(150, 160), &wrong),
            Ok(RuleApplicabilityResult::NoPublishedRuleForValidTime {
                draft_revision_count: 0,
            })
        );
    }

    #[test]
    fn effective_window_is_start_inclusive_and_end_exclusive() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, Some(200), None),
            revision("v2", "code/rev2", NEW_SHA, 180, 185, 200, None, Some("v1")),
        ];
        let before = resolve_applicable_rule(query(199, 220), &rules).unwrap();
        let RuleApplicabilityResult::Applicable(before) = before else {
            panic!("v1 remains effective up to but not including the v2 boundary");
        };
        assert_eq!(before.revision_id, "v1");

        let at_boundary = resolve_applicable_rule(query(200, 220), &rules).unwrap();
        let RuleApplicabilityResult::Applicable(at_boundary) = at_boundary else {
            panic!("v2 becomes effective at its start boundary");
        };
        assert_eq!(at_boundary.revision_id, "v2");
    }

    #[test]
    fn not_yet_effective_published_amendment_does_not_displace_current_rule() {
        let rules = [
            revision("v1", "code/rev1", OLD_SHA, 90, 92, 100, None, None),
            revision("v2", "code/rev2", NEW_SHA, 180, 185, 200, None, Some("v1")),
        ];
        let result = resolve_applicable_rule(query(199, 190), &rules).unwrap();
        let RuleApplicabilityResult::Applicable(selected) = result else {
            panic!("v2 is published but not effective for valid time 199");
        };
        assert_eq!(selected.revision_id, "v1");
    }

    #[test]
    fn draft_or_published_state_metadata_must_be_consistent() {
        let draft_with_publication = [RuleRevisionEvidence {
            publication_state: RulePublicationState::Draft,
            published_at_utc: Some(90),
            ..revision("draft-bad", "draft/bad", OLD_SHA, 90, 92, 100, None, None)
        }];
        assert_eq!(
            resolve_applicable_rule(query(150, 160), &draft_with_publication),
            Err(RuleApplicabilityError::DraftHasPublicationTimestamp)
        );

        let published_without_publication = [RuleRevisionEvidence {
            published_at_utc: None,
            ..revision(
                "published-bad",
                "published/bad",
                OLD_SHA,
                90,
                92,
                100,
                None,
                None,
            )
        }];
        assert_eq!(
            resolve_applicable_rule(query(150, 160), &published_without_publication),
            Err(RuleApplicabilityError::PublishedRuleMissingPublicationTimestamp)
        );
    }

    #[test]
    fn inconsistent_source_digest_and_invalid_publication_order_are_rejected() {
        let inconsistent = [
            revision("v1", "same/source", OLD_SHA, 90, 92, 100, None, None),
            revision(
                "v2",
                "same/source",
                NEW_SHA,
                180,
                185,
                200,
                None,
                Some("v1"),
            ),
        ];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &inconsistent),
            Err(RuleApplicabilityError::DuplicateSourceReferenceWithDifferentDigest)
        );

        let before_publication = [revision(
            "future-effective",
            "rule/future",
            OLD_SHA,
            180,
            179,
            200,
            None,
            None,
        )];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &before_publication),
            Err(RuleApplicabilityError::AvailableBeforePublication)
        );
    }

    #[test]
    fn duplicate_revision_ids_and_invalid_windows_fail_closed() {
        let duplicate = [
            revision("same", "rule/one", OLD_SHA, 90, 92, 100, None, None),
            revision("same", "rule/two", NEW_SHA, 100, 102, 110, None, None),
        ];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &duplicate),
            Err(RuleApplicabilityError::DuplicateRevisionId)
        );

        let invalid_window = [revision(
            "bad-window",
            "rule/bad",
            OLD_SHA,
            90,
            92,
            100,
            Some(100),
            None,
        )];
        assert_eq!(
            resolve_applicable_rule(query(220, 220), &invalid_window),
            Err(RuleApplicabilityError::InvalidEffectiveWindow)
        );
    }
}
