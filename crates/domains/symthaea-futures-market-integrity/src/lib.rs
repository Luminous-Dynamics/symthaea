// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic-only integrity primitives for prediction-market research.
//!
//! This crate audits three narrow failure modes: repeated accounts that share a declared
//! information-source group, turnover whose buyer and seller map to the same declared beneficial
//! owner, and incomplete event-resolution policies. It deliberately does not connect to a venue,
//! infer real identities, classify intent, recommend trades, or claim independent information.
//!
//! A successful result means only that these local structural checks passed. Source-group labels
//! and beneficial-owner labels must be qualified by a separate evidence system; this crate does
//! not authenticate them.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};

/// Structural failure detected by the synthetic integrity harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityError {
    /// The forecast set contained no observations.
    EmptySignalSet,
    /// A required identifier was empty or whitespace-only.
    EmptyIdentifier(&'static str),
    /// A probability was non-finite or outside the closed interval [0, 1].
    InvalidProbability,
    /// The same account identifier occurred more than once in one forecast snapshot.
    DuplicateAccount(String),
    /// A trade fill had zero notional.
    ZeroNotional,
    /// A trade fill identifier occurred more than once in one audit window.
    DuplicateFill(String),
    /// Checked arithmetic failed.
    ArithmeticOverflow,
    /// A required resolution-policy field was absent or empty.
    MissingResolutionField(&'static str),
    /// The Unix-second deadline was zero or missing.
    InvalidDeadline,
    /// The dispute window was zero or missing.
    InvalidDisputeWindow,
}

/// A finite probability in the closed interval [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Validate a raw probability before it enters a forecast summary.
    pub fn new(value: f64) -> Result<Self, IntegrityError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(IntegrityError::InvalidProbability)
        }
    }

    /// Return the validated probability.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A forecast submitted by one account, tagged with its declared information-source cluster.
///
/// Source-cluster labels are an input to a sensitivity analysis, not authenticated facts.
/// Unknown or unclassified sources should be placed in an explicit shared unknown group rather
/// than silently treating every account as independent.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastSignal {
    account_id: String,
    source_group_id: String,
    probability: Probability,
}

impl ForecastSignal {
    /// Build a forecast signal after validating its identifiers and probability.
    pub fn new(
        account_id: impl Into<String>,
        source_group_id: impl Into<String>,
        probability: f64,
    ) -> Result<Self, IntegrityError> {
        let account_id = required_identifier(account_id.into(), "account_id")?;
        let source_group_id =
            required_identifier(source_group_id.into(), "source_group_id")?;
        Ok(Self {
            account_id,
            source_group_id,
            probability: Probability::new(probability)?,
        })
    }

    /// Account identifier supplied for this snapshot.
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Declared information-source cluster supplied for this snapshot.
    pub fn source_group_id(&self) -> &str {
        &self.source_group_id
    }

    /// Validated probability supplied for this snapshot.
    pub fn probability(&self) -> Probability {
        self.probability
    }
}

/// Forecast aggregation diagnostics that keep account count separate from source concentration.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalIntegritySummary {
    /// Number of unique accounts in the snapshot.
    pub account_count: usize,
    /// Number of declared source groups. This is not a verified independent-source count.
    pub declared_source_group_count: usize,
    /// Equal-account mean; vulnerable to repeated accounts sharing one source.
    pub account_weighted_mean: f64,
    /// Mean of within-group means, giving each declared source group one vote.
    pub source_capped_mean: f64,
    /// Always false: source independence is not established by this crate.
    pub source_independence_verified: bool,
}

/// Summarize a forecast snapshot with both naive and declared-source-capped aggregation.
///
/// The source-capped mean first averages all reports within each declared group and then averages
/// those group means equally. This is a diagnostic baseline, not a universally optimal pooling
/// rule: the grouping may be wrong, sources may differ in quality, and equal source weight can
/// discard useful information. The function explicitly does not verify independence.
pub fn summarize_signals(
    signals: &[ForecastSignal],
) -> Result<SignalIntegritySummary, IntegrityError> {
    if signals.is_empty() {
        return Err(IntegrityError::EmptySignalSet);
    }

    let mut accounts = BTreeSet::new();
    let mut group_totals: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    let mut account_total = 0.0_f64;

    for signal in signals {
        if !accounts.insert(signal.account_id.as_str()) {
            return Err(IntegrityError::DuplicateAccount(
                signal.account_id.clone(),
            ));
        }

        let p = signal.probability.get();
        account_total += p;
        let entry = group_totals
            .entry(signal.source_group_id.as_str())
            .or_insert((0.0, 0));
        entry.0 += p;
        entry.1 += 1;
    }

    let account_weighted_mean = account_total / signals.len() as f64;
    let source_capped_mean = group_totals
        .values()
        .map(|(total, count)| total / *count as f64)
        .sum::<f64>()
        / group_totals.len() as f64;

    Ok(SignalIntegritySummary {
        account_count: signals.len(),
        declared_source_group_count: group_totals.len(),
        account_weighted_mean,
        source_capped_mean,
        source_independence_verified: false,
    })
}

/// One filled trade, using opaque beneficial-owner identifiers supplied by an external identity
/// or surveillance process and integer minor currency units for notional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradeFill {
    fill_id: String,
    buyer_owner_id: String,
    seller_owner_id: String,
    notional_minor_units: u64,
}

impl TradeFill {
    /// Construct a fill. A zero-notional fill is rejected rather than counted as activity.
    pub fn new(
        fill_id: impl Into<String>,
        buyer_owner_id: impl Into<String>,
        seller_owner_id: impl Into<String>,
        notional_minor_units: u64,
    ) -> Result<Self, IntegrityError> {
        let fill_id = required_identifier(fill_id.into(), "fill_id")?;
        let buyer_owner_id =
            required_identifier(buyer_owner_id.into(), "buyer_owner_id")?;
        let seller_owner_id =
            required_identifier(seller_owner_id.into(), "seller_owner_id")?;

        if notional_minor_units == 0 {
            return Err(IntegrityError::ZeroNotional);
        }

        Ok(Self {
            fill_id,
            buyer_owner_id,
            seller_owner_id,
            notional_minor_units,
        })
    }
}

/// Turnover diagnostics for an execution window.
///
/// Same-owner turnover is a lower-bound structural alert when beneficial-owner labels are sound.
/// This does not detect wash trading between distinct owners, prove intent, or establish that
/// the remaining notional represents economically meaningful risk transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeIntegritySummary {
    /// Sum of all fill notional in integer minor currency units.
    pub gross_notional_minor_units: u128,
    /// Notional where the buyer and seller have the same supplied beneficial-owner identifier.
    pub same_owner_notional_minor_units: u128,
    /// Gross notional minus same-owner notional; not a proof of net risk transfer.
    pub non_same_owner_notional_minor_units: u128,
    /// Number of unique unordered pairs of distinct buyer/seller owner identifiers.
    pub distinct_external_owner_pairs: usize,
    /// Same-owner notional as parts per million of gross notional; zero when gross is zero.
    pub same_owner_share_ppm: u32,
    /// Always false: economic risk transfer is not proven by fill aggregation.
    pub net_risk_transfer_verified: bool,
}

/// Audit fill identifiers, gross turnover and same-owner turnover with checked arithmetic.
pub fn audit_trade_fills(
    fills: &[TradeFill],
) -> Result<TradeIntegritySummary, IntegrityError> {
    let mut fill_ids = BTreeSet::new();
    let mut pairs: BTreeSet<(String, String)> = BTreeSet::new();
    let mut gross = 0_u128;
    let mut same_owner = 0_u128;

    for fill in fills {
        if !fill_ids.insert(fill.fill_id.as_str()) {
            return Err(IntegrityError::DuplicateFill(fill.fill_id.clone()));
        }

        gross = gross
            .checked_add(u128::from(fill.notional_minor_units))
            .ok_or(IntegrityError::ArithmeticOverflow)?;

        if fill.buyer_owner_id == fill.seller_owner_id {
            same_owner = same_owner
                .checked_add(u128::from(fill.notional_minor_units))
                .ok_or(IntegrityError::ArithmeticOverflow)?;
        } else {
            let (a, b) = if fill.buyer_owner_id < fill.seller_owner_id {
                (
                    fill.buyer_owner_id.clone(),
                    fill.seller_owner_id.clone(),
                )
            } else {
                (
                    fill.seller_owner_id.clone(),
                    fill.buyer_owner_id.clone(),
                )
            };
            pairs.insert((a, b));
        }
    }

    let non_same_owner = gross
        .checked_sub(same_owner)
        .ok_or(IntegrityError::ArithmeticOverflow)?;
    let share_ppm = if gross == 0 {
        0
    } else {
        ((same_owner as f64 / gross as f64) * 1_000_000.0).round() as u32
    };

    Ok(TradeIntegritySummary {
        gross_notional_minor_units: gross,
        same_owner_notional_minor_units: same_owner,
        non_same_owner_notional_minor_units: non_same_owner,
        distinct_external_owner_pairs: pairs.len(),
        same_owner_share_ppm: share_ppm,
        net_risk_transfer_verified: false,
    })
}

/// Candidate event-resolution policy, representing raw input before admission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolutionPolicyCandidate {
    /// Exact question/contract text.
    pub contract_text: Option<String>,
    /// Identifier for the predeclared primary resolution source.
    pub primary_source_id: Option<String>,
    /// Explicit fallback or no-fallback/void-market rule.
    pub fallback_rule: Option<String>,
    /// Resolution deadline as Unix seconds in UTC.
    pub deadline_unix_seconds: Option<u64>,
    /// Rule for corrections, restatements, and revisions to the primary source.
    pub source_revision_policy: Option<String>,
    /// Minimum dispute window in seconds.
    pub dispute_window_seconds: Option<u64>,
    /// Named independent-adjudication and appeal policy identifier.
    pub adjudication_policy: Option<String>,
}

/// A resolution policy whose required fields passed local structural admission.
///
/// This type prevents a caller from accidentally confusing a raw incomplete candidate with a
/// structurally complete policy. It does not freeze the policy, authenticate the source, prove
/// adjudicator independence, or determine whether a real-world event occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedResolutionPolicy {
    contract_text: String,
    primary_source_id: String,
    fallback_rule: String,
    deadline_unix_seconds: u64,
    source_revision_policy: String,
    dispute_window_seconds: u64,
    adjudication_policy: String,
}

impl QualifiedResolutionPolicy {
    /// Exact contract text recorded in the admitted policy.
    pub fn contract_text(&self) -> &str {
        &self.contract_text
    }

    /// Primary source identifier recorded in the admitted policy.
    pub fn primary_source_id(&self) -> &str {
        &self.primary_source_id
    }

    /// Fallback or explicit no-fallback rule recorded in the admitted policy.
    pub fn fallback_rule(&self) -> &str {
        &self.fallback_rule
    }

    /// Deadline in Unix seconds (UTC).
    pub fn deadline_unix_seconds(&self) -> u64 {
        self.deadline_unix_seconds
    }

    /// Source revision policy.
    pub fn source_revision_policy(&self) -> &str {
        &self.source_revision_policy
    }

    /// Challenge/dispute window in seconds.
    pub fn dispute_window_seconds(&self) -> u64 {
        self.dispute_window_seconds
    }

    /// Adjudication and appeal policy identifier.
    pub fn adjudication_policy(&self) -> &str {
        &self.adjudication_policy
    }
}

/// Admit a resolution-policy candidate only when all required semantic fields are present.
///
/// Completeness is not correctness: external qualification must still check source authority,
/// exact contract bytes, policy versioning, adjudicator conflicts, and the relevant legal rules.
pub fn qualify_resolution_policy(
    candidate: &ResolutionPolicyCandidate,
) -> Result<QualifiedResolutionPolicy, IntegrityError> {
    let contract_text = required_field(
        candidate.contract_text.as_deref(),
        "contract_text",
    )?;
    let primary_source_id = required_field(
        candidate.primary_source_id.as_deref(),
        "primary_source_id",
    )?;
    let fallback_rule =
        required_field(candidate.fallback_rule.as_deref(), "fallback_rule")?;
    let source_revision_policy = required_field(
        candidate.source_revision_policy.as_deref(),
        "source_revision_policy",
    )?;
    let adjudication_policy = required_field(
        candidate.adjudication_policy.as_deref(),
        "adjudication_policy",
    )?;

    let deadline_unix_seconds = candidate
        .deadline_unix_seconds
        .filter(|deadline| *deadline > 0)
        .ok_or(IntegrityError::InvalidDeadline)?;
    let dispute_window_seconds = candidate
        .dispute_window_seconds
        .filter(|window| *window > 0)
        .ok_or(IntegrityError::InvalidDisputeWindow)?;

    Ok(QualifiedResolutionPolicy {
        contract_text,
        primary_source_id,
        fallback_rule,
        deadline_unix_seconds,
        source_revision_policy,
        dispute_window_seconds,
        adjudication_policy,
    })
}

fn required_identifier(value: String, field: &'static str) -> Result<String, IntegrityError> {
    if value.trim().is_empty() {
        return Err(IntegrityError::EmptyIdentifier(field));
    }
    Ok(value.trim().to_owned())
}

fn required_field(
    value: Option<&str>,
    field: &'static str,
) -> Result<String, IntegrityError> {
    let value = value.ok_or(IntegrityError::MissingResolutionField(field))?;
    if value.trim().is_empty() {
        return Err(IntegrityError::MissingResolutionField(field));
    }
    Ok(value.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(account: &str, source: &str, probability: f64) -> ForecastSignal {
        ForecastSignal::new(account, source, probability).expect("valid fixture signal")
    }

    fn complete_policy() -> ResolutionPolicyCandidate {
        ResolutionPolicyCandidate {
            contract_text: Some(
                "Will event X occur by 2030-01-01T00:00:00Z?".to_owned(),
            ),
            primary_source_id: Some("official-source-v1".to_owned()),
            fallback_rule: Some("void if both named sources are unavailable".to_owned()),
            deadline_unix_seconds: Some(1_893_456_000),
            source_revision_policy: Some(
                "use official revision published within 72 hours".to_owned(),
            ),
            dispute_window_seconds: Some(86_400),
            adjudication_policy: Some("independent-review-v1".to_owned()),
        }
    }

    #[test]
    fn probability_rejects_non_finite_and_out_of_range_values() {
        assert_eq!(Probability::new(f64::NAN), Err(IntegrityError::InvalidProbability));
        assert_eq!(Probability::new(f64::INFINITY), Err(IntegrityError::InvalidProbability));
        assert_eq!(Probability::new(-0.01), Err(IntegrityError::InvalidProbability));
        assert_eq!(Probability::new(1.01), Err(IntegrityError::InvalidProbability));
        assert_eq!(Probability::new(0.0).unwrap().get(), 0.0);
        assert_eq!(Probability::new(1.0).unwrap().get(), 1.0);
    }

    #[test]
    fn empty_forecast_snapshot_is_rejected() {
        assert_eq!(summarize_signals(&[]), Err(IntegrityError::EmptySignalSet));
    }

    #[test]
    fn duplicate_account_in_one_snapshot_is_rejected() {
        let signals = vec![
            signal("account-1", "source-a", 0.3),
            signal("account-1", "source-b", 0.7),
        ];
        assert_eq!(
            summarize_signals(&signals),
            Err(IntegrityError::DuplicateAccount("account-1".to_owned()))
        );
    }

    #[test]
    fn source_capping_prevents_one_declared_source_from_winning_by_account_count() {
        let mut signals = (0..100)
            .map(|i| signal(&format!("copied-account-{i}"), "copied-source", 0.99))
            .collect::<Vec<_>>();
        signals.push(signal("independent-account", "other-source", 0.20));

        let summary = summarize_signals(&signals).unwrap();
        assert_eq!(summary.account_count, 101);
        assert_eq!(summary.declared_source_group_count, 2);
        assert!(summary.account_weighted_mean > 0.98);
        assert!((summary.source_capped_mean - 0.595).abs() < 1e-12);
        assert!(!summary.source_independence_verified);
    }

    #[test]
    fn copying_accounts_within_a_source_group_does_not_change_source_capped_mean() {
        for copies in [1, 2, 10, 100, 1_000] {
            let mut signals = (0..copies)
                .map(|i| signal(&format!("copy-{i}"), "same-source", 0.84))
                .collect::<Vec<_>>();
            signals.push(signal("other", "different-declared-source", 0.24));
            let summary = summarize_signals(&signals).unwrap();
            assert!((summary.source_capped_mean - 0.54).abs() < 1e-12);
            assert!(!summary.source_independence_verified);
        }
    }

    #[test]
    fn equal_source_groups_are_averaged_after_within_group_aggregation() {
        let signals = vec![
            signal("a1", "a", 0.2),
            signal("a2", "a", 0.4),
            signal("b1", "b", 0.8),
        ];
        let summary = summarize_signals(&signals).unwrap();
        assert!((summary.account_weighted_mean - (1.4 / 3.0)).abs() < 1e-12);
        assert!((summary.source_capped_mean - 0.55).abs() < 1e-12);
    }

    #[test]
    fn identifiers_must_not_be_blank() {
        assert_eq!(
            ForecastSignal::new("   ", "source", 0.5),
            Err(IntegrityError::EmptyIdentifier("account_id"))
        );
        assert_eq!(
            TradeFill::new("fill", "buyer", " ", 1),
            Err(IntegrityError::EmptyIdentifier("seller_owner_id"))
        );
    }

    #[test]
    fn same_owner_turnover_is_separated_from_other_owner_turnover() {
        let fills = vec![
            TradeFill::new("f1", "owner-a", "owner-a", 9_000).unwrap(),
            TradeFill::new("f2", "owner-a", "owner-b", 700).unwrap(),
            TradeFill::new("f3", "owner-b", "owner-a", 300).unwrap(),
        ];
        let summary = audit_trade_fills(&fills).unwrap();
        assert_eq!(summary.gross_notional_minor_units, 10_000);
        assert_eq!(summary.same_owner_notional_minor_units, 9_000);
        assert_eq!(summary.non_same_owner_notional_minor_units, 1_000);
        assert_eq!(summary.distinct_external_owner_pairs, 1);
        assert_eq!(summary.same_owner_share_ppm, 900_000);
        assert!(!summary.net_risk_transfer_verified);
    }

    #[test]
    fn repeated_fill_identifier_is_rejected() {
        let fills = vec![
            TradeFill::new("same", "a", "b", 10).unwrap(),
            TradeFill::new("same", "b", "c", 20).unwrap(),
        ];
        assert_eq!(
            audit_trade_fills(&fills),
            Err(IntegrityError::DuplicateFill("same".to_owned()))
        );
    }

    #[test]
    fn zero_notional_fills_are_rejected() {
        assert_eq!(
            TradeFill::new("f", "a", "b", 0),
            Err(IntegrityError::ZeroNotional)
        );
    }

    #[test]
    fn empty_trade_window_has_zero_metrics_without_claiming_verified_transfer() {
        let summary = audit_trade_fills(&[]).unwrap();
        assert_eq!(summary.gross_notional_minor_units, 0);
        assert_eq!(summary.same_owner_notional_minor_units, 0);
        assert_eq!(summary.same_owner_share_ppm, 0);
        assert!(!summary.net_risk_transfer_verified);
    }

    #[test]
    fn trade_summary_is_independent_of_fill_order() {
        let a = TradeFill::new("f1", "owner-a", "owner-b", 100).unwrap();
        let b = TradeFill::new("f2", "owner-b", "owner-c", 200).unwrap();
        let c = TradeFill::new("f3", "owner-c", "owner-b", 300).unwrap();
        assert_eq!(
            audit_trade_fills(&[a.clone(), b.clone(), c.clone()]).unwrap(),
            audit_trade_fills(&[c, a, b]).unwrap()
        );
    }

    #[test]
    fn resolution_policy_with_a_missing_field_is_rejected() {
        let mut candidate = complete_policy();
        candidate.fallback_rule = None;
        assert_eq!(
            qualify_resolution_policy(&candidate),
            Err(IntegrityError::MissingResolutionField("fallback_rule"))
        );
    }

    #[test]
    fn every_required_resolution_field_is_checked_when_missing() {
        let mutations: [(&str, fn(&mut ResolutionPolicyCandidate)); 7] = [
            ("contract_text", |c| c.contract_text = None),
            ("primary_source_id", |c| c.primary_source_id = None),
            ("fallback_rule", |c| c.fallback_rule = None),
            ("deadline_unix_seconds", |c| c.deadline_unix_seconds = None),
            ("source_revision_policy", |c| c.source_revision_policy = None),
            ("dispute_window_seconds", |c| c.dispute_window_seconds = None),
            ("adjudication_policy", |c| c.adjudication_policy = None),
        ];

        for (field, mutate) in mutations {
            let mut candidate = complete_policy();
            mutate(&mut candidate);
            let result = qualify_resolution_policy(&candidate);
            match field {
                "deadline_unix_seconds" => {
                    assert_eq!(result, Err(IntegrityError::InvalidDeadline));
                }
                "dispute_window_seconds" => {
                    assert_eq!(result, Err(IntegrityError::InvalidDisputeWindow));
                }
                _ => {
                    assert_eq!(result, Err(IntegrityError::MissingResolutionField(field)));
                }
            }
        }
    }

    #[test]
    fn whitespace_resolution_fields_are_rejected() {
        let mut candidate = complete_policy();
        candidate.primary_source_id = Some("   ".to_owned());
        assert_eq!(
            qualify_resolution_policy(&candidate),
            Err(IntegrityError::MissingResolutionField("primary_source_id"))
        );
    }

    #[test]
    fn zero_deadline_is_rejected() {
        let mut candidate = complete_policy();
        candidate.deadline_unix_seconds = Some(0);
        assert_eq!(
            qualify_resolution_policy(&candidate),
            Err(IntegrityError::InvalidDeadline)
        );
    }

    #[test]
    fn zero_dispute_window_is_rejected() {
        let mut candidate = complete_policy();
        candidate.dispute_window_seconds = Some(0);
        assert_eq!(
            qualify_resolution_policy(&candidate),
            Err(IntegrityError::InvalidDisputeWindow)
        );
    }

    #[test]
    fn complete_policy_can_be_admitted_but_policy_truth_is_not_implied() {
        let policy = qualify_resolution_policy(&complete_policy()).unwrap();
        assert!(policy.contract_text().starts_with("Will event X"));
        assert_eq!(policy.primary_source_id(), "official-source-v1");
        assert_eq!(policy.deadline_unix_seconds(), 1_893_456_000);
        assert_eq!(policy.dispute_window_seconds(), 86_400);
    }
}
