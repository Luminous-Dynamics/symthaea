//! Driver-owned replay boundary for delayed-outcome policy experiments.
//!
//! This module is an API/contract prototype. A policy adapter receives only newly delivered
//! features, newly revealed outcomes, and identities it could already have observed. Hidden
//! targets and future source events are held by ReplayScenario and never placed in the observation.
//! This limits accidental leakage through the typed interface; it is not a process sandbox and
//! does not protect against malicious code with independent access to the source dataset.
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeatureEvent {
    pub event_id: String,
    pub source_tick: u64,
    pub available_at_tick: u64,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RevealedOutcome {
    pub outcome_id: String,
    pub prediction_tick: u64,
    pub available_at_tick: u64,
    pub value: f64,
}

/// The only per-tick data passed to a deployable policy adapter.
/// The adapter can retain earlier deliveries in its own state, but cannot inspect hidden targets.
#[derive(Debug, Clone)]
pub struct PolicyObservation {
    pub decision_tick: u64,
    /// Features that first become available at this tick, not the hidden/future feature schedule.
    pub delivered_features: Vec<FeatureEvent>,
    /// Labels revealed at this tick, after their configured delay.
    pub newly_revealed_outcomes: Vec<RevealedOutcome>,
    /// IDs delivered no later than this tick. These allow explicit lineage for retained state.
    pub eligible_feature_ids: Vec<String>,
    pub eligible_outcome_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyAction {
    pub prediction: f64,
    pub consumed_feature_ids: Vec<String>,
    pub consumed_outcome_ids: Vec<String>,
}

pub trait PolicyAdapter {
    fn policy_id(&self) -> &str;
    fn decide(&mut self, observation: &PolicyObservation) -> anyhow::Result<PolicyAction>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverDecision {
    pub policy_id: String,
    pub decision_tick: u64,
    pub prediction: f64,
    pub delivered_feature_ids: Vec<String>,
    pub delivered_outcome_ids: Vec<String>,
    pub eligible_feature_ids: Vec<String>,
    pub eligible_outcome_ids: Vec<String>,
    pub consumed_feature_ids: Vec<String>,
    pub consumed_outcome_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayRunReceipt {
    pub schema_version: u32,
    pub stream_id: String,
    pub stream_length_ticks: u64,
    pub label_delay_ticks: u64,
    /// The feature-side source stream; contains no target/outcome values.
    pub feature_events: Vec<FeatureEvent>,
    /// Only labels that actually became available during the replay horizon.
    pub revealed_outcomes: Vec<RevealedOutcome>,
    pub decisions: Vec<DriverDecision>,
}

#[derive(Debug, Clone)]
pub struct ReplayScenario {
    stream_id: String,
    feature_events: Vec<FeatureEvent>,
    hidden_targets: Vec<f64>,
    label_delay_ticks: u64,
}

impl ReplayScenario {
    /// Construct the trusted driver-side scenario. Keep hidden_targets in the driver layer;
    /// do not pass this scenario or its backing dataset object to policy adapters.
    pub fn new(
        stream_id: String,
        mut feature_events: Vec<FeatureEvent>,
        hidden_targets: Vec<f64>,
        label_delay_ticks: u64,
    ) -> anyhow::Result<Self> {
        ensure!(!stream_id.trim().is_empty(), "stream_id must not be empty");
        ensure!(!hidden_targets.is_empty(), "hidden_targets must not be empty");
        ensure!(label_delay_ticks > 0, "label_delay_ticks must be positive");
        ensure!(
            hidden_targets.iter().all(|value| value.is_finite()),
            "hidden_targets must contain only finite values"
        );
        let ticks = hidden_targets.len() as u64;
        let mut ids = BTreeSet::new();
        for event in &feature_events {
            ensure!(!event.event_id.trim().is_empty(), "feature event_id must not be empty");
            ensure!(ids.insert(event.event_id.as_str()), "feature event IDs must be unique");
            ensure!(event.source_tick < ticks, "feature source_tick is outside the stream");
            ensure!(
                event.available_at_tick >= event.source_tick,
                "feature cannot be available before its source tick"
            );
            ensure!(
                event.available_at_tick < ticks,
                "feature availability tick is outside the replay horizon"
            );
            ensure!(event.value.is_finite(), "feature values must be finite");
        }
        feature_events.sort_by(|a, b| {
            (a.available_at_tick, a.source_tick, &a.event_id)
                .cmp(&(b.available_at_tick, b.source_tick, &b.event_id))
        });
        Ok(Self { stream_id, feature_events, hidden_targets, label_delay_ticks })
    }

    /// Execute one adapter from a clean policy state against the driver's immutable source stream.
    pub fn run<P: PolicyAdapter>(&self, policy: &mut P) -> anyhow::Result<ReplayRunReceipt> {
        let ticks = self.hidden_targets.len() as u64;
        ensure!(!policy.policy_id().trim().is_empty(), "policy_id must not be empty");
        let mut revealed_outcomes = Vec::<RevealedOutcome>::new();
        let mut decisions = Vec::with_capacity(ticks as usize);
        let mut delivered_feature_ids = BTreeSet::<String>::new();
        let mut delivered_outcome_ids = BTreeSet::<String>::new();

        for tick in 0..ticks {
            let delivered_features: Vec<FeatureEvent> = self.feature_events.iter()
                .filter(|event| event.available_at_tick == tick)
                .cloned()
                .collect();
            for event in &delivered_features {
                ensure!(
                    delivered_feature_ids.insert(event.event_id.clone()),
                    "driver attempted to deliver feature {} more than once",
                    event.event_id
                );
            }

            let newly_revealed_outcomes = if let Some(source_tick) = tick.checked_sub(self.label_delay_ticks) {
                let reveal = RevealedOutcome {
                    outcome_id: format!("outcome-{source_tick:08}"),
                    prediction_tick: source_tick,
                    available_at_tick: tick,
                    value: self.hidden_targets[source_tick as usize],
                };
                ensure!(
                    delivered_outcome_ids.insert(reveal.outcome_id.clone()),
                    "driver attempted to reveal outcome {} more than once",
                    reveal.outcome_id
                );
                revealed_outcomes.push(reveal.clone());
                vec![reveal]
            } else {
                Vec::new()
            };

            let observation = PolicyObservation {
                decision_tick: tick,
                delivered_features: delivered_features.clone(),
                newly_revealed_outcomes: newly_revealed_outcomes.clone(),
                eligible_feature_ids: delivered_feature_ids.iter().cloned().collect(),
                eligible_outcome_ids: delivered_outcome_ids.iter().cloned().collect(),
            };
            let action = policy.decide(&observation)
                .with_context(|| format!("policy {} failed at tick {tick}", policy.policy_id()))?;
            ensure!(action.prediction.is_finite(), "policy prediction must be finite at tick {tick}");
            validate_consumed_ids(
                &action.consumed_feature_ids,
                &observation.eligible_feature_ids,
                "feature",
                tick,
            )?;
            validate_consumed_ids(
                &action.consumed_outcome_ids,
                &observation.eligible_outcome_ids,
                "outcome",
                tick,
            )?;

            decisions.push(DriverDecision {
                policy_id: policy.policy_id().to_owned(),
                decision_tick: tick,
                prediction: action.prediction,
                delivered_feature_ids: delivered_features.iter().map(|event| event.event_id.clone()).collect(),
                delivered_outcome_ids: newly_revealed_outcomes.iter().map(|outcome| outcome.outcome_id.clone()).collect(),
                eligible_feature_ids: observation.eligible_feature_ids,
                eligible_outcome_ids: observation.eligible_outcome_ids,
                consumed_feature_ids: action.consumed_feature_ids,
                consumed_outcome_ids: action.consumed_outcome_ids,
            });
        }

        let receipt = ReplayRunReceipt {
            schema_version: 1,
            stream_id: self.stream_id.clone(),
            stream_length_ticks: ticks,
            label_delay_ticks: self.label_delay_ticks,
            feature_events: self.feature_events.clone(),
            revealed_outcomes,
            decisions,
        };
        audit_receipt(&receipt)?;
        Ok(receipt)
    }
}

/// Independently checks that each recorded policy consumption was delivered by that decision tick.
pub fn audit_receipt(receipt: &ReplayRunReceipt) -> anyhow::Result<()> {
    ensure!(receipt.schema_version == 1, "replay driver receipt schema_version must be 1");
    ensure!(!receipt.stream_id.trim().is_empty(), "stream_id must not be empty");
    ensure!(receipt.stream_length_ticks > 0, "stream_length_ticks must be positive");
    ensure!(receipt.label_delay_ticks > 0, "label_delay_ticks must be positive");
    ensure!(
        receipt.decisions.len() as u64 == receipt.stream_length_ticks,
        "receipt must contain one decision per stream tick"
    );

    let mut feature_ids = BTreeSet::new();
    for event in &receipt.feature_events {
        ensure!(!event.event_id.trim().is_empty(), "feature event_id must not be empty");
        ensure!(feature_ids.insert(event.event_id.as_str()), "duplicate feature event ID");
        ensure!(event.value.is_finite(), "feature values must be finite");
        ensure!(event.source_tick < receipt.stream_length_ticks, "feature source tick is out of range");
        ensure!(event.available_at_tick >= event.source_tick, "feature is available before source tick");
        ensure!(event.available_at_tick < receipt.stream_length_ticks, "feature availability is out of range");
    }

    let mut outcome_ids = BTreeSet::new();
    for outcome in &receipt.revealed_outcomes {
        ensure!(!outcome.outcome_id.trim().is_empty(), "outcome_id must not be empty");
        ensure!(outcome_ids.insert(outcome.outcome_id.as_str()), "duplicate outcome ID");
        ensure!(outcome.value.is_finite(), "revealed outcome must be finite");
        ensure!(outcome.prediction_tick < receipt.stream_length_ticks, "outcome prediction tick is out of range");
        ensure!(
            outcome.outcome_id == format!("outcome-{:08}", outcome.prediction_tick),
            "outcome ID does not bind to its prediction tick"
        );
        let expected_reveal_tick = outcome.prediction_tick.checked_add(receipt.label_delay_ticks)
            .context("outcome reveal tick overflow")?;
        ensure!(
            outcome.available_at_tick == expected_reveal_tick,
            "outcome reveal tick does not match the registered label delay"
        );
        ensure!(outcome.available_at_tick < receipt.stream_length_ticks,
            "receipt contains an outcome that was not available during the replay horizon");
    }

    let expected_policy_id = receipt.decisions.first()
        .map(|decision| decision.policy_id.as_str())
        .unwrap_or_default();
    ensure!(!expected_policy_id.trim().is_empty(), "decision policy_id must not be empty");
    for (index, decision) in receipt.decisions.iter().enumerate() {
        let tick = index as u64;
        ensure!(decision.decision_tick == tick, "decision ticks must be contiguous and ordered");
        ensure!(decision.policy_id == expected_policy_id, "policy_id changes within one replay receipt");
        ensure!(decision.prediction.is_finite(), "decision prediction must be finite");
        let expected_features: BTreeSet<String> = receipt.feature_events.iter()
            .filter(|event| event.available_at_tick <= tick)
            .map(|event| event.event_id.clone())
            .collect();
        let recorded_features: BTreeSet<String> = decision.eligible_feature_ids.iter().cloned().collect();
        ensure!(recorded_features.len() == decision.eligible_feature_ids.len(),
            "duplicate eligible feature ID at tick {tick}");
        ensure!(recorded_features == expected_features,
            "eligible feature set does not match driver delivery history at tick {tick}");
        let expected_outcomes: BTreeSet<String> = receipt.revealed_outcomes.iter()
            .filter(|outcome| outcome.available_at_tick <= tick)
            .map(|outcome| outcome.outcome_id.clone())
            .collect();
        let recorded_outcomes: BTreeSet<String> = decision.eligible_outcome_ids.iter().cloned().collect();
        ensure!(recorded_outcomes.len() == decision.eligible_outcome_ids.len(),
            "duplicate eligible outcome ID at tick {tick}");
        ensure!(recorded_outcomes == expected_outcomes,
            "eligible outcome set does not match reveal history at tick {tick}");

        let newly_delivered_features: BTreeSet<String> = receipt.feature_events.iter()
            .filter(|event| event.available_at_tick == tick)
            .map(|event| event.event_id.clone())
            .collect();
        ensure!(
            set_from_vec(&decision.delivered_feature_ids)? == newly_delivered_features,
            "delivered feature set differs from driver source at tick {tick}"
        );
        let newly_delivered_outcomes: BTreeSet<String> = receipt.revealed_outcomes.iter()
            .filter(|outcome| outcome.available_at_tick == tick)
            .map(|outcome| outcome.outcome_id.clone())
            .collect();
        ensure!(
            set_from_vec(&decision.delivered_outcome_ids)? == newly_delivered_outcomes,
            "delivered outcome set differs from driver reveal log at tick {tick}"
        );

        validate_consumed_ids(&decision.consumed_feature_ids, &decision.eligible_feature_ids, "feature", tick)?;
        validate_consumed_ids(&decision.consumed_outcome_ids, &decision.eligible_outcome_ids, "outcome", tick)?;
    }
    Ok(())
}

fn set_from_vec(values: &[String]) -> anyhow::Result<BTreeSet<String>> {
    let set: BTreeSet<String> = values.iter().cloned().collect();
    ensure!(set.len() == values.len(), "duplicate ID in driver receipt");
    Ok(set)
}

fn validate_consumed_ids(
    consumed: &[String],
    eligible: &[String],
    kind: &str,
    tick: u64,
) -> anyhow::Result<()> {
    let eligible: BTreeSet<&str> = eligible.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    for id in consumed {
        ensure!(!id.trim().is_empty(), "consumed {kind} ID must not be empty at tick {tick}");
        ensure!(seen.insert(id.as_str()), "duplicate consumed {kind} ID {id} at tick {tick}");
        ensure!(
            eligible.contains(id.as_str()),
            "policy consumed unknown or not-yet-delivered {kind} ID {id} at tick {tick}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConsumeDeliveredPolicy;
    impl PolicyAdapter for ConsumeDeliveredPolicy {
        fn policy_id(&self) -> &str { "consume-delivered" }
        fn decide(&mut self, observation: &PolicyObservation) -> anyhow::Result<PolicyAction> {
            Ok(PolicyAction {
                prediction: observation.newly_revealed_outcomes.last().map(|outcome| outcome.value).unwrap_or(0.0),
                consumed_feature_ids: observation.delivered_features.iter().map(|event| event.event_id.clone()).collect(),
                consumed_outcome_ids: observation.newly_revealed_outcomes.iter().map(|outcome| outcome.outcome_id.clone()).collect(),
            })
        }
    }

    struct FutureClaimingPolicy;
    impl PolicyAdapter for FutureClaimingPolicy {
        fn policy_id(&self) -> &str { "future-claiming" }
        fn decide(&mut self, _observation: &PolicyObservation) -> anyhow::Result<PolicyAction> {
            Ok(PolicyAction {
                prediction: 0.0,
                consumed_feature_ids: Vec::new(),
                // This is a real outcome ID for a future reveal, not an arbitrary unknown ID.
                consumed_outcome_ids: vec!["outcome-00000000".to_owned()],
            })
        }
    }

    fn features(values: &[f64]) -> Vec<FeatureEvent> {
        values.iter().enumerate().map(|(tick, value)| FeatureEvent {
            event_id: format!("feature-{tick:08}"),
            source_tick: tick as u64,
            available_at_tick: tick as u64,
            value: *value,
        }).collect()
    }

    fn scenario(targets: Vec<f64>) -> ReplayScenario {
        let feature_values: Vec<f64> = (0..targets.len()).map(|tick| tick as f64).collect();
        ReplayScenario::new("driver-contract-test".to_owned(), features(&feature_values), targets, 2)
            .expect("valid scenario")
    }

    #[test]
    fn delayed_outcomes_are_not_visible_before_the_registered_reveal_tick() {
        let mut policy = ConsumeDeliveredPolicy;
        let receipt = scenario(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
            .run(&mut policy).expect("driver run");
        assert!(receipt.decisions[0].eligible_outcome_ids.is_empty());
        assert!(receipt.decisions[1].eligible_outcome_ids.is_empty());
        assert_eq!(receipt.decisions[2].eligible_outcome_ids, vec!["outcome-00000000"]);
        assert_eq!(receipt.decisions[2].delivered_outcome_ids, vec!["outcome-00000000"]);
        assert_eq!(receipt.revealed_outcomes[0].available_at_tick, 2);
    }

    #[test]
    fn policy_cannot_claim_an_unknown_or_future_outcome() {
        let mut policy = FutureClaimingPolicy;
        let error = scenario(vec![0.0; 6]).run(&mut policy)
            .expect_err("unknown/future outcome reference must be rejected");
        assert!(error.to_string().contains("unknown or not-yet-delivered outcome ID"));
    }

    #[test]
    fn policy_cannot_consume_a_feature_before_its_delivery_tick() {
        struct ClaimDelayedFeature;
        impl PolicyAdapter for ClaimDelayedFeature {
            fn policy_id(&self) -> &str { "claim-delayed-feature" }
            fn decide(&mut self, _observation: &PolicyObservation) -> anyhow::Result<PolicyAction> {
                Ok(PolicyAction {
                    prediction: 0.0,
                    consumed_feature_ids: vec!["feature-arrives-later".to_owned()],
                    consumed_outcome_ids: Vec::new(),
                })
            }
        }
        let events = vec![
            FeatureEvent {
                event_id: "feature-now".to_owned(),
                source_tick: 0,
                available_at_tick: 0,
                value: 1.0,
            },
            FeatureEvent {
                event_id: "feature-arrives-later".to_owned(),
                source_tick: 0,
                available_at_tick: 3,
                value: 99.0,
            },
        ];
        let scenario = ReplayScenario::new(
            "delayed-feature-test".to_owned(), events, vec![0.0; 5], 1,
        ).expect("valid delayed-feature scenario");
        let error = scenario.run(&mut ClaimDelayedFeature)
            .expect_err("future feature consumption must fail");
        assert!(error.to_string().contains("unknown or not-yet-delivered feature ID"));
    }

    #[test]
    fn receipt_audit_rejects_a_future_outcome_backfilled_into_an_earlier_decision() {
        let mut policy = ConsumeDeliveredPolicy;
        let mut receipt = scenario(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
            .run(&mut policy).expect("driver run");
        receipt.decisions[0].eligible_outcome_ids.push("outcome-00000000".to_owned());
        let error = audit_receipt(&receipt).expect_err("receipt must reject future outcome");
        assert!(error.to_string().contains("eligible outcome set does not match reveal history"));
    }

    #[test]
    fn changing_hidden_future_targets_cannot_change_earlier_decisions() {
        let original_targets = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut changed_targets = original_targets.clone();
        changed_targets[4] = -400.0;
        changed_targets[5] = 900.0;
        let mut first_policy = ConsumeDeliveredPolicy;
        let mut second_policy = ConsumeDeliveredPolicy;
        let first = scenario(original_targets).run(&mut first_policy).expect("first replay");
        let second = scenario(changed_targets).run(&mut second_policy).expect("second replay");
        // Outcomes from ticks 4 and 5 arrive at ticks 6 and 7, respectively.
        let first_prefix = serde_json::to_value(&first.decisions[..6]).unwrap();
        let second_prefix = serde_json::to_value(&second.decisions[..6]).unwrap();
        assert_eq!(first_prefix, second_prefix);
        assert_ne!(first.decisions[6].prediction, second.decisions[6].prediction);
    }

    #[test]
    fn receipt_audit_rejects_duplicated_consumption_ids() {
        let mut policy = ConsumeDeliveredPolicy;
        let mut receipt = scenario(vec![0.0; 6]).run(&mut policy).expect("driver run");
        receipt.decisions[1].consumed_feature_ids = vec![
            "feature-00000001".to_owned(),
            "feature-00000001".to_owned(),
        ];
        let error = audit_receipt(&receipt).expect_err("duplicated consumption must fail");
        assert!(error.to_string().contains("duplicate consumed feature ID"));
    }
}
