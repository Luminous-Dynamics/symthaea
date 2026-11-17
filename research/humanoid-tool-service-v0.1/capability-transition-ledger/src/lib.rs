//! Evidence-preserving capability-transition metrics for humanoid tool-use episodes.
//!
//! Missing instrumentation, instrumented absence and observed occurrence remain
//! distinct. Events are not assumed to form a strict funnel: co-occurrence and
//! temporally ordered conversion are computed separately.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

pub const EVENT_COUNT: usize = 5;
pub const TRANSITION_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventKind {
    CandidateContact,
    CorrectFirstContact,
    CorrectToolLift,
    ToolTargetContact,
    TaskSuccess,
}

pub const EVENT_KINDS: [EventKind; EVENT_COUNT] = [
    EventKind::CandidateContact,
    EventKind::CorrectFirstContact,
    EventKind::CorrectToolLift,
    EventKind::ToolTargetContact,
    EventKind::TaskSuccess,
];

pub const EVENT_TRANSITIONS: [(EventKind, EventKind); TRANSITION_COUNT] = [
    (EventKind::CandidateContact, EventKind::CorrectFirstContact),
    (EventKind::CorrectFirstContact, EventKind::CorrectToolLift),
    (EventKind::CorrectToolLift, EventKind::ToolTargetContact),
    (EventKind::ToolTargetContact, EventKind::TaskSuccess),
];

impl EventKind {
    pub const fn index(self) -> usize {
        match self {
            Self::CandidateContact => 0,
            Self::CorrectFirstContact => 1,
            Self::CorrectToolLift => 2,
            Self::ToolTargetContact => 3,
            Self::TaskSuccess => 4,
        }
    }
}

/// The evaluator must never turn missing telemetry into a negative event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "timestamp_ns", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventEvidence {
    /// This event was not instrumented for the episode.
    NotInstrumented,
    /// Instrumentation was active and the event did not occur.
    InstrumentedAbsent,
    /// The event occurred at this episode-clock timestamp.
    OccurredAtNs(u64),
}

impl EventEvidence {
    pub const fn is_instrumented(self) -> bool {
        !matches!(self, Self::NotInstrumented)
    }

    pub const fn timestamp_ns(self) -> Option<u64> {
        match self {
            Self::OccurredAtNs(timestamp) => Some(timestamp),
            Self::NotInstrumented | Self::InstrumentedAbsent => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExecutionLevel {
    L0SelectionPickup,
    L1StationaryUse,
    L2MobileUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ToolSetMode {
    Standard,
    Decoy,
}

/// Named wire-format event collection for capability-transition-ledger-v0.1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventReceiptSet {
    pub candidate_contact: EventEvidence,
    pub correct_first_contact: EventEvidence,
    pub correct_tool_lift: EventEvidence,
    pub tool_target_contact: EventEvidence,
    pub task_success: EventEvidence,
}

impl EventReceiptSet {
    pub fn into_array(self) -> [EventEvidence; EVENT_COUNT] {
        [
            self.candidate_contact,
            self.correct_first_contact,
            self.correct_tool_lift,
            self.tool_target_contact,
            self.task_success,
        ]
    }
}

/// JSON receipt matching schema-v0.1.json. The schema version is checked
/// before conversion into the evaluator's internal representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventLedgerReceiptV0_1 {
    pub schema_version: String,
    pub episode_id: String,
    pub policy_id: String,
    pub split_id: String,
    pub scenario_id: String,
    pub execution_level: ExecutionLevel,
    pub tool_set_mode: ToolSetMode,
    pub clock_domain_id: String,
    pub source_id: String,
    pub independent_verifier_id: Option<String>,
    pub events: EventReceiptSet,
}

impl TryFrom<EventLedgerReceiptV0_1> for EpisodeEventLedger {
    type Error = String;

    fn try_from(receipt: EventLedgerReceiptV0_1) -> Result<Self, Self::Error> {
        if receipt.schema_version != "0.1" {
            return Err(format!(
                "unsupported capability-transition receipt schema_version {:?}",
                receipt.schema_version
            ));
        }
        let ledger = EpisodeEventLedger {
            episode_id: receipt.episode_id,
            policy_id: receipt.policy_id,
            split_id: receipt.split_id,
            scenario_id: receipt.scenario_id,
            execution_level: receipt.execution_level,
            tool_set_mode: receipt.tool_set_mode,
            clock_domain_id: receipt.clock_domain_id,
            source_id: receipt.source_id,
            independent_verifier_id: receipt.independent_verifier_id,
            events: receipt.events.into_array(),
        };
        ledger.validate()?;
        Ok(ledger)
    }
}

/// Deserialize a JSON array of v0.1 episode receipts into validated evaluator rows.
///
/// This function does not infer missing events or synthesize timestamps. A caller
/// must supply source-provided timestamps in the episode's declared clock domain.
pub fn deserialize_json_receipts(input: &str) -> Result<Vec<EpisodeEventLedger>, String> {
    let receipts: Vec<EventLedgerReceiptV0_1> = serde_json::from_str(input)
        .map_err(|error| format!("invalid capability-transition receipt JSON: {error}"))?;
    receipts.into_iter().map(EpisodeEventLedger::try_from).collect()
}

/// One episode's event evidence, all timestamps belonging to clock_domain_id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeEventLedger {
    pub episode_id: String,
    pub policy_id: String,
    pub split_id: String,
    pub scenario_id: String,
    pub execution_level: ExecutionLevel,
    pub tool_set_mode: ToolSetMode,
    pub clock_domain_id: String,
    pub source_id: String,
    /// Separate evaluator/verifier identity for the task-success channel.
    /// This is provenance, not proof of organizational independence.
    pub independent_verifier_id: Option<String>,
    pub events: [EventEvidence; EVENT_COUNT],
}

impl EpisodeEventLedger {
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("episode_id", self.episode_id.as_str()),
            ("policy_id", self.policy_id.as_str()),
            ("split_id", self.split_id.as_str()),
            ("scenario_id", self.scenario_id.as_str()),
            ("clock_domain_id", self.clock_domain_id.as_str()),
            ("source_id", self.source_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{field} must not be empty"));
            }
        }
        if self.events.iter().all(|event| !event.is_instrumented()) {
            return Err("episode has no instrumented events".into());
        }
        if self.events[EventKind::TaskSuccess.index()].is_instrumented() {
            let verifier = self.independent_verifier_id.as_deref()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "instrumented task-success evidence requires independent_verifier_id".to_string())?;
            if verifier == self.source_id || verifier == self.policy_id {
                return Err("task-success verifier identity must differ from policy_id and event source_id".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellKey {
    pub policy_id: String,
    pub split_id: String,
    pub scenario_id: String,
    pub execution_level: ExecutionLevel,
    pub tool_set_mode: ToolSetMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventRateSummary {
    pub event: EventKind,
    /// Number of episodes where this event was actually instrumented.
    pub instrumented_episodes: usize,
    /// Number of instrumented episodes where the event occurred.
    pub occurred_episodes: usize,
    /// occurred / instrumented; None when no episode instrumented this event.
    pub incidence_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionSummary {
    pub from_event: EventKind,
    pub to_event: EventKind,
    /// Episodes where both event channels were instrumented.
    pub jointly_instrumented_episodes: usize,
    /// Episodes where the source event occurred and both channels were instrumented.
    pub from_occurred_episodes: usize,
    /// Source and target both occurred, irrespective of timestamp order.
    pub cooccurrence_episodes: usize,
    /// Both occurred and target timestamp was >= source timestamp.
    pub temporally_ordered_episodes: usize,
    /// Mean (target - source) timestamp among temporally ordered co-occurrences.
    pub mean_latency_ns: Option<f64>,
    /// cooccurrence / from_occurred; None when source event has no eligible occurrence.
    pub conditional_cooccurrence_rate: Option<f64>,
    /// ordered / from_occurred; None when source event has no eligible occurrence.
    pub ordered_transition_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellSummary {
    pub key: CellKey,
    pub episode_count: usize,
    pub event_rates: Vec<EventRateSummary>,
    pub transitions: Vec<TransitionSummary>,
}

/// Summarize one set of episodes, preserving policy/task/split strata.
///
/// Rows that differ on any CellKey field are never pooled together. Duplicate
/// episode IDs are rejected globally rather than silently double-counted.
pub fn summarize_by_cell(rows: &[EpisodeEventLedger]) -> Result<Vec<CellSummary>, String> {
    let mut episode_ids = HashSet::with_capacity(rows.len());
    let mut grouped: BTreeMap<CellKey, Vec<&EpisodeEventLedger>> = BTreeMap::new();

    for row in rows {
        row.validate()?;
        if !episode_ids.insert(row.episode_id.as_str()) {
            return Err(format!("duplicate episode_id {:?}", row.episode_id));
        }
        let key = CellKey {
            policy_id: row.policy_id.clone(),
            split_id: row.split_id.clone(),
            scenario_id: row.scenario_id.clone(),
            execution_level: row.execution_level,
            tool_set_mode: row.tool_set_mode,
        };
        grouped.entry(key).or_default().push(row);
    }

    let mut summaries = Vec::with_capacity(grouped.len());
    for (key, episodes) in grouped {
        let event_rates = EVENT_KINDS
            .iter()
            .copied()
            .map(|event| summarize_event(event, &episodes))
            .collect();
        let transitions = EVENT_TRANSITIONS
            .iter()
            .copied()
            .map(|(from, to)| summarize_transition(from, to, &episodes))
            .collect();

        summaries.push(CellSummary {
            key,
            episode_count: episodes.len(),
            event_rates,
            transitions,
        });
    }
    Ok(summaries)
}

fn summarize_event(
    event: EventKind,
    episodes: &[&EpisodeEventLedger],
) -> EventRateSummary {
    let mut instrumented = 0usize;
    let mut occurred = 0usize;
    for episode in episodes {
        let evidence = episode.events[event.index()];
        if evidence.is_instrumented() {
            instrumented += 1;
            if evidence.timestamp_ns().is_some() {
                occurred += 1;
            }
        }
    }
    EventRateSummary {
        event,
        instrumented_episodes: instrumented,
        occurred_episodes: occurred,
        incidence_rate: ratio(occurred, instrumented),
    }
}

fn summarize_transition(
    from_event: EventKind,
    to_event: EventKind,
    episodes: &[&EpisodeEventLedger],
) -> TransitionSummary {
    let mut jointly_instrumented = 0usize;
    let mut from_occurred = 0usize;
    let mut cooccurrence = 0usize;
    let mut ordered = 0usize;
    let mut latency_sum_ns = 0u128;

    for episode in episodes {
        let from = episode.events[from_event.index()];
        let to = episode.events[to_event.index()];
        if !from.is_instrumented() || !to.is_instrumented() {
            continue;
        }
        jointly_instrumented += 1;

        let Some(from_time) = from.timestamp_ns() else {
            continue;
        };
        from_occurred += 1;

        let Some(to_time) = to.timestamp_ns() else {
            continue;
        };
        cooccurrence += 1;

        if to_time >= from_time {
            ordered += 1;
            latency_sum_ns += u128::from(to_time - from_time);
        }
    }

    TransitionSummary {
        from_event,
        to_event,
        jointly_instrumented_episodes: jointly_instrumented,
        from_occurred_episodes: from_occurred,
        cooccurrence_episodes: cooccurrence,
        temporally_ordered_episodes: ordered,
        mean_latency_ns: if ordered == 0 {
            None
        } else {
            Some(latency_sum_ns as f64 / ordered as f64)
        },
        conditional_cooccurrence_rate: ratio(cooccurrence, from_occurred),
        ordered_transition_rate: ratio(ordered, from_occurred),
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        episode_id: &str,
        level: ExecutionLevel,
        mode: ToolSetMode,
        events: [EventEvidence; EVENT_COUNT],
    ) -> EpisodeEventLedger {
        EpisodeEventLedger {
            episode_id: episode_id.into(),
            policy_id: "policy-a".into(),
            split_id: "heldout-v1".into(),
            scenario_id: "BallMove".into(),
            execution_level: level,
            tool_set_mode: mode,
            clock_domain_id: format!("sim-clock-{episode_id}"),
            source_id: "episode-event-recorder".into(),
            independent_verifier_id: Some("external-task-oracle".into()),
            events,
        }
    }

    fn ordered_success() -> [EventEvidence; EVENT_COUNT] {
        [
            EventEvidence::OccurredAtNs(10),
            EventEvidence::OccurredAtNs(20),
            EventEvidence::OccurredAtNs(30),
            EventEvidence::OccurredAtNs(40),
            EventEvidence::OccurredAtNs(50),
        ]
    }

    #[test]
    fn json_fixtures_deserialize_and_preserve_cell_strata() {
        let rows = deserialize_json_receipts(include_str!("../fixtures-v0.1.json")).unwrap();
        assert_eq!(rows.len(), 6);
        let summaries = summarize_by_cell(&rows).unwrap();
        assert_eq!(summaries.len(), 5);
    }

    #[test]
    fn json_receipts_reject_unknown_schema_versions_and_unknown_fields() {
        let unknown_version = r#"[{
            "schema_version":"9.9",
            "episode_id":"episode-1",
            "policy_id":"policy-a",
            "split_id":"heldout",
            "scenario_id":"BallMove",
            "execution_level":"L1_STATIONARY_USE",
            "tool_set_mode":"STANDARD",
            "clock_domain_id":"sim-clock",
            "source_id":"event-recorder",
            "independent_verifier_id":"task-oracle",
            "events":{
                "candidate_contact":{"status":"NOT_INSTRUMENTED"},
                "correct_first_contact":{"status":"NOT_INSTRUMENTED"},
                "correct_tool_lift":{"status":"NOT_INSTRUMENTED"},
                "tool_target_contact":{"status":"NOT_INSTRUMENTED"},
                "task_success":{"status":"NOT_INSTRUMENTED"}
            }
        }]"#;
        assert!(deserialize_json_receipts(unknown_version).is_err());

        let mut valid = include_str!("../fixtures-v0.1.json").to_string();
        valid = valid.replacen("\"episode_id\": \"ct-001\",", "\"unexpected\": true,\"episode_id\": \"ct-001\",", 1);
        assert!(deserialize_json_receipts(&valid).is_err());
    }

    #[test]
    fn missing_instrumentation_is_not_counted_as_absence() {
        let episode = row(
            "episode-1",
            ExecutionLevel::L1StationaryUse,
            ToolSetMode::Standard,
            [
                EventEvidence::OccurredAtNs(10),
                EventEvidence::InstrumentedAbsent,
                EventEvidence::NotInstrumented,
                EventEvidence::NotInstrumented,
                EventEvidence::NotInstrumented,
            ],
        );
        let summary = summarize_by_cell(&[episode]).unwrap();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].event_rates[0].incidence_rate, Some(1.0));
        assert_eq!(summary[0].event_rates[1].incidence_rate, Some(0.0));
        assert_eq!(summary[0].event_rates[2].instrumented_episodes, 0);
        assert_eq!(summary[0].event_rates[2].incidence_rate, None);
    }

    #[test]
    fn cooccurrence_and_temporal_order_are_reported_separately() {
        let first = row(
            "episode-1",
            ExecutionLevel::L1StationaryUse,
            ToolSetMode::Standard,
            ordered_success(),
        );
        let second = row(
            "episode-2",
            ExecutionLevel::L1StationaryUse,
            ToolSetMode::Standard,
            [
                EventEvidence::OccurredAtNs(10),
                EventEvidence::OccurredAtNs(30),
                EventEvidence::OccurredAtNs(20),
                EventEvidence::OccurredAtNs(40),
                EventEvidence::InstrumentedAbsent,
            ],
        );
        let summary = summarize_by_cell(&[first, second]).unwrap();
        let lift_transition = &summary[0].transitions[1];
        assert_eq!(lift_transition.from_occurred_episodes, 2);
        assert_eq!(lift_transition.cooccurrence_episodes, 2);
        assert_eq!(lift_transition.temporally_ordered_episodes, 1);
        assert_eq!(lift_transition.conditional_cooccurrence_rate, Some(1.0));
        assert_eq!(lift_transition.ordered_transition_rate, Some(0.5));
        assert_eq!(lift_transition.mean_latency_ns, Some(10.0));
    }

    #[test]
    fn task_cells_are_never_pooled_together() {
        let stationary = row(
            "episode-1",
            ExecutionLevel::L1StationaryUse,
            ToolSetMode::Standard,
            ordered_success(),
        );
        let mobile_decoy = row(
            "episode-2",
            ExecutionLevel::L2MobileUse,
            ToolSetMode::Decoy,
            ordered_success(),
        );
        let summary = summarize_by_cell(&[stationary, mobile_decoy]).unwrap();
        assert_eq!(summary.len(), 2);
        assert_ne!(summary[0].key, summary[1].key);
    }

    #[test]
    fn duplicate_episode_identity_is_rejected() {
        let first = row(
            "same-episode",
            ExecutionLevel::L0SelectionPickup,
            ToolSetMode::Standard,
            ordered_success(),
        );
        let second = row(
            "same-episode",
            ExecutionLevel::L0SelectionPickup,
            ToolSetMode::Standard,
            ordered_success(),
        );
        assert!(summarize_by_cell(&[first, second]).is_err());
    }

    #[test]
    fn non_funnel_event_order_is_preserved_not_rejected() {
        let episode = row(
            "episode-1",
            ExecutionLevel::L2MobileUse,
            ToolSetMode::Decoy,
            [
                EventEvidence::InstrumentedAbsent,
                EventEvidence::OccurredAtNs(30),
                EventEvidence::OccurredAtNs(20),
                EventEvidence::OccurredAtNs(40),
                EventEvidence::InstrumentedAbsent,
            ],
        );
        let summary = summarize_by_cell(&[episode]).unwrap();
        let contact_to_lift = &summary[0].transitions[1];
        assert_eq!(contact_to_lift.cooccurrence_episodes, 1);
        assert_eq!(contact_to_lift.temporally_ordered_episodes, 0);
        assert_eq!(contact_to_lift.conditional_cooccurrence_rate, Some(1.0));
        assert_eq!(contact_to_lift.ordered_transition_rate, Some(0.0));
    }

    #[test]
    fn an_episode_with_no_instrumented_event_is_rejected() {
        let episode = row(
            "episode-1",
            ExecutionLevel::L0SelectionPickup,
            ToolSetMode::Standard,
            [EventEvidence::NotInstrumented; EVENT_COUNT],
        );
        assert!(summarize_by_cell(&[episode]).is_err());
    }

    #[test]
    fn task_success_requires_a_separate_verifier_identity() {
        let mut episode = row(
            "episode-1",
            ExecutionLevel::L1StationaryUse,
            ToolSetMode::Standard,
            ordered_success(),
        );
        episode.independent_verifier_id = None;
        assert!(summarize_by_cell(&[episode.clone()]).is_err());
        episode.independent_verifier_id = Some(episode.policy_id.clone());
        assert!(summarize_by_cell(&[episode.clone()]).is_err());
        episode.independent_verifier_id = Some(episode.source_id.clone());
        assert!(summarize_by_cell(&[episode]).is_err());
    }

    #[test]
    fn absent_success_requires_verifier_but_uninstrumented_success_does_not() {
        let mut episode = row(
            "episode-1",
            ExecutionLevel::L1StationaryUse,
            ToolSetMode::Standard,
            [
                EventEvidence::OccurredAtNs(10),
                EventEvidence::InstrumentedAbsent,
                EventEvidence::NotInstrumented,
                EventEvidence::NotInstrumented,
                EventEvidence::InstrumentedAbsent,
            ],
        );
        episode.independent_verifier_id = None;
        assert!(summarize_by_cell(&[episode.clone()]).is_err());

        episode.events[EventKind::TaskSuccess.index()] = EventEvidence::NotInstrumented;
        assert!(summarize_by_cell(&[episode]).is_ok());
    }

    #[test]
    fn empty_provenance_is_rejected() {
        let mut episode = row(
            "episode-1",
            ExecutionLevel::L0SelectionPickup,
            ToolSetMode::Standard,
            ordered_success(),
        );
        episode.clock_domain_id.clear();
        assert!(summarize_by_cell(&[episode]).is_err());
    }
}
