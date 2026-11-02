#![forbid(unsafe_code)]

//! SPECTRA-CARD-001 synthetic design/evaluator.
//!
//! This task-specific simulator separates (A) enclosure context, (B) an
//! unenergized enclosure, and (C) an inside electrical-state active/sham
//! contrast. It is synthetic only. It does not specify or authorize human
//! electrical exposure, does not infer ESP, and does not implement inferential
//! statistics. Participant-level descriptive contrasts are emitted for the
//! existing blocked-randomization analysis owner.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA_V0_1: &str = "symthaea.spectra_card.synthetic_design.v0.1";
pub const GENERATOR_V0_1: &str = "splitmix64-card-blocked-v0.1";
pub const MAX_PARTICIPANTS: u16 = 256;
pub const MIN_PARTICIPANTS: u16 = 8;
pub const MAX_CYCLES: u16 = 64;
pub const MIN_CYCLES: u16 = 4;
pub const CONDITIONS_PER_CYCLE: usize = 4;
pub const MAX_TRIALS: usize = 100_000;
pub const MAX_PERIOD_INDEX: u8 = 3;
pub const DEFAULT_OPAQUE_CAMPAIGN_ID: u64 = 600_000_701;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardConditionV1 {
    OutsideReference,
    InsideUnenergized,
    InsideActive,
    InsideSham,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardWorldV1 {
    Null,
    EnclosureContextOnly,
    ElectricalIncrementOnly,
    BothEffects,
    PracticeOnly,
    OrderConfounded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardConfigV1 {
    /// Opaque caller-supplied identity. Not derived from seed, world or outcomes.
    pub campaign_instance_id: u64,
    pub participant_count: u16,
    pub cycles_per_participant: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCardTrialV1 {
    pub trial_id: String,
    pub participant_id: String,
    pub session_id: String,
    pub cycle_index: u16,
    pub period_index: u8,
    /// Randomized condition code. Its mapping to active/sham is evaluator-only.
    pub blinded_condition_code: u8,
    /// Context that is physically apparent to the participant: outside vs inside.
    pub inside_context_visible: bool,
    /// Participant's red/black response: 0=red, 1=black. Target remains hidden.
    pub response_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCardCampaignV1 {
    pub schema: String,
    pub generator_version: String,
    pub campaign_id: String,
    pub participant_count: u16,
    pub cycles_per_participant: u16,
    pub trials: Vec<PublicCardTrialV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardEvaluatorTruthV1 {
    pub schema: String,
    pub world: CardWorldV1,
    pub seed: u64,
    pub target_indices: Vec<u8>,
    /// Array index is the public blinded_condition_code.
    pub condition_codebook: [CardConditionV1; CONDITIONS_PER_CYCLE],
    /// World probability at period 1, indexed by blinded public condition code.
    pub response_success_probability_permille_at_period_one_by_code: [u16; CONDITIONS_PER_CYCLE],
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedCardCampaignV1 {
    pub public: PublicCardCampaignV1,
    pub evaluator_truth: CardEvaluatorTruthV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardErrorV1 {
    InvalidCampaignIdentity,
    InvalidParticipantCount,
    InvalidCycleCount,
    CampaignTooLarge,
    TruthLengthMismatch,
    DuplicateTrialId,
    InvalidResponse,
    InvalidTarget,
    InvalidConditionCode,
    InvalidConditionCodebook,
    InsideContextMismatch,
    ParticipantBlockIncomplete,
    ParticipantConditionMissing,
    EmptyParticipantSet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionSummaryV1 {
    pub condition: CardConditionV1,
    pub trial_count: u64,
    pub participant_count: u64,
    /// Mean of participant-level accuracies; participants receive equal weight.
    pub mean_accuracy_permille: f64,
    /// Binary signal-detection counts, with black (index 1) as the positive class.
    pub hits: u64,
    pub misses: u64,
    pub false_alarms: u64,
    pub correct_rejections: u64,
    /// Mean period position. A large active/sham difference can reveal order confounding.
    pub mean_period_index: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantContrastV1 {
    pub participant_id: String,
    /// InsideUnenergized minus OutsideReference.
    pub enclosure_context_effect_permille: f64,
    /// InsideActive minus InsideSham.
    pub electrical_increment_effect_permille: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesignDiagnosticV1 {
    Counterbalanced,
    ConditionOrderConfounded,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardReportV1 {
    pub schema: String,
    pub campaign_id: String,
    pub participant_count: u64,
    pub trial_count: u64,
    /// Binary signal-detection positive class: black card / response index 1.
    pub signal_positive_index: u8,
    pub condition_summaries: Vec<ConditionSummaryV1>,
    pub participant_contrasts: Vec<ParticipantContrastV1>,
    /// Equal-weight mean of participant contrasts; descriptive only.
    pub mean_enclosure_context_effect_permille: f64,
    /// Equal-weight mean of participant contrasts; descriptive only.
    pub mean_electrical_increment_effect_permille: f64,
    /// Positive means Active was systematically scheduled later than Sham.
    pub active_minus_sham_mean_period_index: f64,
    pub design_diagnostic: DesignDiagnosticV1,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bounded(&mut self, upper: u64) -> u64 {
        assert!(upper > 0);
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold { return value % upper; }
        }
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.bounded((i + 1) as u64) as usize;
            items.swap(i, j);
        }
    }
}

const CONDITIONS: [CardConditionV1; CONDITIONS_PER_CYCLE] = [
    CardConditionV1::OutsideReference,
    CardConditionV1::InsideUnenergized,
    CardConditionV1::InsideActive,
    CardConditionV1::InsideSham,
];

fn condition_index(condition: CardConditionV1) -> usize {
    match condition {
        CardConditionV1::OutsideReference => 0,
        CardConditionV1::InsideUnenergized => 1,
        CardConditionV1::InsideActive => 2,
        CardConditionV1::InsideSham => 3,
    }
}

fn validate_config(config: CardConfigV1) -> Result<(), CardErrorV1> {
    if config.campaign_instance_id == 0 { return Err(CardErrorV1::InvalidCampaignIdentity); }
    if !(MIN_PARTICIPANTS..=MAX_PARTICIPANTS).contains(&config.participant_count) {
        return Err(CardErrorV1::InvalidParticipantCount);
    }
    if !(MIN_CYCLES..=MAX_CYCLES).contains(&config.cycles_per_participant) {
        return Err(CardErrorV1::InvalidCycleCount);
    }
    let total = config.participant_count as usize
        * config.cycles_per_participant as usize
        * CONDITIONS_PER_CYCLE;
    if total > MAX_TRIALS { return Err(CardErrorV1::CampaignTooLarge); }
    Ok(())
}

fn probability_for(
    world: CardWorldV1,
    condition: CardConditionV1,
    period_index: u8,
) -> u16 {
    match world {
        CardWorldV1::Null => 500,
        CardWorldV1::EnclosureContextOnly => match condition {
            CardConditionV1::OutsideReference => 500,
            CardConditionV1::InsideUnenergized
            | CardConditionV1::InsideActive
            | CardConditionV1::InsideSham => 680,
        },
        CardWorldV1::ElectricalIncrementOnly => match condition {
            CardConditionV1::InsideActive => 680,
            _ => 500,
        },
        CardWorldV1::BothEffects => match condition {
            CardConditionV1::OutsideReference => 500,
            CardConditionV1::InsideUnenergized | CardConditionV1::InsideSham => 620,
            CardConditionV1::InsideActive => 740,
        },
        // Condition-independent practice curve; random order should prevent
        // systematic condition attribution in a counterbalanced design.
        CardWorldV1::PracticeOnly => 500 + period_index as u16 * 100,
        // Deliberately fixed order below makes practice masquerade as condition.
        CardWorldV1::OrderConfounded => 500 + period_index as u16 * 100,
    }
}

fn validate_codebook(codebook: &[CardConditionV1; CONDITIONS_PER_CYCLE]) -> bool {
    let set: BTreeSet<usize> = codebook.iter().map(|c| condition_index(*c)).collect();
    set.len() == CONDITIONS_PER_CYCLE
}

fn accuracy(correct: u64, total: u64) -> f64 {
    if total == 0 { 0.0 } else { correct as f64 * 1_000.0 / total as f64 }
}

/// Generate a safe synthetic campaign only. "Active" and "sham" are abstract
/// evaluator conditions; this generator does not specify a real field source,
/// exposure, physical device or human electrical intervention.
pub fn generate_card_campaign_v1(
    config: CardConfigV1,
    seed: u64,
    world: CardWorldV1,
) -> Result<GeneratedCardCampaignV1, CardErrorV1> {
    validate_config(config)?;
    let mut rng = SplitMix64::new(seed);

    let mut codebook = CONDITIONS;
    rng.shuffle(&mut codebook);
    let mut trials = Vec::with_capacity(
        config.participant_count as usize
            * config.cycles_per_participant as usize
            * CONDITIONS_PER_CYCLE
    );
    let mut targets = Vec::with_capacity(trials.capacity());

    for participant in 0..config.participant_count {
        for cycle in 0..config.cycles_per_participant {
            let mut schedule = CONDITIONS;
            if world == CardWorldV1::OrderConfounded {
                schedule = [
                    CardConditionV1::OutsideReference,
                    CardConditionV1::InsideUnenergized,
                    CardConditionV1::InsideSham,
                    CardConditionV1::InsideActive,
                ];
            } else {
                rng.shuffle(&mut schedule);
            }

            // Exactly two red and two black targets per participant × cycle;
            // shuffle before assignment so target count cannot manufacture a
            // condition effect within a block.
            let mut cycle_targets = [0_u8, 0, 1, 1];
            rng.shuffle(&mut cycle_targets);

            for (period, condition) in schedule.iter().enumerate() {
                let target = cycle_targets[period];
                let probability = probability_for(world, *condition, period as u8);
                let correct = rng.bounded(1_000) < probability as u64;
                let response = if correct { target } else { 1 - target };
                let code = codebook.iter().position(|c| c == condition)
                    .ok_or(CardErrorV1::InvalidConditionCode)? as u8;
                let inside = !matches!(condition, CardConditionV1::OutsideReference);
                let global_index = trials.len() as u64;
                trials.push(PublicCardTrialV1 {
                    trial_id: format!("card-trial-{global_index:08}"),
                    participant_id: format!("participant-{participant:04}"),
                    session_id: format!("participant-{participant:04}-cycle-{cycle:03}"),
                    cycle_index: cycle,
                    period_index: period as u8,
                    blinded_condition_code: code,
                    inside_context_visible: inside,
                    response_index: response,
                });
                targets.push(target);
            }
        }
    }

    let probabilities = codebook.map(|condition| {
        // Fixture reference at period 1; world-specific expected condition
        // rates are evaluator metadata, not public campaign information.
        probability_for(world, condition, 1)
    });

    Ok(GeneratedCardCampaignV1 {
        public: PublicCardCampaignV1 {
            schema: SCHEMA_V0_1.to_owned(),
            generator_version: GENERATOR_V0_1.to_owned(),
            campaign_id: format!("spectra-card-{0:016x}", config.campaign_instance_id),
            participant_count: config.participant_count,
            cycles_per_participant: config.cycles_per_participant,
            trials,
        },
        evaluator_truth: CardEvaluatorTruthV1 {
            schema: SCHEMA_V0_1.to_owned(),
            world,
            seed,
            target_indices: targets,
            condition_codebook: codebook,
            response_success_probability_permille_at_period_one_by_code: probabilities,
            claim_ceiling: "synthetic_only".to_owned(),
        },
    })
}

/// Evaluate a synthetic campaign. This function emits descriptive paired
/// participant contrasts only; it does not calculate p-values, confidence
/// intervals, or confirmatory inference.
pub fn evaluate_card_campaign_v1(
    public: &PublicCardCampaignV1,
    truth: &CardEvaluatorTruthV1,
) -> Result<CardReportV1, CardErrorV1> {
    if public.trials.len() != truth.target_indices.len() {
        return Err(CardErrorV1::TruthLengthMismatch);
    }
    let expected_trial_count = public.participant_count as usize
        * public.cycles_per_participant as usize
        * CONDITIONS_PER_CYCLE;
    if public.participant_count < MIN_PARTICIPANTS
        || public.participant_count > MAX_PARTICIPANTS
        || public.cycles_per_participant < MIN_CYCLES
        || public.cycles_per_participant > MAX_CYCLES
        || public.trials.len() != expected_trial_count
        || expected_trial_count > MAX_TRIALS
    {
        return Err(CardErrorV1::ParticipantBlockIncomplete);
    }
    if !validate_codebook(&truth.condition_codebook) {
        return Err(CardErrorV1::InvalidConditionCodebook);
    }
    if public.trials.is_empty() {
        return Err(CardErrorV1::EmptyParticipantSet);
    }

    let mut trial_ids = BTreeSet::new();
    // participant -> condition -> (correct, n, sum of period positions)
    let mut per_participant = BTreeMap::<String, [ (u64, u64, u64); 4]>::new();
    // Tuple: correct, total, period sum, hits, misses, false alarms, correct rejections.
    let mut condition_totals = [(0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64); 4];
    let mut active_period_sum = 0_u64;
    let mut sham_period_sum = 0_u64;
    let mut active_period_count = 0_u64;
    let mut sham_period_count = 0_u64;
    let mut participant_cycles = BTreeMap::<
        String,
        BTreeMap<u16, ([u8; CONDITIONS_PER_CYCLE], BTreeSet<u8>)>,
    >::new();

    for (index, row) in public.trials.iter().enumerate() {
        if !trial_ids.insert(row.trial_id.as_str()) { return Err(CardErrorV1::DuplicateTrialId); }
        if row.response_index > 1 { return Err(CardErrorV1::InvalidResponse); }
        let target = truth.target_indices[index];
        if target > 1 { return Err(CardErrorV1::InvalidTarget); }
        if row.blinded_condition_code as usize >= CONDITIONS_PER_CYCLE {
            return Err(CardErrorV1::InvalidConditionCode);
        }
        if row.period_index > MAX_PERIOD_INDEX { return Err(CardErrorV1::ParticipantBlockIncomplete); }
        let condition = truth.condition_codebook[row.blinded_condition_code as usize];
        let ci = condition_index(condition);
        let expected_inside = !matches!(condition, CardConditionV1::OutsideReference);
        if row.inside_context_visible != expected_inside {
            return Err(CardErrorV1::InsideContextMismatch);
        }
        if row.cycle_index >= public.cycles_per_participant
            || row.session_id != format!("{}-cycle-{:03}", row.participant_id, row.cycle_index)
        {
            return Err(CardErrorV1::ParticipantBlockIncomplete);
        }
        let correct = u64::from(row.response_index == target);
        let entry = per_participant.entry(row.participant_id.clone())
            .or_insert([(0, 0, 0); 4]);
        entry[ci].0 += correct;
        entry[ci].1 += 1;
        entry[ci].2 += row.period_index as u64;
        condition_totals[ci].0 += correct;
        condition_totals[ci].1 += 1;
        condition_totals[ci].2 += row.period_index as u64;
        match (target, row.response_index) {
            (1, 1) => condition_totals[ci].3 += 1, // hit
            (1, 0) => condition_totals[ci].4 += 1, // miss
            (0, 1) => condition_totals[ci].5 += 1, // false alarm
            (0, 0) => condition_totals[ci].6 += 1, // correct rejection
            _ => return Err(CardErrorV1::InvalidResponse),
        }
        let (condition_counts, periods) = participant_cycles
            .entry(row.participant_id.clone())
            .or_default()
            .entry(row.cycle_index)
            .or_insert(([0; CONDITIONS_PER_CYCLE], BTreeSet::new()));
        condition_counts[ci] = condition_counts[ci].saturating_add(1);
        periods.insert(row.period_index);

        if condition == CardConditionV1::InsideActive {
            active_period_sum += row.period_index as u64;
            active_period_count += 1;
        } else if condition == CardConditionV1::InsideSham {
            sham_period_sum += row.period_index as u64;
            sham_period_count += 1;
        }
    }

    if participant_cycles.len() != public.participant_count as usize {
        return Err(CardErrorV1::ParticipantBlockIncomplete);
    }
    for cycles in participant_cycles.values() {
        if cycles.len() != public.cycles_per_participant as usize {
            return Err(CardErrorV1::ParticipantBlockIncomplete);
        }
        for (condition_counts, periods) in cycles.values() {
            if *condition_counts != [1; CONDITIONS_PER_CYCLE]
                || periods.len() != CONDITIONS_PER_CYCLE
            {
                return Err(CardErrorV1::ParticipantBlockIncomplete);
            }
        }
    }

    if per_participant.is_empty() { return Err(CardErrorV1::EmptyParticipantSet); }

    let mut participant_contrasts = Vec::with_capacity(per_participant.len());
    let mut context_sum = 0.0;
    let mut electrical_sum = 0.0;
    let mut context_n = 0_u64;
    let mut electrical_n = 0_u64;

    for (participant_id, stats) in &per_participant {
        for ci in 0..CONDITIONS_PER_CYCLE {
            if stats[ci].1 == 0 { return Err(CardErrorV1::ParticipantConditionMissing); }
        }
        let outside = accuracy(stats[0].0, stats[0].1);
        let inside_unenergized = accuracy(stats[1].0, stats[1].1);
        let active = accuracy(stats[2].0, stats[2].1);
        let sham = accuracy(stats[3].0, stats[3].1);
        let context_effect = inside_unenergized - outside;
        let electrical_effect = active - sham;
        context_sum += context_effect;
        electrical_sum += electrical_effect;
        context_n += 1;
        electrical_n += 1;
        participant_contrasts.push(ParticipantContrastV1 {
            participant_id: participant_id.clone(),
            enclosure_context_effect_permille: context_effect,
            electrical_increment_effect_permille: electrical_effect,
        });
    }

    let condition_summaries = CONDITIONS.iter().enumerate().map(|(ci, condition)| {
        let participants: Vec<&(u64, u64, u64)> = per_participant.values()
            .map(|stats| &stats[ci])
            .collect();
        let mean_accuracy = participants.iter()
            .map(|stats| accuracy(stats.0, stats.1))
            .sum::<f64>() / participants.len() as f64;
        let mean_period = condition_totals[ci].2 as f64 / condition_totals[ci].1 as f64;
        ConditionSummaryV1 {
            condition: *condition,
            trial_count: condition_totals[ci].1,
            participant_count: participants.len() as u64,
            mean_accuracy_permille: mean_accuracy,
            hits: condition_totals[ci].3,
            misses: condition_totals[ci].4,
            false_alarms: condition_totals[ci].5,
            correct_rejections: condition_totals[ci].6,
            mean_period_index: mean_period,
        }
    }).collect::<Vec<_>>();

    let active_mean_period = if active_period_count == 0 { 0.0 } else {
        active_period_sum as f64 / active_period_count as f64
    };
    let sham_mean_period = if sham_period_count == 0 { 0.0 } else {
        sham_period_sum as f64 / sham_period_count as f64
    };
    let order_difference = active_mean_period - sham_mean_period;
    let diagnostic = if order_difference.abs() > 0.5 {
        DesignDiagnosticV1::ConditionOrderConfounded
    } else {
        DesignDiagnosticV1::Counterbalanced
    };

    Ok(CardReportV1 {
        schema: "symthaea.spectra_card.report.v0.1".to_owned(),
        campaign_id: public.campaign_id.clone(),
        participant_count: per_participant.len() as u64,
        trial_count: public.trials.len() as u64,
        signal_positive_index: 1,
        condition_summaries,
        participant_contrasts,
        mean_enclosure_context_effect_permille: context_sum / context_n as f64,
        mean_electrical_increment_effect_permille: electrical_sum / electrical_n as f64,
        active_minus_sham_mean_period_index: order_difference,
        design_diagnostic: diagnostic,
        claim_ceiling: "synthetic_descriptive_only".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CardConfigV1 {
        CardConfigV1 {
            campaign_instance_id: 600_000_701,
            participant_count: 32,
            cycles_per_participant: 8,
        }
    }

    fn report(world: CardWorldV1, seed: u64) -> CardReportV1 {
        let campaign = generate_card_campaign_v1(config(), seed, world).unwrap();
        evaluate_card_campaign_v1(&campaign.public, &campaign.evaluator_truth).unwrap()
    }

    #[test]
    fn evaluator_rejects_duplicate_condition_assignment_and_context_mismatch() {
        let campaign = generate_card_campaign_v1(config(), 987, CardWorldV1::Null).unwrap();

        let mut duplicate = campaign.public.clone();
        duplicate.trials[1].blinded_condition_code = duplicate.trials[0].blinded_condition_code;
        assert_eq!(
            evaluate_card_campaign_v1(&duplicate, &campaign.evaluator_truth),
            Err(CardErrorV1::ParticipantBlockIncomplete)
        );

        let mut context_mismatch = campaign.public.clone();
        context_mismatch.trials[0].inside_context_visible =
            !context_mismatch.trials[0].inside_context_visible;
        assert_eq!(
            evaluate_card_campaign_v1(&context_mismatch, &campaign.evaluator_truth),
            Err(CardErrorV1::InsideContextMismatch)
        );
    }

    #[test]
    fn public_campaign_has_no_target_or_world_truth() {
        let campaign = generate_card_campaign_v1(config(), 123, CardWorldV1::Null).unwrap();
        let json = serde_json::to_string(&campaign.public).unwrap();
        assert!(!json.contains("target_indices"));
        assert!(!json.contains("response_correct"));
        assert!(!json.contains("\"world\""));
        assert!(!json.contains("\"seed\""));
    }

    #[test]
    fn every_participant_cycle_has_all_four_conditions_and_balanced_targets() {
        let campaign = generate_card_campaign_v1(config(), 456, CardWorldV1::Null).unwrap();
        for p in 0..config().participant_count {
            for cycle in 0..config().cycles_per_participant {
                let indexes: Vec<usize> = campaign.public.trials.iter().enumerate()
                    .filter(|(_, row)| row.participant_id == format!("participant-{p:04}")
                        && row.cycle_index == cycle)
                    .map(|(index, _)| index).collect();
                assert_eq!(indexes.len(), 4);
                let mut codes = BTreeSet::new();
                let mut targets = [0_u8; 2];
                for index in indexes {
                    codes.insert(campaign.public.trials[index].blinded_condition_code);
                    targets[campaign.evaluator_truth.target_indices[index] as usize] += 1;
                }
                assert_eq!(codes.len(), 4);
                assert_eq!(targets, [2, 2]);
            }
        }
    }

    #[test]
    fn enclosure_only_world_separates_context_from_electrical_increment() {
        let r = report(CardWorldV1::EnclosureContextOnly, 0x1001);
        assert!(r.mean_enclosure_context_effect_permille > 0.0);
        assert!(r.mean_electrical_increment_effect_permille.abs() < 100.0);
        assert_eq!(r.design_diagnostic, DesignDiagnosticV1::Counterbalanced);
    }

    #[test]
    fn electrical_only_world_separates_active_increment_from_context() {
        let r = report(CardWorldV1::ElectricalIncrementOnly, 0x1002);
        assert!(r.mean_electrical_increment_effect_permille > 0.0);
        assert!(r.mean_enclosure_context_effect_permille.abs() < 100.0);
        assert_eq!(r.design_diagnostic, DesignDiagnosticV1::Counterbalanced);
    }

    #[test]
    fn both_effects_world_has_two_distinct_positive_contrasts() {
        let r = report(CardWorldV1::BothEffects, 0x1003);
        assert!(r.mean_enclosure_context_effect_permille > 0.0);
        assert!(r.mean_electrical_increment_effect_permille > 0.0);
    }

    #[test]
    fn practice_only_world_is_not_systematically_attributed_to_active_condition() {
        let r = report(CardWorldV1::PracticeOnly, 0x1004);
        assert!(r.active_minus_sham_mean_period_index.abs() < 0.5);
        assert_eq!(r.design_diagnostic, DesignDiagnosticV1::Counterbalanced);
        assert!(r.mean_electrical_increment_effect_permille.abs() < 120.0);
    }

    #[test]
    fn fixed_order_practice_world_is_flagged_as_condition_order_confounded() {
        let r = report(CardWorldV1::OrderConfounded, 0x1005);
        assert_eq!(r.design_diagnostic, DesignDiagnosticV1::ConditionOrderConfounded);
        assert!(r.active_minus_sham_mean_period_index > 0.5);
    }


    #[test]
    fn six_world_known_answers_replay_exactly() {
        let fixture: serde_json::Value = serde_json::from_str(
            include_str!("../../../research/spectra-card-001/fixtures/known-answer-v0.1.json")
        ).expect("card known-answer fixture");
        for scenario in fixture["scenarios"].as_array().unwrap() {
            let world = match scenario["world"].as_str().unwrap() {
                "null" => CardWorldV1::Null,
                "enclosure_context_only" => CardWorldV1::EnclosureContextOnly,
                "electrical_increment_only" => CardWorldV1::ElectricalIncrementOnly,
                "both_effects" => CardWorldV1::BothEffects,
                "practice_only" => CardWorldV1::PracticeOnly,
                "order_confounded" => CardWorldV1::OrderConfounded,
                other => panic!("unknown card fixture world: {other}"),
            };
            let seed = u64::from_str_radix(
                scenario["seed_hex"].as_str().unwrap().trim_start_matches("0x"), 16
            ).unwrap();
            let campaign = generate_card_campaign_v1(config(), seed, world).unwrap();
            let report = evaluate_card_campaign_v1(&campaign.public, &campaign.evaluator_truth).unwrap();
            let expected = &scenario["expected"];
            assert_eq!(report.trial_count, expected["trial_count"].as_u64().unwrap());
            assert_eq!(report.participant_count, expected["participant_count"].as_u64().unwrap());
            assert!((report.mean_enclosure_context_effect_permille
                - expected["enclosure_context_effect_permille"].as_f64().unwrap()).abs() < 1.0e-9);
            assert!((report.mean_electrical_increment_effect_permille
                - expected["electrical_increment_effect_permille"].as_f64().unwrap()).abs() < 1.0e-9);
            assert!((report.active_minus_sham_mean_period_index
                - expected["active_minus_sham_mean_period_index"].as_f64().unwrap()).abs() < 1.0e-9);
            let expected_diagnostic = match expected["design_diagnostic"].as_str().unwrap() {
                "counterbalanced" => DesignDiagnosticV1::Counterbalanced,
                "condition_order_confounded" => DesignDiagnosticV1::ConditionOrderConfounded,
                other => panic!("unknown card diagnostic: {other}"),
            };
            assert_eq!(report.design_diagnostic, expected_diagnostic);
            for (index, condition) in report.condition_summaries.iter().enumerate() {
                assert!((condition.mean_accuracy_permille
                    - expected["condition_mean_accuracy_permille"][index].as_f64().unwrap()).abs() < 1.0e-9);
                for (field, actual) in [
                    ("hits", condition.hits),
                    ("misses", condition.misses),
                    ("false_alarms", condition.false_alarms),
                    ("correct_rejections", condition.correct_rejections),
                ] {
                    assert_eq!(actual, expected["condition_signal_detection_counts"][index][field].as_u64().unwrap());
                }
            }
        }
    }

    #[test]
    fn invalid_campaign_parameters_fail_closed() {
        let mut c = config();
        c.campaign_instance_id = 0;
        assert_eq!(
            generate_card_campaign_v1(c, 1, CardWorldV1::Null),
            Err(CardErrorV1::InvalidCampaignIdentity)
        );
        c = config();
        c.participant_count = 2;
        assert_eq!(
            generate_card_campaign_v1(c, 1, CardWorldV1::Null),
            Err(CardErrorV1::InvalidParticipantCount)
        );
    }
}
