//! Verifier-grounded direction-board allocation for open-ended architecture search.
//!
//! A direction is a canonical research hypothesis defined by its target capability,
//! solution object, construction principle, and scope. A method is a concrete candidate
//! delta within one direction. The board allocates a fixed evaluation budget across
//! directions and an explicit NEW-direction arm.
//!
//! The board is advisory. It does not execute candidates, authenticate evaluator receipts,
//! inspect hidden labels, or grant repository/production authority. Callers must supply
//! records produced by an independently controlled verifier.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BOARD_SCHEMA: &str = "symthaea.architecture-engineering.direction-board";
const BOARD_VERSION: &str = "0.1.0";
const DIRECTION_DOMAIN: &[u8] = b"symthaea.direction.v0.1\0";
const BOARD_DOMAIN: &[u8] = b"symthaea.direction-board.v0.1\0";
const PLAN_DOMAIN: &[u8] = b"symthaea.direction-allocation-plan.v0.1\0";

/// Canonical identity of a research direction.
///
/// These fields separate the capability being pursued, the object being changed,
/// and the construction/search principle. A source refactor alone cannot create
/// a distinct direction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionDescriptor {
    pub target_capability: String,
    pub solution_object: String,
    pub construction_principle: String,
    pub scope: String,
}

impl DirectionDescriptor {
    /// Normalize labels before hashing so case and whitespace aliases collapse.
    pub fn canonicalized(&self) -> Result<Self, DirectionBoardError> {
        Ok(Self {
            target_capability: normalize_label("target_capability", &self.target_capability)?,
            solution_object: normalize_label("solution_object", &self.solution_object)?,
            construction_principle: normalize_label(
                "construction_principle",
                &self.construction_principle,
            )?,
            scope: normalize_label("scope", &self.scope)?,
        })
    }

    /// Stable, domain-separated digest of the canonical direction identity.
    pub fn digest(&self) -> Result<String, DirectionBoardError> {
        let canonical = self.canonicalized()?;
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| DirectionBoardError::new("direction", error.to_string()))?;
        Ok(domain_digest(DIRECTION_DOMAIN, &bytes))
    }
}

/// Fixed comparison profile for every score on one board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionBoardConfig {
    pub comparison_profile_sha256: String,
    pub metric_id: String,
    /// UCB exploration coefficient. Set before any direction outcomes are observed.
    pub exploration_coefficient: f64,
    /// Prior reward per cost unit for the NEW-direction arm before its first proposal.
    pub new_direction_prior_yield: f64,
    /// Verifier-admitted novel directions receive this normalized credit per admission.
    /// Keep this on the same declared utility scale as local-record score improvements.
    pub new_direction_admission_credit: f64,
    /// Each arm receives this many initial evaluations before pure proportional allocation,
    /// whenever the requested budget is large enough to cover all arms.
    pub minimum_initial_evaluations: u64,
}

impl DirectionBoardConfig {
    fn validate(&self) -> Result<(), DirectionBoardError> {
        validate_sha256("comparison_profile_sha256", &self.comparison_profile_sha256)?;
        if self.metric_id.trim().is_empty() {
            return Err(DirectionBoardError::new("metric_id", "must be non-empty"));
        }
        if !self.exploration_coefficient.is_finite() || self.exploration_coefficient < 0.0 {
            return Err(DirectionBoardError::new(
                "exploration_coefficient",
                "must be finite and non-negative",
            ));
        }
        if !self.new_direction_prior_yield.is_finite() || self.new_direction_prior_yield < 0.0 {
            return Err(DirectionBoardError::new(
                "new_direction_prior_yield",
                "must be finite and non-negative",
            ));
        }
        if !self.new_direction_admission_credit.is_finite()
            || !(0.0..=1.0).contains(&self.new_direction_admission_credit)
        {
            return Err(DirectionBoardError::new(
                "new_direction_admission_credit",
                "must be finite and within [0,1]",
            ));
        }
        Ok(())
    }
}

/// The verifier disposition for one candidate evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    Verified,
    Rejected,
    Inconclusive,
}

/// One independently evaluated method within a direction.
///
/// This record is accepted only after the external verifier has produced a receipt.
/// The board checks digest shape and profile consistency, but does not authenticate
/// the receipt or recompute the candidate/evaluator artifacts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionAttempt {
    pub attempt_id: String,
    pub direction_sha256: String,
    pub method_id: String,
    pub parent_candidate_sha256: String,
    pub candidate_sha256: String,
    pub semantic_delta_sha256: String,
    pub comparison_profile_sha256: String,
    pub metric_id: String,
    pub evaluation_cost_units: u64,
    pub disposition: AttemptDisposition,
    /// Normalized score in [0,1], present only for Verified attempts.
    pub score: Option<f64>,
    pub verifier_receipt_sha256: String,
}

/// Outcome of asking the verifier to admit a genuinely new direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NewDirectionDisposition {
    Admitted,
    Duplicate,
    Rejected,
    Inconclusive,
}

/// Paid attempt to open a new direction.
///
/// Admitted/duplicate status is checked against the board's canonical direction IDs.
/// Rejected and inconclusive proposals remain in the cost/attempt denominator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewDirectionProposal {
    pub proposal_id: String,
    pub descriptor: DirectionDescriptor,
    pub proposal_sha256: String,
    pub comparison_profile_sha256: String,
    pub metric_id: String,
    pub evaluation_cost_units: u64,
    pub disposition: NewDirectionDisposition,
    pub verifier_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionStats {
    pub direction_sha256: String,
    pub descriptor: DirectionDescriptor,
    pub verifier_receipt_sha256: String,
    pub attempted_evaluations: u64,
    pub verified_evaluations: u64,
    pub rejected_evaluations: u64,
    pub inconclusive_evaluations: u64,
    pub total_cost_units: u64,
    pub total_record_improvement: f64,
    pub local_best_score: Option<f64>,
    /// Cumulative local-record improvement divided by *all* attempted evaluation cost.
    pub improvement_yield_per_cost: f64,
}

impl DirectionStats {
    fn new(
        direction_sha256: String,
        descriptor: DirectionDescriptor,
        verifier_receipt_sha256: String,
    ) -> Self {
        Self {
            direction_sha256,
            descriptor,
            verifier_receipt_sha256,
            attempted_evaluations: 0,
            verified_evaluations: 0,
            rejected_evaluations: 0,
            inconclusive_evaluations: 0,
            total_cost_units: 0,
            total_record_improvement: 0.0,
            local_best_score: None,
            improvement_yield_per_cost: 0.0,
        }
    }

    fn record_attempt(&mut self, attempt: &DirectionAttempt) -> Result<(), DirectionBoardError> {
        self.attempted_evaluations = checked_add(
            "attempted_evaluations",
            self.attempted_evaluations,
            1,
        )?;
        self.total_cost_units = checked_add(
            "total_cost_units",
            self.total_cost_units,
            attempt.evaluation_cost_units,
        )?;

        match attempt.disposition {
            AttemptDisposition::Verified => {
                self.verified_evaluations = checked_add(
                    "verified_evaluations",
                    self.verified_evaluations,
                    1,
                )?;
                let score = attempt.score.ok_or_else(|| {
                    DirectionBoardError::new("score", "verified attempts require a score")
                })?;
                let previous = self.local_best_score.unwrap_or(0.0);
                if self.local_best_score.is_none() || score > previous {
                    self.total_record_improvement += score - previous;
                    self.local_best_score = Some(score);
                }
            }
            AttemptDisposition::Rejected => {
                self.rejected_evaluations = checked_add(
                    "rejected_evaluations",
                    self.rejected_evaluations,
                    1,
                )?;
            }
            AttemptDisposition::Inconclusive => {
                self.inconclusive_evaluations = checked_add(
                    "inconclusive_evaluations",
                    self.inconclusive_evaluations,
                    1,
                )?;
            }
        }
        self.improvement_yield_per_cost = if self.total_cost_units == 0 {
            0.0
        } else {
            self.total_record_improvement / self.total_cost_units as f64
        };
        if !self.improvement_yield_per_cost.is_finite() {
            return Err(DirectionBoardError::new(
                "improvement_yield_per_cost",
                "non-finite result",
            ));
        }
        Ok(())
    }

    fn mean_yield(&self) -> f64 {
        self.improvement_yield_per_cost
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewDirectionStats {
    pub attempted_proposals: u64,
    pub admitted_directions: u64,
    pub duplicate_proposals: u64,
    pub rejected_proposals: u64,
    pub inconclusive_proposals: u64,
    pub total_cost_units: u64,
    pub total_admission_credit: f64,
    pub admissions_per_cost: f64,
}

impl Default for NewDirectionStats {
    fn default() -> Self {
        Self {
            attempted_proposals: 0,
            admitted_directions: 0,
            duplicate_proposals: 0,
            rejected_proposals: 0,
            inconclusive_proposals: 0,
            total_cost_units: 0,
            total_admission_credit: 0.0,
            admissions_per_cost: 0.0,
        }
    }
}

/// The arm receiving the advisory evaluation allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "direction_sha256", rename_all = "snake_case")]
pub enum AllocationArm {
    Direction(String),
    NewDirection,
}

/// One arm's requested share of the next finite evaluation budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmAllocation {
    pub arm: AllocationArm,
    pub recommended_evaluations: usize,
    pub prior_attempts: u64,
    pub estimated_yield_per_cost: f64,
    pub exploration_bonus: f64,
    pub potential: f64,
    pub protected_initial_allocation: bool,
}

/// Deterministic, advisory plan. The caller must record actual work separately;
/// a recommendation is not evidence that the suggested work was executed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionAllocationPlan {
    pub schema: String,
    pub version: String,
    pub comparison_profile_sha256: String,
    pub metric_id: String,
    pub source_board_sha256: String,
    pub evaluation_budget: usize,
    pub allocations: Vec<ArmAllocation>,
    pub plan_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BoardDigestPayload<'a> {
    schema: &'static str,
    version: &'static str,
    config: &'a DirectionBoardConfig,
    directions: &'a BTreeMap<String, DirectionStats>,
    new_direction: &'a NewDirectionStats,
    attempt_ids: &'a BTreeSet<String>,
    proposal_ids: &'a BTreeSet<String>,
    attempts: &'a BTreeMap<String, DirectionAttempt>,
    proposals: &'a BTreeMap<String, NewDirectionProposal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionBoard {
    pub schema: String,
    pub version: String,
    pub config: DirectionBoardConfig,
    directions: BTreeMap<String, DirectionStats>,
    new_direction: NewDirectionStats,
    attempt_ids: BTreeSet<String>,
    proposal_ids: BTreeSet<String>,
    attempts: BTreeMap<String, DirectionAttempt>,
    proposals: BTreeMap<String, NewDirectionProposal>,
}

impl DirectionBoard {
    pub fn new(config: DirectionBoardConfig) -> Result<Self, DirectionBoardError> {
        config.validate()?;
        Ok(Self {
            schema: BOARD_SCHEMA.to_owned(),
            version: BOARD_VERSION.to_owned(),
            config,
            directions: BTreeMap::new(),
            new_direction: NewDirectionStats::default(),
            attempt_ids: BTreeSet::new(),
            proposal_ids: BTreeSet::new(),
            attempts: BTreeMap::new(),
            proposals: BTreeMap::new(),
        })
    }

    /// Register a direction only after the independent verifier has canonicalized it.
    pub fn register_verified_direction(
        &mut self,
        descriptor: DirectionDescriptor,
        verifier_receipt_sha256: &str,
    ) -> Result<String, DirectionBoardError> {
        validate_sha256("verifier_receipt_sha256", verifier_receipt_sha256)?;
        let canonical = descriptor.canonicalized()?;
        let digest = canonical.digest()?;
        if self.directions.contains_key(&digest) {
            return Err(DirectionBoardError::new(
                "direction",
                format!("duplicate canonical direction {digest}"),
            ));
        }
        self.directions.insert(
            digest.clone(),
            DirectionStats::new(
                digest.clone(),
                canonical,
                verifier_receipt_sha256.to_owned(),
            ),
        );
        Ok(digest)
    }

    /// Record a paid proposal to create a new direction.
    pub fn record_new_direction_proposal(
        &mut self,
        proposal: NewDirectionProposal,
    ) -> Result<Option<String>, DirectionBoardError> {
        self.validate_proposal(&proposal)?;
        if self.proposal_ids.contains(&proposal.proposal_id) {
            return Err(DirectionBoardError::new(
                "proposal_id",
                format!("duplicate proposal ID {:?}", proposal.proposal_id),
            ));
        }

        let canonical = proposal.descriptor.canonicalized()?;
        let direction_sha256 = canonical.digest()?;
        let known = self.directions.contains_key(&direction_sha256);

        match proposal.disposition {
            NewDirectionDisposition::Admitted if known => {
                return Err(DirectionBoardError::new(
                    "disposition",
                    "verifier marked a known direction as newly admitted",
                ));
            }
            NewDirectionDisposition::Duplicate if !known => {
                return Err(DirectionBoardError::new(
                    "disposition",
                    "verifier marked an unknown direction as a duplicate",
                ));
            }
            _ => {}
        }

        let mut next_stats = self.new_direction.clone();
        next_stats.attempted_proposals = checked_add(
            "new_direction.attempted_proposals",
            next_stats.attempted_proposals,
            1,
        )?;
        next_stats.total_cost_units = checked_add(
            "new_direction.total_cost_units",
            next_stats.total_cost_units,
            proposal.evaluation_cost_units,
        )?;

        let admitted = match proposal.disposition {
            NewDirectionDisposition::Admitted => {
                next_stats.admitted_directions = checked_add(
                    "new_direction.admitted_directions",
                    next_stats.admitted_directions,
                    1,
                )?;
                next_stats.total_admission_credit +=
                    self.config.new_direction_admission_credit;
                true
            }
            NewDirectionDisposition::Duplicate => {
                next_stats.duplicate_proposals = checked_add(
                    "new_direction.duplicate_proposals",
                    next_stats.duplicate_proposals,
                    1,
                )?;
                false
            }
            NewDirectionDisposition::Rejected => {
                next_stats.rejected_proposals = checked_add(
                    "new_direction.rejected_proposals",
                    next_stats.rejected_proposals,
                    1,
                )?;
                false
            }
            NewDirectionDisposition::Inconclusive => {
                next_stats.inconclusive_proposals = checked_add(
                    "new_direction.inconclusive_proposals",
                    next_stats.inconclusive_proposals,
                    1,
                )?;
                false
            }
        };

        next_stats.admissions_per_cost = if next_stats.total_cost_units == 0 {
            0.0
        } else {
            next_stats.total_admission_credit / next_stats.total_cost_units as f64
        };
        if !next_stats.admissions_per_cost.is_finite() {
            return Err(DirectionBoardError::new(
                "new_direction.admissions_per_cost",
                "non-finite result",
            ));
        }

        if admitted {
            self.directions.insert(
                direction_sha256.clone(),
                DirectionStats::new(
                    direction_sha256.clone(),
                    canonical,
                    proposal.verifier_receipt_sha256.clone(),
                ),
            );
        }
        self.proposal_ids.insert(proposal.proposal_id.clone());
        self.proposals.insert(proposal.proposal_id.clone(), proposal);
        self.new_direction = next_stats;

        Ok(if admitted {
            Some(direction_sha256)
        } else {
            None
        })
    }

    pub fn record_attempt(
        &mut self,
        attempt: DirectionAttempt,
    ) -> Result<(), DirectionBoardError> {
        self.validate_attempt(&attempt)?;
        if self.attempt_ids.contains(&attempt.attempt_id) {
            return Err(DirectionBoardError::new(
                "attempt_id",
                format!("duplicate attempt ID {:?}", attempt.attempt_id),
            ));
        }
        let direction = self.directions.get(&attempt.direction_sha256).ok_or_else(|| {
            DirectionBoardError::new(
                "direction_sha256",
                format!("unknown direction {:?}", attempt.direction_sha256),
            )
        })?;
        let mut updated = direction.clone();
        updated.record_attempt(&attempt)?;
        self.directions.insert(attempt.direction_sha256.clone(), updated);
        self.attempt_ids.insert(attempt.attempt_id.clone());
        self.attempts.insert(attempt.attempt_id.clone(), attempt);
        Ok(())
    }

    pub fn direction_stats(&self) -> Vec<DirectionStats> {
        self.directions.values().cloned().collect()
    }

    pub fn new_direction_stats(&self) -> &NewDirectionStats {
        &self.new_direction
    }

    /// Return a deterministic UCB-style recommendation over registered directions + NEW.
    ///
    /// If the budget can cover every untried arm, one initial evaluation is protected for
    /// each arm. Remaining evaluations are assigned by largest-remainder proportional
    /// allocation over local-record yield plus the exploration bonus.
    pub fn recommend(&self, evaluation_budget: usize) -> Result<DirectionAllocationPlan, DirectionBoardError> {
        if evaluation_budget == 0 {
            return Err(DirectionBoardError::new(
                "evaluation_budget",
                "must be greater than zero",
            ));
        }

        let mut arms = Vec::with_capacity(self.directions.len() + 1);
        let total_attempts = self
            .directions
            .values()
            .fold(0u64, |total, stats| {
                total.saturating_add(stats.attempted_evaluations)
            })
            .saturating_add(self.new_direction.attempted_proposals);
        let log_total = ((total_attempts as f64) + 2.0).ln();

        for (direction_sha256, stats) in &self.directions {
            let estimated_yield = stats.mean_yield();
            let exploration_bonus = self.config.exploration_coefficient
                * (log_total / (stats.attempted_evaluations as f64 + 1.0)).sqrt();
            arms.push(ArmState {
                arm: AllocationArm::Direction(direction_sha256.clone()),
                attempts: stats.attempted_evaluations,
                estimated_yield,
                exploration_bonus,
                potential: estimated_yield + exploration_bonus,
            });
        }

        let new_bonus = self.config.exploration_coefficient
            * (log_total / (self.new_direction.attempted_proposals as f64 + 1.0)).sqrt();
        arms.push(ArmState {
            arm: AllocationArm::NewDirection,
            attempts: self.new_direction.attempted_proposals,
            estimated_yield: if self.new_direction.total_cost_units == 0 {
                self.config.new_direction_prior_yield
            } else {
                self.new_direction.admissions_per_cost
            },
            exploration_bonus: new_bonus,
            potential: (if self.new_direction.total_cost_units == 0 {
                self.config.new_direction_prior_yield
            } else {
                self.new_direction.admissions_per_cost
            }) + new_bonus,
        });

        arms.sort_by(|a, b| arm_key(&a.arm).cmp(&arm_key(&b.arm)));

        let mut allocations = vec![0usize; arms.len()];
        let mut remaining = evaluation_budget;

        let protected: Vec<usize> = arms
            .iter()
            .enumerate()
            .filter(|(_, arm)| arm.attempts < self.config.minimum_initial_evaluations)
            .map(|(index, _)| index)
            .collect();

        if evaluation_budget >= protected.len() {
            for index in protected {
                allocations[index] = 1;
                remaining -= 1;
            }
        } else {
            let mut ranked: Vec<usize> = protected;
            ranked.sort_by(|&left, &right| {
                arms[right]
                    .potential
                    .total_cmp(&arms[left].potential)
                    .then_with(|| arm_key(&arms[left].arm).cmp(&arm_key(&arms[right].arm)))
            });
            for index in ranked.into_iter().take(evaluation_budget) {
                allocations[index] = 1;
                remaining -= 1;
            }
        }

        if remaining > 0 {
            allocate_largest_remainder(&arms, &mut allocations, remaining);
        }

        let mut plan_allocations = Vec::with_capacity(arms.len());
        for (index, arm) in arms.into_iter().enumerate() {
            plan_allocations.push(ArmAllocation {
                arm: arm.arm,
                recommended_evaluations: allocations[index],
                prior_attempts: arm.attempts,
                estimated_yield_per_cost: arm.estimated_yield,
                exploration_bonus: arm.exploration_bonus,
                potential: arm.potential,
                protected_initial_allocation: arm.attempts
                    < self.config.minimum_initial_evaluations
                    && allocations[index] > 0,
            });
        }

        let source_board_sha256 = self.digest()?;
        let mut plan = DirectionAllocationPlan {
            schema: BOARD_SCHEMA.to_owned(),
            version: BOARD_VERSION.to_owned(),
            comparison_profile_sha256: self.config.comparison_profile_sha256.clone(),
            metric_id: self.config.metric_id.clone(),
            source_board_sha256,
            evaluation_budget,
            allocations: plan_allocations,
            plan_sha256: String::new(),
        };
        let bytes = serde_json::to_vec(&plan)
            .map_err(|error| DirectionBoardError::new("plan", error.to_string()))?;
        plan.plan_sha256 = domain_digest(PLAN_DOMAIN, &bytes);
        Ok(plan)
    }

    pub fn digest(&self) -> Result<String, DirectionBoardError> {
        let payload = BoardDigestPayload {
            schema: BOARD_SCHEMA,
            version: BOARD_VERSION,
            config: &self.config,
            directions: &self.directions,
            new_direction: &self.new_direction,
            attempt_ids: &self.attempt_ids,
            proposal_ids: &self.proposal_ids,
            attempts: &self.attempts,
            proposals: &self.proposals,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| DirectionBoardError::new("board", error.to_string()))?;
        Ok(domain_digest(BOARD_DOMAIN, &bytes))
    }

    fn validate_attempt(&self, attempt: &DirectionAttempt) -> Result<(), DirectionBoardError> {
        if attempt.attempt_id.trim().is_empty() || attempt.method_id.trim().is_empty() {
            return Err(DirectionBoardError::new(
                "attempt",
                "attempt_id and method_id must be non-empty",
            ));
        }
        validate_sha256("direction_sha256", &attempt.direction_sha256)?;
        validate_sha256("parent_candidate_sha256", &attempt.parent_candidate_sha256)?;
        validate_sha256("candidate_sha256", &attempt.candidate_sha256)?;
        validate_sha256("semantic_delta_sha256", &attempt.semantic_delta_sha256)?;
        validate_sha256("comparison_profile_sha256", &attempt.comparison_profile_sha256)?;
        validate_sha256("verifier_receipt_sha256", &attempt.verifier_receipt_sha256)?;
        self.validate_profile(&attempt.comparison_profile_sha256, &attempt.metric_id)?;
        if attempt.evaluation_cost_units == 0 {
            return Err(DirectionBoardError::new(
                "evaluation_cost_units",
                "must be greater than zero",
            ));
        }
        match attempt.disposition {
            AttemptDisposition::Verified => {
                let score = attempt.score.ok_or_else(|| {
                    DirectionBoardError::new("score", "verified attempt requires a score")
                })?;
                if !score.is_finite() || !(0.0..=1.0).contains(&score) {
                    return Err(DirectionBoardError::new(
                        "score",
                        "verified score must be finite and in [0,1]",
                    ));
                }
            }
            AttemptDisposition::Rejected | AttemptDisposition::Inconclusive => {
                if attempt.score.is_some() {
                    return Err(DirectionBoardError::new(
                        "score",
                        "rejected/inconclusive attempts must not carry a qualification score",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_proposal(&self, proposal: &NewDirectionProposal) -> Result<(), DirectionBoardError> {
        if proposal.proposal_id.trim().is_empty() {
            return Err(DirectionBoardError::new("proposal_id", "must be non-empty"));
        }
        validate_sha256("proposal_sha256", &proposal.proposal_sha256)?;
        validate_sha256(
            "comparison_profile_sha256",
            &proposal.comparison_profile_sha256,
        )?;
        validate_sha256("verifier_receipt_sha256", &proposal.verifier_receipt_sha256)?;
        self.validate_profile(&proposal.comparison_profile_sha256, &proposal.metric_id)?;
        if proposal.evaluation_cost_units == 0 {
            return Err(DirectionBoardError::new(
                "evaluation_cost_units",
                "must be greater than zero",
            ));
        }
        proposal.descriptor.canonicalized()?;
        Ok(())
    }

    fn validate_profile(
        &self,
        comparison_profile_sha256: &str,
        metric_id: &str,
    ) -> Result<(), DirectionBoardError> {
        if comparison_profile_sha256 != self.config.comparison_profile_sha256 {
            return Err(DirectionBoardError::new(
                "comparison_profile_sha256",
                "does not match the board's frozen comparison profile",
            ));
        }
        if metric_id != self.config.metric_id {
            return Err(DirectionBoardError::new(
                "metric_id",
                "does not match the board's frozen metric",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ArmState {
    arm: AllocationArm,
    attempts: u64,
    estimated_yield: f64,
    exploration_bonus: f64,
    potential: f64,
}

fn arm_key(arm: &AllocationArm) -> &str {
    match arm {
        AllocationArm::Direction(digest) => digest,
        AllocationArm::NewDirection => "~NEW",
    }
}

fn allocate_largest_remainder(arms: &[ArmState], allocations: &mut [usize], remaining: usize) {
    if arms.is_empty() || remaining == 0 {
        return;
    }
    let weights: Vec<f64> = arms.iter().map(|arm| arm.potential.max(1e-12)).collect();
    let weight_sum = weights.iter().sum::<f64>();
    let mut fractions = Vec::with_capacity(arms.len());
    let mut assigned = 0usize;

    for (index, weight) in weights.iter().enumerate() {
        let exact = remaining as f64 * weight / weight_sum;
        let whole = exact.floor() as usize;
        allocations[index] += whole;
        assigned += whole;
        fractions.push((index, exact - whole as f64));
    }

    fractions.sort_by(|(index_a, fraction_a), (index_b, fraction_b)| {
        fraction_b
            .total_cmp(fraction_a)
            .then_with(|| arm_key(&arms[*index_a].arm).cmp(arm_key(&arms[*index_b].arm)))
    });
    for (index, _) in fractions.into_iter().take(remaining - assigned) {
        allocations[index] += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectionBoardError {
    pub path: String,
    pub message: String,
}

impl DirectionBoardError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for DirectionBoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for DirectionBoardError {}

fn normalize_label(field: &str, value: &str) -> Result<String, DirectionBoardError> {
    let normalized = value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(DirectionBoardError::new(field, "must be non-empty"));
    }
    Ok(normalized)
}

fn validate_sha256(field: &str, value: &str) -> Result<(), DirectionBoardError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(DirectionBoardError::new(
            field,
            "expected lowercase 64-character SHA-256 hex digest",
        ))
    }
}

fn checked_add(field: &str, left: u64, right: u64) -> Result<u64, DirectionBoardError> {
    left.checked_add(right)
        .ok_or_else(|| DirectionBoardError::new(field, "counter overflow"))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const RECEIPT: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CANDIDATE_A: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CANDIDATE_B: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DELTA: &str =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const PROPOSAL: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn config() -> DirectionBoardConfig {
        DirectionBoardConfig {
            comparison_profile_sha256: PROFILE.to_owned(),
            metric_id: "held_out_task_quality".to_owned(),
            exploration_coefficient: 0.25,
            new_direction_prior_yield: 0.02,
            new_direction_admission_credit: 0.10,
            minimum_initial_evaluations: 1,
        }
    }

    fn descriptor(target: &str, object: &str, method: &str) -> DirectionDescriptor {
        DirectionDescriptor {
            target_capability: target.to_owned(),
            solution_object: object.to_owned(),
            construction_principle: method.to_owned(),
            scope: "task-free continual learning".to_owned(),
        }
    }

    fn admitted_board() -> (DirectionBoard, String, String) {
        let mut board = DirectionBoard::new(config()).expect("valid config");
        let first = board
            .register_verified_direction(
                descriptor("transfer", "memory topology", "typed graph rewrite"),
                RECEIPT,
            )
            .expect("first direction");
        let second = board
            .register_verified_direction(
                descriptor("transfer", "update policy", "event-triggered adaptation"),
                RECEIPT,
            )
            .expect("second direction");
        (board, first, second)
    }

    fn attempt(
        id: &str,
        direction: &str,
        score: Option<f64>,
        disposition: AttemptDisposition,
        cost: u64,
    ) -> DirectionAttempt {
        DirectionAttempt {
            attempt_id: id.to_owned(),
            direction_sha256: direction.to_owned(),
            method_id: format!("method-{id}"),
            parent_candidate_sha256: CANDIDATE_A.to_owned(),
            candidate_sha256: CANDIDATE_B.to_owned(),
            semantic_delta_sha256: DELTA.to_owned(),
            comparison_profile_sha256: PROFILE.to_owned(),
            metric_id: "held_out_task_quality".to_owned(),
            evaluation_cost_units: cost,
            disposition,
            score,
            verifier_receipt_sha256: RECEIPT.to_owned(),
        }
    }

    #[test]
    fn canonical_direction_identity_collapses_case_and_whitespace_aliases() {
        let left = descriptor("Causal Transfer", "Memory Topology", "Typed Graph Rewrite");
        let right = descriptor(" causal   transfer ", "MEMORY topology", "typed graph rewrite");
        assert_eq!(left.digest().unwrap(), right.digest().unwrap());
    }

    #[test]
    fn registering_duplicate_canonical_direction_fails_closed() {
        let mut board = DirectionBoard::new(config()).unwrap();
        board
            .register_verified_direction(
                descriptor("transfer", "memory topology", "typed graph rewrite"),
                RECEIPT,
            )
            .unwrap();
        let error = board
            .register_verified_direction(
                descriptor("TRANSFER", "  MEMORY   topology ", "typed graph rewrite"),
                RECEIPT,
            )
            .unwrap_err();
        assert!(error.to_string().contains("duplicate canonical direction"));
    }

    #[test]
    fn local_best_and_yield_credit_only_new_verified_records() {
        let (mut board, direction, _) = admitted_board();
        board
            .record_attempt(attempt(
                "a1",
                &direction,
                Some(0.60),
                AttemptDisposition::Verified,
                2,
            ))
            .unwrap();
        board
            .record_attempt(attempt(
                "a2",
                &direction,
                Some(0.55),
                AttemptDisposition::Verified,
                2,
            ))
            .unwrap();
        board
            .record_attempt(attempt(
                "a3",
                &direction,
                Some(0.80),
                AttemptDisposition::Verified,
                2,
            ))
            .unwrap();
        let stats = board
            .direction_stats()
            .into_iter()
            .find(|stats| stats.direction_sha256 == direction)
            .unwrap();
        assert_eq!(stats.local_best_score, Some(0.80));
        assert!((stats.improvement_yield_per_cost - (0.80 / 6.0)).abs() < 1e-12);
        assert_eq!(stats.verified_evaluations, 3);
    }

    #[test]
    fn rejected_and_inconclusive_attempts_remain_in_cost_denominator() {
        let (mut board, direction, _) = admitted_board();
        board
            .record_attempt(attempt(
                "a1",
                &direction,
                Some(0.60),
                AttemptDisposition::Verified,
                2,
            ))
            .unwrap();
        board
            .record_attempt(attempt(
                "a2",
                &direction,
                None,
                AttemptDisposition::Rejected,
                2,
            ))
            .unwrap();
        board
            .record_attempt(attempt(
                "a3",
                &direction,
                None,
                AttemptDisposition::Inconclusive,
                2,
            ))
            .unwrap();
        let stats = board
            .direction_stats()
            .into_iter()
            .find(|stats| stats.direction_sha256 == direction)
            .unwrap();
        assert_eq!(stats.attempted_evaluations, 3);
        assert_eq!(stats.rejected_evaluations, 1);
        assert_eq!(stats.inconclusive_evaluations, 1);
        assert!((stats.improvement_yield_per_cost - 0.10).abs() < 1e-12);
    }

    #[test]
    fn new_direction_admission_duplicate_and_rejection_are_counted() {
        let mut board = DirectionBoard::new(config()).unwrap();
        let new = descriptor("continual transfer", "context remapping", "latent regime routing");
        let new_id = board
            .record_new_direction_proposal(NewDirectionProposal {
                proposal_id: "new-1".to_owned(),
                descriptor: new.clone(),
                proposal_sha256: PROPOSAL.to_owned(),
                comparison_profile_sha256: PROFILE.to_owned(),
                metric_id: "held_out_task_quality".to_owned(),
                evaluation_cost_units: 4,
                disposition: NewDirectionDisposition::Admitted,
                verifier_receipt_sha256: RECEIPT.to_owned(),
            })
            .unwrap()
            .unwrap();
        assert_eq!(new_id, new.digest().unwrap());

        assert!(board
            .record_new_direction_proposal(NewDirectionProposal {
                proposal_id: "new-2".to_owned(),
                descriptor: new,
                proposal_sha256: PROPOSAL.to_owned(),
                comparison_profile_sha256: PROFILE.to_owned(),
                metric_id: "held_out_task_quality".to_owned(),
                evaluation_cost_units: 4,
                disposition: NewDirectionDisposition::Duplicate,
                verifier_receipt_sha256: RECEIPT.to_owned(),
            })
            .unwrap()
            .is_none());

        let stats = board.new_direction_stats();
        assert!(board
            .record_new_direction_proposal(NewDirectionProposal {
                proposal_id: "new-3".to_owned(),
                descriptor: descriptor("other target", "new object", "new construction"),
                proposal_sha256: PROPOSAL.to_owned(),
                comparison_profile_sha256: PROFILE.to_owned(),
                metric_id: "held_out_task_quality".to_owned(),
                evaluation_cost_units: 4,
                disposition: NewDirectionDisposition::Rejected,
                verifier_receipt_sha256: RECEIPT.to_owned(),
            })
            .unwrap()
            .is_none());

        let stats = board.new_direction_stats();
        assert_eq!(stats.attempted_proposals, 3);
        assert_eq!(stats.admitted_directions, 1);
        assert_eq!(stats.duplicate_proposals, 1);
        assert_eq!(stats.rejected_proposals, 1);
        assert!((stats.admissions_per_cost - (0.10 / 12.0)).abs() < 1e-12);
    }

    #[test]
    fn allocation_is_deterministic_and_conserves_budget() {
        let (board, _, _) = admitted_board();
        let left = board.recommend(17).unwrap();
        let right = board.recommend(17).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            left.allocations
                .iter()
                .map(|allocation| allocation.recommended_evaluations)
                .sum::<usize>(),
            17
        );
        assert_eq!(left.allocations.len(), 3); // two registered directions + NEW
    }

    #[test]
    fn untried_direction_and_new_arm_receive_protected_initial_exploration() {
        let (board, first, second) = admitted_board();
        let plan = board.recommend(3).unwrap();
        let counts: BTreeMap<String, usize> = plan
            .allocations
            .iter()
            .map(|allocation| (arm_key(&allocation.arm).to_owned(), allocation.recommended_evaluations))
            .collect();
        assert_eq!(counts[&first], 1);
        assert_eq!(counts[&second], 1);
        assert_eq!(counts["~NEW"], 1);
    }

    #[test]
    fn delayed_payoff_direction_is_not_permanently_eliminated() {
        let (mut board, productive, delayed) = admitted_board();
        // One direction finds an early local record; the other has three expensive
        // unproductive attempts. It remains an explicit arm and receives exploration.
        for index in 0..4 {
            board
                .record_attempt(attempt(
                    &format!("fast-{index}"),
                    &productive,
                    Some(0.70 + index as f64 * 0.05),
                    AttemptDisposition::Verified,
                    1,
                ))
                .unwrap();
        }
        for index in 0..3 {
            board
                .record_attempt(attempt(
                    &format!("slow-{index}"),
                    &delayed,
                    Some(0.10),
                    AttemptDisposition::Verified,
                    4,
                ))
                .unwrap();
        }
        let plan = board.recommend(20).unwrap();
        let delayed_allocation = plan
            .allocations
            .iter()
            .find(|allocation| allocation.arm == AllocationArm::Direction(delayed.clone()))
            .unwrap();
        assert!(delayed_allocation.recommended_evaluations > 0);
        assert!(delayed_allocation.exploration_bonus > 0.0);
    }

    #[test]
    fn rejects_attempt_from_different_metric_or_comparison_profile() {
        let (mut board, direction, _) = admitted_board();
        let mut wrong_metric = attempt(
            "bad-metric",
            &direction,
            Some(0.5),
            AttemptDisposition::Verified,
            1,
        );
        wrong_metric.metric_id = "different_metric".to_owned();
        assert!(board.record_attempt(wrong_metric).is_err());

        let mut wrong_profile = attempt(
            "bad-profile",
            &direction,
            Some(0.5),
            AttemptDisposition::Verified,
            1,
        );
        wrong_profile.comparison_profile_sha256 = CANDIDATE_A.to_owned();
        assert!(board.record_attempt(wrong_profile).is_err());
    }

    #[test]
    fn rejects_duplicate_attempt_and_zero_cost() {
        let (mut board, direction, _) = admitted_board();
        let valid = attempt(
            "same",
            &direction,
            Some(0.4),
            AttemptDisposition::Verified,
            1,
        );
        board.record_attempt(valid.clone()).unwrap();
        assert!(board.record_attempt(valid).is_err());

        let zero_cost = attempt(
            "zero-cost",
            &direction,
            Some(0.4),
            AttemptDisposition::Verified,
            0,
        );
        assert!(board.record_attempt(zero_cost).is_err());
    }

    #[test]
    fn invalid_disposition_score_combination_fails_closed() {
        let (mut board, direction, _) = admitted_board();
        let rejected_with_score = attempt(
            "bad-disposition",
            &direction,
            Some(0.7),
            AttemptDisposition::Rejected,
            1,
        );
        assert!(board.record_attempt(rejected_with_score).is_err());
    }

    #[test]
    fn board_digest_changes_after_verified_evidence() {
        let (mut board, direction, _) = admitted_board();
        let before = board.digest().unwrap();
        board
            .record_attempt(attempt(
                "a1",
                &direction,
                Some(0.7),
                AttemptDisposition::Verified,
                1,
            ))
            .unwrap();
        let after = board.digest().unwrap();
        assert_ne!(before, after);
    }
}
