//! Deterministic 3^4 factorial design and orthogonal contrast analysis.
//!
//! This crate deliberately has no external dependencies. It is a research
//! protocol kernel, not a music generator or a substitute for the Muse
//! composer. The future composer adapter must execute the emitted cases and
//! supply one measured scalar per case and metric.
//!
//! The design uses three equally spaced levels for four intent factors.
//! Linear and quadratic contrast bases are orthogonal:
//!   linear    = [-1, 0, 1]
//!   quadratic = [ 1,-2, 1]
//!
//! For each complete style/seed/metric block, this module computes 8 main
//! effects and 24 pairwise interaction coefficients. It intentionally does
//! not emit p-values or claim causal identification outside the frozen,
//! deterministic intervention design.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

pub const LEVEL_COUNT: usize = 3;
pub const CONDITIONS_PER_BLOCK: usize = 81;
pub const CONTRASTS_PER_BLOCK: usize = 32;

pub const DISCOVERY_STYLES: [&str; 29] = [
    "Classical",
    "Waltz",
    "Folk",
    "Cinematic",
    "Playful",
    "Nocturne",
    "March",
    "Lullaby",
    "ModalFolk",
    "Fugue",
    "Passacaglia",
    "Tango",
    "Celtic",
    "Blues",
    "Impressionism",
    "SacredChoral",
    "Minimalism",
    "JazzBallad",
    "BaroqueSuite",
    "ProgFolk",
    "Ambient",
    "Sonata",
    "RenaissancePolyphony",
    "AfroCuban",
    "Flamenco",
    "BossaNova",
    "Opera",
    "IrishTraditional",
    "HindustaniInspired",
];

/// Coverage-selected confirmation panel. It includes the shared period
/// pipeline, dedicated grammar families, and one known zero-axis style.
pub const CONFIRMATION_STYLES: [&str; 6] = [
    "Classical",
    "Fugue",
    "Passacaglia",
    "Blues",
    "AfroCuban",
    "HindustaniInspired",
];

pub const DISCOVERY_SEEDS: [u64; 4] = [90_210, 130_363, 222_731, 918_273];

pub const CONFIRMATION_SEEDS: [u64; 12] = [
    1_000_003,
    1_000_033,
    1_000_037,
    1_000_039,
    1_000_081,
    1_000_099,
    1_000_117,
    1_000_121,
    1_000_133,
    1_000_151,
    1_000_159,
    1_000_171,
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Cohort {
    Discovery,
    Confirmation,
}

impl Cohort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Discovery => "DISCOVERY",
            Self::Confirmation => "CONFIRMATION",
        }
    }

    pub fn styles(self) -> &'static [&'static str] {
        match self {
            Self::Discovery => &DISCOVERY_STYLES,
            Self::Confirmation => &CONFIRMATION_STYLES,
        }
    }

    pub fn seeds(self) -> &'static [u64] {
        match self {
            Self::Discovery => &DISCOVERY_SEEDS,
            Self::Confirmation => &CONFIRMATION_SEEDS,
        }
    }

    pub fn expected_case_count(self) -> usize {
        self.styles().len() * self.seeds().len() * CONDITIONS_PER_BLOCK
    }
}

/// Each factor code is 0=low, 1=centre, 2=high.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Cell {
    pub valence: u8,
    pub arousal: u8,
    pub energy: u8,
    pub bars: u8,
}

impl Cell {
    pub fn all() -> Vec<Self> {
        let mut cells = Vec::with_capacity(CONDITIONS_PER_BLOCK);
        for valence in 0_u8..3 {
            for arousal in 0_u8..3 {
                for energy in 0_u8..3 {
                    for bars in 0_u8..3 {
                        cells.push(Self {
                            valence,
                            arousal,
                            energy,
                            bars,
                        });
                    }
                }
            }
        }
        cells
    }

    fn level(self, factor: Factor) -> u8 {
        match factor {
            Factor::Valence => self.valence,
            Factor::Arousal => self.arousal,
            Factor::Energy => self.energy,
            Factor::Bars => self.bars,
        }
    }

    fn levels_are_valid(self) -> bool {
        self.valence < 3 && self.arousal < 3 && self.energy < 3 && self.bars < 3
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Case {
    pub cohort: Cohort,
    pub style: String,
    pub seed: u64,
    pub cell: Cell,
    pub case_id: String,
}

pub fn case_id(cohort: Cohort, style: &str, seed: u64, cell: Cell) -> String {
    format!(
        "{}__{}__s{}__v{}a{}e{}b{}",
        cohort.as_str(),
        style,
        seed,
        cell.valence,
        cell.arousal,
        cell.energy,
        cell.bars
    )
}

pub fn generate_cases(cohort: Cohort) -> Vec<Case> {
    let mut cases = Vec::with_capacity(cohort.expected_case_count());
    for style in cohort.styles() {
        for seed in cohort.seeds() {
            for cell in Cell::all() {
                cases.push(Case {
                    cohort,
                    style: (*style).to_string(),
                    seed: *seed,
                    cell,
                    case_id: case_id(cohort, style, *seed, cell),
                });
            }
        }
    }
    cases
}

/// Compare a materialized case manifest to the complete frozen design.
/// Missing, duplicated, unexpected, or mutated cases are errors.
pub fn validate_case_set(cohort: Cohort, actual: &[Case]) -> Result<(), AuditError> {
    let expected = generate_cases(cohort);
    let expected_by_id: HashMap<&str, &Case> = expected
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();
    let mut seen = HashSet::with_capacity(actual.len());

    for case in actual {
        if !seen.insert(case.case_id.as_str()) {
            return Err(AuditError::DuplicateCase(case.case_id.clone()));
        }
        let Some(expected_case) = expected_by_id.get(case.case_id.as_str()) else {
            return Err(AuditError::UnexpectedCase(case.case_id.clone()));
        };
        if *expected_case != case {
            return Err(AuditError::MutatedCase(case.case_id.clone()));
        }
    }

    if actual.len() != expected.len() {
        return Err(AuditError::WrongCaseCount {
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    for case in &expected {
        if !seen.contains(case.case_id.as_str()) {
            return Err(AuditError::MissingCase(case.case_id.clone()));
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Factor {
    Valence,
    Arousal,
    Energy,
    Bars,
}

impl Factor {
    const ALL: [Self; 4] = [Self::Valence, Self::Arousal, Self::Energy, Self::Bars];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Valence => "valence",
            Self::Arousal => "arousal",
            Self::Energy => "energy",
            Self::Bars => "bars",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContrastKind {
    Linear,
    Quadratic,
}

impl ContrastKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
        }
    }

    fn coefficient(self, level: u8) -> f64 {
        match (self, level) {
            (Self::Linear, 0) => -1.0,
            (Self::Linear, 1) => 0.0,
            (Self::Linear, 2) => 1.0,
            (Self::Quadratic, 0) => 1.0,
            (Self::Quadratic, 1) => -2.0,
            (Self::Quadratic, 2) => 1.0,
            _ => f64::NAN,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FactorContrast {
    pub factor: Factor,
    pub kind: ContrastKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContrastTerm {
    pub components: Vec<FactorContrast>,
}

impl ContrastTerm {
    pub fn id(&self) -> String {
        self.components
            .iter()
            .map(|component| {
                format!(
                    "{}_{}",
                    component.factor.as_str(),
                    component.kind.as_str()
                )
            })
            .collect::<Vec<_>>()
            .join("__")
    }

    fn weight(&self, cell: Cell) -> f64 {
        self.components
            .iter()
            .map(|component| {
                component
                    .kind
                    .coefficient(cell.level(component.factor))
            })
            .product()
    }
}

/// Eight main-effect contrasts plus four contrast products for each of the
/// six factor pairs: 8 + (6 * 4) = 32 terms.
pub fn contrast_terms() -> Vec<ContrastTerm> {
    let kinds = [ContrastKind::Linear, ContrastKind::Quadratic];
    let mut terms = Vec::with_capacity(CONTRASTS_PER_BLOCK);

    for factor in Factor::ALL {
        for kind in kinds {
            terms.push(ContrastTerm {
                components: vec![FactorContrast { factor, kind }],
            });
        }
    }

    for left_index in 0..Factor::ALL.len() {
        for right_index in (left_index + 1)..Factor::ALL.len() {
            for left_kind in kinds {
                for right_kind in kinds {
                    terms.push(ContrastTerm {
                        components: vec![
                            FactorContrast {
                                factor: Factor::ALL[left_index],
                                kind: left_kind,
                            },
                            FactorContrast {
                                factor: Factor::ALL[right_index],
                                kind: right_kind,
                            },
                        ],
                    });
                }
            }
        }
    }

    terms
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub cohort: Cohort,
    pub style: String,
    pub seed: u64,
    pub metric: String,
    pub cell: Cell,
    pub case_id: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContrastEstimate {
    pub term: String,
    pub estimate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EffectSummary {
    pub term: String,
    pub seed_count: usize,
    pub mean: f64,
    /// Sample standard deviation across seed-block estimates.
    /// None is used for a single seed; it is not reported as zero.
    pub sample_sd: Option<f64>,
    pub minimum: f64,
    pub maximum: f64,
}

/// Estimate all predeclared main and pairwise interaction contrasts for one
/// style/seed/metric. Every factorial cell must appear exactly once.
pub fn estimate_block(
    cohort: Cohort,
    style: &str,
    seed: u64,
    metric: &str,
    observations: &[Observation],
) -> Result<Vec<ContrastEstimate>, AuditError> {
    if !cohort.styles().contains(&style) || !cohort.seeds().contains(&seed) {
        return Err(AuditError::UnexpectedBlock {
            cohort,
            style: style.to_string(),
            seed,
        });
    }
    if observations.len() != CONDITIONS_PER_BLOCK {
        return Err(AuditError::WrongObservationCount {
            expected: CONDITIONS_PER_BLOCK,
            actual: observations.len(),
        });
    }

    let mut seen = HashSet::with_capacity(CONDITIONS_PER_BLOCK);
    for observation in observations {
        if observation.cohort != cohort
            || observation.style != style
            || observation.seed != seed
            || observation.metric != metric
        {
            return Err(AuditError::MixedBlock);
        }
        if !observation.cell.levels_are_valid() {
            return Err(AuditError::OutOfRangeCell);
        }
        if !observation.value.is_finite() {
            return Err(AuditError::NonFiniteValue(observation.case_id.clone()));
        }
        let expected_id = case_id(cohort, style, seed, observation.cell);
        if observation.case_id != expected_id {
            return Err(AuditError::MutatedCase(observation.case_id.clone()));
        }
        if !seen.insert(observation.cell) {
            return Err(AuditError::DuplicateCell(observation.cell));
        }
    }

    if seen.len() != CONDITIONS_PER_BLOCK {
        return Err(AuditError::IncompleteBlock {
            expected: CONDITIONS_PER_BLOCK,
            actual: seen.len(),
        });
    }

    let terms = contrast_terms();
    let mut estimates = Vec::with_capacity(terms.len());
    for term in terms {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for observation in observations {
            let weight = term.weight(observation.cell);
            numerator += observation.value * weight;
            denominator += weight * weight;
        }
        if denominator <= f64::EPSILON {
            return Err(AuditError::DegenerateContrast(term.id()));
        }
        estimates.push(ContrastEstimate {
            term: term.id(),
            estimate: numerator / denominator,
        });
    }

    Ok(estimates)
}

/// Aggregate equivalent contrast terms across independently executed seed
/// blocks. This reports replication dispersion, not a confidence interval or
/// a hypothesis-test p-value.
pub fn summarize_seed_effects(
    seed_blocks: &[Vec<ContrastEstimate>],
) -> Result<Vec<EffectSummary>, AuditError> {
    if seed_blocks.is_empty() {
        return Err(AuditError::NoSeedBlocks);
    }

    let expected_terms: Vec<String> = seed_blocks[0]
        .iter()
        .map(|estimate| estimate.term.clone())
        .collect();
    let mut seen_terms = HashSet::new();
    for term in &expected_terms {
        if !seen_terms.insert(term.as_str()) {
            return Err(AuditError::DuplicateContrast(term.clone()));
        }
    }

    for block in seed_blocks {
        if block.len() != expected_terms.len() {
            return Err(AuditError::InconsistentContrastSet);
        }
        let observed: HashSet<&str> = block.iter().map(|item| item.term.as_str()).collect();
        if observed.len() != block.len()
            || expected_terms.iter().any(|term| !observed.contains(term.as_str()))
        {
            return Err(AuditError::InconsistentContrastSet);
        }
        if block.iter().any(|item| !item.estimate.is_finite()) {
            return Err(AuditError::NonFiniteEffect);
        }
    }

    let mut summaries = Vec::with_capacity(expected_terms.len());
    for term in expected_terms {
        let values: Vec<f64> = seed_blocks
            .iter()
            .map(|block| {
                block
                    .iter()
                    .find(|estimate| estimate.term == term)
                    .expect("contrast membership validated above")
                    .estimate
            })
            .collect();
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let sample_sd = if count > 1 {
            let sum_squares = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>();
            Some((sum_squares / (count - 1) as f64).sqrt())
        } else {
            None
        };
        let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        summaries.push(EffectSummary {
            term,
            seed_count: count,
            mean,
            sample_sd,
            minimum,
            maximum,
        });
    }

    Ok(summaries)
}

#[derive(Clone, Debug, PartialEq)]
pub enum AuditError {
    DuplicateCase(String),
    UnexpectedCase(String),
    MutatedCase(String),
    MissingCase(String),
    WrongCaseCount { expected: usize, actual: usize },
    UnexpectedBlock { cohort: Cohort, style: String, seed: u64 },
    WrongObservationCount { expected: usize, actual: usize },
    MixedBlock,
    OutOfRangeCell,
    NonFiniteValue(String),
    DuplicateCell(Cell),
    IncompleteBlock { expected: usize, actual: usize },
    DegenerateContrast(String),
    NoSeedBlocks,
    DuplicateContrast(String),
    InconsistentContrastSet,
    NonFiniteEffect,
}

impl fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for AuditError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations_for(
        cohort: Cohort,
        style: &str,
        seed: u64,
        response: impl Fn(Cell) -> f64,
    ) -> Vec<Observation> {
        Cell::all()
            .into_iter()
            .map(|cell| Observation {
                cohort,
                style: style.to_string(),
                seed,
                metric: "synthetic_response".to_string(),
                cell,
                case_id: case_id(cohort, style, seed, cell),
                value: response(cell),
            })
            .collect()
    }

    fn find<'a>(items: &'a [ContrastEstimate], term: &str) -> &'a ContrastEstimate {
        items.iter().find(|item| item.term == term).unwrap()
    }

    #[test]
    fn discovery_and_confirmation_case_counts_are_frozen() {
        assert_eq!(generate_cases(Cohort::Discovery).len(), 9_396);
        assert_eq!(generate_cases(Cohort::Confirmation).len(), 5_832);
        assert_eq!(Cohort::Discovery.expected_case_count(), 9_396);
        assert_eq!(Cohort::Confirmation.expected_case_count(), 5_832);
    }

    #[test]
    fn case_ids_are_unique_and_confirmation_seeds_are_disjoint() {
        for cohort in [Cohort::Discovery, Cohort::Confirmation] {
            let cases = generate_cases(cohort);
            let ids: HashSet<&str> = cases.iter().map(|case| case.case_id.as_str()).collect();
            assert_eq!(ids.len(), cases.len());
        }
        for seed in CONFIRMATION_SEEDS {
            assert!(!DISCOVERY_SEEDS.contains(&seed));
        }
    }

    #[test]
    fn style_inventory_matches_source_derived_style_count() {
        assert_eq!(DISCOVERY_STYLES.len(), 29);
        assert_eq!(CONFIRMATION_STYLES.len(), 6);
        assert!(CONFIRMATION_STYLES.iter().all(|style| DISCOVERY_STYLES.contains(style)));
    }

    #[test]
    fn exact_manifest_validation_rejects_missing_duplicate_and_mutated_cases() {
        let expected = generate_cases(Cohort::Confirmation);
        validate_case_set(Cohort::Confirmation, &expected).unwrap();

        let mut missing = expected.clone();
        missing.pop();
        assert!(matches!(
            validate_case_set(Cohort::Confirmation, &missing),
            Err(AuditError::WrongCaseCount { .. })
        ));

        let mut duplicate = expected.clone();
        duplicate[0] = duplicate[1].clone();
        assert!(matches!(
            validate_case_set(Cohort::Confirmation, &duplicate),
            Err(AuditError::DuplicateCase(_))
        ));

        let mut mutated = expected;
        mutated[0].cell.energy = 2;
        assert!(matches!(
            validate_case_set(Cohort::Confirmation, &mutated),
            Err(AuditError::MutatedCase(_))
        ));
    }

    #[test]
    fn complete_design_has_eight_main_and_twenty_four_pairwise_terms() {
        let terms = contrast_terms();
        assert_eq!(terms.len(), CONTRASTS_PER_BLOCK);
        assert_eq!(terms.iter().filter(|term| term.components.len() == 1).count(), 8);
        assert_eq!(terms.iter().filter(|term| term.components.len() == 2).count(), 24);
        assert!(terms.iter().all(|term| term.components.len() <= 2));
    }

    #[test]
    fn orthogonal_contrasts_recover_known_main_and_interaction_effects() {
        let observations = observations_for(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            |cell| {
                let linear = [-1.0, 0.0, 1.0];
                let quadratic = [1.0, -2.0, 1.0];
                7.0
                    + 2.0 * linear[cell.valence as usize]
                    + 3.0 * quadratic[cell.arousal as usize]
                    + 4.0
                        * linear[cell.valence as usize]
                        * quadratic[cell.energy as usize]
            },
        );
        let estimates = estimate_block(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            "synthetic_response",
            &observations,
        )
        .unwrap();

        assert!((find(&estimates, "valence_linear").estimate - 2.0).abs() < 1.0e-12);
        assert!((find(&estimates, "arousal_quadratic").estimate - 3.0).abs() < 1.0e-12);
        assert!(
            (find(
                &estimates,
                "valence_linear__energy_quadratic"
            )
            .estimate
                - 4.0)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn incomplete_duplicate_and_nonfinite_observation_blocks_fail_closed() {
        let mut observations = observations_for(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            |_| 1.0,
        );
        observations.pop();
        assert!(matches!(
            estimate_block(
                Cohort::Discovery,
                "Classical",
                DISCOVERY_SEEDS[0],
                "synthetic_response",
                &observations
            ),
            Err(AuditError::WrongObservationCount { .. })
        ));

        let mut observations = observations_for(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            |_| 1.0,
        );
        observations[0].cell = observations[1].cell;
        observations[0].case_id = observations[1].case_id.clone();
        assert!(matches!(
            estimate_block(
                Cohort::Discovery,
                "Classical",
                DISCOVERY_SEEDS[0],
                "synthetic_response",
                &observations
            ),
            Err(AuditError::DuplicateCell(_))
        ));

        let mut observations = observations_for(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            |_| 1.0,
        );
        observations[0].value = f64::NAN;
        assert!(matches!(
            estimate_block(
                Cohort::Discovery,
                "Classical",
                DISCOVERY_SEEDS[0],
                "synthetic_response",
                &observations
            ),
            Err(AuditError::NonFiniteValue(_))
        ));
    }

    #[test]
    fn seed_aggregation_reports_dispersion_not_p_values() {
        let first = observations_for(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            |cell| [-1.0, 0.0, 1.0][cell.valence as usize],
        );
        let second = observations_for(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[1],
            |cell| 3.0 * [-1.0, 0.0, 1.0][cell.valence as usize],
        );
        let first_effects = estimate_block(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[0],
            "synthetic_response",
            &first,
        )
        .unwrap();
        let second_effects = estimate_block(
            Cohort::Discovery,
            "Classical",
            DISCOVERY_SEEDS[1],
            "synthetic_response",
            &second,
        )
        .unwrap();
        let summaries = summarize_seed_effects(&[first_effects, second_effects]).unwrap();
        let valence = summaries
            .iter()
            .find(|summary| summary.term == "valence_linear")
            .unwrap();
        assert_eq!(valence.seed_count, 2);
        assert!((valence.mean - 2.0).abs() < 1.0e-12);
        assert!(valence.sample_sd.unwrap() > 0.0);
    }

    #[test]
    fn centre_cell_is_the_null_reference_condition() {
        let centre = Cell {
            valence: 1,
            arousal: 1,
            energy: 1,
            bars: 1,
        };
        assert!(Cell::all().contains(&centre));
        assert_eq!(
            case_id(Cohort::Discovery, "Classical", DISCOVERY_SEEDS[0], centre),
            format!(
                "DISCOVERY__Classical__s{}__v1a1e1b1",
                DISCOVERY_SEEDS[0]
            )
        );
    }
}
