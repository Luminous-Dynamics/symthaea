// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pluralistic and personalized ordinal aesthetics scoring primitives.
//!
//! These are evaluator-side point-estimate scorers, not a complete benchmark,
//! not human-normed results, and not a universal model of beauty. Ratings use
//! zero-based ordinal categories. The population scorer evaluates a predicted
//! distribution against multiple ratings per item; the personalized scorer
//! evaluates a distinct prediction for each (item, rater) pair.
//!
//! Confidence intervals must be computed at the correct sampling unit
//! (participant/rater and item), e.g. with a two-way cluster bootstrap. Treating
//! every rating as independent will generally understate uncertainty.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// One human ordinal rating for one item by one rater.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrdinalRating {
    pub item_id: String,
    pub rater_id: String,
    /// Zero-based ordinal category in [0, category_count).
    pub category: usize,
}

/// One population-level predicted distribution for an item.
///
/// This asks: "What distribution of human ratings should this item receive?"
/// It does not claim that all raters share the same preference.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationOrdinalPrediction {
    pub item_id: String,
    /// Probabilities in ascending category order; must sum to one.
    pub probabilities: Vec<f64>,
}

/// One personalized prediction for a particular rater-item pair.
///
/// This asks: "What rating is this identified participant likely to give?"
#[derive(Debug, Clone, PartialEq)]
pub struct PersonalizedOrdinalPrediction {
    pub item_id: String,
    pub rater_id: String,
    /// Probabilities in ascending category order; must sum to one.
    pub probabilities: Vec<f64>,
}

/// Population-level distribution prediction metrics.
///
/// Brier and log loss are calculated at rating-row level. Earth mover distance
/// compares each item's predicted rating distribution with its empirical
/// multi-rater distribution, then macro-averages across items.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationOrdinalMetrics {
    pub items: usize,
    pub raters: usize,
    pub ratings: usize,
    /// Mean multiclass Brier score; lower is better.
    pub brier_score: f64,
    /// Mean negative log probability of the observed category; lower is better.
    pub log_loss: f64,
    /// Modal-category accuracy; secondary and not a distributional score.
    pub modal_accuracy: f64,
    /// Mean normalized ordinal Earth mover distance; lower is better.
    pub mean_item_earth_mover_distance: f64,
}

/// Rater-conditioned prediction metrics for known users.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonalizedOrdinalMetrics {
    pub items: usize,
    pub raters: usize,
    pub ratings: usize,
    /// Mean multiclass Brier score; lower is better.
    pub brier_score: f64,
    /// Mean negative log probability of the observed category; lower is better.
    pub log_loss: f64,
    /// Modal-category accuracy; secondary and not a distributional score.
    pub modal_accuracy: f64,
}

/// Invalid scorer input. Errors are fail-closed: malformed or incomplete
/// predictions are never silently dropped, clamped, or imputed.
#[derive(Debug, Clone, PartialEq)]
pub enum OrdinalScoringError {
    EmptyRatings,
    EmptyPredictions,
    TooFewCategories { category_count: usize },
    RatingOutOfRange {
        item_id: String,
        rater_id: String,
        category: usize,
        category_count: usize,
    },
    DuplicateRating { item_id: String, rater_id: String },
    DuplicatePopulationPrediction { item_id: String },
    DuplicatePersonalizedPrediction { item_id: String, rater_id: String },
    MissingPopulationPrediction { item_id: String },
    UnexpectedPopulationPrediction { item_id: String },
    MissingPersonalizedPrediction { item_id: String, rater_id: String },
    UnexpectedPersonalizedPrediction { item_id: String, rater_id: String },
    ProbabilityCountMismatch {
        item_id: String,
        expected: usize,
        actual: usize,
    },
    InvalidProbability {
        item_id: String,
        index: usize,
        value: f64,
    },
    ProbabilityMassMismatch { item_id: String, sum: f64 },
}

impl fmt::Display for OrdinalScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRatings => write!(f, "at least one human rating is required"),
            Self::EmptyPredictions => write!(f, "at least one prediction is required"),
            Self::TooFewCategories { category_count } => {
                write!(f, "category_count must be at least 2, got {category_count}")
            }
            Self::RatingOutOfRange {
                item_id,
                rater_id,
                category,
                category_count,
            } => write!(
                f,
                "rating {category} for item {item_id} by rater {rater_id} is outside 0..{category_count}"
            ),
            Self::DuplicateRating { item_id, rater_id } => {
                write!(f, "duplicate rating for item {item_id} by rater {rater_id}")
            }
            Self::DuplicatePopulationPrediction { item_id } => {
                write!(f, "duplicate population prediction for item {item_id}")
            }
            Self::DuplicatePersonalizedPrediction { item_id, rater_id } => write!(
                f,
                "duplicate personalized prediction for item {item_id} by rater {rater_id}"
            ),
            Self::MissingPopulationPrediction { item_id } => {
                write!(f, "missing population prediction for item {item_id}")
            }
            Self::UnexpectedPopulationPrediction { item_id } => {
                write!(f, "population prediction has no human ratings for item {item_id}")
            }
            Self::MissingPersonalizedPrediction { item_id, rater_id } => write!(
                f,
                "missing personalized prediction for item {item_id} by rater {rater_id}"
            ),
            Self::UnexpectedPersonalizedPrediction { item_id, rater_id } => write!(
                f,
                "personalized prediction has no matching rating for item {item_id} by rater {rater_id}"
            ),
            Self::ProbabilityCountMismatch {
                item_id,
                expected,
                actual,
            } => write!(
                f,
                "prediction for item {item_id} has {actual} probabilities; expected {expected}"
            ),
            Self::InvalidProbability {
                item_id,
                index,
                value,
            } => write!(
                f,
                "probability {index} for item {item_id} is invalid: {value}"
            ),
            Self::ProbabilityMassMismatch { item_id, sum } => write!(
                f,
                "probabilities for item {item_id} sum to {sum}, expected 1"
            ),
        }
    }
}

impl std::error::Error for OrdinalScoringError {}

const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;
const LOG_LOSS_FLOOR: f64 = 1e-15;

/// Score population-level ordinal distributions against all observed ratings.
///
/// Ratings are retained as participant/item observations. The empirical
/// distribution is not collapsed to a mean before calculating Brier or log
/// loss, so disagreement between raters remains visible. EMD is normalized to
/// [0, 1] for equally spaced ordinal categories.
///
/// Predictions must cover exactly the set of rated items.
pub fn score_population_ordinal(
    ratings: &[OrdinalRating],
    predictions: &[PopulationOrdinalPrediction],
    category_count: usize,
) -> Result<PopulationOrdinalMetrics, OrdinalScoringError> {
    validate_common_inputs(ratings, category_count)?;
    if predictions.is_empty() {
        return Err(OrdinalScoringError::EmptyPredictions);
    }

    let mut prediction_by_item: HashMap<String, Vec<f64>> = HashMap::new();
    for prediction in predictions {
        validate_probabilities(&prediction.item_id, &prediction.probabilities, category_count)?;
        if prediction_by_item
            .insert(prediction.item_id.clone(), prediction.probabilities.clone())
            .is_some()
        {
            return Err(OrdinalScoringError::DuplicatePopulationPrediction {
                item_id: prediction.item_id.clone(),
            });
        }
    }

    let mut ratings_by_item: HashMap<String, Vec<usize>> = HashMap::new();
    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();
    let mut raters = HashSet::new();
    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut modal_correct = 0usize;

    for rating in ratings {
        let pair = (rating.item_id.clone(), rating.rater_id.clone());
        if !seen_pairs.insert(pair) {
            return Err(OrdinalScoringError::DuplicateRating {
                item_id: rating.item_id.clone(),
                rater_id: rating.rater_id.clone(),
            });
        }
        raters.insert(rating.rater_id.clone());
        ratings_by_item
            .entry(rating.item_id.clone())
            .or_default()
            .push(rating.category);

        let probabilities = prediction_by_item.get(&rating.item_id).ok_or_else(|| {
            OrdinalScoringError::MissingPopulationPrediction {
                item_id: rating.item_id.clone(),
            }
        })?;
        brier_sum += multiclass_brier(probabilities, rating.category);
        log_loss_sum += -probabilities[rating.category].max(LOG_LOSS_FLOOR).ln();
        if modal_category(probabilities) == rating.category {
            modal_correct += 1;
        }
    }

    for item_id in prediction_by_item.keys() {
        if !ratings_by_item.contains_key(item_id) {
            return Err(OrdinalScoringError::UnexpectedPopulationPrediction {
                item_id: item_id.clone(),
            });
        }
    }

    let mut emd_sum = 0.0;
    for (item_id, observed_categories) in &ratings_by_item {
        let probabilities = &prediction_by_item[item_id];
        emd_sum += ordinal_earth_mover_distance(probabilities, observed_categories, category_count);
    }

    let rating_count = ratings.len() as f64;
    let item_count = ratings_by_item.len();

    Ok(PopulationOrdinalMetrics {
        items: item_count,
        raters: raters.len(),
        ratings: ratings.len(),
        brier_score: brier_sum / rating_count,
        log_loss: log_loss_sum / rating_count,
        modal_accuracy: modal_correct as f64 / rating_count,
        mean_item_earth_mover_distance: emd_sum / item_count as f64,
    })
}

/// Score user-conditioned ordinal predictions against exact rater-item matches.
///
/// This is the known-user personalization track. An unseen-user evaluation
/// should instead use a population prior or a separately trained cold-start
/// model and be reported as a different protocol.
pub fn score_personalized_ordinal(
    ratings: &[OrdinalRating],
    predictions: &[PersonalizedOrdinalPrediction],
    category_count: usize,
) -> Result<PersonalizedOrdinalMetrics, OrdinalScoringError> {
    validate_common_inputs(ratings, category_count)?;
    if predictions.is_empty() {
        return Err(OrdinalScoringError::EmptyPredictions);
    }

    let mut prediction_by_pair: HashMap<(String, String), Vec<f64>> = HashMap::new();
    for prediction in predictions {
        validate_probabilities(&prediction.item_id, &prediction.probabilities, category_count)?;
        let pair = (prediction.item_id.clone(), prediction.rater_id.clone());
        if prediction_by_pair
            .insert(pair, prediction.probabilities.clone())
            .is_some()
        {
            return Err(OrdinalScoringError::DuplicatePersonalizedPrediction {
                item_id: prediction.item_id.clone(),
                rater_id: prediction.rater_id.clone(),
            });
        }
    }

    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();
    let mut items = HashSet::new();
    let mut raters = HashSet::new();
    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut modal_correct = 0usize;

    for rating in ratings {
        let pair = (rating.item_id.clone(), rating.rater_id.clone());
        if !seen_pairs.insert(pair.clone()) {
            return Err(OrdinalScoringError::DuplicateRating {
                item_id: rating.item_id.clone(),
                rater_id: rating.rater_id.clone(),
            });
        }
        items.insert(rating.item_id.clone());
        raters.insert(rating.rater_id.clone());

        let probabilities = prediction_by_pair.get(&pair).ok_or_else(|| {
            OrdinalScoringError::MissingPersonalizedPrediction {
                item_id: rating.item_id.clone(),
                rater_id: rating.rater_id.clone(),
            }
        })?;
        brier_sum += multiclass_brier(probabilities, rating.category);
        log_loss_sum += -probabilities[rating.category].max(LOG_LOSS_FLOOR).ln();
        if modal_category(probabilities) == rating.category {
            modal_correct += 1;
        }
    }

    for (item_id, rater_id) in prediction_by_pair.keys() {
        if !seen_pairs.contains(&(item_id.clone(), rater_id.clone())) {
            return Err(OrdinalScoringError::UnexpectedPersonalizedPrediction {
                item_id: item_id.clone(),
                rater_id: rater_id.clone(),
            });
        }
    }

    let rating_count = ratings.len() as f64;
    Ok(PersonalizedOrdinalMetrics {
        items: items.len(),
        raters: raters.len(),
        ratings: ratings.len(),
        brier_score: brier_sum / rating_count,
        log_loss: log_loss_sum / rating_count,
        modal_accuracy: modal_correct as f64 / rating_count,
    })
}

fn validate_common_inputs(
    ratings: &[OrdinalRating],
    category_count: usize,
) -> Result<(), OrdinalScoringError> {
    if ratings.is_empty() {
        return Err(OrdinalScoringError::EmptyRatings);
    }
    if category_count < 2 {
        return Err(OrdinalScoringError::TooFewCategories { category_count });
    }
    for rating in ratings {
        if rating.category >= category_count {
            return Err(OrdinalScoringError::RatingOutOfRange {
                item_id: rating.item_id.clone(),
                rater_id: rating.rater_id.clone(),
                category: rating.category,
                category_count,
            });
        }
    }
    Ok(())
}

fn validate_probabilities(
    item_id: &str,
    probabilities: &[f64],
    category_count: usize,
) -> Result<(), OrdinalScoringError> {
    if probabilities.len() != category_count {
        return Err(OrdinalScoringError::ProbabilityCountMismatch {
            item_id: item_id.to_owned(),
            expected: category_count,
            actual: probabilities.len(),
        });
    }
    let mut sum = 0.0;
    for (index, &value) in probabilities.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(OrdinalScoringError::InvalidProbability {
                item_id: item_id.to_owned(),
                index,
                value,
            });
        }
        sum += value;
    }
    if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(OrdinalScoringError::ProbabilityMassMismatch {
            item_id: item_id.to_owned(),
            sum,
        });
    }
    Ok(())
}

fn multiclass_brier(probabilities: &[f64], observed_category: usize) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, probability)| {
            let outcome = if index == observed_category { 1.0 } else { 0.0 };
            (probability - outcome).powi(2)
        })
        .sum()
}

fn modal_category(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn ordinal_earth_mover_distance(
    predicted: &[f64],
    observed: &[usize],
    category_count: usize,
) -> f64 {
    let mut observed_counts = vec![0usize; category_count];
    for &category in observed {
        observed_counts[category] += 1;
    }

    let observed_count = observed.len() as f64;
    let mut predicted_cdf = 0.0;
    let mut observed_cdf = 0.0;
    let mut distance = 0.0;
    for index in 0..(category_count - 1) {
        predicted_cdf += predicted[index];
        observed_cdf += observed_counts[index] as f64 / observed_count;
        distance += (predicted_cdf - observed_cdf).abs();
    }
    distance / (category_count - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rating(item: &str, rater: &str, category: usize) -> OrdinalRating {
        OrdinalRating {
            item_id: item.to_owned(),
            rater_id: rater.to_owned(),
            category,
        }
    }

    #[test]
    fn population_scorer_preserves_bimodal_preferences() {
        let ratings = vec![
            rating("landscape", "r1", 0),
            rating("landscape", "r2", 0),
            rating("landscape", "r3", 4),
            rating("landscape", "r4", 4),
        ];
        let predictions = vec![PopulationOrdinalPrediction {
            item_id: "landscape".to_owned(),
            probabilities: vec![0.5, 0.0, 0.0, 0.0, 0.5],
        }];

        let metrics = score_population_ordinal(&ratings, &predictions, 5).unwrap();
        assert_eq!(metrics.items, 1);
        assert_eq!(metrics.raters, 4);
        assert_eq!(metrics.ratings, 4);
        assert!(metrics.mean_item_earth_mover_distance.abs() < 1e-12);
        assert!(metrics.modal_accuracy <= 0.5);
    }

    #[test]
    fn personalized_scorer_requires_rater_item_predictions() {
        let ratings = vec![rating("poem", "r1", 0), rating("poem", "r2", 4)];
        let predictions = vec![
            PersonalizedOrdinalPrediction {
                item_id: "poem".to_owned(),
                rater_id: "r1".to_owned(),
                probabilities: vec![0.9, 0.025, 0.025, 0.025, 0.025],
            },
            PersonalizedOrdinalPrediction {
                item_id: "poem".to_owned(),
                rater_id: "r2".to_owned(),
                probabilities: vec![0.025, 0.025, 0.025, 0.025, 0.9],
            },
        ];

        let metrics = score_personalized_ordinal(&ratings, &predictions, 5).unwrap();
        assert_eq!(metrics.ratings, 2);
        assert_eq!(metrics.raters, 2);
        assert_eq!(metrics.modal_accuracy, 1.0);
        assert!(metrics.log_loss < 0.11);
    }

    #[test]
    fn rejects_probability_mass_that_does_not_sum_to_one() {
        let ratings = vec![rating("item", "r1", 0)];
        let predictions = vec![PopulationOrdinalPrediction {
            item_id: "item".to_owned(),
            probabilities: vec![0.2, 0.2, 0.2, 0.2, 0.1],
        }];

        let error = score_population_ordinal(&ratings, &predictions, 5).unwrap_err();
        assert!(matches!(
            error,
            OrdinalScoringError::ProbabilityMassMismatch { .. }
        ));
    }

    #[test]
    fn rejects_missing_personalized_predictions() {
        let ratings = vec![rating("item", "r1", 0)];
        let error = score_personalized_ordinal(&ratings, &[], 5).unwrap_err();
        assert_eq!(error, OrdinalScoringError::EmptyPredictions);
    }

    #[test]
    fn rejects_duplicate_human_rater_item_records() {
        let ratings = vec![rating("item", "r1", 0), rating("item", "r1", 1)];
        let predictions = vec![PopulationOrdinalPrediction {
            item_id: "item".to_owned(),
            probabilities: vec![0.5, 0.5],
        }];

        let error = score_population_ordinal(&ratings, &predictions, 2).unwrap_err();
        assert!(matches!(error, OrdinalScoringError::DuplicateRating { .. }));
    }
}
