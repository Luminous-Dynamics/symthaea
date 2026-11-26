//! Representative, informative, and de-amplifying (R-IDeA) acquisition
//! reference controls for SDCB.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_ridea.rs -o /tmp/sdcb-ridea
//!
//! The equations follow Tang, Sloman & Kaski (AISTATS 2026), Section 4:
//! R-I = EIG * (1 - lambda * MMD(history + candidate, target) /
//!                    MMD(history, target))
//! DeA = sigmoid((abs(f_hat(candidate) - g(candidate)) - tau) / kappa)
//! R-IDeA = R-I * DeA
//!
//! This reference uses the paper's biased empirical MMD^2 estimator with an
//! RBF kernel (fixed sigma is caller-supplied), then its square root as MMD.
//! It accepts a proxy prediction g from an external proxy model: it does not
//! train that proxy. Cost-per-unit ranking is reported as a separate project
//! extension and is NOT part of the original R-IDeA acquisition equation.

const EPS: f64 = 1e-12;
const MAX_SAMPLES: usize = 1024;
const MAX_DIMENSION: usize = 64;
const MAX_CANDIDATES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RideaError {
    EmptyHistoryOrTarget,
    InvalidDimension,
    InvalidSample,
    InvalidKernelBandwidth,
    InvalidCandidate,
    InvalidConfig,
    ZeroBaselineMmd,
    NumericalInconsistency,
    DuplicateCandidateId,
    SearchSpaceTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionKind {
    RepresentativeInformative,
    RepresentativeInformativeDeAmplifying,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RideaConfig {
    /// lambda in the R-I robust ratio; the reference constrains it to [0, 1].
    pub representativeness_weight: f64,
    /// tau in the DeA threshold.
    pub deamplification_threshold: f64,
    /// kappa > 0, smoothing the DeA sigmoid.
    pub sigmoid_scale: f64,
    /// RBF kernel bandwidth sigma, following the paper's appendix setup.
    pub rbf_sigma: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RideaCandidate {
    pub id: String,
    pub design: Vec<f64>,
    /// EIG about the declared parameter/hypothesis (units are caller-declared).
    pub expected_information_gain: f64,
    pub cost_units: u64,
    /// Current predictor f_hat at the candidate design.
    pub current_prediction: f64,
    /// Proxy g at the candidate design (estimated outside this reference).
    pub proxy_prediction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RideaScore {
    pub candidate_id: String,
    pub mmd_before: f64,
    pub mmd_after: f64,
    pub robust_ratio: f64,
    pub representative_informative: f64,
    pub deamplifying_weight: f64,
    pub ridea_score: f64,
    pub cost_units: u64,
    /// Project extension for budget comparison; original R-IDeA is ridea_score.
    pub ridea_per_cost_unit: f64,
}

fn validate_samples(samples: &[Vec<f64>]) -> Result<usize, RideaError> {
    if samples.is_empty() || samples.len() > MAX_SAMPLES {
        return Err(RideaError::EmptyHistoryOrTarget);
    }
    let dimension = samples[0].len();
    if dimension == 0 || dimension > MAX_DIMENSION {
        return Err(RideaError::InvalidDimension);
    }
    for row in samples {
        if row.len() != dimension {
            return Err(RideaError::InvalidDimension);
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(RideaError::InvalidSample);
        }
    }
    Ok(dimension)
}

fn rbf_kernel(left: &[f64], right: &[f64], sigma: f64) -> f64 {
    let squared_distance: f64 = left
        .iter()
        .zip(right)
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    (-squared_distance / (2.0 * sigma * sigma)).exp()
}

/// Biased empirical MMD^2 with an RBF kernel. Diagonal terms are included,
/// matching the empirical estimator described in Tang et al., Appendix D.
pub fn mmd_squared_rbf(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    sigma: f64,
) -> Result<f64, RideaError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(RideaError::InvalidKernelBandwidth);
    }
    let left_dimension = validate_samples(left)?;
    let right_dimension = validate_samples(right)?;
    if left_dimension != right_dimension {
        return Err(RideaError::InvalidDimension);
    }

    let left_sum: f64 = left
        .iter()
        .flat_map(|a| left.iter().map(move |b| rbf_kernel(a, b, sigma)))
        .sum();
    let right_sum: f64 = right
        .iter()
        .flat_map(|a| right.iter().map(move |b| rbf_kernel(a, b, sigma)))
        .sum();
    let cross_sum: f64 = left
        .iter()
        .flat_map(|a| right.iter().map(move |b| rbf_kernel(a, b, sigma)))
        .sum();

    let left_n = left.len() as f64;
    let right_n = right.len() as f64;
    let value = left_sum / left_n.powi(2)
        + right_sum / right_n.powi(2)
        - 2.0 * cross_sum / (left_n * right_n);

    if !value.is_finite() || value < -EPS {
        return Err(RideaError::NumericalInconsistency);
    }
    Ok(value.max(0.0))
}

pub fn mmd_rbf(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    sigma: f64,
) -> Result<f64, RideaError> {
    Ok(mmd_squared_rbf(left, right, sigma)?.sqrt())
}

fn stable_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn validate_config(config: &RideaConfig) -> Result<(), RideaError> {
    if !config.representativeness_weight.is_finite()
        || !(0.0..=1.0).contains(&config.representativeness_weight)
        || !config.deamplification_threshold.is_finite()
        || config.deamplification_threshold < 0.0
        || !config.sigmoid_scale.is_finite()
        || config.sigmoid_scale <= 0.0
        || !config.rbf_sigma.is_finite()
        || config.rbf_sigma <= 0.0
    {
        return Err(RideaError::InvalidConfig);
    }
    Ok(())
}

fn validate_candidate(candidate: &RideaCandidate, dimension: usize) -> Result<(), RideaError> {
    if candidate.id.trim().is_empty()
        || candidate.design.len() != dimension
        || candidate.cost_units == 0
        || !candidate.expected_information_gain.is_finite()
        || candidate.expected_information_gain < 0.0
        || !candidate.current_prediction.is_finite()
        || !candidate.proxy_prediction.is_finite()
        || candidate.design.iter().any(|value| !value.is_finite())
    {
        return Err(RideaError::InvalidCandidate);
    }
    Ok(())
}

/// Score a single design. The representativeness term uses MMD(history,target)
/// before and after appending the candidate to the selected-design history.
///
/// If the baseline history already has zero MMD to target, the paper's ratio
/// is undefined. We fail closed instead of silently substituting EIG.
pub fn score_ridea_candidate(
    history: &[Vec<f64>],
    target_distribution: &[Vec<f64>],
    candidate: &RideaCandidate,
    config: &RideaConfig,
) -> Result<RideaScore, RideaError> {
    validate_config(config)?;
    let dimension = validate_samples(history)?;
    if validate_samples(target_distribution)? != dimension {
        return Err(RideaError::InvalidDimension);
    }
    validate_candidate(candidate, dimension)?;

    let mmd_before = mmd_rbf(history, target_distribution, config.rbf_sigma)?;
    if mmd_before <= EPS {
        return Err(RideaError::ZeroBaselineMmd);
    }

    let mut augmented_history = history.to_vec();
    augmented_history.push(candidate.design.clone());
    if augmented_history.len() > MAX_SAMPLES {
        return Err(RideaError::SearchSpaceTooLarge);
    }
    let mmd_after = mmd_rbf(
        &augmented_history,
        target_distribution,
        config.rbf_sigma,
    )?;

    let robust_ratio =
        1.0 - config.representativeness_weight * (mmd_after / mmd_before);
    let representative_informative =
        candidate.expected_information_gain * robust_ratio;

    let standardized_disagreement =
        ((candidate.current_prediction - candidate.proxy_prediction).abs()
            - config.deamplification_threshold)
            / config.sigmoid_scale;
    let deamplifying_weight = stable_sigmoid(standardized_disagreement);
    let ridea_score = representative_informative * deamplifying_weight;
    let ridea_per_cost_unit = ridea_score / candidate.cost_units as f64;

    if !mmd_before.is_finite()
        || !mmd_after.is_finite()
        || !robust_ratio.is_finite()
        || !representative_informative.is_finite()
        || !deamplifying_weight.is_finite()
        || !ridea_score.is_finite()
        || !ridea_per_cost_unit.is_finite()
    {
        return Err(RideaError::NumericalInconsistency);
    }

    Ok(RideaScore {
        candidate_id: candidate.id.clone(),
        mmd_before,
        mmd_after,
        robust_ratio,
        representative_informative,
        deamplifying_weight,
        ridea_score,
        cost_units: candidate.cost_units,
        ridea_per_cost_unit,
    })
}

/// Rank candidates by either R-I or R-IDeA per declared cost unit.
/// Negative/non-positive scores are not treated as useful acquisitions.
pub fn rank_ridea_candidates(
    history: &[Vec<f64>],
    target_distribution: &[Vec<f64>],
    candidates: &[RideaCandidate],
    config: &RideaConfig,
    kind: AcquisitionKind,
) -> Result<Vec<RideaScore>, RideaError> {
    if candidates.len() > MAX_CANDIDATES {
        return Err(RideaError::SearchSpaceTooLarge);
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index].iter().any(|old| old.id == candidate.id) {
            return Err(RideaError::DuplicateCandidateId);
        }
        scores.push(score_ridea_candidate(
            history,
            target_distribution,
            candidate,
            config,
        )?);
    }
    scores.sort_by(|a, b| {
        let left = match kind {
            AcquisitionKind::RepresentativeInformative => {
                a.representative_informative / a.cost_units as f64
            }
            AcquisitionKind::RepresentativeInformativeDeAmplifying => {
                a.ridea_per_cost_unit
            }
        };
        let right = match kind {
            AcquisitionKind::RepresentativeInformative => {
                b.representative_informative / b.cost_units as f64
            }
            AcquisitionKind::RepresentativeInformativeDeAmplifying => {
                b.ridea_per_cost_unit
            }
        };
        right.total_cmp(&left).then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });
    Ok(scores)
}

pub fn select_ridea_candidate(
    history: &[Vec<f64>],
    target_distribution: &[Vec<f64>],
    candidates: &[RideaCandidate],
    config: &RideaConfig,
    kind: AcquisitionKind,
) -> Result<Option<RideaScore>, RideaError> {
    let ranked = rank_ridea_candidates(history, target_distribution, candidates, config, kind)?;
    Ok(ranked.into_iter().find(|score| {
        let value = match kind {
            AcquisitionKind::RepresentativeInformative => {
                score.representative_informative / score.cost_units as f64
            }
            AcquisitionKind::RepresentativeInformativeDeAmplifying => {
                score.ridea_per_cost_unit
            }
        };
        value > EPS
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RideaConfig {
        RideaConfig {
            representativeness_weight: 0.5,
            deamplification_threshold: 0.5,
            sigmoid_scale: 0.5,
            rbf_sigma: 1.0,
        }
    }

    fn history() -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![2.0]]
    }

    fn target() -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![2.0]]
    }

    fn representative_candidate() -> RideaCandidate {
        RideaCandidate {
            id: "representative-informative-deamplifying".into(),
            design: vec![1.0],
            expected_information_gain: 0.8,
            cost_units: 1,
            current_prediction: 2.0,
            proxy_prediction: 0.5,
        }
    }

    fn extrapolating_candidate() -> RideaCandidate {
        RideaCandidate {
            id: "high-eig-extrapolation".into(),
            design: vec![4.0],
            expected_information_gain: 1.2,
            cost_units: 1,
            current_prediction: 0.9,
            proxy_prediction: 0.8,
        }
    }

    #[test]
    fn identical_empirical_distributions_have_zero_mmd() {
        let samples = vec![vec![0.0], vec![1.0], vec![2.0]];
        assert!(mmd_rbf(&samples, &samples, 1.0).unwrap().abs() < EPS);
    }

    #[test]
    fn representative_design_reduces_mmd_to_target() {
        let before = mmd_rbf(&history(), &target(), 1.0).unwrap();
        let mut augmented = history();
        augmented.push(vec![1.0]);
        let after = mmd_rbf(&augmented, &target(), 1.0).unwrap();
        assert!(after + EPS < before);
        assert!(after.abs() < EPS);
    }

    #[test]
    fn paper_r_i_ratio_matches_hand_fixture() {
        let score = score_ridea_candidate(
            &history(), &target(), &representative_candidate(), &config()
        ).unwrap();
        assert!((score.mmd_before - 0.19849610189092715).abs() < 1e-12);
        assert!(score.mmd_after.abs() < EPS);
        assert!((score.robust_ratio - 1.0).abs() < EPS);
        assert!((score.representative_informative - 0.8).abs() < EPS);
    }

    #[test]
    fn paper_deamplification_sigmoid_matches_hand_fixture() {
        let score = score_ridea_candidate(
            &history(), &target(), &representative_candidate(), &config()
        ).unwrap();
        assert!((score.deamplifying_weight - 0.8807970779778823).abs() < 1e-12);
        assert!((score.ridea_score - 0.7046376623823059).abs() < 1e-12);
    }

    #[test]
    fn ridea_selects_representative_design_over_higher_eig_extrapolation() {
        let selected = select_ridea_candidate(
            &history(),
            &target(),
            &[extrapolating_candidate(), representative_candidate()],
            &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying,
        ).unwrap().unwrap();
        assert_eq!(selected.candidate_id, "representative-informative-deamplifying");
    }

    #[test]
    fn r_i_penalizes_distribution_shift_even_without_deamplification() {
        let score = score_ridea_candidate(
            &history(), &target(), &extrapolating_candidate(), &config()
        ).unwrap();
        assert!(score.mmd_after > score.mmd_before);
        assert!(score.representative_informative < score.expected_information_gain_fallback());
    }

    #[test]
    fn zero_lambda_recovers_raw_eig_in_r_i() {
        let mut cfg = config();
        cfg.representativeness_weight = 0.0;
        let score = score_ridea_candidate(
            &history(), &target(), &extrapolating_candidate(), &cfg
        ).unwrap();
        assert!((score.representative_informative - 1.2).abs() < 1e-12);
    }

    #[test]
    fn zero_baseline_mmd_is_rejected_as_undefined_ratio() {
        let samples = target();
        assert_eq!(
            score_ridea_candidate(
                &samples,
                &target(),
                &representative_candidate(),
                &config(),
            ),
            Err(RideaError::ZeroBaselineMmd)
        );
    }

    #[test]
    fn invalid_sigmoid_scale_is_rejected() {
        let mut cfg = config();
        cfg.sigmoid_scale = 0.0;
        assert_eq!(
            score_ridea_candidate(
                &history(), &target(), &representative_candidate(), &cfg
            ),
            Err(RideaError::InvalidConfig)
        );
    }

    #[test]
    fn candidate_cost_is_reported_separately_from_paper_score() {
        let mut candidate = representative_candidate();
        candidate.cost_units = 2;
        let score = score_ridea_candidate(&history(), &target(), &candidate, &config()).unwrap();
        assert!((score.ridea_score - 0.7046376623823059).abs() < 1e-12);
        assert!((score.ridea_per_cost_unit - 0.35231883119115295).abs() < 1e-12);
    }

    #[test]
    fn negative_robust_ratio_is_not_hidden() {
        let score = score_ridea_candidate(
            &history(), &target(), &extrapolating_candidate(), &config()
        ).unwrap();
        assert!(score.robust_ratio < 0.0);
        assert!(score.ridea_score < 0.0);
        assert_eq!(
            select_ridea_candidate(
                &history(), &target(), &[extrapolating_candidate()], &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying
            ).unwrap(),
            None
        );
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let mut second = representative_candidate();
        second.design[0] = 4.0;
        assert_eq!(
            rank_ridea_candidates(
                &history(), &target(), &[representative_candidate(), second], &config(),
                AcquisitionKind::RepresentativeInformative
            ),
            Err(RideaError::DuplicateCandidateId)
        );
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let mut candidate = representative_candidate();
        candidate.design.push(4.0);
        assert_eq!(
            score_ridea_candidate(&history(), &target(), &candidate, &config()),
            Err(RideaError::InvalidDimension)
        );
    }

    #[test]
    fn non_finite_candidate_values_are_rejected() {
        let mut candidate = representative_candidate();
        candidate.current_prediction = f64::NAN;
        assert_eq!(
            score_ridea_candidate(&history(), &target(), &candidate, &config()),
            Err(RideaError::InvalidCandidate)
        );
    }

    #[test]
    fn selector_cost_normalization_changes_ranking_without_mutating_raw_scores() {
        let mut expensive = representative_candidate();
        expensive.cost_units = 4;
        let mut cheap = representative_candidate();
        cheap.id = "cheaper".into();
        cheap.design = vec![1.0];
        cheap.cost_units = 1;
        let ranked = rank_ridea_candidates(
            &history(), &target(), &[expensive.clone(), cheap.clone()], &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying
        ).unwrap();
        assert_eq!(ranked[0].candidate_id, "cheaper");
        assert!((ranked.iter().find(|x| x.candidate_id == expensive.id).unwrap().ridea_score
            - ranked.iter().find(|x| x.candidate_id == cheap.id).unwrap().ridea_score).abs() < EPS);
    }

    #[test]
    fn invalid_lambda_range_is_rejected() {
        let mut cfg = config();
        cfg.representativeness_weight = 1.5;
        assert_eq!(
            score_ridea_candidate(&history(), &target(), &representative_candidate(), &cfg),
            Err(RideaError::InvalidConfig)
        );
    }
}
