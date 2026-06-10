// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observational data and causal effect estimation.
//!
//! Provides data structures for observational data and multiple estimation
//! methods including regression, IPW, and doubly robust estimation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::dag::{
    CausalDAG, CausalEstimand, CausalQuery, CausalQueryOutcome, IdentificationMethod,
};
use super::reasoner::CounterfactualReasoner;

// ─────────────────────────────────────────────────────────────────────────────
// Observational Data and Effect Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Observational data for effect estimation.
///
/// Each observation is a vector of variable values, indexed by node index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationalData {
    /// Variable names (indexed by node).
    pub variables: Vec<String>,
    /// Observations: each row contains values for all variables.
    pub observations: Vec<Vec<f64>>,
}

impl ObservationalData {
    /// Create new observational data with variable names.
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            observations: Vec::new(),
        }
    }

    /// Add an observation (row of values).
    pub fn add_observation(&mut self, values: Vec<f64>) {
        assert_eq!(
            values.len(),
            self.variables.len(),
            "Value count must match variable count"
        );
        self.observations.push(values);
    }

    /// Number of observations.
    pub fn n(&self) -> usize {
        self.observations.len()
    }

    /// Get mean of a variable.
    pub fn mean(&self, var_idx: usize) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.observations.iter().map(|row| row[var_idx]).sum();
        sum / self.observations.len() as f64
    }

    /// Get variance of a variable.
    pub fn variance(&self, var_idx: usize) -> f64 {
        if self.observations.len() < 2 {
            return 0.0;
        }
        let mean = self.mean(var_idx);
        let sum_sq: f64 = self
            .observations
            .iter()
            .map(|row| (row[var_idx] - mean).powi(2))
            .sum();
        sum_sq / (self.observations.len() - 1) as f64
    }

    /// Get covariance between two variables.
    pub fn covariance(&self, var1: usize, var2: usize) -> f64 {
        if self.observations.len() < 2 {
            return 0.0;
        }
        let mean1 = self.mean(var1);
        let mean2 = self.mean(var2);
        let sum: f64 = self
            .observations
            .iter()
            .map(|row| (row[var1] - mean1) * (row[var2] - mean2))
            .sum();
        sum / (self.observations.len() - 1) as f64
    }

    /// Filter observations by a condition on one variable.
    pub fn filter(&self, var_idx: usize, predicate: impl Fn(f64) -> bool) -> ObservationalData {
        let filtered: Vec<Vec<f64>> = self
            .observations
            .iter()
            .filter(|row| predicate(row[var_idx]))
            .cloned()
            .collect();
        ObservationalData {
            variables: self.variables.clone(),
            observations: filtered,
        }
    }

    /// Group observations by discrete values of a variable.
    pub fn group_by(&self, var_idx: usize, bins: &[f64]) -> HashMap<usize, ObservationalData> {
        let mut groups: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();

        for row in &self.observations {
            let value = row[var_idx];
            let bin = bins.iter().position(|&b| value < b).unwrap_or(bins.len());
            groups.entry(bin).or_default().push(row.clone());
        }

        groups
            .into_iter()
            .map(|(bin, obs)| {
                (
                    bin,
                    ObservationalData {
                        variables: self.variables.clone(),
                        observations: obs,
                    },
                )
            })
            .collect()
    }
}

/// Effect estimator using identified adjustment formulas.
#[derive(Debug, Clone)]
pub struct EffectEstimator {
    /// Reasoner for identification.
    reasoner: CounterfactualReasoner,
}

impl EffectEstimator {
    pub fn new() -> Self {
        Self {
            reasoner: CounterfactualReasoner::new(),
        }
    }

    /// Estimate causal effect with observational data.
    ///
    /// Returns `CausalQueryOutcome` with the estimated effect filled in.
    pub fn estimate(
        &self,
        dag: &CausalDAG,
        query: &CausalQuery,
        data: &ObservationalData,
    ) -> CausalQueryOutcome {
        // First, identify the causal effect
        let outcome = self.reasoner.query(dag, query);

        match &outcome {
            CausalQueryOutcome::Identified {
                estimand,
                method,
                confidence,
            } => {
                // Compute actual effect based on method
                let effect = match method {
                    IdentificationMethod::BackdoorAdjustment => {
                        self.estimate_backdoor(query, &estimand.adjustment_set, data)
                    }
                    IdentificationMethod::FrontdoorCriterion => {
                        self.estimate_frontdoor(query, &estimand.adjustment_set, data)
                    }
                    IdentificationMethod::DSeparation => {
                        // If d-separated, effect is 0
                        0.0
                    }
                    IdentificationMethod::Rule2ActionObservation
                    | IdentificationMethod::Rule3ActionDeletion
                    | IdentificationMethod::IDAlgorithm => {
                        // Use linear regression as fallback
                        self.estimate_regression(query.treatment, query.outcome, data)
                    }
                };

                CausalQueryOutcome::Identified {
                    estimand: CausalEstimand {
                        effect,
                        adjustment_set: estimand.adjustment_set.clone(),
                        description: estimand.description.clone(),
                    },
                    method: *method,
                    confidence: *confidence,
                }
            }
            _ => outcome,
        }
    }

    /// Estimate effect using backdoor adjustment.
    ///
    /// Formula: E[Y|do(X)] = Σ_z E[Y|X,Z=z] P(Z=z)
    ///
    /// For continuous variables, we use regression adjustment:
    /// ACE = Cov(Y, X) / Var(X) after regressing out Z
    pub(crate) fn estimate_backdoor(
        &self,
        query: &CausalQuery,
        adjustment_set: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if data.n() < 2 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;

        if adjustment_set.is_empty() {
            // No confounders: simple regression
            return self.estimate_regression(x, y, data);
        }

        // Residualize Y and X on the adjustment set, then compute covariance
        let y_residuals = self.residualize(y, adjustment_set, data);
        let x_residuals = self.residualize(x, adjustment_set, data);

        // Compute effect as Cov(Y_res, X_res) / Var(X_res)
        let n = data.n() as f64;
        let mean_y = y_residuals.iter().sum::<f64>() / n;
        let mean_x = x_residuals.iter().sum::<f64>() / n;

        let cov: f64 = y_residuals
            .iter()
            .zip(x_residuals.iter())
            .map(|(yi, xi)| (yi - mean_y) * (xi - mean_x))
            .sum();

        let var_x: f64 = x_residuals.iter().map(|xi| (xi - mean_x).powi(2)).sum();

        if var_x.abs() < 1e-10 {
            return 0.0;
        }

        cov / var_x
    }

    /// Estimate effect using frontdoor adjustment.
    ///
    /// Formula: P(Y|do(X)) = Σ_m P(M=m|X) Σ_x' P(Y|M=m,X=x') P(X=x')
    ///
    /// For continuous variables, we use the product of path coefficients:
    /// ACE = (Cov(M,X)/Var(X)) * (Cov(Y,M)/Var(M))
    fn estimate_frontdoor(
        &self,
        query: &CausalQuery,
        mediator_set: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if mediator_set.is_empty() || data.n() < 2 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;

        // Use first mediator (simplification)
        let m = mediator_set[0];

        // Effect X→M
        let effect_xm = self.estimate_regression(x, m, data);

        // Effect M→Y (controlling for X)
        let effect_my = self.estimate_regression_controlled(m, y, x, data);

        // Frontdoor effect is product of path coefficients
        effect_xm * effect_my
    }

    /// Simple linear regression coefficient: Cov(Y,X) / Var(X).
    pub(crate) fn estimate_regression(&self, x: usize, y: usize, data: &ObservationalData) -> f64 {
        let var_x = data.variance(x);
        if var_x.abs() < 1e-10 {
            return 0.0;
        }
        data.covariance(y, x) / var_x
    }

    /// Regression coefficient of X on Y, controlling for Z.
    fn estimate_regression_controlled(
        &self,
        x: usize,
        y: usize,
        control: usize,
        data: &ObservationalData,
    ) -> f64 {
        // Residualize both X and Y on control variable
        let y_residuals = self.residualize(y, &[control], data);
        let x_residuals = self.residualize(x, &[control], data);

        let n = data.n() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_y = y_residuals.iter().sum::<f64>() / n;
        let mean_x = x_residuals.iter().sum::<f64>() / n;

        let cov: f64 = y_residuals
            .iter()
            .zip(x_residuals.iter())
            .map(|(yi, xi)| (yi - mean_y) * (xi - mean_x))
            .sum();

        let var_x: f64 = x_residuals.iter().map(|xi| (xi - mean_x).powi(2)).sum();

        if var_x.abs() < 1e-10 {
            return 0.0;
        }

        cov / var_x
    }

    /// Compute residuals of variable after regressing out controls.
    fn residualize(&self, target: usize, controls: &[usize], data: &ObservationalData) -> Vec<f64> {
        if controls.is_empty() {
            return data.observations.iter().map(|row| row[target]).collect();
        }

        // Simple approach: subtract predicted value based on linear regression on controls
        // For single control: residual = Y - beta * (Z - mean_Z)
        let n = data.n();
        if n < 2 {
            return vec![0.0; n];
        }

        // Use mean-centering approach for simplicity
        let target_values: Vec<f64> = data.observations.iter().map(|row| row[target]).collect();
        let target_mean: f64 = target_values.iter().sum::<f64>() / n as f64;

        // Compute residual by regressing out each control sequentially
        let mut residuals = target_values.clone();

        for &control in controls {
            let control_mean = data.mean(control);
            let control_var = data.variance(control);

            if control_var.abs() < 1e-10 {
                continue;
            }

            // Coefficient for this control
            let cov_tc: f64 = residuals
                .iter()
                .zip(data.observations.iter())
                .map(|(r, row)| (*r - target_mean) * (row[control] - control_mean))
                .sum::<f64>()
                / (n - 1) as f64;

            let beta = cov_tc / control_var;

            // Subtract prediction
            for (i, row) in data.observations.iter().enumerate() {
                residuals[i] -= beta * (row[control] - control_mean);
            }
        }

        residuals
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Doubly Robust Estimation
    // ─────────────────────────────────────────────────────────────────────────────

    /// Estimate propensity scores P(X=1|Z) for binary treatment.
    ///
    /// Uses logistic regression approximation via sigmoid of linear predictor.
    fn estimate_propensity_scores(
        &self,
        treatment: usize,
        confounders: &[usize],
        data: &ObservationalData,
    ) -> Vec<f64> {
        if data.n() < 2 || confounders.is_empty() {
            // No confounders: assume uniform propensity
            let p = data.mean(treatment);
            return vec![p.clamp(0.01, 0.99); data.n()];
        }

        // Compute linear predictor: β₀ + Σ βᵢ * Zᵢ
        // Using simple approach: regress treatment on confounders
        let treatment_mean = data.mean(treatment);

        let mut scores = Vec::with_capacity(data.n());

        for row in &data.observations {
            // Linear predictor
            let mut lp = 0.0;

            for &z in confounders {
                let z_mean = data.mean(z);
                let z_var = data.variance(z);
                if z_var.abs() > 1e-10 {
                    let beta = data.covariance(treatment, z) / z_var;
                    lp += beta * (row[z] - z_mean);
                }
            }

            // Sigmoid to get probability
            let prob = 1.0 / (1.0 + (-lp).exp());
            // Clamp to avoid extreme weights
            scores.push(prob.clamp(0.01, 0.99));
        }

        // Normalize to have mean = treatment_mean
        let score_mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let ratio = treatment_mean / score_mean;
        for s in &mut scores {
            *s = (*s * ratio).clamp(0.01, 0.99);
        }

        scores
    }

    /// Inverse Probability Weighting (IPW) estimator.
    ///
    /// Formula: ATE = E[Y * X / e(Z)] - E[Y * (1-X) / (1-e(Z))]
    /// where e(Z) = P(X=1|Z) is the propensity score.
    ///
    /// This estimator is consistent if the propensity model is correct.
    pub fn estimate_ipw(
        &self,
        query: &CausalQuery,
        confounders: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if data.n() < 10 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;

        // Get propensity scores
        let propensity = self.estimate_propensity_scores(x, confounders, data);

        // IPW estimator
        let mut sum_treated = 0.0;
        let mut sum_control = 0.0;
        let mut weight_treated = 0.0;
        let mut weight_control = 0.0;

        for (i, row) in data.observations.iter().enumerate() {
            let x_val = row[x];
            let y_val = row[y];
            let e = propensity[i];

            if x_val > 0.5 {
                // Treated unit
                let w = 1.0 / e;
                sum_treated += y_val * w;
                weight_treated += w;
            } else {
                // Control unit
                let w = 1.0 / (1.0 - e);
                sum_control += y_val * w;
                weight_control += w;
            }
        }

        if weight_treated < 1e-10 || weight_control < 1e-10 {
            return 0.0;
        }

        // Normalized IPW (Hajek estimator)
        let mean_treated = sum_treated / weight_treated;
        let mean_control = sum_control / weight_control;

        mean_treated - mean_control
    }

    /// Doubly Robust (DR) estimator combining regression and IPW.
    ///
    /// Formula: DR = E[(X/e - (1-X)/(1-e)) * (Y - μ(X,Z)) + μ(1,Z) - μ(0,Z)]
    ///
    /// This estimator is consistent if EITHER the outcome model OR the
    /// propensity model is correct (hence "doubly robust").
    pub fn estimate_doubly_robust(
        &self,
        query: &CausalQuery,
        confounders: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if data.n() < 10 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;
        let n = data.n() as f64;

        // Get propensity scores
        let propensity = self.estimate_propensity_scores(x, confounders, data);

        // Compute outcome model predictions: E[Y|X,Z]
        // Use linear regression: Y = α + β*X + Σγᵢ*Zᵢ
        let y_mean = data.mean(y);
        let x_mean = data.mean(x);

        // Get regression coefficients
        let beta_x = if data.variance(x) > 1e-10 {
            data.covariance(y, x) / data.variance(x)
        } else {
            0.0
        };

        let mut gamma = Vec::new();
        for &z in confounders {
            let var_z = data.variance(z);
            let g = if var_z > 1e-10 {
                data.covariance(y, z) / var_z
            } else {
                0.0
            };
            gamma.push(g);
        }

        // Compute DR estimator
        let mut dr_sum = 0.0;

        for (i, row) in data.observations.iter().enumerate() {
            let x_val = row[x];
            let y_val = row[y];
            let e = propensity[i];

            // Predicted outcome given actual treatment
            let mut mu_x = y_mean + beta_x * (x_val - x_mean);
            for (j, &z) in confounders.iter().enumerate() {
                mu_x += gamma[j] * (row[z] - data.mean(z));
            }

            // Predicted outcome if treated (X=1)
            let mut mu_1 = y_mean + beta_x * (1.0 - x_mean);
            for (j, &z) in confounders.iter().enumerate() {
                mu_1 += gamma[j] * (row[z] - data.mean(z));
            }

            // Predicted outcome if control (X=0)
            let mut mu_0 = y_mean + beta_x * (0.0 - x_mean);
            for (j, &z) in confounders.iter().enumerate() {
                mu_0 += gamma[j] * (row[z] - data.mean(z));
            }

            // DR term for this observation
            let ipw_term = if x_val > 0.5 {
                (y_val - mu_x) / e
            } else {
                -(y_val - mu_x) / (1.0 - e)
            };

            let dr_i = ipw_term + mu_1 - mu_0;
            dr_sum += dr_i;
        }

        dr_sum / n
    }

    /// Estimate with all methods and return most robust result.
    ///
    /// Computes regression, IPW, and doubly robust estimates,
    /// then returns the DR estimate with diagnostics.
    pub fn estimate_robust(
        &self,
        dag: &CausalDAG,
        query: &CausalQuery,
        data: &ObservationalData,
    ) -> RobustEstimate {
        // First identify the causal effect
        let outcome = self.reasoner.query(dag, query);

        let (adjustment_set, method) = match &outcome {
            CausalQueryOutcome::Identified {
                estimand, method, ..
            } => (estimand.adjustment_set.clone(), *method),
            _ => {
                return RobustEstimate {
                    effect: 0.0,
                    regression_estimate: 0.0,
                    ipw_estimate: 0.0,
                    dr_estimate: 0.0,
                    method: IdentificationMethod::DSeparation, // Default for unidentified
                    is_identified: false,
                };
            }
        };

        // Compute all estimates
        let regression = self.estimate_backdoor(query, &adjustment_set, data);
        let ipw = self.estimate_ipw(query, &adjustment_set, data);
        let dr = self.estimate_doubly_robust(query, &adjustment_set, data);

        RobustEstimate {
            effect: dr, // Use DR as primary estimate
            regression_estimate: regression,
            ipw_estimate: ipw,
            dr_estimate: dr,
            method,
            is_identified: true,
        }
    }
}

/// Result of robust effect estimation with multiple methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustEstimate {
    /// Primary effect estimate (doubly robust).
    pub effect: f64,
    /// Regression-based estimate.
    pub regression_estimate: f64,
    /// IPW estimate.
    pub ipw_estimate: f64,
    /// Doubly robust estimate.
    pub dr_estimate: f64,
    /// Identification method used.
    pub method: IdentificationMethod,
    /// Whether the effect was identified.
    pub is_identified: bool,
}

impl RobustEstimate {
    /// Check if estimates agree (diagnostic for model misspecification).
    ///
    /// Large disagreement between methods suggests model problems.
    pub fn estimates_agree(&self, tolerance: f64) -> bool {
        let max_diff = (self.regression_estimate - self.ipw_estimate)
            .abs()
            .max((self.regression_estimate - self.dr_estimate).abs())
            .max((self.ipw_estimate - self.dr_estimate).abs());
        max_diff < tolerance
    }

    /// Get confidence based on estimate agreement.
    pub fn confidence(&self) -> f64 {
        if !self.is_identified {
            return 0.0;
        }

        let spread = (self.regression_estimate - self.ipw_estimate)
            .abs()
            .max((self.regression_estimate - self.dr_estimate).abs());

        // Higher agreement = higher confidence
        (1.0 / (1.0 + spread)).min(0.95)
    }

    /// Compute E-value for sensitivity analysis.
    ///
    /// The E-value quantifies the minimum strength of association that an unmeasured
    /// confounder would need with both treatment and outcome to fully explain away
    /// the observed effect.
    ///
    /// Interpretation:
    /// - E-value = 1.0: No unmeasured confounding needed (null effect)
    /// - E-value = 2.0: Confounder needs RR ≥ 2 with both T and Y to explain away
    /// - E-value > 3.0: Strong robustness to unmeasured confounding
    ///
    /// Reference: VanderWeele & Ding (2017). Annals of Internal Medicine.
    pub fn e_value(&self) -> f64 {
        // Convert effect (assumed standardized mean difference) to risk ratio
        // Using the approximation: RR ≈ exp(0.91 * d) for d in reasonable range
        let rr = self.effect_to_risk_ratio(self.effect.abs());

        // E-value formula: RR + sqrt(RR * (RR - 1))
        if rr <= 1.0 {
            1.0 // No unmeasured confounding needed
        } else {
            rr + (rr * (rr - 1.0)).sqrt()
        }
    }

    /// E-value for the confidence interval bound.
    ///
    /// This is the E-value for the CI bound closest to null.
    /// More conservative: how strong must confounding be to shift CI to include null?
    pub fn e_value_ci(&self, ci_lower: f64, ci_upper: f64) -> f64 {
        // Find CI bound closest to null
        let bound_closest_to_null = if ci_lower.abs() < ci_upper.abs() {
            ci_lower.abs()
        } else {
            ci_upper.abs()
        };

        // If CI crosses null, E-value_CI = 1
        if ci_lower * ci_upper < 0.0 {
            return 1.0;
        }

        let rr = self.effect_to_risk_ratio(bound_closest_to_null);
        if rr <= 1.0 {
            1.0
        } else {
            rr + (rr * (rr - 1.0)).sqrt()
        }
    }

    /// Convert standardized mean difference to risk ratio.
    ///
    /// Uses the approximation from Chinn (2000):
    /// RR ≈ exp(d * π / sqrt(3)) ≈ exp(0.91 * d)
    fn effect_to_risk_ratio(&self, d: f64) -> f64 {
        // exp(d * π / sqrt(3)) ≈ exp(1.814 * d)
        // Using more conservative conversion: exp(0.91 * d)
        (0.91 * d).exp()
    }

    /// Compute robustness to unmeasured confounding.
    ///
    /// Returns a sensitivity analysis summary including:
    /// - E-value (point estimate)
    /// - Required confounder-treatment RR
    /// - Required confounder-outcome RR
    /// - Interpretation
    pub fn sensitivity_analysis(&self) -> SensitivityAnalysis {
        let e_value = self.e_value();

        let interpretation = if e_value < 1.5 {
            "Weak: Small unmeasured confounding could explain effect"
        } else if e_value < 2.0 {
            "Moderate: Medium confounding needed to explain effect"
        } else if e_value < 3.0 {
            "Good: Strong confounding needed to explain effect"
        } else {
            "Robust: Very strong confounding needed to explain effect"
        };

        SensitivityAnalysis {
            e_value,
            e_value_interpretation: interpretation.to_string(),
            robustness_score: (e_value - 1.0).min(5.0) / 5.0, // Normalized 0-1
            min_confounder_rr_treatment: e_value,
            min_confounder_rr_outcome: e_value,
        }
    }
}

/// Sensitivity analysis results for unmeasured confounding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// E-value: minimum confounder strength to explain away effect.
    pub e_value: f64,
    /// Human-readable interpretation.
    pub e_value_interpretation: String,
    /// Robustness score (0-1, higher = more robust).
    pub robustness_score: f64,
    /// Minimum RR between confounder and treatment.
    pub min_confounder_rr_treatment: f64,
    /// Minimum RR between confounder and outcome.
    pub min_confounder_rr_outcome: f64,
}

impl Default for EffectEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // ObservationalData Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_data_new_and_add_observation() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        assert_eq!(data.n(), 0);
        assert_eq!(data.variables.len(), 2);

        data.add_observation(vec![1.0, 2.0]);
        assert_eq!(data.n(), 1);

        data.add_observation(vec![3.0, 4.0]);
        assert_eq!(data.n(), 2);
    }

    #[test]
    #[should_panic(expected = "Value count must match variable count")]
    fn test_data_add_observation_wrong_length() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        data.add_observation(vec![1.0]); // Wrong length
    }

    #[test]
    fn test_data_mean_empty() {
        let data = ObservationalData::new(vec!["X".into()]);
        assert!((data.mean(0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_data_mean_computed_correctly() {
        let mut data = ObservationalData::new(vec!["X".into()]);
        data.add_observation(vec![2.0]);
        data.add_observation(vec![4.0]);
        data.add_observation(vec![6.0]);
        assert!((data.mean(0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_data_variance_single_obs() {
        let mut data = ObservationalData::new(vec!["X".into()]);
        data.add_observation(vec![5.0]);
        assert!((data.variance(0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_data_variance_computed_correctly() {
        let mut data = ObservationalData::new(vec!["X".into()]);
        // Values: 2, 4, 6, 8 → mean=5, var = (9+1+1+9)/3 = 20/3
        data.add_observation(vec![2.0]);
        data.add_observation(vec![4.0]);
        data.add_observation(vec![6.0]);
        data.add_observation(vec![8.0]);
        let expected_var = 20.0 / 3.0;
        assert!(
            (data.variance(0) - expected_var).abs() < 1e-10,
            "Expected variance {}, got {}",
            expected_var,
            data.variance(0)
        );
    }

    #[test]
    fn test_data_covariance_perfect_positive() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..10 {
            let x = i as f64;
            data.add_observation(vec![x, 2.0 * x]);
        }
        let cov = data.covariance(0, 1);
        // Cov(X, 2X) = 2 * Var(X)
        let var_x = data.variance(0);
        assert!(
            (cov - 2.0 * var_x).abs() < 1e-10,
            "Cov(X, 2X) should equal 2*Var(X)"
        );
    }

    #[test]
    fn test_data_covariance_independent() {
        // Constant Y → zero covariance
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..20 {
            data.add_observation(vec![i as f64, 5.0]);
        }
        assert!(
            data.covariance(0, 1).abs() < 1e-10,
            "Covariance with constant should be zero"
        );
    }

    #[test]
    fn test_data_filter() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..10 {
            data.add_observation(vec![i as f64, (i * 2) as f64]);
        }
        let filtered = data.filter(0, |x| x >= 5.0);
        assert_eq!(filtered.n(), 5);
        assert_eq!(filtered.variables.len(), 2);
        // Mean of 5,6,7,8,9 = 7
        assert!((filtered.mean(0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_data_filter_none_pass() {
        let mut data = ObservationalData::new(vec!["X".into()]);
        data.add_observation(vec![1.0]);
        data.add_observation(vec![2.0]);
        let filtered = data.filter(0, |x| x > 100.0);
        assert_eq!(filtered.n(), 0);
    }

    #[test]
    fn test_data_group_by() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..20 {
            data.add_observation(vec![i as f64, i as f64 * 2.0]);
        }
        let groups = data.group_by(0, &[5.0, 10.0, 15.0]);
        // Bins: [<5), [5,10), [10,15), [15+)
        assert!(groups.contains_key(&0)); // 0-4
        assert!(groups.contains_key(&1)); // 5-9
        assert!(groups.contains_key(&2)); // 10-14
        assert!(groups.contains_key(&3)); // 15-19
        assert_eq!(groups[&0].n(), 5);
        assert_eq!(groups[&1].n(), 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EffectEstimator Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_estimator_new_and_default() {
        let est1 = EffectEstimator::new();
        let est2 = EffectEstimator::default();
        // Both should construct successfully
        assert!(std::mem::size_of_val(&est1) > 0);
        assert!(std::mem::size_of_val(&est2) > 0);
    }

    #[test]
    fn test_regression_estimate_perfect_linear() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..100 {
            let x = i as f64 / 10.0;
            let y = 3.0 * x + 1.0;
            data.add_observation(vec![x, y]);
        }
        let est = EffectEstimator::new();
        let beta = est.estimate_regression(0, 1, &data);
        assert!(
            (beta - 3.0).abs() < 1e-6,
            "Regression coefficient should be 3.0, got {}",
            beta
        );
    }

    #[test]
    fn test_regression_estimate_zero_variance() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for _ in 0..10 {
            data.add_observation(vec![5.0, 10.0]); // Constant X
        }
        let est = EffectEstimator::new();
        let beta = est.estimate_regression(0, 1, &data);
        assert!(
            (beta - 0.0).abs() < 1e-10,
            "Zero variance X should return 0"
        );
    }

    #[test]
    fn test_backdoor_estimate_no_confounders() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..100 {
            let x = i as f64 / 50.0;
            let y = 2.5 * x + 0.01 * (i % 7) as f64;
            data.add_observation(vec![x, y]);
        }
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let effect = est.estimate_backdoor(&query, &[], &data);
        assert!(
            (effect - 2.5).abs() < 0.1,
            "No-confounder backdoor should be ~2.5, got {}",
            effect
        );
    }

    #[test]
    fn test_backdoor_estimate_with_confounder() {
        // Z confounds X and Y
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);
        for i in 0..200 {
            let z = (i % 5) as f64;
            let x = 0.5 * z + 0.01 * (i % 3) as f64;
            let y = 2.0 * x + 1.0 * z + 0.01 * (i % 7) as f64;
            data.add_observation(vec![x, y, z]);
        }
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let effect = est.estimate_backdoor(&query, &[2], &data);
        // After adjusting for Z, should get closer to true effect of 2.0
        assert!(
            (effect - 2.0).abs() < 1.0,
            "Backdoor with confounder should be ~2.0, got {}",
            effect
        );
    }

    #[test]
    fn test_backdoor_estimate_insufficient_data() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        data.add_observation(vec![1.0, 2.0]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let effect = est.estimate_backdoor(&query, &[], &data);
        assert!((effect - 0.0).abs() < 1e-10, "Single obs should return 0");
    }

    #[test]
    fn test_estimate_identifies_and_computes() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..100 {
            let x = i as f64 / 50.0;
            let y = 2.0 * x + 0.01 * (i % 5) as f64;
            data.add_observation(vec![x, y]);
        }
        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let result = est.estimate(&dag, &query, &data);
        match result {
            CausalQueryOutcome::Identified { estimand, .. } => {
                assert!(
                    (estimand.effect - 2.0).abs() < 0.1,
                    "Estimated effect should be ~2.0, got {}",
                    estimand.effect
                );
            }
            _ => panic!("Expected Identified result"),
        }
    }

    #[test]
    fn test_estimate_unidentified_passthrough() {
        // Disconnected nodes → Unidentified
        let data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let result = est.estimate(&dag, &query, &data);
        assert!(matches!(result, CausalQueryOutcome::Unidentified { .. }));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IPW Estimation Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_ipw_insufficient_data() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..5 {
            data.add_observation(vec![i as f64, i as f64 * 2.0]);
        }
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let ipw = est.estimate_ipw(&query, &[], &data);
        assert!(
            (ipw - 0.0).abs() < 1e-10,
            "IPW with <10 obs should return 0"
        );
    }

    #[test]
    fn test_ipw_binary_treatment() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);
        for i in 0..200 {
            let z = (i % 2) as f64;
            let x = if (i % 5) < 3 { z } else { 1.0 - z };
            let y = 2.0 * x + 0.5 * z + 0.05 * (i % 7) as f64;
            data.add_observation(vec![x, y, z]);
        }
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let ipw = est.estimate_ipw(&query, &[2], &data);
        assert!(ipw.is_finite(), "IPW estimate should be finite");
        assert!(
            (ipw - 2.0).abs() < 2.0,
            "IPW estimate should be in reasonable range, got {}",
            ipw
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Doubly Robust Estimation Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dr_insufficient_data() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..5 {
            data.add_observation(vec![i as f64, i as f64 * 2.0]);
        }
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let dr = est.estimate_doubly_robust(&query, &[], &data);
        assert!((dr - 0.0).abs() < 1e-10, "DR with <10 obs should return 0");
    }

    #[test]
    fn test_dr_binary_treatment_estimate() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);
        for i in 0..300 {
            let z = (i % 3) as f64 / 2.0;
            let x = if z + 0.1 * (i % 5) as f64 > 0.5 {
                1.0
            } else {
                0.0
            };
            let y = 1.5 * x + 0.8 * z + 0.02 * (i % 11) as f64;
            data.add_observation(vec![x, y, z]);
        }
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let dr = est.estimate_doubly_robust(&query, &[2], &data);
        assert!(dr.is_finite(), "DR estimate should be finite");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Robust Estimation Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_robust_estimate_all_methods() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);
        for i in 0..300 {
            let z = (i % 4) as f64 / 3.0;
            let x = if z + 0.1 * (i % 5) as f64 > 0.4 {
                1.0
            } else {
                0.0
            };
            let y = 2.0 * x + 0.5 * z + 0.02 * (i % 7) as f64;
            data.add_observation(vec![x, y, z]);
        }
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (2, 0), (2, 1)],
        );
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let robust = est.estimate_robust(&dag, &query, &data);

        assert!(robust.is_identified);
        assert!(robust.regression_estimate.is_finite());
        assert!(robust.ipw_estimate.is_finite());
        assert!(robust.dr_estimate.is_finite());
        assert!(
            (robust.effect - robust.dr_estimate).abs() < 1e-10,
            "Primary effect should be DR"
        );
    }

    #[test]
    fn test_robust_estimate_unidentified() {
        let data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let est = EffectEstimator::new();
        let robust = est.estimate_robust(&dag, &query, &data);
        assert!(!robust.is_identified);
        assert!((robust.effect - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_robust_estimate_agreement() {
        let estimate = RobustEstimate {
            effect: 1.0,
            regression_estimate: 1.0,
            ipw_estimate: 1.05,
            dr_estimate: 1.02,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        assert!(estimate.estimates_agree(0.1));
        assert!(!estimate.estimates_agree(0.01));
    }

    #[test]
    fn test_robust_estimate_disagreement() {
        let estimate = RobustEstimate {
            effect: 2.0,
            regression_estimate: 1.0,
            ipw_estimate: 3.0,
            dr_estimate: 2.0,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        assert!(!estimate.estimates_agree(1.0));
    }

    #[test]
    fn test_robust_confidence_identified() {
        let estimate = RobustEstimate {
            effect: 1.0,
            regression_estimate: 1.0,
            ipw_estimate: 1.0,
            dr_estimate: 1.0,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        assert!(
            estimate.confidence() > 0.9,
            "Perfect agreement should have high confidence"
        );
    }

    #[test]
    fn test_robust_confidence_unidentified() {
        let estimate = RobustEstimate {
            effect: 0.0,
            regression_estimate: 0.0,
            ipw_estimate: 0.0,
            dr_estimate: 0.0,
            method: IdentificationMethod::DSeparation,
            is_identified: false,
        };
        assert!((estimate.confidence() - 0.0).abs() < 1e-10);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // E-Value and Sensitivity Analysis Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_e_value_null_effect() {
        let estimate = RobustEstimate {
            effect: 0.0,
            regression_estimate: 0.0,
            ipw_estimate: 0.0,
            dr_estimate: 0.0,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        assert!(
            (estimate.e_value() - 1.0).abs() < 1e-10,
            "Null effect should have E-value = 1.0"
        );
    }

    #[test]
    fn test_e_value_large_effect() {
        let estimate = RobustEstimate {
            effect: 2.0,
            regression_estimate: 2.0,
            ipw_estimate: 2.0,
            dr_estimate: 2.0,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        let e = estimate.e_value();
        assert!(e > 3.0, "Large effect should have E-value > 3, got {}", e);
    }

    #[test]
    fn test_e_value_ci_crosses_null() {
        let estimate = RobustEstimate {
            effect: 0.5,
            regression_estimate: 0.5,
            ipw_estimate: 0.5,
            dr_estimate: 0.5,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        // CI crosses null: [-0.1, 1.1]
        let e_ci = estimate.e_value_ci(-0.1, 1.1);
        assert!(
            (e_ci - 1.0).abs() < 1e-10,
            "CI crossing null should have E-value_CI = 1.0"
        );
    }

    #[test]
    fn test_e_value_ci_doesnt_cross_null() {
        let estimate = RobustEstimate {
            effect: 2.0,
            regression_estimate: 2.0,
            ipw_estimate: 2.0,
            dr_estimate: 2.0,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        // CI doesn't cross null: [0.5, 3.5]
        let e_ci = estimate.e_value_ci(0.5, 3.5);
        assert!(
            e_ci > 1.0,
            "CI not crossing null should have E-value_CI > 1"
        );
    }

    #[test]
    fn test_sensitivity_analysis_weak() {
        let estimate = RobustEstimate {
            effect: 0.1,
            regression_estimate: 0.1,
            ipw_estimate: 0.1,
            dr_estimate: 0.1,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        let sa = estimate.sensitivity_analysis();
        assert!(sa.e_value >= 1.0);
        assert!(sa.robustness_score >= 0.0 && sa.robustness_score <= 1.0);
        assert!(!sa.e_value_interpretation.is_empty());
    }

    #[test]
    fn test_sensitivity_analysis_robust() {
        let estimate = RobustEstimate {
            effect: 3.0,
            regression_estimate: 3.0,
            ipw_estimate: 3.0,
            dr_estimate: 3.0,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };
        let sa = estimate.sensitivity_analysis();
        assert!(sa.e_value > 3.0, "Strong effect should have E-value > 3");
        assert!(sa.e_value_interpretation.contains("Robust"));
        assert!(sa.min_confounder_rr_treatment > 1.0);
        assert!(sa.min_confounder_rr_outcome > 1.0);
    }
}
