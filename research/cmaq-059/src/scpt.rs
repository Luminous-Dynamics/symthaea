#![forbid(unsafe_code)]

//! SCPT-style sequential conditional paired-sample reference implementation.
//! This reproduces the published sequential KNN payoff, past-only betting
//! fraction, and equal-weight mixture over a={2,4,6}. The warm-up constructor
//! estimates a Gaussian bandwidth from the pre-monitoring block; warm-up
//! observations contribute no e-capital.

use std::collections::HashSet;
use sha2::{Digest, Sha256};
use crate::scenarios::PairedObservation;

const DEFAULT_A: [f64; 3] = [2.0, 4.0, 6.0];
const DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-059/SCPT-REFERENCE/V1\0";
const MAX_HISTORY: usize = 100_000;
const MAX_ID: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScptError {
    InvalidAlpha, InvalidBandwidth, InvalidTruncation, InvalidConfiguration,
    EmptyWarmup, WarmupDimensionsMismatch, WarmupSequenceMismatch,
    InputDimensionMismatch, NonFiniteInput, InvalidPairId, PairIdTooLong,
    DuplicatePairId, HistoryFull, SequenceMismatch, AlreadyQuarantined,
    InvalidPayoff, InvalidFactor, ArithmeticOverflow,
}
impl std::fmt::Display for ScptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for ScptError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ScptConfigurationUpdate {
    pub neighborhood_coefficient: f64,
    pub neighbor_count: usize,
    pub betting_fraction: f64,
    pub payoff: f64,
    pub factor: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ScptUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub mixture_factor: f64,
    pub log_e_value: f64,
    pub log_threshold: f64,
    pub quarantined: bool,
    pub configurations: Vec<ScptConfigurationUpdate>,
    pub trace_sha256: String,
}
#[derive(Clone, Debug)]
struct History {
    covariate: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    payoffs: Vec<f64>,
}

/// Faithful reference arm with w(Z)=1, Euclidean KNN, Gaussian response kernel,
/// local betting fractions and one equal-weight factor mixture.
#[derive(Clone, Debug)]
pub struct ScptReferenceMonitor {
    sigma: f64,
    eta: f64,
    alpha: f64,
    log_threshold: f64,
    coefficients: Vec<f64>,
    seen: u64,
    history: Vec<History>,
    pair_ids: HashSet<String>,
    log_e: f64,
    quarantined: bool,
    trace: String,
    covariate_dim: Option<usize>,
    response_dim: Option<usize>,
}

impl ScptReferenceMonitor {
    pub fn paper_default(sigma: f64, eta: f64, alpha: f64) -> Result<Self, ScptError> {
        Self::new(sigma, eta, alpha, DEFAULT_A.to_vec())
    }

    pub fn new(sigma: f64, eta: f64, alpha: f64, coefficients: Vec<f64>) -> Result<Self, ScptError> {
        if !sigma.is_finite() || sigma <= 0.0 { return Err(ScptError::InvalidBandwidth); }
        if !eta.is_finite() || eta <= 0.0 { return Err(ScptError::InvalidTruncation); }
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 { return Err(ScptError::InvalidAlpha); }
        if coefficients.is_empty() || coefficients.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(ScptError::InvalidConfiguration);
        }
        let mut result = Self {
            sigma, eta, alpha, log_threshold: -alpha.ln(), coefficients,
            seen: 0, history: Vec::new(), pair_ids: HashSet::new(),
            log_e: 0.0, quarantined: false, trace: String::new(),
            covariate_dim: None, response_dim: None,
        };
        result.trace = result.initial_trace();
        Ok(result)
    }

    /// Uses pooled X/Y responses from a pre-monitoring block for the median
    /// bandwidth heuristic. No test capital is accumulated in that block.
    pub fn from_warmup(warmup: &[PairedObservation], eta: f64, alpha: f64) -> Result<Self, ScptError> {
        if warmup.is_empty() { return Err(ScptError::EmptyWarmup); }
        let sigma = median_heuristic_sigma(warmup)?;
        let mut monitor = Self::paper_default(sigma, eta, alpha)?;
        for (index, obs) in warmup.iter().enumerate() {
            let sequence = index as u64 + 1;
            if obs.sequence != sequence { return Err(ScptError::WarmupSequenceMismatch); }
            validate_obs(obs, monitor.covariate_dim, monitor.response_dim)?;
            let payoffs = monitor.compute_payoffs(sequence, obs)?;
            monitor.covariate_dim = Some(obs.covariate.len());
            monitor.response_dim = Some(obs.x.len());
            monitor.push(obs, payoffs);
            monitor.seen = sequence;
        }
        monitor.trace = warmup_trace(&monitor.trace, warmup, sigma, eta, alpha);
        Ok(monitor)
    }

    pub fn observe_pair(
        &mut self, sequence: u64, pair_id: &str, covariate: &[f64], x: &[f64], y: &[f64],
    ) -> Result<ScptUpdate, ScptError> {
        if self.quarantined { return Err(ScptError::AlreadyQuarantined); }
        if pair_id.trim().is_empty() { return Err(ScptError::InvalidPairId); }
        if pair_id.len() > MAX_ID { return Err(ScptError::PairIdTooLong); }
        if self.pair_ids.contains(pair_id) { return Err(ScptError::DuplicatePairId); }
        if self.history.len() >= MAX_HISTORY { return Err(ScptError::HistoryFull); }
        let expected = self.seen.checked_add(1).ok_or(ScptError::ArithmeticOverflow)?;
        if sequence != expected { return Err(ScptError::SequenceMismatch); }

        let obs = PairedObservation { sequence, covariate: covariate.to_vec(), x: x.to_vec(), y: y.to_vec() };
        validate_obs(&obs, self.covariate_dim, self.response_dim)?;
        let mut payoffs = Vec::with_capacity(self.coefficients.len());
        let mut updates = Vec::with_capacity(self.coefficients.len());
        let mut factor_sum = 0.0;

        for (j, a) in self.coefficients.iter().enumerate() {
            let neighbors = self.neighbors(sequence, covariate, *a);
            let (past_sum, past_squares) = self.past_moments(&neighbors, j);
            // Equation (16), w(Z)=1. Finish the predictable bet decision
            // using only prior payoffs and the pre-pair covariate.
            let positive_sum = past_sum.max(0.0);
            let denominator = past_sum + past_squares;
            let lambda = if positive_sum == 0.0 || denominator <= 0.0 {
                0.0
            } else { positive_sum / denominator };
            if !lambda.is_finite() || !(0.0..1.0).contains(&lambda) {
                return Err(ScptError::InvalidFactor);
            }

            // Only after the predictable bet is fixed do current X/Y values
            // enter the RKHS payoff calculation.
            let payoff = self.payoff(&obs, &neighbors)?;
            let factor = 1.0 + lambda * payoff;
            if !factor.is_finite() || factor <= 0.0 { return Err(ScptError::InvalidFactor); }
            factor_sum += factor;
            payoffs.push(payoff);
            updates.push(ScptConfigurationUpdate {
                neighborhood_coefficient: *a, neighbor_count: neighbors.len(),
                betting_fraction: lambda, payoff, factor,
            });
        }

        // Mixture is the arithmetic mean of predictable factors, not a
        // same-time maximum or a mixture of post-hoc selected e-values.
        let mixture_factor = factor_sum / updates.len() as f64;
        if !mixture_factor.is_finite() || mixture_factor <= 0.0 {
            return Err(ScptError::InvalidFactor);
        }
        let next_log_e = self.log_e + mixture_factor.ln();
        if !next_log_e.is_finite() { return Err(ScptError::ArithmeticOverflow); }
        let trace = extend_trace(&self.trace, sequence, pair_id, &updates, mixture_factor, next_log_e);

        // Atomic commit after all checks.
        self.covariate_dim = Some(obs.covariate.len());
        self.response_dim = Some(obs.x.len());
        self.push(&obs, payoffs);
        self.seen = sequence;
        self.pair_ids.insert(pair_id.to_string());
        self.log_e = next_log_e;
        self.quarantined = next_log_e >= self.log_threshold;
        self.trace = trace.clone();

        Ok(ScptUpdate {
            sequence, pair_id: pair_id.to_string(), mixture_factor,
            log_e_value: next_log_e, log_threshold: self.log_threshold,
            quarantined: self.quarantined, configurations: updates, trace_sha256: trace,
        })
    }

    pub fn sigma(&self) -> f64 { self.sigma }
    pub fn observations_seen(&self) -> u64 { self.seen }
    pub fn log_e_value(&self) -> f64 { self.log_e }
    pub fn trace_sha256(&self) -> &str { &self.trace }

    fn compute_payoffs(&self, sequence: u64, obs: &PairedObservation) -> Result<Vec<f64>, ScptError> {
        self.coefficients.iter().map(|a| {
            let neighbors = self.neighbors(sequence, &obs.covariate, *a);
            self.payoff(obs, &neighbors)
        }).collect()
    }

    fn neighbors(&self, sequence: u64, covariate: &[f64], a: f64) -> Vec<usize> {
        if self.history.is_empty() { return Vec::new(); }
        let requested = (a * (sequence as f64).sqrt()).ceil() as usize;
        let count = requested.min(self.history.len());
        let mut distances: Vec<(f64, usize)> = self.history.iter().enumerate()
            .map(|(i, h)| (sqdist(covariate, &h.covariate), i)).collect();
        distances.sort_by(|l, r| l.0.total_cmp(&r.0).then_with(|| l.1.cmp(&r.1)));
        distances.into_iter().take(count).map(|(_, i)| i).collect()
    }

    fn past_moments(&self, neighbors: &[usize], config: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut squares = 0.0;
        for i in neighbors {
            let s = self.history[*i].payoffs[config];
            sum += s;
            squares += s * s;
        }
        (sum, squares)
    }

    fn payoff(&self, current: &PairedObservation, neighbors: &[usize]) -> Result<f64, ScptError> {
        if neighbors.is_empty() { return Ok(0.0); }
        let mut numerator = 0.0;
        for i in neighbors {
            let h = &self.history[*i];
            numerator += gaussian(&h.x, &current.x, self.sigma)?;
            numerator += gaussian(&h.y, &current.y, self.sigma)?;
            numerator -= gaussian(&h.x, &current.y, self.sigma)?;
            numerator -= gaussian(&h.y, &current.x, self.sigma)?;
        }
        let mut norm_squared = 0.0;
        for i in neighbors {
            let left = &self.history[*i];
            for j in neighbors {
                let right = &self.history[*j];
                norm_squared += gaussian(&left.x, &right.x, self.sigma)?;
                norm_squared += gaussian(&left.y, &right.y, self.sigma)?;
                norm_squared -= gaussian(&left.x, &right.y, self.sigma)?;
                norm_squared -= gaussian(&left.y, &right.x, self.sigma)?;
            }
        }
        let k = neighbors.len() as f64;
        if norm_squared < -1e-10 * k * k { return Err(ScptError::InvalidPayoff); }
        let denominator = std::f64::consts::SQRT_2 * norm_squared.max(0.0).sqrt().max(self.eta * k);
        let payoff = numerator / denominator;
        if !payoff.is_finite() || payoff.abs() > 1.0 + 1e-8 { return Err(ScptError::InvalidPayoff); }
        Ok(payoff.clamp(-1.0, 1.0))
    }

    fn push(&mut self, obs: &PairedObservation, payoffs: Vec<f64>) {
        self.history.push(History {
            covariate: obs.covariate.clone(), x: obs.x.clone(), y: obs.y.clone(), payoffs,
        });
    }

    fn initial_trace(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        h.update(b"w(z)=1;knn-euclidean;equal-factor-mixture-v1");
        h.update(self.sigma.to_bits().to_be_bytes());
        h.update(self.eta.to_bits().to_be_bytes());
        h.update(self.alpha.to_bits().to_be_bytes());
        for a in &self.coefficients { h.update(a.to_bits().to_be_bytes()); }
        hex(&h.finalize())
    }
}

fn validate_obs(
    obs: &PairedObservation, cov_dim: Option<usize>, response_dim: Option<usize>,
) -> Result<(), ScptError> {
    if obs.covariate.is_empty() || obs.x.is_empty() || obs.x.len() != obs.y.len() {
        return Err(ScptError::InputDimensionMismatch);
    }
    if cov_dim.is_some_and(|d| d != obs.covariate.len())
        || response_dim.is_some_and(|d| d != obs.x.len()) {
        return Err(ScptError::InputDimensionMismatch);
    }
    if obs.covariate.iter().chain(&obs.x).chain(&obs.y).any(|v| !v.is_finite()) {
        return Err(ScptError::NonFiniteInput);
    }
    Ok(())
}

fn gaussian(x: &[f64], y: &[f64], sigma: f64) -> Result<f64, ScptError> {
    if x.is_empty() || x.len() != y.len() { return Err(ScptError::InputDimensionMismatch); }
    let v = (-sqdist(x, y) / (2.0 * sigma * sigma)).exp();
    if v.is_finite() && (0.0..=1.0).contains(&v) { Ok(v) } else { Err(ScptError::ArithmeticOverflow) }
}

fn sqdist(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a,b)| (a-b)*(a-b)).sum()
}

/// Median of pairwise distances among the pooled 2n warm-up responses.
pub fn median_heuristic_sigma(warmup: &[PairedObservation]) -> Result<f64, ScptError> {
    if warmup.len() < 2 { return Err(ScptError::EmptyWarmup); }
    let d = warmup[0].x.len();
    if d == 0 { return Err(ScptError::InputDimensionMismatch); }
    let mut points: Vec<&[f64]> = Vec::with_capacity(2 * warmup.len());
    for obs in warmup {
        if obs.x.len() != d || obs.y.len() != d ||
            obs.x.iter().chain(&obs.y).any(|v| !v.is_finite()) {
            return Err(ScptError::WarmupDimensionsMismatch);
        }
        points.push(&obs.x); points.push(&obs.y);
    }
    let mut distances = Vec::new();
    for i in 0..points.len() {
        for j in (i+1)..points.len() {
            let d = sqdist(points[i], points[j]).sqrt();
            if d.is_finite() && d > 0.0 { distances.push(d); }
        }
    }
    if distances.is_empty() { return Err(ScptError::InvalidBandwidth); }
    distances.sort_by(f64::total_cmp);
    let m = distances.len()/2;
    let median = if distances.len()%2 == 0 {
        0.5*(distances[m-1]+distances[m])
    } else { distances[m] };
    if median.is_finite() && median > 0.0 { Ok(median) } else { Err(ScptError::InvalidBandwidth) }
}

fn warmup_trace(prior: &str, warmup: &[PairedObservation], sigma: f64, eta: f64, alpha: f64) -> String {
    let mut h = Sha256::new();
    h.update(DOMAIN); h.update(prior.as_bytes());
    h.update(b"first-50-pooled-x-y-median-warmup-v1");
    h.update(sigma.to_bits().to_be_bytes()); h.update(eta.to_bits().to_be_bytes());
    h.update(alpha.to_bits().to_be_bytes());
    for obs in warmup {
        h.update(obs.sequence.to_be_bytes());
        for value in obs.covariate.iter().chain(&obs.x).chain(&obs.y) {
            h.update(value.to_bits().to_be_bytes());
        }
    }
    hex(&h.finalize())
}

fn extend_trace(
    prior: &str, sequence: u64, id: &str, configs: &[ScptConfigurationUpdate],
    factor: f64, log_e: f64,
) -> String {
    let mut h = Sha256::new();
    h.update(DOMAIN); h.update(prior.as_bytes()); h.update(sequence.to_be_bytes());
    h.update((id.len() as u64).to_be_bytes()); h.update(id.as_bytes());
    for c in configs {
        h.update(c.neighborhood_coefficient.to_bits().to_be_bytes());
        h.update((c.neighbor_count as u64).to_be_bytes());
        h.update(c.betting_fraction.to_bits().to_be_bytes());
        h.update(c.payoff.to_bits().to_be_bytes()); h.update(c.factor.to_bits().to_be_bytes());
    }
    h.update(factor.to_bits().to_be_bytes()); h.update(log_e.to_bits().to_be_bytes());
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8;16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len()*2);
    for b in bytes { out.push(DIGITS[(b>>4) as usize] as char); out.push(DIGITS[(b&15) as usize] as char); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn warmup() -> Vec<PairedObservation> {
        (0..50u64).map(|i| PairedObservation {
            sequence:i+1,
            covariate:vec![(i as f64/7.0).sin(),(i as f64/11.0).cos()],
            x:vec![(i as f64/5.0).sin()], y:vec![(i as f64/9.0).cos()],
        }).collect()
    }

    #[test]
    fn three_scale_mixture_uses_past_knn_counts() {
        let mut m = ScptReferenceMonitor::from_warmup(&warmup(),1e-6,0.05).unwrap();
        let u=m.observe_pair(51,"first",&[0.25,-0.5],&[0.1],&[0.2]).unwrap();
        assert_eq!(u.configurations.iter().map(|c| c.neighbor_count).collect::<Vec<_>>(),vec![15,29,43]);
        let avg=u.configurations.iter().map(|c|c.factor).sum::<f64>()/3.0;
        assert!((avg-u.mixture_factor).abs()<1e-12);
        assert!((u.log_e_value-avg.ln()).abs()<1e-12);
    }

    #[test]
    fn equal_current_responses_have_zero_payoffs_and_unit_wealth_factor() {
        let mut m=ScptReferenceMonitor::from_warmup(&warmup(),1e-6,0.05).unwrap();
        let u=m.observe_pair(51,"identical",&[0.25,-0.5],&[0.375],&[0.375]).unwrap();
        assert!(u.configurations.iter().all(|c|c.payoff.abs()<1e-12 && (c.factor-1.0).abs()<1e-12));
        assert!((u.mixture_factor-1.0).abs()<1e-12);
        assert!(u.log_e_value.abs()<1e-12);
    }

    #[test]
    fn betting_fractions_do_not_use_current_response_pair() {
        let baseline=ScptReferenceMonitor::from_warmup(&warmup(),1e-6,0.05).unwrap();
        let mut left=baseline.clone(); let mut right=baseline;
        let a=left.observe_pair(51,"left",&[0.25,-0.5],&[0.1],&[0.2]).unwrap();
        let b=right.observe_pair(51,"right",&[0.25,-0.5],&[2.0],&[-1.0]).unwrap();
        assert_eq!(
            a.configurations.iter().map(|c|c.betting_fraction.to_bits()).collect::<Vec<_>>(),
            b.configurations.iter().map(|c|c.betting_fraction.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn sequence_and_replay_fail_without_partial_commit() {
        let mut m=ScptReferenceMonitor::from_warmup(&warmup(),1e-6,0.05).unwrap();
        assert_eq!(m.observe_pair(52,"late",&[0.0,0.0],&[0.0],&[0.0]),Err(ScptError::SequenceMismatch));
        assert_eq!(m.observations_seen(),50); assert_eq!(m.log_e_value(),0.0);
        m.observe_pair(51,"once",&[0.0,0.0],&[0.0],&[0.0]).unwrap();
        assert_eq!(m.observe_pair(52,"once",&[0.0,0.0],&[0.0],&[0.0]),Err(ScptError::DuplicatePairId));
        assert_eq!(m.observations_seen(),51);
    }
}
