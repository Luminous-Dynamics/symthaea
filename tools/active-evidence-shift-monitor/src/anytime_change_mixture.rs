//! Anytime-valid change-point mixture for Bernoulli streams under a composite conditional null.
//!
//! Under the null, each predictable conditional event rate p_t is in [0.48, 0.52]. For an
//! upward alternative q > 0.52, use the Bernoulli likelihood ratio against p=0.52; for a
//! downward alternative q < 0.48, use the likelihood ratio against p=0.48. Each increment has
//! conditional expectation at most one under the composite null.
//!
//! For each (direction, q, hazard) component, mix those likelihood ratios over every possible
//! change start k using pi_k = h(1-h)^(k-1). The component keeps its dormant future-start mass
//! (1-h)^t, so it starts at e=1 and remains a nonnegative supermartingale. A fixed mixture over
//! directions, alternatives and hazards is therefore an e-process too. Ville's inequality gives
//! P(sup_t E_t >= 1/alpha) <= alpha under the stated conditional null.
//!
//! This is a discrete Bernoulli reference implementation, not a claim of a new general e-detector
//! theory. All arithmetic is in log space so long streams cannot overflow.

pub const NULL_LOWER: f64 = 0.48;
pub const NULL_UPPER: f64 = 0.52;
pub const UPWARD_ALTERNATIVES: [f64; 5] = [0.55, 0.60, 0.70, 0.80, 0.90];
pub const DOWNWARD_ALTERNATIVES: [f64; 5] = [0.45, 0.40, 0.30, 0.20, 0.10];
pub const CHANGE_HAZARDS: [f64; 3] = [0.04, 0.01, 0.0025];

const COMPONENT_COUNT: usize =
    CHANGE_HAZARDS.len() * (UPWARD_ALTERNATIVES.len() + DOWNWARD_ALTERNATIVES.len());

#[derive(Clone, Debug)]
struct Component {
    log_active: f64,
    log_tail: f64,
    log_hazard: f64,
    log_survival: f64,
    log_lr_success: f64,
    log_lr_failure: f64,
}

impl Component {
    fn new(hazard: f64, alternative: f64, null_edge: f64) -> Self {
        debug_assert!((0.0..1.0).contains(&hazard));
        debug_assert!((0.0..1.0).contains(&alternative));
        debug_assert!((0.0..1.0).contains(&null_edge));
        Self {
            log_active: f64::NEG_INFINITY,
            log_tail: 0.0,
            log_hazard: hazard.ln(),
            log_survival: (-hazard).ln_1p(),
            log_lr_success: (alternative / null_edge).ln(),
            log_lr_failure: ((1.0 - alternative) / (1.0 - null_edge)).ln(),
        }
    }

    fn observe(&mut self, step: u64, event: bool) {
        // pi_t = h (1-h)^(t-1), and dormant future-start mass is (1-h)^t.
        let log_start_weight = self.log_hazard + (step - 1) as f64 * self.log_survival;
        let log_prior_active = log_add_exp(self.log_active, log_start_weight);
        let log_lr = if event {
            self.log_lr_success
        } else {
            self.log_lr_failure
        };
        self.log_active = log_lr + log_prior_active;
        self.log_tail = step as f64 * self.log_survival;
    }

    fn log_e_value(&self) -> f64 {
        log_add_exp(self.log_active, self.log_tail)
    }
}

/// A fixed equal-weight portfolio over direction × alternative rate × change-time scale.
#[derive(Clone, Debug)]
pub struct AnytimeChangeMixture {
    steps: u64,
    components: Vec<Component>,
}

impl Default for AnytimeChangeMixture {
    fn default() -> Self {
        Self::new()
    }
}

impl AnytimeChangeMixture {
    pub fn new() -> Self {
        let mut components = Vec::with_capacity(COMPONENT_COUNT);
        for hazard in CHANGE_HAZARDS {
            for alternative in UPWARD_ALTERNATIVES {
                components.push(Component::new(hazard, alternative, NULL_UPPER));
            }
            for alternative in DOWNWARD_ALTERNATIVES {
                components.push(Component::new(hazard, alternative, NULL_LOWER));
            }
        }
        debug_assert_eq!(components.len(), COMPONENT_COUNT);
        Self { steps: 0, components }
    }

    /// Update the monitor with one Bernoulli event and return log portfolio e-value.
    pub fn observe(&mut self, event: bool) -> f64 {
        self.steps = self
            .steps
            .checked_add(1)
            .expect("anytime-mixture step counter exhausted; fail closed rather than wrap");
        for component in &mut self.components {
            component.observe(self.steps, event);
        }
        self.log_e_value()
    }

    /// Natural log of the equal-weight e-process portfolio, initialized at log(1)=0.
    pub fn log_e_value(&self) -> f64 {
        let mut maximum = f64::NEG_INFINITY;
        for component in &self.components {
            maximum = maximum.max(component.log_e_value());
        }
        let scaled_sum: f64 = self
            .components
            .iter()
            .map(|component| (component.log_e_value() - maximum).exp())
            .sum();
        maximum + scaled_sum.ln() - (self.components.len() as f64).ln()
    }

    /// Whether the time-uniform e-process crossing threshold has been reached.
    pub fn crossed(&self, alpha: f64) -> bool {
        assert!(alpha.is_finite() && alpha > 0.0 && alpha < 1.0);
        self.log_e_value() >= -alpha.ln()
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

/// Counter-based SplitMix64 fixture generator. It is deterministic and intended only for
/// reproducible synthetic traces, never for security or production randomization.
pub fn probe_bernoulli(seed: u64, step: u64, probability: f64) -> bool {
    assert!(probability.is_finite() && (0.0..=1.0).contains(&probability));
    if probability == 0.0 {
        return false;
    }
    if probability == 1.0 {
        return true;
    }
    let mut value = seed ^ step.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let cutoff = (probability * u64::MAX as f64) as u64;
    value <= cutoff
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ALPHA: f64 = 0.05 / 12.0;

    fn assert_one_step_supermartingale_bound(state: &AnytimeChangeMixture, probability: f64) {
        let current = state.log_e_value().exp();
        let mut zero = state.clone();
        let zero_next = zero.observe(false).exp();
        let mut one = state.clone();
        let one_next = one.observe(true).exp();
        let conditional_expectation = (1.0 - probability) * zero_next + probability * one_next;
        assert!(
            conditional_expectation <= current + 1.0e-10 * current.max(1.0),
            "conditional expectation {conditional_expectation} exceeded current e-value {current} at p={probability}"
        );
    }

    #[test]
    fn initializes_at_one_and_mixes_all_prespecified_components() {
        let monitor = AnytimeChangeMixture::new();
        assert_eq!(monitor.steps(), 0);
        assert_eq!(monitor.component_count(), 30);
        assert!(monitor.log_e_value().abs() <= 1.0e-12);
    }

    #[test]
    fn conditional_supermartingale_bound_holds_on_exhaustive_short_histories() {
        fn recurse(state: AnytimeChangeMixture, depth: usize, max_depth: usize) {
            for probability in [NULL_LOWER, 0.49, 0.50, 0.51, NULL_UPPER] {
                assert_one_step_supermartingale_bound(&state, probability);
            }
            if depth == max_depth {
                return;
            }
            let mut zero = state.clone();
            zero.observe(false);
            recurse(zero, depth + 1, max_depth);
            let mut one = state;
            one.observe(true);
            recurse(one, depth + 1, max_depth);
        }

        recurse(AnytimeChangeMixture::new(), 0, 10);
    }

    #[test]
    fn detects_persistent_extreme_upward_and_downward_traces() {
        for event in [true, false] {
            let mut monitor = AnytimeChangeMixture::new();
            let first_crossing = (1..=200)
                .find(|_| monitor.observe(event) >= -TEST_ALPHA.ln());
            assert!(
                first_crossing.is_some(),
                "persistent event={event} did not cross the prespecified threshold"
            );
        }
    }

    #[test]
    fn dormant_start_time_mass_remains_normalized() {
        for hazard in CHANGE_HAZARDS {
            let mut cumulative = 0.0;
            for step in 1..=10_000 {
                cumulative += hazard * (1.0 - hazard).powi((step - 1) as i32);
                let tail = (1.0 - hazard).powi(step as i32);
                assert!((cumulative + tail - 1.0).abs() <= 1.0e-10);
            }
        }
    }

    #[test]
    fn deterministic_probe_stream_is_reproducible_and_rate_sensitive() {
        let first: Vec<bool> = (1..=128)
            .map(|step| probe_bernoulli(0xA11C_E5EED, step, 0.60))
            .collect();
        let second: Vec<bool> = (1..=128)
            .map(|step| probe_bernoulli(0xA11C_E5EED, step, 0.60))
            .collect();
        assert_eq!(first, second);
        assert!(first.iter().filter(|event| **event).count() > 45);
        assert!(first.iter().filter(|event| **event).count() < 105);
    }
}
