//! Anytime-valid bounded-risk e-process for sequential runtime monitoring.
//!
//! Under the null hypothesis E[loss_t | history before t] <= risk_bound, the
//! factor 1 + bet * (loss_t - risk_bound) is a nonnegative supermartingale
//! factor when bet is predictable and 0 <= bet * risk_bound < 1. Ville's
//! inequality bounds the probability that the process ever crosses 1/delta.
//! For N parallel monitors, this type uses N/delta as each threshold so the
//! union bound allocates the total false-alarm budget across them.
//!
//! This is a statistical risk monitor, not an OOD detector or safety certificate.
//! Its validity depends on the conditional-mean null assumption; nominal
//! calibration alone does not establish that assumption under arbitrary drift.

/// Log threshold for episode-reset e-processes with a countable lifetime budget.
///
/// Episode j receives delta_j = total_delta / (K * j * (j + 1)); summing over
/// j >= 1 and K monitors allocates at most total_delta. The episode index must
/// be fixed before outcomes are observed.
pub fn alpha_spending_log_threshold(
    total_delta: f64,
    simultaneous_monitors: usize,
    episode_index: u64,
) -> Option<f64> {
    if !total_delta.is_finite()
        || total_delta <= 0.0
        || total_delta >= 1.0
        || simultaneous_monitors == 0
        || episode_index == 0
    {
        return None;
    }
    let next_episode = episode_index.checked_add(1)?;
    let log_threshold = (simultaneous_monitors as f64).ln()
        - total_delta.ln()
        + (episode_index as f64).ln()
        + (next_episode as f64).ln();
    if !log_threshold.is_finite() || !log_threshold.exp().is_finite() {
        return None;
    }
    Some(log_threshold)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedRiskEProcess {
    risk_bound: f64,
    bet: f64,
    total_delta: f64,
    simultaneous_monitors: usize,
    episode_budget_index: Option<u64>,
    log_alarm_threshold: f64,
    log_e_value: f64,
    alarmed: bool,
}

impl BoundedRiskEProcess {
    /// Construct a fixed-bet e-process.
    ///
    /// risk_bound is the tolerated upper bound on the conditional expected
    /// bounded loss under the null. bet must be predictable with respect to
    /// the current loss; this implementation uses a fixed bet. total_delta
    /// is the overall false-alarm budget allocated by a union bound across
    /// simultaneous_monitors parallel processes.
    pub fn new(
        risk_bound: f64,
        bet: f64,
        total_delta: f64,
        simultaneous_monitors: usize,
    ) -> Option<Self> {
        if !risk_bound.is_finite()
            || risk_bound <= 0.0
            || risk_bound >= 1.0
            || !bet.is_finite()
            || bet <= 0.0
            || bet * risk_bound >= 1.0
            || !total_delta.is_finite()
            || total_delta <= 0.0
            || total_delta >= 1.0
            || simultaneous_monitors == 0
        {
            return None;
        }

        let e_value_threshold = simultaneous_monitors as f64 / total_delta;
        if !e_value_threshold.is_finite() {
            return None;
        }

        Some(Self {
            risk_bound,
            bet,
            total_delta,
            simultaneous_monitors,
            episode_budget_index: None,
            log_alarm_threshold: e_value_threshold.ln(),
            log_e_value: 0.0,
            alarmed: false,
        })
    }

    /// Use a predeclared summable per-episode allocation so resets do not
    /// silently reuse the same lifetime false-alarm budget.
    pub fn new_with_alpha_spending(
        risk_bound: f64,
        bet: f64,
        total_delta: f64,
        simultaneous_monitors: usize,
        episode_index: u64,
    ) -> Option<Self> {
        let mut process = Self::new(risk_bound, bet, total_delta, simultaneous_monitors)?;
        process.log_alarm_threshold =
            alpha_spending_log_threshold(total_delta, simultaneous_monitors, episode_index)?;
        process.episode_budget_index = Some(episode_index);
        Some(process)
    }

    pub const fn episode_budget_index(self) -> Option<u64> {
        self.episode_budget_index
    }

    /// Encode the live monitor state as a versioned canonical text checkpoint.
    ///
    /// The checksum detects accidental corruption and truncation; it is not an
    /// authentication mechanism. Use an authenticated, access-controlled store
    /// when tamper resistance is required.
    pub fn encode_checkpoint(self) -> String {
        let episode_index = self
            .episode_budget_index
            .map_or_else(|| "none".to_string(), |index| index.to_string());
        let payload = format!(
            "{MIXTURE_CHECKPOINT_MAGIC}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
            checkpoint_float("risk_bound", self.risk_bound),
            checkpoint_float("bet_fraction_1", self.bet_fractions[0]),
            checkpoint_float("bet_fraction_2", self.bet_fractions[1]),
            checkpoint_float("bet_fraction_3", self.bet_fractions[2]),
            checkpoint_float("total_delta", self.total_delta),
            format!("simultaneous_monitors={}", self.simultaneous_monitors),
            format!("episode_budget_index={episode_index}"),
            checkpoint_float("log_alarm_threshold", self.log_alarm_threshold),
            checkpoint_float("component_log_e_1", self.component_log_e_values[0]),
            checkpoint_float("component_log_e_2", self.component_log_e_values[1]),
            checkpoint_float("component_log_e_3", self.component_log_e_values[2]),
            checkpoint_float("log_e_value", self.log_e_value),
            format!("alarmed={}", u8::from(self.alarmed)),
        );
        let checksum = checkpoint_checksum(payload.as_bytes());
        format!("{payload}checksum={checksum:016x}\n")
    }

    /// Restore a mixture monitor only if the version, checksum, parameters,
    /// threshold, component evidence and mixture evidence are consistent.
    /// An incompatible or corrupted checkpoint fails closed.
    pub fn from_checkpoint(encoded: &str) -> Option<Self> {
        if !encoded.ends_with('\n') || encoded.contains('\r') {
            return None;
        }
        let lines: Vec<&str> = encoded.strip_suffix('\n')?.split('\n').collect();
        if lines.len() != 15 || lines[0] != MIXTURE_CHECKPOINT_MAGIC {
            return None;
        }
        let checksum_line = lines[14].strip_prefix("checksum=")?;
        if checksum_line.len() != 16 {
            return None;
        }
        let expected_checksum = u64::from_str_radix(checksum_line, 16).ok()?;
        let payload = format!("{}\n", lines[..14].join("\n"));
        if checkpoint_checksum(payload.as_bytes()) != expected_checksum {
            return None;
        }

        let risk_bound = parse_checkpoint_float(lines[1], "risk_bound")?;
        let bet_fractions = [
            parse_checkpoint_float(lines[2], "bet_fraction_1")?,
            parse_checkpoint_float(lines[3], "bet_fraction_2")?,
            parse_checkpoint_float(lines[4], "bet_fraction_3")?,
        ];
        let total_delta = parse_checkpoint_float(lines[5], "total_delta")?;
        let simultaneous_monitors = lines[6]
            .strip_prefix("simultaneous_monitors=")?
            .parse::<usize>()
            .ok()?;
        let episode_index_text = lines[7].strip_prefix("episode_budget_index=")?;
        let episode_budget_index = if episode_index_text == "none" {
            None
        } else {
            Some(episode_index_text.parse::<u64>().ok()?)
        };
        let saved_threshold = parse_checkpoint_float(lines[8], "log_alarm_threshold")?;
        let component_log_e_values = [
            parse_checkpoint_float(lines[9], "component_log_e_1")?,
            parse_checkpoint_float(lines[10], "component_log_e_2")?,
            parse_checkpoint_float(lines[11], "component_log_e_3")?,
        ];
        let saved_log_e_value = parse_checkpoint_float(lines[12], "log_e_value")?;
        let alarmed = match lines[13].strip_prefix("alarmed=")? {
            "0" => false,
            "1" => true,
            _ => return None,
        };

        let mut restored = if let Some(index) = episode_budget_index {
            Self::new_with_alpha_spending(
                risk_bound,
                bet_fractions,
                total_delta,
                simultaneous_monitors,
                index,
            )?
        } else {
            Self::new(risk_bound, bet_fractions, total_delta, simultaneous_monitors)?
        };
        if !checkpoint_close(saved_threshold, restored.log_alarm_threshold) {
            return None;
        }

        let maximum = component_log_e_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_exp = component_log_e_values
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f64>()
            / component_log_e_values.len() as f64;
        let reconstructed_log_e = maximum + mean_exp.ln();
        if !reconstructed_log_e.is_finite()
            || !checkpoint_close(saved_log_e_value, reconstructed_log_e)
            || (!alarmed && saved_log_e_value >= saved_threshold)
        {
            return None;
        }

        restored.component_log_e_values = component_log_e_values;
        restored.log_e_value = saved_log_e_value;
        restored.alarmed = alarmed;
        Some(restored)
    }

    pub const fn risk_bound(self) -> f64 {
        self.risk_bound
    }

    pub const fn bet(self) -> f64 {
        self.bet
    }

    pub const fn total_delta(self) -> f64 {
        self.total_delta
    }

    pub const fn simultaneous_monitors(self) -> usize {
        self.simultaneous_monitors
    }

    /// E-value threshold after Bonferroni allocation of the global delta.
    pub fn e_value_alarm_threshold(self) -> f64 {
        self.log_alarm_threshold.exp()
    }

    /// Log of the E-value threshold, used for stable comparisons in log space.
    pub const fn log_alarm_threshold(self) -> f64 {
        self.log_alarm_threshold
    }

    pub const fn log_e_value(self) -> f64 {
        self.log_e_value
    }

    /// Exponentiated e-value for reporting. The monitor itself works in log
    /// space so very long streams do not silently underflow or overflow.
    pub fn e_value(self) -> f64 {
        self.log_e_value.exp()
    }

    pub const fn alarmed(self) -> bool {
        self.alarmed
    }

    /// Observe a bounded loss in [0, 1]. Returns None on invalid input without
    /// mutating the process. Overflow is treated as decisive evidence and latches
    /// an alarm, rather than wrapping or returning a misleading finite value.
    pub fn observe(&mut self, loss: f64) -> Option<bool> {
        if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
            return None;
        }

        let factor = 1.0 + self.bet * (loss - self.risk_bound);
        if !factor.is_finite() || factor <= 0.0 {
            return None;
        }

        let next_log_value = self.log_e_value + factor.ln();
        if next_log_value.is_nan() {
            return None;
        }
        if next_log_value == f64::INFINITY {
            self.log_e_value = f64::MAX;
            self.alarmed = true;
            return Some(true);
        }
        if next_log_value == f64::NEG_INFINITY {
            self.log_e_value = -f64::MAX;
            return Some(self.alarmed);
        }

        self.log_e_value = next_log_value;
        self.alarmed |= self.log_e_value >= self.log_alarm_threshold();
        Some(self.alarmed)
    }

    /// Reset the evidence stream within the same preallocated episode budget.
    ///
    /// A reset does not grant a new false-alarm budget. For a new episode, build
    /// a new monitor with a fresh alpha-spending index; do not call reset to
    /// restart a detector while reusing the same index.
    pub fn reset(&mut self) {
        self.log_e_value = 0.0;
        self.alarmed = false;
    }
}

/// Fixed-grid mixture of bounded-risk e-processes for unknown shift magnitude.
///
/// Each component uses a predeclared predictable bet fraction in (0, 1), with
/// bet = fraction / risk_bound. The arithmetic mixture of nonnegative
/// supermartingales is itself a nonnegative supermartingale under the same
/// conditional-risk null. This avoids selecting one bet using evaluation data.
const MIXTURE_CHECKPOINT_MAGIC: &str = "SYMTHEA_MIXTURE_EPROCESS_CHECKPOINT_V1";

/// FNV-1a detects accidental checkpoint corruption; it is not a MAC/signature
/// and does not protect against an attacker who can rewrite the checkpoint.
fn checkpoint_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn checkpoint_float(name: &str, value: f64) -> String {
    format!("{name}={:016x}", value.to_bits())
}

fn parse_checkpoint_float(line: &str, name: &str) -> Option<f64> {
    let encoded = line.strip_prefix(&format!("{name}="))?;
    if encoded.len() != 16 {
        return None;
    }
    let bits = u64::from_str_radix(encoded, 16).ok()?;
    let value = f64::from_bits(bits);
    value.is_finite().then_some(value)
}

fn checkpoint_close(left: f64, right: f64) -> bool {
    if left.to_bits() == right.to_bits() {
        return true;
    }
    (left - right).abs() <= 1e-10 * left.abs().max(right.abs()).max(1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixtureBoundedRiskEProcess {
    risk_bound: f64,
    bet_fractions: [f64; 3],
    total_delta: f64,
    simultaneous_monitors: usize,
    episode_budget_index: Option<u64>,
    log_alarm_threshold: f64,
    component_log_e_values: [f64; 3],
    log_e_value: f64,
    alarmed: bool,
}

impl MixtureBoundedRiskEProcess {
    pub fn new(
        risk_bound: f64,
        bet_fractions: [f64; 3],
        total_delta: f64,
        simultaneous_monitors: usize,
    ) -> Option<Self> {
        if !risk_bound.is_finite()
            || risk_bound <= 0.0
            || risk_bound >= 1.0
            || bet_fractions
                .iter()
                .any(|fraction| !fraction.is_finite() || *fraction <= 0.0 || *fraction >= 1.0)
            || !total_delta.is_finite()
            || total_delta <= 0.0
            || total_delta >= 1.0
            || simultaneous_monitors == 0
        {
            return None;
        }

        if bet_fractions
            .iter()
            .any(|fraction| !(fraction / risk_bound).is_finite())
            || !(simultaneous_monitors as f64 / total_delta).is_finite()
        {
            return None;
        }

        Some(Self {
            risk_bound,
            bet_fractions,
            total_delta,
            simultaneous_monitors,
            episode_budget_index: None,
            log_alarm_threshold: (simultaneous_monitors as f64 / total_delta).ln(),
            component_log_e_values: [0.0; 3],
            log_e_value: 0.0,
            alarmed: false,
        })
    }

    /// Use summable alpha spending for repeated, episode-reset mixture monitors.
    pub fn new_with_alpha_spending(
        risk_bound: f64,
        bet_fractions: [f64; 3],
        total_delta: f64,
        simultaneous_monitors: usize,
        episode_index: u64,
    ) -> Option<Self> {
        let mut process = Self::new(
            risk_bound,
            bet_fractions,
            total_delta,
            simultaneous_monitors,
        )?;
        process.log_alarm_threshold =
            alpha_spending_log_threshold(total_delta, simultaneous_monitors, episode_index)?;
        process.episode_budget_index = Some(episode_index);
        Some(process)
    }

    pub const fn episode_budget_index(self) -> Option<u64> {
        self.episode_budget_index
    }

    pub const fn risk_bound(self) -> f64 {
        self.risk_bound
    }

    pub const fn bet_fractions(self) -> [f64; 3] {
        self.bet_fractions
    }

    pub fn bet_values(self) -> [f64; 3] {
        self.bet_fractions.map(|fraction| fraction / self.risk_bound)
    }

    pub const fn total_delta(self) -> f64 {
        self.total_delta
    }

    pub const fn simultaneous_monitors(self) -> usize {
        self.simultaneous_monitors
    }

    pub const fn log_alarm_threshold(self) -> f64 {
        self.log_alarm_threshold
    }

    pub const fn component_log_e_values(self) -> [f64; 3] {
        self.component_log_e_values
    }

    pub const fn log_e_value(self) -> f64 {
        self.log_e_value
    }

    pub fn e_value(self) -> f64 {
        self.log_e_value.exp()
    }

    pub const fn alarmed(self) -> bool {
        self.alarmed
    }

    pub fn observe(&mut self, loss: f64) -> Option<bool> {
        if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
            return None;
        }

        let mut next_components = self.component_log_e_values;
        for (index, fraction) in self.bet_fractions.iter().copied().enumerate() {
            let bet = fraction / self.risk_bound;
            let factor = 1.0 + bet * (loss - self.risk_bound);
            if !factor.is_finite() || factor <= 0.0 {
                return None;
            }
            let candidate = next_components[index] + factor.ln();
            if candidate.is_nan() {
                return None;
            }
            if candidate == f64::INFINITY {
                next_components[index] = f64::MAX;
                self.component_log_e_values = next_components;
                self.log_e_value = f64::MAX;
                self.alarmed = true;
                return Some(true);
            }
            next_components[index] = if candidate == f64::NEG_INFINITY {
                -f64::MAX
            } else {
                candidate
            };
        }
        self.component_log_e_values = next_components;

        let max_log_value = self
            .component_log_e_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_exp = self
            .component_log_e_values
            .iter()
            .map(|value| (*value - max_log_value).exp())
            .sum::<f64>()
            / self.component_log_e_values.len() as f64;
        let next_log_value = max_log_value + mean_exp.ln();

        if !next_log_value.is_finite() {
            self.log_e_value = f64::MAX;
            self.alarmed = true;
            return Some(true);
        }

        self.log_e_value = next_log_value;
        self.alarmed |= self.log_e_value >= self.log_alarm_threshold();
        Some(self.alarmed)
    }

    /// Reset evidence within the same episode budget only. A fresh episode must
    /// construct a new monitor with its newly allocated alpha-spending index.
    pub fn reset(&mut self) {
        self.component_log_e_values = [0.0; 3];
        self.log_e_value = 0.0;
        self.alarmed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_of_bets_allocates_delta_and_rejects_invalid_components() {
        assert!(MixtureBoundedRiskEProcess::new(0.15, [0.25, 0.5, 1.0], 0.05, 5).is_none());
        assert!(MixtureBoundedRiskEProcess::new(0.15, [0.25, 0.0, 0.75], 0.05, 5).is_none());
        let process = MixtureBoundedRiskEProcess::new(0.15, [0.25, 0.5, 0.75], 0.05, 5)
            .expect("valid bet mixture");
        assert!((process.log_alarm_threshold().exp() - 100.0).abs() < 1e-10);
        assert_eq!(process.bet_fractions(), [0.25, 0.5, 0.75]);
        assert_eq!(process.bet_values(), [
            0.25 / 0.15,
            0.5 / 0.15,
            0.75 / 0.15,
        ]);
    }

    #[test]
    fn mixture_preserves_nominal_evidence_and_detects_gradual_excess() {
        let mut nominal = MixtureBoundedRiskEProcess::new(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .expect("valid mixture");
        for loss in [0.06, 0.08, 0.10, 0.12, 0.09, 0.13, 0.07, 0.10] {
            assert_eq!(nominal.observe(loss), Some(false));
        }
        assert!(nominal.log_e_value() < nominal.log_alarm_threshold());

        let mut gradual = MixtureBoundedRiskEProcess::new(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .expect("valid mixture");
        let mut first_alarm = None;
        for (step, score) in [0.009, 0.012, 0.015, 0.018, 0.021, 0.025, 0.028, 0.031]
            .into_iter()
            .enumerate()
        {
            if gradual.observe((score / 0.05).min(1.0)) == Some(true) {
                first_alarm.get_or_insert(step);
            }
        }
        assert!(first_alarm.is_some());
        assert!(gradual.log_e_value() >= gradual.log_alarm_threshold());
    }

    #[test]
    fn mixture_logsumexp_matches_independent_component_recurrence() {
        let risk_bound = 0.15;
        let fractions = [0.25, 0.5, 0.75];
        let mut process = MixtureBoundedRiskEProcess::new(
            risk_bound,
            fractions,
            0.05,
            5,
        )
        .expect("valid mixture");

        let losses = [0.12, 0.20, 0.24, 0.08, 0.31];
        let mut expected_components = [0.0_f64; 3];
        for loss in losses {
            assert!(process.observe(loss).is_some());
            for (index, fraction) in fractions.iter().copied().enumerate() {
                let factor = 1.0 + (fraction / risk_bound) * (loss - risk_bound);
                expected_components[index] += factor.ln();
            }
            for (actual, expected) in process
                .component_log_e_values()
                .iter()
                .zip(expected_components.iter())
            {
                assert!((actual - expected).abs() < 1e-12);
            }
            let max_component = expected_components
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let expected_mix = max_component
                + ((expected_components[0] - max_component).exp()
                    + (expected_components[1] - max_component).exp()
                    + (expected_components[2] - max_component).exp())
                    .ln()
                - 3.0_f64.ln();
            assert!((process.log_e_value() - expected_mix).abs() < 1e-12);
        }
    }

    #[test]
    fn checkpoint_resume_is_bitwise_equivalent_across_alpha_spent_episode() {
        let losses = [
            0.05, 0.10, 0.17, 0.22, 0.07, 0.31, 0.42, 0.12, 0.15, 0.28, 0.06,
        ];
        let split = 5;
        let mut uninterrupted = MixtureBoundedRiskEProcess::new_with_alpha_spending(
            0.15,
            [0.25, 0.50, 0.75],
            0.05,
            20,
            37,
        )
        .expect("valid alpha-spent stream");
        for loss in &losses[..split] {
            assert!(uninterrupted.observe(*loss).is_some());
        }

        let encoded = uninterrupted.encode_checkpoint();
        let mut resumed =
            MixtureBoundedRiskEProcess::from_checkpoint(&encoded).expect("valid checkpoint");
        assert_eq!(resumed.episode_budget_index(), Some(37));
        assert_eq!(
            resumed.component_log_e_values(),
            uninterrupted.component_log_e_values()
        );
        assert_eq!(
            resumed.log_e_value().to_bits(),
            uninterrupted.log_e_value().to_bits()
        );
        assert_eq!(resumed.alarmed(), uninterrupted.alarmed());

        for loss in &losses[split..] {
            assert_eq!(resumed.observe(*loss), uninterrupted.observe(*loss));
            assert_eq!(
                resumed.component_log_e_values(),
                uninterrupted.component_log_e_values()
            );
            assert_eq!(
                resumed.log_e_value().to_bits(),
                uninterrupted.log_e_value().to_bits()
            );
            assert_eq!(resumed.alarmed(), uninterrupted.alarmed());
        }
    }

    #[test]
    fn checkpoint_restore_fails_closed_on_corruption_truncation_and_wrong_version() {
        let mut monitor = MixtureBoundedRiskEProcess::new_with_alpha_spending(
            0.15,
            [0.25, 0.50, 0.75],
            0.05,
            20,
            4,
        )
        .expect("valid stream");
        for loss in [0.12, 0.18, 0.21] {
            assert!(monitor.observe(loss).is_some());
        }
        let encoded = monitor.encode_checkpoint();
        assert!(MixtureBoundedRiskEProcess::from_checkpoint(&encoded).is_some());
        assert!(MixtureBoundedRiskEProcess::from_checkpoint(
            &encoded[..encoded.len() - 4]
        )
        .is_none());

        let mut corrupted = encoded.clone();
        let position = corrupted
            .find("component_log_e_2=")
            .expect("checkpoint field exists")
            + "component_log_e_2=".len();
        let replacement = if corrupted.as_bytes()[position] == b'0' {
            '1'
        } else {
            '0'
        };
        corrupted.replace_range(position..position + 1, &replacement.to_string());
        assert!(MixtureBoundedRiskEProcess::from_checkpoint(&corrupted).is_none());

        let mut wrong_version = encoded;
        wrong_version.replace_range(0..MIXTURE_CHECKPOINT_MAGIC.len(), "BROKEN_CHECKPOINT_V1");
        assert!(MixtureBoundedRiskEProcess::from_checkpoint(&wrong_version).is_none());
    }

    #[test]
    fn checkpoint_preserves_latched_alarm_and_alpha_budget_boundary() {
        let mut monitor = MixtureBoundedRiskEProcess::new_with_alpha_spending(
            0.15,
            [0.25, 0.50, 0.75],
            0.05,
            1,
            1,
        )
        .expect("valid stream");
        for _ in 0..12 {
            assert!(monitor.observe(0.85).is_some());
        }
        assert!(monitor.alarmed());
        let encoded = monitor.encode_checkpoint();
        let restored =
            MixtureBoundedRiskEProcess::from_checkpoint(&encoded).expect("valid latched state");
        assert!(restored.alarmed());
        assert_eq!(
            restored.log_alarm_threshold().to_bits(),
            monitor.log_alarm_threshold().to_bits()
        );
        assert_eq!(
            restored.log_e_value().to_bits(),
            monitor.log_e_value().to_bits()
        );
    }

    #[test]
    fn mixture_rejects_invalid_loss_without_mutating_state_and_can_reset() {
        let mut process = MixtureBoundedRiskEProcess::new(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .expect("valid mixture");
        let initial = process.component_log_e_values();
        assert!(process.observe(f64::NAN).is_none());
        assert_eq!(process.component_log_e_values(), initial);

        for _ in 0..8 {
            assert!(process.observe(0.65).is_some());
        }
        assert!(process.alarmed());
        process.reset();
        assert_eq!(process.component_log_e_values(), [0.0; 3]);
        assert_eq!(process.log_e_value(), 0.0);
        assert!(!process.alarmed());
    }

    #[test]
    fn alpha_spending_thresholds_allocate_a_lifetime_false_alarm_budget() {
        let first = alpha_spending_log_threshold(0.05, 5, 1).expect("first budget");
        let second = alpha_spending_log_threshold(0.05, 5, 2).expect("second budget");
        let thousand = alpha_spending_log_threshold(0.05, 5, 1_000).expect("late budget");
        assert!((first.exp() - 200.0).abs() < 1e-10);
        assert!((second.exp() - 600.0).abs() < 1e-10);
        assert!(second > first);
        assert!(thousand > second);
        assert!(alpha_spending_log_threshold(0.05, 5, 0).is_none());
        assert!(alpha_spending_log_threshold(0.05, 5, u64::MAX).is_none());

        let allocated: f64 = (1..=100_000)
            .map(|episode| 0.05 / (5.0 * episode as f64 * (episode + 1) as f64))
            .sum();
        assert!(allocated <= 0.05 / 5.0);
        assert!(allocated > 0.0098);
    }

    #[test]
    fn alpha_spending_is_monotone_for_fixed_bet_and_mixture_monitors() {
        let fixed_first =
            BoundedRiskEProcess::new_with_alpha_spending(0.15, 0.5 / 0.15, 0.05, 5, 1)
                .expect("valid alpha-spent fixed monitor");
        let fixed_later =
            BoundedRiskEProcess::new_with_alpha_spending(0.15, 0.5 / 0.15, 0.05, 5, 300)
                .expect("valid later fixed monitor");
        let mixture_first = MixtureBoundedRiskEProcess::new_with_alpha_spending(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
            1,
        )
        .expect("valid alpha-spent mixture");
        let mixture_later = MixtureBoundedRiskEProcess::new_with_alpha_spending(
            0.15,
            [0.25, 0.5, 0.75],
            0.05,
            5,
            300,
        )
        .expect("valid later mixture");

        assert_eq!(fixed_first.episode_budget_index(), Some(1));
        assert_eq!(mixture_later.episode_budget_index(), Some(300));
        assert!(fixed_later.log_alarm_threshold() > fixed_first.log_alarm_threshold());
        assert!(mixture_later.log_alarm_threshold() > mixture_first.log_alarm_threshold());
        assert_eq!(fixed_first.log_alarm_threshold(), mixture_first.log_alarm_threshold());
        assert_eq!(fixed_later.log_alarm_threshold(), mixture_later.log_alarm_threshold());
    }

    #[test]
    fn rejects_invalid_parameters_and_losses() {
        assert!(BoundedRiskEProcess::new(0.0, 1.0, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(1.0, 0.5, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 5.0, 0.05, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 1.0, 1.0, 1).is_none());
        assert!(BoundedRiskEProcess::new(0.2, 1.0, 0.05, 0).is_none());

        let mut process =
            BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1).expect("valid process");
        let before = process.log_e_value();
        assert!(process.observe(f64::NAN).is_none());
        assert!(process.observe(-0.01).is_none());
        assert!(process.observe(1.01).is_none());
        assert_eq!(process.log_e_value(), before);
    }

    #[test]
    fn rejects_non_finite_e_value_thresholds_at_construction() {
        assert!(BoundedRiskEProcess::new(
            0.2,
            2.0,
            f64::MIN_POSITIVE,
            usize::MAX,
        )
        .is_none());
        assert!(MixtureBoundedRiskEProcess::new(
            f64::MIN_POSITIVE,
            [0.25, 0.5, 0.75],
            0.05,
            5,
        )
        .is_none());
    }

    #[test]
    fn allocates_global_delta_across_parallel_monitors() {
        let process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 5).expect("valid process");
        assert!((process.e_value_alarm_threshold() - 100.0).abs() < 1e-10);
        assert_eq!(process.simultaneous_monitors(), 5);
        assert_eq!(process.total_delta(), 0.05);
    }

    #[test]
    fn repeated_low_losses_reduce_evidence_without_alarms() {
        let mut process =
            BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1).expect("valid process");
        for _ in 0..30 {
            assert_eq!(process.observe(0.05), Some(false));
        }
        assert!(process.log_e_value() < 0.0);
        assert!(!process.alarmed());
    }

    #[test]
    fn sustained_high_losses_cross_the_anytime_alarm_boundary() {
        let mut process =
            BoundedRiskEProcess::new(0.15, 0.5 / 0.15, 0.05, 5).expect("valid process");
        let mut first_alarm = None;
        for step in 0..8 {
            if process.observe(0.65) == Some(true) {
                first_alarm.get_or_insert(step);
            }
        }
        assert!(first_alarm.is_some());
        assert!(process.log_e_value() >= process.log_alarm_threshold());
        assert!(process.alarmed());
    }

    #[test]
    fn valid_below_bound_losses_preserve_positive_evidence() {
        let mut process = BoundedRiskEProcess::new(0.2, 2.0, 0.05, 1)
            .expect("valid process");
        for loss in [0.05, 0.10, 0.05, 0.10, 0.05, 0.10] {
            assert!(process.observe(loss).is_some());
        }
        assert!(process.e_value() > 0.0);
    }

    #[test]
    fn reset_starts_a_fresh_evidence_stream() {
        let mut process =
            BoundedRiskEProcess::new(0.15, 0.5 / 0.15, 0.05, 1).expect("valid process");
        for _ in 0..8 {
            assert!(process.observe(0.65).is_some());
        }
        assert!(process.alarmed());
        process.reset();
        assert_eq!(process.log_e_value(), 0.0);
        assert!(!process.alarmed());
    }
}
