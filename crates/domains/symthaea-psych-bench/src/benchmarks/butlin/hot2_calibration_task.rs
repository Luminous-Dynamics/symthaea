//! Architecture-agnostic metacognitive calibration task for HOT-2.
//!
//! A runner supplies ground-truth correctness, reported confidence, and
//! whether the system flagged its own confidence as suspect. This module
//! computes calibration/error metrics only; it grants no support tier.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationTrial {
    pub correct: bool,
    pub confidence: f64,
    pub confidence_was_corrupted: bool,
    pub mismatch_flagged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationMetrics {
    pub brier_score: f64,
    pub corruption_detection_rate: f64,
    pub false_alarm_rate: f64,
}

pub fn score(trials: &[CalibrationTrial]) -> Result<CalibrationMetrics, String> {
    if trials.is_empty() {
        return Err("calibration task requires at least one trial".into());
    }
    if trials
        .iter()
        .any(|t| !t.confidence.is_finite() || !(0.0..=1.0).contains(&t.confidence))
    {
        return Err("confidence must be finite and inside [0,1]".into());
    }

    let brier = trials
        .iter()
        .map(|t| {
            let y = if t.correct { 1.0 } else { 0.0 };
            (t.confidence - y).powi(2)
        })
        .sum::<f64>()
        / trials.len() as f64;

    let corrupted: Vec<_> = trials
        .iter()
        .filter(|t| t.confidence_was_corrupted)
        .collect();
    let clean: Vec<_> = trials
        .iter()
        .filter(|t| !t.confidence_was_corrupted)
        .collect();

    let detection = if corrupted.is_empty() {
        0.0
    } else {
        corrupted.iter().filter(|t| t.mismatch_flagged).count() as f64 / corrupted.len() as f64
    };
    let false_alarm = if clean.is_empty() {
        0.0
    } else {
        clean.iter().filter(|t| t.mismatch_flagged).count() as f64 / clean.len() as f64
    };

    Ok(CalibrationMetrics {
        brier_score: brier,
        corruption_detection_rate: detection,
        false_alarm_rate: false_alarm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_known_corruption_without_rewarding_false_alarms() {
        let trials = [
            CalibrationTrial { correct: true, confidence: 0.9, confidence_was_corrupted: false, mismatch_flagged: false },
            CalibrationTrial { correct: false, confidence: 0.9, confidence_was_corrupted: true, mismatch_flagged: true },
            CalibrationTrial { correct: true, confidence: 0.1, confidence_was_corrupted: true, mismatch_flagged: true },
            CalibrationTrial { correct: false, confidence: 0.1, confidence_was_corrupted: false, mismatch_flagged: false },
        ];
        let m = score(&trials).unwrap();
        assert_eq!(m.corruption_detection_rate, 1.0);
        assert_eq!(m.false_alarm_rate, 0.0);
        assert!(m.brier_score > 0.0);
    }

    #[test]
    fn malformed_confidence_fails_closed() {
        let trials = [CalibrationTrial {
            correct: true,
            confidence: 1.1,
            confidence_was_corrupted: false,
            mismatch_flagged: false,
        }];
        assert!(score(&trials).is_err());
    }
}
