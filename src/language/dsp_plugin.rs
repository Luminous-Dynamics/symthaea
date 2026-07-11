// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! DSP domain plugin — Nyquist frequency / anti-alias check for a sample rate.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::value_for_unit;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_dsp::{nyquist_frequency, will_alias};

pub struct DspDomainPlugin;

const CUES: &[&str] = &[
    "nyquist",
    "aliasing",
    "alias",
    "sample rate",
    "sampling rate",
];

fn result(answer: String) -> ComputedResult {
    ComputedResult {
        answer,
        cube: EpistemicCube {
            e: ETier::E4,
            n: NTier::N3,
            m: MTier::M3,
            h: None,
        },
        psi: 0.0,
        proof_available: false,
    }
}

impl DspDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for DspDomainPlugin {
    fn domain_name(&self) -> &str {
        "dsp"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "nyquist",
            "aliasing",
            "sample",
            "rate",
            "signal",
            "frequency",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let all = crate::language::plugin_parse::values_for_unit(
            input,
            &["hz", "hertz", "khz", "ksps", "sps"],
        );
        // Need at least the sample rate. Convention: the largest value is the
        // sample rate; a smaller one (if present) is the signal frequency.
        let sample_rate = all.iter().cloned().fold(f64::MIN, f64::max);
        if sample_rate <= 0.0 || sample_rate == f64::MIN {
            return None;
        }
        let nyq = nyquist_frequency(sample_rate);
        if let Some(&sig) = all
            .iter()
            .filter(|&&v| v < sample_rate)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            let aliases = will_alias(sig, sample_rate);
            return Some(result(format!(
                "Nyquist frequency for a {sample_rate} Hz sample rate is {nyq} Hz; a {sig} Hz \
                 signal {} alias.",
                if aliases { "WILL" } else { "will NOT" }
            )));
        }
        Some(result(format!(
            "The Nyquist frequency for a {sample_rate} Hz sample rate is {nyq} Hz \
             (the highest signal frequency captured without aliasing)."
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nyquist_of_cd_audio() {
        let p = DspDomainPlugin;
        let r = p
            .compute(
                "what is the nyquist frequency for a 44100 Hz sample rate?",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("22050"), "{}", r.answer);
    }

    #[test]
    fn detects_aliasing() {
        let p = DspDomainPlugin;
        // 30 kHz signal sampled at 44100 Hz → above Nyquist → aliases.
        let r = p
            .compute("sampling rate 44100 Hz, does a 30000 Hz signal alias?", &[])
            .unwrap();
        assert!(r.answer.contains("WILL"), "{}", r.answer);
    }

    #[test]
    fn no_cue_none() {
        let p = DspDomainPlugin;
        assert!(p.compute("play the song at 44100", &[]).is_none());
    }
}
