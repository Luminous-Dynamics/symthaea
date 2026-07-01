//! Stable fixture catalog for local alpha verification.
//!
//! Fixtures are small, named experiment shapes with explicit interpretation
//! limits. They are not golden scientific results. They are useful for smoke
//! tests, tutorials, notebooks, and downstream scripts that need stable local
//! inputs before running larger experiments.

use crate::presets::RunPreset;
use crate::probe::BindingProbeConfig;

/// Interpretation status for a fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureIntent {
    /// Tiny run intended to verify wiring only.
    Smoke,
    /// Small local run intended for notebook and documentation examples.
    Demonstration,
    /// Pilot run intended to check whether a larger study is worth running.
    Pilot,
}

/// Expected qualitative behavior for a fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixtureExpectation {
    /// Minimum expected clean classical recovery similarity.
    pub minimum_classical_recovery: f32,
    /// Minimum expected clean phase recovery similarity.
    pub minimum_phase_recovery: f32,
    /// Minimum expected clean correlation recovery similarity.
    pub minimum_correlation_recovery: f32,
    /// Maximum acceptable noisy similarity collapse warning threshold.
    pub noisy_similarity_floor: f32,
}

impl FixtureExpectation {
    /// Conservative default expectation for implementation sanity checks.
    pub fn implementation_sanity() -> Self {
        Self {
            minimum_classical_recovery: 0.95,
            minimum_phase_recovery: 0.95,
            minimum_correlation_recovery: 0.95,
            noisy_similarity_floor: 0.50,
        }
    }
}

/// Named local fixture specification.
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureSpec {
    /// Stable fixture name.
    pub name: &'static str,
    /// Human-readable purpose.
    pub purpose: &'static str,
    /// Intended interpretation level.
    pub intent: FixtureIntent,
    /// Binding probe configuration used by the fixture.
    pub config: BindingProbeConfig,
    /// Qualitative expectations for implementation sanity.
    pub expectation: FixtureExpectation,
    /// Required caveat for reports using this fixture.
    pub caveat: &'static str,
}

impl FixtureSpec {
    /// Returns a line-oriented fixture summary.
    pub fn to_text(&self) -> String {
        format!(
            "fixture={} intent={:?} dimension={} trials={} noise={} seed={} purpose={} caveat={}",
            self.name,
            self.intent,
            self.config.dimension,
            self.config.trials,
            self.config.noise,
            self.config.seed,
            self.purpose,
            self.caveat,
        )
    }
}

/// Returns the stable fixture names known in alpha.10.
pub fn fixture_names() -> &'static [&'static str] {
    &["smoke-binding", "demo-binding", "pilot-binding"]
}

/// Looks up a named fixture.
pub fn named_fixture(name: &str) -> Option<FixtureSpec> {
    match name {
        "smoke-binding" => Some(FixtureSpec {
            name: "smoke-binding",
            purpose: "minimal binding probe used for CLI and CI wiring checks",
            intent: FixtureIntent::Smoke,
            config: RunPreset::Smoke.binding_config(),
            expectation: FixtureExpectation::implementation_sanity(),
            caveat: "smoke fixture only; do not report as benchmark evidence",
        }),
        "demo-binding" => Some(FixtureSpec {
            name: "demo-binding",
            purpose: "small local notebook demonstration of binding report shape",
            intent: FixtureIntent::Demonstration,
            config: RunPreset::LocalResearch.binding_config(),
            expectation: FixtureExpectation::implementation_sanity(),
            caveat: "local demonstration only; not a quantum backend observation",
        }),
        "pilot-binding" => Some(FixtureSpec {
            name: "pilot-binding",
            purpose: "pilot-sized binding probe for checking larger experiment readiness",
            intent: FixtureIntent::Pilot,
            config: RunPreset::PilotMatrix.binding_config(),
            expectation: FixtureExpectation::implementation_sanity(),
            caveat: "pilot fixture only; needs replicated matrix before interpretation",
        }),
        _ => None,
    }
}

/// Returns all alpha.10 fixtures.
pub fn fixture_catalog() -> Vec<FixtureSpec> {
    fixture_names()
        .iter()
        .filter_map(|name| named_fixture(name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_lookup_is_stable() {
        let fixture = named_fixture("smoke-binding").unwrap();
        assert_eq!(fixture.intent, FixtureIntent::Smoke);
        assert!(fixture.to_text().contains("smoke-binding"));
        assert_eq!(fixture_catalog().len(), fixture_names().len());
    }
}
