// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feeds web-research-acquired numeric data into the Conjecture Engine.
//!
//! # Why this module exists
//!
//! `symthaea_core::hdc::conjecture_engine` is a real symbolic-regression /
//! genetic-programming discovery engine — it takes [`ObservedSequence`]
//! inputs and searches for closed-form conjectures (verified numerically,
//! symbolically, and via Z3). Until now it has only ever been fed
//! synthetic or simulated data: ODE trajectories, number-theoretic
//! generators, physics-catalog formulas (see `conjecture_engine/autonomous.rs`
//! and the Ramanujan paper examples).
//!
//! Separately, [`crate::web_research`] is a real fetch → extract →
//! epistemic-verify → integrate pipeline, but nothing in it ever reached
//! the Conjecture Engine — the two were completely orphaned from each
//! other (Tier 1.3 of `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`).
//!
//! This module is the seam that joins them: it converts numeric data
//! extracted from research content (via
//! [`super::extractor::ContentExtractor::extract_numeric_series`]) plus its
//! epistemic verification (via [`super::verifier::EpistemicVerifier`]) into
//! an [`ObservedSequence`] the engine can search over, **without losing the
//! epistemic pedigree** — every conjecture later produced from
//! research-acquired data can be traced back to which claim/URL it came
//! from and how trustworthy that source was judged to be.
//!
//! # Crate boundary
//!
//! `symthaea-core` (home of the Conjecture Engine) is a dependency of
//! `symthaea` (home of web research), never the reverse. This module lives
//! in `symthaea` and only ever calls *into* `symthaea_core`.

use std::collections::HashMap;
use std::fmt;

use symthaea_core::hdc::conjecture_engine::{
    Conjecture, ConjectureEngine, MacroPromotionTier, MathDomain, ObservedSequence,
};

use super::extractor::NumericSeries;
use super::types::{EpistemicStatus, ResearchSource};
use super::verifier::VerificationResult;

/// A numeric dataset pulled out of research content (a table, a list of
/// measurements, a sequence mentioned in prose) together with enough
/// context to become a valid [`ObservedSequence`].
#[derive(Debug, Clone)]
pub struct ResearchNumericDataset {
    /// Short human-readable label for what this sequence represents,
    /// e.g. `"boiling_point_vs_pressure"` or `"fibonacci_ratio(n)"`.
    pub label: String,
    /// Best-effort math/science domain classification for this data.
    pub domain: MathDomain,
    /// (x, y) pairs to mine for structure.
    pub data: Vec<(f64, f64)>,
    /// The sentence/line the numbers were extracted from (audit trail).
    pub claim_text: String,
    /// URL of the page the data was extracted from.
    pub source_url: String,
}

impl ResearchNumericDataset {
    pub fn new(
        label: impl Into<String>,
        domain: MathDomain,
        data: Vec<(f64, f64)>,
        claim_text: impl Into<String>,
        source_url: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            data,
            claim_text: claim_text.into(),
            source_url: source_url.into(),
        }
    }

    /// Build a dataset from a [`NumericSeries`] found by
    /// [`super::extractor::ContentExtractor::extract_numeric_series`].
    pub fn from_numeric_series(
        label: impl Into<String>,
        domain: MathDomain,
        series: &NumericSeries,
        source_url: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            data: series.values.clone(),
            claim_text: series.context.clone(),
            source_url: source_url.into(),
        }
    }
}

/// Why a [`ResearchNumericDataset`] was rejected before ever reaching the
/// Conjecture Engine.
#[derive(Debug, Clone, PartialEq)]
pub enum ConjectureFeedError {
    /// Fewer than 3 data points — not enough for the engine's
    /// train/test split (`ObservedSequence::train_test_split`) to be
    /// meaningful.
    InsufficientData { label: String, len: usize },
    /// Data contained NaN/infinite values, which would silently poison
    /// GP fitness evaluation downstream.
    NonFiniteValues { label: String },
}

impl fmt::Display for ConjectureFeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData { label, len } => write!(
                f,
                "dataset '{label}' has only {len} point(s); need at least 3"
            ),
            Self::NonFiniteValues { label } => {
                write!(f, "dataset '{label}' contains NaN/infinite values")
            }
        }
    }
}

impl std::error::Error for ConjectureFeedError {}

/// Epistemic pedigree of a research-acquired dataset. Carried *alongside*
/// any [`Conjecture`] generated from it — never collapsed into a single
/// opaque number before the caller has a chance to see it.
#[derive(Debug, Clone)]
pub struct ResearchProvenance {
    pub source_url: String,
    pub claim_text: String,
    pub source_type: ResearchSource,
    pub epistemic_status: EpistemicStatus,
    /// Confidence assigned by [`super::verifier::EpistemicVerifier`] (0.0-1.0).
    pub confidence: f32,
    pub supporting_sources: Vec<String>,
    pub contradicting_sources: Vec<String>,
}

impl ResearchProvenance {
    /// Build provenance directly from a verifier [`VerificationResult`].
    pub fn from_verification(
        dataset: &ResearchNumericDataset,
        source_type: ResearchSource,
        verification: &VerificationResult,
    ) -> Self {
        Self {
            source_url: dataset.source_url.clone(),
            claim_text: dataset.claim_text.clone(),
            source_type,
            epistemic_status: verification.status,
            confidence: verification.confidence,
            supporting_sources: verification.supporting_sources.clone(),
            contradicting_sources: verification.contradicting_sources.clone(),
        }
    }

    /// Construct provenance directly, e.g. when the caller already has an
    /// `EpistemicStatus`/confidence pair rather than a full
    /// `VerificationResult` (for example, a single-source unverified pull).
    pub fn new(
        source_url: impl Into<String>,
        claim_text: impl Into<String>,
        source_type: ResearchSource,
        epistemic_status: EpistemicStatus,
        confidence: f32,
    ) -> Self {
        Self {
            source_url: source_url.into(),
            claim_text: claim_text.into(),
            source_type,
            epistemic_status,
            confidence: confidence.clamp(0.0, 1.0),
            supporting_sources: Vec::new(),
            contradicting_sources: Vec::new(),
        }
    }

    /// Whether the underlying evidence is trustworthy enough that
    /// conjectures derived from it may be considered for macro promotion
    /// (contributing reusable sub-expressions to the engine's grammar).
    pub fn allows_macro_promotion(&self) -> bool {
        self.epistemic_status.is_trustworthy()
    }

    /// Short, filesystem/log-safe tag summarizing epistemic status —
    /// embedded in the `ObservedSequence` name so provenance is
    /// human-visible even without going through [`ConjectureFeeder`].
    fn status_tag(&self) -> &'static str {
        match self.epistemic_status {
            EpistemicStatus::HighConfidence => "high",
            EpistemicStatus::ModerateConfidence => "moderate",
            EpistemicStatus::LowConfidence => "low",
            EpistemicStatus::InsufficientEvidence => "insufficient",
            EpistemicStatus::Contradicted => "contradicted",
            EpistemicStatus::False => "false",
        }
    }
}

/// A conjecture produced from research-acquired data, paired with the full
/// epistemic provenance of the data it was fit to.
#[derive(Debug, Clone)]
pub struct ProvenancedConjecture {
    pub conjecture: Conjecture,
    pub provenance: ResearchProvenance,
}

/// Converts research-acquired numeric datasets into `ObservedSequence`
/// inputs for a [`ConjectureEngine`], keeping a side-table from each
/// generated sequence name back to its full [`ResearchProvenance`] so that
/// after `engine.generate_conjectures(..)` runs, every resulting
/// [`Conjecture`] can be re-paired with the epistemic pedigree of the data
/// that produced it.
///
/// This is the engine's web-research data-acquisition arm: previously the
/// only way to get data into the Conjecture Engine was a synthetic
/// generator (`observe_*` functions) or an ODE trajectory. Now a claim
/// pulled from a web page, with a table of numbers in it, can become a
/// conjecture too — honestly labeled with how much that page should be
/// trusted.
#[derive(Debug, Default)]
pub struct ConjectureFeeder {
    provenance_by_sequence_name: HashMap<String, ResearchProvenance>,
    next_id: u64,
}

impl ConjectureFeeder {
    pub fn new() -> Self {
        Self {
            provenance_by_sequence_name: HashMap::new(),
            next_id: 0,
        }
    }

    /// Convert a research-acquired dataset into an [`ObservedSequence`],
    /// register it with `engine` via [`ConjectureEngine::observe`], and
    /// remember its provenance for later lookup. Returns the assigned
    /// sequence name (also `Conjecture::source` on anything the engine
    /// derives from it).
    pub fn feed(
        &mut self,
        engine: &mut ConjectureEngine,
        dataset: ResearchNumericDataset,
        provenance: ResearchProvenance,
    ) -> Result<String, ConjectureFeedError> {
        if dataset.data.len() < 3 {
            return Err(ConjectureFeedError::InsufficientData {
                label: dataset.label.clone(),
                len: dataset.data.len(),
            });
        }
        if dataset
            .data
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err(ConjectureFeedError::NonFiniteValues {
                label: dataset.label.clone(),
            });
        }

        self.next_id += 1;
        // Self-describing, unique sequence name: provenance is trivially
        // recoverable from `Conjecture::source` even by callers that never
        // touch `ConjectureFeeder` directly (e.g. offline log inspection).
        let seq_name = format!(
            "web_research#{}:{}:{}",
            self.next_id,
            dataset.label,
            provenance.status_tag()
        );

        let seq = ObservedSequence::new(&seq_name, dataset.domain, dataset.data);
        engine.observe(seq);
        self.provenance_by_sequence_name
            .insert(seq_name.clone(), provenance);
        Ok(seq_name)
    }

    /// After `engine.generate_conjectures(..)` has run, pair every
    /// conjecture whose `source` matches a dataset fed through this
    /// feeder with its full [`ResearchProvenance`].
    ///
    /// Pedigree rule applied on the way out (never destructively — the
    /// original engine-computed fields are still visible on
    /// `conjecture.fitness` / `conjecture.training_mse`):
    /// - A conjecture can never be macro-promotion-eligible beyond what
    ///   its data source earned: if the provenance is not
    ///   [`EpistemicStatus::is_trustworthy`], the tier is forced down to
    ///   [`MacroPromotionTier::Quarantined`] regardless of numeric fit
    ///   quality.
    /// - `confidence` is capped by the source's verified confidence — a
    ///   conjecture can never be more trustworthy than the data it was
    ///   fit to.
    pub fn provenanced_conjectures(&self, engine: &ConjectureEngine) -> Vec<ProvenancedConjecture> {
        engine
            .conjectures
            .iter()
            .filter_map(|c| {
                self.provenance_by_sequence_name.get(&c.source).map(|prov| {
                    let mut conjecture = c.clone();
                    if !prov.allows_macro_promotion() {
                        conjecture.macro_promotion_tier = MacroPromotionTier::Quarantined;
                    }
                    conjecture.confidence = conjecture.confidence.min(prov.confidence as f64);
                    ProvenancedConjecture {
                        conjecture,
                        provenance: prov.clone(),
                    }
                })
            })
            .collect()
    }

    /// Number of datasets currently registered with this feeder.
    pub fn dataset_count(&self) -> usize {
        self.provenance_by_sequence_name.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::conjecture_engine::RegressorConfig;

    fn fast_engine() -> ConjectureEngine {
        // Small population/generations: this is an integration smoke test
        // proving data flow, not a search-quality benchmark. Mirrors the
        // fast-test convention used throughout conjecture_engine.rs
        // (population_size: 60, generations: 2).
        ConjectureEngine::with_config(RegressorConfig {
            population_size: 60,
            generations: 2,
            max_depth: 4,
            max_complexity: 15,
            seed: 4242,
            ..Default::default()
        })
    }

    fn trustworthy_provenance() -> ResearchProvenance {
        ResearchProvenance::new(
            "https://en.wikipedia.org/wiki/Triangular_number",
            "The triangular numbers are 1, 3, 6, 10, 15, 21, 28, ...",
            ResearchSource::Web,
            EpistemicStatus::HighConfidence,
            0.9,
        )
    }

    fn unverified_provenance() -> ResearchProvenance {
        ResearchProvenance::new(
            "https://some-random-forum.example/post/1",
            "someone claimed these numbers follow a pattern",
            ResearchSource::Web,
            EpistemicStatus::InsufficientEvidence,
            0.2,
        )
    }

    /// Triangular numbers T(n) = n(n+1)/2 — cheap for the GP to find, a
    /// clean end-to-end fixture that doesn't require network access.
    fn triangular_dataset(label: &str) -> ResearchNumericDataset {
        let data: Vec<(f64, f64)> = (1..=12)
            .map(|n| (n as f64, (n * (n + 1) / 2) as f64))
            .collect();
        ResearchNumericDataset::new(
            label,
            MathDomain::Combinatorics,
            data,
            "The triangular numbers are 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78.",
            "https://en.wikipedia.org/wiki/Triangular_number",
        )
    }

    #[test]
    fn test_feed_rejects_insufficient_data() {
        let mut engine = fast_engine();
        let mut feeder = ConjectureFeeder::new();
        let dataset = ResearchNumericDataset::new(
            "too_short",
            MathDomain::Physics,
            vec![(1.0, 2.0), (2.0, 4.0)],
            "claim",
            "https://example.com",
        );
        let result = feeder.feed(&mut engine, dataset, trustworthy_provenance());
        assert!(matches!(
            result,
            Err(ConjectureFeedError::InsufficientData { .. })
        ));
        assert_eq!(feeder.dataset_count(), 0);
    }

    #[test]
    fn test_feed_rejects_non_finite_values() {
        let mut engine = fast_engine();
        let mut feeder = ConjectureFeeder::new();
        let dataset = ResearchNumericDataset::new(
            "has_nan",
            MathDomain::Physics,
            vec![(1.0, 2.0), (2.0, f64::NAN), (3.0, 6.0), (4.0, 8.0)],
            "claim",
            "https://example.com",
        );
        let result = feeder.feed(&mut engine, dataset, trustworthy_provenance());
        assert!(matches!(
            result,
            Err(ConjectureFeedError::NonFiniteValues { .. })
        ));
    }

    /// The Tier 1.3 acceptance criterion: a web-research-acquired dataset,
    /// tagged with its epistemic provenance, flows end-to-end into the
    /// Conjecture Engine and the engine actually runs conjecture
    /// generation on it. We don't require a specific formula to be
    /// found — proving the loop runs cleanly (produces some result, with
    /// provenance intact) is the point.
    #[test]
    fn test_research_dataset_flows_into_conjecture_engine() {
        let mut engine = fast_engine();
        let mut feeder = ConjectureFeeder::new();

        let seq_name = feeder
            .feed(
                &mut engine,
                triangular_dataset("triangular_numbers"),
                trustworthy_provenance(),
            )
            .expect("well-formed dataset should be accepted");

        assert_eq!(feeder.dataset_count(), 1);
        assert_eq!(engine.observations.len(), 1);
        assert_eq!(engine.observations[0].name, seq_name);

        // Run the engine — this is the Conjecture Engine actually
        // attempting conjecture generation on web-research-acquired data,
        // the thing that has never happened before this module existed.
        let conjectures = engine.generate_conjectures(3);
        assert!(
            !conjectures.is_empty(),
            "engine should produce at least one candidate for a clean \
             quadratic sequence like triangular numbers"
        );

        let provenanced = feeder.provenanced_conjectures(&engine);
        assert!(
            !provenanced.is_empty(),
            "at least one conjecture should re-pair with its provenance"
        );
        for pc in &provenanced {
            assert_eq!(pc.conjecture.source, seq_name);
            assert_eq!(
                pc.provenance.source_url,
                "https://en.wikipedia.org/wiki/Triangular_number"
            );
            assert_eq!(
                pc.provenance.epistemic_status,
                EpistemicStatus::HighConfidence
            );
            // Confidence can never exceed what the source earned.
            assert!(pc.conjecture.confidence <= pc.provenance.confidence as f64 + 1e-9);
        }
    }

    /// Data from an unverified/low-trust source must never be eligible
    /// for macro promotion, no matter how good the numeric fit is — this
    /// is the guard against silently laundering unverified web content
    /// into the engine's reusable-subexpression grammar.
    #[test]
    fn test_unverified_provenance_forces_quarantine() {
        let mut engine = fast_engine();
        let mut feeder = ConjectureFeeder::new();

        feeder
            .feed(
                &mut engine,
                triangular_dataset("suspicious_numbers"),
                unverified_provenance(),
            )
            .unwrap();

        engine.generate_conjectures(3);
        let provenanced = feeder.provenanced_conjectures(&engine);
        assert!(!provenanced.is_empty());
        for pc in &provenanced {
            assert_eq!(
                pc.conjecture.macro_promotion_tier,
                MacroPromotionTier::Quarantined
            );
            // Tolerance widened from 1e-9: `prov.confidence` is f32 (0.2f32
            // widens to 0.20000000298023224f64), so 1e-9 was tighter than
            // the f32->f64 cast's own representation error and failed
            // spuriously. 1e-6 comfortably covers single-precision rounding
            // for any confidence in [0,1] while still catching real bugs.
            assert!(pc.conjecture.confidence <= 0.2 + 1e-6);
        }
    }

    #[test]
    fn test_dataset_from_numeric_series_roundtrips_context() {
        let extractor = super::super::extractor::ContentExtractor::new();
        let text = "Fibonacci ratios: 1, 1, 2, 3, 5, 8, 13, 21.";
        let found = extractor.extract_numeric_series(text, 3);
        assert_eq!(found.len(), 1);

        let dataset = ResearchNumericDataset::from_numeric_series(
            "fibonacci_like",
            MathDomain::NumberTheory,
            &found[0],
            "https://oeis.org/A000045",
        );
        assert_eq!(dataset.claim_text, found[0].context);
        assert_eq!(dataset.data, found[0].values);
        assert_eq!(dataset.source_url, "https://oeis.org/A000045");
    }
}
