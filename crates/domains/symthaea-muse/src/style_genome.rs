// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Objective, provenance-bearing StyleGenome evaluation and Pareto archive.
//!
//! Generator fitness is structural and evidence-based. Human preference is
//! intentionally absent from this module. A candidate is admitted only when
//! universal score invariants pass, the grammar emits a non-empty trace, all
//! trace references resolve against the exact score, and fingerprints are
//! finite. Trace integrity in this v1 is referential/order integrity only;
//! it does not elevate composer assertions into independently verified
//! musical semantics. Style-specific musical judgments remain reported separately;
//! this module does not force every tradition through common-practice tonal
//! rules.
//!
//! This v2 uses the existing hybrid compiler, typed family-native grammar
//! plans, native trace, score validator and structural fingerprint. It does
//! not duplicate those mechanisms; it fingerprints their emitted evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use symthaea_muse_protocol::{
    EvidenceBasis, EvidenceStatus, HybridStyleSpec, ListenCompositionBundle,
    MeterPoint, MotifDefinition, MusicalTime, SymbolicNoteEvent, TempoPoint,
};
use symthaea_music_theory::{
    GrammarPlanEvidence, GrammarStructuralTrace, MusicalIntent,
    Score, ScoreValidationConfig, fingerprint, compose_with_grammar_plan,
    score_validation::{ValidationSeverity, is_universal_invariant, validate_score},
};

use crate::analyst::{composer_trace_from_theory, verify_composer_trace};
use crate::style_mixer::compile_hybrid_style;

pub const STYLE_GENOME_SCHEMA: &str = "muse-style-genome-v1";
pub const STYLE_GENOME_EVALUATION_SCHEMA: &str = "muse-style-genome-evaluation-v2";
pub const STYLE_GENOME_ARCHIVE_SCHEMA: &str = "muse-style-genome-pareto-archive-v2";
pub const STYLE_GENOME_TRACE_REFERENCE_AUDIT_VERSION: &str = "muse-style-genome-trace-reference-audit-v1";
pub const STYLE_GENOME_FINGERPRINT_VERSION: &str = "symthaea-music-theory-structural-fingerprint-v1";
pub const STYLE_GENOME_GRAMMAR_PLAN_FINGERPRINT_VERSION: &str = "muse-grammar-plan-fingerprint-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenomeAxis {
    Structure,
    Melody,
    Rhythm,
    Harmony,
    Ensemble,
    Performance,
    Production,
}

impl GenomeAxis {
    fn catalog_id(self, recipe: &HybridStyleSpec) -> Option<u16> {
        match self {
            Self::Structure => Some(recipe.structural_style_id),
            Self::Melody => recipe.melodic_style_id,
            Self::Rhythm => recipe.rhythmic_style_id,
            Self::Harmony => recipe.harmonic_style_id,
            Self::Ensemble => recipe.ensemble_style_id,
            Self::Performance => recipe.performance_style_id,
            Self::Production => recipe.production_style_id,
        }
    }

    fn set_catalog_id(self, recipe: &mut HybridStyleSpec, catalog_id: Option<u16>) -> Result<(), GenomeIssue> {
        match self {
            Self::Structure => {
                let Some(value) = catalog_id else {
                    return Err(GenomeIssue::new(
                        "structural_gene_cannot_be_empty",
                        "the structural owner is mandatory",
                    ));
                };
                recipe.structural_style_id = value;
            }
            Self::Melody => recipe.melodic_style_id = catalog_id,
            Self::Rhythm => recipe.rhythmic_style_id = catalog_id,
            Self::Harmony => recipe.harmonic_style_id = catalog_id,
            Self::Ensemble => recipe.ensemble_style_id = catalog_id,
            Self::Performance => recipe.performance_style_id = catalog_id,
            Self::Production => recipe.production_style_id = catalog_id,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenomeMutation {
    pub axis: GenomeAxis,
    pub previous_catalog_id: Option<u16>,
    pub next_catalog_id: Option<u16>,
    pub operator: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleGenome {
    pub schema: String,
    pub compiler_schema: String,
    /// Content-derived identifier; changing recipe, compiler schema,
    /// parentage or mutation history changes the ID. No wall clock or UUID.
    pub genome_id: String,
    pub recipe: HybridStyleSpec,
    pub parent_genome_ids: Vec<String>,
    pub mutation_history: Vec<GenomeMutation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct GenomeIdentity<'a> {
    schema: &'a str,
    compiler_schema: &'a str,
    recipe: &'a HybridStyleSpec,
    parent_genome_ids: &'a [String],
    mutation_history: &'a [GenomeMutation],
}

impl StyleGenome {
    pub fn new(recipe: HybridStyleSpec) -> Result<Self, GenomeIssue> {
        Self::from_parts(recipe, Vec::new(), Vec::new())
    }

    fn from_parts(
        recipe: HybridStyleSpec,
        parent_genome_ids: Vec<String>,
        mutation_history: Vec<GenomeMutation>,
    ) -> Result<Self, GenomeIssue> {
        let mut genome = Self {
            schema: STYLE_GENOME_SCHEMA.into(),
            compiler_schema: crate::style_mixer::HYBRID_STYLE_COMPILER_SCHEMA.into(),
            genome_id: String::new(),
            recipe,
            parent_genome_ids,
            mutation_history,
        };
        genome.genome_id = genome.compute_id()?;
        Ok(genome)
    }

    pub fn verify_id(&self) -> bool {
        self.schema == STYLE_GENOME_SCHEMA
            && self.compiler_schema == crate::style_mixer::HYBRID_STYLE_COMPILER_SCHEMA
            && self.compute_id().is_ok_and(|expected| expected == self.genome_id)
    }

    fn compute_id(&self) -> Result<String, GenomeIssue> {
        let identity = GenomeIdentity {
            schema: &self.schema,
            compiler_schema: &self.compiler_schema,
            recipe: &self.recipe,
            parent_genome_ids: &self.parent_genome_ids,
            mutation_history: &self.mutation_history,
        };
        let bytes = serde_json::to_vec(&identity)
            .map_err(|error| GenomeIssue::new("genome_serialization_failed", error.to_string()))?;
        Ok(format!("sha256:{}", domain_sha256(b"symthaea:muse-style-genome:v1\n", &bytes)))
    }

    /// Replace one explicit source gene. The old genome is immutable; the
    /// returned genome carries parent lineage and an inspectable mutation.
    /// Compatibility is decided later by the hybrid compiler, not guessed here.
    pub fn replace_gene(
        &self,
        axis: GenomeAxis,
        next_catalog_id: Option<u16>,
    ) -> Result<Self, GenomeIssue> {
        let previous = axis.catalog_id(&self.recipe);
        if previous == next_catalog_id {
            return Err(GenomeIssue::new(
                "no_op_mutation",
                format!("{} already has catalogue ID {:?}", axis.layer_name(), previous),
            ));
        }
        let mut recipe = self.recipe.clone();
        axis.set_catalog_id(&mut recipe, next_catalog_id)?;
        let mut parents = self.parent_genome_ids.clone();
        if !parents.contains(&self.genome_id) {
            parents.push(self.genome_id.clone());
        }
        parents.sort();
        let mut mutations = self.mutation_history.clone();
        mutations.push(GenomeMutation {
            axis,
            previous_catalog_id: previous,
            next_catalog_id,
            operator: "replace_catalogue_layer_source_v1".into(),
        });
        Self::from_parts(recipe, parents, mutations)
    }
}

impl GenomeAxis {
    fn layer_name(self) -> &'static str {
        match self {
            Self::Structure => "structure",
            Self::Melody => "melody",
            Self::Rhythm => "rhythm",
            Self::Harmony => "harmony",
            Self::Ensemble => "ensemble",
            Self::Performance => "performance",
            Self::Production => "production",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenomeIssue {
    pub code: String,
    pub message: String,
}

impl GenomeIssue {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeEvaluationConfig {
    /// Use at least two seeds; the archive evaluates a genome's behavior,
    /// not one cherry-picked generation.
    pub seeds: Vec<u64>,
    pub score_validation: ScoreValidationConfig,
    /// Number of nearest reference fingerprints used for novelty.
    pub novelty_neighbours: usize,
}

impl Default for GenomeEvaluationConfig {
    fn default() -> Self {
        Self {
            seeds: vec![1, 7, 19, 42, 101],
            score_validation: ScoreValidationConfig::default(),
            novelty_neighbours: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceIntegrity {
    pub assertion_count: usize,
    pub event_reference_count: usize,
    pub valid_event_reference_count: usize,
    pub invalid_event_references: Vec<String>,
    pub spans_ordered: bool,
}

impl TraceIntegrity {
    pub fn is_complete(&self) -> bool {
        self.assertion_count > 0
            && self.spans_ordered
            && self.invalid_event_references.is_empty()
            && self.event_reference_count == self.valid_event_reference_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenomeTraceIssue {
    pub code: String,
    pub record_id: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeTrial {
    pub seed: u64,
    pub score_sha256: String,
    pub exact_score_fingerprint: u64,
    pub structural_fingerprint: Vec<f64>,
    pub universal_invariant_passed: bool,
    pub universal_failure_codes: Vec<String>,
    /// Style-dependent theory findings are exposed but not treated as
    /// universal errors. Family-specific verification may interpret them.
    pub style_dependent_fatal_count: usize,
    pub warning_count: usize,
    pub trace_integrity: TraceIntegrity,
    pub semantic_trace_verified: bool,
    pub semantic_trace_verifier_version: String,
    pub semantic_trace_checked_records: usize,
    pub semantic_trace_issues: Vec<GenomeTraceIssue>,
    pub grammar_plan_kind: String,
    pub grammar_plan_well_formed: bool,
    /// Domain-separated, seed-normalized fingerprint of the family-native plan.
    pub grammar_plan_signature: Option<String>,
    /// Named operations extracted from the typed family plan.
    pub grammar_transformations: Vec<String>,
    /// Legacy topology signature for the generic Form representation.
    pub form_signature: Option<String>,
    /// Motif-operation labels emitted by the generic structural trace.
    pub motif_transformations: Vec<String>,
    pub note_count: usize,
    pub score_duration_beats: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeObjectives {
    /// Mean k-nearest structural distance from a separately supplied
    /// reference population. None means no reference population was supplied.
    pub novelty_distance: Option<f64>,
    /// Mean pairwise structural distance between this genome's seeded trials.
    pub within_genome_diversity: f64,
    /// Fraction of the four generic motif-transformation labels observed.
    /// None means the family did not emit that trace representation.
    pub motif_transformation_richness: Option<f64>,
    /// Fraction of configured generic-Form alternatives observed. None means
    /// this grammar has no generic Form object; dedicated plan topology is
    /// measured separately below.
    pub form_coverage: Option<f64>,
    /// Fraction of distinct family-native grammar-plan topologies observed
    /// across deterministic trials. Applicable to every qualified plan family.
    pub grammar_topology_variation: f64,
    /// Mean pairwise Jaccard distance between each trial's named grammar
    /// operations. This compares process/chorus/cycle/fugal/dramatic operations
    /// without converting them into listener preference.
    pub grammar_operation_diversity: f64,
}

impl GenomeObjectives {
    /// Objective axes are maximized separately; incomparable applicability
    /// states remain incomparable instead of being silently scored as zero.
    pub fn axes(&self) -> [Option<f64>; 6] {
        [
            self.novelty_distance,
            Some(self.within_genome_diversity),
            self.motif_transformation_richness,
            self.form_coverage,
            Some(self.grammar_topology_variation),
            Some(self.grammar_operation_diversity),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeEvaluation {
    pub schema: String,
    pub genome: StyleGenome,
    pub compiler_schema: String,
    pub theory_validation_version: String,
    pub trace_reference_audit_version: String,
    pub fingerprint_version: String,
    pub grammar_plan_fingerprint_version: String,
    pub compiled_spec_sha256: String,
    pub evaluation_config: GenomeEvaluationConfig,
    pub evaluation_config_sha256: String,
    pub reference_population_sha256: Option<String>,
    pub reference_population_member_count: usize,
    pub seed_count: usize,
    pub eligible_for_archive: bool,
    pub abstention_reasons: Vec<GenomeIssue>,
    pub objectives: GenomeObjectives,
    pub mean_structural_fingerprint: Vec<f64>,
    pub layer_coverage: Vec<LayerCoverage>,
    pub trials: Vec<GenomeTrial>,
    /// Hash binds the genome, evaluation config and all trial receipts.
    pub evaluation_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerCoverage {
    pub layer: String,
    pub mean_distance_to_population: Option<f64>,
    pub nonzero_across_trials: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeArchiveConfig {
    pub max_entries: usize,
    /// Normalized Euclidean distance. Near-clone rejection is a duplicate
    /// control only; its threshold must be calibrated on null pairs before
    /// confirmatory discovery claims.
    pub near_clone_threshold: Option<f64>,
}

impl Default for GenomeArchiveConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            near_clone_threshold: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeArchiveEntry {
    pub genome_id: String,
    pub evaluation_sha256: String,
    pub mean_structural_fingerprint: Vec<f64>,
    pub objectives: GenomeObjectives,
    pub evaluation: GenomeEvaluation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeArchive {
    pub schema: String,
    pub config: GenomeArchiveConfig,
    /// The reference population is frozen for the lifetime of an archive.
    /// Pareto comparisons are invalid when novelty scores use different baselines.
    pub reference_population_sha256: Option<String>,
    pub reference_population_bound: bool,
    pub entries: Vec<GenomeArchiveEntry>,
    /// Domain-separated content digest with this field cleared during hashing.
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveInsertDisposition {
    Accepted,
    ReplacedDominatedEntries,
    RejectedIneligible,
    RejectedExactDuplicate,
    RejectedNearClone,
    RejectedDominated,
    RejectedCapacity,
    RejectedReferencePopulationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveInsertReceipt {
    pub disposition: ArchiveInsertDisposition,
    pub genome_id: String,
    pub removed_genome_ids: Vec<String>,
    pub reason: String,
}

impl GenomeArchive {
    pub fn new(config: GenomeArchiveConfig) -> Result<Self, GenomeIssue> {
        if config.max_entries == 0 {
            return Err(GenomeIssue::new("invalid_archive_capacity", "max_entries must be positive"));
        }
        if config.near_clone_threshold.is_some_and(|threshold| {
            !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)
        }) {
            return Err(GenomeIssue::new(
                "invalid_near_clone_threshold",
                "near_clone_threshold must be None or finite and in [0,1]",
            ));
        }
        let mut archive = Self {
            schema: STYLE_GENOME_ARCHIVE_SCHEMA.into(),
            config,
            reference_population_sha256: None,
            reference_population_bound: false,
            entries: Vec::new(),
            archive_sha256: String::new(),
        };
        archive.archive_sha256 = archive.compute_digest()?;
        Ok(archive)
    }

    fn compute_digest(&self) -> Result<String, GenomeIssue> {
        let mut copy = self.clone();
        copy.archive_sha256.clear();
        let bytes = serde_json::to_vec(&copy)
            .map_err(|error| GenomeIssue::new("archive_serialization_failed", error.to_string()))?;
        Ok(domain_sha256(b"symthaea:muse-style-genome-archive:v2\n", &bytes))
    }

    /// Detect tampering, duplicate IDs, changed candidate receipts, mixed
    /// novelty baselines, and a persisted set that is no longer Pareto-valid.
    pub fn verify_digest(&self) -> bool {
        if self.schema != STYLE_GENOME_ARCHIVE_SCHEMA
            || self.config.max_entries == 0
            || self.config.near_clone_threshold.is_some_and(|threshold| {
                !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)
            })
            || self.entries.len() > self.config.max_entries
            || !self.reference_population_bound && !self.entries.is_empty()
        {
            return false;
        }
        let mut previous_id: Option<&str> = None;
        let mut ids = BTreeSet::new();
        for entry in &self.entries {
            if !ids.insert(entry.genome_id.as_str())
                || previous_id.is_some_and(|previous| previous >= entry.genome_id.as_str())
                || !entry.evaluation.verify_digest()
                || !entry.evaluation.eligible_for_archive
                || entry.genome_id != entry.evaluation.genome.genome_id
                || entry.evaluation_sha256 != entry.evaluation.evaluation_sha256
                || entry.objectives != entry.evaluation.objectives
                || entry.mean_structural_fingerprint != entry.evaluation.mean_structural_fingerprint
                || entry.evaluation.reference_population_sha256 != self.reference_population_sha256
            {
                return false;
            }
            previous_id = Some(entry.genome_id.as_str());
        }
        for left in 0..self.entries.len() {
            for right in (left + 1)..self.entries.len() {
                if objective_dominates(&self.entries[left].objectives, &self.entries[right].objectives)
                    || objective_dominates(&self.entries[right].objectives, &self.entries[left].objectives)
                {
                    return false;
                }
            }
        }
        self.compute_digest().is_ok_and(|expected| expected == self.archive_sha256)
    }

    pub fn insert(&mut self, evaluation: GenomeEvaluation) -> ArchiveInsertReceipt {
        let genome_id = evaluation.genome.genome_id.clone();
        if !evaluation.genome.verify_id() || !evaluation.verify_digest() {
            return receipt(
                ArchiveInsertDisposition::RejectedIneligible,
                &genome_id,
                Vec::new(),
                "genome identity or evaluation receipt digest did not verify",
            );
        }
        if self.reference_population_bound
            && self.reference_population_sha256 != evaluation.reference_population_sha256
        {
            return receipt(
                ArchiveInsertDisposition::RejectedReferencePopulationMismatch,
                &genome_id,
                Vec::new(),
                "Pareto candidates must use the same frozen reference-population digest",
            );
        }
        if !evaluation.eligible_for_archive {
            return receipt(
                ArchiveInsertDisposition::RejectedIneligible,
                &genome_id,
                Vec::new(),
                "hard score/trace invariants failed or the candidate abstained",
            );
        }
        if self.entries.iter().any(|entry| entry.genome_id == genome_id) {
            return receipt(
                ArchiveInsertDisposition::RejectedExactDuplicate,
                &genome_id,
                Vec::new(),
                "genome ID already exists in the archive",
            );
        }
        let Some(candidate) = mean_fingerprint_array(&evaluation.mean_structural_fingerprint) else {
            return receipt(
                ArchiveInsertDisposition::RejectedIneligible,
                &genome_id,
                Vec::new(),
                "mean structural fingerprint has an invalid dimension or non-finite value",
            );
        };
        let candidate_scores: BTreeSet<_> = evaluation.trials.iter()
            .map(|trial| trial.score_sha256.as_str())
            .collect();
        if !candidate_scores.is_empty()
            && self.entries.iter().any(|entry| {
                let existing_scores: BTreeSet<_> = entry.evaluation.trials.iter()
                    .map(|trial| trial.score_sha256.as_str())
                    .collect();
                candidate_scores == existing_scores
            })
        {
            return receipt(
                ArchiveInsertDisposition::RejectedExactDuplicate,
                &genome_id,
                Vec::new(),
                "all generated score digests match an existing genome's trial set",
            );
        }

        if let Some(entry) = self.entries.iter()
            .find(|entry| objective_dominates(&entry.objectives, &evaluation.objectives))
        {
            return receipt(
                ArchiveInsertDisposition::RejectedDominated,
                &genome_id,
                Vec::new(),
                format!("candidate is Pareto-dominated by {}", entry.genome_id),
            );
        }

        let removed: Vec<String> = self.entries.iter()
            .filter(|entry| objective_dominates(&evaluation.objectives, &entry.objectives))
            .map(|entry| entry.genome_id.clone())
            .collect();
        let mut next_entries: Vec<GenomeArchiveEntry> = self.entries.iter()
            .filter(|entry| !removed.contains(&entry.genome_id))
            .cloned()
            .collect();

        if removed.is_empty() && self.config.near_clone_threshold.is_some_and(|threshold| {
            next_entries.iter().any(|entry| {
                mean_fingerprint_array(&entry.mean_structural_fingerprint)
                    .is_some_and(|existing| normalized_distance(&candidate, &existing) <= threshold)
            })
        }) {
            return receipt(
                ArchiveInsertDisposition::RejectedNearClone,
                &genome_id,
                Vec::new(),
                "candidate is structurally near-identical to an archived genome and does not Pareto-dominate it",
            );
        }
        if next_entries.len() >= self.config.max_entries {
            return receipt(
                ArchiveInsertDisposition::RejectedCapacity,
                &genome_id,
                Vec::new(),
                "archive is full; no scalar score is used to evict an existing Pareto member",
            );
        }

        next_entries.push(GenomeArchiveEntry {
            genome_id: genome_id.clone(),
            evaluation_sha256: evaluation.evaluation_sha256.clone(),
            mean_structural_fingerprint: evaluation.mean_structural_fingerprint.clone(),
            objectives: evaluation.objectives.clone(),
            evaluation,
        });
        next_entries.sort_by(|left, right| left.genome_id.cmp(&right.genome_id));

        // Build and hash a complete candidate snapshot before mutating self.
        // Any serialization or invariant failure leaves the original archive
        // byte-for-byte unchanged.
        let mut next_archive = Self {
            schema: self.schema.clone(),
            config: self.config.clone(),
            reference_population_sha256: self.reference_population_sha256.clone()
                .or_else(|| next_entries.first().and_then(|entry| {
                    entry.evaluation.reference_population_sha256.clone()
                })),
            reference_population_bound: self.reference_population_bound || !next_entries.is_empty(),
            entries: next_entries,
            archive_sha256: String::new(),
        };
        let digest = match next_archive.compute_digest() {
            Ok(digest) => digest,
            Err(error) => {
                return receipt(
                    ArchiveInsertDisposition::RejectedIneligible,
                    &genome_id,
                    Vec::new(),
                    format!("archive digest could not be recomputed: {}", error.message),
                );
            }
        };
        next_archive.archive_sha256 = digest;
        if !next_archive.verify_digest() {
            return receipt(
                ArchiveInsertDisposition::RejectedIneligible,
                &genome_id,
                Vec::new(),
                "candidate archive snapshot failed its own integrity/non-dominance checks",
            );
        }
        *self = next_archive;

        receipt(
            if removed.is_empty() {
                ArchiveInsertDisposition::Accepted
            } else {
                ArchiveInsertDisposition::ReplacedDominatedEntries
            },
            &genome_id,
            removed,
            "candidate retained as a non-dominated member of the multi-objective archive",
        )
    }
}

pub fn evaluate_genome(
    genome: &StyleGenome,
    base_intent: &MusicalIntent,
    config: &GenomeEvaluationConfig,
    reference_population: &[[f64; fingerprint::STRUCT_DIMS]],
) -> Result<GenomeEvaluation, Vec<GenomeIssue>> {
    let mut issues = Vec::new();
    if config.seeds.len() < 2 {
        return Err(vec![GenomeIssue::new(
            "insufficient_seed_replicates",
            "at least two deterministic seeds are required for a genome evaluation",
        )]);
    }
    if config.seeds.iter().copied().collect::<BTreeSet<_>>().len() != config.seeds.len() {
        return Err(vec![GenomeIssue::new(
            "duplicate_seed",
            "evaluation seeds must be unique",
        )]);
    }
    if reference_population.iter().any(|vector| vector.iter().any(|value| !value.is_finite())) {
        return Err(vec![GenomeIssue::new(
            "invalid_reference_population",
            "reference fingerprints must contain only finite values",
        )]);
    }
    let evaluation_config_sha256 = config_digest(config)?;
    let reference_population_sha256 = reference_population_digest(reference_population)?;
    let compiled = compile_hybrid_style(&genome.recipe).map_err(|mix_issues| {
        mix_issues.into_iter().map(|problem| {
            GenomeIssue::new(&problem.code, problem.message)
        }).collect::<Vec<_>>()
    })?;
    let mut trials = Vec::with_capacity(config.seeds.len());

    for seed in &config.seeds {
        let mut intent = *base_intent;
        intent.seed = *seed;
        let realization = compose_with_grammar_plan(
            compiled.structural_style.grammar_profile(),
            &intent,
            &compiled.spec,
        );
        let report = validate_score(&realization.score, &config.score_validation);
        let universal_failures: Vec<_> = report.issues.iter()
            .filter(|problem| {
                problem.severity == ValidationSeverity::Fatal
                    && is_universal_invariant(problem.rule)
            })
            .collect();
        let universal_failure_codes = universal_failures.iter()
            .map(|problem| format!("{:?}", problem.rule))
            .collect::<Vec<_>>();
        let style_dependent_fatal_count = report.issues.iter()
            .filter(|problem| {
                problem.severity == ValidationSeverity::Fatal
                    && !is_universal_invariant(problem.rule)
            })
            .count();
        let warning_count = report.warning_count();
        let trace_integrity = audit_trace(&realization.score, &realization.trace);
        let score_bytes = serde_json::to_vec(&realization.score).map_err(|error| {
            vec![GenomeIssue::new("score_serialization_failed", error.to_string())]
        })?;
        let score_sha256 = domain_sha256(
            b"symthaea:muse-style-genome-score:v1\n",
            &score_bytes,
        );
        let source_motif = compiled.spec.motif(intent.arousal, intent.seed);
        let source_motif = if compiled.spec.texture.hook_cell {
            symthaea_music_theory::hook::graft_hook(
                &source_motif,
                &symthaea_music_theory::hook::HookCell::generate_with(
                    &compiled.spec.melody,
                    intent.seed,
                    compiled.spec.meter as f64,
                ),
                compiled.spec.meter as f64,
            )
        } else {
            source_motif
        };
        let semantic_bundle = listen_bundle_for_score(
            &realization.score,
            &source_motif,
            &realization.plan,
        );
        let composer_trace = composer_trace_from_theory(
            &realization.trace,
            &score_sha256,
            0,
            Some(("motif-primary", 1)),
        );
        let semantic_verification = verify_composer_trace(
            &composer_trace,
            &semantic_bundle,
            &score_sha256,
            0,
        );
        let semantic_trace_issues = semantic_verification.issues.iter()
            .map(|issue| GenomeTraceIssue {
                code: issue.code.clone(),
                record_id: issue.record_id.clone(),
                message: issue.message.clone(),
            })
            .collect::<Vec<_>>();
        let semantic_trace_verified = semantic_verification.issues.is_empty()
            && semantic_verification.checked_records == semantic_verification.verified_records
            && semantic_verification.checked_records > 0;
        let grammar_plan_well_formed = grammar_plan_well_formed(&realization.plan);
        let grammar_plan_signature = grammar_plan_signature(&realization.plan);
        let grammar_transformations = grammar_transformations(&realization.plan);
        if grammar_plan_signature.is_none() {
            issues.push(GenomeIssue::new(
                "grammar_plan_fingerprint_missing",
                format!("seed {seed} emitted no family-native grammar plan fingerprint"),
            ));
        }
        let structural_fingerprint = fingerprint::structural_fingerprint(
            &realization.score,
            &realization.form,
        );
        if structural_fingerprint.iter().any(|value| !value.is_finite()) {
            issues.push(GenomeIssue::new(
                "non_finite_structural_fingerprint",
                format!("seed {seed} produced a non-finite structural fingerprint"),
            ));
        }
        // Form variation is topology variation, not arbitrary differences
        // in notes, key spelling or progression values. Hash only section
        // role order and meter so equivalent forms remain in the same cell.
        let form_signature = realization.form.as_ref().map(|form| {
            let role_sequence = form.sections.iter()
                .map(|section| format!("{:?}", section.role))
                .collect::<Vec<_>>();
            let topology = serde_json::to_vec(&(realization.score.meter, role_sequence))
                .unwrap_or_default();
            domain_sha256(b"symthaea:muse-style-genome-form-topology:v1\n", &topology)
        });
        let motif_transformations = realization.trace.motif_occurrences.iter()
            .map(|occurrence| occurrence.transformation.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        trials.push(GenomeTrial {
            seed: *seed,
            score_sha256,
            exact_score_fingerprint: fingerprint::exact_fingerprint(&realization.score),
            structural_fingerprint: structural_fingerprint.to_vec(),
            universal_invariant_passed: universal_failures.is_empty(),
            universal_failure_codes,
            style_dependent_fatal_count,
            warning_count,
            trace_integrity,
            semantic_trace_verified,
            semantic_trace_verifier_version: semantic_verification.verifier_version.clone(),
            semantic_trace_checked_records: semantic_verification.checked_records,
            semantic_trace_issues,
            grammar_plan_kind: grammar_plan_name(&realization.plan),
            grammar_plan_well_formed,
            grammar_plan_signature,
            grammar_transformations,
            form_signature,
            motif_transformations,
            note_count: realization.score.notes.len(),
            score_duration_beats: realization.score.total_beats.beats(),
        });
    }

    let eligible_for_archive = issues.is_empty()
        && trials.iter().all(|trial| {
            trial.universal_invariant_passed
                && trial.trace_integrity.is_complete()
                && trial.semantic_trace_verified
                && trial.grammar_plan_well_formed
                && trial.structural_fingerprint.iter().all(|value| value.is_finite())
        });
    if !eligible_for_archive {
        if trials.iter().any(|trial| !trial.universal_invariant_passed) {
            issues.push(GenomeIssue::new(
                "universal_score_invariant_failed",
                "one or more seeded trials failed universal score invariants",
            ));
        }
        if trials.iter().any(|trial| !trial.trace_integrity.is_complete()) {
            issues.push(GenomeIssue::new(
                "structural_trace_incomplete",
                "one or more seeded trials emitted an empty, malformed or dangling structural trace",
            ));
        }
        if trials.iter().any(|trial| !trial.semantic_trace_verified) {
            issues.push(GenomeIssue::new(
                "independent_trace_verification_failed",
                "one or more seeded trials failed Muse Analyst's independent structural-trace verifier",
            ));
        }
        if trials.iter().any(|trial| !trial.grammar_plan_well_formed) {
            issues.push(GenomeIssue::new(
                "grammar_plan_incomplete",
                "one or more seeded trials emitted an incomplete grammar plan",
            ));
        }
    }

    let trial_fingerprints = trials.iter()
        .filter_map(|trial| mean_fingerprint_array(&trial.structural_fingerprint))
        .collect::<Vec<_>>();
    let within_genome_diversity = mean_pairwise_distance(&trial_fingerprints);
    let novelty_distance = novelty_distance(&trial_fingerprints, reference_population, config.novelty_neighbours);
    let motif_transformations = trials.iter()
        .flat_map(|trial| trial.motif_transformations.iter().cloned())
        .collect::<BTreeSet<_>>();
    let motif_transformation_richness = if motif_transformations.is_empty() {
        None
    } else {
        Some((motif_transformations.len() as f64 / 4.0).min(1.0))
    };
    let distinct_form_topologies = trials.iter()
        .filter_map(|trial| trial.form_signature.as_ref())
        .collect::<BTreeSet<_>>()
        .len();
    let form_coverage = if trials.iter().any(|trial| trial.form_signature.is_some()) {
        let denominator = compiled.spec.form_pool.len().min(trials.len()).max(1);
        Some(distinct_form_topologies as f64 / denominator as f64)
    } else {
        None
    };
    let grammar_topology_variation = pairwise_signature_diversity(
        &trials
            .iter()
            .filter_map(|trial| trial.grammar_plan_signature.as_deref())
            .collect::<Vec<_>>(),
    );
    let grammar_operation_diversity = mean_pairwise_operation_diversity(&trials);

    let mut mean_fingerprint = vec![0.0; fingerprint::STRUCT_DIMS];
    if !trial_fingerprints.is_empty() {
        for vector in &trial_fingerprints {
            for (sum, value) in mean_fingerprint.iter_mut().zip(vector.iter()) {
                *sum += *value / trial_fingerprints.len() as f64;
            }
        }
    }
    let layer_coverage = layer_coverage(&trial_fingerprints, reference_population);
    let mut evaluation = GenomeEvaluation {
        schema: STYLE_GENOME_EVALUATION_SCHEMA.into(),
        genome: genome.clone(),
        compiler_schema: crate::style_mixer::HYBRID_STYLE_COMPILER_SCHEMA.into(),
        theory_validation_version: symthaea_music_theory::THEORY_VALIDATION_VERSION.into(),
        trace_reference_audit_version: STYLE_GENOME_TRACE_REFERENCE_AUDIT_VERSION.into(),
        fingerprint_version: STYLE_GENOME_FINGERPRINT_VERSION.into(),
        grammar_plan_fingerprint_version: STYLE_GENOME_GRAMMAR_PLAN_FINGERPRINT_VERSION.into(),
        compiled_spec_sha256: compiled.provenance.compiled_spec_sha256.clone(),
        evaluation_config: config.clone(),
        evaluation_config_sha256,
        reference_population_sha256,
        reference_population_member_count: reference_population.len(),
        seed_count: config.seeds.len(),
        eligible_for_archive,
        abstention_reasons: issues,
        objectives: GenomeObjectives {
            novelty_distance,
            within_genome_diversity,
            motif_transformation_richness,
            form_coverage,
            grammar_topology_variation,
            grammar_operation_diversity,
        },
        mean_structural_fingerprint: mean_fingerprint,
        layer_coverage,
        trials,
        evaluation_sha256: String::new(),
    };
    evaluation.evaluation_sha256 = evaluation_digest(&evaluation)?;
    Ok(evaluation)
}

fn listen_bundle_for_score(
    score: &Score,
    motif: &symthaea_music_theory::Motif,
    plan: &GrammarPlanEvidence,
) -> ListenCompositionBundle {
    const TICKS_PER_BEAT: f64 = 480.0;
    let seconds_per_beat = 60.0 / score.tempo_bpm as f64;
    let time = |beats: f64| MusicalTime {
        tick: (beats * TICKS_PER_BEAT).round().max(0.0) as u64,
        beats,
        seconds: beats * seconds_per_beat,
    };
    let mut notes = Vec::with_capacity(score.notes.len());
    for role in [
        symthaea_music_theory::VoiceRole::Melody,
        symthaea_music_theory::VoiceRole::Harmony,
        symthaea_music_theory::VoiceRole::Bass,
        symthaea_music_theory::VoiceRole::CounterMelody,
    ] {
        let role_name = format!("{role:?}");
        let id_role = match role {
            symthaea_music_theory::VoiceRole::Melody => "melody",
            symthaea_music_theory::VoiceRole::Harmony => "harmony",
            symthaea_music_theory::VoiceRole::Bass => "bass",
            symthaea_music_theory::VoiceRole::CounterMelody => "counter",
        };
        for (index, note) in score.voice(role).into_iter().enumerate() {
            let onset = note.onset.beats();
            let duration = note.duration.beats();
            notes.push(SymbolicNoteEvent {
                id: format!("score-{id_role}-{index}"),
                midi: note.pitch.midi(),
                pitch_name: format!("MIDI{}", note.pitch.midi()),
                onset: time(onset),
                duration_ticks: (duration * TICKS_PER_BEAT).round().max(0.0) as u64,
                duration_beats: duration,
                duration_seconds: duration * seconds_per_beat,
                velocity: note.velocity,
                voice_role: role_name.clone(),
                emphasis: format!("{:?}", note.emphasis).to_ascii_lowercase(),
                section_intensity: note.section_intensity,
            });
        }
    }
    notes.sort_by(|left, right| left.onset.beats.total_cmp(&right.onset.beats));
    let motif_notes: Vec<_> = motif.notes.iter()
        .filter_map(|note| note.degree.map(|degree| (degree, note.duration.beats())))
        .collect();
    ListenCompositionBundle {
        ticks_per_beat: TICKS_PER_BEAT as u32,
        duration_ticks: (score.total_beats.beats() * TICKS_PER_BEAT).round().max(0.0) as u64,
        duration_beats: score.total_beats.beats(),
        duration_seconds: score.seconds(),
        form_kind: format!("{plan:?}"),
        tempo_map: vec![TempoPoint {
            at: time(0.0),
            bpm: score.tempo_bpm,
        }],
        meter_map: vec![MeterPoint {
            at: time(0.0),
            numerator: score.meter,
            denominator: 4,
        }],
        sections: Vec::new(),
        phrases: Vec::new(),
        notes,
        motif_definitions: vec![MotifDefinition {
            id: "motif-primary".into(),
            label: "recipe-source-motif".into(),
            degrees: motif_notes.iter().map(|(degree, _)| *degree).collect(),
            durations_beats: motif_notes.iter().map(|(_, duration)| *duration).collect(),
            basis: EvidenceBasis {
                status: EvidenceStatus::Reconstructed,
                source_method: "style-genome-recipe-motif-v1".into(),
                confidence: None,
                limitations: vec![
                    "This is the exact recipe-selected motif used by the grammar planner; motif occurrences remain composer assertions until Analyst verification.".into(),
                ],
            },
        }],
        motif_occurrences: Vec::new(),
        cadences: Vec::new(),
        sonorities: Vec::new(),
        orchestration: Vec::new(),
        resonance: None,
    }
}

fn audit_trace(score: &Score, trace: &GrammarStructuralTrace) -> TraceIntegrity {
    let mut assertion_count = 0usize;
    let mut event_reference_count = 0usize;
    let mut valid_event_reference_count = 0usize;
    let mut invalid = Vec::new();
    let mut spans_ordered = true;

    let mut check_ref = |reference: symthaea_music_theory::ScoreEventRef, label: String| {
        event_reference_count += 1;
        match score.voice(reference.role).get(reference.role_index) {
            Some(_) => valid_event_reference_count += 1,
            None => invalid.push(label),
        }
    };

    for span in &trace.structures {
        assertion_count += 1;
        check_ref(span.start, format!("structure:{}:start", span.id));
        check_ref(span.end, format!("structure:{}:end", span.id));
        if let (Some(start), Some(end)) = (
            score.voice(span.start.role).get(span.start.role_index),
            score.voice(span.end.role).get(span.end.role_index),
        ) {
            if start.onset.beats() > end.onset.beats() {
                spans_ordered = false;
                invalid.push(format!("structure:{}:reversed", span.id));
            }
        }
    }
    for motif in &trace.motif_occurrences {
        assertion_count += 1;
        for (index, reference) in motif.events.iter().enumerate() {
            check_ref(*reference, format!("motif:{}:{index}", motif.occurrence_id));
        }
    }
    for cadence in &trace.cadences {
        assertion_count += 1;
        for (index, reference) in cadence.preparation.iter().enumerate() {
            check_ref(*reference, format!("cadence:{}:preparation:{index}", cadence.cadence_id));
        }
        check_ref(cadence.arrival, format!("cadence:{}:arrival", cadence.cadence_id));
    }
    for obligation in &trace.obligation_transitions {
        assertion_count += 1;
        for (index, reference) in obligation.evidence.iter().enumerate() {
            check_ref(*reference, format!("obligation:{}:{index}", obligation.obligation_id));
        }
    }
    TraceIntegrity {
        assertion_count,
        event_reference_count,
        valid_event_reference_count,
        invalid_event_references: invalid,
        spans_ordered,
    }
}

fn grammar_plan_well_formed(plan: &GrammarPlanEvidence) -> bool {
    use symthaea_music_theory::grammar::GrammarPlanEvidence as G;
    match plan {
        G::PeriodSentence(plan) => {
            plan.section_count > 0
                && !plan.phrase_model.trim().is_empty()
                && !plan.cadence_strategy.trim().is_empty()
                && !plan.continuation_strategy.trim().is_empty()
                && !plan.return_obligation.trim().is_empty()
                && !plan.ending_strategy.trim().is_empty()
        }
        G::Contrapuntal(plan) => {
            plan.engine_version == "compact-fugue-engine-v1"
                && !plan.answer_class.trim().is_empty()
                && plan.subject_entry_order.len() >= 3
                && plan.countersubject
                && plan.inversion
                && plan.augmentation
                && plan.stretto
                && plan.controlled_dissonance
                && plan.final_combination
        }
        G::GrooveCycle(plan) => {
            plan.cycle_beats.is_finite()
                && plan.cycle_beats > 0.0
                && plan.total_cycles > 0
                && !plan.sections.is_empty()
                && plan.sections.first().is_some_and(|section| section.start_cycle == 0)
                && plan.sections.iter().all(|section| section.cycles > 0)
                && plan.sections.windows(2).all(|pair| {
                    pair[0].start_cycle + pair[0].cycles == pair[1].start_cycle
                })
                && plan.sections.last().is_some_and(|section| {
                    section.start_cycle + section.cycles == plan.total_cycles
                })
                && plan.phrase_boundary_cycles.iter().all(|cycle| *cycle < plan.total_cycles)
                && plan.phrase_boundary_cycles.windows(2).all(|pair| pair[0] < pair[1])
                && plan.phase_continuous
                && !plan.obligations.is_empty()
                && plan.obligations.iter().all(|obligation| obligation.fulfilled)
        }
        G::AdditiveProcess(plan) => {
            !plan.prefix_lengths.is_empty()
                && plan.prefix_lengths.len() == plan.cycle_starts.len()
                && plan.prefix_lengths.iter().all(|length| *length > 0)
                && plan.cycle_starts.first().is_some_and(|start| start.beats() == 0.0)
                && plan.cycle_starts.iter().all(|start| start.beats().is_finite())
                && plan.cycle_starts.windows(2).all(|pair| {
                    pair[0].beats() < pair[1].beats()
                })
        }
        G::ModalArc(plan) => {
            plan.stage_repetitions.iter().all(|count| *count > 0)
                && plan.alap_end.beats().is_finite()
                && plan.jor_end.beats().is_finite()
                && plan.jhala_end.beats().is_finite()
                && 0.0 < plan.alap_end.beats()
                && plan.alap_end.beats() < plan.jor_end.beats()
                && plan.jor_end.beats() < plan.jhala_end.beats()
        }
        G::CallResponse(plan) => {
            plan.requested_bars > 0
                && plan.choruses > 0
                && plan.bars_per_chorus == 12
                && plan.realized_bars == plan.choruses * plan.bars_per_chorus
                && plan.realized_bars >= plan.requested_bars
                && plan.trajectory.len() == plan.choruses
                && plan.trajectory.first().is_some_and(|role| {
                    *role == symthaea_music_theory::call_response::ChorusRole::Statement
                })
                && plan.call_starts == vec![0, 4, 8]
                && plan.response_starts == vec![2, 6, 10]
        }
        G::JazzChorus(plan) => {
            plan.requested_bars > 0
                && plan.choruses > 0
                && plan.bars_per_chorus == 8
                && plan.realized_bars == plan.choruses * plan.bars_per_chorus
                && plan.realized_bars >= plan.requested_bars
                && plan.trajectory.len() == plan.choruses
                && plan.trajectory.first().is_some_and(|role| {
                    *role == symthaea_music_theory::call_response::ChorusRole::Statement
                })
        }
        G::ProgSuite(evidence) => {
            let plan = &evidence.plan;
            evidence.verification.all_verified()
                && plan.sections.len() == 4
                && plan.sections.iter().map(|section| section.meter).eq([4, 7, 5, 4])
                && plan.sections.windows(2).all(|pair| {
                    (pair[0].end.beats() - pair[1].start.beats()).abs() <= 1e-7
                })
        }
        G::Sonata(evidence) => {
            evidence.plan.sections.len() == 5
                && !evidence.verification.is_empty()
                && evidence.verification.iter().all(|item| item.verified)
        }
        G::Renaissance(evidence) => {
            evidence.plan.total_bars == 9 && evidence.verification.all_verified()
        }
        G::Opera(evidence) => {
            evidence.plan.total_bars == 9 && evidence.verification.all_verified()
        }
        G::Compatibility { .. } => false,
    }
}

fn grammar_plan_name(plan: &GrammarPlanEvidence) -> String {
    use symthaea_music_theory::grammar::GrammarPlanEvidence as G;
    match plan {
        G::PeriodSentence(_) => "period_sentence".into(),
        G::Contrapuntal(_) => "contrapuntal".into(),
        G::GrooveCycle(_) => "groove_cycle".into(),
        G::AdditiveProcess(_) => "additive_process".into(),
        G::ModalArc(_) => "modal_arc".into(),
        G::CallResponse(_) => "call_response".into(),
        G::JazzChorus(_) => "jazz_chorus".into(),
        G::ProgSuite(_) => "prog_suite".into(),
        G::Sonata(_) => "sonata".into(),
        G::Renaissance(_) => "renaissance".into(),
        G::Opera(_) => "opera".into(),
        G::Compatibility { family, .. } => format!("compatibility:{family:?}"),
    }
}

fn strip_seed_and_dynamic_fields(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            object.remove("seed");
            object.remove("energy");
            object.remove("section_intensity");
            // Absolute transposition is a different musical realization,
            // not a different formal topology. Relative key-context labels
            // (e.g. Opera home/relative) remain in the typed plan.
            object.remove("key");
            object.remove("home_key");
            object.remove("contrast_key");
            object.remove("development_key");
            for child in object.values_mut() {
                strip_seed_and_dynamic_fields(child);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                strip_seed_and_dynamic_fields(child);
            }
        }
        _ => {}
    }
}

/// Canonical, family-native fingerprint of a typed grammar plan. Random seed
/// and non-structural intensity values are excluded; structural role order,
/// cycle/chorus counts, section boundaries, transformations and obligations
/// remain bound. Compatibility plans are deliberately unqualified.
fn grammar_plan_signature(plan: &GrammarPlanEvidence) -> Option<String> {
    if matches!(plan, GrammarPlanEvidence::Compatibility { .. }) {
        return None;
    }
    let mut value = serde_json::to_value(plan).ok()?;
    strip_seed_and_dynamic_fields(&mut value);
    let bytes = serde_json::to_vec(&value).ok()?;
    Some(domain_sha256(
        b"symthaea:muse-grammar-plan-topology:v1\n",
        &bytes,
    ))
}

fn grammar_transformations(plan: &GrammarPlanEvidence) -> Vec<String> {
    use symthaea_music_theory::call_response::ChorusRole;
    use symthaea_music_theory::grammar::GrammarPlanEvidence as G;

    let mut operations = BTreeSet::new();
    match plan {
        G::PeriodSentence(plan) => {
            operations.insert(format!("phrase-model:{}", plan.phrase_model));
            operations.insert(format!("continuation:{}", plan.continuation_strategy));
        }
        G::Contrapuntal(plan) => {
            for (enabled, name) in [
                (plan.countersubject, "countersubject"),
                (plan.inversion, "inversion"),
                (plan.augmentation, "augmentation"),
                (plan.diminution, "diminution"),
                (plan.stretto, "stretto"),
                (plan.controlled_dissonance, "controlled_dissonance"),
                (plan.final_combination, "final_combination"),
            ] {
                if enabled {
                    operations.insert(format!("fugue:{name}"));
                }
            }
        }
        G::GrooveCycle(plan) => {
            operations.insert(format!("cycle-kind:{:?}", plan.kind));
            for section in &plan.sections {
                operations.insert(format!("cycle-role:{:?}", section.role));
            }
            for obligation in &plan.obligations {
                if obligation.fulfilled {
                    operations.insert(format!("cycle-obligation:{}", obligation.code));
                }
            }
        }
        G::AdditiveProcess(plan) => {
            operations.insert(format!("process-trajectory:{:?}", plan.trajectory_kind));
            if plan.prefix_lengths.windows(2).any(|pair| pair[1] > pair[0]) {
                operations.insert("process:prefix-growth".into());
            }
            if plan.prefix_lengths.windows(2).any(|pair| pair[1] < pair[0]) {
                operations.insert("process:prefix-release".into());
            }
            if plan.prefix_lengths.windows(2).any(|pair| pair[1] == pair[0]) {
                operations.insert("process:prefix-hold".into());
            }
        }
        G::ModalArc(plan) => {
            operations.insert(format!("modal-arc-profile:{:?}", plan.profile));
            operations.insert("modal-arc:alap".into());
            operations.insert("modal-arc:jor".into());
            operations.insert("modal-arc:jhala".into());
        }
        G::CallResponse(plan) => {
            for role in &plan.trajectory {
                let label = match role {
                    ChorusRole::Statement => "statement",
                    ChorusRole::Variation => "variation",
                    ChorusRole::Intensification => "intensification",
                    ChorusRole::Breakdown => "breakdown",
                    ChorusRole::Reconciliation => "reconciliation",
                    ChorusRole::Coda => "coda",
                };
                operations.insert(format!("blues-chorus:{label}"));
            }
            operations.insert("blues:call-response".into());
        }
        G::JazzChorus(plan) => {
            for role in &plan.trajectory {
                let label = match role {
                    ChorusRole::Statement => "statement",
                    ChorusRole::Variation => "variation",
                    ChorusRole::Intensification => "intensification",
                    ChorusRole::Breakdown => "breakdown",
                    ChorusRole::Reconciliation => "reconciliation",
                    ChorusRole::Coda => "coda",
                };
                operations.insert(format!("jazz-chorus:{label}"));
            }
            operations.insert("jazz:head-and-chorus-development".into());
        }
        G::ProgSuite(evidence) => {
            for section in &evidence.plan.sections {
                operations.insert(format!("prog-section:{:?}", section.role));
                if let Some(transform) = section.transform {
                    let label = match transform {
                        0 => "inversion",
                        1 => "retrograde",
                        2 => "retrograde-inversion",
                        _ => "other",
                    };
                    operations.insert(format!("prog-transform:{label}"));
                }
            }
        }
        G::Sonata(evidence) => {
            for section in &evidence.plan.sections {
                operations.insert(format!("sonata-section:{:?}", section.kind));
            }
            for item in &evidence.verification {
                if item.verified {
                    operations.insert(format!("sonata-obligation:{}", item.obligation_id));
                }
            }
        }
        G::Renaissance(evidence) => {
            operations.insert("renaissance:equal-voice-imitation".into());
            operations.insert(format!("renaissance:point-one-{}", evidence.plan.point_one_entry_order.join("-")));
            operations.insert(format!("renaissance:point-two-{}", evidence.plan.point_two_entry_order.join("-")));
            operations.insert("renaissance:suspension-7-6".into());
            operations.insert("renaissance:landini-7-6-8".into());
            operations.insert("renaissance:tonic-close".into());
        }
        G::Opera(evidence) => {
            for section in &evidence.plan.sections {
                operations.insert(format!("opera-section:{}", section.id));
            }
            operations.insert("opera:two-theme-dialogue".into());
            operations.insert("opera:one-bar-turn-taking".into());
            operations.insert("opera:interruption".into());
            operations.insert("opera:cadential-tag".into());
        }
        G::Compatibility { .. } => {}
    }
    operations.into_iter().collect()
}

fn jaccard_distance(left: &[String], right: &[String]) -> f64 {
    let left: BTreeSet<_> = left.iter().collect();
    let right: BTreeSet<_> = right.iter().collect();
    let union = left.union(&right).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = left.intersection(&right).count();
    1.0 - intersection as f64 / union as f64
}

fn pairwise_signature_diversity(signatures: &[&str]) -> f64 {
    if signatures.len() < 2 {
        return 0.0;
    }
    let mut different_pairs = 0usize;
    let mut pair_count = 0usize;
    for left in 0..signatures.len() {
        for right in left + 1..signatures.len() {
            if signatures[left] != signatures[right] {
                different_pairs += 1;
            }
            pair_count += 1;
        }
    }
    different_pairs as f64 / pair_count.max(1) as f64
}

fn mean_pairwise_operation_diversity(trials: &[GenomeTrial]) -> f64 {
    if trials.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for left in 0..trials.len() {
        for right in left + 1..trials.len() {
            sum += jaccard_distance(
                &trials[left].grammar_transformations,
                &trials[right].grammar_transformations,
            );
            count += 1;
        }
    }
    sum / count.max(1) as f64
}


fn mean_pairwise_distance(fingerprints: &[[f64; fingerprint::STRUCT_DIMS]]) -> f64 {
    if fingerprints.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for left in 0..fingerprints.len() {
        for right in left + 1..fingerprints.len() {
            sum += normalized_distance(&fingerprints[left], &fingerprints[right]);
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

fn novelty_distance(
    fingerprints: &[[f64; fingerprint::STRUCT_DIMS]],
    reference_population: &[[f64; fingerprint::STRUCT_DIMS]],
    neighbours: usize,
) -> Option<f64> {
    if fingerprints.is_empty() || reference_population.is_empty() {
        return None;
    }
    let k = neighbours.max(1).min(reference_population.len());
    let sum = fingerprints.iter().map(|candidate| {
        let mut distances = reference_population.iter()
            .map(|reference| normalized_distance(candidate, reference))
            .collect::<Vec<_>>();
        distances.sort_by(f64::total_cmp);
        distances.iter().take(k).sum::<f64>() / k as f64
    }).sum::<f64>();
    Some(sum / fingerprints.len() as f64)
}

fn layer_coverage(
    fingerprints: &[[f64; fingerprint::STRUCT_DIMS]],
    references: &[[f64; fingerprint::STRUCT_DIMS]],
) -> Vec<LayerCoverage> {
    fingerprint::LAYERS.iter().enumerate().map(|(index, (name, _, _))| {
        let nonzero_across_trials = fingerprints.iter().any(|value| {
            fingerprint::layer_dists(value, &[0.0; fingerprint::STRUCT_DIMS])[index] > 1e-12
        });
        let mean_distance_to_population = if fingerprints.is_empty() || references.is_empty() {
            None
        } else {
            let mut distances = Vec::new();
            for value in fingerprints {
                for reference in references {
                    distances.push(fingerprint::layer_dists(value, reference)[index]);
                }
            }
            Some(distances.iter().sum::<f64>() / distances.len() as f64)
        };
        LayerCoverage {
            layer: (*name).into(),
            mean_distance_to_population,
            nonzero_across_trials,
        }
    }).collect()
}

fn normalized_distance(
    left: &[f64; fingerprint::STRUCT_DIMS],
    right: &[f64; fingerprint::STRUCT_DIMS],
) -> f64 {
    fingerprint::dist(left, right) / (fingerprint::STRUCT_DIMS as f64).sqrt()
}

fn mean_fingerprint_array(values: &[f64]) -> Option<[f64; fingerprint::STRUCT_DIMS]> {
    if values.len() != fingerprint::STRUCT_DIMS || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut result = [0.0; fingerprint::STRUCT_DIMS];
    result.copy_from_slice(values);
    Some(result)
}

fn objective_dominates(left: &GenomeObjectives, right: &GenomeObjectives) -> bool {
    let left_axes = left.axes();
    let right_axes = right.axes();
    let mut strictly_better = false;
    for index in 0..left_axes.len() {
        match (left_axes[index], right_axes[index]) {
            (Some(a), Some(b)) => {
                if a < b {
                    return false;
                }
                strictly_better |= a > b;
            }
            (None, None) => {}
            // Candidates evaluated against different/missing reference
            // populations are incomparable on novelty, not silently equated.
            _ => return false,
        }
    }
    strictly_better
}

fn config_digest(config: &GenomeEvaluationConfig) -> Result<String, Vec<GenomeIssue>> {
    let bytes = serde_json::to_vec(config).map_err(|error| {
        vec![GenomeIssue::new("evaluation_config_serialization_failed", error.to_string())]
    })?;
    Ok(domain_sha256(b"symthaea:muse-style-genome-config:v1\n", &bytes))
}

fn evaluation_digest(evaluation: &GenomeEvaluation) -> Result<String, Vec<GenomeIssue>> {
    let mut copy = evaluation.clone();
    copy.evaluation_sha256.clear();
    let bytes = serde_json::to_vec(&copy)
        .map_err(|error| vec![GenomeIssue::new("evaluation_serialization_failed", error.to_string())])?;
    Ok(domain_sha256(b"symthaea:muse-style-genome-evaluation:v2\n", &bytes))
}

pub fn reference_population_digest(
    reference_population: &[[f64; fingerprint::STRUCT_DIMS]],
) -> Result<Option<String>, Vec<GenomeIssue>> {
    if reference_population.is_empty() {
        return Ok(None);
    }
    let mut members = Vec::with_capacity(reference_population.len());
    for vector in reference_population {
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(vec![GenomeIssue::new(
                "invalid_reference_population",
                "reference fingerprints must contain only finite values",
            )]);
        }
        let bytes = serde_json::to_vec(vector).map_err(|error| {
            vec![GenomeIssue::new("reference_population_serialization_failed", error.to_string())]
        })?;
        members.push(domain_sha256(
            b"symthaea:muse-style-genome-reference-member:v1\n",
            &bytes,
        ));
    }
    members.sort();
    let bytes = serde_json::to_vec(&members).map_err(|error| {
        vec![GenomeIssue::new("reference_population_serialization_failed", error.to_string())]
    })?;
    Ok(Some(domain_sha256(
        b"symthaea:muse-style-genome-reference-population:v1\n",
        &bytes,
    )))
}

impl GenomeEvaluation {
    pub fn verify_digest(&self) -> bool {
        let reference_binding_valid = match &self.reference_population_sha256 {
            Some(digest) => {
                self.reference_population_member_count > 0
                    && is_sha256_hex(digest)
            }
            None => self.reference_population_member_count == 0,
        };
        self.schema == STYLE_GENOME_EVALUATION_SCHEMA
            && self.compiler_schema == crate::style_mixer::HYBRID_STYLE_COMPILER_SCHEMA
            && self.theory_validation_version == symthaea_music_theory::THEORY_VALIDATION_VERSION
            && self.trace_reference_audit_version == STYLE_GENOME_TRACE_REFERENCE_AUDIT_VERSION
            && self.fingerprint_version == STYLE_GENOME_FINGERPRINT_VERSION
            && self.grammar_plan_fingerprint_version == STYLE_GENOME_GRAMMAR_PLAN_FINGERPRINT_VERSION
            && self.genome.verify_id()
            && is_sha256_hex(&self.evaluation_config_sha256)
            && config_digest(&self.evaluation_config).is_ok_and(|expected| {
                expected == self.evaluation_config_sha256
            })
            && reference_binding_valid
            && self.seed_count == self.evaluation_config.seeds.len()
            && self.seed_count == self.trials.len()
            && self.trials.iter().all(|trial| {
                is_sha256_hex(&trial.score_sha256)
                    && (!self.eligible_for_archive
                        || trial.grammar_plan_signature.as_ref().is_some_and(|value| is_sha256_hex(value)))
                    && trial.structural_fingerprint.len() == fingerprint::STRUCT_DIMS
                    && trial.structural_fingerprint.iter().all(|value| value.is_finite())
                    && trial.score_duration_beats.is_finite()
                    && trial.score_duration_beats > 0.0
            })
            && self.mean_structural_fingerprint.len() == fingerprint::STRUCT_DIMS
            && self.mean_structural_fingerprint.iter().all(|value| value.is_finite())
            && self.objectives.within_genome_diversity.is_finite()
            && self.objectives.motif_transformation_richness
                .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            && self.objectives.form_coverage
                .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            && self.objectives.grammar_topology_variation.is_finite()
            && (0.0..=1.0).contains(&self.objectives.grammar_topology_variation)
            && self.objectives.grammar_operation_diversity.is_finite()
            && (0.0..=1.0).contains(&self.objectives.grammar_operation_diversity)
            && self.objectives.novelty_distance.is_none_or(|value| value.is_finite() && value >= 0.0)
            && evaluation_digest(self).is_ok_and(|expected| expected == self.evaluation_sha256)
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn receipt(
    disposition: ArchiveInsertDisposition,
    genome_id: &str,
    removed_genome_ids: Vec<String>,
    reason: impl Into<String>,
) -> ArchiveInsertReceipt {
    ArchiveInsertReceipt {
        disposition,
        genome_id: genome_id.into(),
        removed_genome_ids,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classical_genome() -> StyleGenome {
        let find = |name: &str| {
            (1..=152)
                .find(|id| symthaea_muse_protocol::catalog::catalog_entry(*id)
                    .is_some_and(|entry| entry.composer_style == Some(name)))
                .unwrap()
        };
        StyleGenome::new(HybridStyleSpec {
            structural_style_id: find("Classical"),
            melodic_style_id: Some(find("Blues")),
            rhythmic_style_id: Some(find("Tango")),
            harmonic_style_id: None,
            ensemble_style_id: None,
            performance_style_id: None,
            production_style_id: None,
            culturally_qualified_sources_acknowledged: false,
        }).unwrap()
    }

    #[test]
    fn genome_ids_are_deterministic_and_gene_replacements_preserve_lineage() {
        let genome = classical_genome();
        let again = classical_genome();
        assert_eq!(genome.genome_id, again.genome_id);
        let next = genome.replace_gene(GenomeAxis::Melody, None).unwrap();
        assert_ne!(genome.genome_id, next.genome_id);
        assert_eq!(next.parent_genome_ids, vec![genome.genome_id.clone()]);
        assert_eq!(next.mutation_history.len(), 1);
        assert_eq!(next.mutation_history[0].previous_catalog_id, genome.recipe.melodic_style_id);
        assert_eq!(next.mutation_history[0].next_catalog_id, None);
    }

    #[test]
    fn structural_owner_gene_cannot_be_erased() {
        let genome = classical_genome();
        let result = genome.replace_gene(GenomeAxis::Structure, None);
        assert_eq!(result.unwrap_err().code, "structural_gene_cannot_be_empty");
    }

    #[test]
    fn candidate_evaluation_is_deterministic_and_independent_of_preference() {
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![3, 7, 11],
            ..GenomeEvaluationConfig::default()
        };
        let left = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        let right = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        assert_eq!(left, right);
        assert!(left.eligible_for_archive, "{:?}", left.abstention_reasons);
        assert_eq!(left.seed_count, 3);
        assert!(left.objectives.novelty_distance.is_none());
        assert!(left.evaluation_sha256.len() == 64);
    }

    #[test]
    fn dedicated_grammar_plan_fingerprints_ignore_seed_but_bind_structure() {
        use symthaea_music_theory::{GrammarPlanEvidence, Style};

        let opera = Style::Opera;
        let opera_spec = opera.spec();
        let mut intent_a = MusicalIntent::default();
        intent_a.seed = 3;
        let mut intent_b = MusicalIntent::default();
        intent_b.seed = 4;
        let plan_a = compose_with_grammar_plan(
            opera.grammar_profile(),
            &intent_a,
            &opera_spec,
        ).plan;
        let plan_b = compose_with_grammar_plan(
            opera.grammar_profile(),
            &intent_b,
            &opera_spec,
        ).plan;

        assert!(grammar_plan_well_formed(&plan_a));
        assert_eq!(grammar_plan_signature(&plan_a), grammar_plan_signature(&plan_b));
        assert_eq!(grammar_transformations(&plan_a), grammar_transformations(&plan_b));

        let mut changed = plan_a.clone();
        match &mut changed {
            GrammarPlanEvidence::Opera(evidence) => {
                evidence.plan.sections[0].end_bar_exclusive += 1;
            }
            other => panic!("expected Opera plan, got {other:?}"),
        }
        assert_ne!(grammar_plan_signature(&plan_a), grammar_plan_signature(&changed));
    }

    #[test]
    fn process_plan_fingerprint_tracks_trajectory_not_random_seed_field() {
        use symthaea_music_theory::Style;

        let style = Style::Minimalism;
        let spec = style.spec();
        let intent_a = MusicalIntent { seed: 0, ..Default::default() };
        let intent_b = MusicalIntent { seed: 1, ..Default::default() };
        let plan_a = compose_with_grammar_plan(style.grammar_profile(), &intent_a, &spec).plan;
        let plan_b = compose_with_grammar_plan(style.grammar_profile(), &intent_b, &spec).plan;

        assert!(grammar_plan_well_formed(&plan_a));
        assert!(grammar_plan_well_formed(&plan_b));
        assert_ne!(grammar_plan_signature(&plan_a), grammar_plan_signature(&plan_b));
        assert_ne!(grammar_transformations(&plan_a), grammar_transformations(&plan_b));
    }

    #[test]
    fn compatibility_plans_abstain_from_native_topology_fitness() {
        let compatibility = GrammarPlanEvidence::Compatibility {
            family: symthaea_music_theory::grammar::GrammarFamily::AmbientTextural,
            form_available: true,
        };
        assert!(!grammar_plan_well_formed(&compatibility));
        assert!(grammar_plan_signature(&compatibility).is_none());
        assert!(grammar_transformations(&compatibility).is_empty());
    }

    #[test]
    fn topology_and_operation_diversity_are_seed_count_stable() {
        let identical = ["same", "same", "same", "same"];
        let all_different = ["a", "b", "c", "d"];
        let half = ["a", "a", "b", "b"];
        assert_eq!(pairwise_signature_diversity(&identical), 0.0);
        assert_eq!(pairwise_signature_diversity(&all_different), 1.0);
        assert!((pairwise_signature_diversity(&half) - (4.0 / 6.0)).abs() < 1e-12);

        assert_eq!(jaccard_distance(&[], &[]), 0.0);
        assert_eq!(
            jaccard_distance(&["inversion".into()], &["retrograde".into()]),
            1.0
        );
        assert_eq!(
            jaccard_distance(&["inversion".into()], &["inversion".into()]),
            0.0
        );
    }

    #[test]
    fn trace_reference_gate_is_separate_from_style_dependent_theory_findings() {
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![1, 2],
            ..GenomeEvaluationConfig::default()
        };
        let evaluation = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        assert!(evaluation.trials.iter().all(|trial| trial.trace_integrity.is_complete()));
        assert!(evaluation.trials.iter().all(|trial| trial.semantic_trace_verified), "{:?}", evaluation.trials.iter().map(|trial| &trial.semantic_trace_issues).collect::<Vec<_>>());
        assert!(evaluation.trials.iter().all(|trial| trial.universal_invariant_passed));
        // The archive gate does not incorrectly treat a classical-rule
        // warning/failure as a universal structural defect in another style.
        assert_eq!(evaluation.abstention_reasons.len(), 0);
    }

    #[test]
    fn reference_population_digest_is_order_independent_and_content_bound() {
        let first = [0.0; fingerprint::STRUCT_DIMS];
        let mut second = [0.0; fingerprint::STRUCT_DIMS];
        second[0] = 0.25;
        let left = reference_population_digest(&[first, second]).unwrap();
        let reordered = reference_population_digest(&[second, first]).unwrap();
        let changed = {
            let mut third = second;
            third[0] = 0.5;
            reference_population_digest(&[first, third]).unwrap()
        };
        assert_eq!(left, reordered);
        assert_ne!(left, changed);
    }

    #[test]
    fn novelty_uses_only_separate_reference_population() {
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![1, 2, 3],
            ..GenomeEvaluationConfig::default()
        };
        let pilot = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        let references = pilot.trials.iter()
            .map(|trial| mean_fingerprint_array(&trial.structural_fingerprint).unwrap())
            .collect::<Vec<_>>();
        let evaluation = evaluate_genome(&genome, &MusicalIntent::default(), &config, &references).unwrap();
        assert!(evaluation.objectives.novelty_distance.is_some());
    }

    #[test]
    fn evaluation_digest_binds_the_objectives_and_reference_population() {
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![1, 2, 3],
            ..GenomeEvaluationConfig::default()
        };
        let baseline = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        assert!(baseline.verify_digest());
        let mut tampered = baseline.clone();
        tampered.objectives.within_genome_diversity += 0.1;
        assert!(!tampered.verify_digest());
    }

    #[test]
    fn archive_rejects_candidates_evaluated_against_a_different_reference_population() {
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![1, 2, 3],
            ..GenomeEvaluationConfig::default()
        };
        let without_reference = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        let references = without_reference.trials.iter()
            .map(|trial| mean_fingerprint_array(&trial.structural_fingerprint).unwrap())
            .collect::<Vec<_>>();
        let with_reference = evaluate_genome(
            &genome,
            &MusicalIntent::default(),
            &config,
            &references,
        ).unwrap();
        let mut archive = GenomeArchive::new(GenomeArchiveConfig::default()).unwrap();
        assert_eq!(
            archive.insert(without_reference).disposition,
            ArchiveInsertDisposition::Accepted
        );
        assert_eq!(
            archive.insert(with_reference).disposition,
            ArchiveInsertDisposition::RejectedReferencePopulationMismatch
        );
    }

    #[test]
    fn archive_digest_verifies_and_detects_objective_tampering() {
        let mut archive = GenomeArchive::new(GenomeArchiveConfig::default()).unwrap();
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![1, 2, 3],
            ..GenomeEvaluationConfig::default()
        };
        let evaluation = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        let receipt = archive.insert(evaluation);
        assert_eq!(receipt.disposition, ArchiveInsertDisposition::Accepted);
        assert!(archive.verify_digest());

        let mut tampered = archive.clone();
        tampered.entries[0].objectives.within_genome_diversity += 0.1;
        assert!(!tampered.verify_digest());
    }

    #[test]
    fn pareto_archive_keeps_non_dominated_candidates_without_a_scalar_reward() {
        let mut archive = GenomeArchive::new(GenomeArchiveConfig::default()).unwrap();
        let genome = classical_genome();
        let config = GenomeEvaluationConfig {
            seeds: vec![1, 2, 3],
            ..GenomeEvaluationConfig::default()
        };
        let first = evaluate_genome(&genome, &MusicalIntent::default(), &config, &[]).unwrap();
        let receipt = archive.insert(first);
        assert_eq!(receipt.disposition, ArchiveInsertDisposition::Accepted);
        assert_eq!(archive.entries.len(), 1);
    }
}
