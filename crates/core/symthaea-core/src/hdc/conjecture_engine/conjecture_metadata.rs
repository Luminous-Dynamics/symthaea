use super::*;
use symthaea_types::{
    ModelMaturity, PhysicalType, ScientificHypothesisHandoff,
};

fn canonical_expr(expr: &Expr, out: &mut String) {
    match expr {
        Expr::Var(name) => {
            out.push_str("V:");
            out.push_str(&name.len().to_string());
            out.push(':');
            out.push_str(name);
            out.push(';');
        }
        Expr::Const(value) => {
            out.push_str("C:");
            out.push_str(&value.to_bits().to_string());
            out.push(';');
        }
        Expr::BinOp(op, left, right) => {
            out.push_str("B:");
            out.push_str(match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                BinOp::Div => "div",
                BinOp::Pow => "pow",
            });
            out.push('(');
            canonical_expr(left, out);
            out.push(',');
            canonical_expr(right, out);
            out.push_str(");");
        }
        Expr::Func(func, arg) => {
            out.push_str("F:");
            out.push_str(match func {
                UnaryFn::Sqrt => "sqrt",
                UnaryFn::Log => "log",
                UnaryFn::Exp => "exp",
                UnaryFn::Sin => "sin",
                UnaryFn::Cos => "cos",
                UnaryFn::Abs => "abs",
                UnaryFn::Floor => "floor",
            });
            out.push('(');
            canonical_expr(arg, out);
            out.push_str(");");
        }
        Expr::Sum(body, variable) => {
            out.push_str("S:");
            out.push_str(&variable.len().to_string());
            out.push(':');
            out.push_str(variable);
            out.push('(');
            canonical_expr(body, out);
            out.push_str(");");
        }
    }
}

fn candidate_identity_digest(conjecture: &Conjecture) -> String {
    // Candidate identity binds the exact discovered mathematical object and
    // source/domain identity. Mutable confidence and downstream evidence are
    // deliberately excluded so qualification cannot retroactively mutate it.
    let mut canonical = String::from("candidate-v1|source=");
    canonical.push_str(&conjecture.source.len().to_string());
    canonical.push(':');
    canonical.push_str(&conjecture.source);
    canonical.push_str("|domain=");
    canonical.push_str(match conjecture.domain {
        MathDomain::NumberTheory => "number_theory",
        MathDomain::Combinatorics => "combinatorics",
        MathDomain::AlgebraicComplexity => "algebraic_complexity",
        MathDomain::DynamicalSystems => "dynamical_systems",
        MathDomain::SpectralAnalysis => "spectral_analysis",
        MathDomain::Chemistry => "chemistry",
        MathDomain::Biology => "biology",
        MathDomain::Ecology => "ecology",
        MathDomain::Economics => "economics",
        MathDomain::Physics => "physics",
        MathDomain::InformationTheory => "information_theory",
    });
    canonical.push_str("|complexity=");
    canonical.push_str(&conjecture.complexity.to_string());
    canonical.push_str("|formula=");
    canonical_expr(&conjecture.formula, &mut canonical);
    blake3::hash(canonical.as_bytes()).to_hex().to_string()
}

impl Conjecture {
    /// Export this conjecture as an evidence-neutral scientific hypothesis.
    ///
    /// EUREKA is intentionally not consulted here: this is a one-way identity
    /// handoff, not a qualification call.
    pub fn scientific_hypothesis_handoff(
        &self,
        physical_type: &PhysicalType,
        model_maturity: ModelMaturity,
        source_observation_digest: impl Into<String>,
        search_configuration_digest: impl Into<String>,
        search_seed: u64,
        discovery_manifest_digest: impl Into<String>,
        non_claims: Vec<String>,
    ) -> ScientificHypothesisHandoff {
        ScientificHypothesisHandoff::new(
            candidate_identity_digest(self),
            physical_type,
            model_maturity,
            source_observation_digest,
            search_configuration_digest,
            search_seed,
            discovery_manifest_digest,
            self.complexity as u64,
            "ramanujan",
            non_claims,
        )
    }
}

pub fn conjecture_has_verified_eml_backend(conjecture: &Conjecture) -> bool {
    conjecture.eml_compiled.is_some()
        || conjecture.eml_constructive_compiled.is_some()
        || conjecture.eml_verified_real == Some(true)
        || conjecture.eml_verified_complex == Some(true)
        || conjecture.eml_verified_constructive_real == Some(true)
}

pub(crate) fn preferred_eml_backend_rank(conjecture: &Conjecture) -> u8 {
    match conjecture.preferred_eml_backend() {
        Some(PreferredEmlBackend::StrictRealAndComplex) => {
            if conjecture
                .eml_real_domain
                .is_some_and(EmlRealDomainAssumption::is_unconstrained)
            {
                0
            } else {
                1
            }
        }
        Some(PreferredEmlBackend::StrictReal) => {
            if conjecture
                .eml_real_domain
                .is_some_and(EmlRealDomainAssumption::is_unconstrained)
            {
                2
            } else {
                3
            }
        }
        Some(PreferredEmlBackend::StrictComplex) => 4,
        Some(PreferredEmlBackend::StrictUnverified) => 5,
        Some(PreferredEmlBackend::ConstructiveReal) => 6,
        None => 7,
    }
}

pub(crate) fn compare_conjectures_for_selection(
    a: &Conjecture,
    b: &Conjecture,
) -> std::cmp::Ordering {
    a.training_mse
        .partial_cmp(&b.training_mse)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| preferred_eml_backend_rank(a).cmp(&preferred_eml_backend_rank(b)))
        .then_with(|| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.complexity.cmp(&b.complexity))
        .then_with(|| a.formula_str.cmp(&b.formula_str))
}

pub(crate) fn dedupe_conjectures_by_preferred_backend(conjectures: &mut Vec<Conjecture>) {
    let mut seen_backend_keys = std::collections::HashSet::new();
    conjectures.retain(|conjecture| {
        if let Some(canonical) = conjecture.preferred_eml_canonical_form() {
            let key = format!("{}::{canonical}", conjecture.source);
            seen_backend_keys.insert(key)
        } else {
            true
        }
    });
}

pub(crate) fn finalize_conjectures_after_eml(conjectures: &mut Vec<Conjecture>) {
    conjectures.sort_by(compare_conjectures_for_selection);
    dedupe_conjectures_by_preferred_backend(conjectures);
}

pub(crate) fn elevate_macro_promotion_tier(conjecture: &mut Conjecture, tier: MacroPromotionTier) {
    if tier > conjecture.macro_promotion_tier {
        conjecture.macro_promotion_tier = tier;
    }
}

pub fn attach_eml_metadata(conjecture: &mut Conjecture) {
    let cache_key = format!("{}", conjecture.formula);
    if let Some(snapshot) = EML_METADATA_CACHE.read().get(&cache_key).cloned() {
        snapshot.apply_to(conjecture);
        return;
    }

    let mut snapshot = EmlMetadataSnapshot::default();

    if let Ok(compiled) = eml::compile_expr(&conjecture.formula) {
        let metrics = compiled.metrics();
        let real_report =
            eml::verify_expr_compilation(&conjecture.formula, &compiled, EmlEvalMode::RealIeee);
        let real_ok = real_report.passed;
        let complex_ok = eml::verify_expr_compilation(
            &conjecture.formula,
            &compiled,
            EmlEvalMode::ComplexPrincipal,
        )
        .passed;
        snapshot.eml_metrics = Some(metrics);
        snapshot.eml_verified_real = Some(real_ok);
        snapshot.eml_real_domain = real_report.real_domain_assumption.filter(|_| real_ok);
        snapshot.eml_verified_complex = Some(complex_ok);
        if real_ok || complex_ok {
            snapshot.eml_compiled = Some(compiled);
        }
    }

    if let Ok(compiled) = eml::compile_expr_constructive(&conjecture.formula) {
        let metrics = compiled.metrics();
        let constructive_ok = eml::verify_expr_compilation(
            &conjecture.formula,
            &compiled,
            EmlEvalMode::RealConstructive,
        )
        .passed;
        snapshot.eml_constructive_metrics = Some(metrics);
        snapshot.eml_verified_constructive_real = Some(constructive_ok);
        if constructive_ok {
            snapshot.eml_constructive_compiled = Some(compiled);
        }
    }

    snapshot.apply_to(conjecture);
    EML_METADATA_CACHE.write().insert(cache_key, snapshot);
}

#[derive(Debug, Clone, Default)]
struct EmlMetadataSnapshot {
    eml_compiled: Option<EmlExpr>,
    eml_metrics: Option<EmlMetrics>,
    eml_verified_real: Option<bool>,
    eml_real_domain: Option<EmlRealDomainAssumption>,
    eml_verified_complex: Option<bool>,
    eml_constructive_compiled: Option<EmlExpr>,
    eml_constructive_metrics: Option<EmlMetrics>,
    eml_verified_constructive_real: Option<bool>,
}

impl EmlMetadataSnapshot {
    fn apply_to(&self, conjecture: &mut Conjecture) {
        conjecture.eml_compiled = self.eml_compiled.clone();
        conjecture.eml_metrics = self.eml_metrics;
        conjecture.eml_verified_real = self.eml_verified_real;
        conjecture.eml_real_domain = self.eml_real_domain;
        conjecture.eml_verified_complex = self.eml_verified_complex;
        conjecture.eml_constructive_compiled = self.eml_constructive_compiled.clone();
        conjecture.eml_constructive_metrics = self.eml_constructive_metrics;
        conjecture.eml_verified_constructive_real = self.eml_verified_constructive_real;
    }
}

static EML_METADATA_CACHE: Lazy<RwLock<std::collections::HashMap<String, EmlMetadataSnapshot>>> =
    Lazy::new(|| RwLock::new(std::collections::HashMap::new()));

#[cfg(test)]
#[cfg(test)]
mod hypothesis_handoff_tests {
    use super::*;
    use symthaea_types::{PhysicalDimension, QuantityKind};

    #[test]
    fn handoff_is_independent_of_mutable_confidence() {
        let formula = Expr::Var("n".into());
        let mut a = Conjecture {
            formula: formula.clone(),
            formula_str: "n".into(),
            source: "test-seq".into(),
            domain: MathDomain::Physics,
            training_mse: 0.0,
            complexity: 1,
            fitness: 0.0,
            status: ConjectureStatus::Proposed,
            confidence: 0.1,
            macro_promotion_tier: MacroPromotionTier::RecurrentNumerical,
            eml_compiled: None,
            eml_metrics: None,
            eml_verified_real: None,
            eml_real_domain: None,
            eml_verified_complex: None,
            eml_constructive_compiled: None,
            eml_constructive_metrics: None,
            eml_verified_constructive_real: None,
        };
        let mut b = a.clone();
        b.confidence = 0.99;

        let energy = PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY);
        let ha = a.scientific_hypothesis_handoff(
            &energy,
            ModelMaturity::ResearchPrototype,
            "obs",
            "search",
            7,
            "manifest",
            vec!["not qualification".into()],
        );
        let hb = b.scientific_hypothesis_handoff(
            &energy,
            ModelMaturity::ResearchPrototype,
            "obs",
            "search",
            7,
            "manifest",
            vec!["not qualification".into()],
        );
        assert_eq!(ha.candidate_digest, hb.candidate_digest);
        assert_eq!(ha.physical_type_digest, energy.digest_hex());
        assert!(ha.validate().is_ok());
        a.confidence = 0.77;
        assert_eq!(
            a.scientific_hypothesis_handoff(
                &energy,
                ModelMaturity::ResearchPrototype,
                "obs",
                "search",
                7,
                "manifest",
                vec!["not qualification".into()],
            )
            .candidate_digest,
            ha.candidate_digest
        );
    }
}

pub(crate) fn clear_eml_metadata_cache() {
    EML_METADATA_CACHE.write().clear();
}

/// Beta-distribution confidence tracker for conjectures.
#[derive(Debug, Clone)]
pub struct BayesianConfidence {
    pub alpha: f64,
    pub beta: f64,
}

impl BayesianConfidence {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn record_success(&mut self, weight: f64) {
        self.alpha += weight.max(0.0);
    }

    pub fn record_failure(&mut self, weight: f64) {
        self.beta += weight.max(0.0);
    }
}

impl ConjectureEngine {
    /// Verify all conjectures using Bayesian confidence updating.
    pub fn verify_bayesian(&mut self, max_n: usize) {
        let observations = self.observations.clone();

        for conjecture in &mut self.conjectures {
            let mut confidence = BayesianConfidence::new();

            if conjecture.training_mse < 1e-6 {
                confidence.record_success(1.0);
            } else if conjecture.training_mse < 1.0 {
                confidence.record_success(0.5);
            }

            if let Some(sequence) = observations.iter().find(|s| s.name == conjecture.source) {
                let (_, test) = sequence.train_test_split();
                if !test.is_empty() {
                    let test_mse = compute_mse(&conjecture.formula, &test);
                    if test_mse.is_finite() && test_mse < conjecture.training_mse * 2.0 {
                        confidence.record_success(1.0);
                        conjecture.status = ConjectureStatus::NumericallyTested { test_mse };
                        elevate_macro_promotion_tier(
                            conjecture,
                            MacroPromotionTier::RecurrentNumerical,
                        );
                    } else if test_mse.is_finite() {
                        confidence.record_failure(1.0);
                    }
                }
            }

            let mut all_match = true;
            let mut checked = 0;
            if let Some(sequence) = observations.iter().find(|s| s.name == conjecture.source) {
                for &(x, y) in &sequence.data {
                    if (x as usize) < 1 || x > max_n as f64 {
                        continue;
                    }
                    let predicted = conjecture.formula.eval(&[("n", x)]);
                    if !predicted.is_finite() || (predicted - y).abs() > y.abs() * 0.01 + 1e-10 {
                        all_match = false;
                        conjecture.status = ConjectureStatus::Refuted { counterexample: x };
                        conjecture.macro_promotion_tier = MacroPromotionTier::Quarantined;
                        confidence.record_failure(5.0);
                        break;
                    }
                    checked += 1;
                }
            }

            if all_match && checked > 10 {
                confidence.record_success(3.0);
                conjecture.status = ConjectureStatus::BoundedChecked {
                    checked_points: checked,
                    max_n,
                };
                elevate_macro_promotion_tier(conjecture, MacroPromotionTier::RecurrentNumerical);
            }

            conjecture.confidence = confidence.mean();
        }
    }
}

pub(crate) fn identify_constant(val: f64) -> Option<String> {
    let candidates: &[(&str, f64)] = &[
        ("π", std::f64::consts::PI),
        ("e", std::f64::consts::E),
        ("φ", (1.0 + 5.0_f64.sqrt()) / 2.0),
        ("1/e", 1.0 / std::f64::consts::E),
        ("√2", std::f64::consts::SQRT_2),
        ("1/√2", std::f64::consts::FRAC_1_SQRT_2),
        ("ln(2)", std::f64::consts::LN_2),
        ("π²/6", std::f64::consts::PI * std::f64::consts::PI / 6.0),
        ("1/π", std::f64::consts::FRAC_1_PI),
        ("2/π", std::f64::consts::FRAC_2_PI),
        ("1/√π", 1.0 / std::f64::consts::PI.sqrt()),
        ("√3", 3.0_f64.sqrt()),
        ("1/√3", 1.0 / 3.0_f64.sqrt()),
        ("γ (Euler-Mascheroni)", 0.5772156649015329),
        ("Catalan", 0.915_965_594_177_219),
        ("Apéry ζ(3)", 1.2020569031595942),
    ];
    for (name, known) in candidates {
        if (val - known).abs() < known.abs().max(1.0) * 1e-4 {
            return Some(name.to_string());
        }
    }
    for d in 1..=12 {
        for n in 0..=d * 3 {
            let frac = n as f64 / d as f64;
            if (val - frac).abs() < 1e-6 && d > 1 {
                return Some(format!("{}/{}", n, d));
            }
        }
    }
    None
}

pub fn annotate_conjecture(conjecture: &Conjecture) -> String {
    let mut annotations = Vec::new();

    let consts = collect_constants(&conjecture.formula);
    for c in &consts {
        if let Some(name) = identify_constant(*c) {
            annotations.push(format!("{}≈{:.4}", name, c));
        }
    }

    let limit = conjecture.formula.eval(&[("n", 1000.0)]);
    if limit.is_finite()
        && let Some(name) = identify_constant(limit)
    {
        annotations.push(format!("limit→{}", name));
    }

    if let Some(tag) = eml_backend_annotation(conjecture) {
        annotations.push(tag);
    }

    if annotations.is_empty() {
        String::new()
    } else {
        format!(" [{}]", annotations.join(", "))
    }
}

fn eml_backend_annotation(conjecture: &Conjecture) -> Option<String> {
    conjecture
        .preferred_eml_backend()
        .map(|backend| match backend {
            PreferredEmlBackend::StrictRealAndComplex => format!(
                "eml=strict:real+complex{}",
                eml_real_domain_annotation_suffix(conjecture)
            ),
            PreferredEmlBackend::StrictReal => {
                format!(
                    "eml=strict:real{}",
                    eml_real_domain_annotation_suffix(conjecture)
                )
            }
            PreferredEmlBackend::StrictComplex => "eml=strict:complex".to_string(),
            PreferredEmlBackend::StrictUnverified => "eml=strict".to_string(),
            PreferredEmlBackend::ConstructiveReal => "eml=constructive".to_string(),
        })
}

pub(crate) fn eml_backend_label(conjecture: &Conjecture) -> Option<String> {
    conjecture
        .preferred_eml_backend()
        .map(|backend| match backend {
            PreferredEmlBackend::StrictRealAndComplex => format!(
                "EML strict real+complex{}",
                eml_real_domain_label_suffix(conjecture)
            ),
            PreferredEmlBackend::StrictReal => {
                format!(
                    "EML strict real{}",
                    eml_real_domain_label_suffix(conjecture)
                )
            }
            PreferredEmlBackend::StrictComplex => "EML strict complex".to_string(),
            PreferredEmlBackend::StrictUnverified => "EML strict".to_string(),
            PreferredEmlBackend::ConstructiveReal => "EML constructive".to_string(),
        })
}

fn eml_real_domain_annotation_suffix(conjecture: &Conjecture) -> String {
    match conjecture.eml_real_domain {
        Some(domain) if !domain.is_unconstrained() => format!("@{}", domain.short_tag()),
        _ => String::new(),
    }
}

fn eml_real_domain_label_suffix(conjecture: &Conjecture) -> String {
    match conjecture.eml_real_domain {
        Some(domain) if !domain.is_unconstrained() => format!(" ({})", domain.display_label()),
        _ => String::new(),
    }
}
