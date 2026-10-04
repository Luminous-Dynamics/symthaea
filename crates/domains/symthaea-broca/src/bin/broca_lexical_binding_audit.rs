// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic corpus audit for the lexical/morphosyntactic Broca boundary.
//!
//! This binary intentionally uses fixed semantic fixtures and explicit lexical data.
//! It does not generate words from scratch. Positive cases must bind all semantic
//! constituents; negative cases must fail closed when lineage, coverage, or agreement
//! evidence is incomplete.

use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use symthaea_broca::{
    AgreementConstraint, ConstituentDependency, GrammaticalFunction, LanguageRuleBinding,
    LanguageRuleStatus, LexemeBinding, LexicalBindingError, LexicalMorphosyntacticBinding,
    LexicalSource, LinguisticFrame, MorphologicalFeature, SpeechPlan, StructuredDecoder,
    ThoughtChannels,
};
use symthaea_core::genesis::GenesisSeed;

const SCHEMA_VERSION: u32 = 1;
const SEED: &str = "broca-lexical-binding-audit-v1";

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    evidence_level: &'static str,
    seed_phrase: &'static str,
    corpus_cases: usize,
    positive_passes: usize,
    negative_passes: usize,
    semantic_coverage_pass: bool,
    provenance_pass: bool,
    grammar_pass: bool,
    no_invention_pass: bool,
    unsupported_rule_unbound_pass: bool,
    stable_repeat_pass: bool,
    cases: Vec<AuditCase>,
}

#[derive(Debug, Serialize)]
struct AuditCase {
    name: &'static str,
    polarity: &'static str,
    passed: bool,
    expected_failure: Option<&'static str>,
    observed_error: Option<String>,
    provenance_token: Option<String>,
    semantic_items: usize,
    inserted_function_words: usize,
}

fn statement_frame() -> LinguisticFrame {
    let genesis = GenesisSeed::from_phrase(SEED);
    let decoder = StructuredDecoder::new(&genesis);
    let channels = ThoughtChannels::with_intent(4);
    let readout = decoder.decode(&channels);
    LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout))
}

fn question_frame() -> LinguisticFrame {
    let genesis = GenesisSeed::from_phrase(SEED);
    let decoder = StructuredDecoder::new(&genesis);
    let channels = ThoughtChannels::with_intent(3);
    let readout = decoder.decode(&channels);
    SpeechPlan::from_readout(&channels, &readout).into()
}

fn explicit_language(status: LanguageRuleStatus) -> LanguageRuleBinding {
    match status {
        LanguageRuleStatus::Bound => LanguageRuleBinding {
            language_tag: "en".into(),
            status,
            rule_id: Some("audit-en-svo-v1".into()),
            provenance: Some("audit-fixture:english-rules-v1".into()),
            unbound_reason: None,
        },
        LanguageRuleStatus::Unbound => LanguageRuleBinding {
            language_tag: "xx".into(),
            status,
            rule_id: None,
            provenance: None,
            unbound_reason: Some("audit-no-supported-language-rule-set".into()),
        },
    }
}

fn grammatical_function(role: &str) -> GrammaticalFunction {
    match role {
        "AGENT" => GrammaticalFunction::Subject,
        "ACTION" => GrammaticalFunction::Verb,
        "PATIENT" => GrammaticalFunction::Object,
        "PREDICATE" | "EVALUATOR" => GrammaticalFunction::Predicate,
        "LOCATION" | "TIME" | "REASON" => GrammaticalFunction::Oblique,
        other => GrammaticalFunction::Other(other.to_string()),
    }
}

fn base_bindings(frame: &LinguisticFrame) -> Vec<LexemeBinding> {
    frame
        .constituents
        .iter()
        .enumerate()
        .map(|(position, slot)| {
            let morphology = match slot.role.as_str() {
                "AGENT" | "ACTION" => vec![
                    MorphologicalFeature {
                        category: "person".into(),
                        value: "1".into(),
                    },
                    MorphologicalFeature {
                        category: "number".into(),
                        value: "singular".into(),
                    },
                ],
                _ => Vec::new(),
            };

            LexemeBinding {
                position,
                source: LexicalSource::SemanticConstituent {
                    role: slot.role.clone(),
                    prime: slot.prime.clone(),
                },
                lemma: slot.prime.to_ascii_lowercase(),
                lexeme_id: format!("audit:en:{}:{}", slot.role, slot.prime),
                grammatical_function: grammatical_function(&slot.role),
                morphology,
                morphophonological_form: Some(slot.prime.to_ascii_lowercase()),
                provenance: "audit-fixture:lexicon-v1".into(),
                semantic_payload: true,
            }
        })
        .collect()
}

fn dependencies(bindings: &[LexemeBinding]) -> Vec<ConstituentDependency> {
    let action = bindings
        .iter()
        .position(|item| matches!(&item.source, LexicalSource::SemanticConstituent { role, .. } if role == "ACTION"));
    let agent = bindings
        .iter()
        .position(|item| matches!(&item.source, LexicalSource::SemanticConstituent { role, .. } if role == "AGENT"));
    let patient = bindings
        .iter()
        .position(|item| matches!(&item.source, LexicalSource::SemanticConstituent { role, .. } if role == "PATIENT"));

    let mut edges = Vec::new();
    if let (Some(governor), Some(dependent)) = (action, agent) {
        edges.push(ConstituentDependency {
            governor_position: governor,
            dependent_position: dependent,
            relation: "subject".into(),
        });
    }
    if let (Some(governor), Some(dependent)) = (action, patient) {
        edges.push(ConstituentDependency {
            governor_position: governor,
            dependent_position: dependent,
            relation: "object".into(),
        });
    }
    edges
}

fn agreement(bindings: &[LexemeBinding]) -> Vec<AgreementConstraint> {
    let agent = bindings
        .iter()
        .position(|item| matches!(&item.source, LexicalSource::SemanticConstituent { role, .. } if role == "AGENT"));
    let action = bindings
        .iter()
        .position(|item| matches!(&item.source, LexicalSource::SemanticConstituent { role, .. } if role == "ACTION"));

    match (agent, action) {
        (Some(controller), Some(target)) => vec![AgreementConstraint {
            controller_position: controller,
            target_position: target,
            feature: MorphologicalFeature {
                category: "person".into(),
                value: "1".into(),
            },
        }],
        _ => Vec::new(),
    }
}

fn bind(
    frame: &LinguisticFrame,
    language: LanguageRuleBinding,
    bindings: Vec<LexemeBinding>,
) -> Result<LexicalMorphosyntacticBinding, LexicalBindingError> {
    LexicalMorphosyntacticBinding::new(
        frame,
        language,
        bindings,
        Vec::new(),
        Vec::new(),
    )
}

fn run_positive(
    name: &'static str,
    frame: &LinguisticFrame,
    language: LanguageRuleBinding,
    mut bindings: Vec<LexemeBinding>,
    add_function_word: bool,
) -> AuditCase {
    if add_function_word {
        bindings.insert(
            0,
            LexemeBinding {
                position: 0,
                source: LexicalSource::InsertedFunctionWord {
                    insertion_reason: "audit-fixture:explicit-function-word".into(),
                },
                lemma: "the".into(),
                lexeme_id: "audit:en:function:the".into(),
                grammatical_function: GrammaticalFunction::FunctionWord,
                morphology: Vec::new(),
                morphophonological_form: Some("the".into()),
                provenance: "audit-fixture:lexicon-v1".into(),
                semantic_payload: false,
            },
        );
    }

    for (position, item) in bindings.iter_mut().enumerate() {
        item.position = position;
    }

    match LexicalMorphosyntacticBinding::new(
        frame,
        language,
        bindings.clone(),
        dependencies(&bindings),
        agreement(&bindings),
    ) {
        Ok(binding) => AuditCase {
            name,
            polarity: "positive",
            passed: binding.validate_against_frame(frame).is_ok(),
            expected_failure: None,
            observed_error: None,
            provenance_token: Some(binding.provenance_token()),
            semantic_items: bindings
                .iter()
                .filter(|item| item.semantic_payload)
                .count(),
            inserted_function_words: bindings
                .iter()
                .filter(|item| !item.semantic_payload)
                .count(),
        },
        Err(error) => AuditCase {
            name,
            polarity: "positive",
            passed: false,
            expected_failure: None,
            observed_error: Some(error.to_string()),
            provenance_token: None,
            semantic_items: bindings
                .iter()
                .filter(|item| item.semantic_payload)
                .count(),
            inserted_function_words: bindings
                .iter()
                .filter(|item| !item.semantic_payload)
                .count(),
        },
    }
}

fn run_negative(
    name: &'static str,
    frame: &LinguisticFrame,
    language: LanguageRuleBinding,
    bindings: Vec<LexemeBinding>,
    expected_failure: &'static str,
) -> AuditCase {
    match bind(frame, language, bindings) {
        Ok(_) => AuditCase {
            name,
            polarity: "negative",
            passed: false,
            expected_failure: Some(expected_failure),
            observed_error: None,
            provenance_token: None,
            semantic_items: 0,
            inserted_function_words: 0,
        },
        Err(error) => AuditCase {
            name,
            polarity: "negative",
            passed: error.to_string().contains(expected_failure),
            expected_failure: Some(expected_failure),
            observed_error: Some(error.to_string()),
            provenance_token: None,
            semantic_items: 0,
            inserted_function_words: 0,
        },
    }
}

fn parse_json_out() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--json-out" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn write_report(path: Option<&Path>, report: &AuditReport) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(path, serde_json::to_vec_pretty(report)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let statement = statement_frame();
    let question = question_frame();

    let statement_bindings = base_bindings(&statement);
    let question_bindings = base_bindings(&question);

    let mut cases = Vec::new();
    cases.push(run_positive(
        "statement-bound",
        &statement,
        explicit_language(LanguageRuleStatus::Bound),
        statement_bindings.clone(),
        false,
    ));
    cases.push(run_positive(
        "statement-function-word",
        &statement,
        explicit_language(LanguageRuleStatus::Bound),
        statement_bindings.clone(),
        true,
    ));
    cases.push(run_positive(
        "question-rule-unbound",
        &question,
        explicit_language(LanguageRuleStatus::Unbound),
        question_bindings.clone(),
        false,
    ));

    let mut missing = statement_bindings.clone();
    missing.pop();
    cases.push(run_negative(
        "missing-semantic-source",
        &statement,
        explicit_language(LanguageRuleStatus::Bound),
        missing,
        "not lexically covered",
    ));

    let mut agreement_gap = statement_bindings.clone();
    if let Some(action) = agreement_gap.iter_mut().find(|item| {
        matches!(&item.source, LexicalSource::SemanticConstituent { role, .. } if role == "ACTION")
    }) {
        action.morphology.clear();
    }
    cases.push({
        let result = LexicalMorphosyntacticBinding::new(
            &statement,
            explicit_language(LanguageRuleStatus::Bound),
            agreement_gap.clone(),
            dependencies(&agreement_gap),
            agreement(&statement_bindings),
        );
        let (passed, error) = match result {
            Err(error) => (error.to_string().contains("agreement feature"), Some(error.to_string())),
            Ok(_) => (false, None),
        };
        AuditCase {
            name: "agreement-gap",
            polarity: "negative",
            passed,
            expected_failure: Some("agreement feature"),
            observed_error: error,
            provenance_token: None,
            semantic_items: agreement_gap.iter().filter(|item| item.semantic_payload).count(),
            inserted_function_words: 0,
        }
    });

    let mut stale = LexicalMorphosyntacticBinding::new(
        &statement,
        explicit_language(LanguageRuleStatus::Bound),
        statement_bindings.clone(),
        dependencies(&statement_bindings),
        agreement(&statement_bindings),
    )
    .expect("base fixture must bind");
    stale.source_frame_grounding.push_str(";tampered");
    cases.push(match stale.validate_against_frame(&statement) {
        Err(error) => AuditCase {
            name: "stale-lineage",
            polarity: "negative",
            passed: matches!(error, LexicalBindingError::UpstreamMismatch),
            expected_failure: Some("upstream frame"),
            observed_error: Some(error.to_string()),
            provenance_token: None,
            semantic_items: statement_bindings.len(),
            inserted_function_words: 0,
        },
        Ok(_) => AuditCase {
            name: "stale-lineage",
            polarity: "negative",
            passed: false,
            expected_failure: Some("upstream frame"),
            observed_error: None,
            provenance_token: None,
            semantic_items: statement_bindings.len(),
            inserted_function_words: 0,
        },
    });

    let abstention = {
        let genesis = GenesisSeed::from_phrase(SEED);
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = ThoughtChannels::with_intent(7);
        channels.set_epistemic(3.0);
        let readout = decoder.decode(&channels);
        SpeechPlan::from_readout(&channels, &readout)
    };
    let abstention_frame = LinguisticFrame::from_speech_plan(&abstention);
    let abstention_result = bind(
        &abstention_frame,
        explicit_language(LanguageRuleStatus::Bound),
        vec![LexemeBinding {
            position: 0,
            source: LexicalSource::SemanticConstituent {
                role: "AGENT".into(),
                prime: "I".into(),
            },
            lemma: "i".into(),
            lexeme_id: "audit:en:pronoun:I".into(),
            grammatical_function: GrammaticalFunction::Subject,
            morphology: Vec::new(),
            morphophonological_form: Some("i".into()),
            provenance: "audit-fixture:lexicon-v1".into(),
            semantic_payload: true,
        }],
    );
    cases.push(match abstention_result {
        Err(error) => AuditCase {
            name: "abstention-cannot-bind",
            polarity: "negative",
            passed: true,
            expected_failure: Some("linguistic"),
            observed_error: Some(error.to_string()),
            provenance_token: None,
            semantic_items: 1,
            inserted_function_words: 0,
        },
        Ok(_) => AuditCase {
            name: "abstention-cannot-bind",
            polarity: "negative",
            passed: false,
            expected_failure: Some("linguistic"),
            observed_error: None,
            provenance_token: None,
            semantic_items: 1,
            inserted_function_words: 0,
        },
    });

    let repeat_a = {
        let frame = statement_frame();
        let bindings = base_bindings(&frame);
        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            explicit_language(LanguageRuleStatus::Bound),
            bindings.clone(),
            dependencies(&bindings),
            agreement(&bindings),
        )?;
        binding.provenance_token()
    };
    let repeat_b = {
        let frame = statement_frame();
        let bindings = base_bindings(&frame);
        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            explicit_language(LanguageRuleStatus::Bound),
            bindings.clone(),
            dependencies(&bindings),
            agreement(&bindings),
        )?;
        binding.provenance_token()
    };
    let stable_repeat_pass = repeat_a == repeat_b;

    let positive_passes = cases
        .iter()
        .filter(|case| case.polarity == "positive" && case.passed)
        .count();
    let negative_passes = cases
        .iter()
        .filter(|case| case.polarity == "negative" && case.passed)
        .count();

    let semantic_coverage_pass = cases
        .iter()
        .find(|case| case.name == "statement-bound")
        .is_some_and(|case| case.passed && case.semantic_items > 0);
    let provenance_pass = cases.iter().all(|case| {
        case.polarity == "negative"
            || (case.passed && case.provenance_token.as_ref().is_some_and(|token| token.len() == 64))
    });
    let grammar_pass = cases
        .iter()
        .filter(|case| case.polarity == "positive")
        .all(|case| case.passed);
    let no_invention_pass = cases
        .iter()
        .find(|case| case.name == "missing-semantic-source")
        .is_some_and(|case| case.passed)
        && cases
            .iter()
            .find(|case| case.name == "agreement-gap")
            .is_some_and(|case| case.passed);
    let unsupported_rule_unbound_pass = cases
        .iter()
        .find(|case| case.name == "question-rule-unbound")
        .is_some_and(|case| case.passed);

    let report = AuditReport {
        schema_version: SCHEMA_VERSION,
        evidence_level: "deterministic-lexical-morphosyntactic-corpus-v1",
        seed_phrase: SEED,
        corpus_cases: cases.len(),
        positive_passes,
        negative_passes,
        semantic_coverage_pass,
        provenance_pass,
        grammar_pass,
        no_invention_pass,
        unsupported_rule_unbound_pass,
        stable_repeat_pass,
        cases,
    };

    write_report(parse_json_out().as_deref(), &report)?;

    let all_pass = report.semantic_coverage_pass
        && report.provenance_pass
        && report.grammar_pass
        && report.no_invention_pass
        && report.unsupported_rule_unbound_pass
        && report.stable_repeat_pass;

    println!("{}", serde_json::to_string_pretty(&report)?);

    if all_pass {
        Ok(())
    } else {
        anyhow::bail!("lexical/morphosyntactic corpus audit failed")
    }
}
