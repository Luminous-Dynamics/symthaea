// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sequencer Training — Generate training data and train the CfC code sequencer
//!
//! Approach: Instead of manually mapping 250+ keyword patterns to plan sequences,
//! we derive the correct plan for each purpose by analyzing what the emitter
//! actually needs. The emitter's `infer_rust_body()` produces code from purpose
//! strings — we analyze the resulting code to determine which PlanActions were
//! effectively executed.
//!
//! # Training Pipeline
//!
//! 1. `generate_training_data()` — produces (intent_hv, target_actions) pairs
//! 2. `train_sequencer()` — BPTT training loop with early stopping
//! 3. `evaluate_sequencer()` — measure accuracy on held-out data

use symthaea_core::hdc::ContinuousHV;

use super::sequencer_benchmark::{BenchmarkTask, TaskCategory, benchmark_tasks};
use crate::dynamics::cfc_code_sequencer::{CfCCodeSequencer, CodePlanStep, PlanAction};
use crate::hdc::code_encoder::CodeHDEncoder;

/// A single training example: intent HV → target plan action sequence
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub purpose: String,
    pub intent_hv: ContinuousHV,
    pub target_actions: Vec<PlanAction>,
    pub category: TaskCategory,
}

/// Infer the correct PlanAction sequence for a given task based on its
/// signature and purpose.
///
/// This analyzes what structural elements the function needs:
/// - Has parameters? → AddParameter
/// - Has return type? → SetReturnType
/// - Contains error handling (Result)? → AddErrorHandling
/// - Needs imports? → AddImport
/// - etc.
fn infer_target_plan(task: &BenchmarkTask) -> Vec<PlanAction> {
    let mut actions = Vec::new();
    let sig = &task.signature;
    let purpose = task.purpose.to_lowercase();

    // Every function starts with DefineFunction
    actions.push(PlanAction::DefineFunction);

    // Count parameters (everything between '(' and ')' split by ',')
    if let Some(params_start) = sig.find('(') {
        if let Some(params_end) = sig.find(')') {
            let params_str = &sig[params_start + 1..params_end];
            let param_count = if params_str.trim().is_empty() {
                0
            } else {
                params_str.split(',').count()
            };
            for _ in 0..param_count {
                actions.push(PlanAction::AddParameter);
            }
        }
    }

    // Has explicit return type (-> something)?
    if sig.contains("->") {
        actions.push(PlanAction::SetReturnType);
    }

    // Needs error handling (Result type)?
    if sig.contains("Result<") || purpose.contains("error") || purpose.contains("safe") {
        actions.push(PlanAction::AddErrorHandling);
    }

    // Needs imports (HashMap, HashSet, etc.)?
    if purpose.contains("mode")
        || purpose.contains("frequency")
        || purpose.contains("duplicate")
        || purpose.contains("unique")
    {
        actions.push(PlanAction::AddImport);
    }

    // Complex algorithms might need a method body annotation
    if purpose.contains("sort")
        || purpose.contains("search")
        || purpose.contains("fibonacci")
        || purpose.contains("factorial")
        || purpose.contains("gcd")
        || purpose.contains("collatz")
        || purpose.contains("palindrome")
        || purpose.contains("median")
        || purpose.contains("mode")
    {
        actions.push(PlanAction::AddMethod);
    }

    // Phase 4: detect new action types from purpose/signature
    if purpose.contains("match") || purpose.contains("pattern") {
        actions.push(PlanAction::MatchExpression);
    }
    if purpose.contains("for each") || purpose.contains("iterate") || purpose.contains("loop") {
        actions.push(PlanAction::ForLoop);
    }
    if purpose.contains("filter")
        || purpose.contains("map")
        || purpose.contains("chain")
        || purpose.contains("collect")
    {
        actions.push(PlanAction::IteratorChain);
    }
    if purpose.contains("closure") || purpose.contains("callback") || purpose.contains("lambda") {
        actions.push(PlanAction::ClosureDefine);
    }
    if sig.contains("Result<") && (purpose.contains("propagat") || purpose.contains("?")) {
        actions.push(PlanAction::ErrorPropagation);
    }
    if sig.contains("<T") || sig.contains("<T>") || purpose.contains("generic") {
        actions.push(PlanAction::GenericParam);
    }
    if sig.contains("'a") || purpose.contains("lifetime") {
        actions.push(PlanAction::LifetimeAnnotation);
    }
    if purpose.contains("derive") || purpose.contains("struct") {
        actions.push(PlanAction::DeriveAttribute);
    }
    if purpose.contains("test") {
        actions.push(PlanAction::TestModule);
    }
    if purpose.contains("constant") || purpose.contains("const ") {
        actions.push(PlanAction::ConstDefinition);
    }
    if purpose.contains("type alias") || purpose.contains("type ") {
        actions.push(PlanAction::TypeAlias);
    }

    // Documentation for complex functions
    if purpose.len() > 30 || purpose.contains("compute") || purpose.contains("calculate") {
        actions.push(PlanAction::AddDocumentation);
    }

    actions.push(PlanAction::Complete);
    actions
}

/// Generate additional training examples beyond the benchmark tasks.
///
/// These cover patterns from the emitter that aren't in the 100-function benchmark:
/// struct definitions, trait impls, enum definitions, etc.
fn generate_structural_examples(encoder: &CodeHDEncoder) -> Vec<TrainingExample> {
    let mut examples = Vec::new();

    // Struct definitions
    let struct_purposes = [
        "Define a Point struct with x and y coordinates",
        "Create a Rectangle struct with width and height",
        "Define a Person struct with name and age fields",
        "Create a Config struct with host, port, and timeout",
    ];
    for purpose in struct_purposes {
        let hv = encoder.encode_name(purpose);
        examples.push(TrainingExample {
            purpose: purpose.to_string(),
            intent_hv: hv,
            target_actions: vec![
                PlanAction::DefineStruct,
                PlanAction::AddField,
                PlanAction::AddField,
                PlanAction::AddDocumentation,
                PlanAction::Complete,
            ],
            category: TaskCategory::MultiStep,
        });
    }

    // Struct with methods
    let struct_method_purposes = [
        "Define a Counter struct with increment and get methods",
        "Create a Stack struct with push, pop, and peek methods",
        "Define a Circle with area and circumference methods",
    ];
    for purpose in struct_method_purposes {
        let hv = encoder.encode_name(purpose);
        examples.push(TrainingExample {
            purpose: purpose.to_string(),
            intent_hv: hv,
            target_actions: vec![
                PlanAction::DefineStruct,
                PlanAction::AddField,
                PlanAction::AddMethod,
                PlanAction::AddMethod,
                PlanAction::AddDocumentation,
                PlanAction::Complete,
            ],
            category: TaskCategory::MultiStep,
        });
    }

    // Trait definitions
    let trait_purposes = [
        "Define a Drawable trait with a draw method",
        "Create a Serializable trait with serialize and deserialize",
        "Define a Validator trait with validate method returning Result",
    ];
    for purpose in trait_purposes {
        let hv = encoder.encode_name(purpose);
        examples.push(TrainingExample {
            purpose: purpose.to_string(),
            intent_hv: hv,
            target_actions: vec![
                PlanAction::DefineTrait,
                PlanAction::AddMethod,
                PlanAction::AddDocumentation,
                PlanAction::Complete,
            ],
            category: TaskCategory::MultiStep,
        });
    }

    // Trait implementations
    let impl_purposes = [
        "Implement Display trait for Point struct",
        "Implement Iterator for Range struct",
        "Implement From<String> for MyError",
    ];
    for purpose in impl_purposes {
        let hv = encoder.encode_name(purpose);
        examples.push(TrainingExample {
            purpose: purpose.to_string(),
            intent_hv: hv,
            target_actions: vec![
                PlanAction::ImplTrait,
                PlanAction::AddMethod,
                PlanAction::Complete,
            ],
            category: TaskCategory::MultiStep,
        });
    }

    // Enum definitions
    let enum_purposes = [
        "Define a Color enum with Red, Green, Blue variants",
        "Create a Direction enum with North, South, East, West",
        "Define an Error enum with NotFound, PermissionDenied, Timeout",
    ];
    for purpose in enum_purposes {
        let hv = encoder.encode_name(purpose);
        examples.push(TrainingExample {
            purpose: purpose.to_string(),
            intent_hv: hv,
            target_actions: vec![
                PlanAction::DefineEnum,
                PlanAction::AddDocumentation,
                PlanAction::Complete,
            ],
            category: TaskCategory::MultiStep,
        });
    }

    examples
}

/// Generate the complete training dataset from benchmark tasks + structural examples.
///
/// Returns (train_set, eval_set) split 80/20.
pub fn generate_training_data(
    encoder: &CodeHDEncoder,
) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
    let mut all_examples = Vec::new();

    // 1. Generate from benchmark tasks (100 function-level examples)
    for task in benchmark_tasks() {
        let hv = encoder.encode_name(&task.purpose);
        let target_actions = infer_target_plan(&task);
        all_examples.push(TrainingExample {
            purpose: task.purpose.clone(),
            intent_hv: hv,
            target_actions,
            category: task.category,
        });
    }

    // 2. Generate structural examples (structs, traits, enums, impls)
    all_examples.extend(generate_structural_examples(encoder));

    // 3. Generate augmented examples with synonym variations
    let augmented = augment_with_synonyms(&all_examples, encoder);
    all_examples.extend(augmented);

    // Shuffle deterministically (use purpose hash as seed)
    all_examples.sort_by(|a, b| {
        let hash_a: u64 = a
            .purpose
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let hash_b: u64 = b
            .purpose
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        hash_a.cmp(&hash_b)
    });

    // 80/20 split
    let split = (all_examples.len() * 80) / 100;
    let eval = all_examples.split_off(split);
    (all_examples, eval)
}

/// Augment training data with synonym variations.
///
/// For each example, generate variants with replaced keywords:
/// "add" → "sum", "sort" → "order", etc.
fn augment_with_synonyms(
    examples: &[TrainingExample],
    encoder: &CodeHDEncoder,
) -> Vec<TrainingExample> {
    let synonyms: &[(&str, &[&str])] = &[
        ("add", &["sum", "plus", "total"]),
        ("subtract", &["minus", "difference"]),
        ("multiply", &["product", "times"]),
        ("divide", &["quotient", "split by"]),
        ("sort", &["order", "arrange", "rank"]),
        ("reverse", &["flip", "invert"]),
        ("filter", &["select", "keep"]),
        ("find", &["search", "locate", "look up"]),
        ("maximum", &["max", "largest", "biggest"]),
        ("minimum", &["min", "smallest"]),
        ("length", &["size", "count"]),
        ("remove", &["delete", "drop"]),
        ("check", &["verify", "test", "validate"]),
        ("convert", &["transform", "map"]),
        ("compute", &["calculate", "determine"]),
    ];

    let mut augmented = Vec::new();

    for example in examples {
        let purpose_lower = example.purpose.to_lowercase();
        for (word, replacements) in synonyms {
            if purpose_lower.contains(word) {
                for replacement in *replacements {
                    let new_purpose = example.purpose.to_lowercase().replace(word, replacement);
                    let hv = encoder.encode_name(&new_purpose);
                    augmented.push(TrainingExample {
                        purpose: new_purpose,
                        intent_hv: hv,
                        target_actions: example.target_actions.clone(),
                        category: example.category,
                    });
                }
            }
        }
    }

    augmented
}

/// Training result with metrics
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub epochs: usize,
    pub final_train_loss: f32,
    pub final_eval_loss: f32,
    pub best_eval_loss: f32,
    pub best_epoch: usize,
}

/// Train the CfC code sequencer on the generated training data.
///
/// Uses BPTT with Adam optimizer and early stopping.
///
/// # Arguments
/// * `sequencer` — the sequencer to train (weights updated in-place)
/// * `train_data` — training examples
/// * `eval_data` — evaluation examples for early stopping
/// * `epochs` — maximum training epochs
/// * `learning_rate` — Adam learning rate
/// * `patience` — early stopping patience (epochs without improvement)
pub fn train_sequencer(
    sequencer: &CfCCodeSequencer,
    train_data: &[TrainingExample],
    eval_data: &[TrainingExample],
    epochs: usize,
    learning_rate: f32,
    patience: usize,
) -> TrainingResult {
    let mut best_eval_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut patience_counter = 0;
    let mut final_train_loss = 0.0f32;
    let mut final_eval_loss = 0.0f32;

    // Save best weights for restoration
    let mut best_weights = sequencer.export_weights();

    for epoch in 0..epochs {
        // Training pass
        let mut train_loss_sum = 0.0f32;
        let mut train_count = 0usize;

        for example in train_data {
            match sequencer.train_sequence(
                &example.intent_hv,
                &example.target_actions,
                learning_rate,
            ) {
                Ok(loss) if loss.is_finite() => {
                    train_loss_sum += loss;
                    train_count += 1;
                }
                _ => {} // Skip NaN/Inf losses
            }
        }

        final_train_loss = if train_count > 0 {
            train_loss_sum / train_count as f32
        } else {
            f32::INFINITY
        };

        // Evaluation pass (no gradients)
        let mut eval_loss_sum = 0.0f32;
        let mut eval_count = 0usize;

        for example in eval_data {
            // Evaluate by doing a forward pass and comparing
            let plan = sequencer.plan_structure(&example.intent_hv, &[], &crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default());
            let plan_actions: Vec<PlanAction> = plan.iter().map(|s| s.action.clone()).collect();

            // Compute action-level accuracy as a proxy for loss
            let max_len = plan_actions.len().max(example.target_actions.len());
            if max_len > 0 {
                let matches: usize = plan_actions
                    .iter()
                    .zip(example.target_actions.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                let accuracy = matches as f32 / max_len as f32;
                eval_loss_sum += 1.0 - accuracy; // loss = 1 - accuracy
                eval_count += 1;
            }
        }

        final_eval_loss = if eval_count > 0 {
            eval_loss_sum / eval_count as f32
        } else {
            f32::INFINITY
        };

        // Early stopping
        if final_eval_loss < best_eval_loss {
            best_eval_loss = final_eval_loss;
            best_epoch = epoch;
            best_weights = sequencer.export_weights();
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= patience {
                // Restore best weights
                let _ = sequencer.import_weights(&best_weights);
                return TrainingResult {
                    epochs: epoch + 1,
                    final_train_loss,
                    final_eval_loss: best_eval_loss,
                    best_eval_loss,
                    best_epoch,
                };
            }
        }
    }

    // Restore best weights
    let _ = sequencer.import_weights(&best_weights);

    TrainingResult {
        epochs,
        final_train_loss,
        final_eval_loss: best_eval_loss,
        best_eval_loss,
        best_epoch,
    }
}

/// Evaluate the sequencer on a dataset and return per-category accuracy.
pub fn evaluate_sequencer(
    sequencer: &CfCCodeSequencer,
    data: &[TrainingExample],
) -> Vec<(TaskCategory, f32)> {
    use std::collections::HashMap;

    let mut cat_correct: HashMap<TaskCategory, (usize, usize)> = HashMap::new();

    for example in data {
        let plan = sequencer.plan_structure(&example.intent_hv, &[], &crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default());
        let plan_actions: Vec<PlanAction> = plan.iter().map(|s| s.action.clone()).collect();

        // Check if first action matches (most important for emitter routing)
        let first_correct = plan_actions
            .first()
            .zip(example.target_actions.first())
            .is_some_and(|(a, b)| a == b);

        let entry = cat_correct.entry(example.category).or_insert((0, 0));
        entry.0 += 1; // total
        if first_correct {
            entry.1 += 1; // correct
        }
    }

    let mut results: Vec<(TaskCategory, f32)> = cat_correct
        .into_iter()
        .map(|(cat, (total, correct))| {
            let accuracy = if total > 0 {
                correct as f32 / total as f32
            } else {
                0.0
            };
            (cat, accuracy)
        })
        .collect();

    results.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_training_data_produces_examples() {
        let encoder = CodeHDEncoder::new(512);
        let (train, eval) = generate_training_data(&encoder);

        assert!(!train.is_empty(), "Training set should not be empty");
        assert!(!eval.is_empty(), "Eval set should not be empty");
        assert!(
            train.len() > eval.len(),
            "Training set should be larger than eval"
        );

        // Every example should have at least 2 actions (action + Complete)
        for ex in &train {
            assert!(
                ex.target_actions.len() >= 2,
                "Example '{}' should have at least 2 actions",
                ex.purpose
            );
            assert_eq!(
                *ex.target_actions.last().unwrap(),
                PlanAction::Complete,
                "Last action should be Complete"
            );
        }
    }

    #[test]
    fn test_infer_target_plan_simple_function() {
        let task = BenchmarkTask {
            name: "add".into(),
            purpose: "Add two integers".into(),
            signature: "fn add(a: i32, b: i32) -> i32".into(),
            category: TaskCategory::Arithmetic,
        };
        let plan = infer_target_plan(&task);
        assert_eq!(plan[0], PlanAction::DefineFunction);
        assert!(plan.contains(&PlanAction::AddParameter));
        assert!(plan.contains(&PlanAction::SetReturnType));
        assert_eq!(*plan.last().unwrap(), PlanAction::Complete);
    }

    #[test]
    fn test_infer_target_plan_error_handling() {
        let task = BenchmarkTask {
            name: "parse_int".into(),
            purpose: "Parse string to integer with error handling".into(),
            signature: "fn parse_int(s: &str) -> Result<i32, String>".into(),
            category: TaskCategory::ErrorHandling,
        };
        let plan = infer_target_plan(&task);
        assert!(plan.contains(&PlanAction::AddErrorHandling));
    }

    #[test]
    fn test_augment_produces_synonyms() {
        let encoder = CodeHDEncoder::new(512);
        let examples = vec![TrainingExample {
            purpose: "Add two numbers".into(),
            intent_hv: encoder.encode_name("Add two numbers"),
            target_actions: vec![PlanAction::DefineFunction, PlanAction::Complete],
            category: TaskCategory::Arithmetic,
        }];
        let augmented = augment_with_synonyms(&examples, &encoder);
        assert!(
            augmented.len() >= 2,
            "Should produce at least 2 synonym variants for 'add'"
        );
        assert!(augmented.iter().any(|e| e.purpose.contains("sum")));
        assert!(augmented.iter().any(|e| e.purpose.contains("plus")));
    }

    #[test]
    fn test_train_sequencer_runs() {
        let encoder = CodeHDEncoder::new(512);
        let (train, eval) = generate_training_data(&encoder);

        // Use small subset for speed
        let small_train: Vec<_> = train.into_iter().take(10).collect();
        let small_eval: Vec<_> = eval.into_iter().take(5).collect();

        let sequencer = CfCCodeSequencer::default();
        let result = train_sequencer(&sequencer, &small_train, &small_eval, 3, 0.01, 2);

        assert!(result.epochs >= 1);
        assert!(result.final_train_loss.is_finite());
    }
}
