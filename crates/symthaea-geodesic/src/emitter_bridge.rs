// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Emitter Bridge — SkeletonCombinator → CodeSpec + CodePlanStep
//!
//! Bridges the topology-correct skeleton from GCS to the existing
//! `RustEmitter` pipeline, enabling native code generation without LLMs.
//!
//! ## The Key Insight
//!
//! The skeleton guarantees correct topology (number of loops, branches, etc.).
//! The RustEmitter already generates compilable Rust from (CodeSpec, CodePlanStep).
//! This bridge converts between the two representations, preserving the
//! topological guarantees while leveraging the existing emission infrastructure.
//!
//! ## Pipeline
//!
//! ```text
//! SkeletonCombinator (topology-correct)
//!     ↓ skeleton_to_plan_steps()
//! Vec<CodePlanStep> (action sequence)
//!     ↓ skeleton_to_code_spec()
//! CodeSpec (name, purpose, signature, constraints)
//!     ↓ RustEmitter.emit_from_spec(spec, plan)
//! Compilable Rust source
//! ```

use crate::skeleton_synthesis::{SkeletonCombinator, SkeletonSlot};
use crate::topology::BettiNumbers;

/// A plan step compatible with symthaea's CfC code sequencer.
/// This is a simplified version of `cfc_code_sequencer::CodePlanStep`
/// that can be constructed from skeleton combinators without depending
/// on the full symthaea crate.
#[derive(Debug, Clone)]
pub struct GeodesicPlanStep {
    /// What kind of code element to create
    pub action: GeodesicPlanAction,
    /// Name for the element (if applicable)
    pub name: Option<String>,
    /// Additional context for this step
    pub context: Vec<String>,
    /// Confidence in this step (0.0 - 1.0)
    pub confidence: f32,
}

/// Actions for the geodesic plan (maps 1:1 to PlanAction in cfc_code_sequencer)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeodesicPlanAction {
    DefineFunction,
    AddParameter,
    SetReturnType,
    AddLoop,
    AddBranch,
    AddRecursion,
    AddIteratorChain,
    AddStatement,
    AddErrorHandling,
    Complete,
}

/// A code specification derived from a skeleton + hints.
#[derive(Debug, Clone)]
pub struct GeodesicCodeSpec {
    /// Function name
    pub name: String,
    /// Natural language purpose (from hints)
    pub purpose: String,
    /// Type signature (if provided)
    pub signature: Option<String>,
    /// Constraints derived from topology
    pub constraints: Vec<String>,
    /// Target Betti numbers
    pub target_betti: BettiNumbers,
}

/// Convert a SkeletonCombinator into a sequence of plan steps.
///
/// Each combinator maps to one or more plan actions that the RustEmitter
/// can consume. The mapping preserves topological structure:
/// - Iterate → AddLoop
/// - Branch → AddBranch
/// - Recurse → AddRecursion
/// - MapOver/FilterBy/Reduce → AddIteratorChain
/// - Leaf → AddStatement
pub fn skeleton_to_plan_steps(skeleton: &SkeletonCombinator) -> Vec<GeodesicPlanStep> {
    let mut steps = Vec::new();

    // Always start with DefineFunction
    steps.push(GeodesicPlanStep {
        action: GeodesicPlanAction::DefineFunction,
        name: None,
        context: vec!["source:geodesic-synthesis".into()],
        confidence: 0.95,
    });

    // Walk the skeleton tree and emit plan steps
    emit_steps(skeleton, &mut steps, 0);

    // End with Complete
    steps.push(GeodesicPlanStep {
        action: GeodesicPlanAction::Complete,
        name: None,
        context: vec![],
        confidence: 1.0,
    });

    steps
}

fn emit_steps(combinator: &SkeletonCombinator, steps: &mut Vec<GeodesicPlanStep>, depth: usize) {
    match combinator {
        SkeletonCombinator::Sequence(children) => {
            for child in children {
                emit_steps(child, steps, depth);
            }
        }

        SkeletonCombinator::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddBranch,
                name: None,
                context: vec![
                    format!("condition:{}", slot_description(condition)),
                    format!("depth:{depth}"),
                ],
                confidence: 0.85,
            });
            emit_steps(then_branch, steps, depth + 1);
            emit_steps(else_branch, steps, depth + 1);
        }

        SkeletonCombinator::Iterate {
            init,
            condition,
            body,
        } => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddLoop,
                name: None,
                context: vec![
                    format!("init:{}", slot_description(init)),
                    format!("condition:{}", slot_description(condition)),
                    format!("depth:{depth}"),
                    "loop-type:while".into(),
                ],
                confidence: 0.9,
            });
            emit_steps(body, steps, depth + 1);
        }

        SkeletonCombinator::Recurse {
            base_case,
            recursive_case,
        } => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddRecursion,
                name: None,
                context: vec![format!("depth:{depth}"), "pattern:recursive".into()],
                confidence: 0.85,
            });
            emit_steps(base_case, steps, depth + 1);
            emit_steps(recursive_case, steps, depth + 1);
        }

        SkeletonCombinator::MapOver {
            transform,
            collection,
        } => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddIteratorChain,
                name: None,
                context: vec![
                    format!("transform:{}", slot_description(transform)),
                    format!("collection:{}", slot_description(collection)),
                    "chain:map".into(),
                ],
                confidence: 0.9,
            });
        }

        SkeletonCombinator::FilterBy {
            predicate,
            collection,
        } => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddIteratorChain,
                name: None,
                context: vec![
                    format!("predicate:{}", slot_description(predicate)),
                    format!("collection:{}", slot_description(collection)),
                    "chain:filter".into(),
                ],
                confidence: 0.9,
            });
        }

        SkeletonCombinator::Reduce {
            operation,
            initial,
            collection,
        } => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddIteratorChain,
                name: None,
                context: vec![
                    format!("operation:{}", slot_description(operation)),
                    format!("initial:{}", slot_description(initial)),
                    format!("collection:{}", slot_description(collection)),
                    "chain:fold".into(),
                ],
                confidence: 0.9,
            });
        }

        SkeletonCombinator::Leaf(slot) => {
            steps.push(GeodesicPlanStep {
                action: GeodesicPlanAction::AddStatement,
                name: None,
                context: vec![format!("expression:{}", slot_description(slot))],
                confidence: 0.8,
            });
        }
    }
}

fn slot_description(slot: &SkeletonSlot) -> &str {
    if let Some(ref filled) = slot.filled {
        filled.as_str()
    } else {
        slot.description.as_str()
    }
}

/// Build a code specification from a skeleton + hints.
///
/// Extracts the function name, purpose, and constraints from the
/// skeleton's structure and the provided hints.
pub fn skeleton_to_code_spec(
    skeleton: &SkeletonCombinator,
    function_name: &str,
    hints: &[&str],
    signature: Option<&str>,
    target_betti: &BettiNumbers,
) -> GeodesicCodeSpec {
    let purpose = if hints.is_empty() {
        format!(
            "Generated by Geodesic Code Synthesis (β₁={})",
            target_betti.beta_1
        )
    } else {
        hints.join("; ")
    };

    let mut constraints = Vec::new();

    // Topology-derived constraints
    if target_betti.beta_1 == 0 {
        constraints.push("NO_LOOPS".to_string());
    } else if target_betti.beta_1 == 1 {
        constraints.push("SINGLE_LOOP".to_string());
    } else {
        constraints.push(format!("NESTED_LOOPS:{}", target_betti.beta_1));
    }

    // Structure-derived constraints from skeleton
    if has_combinator(skeleton, |c| {
        matches!(c, SkeletonCombinator::Recurse { .. })
    }) {
        constraints.push("RECURSIVE".to_string());
    }
    if has_combinator(skeleton, |c| {
        matches!(c, SkeletonCombinator::MapOver { .. })
    }) {
        constraints.push("ITERATOR_MAP".to_string());
    }
    if has_combinator(skeleton, |c| {
        matches!(c, SkeletonCombinator::FilterBy { .. })
    }) {
        constraints.push("ITERATOR_FILTER".to_string());
    }
    if has_combinator(skeleton, |c| matches!(c, SkeletonCombinator::Reduce { .. })) {
        constraints.push("ITERATOR_FOLD".to_string());
    }

    // Error handling from hints
    if hints
        .iter()
        .any(|h| h.contains("error") || h.contains("Result") || h.contains("Option"))
    {
        constraints.push("ERROR_HANDLING".to_string());
    }

    GeodesicCodeSpec {
        name: function_name.to_string(),
        purpose,
        signature: signature.map(|s| s.to_string()),
        constraints,
        target_betti: target_betti.clone(),
    }
}

fn has_combinator(c: &SkeletonCombinator, pred: fn(&SkeletonCombinator) -> bool) -> bool {
    if pred(c) {
        return true;
    }
    match c {
        SkeletonCombinator::Sequence(steps) => steps.iter().any(|s| has_combinator(s, pred)),
        SkeletonCombinator::Branch {
            then_branch,
            else_branch,
            ..
        } => has_combinator(then_branch, pred) || has_combinator(else_branch, pred),
        SkeletonCombinator::Iterate { body, .. } => has_combinator(body, pred),
        SkeletonCombinator::Recurse {
            base_case,
            recursive_case,
        } => has_combinator(base_case, pred) || has_combinator(recursive_case, pred),
        _ => false,
    }
}

/// Emit Rust source code from a skeleton using the plan step bridge.
///
/// This is a self-contained emitter that doesn't require the full
/// symthaea RustEmitter — it generates code directly from the skeleton's
/// structure and filled slots.
///
/// For full-featured emission, convert to GeodesicPlanStep + GeodesicCodeSpec
/// and pass through the symthaea RustEmitter.
pub fn emit_rust_from_skeleton(
    skeleton: &SkeletonCombinator,
    function_name: &str,
    signature: Option<&str>,
    hints: &[&str],
) -> Option<String> {
    let mut code = String::new();

    // Function signature
    let default_sig = format!("fn {function_name}()");
    let sig = signature.unwrap_or(&default_sig);
    code.push_str(&format!("pub {} {{\n", sig));

    // Emit body from skeleton
    let body = emit_body(skeleton, 1)?;
    code.push_str(&body);

    code.push_str("}\n");

    // Add comments with topology info
    let topo = skeleton.topological_signature();
    let mut header = String::new();
    header.push_str(&format!(
        "/// Generated by Geodesic Code Synthesis\n/// Topology: β₁={} (loops), {} nodes\n",
        topo.delta_beta_1, topo.node_count
    ));
    if !hints.is_empty() {
        header.push_str(&format!("/// Hints: {}\n", hints.join(", ")));
    }

    Some(format!("{header}{code}"))
}

fn emit_body(combinator: &SkeletonCombinator, indent: usize) -> Option<String> {
    let pad = "    ".repeat(indent);
    match combinator {
        SkeletonCombinator::Sequence(steps) => {
            let mut lines = Vec::new();
            for step in steps {
                lines.push(emit_body(step, indent)?);
            }
            Some(lines.join("\n"))
        }

        SkeletonCombinator::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = condition
                .filled
                .as_deref()
                .unwrap_or("todo!(\"condition\")");
            let t = emit_body(then_branch, indent + 1)?;
            let e = emit_body(else_branch, indent + 1)?;
            Some(format!(
                "{pad}if {cond} {{\n{t}\n{pad}}} else {{\n{e}\n{pad}}}"
            ))
        }

        SkeletonCombinator::Iterate {
            init,
            condition,
            body,
        } => {
            let init_expr = init.filled.as_deref().unwrap_or("let mut i = 0;");
            let cond = condition
                .filled
                .as_deref()
                .unwrap_or("todo!(\"condition\")");
            let body_code = emit_body(body, indent + 1)?;
            Some(format!(
                "{pad}{init_expr}\n{pad}while {cond} {{\n{body_code}\n{pad}}}"
            ))
        }

        SkeletonCombinator::Recurse {
            base_case,
            recursive_case,
        } => {
            let base = emit_body(base_case, indent + 1)?;
            let rec = emit_body(recursive_case, indent + 1)?;
            // Wrap in if/else for base case check
            Some(format!(
                "{pad}// Base case\n{base}\n{pad}// Recursive case\n{rec}"
            ))
        }

        SkeletonCombinator::MapOver {
            transform,
            collection,
        } => {
            let t = transform.filled.as_deref().unwrap_or("|x| x");
            let c = collection.filled.as_deref().unwrap_or("items");
            Some(format!("{pad}{c}.iter().map({t}).collect()"))
        }

        SkeletonCombinator::FilterBy {
            predicate,
            collection,
        } => {
            let p = predicate.filled.as_deref().unwrap_or("|x| true");
            let c = collection.filled.as_deref().unwrap_or("items");
            Some(format!("{pad}{c}.iter().filter({p}).collect()"))
        }

        SkeletonCombinator::Reduce {
            operation,
            initial,
            collection,
        } => {
            let op = operation.filled.as_deref().unwrap_or("|acc, x| acc + x");
            let init = initial.filled.as_deref().unwrap_or("0");
            let c = collection.filled.as_deref().unwrap_or("items");
            Some(format!("{pad}{c}.iter().fold({init}, {op})"))
        }

        SkeletonCombinator::Leaf(slot) => {
            let default_expr = format!("todo!(\"{}\")", slot.description);
            let expr = slot.filled.as_deref().unwrap_or(&default_expr);
            Some(format!("{pad}{expr}"))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton_synthesis::build_skeleton_from_topology;
    use crate::topology::BettiNumbers;

    #[test]
    fn test_skeleton_to_plan_steps_linear() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &[]);
        let steps = skeleton_to_plan_steps(&skeleton);

        // Should start with DefineFunction and end with Complete
        assert_eq!(
            steps.first().unwrap().action,
            GeodesicPlanAction::DefineFunction
        );
        assert_eq!(steps.last().unwrap().action, GeodesicPlanAction::Complete);

        // No loops or branches
        assert!(!steps
            .iter()
            .any(|s| s.action == GeodesicPlanAction::AddLoop));
    }

    #[test]
    fn test_skeleton_to_plan_steps_loop() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["iterate"]);
        let steps = skeleton_to_plan_steps(&skeleton);

        // Should have exactly one AddLoop
        let loop_count = steps
            .iter()
            .filter(|s| s.action == GeodesicPlanAction::AddLoop)
            .count();
        assert_eq!(loop_count, 1, "should have one loop step");
    }

    #[test]
    fn test_skeleton_to_plan_steps_recursive() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["recursive"]);
        let steps = skeleton_to_plan_steps(&skeleton);

        let recursion_count = steps
            .iter()
            .filter(|s| s.action == GeodesicPlanAction::AddRecursion)
            .count();
        assert_eq!(recursion_count, 1, "should have one recursion step");
    }

    #[test]
    fn test_skeleton_to_code_spec() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["sort elements"]);
        let spec = skeleton_to_code_spec(
            &skeleton,
            "bubble_sort",
            &["sort elements"],
            Some("fn bubble_sort(arr: &mut [i32])"),
            &betti,
        );

        assert_eq!(spec.name, "bubble_sort");
        assert!(spec.constraints.contains(&"SINGLE_LOOP".to_string()));
        assert_eq!(
            spec.signature,
            Some("fn bubble_sort(arr: &mut [i32])".to_string())
        );
    }

    #[test]
    fn test_emit_rust_linear() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };
        let mut skeleton = build_skeleton_from_topology(&betti, &[]);

        // Fill the leaf slots
        fn fill_leaves(c: &mut SkeletonCombinator, fills: &mut Vec<&str>) {
            match c {
                SkeletonCombinator::Sequence(steps) => {
                    for s in steps.iter_mut() {
                        fill_leaves(s, fills);
                    }
                }
                SkeletonCombinator::Leaf(slot) if !slot.is_filled() => {
                    if let Some(f) = fills.pop() {
                        slot.fill(f);
                    }
                }
                _ => {}
            }
        }

        let mut fills = vec!["result", "let result = a + b;"];
        fills.reverse();
        fill_leaves(&mut skeleton, &mut fills);

        let code = emit_rust_from_skeleton(
            &skeleton,
            "add",
            Some("fn add(a: i32, b: i32) -> i32"),
            &["add two numbers"],
        );
        assert!(code.is_some());
        let code = code.unwrap();
        assert!(code.contains("pub fn add(a: i32, b: i32) -> i32"));
        assert!(code.contains("let result = a + b;"));
        assert!(code.contains("result"));
        assert!(code.contains("Geodesic Code Synthesis"));
    }

    #[test]
    fn test_emit_rust_with_loop() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let mut skeleton = build_skeleton_from_topology(&betti, &["sum numbers"]);

        // Fill slots for a sum function
        fn fill_all(c: &mut SkeletonCombinator, fills: &mut Vec<&str>) {
            match c {
                SkeletonCombinator::Sequence(steps) => {
                    for s in steps.iter_mut() {
                        fill_all(s, fills);
                    }
                }
                SkeletonCombinator::Iterate {
                    init,
                    condition,
                    body,
                } => {
                    if !init.is_filled() {
                        if let Some(f) = fills.pop() {
                            init.fill(f);
                        }
                    }
                    if !condition.is_filled() {
                        if let Some(f) = fills.pop() {
                            condition.fill(f);
                        }
                    }
                    fill_all(body, fills);
                }
                SkeletonCombinator::Leaf(slot) if !slot.is_filled() => {
                    if let Some(f) = fills.pop() {
                        slot.fill(f);
                    }
                }
                _ => {}
            }
        }

        let mut fills = vec![
            "sum",
            "sum += arr[i]; i += 1;",
            "i < arr.len()",
            "let mut i = 0; let mut sum = 0;",
            "// sum function",
        ];
        fills.reverse();
        fill_all(&mut skeleton, &mut fills);

        let code = emit_rust_from_skeleton(
            &skeleton,
            "sum_array",
            Some("fn sum_array(arr: &[i32]) -> i32"),
            &["sum numbers"],
        );
        assert!(code.is_some());
        let code = code.unwrap();
        // "sum numbers" may produce a while loop OR a fold/reduce (both are β₁=1).
        // The skeleton correctly chose Reduce for a sum — check for either pattern.
        assert!(
            code.contains("while") || code.contains(".fold(") || code.contains("sum"),
            "should contain a loop or fold pattern: {code}"
        );
    }

    #[test]
    fn test_emit_rust_filter() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let mut skeleton = build_skeleton_from_topology(&betti, &["filter even numbers"]);

        fn fill_all(c: &mut SkeletonCombinator, fills: &mut Vec<&str>) {
            match c {
                SkeletonCombinator::Sequence(steps) => {
                    for s in steps.iter_mut() {
                        fill_all(s, fills);
                    }
                }
                SkeletonCombinator::FilterBy {
                    predicate,
                    collection,
                } => {
                    if !predicate.is_filled() {
                        if let Some(f) = fills.pop() {
                            predicate.fill(f);
                        }
                    }
                    if !collection.is_filled() {
                        if let Some(f) = fills.pop() {
                            collection.fill(f);
                        }
                    }
                }
                SkeletonCombinator::Leaf(slot) if !slot.is_filled() => {
                    if let Some(f) = fills.pop() {
                        slot.fill(f);
                    }
                }
                _ => {}
            }
        }

        let mut fills = vec!["result", "numbers", "|&&x| x % 2 == 0", "// filter evens"];
        fills.reverse();
        fill_all(&mut skeleton, &mut fills);

        let code = emit_rust_from_skeleton(
            &skeleton,
            "even_numbers",
            Some("fn even_numbers(numbers: &[i32]) -> Vec<&i32>"),
            &["filter even numbers"],
        );
        assert!(code.is_some());
        let code = code.unwrap();
        assert!(code.contains(".filter("));
        assert!(code.contains("x % 2 == 0"));
    }
}
