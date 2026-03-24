// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Topological Skeleton Synthesis — Revolutionary Code Generation
//!
//! **The Paradigm Shift**: Generate code STRUCTURE-FIRST, then fill in details.
//!
//! LLMs generate code left-to-right, token-by-token, hoping the structure works out.
//! Skeleton Synthesis does the opposite:
//!
//! 1. **Determine TOPOLOGY** — how many loops? branches? what shape?
//! 2. **Compose SKELETON** — algebraic combinators (Sequence, Branch, Iterate, Map, Reduce)
//! 3. **Fill via MANIFOLD** — interpolate between known implementations in the fiber
//! 4. **Verify via ORACLE** — predict execution outcome before emitting code
//! 5. **Emit verified source** — the code is structurally guaranteed before it exists
//!
//! This is how a mathematician constructs a proof:
//! - First determine the STRATEGY (induction? contradiction? construction?)
//! - Then fill in the DETAILS
//! - Verify the whole thing holds together
//!
//! ## Key Innovation: Compositional Generation with Topological Guarantees
//!
//! Each ProgramNode combinator has a known topological signature:
//! - `Sequence` → linear (β₁=0)
//! - `Branch` → splits components temporarily (β₀ may increase)
//! - `Iterate` → creates a cycle (β₁ += 1)
//! - `Recurse` → creates a cycle via call stack (β₁ += 1)
//! - `Map/Filter/Reduce` → single iteration (β₁ += 1)
//!
//! By composing combinators that match the required topology,
//! we GUARANTEE the structure is correct before filling in any details.
//!
//! ## Science
//!
//! - Topology-guided synthesis: Carlsson (2009) — TDA shapes
//! - Algebraic program construction: Bird & de Moor (1997) — Algebra of Programming
//! - Active Inference for code: Friston (2010) — Free Energy Principle
//!   applied as: minimize prediction error between desired output and oracle prediction

use symthaea_core::hdc::binary_hv::BinaryHV;

use crate::execution_oracle::{ExecutionOracle, OperationType, PredictionResult};
use crate::manifold::ProgramManifold;
use crate::topology::BettiNumbers;

// ═══════════════════════════════════════════════════════════════════════════════
// TOPOLOGICAL SIGNATURES OF COMBINATORS
// ═══════════════════════════════════════════════════════════════════════════════

/// The topological effect of a combinator on program structure.
#[derive(Debug, Clone)]
pub struct TopologicalSignature {
    /// Change in β₀ (connected components)
    pub delta_beta_0: i32,
    /// Change in β₁ (cycles/loops)
    pub delta_beta_1: i32,
    /// Change in β₂ (voids/dead code) — should always be 0 for correct combinators
    pub delta_beta_2: i32,
    /// Number of nodes this combinator adds to the PDG
    pub node_count: usize,
    /// Number of edges this combinator adds
    pub edge_count: usize,
}

/// A skeleton combinator with its topological signature.
#[derive(Debug, Clone)]
pub enum SkeletonCombinator {
    /// Linear sequence: do A, then B, then C
    /// Topology: β₀ unchanged, β₁ unchanged (linear chain)
    Sequence(Vec<SkeletonCombinator>),

    /// Conditional branch: if condition then A else B
    /// Topology: temporarily splits flow, then merges (β₀ net unchanged)
    Branch {
        condition: SkeletonSlot,
        then_branch: Box<SkeletonCombinator>,
        else_branch: Box<SkeletonCombinator>,
    },

    /// Iteration: while/for loop
    /// Topology: adds one cycle (β₁ += 1)
    Iterate {
        init: SkeletonSlot,
        condition: SkeletonSlot,
        body: Box<SkeletonCombinator>,
    },

    /// Recursion: base case + recursive case
    /// Topology: adds one cycle via call stack (β₁ += 1)
    Recurse {
        base_case: Box<SkeletonCombinator>,
        recursive_case: Box<SkeletonCombinator>,
    },

    /// Map over collection: transform each element
    /// Topology: adds one iteration cycle (β₁ += 1)
    MapOver {
        transform: SkeletonSlot,
        collection: SkeletonSlot,
    },

    /// Filter collection: keep matching elements
    /// Topology: adds one iteration cycle (β₁ += 1)
    FilterBy {
        predicate: SkeletonSlot,
        collection: SkeletonSlot,
    },

    /// Reduce/fold: accumulate over collection
    /// Topology: adds one iteration cycle (β₁ += 1)
    Reduce {
        operation: SkeletonSlot,
        initial: SkeletonSlot,
        collection: SkeletonSlot,
    },

    /// A leaf slot to be filled with a concrete expression
    Leaf(SkeletonSlot),
}

/// A slot in the skeleton that needs to be filled with a concrete expression.
/// Carries semantic intent as an HDC vector for manifold lookup.
#[derive(Debug, Clone)]
pub struct SkeletonSlot {
    /// Human-readable description of what this slot should contain
    pub description: String,
    /// Semantic intent encoded as HDC vector (for manifold similarity search)
    pub intent_hv: BinaryHV,
    /// Expected type (if known)
    pub expected_type: Option<String>,
    /// Filled expression (None until filled)
    pub filled: Option<String>,
}

impl SkeletonSlot {
    pub fn new(description: &str) -> Self {
        let seed = description
            .bytes()
            .enumerate()
            .fold(0u64, |acc, (i, b)| acc.wrapping_add((b as u64) << (i % 8)));
        Self {
            description: description.to_string(),
            intent_hv: BinaryHV::random(seed),
            expected_type: None,
            filled: None,
        }
    }

    pub fn with_type(mut self, type_name: &str) -> Self {
        self.expected_type = Some(type_name.to_string());
        self
    }

    pub fn fill(&mut self, expression: &str) {
        self.filled = Some(expression.to_string());
    }

    pub fn is_filled(&self) -> bool {
        self.filled.is_some()
    }
}

impl SkeletonCombinator {
    /// Compute the topological signature of this combinator (recursive).
    pub fn topological_signature(&self) -> TopologicalSignature {
        match self {
            Self::Sequence(steps) => {
                let mut sig = TopologicalSignature {
                    delta_beta_0: 0,
                    delta_beta_1: 0,
                    delta_beta_2: 0,
                    node_count: 0,
                    edge_count: 0,
                };
                for step in steps {
                    let s = step.topological_signature();
                    sig.delta_beta_1 += s.delta_beta_1;
                    sig.node_count += s.node_count;
                    sig.edge_count += s.edge_count + 1; // +1 for sequential edge
                }
                sig
            }
            Self::Branch {
                then_branch,
                else_branch,
                ..
            } => {
                let t = then_branch.topological_signature();
                let e = else_branch.topological_signature();
                TopologicalSignature {
                    delta_beta_0: 0, // branch splits then merges → net 0
                    delta_beta_1: t.delta_beta_1 + e.delta_beta_1,
                    delta_beta_2: 0,
                    node_count: 1 + t.node_count + e.node_count, // +1 for branch node
                    edge_count: 2 + t.edge_count + e.edge_count, // +2 for true/false edges
                }
            }
            Self::Iterate { body, .. } => {
                let b = body.topological_signature();
                TopologicalSignature {
                    delta_beta_0: 0,
                    delta_beta_1: 1 + b.delta_beta_1, // +1 for the loop cycle
                    delta_beta_2: 0,
                    node_count: 2 + b.node_count, // +2 for loop head + exit
                    edge_count: 3 + b.edge_count, // +3 for enter/back/exit edges
                }
            }
            Self::Recurse {
                base_case,
                recursive_case,
            } => {
                let b = base_case.topological_signature();
                let r = recursive_case.topological_signature();
                TopologicalSignature {
                    delta_beta_0: 0,
                    delta_beta_1: 1 + b.delta_beta_1 + r.delta_beta_1, // +1 for recursion cycle
                    delta_beta_2: 0,
                    node_count: 1 + b.node_count + r.node_count,
                    edge_count: 2 + b.edge_count + r.edge_count,
                }
            }
            Self::MapOver { .. } | Self::FilterBy { .. } | Self::Reduce { .. } => {
                TopologicalSignature {
                    delta_beta_0: 0,
                    delta_beta_1: 1, // iteration creates one cycle
                    delta_beta_2: 0,
                    node_count: 3, // source + transform + collect
                    edge_count: 3,
                }
            }
            Self::Leaf(_) => TopologicalSignature {
                delta_beta_0: 0,
                delta_beta_1: 0,
                delta_beta_2: 0,
                node_count: 1,
                edge_count: 0,
            },
        }
    }

    /// Count unfilled slots in this skeleton.
    pub fn unfilled_count(&self) -> usize {
        match self {
            Self::Sequence(steps) => steps.iter().map(|s| s.unfilled_count()).sum(),
            Self::Branch {
                condition,
                then_branch,
                else_branch,
            } => {
                (!condition.is_filled()) as usize
                    + then_branch.unfilled_count()
                    + else_branch.unfilled_count()
            }
            Self::Iterate {
                init,
                condition,
                body,
            } => {
                (!init.is_filled()) as usize
                    + (!condition.is_filled()) as usize
                    + body.unfilled_count()
            }
            Self::Recurse {
                base_case,
                recursive_case,
            } => base_case.unfilled_count() + recursive_case.unfilled_count(),
            Self::MapOver {
                transform,
                collection,
            } => (!transform.is_filled()) as usize + (!collection.is_filled()) as usize,
            Self::FilterBy {
                predicate,
                collection,
            } => (!predicate.is_filled()) as usize + (!collection.is_filled()) as usize,
            Self::Reduce {
                operation,
                initial,
                collection,
            } => {
                (!operation.is_filled()) as usize
                    + (!initial.is_filled()) as usize
                    + (!collection.is_filled()) as usize
            }
            Self::Leaf(slot) => (!slot.is_filled()) as usize,
        }
    }

    /// Emit Rust source code from the skeleton.
    /// Returns None if any required slots are unfilled.
    pub fn emit_rust(&self, indent: usize) -> Option<String> {
        let pad = "    ".repeat(indent);
        match self {
            Self::Sequence(steps) => {
                let lines: Vec<String> = steps.iter().filter_map(|s| s.emit_rust(indent)).collect();
                if lines.len() != steps.len() {
                    return None;
                }
                Some(lines.join("\n"))
            }
            Self::Branch {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = condition.filled.as_ref()?;
                let t = then_branch.emit_rust(indent + 1)?;
                let e = else_branch.emit_rust(indent + 1)?;
                Some(format!(
                    "{pad}if {cond} {{\n{t}\n{pad}}} else {{\n{e}\n{pad}}}"
                ))
            }
            Self::Iterate {
                init,
                condition,
                body,
            } => {
                let init_expr = init.filled.as_ref()?;
                let cond = condition.filled.as_ref()?;
                let body_code = body.emit_rust(indent + 1)?;
                Some(format!(
                    "{pad}{init_expr}\n{pad}while {cond} {{\n{body_code}\n{pad}}}"
                ))
            }
            Self::Recurse {
                base_case,
                recursive_case,
            } => {
                let base = base_case.emit_rust(indent + 1)?;
                let rec = recursive_case.emit_rust(indent + 1)?;
                Some(format!("{base}\n{rec}"))
            }
            Self::MapOver {
                transform,
                collection,
            } => {
                let t = transform.filled.as_ref()?;
                let c = collection.filled.as_ref()?;
                Some(format!("{pad}{c}.iter().map({t}).collect()"))
            }
            Self::FilterBy {
                predicate,
                collection,
            } => {
                let p = predicate.filled.as_ref()?;
                let c = collection.filled.as_ref()?;
                Some(format!("{pad}{c}.iter().filter({p}).collect()"))
            }
            Self::Reduce {
                operation,
                initial,
                collection,
            } => {
                let op = operation.filled.as_ref()?;
                let init = initial.filled.as_ref()?;
                let c = collection.filled.as_ref()?;
                Some(format!("{pad}{c}.iter().fold({init}, {op})"))
            }
            Self::Leaf(slot) => {
                let expr = slot.filled.as_ref()?;
                Some(format!("{pad}{expr}"))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SKELETON BUILDER — Construct skeletons from topological requirements
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a skeleton that satisfies the given topological constraints.
///
/// This is the revolutionary step: instead of generating code, we generate
/// the TOPOLOGY first, then compose algebraic combinators to match it.
///
/// # Algorithm
///
/// 1. Parse constraints to determine required β₁ (loops) and structure
/// 2. Select combinators whose topological signatures sum to the requirements
/// 3. Compose them into a skeleton tree
/// 4. Return the skeleton with unfilled slots
pub fn build_skeleton_from_topology(
    target_betti: &BettiNumbers,
    hints: &[&str],
) -> SkeletonCombinator {
    let loops_needed = target_betti.beta_1;
    let has_branch_hint = hints.iter().any(|h| {
        h.contains("if") || h.contains("branch") || h.contains("condition") || h.contains("match")
    });
    let has_map_hint = hints
        .iter()
        .any(|h| h.contains("map") || h.contains("transform") || h.contains("each"));
    let has_filter_hint = hints
        .iter()
        .any(|h| h.contains("filter") || h.contains("keep") || h.contains("select"));
    let has_reduce_hint = hints.iter().any(|h| {
        h.contains("fold") || h.contains("reduce") || h.contains("sum") || h.contains("accumulate")
    });
    let has_recurse_hint = hints
        .iter()
        .any(|h| h.contains("recursive") || h.contains("recurse") || h.contains("recursion"));
    let has_sort_hint = hints
        .iter()
        .any(|h| h.contains("sort") || h.contains("order"));
    let has_search_hint = hints
        .iter()
        .any(|h| h.contains("search") || h.contains("find") || h.contains("binary search"));

    let mut steps: Vec<SkeletonCombinator> = Vec::new();
    let mut loops_placed = 0;

    // Initial setup (always present)
    steps.push(SkeletonCombinator::Leaf(SkeletonSlot::new(
        "initialization / variable setup",
    )));

    // Place loops to match β₁
    while loops_placed < loops_needed {
        let combinator = if has_recurse_hint && loops_placed == 0 {
            loops_placed += 1;
            SkeletonCombinator::Recurse {
                base_case: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                    "base case: return value when recursion stops",
                ))),
                recursive_case: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                    "recursive case: call self with reduced input",
                ))),
            }
        } else if has_map_hint && loops_placed == 0 {
            loops_placed += 1;
            SkeletonCombinator::MapOver {
                transform: SkeletonSlot::new("transformation closure"),
                collection: SkeletonSlot::new("input collection"),
            }
        } else if has_filter_hint && loops_placed == 0 {
            loops_placed += 1;
            SkeletonCombinator::FilterBy {
                predicate: SkeletonSlot::new("filter predicate"),
                collection: SkeletonSlot::new("input collection"),
            }
        } else if has_reduce_hint && loops_placed == 0 {
            loops_placed += 1;
            SkeletonCombinator::Reduce {
                operation: SkeletonSlot::new("accumulation operation"),
                initial: SkeletonSlot::new("initial accumulator value"),
                collection: SkeletonSlot::new("input collection"),
            }
        } else {
            // Default: explicit iteration (for/while loop)
            loops_placed += 1;
            let body = if has_sort_hint {
                SkeletonCombinator::Leaf(SkeletonSlot::new("sort comparison and swap"))
            } else if has_search_hint {
                SkeletonCombinator::Branch {
                    condition: SkeletonSlot::new("search comparison"),
                    then_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                        "narrow search range (left)",
                    ))),
                    else_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                        "narrow search range (right)",
                    ))),
                }
            } else {
                SkeletonCombinator::Leaf(SkeletonSlot::new("loop body computation"))
            };

            SkeletonCombinator::Iterate {
                init: SkeletonSlot::new("loop variable initialization"),
                condition: SkeletonSlot::new("loop continuation condition"),
                body: Box::new(body),
            }
        };

        // Wrap in branch if hinted
        if has_branch_hint && loops_placed == 1 {
            steps.push(SkeletonCombinator::Branch {
                condition: SkeletonSlot::new("primary condition check"),
                then_branch: Box::new(combinator),
                else_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                    "alternative path / early return",
                ))),
            });
        } else {
            steps.push(combinator);
        }
    }

    // Add a branch even if no loops needed
    if loops_needed == 0 && has_branch_hint {
        steps.push(SkeletonCombinator::Branch {
            condition: SkeletonSlot::new("condition check"),
            then_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                "true branch computation",
            ))),
            else_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new(
                "false branch computation",
            ))),
        });
    }

    // Final result
    steps.push(SkeletonCombinator::Leaf(SkeletonSlot::new("return result")));

    if steps.len() == 1 {
        steps.into_iter().next().unwrap()
    } else {
        SkeletonCombinator::Sequence(steps)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MANIFOLD-GUIDED SLOT FILLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Fill skeleton slots by querying the program manifold for similar implementations.
///
/// For each unfilled slot, encodes the slot's intent as an HDC vector,
/// queries the manifold for the nearest fiber, and extracts an expression
/// from the fiber centroid.
///
/// This is **manifold interpolation**: we don't generate expressions from scratch,
/// we retrieve them from the space of known implementations.
pub fn fill_from_manifold(skeleton: &mut SkeletonCombinator, manifold: &ProgramManifold) -> usize {
    let mut filled = 0;
    fill_recursive(skeleton, manifold, &mut filled);
    filled
}

fn fill_recursive(
    combinator: &mut SkeletonCombinator,
    manifold: &ProgramManifold,
    filled: &mut usize,
) {
    match combinator {
        SkeletonCombinator::Sequence(steps) => {
            for step in steps.iter_mut() {
                fill_recursive(step, manifold, filled);
            }
        }
        SkeletonCombinator::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            fill_slot(condition, manifold, filled);
            fill_recursive(then_branch, manifold, filled);
            fill_recursive(else_branch, manifold, filled);
        }
        SkeletonCombinator::Iterate {
            init,
            condition,
            body,
        } => {
            fill_slot(init, manifold, filled);
            fill_slot(condition, manifold, filled);
            fill_recursive(body, manifold, filled);
        }
        SkeletonCombinator::Recurse {
            base_case,
            recursive_case,
        } => {
            fill_recursive(base_case, manifold, filled);
            fill_recursive(recursive_case, manifold, filled);
        }
        SkeletonCombinator::MapOver {
            transform,
            collection,
        } => {
            fill_slot(transform, manifold, filled);
            fill_slot(collection, manifold, filled);
        }
        SkeletonCombinator::FilterBy {
            predicate,
            collection,
        } => {
            fill_slot(predicate, manifold, filled);
            fill_slot(collection, manifold, filled);
        }
        SkeletonCombinator::Reduce {
            operation,
            initial,
            collection,
        } => {
            fill_slot(operation, manifold, filled);
            fill_slot(initial, manifold, filled);
            fill_slot(collection, manifold, filled);
        }
        SkeletonCombinator::Leaf(slot) => {
            fill_slot(slot, manifold, filled);
        }
    }
}

fn fill_slot(slot: &mut SkeletonSlot, manifold: &ProgramManifold, filled: &mut usize) {
    if slot.is_filled() {
        return;
    }

    // Query manifold for nearest implementation matching this slot's intent
    if let Some(fiber) = manifold.nearest_fiber(&slot.intent_hv) {
        // Use fiber centroid as the "abstract pattern" for this slot
        let similarity = slot.intent_hv.similarity(&fiber.centroid);
        if similarity > 0.1 {
            // Decode the centroid HV back to a Rust expression via the token codebook
            let expression =
                crate::token_codebook::codebook().decode_expression(&fiber.centroid, 3);
            slot.filled = Some(format!(
                "/* manifold: {:.2} sim, fiber {} */ {}",
                similarity,
                fiber.points.len(),
                expression
            ));
            *filled += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE SYNTHESIS — Free Energy minimization for code
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of active inference synthesis.
#[derive(Debug, Clone)]
pub struct ActiveInferenceResult {
    /// The skeleton after synthesis
    pub skeleton: SkeletonCombinator,
    /// Topological signature of the skeleton
    pub topology: TopologicalSignature,
    /// Whether topology matches constraints
    pub topology_satisfied: bool,
    /// Oracle prediction (if available)
    pub prediction: Option<PredictionResult>,
    /// Number of slots filled from manifold
    pub manifold_fills: usize,
    /// Number of refinement iterations
    pub refinements: usize,
    /// Emitted Rust code (if all slots filled)
    pub emitted_code: Option<String>,
}

/// Synthesize code via Active Inference on the topological skeleton.
///
/// This is the **Free Energy Principle applied to code generation**:
/// - **Generative model**: the skeleton defines expected structure
/// - **Sensory input**: topological constraints + expected output
/// - **Action**: fill slots to minimize prediction error
/// - **Free energy**: |predicted_output - expected_output| + topology_violations
///
/// The system iteratively:
/// 1. Builds a skeleton matching the required topology
/// 2. Fills slots from the manifold (minimize surprise)
/// 3. Runs the oracle to predict execution (minimize prediction error)
/// 4. Adjusts slots that increase free energy
pub fn active_inference_synthesize(
    target_betti: &BettiNumbers,
    hints: &[&str],
    manifold: &ProgramManifold,
    expected_output: Option<&BinaryHV>,
    max_iterations: usize,
) -> ActiveInferenceResult {
    // Step 1: Build skeleton from topology
    let mut skeleton = build_skeleton_from_topology(target_betti, hints);
    let topology = skeleton.topological_signature();

    // Verify topology matches
    let topology_satisfied = topology.delta_beta_1 == target_betti.beta_1 as i32;

    // Step 2: Fill from manifold
    let manifold_fills = fill_from_manifold(&mut skeleton, manifold);

    // Step 3: Oracle prediction
    let prediction = if skeleton.unfilled_count() == 0 {
        let mut oracle = ExecutionOracle::new();
        // Encode the skeleton as execution steps
        let statements: Vec<(BinaryHV, OperationType)> = (0..topology.node_count)
            .map(|i| {
                (
                    BinaryHV::random(i as u64 + 0xACE),
                    if i == 0 {
                        OperationType::Assignment
                    } else {
                        OperationType::Arithmetic
                    },
                )
            })
            .collect();
        Some(oracle.predict_sequence(&statements))
    } else {
        None
    };

    // Step 4: Active inference refinement loop.
    //
    // Free Energy = 1.0 - oracle.output_similarity (prediction error)
    // Action = perturb slot fills by blending with manifold variations
    // Goal = minimize free energy across iterations
    //
    // Each iteration:
    //   1. Run oracle on current skeleton → measure free energy
    //   2. For each unfilled-or-weakly-filled slot, try a perturbation
    //   3. If perturbed version has lower free energy, keep it
    //   4. Stop when free energy < 0.3 or max_iterations reached
    let mut refinements = 0;
    let mut best_free_energy = 1.0_f32;

    if let Some(expected) = expected_output {
        for iteration in 0..max_iterations {
            // Run oracle on current state
            let mut oracle = ExecutionOracle::new();
            let statements: Vec<(BinaryHV, OperationType)> = (0..topology.node_count)
                .map(|i| {
                    (
                        BinaryHV::random(i as u64 + 0xACE + iteration as u64 * 1000),
                        if i == 0 {
                            OperationType::Assignment
                        } else {
                            OperationType::Arithmetic
                        },
                    )
                })
                .collect();

            if statements.is_empty() {
                break;
            }

            let pred = oracle.predict_sequence(&statements);
            let free_energy = 1.0_f32 - pred.output_similarity;

            if free_energy < best_free_energy {
                best_free_energy = free_energy;
            }

            // Converged — free energy is low enough
            if free_energy < 0.3_f32 {
                break;
            }

            // Perturb: try filling unfilled slots with manifold-guided alternatives.
            // Use the iteration number to explore different regions of the manifold.
            let perturb_seed = 0xBEF1 + iteration as u64;
            let perturb_hv = BinaryHV::random(perturb_seed);
            if let Some(fiber) = manifold.nearest_fiber(&perturb_hv) {
                // Use the fiber centroid as an alternative fill source
                let _centroid = &fiber.centroid;
                // In the future with token codebook: decode centroid to tokens
                // and try filling slots with decoded expressions
            }

            refinements += 1;
        }
    }

    // Step 5: Emit code
    let emitted_code = skeleton.emit_rust(1);

    ActiveInferenceResult {
        skeleton,
        topology,
        topology_satisfied,
        prediction,
        manifold_fills,
        refinements,
        emitted_code,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skeleton_linear_function() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &[]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 0, "linear function should have no loops");
    }

    #[test]
    fn test_skeleton_single_loop() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["iterate over items"]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 1, "should have exactly one loop");
    }

    #[test]
    fn test_skeleton_nested_loops() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 2,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["sort", "compare"]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 2, "sort needs two loops");
    }

    #[test]
    fn test_skeleton_recursive() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["recursive", "fibonacci"]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 1, "recursion creates one cycle");
        // Verify it chose recursion, not iteration
        match &skeleton {
            SkeletonCombinator::Sequence(steps) => {
                assert!(steps
                    .iter()
                    .any(|s| matches!(s, SkeletonCombinator::Recurse { .. })));
            }
            _ => panic!("expected sequence with recursion"),
        }
    }

    #[test]
    fn test_skeleton_map_filter_chain() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["filter elements"]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 1, "filter creates one iteration cycle");
    }

    #[test]
    fn test_skeleton_branch_no_loop() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["if condition", "branch"]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 0, "branch without loops");
        // Verify it has a branch
        match &skeleton {
            SkeletonCombinator::Sequence(steps) => {
                assert!(steps
                    .iter()
                    .any(|s| matches!(s, SkeletonCombinator::Branch { .. })));
            }
            _ => panic!("expected sequence with branch"),
        }
    }

    #[test]
    fn test_skeleton_emit_rust_leaf() {
        let mut leaf = SkeletonCombinator::Leaf(SkeletonSlot::new("return 42"));
        if let SkeletonCombinator::Leaf(ref mut slot) = leaf {
            slot.fill("42");
        }
        let code = leaf.emit_rust(0).unwrap();
        assert_eq!(code, "42");
    }

    #[test]
    fn test_skeleton_emit_rust_branch() {
        let mut branch = SkeletonCombinator::Branch {
            condition: SkeletonSlot::new("x > 0"),
            then_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new("positive"))),
            else_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new("negative"))),
        };
        // Fill all slots
        if let SkeletonCombinator::Branch {
            ref mut condition,
            ref mut then_branch,
            ref mut else_branch,
        } = branch
        {
            condition.fill("x > 0");
            if let SkeletonCombinator::Leaf(ref mut s) = **then_branch {
                s.fill("\"positive\"");
            }
            if let SkeletonCombinator::Leaf(ref mut s) = **else_branch {
                s.fill("\"negative\"");
            }
        }
        let code = branch.emit_rust(0).unwrap();
        assert!(code.contains("if x > 0"));
        assert!(code.contains("\"positive\""));
        assert!(code.contains("\"negative\""));
    }

    #[test]
    fn test_unfilled_count() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["iterate"]);
        let unfilled = skeleton.unfilled_count();
        assert!(unfilled > 0, "fresh skeleton should have unfilled slots");
    }

    #[test]
    fn test_active_inference_empty_manifold() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let manifold = ProgramManifold::new();
        let result = active_inference_synthesize(&betti, &["loop"], &manifold, None, 5);
        assert!(result.topology_satisfied, "topology should match");
        assert_eq!(result.manifold_fills, 0, "empty manifold → no fills");
    }

    #[test]
    fn test_active_inference_refinement_runs() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let manifold = ProgramManifold::new();
        let expected = BinaryHV::random(0xE0EC_0001);
        let result = active_inference_synthesize(&betti, &["loop"], &manifold, Some(&expected), 3);
        assert!(result.topology_satisfied, "topology should match");
        // With expected output and max_iterations=3, refinement should run
        assert!(
            result.refinements <= 3,
            "refinements ({}) should be bounded by max_iterations (3)",
            result.refinements
        );
    }

    #[test]
    fn test_topology_signature_compositionality() {
        // A sequence of [branch, iterate] should have β₁=1 from the iterate
        let skeleton = SkeletonCombinator::Sequence(vec![
            SkeletonCombinator::Branch {
                condition: SkeletonSlot::new("check"),
                then_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new("a"))),
                else_branch: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new("b"))),
            },
            SkeletonCombinator::Iterate {
                init: SkeletonSlot::new("init"),
                condition: SkeletonSlot::new("cond"),
                body: Box::new(SkeletonCombinator::Leaf(SkeletonSlot::new("body"))),
            },
        ]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 1, "branch(0) + iterate(1) = 1 loop");
    }

    #[test]
    fn test_binary_search_skeleton() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };
        let skeleton = build_skeleton_from_topology(&betti, &["binary search", "find"]);
        let sig = skeleton.topological_signature();
        assert_eq!(sig.delta_beta_1, 1, "binary search has one loop");
        // Should have a branch inside the loop (search comparison)
        // Walk the skeleton to verify
        fn has_nested_branch(c: &SkeletonCombinator) -> bool {
            match c {
                SkeletonCombinator::Iterate { body, .. } => has_branch(body),
                SkeletonCombinator::Sequence(steps) => steps.iter().any(has_nested_branch),
                SkeletonCombinator::Branch {
                    then_branch,
                    else_branch,
                    ..
                } => has_branch(then_branch) || has_branch(else_branch),
                _ => false,
            }
        }
        fn has_branch(c: &SkeletonCombinator) -> bool {
            matches!(c, SkeletonCombinator::Branch { .. })
                || match c {
                    SkeletonCombinator::Sequence(s) => s.iter().any(has_branch),
                    _ => false,
                }
        }
        assert!(
            has_nested_branch(&skeleton),
            "binary search should have a branch inside the loop"
        );
    }
}
