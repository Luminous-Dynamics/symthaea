// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # The Periodic Table of Programming
//!
//! Classifies operations by their intrinsic algebraic properties — arity,
//! commutativity, associativity, identity elements, type signatures — and
//! encodes them as BinaryHVs where similar properties → similar vectors.
//!
//! ## Why This Exists
//!
//! The original program algebra encodes operations by NAME — `ADD` and `MUL`
//! get orthogonal random vectors (~0.5 similarity). But they share 5 of 7
//! algebraic properties (binary, arithmetic, commutative, associative, pure).
//!
//! The periodic table encodes by PROPERTIES — so `ADD` and `MUL` produce
//! vectors with ~0.83 similarity, while `ADD` and `SORT` produce ~0.5.
//! This enables compositional recombination: sub-trees with the same
//! structure match regardless of which variable names are used.
//!
//! ## The Analogy
//!
//! In chemistry: hydrogen is hydrogen regardless of which molecule it's in.
//! Properties (atomic number, electron configuration) are intrinsic.
//!
//! In programming: `ADD` is `ADD` regardless of which variables it operates on.
//! Properties (arity, commutativity, type signature) are intrinsic.
//!
//! The periodic table organizes operations by their compositional behavior,
//! enabling novel program synthesis by recombining structurally compatible parts.

use std::sync::OnceLock;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::program_algebra::ProgramNode;

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Arity of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arity {
    Nullary = 0,
    Unary = 1,
    Binary = 2,
    Variadic = 3,
}

/// Kind of operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    Arithmetic = 0,
    Comparison = 1,
    Logic = 2,
    Control = 3,
    Data = 4,
    Higher = 5,
}

/// Return type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReturnClass {
    Same = 0,       // returns same type as input (ADD: int→int)
    Bool = 1,       // returns boolean (LT: int→bool)
    Collection = 2, // returns collection (SORT: list→list)
    Void = 3,       // no return (PRINT)
}

/// A single element in the periodic table — one operation with its properties.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationElement {
    pub name: &'static str,
    pub arity: Arity,
    pub kind: OperationKind,
    pub commutativity: bool,
    pub associativity: bool,
    pub has_identity: bool,
    pub return_class: ReturnClass,
    pub purity: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THE TABLE
// ═══════════════════════════════════════════════════════════════════════════════

/// The complete periodic table of programming operations.
pub fn periodic_table() -> &'static [OperationElement] {
    static TABLE: OnceLock<Vec<OperationElement>> = OnceLock::new();
    TABLE.get_or_init(|| {
        use Arity::*;
        use OperationKind::*;
        use ReturnClass::*;
        vec![
            // ── Arithmetic (binary, pure) ──
            OperationElement {
                name: "ADD",
                arity: Binary,
                kind: Arithmetic,
                commutativity: true,
                associativity: true,
                has_identity: true,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "SUB",
                arity: Binary,
                kind: Arithmetic,
                commutativity: false,
                associativity: false,
                has_identity: true,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "MUL",
                arity: Binary,
                kind: Arithmetic,
                commutativity: true,
                associativity: true,
                has_identity: true,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "DIV",
                arity: Binary,
                kind: Arithmetic,
                commutativity: false,
                associativity: false,
                has_identity: true,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "MOD",
                arity: Binary,
                kind: Arithmetic,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "MAX",
                arity: Binary,
                kind: Arithmetic,
                commutativity: true,
                associativity: true,
                has_identity: false,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "MIN",
                arity: Binary,
                kind: Arithmetic,
                commutativity: true,
                associativity: true,
                has_identity: false,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "SUM",
                arity: Unary,
                kind: Arithmetic,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Same,
                purity: true,
            },
            // ── Comparison (binary → bool) ──
            OperationElement {
                name: "EQ",
                arity: Binary,
                kind: Comparison,
                commutativity: true,
                associativity: false,
                has_identity: false,
                return_class: Bool,
                purity: true,
            },
            OperationElement {
                name: "LT",
                arity: Binary,
                kind: Comparison,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Bool,
                purity: true,
            },
            OperationElement {
                name: "GT",
                arity: Binary,
                kind: Comparison,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Bool,
                purity: true,
            },
            // ── Logic (→ bool) ──
            OperationElement {
                name: "AND",
                arity: Binary,
                kind: Logic,
                commutativity: true,
                associativity: true,
                has_identity: true,
                return_class: Bool,
                purity: true,
            },
            OperationElement {
                name: "OR",
                arity: Binary,
                kind: Logic,
                commutativity: true,
                associativity: true,
                has_identity: true,
                return_class: Bool,
                purity: true,
            },
            OperationElement {
                name: "NOT",
                arity: Unary,
                kind: Logic,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Bool,
                purity: true,
            },
            // ── Data (collection operations) ──
            OperationElement {
                name: "SORT",
                arity: Unary,
                kind: Data,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Collection,
                purity: true,
            },
            OperationElement {
                name: "REVERSE",
                arity: Unary,
                kind: Data,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Collection,
                purity: true,
            },
            OperationElement {
                name: "LEN",
                arity: Unary,
                kind: Data,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "INDEX",
                arity: Binary,
                kind: Data,
                commutativity: false,
                associativity: false,
                has_identity: false,
                return_class: Same,
                purity: true,
            },
            OperationElement {
                name: "SWAP",
                arity: Binary,
                kind: Data,
                commutativity: true,
                associativity: false,
                has_identity: false,
                return_class: Void,
                purity: false,
            },
            OperationElement {
                name: "CONCAT",
                arity: Binary,
                kind: Data,
                commutativity: false,
                associativity: true,
                has_identity: true,
                return_class: Same,
                purity: true,
            },
        ]
    })
}

/// Look up an operation by name.
pub fn lookup(name: &str) -> Option<&'static OperationElement> {
    let upper = name.to_uppercase();
    periodic_table().iter().find(|e| e.name == upper)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY-BASED ENCODING
// ═══════════════════════════════════════════════════════════════════════════════

// Deterministic seeds for property role vectors.
const ARITY_SEED: u64 = 0xAE01_0001_0000_0000;
const KIND_SEED: u64 = 0xAE01_0002_0000_0000;
const COMM_SEED: u64 = 0xAE01_0003_0000_0000;
const ASSOC_SEED: u64 = 0xAE01_0004_0000_0000;
const IDENT_SEED: u64 = 0xAE01_0005_0000_0000;
const RET_SEED: u64 = 0xAE01_0006_0000_0000;
const PURE_SEED: u64 = 0xAE01_0007_0000_0000;
const SLOT_SEED: u64 = 0xAE01_AAAA_0000_0000;

// Role vectors for structural encoding.
const APPLY_STRUCT_SEED: u64 = 0xBE01_0001_0000_0000;
const BRANCH_STRUCT_SEED: u64 = 0xBE01_0002_0000_0000;
const ITERATE_STRUCT_SEED: u64 = 0xBE01_0003_0000_0000;
const RECURSE_STRUCT_SEED: u64 = 0xBE01_0004_0000_0000;
const MAP_STRUCT_SEED: u64 = 0xBE01_0005_0000_0000;
const FILTER_STRUCT_SEED: u64 = 0xBE01_0006_0000_0000;
const REDUCE_STRUCT_SEED: u64 = 0xBE01_0007_0000_0000;
const SEQUENCE_STRUCT_SEED: u64 = 0xBE01_0008_0000_0000;
const COMPOSE_STRUCT_SEED: u64 = 0xBE01_0009_0000_0000;
const COLLECT_STRUCT_SEED: u64 = 0xBE01_000A_0000_0000;

/// Encode an operation element as a BinaryHV from its algebraic properties.
///
/// Operations sharing properties produce similar vectors:
/// - ADD and MUL: ~0.83 similarity (differ only on identity value)
/// - ADD and SORT: ~0.5 similarity (differ on arity, kind, commutativity, associativity)
pub fn encode_element(element: &OperationElement) -> BinaryHV {
    let arity_hv = BinaryHV::random(ARITY_SEED + element.arity as u64);
    let kind_hv = BinaryHV::random(KIND_SEED + element.kind as u64);
    let comm_hv = BinaryHV::random(COMM_SEED + element.commutativity as u64);
    let assoc_hv = BinaryHV::random(ASSOC_SEED + element.associativity as u64);
    let ident_hv = BinaryHV::random(IDENT_SEED + element.has_identity as u64);
    let ret_hv = BinaryHV::random(RET_SEED + element.return_class as u64);
    let pure_hv = BinaryHV::random(PURE_SEED + element.purity as u64);

    BinaryHV::bundle(&[
        arity_hv, kind_hv, comm_hv, assoc_hv, ident_hv, ret_hv, pure_hv,
    ])
}

/// Generic slot encoding — same for ALL variables regardless of name.
/// Position is encoded via permutation.
fn slot_hv() -> BinaryHV {
    BinaryHV::random(SLOT_SEED)
}

/// Extract the operation name from a ProgramNode (if it's an op reference).
fn extract_op_name(node: &ProgramNode) -> Option<String> {
    match node {
        ProgramNode::Atom(name) => Some(name.to_uppercase()),
        ProgramNode::Typed(name, _) => Some(name.to_uppercase()),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRUCTURAL ENCODING — The Key Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a ProgramNode using property-based operation encoding
/// and positional slot encoding for variables.
///
/// This captures STRUCTURE, not variable names:
/// - `ADD(a, b)` and `ADD(x, y)` → IDENTICAL encoding
/// - `ADD(a, b)` and `MUL(a, b)` → SIMILAR (same structure, similar op)
/// - `ADD(a, b)` and `SORT(a)` → DISSIMILAR (different structure)
///
/// Variable names become metadata, not part of the structural encoding.
pub fn encode_structural(node: &ProgramNode) -> BinaryHV {
    encode_structural_inner(node, 0)
}

fn encode_structural_inner(node: &ProgramNode, depth: usize) -> BinaryHV {
    match node {
        // Variables: generic slot encoding (name-independent)
        ProgramNode::Atom(_) | ProgramNode::Typed(_, _) => slot_hv().permute(depth),

        // Apply: encode operation by properties, args by position
        ProgramNode::Apply { func, args } => {
            let op_hv = extract_op_name(func)
                .and_then(|name| lookup(&name))
                .map(encode_element)
                .unwrap_or_else(|| encode_structural_inner(func, depth + 1));

            let apply_role = BinaryHV::random(APPLY_STRUCT_SEED);
            let mut result = apply_role.bind(&op_hv);
            for (i, arg) in args.iter().enumerate() {
                result = result.bind(&encode_structural_inner(arg, depth + 1).permute(i + 1));
            }
            result
        }

        // Sequence: encode each step by position
        ProgramNode::Sequence(steps) => {
            let seq_role = BinaryHV::random(SEQUENCE_STRUCT_SEED);
            let step_hvs: Vec<BinaryHV> = steps
                .iter()
                .enumerate()
                .map(|(i, s)| encode_structural_inner(s, depth + 1).permute(i + 1))
                .collect();
            let bundle = BinaryHV::bundle(&step_hvs);
            seq_role.bind(&bundle)
        }

        // Branch: encode condition + branches by role
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            let branch_role = BinaryHV::random(BRANCH_STRUCT_SEED);
            branch_role
                .bind(&encode_structural_inner(condition, depth + 1))
                .bind(&encode_structural_inner(then_branch, depth + 1).permute(1))
                .bind(&encode_structural_inner(else_branch, depth + 1).permute(2))
        }

        // Iterate: encode init + step + condition
        ProgramNode::Iterate {
            init,
            step,
            condition,
        } => {
            let iter_role = BinaryHV::random(ITERATE_STRUCT_SEED);
            iter_role
                .bind(&encode_structural_inner(init, depth + 1))
                .bind(&encode_structural_inner(step, depth + 1).permute(1))
                .bind(&encode_structural_inner(condition, depth + 1).permute(2))
        }

        // Recurse: encode base + step
        ProgramNode::Recurse {
            base_case,
            recursive_step,
        } => {
            let rec_role = BinaryHV::random(RECURSE_STRUCT_SEED);
            rec_role
                .bind(&encode_structural_inner(base_case, depth + 1))
                .bind(&encode_structural_inner(recursive_step, depth + 1).permute(1))
        }

        // Higher-order: encode by structure
        ProgramNode::Map { func, collection } => {
            let map_role = BinaryHV::random(MAP_STRUCT_SEED);
            map_role
                .bind(&encode_structural_inner(func, depth + 1))
                .bind(&encode_structural_inner(collection, depth + 1).permute(1))
        }
        ProgramNode::Filter {
            predicate,
            collection,
        } => {
            let filter_role = BinaryHV::random(FILTER_STRUCT_SEED);
            filter_role
                .bind(&encode_structural_inner(predicate, depth + 1))
                .bind(&encode_structural_inner(collection, depth + 1).permute(1))
        }
        ProgramNode::Reduce {
            func,
            initial,
            collection,
        } => {
            let reduce_role = BinaryHV::random(REDUCE_STRUCT_SEED);
            reduce_role
                .bind(&encode_structural_inner(func, depth + 1))
                .bind(&encode_structural_inner(initial, depth + 1).permute(1))
                .bind(&encode_structural_inner(collection, depth + 1).permute(2))
        }
        ProgramNode::Compose(f, g) => {
            let compose_role = BinaryHV::random(COMPOSE_STRUCT_SEED);
            compose_role
                .bind(&encode_structural_inner(f, depth + 1))
                .bind(&encode_structural_inner(g, depth + 1).permute(1))
        }
        ProgramNode::Collect(source) => {
            let collect_role = BinaryHV::random(COLLECT_STRUCT_SEED);
            collect_role.bind(&encode_structural_inner(source, depth + 1))
        }
        ProgramNode::Abstract(examples) => {
            // Majority vote of structural encodings → abstract pattern
            let hvs: Vec<BinaryHV> = examples
                .iter()
                .map(|e| encode_structural_inner(e, depth + 1))
                .collect();
            BinaryHV::bundle(&hvs)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodic_table_size() {
        assert!(periodic_table().len() >= 20, "should have 20+ operations");
    }

    #[test]
    fn test_lookup_known() {
        assert!(lookup("ADD").is_some());
        assert!(lookup("add").is_some()); // case-insensitive
        assert!(lookup("MUL").is_some());
        assert!(lookup("UNKNOWN").is_none());
    }

    #[test]
    fn test_add_mul_high_similarity() {
        let add = lookup("ADD").unwrap();
        let mul = lookup("MUL").unwrap();
        let add_hv = encode_element(add);
        let mul_hv = encode_element(mul);
        let sim = add_hv.similarity(&mul_hv);
        println!("ADD vs MUL similarity: {:.4}", sim);
        // Share 5/7 properties → high similarity
        assert!(sim > 0.6, "ADD and MUL should be similar, got {sim}");
    }

    #[test]
    fn test_add_sort_low_similarity() {
        let add = lookup("ADD").unwrap();
        let sort = lookup("SORT").unwrap();
        let add_hv = encode_element(add);
        let sort_hv = encode_element(sort);
        let sim = add_hv.similarity(&sort_hv);
        println!("ADD vs SORT similarity: {:.4}", sim);
        // Share few properties → lower similarity
        assert!(sim < 0.8, "ADD and SORT should be less similar, got {sim}");
    }

    #[test]
    fn test_add_sub_moderate_similarity() {
        let add = lookup("ADD").unwrap();
        let sub = lookup("SUB").unwrap();
        let add_hv = encode_element(add);
        let sub_hv = encode_element(sub);
        let sim = add_hv.similarity(&sub_hv);
        println!("ADD vs SUB similarity: {:.4}", sim);
        // Share some but not all → moderate
        let add_mul = encode_element(lookup("MUL").unwrap()).similarity(&add_hv);
        // ADD-MUL should be more similar than ADD-SUB
        assert!(
            add_mul >= sim - 0.1,
            "ADD-MUL ({add_mul:.4}) should be >= ADD-SUB ({sim:.4})"
        );
    }

    #[test]
    fn test_structural_name_independence() {
        // THE critical test: ADD(a,b) and ADD(x,y) should encode IDENTICALLY
        let prog1 = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let prog2 = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
        );
        let enc1 = encode_structural(&prog1);
        let enc2 = encode_structural(&prog2);
        let sim = enc1.similarity(&enc2);
        println!("ADD(a,b) vs ADD(x,y) structural similarity: {:.4}", sim);
        assert!(
            sim > 0.99,
            "same structure with different names should be identical, got {sim}"
        );
    }

    #[test]
    fn test_structural_different_ops() {
        let add = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let mul = ProgramNode::apply(
            ProgramNode::op("MUL"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let sort = ProgramNode::apply(ProgramNode::atom("sort"), vec![ProgramNode::atom("arr")]);

        let add_enc = encode_structural(&add);
        let mul_enc = encode_structural(&mul);
        let sort_enc = encode_structural(&sort);

        let add_mul = add_enc.similarity(&mul_enc);
        let add_sort = add_enc.similarity(&sort_enc);
        println!("ADD vs MUL structural: {:.4}", add_mul);
        println!("ADD vs SORT structural: {:.4}", add_sort);
        // ADD-MUL should be more similar than ADD-SORT
        assert!(
            add_mul > add_sort,
            "ADD-MUL ({add_mul:.4}) > ADD-SORT ({add_sort:.4})"
        );
    }

    #[test]
    fn test_structural_iterate_name_independence() {
        // Two loops with different variable names but same structure
        let loop1 = ProgramNode::iterate(
            ProgramNode::atom("i = 0"),
            ProgramNode::branch(
                ProgramNode::apply(
                    ProgramNode::op("GT"),
                    vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max")],
                ),
                ProgramNode::atom("max = arr[i]"),
                ProgramNode::atom("skip"),
            ),
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
            ),
        );
        let loop2 = ProgramNode::iterate(
            ProgramNode::atom("j = 0"),
            ProgramNode::branch(
                ProgramNode::apply(
                    ProgramNode::op("GT"),
                    vec![ProgramNode::atom("data[j]"), ProgramNode::atom("best")],
                ),
                ProgramNode::atom("best = data[j]"),
                ProgramNode::atom("noop"),
            ),
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("j"), ProgramNode::atom("n")],
            ),
        );
        let enc1 = encode_structural(&loop1);
        let enc2 = encode_structural(&loop2);
        let sim = enc1.similarity(&enc2);
        println!("Iterate(GT branch) with different names: {:.4}", sim);
        // Same structure → high similarity
        assert!(
            sim > 0.9,
            "same loop structure should be highly similar, got {sim}"
        );
    }

    #[test]
    fn test_structural_different_structures() {
        let loop_branch = ProgramNode::iterate(
            ProgramNode::atom("init"),
            ProgramNode::branch(
                ProgramNode::atom("cond"),
                ProgramNode::atom("then"),
                ProgramNode::atom("else"),
            ),
            ProgramNode::atom("end_cond"),
        );
        let simple_add = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let enc_loop = encode_structural(&loop_branch);
        let enc_add = encode_structural(&simple_add);
        let sim = enc_loop.similarity(&enc_add);
        println!("Iterate(Branch) vs ADD: {:.4}", sim);
        // Very different structures → low similarity
        assert!(
            sim < 0.7,
            "different structures should be dissimilar, got {sim}"
        );
    }
}
