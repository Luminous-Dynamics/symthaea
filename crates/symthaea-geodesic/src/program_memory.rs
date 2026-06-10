// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Program Memory — structured library of ProgramNode compositions with pre-computed encodings.
//!
//! Provides a searchable library of program patterns (arithmetic, control flow, recursion,
//! higher-order) that the resonant explorer can use as a vocabulary for noise-driven search.
//! Each entry stores both the symbolic tree and its HDC encoding for O(1) nearest-neighbour
//! lookup.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::program_algebra::ProgramNode;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A single entry in the program memory.
#[derive(Clone)]
pub struct ProgramMemoryEntry {
    /// Human-readable name (e.g. "add", "factorial").
    pub name: String,
    /// The program tree.
    pub node: ProgramNode,
    /// Pre-computed HDC encoding of `node`.
    pub encoding: BinaryHV,
    /// Optional provenance (e.g. "basic", "learned", "standard_library").
    pub source: Option<String>,
    /// Sub-patterns extracted by decomposing `node`.
    pub sub_patterns: Vec<(String, ProgramNode, BinaryHV)>,
}

/// Searchable library of ProgramNode compositions.
#[derive(Clone)]
pub struct ProgramMemory {
    entries: Vec<ProgramMemoryEntry>,
}

impl ProgramMemory {
    /// Build a minimal library with fundamental arithmetic/control patterns.
    ///
    /// Constructs ~20 patterns directly from ProgramNode constructors — does NOT
    /// depend on ProgramPatternLibrary (which lives in the main symthaea crate).
    pub fn basic() -> Self {
        let mut mem = Self {
            entries: Vec::with_capacity(20),
        };

        // ── Arithmetic ──
        mem.add_basic(
            "add",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "sub",
            ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "mul",
            ProgramNode::apply(
                ProgramNode::op("MUL"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "div",
            ProgramNode::apply(
                ProgramNode::op("DIV"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "mod",
            ProgramNode::apply(
                ProgramNode::op("MOD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );

        // ── Comparison ──
        mem.add_basic(
            "eq",
            ProgramNode::apply(
                ProgramNode::op("EQ"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "lt",
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "gt",
            ProgramNode::apply(
                ProgramNode::op("GT"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );

        // ── Logic ──
        mem.add_basic(
            "and",
            ProgramNode::apply(
                ProgramNode::op("AND"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "or",
            ProgramNode::apply(
                ProgramNode::op("OR"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        mem.add_basic(
            "not",
            ProgramNode::apply(ProgramNode::op("NOT"), vec![ProgramNode::atom("a")]),
        );

        // ── Iteration ──
        mem.add_basic(
            "count_loop",
            ProgramNode::iterate(
                ProgramNode::atom("i = 0"),
                ProgramNode::atom("i += 1"),
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("i"), ProgramNode::atom("n")],
                ),
            ),
        );

        // ── Accumulation ──
        mem.add_basic(
            "sum_reduce",
            ProgramNode::reduce(
                ProgramNode::op("ADD"),
                ProgramNode::typed("0", "INT"),
                ProgramNode::atom("arr"),
            ),
        );

        // ── Recursion ──
        mem.add_basic(
            "factorial",
            ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("EQ"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("0", "INT")],
                    ),
                    ProgramNode::typed("1", "INT"),
                    ProgramNode::apply(
                        ProgramNode::op("MUL"),
                        vec![
                            ProgramNode::atom("n"),
                            ProgramNode::apply(
                                ProgramNode::atom("factorial"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")],
                                )],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("factorial"),
            ),
        );

        // ── Higher-order ──
        mem.add_basic(
            "map_transform",
            ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("items")),
        );
        mem.add_basic(
            "filter_predicate",
            ProgramNode::filter(ProgramNode::atom("predicate"), ProgramNode::atom("items")),
        );
        mem.add_basic(
            "compose_fg",
            ProgramNode::compose(ProgramNode::atom("f"), ProgramNode::atom("g")),
        );

        // ── Branching ──
        mem.add_basic(
            "if_then_else",
            ProgramNode::branch(
                ProgramNode::atom("condition"),
                ProgramNode::atom("then_value"),
                ProgramNode::atom("else_value"),
            ),
        );

        // ── Collection ──
        mem.add_basic(
            "collect_source",
            ProgramNode::collect(ProgramNode::atom("source")),
        );

        // ── Sequence ──
        mem.add_basic(
            "sequence_ab",
            ProgramNode::seq(vec![
                ProgramNode::atom("step_a"),
                ProgramNode::atom("step_b"),
            ]),
        );

        // ── Extended: arithmetic with different variable names ──
        // (Fix 3: atom names matter in encoding, so include common variants)
        mem.add_basic(
            "add_xy",
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        );
        mem.add_basic(
            "mul_xy",
            ProgramNode::apply(
                ProgramNode::op("MUL"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        );
        mem.add_basic(
            "sub_xy",
            ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        );
        mem.add_basic(
            "lt_xy",
            ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
        );

        // ── Extended: multi-step compositions ──
        // Accumulate pattern: init → loop → accumulate → return
        mem.add_basic(
            "sum_loop",
            ProgramNode::seq(vec![
                ProgramNode::typed("sum", "INT"),
                ProgramNode::iterate(
                    ProgramNode::atom("sum = 0"),
                    ProgramNode::apply(
                        ProgramNode::op("ADD"),
                        vec![ProgramNode::atom("sum"), ProgramNode::atom("arr[i]")],
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("sum"),
            ]),
        );

        // Find max: branch inside loop
        mem.add_basic(
            "find_max",
            ProgramNode::seq(vec![
                ProgramNode::atom("max = arr[0]"),
                ProgramNode::iterate(
                    ProgramNode::atom("i = 1"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("GT"),
                            vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max")],
                        ),
                        ProgramNode::atom("max = arr[i]"),
                        ProgramNode::atom("/* no-op */"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("max"),
            ]),
        );

        // Contains: loop with early return
        mem.add_basic(
            "contains",
            ProgramNode::seq(vec![
                ProgramNode::iterate(
                    ProgramNode::atom("i = 0"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("EQ"),
                            vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("target")],
                        ),
                        ProgramNode::atom("return true"),
                        ProgramNode::atom("i += 1"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("false"),
            ]),
        );

        // Fibonacci recursive
        mem.add_basic(
            "fibonacci",
            ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                    ),
                    ProgramNode::atom("n"),
                    ProgramNode::apply(
                        ProgramNode::op("ADD"),
                        vec![
                            ProgramNode::apply(
                                ProgramNode::atom("fib"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")],
                                )],
                            ),
                            ProgramNode::apply(
                                ProgramNode::atom("fib"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                                )],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("fib"),
            ),
        );

        // Filter + map chain
        mem.add_basic(
            "filter_map",
            ProgramNode::seq(vec![
                ProgramNode::filter(ProgramNode::atom("predicate"), ProgramNode::atom("items")),
                ProgramNode::map(
                    ProgramNode::atom("transform"),
                    ProgramNode::atom("filtered"),
                ),
            ]),
        );

        // Reduce with multiply (product)
        mem.add_basic(
            "product_reduce",
            ProgramNode::reduce(
                ProgramNode::op("MUL"),
                ProgramNode::typed("1", "INT"),
                ProgramNode::atom("arr"),
            ),
        );

        mem
    }

    /// Add a learned pattern to the library.
    pub fn learn(&mut self, name: &str, node: ProgramNode, source: Option<&str>) {
        let encoding = node.encode();
        let sub_patterns = Self::decompose(name, &node);
        self.entries.push(ProgramMemoryEntry {
            name: name.to_string(),
            node,
            encoding,
            source: source.map(|s| s.to_string()),
            sub_patterns,
        });
    }

    /// Find the nearest entry by encoding similarity.
    ///
    /// Returns `None` only if the library is empty.
    pub fn nearest(&self, query: &BinaryHV) -> Option<&ProgramMemoryEntry> {
        self.entries.iter().max_by(|a, b| {
            a.encoding
                .similarity(query)
                .partial_cmp(&b.encoding.similarity(query))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all entries as (ProgramNode, BinaryHV) pairs for the explorer.
    pub fn as_library(&self) -> Vec<(ProgramNode, BinaryHV)> {
        self.entries
            .iter()
            .map(|e| (e.node.clone(), e.encoding))
            .collect()
    }

    /// Find the top-k nearest entries by encoding similarity.
    pub fn nearest_k(&self, query: &BinaryHV, k: usize) -> Vec<(&ProgramMemoryEntry, f32)> {
        let mut scored: Vec<_> = self
            .entries
            .iter()
            .map(|e| (e, e.encoding.similarity(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Find the nearest sub-pattern across all entries.
    /// Searches decomposed sub-trees for fine-grained matching.
    pub fn nearest_sub_pattern(&self, query: &BinaryHV) -> Option<(&ProgramNode, f32)> {
        let mut best: Option<(&ProgramNode, f32)> = None;
        for entry in &self.entries {
            for (_, node, enc) in &entry.sub_patterns {
                let sim = enc.similarity(query);
                if best.map_or(true, |(_, s)| sim > s) {
                    best = Some((node, sim));
                }
            }
        }
        best
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ── Internal ──

    /// Add a pattern with "basic" source tag and auto-decomposition.
    fn add_basic(&mut self, name: &str, node: ProgramNode) {
        let encoding = node.encode();
        let sub_patterns = Self::decompose(name, &node);
        self.entries.push(ProgramMemoryEntry {
            name: name.to_string(),
            node,
            encoding,
            source: Some("basic".to_string()),
            sub_patterns,
        });
    }

    /// Walk a ProgramNode tree and extract non-Atom sub-trees as named sub-patterns.
    fn decompose(parent_name: &str, node: &ProgramNode) -> Vec<(String, ProgramNode, BinaryHV)> {
        let mut subs = Vec::new();
        Self::decompose_inner(parent_name, node, &mut subs, 0);
        subs
    }

    fn decompose_inner(
        parent: &str,
        node: &ProgramNode,
        out: &mut Vec<(String, ProgramNode, BinaryHV)>,
        depth: usize,
    ) {
        // Only decompose compound nodes, skip atoms
        match node {
            ProgramNode::Atom(_) | ProgramNode::Typed(_, _) => {}

            ProgramNode::Apply { func, args } => {
                let name = format!("{parent}/apply_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, func, out, depth + 1);
                for (i, arg) in args.iter().enumerate() {
                    Self::decompose_inner(&name, arg, out, depth + 1 + i);
                }
            }
            ProgramNode::Sequence(steps) => {
                let name = format!("{parent}/seq_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                for (i, step) in steps.iter().enumerate() {
                    Self::decompose_inner(&name, step, out, depth + 1 + i);
                }
            }
            ProgramNode::Branch {
                condition,
                then_branch,
                else_branch,
            } => {
                let name = format!("{parent}/branch_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, condition, out, depth + 1);
                Self::decompose_inner(&name, then_branch, out, depth + 2);
                Self::decompose_inner(&name, else_branch, out, depth + 3);
            }
            ProgramNode::Map { func, collection } => {
                let name = format!("{parent}/map_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, func, out, depth + 1);
                Self::decompose_inner(&name, collection, out, depth + 2);
            }
            ProgramNode::Reduce {
                func,
                initial,
                collection,
            } => {
                let name = format!("{parent}/reduce_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, func, out, depth + 1);
                Self::decompose_inner(&name, initial, out, depth + 2);
                Self::decompose_inner(&name, collection, out, depth + 3);
            }
            ProgramNode::Filter {
                predicate,
                collection,
            } => {
                let name = format!("{parent}/filter_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, predicate, out, depth + 1);
                Self::decompose_inner(&name, collection, out, depth + 2);
            }
            ProgramNode::Compose(f, g) => {
                let name = format!("{parent}/compose_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, f, out, depth + 1);
                Self::decompose_inner(&name, g, out, depth + 2);
            }
            ProgramNode::Recurse {
                base_case,
                recursive_step,
            } => {
                let name = format!("{parent}/recurse_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, base_case, out, depth + 1);
                Self::decompose_inner(&name, recursive_step, out, depth + 2);
            }
            ProgramNode::Iterate {
                init,
                step,
                condition,
            } => {
                let name = format!("{parent}/iterate_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, init, out, depth + 1);
                Self::decompose_inner(&name, step, out, depth + 2);
                Self::decompose_inner(&name, condition, out, depth + 3);
            }
            ProgramNode::Collect(source) => {
                let name = format!("{parent}/collect_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                Self::decompose_inner(&name, source, out, depth + 1);
            }
            ProgramNode::Abstract(examples) => {
                let name = format!("{parent}/abstract_{depth}");
                out.push((name.clone(), node.clone(), node.encode()));
                for (i, ex) in examples.iter().enumerate() {
                    Self::decompose_inner(&name, ex, out, depth + 1 + i);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_library_size() {
        let mem = ProgramMemory::basic();
        assert!(
            mem.len() >= 15,
            "basic library should have at least 15 entries, got {}",
            mem.len()
        );
    }

    #[test]
    fn test_nearest_finds_similar() {
        let mem = ProgramMemory::basic();

        // Encode an ADD pattern and query for it — nearest should return "add"
        let add_node = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let add_hv = add_node.encode();

        let nearest = mem.nearest(&add_hv).expect("library not empty");
        assert_eq!(
            nearest.name, "add",
            "nearest to ADD encoding should be 'add'"
        );
    }

    #[test]
    fn test_decompose() {
        let seq = ProgramNode::seq(vec![
            ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("x"), ProgramNode::atom("y")],
            ),
            ProgramNode::atom("z"),
        ]);
        let subs = ProgramMemory::decompose("test", &seq);
        // Should have at least: the sequence itself + the Apply sub-node
        assert!(
            subs.len() >= 2,
            "decompose of Sequence(Apply, Atom) should produce >= 2 sub-patterns, got {}",
            subs.len()
        );
    }

    #[test]
    fn test_learn_adds_entry() {
        let mut mem = ProgramMemory::basic();
        let before = mem.len();
        mem.learn("custom", ProgramNode::atom("hello"), Some("test"));
        assert_eq!(mem.len(), before + 1);
    }

    #[test]
    fn test_as_library() {
        let mem = ProgramMemory::basic();
        let lib = mem.as_library();
        assert_eq!(lib.len(), mem.len());
    }
}
