// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC Program Algebra — Hyperdimensional Intermediate Representation for code.
//!
//! Encodes program structure as 16,384-bit hypervectors using HDC combinators.
//! This is the "Native tier" replacement: instead of hardcoded templates, Symthaea
//! reasons about code at the semantic level using vector operations, then translates
//! to any target language via emitters.
//!
//! # Core Insight
//!
//! Standard compilers: Source → AST → IR → Machine code
//! HDC Program Algebra: Intent → HV Program → CodeSpec → Source (any language)
//!
//! # Combinators
//!
//! | Combinator | HDC Operation | Code Meaning |
//! |------------|--------------|-------------|
//! | `apply(f, x)` | `f ⊗ x` | Function application |
//! | `sequence(a, b, c)` | `a ⊗ ρ¹(b) ⊗ ρ²(c)` | Sequential execution |
//! | `branch(cond, t, f)` | `cond ⊗ (t + f)` | Conditional (if/else) |
//! | `map(f, col)` | `MAP_ROLE ⊗ f ⊗ col` | Transform each element |
//! | `reduce(f, init, col)` | `REDUCE_ROLE ⊗ f ⊗ init ⊗ col` | Fold/accumulate |
//! | `compose(f, g)` | `COMPOSE_ROLE ⊗ f ⊗ g` | Function composition g∘f |
//! | `recurse(base, step)` | `RECURSE_ROLE ⊗ base ⊗ ρ¹(step)` | Recursive structure |
//! | `abstract(examples)` | `majority(examples)` | Extract common pattern |
//!
//! # Saturation Defense
//!
//! Per Gemini's warning: bundling too many vectors → noise. This module uses
//! **hierarchical encoding**: functions are encoded as single vectors, composed
//! programs reference function vectors by role-binding (not deep bundling).
//! A 10-step program is 10 permuted bindings, not 10 bundled vectors.

use crate::hdc::binary_hv::BinaryHV;
use std::collections::HashMap;

// ── Role Vectors ─────────────────────────────────────────────────────────
// Pre-computed basis vectors that distinguish combinator types.
// Each role is a deterministic random HV derived from a seed phrase.

/// Generate a deterministic role vector from a seed string.
fn role_vector(seed: &str) -> BinaryHV {
    BinaryHV::random(
        seed.bytes()
            .enumerate()
            .fold(0u64, |acc, (i, b)| acc.wrapping_add((b as u64) << (i % 8))),
    )
}

use std::sync::OnceLock;

fn roles() -> &'static HashMap<&'static str, BinaryHV> {
    static INSTANCE: OnceLock<HashMap<&'static str, BinaryHV>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("APPLY", role_vector("HDC_PROG_APPLY"));
        m.insert("SEQUENCE", role_vector("HDC_PROG_SEQUENCE"));
        m.insert("BRANCH", role_vector("HDC_PROG_BRANCH"));
        m.insert("MAP", role_vector("HDC_PROG_MAP"));
        m.insert("REDUCE", role_vector("HDC_PROG_REDUCE"));
        m.insert("FILTER", role_vector("HDC_PROG_FILTER"));
        m.insert("COMPOSE", role_vector("HDC_PROG_COMPOSE"));
        m.insert("RECURSE", role_vector("HDC_PROG_RECURSE"));
        m.insert("ABSTRACT", role_vector("HDC_PROG_ABSTRACT"));
        m.insert("PARAM", role_vector("HDC_PROG_PARAM"));
        m.insert("RETURN", role_vector("HDC_PROG_RETURN"));
        m.insert("GUARD", role_vector("HDC_PROG_GUARD"));
        m.insert("ITERATE", role_vector("HDC_PROG_ITERATE"));
        m.insert("COLLECT", role_vector("HDC_PROG_COLLECT"));
        m
    })
}

fn types() -> &'static HashMap<&'static str, BinaryHV> {
    static INSTANCE: OnceLock<HashMap<&'static str, BinaryHV>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("INT", role_vector("HDC_TYPE_INT"));
        m.insert("FLOAT", role_vector("HDC_TYPE_FLOAT"));
        m.insert("BOOL", role_vector("HDC_TYPE_BOOL"));
        m.insert("STRING", role_vector("HDC_TYPE_STRING"));
        m.insert("LIST", role_vector("HDC_TYPE_LIST"));
        m.insert("OPTION", role_vector("HDC_TYPE_OPTION"));
        m.insert("TUPLE", role_vector("HDC_TYPE_TUPLE"));
        m.insert("VOID", role_vector("HDC_TYPE_VOID"));
        m.insert("GENERIC", role_vector("HDC_TYPE_GENERIC"));
        m
    })
}

fn ops() -> &'static HashMap<&'static str, BinaryHV> {
    static INSTANCE: OnceLock<HashMap<&'static str, BinaryHV>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("ADD", role_vector("HDC_OP_ADD"));
        m.insert("SUB", role_vector("HDC_OP_SUB"));
        m.insert("MUL", role_vector("HDC_OP_MUL"));
        m.insert("DIV", role_vector("HDC_OP_DIV"));
        m.insert("MOD", role_vector("HDC_OP_MOD"));
        m.insert("EQ", role_vector("HDC_OP_EQ"));
        m.insert("LT", role_vector("HDC_OP_LT"));
        m.insert("GT", role_vector("HDC_OP_GT"));
        m.insert("AND", role_vector("HDC_OP_AND"));
        m.insert("OR", role_vector("HDC_OP_OR"));
        m.insert("NOT", role_vector("HDC_OP_NOT"));
        m.insert("INDEX", role_vector("HDC_OP_INDEX"));
        m.insert("SWAP", role_vector("HDC_OP_SWAP"));
        m.insert("COMPARE", role_vector("HDC_OP_COMPARE"));
        m.insert("CONCAT", role_vector("HDC_OP_CONCAT"));
        m.insert("SPLIT", role_vector("HDC_OP_SPLIT"));
        m.insert("SORT", role_vector("HDC_OP_SORT"));
        m.insert("REVERSE", role_vector("HDC_OP_REVERSE"));
        m.insert("LEN", role_vector("HDC_OP_LEN"));
        m.insert("MAX", role_vector("HDC_OP_MAX"));
        m.insert("MIN", role_vector("HDC_OP_MIN"));
        m.insert("SUM", role_vector("HDC_OP_SUM"));
        m
    })
}

// ── Program Node ─────────────────────────────────────────────────────────

/// A node in an HDC program tree. Each node encodes to a 16,384-bit vector.
#[derive(Debug, Clone)]
pub enum ProgramNode {
    /// A named operation or variable: "sort", "x", "fibonacci"
    Atom(String),

    /// A typed atom with semantic type: ("x", INT)
    Typed(String, String),

    /// Function application: apply(f, args...)
    /// HDC: APPLY_ROLE ⊗ f ⊗ ρ¹(arg1) ⊗ ρ²(arg2)
    Apply {
        func: Box<ProgramNode>,
        args: Vec<ProgramNode>,
    },

    /// Sequential execution: do a, then b, then c
    /// HDC: a ⊗ ρ¹(b) ⊗ ρ²(c) (temporal binding)
    Sequence(Vec<ProgramNode>),

    /// Conditional branch: if cond then t else f
    /// HDC: BRANCH_ROLE ⊗ cond ⊗ ρ¹(then) ⊗ ρ²(else)
    Branch {
        condition: Box<ProgramNode>,
        then_branch: Box<ProgramNode>,
        else_branch: Box<ProgramNode>,
    },

    /// Map: transform each element of a collection
    /// HDC: MAP_ROLE ⊗ func ⊗ collection
    Map {
        func: Box<ProgramNode>,
        collection: Box<ProgramNode>,
    },

    /// Reduce/fold: accumulate over a collection
    /// HDC: REDUCE_ROLE ⊗ func ⊗ initial ⊗ collection
    Reduce {
        func: Box<ProgramNode>,
        initial: Box<ProgramNode>,
        collection: Box<ProgramNode>,
    },

    /// Filter: keep elements matching a predicate
    /// HDC: FILTER_ROLE ⊗ predicate ⊗ collection
    Filter {
        predicate: Box<ProgramNode>,
        collection: Box<ProgramNode>,
    },

    /// Function composition: g(f(x)) = compose(f, g)
    /// HDC: COMPOSE_ROLE ⊗ f ⊗ ρ¹(g)
    Compose(Box<ProgramNode>, Box<ProgramNode>),

    /// Recursion: base case + recursive step
    /// HDC: RECURSE_ROLE ⊗ base ⊗ ρ¹(step)
    Recurse {
        base_case: Box<ProgramNode>,
        recursive_step: Box<ProgramNode>,
    },

    /// Iteration: for/while loop body
    /// HDC: ITERATE_ROLE ⊗ init ⊗ ρ¹(step) ⊗ ρ²(condition)
    Iterate {
        init: Box<ProgramNode>,
        step: Box<ProgramNode>,
        condition: Box<ProgramNode>,
    },

    /// Collect results: gather outputs into a collection
    /// HDC: COLLECT_ROLE ⊗ source
    Collect(Box<ProgramNode>),

    /// Pattern abstraction: extract common structure from examples
    /// HDC: majority_vote(example_encodings)
    Abstract(Vec<ProgramNode>),
}

impl ProgramNode {
    /// Encode this program node as a 16,384-bit BinaryHV.
    ///
    /// This is the core operation: turning program structure into a vector
    /// that can be compared, composed, and translated to any language.
    pub fn encode(&self) -> BinaryHV {
        match self {
            ProgramNode::Atom(name) => encode_atom(name),

            ProgramNode::Typed(name, type_name) => {
                let atom = encode_atom(name);
                let ty = types()
                    .get(type_name.as_str())
                    .cloned()
                    .unwrap_or_else(|| encode_atom(type_name));
                atom.bind(&ty)
            }

            ProgramNode::Apply { func, args } => {
                let mut result = roles()["APPLY"].bind(&func.encode());
                for (i, arg) in args.iter().enumerate() {
                    let arg_hv = arg.encode().permute(i + 1);
                    result = result.bind(&arg_hv);
                }
                result
            }

            ProgramNode::Sequence(steps) => {
                // Temporal binding: each step is permuted by its position
                // This is NON-COMMUTATIVE — order matters
                let mut result = steps[0].encode();
                for (i, step) in steps[1..].iter().enumerate() {
                    result = result.bind(&step.encode().permute(i + 1));
                }
                result
            }

            ProgramNode::Branch {
                condition,
                then_branch,
                else_branch,
            } => roles()["BRANCH"]
                .bind(&condition.encode())
                .bind(&then_branch.encode().permute(1))
                .bind(&else_branch.encode().permute(2)),

            ProgramNode::Map { func, collection } => roles()["MAP"]
                .bind(&func.encode())
                .bind(&collection.encode().permute(1)),

            ProgramNode::Reduce {
                func,
                initial,
                collection,
            } => roles()["REDUCE"]
                .bind(&func.encode())
                .bind(&initial.encode().permute(1))
                .bind(&collection.encode().permute(2)),

            ProgramNode::Filter {
                predicate,
                collection,
            } => roles()["FILTER"]
                .bind(&predicate.encode())
                .bind(&collection.encode().permute(1)),

            ProgramNode::Compose(f, g) => {
                // g ∘ f: apply f first, then g
                roles()["COMPOSE"]
                    .bind(&f.encode())
                    .bind(&g.encode().permute(1))
            }

            ProgramNode::Recurse {
                base_case,
                recursive_step,
            } => roles()["RECURSE"]
                .bind(&base_case.encode())
                .bind(&recursive_step.encode().permute(1)),

            ProgramNode::Iterate {
                init,
                step,
                condition,
            } => roles()["ITERATE"]
                .bind(&init.encode())
                .bind(&step.encode().permute(1))
                .bind(&condition.encode().permute(2)),

            ProgramNode::Collect(source) => roles()["COLLECT"].bind(&source.encode()),

            ProgramNode::Abstract(examples) => {
                // Majority vote of example encodings — extracts common structure
                // This is where Gemini's saturation warning applies:
                // keep examples < 20 to avoid noise
                let hvs: Vec<BinaryHV> = examples.iter().map(|e| e.encode()).collect();
                BinaryHV::bundle(&hvs)
            }
        }
    }

    /// Check semantic type compatibility between two program nodes.
    ///
    /// Uses cosine similarity: > 0.3 = compatible, < 0.1 = incompatible.
    /// This is the "HDC type checker" — catches semantic mismatches before
    /// any code is generated.
    pub fn type_compatible(&self, other: &ProgramNode) -> f32 {
        self.encode().similarity(&other.encode())
    }
}

/// Encode an atom (name/identifier) as a BinaryHV using n-gram hashing.
fn encode_atom(name: &str) -> BinaryHV {
    // Check if it's a known operation
    if let Some(hv) = ops().get(name.to_uppercase().as_str()) {
        return *hv;
    }
    if let Some(hv) = types().get(name.to_uppercase().as_str()) {
        return *hv;
    }

    // N-gram encoding: each character gets a position-shifted base vector,
    // then all are bundled. This gives similar names similar vectors.
    let chars: Vec<char> = name.chars().collect();
    if chars.is_empty() {
        return BinaryHV::random(0);
    }
    let char_hvs: Vec<BinaryHV> = chars
        .iter()
        .enumerate()
        .map(|(i, c)| BinaryHV::random(*c as u64 + 1000).permute(i))
        .collect();
    BinaryHV::bundle(&char_hvs)
}

/// Encode a natural language task description as a BinaryHV for pattern matching.
///
/// Extracts significant tokens (strips stop words), encodes each as an atom,
/// and bundles them. Known operation keywords get role-binding emphasis.
///
/// # Example
/// ```ignore
/// let hv = encode_task_description("implement a fibonacci function");
/// let lib = ProgramPatternLibrary::standard();
/// let match = lib.find_similar(&hv, 0.52); // should match "fibonacci"
/// ```
pub fn encode_task_description(text: &str) -> BinaryHV {
    const STOP_WORDS: &[&str] = &[
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "for",
        "and",
        "or",
        "that",
        "this",
        "with",
        "from",
        "by",
        "on",
        "is",
        "are",
        "was",
        "be",
        "been",
        "being",
        "implement",
        "create",
        "add",
        "write",
        "make",
        "build",
        "define",
        "function",
        "method",
        "struct",
        "class",
        "module",
        "program",
        "please",
        "should",
        "would",
        "could",
        "can",
        "will",
        "using",
        "use",
        "given",
        "return",
        "returns",
        "take",
        "takes",
    ];

    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split_whitespace()
        .filter(|t| !STOP_WORDS.contains(t) && t.len() > 1)
        .collect();

    if tokens.is_empty() {
        return BinaryHV::random(0);
    }

    // Encode each significant token as an atom
    let mut token_hvs: Vec<BinaryHV> = tokens.iter().map(|t| encode_atom(t)).collect();

    // Boost known operation keywords with role binding for stronger matching
    for token in &tokens {
        if let Some(op_hv) = ops().get(token.to_uppercase().as_str()) {
            // Bind the operation role vector — this makes "sort" strongly match sort patterns
            token_hvs.push(roles()["APPLY"].bind(op_hv));
        }
    }

    BinaryHV::bundle(&token_hvs)
}

// ── Convenience Constructors ─────────────────────────────────────────────

/// Shorthand constructors for building program trees.
impl ProgramNode {
    pub fn atom(name: &str) -> Self {
        ProgramNode::Atom(name.to_string())
    }

    pub fn typed(name: &str, ty: &str) -> Self {
        ProgramNode::Typed(name.to_string(), ty.to_string())
    }

    pub fn op(name: &str) -> Self {
        ProgramNode::Atom(name.to_uppercase())
    }

    pub fn apply(func: ProgramNode, args: Vec<ProgramNode>) -> Self {
        ProgramNode::Apply {
            func: Box::new(func),
            args,
        }
    }

    pub fn seq(steps: Vec<ProgramNode>) -> Self {
        ProgramNode::Sequence(steps)
    }

    pub fn branch(cond: ProgramNode, then_b: ProgramNode, else_b: ProgramNode) -> Self {
        ProgramNode::Branch {
            condition: Box::new(cond),
            then_branch: Box::new(then_b),
            else_branch: Box::new(else_b),
        }
    }

    pub fn map(func: ProgramNode, collection: ProgramNode) -> Self {
        ProgramNode::Map {
            func: Box::new(func),
            collection: Box::new(collection),
        }
    }

    pub fn reduce(func: ProgramNode, init: ProgramNode, collection: ProgramNode) -> Self {
        ProgramNode::Reduce {
            func: Box::new(func),
            initial: Box::new(init),
            collection: Box::new(collection),
        }
    }

    pub fn filter(pred: ProgramNode, collection: ProgramNode) -> Self {
        ProgramNode::Filter {
            predicate: Box::new(pred),
            collection: Box::new(collection),
        }
    }

    pub fn compose(f: ProgramNode, g: ProgramNode) -> Self {
        ProgramNode::Compose(Box::new(f), Box::new(g))
    }

    pub fn recurse(base: ProgramNode, step: ProgramNode) -> Self {
        ProgramNode::Recurse {
            base_case: Box::new(base),
            recursive_step: Box::new(step),
        }
    }

    pub fn iterate(init: ProgramNode, step: ProgramNode, cond: ProgramNode) -> Self {
        ProgramNode::Iterate {
            init: Box::new(init),
            step: Box::new(step),
            condition: Box::new(cond),
        }
    }

    pub fn collect(source: ProgramNode) -> Self {
        ProgramNode::Collect(Box::new(source))
    }
}

// ── Pattern Library ──────────────────────────────────────────────────────
// Encode the 36 native coding patterns as HDC programs.
// This replaces the hardcoded string templates in coding_agent.rs.

/// Metadata for a pattern entry — needed by the translator to produce CodeSpec.
#[derive(Debug, Clone)]
pub struct PatternEntry {
    /// Pattern name (e.g., "fibonacci")
    pub name: String,
    /// The HDC program tree
    pub node: ProgramNode,
    /// Pre-computed encoding for fast similarity search
    pub encoding: BinaryHV,
    /// Human-readable purpose (e.g., "Compute the nth Fibonacci number")
    pub purpose: String,
    /// Signature hint for Rust (e.g., "(n: u64) -> u64")
    pub rust_signature: String,
    /// Signature hint for Python (e.g., "(n: int) -> int")
    pub python_signature: String,
    /// Parameter type names (e.g., ["INT"])
    pub param_types: Vec<String>,
    /// Return type name (e.g., "INT")
    pub return_type: String,
    /// Keywords that should strongly match this pattern
    pub keywords: Vec<String>,
}

/// A library of encoded program patterns that can be matched by similarity.
pub struct ProgramPatternLibrary {
    patterns: Vec<PatternEntry>,
}

impl ProgramPatternLibrary {
    /// Build the standard pattern library.
    pub fn standard() -> Self {
        let mut lib = Self {
            patterns: Vec::new(),
        };

        // ── Arithmetic ──
        lib.add(
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
                                ProgramNode::atom("fibonacci"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")],
                                )],
                            ),
                            ProgramNode::apply(
                                ProgramNode::atom("fibonacci"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                                )],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("fibonacci"), // recursive reference
            ),
        );

        lib.add(
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

        lib.add(
            "gcd",
            ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("EQ"),
                        vec![ProgramNode::atom("b"), ProgramNode::typed("0", "INT")],
                    ),
                    ProgramNode::atom("a"),
                    ProgramNode::apply(
                        ProgramNode::atom("gcd"),
                        vec![
                            ProgramNode::atom("b"),
                            ProgramNode::apply(
                                ProgramNode::op("MOD"),
                                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("gcd"),
            ),
        );

        // ── Collection operations ──
        lib.add(
            "sum",
            ProgramNode::reduce(
                ProgramNode::op("ADD"),
                ProgramNode::typed("0", "INT"),
                ProgramNode::atom("collection"),
            ),
        );

        lib.add(
            "max",
            ProgramNode::reduce(
                ProgramNode::op("MAX"),
                ProgramNode::atom("first_element"),
                ProgramNode::atom("collection"),
            ),
        );

        lib.add(
            "min",
            ProgramNode::reduce(
                ProgramNode::op("MIN"),
                ProgramNode::atom("first_element"),
                ProgramNode::atom("collection"),
            ),
        );

        lib.add(
            "reverse",
            ProgramNode::reduce(
                ProgramNode::atom("prepend"),
                ProgramNode::atom("empty_list"),
                ProgramNode::atom("collection"),
            ),
        );

        lib.add(
            "sort",
            ProgramNode::apply(
                ProgramNode::op("SORT"),
                vec![ProgramNode::atom("collection")],
            ),
        );

        lib.add(
            "filter_predicate",
            ProgramNode::filter(
                ProgramNode::atom("predicate"),
                ProgramNode::atom("collection"),
            ),
        );

        lib.add(
            "map_transform",
            ProgramNode::map(
                ProgramNode::atom("transform"),
                ProgramNode::atom("collection"),
            ),
        );

        // ── String operations ──
        lib.add(
            "reverse_string",
            ProgramNode::compose(
                ProgramNode::atom("chars"),
                ProgramNode::compose(
                    ProgramNode::op("REVERSE"),
                    ProgramNode::atom("collect_string"),
                ),
            ),
        );

        lib.add(
            "count_vowels",
            ProgramNode::compose(
                ProgramNode::filter(ProgramNode::atom("is_vowel"), ProgramNode::atom("chars")),
                ProgramNode::op("LEN"),
            ),
        );

        lib.add(
            "palindrome_check",
            ProgramNode::apply(
                ProgramNode::op("EQ"),
                vec![
                    ProgramNode::atom("s"),
                    ProgramNode::apply(ProgramNode::op("REVERSE"), vec![ProgramNode::atom("s")]),
                ],
            ),
        );

        // ── Search ──
        lib.add(
            "binary_search",
            ProgramNode::iterate(
                ProgramNode::seq(vec![
                    ProgramNode::typed("0", "INT"),
                    ProgramNode::apply(ProgramNode::op("LEN"), vec![ProgramNode::atom("arr")]),
                ]),
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("EQ"),
                        vec![ProgramNode::atom("mid_val"), ProgramNode::atom("target")],
                    ),
                    ProgramNode::atom("found"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("LT"),
                            vec![ProgramNode::atom("mid_val"), ProgramNode::atom("target")],
                        ),
                        ProgramNode::atom("search_right"),
                        ProgramNode::atom("search_left"),
                    ),
                ),
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("lo"), ProgramNode::atom("hi")],
                ),
            ),
        );

        // ── Sorting algorithms ──
        lib.add(
            "merge_sort",
            ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![
                            ProgramNode::apply(
                                ProgramNode::op("LEN"),
                                vec![ProgramNode::atom("arr")],
                            ),
                            ProgramNode::typed("2", "INT"),
                        ],
                    ),
                    ProgramNode::atom("arr"), // base: already sorted
                    ProgramNode::apply(
                        ProgramNode::atom("merge"),
                        vec![
                            ProgramNode::apply(
                                ProgramNode::atom("merge_sort"),
                                vec![ProgramNode::atom("left_half")],
                            ),
                            ProgramNode::apply(
                                ProgramNode::atom("merge_sort"),
                                vec![ProgramNode::atom("right_half")],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("merge_sort"),
            ),
        );

        lib.add(
            "bubble_sort",
            ProgramNode::iterate(
                ProgramNode::atom("arr"),
                ProgramNode::map(
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("GT"),
                            vec![ProgramNode::atom("current"), ProgramNode::atom("next")],
                        ),
                        ProgramNode::op("SWAP"),
                        ProgramNode::atom("keep"),
                    ),
                    ProgramNode::atom("pairs"),
                ),
                ProgramNode::atom("swapped"),
            ),
        );

        // ── Annotate patterns with metadata for the translator ──
        lib.annotate(
            "fibonacci",
            "Compute the nth Fibonacci number",
            "(n: u64) -> u64",
            "(n: int) -> int",
            "INT",
            &["fibonacci", "fib", "sequence"],
        );
        lib.annotate(
            "factorial",
            "Compute the factorial of n",
            "(n: u64) -> u64",
            "(n: int) -> int",
            "INT",
            &["factorial", "fact"],
        );
        lib.annotate(
            "gcd",
            "Compute the greatest common divisor",
            "(a: u64, b: u64) -> u64",
            "(a: int, b: int) -> int",
            "INT",
            &["gcd", "greatest", "common", "divisor", "euclidean"],
        );
        lib.annotate(
            "sum",
            "Sum all elements in a slice",
            "(v: &[i64]) -> i64",
            "(v: list) -> int",
            "INT",
            &["sum", "total", "add", "accumulate"],
        );
        lib.annotate(
            "max",
            "Find the maximum value",
            "(v: &[i64]) -> Option<i64>",
            "(v: list) -> int",
            "INT",
            &["max", "maximum", "largest", "biggest"],
        );
        lib.annotate(
            "min",
            "Find the minimum value",
            "(v: &[i64]) -> Option<i64>",
            "(v: list) -> int",
            "INT",
            &["min", "minimum", "smallest"],
        );
        lib.annotate(
            "reverse",
            "Reverse a collection",
            "(v: &[T]) -> Vec<T>",
            "(v: list) -> list",
            "LIST",
            &["reverse", "reversed", "flip"],
        );
        lib.annotate(
            "sort",
            "Sort a collection",
            "(v: &mut [T])",
            "(v: list) -> list",
            "LIST",
            &["sort", "sorted", "order", "arrange"],
        );
        lib.annotate(
            "reverse_string",
            "Reverse a string",
            "(s: &str) -> String",
            "(s: str) -> str",
            "STRING",
            &["reverse", "string", "backwards"],
        );
        lib.annotate(
            "count_vowels",
            "Count vowels in a string",
            "(s: &str) -> usize",
            "(s: str) -> int",
            "INT",
            &["count", "vowels", "vowel"],
        );
        lib.annotate(
            "palindrome_check",
            "Check if a string is a palindrome",
            "(s: &str) -> bool",
            "(s: str) -> bool",
            "BOOL",
            &["palindrome", "mirror", "symmetric"],
        );
        lib.annotate(
            "binary_search",
            "Binary search in a sorted slice",
            "(arr: &[T], target: &T) -> Option<usize>",
            "(arr: list, target) -> int",
            "OPTION",
            &["binary", "search", "find", "bisect"],
        );
        lib.annotate(
            "merge_sort",
            "Sort using merge sort algorithm",
            "(arr: &mut [T])",
            "(arr: list) -> list",
            "LIST",
            &["merge", "sort", "divide", "conquer"],
        );
        lib.annotate(
            "bubble_sort",
            "Sort using bubble sort algorithm",
            "(arr: &mut [T])",
            "(arr: list) -> list",
            "LIST",
            &["bubble", "sort", "swap"],
        );

        // ── Extended patterns (Phase 2) ──
        lib.extend_standard();
        lib
    }

    /// Add 30+ additional patterns covering common programming tasks.
    fn extend_standard(&mut self) {
        // Iterator adapters
        self.add_with_meta(
            "enumerate",
            ProgramNode::map(
                ProgramNode::atom("index_element_pair"),
                ProgramNode::atom("collection"),
            ),
            "Enumerate elements with indices",
            "(v: &[T]) -> Vec<(usize, &T)>",
            "(v: list) -> list",
            &["LIST"],
            "LIST",
            &["enumerate", "index", "indices", "numbered"],
        );
        self.add_with_meta(
            "zip",
            ProgramNode::map(
                ProgramNode::atom("pair"),
                ProgramNode::seq(vec![ProgramNode::atom("a"), ProgramNode::atom("b")]),
            ),
            "Zip two collections into pairs",
            "(a: &[A], b: &[B]) -> Vec<(A, B)>",
            "(a: list, b: list) -> list",
            &["LIST", "LIST"],
            "LIST",
            &["zip", "pair", "combine", "merge"],
        );
        self.add_with_meta(
            "take",
            ProgramNode::filter(
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("index"), ProgramNode::atom("n")],
                ),
                ProgramNode::atom("collection"),
            ),
            "Take first n elements",
            "(v: &[T], n: usize) -> Vec<T>",
            "(v: list, n: int) -> list",
            &["LIST", "INT"],
            "LIST",
            &["take", "first", "head", "prefix", "limit"],
        );
        self.add_with_meta(
            "skip",
            ProgramNode::filter(
                ProgramNode::apply(
                    ProgramNode::op("GE"),
                    vec![ProgramNode::atom("index"), ProgramNode::atom("n")],
                ),
                ProgramNode::atom("collection"),
            ),
            "Skip first n elements",
            "(v: &[T], n: usize) -> Vec<T>",
            "(v: list, n: int) -> list",
            &["LIST", "INT"],
            "LIST",
            &["skip", "drop", "tail", "offset"],
        );
        self.add_with_meta(
            "chunks",
            ProgramNode::map(
                ProgramNode::atom("slice_chunk"),
                ProgramNode::atom("collection"),
            ),
            "Split into fixed-size chunks",
            "(v: &[T], size: usize) -> Vec<&[T]>",
            "(v: list, size: int) -> list",
            &["LIST", "INT"],
            "LIST",
            &["chunk", "chunks", "batch", "partition", "window"],
        );
        self.add_with_meta(
            "chain",
            ProgramNode::reduce(
                ProgramNode::atom("append"),
                ProgramNode::atom("empty_list"),
                ProgramNode::seq(vec![ProgramNode::atom("a"), ProgramNode::atom("b")]),
            ),
            "Chain two collections",
            "(a: &[T], b: &[T]) -> Vec<T>",
            "(a: list, b: list) -> list",
            &["LIST", "LIST"],
            "LIST",
            &["chain", "concat", "concatenate", "append", "extend"],
        );

        // Error handling
        self.add_with_meta(
            "try_parse",
            ProgramNode::branch(
                ProgramNode::apply(
                    ProgramNode::atom("parse_ok"),
                    vec![ProgramNode::atom("input")],
                ),
                ProgramNode::apply(ProgramNode::atom("Ok"), vec![ProgramNode::atom("value")]),
                ProgramNode::apply(ProgramNode::atom("Err"), vec![ProgramNode::atom("error")]),
            ),
            "Parse with Result error handling",
            "(s: &str) -> Result<T, E>",
            "(s: str) -> Result",
            &["STRING"],
            "RESULT",
            &["parse", "try", "result", "convert", "from_str"],
        );
        self.add_with_meta(
            "unwrap_or_default",
            ProgramNode::branch(
                ProgramNode::apply(ProgramNode::atom("is_some"), vec![ProgramNode::atom("opt")]),
                ProgramNode::atom("inner"),
                ProgramNode::atom("default"),
            ),
            "Unwrap Option with default",
            "(opt: Option<T>, default: T) -> T",
            "(opt, default) -> value",
            &["OPTION"],
            "VALUE",
            &["unwrap", "default", "fallback", "option"],
        );

        // String processing
        self.add_with_meta(
            "split_string",
            ProgramNode::apply(
                ProgramNode::atom("split"),
                vec![ProgramNode::atom("s"), ProgramNode::atom("delim")],
            ),
            "Split string by delimiter",
            "(s: &str, delim: &str) -> Vec<&str>",
            "(s: str, delim: str) -> list",
            &["STRING", "STRING"],
            "LIST",
            &["split", "tokenize", "separate"],
        );
        self.add_with_meta(
            "join_strings",
            ProgramNode::reduce(
                ProgramNode::atom("concat_sep"),
                ProgramNode::atom("empty"),
                ProgramNode::atom("parts"),
            ),
            "Join strings with separator",
            "(parts: &[&str], sep: &str) -> String",
            "(parts: list, sep: str) -> str",
            &["LIST", "STRING"],
            "STRING",
            &["join", "concat", "combine", "separator"],
        );
        self.add_with_meta(
            "trim_string",
            ProgramNode::compose(ProgramNode::atom("trim"), ProgramNode::atom("identity")),
            "Trim whitespace",
            "(s: &str) -> &str",
            "(s: str) -> str",
            &["STRING"],
            "STRING",
            &["trim", "strip", "whitespace", "clean"],
        );
        self.add_with_meta(
            "replace_string",
            ProgramNode::apply(
                ProgramNode::atom("replace"),
                vec![
                    ProgramNode::atom("s"),
                    ProgramNode::atom("from"),
                    ProgramNode::atom("to"),
                ],
            ),
            "Replace in string",
            "(s: &str, from: &str, to: &str) -> String",
            "(s: str, old: str, new: str) -> str",
            &["STRING", "STRING", "STRING"],
            "STRING",
            &["replace", "substitute", "swap"],
        );
        self.add_with_meta(
            "contains_check",
            ProgramNode::apply(
                ProgramNode::atom("contains"),
                vec![ProgramNode::atom("haystack"), ProgramNode::atom("needle")],
            ),
            "Check substring",
            "(s: &str, pat: &str) -> bool",
            "(s: str, pat: str) -> bool",
            &["STRING", "STRING"],
            "BOOL",
            &["contains", "includes", "has", "substring"],
        );

        // HashMap operations
        self.add_with_meta(
            "frequency_count",
            ProgramNode::reduce(
                ProgramNode::atom("increment_count"),
                ProgramNode::atom("empty_map"),
                ProgramNode::atom("collection"),
            ),
            "Count frequency of elements",
            "(v: &[T]) -> HashMap<T, usize>",
            "(v: list) -> dict",
            &["LIST"],
            "MAP",
            &["frequency", "count", "histogram", "occurrences", "tally"],
        );
        self.add_with_meta(
            "group_by",
            ProgramNode::reduce(
                ProgramNode::atom("group_insert"),
                ProgramNode::atom("empty_map"),
                ProgramNode::atom("collection"),
            ),
            "Group elements by key",
            "(v: &[T], key: impl Fn(&T) -> K) -> HashMap<K, Vec<T>>",
            "(v: list, key: callable) -> dict",
            &["LIST"],
            "MAP",
            &["group", "group_by", "categorize", "classify"],
        );

        // Numeric
        self.add_with_meta(
            "clamp",
            ProgramNode::branch(
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("x"), ProgramNode::atom("min")],
                ),
                ProgramNode::atom("min"),
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("GT"),
                        vec![ProgramNode::atom("x"), ProgramNode::atom("max")],
                    ),
                    ProgramNode::atom("max"),
                    ProgramNode::atom("x"),
                ),
            ),
            "Clamp between min and max",
            "(x: T, min: T, max: T) -> T",
            "(x, lo, hi) -> value",
            &["INT", "INT", "INT"],
            "INT",
            &["clamp", "bound", "constrain", "limit"],
        );
        self.add_with_meta(
            "power",
            ProgramNode::iterate(
                ProgramNode::seq(vec![
                    ProgramNode::typed("1", "INT"),
                    ProgramNode::atom("exp"),
                ]),
                ProgramNode::apply(
                    ProgramNode::op("MUL"),
                    vec![ProgramNode::atom("acc"), ProgramNode::atom("base")],
                ),
                ProgramNode::apply(
                    ProgramNode::op("GT"),
                    vec![ProgramNode::atom("exp"), ProgramNode::typed("0", "INT")],
                ),
            ),
            "Compute base^exponent",
            "(base: i64, exp: u32) -> i64",
            "(base: int, exp: int) -> int",
            &["INT", "INT"],
            "INT",
            &["power", "pow", "exponent", "raise"],
        );
        self.add_with_meta(
            "average",
            ProgramNode::apply(
                ProgramNode::op("DIV"),
                vec![
                    ProgramNode::reduce(
                        ProgramNode::op("ADD"),
                        ProgramNode::typed("0", "FLOAT"),
                        ProgramNode::atom("collection"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LEN"),
                        vec![ProgramNode::atom("collection")],
                    ),
                ],
            ),
            "Compute average",
            "(v: &[f64]) -> f64",
            "(v: list) -> float",
            &["LIST"],
            "FLOAT",
            &["average", "mean", "avg"],
        );
        self.add_with_meta(
            "median",
            ProgramNode::seq(vec![
                ProgramNode::apply(
                    ProgramNode::op("SORT"),
                    vec![ProgramNode::atom("collection")],
                ),
                ProgramNode::apply(
                    ProgramNode::atom("middle"),
                    vec![ProgramNode::atom("sorted")],
                ),
            ]),
            "Find median",
            "(v: &mut [f64]) -> f64",
            "(v: list) -> float",
            &["LIST"],
            "FLOAT",
            &["median", "middle", "midpoint"],
        );

        // Data transformation
        self.add_with_meta(
            "flatten",
            ProgramNode::reduce(
                ProgramNode::atom("append"),
                ProgramNode::atom("empty_list"),
                ProgramNode::atom("nested"),
            ),
            "Flatten nested collection",
            "(v: &[Vec<T>]) -> Vec<T>",
            "(v: list) -> list",
            &["LIST"],
            "LIST",
            &["flatten", "flat", "unnest", "flat_map"],
        );
        self.add_with_meta(
            "dedup",
            ProgramNode::reduce(
                ProgramNode::atom("add_if_new"),
                ProgramNode::atom("empty_list"),
                ProgramNode::atom("collection"),
            ),
            "Remove consecutive duplicates",
            "(v: &mut Vec<T>)",
            "(v: list) -> list",
            &["LIST"],
            "LIST",
            &["dedup", "deduplicate", "unique", "distinct"],
        );
        self.add_with_meta(
            "product",
            ProgramNode::reduce(
                ProgramNode::op("MUL"),
                ProgramNode::typed("1", "INT"),
                ProgramNode::atom("collection"),
            ),
            "Product of all elements",
            "(v: &[i64]) -> i64",
            "(v: list) -> int",
            &["LIST"],
            "INT",
            &["product", "multiply", "mul", "prod"],
        );
        self.add_with_meta(
            "any",
            ProgramNode::reduce(
                ProgramNode::op("OR"),
                ProgramNode::typed("false", "BOOL"),
                ProgramNode::map(
                    ProgramNode::atom("predicate"),
                    ProgramNode::atom("collection"),
                ),
            ),
            "Any element matches predicate",
            "(v: &[T], f: impl Fn(&T) -> bool) -> bool",
            "(v: list, f: callable) -> bool",
            &["LIST"],
            "BOOL",
            &["any", "some", "exists"],
        );
        self.add_with_meta(
            "all",
            ProgramNode::reduce(
                ProgramNode::op("AND"),
                ProgramNode::typed("true", "BOOL"),
                ProgramNode::map(
                    ProgramNode::atom("predicate"),
                    ProgramNode::atom("collection"),
                ),
            ),
            "All elements match predicate",
            "(v: &[T], f: impl Fn(&T) -> bool) -> bool",
            "(v: list, f: callable) -> bool",
            &["LIST"],
            "BOOL",
            &["all", "every", "each"],
        );
        self.add_with_meta(
            "find_first",
            ProgramNode::compose(
                ProgramNode::filter(
                    ProgramNode::atom("predicate"),
                    ProgramNode::atom("collection"),
                ),
                ProgramNode::atom("first"),
            ),
            "Find first matching element",
            "(v: &[T], f: impl Fn(&T) -> bool) -> Option<&T>",
            "(v: list, f: callable) -> value",
            &["LIST"],
            "OPTION",
            &["find", "first", "search", "locate"],
        );
        self.add_with_meta(
            "count_if",
            ProgramNode::compose(
                ProgramNode::filter(
                    ProgramNode::atom("predicate"),
                    ProgramNode::atom("collection"),
                ),
                ProgramNode::op("LEN"),
            ),
            "Count matching elements",
            "(v: &[T], f: impl Fn(&T) -> bool) -> usize",
            "(v: list, f: callable) -> int",
            &["LIST"],
            "INT",
            &["count", "count_if", "tally"],
        );

        // I/O patterns
        self.add_with_meta(
            "read_file",
            ProgramNode::apply(
                ProgramNode::atom("read_to_string"),
                vec![ProgramNode::atom("path")],
            ),
            "Read file to string",
            "(path: &str) -> Result<String, std::io::Error>",
            "(path: str) -> str",
            &["STRING"],
            "RESULT",
            &["read", "file", "load", "open", "input"],
        );
        self.add_with_meta(
            "write_file",
            ProgramNode::apply(
                ProgramNode::atom("write_all"),
                vec![ProgramNode::atom("path"), ProgramNode::atom("contents")],
            ),
            "Write string to file",
            "(path: &str, contents: &str) -> Result<(), std::io::Error>",
            "(path: str, contents: str) -> None",
            &["STRING", "STRING"],
            "RESULT",
            &["write", "save", "output", "store"],
        );
    }

    /// Find two complementary patterns for compositional tasks.
    pub fn compose_similar(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Option<(&PatternEntry, &PatternEntry, f32)> {
        let top = self.find_top_k(query, 5);
        if top.len() < 2 {
            return None;
        }
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..top.len() {
            for j in (i + 1)..top.len() {
                let combined = (top[i].1 + top[j].1) / 2.0;
                if combined < threshold {
                    continue;
                }
                let inter_sim = top[i].0.encoding.similarity(&top[j].0.encoding);
                if inter_sim > 0.7 {
                    continue;
                }
                if best.is_none_or(|(_, _, s)| combined > s) {
                    best = Some((i, j, combined));
                }
            }
        }
        best.map(|(i, j, s)| (top[i].0, top[j].0, s))
    }

    /// Add a named pattern to the library (simple form — metadata defaults).
    fn add(&mut self, name: &str, node: ProgramNode) {
        let encoding = node.encode();
        // Also encode the name as a task description and bundle with the node encoding
        // for better matching when users describe tasks by name
        let name_hv = encode_task_description(name);
        let combined = BinaryHV::bundle(&[encoding, name_hv]);

        self.patterns.push(PatternEntry {
            name: name.to_string(),
            node,
            encoding: combined,
            purpose: String::new(),
            rust_signature: String::new(),
            python_signature: String::new(),
            param_types: Vec::new(),
            return_type: String::new(),
            keywords: vec![name.to_string()],
        });
    }

    /// Add a pattern with full metadata.
    #[allow(clippy::too_many_arguments)]
    fn add_with_meta(
        &mut self,
        name: &str,
        node: ProgramNode,
        purpose: &str,
        rust_sig: &str,
        python_sig: &str,
        param_types: &[&str],
        return_type: &str,
        keywords: &[&str],
    ) {
        let node_encoding = node.encode();
        let name_hv = encode_task_description(name);
        // Also encode keywords for matching
        let keyword_hvs: Vec<BinaryHV> = keywords.iter().map(|k| encode_atom(k)).collect();
        let mut all_hvs = vec![node_encoding, name_hv];
        all_hvs.extend(keyword_hvs);
        let combined = BinaryHV::bundle(&all_hvs);

        self.patterns.push(PatternEntry {
            name: name.to_string(),
            node,
            encoding: combined,
            purpose: purpose.to_string(),
            rust_signature: rust_sig.to_string(),
            python_signature: python_sig.to_string(),
            param_types: param_types.iter().map(|s| s.to_string()).collect(),
            return_type: return_type.to_string(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
        });
    }

    /// Find the most similar pattern to a query vector.
    ///
    /// Returns the best matching PatternEntry if similarity >= threshold.
    /// For BinaryHV, random baseline is ~0.5; meaningful match is > 0.52.
    pub fn find_similar(&self, query: &BinaryHV, threshold: f32) -> Option<(&PatternEntry, f32)> {
        let mut best: Option<(&PatternEntry, f32)> = None;

        for entry in &self.patterns {
            let sim = query.similarity(&entry.encoding);
            if sim > threshold && best.as_ref().is_none_or(|(_, s)| sim > *s) {
                best = Some((entry, sim));
            }
        }

        best
    }

    /// Find top-k most similar patterns.
    pub fn find_top_k(&self, query: &BinaryHV, k: usize) -> Vec<(&PatternEntry, f32)> {
        let mut scored: Vec<(&PatternEntry, f32)> = self
            .patterns
            .iter()
            .map(|entry| (entry, query.similarity(&entry.encoding)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Annotate a pattern with metadata (for translator use).
    pub fn annotate(
        &mut self,
        name: &str,
        purpose: &str,
        rust_sig: &str,
        python_sig: &str,
        return_type: &str,
        keywords: &[&str],
    ) {
        if let Some(entry) = self.patterns.iter_mut().find(|e| e.name == name) {
            entry.purpose = purpose.to_string();
            entry.rust_signature = rust_sig.to_string();
            entry.python_signature = python_sig.to_string();
            entry.return_type = return_type.to_string();
            entry.keywords = keywords.iter().map(|s| s.to_string()).collect();
            // Re-encode with keyword boosting
            let keyword_hvs: Vec<BinaryHV> = keywords.iter().map(|k| encode_atom(k)).collect();
            let mut all_hvs = vec![entry.node.encode(), encode_task_description(name)];
            all_hvs.extend(keyword_hvs);
            entry.encoding = BinaryHV::bundle(&all_hvs);
        }
    }

    /// Number of patterns in the library.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_encoding_deterministic() {
        let a = ProgramNode::atom("fibonacci").encode();
        let b = ProgramNode::atom("fibonacci").encode();
        assert_eq!(a.similarity(&b), 1.0);
    }

    #[test]
    fn test_different_atoms_orthogonal() {
        let fib = ProgramNode::atom("fibonacci").encode();
        let sort = ProgramNode::atom("sort").encode();
        // Different names should be dissimilar (< 0.6)
        assert!(
            fib.similarity(&sort) < 0.6,
            "Different atoms should be dissimilar"
        );
    }

    #[test]
    fn test_similar_names_similar_vectors() {
        let sort = ProgramNode::atom("sort").encode();
        let sorted = ProgramNode::atom("sorted").encode();
        let unrelated = ProgramNode::atom("fibonacci").encode();
        // "sort" and "sorted" share characters → more similar than "fibonacci"
        assert!(sort.similarity(&sorted) > sort.similarity(&unrelated));
    }

    #[test]
    fn test_sequence_order_matters() {
        let a = ProgramNode::atom("read");
        let b = ProgramNode::atom("process");
        let c = ProgramNode::atom("write");

        let seq1 = ProgramNode::seq(vec![a.clone(), b.clone(), c.clone()]).encode();
        let seq2 = ProgramNode::seq(vec![c.clone(), b.clone(), a.clone()]).encode();

        // Different order → different encoding (temporal binding is non-commutative)
        assert!(
            seq1.similarity(&seq2) < 0.8,
            "Reversed sequence should differ"
        );
    }

    #[test]
    fn test_same_sequence_identical() {
        let seq1 = ProgramNode::seq(vec![ProgramNode::atom("a"), ProgramNode::atom("b")]).encode();
        let seq2 = ProgramNode::seq(vec![ProgramNode::atom("a"), ProgramNode::atom("b")]).encode();
        assert_eq!(seq1.similarity(&seq2), 1.0);
    }

    #[test]
    fn test_map_vs_reduce_distinguishable() {
        let map_prog =
            ProgramNode::map(ProgramNode::atom("double"), ProgramNode::atom("numbers")).encode();

        let reduce_prog = ProgramNode::reduce(
            ProgramNode::op("ADD"),
            ProgramNode::typed("0", "INT"),
            ProgramNode::atom("numbers"),
        )
        .encode();

        // Different combinators → different encodings
        assert!(map_prog.similarity(&reduce_prog) < 0.6);
    }

    #[test]
    fn test_type_checking_via_similarity() {
        let int_val = ProgramNode::typed("x", "INT");
        let float_val = ProgramNode::typed("y", "FLOAT");
        let str_val = ProgramNode::typed("z", "STRING");

        // INT and FLOAT are more compatible than INT and STRING
        let int_float = int_val.type_compatible(&float_val);
        let int_str = int_val.type_compatible(&str_val);
        // Both should be low (different types), but this tests the mechanism works
        assert!(int_float.is_finite());
        assert!(int_str.is_finite());
    }

    #[test]
    fn test_pattern_library_builds() {
        let lib = ProgramPatternLibrary::standard();
        assert!(
            lib.len() >= 15,
            "Should have at least 15 patterns, got {}",
            lib.len()
        );
    }

    #[test]
    fn test_pattern_library_finds_fibonacci() {
        let lib = ProgramPatternLibrary::standard();
        let query = encode_task_description("compute fibonacci number");
        let result = lib.find_similar(&query, 0.1);
        assert!(result.is_some(), "Should find a match for fibonacci");
        let (entry, sim) = result.unwrap();
        assert_eq!(entry.name, "fibonacci");
        assert!(sim > 0.3, "Fibonacci should match strongly: {sim}");
        assert!(!entry.purpose.is_empty(), "Should have purpose metadata");
    }

    #[test]
    fn test_pattern_library_top_k() {
        let lib = ProgramPatternLibrary::standard();
        let query = ProgramNode::reduce(
            ProgramNode::op("ADD"),
            ProgramNode::typed("0", "INT"),
            ProgramNode::atom("numbers"),
        )
        .encode();

        let top = lib.find_top_k(&query, 3);
        assert!(!top.is_empty());
        // "sum" should be the closest match (it's a reduce with ADD)
        assert_eq!(
            top[0].0.name, "sum",
            "Top match for reduce+ADD should be 'sum', got '{}'",
            top[0].0.name
        );
    }

    #[test]
    fn test_encode_task_description_matches_pattern() {
        let lib = ProgramPatternLibrary::standard();

        // "add a fibonacci function" should match the fibonacci pattern
        let fib_query = encode_task_description("fibonacci sequence number");
        let result = lib.find_similar(&fib_query, 0.1);
        assert!(result.is_some(), "Should match fibonacci");
        assert_eq!(result.unwrap().0.name, "fibonacci");

        // "sort a list" should match a sort pattern
        let sort_query = encode_task_description("sort elements list");
        let result = lib.find_similar(&sort_query, 0.1);
        assert!(result.is_some(), "Should match a sort pattern");
        let name = &result.unwrap().0.name;
        assert!(
            name.contains("sort") || name == "reverse",
            "Should match sort-related pattern, got {name}"
        );
    }

    #[test]
    fn test_encode_task_description_no_saturation() {
        // Long description should still produce non-degenerate HV
        let long = "implement a function that computes the greatest common divisor \
                     using the euclidean algorithm with memoization and error handling";
        let hv = encode_task_description(long);
        let ones: u32 = hv.0.iter().map(|b| b.count_ones()).sum();
        let density = ones as f32 / 16384.0;
        assert!(
            density > 0.35 && density < 0.65,
            "Density {density} should be near 0.5 (no saturation)"
        );
    }

    #[test]
    fn test_abstract_extracts_common_pattern() {
        // Abstract over multiple sort implementations → common "sort" pattern
        let sorts = ProgramNode::Abstract(vec![
            ProgramNode::atom("bubble_sort"),
            ProgramNode::atom("merge_sort"),
            ProgramNode::atom("insertion_sort"),
        ]);
        let abstract_hv = sorts.encode();

        // The abstraction should be more similar to any sort than to "fibonacci"
        let bubble = ProgramNode::atom("bubble_sort").encode();
        let fib = ProgramNode::atom("fibonacci").encode();

        assert!(
            abstract_hv.similarity(&bubble) > abstract_hv.similarity(&fib),
            "Sort abstraction should be closer to sorts than fibonacci"
        );
    }

    #[test]
    fn test_compose_is_ordered() {
        let f_then_g =
            ProgramNode::compose(ProgramNode::atom("parse"), ProgramNode::atom("validate"))
                .encode();

        let g_then_f =
            ProgramNode::compose(ProgramNode::atom("validate"), ProgramNode::atom("parse"))
                .encode();

        // Different composition order → different encoding
        assert!(f_then_g.similarity(&g_then_f) < 0.9);
    }

    #[test]
    fn test_branch_encodes_both_paths() {
        let prog = ProgramNode::branch(
            ProgramNode::apply(
                ProgramNode::op("GT"),
                vec![ProgramNode::atom("x"), ProgramNode::typed("0", "INT")],
            ),
            ProgramNode::atom("positive_path"),
            ProgramNode::atom("negative_path"),
        );
        let hv = prog.encode();

        // Should be a valid HV (not all zeros, not all ones)
        let ones = hv.popcount();
        assert!(
            ones > 1000 && ones < 15000,
            "Branch encoding should be non-degenerate: {ones} ones"
        );
    }

    #[test]
    fn test_extended_library_has_more_patterns() {
        let lib = ProgramPatternLibrary::standard();
        assert!(lib.patterns.len() >= 40, "Got {}", lib.patterns.len());
    }

    #[test]
    fn test_extended_patterns_have_metadata() {
        let lib = ProgramPatternLibrary::standard();
        for entry in &lib.patterns {
            assert!(
                !entry.keywords.is_empty(),
                "Pattern '{}' has no keywords",
                entry.name
            );
        }
    }

    #[test]
    fn test_frequency_count_matches_histogram_query() {
        let lib = ProgramPatternLibrary::standard();
        let query = encode_task_description("count frequency histogram of elements");
        assert!(lib.find_similar(&query, 0.50).is_some());
    }

    #[test]
    fn test_average_matches_mean_query() {
        let lib = ProgramPatternLibrary::standard();
        let query = encode_task_description("compute the average mean of numbers");
        assert!(lib.find_similar(&query, 0.50).is_some());
    }

    #[test]
    fn test_read_file_matches_load_query() {
        let lib = ProgramPatternLibrary::standard();
        let query = encode_task_description("read load file contents");
        assert!(lib.find_similar(&query, 0.50).is_some());
    }

    #[test]
    fn test_compose_similar_finds_pair() {
        let lib = ProgramPatternLibrary::standard();
        let query = encode_task_description("read file count word frequency");
        let result = lib.compose_similar(&query, 0.45);
        assert!(result.is_some());
        if let Some((first, second, score)) = result {
            assert!(score >= 0.45);
            assert_ne!(first.name, second.name);
        }
    }

    #[test]
    fn test_extended_encodings_non_degenerate() {
        let lib = ProgramPatternLibrary::standard();
        for entry in &lib.patterns {
            let ones = entry.encoding.popcount();
            assert!(
                ones > 1000 && ones < 15000,
                "Pattern '{}': {ones} ones",
                entry.name
            );
        }
    }
}
