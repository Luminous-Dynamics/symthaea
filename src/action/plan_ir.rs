// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, serializable execution-plan IR with stable content identity.
//!
//! [`Molecule`](super::primitives::Molecule) remains the ergonomic in-process
//! composition type. It can contain opaque Rust closures, however, so not every
//! molecule can be given an exact authority/evidence identity. This module is
//! the declarative boundary: every execution-relevant field is data, canonical
//! serialization is deterministic, and legacy closure-bearing control flow
//! fails closed during conversion.

use super::primitives::{Atom, DispatchTier, Molecule};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// Current schema version for the declarative plan document.
pub const PLAN_IR_VERSION: u32 = 1;
const PLAN_ID_DOMAIN: &[u8] = b"symthaea.action.plan-ir.v1\0";

/// Content identity of one validated [`PlanDocument`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlanId([u8; 32]);

impl PlanId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A path whose exact UTF-8 spelling is identity-significant.
///
/// The transaction layer may later constrain paths to a repository-relative
/// namespace. Plan IR itself deliberately does not rewrite absolute paths or
/// resolve symlinks because doing so would mutate the caller's declared intent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlanPath(String);

impl PlanPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_legacy(path: &Path, atom: &'static str) -> Result<Self, PlanIrError> {
        let value = path
            .to_str()
            .ok_or(PlanIrError::NonUtf8Path { atom })?;
        Ok(Self(value.to_string()))
    }
}

/// Versioned declarative plan whose identity can be bound to authorization,
/// execution observations, and evidence receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDocument {
    pub version: u32,
    pub root: PlanNode,
}

impl PlanDocument {
    pub fn new(root: PlanNode) -> Self {
        Self {
            version: PLAN_IR_VERSION,
            root,
        }
    }

    /// Convert a legacy molecule only when all of its semantics are explicit.
    pub fn try_from_legacy(molecule: &Molecule) -> Result<Self, PlanIrError> {
        Ok(Self::new(PlanNode::try_from(molecule)?))
    }

    /// Validate structural invariants before hashing or execution.
    pub fn validate(&self) -> Result<(), PlanIrError> {
        if self.version != PLAN_IR_VERSION {
            return Err(PlanIrError::UnsupportedVersion(self.version));
        }
        self.root.validate()
    }

    /// Domain-separated BLAKE3 identity over deterministic JSON bytes.
    ///
    /// All maps in the IR are [`BTreeMap`]s and struct/enum field order is
    /// schema-defined, so `serde_json` provides a deterministic representation
    /// for a fixed Plan IR version. A future schema that changes canonical bytes
    /// must use a new version/domain.
    pub fn id(&self) -> Result<PlanId, PlanIrError> {
        self.validate()?;
        let canonical = serde_json::to_vec(self)
            .map_err(|error| PlanIrError::Serialization(error.to_string()))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PLAN_ID_DOMAIN);
        hasher.update(&canonical);
        Ok(PlanId(*hasher.finalize().as_bytes()))
    }
}

/// Declarative control-flow tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanNode {
    Atom(PlanAtom),
    Sequence {
        steps: Vec<PlanNode>,
    },
    Parallel {
        branches: Vec<PlanNode>,
    },
    Conditional {
        predicate: PlanPredicate,
        then_plan: Box<PlanNode>,
        else_plan: Box<PlanNode>,
    },
    Until {
        body: Box<PlanNode>,
        predicate: PlanPredicate,
        max_iterations: usize,
    },
    Recovery {
        body: Box<PlanNode>,
        on_error: Box<PlanNode>,
    },
}

impl PlanNode {
    fn validate(&self) -> Result<(), PlanIrError> {
        match self {
            Self::Atom(_) => Ok(()),
            Self::Sequence { steps } => {
                if steps.is_empty() {
                    return Err(PlanIrError::EmptySequence);
                }
                for step in steps {
                    step.validate()?;
                }
                Ok(())
            }
            Self::Parallel { branches } => {
                if branches.is_empty() {
                    return Err(PlanIrError::EmptyParallel);
                }
                for branch in branches {
                    branch.validate()?;
                }
                Ok(())
            }
            Self::Conditional {
                then_plan,
                else_plan,
                ..
            } => {
                then_plan.validate()?;
                else_plan.validate()
            }
            Self::Until {
                body,
                max_iterations,
                ..
            } => {
                if *max_iterations == 0 {
                    return Err(PlanIrError::ZeroIterations);
                }
                body.validate()
            }
            Self::Recovery { body, on_error } => {
                body.validate()?;
                on_error.validate()
            }
        }
    }
}

/// Typed predicate vocabulary for content-addressable control flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum PlanPredicate {
    Always,
    Never,
    LastResultSucceeded,
    LastResultFailed,
    LastExitCodeEq(i32),
    LastBoolEq(bool),
    LastTextContains(String),
}

/// Declarative atom. Unlike lowering to `ActionIR`, internal effects such as
/// backend dispatch remain explicit and identity-significant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanAtom {
    ReadFile {
        path: PlanPath,
    },
    WriteFile {
        path: PlanPath,
        content: Vec<u8>,
    },
    ListDir {
        path: PlanPath,
        recursive: bool,
    },
    Exec {
        program: String,
        args: Vec<String>,
        working_dir: Option<PlanPath>,
        env: BTreeMap<String, String>,
    },
    Parse {
        path: PlanPath,
    },
    Encode {
        text: String,
    },
    Search {
        query: String,
        top_k: usize,
    },
    EditFile {
        path: PlanPath,
        old: String,
        new: String,
    },
    GrepFiles {
        pattern: String,
        path: PlanPath,
        recursive: bool,
    },
    DeleteFile {
        path: PlanPath,
    },
    Dispatch {
        prompt: String,
        tier: PlanDispatchTier,
    },
    Noop,
}

/// Serializable counterpart of [`DispatchTier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanDispatchTier {
    Native,
    LocalLlm,
    CloudLlm,
}

impl From<DispatchTier> for PlanDispatchTier {
    fn from(value: DispatchTier) -> Self {
        match value {
            DispatchTier::Native => Self::Native,
            DispatchTier::LocalLlm => Self::LocalLlm,
            DispatchTier::CloudLlm => Self::CloudLlm,
        }
    }
}

impl TryFrom<&Atom> for PlanAtom {
    type Error = PlanIrError;

    fn try_from(atom: &Atom) -> Result<Self, Self::Error> {
        match atom {
            Atom::ReadFile { path } => Ok(Self::ReadFile {
                path: PlanPath::from_legacy(path, "ReadFile")?,
            }),
            Atom::WriteFile { path, content } => Ok(Self::WriteFile {
                path: PlanPath::from_legacy(path, "WriteFile")?,
                content: content.clone(),
            }),
            Atom::ListDir { path, recursive } => Ok(Self::ListDir {
                path: PlanPath::from_legacy(path, "ListDir")?,
                recursive: *recursive,
            }),
            Atom::Exec {
                program,
                args,
                working_dir,
                env,
            } => Ok(Self::Exec {
                program: program.clone(),
                args: args.clone(),
                working_dir: working_dir
                    .as_deref()
                    .map(|path| PlanPath::from_legacy(path, "Exec"))
                    .transpose()?,
                env: env.clone(),
            }),
            Atom::Parse { path } => Ok(Self::Parse {
                path: PlanPath::from_legacy(path, "Parse")?,
            }),
            Atom::Encode { text } => Ok(Self::Encode { text: text.clone() }),
            Atom::Search { query, top_k } => Ok(Self::Search {
                query: query.clone(),
                top_k: *top_k,
            }),
            Atom::EditFile { path, old, new } => Ok(Self::EditFile {
                path: PlanPath::from_legacy(path, "EditFile")?,
                old: old.clone(),
                new: new.clone(),
            }),
            Atom::GrepFiles {
                pattern,
                path,
                recursive,
            } => Ok(Self::GrepFiles {
                pattern: pattern.clone(),
                path: PlanPath::from_legacy(path, "GrepFiles")?,
                recursive: *recursive,
            }),
            Atom::DeleteFile { path } => Ok(Self::DeleteFile {
                path: PlanPath::from_legacy(path, "DeleteFile")?,
            }),
            Atom::Dispatch { prompt, tier } => Ok(Self::Dispatch {
                prompt: prompt.clone(),
                tier: (*tier).into(),
            }),
            Atom::Noop => Ok(Self::Noop),
        }
    }
}

impl TryFrom<&Molecule> for PlanNode {
    type Error = PlanIrError;

    fn try_from(molecule: &Molecule) -> Result<Self, Self::Error> {
        match molecule {
            Molecule::Atom(atom) => Ok(Self::Atom(PlanAtom::try_from(atom)?)),
            Molecule::Sequence(first, second) => Ok(Self::Sequence {
                steps: vec![Self::try_from(first.as_ref())?, Self::try_from(second.as_ref())?],
            }),
            Molecule::Parallel(first, second) => Ok(Self::Parallel {
                branches: vec![Self::try_from(first.as_ref())?, Self::try_from(second.as_ref())?],
            }),
            Molecule::Conditional { .. } => Err(PlanIrError::OpaqueControlFlow(
                OpaqueControlFlowKind::Conditional,
            )),
            Molecule::Until { .. } => {
                Err(PlanIrError::OpaqueControlFlow(OpaqueControlFlowKind::Until))
            }
            Molecule::Recovery { .. } => Err(PlanIrError::OpaqueControlFlow(
                OpaqueControlFlowKind::Recovery,
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpaqueControlFlowKind {
    Conditional,
    Until,
    Recovery,
}

impl fmt::Display for OpaqueControlFlowKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conditional => write!(f, "conditional predicate"),
            Self::Until => write!(f, "until predicate"),
            Self::Recovery => write!(f, "recovery handler"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanIrError {
    UnsupportedVersion(u32),
    EmptySequence,
    EmptyParallel,
    ZeroIterations,
    NonUtf8Path { atom: &'static str },
    OpaqueControlFlow(OpaqueControlFlowKind),
    Serialization(String),
}

impl fmt::Display for PlanIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported Plan IR version {version}")
            }
            Self::EmptySequence => write!(f, "Plan IR sequence must contain at least one step"),
            Self::EmptyParallel => write!(f, "Plan IR parallel node must contain at least one branch"),
            Self::ZeroIterations => write!(f, "Plan IR until node must allow at least one iteration"),
            Self::NonUtf8Path { atom } => {
                write!(f, "legacy {atom} path is not valid UTF-8 and cannot be content-addressed")
            }
            Self::OpaqueControlFlow(kind) => write!(
                f,
                "legacy molecule contains opaque {kind}; migrate it to a typed PlanPredicate/PlanNode"
            ),
            Self::Serialization(error) => write!(f, "serialize Plan IR: {error}"),
        }
    }
}

impl std::error::Error for PlanIrError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn exec_plan(arg: &str) -> PlanDocument {
        PlanDocument::new(PlanNode::Atom(PlanAtom::Exec {
            program: "cargo".into(),
            args: vec![arg.into()],
            working_dir: Some(PlanPath::new(".")),
            env: BTreeMap::new(),
        }))
    }

    #[test]
    fn identical_plans_have_identical_ids() {
        assert_eq!(exec_plan("check").id().unwrap(), exec_plan("check").id().unwrap());
    }

    #[test]
    fn one_changed_argument_changes_id() {
        assert_ne!(exec_plan("check").id().unwrap(), exec_plan("test").id().unwrap());
    }

    #[test]
    fn sequence_order_changes_id() {
        let a = PlanNode::Atom(PlanAtom::ReadFile {
            path: PlanPath::new("a.rs"),
        });
        let b = PlanNode::Atom(PlanAtom::ReadFile {
            path: PlanPath::new("b.rs"),
        });
        let ab = PlanDocument::new(PlanNode::Sequence {
            steps: vec![a.clone(), b.clone()],
        });
        let ba = PlanDocument::new(PlanNode::Sequence {
            steps: vec![b, a],
        });
        assert_ne!(ab.id().unwrap(), ba.id().unwrap());
    }

    #[test]
    fn dispatch_prompt_and_tier_are_identity_significant() {
        let base = PlanDocument::new(PlanNode::Atom(PlanAtom::Dispatch {
            prompt: "repair E0308".into(),
            tier: PlanDispatchTier::Native,
        }));
        let changed_prompt = PlanDocument::new(PlanNode::Atom(PlanAtom::Dispatch {
            prompt: "repair E0382".into(),
            tier: PlanDispatchTier::Native,
        }));
        let changed_tier = PlanDocument::new(PlanNode::Atom(PlanAtom::Dispatch {
            prompt: "repair E0308".into(),
            tier: PlanDispatchTier::LocalLlm,
        }));
        assert_ne!(base.id().unwrap(), changed_prompt.id().unwrap());
        assert_ne!(base.id().unwrap(), changed_tier.id().unwrap());
    }

    #[test]
    fn typed_predicate_is_identity_significant() {
        let branch = |predicate| {
            PlanDocument::new(PlanNode::Conditional {
                predicate,
                then_plan: Box::new(PlanNode::Atom(PlanAtom::Noop)),
                else_plan: Box::new(PlanNode::Atom(PlanAtom::ReadFile {
                    path: PlanPath::new("Cargo.toml"),
                })),
            })
        };
        assert_ne!(
            branch(PlanPredicate::LastBoolEq(true)).id().unwrap(),
            branch(PlanPredicate::LastBoolEq(false)).id().unwrap()
        );
    }

    #[test]
    fn serialization_round_trip_preserves_id() {
        let plan = PlanDocument::new(PlanNode::Until {
            body: Box::new(PlanNode::Atom(PlanAtom::Noop)),
            predicate: PlanPredicate::LastResultSucceeded,
            max_iterations: 3,
        });
        let encoded = serde_json::to_vec(&plan).unwrap();
        let decoded: PlanDocument = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(plan, decoded);
        assert_eq!(plan.id().unwrap(), decoded.id().unwrap());
    }

    #[test]
    fn closure_free_legacy_molecule_preserves_atoms() {
        let molecule = Molecule::atom(Atom::read("src/lib.rs")).then(Molecule::atom(
            Atom::cargo_check(PathBuf::from(".")),
        ));
        let plan = PlanDocument::try_from_legacy(&molecule).unwrap();
        let PlanNode::Sequence { steps } = plan.root else {
            panic!("expected sequence");
        };
        assert!(matches!(
            &steps[0],
            PlanNode::Atom(PlanAtom::ReadFile { path }) if path.as_str() == "src/lib.rs"
        ));
        assert!(matches!(
            &steps[1],
            PlanNode::Atom(PlanAtom::Exec { program, args, .. })
                if program == "cargo" && args.first().map(String::as_str) == Some("check")
        ));
    }

    #[test]
    fn legacy_dispatch_is_not_lowered_to_noop() {
        let molecule = Molecule::atom(Atom::dispatch_local("implement the selected repair"));
        let plan = PlanDocument::try_from_legacy(&molecule).unwrap();
        assert!(matches!(
            plan.root,
            PlanNode::Atom(PlanAtom::Dispatch {
                tier: PlanDispatchTier::LocalLlm,
                ..
            })
        ));
    }

    #[test]
    fn opaque_legacy_control_flow_fails_closed() {
        let molecule = Molecule::branch(
            |_| true,
            Molecule::atom(Atom::Noop),
            Molecule::atom(Atom::Noop),
        );
        assert!(matches!(
            PlanDocument::try_from_legacy(&molecule),
            Err(PlanIrError::OpaqueControlFlow(
                OpaqueControlFlowKind::Conditional
            ))
        ));
    }

    #[test]
    fn invalid_structures_do_not_receive_ids() {
        let empty = PlanDocument::new(PlanNode::Sequence { steps: vec![] });
        assert!(matches!(empty.id(), Err(PlanIrError::EmptySequence)));

        let zero_loop = PlanDocument::new(PlanNode::Until {
            body: Box::new(PlanNode::Atom(PlanAtom::Noop)),
            predicate: PlanPredicate::Always,
            max_iterations: 0,
        });
        assert!(matches!(zero_loop.id(), Err(PlanIrError::ZeroIterations)));
    }
}
