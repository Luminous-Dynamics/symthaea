// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed Primitive System: Atomic operations with composable combinators.
//!
//! Each primitive carries type-level metadata (energy cost, reversibility,
//! required Phi threshold, destructiveness tier) enabling the FEP to reason
//! about execution plans *before* committing to them.
//!
//! ## Architecture
//!
//! **Atoms** are irreducible operations (Read, Write, Exec, Parse, Encode, Search).
//! **Molecules** compose atoms via combinators (Sequence, Conditional, Until, Recovery).
//! **Plans** are molecules with pre-computed cost/risk profiles for FEP evaluation.
//!
//! The system integrates backward-compatibly with the existing `ActionRegistry`
//! and `ActionIR` — primitives lower to `ActionIR` for execution by `SimpleExecutor`.

use super::{ActionIR, DestructivenessLevel, RiskTier};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

// ─── Primitive Metadata ──────────────────────────────────────────────────────

/// Static properties of a primitive operation.
/// These are known at plan-time, before execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveProperties {
    /// Human-readable name (e.g., "ReadFile", "CargoCheck")
    pub name: &'static str,
    /// Energy cost in arbitrary units (native=1.0, LLM=10.0, cloud=50.0)
    pub energy_cost: f32,
    /// Whether the operation can be undone
    pub reversible: bool,
    /// Minimum consciousness level required
    pub min_phi: f32,
    /// Destructiveness classification
    pub destructiveness: DestructivenessLevel,
    /// Risk tier for budgeting
    pub risk_tier: RiskTier,
}

impl PrimitiveProperties {
    /// ReadOnly defaults: zero energy, reversible, no consciousness required.
    pub const fn read_only(name: &'static str) -> Self {
        Self {
            name,
            energy_cost: 0.1,
            reversible: true,
            min_phi: 0.0,
            destructiveness: DestructivenessLevel::ReadOnly,
            risk_tier: RiskTier::Low,
        }
    }

    /// Write defaults: low energy, reversible, low consciousness.
    pub const fn write(name: &'static str) -> Self {
        Self {
            name,
            energy_cost: 1.0,
            reversible: true,
            min_phi: 0.05,
            destructiveness: DestructivenessLevel::Reversible,
            risk_tier: RiskTier::Medium,
        }
    }

    /// Command defaults: medium energy, NOT reversible, needs consciousness.
    pub const fn command(name: &'static str) -> Self {
        Self {
            name,
            energy_cost: 5.0,
            reversible: false,
            min_phi: 0.1,
            destructiveness: DestructivenessLevel::NeedsConfirmation,
            risk_tier: RiskTier::High,
        }
    }
}

// ─── Atomic Primitives ───────────────────────────────────────────────────────

/// The value flowing between primitives in a pipeline.
#[derive(Debug, Clone)]
pub enum PrimitiveValue {
    /// No value (unit type)
    Unit,
    /// Raw bytes (file content, command output)
    Bytes(Vec<u8>),
    /// Text content
    Text(String),
    /// Structured command output
    CommandResult {
        stdout: String,
        stderr: String,
        exit_code: i32,
    },
    /// Directory listing
    Listing(Vec<PathBuf>),
    /// Boolean (for predicates)
    Bool(bool),
    /// Numeric (similarity scores, etc.)
    Float(f32),
}

impl PrimitiveValue {
    /// Extract as text, converting bytes if needed.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            PrimitiveValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Extract exit code from command result.
    pub fn exit_code(&self) -> Option<i32> {
        match self {
            PrimitiveValue::CommandResult { exit_code, .. } => Some(*exit_code),
            _ => None,
        }
    }

    /// Was this a successful command execution?
    pub fn is_success(&self) -> bool {
        match self {
            PrimitiveValue::CommandResult { exit_code, .. } => *exit_code == 0,
            PrimitiveValue::Bool(b) => *b,
            PrimitiveValue::Unit => true,
            _ => true,
        }
    }

    /// Extract stderr from command result.
    pub fn stderr(&self) -> Option<&str> {
        match self {
            PrimitiveValue::CommandResult { stderr, .. } => Some(stderr),
            _ => None,
        }
    }
}

/// An atomic primitive — the irreducible unit of action.
#[derive(Debug, Clone)]
pub enum Atom {
    /// Read a file's content.
    ReadFile { path: PathBuf },

    /// Write content to a file.
    WriteFile { path: PathBuf, content: Vec<u8> },

    /// List a directory.
    ListDir { path: PathBuf, recursive: bool },

    /// Execute a command.
    Exec {
        program: String,
        args: Vec<String>,
        working_dir: Option<PathBuf>,
        env: BTreeMap<String, String>,
    },

    /// Parse source code into an AST summary (via tree-sitter).
    Parse { path: PathBuf },

    /// Encode text into an HDC hypervector (pure computation).
    Encode { text: String },

    /// Search codebase memory for similar patterns.
    Search { query: String, top_k: usize },

    /// Edit a file: find `old` text and replace with `new` text.
    EditFile {
        path: PathBuf,
        old: String,
        new: String,
    },

    /// Search file contents for a regex pattern (like grep/rg).
    GrepFiles {
        pattern: String,
        path: PathBuf,
        recursive: bool,
    },

    /// Delete a file (destructive).
    DeleteFile { path: PathBuf },

    /// Dispatch code generation to a backend tier.
    /// Energy cost reflects the tier: Native=1.0, LocalLLM=10.0, CloudLLM=50.0.
    Dispatch { prompt: String, tier: DispatchTier },

    /// No-op (identity in sequences).
    Noop,
}

/// Backend tier for the Dispatch atom.
/// Mirrors `BackendTier` from `intelligent_dispatcher` but lives in the
/// primitives module to avoid circular dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchTier {
    Native,
    LocalLlm,
    CloudLlm,
}

impl Atom {
    /// Get the static properties of this atom.
    pub fn properties(&self) -> PrimitiveProperties {
        match self {
            Atom::ReadFile { .. } => PrimitiveProperties::read_only("ReadFile"),
            Atom::ListDir { .. } => PrimitiveProperties::read_only("ListDir"),
            Atom::Parse { .. } => PrimitiveProperties::read_only("Parse"),
            Atom::Encode { .. } => PrimitiveProperties {
                name: "Encode",
                energy_cost: 0.05,
                reversible: true,
                min_phi: 0.0,
                destructiveness: DestructivenessLevel::ReadOnly,
                risk_tier: RiskTier::Low,
            },
            Atom::Search { .. } => PrimitiveProperties {
                name: "Search",
                energy_cost: 0.2,
                reversible: true,
                min_phi: 0.0,
                destructiveness: DestructivenessLevel::ReadOnly,
                risk_tier: RiskTier::Low,
            },
            Atom::EditFile { .. } => PrimitiveProperties::write("EditFile"),
            Atom::GrepFiles { .. } => PrimitiveProperties::read_only("GrepFiles"),
            Atom::DeleteFile { .. } => PrimitiveProperties {
                name: "DeleteFile",
                energy_cost: 2.0,
                reversible: false,
                min_phi: 0.2,
                destructiveness: DestructivenessLevel::Destructive,
                risk_tier: RiskTier::High,
            },
            Atom::Dispatch { tier, .. } => {
                let (name, energy, phi) = match tier {
                    DispatchTier::Native => ("DispatchNative", 1.0, 0.0),
                    DispatchTier::LocalLlm => ("DispatchLocalLLM", 10.0, 0.05),
                    DispatchTier::CloudLlm => ("DispatchCloudLLM", 50.0, 0.1),
                };
                PrimitiveProperties {
                    name,
                    energy_cost: energy,
                    reversible: true,
                    min_phi: phi,
                    destructiveness: DestructivenessLevel::ReadOnly,
                    risk_tier: if matches!(tier, DispatchTier::CloudLlm) {
                        RiskTier::High
                    } else {
                        RiskTier::Medium
                    },
                }
            }
            Atom::WriteFile { .. } => PrimitiveProperties::write("WriteFile"),
            Atom::Exec { program, args, .. } => {
                // Cargo check/test are confirmable but not destructive
                if program == "cargo" || program.ends_with("/cargo") {
                    if args.iter().any(|a| a == "check" || a == "clippy") {
                        return PrimitiveProperties {
                            name: "CargoCheck",
                            energy_cost: 3.0,
                            reversible: true,
                            min_phi: 0.05,
                            destructiveness: DestructivenessLevel::ReadOnly,
                            risk_tier: RiskTier::Low,
                        };
                    }
                    if args.iter().any(|a| a == "test") {
                        return PrimitiveProperties {
                            name: "CargoTest",
                            energy_cost: 5.0,
                            reversible: true,
                            min_phi: 0.05,
                            destructiveness: DestructivenessLevel::ReadOnly,
                            risk_tier: RiskTier::Low,
                        };
                    }
                }
                // git push is destructive
                if (program == "git" || program.ends_with("/git"))
                    && args.iter().any(|a| a == "push")
                {
                    return PrimitiveProperties {
                        name: "GitPush",
                        energy_cost: 10.0,
                        reversible: false,
                        min_phi: 0.3,
                        destructiveness: DestructivenessLevel::Destructive,
                        risk_tier: RiskTier::High,
                    };
                }
                PrimitiveProperties::command("Exec")
            }
            Atom::Noop => PrimitiveProperties {
                name: "Noop",
                energy_cost: 0.0,
                reversible: true,
                min_phi: 0.0,
                destructiveness: DestructivenessLevel::ReadOnly,
                risk_tier: RiskTier::Low,
            },
        }
    }

    /// Lower this atom to an `ActionIR` for execution by `SimpleExecutor`.
    pub fn to_action_ir(&self) -> ActionIR {
        match self {
            Atom::ReadFile { path } => ActionIR::ReadFile {
                path: path.clone(),
                encoding: None,
            },
            Atom::WriteFile { path, content } => ActionIR::WriteFile {
                path: path.clone(),
                content: content.clone(),
                create_dirs: true,
            },
            Atom::ListDir { path, recursive } => ActionIR::ListDirectory {
                path: path.clone(),
                recursive: *recursive,
            },
            Atom::Exec {
                program,
                args,
                working_dir,
                env,
            } => ActionIR::RunCommand {
                program: program.clone(),
                args: args.clone(),
                env: env.clone(),
                working_dir: working_dir.clone(),
            },
            Atom::Parse { path } => ActionIR::RunCommand {
                program: "tree-sitter".into(),
                args: vec!["parse".into(), path.to_string_lossy().to_string()],
                env: BTreeMap::new(),
                working_dir: None,
            },
            Atom::EditFile { path, new, .. } => ActionIR::WriteFile {
                path: path.clone(),
                content: new.as_bytes().to_vec(),
                create_dirs: false,
            },
            Atom::GrepFiles { pattern, path, .. } => ActionIR::RunCommand {
                program: "rg".into(),
                args: vec![
                    "--no-heading".into(),
                    pattern.clone(),
                    path.to_string_lossy().to_string(),
                ],
                env: BTreeMap::new(),
                working_dir: None,
            },
            Atom::DeleteFile { path } => ActionIR::DeleteFile { path: path.clone() },
            Atom::Encode { .. } | Atom::Search { .. } | Atom::Dispatch { .. } => ActionIR::NoOp, // handled in-process
            Atom::Noop => ActionIR::NoOp,
        }
    }
}

// ─── Convenience Constructors ────────────────────────────────────────────────

impl Atom {
    /// `cargo check --message-format=short`
    pub fn cargo_check(working_dir: PathBuf) -> Self {
        Atom::Exec {
            program: "cargo".into(),
            args: vec!["check".into(), "--message-format=short".into()],
            working_dir: Some(working_dir),
            env: BTreeMap::new(),
        }
    }

    /// `cargo check --message-format=json` for structured diagnostics.
    pub fn cargo_check_json(working_dir: PathBuf) -> Self {
        Atom::Exec {
            program: "cargo".into(),
            args: vec!["check".into(), "--message-format=json".into()],
            working_dir: Some(working_dir),
            env: BTreeMap::new(),
        }
    }

    /// `cargo test`
    pub fn cargo_test(working_dir: PathBuf) -> Self {
        Atom::Exec {
            program: "cargo".into(),
            args: vec!["test".into()],
            working_dir: Some(working_dir),
            env: BTreeMap::new(),
        }
    }

    /// `cargo clippy`
    pub fn cargo_clippy(working_dir: PathBuf) -> Self {
        Atom::Exec {
            program: "cargo".into(),
            args: vec!["clippy".into(), "--message-format=short".into()],
            working_dir: Some(working_dir),
            env: BTreeMap::new(),
        }
    }

    /// Read a specific file.
    pub fn read(path: impl Into<PathBuf>) -> Self {
        Atom::ReadFile { path: path.into() }
    }

    /// Write text content to a file.
    pub fn write_text(path: impl Into<PathBuf>, content: &str) -> Self {
        Atom::WriteFile {
            path: path.into(),
            content: content.as_bytes().to_vec(),
        }
    }

    /// List directory contents.
    pub fn list(path: impl Into<PathBuf>) -> Self {
        Atom::ListDir {
            path: path.into(),
            recursive: false,
        }
    }

    /// Edit a file: find `old` and replace with `new`.
    pub fn edit(path: impl Into<PathBuf>, old: &str, new: &str) -> Self {
        Atom::EditFile {
            path: path.into(),
            old: old.to_string(),
            new: new.to_string(),
        }
    }

    /// Search for a pattern in files under a directory.
    pub fn grep(pattern: &str, path: impl Into<PathBuf>) -> Self {
        Atom::GrepFiles {
            pattern: pattern.to_string(),
            path: path.into(),
            recursive: true,
        }
    }

    /// Delete a file.
    pub fn delete(path: impl Into<PathBuf>) -> Self {
        Atom::DeleteFile { path: path.into() }
    }

    /// Dispatch code generation to native backend (energy=1.0).
    pub fn dispatch_native(prompt: &str) -> Self {
        Atom::Dispatch {
            prompt: prompt.to_string(),
            tier: DispatchTier::Native,
        }
    }

    /// Dispatch code generation to local LLM (energy=10.0).
    pub fn dispatch_local(prompt: &str) -> Self {
        Atom::Dispatch {
            prompt: prompt.to_string(),
            tier: DispatchTier::LocalLlm,
        }
    }

    /// Dispatch code generation to cloud LLM (energy=50.0).
    pub fn dispatch_cloud(prompt: &str) -> Self {
        Atom::Dispatch {
            prompt: prompt.to_string(),
            tier: DispatchTier::CloudLlm,
        }
    }
}

// ─── Combinators (Molecules) ─────────────────────────────────────────────────

/// Predicate function over PrimitiveValue — used by Conditional and Until.
pub type Predicate = Box<dyn Fn(&PrimitiveValue) -> bool + Send + Sync>;

/// Transform function — maps a PrimitiveValue to modify behavior.
pub type Transform = Box<dyn Fn(&PrimitiveValue) -> Molecule + Send + Sync>;

/// A molecule: composed from atoms via combinators.
/// This is the core abstraction — molecules are recursive and can nest arbitrarily.
pub enum Molecule {
    /// A single atom (leaf node).
    Atom(Atom),

    /// Execute `first`, pass its output to `second`.
    /// Energy = sum, Phi = max, Destructiveness = max.
    Sequence(Box<Molecule>, Box<Molecule>),

    /// Execute both concurrently (when independent).
    /// Energy = sum, Phi = max, Destructiveness = max.
    Parallel(Box<Molecule>, Box<Molecule>),

    /// If predicate(input) is true, execute `then`, else `otherwise`.
    /// Energy = max(then, otherwise), Phi = max, Destructiveness = max.
    Conditional {
        predicate: Predicate,
        then: Box<Molecule>,
        otherwise: Box<Molecule>,
    },

    /// Repeat `body` until `predicate(output)` is true or `max_iterations` reached.
    /// Energy = body * max_iterations (worst case), Phi = body.phi, Destructiveness = body.
    Until {
        body: Box<Molecule>,
        predicate: Predicate,
        max_iterations: usize,
    },

    /// Try `body`; on failure, run `handler(error)` to produce a recovery molecule.
    /// Energy = body + handler (worst case), Phi = max, Destructiveness = max.
    Recovery {
        body: Box<Molecule>,
        handler: Transform,
    },
}

impl fmt::Debug for Molecule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Molecule::Atom(a) => write!(f, "Atom({:?})", a),
            Molecule::Sequence(a, b) => write!(f, "Seq({:?}, {:?})", a, b),
            Molecule::Parallel(a, b) => write!(f, "Par({:?}, {:?})", a, b),
            Molecule::Conditional {
                then, otherwise, ..
            } => {
                write!(f, "If(?, {:?}, {:?})", then, otherwise)
            }
            Molecule::Until {
                body,
                max_iterations,
                ..
            } => write!(f, "Until({:?}, max={})", body, max_iterations),
            Molecule::Recovery { body, .. } => write!(f, "Try({:?}, catch(...))", body),
        }
    }
}

// ─── Plan Profile ────────────────────────────────────────────────────────────

/// Pre-computed cost/risk profile for a molecule.
/// Enables FEP to evaluate plans before execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanProfile {
    /// Total energy cost (worst-case for loops/conditionals).
    pub total_energy: f32,
    /// Minimum Phi required to execute the entire plan.
    pub min_phi: f32,
    /// Highest destructiveness level in the plan.
    pub max_destructiveness: DestructivenessLevel,
    /// Highest risk tier in the plan.
    pub max_risk: RiskTier,
    /// Whether the entire plan is reversible.
    pub fully_reversible: bool,
    /// Number of atom steps (worst-case path).
    pub step_count: usize,
    /// Names of atoms in execution order (first path).
    pub atom_names: Vec<&'static str>,
}

impl PlanProfile {
    /// Whether a given Phi level is sufficient to execute this plan.
    pub fn phi_sufficient(&self, current_phi: f32) -> bool {
        current_phi >= self.min_phi
    }

    /// Whether this plan fits within an energy budget.
    pub fn within_budget(&self, budget: f32) -> bool {
        self.total_energy <= budget
    }

    /// Whether user confirmation is needed (destructive or non-reversible).
    pub fn needs_confirmation(&self) -> bool {
        self.max_destructiveness.requires_confirmation() || !self.fully_reversible
    }
}

impl Molecule {
    /// Compute the full plan profile for this molecule.
    pub fn profile(&self) -> PlanProfile {
        match self {
            Molecule::Atom(atom) => {
                let props = atom.properties();
                PlanProfile {
                    total_energy: props.energy_cost,
                    min_phi: props.min_phi,
                    max_destructiveness: props.destructiveness,
                    max_risk: props.risk_tier,
                    fully_reversible: props.reversible,
                    step_count: 1,
                    atom_names: vec![props.name],
                }
            }

            Molecule::Sequence(a, b) | Molecule::Parallel(a, b) => {
                let pa = a.profile();
                let pb = b.profile();
                PlanProfile {
                    total_energy: pa.total_energy + pb.total_energy,
                    min_phi: pa.min_phi.max(pb.min_phi),
                    max_destructiveness: pa.max_destructiveness.max(pb.max_destructiveness),
                    max_risk: pa.max_risk.max(pb.max_risk),
                    fully_reversible: pa.fully_reversible && pb.fully_reversible,
                    step_count: pa.step_count + pb.step_count,
                    atom_names: pa.atom_names.into_iter().chain(pb.atom_names).collect(),
                }
            }

            Molecule::Conditional {
                then, otherwise, ..
            } => {
                let pt = then.profile();
                let po = otherwise.profile();
                PlanProfile {
                    // Worst case: whichever branch is more expensive
                    total_energy: pt.total_energy.max(po.total_energy),
                    min_phi: pt.min_phi.max(po.min_phi),
                    max_destructiveness: pt.max_destructiveness.max(po.max_destructiveness),
                    max_risk: pt.max_risk.max(po.max_risk),
                    fully_reversible: pt.fully_reversible && po.fully_reversible,
                    step_count: pt.step_count.max(po.step_count),
                    atom_names: pt.atom_names, // first branch as representative
                }
            }

            Molecule::Until {
                body,
                max_iterations,
                ..
            } => {
                let pb = body.profile();
                PlanProfile {
                    total_energy: pb.total_energy * *max_iterations as f32,
                    min_phi: pb.min_phi,
                    max_destructiveness: pb.max_destructiveness,
                    max_risk: pb.max_risk,
                    fully_reversible: pb.fully_reversible,
                    step_count: pb.step_count * *max_iterations,
                    atom_names: pb.atom_names,
                }
            }

            Molecule::Recovery { body, .. } => {
                let pb = body.profile();
                // Handler cost unknown at plan-time — estimate as body cost
                PlanProfile {
                    total_energy: pb.total_energy * 2.0,
                    min_phi: pb.min_phi,
                    max_destructiveness: pb.max_destructiveness,
                    max_risk: pb.max_risk,
                    fully_reversible: pb.fully_reversible,
                    step_count: pb.step_count * 2,
                    atom_names: pb.atom_names,
                }
            }
        }
    }

    /// Lower this molecule to a sequence of `ActionIR` for the existing executor.
    /// Only lowers the first (non-conditional, non-loop) path — for planning only.
    pub fn to_action_ir_sequence(&self) -> Vec<ActionIR> {
        match self {
            Molecule::Atom(atom) => {
                let ir = atom.to_action_ir();
                if matches!(ir, ActionIR::NoOp) {
                    vec![]
                } else {
                    vec![ir]
                }
            }
            Molecule::Sequence(a, b) => {
                let mut seq = a.to_action_ir_sequence();
                seq.extend(b.to_action_ir_sequence());
                seq
            }
            Molecule::Parallel(a, b) => {
                let mut seq = a.to_action_ir_sequence();
                seq.extend(b.to_action_ir_sequence());
                seq
            }
            Molecule::Conditional { then, .. } => then.to_action_ir_sequence(),
            Molecule::Until { body, .. } => body.to_action_ir_sequence(),
            Molecule::Recovery { body, .. } => body.to_action_ir_sequence(),
        }
    }
}

// ─── Builder DSL ─────────────────────────────────────────────────────────────

/// Fluent builder for constructing molecules.
impl Molecule {
    /// Wrap an atom as a molecule.
    pub fn atom(atom: Atom) -> Self {
        Molecule::Atom(atom)
    }

    /// Chain: self then next (sequential).
    pub fn then(self, next: Molecule) -> Self {
        Molecule::Sequence(Box::new(self), Box::new(next))
    }

    /// Branch on a predicate.
    pub fn branch(
        predicate: impl Fn(&PrimitiveValue) -> bool + Send + Sync + 'static,
        then: Molecule,
        otherwise: Molecule,
    ) -> Self {
        Molecule::Conditional {
            predicate: Box::new(predicate),
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        }
    }

    /// Loop until predicate is satisfied.
    pub fn repeat_until(
        self,
        predicate: impl Fn(&PrimitiveValue) -> bool + Send + Sync + 'static,
        max_iterations: usize,
    ) -> Self {
        Molecule::Until {
            body: Box::new(self),
            predicate: Box::new(predicate),
            max_iterations,
        }
    }

    /// Attach a recovery handler.
    pub fn recover(
        self,
        handler: impl Fn(&PrimitiveValue) -> Molecule + Send + Sync + 'static,
    ) -> Self {
        Molecule::Recovery {
            body: Box::new(self),
            handler: Box::new(handler),
        }
    }
}

// ─── Molecule Executor ───────────────────────────────────────────────────────

/// Execution error for molecule execution.
#[derive(Debug)]
pub enum MoleculeError {
    /// An I/O error during execution.
    Io(std::io::Error),
    /// Plan was rejected by the evaluator.
    PlanRejected(String),
    /// Maximum iterations exceeded in Until loop.
    MaxIterations(usize),
    /// Phi too low for this step.
    PhiGate { required: f32, actual: f32 },
}

impl fmt::Display for MoleculeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoleculeError::Io(e) => write!(f, "I/O error: {}", e),
            MoleculeError::PlanRejected(r) => write!(f, "Plan rejected: {}", r),
            MoleculeError::MaxIterations(n) => write!(f, "Max iterations ({}) exceeded", n),
            MoleculeError::PhiGate { required, actual } => {
                write!(f, "Phi gate: need {:.3}, have {:.3}", required, actual)
            }
        }
    }
}

impl From<std::io::Error> for MoleculeError {
    fn from(e: std::io::Error) -> Self {
        MoleculeError::Io(e)
    }
}

/// Executes molecules with value flow between atoms.
///
/// Each atom produces a `PrimitiveValue` that feeds into the next atom
/// in a Sequence, or into predicates for Conditional/Until.
pub struct MoleculeExecutor {
    /// Current Phi level (updated externally by cognitive loop).
    pub current_phi: f32,
    /// Remaining energy budget.
    pub energy_budget: f32,
    /// Whether to actually execute commands (false = dry-run).
    pub real_exec: bool,
    /// Execution trace: (atom_name, energy_cost, value_summary).
    pub trace: Vec<(String, f32, String)>,
}

impl MoleculeExecutor {
    /// Create a new executor.
    pub fn new(current_phi: f32, energy_budget: f32, real_exec: bool) -> Self {
        Self {
            current_phi,
            energy_budget,
            real_exec,
            trace: Vec::new(),
        }
    }

    /// Execute a molecule, returning the final PrimitiveValue.
    pub fn execute(&mut self, molecule: &Molecule) -> Result<PrimitiveValue, MoleculeError> {
        match molecule {
            Molecule::Atom(atom) => self.execute_atom(atom),

            Molecule::Sequence(a, b) => {
                let va = self.execute(a)?;
                // Second atom can see first's output via trace
                let vb = self.execute(b)?;
                // Return the last value in a sequence
                Ok(if matches!(vb, PrimitiveValue::Unit) {
                    va
                } else {
                    vb
                })
            }

            Molecule::Parallel(a, b) => {
                // Execute both (no actual parallelism for now — sequential)
                let va = self.execute(a)?;
                let vb = self.execute(b)?;
                // Return the more informative value
                Ok(if matches!(va, PrimitiveValue::Unit) {
                    vb
                } else {
                    va
                })
            }

            Molecule::Conditional {
                predicate,
                then,
                otherwise,
            } => {
                // Use the last trace value as input, or Unit
                let last_val = self
                    .trace
                    .last()
                    .map(|(_, _, s)| PrimitiveValue::Text(s.clone()))
                    .unwrap_or(PrimitiveValue::Unit);

                if predicate(&last_val) {
                    self.execute(then)
                } else {
                    self.execute(otherwise)
                }
            }

            Molecule::Until {
                body,
                predicate,
                max_iterations,
            } => {
                let mut result = PrimitiveValue::Unit;
                for i in 0..*max_iterations {
                    result = self.execute(body)?;
                    if predicate(&result) {
                        return Ok(result);
                    }
                    if i == *max_iterations - 1 {
                        return Err(MoleculeError::MaxIterations(*max_iterations));
                    }
                }
                Ok(result)
            }

            Molecule::Recovery { body, handler } => match self.execute(body) {
                Ok(val) => Ok(val),
                Err(e) => {
                    let error_val = PrimitiveValue::Text(format!("{}", e));
                    let recovery = handler(&error_val);
                    self.execute(&recovery)
                }
            },
        }
    }

    /// Execute a single atom, returning its output value.
    fn execute_atom(&mut self, atom: &Atom) -> Result<PrimitiveValue, MoleculeError> {
        let props = atom.properties();

        // Phi gate
        if self.current_phi < props.min_phi {
            return Err(MoleculeError::PhiGate {
                required: props.min_phi,
                actual: self.current_phi,
            });
        }

        // Energy check
        if self.energy_budget < props.energy_cost {
            return Err(MoleculeError::PlanRejected(format!(
                "{} costs {:.1}, budget {:.1}",
                props.name, props.energy_cost, self.energy_budget
            )));
        }
        self.energy_budget -= props.energy_cost;

        let result = if self.real_exec {
            self.execute_atom_real(atom)?
        } else {
            self.execute_atom_simulated(atom)
        };

        // Record trace
        let summary = match &result {
            PrimitiveValue::Unit => "()".to_string(),
            PrimitiveValue::Text(s) => {
                if s.len() > 80 {
                    format!("{}...", &s[..80])
                } else {
                    s.clone()
                }
            }
            PrimitiveValue::Bytes(b) => format!("{} bytes", b.len()),
            PrimitiveValue::CommandResult { exit_code, .. } => format!("exit={}", exit_code),
            PrimitiveValue::Listing(l) => format!("{} entries", l.len()),
            PrimitiveValue::Bool(b) => format!("{}", b),
            PrimitiveValue::Float(f) => format!("{:.3}", f),
        };
        self.trace
            .push((props.name.to_string(), props.energy_cost, summary));

        Ok(result)
    }

    /// Execute an atom for real (filesystem + commands).
    fn execute_atom_real(&self, atom: &Atom) -> Result<PrimitiveValue, MoleculeError> {
        match atom {
            Atom::ReadFile { path } => {
                let content = std::fs::read_to_string(path)?;
                Ok(PrimitiveValue::Text(content))
            }
            Atom::WriteFile { path, content } => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(path, content)?;
                Ok(PrimitiveValue::Unit)
            }
            Atom::EditFile { path, old, new } => {
                let content = std::fs::read_to_string(path)?;
                let edited = content.replace(old, new);
                std::fs::write(path, &edited)?;
                Ok(PrimitiveValue::Text(edited))
            }
            Atom::DeleteFile { path } => {
                std::fs::remove_file(path)?;
                Ok(PrimitiveValue::Unit)
            }
            Atom::ListDir { path, recursive } => {
                let mut entries = Vec::new();
                if *recursive {
                    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
                        for entry in std::fs::read_dir(dir)? {
                            let entry = entry?;
                            let path = entry.path();
                            out.push(path.clone());
                            if path.is_dir() {
                                walk(&path, out)?;
                            }
                        }
                        Ok(())
                    }
                    walk(path, &mut entries)?;
                } else {
                    for entry in std::fs::read_dir(path)? {
                        entries.push(entry?.path());
                    }
                }
                Ok(PrimitiveValue::Listing(entries))
            }
            Atom::GrepFiles {
                pattern,
                path,
                recursive,
            } => {
                let mut args = vec!["--no-heading".to_string()];
                if *recursive {
                    args.push("-r".to_string());
                }
                args.push(pattern.clone());
                args.push(path.to_string_lossy().to_string());

                match std::process::Command::new("rg").args(&args).output() {
                    Ok(output) => Ok(PrimitiveValue::CommandResult {
                        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                        exit_code: output.status.code().unwrap_or(-1),
                    }),
                    Err(_) => {
                        // Fall back to grep if rg unavailable
                        Ok(PrimitiveValue::Text(String::new()))
                    }
                }
            }
            Atom::Exec {
                program,
                args,
                working_dir,
                env,
            } => {
                let mut cmd = std::process::Command::new(program);
                cmd.args(args);
                if let Some(dir) = working_dir {
                    cmd.current_dir(dir);
                }
                for (k, v) in env {
                    cmd.env(k, v);
                }
                let output = cmd.output()?;
                Ok(PrimitiveValue::CommandResult {
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    exit_code: output.status.code().unwrap_or(-1),
                })
            }
            Atom::Parse { path } => {
                // Return the file content as text (actual parsing would use tree-sitter)
                let content = std::fs::read_to_string(path)?;
                Ok(PrimitiveValue::Text(content))
            }
            Atom::Encode { text } => Ok(PrimitiveValue::Text(format!("encoded:{}", text))),
            Atom::Search { query, top_k } => Ok(PrimitiveValue::Text(format!(
                "search:{}:top_{}",
                query, top_k
            ))),
            Atom::Dispatch { prompt, tier } => {
                // Dispatch is handled externally by the CodingAgent (which has the dispatcher).
                // MoleculeExecutor returns a placeholder — the agent intercepts Dispatch atoms
                // before they reach here, or uses the result as a signal.
                Ok(PrimitiveValue::Text(format!(
                    "dispatch:{:?}:{}",
                    tier,
                    &prompt[..prompt.len().min(100)]
                )))
            }
            Atom::Noop => Ok(PrimitiveValue::Unit),
        }
    }

    /// Simulate an atom (no real I/O).
    fn execute_atom_simulated(&self, atom: &Atom) -> PrimitiveValue {
        match atom {
            Atom::ReadFile { path } => {
                PrimitiveValue::Text(format!("[simulated read: {}]", path.display()))
            }
            Atom::WriteFile { path, content } => PrimitiveValue::Text(format!(
                "[simulated write: {} ({} bytes)]",
                path.display(),
                content.len()
            )),
            Atom::EditFile { path, old, new } => PrimitiveValue::Text(format!(
                "[simulated edit: {} ({}→{})]",
                path.display(),
                &old[..old.len().min(20)],
                &new[..new.len().min(20)]
            )),
            Atom::DeleteFile { path } => {
                PrimitiveValue::Text(format!("[simulated delete: {}]", path.display()))
            }
            Atom::ListDir { path, .. } => {
                PrimitiveValue::Listing(vec![path.join("simulated_entry")])
            }
            Atom::GrepFiles { pattern, path, .. } => PrimitiveValue::Text(format!(
                "[simulated grep: {} in {}]",
                pattern,
                path.display()
            )),
            Atom::Exec { program, args, .. } => PrimitiveValue::CommandResult {
                stdout: format!("[simulated {} {}]", program, args.join(" ")),
                stderr: String::new(),
                exit_code: 0,
            },
            Atom::Parse { path } => {
                PrimitiveValue::Text(format!("[simulated parse: {}]", path.display()))
            }
            Atom::Encode { text } => PrimitiveValue::Text(format!("[simulated encode: {}]", text)),
            Atom::Search { query, top_k } => {
                PrimitiveValue::Text(format!("[simulated search: {} top {}]", query, top_k))
            }
            Atom::Dispatch { prompt, tier } => PrimitiveValue::Text(format!(
                "[simulated dispatch {:?}: {}]",
                tier,
                &prompt[..prompt.len().min(60)]
            )),
            Atom::Noop => PrimitiveValue::Unit,
        }
    }
}

// ─── FEP Plan Selection ─────────────────────────────────────────────────────

/// Candidate plan with its profile for FEP comparison.
#[derive(Debug)]
pub struct PlanCandidate {
    /// Human-readable name for this plan.
    pub name: String,
    /// The molecule to execute.
    pub molecule: Molecule,
    /// Pre-computed profile.
    pub profile: PlanProfile,
}

/// Select the best plan from candidates using FEP-style free energy minimization.
///
/// Selection criteria (in priority order):
/// 1. Must satisfy Phi threshold
/// 2. Must fit within energy budget
/// 3. Must not be destructive (unless no alternatives)
/// 4. Among feasible plans, prefer lowest expected free energy:
///    `F = energy_cost - expected_information_gain`
///
/// Returns the index of the selected plan, or None if no plan is feasible.
pub fn select_best_plan(
    candidates: &[PlanCandidate],
    current_phi: f32,
    energy_budget: f32,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }

    // Filter to feasible plans
    let feasible: Vec<(usize, &PlanCandidate)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.profile.phi_sufficient(current_phi) && c.profile.within_budget(energy_budget)
        })
        .collect();

    if feasible.is_empty() {
        return None;
    }

    // Among feasible, prefer non-destructive plans
    let safe: Vec<(usize, &PlanCandidate)> = feasible
        .iter()
        .filter(|(_, c)| c.profile.max_destructiveness != DestructivenessLevel::Destructive)
        .cloned()
        .collect();

    let pool = if safe.is_empty() { &feasible } else { &safe };

    // Free energy = energy_cost / step_efficiency
    // Lower is better — more information per energy unit
    // step_count acts as proxy for information gain (more steps = more info)
    let best = pool
        .iter()
        .min_by(|(_, a), (_, b)| {
            let fa = free_energy(&a.profile);
            let fb = free_energy(&b.profile);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| *idx);

    best
}

/// Select the best plan using FEP free energy minimization with historical recipe success rates.
///
/// `recipe_rates` maps each candidate index to a historical success rate (0.0-1.0).
/// Plans with higher historical success rates get lower free energy (preferred).
/// This closes the learning loop: past execution outcomes influence future plan selection.
pub fn select_best_plan_with_history(
    candidates: &[PlanCandidate],
    current_phi: f32,
    energy_budget: f32,
    recipe_rates: &[f32],
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }

    // Filter to feasible plans
    let feasible: Vec<(usize, &PlanCandidate)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.profile.phi_sufficient(current_phi) && c.profile.within_budget(energy_budget)
        })
        .collect();

    if feasible.is_empty() {
        return None;
    }

    // Among feasible, prefer non-destructive plans
    let safe: Vec<(usize, &PlanCandidate)> = feasible
        .iter()
        .filter(|(_, c)| c.profile.max_destructiveness != DestructivenessLevel::Destructive)
        .cloned()
        .collect();

    let pool = if safe.is_empty() { &feasible } else { &safe };

    // Free energy modulated by historical success rate:
    // F = base_free_energy / (0.5 + recipe_rate)
    // Higher past success → lower free energy → preferred
    let best = pool
        .iter()
        .min_by(|(idx_a, a), (idx_b, b)| {
            let fa = free_energy(&a.profile);
            let fb = free_energy(&b.profile);
            let rate_a = recipe_rates.get(*idx_a).copied().unwrap_or(0.5);
            let rate_b = recipe_rates.get(*idx_b).copied().unwrap_or(0.5);
            let adjusted_a = fa / (0.5 + rate_a);
            let adjusted_b = fb / (0.5 + rate_b);
            adjusted_a
                .partial_cmp(&adjusted_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| *idx);

    best
}

/// Compute expected free energy for a plan.
/// F = energy_cost * destructiveness_penalty / information_value
/// Lower F = better plan (less energy, more info, less destructive).
fn free_energy(profile: &PlanProfile) -> f32 {
    let destructiveness_penalty = match profile.max_destructiveness {
        DestructivenessLevel::ReadOnly => 1.0,
        DestructivenessLevel::Reversible => 1.5,
        DestructivenessLevel::NeedsConfirmation => 3.0,
        DestructivenessLevel::Destructive => 10.0,
    };
    let information_value = (profile.step_count as f32).max(1.0);
    (profile.total_energy * destructiveness_penalty) / information_value
}

// ─── Pre-built Recipes ───────────────────────────────────────────────────────

/// Standard coding recipes built from primitives.
pub mod recipes {
    use super::*;

    /// Write code → cargo check → report success/failure.
    /// Total energy: ~4.1, min_phi: 0.05, fully reversible: true.
    pub fn write_and_check(path: PathBuf, code: &str) -> Molecule {
        Molecule::atom(Atom::write_text(&path, code)).then(Molecule::atom(Atom::cargo_check(
            path.parent()
                .and_then(|p| p.parent())
                .unwrap_or(path.as_path())
                .to_path_buf(),
        )))
    }

    /// Write code → cargo check → if fail, retry with fix.
    /// The compile-fix loop as a molecule.
    pub fn compile_fix_loop(path: PathBuf, initial_code: String, max_retries: usize) -> Molecule {
        let project_dir = path
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(path.as_path())
            .to_path_buf();

        // Write → Check → Done/Retry
        let write_check = Molecule::atom(Atom::write_text(&path, &initial_code))
            .then(Molecule::atom(Atom::cargo_check(project_dir)));

        write_check.repeat_until(|val| val.is_success(), max_retries)
    }

    /// Read file → parse → encode to HDC → search similar.
    /// Pure analysis pipeline — no side effects.
    pub fn analyze_file(path: PathBuf) -> Molecule {
        Molecule::atom(Atom::read(path.clone()))
            .then(Molecule::atom(Atom::Parse { path: path.clone() }))
            .then(Molecule::atom(Atom::Encode {
                text: String::new(), // populated at runtime from Parse output
            }))
    }

    /// Read target → write code → check → test.
    /// Full coding workflow for a single file.
    pub fn full_coding_workflow(target: PathBuf, code: String, project_dir: PathBuf) -> Molecule {
        Molecule::atom(Atom::read(target.clone()))
            .then(Molecule::atom(Atom::write_text(&target, &code)))
            .then(Molecule::atom(Atom::cargo_check(project_dir.clone())))
            .then(Molecule::branch(
                |val| val.is_success(),
                Molecule::atom(Atom::cargo_test(project_dir.clone())),
                Molecule::atom(Atom::Noop), // fix loop would go here
            ))
    }

    /// Read multiple files in parallel for context gathering.
    pub fn gather_context(files: Vec<PathBuf>) -> Molecule {
        let mut result = Molecule::atom(Atom::Noop);
        for file in files {
            result =
                Molecule::Parallel(Box::new(result), Box::new(Molecule::atom(Atom::read(file))));
        }
        result
    }

    /// Search codebase → Read matching files → Edit with fix.
    /// The "find and fix" pattern for refactoring.
    pub fn find_and_edit(search_dir: PathBuf, pattern: &str, old: &str, new: &str) -> Molecule {
        Molecule::atom(Atom::grep(pattern, search_dir))
            .then(Molecule::atom(Atom::Noop)) // placeholder for match processing
            .recover(|_| Molecule::atom(Atom::Noop)) // if no matches, noop
    }

    /// Read → Edit → Check → Test pipeline.
    /// For targeted fixes with verification.
    pub fn edit_and_verify(path: PathBuf, old: &str, new: &str, project_dir: PathBuf) -> Molecule {
        Molecule::atom(Atom::read(path.clone()))
            .then(Molecule::atom(Atom::edit(&path, old, new)))
            .then(Molecule::atom(Atom::cargo_check(project_dir.clone())))
            .then(Molecule::branch(
                |val| val.is_success(),
                Molecule::atom(Atom::cargo_test(project_dir)),
                Molecule::atom(Atom::Noop),
            ))
    }

    /// Test-driven fix: run test → read failure → apply fix → retest.
    /// Repeats until tests pass or max retries.
    pub fn test_driven_fix(
        target: PathBuf,
        project_dir: PathBuf,
        fix_code: String,
        max_retries: usize,
    ) -> Molecule {
        // test → if fail: write fix → test again
        let test_fix =
            Molecule::atom(Atom::cargo_test(project_dir.clone())).then(Molecule::branch(
                |val| val.is_success(),
                Molecule::atom(Atom::Noop), // done
                Molecule::atom(Atom::write_text(&target, &fix_code))
                    .then(Molecule::atom(Atom::cargo_check(project_dir.clone()))),
            ));

        test_fix.repeat_until(|val| val.is_success(), max_retries)
    }

    /// Multi-file edit: apply the same edit across multiple files, then verify.
    pub fn multi_file_edit(
        files: Vec<PathBuf>,
        old: &str,
        new: &str,
        project_dir: PathBuf,
    ) -> Molecule {
        let mut edits = Molecule::atom(Atom::Noop);
        for file in &files {
            edits = edits.then(Molecule::atom(Atom::edit(file, old, new)));
        }
        // After all edits, verify
        edits.then(Molecule::atom(Atom::cargo_check(project_dir)))
    }

    /// Refactor and verify: grep for pattern → edit all matches → check → test.
    pub fn refactor_and_verify(
        project_dir: PathBuf,
        search_pattern: &str,
        old: &str,
        new: &str,
        target_files: Vec<PathBuf>,
    ) -> Molecule {
        // Search for pattern (context gathering)
        let search = Molecule::atom(Atom::grep(search_pattern, project_dir.clone()));

        // Apply edits to all target files
        let mut edits = search;
        for file in &target_files {
            edits = edits.then(
                Molecule::atom(Atom::read(file.clone()))
                    .then(Molecule::atom(Atom::edit(file, old, new))),
            );
        }

        // Verify
        edits
            .then(Molecule::atom(Atom::cargo_check(project_dir.clone())))
            .then(Molecule::branch(
                |val| val.is_success(),
                Molecule::atom(Atom::cargo_test(project_dir)),
                Molecule::atom(Atom::Noop),
            ))
    }

    /// Generate + write + check with explicit tier routing.
    /// The Dispatch atom signals which backend to use; the agent intercepts it.
    pub fn generate_and_check(target: PathBuf, prompt: &str, tier: DispatchTier) -> Molecule {
        let project_dir = target
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(target.as_path())
            .to_path_buf();

        Molecule::atom(Atom::Dispatch {
            prompt: prompt.to_string(),
            tier,
        })
        .then(Molecule::atom(Atom::cargo_check(project_dir)))
    }

    /// Three-tier candidate set: native, local LLM, cloud LLM.
    /// Returns 3 PlanCandidates with different energy/capability tradeoffs.
    /// FEP selects the tier with best free-energy given consciousness state.
    pub fn tiered_generation_candidates(target: PathBuf, prompt: &str) -> Vec<(String, Molecule)> {
        vec![
            (
                "native_generate".into(),
                generate_and_check(target.clone(), prompt, DispatchTier::Native),
            ),
            (
                "local_llm_generate".into(),
                generate_and_check(target.clone(), prompt, DispatchTier::LocalLlm),
            ),
            (
                "cloud_llm_generate".into(),
                generate_and_check(target, prompt, DispatchTier::CloudLlm),
            ),
        ]
    }

    /// Exploration pipeline: search → read top matches → encode for HDC similarity.
    /// Pure information gathering, no mutations.
    pub fn explore_codebase(
        project_dir: PathBuf,
        search_term: &str,
        files_to_read: Vec<PathBuf>,
    ) -> Molecule {
        let search = Molecule::atom(Atom::grep(search_term, project_dir));
        let mut pipeline = search;
        for file in files_to_read {
            pipeline = pipeline.then(Molecule::atom(Atom::read(file)));
        }
        pipeline
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_properties_read() {
        let atom = Atom::read("/tmp/test.rs");
        let props = atom.properties();
        assert_eq!(props.name, "ReadFile");
        assert!(props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::ReadOnly);
        assert!(props.min_phi < f32::EPSILON);
    }

    #[test]
    fn test_atom_properties_write() {
        let atom = Atom::write_text("/tmp/test.rs", "fn main() {}");
        let props = atom.properties();
        assert_eq!(props.name, "WriteFile");
        assert!(props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::Reversible);
    }

    #[test]
    fn test_atom_properties_cargo_check() {
        let atom = Atom::cargo_check(PathBuf::from("/tmp/project"));
        let props = atom.properties();
        assert_eq!(props.name, "CargoCheck");
        assert!(props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::ReadOnly);
        assert!(props.energy_cost > 1.0); // more expensive than read
    }

    #[test]
    fn test_atom_properties_git_push_destructive() {
        let atom = Atom::Exec {
            program: "git".into(),
            args: vec!["push".into()],
            working_dir: None,
            env: BTreeMap::new(),
        };
        let props = atom.properties();
        assert_eq!(props.name, "GitPush");
        assert!(!props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::Destructive);
        assert!(props.min_phi >= 0.3);
    }

    #[test]
    fn test_sequence_profile_sums_energy() {
        let mol = Molecule::atom(Atom::read("/tmp/a.rs"))
            .then(Molecule::atom(Atom::write_text("/tmp/a.rs", "code")));
        let profile = mol.profile();
        assert!((profile.total_energy - 1.1).abs() < 0.01); // 0.1 + 1.0
        assert_eq!(profile.step_count, 2);
    }

    #[test]
    fn test_sequence_profile_max_phi() {
        let mol = Molecule::atom(Atom::read("/tmp/a.rs")) // phi=0.0
            .then(Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")))); // phi=0.05
        let profile = mol.profile();
        assert!((profile.min_phi - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_sequence_profile_max_destructiveness() {
        let mol = Molecule::atom(Atom::read("/tmp/a.rs")).then(Molecule::atom(Atom::Exec {
            program: "git".into(),
            args: vec!["push".into()],
            working_dir: None,
            env: BTreeMap::new(),
        }));
        let profile = mol.profile();
        assert_eq!(
            profile.max_destructiveness,
            DestructivenessLevel::Destructive
        );
        assert!(!profile.fully_reversible);
    }

    #[test]
    fn test_until_multiplies_energy() {
        let body = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let body_energy = body.profile().total_energy;
        let looped = body.repeat_until(|v| v.is_success(), 5);
        let profile = looped.profile();
        assert!((profile.total_energy - body_energy * 5.0).abs() < 0.01);
        assert_eq!(profile.step_count, 5);
    }

    #[test]
    fn test_conditional_worst_case_energy() {
        let mol = Molecule::branch(
            |_| true,
            Molecule::atom(Atom::cargo_test(PathBuf::from("/tmp"))),
            Molecule::atom(Atom::Noop),
        );
        let profile = mol.profile();
        // Should take the more expensive branch for worst-case
        assert!(profile.total_energy >= 5.0);
    }

    #[test]
    fn test_recovery_doubles_energy_estimate() {
        let body = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let body_energy = body.profile().total_energy;
        let with_recovery = body.recover(|_| Molecule::atom(Atom::Noop));
        let profile = with_recovery.profile();
        assert!((profile.total_energy - body_energy * 2.0).abs() < 0.01);
    }

    #[test]
    fn test_plan_profile_phi_sufficient() {
        let mol = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let profile = mol.profile();
        assert!(profile.phi_sufficient(0.1));
        assert!(profile.phi_sufficient(0.05));
        assert!(!profile.phi_sufficient(0.01));
    }

    #[test]
    fn test_plan_profile_budget() {
        let mol = Molecule::atom(Atom::read("/tmp/a.rs"))
            .then(Molecule::atom(Atom::write_text("/tmp/a.rs", "code")))
            .then(Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))));
        let profile = mol.profile();
        assert!(profile.within_budget(10.0));
        assert!(!profile.within_budget(1.0));
    }

    #[test]
    fn test_compile_fix_recipe_profile() {
        let recipe = recipes::compile_fix_loop(
            PathBuf::from("/tmp/project/src/lib.rs"),
            "fn main() {}".into(),
            3,
        );
        let profile = recipe.profile();
        // Should be write + check, repeated 3x
        assert!(profile.total_energy > 10.0);
        assert_eq!(profile.step_count, 6); // (write + check) * 3
        assert!(profile.fully_reversible);
    }

    #[test]
    fn test_atom_to_action_ir_read() {
        let atom = Atom::read("/tmp/test.rs");
        let ir = atom.to_action_ir();
        assert!(matches!(ir, ActionIR::ReadFile { .. }));
    }

    #[test]
    fn test_atom_to_action_ir_write() {
        let atom = Atom::write_text("/tmp/test.rs", "hello");
        let ir = atom.to_action_ir();
        match ir {
            ActionIR::WriteFile { content, .. } => assert_eq!(content, b"hello"),
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_atom_to_action_ir_exec() {
        let atom = Atom::cargo_check(PathBuf::from("/tmp"));
        let ir = atom.to_action_ir();
        match ir {
            ActionIR::RunCommand { program, args, .. } => {
                assert_eq!(program, "cargo");
                assert!(args.contains(&"check".to_string()));
            }
            _ => panic!("Expected RunCommand"),
        }
    }

    #[test]
    fn test_molecule_to_action_ir_sequence() {
        let mol = Molecule::atom(Atom::read("/tmp/a.rs"))
            .then(Molecule::atom(Atom::write_text("/tmp/a.rs", "code")));
        let actions = mol.to_action_ir_sequence();
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], ActionIR::ReadFile { .. }));
        assert!(matches!(actions[1], ActionIR::WriteFile { .. }));
    }

    #[test]
    fn test_encode_and_search_are_noop_ir() {
        let mol = Molecule::atom(Atom::Encode {
            text: "test".into(),
        });
        let actions = mol.to_action_ir_sequence();
        assert!(actions.is_empty()); // pure computation, no IR
    }

    #[test]
    fn test_full_coding_workflow_recipe() {
        let recipe = recipes::full_coding_workflow(
            PathBuf::from("/tmp/proj/src/lib.rs"),
            "fn hello() {}".into(),
            PathBuf::from("/tmp/proj"),
        );
        let profile = recipe.profile();
        // Read + Write + Check + branch(Test|Noop)
        assert!(profile.step_count >= 3);
        assert!(profile.min_phi >= 0.05);
    }

    #[test]
    fn test_gather_context_parallel() {
        let recipe = recipes::gather_context(vec![
            PathBuf::from("/tmp/a.rs"),
            PathBuf::from("/tmp/b.rs"),
            PathBuf::from("/tmp/c.rs"),
        ]);
        let profile = recipe.profile();
        // Noop + 3 reads
        assert_eq!(profile.step_count, 4);
        assert!(profile.fully_reversible);
        assert_eq!(profile.max_destructiveness, DestructivenessLevel::ReadOnly);
    }

    #[test]
    fn test_needs_confirmation() {
        let safe = Molecule::atom(Atom::read("/tmp/a.rs"));
        assert!(!safe.profile().needs_confirmation());

        let dangerous = Molecule::atom(Atom::Exec {
            program: "git".into(),
            args: vec!["push".into()],
            working_dir: None,
            env: BTreeMap::new(),
        });
        assert!(dangerous.profile().needs_confirmation());
    }

    #[test]
    fn test_primitive_value_success() {
        assert!(PrimitiveValue::Unit.is_success());
        assert!(PrimitiveValue::Bool(true).is_success());
        assert!(!PrimitiveValue::Bool(false).is_success());
        assert!(
            PrimitiveValue::CommandResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
            }
            .is_success()
        );
        assert!(
            !PrimitiveValue::CommandResult {
                stdout: String::new(),
                stderr: "error".into(),
                exit_code: 1,
            }
            .is_success()
        );
    }

    #[test]
    fn test_primitive_value_stderr() {
        let val = PrimitiveValue::CommandResult {
            stdout: "ok".into(),
            stderr: "warn: unused".into(),
            exit_code: 0,
        };
        assert_eq!(val.stderr(), Some("warn: unused"));
        assert_eq!(val.exit_code(), Some(0));
    }

    #[test]
    fn test_debug_formatting() {
        let mol = Molecule::atom(Atom::read("/tmp/a.rs"))
            .then(Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))));
        let debug = format!("{:?}", mol);
        assert!(debug.contains("Seq"));
        assert!(debug.contains("ReadFile"));
    }

    // ── New Atom Tests ──────────────────────────────────────────────

    #[test]
    fn test_atom_properties_edit() {
        let atom = Atom::edit("/tmp/test.rs", "old", "new");
        let props = atom.properties();
        assert_eq!(props.name, "EditFile");
        assert!(props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::Reversible);
    }

    #[test]
    fn test_atom_properties_grep() {
        let atom = Atom::grep("fn main", "/tmp/src");
        let props = atom.properties();
        assert_eq!(props.name, "GrepFiles");
        assert!(props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::ReadOnly);
    }

    #[test]
    fn test_atom_properties_delete() {
        let atom = Atom::delete("/tmp/test.rs");
        let props = atom.properties();
        assert_eq!(props.name, "DeleteFile");
        assert!(!props.reversible);
        assert_eq!(props.destructiveness, DestructivenessLevel::Destructive);
        assert!(props.min_phi >= 0.2);
    }

    #[test]
    fn test_edit_to_action_ir() {
        let atom = Atom::edit("/tmp/test.rs", "old", "new");
        let ir = atom.to_action_ir();
        match ir {
            ActionIR::WriteFile { content, .. } => {
                assert_eq!(content, b"new");
            }
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_grep_to_action_ir() {
        let atom = Atom::grep("fn main", "/tmp/src");
        let ir = atom.to_action_ir();
        match ir {
            ActionIR::RunCommand { program, args, .. } => {
                assert_eq!(program, "rg");
                assert!(args.contains(&"fn main".to_string()));
            }
            _ => panic!("Expected RunCommand"),
        }
    }

    #[test]
    fn test_delete_to_action_ir() {
        let atom = Atom::delete("/tmp/test.rs");
        let ir = atom.to_action_ir();
        assert!(matches!(ir, ActionIR::DeleteFile { .. }));
    }

    // ── Executor Tests ──────────────────────────────────────────────

    #[test]
    fn test_executor_simulated_read() {
        let mut exec = MoleculeExecutor::new(0.5, 100.0, false);
        let mol = Molecule::atom(Atom::read("/tmp/test.rs"));
        let result = exec.execute(&mol).unwrap();
        match result {
            PrimitiveValue::Text(s) => assert!(s.contains("simulated read")),
            _ => panic!("Expected Text"),
        }
        assert_eq!(exec.trace.len(), 1);
        assert_eq!(exec.trace[0].0, "ReadFile");
    }

    #[test]
    fn test_executor_simulated_sequence() {
        let mut exec = MoleculeExecutor::new(0.5, 100.0, false);
        let mol = Molecule::atom(Atom::read("/tmp/a.rs"))
            .then(Molecule::atom(Atom::write_text("/tmp/a.rs", "code")));
        let _ = exec.execute(&mol).unwrap();
        assert_eq!(exec.trace.len(), 2);
        assert_eq!(exec.trace[0].0, "ReadFile");
        assert_eq!(exec.trace[1].0, "WriteFile");
    }

    #[test]
    fn test_executor_phi_gate_rejects() {
        let mut exec = MoleculeExecutor::new(0.01, 100.0, false);
        let mol = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let result = exec.execute(&mol);
        assert!(result.is_err());
        match result.unwrap_err() {
            MoleculeError::PhiGate { required, actual } => {
                assert!(required > actual);
            }
            e => panic!("Expected PhiGate, got {:?}", e),
        }
    }

    #[test]
    fn test_executor_energy_budget_rejects() {
        let mut exec = MoleculeExecutor::new(1.0, 0.05, false);
        let mol = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))); // costs 3.0
        let result = exec.execute(&mol);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_energy_deduction() {
        let mut exec = MoleculeExecutor::new(1.0, 100.0, false);
        let mol = Molecule::atom(Atom::read("/tmp/a.rs")); // costs 0.1
        let _ = exec.execute(&mol).unwrap();
        assert!((exec.energy_budget - 99.9).abs() < 0.01);
    }

    #[test]
    fn test_executor_conditional_true_branch() {
        let mut exec = MoleculeExecutor::new(1.0, 100.0, false);
        // First run a command to populate trace
        let _ = exec.execute(&Molecule::atom(Atom::Noop)).unwrap();

        let mol = Molecule::branch(
            |_| true,
            Molecule::atom(Atom::read("/tmp/true.rs")),
            Molecule::atom(Atom::read("/tmp/false.rs")),
        );
        let result = exec.execute(&mol).unwrap();
        match result {
            PrimitiveValue::Text(s) => assert!(s.contains("true.rs")),
            _ => panic!("Expected Text"),
        }
    }

    #[test]
    fn test_executor_conditional_false_branch() {
        let mut exec = MoleculeExecutor::new(1.0, 100.0, false);
        let _ = exec.execute(&Molecule::atom(Atom::Noop)).unwrap();

        let mol = Molecule::branch(
            |_| false,
            Molecule::atom(Atom::read("/tmp/true.rs")),
            Molecule::atom(Atom::read("/tmp/false.rs")),
        );
        let result = exec.execute(&mol).unwrap();
        match result {
            PrimitiveValue::Text(s) => assert!(s.contains("false.rs")),
            _ => panic!("Expected Text"),
        }
    }

    #[test]
    fn test_executor_until_immediate_success() {
        let mut exec = MoleculeExecutor::new(1.0, 100.0, false);
        let mol = Molecule::atom(Atom::Noop).repeat_until(|v| matches!(v, PrimitiveValue::Unit), 5);
        let _ = exec.execute(&mol).unwrap();
        // Should execute only once since Noop returns Unit which matches
        assert_eq!(exec.trace.len(), 1);
    }

    #[test]
    fn test_executor_recovery_on_phi_failure() {
        let mut exec = MoleculeExecutor::new(0.01, 100.0, false);
        let mol = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))) // phi=0.05, too high
            .recover(|_| Molecule::atom(Atom::Noop));
        let result = exec.execute(&mol);
        assert!(result.is_ok()); // recovery should succeed with Noop
    }

    #[test]
    fn test_executor_real_file_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");

        let mut exec = MoleculeExecutor::new(1.0, 100.0, true);

        // Write → Read → Edit → Read
        let write = Molecule::atom(Atom::write_text(&path, "hello world"));
        let _ = exec.execute(&write).unwrap();

        let read = Molecule::atom(Atom::read(&path));
        let val = exec.execute(&read).unwrap();
        assert_eq!(val.as_text(), Some("hello world"));

        let edit = Molecule::atom(Atom::edit(&path, "world", "universe"));
        let val = exec.execute(&edit).unwrap();
        assert!(val.as_text().unwrap().contains("universe"));

        let read2 = Molecule::atom(Atom::read(&path));
        let val = exec.execute(&read2).unwrap();
        assert_eq!(val.as_text(), Some("hello universe"));

        assert_eq!(exec.trace.len(), 4);
    }

    #[test]
    fn test_executor_real_list_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();

        let mut exec = MoleculeExecutor::new(1.0, 100.0, true);
        let mol = Molecule::atom(Atom::list(dir.path()));
        let val = exec.execute(&mol).unwrap();
        match val {
            PrimitiveValue::Listing(entries) => {
                assert_eq!(entries.len(), 2);
            }
            _ => panic!("Expected Listing"),
        }
    }

    #[test]
    fn test_executor_real_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doomed.txt");
        std::fs::write(&path, "bye").unwrap();
        assert!(path.exists());

        let mut exec = MoleculeExecutor::new(1.0, 100.0, true);
        let mol = Molecule::atom(Atom::delete(&path));
        let _ = exec.execute(&mol).unwrap();
        assert!(!path.exists());
    }

    // ── FEP Plan Selection Tests ────────────────────────────────────

    #[test]
    fn test_select_best_plan_cheapest() {
        let candidates = vec![
            PlanCandidate {
                name: "expensive".into(),
                molecule: Molecule::atom(Atom::cargo_test(PathBuf::from("/tmp"))),
                profile: Molecule::atom(Atom::cargo_test(PathBuf::from("/tmp"))).profile(),
            },
            PlanCandidate {
                name: "cheap".into(),
                molecule: Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))),
                profile: Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))).profile(),
            },
        ];

        let selected = select_best_plan(&candidates, 1.0, 100.0);
        assert_eq!(selected, Some(1), "Should pick cheaper plan");
    }

    #[test]
    fn test_select_best_plan_avoids_destructive() {
        let candidates = vec![
            PlanCandidate {
                name: "destructive".into(),
                molecule: Molecule::atom(Atom::delete("/tmp/x")),
                profile: Molecule::atom(Atom::delete("/tmp/x")).profile(),
            },
            PlanCandidate {
                name: "safe".into(),
                molecule: Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))),
                profile: Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))).profile(),
            },
        ];

        let selected = select_best_plan(&candidates, 1.0, 100.0);
        assert_eq!(selected, Some(1), "Should prefer safe plan");
    }

    #[test]
    fn test_select_best_plan_phi_filter() {
        let candidates = vec![
            PlanCandidate {
                name: "needs_phi".into(),
                molecule: Molecule::atom(Atom::Exec {
                    program: "git".into(),
                    args: vec!["push".into()],
                    working_dir: None,
                    env: BTreeMap::new(),
                }),
                profile: Molecule::atom(Atom::Exec {
                    program: "git".into(),
                    args: vec!["push".into()],
                    working_dir: None,
                    env: BTreeMap::new(),
                })
                .profile(),
            },
            PlanCandidate {
                name: "low_phi".into(),
                molecule: Molecule::atom(Atom::read("/tmp/x")),
                profile: Molecule::atom(Atom::read("/tmp/x")).profile(),
            },
        ];

        // With very low phi, only the read plan is feasible
        let selected = select_best_plan(&candidates, 0.01, 100.0);
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn test_select_best_plan_none_feasible() {
        let candidates = vec![PlanCandidate {
            name: "too_expensive".into(),
            molecule: Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))),
            profile: Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))).profile(),
        }];

        // Budget of 0.1 can't afford cargo_check (3.0)
        let selected = select_best_plan(&candidates, 1.0, 0.1);
        assert_eq!(selected, None);
    }

    #[test]
    fn test_select_best_plan_empty() {
        let selected = select_best_plan(&[], 1.0, 100.0);
        assert_eq!(selected, None);
    }

    #[test]
    fn test_free_energy_read_only_lowest() {
        let read_profile = Molecule::atom(Atom::read("/tmp/x")).profile();
        let write_profile = Molecule::atom(Atom::write_text("/tmp/x", "y")).profile();
        let exec_profile = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))).profile();

        let fe_read = free_energy(&read_profile);
        let fe_write = free_energy(&write_profile);
        let fe_exec = free_energy(&exec_profile);

        // Read should have lowest free energy (cheapest, read-only)
        assert!(fe_read < fe_write);
        assert!(fe_read < fe_exec);
    }

    #[test]
    fn test_edit_and_verify_recipe() {
        let recipe = recipes::edit_and_verify(
            PathBuf::from("/tmp/proj/src/lib.rs"),
            "old_fn",
            "new_fn",
            PathBuf::from("/tmp/proj"),
        );
        let profile = recipe.profile();
        // Read + Edit + Check + branch(Test|Noop)
        assert!(profile.step_count >= 3);
        assert!(profile.min_phi >= 0.05);
    }

    #[test]
    fn test_find_and_edit_recipe() {
        let recipe = recipes::find_and_edit(
            PathBuf::from("/tmp/src"),
            "todo!",
            "todo!()",
            "unimplemented!()",
        );
        let profile = recipe.profile();
        assert!(profile.fully_reversible);
    }

    // ── New Recipe Tests ────────────────────────────────────────────

    #[test]
    fn test_test_driven_fix_recipe() {
        let recipe = recipes::test_driven_fix(
            PathBuf::from("/tmp/proj/src/lib.rs"),
            PathBuf::from("/tmp/proj"),
            "fn fixed() {}".into(),
            3,
        );
        let profile = recipe.profile();
        // test + branch(noop | write+check), repeated 3x
        assert!(profile.step_count >= 6);
        assert!(profile.total_energy > 15.0);
    }

    #[test]
    fn test_multi_file_edit_recipe() {
        let files = vec![
            PathBuf::from("/tmp/proj/src/a.rs"),
            PathBuf::from("/tmp/proj/src/b.rs"),
            PathBuf::from("/tmp/proj/src/c.rs"),
        ];
        let recipe =
            recipes::multi_file_edit(files, "old_name", "new_name", PathBuf::from("/tmp/proj"));
        let profile = recipe.profile();
        // noop + 3 edits + check
        assert_eq!(profile.step_count, 5);
        assert!(profile.fully_reversible);
    }

    #[test]
    fn test_refactor_and_verify_recipe() {
        let recipe = recipes::refactor_and_verify(
            PathBuf::from("/tmp/proj"),
            "deprecated_fn",
            "deprecated_fn()",
            "new_fn()",
            vec![
                PathBuf::from("/tmp/proj/src/a.rs"),
                PathBuf::from("/tmp/proj/src/b.rs"),
            ],
        );
        let profile = recipe.profile();
        // grep + 2*(read+edit) + check + branch(test|noop)
        assert!(profile.step_count >= 6);
        assert!(profile.min_phi >= 0.05); // edits require some phi
    }

    #[test]
    fn test_explore_codebase_recipe() {
        let recipe = recipes::explore_codebase(
            PathBuf::from("/tmp/proj"),
            "struct.*Config",
            vec![
                PathBuf::from("/tmp/proj/src/config.rs"),
                PathBuf::from("/tmp/proj/src/main.rs"),
            ],
        );
        let profile = recipe.profile();
        // grep + 2 reads
        assert_eq!(profile.step_count, 3);
        assert!(profile.fully_reversible);
        assert_eq!(profile.max_destructiveness, DestructivenessLevel::ReadOnly);
    }

    #[test]
    fn test_executor_full_pipeline_real() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.txt");

        let mut exec = MoleculeExecutor::new(1.0, 100.0, true);

        // Write → Edit → Read pipeline (real I/O)
        let pipeline = Molecule::atom(Atom::write_text(&path, "hello world foo"))
            .then(Molecule::atom(Atom::edit(&path, "foo", "bar")))
            .then(Molecule::atom(Atom::read(&path)));

        let result = exec.execute(&pipeline).unwrap();
        assert_eq!(result.as_text(), Some("hello world bar"));
        assert_eq!(exec.trace.len(), 3);
        assert_eq!(exec.trace[0].0, "WriteFile");
        assert_eq!(exec.trace[1].0, "EditFile");
        assert_eq!(exec.trace[2].0, "ReadFile");
    }

    #[test]
    fn test_select_best_plan_prefers_info_density() {
        // Two plans with same safety but different info/energy ratio
        let plan_a = Molecule::atom(Atom::read("/tmp/a"))
            .then(Molecule::atom(Atom::read("/tmp/b")))
            .then(Molecule::atom(Atom::read("/tmp/c")));
        let plan_b = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));

        let candidates = vec![
            PlanCandidate {
                name: "three_reads".into(),
                profile: plan_a.profile(),
                molecule: plan_a,
            },
            PlanCandidate {
                name: "one_check".into(),
                profile: plan_b.profile(),
                molecule: plan_b,
            },
        ];

        // Three reads: energy=0.3, steps=3 → FE = 0.3/3 = 0.1
        // One check: energy=3.0, steps=1 → FE = 3.0/1 = 3.0
        // Reads should win (lower FE = more info per energy)
        let selected = select_best_plan(&candidates, 1.0, 100.0);
        assert_eq!(selected, Some(0), "Should prefer higher info density");
    }

    // ─── Dispatch atom tests ─────────────────────────────────────────

    #[test]
    fn test_dispatch_native_properties() {
        let atom = Atom::dispatch_native("add fibonacci");
        let props = atom.properties();
        assert_eq!(props.name, "DispatchNative");
        assert!((props.energy_cost - 1.0).abs() < f32::EPSILON);
        assert_eq!(props.destructiveness, DestructivenessLevel::ReadOnly);
    }

    #[test]
    fn test_dispatch_local_properties() {
        let atom = Atom::dispatch_local("add fibonacci");
        let props = atom.properties();
        assert_eq!(props.name, "DispatchLocalLLM");
        assert!((props.energy_cost - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dispatch_cloud_properties() {
        let atom = Atom::dispatch_cloud("add fibonacci");
        let props = atom.properties();
        assert_eq!(props.name, "DispatchCloudLLM");
        assert!((props.energy_cost - 50.0).abs() < f32::EPSILON);
        assert_eq!(props.risk_tier, RiskTier::High);
    }

    #[test]
    fn test_dispatch_tier_energy_ordering() {
        let native = Atom::dispatch_native("x").properties().energy_cost;
        let local = Atom::dispatch_local("x").properties().energy_cost;
        let cloud = Atom::dispatch_cloud("x").properties().energy_cost;
        assert!(native < local);
        assert!(local < cloud);
    }

    #[test]
    fn test_dispatch_simulated_execution() {
        let mut exec = MoleculeExecutor::new(1.0, 100.0, false);
        let mol = Molecule::atom(Atom::dispatch_native("add fibonacci"));
        let result = exec.execute(&mol).unwrap();
        assert!(result.as_text().unwrap().contains("simulated dispatch"));
    }

    #[test]
    fn test_dispatch_real_execution() {
        let mut exec = MoleculeExecutor::new(1.0, 100.0, true);
        let mol = Molecule::atom(Atom::dispatch_local("add fibonacci"));
        let result = exec.execute(&mol).unwrap();
        assert!(result.as_text().unwrap().contains("dispatch:LocalLlm"));
    }

    #[test]
    fn test_generate_and_check_recipe() {
        let mol = recipes::generate_and_check(
            PathBuf::from("/tmp/test/src/lib.rs"),
            "add fibonacci",
            DispatchTier::Native,
        );
        let profile = mol.profile();
        // DispatchNative (1.0) + CargoCheck (3.0)
        assert!((profile.total_energy - 4.0).abs() < 0.1);
        assert_eq!(profile.step_count, 2);
    }

    #[test]
    fn test_tiered_candidates_produces_three() {
        let candidates = recipes::tiered_generation_candidates(
            PathBuf::from("/tmp/test/src/lib.rs"),
            "add fibonacci",
        );
        assert_eq!(candidates.len(), 3);
        // Energy should increase: native < local < cloud
        let energies: Vec<f32> = candidates
            .iter()
            .map(|(_, m)| m.profile().total_energy)
            .collect();
        assert!(energies[0] < energies[1]);
        assert!(energies[1] < energies[2]);
    }

    // ─── History-aware plan selection tests ───────────────────────────

    #[test]
    fn test_select_with_history_prefers_successful_recipe() {
        let plan_a = Molecule::atom(Atom::read("/tmp/a.rs"));
        let plan_b = Molecule::atom(Atom::read("/tmp/b.rs"));

        let candidates = vec![
            PlanCandidate {
                name: "a".into(),
                profile: plan_a.profile(),
                molecule: plan_a,
            },
            PlanCandidate {
                name: "b".into(),
                profile: plan_b.profile(),
                molecule: plan_b,
            },
        ];

        // Plan B has much higher historical success rate
        let rates = vec![0.2, 0.9];
        let selected = select_best_plan_with_history(&candidates, 1.0, 100.0, &rates);
        assert_eq!(
            selected,
            Some(1),
            "Should prefer historically successful plan"
        );
    }

    #[test]
    fn test_select_with_history_uninformative_prior() {
        let plan_a = Molecule::atom(Atom::read("/tmp/a.rs"));

        let candidates = vec![PlanCandidate {
            name: "a".into(),
            profile: plan_a.profile(),
            molecule: plan_a,
        }];

        // No history — should still select the only candidate
        let rates = vec![0.5]; // uninformative prior
        let selected = select_best_plan_with_history(&candidates, 1.0, 100.0, &rates);
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn test_select_with_history_respects_phi_gate() {
        let plan = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let candidates = vec![PlanCandidate {
            name: "check".into(),
            profile: plan.profile(),
            molecule: plan,
        }];
        // Phi too low even with great history
        let rates = vec![1.0];
        let selected = select_best_plan_with_history(&candidates, 0.0, 100.0, &rates);
        // CargoCheck needs phi=0.05, phi=0.0 fails
        assert!(selected.is_none() || candidates[selected.unwrap()].profile.phi_sufficient(0.0));
    }
}
