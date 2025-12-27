//! # NixOS Configuration Verification with Z3/SMT
//!
//! **PARADIGM SHIFT**: Prove NixOS configurations correct BEFORE applying!
//!
//! ## The Problem
//!
//! NixOS configurations can fail in complex ways:
//! - Service conflicts (two services binding same port)
//! - Missing dependencies (service needs package not installed)
//! - Security violations (insecure options enabled)
//! - Resource conflicts (overlapping file paths)
//! - Circular dependencies
//!
//! ## The Solution: Formal Verification via SMT
//!
//! We encode NixOS configuration constraints as SMT (Satisfiability Modulo Theories)
//! formulas and use Z3 to verify them:
//!
//! 1. **Constraint Encoding**: NixOS semantics → SMT formulas
//! 2. **Verification**: Z3 checks if constraints are satisfiable
//! 3. **Counterexamples**: If unsatisfiable, Z3 provides witness
//! 4. **Repair Suggestions**: HDC suggests fixes based on semantic similarity
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                  NixOS Configuration Verifier                    │
//! │                                                                  │
//! │  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐  │
//! │  │  Constraint     │   │  SMT Encoder     │   │  Z3 Solver   │  │
//! │  │  Extraction     │──►│  (NixOS → SMT)   │──►│  (Verify)    │  │
//! │  └─────────────────┘   └──────────────────┘   └──────────────┘  │
//! │           │                                         │            │
//! │           │                                         ▼            │
//! │           │                              ┌──────────────────┐    │
//! │           │                              │  Counterexample  │    │
//! │           │                              │  Analysis        │    │
//! │           │                              └────────┬─────────┘    │
//! │           │                                       │              │
//! │           ▼                                       ▼              │
//! │  ┌─────────────────┐                   ┌──────────────────────┐  │
//! │  │  HDC Semantic   │◄─────────────────►│  Repair Suggestions  │  │
//! │  │  Understanding  │                   │  (HDC-guided)        │  │
//! │  └─────────────────┘                   └──────────────────────┘  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Verification Categories
//!
//! 1. **Service Constraints**: Port conflicts, dependency ordering
//! 2. **Package Constraints**: Incompatibilities, version conflicts
//! 3. **Security Constraints**: Firewall rules, permissions
//! 4. **Resource Constraints**: File paths, memory limits
//! 5. **Semantic Constraints**: Configuration makes sense (HDC-powered)

pub mod constraints;
pub mod encoder;
pub mod verifier;

pub use constraints::{Constraint, ConstraintKind, ConstraintSet};
pub use encoder::SmtEncoder;
pub use verifier::{Verifier, VerificationResult, VerificationReport};
