// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nix-specific code emitter
//!
//! Generates Nix expressions following nixfmt conventions from
//! abstract code plans and specifications.

use super::super::code_intent::CodeSpec;
use super::CodeEmitter;
use crate::dynamics::cfc_code_sequencer::CodePlanStep;

// ============================================================================
// Nix Emitter
// ============================================================================

/// Emitter for Nix expressions
pub struct NixEmitter;

impl CodeEmitter for NixEmitter {
    fn emit_from_spec(&self, spec: &CodeSpec, _plan: &[CodePlanStep]) -> String {
        let mut parts = Vec::new();

        parts.push(format!("# {}", spec.purpose));

        if spec.name.contains("derivation") || spec.name.contains("package") {
            parts.push("{ lib, stdenv, ... }:".to_string());
            parts.push(String::new());
            parts.push("stdenv.mkDerivation {".to_string());
            parts.push(format!("  pname = \"{}\";", spec.name));
            parts.push("  version = \"0.1.0\";".to_string());
            parts.push(String::new());
            parts.push("  src = ./.;".to_string());
            parts.push(String::new());
            parts.push("  meta = with lib; {".to_string());
            parts.push(format!("    description = \"{}\";", spec.purpose));
            parts.push("    license = licenses.mit;".to_string());
            parts.push("  };".to_string());
            parts.push("}".to_string());
        } else if spec.name.contains("module") || spec.name.contains("config") {
            parts.push("{ config, lib, pkgs, ... }:".to_string());
            parts.push(String::new());
            parts.push("{".to_string());
            parts.push(format!("  # {}", spec.purpose));
            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("  # Constraint: {}", c));
                }
            }
            parts.push("}".to_string());
        } else if spec.name.contains("overlay") {
            parts.push("final: prev: {".to_string());
            parts.push(format!("  # {}", spec.purpose));
            parts.push("}".to_string());
        } else if spec.name.contains("shell") || spec.name.contains("devShell") {
            parts.push("{ pkgs ? import <nixpkgs> {} }:".to_string());
            parts.push(String::new());
            parts.push("pkgs.mkShell {".to_string());
            parts.push("  buildInputs = with pkgs; [".to_string());
            parts.push("  ];".to_string());
            parts.push("}".to_string());
        } else {
            // Default: function or let binding
            parts.push(format!("{} =", spec.name));
            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("  # {}", c));
                }
            }
            parts.push("  null;".to_string());
        }

        parts.join("\n")
    }

    fn emit_function(&self, name: &str, params: &str, _return_type: &str, body: &str) -> String {
        if params.is_empty() {
            format!("{} = {};", name, body)
        } else {
            format!("{} = {}: {};", name, params, body)
        }
    }

    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String {
        let field_lines: Vec<String> = fields
            .iter()
            .map(|(n, _t)| format!("  {} = null;", n))
            .collect();

        format!("{} = {{\n{}\n}};", name, field_lines.join("\n"))
    }

    fn emit_import(&self, module: &str) -> String {
        format!("imports = [ {} ];", module)
    }

    fn language(&self) -> &str {
        "nix"
    }
}
