// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language Gates — Substrate-specific structural boosts
//!
//! Provides optimized gating for specific programming languages (Nix, Rust, Go, etc.)
//! based on intent detection and structural patterns.

use crate::encoder::ThoughtChannels;
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single language-specific gate definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageGate {
    pub name: String,
    pub intent_keywords: Vec<String>,
    pub structural_ids: Vec<u32>,
    pub base_boost: f32,
}

/// Registry of available language gates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguageGateRegistry {
    pub gates: HashMap<String, LanguageGate>,
    pub default_gate: Option<LanguageGate>,
}

impl LanguageGateRegistry {
    pub fn new(tokenizer: &BpeTokenizer) -> Self {
        let mut gates = HashMap::new();

        // 1. Rust Gate
        let rust_keywords = vec![
            "rust".to_string(),
            "cargo".to_string(),
            "derive".to_string(),
            "impl".to_string(),
        ];
        let rust_structural = vec![
            "fn", "pub", "struct", "enum", "impl", "trait", "mod", "use", "where", "for", "match",
            "if", "let", "mut", "ref", "async", "unsafe", "type", "const", "static",
        ];
        gates.insert(
            "rust".to_string(),
            LanguageGate {
                name: "Rust".to_string(),
                intent_keywords: rust_keywords,
                structural_ids: resolve_token_ids(&rust_structural, tokenizer),
                base_boost: 2.2,
            },
        );

        // 2. Nix Gate
        let nix_keywords = vec![
            "nix".to_string(),
            "nixos".to_string(),
            "flake".to_string(),
            "derivation".to_string(),
        ];
        let nix_structural = vec![
            "let", "in", "inherit", "import", "with", "rec", "builtins", "pkgs", "lib", "stdenv",
        ];
        gates.insert(
            "nix".to_string(),
            LanguageGate {
                name: "Nix".to_string(),
                intent_keywords: nix_keywords,
                structural_ids: resolve_token_ids(&nix_structural, tokenizer),
                base_boost: 2.5,
            },
        );

        // 3. Go Gate
        let go_keywords = vec![
            "go".to_string(),
            "golang".to_string(),
            "goroutine".to_string(),
            "channel".to_string(),
        ];
        let go_structural = vec![
            "func",
            "package",
            "import",
            "struct",
            "interface",
            "type",
            "go",
            "chan",
            "select",
            "defer",
            "if",
            "for",
            "range",
            "map",
            "make",
            "new",
        ];
        gates.insert(
            "go".to_string(),
            LanguageGate {
                name: "Go".to_string(),
                intent_keywords: go_keywords,
                structural_ids: resolve_token_ids(&go_structural, tokenizer),
                base_boost: 1.9,
            },
        );

        // 4. Kubernetes Gate
        let k8s_keywords = vec![
            "kubernetes".to_string(),
            "k8s".to_string(),
            "helm".to_string(),
            "kubectl".to_string(),
        ];
        let k8s_structural = vec![
            "apiVersion",
            "kind",
            "metadata",
            "spec",
            "status",
            "items",
            "template",
            "containers",
            "image",
            "ports",
            "env",
            "volumeMounts",
            "volumes",
        ];
        gates.insert(
            "kubernetes".to_string(),
            LanguageGate {
                name: "Kubernetes".to_string(),
                intent_keywords: k8s_keywords,
                structural_ids: resolve_token_ids(&k8s_structural, tokenizer),
                base_boost: 2.3,
            },
        );

        // 5. OpenTofu / Terraform Gate
        let tofu_keywords = vec![
            "opentofu".to_string(),
            "terraform".to_string(),
            "tofu".to_string(),
            "hcl".to_string(),
        ];
        let tofu_structural = vec![
            "resource",
            "variable",
            "output",
            "module",
            "data",
            "provider",
            "terraform",
            "locals",
            "locals",
        ];
        gates.insert(
            "opentofu".to_string(),
            LanguageGate {
                name: "OpenTofu".to_string(),
                intent_keywords: tofu_keywords,
                structural_ids: resolve_token_ids(&tofu_structural, tokenizer),
                base_boost: 2.4,
            },
        );

        // 6. AWS CDK Gate
        let cdk_keywords = vec![
            "cdk".to_string(),
            "aws-cdk".to_string(),
            "construct".to_string(),
            "stack".to_string(),
        ];
        let cdk_structural = vec![
            "Stack",
            "Construct",
            "App",
            "CfnOutput",
            "Duration",
            "RemovalPolicy",
            "Tags",
        ];
        gates.insert(
            "cdk".to_string(),
            LanguageGate {
                name: "AWS CDK".to_string(),
                intent_keywords: cdk_keywords,
                structural_ids: resolve_token_ids(&cdk_structural, tokenizer),
                base_boost: 2.1,
            },
        );

        // 7. Ansible Gate
        let ansible_keywords = vec![
            "ansible".to_string(),
            "playbook".to_string(),
            "inventory".to_string(),
            "role".to_string(),
        ];
        let ansible_structural = vec![
            "hosts",
            "tasks",
            "vars",
            "handlers",
            "name",
            "state",
            "become",
            "include",
            "import",
            "with_items",
            "loop",
        ];
        gates.insert(
            "ansible".to_string(),
            LanguageGate {
                name: "Ansible".to_string(),
                intent_keywords: ansible_keywords,
                structural_ids: resolve_token_ids(&ansible_structural, tokenizer),
                base_boost: 2.0,
            },
        );

        // 8. Pulumi Gate
        let pulumi_keywords = vec![
            "pulumi".to_string(),
            "stack".to_string(),
            "output".to_string(),
            "config".to_string(),
        ];
        let pulumi_structural = vec![
            "pulumi",
            "Stack",
            "Config",
            "export",
            "ComponentResource",
            "Provider",
        ];
        gates.insert(
            "pulumi".to_string(),
            LanguageGate {
                name: "Pulumi".to_string(),
                intent_keywords: pulumi_keywords,
                structural_ids: resolve_token_ids(&pulumi_structural, tokenizer),
                base_boost: 2.2,
            },
        );

        // 9. CloudFormation Gate
        let cfn_keywords = vec![
            "cloudformation".to_string(),
            "cfn".to_string(),
            "template".to_string(),
        ];
        let cfn_structural = vec![
            "AWSTemplateFormatVersion",
            "Description",
            "Parameters",
            "Mappings",
            "Resources",
            "Outputs",
            "Conditions",
        ];
        gates.insert(
            "cloudformation".to_string(),
            LanguageGate {
                name: "CloudFormation".to_string(),
                intent_keywords: cfn_keywords,
                structural_ids: resolve_token_ids(&cfn_structural, tokenizer),
                base_boost: 2.3,
            },
        );

        // 10. Argo CD Gate
        let argo_keywords = vec![
            "argocd".to_string(),
            "argo".to_string(),
            "application".to_string(),
        ];
        let argo_structural = vec![
            "Application",
            "AppProject",
            "repoURL",
            "targetRevision",
            "destination",
            "syncPolicy",
        ];
        gates.insert(
            "argocd".to_string(),
            LanguageGate {
                name: "Argo CD".to_string(),
                intent_keywords: argo_keywords,
                structural_ids: resolve_token_ids(&argo_structural, tokenizer),
                base_boost: 2.4,
            },
        );

        // 11. Crossplane Gate
        let crossplane_keywords = vec![
            "crossplane".to_string(),
            "composition".to_string(),
            "resource-claim".to_string(),
        ];
        let crossplane_structural = vec![
            "Composition",
            "CompositeResourceDefinition",
            "XRD",
            "ProviderConfig",
            "reclaimPolicy",
        ];
        gates.insert(
            "crossplane".to_string(),
            LanguageGate {
                name: "Crossplane".to_string(),
                intent_keywords: crossplane_keywords,
                structural_ids: resolve_token_ids(&crossplane_structural, tokenizer),
                base_boost: 2.5,
            },
        );

        // 12. Bicep Gate
        let bicep_keywords = vec!["bicep".to_string(), "arm".to_string(), "azure".to_string()];
        let bicep_structural = vec![
            "resource",
            "param",
            "var",
            "output",
            "module",
            "targetScope",
            "existing",
        ];
        gates.insert(
            "bicep".to_string(),
            LanguageGate {
                name: "Bicep".to_string(),
                intent_keywords: bicep_keywords,
                structural_ids: resolve_token_ids(&bicep_structural, tokenizer),
                base_boost: 2.2,
            },
        );

        // 13. Helm Gate
        let helm_keywords = vec![
            "helm".to_string(),
            "chart".to_string(),
            "values".to_string(),
        ];
        let helm_structural = vec![
            "Values",
            "Release",
            "Chart",
            "Template",
            "Capabilities",
            "Files",
        ];
        gates.insert(
            "helm".to_string(),
            LanguageGate {
                name: "Helm".to_string(),
                intent_keywords: helm_keywords,
                structural_ids: resolve_token_ids(&helm_structural, tokenizer),
                base_boost: 2.1,
            },
        );

        let default_gate = gates.get("rust").cloned();

        Self {
            gates,
            default_gate,
        }
    }

    /// Detect the language intent from ThoughtChannels or context
    pub fn detect_intent(&self, channels: &ThoughtChannels) -> Option<&LanguageGate> {
        // Simple heuristic: check for intent signals in channels
        // You can expand this with actual intent classification from Broca
        let intent = channels
            .language_intent()
            .unwrap_or_else(|| "rust".to_string())
            .to_lowercase();

        if let Some(gate) = self.gates.get(&intent) {
            return Some(gate);
        }

        // Fallback: keyword matching on any available context
        for gate in self.gates.values() {
            let kws: Vec<&str> = gate.intent_keywords.iter().map(|s| s.as_str()).collect();
            if channels.prompt_contains_any(&kws) {
                return Some(gate);
            }
        }

        self.default_gate.as_ref()
    }

    /// Apply the language gate boost to logits
    /// strength can be modulated by emotional state (see emotional_gating_integration.rs)
    pub fn apply_gate(&self, logits: &mut [f32], gate: &LanguageGate, strength_multiplier: f32) {
        let boost = gate.base_boost * strength_multiplier;
        for &id in &gate.structural_ids {
            if (id as usize) < logits.len() {
                logits[id as usize] += boost;
            }
        }
    }

    pub fn list_gates(&self) -> Vec<&LanguageGate> {
        self.gates.values().collect()
    }

    /// Suppress all tokens associated with other languages except the target
    pub fn suppress_other_languages(
        &self,
        logits: &mut [f32],
        active_language: &str,
        penalty: f32,
    ) {
        let active_lower = active_language.to_lowercase();
        for (name, gate) in &self.gates {
            if name.to_lowercase() != active_lower {
                for &id in &gate.structural_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] -= penalty;
                    }
                }
            }
        }
    }
}

/// Helper to resolve token IDs from strings using the BPE tokenizer
fn resolve_token_ids(keywords: &[&str], tokenizer: &BpeTokenizer) -> Vec<u32> {
    keywords.iter().map(|kw| tokenizer.token_id(kw)).collect()
}
