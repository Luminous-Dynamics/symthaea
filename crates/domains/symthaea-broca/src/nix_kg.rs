// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nix Knowledge Graph — single source of truth for codegen lookup tables.
//!
//! Ported to symthaea-broca for compiler-grounded training.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConflictClaim {
    pub a: String,
    pub b: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServiceKeyword {
    pub keyword: String,
    pub option_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NixKgFile {
    pub version: u32,
    pub option_roots: Vec<String>,
    pub conflicts: Vec<ConflictClaim>,
    pub service_keywords: Vec<ServiceKeyword>,
    pub rag_prefixes: HashMap<String, Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct NixKg {
    pub option_roots: Vec<String>,
    pub conflicts: Vec<ConflictClaim>,
    pub service_keywords: Vec<String>,
}

impl Default for NixKg {
    fn default() -> Self {
        Self {
            option_roots: vec![
                "services".to_string(),
                "hardware".to_string(),
                "networking".to_string(),
            ],
            conflicts: vec![],
            service_keywords: vec![
                "postgresql".to_string(),
                "nginx".to_string(),
                "redis".to_string(),
                "docker".to_string(),
                "tailscale".to_string(),
                "symthaea".to_string(),
            ],
        }
    }
}

impl NixKg {
    pub fn reverse_prompt(&self, fragment: &str) -> Option<String> {
        let lower = fragment.to_lowercase();

        for kw in &self.service_keywords {
            if lower.contains(kw) {
                if lower.contains("options.services") {
                    return Some(format!("define {} service options", kw));
                }
                if lower.contains("enable = true") {
                    return Some(format!("enable {} service", kw));
                }
                return Some(format!("configure {} service", kw));
            }
        }

        if lower.contains("hardware.") {
            return Some("configure hardware".to_string());
        }

        None
    }

    pub fn matching_service_keywords<'a>(&'a self, lower: &str) -> Vec<&'a str> {
        self.service_keywords
            .iter()
            .filter(|k| lower.contains(k.as_str()))
            .map(String::as_str)
            .collect()
    }
}
