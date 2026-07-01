// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Unified Substrate interface for coding tasks (Nix, HCL, YAML).
//!
//! Provides a shared abstraction for scoring, repair, and distillation
//! across different languages.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Substrate {
    Nix,
    Hcl,
    Compose,
    Rust,
    Python,
}

impl Substrate {
    pub fn all() -> &'static [Substrate] {
        &[
            Substrate::Nix,
            Substrate::Hcl,
            Substrate::Compose,
            Substrate::Rust,
            Substrate::Python,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Substrate::Nix => "nix",
            Substrate::Hcl => "hcl",
            Substrate::Compose => "compose",
            Substrate::Rust => "rust",
            Substrate::Python => "python",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Substrate::Nix => "nix",
            Substrate::Hcl => "tf",
            Substrate::Compose => "yaml",
            Substrate::Rust => "rs",
            Substrate::Python => "py",
        }
    }
}

impl std::str::FromStr for Substrate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nix" => Ok(Substrate::Nix),
            "hcl" | "terraform" => Ok(Substrate::Hcl),
            "compose" | "docker-compose" | "yaml" => Ok(Substrate::Compose),
            "rust" | "rs" => Ok(Substrate::Rust),
            "python" | "py" => Ok(Substrate::Python),
            _ => Err(format!("Unknown substrate: {}", s)),
        }
    }
}

/// Unified verdict across all substrates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateVerdict {
    pub substrate: Substrate,
    pub pass: bool,
    pub score: f32,
    pub summary: String,
    pub parse_error: Option<String>,
}

/// Bridge to substrate-specific scoring logic.
pub fn score_substrate(substrate: Substrate, generated: &str, golden: &str) -> SubstrateVerdict {
    match substrate {
        Substrate::Nix => {
            #[cfg(feature = "code_generation")]
            {
                let v = crate::language::nix_scorer::score(generated, golden);
                SubstrateVerdict {
                    substrate,
                    pass: v.pass(),
                    score: v.path_jaccard,
                    summary: v.summary(),
                    parse_error: v.parse_error,
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                SubstrateVerdict {
                    substrate,
                    pass: false,
                    score: 0.0,
                    summary: "Feature code_generation disabled".to_string(),
                    parse_error: None,
                }
            }
        }
        Substrate::Hcl => {
            #[cfg(feature = "code_generation")]
            {
                let v = crate::language::hcl_scorer::score(generated, golden);
                SubstrateVerdict {
                    substrate,
                    pass: v.pass(),
                    score: v.path_jaccard,
                    summary: v.summary(),
                    parse_error: v.parse_error,
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                SubstrateVerdict {
                    substrate,
                    pass: false,
                    score: 0.0,
                    summary: "Feature code_generation disabled".to_string(),
                    parse_error: None,
                }
            }
        }
        Substrate::Compose => {
            #[cfg(feature = "code_generation")]
            {
                let v = crate::language::compose_scorer::score(generated, golden);
                SubstrateVerdict {
                    substrate,
                    pass: v.pass(),
                    score: v.path_jaccard,
                    summary: v.summary(),
                    parse_error: v.parse_error,
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                SubstrateVerdict {
                    substrate,
                    pass: false,
                    score: 0.0,
                    summary: "Feature code_generation disabled".to_string(),
                    parse_error: None,
                }
            }
        }
        Substrate::Rust => {
            #[cfg(feature = "code_generation")]
            {
                let v = crate::language::rust_scorer::score(generated, golden);
                SubstrateVerdict {
                    substrate,
                    pass: v.pass,
                    score: v.score,
                    summary: v.summary,
                    parse_error: v.parse_error,
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                SubstrateVerdict {
                    substrate,
                    pass: false,
                    score: 0.0,
                    summary: "Feature code_generation disabled".to_string(),
                    parse_error: None,
                }
            }
        }
        Substrate::Python => {
            #[cfg(feature = "code_generation")]
            {
                let v = crate::language::python_scorer::score(generated, golden);
                SubstrateVerdict {
                    substrate,
                    pass: v.pass,
                    score: v.score,
                    summary: v.summary,
                    parse_error: v.parse_error,
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                SubstrateVerdict {
                    substrate,
                    pass: false,
                    score: 0.0,
                    summary: "Feature code_generation disabled".to_string(),
                    parse_error: None,
                }
            }
        }
    }
}

/// Unified result for code generation across all substrates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateGenResult {
    pub substrate: Substrate,
    pub prompt: String,
    pub code: String,
    pub verdict: SubstrateVerdict,
    pub iterations: usize,
    pub repair_steps: usize,
    pub channels: Vec<f32>,
}

/// Top-level entrypoint for generated code with optional structural repair.
pub fn generate_with_repair(
    substrate: Substrate,
    prompt: &str,
    golden: Option<&str>,
    max_iters: usize,
) -> SubstrateGenResult {
    match substrate {
        Substrate::Nix => {
            #[cfg(feature = "code_generation")]
            {
                let golden_val = golden.unwrap_or("");
                let res = crate::language::nix_repair::generate_nix_with_scorer_repair(
                    prompt, golden_val, max_iters,
                );
                SubstrateGenResult {
                    substrate,
                    prompt: prompt.to_string(),
                    code: res.code,
                    verdict: SubstrateVerdict {
                        substrate,
                        pass: res.verdict.pass(),
                        score: res.verdict.path_jaccard,
                        summary: res.verdict.summary(),
                        parse_error: res.verdict.parse_error,
                    },
                    iterations: res.iterations,
                    repair_steps: res.steps.len(),
                    channels: res.channels.channels.to_vec(),
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                unimplemented!("code_generation feature required for Nix repair")
            }
        }
        Substrate::Hcl => {
            #[cfg(feature = "code_generation")]
            {
                let generator_result = crate::language::hcl_codegen::generate_hcl(prompt);
                let verdict = if let Some(g) = golden {
                    score_substrate(substrate, &generator_result.code, g)
                } else {
                    SubstrateVerdict {
                        substrate,
                        pass: true,
                        score: 1.0,
                        summary: "No golden provided".to_string(),
                        parse_error: None,
                    }
                };
                // HCL repair not yet implemented — Phase 3.b
                SubstrateGenResult {
                    substrate,
                    prompt: prompt.to_string(),
                    code: generator_result.code,
                    verdict,
                    iterations: 1,
                    repair_steps: 0,
                    // Use zeroed channels for now — Phase 3.c
                    channels: vec![0.0; 43],
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                unimplemented!("code_generation feature required for HCL")
            }
        }
        Substrate::Compose => {
            #[cfg(feature = "code_generation")]
            {
                let generator_result = crate::language::compose_codegen::generate_compose(prompt);
                let verdict = if let Some(g) = golden {
                    score_substrate(substrate, &generator_result.code, g)
                } else {
                    SubstrateVerdict {
                        substrate,
                        pass: true,
                        score: 1.0,
                        summary: "No golden provided".to_string(),
                        parse_error: None,
                    }
                };
                // Compose repair not yet implemented — Phase 3.b
                SubstrateGenResult {
                    substrate,
                    prompt: prompt.to_string(),
                    code: generator_result.code,
                    verdict,
                    iterations: 1,
                    repair_steps: 0,
                    channels: vec![0.0; 43],
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                unimplemented!("code_generation feature required for Compose")
            }
        }
        Substrate::Rust => {
            #[cfg(feature = "code_generation")]
            {
                // For now, fall through to basic code generator (which is logic-heavy)
                // In Phase 4, we will add a Rust repair loop.
                let generator_result = crate::language::code_generator::generate_rust(prompt);
                let verdict = if let Some(g) = golden {
                    score_substrate(substrate, &generator_result, g)
                } else {
                    SubstrateVerdict {
                        substrate,
                        pass: true,
                        score: 1.0,
                        summary: "No golden provided".to_string(),
                        parse_error: None,
                    }
                };
                SubstrateGenResult {
                    substrate,
                    prompt: prompt.to_string(),
                    code: generator_result,
                    verdict,
                    iterations: 1,
                    repair_steps: 0,
                    channels: vec![0.0; 43],
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                unimplemented!("code_generation feature required for Rust")
            }
        }
        Substrate::Python => {
            #[cfg(feature = "code_generation")]
            {
                let code = fallback_python(prompt);
                let verdict = if let Some(g) = golden {
                    score_substrate(substrate, &code, g)
                } else {
                    SubstrateVerdict {
                        substrate,
                        pass: true,
                        score: 1.0,
                        summary: "No golden provided".to_string(),
                        parse_error: None,
                    }
                };
                SubstrateGenResult {
                    substrate,
                    prompt: prompt.to_string(),
                    code,
                    verdict,
                    iterations: 1,
                    repair_steps: 0,
                    channels: vec![0.0; 43],
                }
            }
            #[cfg(not(feature = "code_generation"))]
            {
                unimplemented!("code_generation feature required for Python")
            }
        }
    }
}

#[cfg(feature = "code_generation")]
fn fallback_python(prompt: &str) -> String {
    let fn_name = prompt
        .split_whitespace()
        .find(|word| word.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or("generated_function")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .to_lowercase();
    let fn_name = if fn_name.is_empty() {
        "generated_function".to_string()
    } else {
        fn_name
    };
    format!("def {fn_name}():\n    \"\"\"Generated scaffold for: {prompt}\"\"\"\n    pass\n")
}
