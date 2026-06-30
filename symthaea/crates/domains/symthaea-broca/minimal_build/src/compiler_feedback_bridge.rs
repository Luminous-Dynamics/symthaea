// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compiler Feedback Bridge — Maps substrate failure to cognitive intent.
//!
//! Allows Broca to ingest diagnostic JSON from rustc/cargo and convert
//! compiler violations into high-dimensional 'Logical Debt' hypervectors.

use serde_json::Value;
use symthaea_core::hdc::ContinuousHV;

#[derive(Clone)]
pub struct CompilerFeedbackBridge {
    pub hdc_dim: usize,
}

impl CompilerFeedbackBridge {
    pub fn new(dim: usize) -> Self {
        Self { hdc_dim: dim }
    }

    /// Convert a compiler diagnostic into a 'Logical Debt' hypervector.
    pub fn diagnostic_to_debt(&self, diagnostic: &Value) -> ContinuousHV {
        let message = diagnostic["message"]["message"]
            .as_str()
            .unwrap_or("unknown error");
        let code = diagnostic["message"]["code"]["code"]
            .as_str()
            .unwrap_or("E0000");

        println!(
            "🔧 Feedback Bridge: Mapping {} ({}) to manifold...",
            message, code
        );

        // Derive a debt vector from the error message and code
        let seed = hash_str(&format!("{code}:{message}"));
        let debt = ContinuousHV::random(self.hdc_dim, seed);
        debt.normalize();
        debt
    }
}

fn hash_str(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn diagnostic_to_debt_is_deterministic_for_same_message() {
        let bridge = CompilerFeedbackBridge::new(256);
        let diagnostic = json!({
            "message": {
                "message": "cannot find value `x` in this scope",
                "code": { "code": "E0425" }
            }
        });
        let a = bridge.diagnostic_to_debt(&diagnostic);
        let b = bridge.diagnostic_to_debt(&diagnostic);
        assert!(a.similarity(&b) > 0.999);
    }
}
