// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

use super::engine::{ArithmeticEngine, ArithmeticResult};

/// A mathematical theorem with its proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theorem {
    /// Name of the theorem
    pub name: String,

    /// Statement of the theorem
    pub statement: String,

    /// The proof (sequence of arithmetic results)
    pub proof_steps: Vec<ArithmeticResult>,

    /// Total Φ of understanding the proof
    pub total_phi: f64,

    /// Whether all steps verified
    pub verified: bool,
}

/// Theorem prover using the arithmetic engine
pub struct TheoremProver {
    engine: ArithmeticEngine,
}

impl TheoremProver {
    pub fn new() -> Self {
        Self {
            engine: ArithmeticEngine::new(),
        }
    }

    /// Prove commutativity of addition: a + b = b + a
    pub fn prove_addition_commutative(&mut self, a: u64, b: u64) -> Theorem {
        let result1 = self.engine.add(a, b);
        let result2 = self.engine.add(b, a);

        let verified = result1.result.value == result2.result.value;
        let total_phi = result1.total_phi + result2.total_phi;

        Theorem {
            name: "Addition Commutativity".to_string(),
            statement: format!("{a} + {b} = {b} + {a}"),
            proof_steps: vec![result1, result2],
            total_phi,
            verified,
        }
    }

    /// Prove associativity of addition: (a + b) + c = a + (b + c)
    pub fn prove_addition_associative(&mut self, a: u64, b: u64, c: u64) -> Theorem {
        let ab = self.engine.add(a, b);
        let ab_c = self.engine.add(ab.result.value, c);

        let bc = self.engine.add(b, c);
        let a_bc = self.engine.add(a, bc.result.value);

        let verified = ab_c.result.value == a_bc.result.value;
        let total_phi = ab.total_phi + ab_c.total_phi + bc.total_phi + a_bc.total_phi;

        Theorem {
            name: "Addition Associativity".to_string(),
            statement: format!("({a} + {b}) + {c} = {a} + ({b} + {c})"),
            proof_steps: vec![ab, ab_c, bc, a_bc],
            total_phi,
            verified,
        }
    }

    /// Prove multiplication distributes over addition: a × (b + c) = a × b + a × c
    pub fn prove_distributive(&mut self, a: u64, b: u64, c: u64) -> Theorem {
        // Left side: a × (b + c)
        let bc = self.engine.add(b, c);
        let a_times_bc = self.engine.multiply(a, bc.result.value);

        // Right side: a × b + a × c
        let ab = self.engine.multiply(a, b);
        let ac = self.engine.multiply(a, c);
        let ab_plus_ac = self.engine.add(ab.result.value, ac.result.value);

        let verified = a_times_bc.result.value == ab_plus_ac.result.value;
        let total_phi = bc.total_phi
            + a_times_bc.total_phi
            + ab.total_phi
            + ac.total_phi
            + ab_plus_ac.total_phi;

        Theorem {
            name: "Distributive Law".to_string(),
            statement: format!("{a} × ({b} + {c}) = {a} × {b} + {a} × {c}"),
            proof_steps: vec![bc, a_times_bc, ab, ac, ab_plus_ac],
            total_phi,
            verified,
        }
    }

    /// Get the arithmetic engine (for direct computations)
    pub fn engine(&mut self) -> &mut ArithmeticEngine {
        &mut self.engine
    }
}

impl Default for TheoremProver {
    fn default() -> Self {
        Self::new()
    }
}
