// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral primitives, operators, and base enums for HDC moral reasoning.

use symthaea_core::hdc::ContinuousHV;

/// Default dimension for moral hypervectors
pub const MORAL_DIM: usize = 4096;

// ============================================================================
// Moral Primitives
// ============================================================================

/// The seven semantic role primitives for moral reasoning.
///
/// These are the "nouns" of our moral algebra - they represent the
/// semantic roles that entities can play in a moral scenario.
#[derive(Debug, Clone)]
pub struct MoralPrimitives {
    /// Dimension of all hypervectors
    pub dim: usize,

    /// AGENT - who performs the action
    /// Encodes the actor's identity/role in the scenario
    pub agent: ContinuousHV,

    /// PATIENT - who is affected by the action
    /// Encodes the recipient/target of moral consideration
    pub patient: ContinuousHV,

    /// ACTION - what is being done
    /// Encodes the verb/activity in the scenario
    pub action: ContinuousHV,

    /// INTENT - why the action is performed
    /// Encodes motivation (good/bad/neutral/unknown)
    pub intent: ContinuousHV,

    /// CONSENT - permission state
    /// Encodes whether permission was given/denied/absent
    pub consent: ContinuousHV,

    /// OBLIGATION - duty relationship
    /// Encodes responsibilities and expectations
    pub obligation: ContinuousHV,

    /// MAGNITUDE - scale/proportion
    /// Encodes size, importance, or proportionality
    pub magnitude: ContinuousHV,
}

impl MoralPrimitives {
    /// Create a new set of moral primitives with deterministic seeds.
    ///
    /// Each primitive gets a unique, reproducible hypervector.
    pub fn new(dim: usize) -> Self {
        // Use prime-based seeds for maximum orthogonality
        Self {
            dim,
            agent: ContinuousHV::random(dim, 1000003), // "who acts"
            patient: ContinuousHV::random(dim, 1000033), // "who is affected"
            action: ContinuousHV::random(dim, 1000037), // "what happens"
            intent: ContinuousHV::random(dim, 1000039), // "why"
            consent: ContinuousHV::random(dim, 1000081), // "permission"
            obligation: ContinuousHV::random(dim, 1000099), // "duty"
            magnitude: ContinuousHV::random(dim, 1000117), // "scale"
        }
    }

    /// Create with default dimension (4096)
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }

    /// Verify that primitives are approximately orthogonal
    pub fn verify_orthogonality(&self) -> f32 {
        let primitives = [
            &self.agent,
            &self.patient,
            &self.action,
            &self.intent,
            &self.consent,
            &self.obligation,
            &self.magnitude,
        ];

        let mut max_similarity = 0.0f32;
        for (i, a) in primitives.iter().enumerate() {
            for b in primitives.iter().skip(i + 1) {
                let sim = a.similarity(b).abs();
                if sim > max_similarity {
                    max_similarity = sim;
                }
            }
        }
        max_similarity
    }
}

// ============================================================================
// Moral Operators
// ============================================================================

/// The five compositional operators for moral reasoning.
///
/// These are the "verbs" of our moral algebra - they define how
/// primitives combine to form moral structures.
#[derive(Debug, Clone)]
pub struct MoralOperators {
    /// Dimension of all hypervectors
    pub dim: usize,

    /// CAUSES - causal relationship
    /// A CAUSES B means A brings about B
    pub causes: ContinuousHV,

    /// VIOLATES - rule violation
    /// A VIOLATES R means A breaks rule R
    pub violates: ContinuousHV,

    /// SATISFIES - obligation fulfillment
    /// A SATISFIES O means action A fulfills obligation O
    pub satisfies: ContinuousHV,

    /// PROPORTIONAL - magnitude comparison
    /// Used to encode proportionality between effort and reward
    pub proportional: ContinuousHV,

    /// NEGATES - negation/absence
    /// NEGATES X means "not X" or "X is absent"
    pub negates: ContinuousHV,
}

impl MoralOperators {
    /// Create a new set of moral operators with deterministic seeds.
    pub fn new(dim: usize) -> Self {
        // Use different prime seeds from primitives
        Self {
            dim,
            causes: ContinuousHV::random(dim, 2000003),
            violates: ContinuousHV::random(dim, 2000029),
            satisfies: ContinuousHV::random(dim, 2000039),
            proportional: ContinuousHV::random(dim, 2000081),
            negates: ContinuousHV::random(dim, 2000083),
        }
    }

    /// Create with default dimension
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }
}

// ============================================================================
// Intent and Magnitude Levels
// ============================================================================

/// Moral intent levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoralIntent {
    /// Positive/benevolent intent
    Good,
    /// Negative/malevolent intent
    Bad,
    /// No moral intent
    Neutral,
    /// Unknown or ambiguous
    Unknown,
}

/// Magnitude levels for proportionality reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Magnitude {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

impl Magnitude {
    /// Convert to numeric value for comparison
    pub fn value(&self) -> f32 {
        match self {
            Magnitude::Tiny => 0.1,
            Magnitude::Small => 0.3,
            Magnitude::Medium => 0.5,
            Magnitude::Large => 0.7,
            Magnitude::Huge => 0.9,
        }
    }
}

/// Consent state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsentState {
    /// Explicit consent given
    Given,
    /// Explicit consent denied
    Denied,
    /// No consent requested (absent)
    Absent,
    /// Implicit/assumed consent
    Implied,
}
