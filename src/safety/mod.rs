//! Safety Module - Multi-Layer Defense Systems
//!
//! Week 1 Days 3-4: Amygdala (Visceral Safety) - Fast regex-based pre-cognitive defense
//! Phase 10/11: SafetyGuardrails (Semantic Safety) - HDC-based hypervector safety
//! Week 17: Digital Thymus (Immune System) - Tri-state adaptive semantic immune system
//!
//! ## Architecture
//!
//! The safety module implements a multi-layered defense system:
//!
//! 1. **Amygdala** (Layer 1): Pre-cognitive regex-based pattern matching
//!    - <10ms response time
//!    - Blocks known dangerous patterns instantly
//!    - Simulated cortisol/threat level tracking
//!
//! 2. **SafetyGuardrails** (Layer 2): HDC-based semantic safety
//!    - Hypervector similarity for forbidden content detection
//!    - Categories: SystemDestruction, DataDestruction, PromptInjection, etc.
//!
//! 3. **Thymus** (Layer 3): Adaptive semantic immune system
//!    - Tri-state verification: Allow / Deny / Uncertain (critical fix!)
//!    - T-Cell vectors that mature with confirmed threat detections
//!    - Z-score based statistical confidence
//!    - Epistemic tier mapping (E0-E4)
//!    - Timeout handling with fail-safe to Deny

pub mod amygdala;
pub mod guardrails;
pub mod thymus;

pub use amygdala::{AmygdalaActor, ThreatLevel};
pub use guardrails::{SafetyGuardrails, ForbiddenCategory, SafetyStats};
pub use thymus::{
    Thymus, ThymusConfig, VerificationVerdict, ThreatReport, TCellVector,
};

// Re-export EmpiricalTier as EpistemicTier for semantic naming
// (Both refer to the same z-score based confidence levels E0-E4)
pub use crate::hdc::statistical_retrieval::EmpiricalTier as EpistemicTier;
