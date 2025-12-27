// symthaea_swarm module - Symthaea ↔ Mycelix Protocol integration
//
// This module provides the interface for Symthaea instances to participate
// in the Mycelix collective intelligence network with:
// - DKG (Decentralized Knowledge Graph) for pattern sharing
// - MATL (Mycelix Adaptive Trust Layer) for 45% Byzantine tolerance
// - MFDI (Multi-Factor Decentralized Identity) for Instrumental Actor registration

pub mod api;
pub mod holochain;

// Re-export commonly used types
pub use api::{
    CartelRisk, ClaimId, CompositeTrustScore, Did, DkgClient, EpistemicClaim, EpistemicTierE,
    EpistemicTierN, EvaluatedClaim, Hypervector, MaterialityTierM, MatlClient, MfdiClient,
    MycelixIdentity, NormativeTierN, PatternQuery, SymthaeaPattern, SymthaeaSwarmClient, SwarmError,
};

pub use holochain::HolochainSwarmClient;
