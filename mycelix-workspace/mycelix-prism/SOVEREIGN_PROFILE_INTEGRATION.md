# Sovereign-Aware Security Integration (Proposal)

## Overview
By bridging `mycelix-prism` telemetry with the `sovereign-profile` civic identity, the browser can transition from a static "Reader Mode" to a dynamic, context-aware security agent that adapts to the user's sovereign integrity.

## Architectural Bridge

We will introduce a `ProfileSecurityPolicy` service that maps the 8D identity dimensions to Prism rendering configurations.

```rust
// Proposed mapping concept
pub struct SecurityPosture {
    pub allow_javascript: bool, // Always false (core mandate)
    pub reader_mode_forced: bool,
    pub epistemic_gatekeeping: bool,
}

impl SecurityPosture {
    pub fn from_profile(profile: &SovereignProfile) -> Self {
        // High EpistemicIntegrity? Allow more media.
        // Low EpistemicIntegrity? Stricter sanitization.
        let integrity = profile.score(SovereignDimension::EpistemicIntegrity);
        
        Self {
            allow_javascript: false,
            reader_mode_forced: integrity < 0.6,
            epistemic_gatekeeping: true,
        }
    }
}
```

## Integration Points
1.  **Reflex Arc Enhancement:** The Reflex Arc will consult the `ProfileSecurityPolicy` during the `PostParseVerdict` phase to determine the strictness of sanitization.
2.  **Chrome Chrome Badge:** The browser chrome (the security badge I've been working on) will reflect not just the content threat level, but also the current *Sovereign Posture* (e.g., "Sovereign Mode: Verified Integrity Active").

## Security Rationale
- **Adaptive Sovereignty:** The system becomes *more* open as the user demonstrates *more* epistemic integrity, rewarding virtuous participation in the Mycelix commons with richer browsing capabilities.
- **Privacy Preservation:** The mapping uses aggregated ZK-proofs, meaning Prism never learns the raw identity metrics, only the resulting policy posture.
