# Sovereign-Aware Privacy Policy Integration (Proposal)

## Overview
We will extend `ConsentStore` to be "Sovereign-Aware." Instead of a binary approved/denied list, the `resolve_zone` function will now consider the user's `SovereignProfile` dimensions. This enables the browser to be more permissive with verified, high-integrity identities.

## Design
Add a `SovereignProfile` parameter to `resolve_zone`.

```rust
// Proposed expansion of resolve_zone
pub fn resolve_zone(
    &self, 
    base_zone: ContentZone, 
    domain: &str, 
    profile: &SovereignProfile
) -> ContentZone {
    // 1. User denial always wins
    if self.is_denied(domain) { return ContentZone::Private; }

    // 2. Sovereign-aware logic
    // Example: If DomainCompetence score is high (e.g. > 0.8),
    // and the domain is classified as 'Public' by the reflex arc,
    // we allow it even if not explicitly approved.
    if base_zone == ContentZone::Local && profile.score(SovereignDimension::DomainCompetence) > 0.8 {
        return ContentZone::Public;
    }

    // 3. Fallback to standard consent check
    if self.is_approved(domain) { return ContentZone::Public; }
    base_zone
}
```

## Security Rationale
- **Contextual Privacy:** The browser behaves differently for a verified scientist (high domain competence in health) visiting a medical database vs. a general user, without revealing the user's specific identity details to the site.
- **Dynamic Policy:** The privacy gates are now elastic; as your sovereign civic identity evolves (gains score in specific dimensions), your ability to interact with the decentralized knowledge graph expands proportionally.
- **Privacy Preservation:** The use of ZK-aggregate proofs means the gatekeeper logic (`resolve_zone`) only receives the threshold check, not the raw dimension scores.
