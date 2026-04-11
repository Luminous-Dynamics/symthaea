// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! DSID (Decentralized Sovereign Identity) utilities.
//!
//! The identity DNA currently uses `did:mycelix:` prefix internally.
//! This module provides a display adapter that shows `dsid:mycelix:` in the UI
//! and converts back to `did:mycelix:` for zome calls.
//!
//! Once Phase 2 migrates the zome prefix, the adapter becomes a passthrough.

/// The wire prefix used by the identity DNA zomes.
pub const DID_PREFIX: &str = "did:mycelix:";

/// The display prefix shown in the UI (DSID branding).
pub const DSID_PREFIX: &str = "dsid:mycelix:";

/// Convert a wire DID (`did:mycelix:...`) to display DSID (`dsid:mycelix:...`).
pub fn display_dsid(did: &str) -> String {
    if did.starts_with(DID_PREFIX) {
        format!("{}{}", DSID_PREFIX, &did[DID_PREFIX.len()..])
    } else if did.starts_with(DSID_PREFIX) {
        did.to_string() // already in display format
    } else {
        did.to_string() // unknown format, pass through
    }
}

/// Convert a display DSID (`dsid:mycelix:...`) back to wire DID (`did:mycelix:...`).
pub fn wire_dsid(dsid: &str) -> String {
    if dsid.starts_with(DSID_PREFIX) {
        format!("{}{}", DID_PREFIX, &dsid[DSID_PREFIX.len()..])
    } else if dsid.starts_with(DID_PREFIX) {
        dsid.to_string() // already in wire format
    } else {
        dsid.to_string() // unknown format, pass through
    }
}

/// MFA Assurance levels (E-axis from the identity system).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssuranceLevel {
    /// E0 — No verification. Sybil-susceptible.
    Anonymous = 0,
    /// E1 — 1 factor (typically primary key pair).
    Basic = 1,
    /// E2 — 3+ factors from 2+ categories.
    Verified = 2,
    /// E3 — 5+ factors from 3+ categories.
    HighlyAssured = 3,
    /// E4 — All requirements + post-quantum verified.
    ConstitutionallyCritical = 4,
}

impl AssuranceLevel {
    pub fn from_score(score: f64) -> Self {
        if score >= 1.0 { Self::ConstitutionallyCritical }
        else if score >= 0.75 { Self::HighlyAssured }
        else if score >= 0.5 { Self::Verified }
        else if score >= 0.25 { Self::Basic }
        else { Self::Anonymous }
    }

    pub fn from_u8(level: u8) -> Self {
        match level {
            4 => Self::ConstitutionallyCritical,
            3 => Self::HighlyAssured,
            2 => Self::Verified,
            1 => Self::Basic,
            _ => Self::Anonymous,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Anonymous => "Anonymous",
            Self::Basic => "Basic",
            Self::Verified => "Verified",
            Self::HighlyAssured => "Highly Assured",
            Self::ConstitutionallyCritical => "Constitutional",
        }
    }

    pub fn short_label(&self) -> &'static str {
        match self {
            Self::Anonymous => "E0",
            Self::Basic => "E1",
            Self::Verified => "E2",
            Self::HighlyAssured => "E3",
            Self::ConstitutionallyCritical => "E4",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Anonymous => "\u{26AB}",          // ⚫
            Self::Basic => "\u{1F535}",             // 🔵
            Self::Verified => "\u{2705}",           // ✅
            Self::HighlyAssured => "\u{1F7E2}",     // 🟢
            Self::ConstitutionallyCritical => "\u{2B50}", // ⭐
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Anonymous => "assurance-e0",
            Self::Basic => "assurance-e1",
            Self::Verified => "assurance-e2",
            Self::HighlyAssured => "assurance-e3",
            Self::ConstitutionallyCritical => "assurance-e4",
        }
    }
}

/// Trust tiers for governance and participation rights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustTier {
    /// < 0.3 — Read-only, no voting.
    Observer,
    /// >= 0.3 — Basic participation.
    Basic,
    /// >= 0.4 — Can vote on major proposals.
    Standard,
    /// >= 0.6 — Can propose constitutional changes.
    Elevated,
    /// >= 0.8 — Full governance rights.
    Guardian,
}

impl TrustTier {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 { Self::Guardian }
        else if score >= 0.6 { Self::Elevated }
        else if score >= 0.4 { Self::Standard }
        else if score >= 0.3 { Self::Basic }
        else { Self::Observer }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Observer => "Observer",
            Self::Basic => "Basic",
            Self::Standard => "Standard",
            Self::Elevated => "Elevated",
            Self::Guardian => "Guardian",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Observer => "\u{26AB}",     // ⚫
            Self::Basic => "\u{1F535}",       // 🔵
            Self::Standard => "\u{1F7E2}",    // 🟢
            Self::Elevated => "\u{1F536}",    // 🔶
            Self::Guardian => "\u{1F451}",    // 👑
        }
    }
}

/// Format an agent public key as a truncated DSID for display.
pub fn agent_key_to_dsid_short(agent_key: &str) -> String {
    let suffix = if agent_key.len() > 12 {
        &agent_key[..12]
    } else {
        agent_key
    };
    format!("{}{}...", DSID_PREFIX, suffix)
}
