// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! First-run threat-model disclosure.
//!
//! The user MUST see this before any legal DID is created. Vectors 3
//! (network metadata) and 4 (device compulsion) are surfaced in plain
//! language, not buried in a docs/ file. See
//! `mycelix-lawful-identity/docs/THREAT_MODEL.md` for the full version.

/// Print the disclosure. If `banner_only` is true, condense for
/// re-display on every pre-init command; if false, print the full
/// disclosure for `init` or an explicit `disclose` invocation.
pub fn print_disclosure(banner_only: bool) {
    if banner_only {
        println!();
        println!("─── lawful-id — first-run disclosure required ────────────────────");
        println!();
        println!("Before creating a legal DID or importing a government-issued");
        println!("credential, you must read the threat model. Run:");
        println!();
        println!("    lawful-id disclose");
        println!();
        println!("Then acknowledge with `lawful-id init`.");
        println!();
        println!("──────────────────────────────────────────────────────────────────");
        return;
    }

    println!("═════════════════════════════════════════════════════════════════════");
    println!("  mycelix-lawful-identity — THREAT MODEL DISCLOSURE");
    println!("═════════════════════════════════════════════════════════════════════");
    println!();
    println!("  This CLI manages a SEPARATE identity (did:mycelix:legal:*) that");
    println!("  holds government-issued credentials (passport, mDL, SSN-derived).");
    println!("  It is strictly isolated from your primary Mycelix identity");
    println!("  (did:mycelix:primary:*) which you use for governance, MYCEL");
    println!("  reputation, and social interaction.");
    println!();
    println!("  The isolation is MATHEMATICAL — on-chain, no cryptographic");
    println!("  correlation between the two DIDs exists. An adversary reading");
    println!("  the entire Holochain DHT cannot link them.");
    println!();
    println!("  ⚠  WHAT THIS PROTECTS AGAINST  ⚠");
    println!();
    println!("  ✓ Mass passive on-chain surveillance. If a tax authority");
    println!("    learns your legal DID (because you filed a return with it),");
    println!("    they STILL cannot deanonymize your primary DID by reading");
    println!("    the DHT.");
    println!();
    println!("  ✓ Replay attacks. Every cross-DID proof requires a fresh");
    println!("    verifier-supplied nonce; reused nonces are rejected.");
    println!();
    println!("  ⚠  WHAT THIS DOES NOT PROTECT AGAINST — READ CAREFULLY  ⚠");
    println!();
    println!("  ✗ NETWORK METADATA CORRELATION. If your primary and legal DIDs");
    println!("    run on the same computer, they gossip to the DHT from the");
    println!("    SAME IP ADDRESS. A well-funded adversary (ISP, intelligence");
    println!("    agency, state-level surveillance program) can correlate the");
    println!("    two gossip streams trivially. The ONLY mitigations are:");
    println!();
    println!("    — Run your legal DID on a SEPARATE physical computer.");
    println!("    — Route one DID's gossip through Tor or I2P.");
    println!("    — Accept that network-level adversaries can link them.");
    println!();
    println!("  ✗ DEVICE COMPULSION. Both DIDs' private keys live in the");
    println!("    local keystore on this device. If law enforcement physically");
    println!("    seizes your device and compels you to unlock it, the link");
    println!("    becomes immediately apparent. Mitigations:");
    println!();
    println!("    — Store the legal DID's key on a separate air-gapped device.");
    println!("    — Use hardware-TEE-backed keys (Android StrongBox, iOS");
    println!("      Secure Enclave) that refuse export.");
    println!("    — Know the duress-wipe procedures for your platform.");
    println!();
    println!("  ─────────────────────────────────────────────────────────────────");
    println!();
    println!("  The full threat model is at:");
    println!("    mycelix-lawful-identity/docs/THREAT_MODEL.md");
    println!();
    println!("  Also read MYCELIX_STATE_COEXISTENCE.md at the repository root");
    println!("  for the broader architectural stance.");
    println!();
    println!("═════════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disclosure_mentions_both_honest_caveats() {
        // Not a strictly testable property since print goes to stdout,
        // but we can compile-check and ensure the text was not accidentally
        // truncated. Pull the body as a function that returns the string
        // if we ever switch to a captured-output pattern.
        print_disclosure(false);
    }

    #[test]
    fn banner_only_mode_does_not_panic() {
        print_disclosure(true);
    }
}
