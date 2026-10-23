// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emit a deterministic report of exact semantic-prime collisions in the core lexicon.
//!
//! Run from the workspace root:
//!   cargo run -p symthaea-core --example lexical_alias_audit
//!
//! Shared prime sequences are representation collisions, not synonym judgments.

use serde::Serialize;
use std::error::Error;
use symthaea_core::hdc::grounded_understanding::LexicalGrounding;
use symthaea_core::hdc::universal_semantics::SemanticPrime;

#[derive(Debug, Serialize)]
struct AliasGroup {
    prime_sequence: Vec<SemanticPrime>,
    words: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AliasAuditReport {
    schema_version: u32,
    audit_id: &'static str,
    source_revision: Option<&'static str>,
    lexicon_size: usize,
    collision_group_count: usize,
    words_in_collision_groups: usize,
    groups: Vec<AliasGroup>,
    interpretation: &'static str,
}

fn main() -> Result<(), Box<dyn Error>> {
    let lexicon = LexicalGrounding::new();
    let groups: Vec<AliasGroup> = lexicon
        .identical_prime_groups()
        .into_iter()
        .map(|(prime_sequence, words)| AliasGroup {
            prime_sequence,
            words,
        })
        .collect();
    let report = AliasAuditReport {
        schema_version: 1,
        audit_id: "broca-exact-prime-collisions-v1",
        source_revision: option_env!("GIT_COMMIT_SHA"),
        lexicon_size: lexicon.lexicon_size(),
        collision_group_count: groups.len(),
        words_in_collision_groups: groups.iter().map(|group| group.words.len()).sum(),
        groups,
        interpretation: "Exact ordered prime-sequence collisions only; not synonym or equivalence labels.",
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
