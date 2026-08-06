// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 3 of the test-corpus plan
//! (SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md): does every
//! one of `NixOSCausalPatterns::known_patterns()`'s curated cause->effect
//! edges actually surface via the real query API
//! (`NixCausalGraph::analyze_root_causes`), or are some of them seeded and
//! then structurally unreachable?
//!
//! First draft of this test asserted every pattern's `cause` must appear
//! in `analyze_root_causes(effect)`'s results -- 87/200 failed. Tracing
//! the algorithm found that assumption was too strong for legitimate
//! multi-hop chains: `analyze_root_causes` intentionally collapses to the
//! *ultimate* upstream root, correctly omitting intermediate causes (e.g.
//! `luks.devices -> availableKernelModules -> kernelModules` correctly
//! reports `luks.devices`, not the intermediate `availableKernelModules`,
//! when you query `kernelModules`). That's by design, not a bug.
//!
//! Tracing also found a real, distinct, CONFIRMED-AND-THEN-FIXED bug: any
//! pattern whose edges form a cycle -- which "conflicts" relationships do
//! by nature, since they're seeded as mutual pairs/rings (grub<->systemd-
//! boot, sudo<->doas, ntp->chrony->timesyncd->ntp) -- was structurally
//! invisible to `analyze_root_causes`. The algorithm's root-cause test was
//! "this node has no incoming edges in the discovered chain"; every node
//! in a fully-explored cycle has an incoming edge from its neighbor, so
//! none of them ever qualified, and the whole conflict silently produced
//! zero root causes -- including for anything *downstream* of a cycle
//! member (e.g. `security.sudo.extraRules`, via `security.sudo.enable`).
//!
//! Fixed by collapsing strongly-connected components (Tarjan's algorithm,
//! `compute_scc` in `causal_graph.rs`) before the root-cause check, so
//! only edges crossing *between* components count as "incoming". This
//! generalizes cleanly:
//! - A cycle with no real external cause feeding into it (sudo<->doas,
//!   the ntp/chrony/timesyncd ring) now reports all its members as
//!   co-equal root causes.
//! - A cycle that *does* have real external causes feeding into one of
//!   its members (grub<->systemd-boot, fed by `efiSysMountPoint`,
//!   `boot.loader.grub.device`, `boot.isContainer`) now correctly
//!   reports those deeper external causes instead of the cycle members
//!   themselves -- consistent with the same "report the ultimate root,
//!   not an intermediate" design already established for straight chains.
//! - The symptom itself is explicitly excluded from ever appearing as its
//!   own root cause, even when it's a cycle member.

use nixward::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};

fn seeded_graph() -> NixCausalGraph {
    let mut graph = NixCausalGraph::new(42);
    for (cause, effect, _relationship) in NixOSCausalPatterns::known_patterns() {
        graph.add_structural_edge(cause, effect, 0.5);
    }
    graph
}

#[test]
fn known_patterns_is_nonempty_and_matches_documented_scale() {
    let patterns = NixOSCausalPatterns::known_patterns();
    assert!(
        !patterns.is_empty(),
        "known_patterns() returned zero patterns"
    );
    // CLAUDE.md and this session's review describe "~210 curated static
    // patterns" -- a wide sanity band, not a strict count assertion (the
    // exact number is expected to drift as patterns are added).
    assert!(
        patterns.len() >= 150,
        "expected roughly ~210 curated patterns, found only {}: either the \
         documented scale is stale, or patterns were unintentionally removed",
        patterns.len()
    );
}

/// The achievable invariant: patterns whose `cause` is a true leaf (never
/// itself the `effect` of any other known pattern, so it can never be
/// "collapsed through" by the transitive walk) must be reachable as a
/// root cause when querying their `effect`.
#[test]
fn true_leaf_causes_are_reachable_via_analyze_root_causes() {
    let graph = seeded_graph();
    let patterns = NixOSCausalPatterns::known_patterns();

    let all_effects: std::collections::HashSet<&str> =
        patterns.iter().map(|(_, effect, _)| *effect).collect();

    let leaf_patterns: Vec<_> = patterns
        .iter()
        .filter(|(cause, _, _)| !all_effects.contains(cause))
        .collect();

    assert!(
        !leaf_patterns.is_empty(),
        "expected at least some patterns whose cause is a true leaf"
    );

    let mut unreachable: Vec<(&str, &str)> = Vec::new();
    for (cause, effect, _relationship) in &leaf_patterns {
        let analysis = graph.analyze_root_causes(effect);
        let found = analysis.root_causes.iter().any(|rc| rc.variable == *cause);
        if !found {
            unreachable.push((cause, effect));
        }
    }

    assert!(
        unreachable.is_empty(),
        "{} of {} true-leaf causal patterns are seeded but NOT reachable via \
         analyze_root_causes(), even though they have no incoming edges of \
         their own (so the 'ultimate root' collapsing logic doesn't explain \
         it): {:?}",
        unreachable.len(),
        leaf_patterns.len(),
        &unreachable[..unreachable.len().min(10)]
    );
}

/// A genuine multi-hop chain (cause is itself downstream of something
/// else) must still resolve all the way to *some* non-empty root-cause
/// set when queried at the far end -- exercising the transitive-walk path,
/// not just single-hop edges. Cycle members are included here too now
/// (unlike before the fix) since they no longer poison downstream queries.
#[test]
fn transitive_chains_produce_a_nonempty_root_cause_set() {
    let graph = seeded_graph();
    let patterns = NixOSCausalPatterns::known_patterns();

    let all_effects: std::collections::HashSet<&str> =
        patterns.iter().map(|(_, effect, _)| *effect).collect();

    let mut chain_found = false;
    for (cause, effect, _relationship) in &patterns {
        if all_effects.contains(cause) {
            chain_found = true;
            let analysis = graph.analyze_root_causes(effect);
            assert!(
                !analysis.root_causes.is_empty(),
                "transitive chain ending at '{effect}' (via intermediate '{cause}') \
                 produced zero root causes",
            );
        }
    }

    assert!(
        chain_found,
        "expected at least one real multi-hop chain among the curated patterns"
    );
}

/// FIXED: cyclic "conflicts" relationships (mutual pairs or rings) are no
/// longer structurally invisible to `analyze_root_causes`. A cycle with no
/// real external cause reports all its members as co-equal root causes;
/// a cycle fed by real external causes correctly defers to those deeper
/// causes instead (same "ultimate root" principle as straight chains).
#[test]
fn conflicts_cycles_now_resolve_correctly() {
    let graph = seeded_graph();

    // sudo <-> doas: a direct 2-cycle with no external causes feeding in.
    // Querying either side should surface the OTHER side as a co-equal
    // root cause (never itself).
    let sudo_query = graph.analyze_root_causes("security.sudo.enable");
    assert!(
        sudo_query
            .root_causes
            .iter()
            .any(|rc| rc.variable == "security.doas.enable"),
        "querying sudo.enable should surface doas.enable as a conflicting \
         root cause now that cycle-blindness is fixed. Got: {:?}",
        sudo_query
            .root_causes
            .iter()
            .map(|rc| &rc.variable)
            .collect::<Vec<_>>()
    );
    assert!(
        !sudo_query
            .root_causes
            .iter()
            .any(|rc| rc.variable == "security.sudo.enable"),
        "sudo.enable must never be reported as its own root cause"
    );

    let doas_query = graph.analyze_root_causes("security.doas.enable");
    assert!(
        doas_query
            .root_causes
            .iter()
            .any(|rc| rc.variable == "security.sudo.enable"),
        "querying doas.enable should surface sudo.enable as a conflicting \
         root cause. Got: {:?}",
        doas_query
            .root_causes
            .iter()
            .map(|rc| &rc.variable)
            .collect::<Vec<_>>()
    );

    // A real acyclic descendant of a cycle member (security.sudo.enable is
    // in the sudo<->doas cycle) must no longer come back empty either --
    // this was the broader "poisons downstream queries too" finding.
    let downstream = graph.analyze_root_causes("security.sudo.extraRules");
    assert!(
        !downstream.root_causes.is_empty(),
        "querying a descendant of a cycle member should no longer come back \
         empty now that the cycle itself resolves"
    );
    assert!(
        downstream.root_causes.iter().any(
            |rc| rc.variable == "security.sudo.enable" || rc.variable == "security.doas.enable"
        ),
        "expected the sudo<->doas cycle members to surface as the root cause \
         of sudo.extraRules. Got: {:?}",
        downstream
            .root_causes
            .iter()
            .map(|rc| &rc.variable)
            .collect::<Vec<_>>()
    );

    // ntp -> chrony -> timesyncd -> ntp: a 3-node ring, not just a pair --
    // confirms the fix isn't limited to direct mutual pairs. No external
    // causes feed into this ring in the curated data, so all non-symptom
    // members should surface.
    let chrony_query = graph.analyze_root_causes("services.chrony.enable");
    assert!(
        chrony_query
            .root_causes
            .iter()
            .any(|rc| rc.variable == "services.ntp.enable"),
        "querying chrony.enable should surface ntp.enable as a cyclic root \
         cause. Got: {:?}",
        chrony_query
            .root_causes
            .iter()
            .map(|rc| &rc.variable)
            .collect::<Vec<_>>()
    );

    // grub <-> systemd-boot: a 2-cycle that DOES have real external causes
    // feeding into it (efiSysMountPoint configures systemd-boot.enable;
    // grub.device and boot.isContainer feed grub.enable). The correct
    // outcome is that those deeper external causes surface, NOT the cycle
    // members themselves -- same "ultimate root, not intermediate"
    // principle already established for straight chains.
    let systemd_boot_query = graph.analyze_root_causes("boot.loader.systemd-boot.enable");
    assert!(
        systemd_boot_query
            .root_causes
            .iter()
            .any(|rc| rc.variable == "boot.loader.efi.efiSysMountPoint"),
        "querying systemd-boot.enable should surface its real external cause \
         efiSysMountPoint, since grub<->systemd-boot has genuine upstream \
         causes and shouldn't be reported as the root itself. Got: {:?}",
        systemd_boot_query
            .root_causes
            .iter()
            .map(|rc| &rc.variable)
            .collect::<Vec<_>>()
    );
    assert!(
        !systemd_boot_query
            .root_causes
            .iter()
            .any(|rc| rc.variable == "boot.loader.grub.enable"),
        "grub.enable has real external upstream causes (grub.device, \
         isContainer), so it should NOT itself be reported as a root cause \
         of systemd-boot.enable -- it's a correctly-collapsed intermediate. \
         Got: {:?}",
        systemd_boot_query
            .root_causes
            .iter()
            .map(|rc| &rc.variable)
            .collect::<Vec<_>>()
    );
}
