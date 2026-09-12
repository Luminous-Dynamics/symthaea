// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::collections::{BTreeMap, BTreeSet};
use symthaea_support::{
    build_legacy_five_platform_portfolio_v1, initial_legacy_qualification_manifest_v1,
    plan_legacy_qualification_manifest_captures_v1, SourceSnapshotIdV1,
};

const FETCHED_AT_UNIX_MS: u64 = 1_800_000_000_000;

#[test]
fn procedures_use_nonempty_profile_scoped_non_platform_wide_sources() {
    let (pack, _, _) = build_legacy_five_platform_portfolio_v1(FETCHED_AT_UNIX_MS).unwrap();

    for procedure in &pack.procedures {
        let profile = pack
            .profile(procedure.platform)
            .expect("every legacy procedure platform must have a profile");

        assert!(
            !procedure.source_snapshots.is_empty(),
            "procedure {} has no source snapshots",
            procedure.id
        );
        assert!(
            procedure.source_snapshots.is_subset(&profile.source_snapshots),
            "procedure {} cites a source outside its {:?} profile",
            procedure.id,
            procedure.platform
        );
        assert_ne!(
            &procedure.source_snapshots, &profile.source_snapshots,
            "procedure {} cites the entire {:?} platform source set; add an explicit reviewed exception only if every source is materially required",
            procedure.id, procedure.platform
        );
    }
}

#[test]
fn capture_plan_procedure_fanout_is_exact_inverse_of_active_source_bindings() {
    let (pack, _, _) = build_legacy_five_platform_portfolio_v1(FETCHED_AT_UNIX_MS).unwrap();
    let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
    let plan = plan_legacy_qualification_manifest_captures_v1(&pack, &manifest).unwrap();

    let mut expected_by_snapshot: BTreeMap<SourceSnapshotIdV1, BTreeSet<String>> =
        BTreeMap::new();
    for procedure in &pack.procedures {
        for snapshot in &procedure.source_snapshots {
            expected_by_snapshot
                .entry(snapshot.clone())
                .or_default()
                .insert(procedure.id.clone());
        }
    }

    let planned_snapshots: BTreeSet<_> = plan
        .requests
        .iter()
        .map(|request| request.original_snapshot_id.clone())
        .collect();

    for (snapshot, expected_procedures) in &expected_by_snapshot {
        assert!(
            planned_snapshots.contains(snapshot),
            "active procedure source {} is absent from the manifest-scoped V3 capture plan",
            snapshot.0
        );

        let request = plan
            .requests
            .iter()
            .find(|request| &request.original_snapshot_id == snapshot)
            .unwrap();
        assert_eq!(
            &request.procedure_ids, expected_procedures,
            "capture-plan procedure fanout for {} must be the exact inverse of active procedure source bindings",
            snapshot.0
        );
    }

    for request in &plan.requests {
        let expected = expected_by_snapshot
            .get(&request.original_snapshot_id)
            .cloned()
            .unwrap_or_default();
        assert_eq!(
            &request.procedure_ids, &expected,
            "capture plan contains stale or invented procedure fanout for {}",
            request.original_snapshot_id.0
        );
    }
}
