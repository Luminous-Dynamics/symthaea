// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(test)]
mod tests {
    use crate::{
        seed_legacy_computing_pack_v1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1,
        LegacyPlatformV1, LegacyProcedureAuthorityV1,
    };

    #[test]
    fn all_legacy_platforms_keep_explicit_coverage_gaps_or_source_mappings() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        for platform in LegacyPlatformV1::ALL {
            let profile = pack.profile(platform).unwrap();
            for area in LegacyKnowledgeAreaV1::ALL {
                assert!(matches!(
                    profile.state(area),
                    LegacyCoverageStateV1::Unmapped
                        | LegacyCoverageStateV1::SourceMapped
                        | LegacyCoverageStateV1::ClaimSeeded
                        | LegacyCoverageStateV1::ProcedureSeeded
                ));
            }
        }
    }

    #[test]
    fn legacy_advisory_steps_never_claim_execution_authority() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        assert!(!pack.procedures.is_empty());
        assert!(pack.procedures.iter().all(|procedure| procedure.steps.iter().all(|step| {
            matches!(
                step.authority,
                LegacyProcedureAuthorityV1::ReadOnlyObservation
                    | LegacyProcedureAuthorityV1::ChangeProposalOnly
            )
        })));
    }

    #[test]
    fn every_seeded_procedure_is_bound_to_registered_sources() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        for procedure in &pack.procedures {
            assert!(!procedure.source_snapshots.is_empty());
            for snapshot in &procedure.source_snapshots {
                assert!(pack.sources.snapshot(snapshot).is_some());
            }
        }
    }
}
