use fusion_reject_audit::{
    AcceptedCandidate, AcceptedCandidateManifest, AuditOutcome, FrozenCandidateUniverse,
    RejectAuditPlan, RejectStratum, audit_rejected_candidates, deterministic_audit_sample_ids,
};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn main() {
    let seed = 0x5eed_f00d_cafe_babe;
    let rejected_population: Vec<String> = (0..10_000)
        .map(|index| format!("rejected-candidate-{index:05}"))
        .collect();
    let sample_size = 1_000;
    let selected =
        deterministic_audit_sample_ids("all-rejected", &rejected_population, sample_size, seed)
            .expect("frozen audit sample must be valid");
    let samples = selected
        .into_iter()
        .enumerate()
        .map(|(index, candidate_id)| {
            let outcome = match index {
                0 | 1 => AuditOutcome::ConfirmedFeasible,
                2 => AuditOutcome::NonConverged,
                _ => AuditOutcome::ConfirmedInfeasible,
            };
            fusion_reject_audit::AuditSample {
                candidate_id,
                outcome,
                evidence_ref: format!("synthetic-reference-run-{index:04}"),
            }
        })
        .collect();

    let accepted: Vec<AcceptedCandidate> = (0..100_000)
        .map(|index| AcceptedCandidate {
            candidate_id: format!("accepted-candidate-{index:08}"),
            outcome: if index == 99_999 {
                AuditOutcome::Failed
            } else {
                AuditOutcome::ConfirmedFeasible
            },
            evidence_ref: format!("synthetic-accepted-run-{index:08}"),
        })
        .collect();

    let mut frozen_ids: Vec<String> = accepted
        .iter()
        .map(|candidate| candidate.candidate_id.clone())
        .collect();
    frozen_ids.extend(rejected_population.iter().cloned());
    frozen_ids.sort();

    let plan = RejectAuditPlan {
        frozen_universe: FrozenCandidateUniverse {
            candidate_ids: frozen_ids,
            manifest_sha256: digest('a'),
        },
        sampling_seed: seed,
        seed_commitment_ref: "synthetic-precommitted-seed-record".to_string(),
        seed_committed_before_outcomes: true,
        familywise_confidence_level: 0.95,
        target_feasible_recall: 0.99,
        accepted: AcceptedCandidateManifest {
            candidates: accepted,
            manifest_sha256: digest('b'),
            evidence_ref: "synthetic-accepted-population-manifest".to_string(),
        },
        strata: vec![RejectStratum {
            stratum_id: "all-rejected".to_string(),
            stratum_definition_sha256: digest('c'),
            population_manifest_sha256: digest('d'),
            selection_receipt_ref: "synthetic-selection-receipt".to_string(),
            rejected_candidate_ids: rejected_population,
            sample_size,
            samples,
        }],
    };

    match audit_rejected_candidates(&plan) {
        Ok(report) => {
            println!("method={}", report.bound_method);
            println!("sampling={}", report.sampling_algorithm);
            println!(
                "familywise_confidence={:.3}",
                report.familywise_confidence_level
            );
            println!("frozen_candidates={}", report.frozen_candidate_count);
            println!("accepted_candidates={}", report.accepted_candidate_count);
            println!(
                "accepted_feasible={},accepted_unresolved={}",
                report.accepted_confirmed_feasible_count, report.accepted_unresolved_count
            );
            println!("rejected_population={}", report.total_rejected_population);
            println!("audited_candidates={}", report.total_audited_candidates);
            println!(
                "missed_feasible_or_unresolved_upper_bound={}",
                report.missed_feasible_count_upper_bound
            );
            match report.feasible_recall_lower_bound {
                Some(recall) => println!("feasible_recall_lower_bound={recall:.6}"),
                None => println!("feasible_recall_lower_bound=unavailable"),
            }
            println!("target_met={}", report.target_met);
            for stratum in report.strata {
                println!(
                    "stratum={},sample={}/{},inclusion={:.4},potential_sample={},upper={}",
                    stratum.stratum_id,
                    stratum.sample_size,
                    stratum.population_size,
                    stratum.inclusion_probability,
                    stratum.potential_feasible_in_sample,
                    stratum.upper_missed_feasible_count
                );
            }
            println!(
                "SCOPE: synthetic sampling and conservative bound calculation only; no solver or physical evidence"
            );
        }
        Err(errors) => {
            eprintln!("rejected-candidate audit refused: {errors:?}");
            std::process::exit(1);
        }
    }
}
