use fusion_reject_audit::{
    AuditOutcome, AcceptedFeasibleManifest, RejectAuditPlan, RejectStratum, audit_rejected_candidates,
    deterministic_audit_sample_ids,
};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn main() {
    let seed = 0x5eed_f00d_cafe_babe;
    let population: Vec<String> = (0..10_000)
        .map(|index| format!("rejected-candidate-{index:05}"))
        .collect();
    let selected = deterministic_audit_sample_ids("all-rejected", &population, 1_000, seed)
        .expect("frozen audit sample must be valid");
    let samples = selected
        .into_iter()
        .enumerate()
        .map(|(index, candidate_id)| {
            let outcome = if index < 2 {
                AuditOutcome::ConfirmedInfeasible
            } else {
                AuditOutcome::ConfirmedInfeasible
            };
            fusion_reject_audit::AuditSample {
                candidate_id,
                outcome,
                evidence_ref: format!("synthetic-reference-run-{index:04}"),
            }
        })
        .collect();

    let plan = RejectAuditPlan {
        sampling_seed: seed,
        seed_commitment_ref: "synthetic-precommitted-seed-record".to_string(),
        seed_committed_before_outcomes: true,
        familywise_confidence_level: 0.95,
        target_feasible_recall: 0.99,
        accepted_feasible: AcceptedFeasibleManifest {
            count_lower_bound: 100_000,
            manifest_sha256: digest('a'),
            evidence_ref: "synthetic-accepted-feasible-manifest".to_string(),
        },
        strata: vec![RejectStratum {
            stratum_id: "all-rejected".to_string(),
            stratum_definition_sha256: digest('b'),
            population_manifest_sha256: digest('c'),
            selection_receipt_ref: "synthetic-selection-receipt".to_string(),
            rejected_candidate_ids: population,
            samples,
        }],
    };

    match audit_rejected_candidates(&plan) {
        Ok(report) => {
            println!("method={}", report.bound_method);
            println!("sampling={}", report.sampling_algorithm);
            println!("familywise_confidence={:.3}", report.familywise_confidence_level);
            println!("rejected_population={}", report.total_rejected_population);
            println!("audited_candidates={}", report.total_audited_candidates);
            println!(
                "missed_feasible_upper_bound={}",
                report.missed_feasible_count_upper_bound
            );
            match report.feasible_recall_lower_bound {
                Some(recall) => println!("feasible_recall_lower_bound={recall:.6}"),
                None => println!("feasible_recall_lower_bound=unavailable"),
            }
            println!("target_met={}", report.target_met);
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
