// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent recomputation of a campaign transcript from frozen inputs and receipts.
//!
//! This module does not call the policy selector or the hidden synthetic oracle.
//! It is source-separated within the crate, not a separately built executable or
//! independent third-party verifier.
use super::{
    CampaignManifest, CampaignTranscript, Candidate, EvaluationReport, Feasibility, Fidelity,
    measurement_digest,
};
use std::collections::{BTreeMap, BTreeSet};

pub fn evaluate_transcript(
    manifest: &CampaignManifest,
    transcript: &CampaignTranscript,
) -> Result<EvaluationReport, String> {
    manifest.validate()?;
    let expected_manifest_digest = manifest.digest()?;
    if transcript.manifest_digest != expected_manifest_digest {
        return Err("manifest digest mismatch".into());
    }

    if transcript.records.len() < manifest.initial_actions.len() {
        return Err("transcript omitted required initial observations".into());
    }
    for (index, expected_action) in manifest.initial_actions.iter().enumerate() {
        if transcript.records[index].action != *expected_action {
            return Err("initial observations are missing or out of order".into());
        }
    }

    let candidates: BTreeMap<&str, &Candidate> = manifest
        .candidates
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect();
    let view = manifest.public_view();
    let mut seen_actions = BTreeSet::new();
    let mut cumulative_cost = 0u32;
    let mut first_conforming_cost = None;
    let mut best_utility: Option<i64> = None;
    let mut best_conforming_throughput: Option<u32> = None;
    let mut total_resource = 0u32;
    let mut points = BTreeSet::new();

    for (index, record) in transcript.records.iter().enumerate() {
        if record.step_index != index as u32 {
            return Err("step indices are not contiguous and ordered".into());
        }
        let candidate = candidates
            .get(record.action.candidate_id.as_str())
            .ok_or_else(|| "transcript references a candidate outside the manifest".to_string())?;
        if candidate.feasibility != Feasibility::Feasible {
            return Err("blocked or unknown candidate escaped the feasibility gate".into());
        }
        if candidate.process_profile_id != manifest.process_profile_id {
            return Err("candidate process profile does not match manifest".into());
        }
        let fidelity_available = match record.action.fidelity {
            Fidelity::Proxy => candidate.proxy_available,
            Fidelity::Reference => candidate.reference_available,
        };
        if !fidelity_available {
            return Err("transcript uses unavailable measurement fidelity".into());
        }
        if !seen_actions.insert(record.action.key()) {
            return Err("duplicate candidate/fidelity action in transcript".into());
        }
        if !record.measurements.complete() {
            return Err("missing required measurement; evaluation is not admissible".into());
        }

        let canonical_cost = view.action_cost(candidate, record.action.fidelity);
        if record.reported_cost_units != canonical_cost {
            return Err("reported action cost differs from manifest-defined cost".into());
        }
        cumulative_cost = cumulative_cost
            .checked_add(canonical_cost)
            .ok_or_else(|| "cumulative cost overflow".to_string())?;
        if cumulative_cost > manifest.budget_units {
            return Err("campaign exceeded its declared experiment budget".into());
        }

        let expected_measurement_digest = measurement_digest(
            &expected_manifest_digest,
            &record.action,
            &candidate.process_profile_id,
            &record.measurements,
        )?;
        if record.measurement_digest != expected_measurement_digest {
            return Err("measurement receipt digest mismatch".into());
        }

        let margin = record.measurements.quality_margin_milli.expect("checked complete");
        let throughput = record.measurements.throughput_units.expect("checked complete");
        let resource = record.measurements.resource_units.expect("checked complete");
        let is_conforming = record.action.fidelity == Fidelity::Reference
            && margin >= manifest.minimum_quality_margin_milli;
        if record.claimed_conforming != is_conforming {
            return Err(
                "claimed conformity does not match reference-fidelity measurement".into(),
            );
        }

        if is_conforming {
            if first_conforming_cost.is_none() {
                first_conforming_cost = Some(cumulative_cost);
            }
            let utility = i64::from(throughput) * 1000 - i64::from(resource) * 100
                + i64::from(margin);
            if best_conforming_throughput.is_none_or(|best| throughput > best) {
                best_conforming_throughput = Some(throughput);
            }
            if best_utility.is_none_or(|best| utility > best) {
                best_utility = Some(utility);
            }
        }
        total_resource = total_resource
            .checked_add(resource)
            .ok_or_else(|| "resource total overflow".to_string())?;
        points.insert((candidate.factor_x, candidate.factor_y));
    }

    if transcript.claimed_total_cost_units != cumulative_cost {
        return Err("claimed campaign cost differs from recomputed cost".into());
    }
    if transcript.claimed_first_conforming_cost_units != first_conforming_cost {
        return Err("claimed discovery cost differs from recomputed discovery cost".into());
    }
    if transcript.claimed_best_utility != best_utility {
        return Err("claimed best utility differs from recomputed utility".into());
    }
    let expected_stop_reason = if cumulative_cost >= manifest.budget_units {
        "BudgetExhausted"
    } else {
        "NoEligibleAffordableAction"
    };
    if transcript.stop_reason != expected_stop_reason {
        return Err("stop reason does not match the recomputed campaign outcome".into());
    }

    let blocked_count = manifest
        .candidates
        .iter()
        .filter(|candidate| matches!(&candidate.feasibility, Feasibility::Blocked { .. }))
        .count() as u32;
    let unknown_count = manifest
        .candidates
        .iter()
        .filter(|candidate| matches!(&candidate.feasibility, Feasibility::Unknown { .. }))
        .count() as u32;

    Ok(EvaluationReport {
        benchmark_id: manifest.benchmark_id.clone(),
        scenario: manifest.scenario,
        policy: transcript.policy,
        manifest_digest: expected_manifest_digest,
        experiments: transcript.records.len() as u32,
        total_cost_units: cumulative_cost,
        first_conforming_cost_units: first_conforming_cost,
        best_conforming_utility: best_utility,
        max_conforming_throughput_units: best_conforming_throughput,
        total_resource_units: total_resource,
        blocked_catalog_candidates: blocked_count,
        unknown_catalog_candidates: unknown_count,
        factor_design_full_rank: has_full_rank_factor_design(&points),
    })
}

pub(crate) fn has_full_rank_factor_design(points: &BTreeSet<(i32, i32)>) -> bool {
    let points: Vec<_> = points.iter().copied().collect();
    for a in 0..points.len() {
        for b in (a + 1)..points.len() {
            for c in (b + 1)..points.len() {
                let (x1, y1) = points[a];
                let (x2, y2) = points[b];
                let (x3, y3) = points[c];
                let area_twice =
                    (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
                if area_twice != 0 {
                    return true;
                }
            }
        }
    }
    false
}
