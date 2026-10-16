// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent feasibility and objective verifier for multi-crew repair schedules.
//!
//! This module deliberately does not call the search oracle or a heuristic scheduler.
//! A certificate is a proposed schedule plus its claimed weighted completion loss.
//! Verification checks the schedule against the model inputs and recomputes the loss.
//! It does NOT prove global optimality: an optimality claim still needs an exhaustive
//! search trace, a lower-bound proof, or a separately checkable optimality certificate.

use crate::multi_crew_repair_oracle::ExactMultiCrewRepair;
use crate::repair_crew_scheduling::{RepairCrew, RepairJob};
use std::collections::HashMap;

const OBJECTIVE_RELATIVE_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct RepairScheduleCertificate {
    pub schedule: Vec<ExactMultiCrewRepair>,
    pub claimed_weighted_completion_loss: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedRepairSchedule {
    /// Independently recomputed; this is not copied from the certificate claim.
    pub weighted_completion_loss: f64,
    pub jobs_verified: usize,
    pub crews_used: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RepairScheduleCertificateError {
    EmptyJobSet,
    EmptyCrewSet,
    EmptyIdentifier,
    DuplicateJob(String),
    DuplicateCrew(String),
    InvalidDuration(String),
    InvalidServiceValue(String),
    UnsupportedOutageStart(String),
    UnsupportedCrewAvailability(String),
    UnsupportedParts(String),
    UnknownDependency { job_id: String, dependency_id: String },
    DuplicateDependency { job_id: String, dependency_id: String },
    UnknownScheduledJob(String),
    UnknownCrew(String),
    MissingJob(String),
    DuplicateScheduledJob(String),
    CrewNotQualified { job_id: String, crew_id: String },
    DurationMismatch(String),
    CrewOverlap { crew_id: String, first_job: String, second_job: String },
    PrecedenceViolation { job_id: String, dependency_id: String },
    InvalidClaimedObjective,
    ObjectiveMismatch { claimed: f64, computed: f64 },
}

impl std::fmt::Display for RepairScheduleCertificateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid repair schedule certificate: {self:?}")
    }
}

impl std::error::Error for RepairScheduleCertificateError {}

/// Verify feasibility and recompute weighted completion loss independently.
///
/// This checker intentionally matches the narrow exact-oracle scope: jobs have
/// outage start zero, crews are available at zero, durations are positive integer
/// steps, there are no spare-part requirements, and each job needs one skill.
/// It fails closed on unsupported inputs instead of silently omitting constraints.
///
/// Passing this function means only that the supplied schedule is feasible under
/// these assumptions and its objective claim agrees with recomputation. It is not
/// evidence that the schedule is optimal.
pub fn verify_repair_schedule_certificate(
    jobs: &[RepairJob],
    crews: &[RepairCrew],
    certificate: &RepairScheduleCertificate,
) -> Result<VerifiedRepairSchedule, RepairScheduleCertificateError> {
    use RepairScheduleCertificateError as E;

    if jobs.is_empty() {
        return Err(E::EmptyJobSet);
    }
    if crews.is_empty() {
        return Err(E::EmptyCrewSet);
    }
    if !certificate.claimed_weighted_completion_loss.is_finite() {
        return Err(E::InvalidClaimedObjective);
    }

    let mut job_by_id: HashMap<String, &RepairJob> = HashMap::with_capacity(jobs.len());
    for job in jobs {
        if job.id.trim().is_empty() || job.required_skill.trim().is_empty() {
            return Err(E::EmptyIdentifier);
        }
        if job_by_id.insert(job.id.clone(), job).is_some() {
            return Err(E::DuplicateJob(job.id.clone()));
        }
        if job.duration_steps == 0 {
            return Err(E::InvalidDuration(job.id.clone()));
        }
        if !job.recovered_service_value_per_step.is_finite()
            || job.recovered_service_value_per_step <= 0.0
        {
            return Err(E::InvalidServiceValue(job.id.clone()));
        }
        if job.outage_start_step != 0 {
            return Err(E::UnsupportedOutageStart(job.id.clone()));
        }
        if !job.required_parts.is_empty() {
            return Err(E::UnsupportedParts(job.id.clone()));
        }
    }

    let mut crew_by_id: HashMap<String, &RepairCrew> = HashMap::with_capacity(crews.len());
    for crew in crews {
        if crew.id.trim().is_empty() || crew.skills.iter().any(|skill| skill.trim().is_empty()) {
            return Err(E::EmptyIdentifier);
        }
        if crew_by_id.insert(crew.id.clone(), crew).is_some() {
            return Err(E::DuplicateCrew(crew.id.clone()));
        }
        if crew.available_from_step != 0 {
            return Err(E::UnsupportedCrewAvailability(crew.id.clone()));
        }
    }

    for job in jobs {
        let mut dependencies = std::collections::HashSet::new();
        for dependency_id in &job.depends_on {
            if !job_by_id.contains_key(dependency_id) {
                return Err(E::UnknownDependency {
                    job_id: job.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
            if !dependencies.insert(dependency_id) {
                return Err(E::DuplicateDependency {
                    job_id: job.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
        }
    }

    let mut scheduled_by_id: HashMap<String, &ExactMultiCrewRepair> =
        HashMap::with_capacity(certificate.schedule.len());
    let mut intervals_by_crew: HashMap<String, Vec<&ExactMultiCrewRepair>> = HashMap::new();
    let mut crews_used = std::collections::HashSet::new();

    for entry in &certificate.schedule {
        let job = job_by_id
            .get(&entry.job_id)
            .copied()
            .ok_or_else(|| E::UnknownScheduledJob(entry.job_id.clone()))?;
        let crew = crew_by_id
            .get(&entry.crew_id)
            .copied()
            .ok_or_else(|| E::UnknownCrew(entry.crew_id.clone()))?;

        if scheduled_by_id.insert(entry.job_id.clone(), entry).is_some() {
            return Err(E::DuplicateScheduledJob(entry.job_id.clone()));
        }
        if !crew.skills.iter().any(|skill| skill == &job.required_skill) {
            return Err(E::CrewNotQualified {
                job_id: entry.job_id.clone(),
                crew_id: entry.crew_id.clone(),
            });
        }
        if entry.finish_step.checked_sub(entry.start_step) != Some(job.duration_steps) {
            return Err(E::DurationMismatch(entry.job_id.clone()));
        }
        intervals_by_crew
            .entry(entry.crew_id.clone())
            .or_default()
            .push(entry);
        crews_used.insert(entry.crew_id.clone());
    }

    for job in jobs {
        if !scheduled_by_id.contains_key(&job.id) {
            return Err(E::MissingJob(job.id.clone()));
        }
    }

    // Distinct crews may operate in parallel; a single crew may not overlap jobs.
    for (crew_id, intervals) in intervals_by_crew.iter_mut() {
        intervals.sort_by(|a, b| {
            (a.start_step, a.finish_step, &a.job_id)
                .cmp(&(b.start_step, b.finish_step, &b.job_id))
        });
        for pair in intervals.windows(2) {
            if pair[0].finish_step > pair[1].start_step {
                return Err(E::CrewOverlap {
                    crew_id: crew_id.clone(),
                    first_job: pair[0].job_id.clone(),
                    second_job: pair[1].job_id.clone(),
                });
            }
        }
    }

    // Dependency completion is checked from the submitted schedule, not the search.
    for job in jobs {
        let entry = scheduled_by_id[&job.id];
        for dependency_id in &job.depends_on {
            let dependency = scheduled_by_id[dependency_id];
            if dependency.finish_step > entry.start_step {
                return Err(E::PrecedenceViolation {
                    job_id: job.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
        }
    }

    let mut computed_loss = 0.0_f64;
    for job in jobs {
        computed_loss += scheduled_by_id[&job.id].finish_step as f64
            * job.recovered_service_value_per_step;
        if !computed_loss.is_finite() {
            return Err(E::InvalidClaimedObjective);
        }
    }
    let scale = certificate
        .claimed_weighted_completion_loss
        .abs()
        .max(computed_loss.abs())
        .max(1.0);
    if (certificate.claimed_weighted_completion_loss - computed_loss).abs()
        > OBJECTIVE_RELATIVE_TOLERANCE * scale
    {
        return Err(E::ObjectiveMismatch {
            claimed: certificate.claimed_weighted_completion_loss,
            computed: computed_loss,
        });
    }

    Ok(VerifiedRepairSchedule {
        weighted_completion_loss: computed_loss,
        jobs_verified: jobs.len(),
        crews_used: crews_used.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repair_crew_scheduling::RepairPartRequirement;

    fn job(id: &str, skill: &str, duration: usize, weight: f64, deps: &[&str]) -> RepairJob {
        RepairJob::new(
            id,
            skill,
            duration,
            0,
            weight,
            0,
            deps.iter().copied(),
            Vec::<RepairPartRequirement>::new(),
        )
    }

    fn crew(id: &str, skills: &[&str], available: usize) -> RepairCrew {
        RepairCrew::new(id, skills.iter().copied(), available)
    }

    fn jobs() -> Vec<RepairJob> {
        vec![
            job("a", "x", 4, 1.0, &[]),
            job("b", "y", 3, 5.0, &[]),
            job("c", "x", 2, 10.0, &[]),
        ]
    }

    fn crews() -> Vec<RepairCrew> {
        vec![crew("crew-x", &["x"], 0), crew("crew-y", &["y"], 0)]
    }

    fn schedule() -> Vec<ExactMultiCrewRepair> {
        vec![
            ExactMultiCrewRepair {
                job_id: "c".into(),
                crew_id: "crew-x".into(),
                start_step: 0,
                finish_step: 2,
            },
            ExactMultiCrewRepair {
                job_id: "a".into(),
                crew_id: "crew-x".into(),
                start_step: 2,
                finish_step: 6,
            },
            ExactMultiCrewRepair {
                job_id: "b".into(),
                crew_id: "crew-y".into(),
                start_step: 0,
                finish_step: 3,
            },
        ]
    }

    fn certificate(schedule: Vec<ExactMultiCrewRepair>, loss: f64) -> RepairScheduleCertificate {
        RepairScheduleCertificate {
            schedule,
            claimed_weighted_completion_loss: loss,
        }
    }

    #[test]
    fn accepts_feasible_parallel_schedule_and_recomputes_loss() {
        let verified = verify_repair_schedule_certificate(
            &jobs(), &crews(), &certificate(schedule(), 41.0),
        ).unwrap();
        assert_eq!(verified.jobs_verified, 3);
        assert_eq!(verified.crews_used, 2);
        assert!((verified.weighted_completion_loss - 41.0).abs() < 1.0e-9);
    }

    #[test]
    fn rejects_a_missing_job() {
        let mut proposed = schedule();
        proposed.pop();
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(proposed, 41.0),
            ),
            Err(RepairScheduleCertificateError::MissingJob(_))
        ));
    }

    #[test]
    fn rejects_a_job_scheduled_twice() {
        let mut proposed = schedule();
        proposed.push(proposed[0].clone());
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(proposed, 41.0),
            ),
            Err(RepairScheduleCertificateError::DuplicateScheduledJob(_))
        ));
    }

    #[test]
    fn rejects_an_unknown_scheduled_job() {
        let mut proposed = schedule();
        proposed[0].job_id = "not-in-input".into();
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(proposed, 41.0),
            ),
            Err(RepairScheduleCertificateError::UnknownScheduledJob(_))
        ));
    }

    #[test]
    fn rejects_a_crew_without_the_required_skill() {
        let mut proposed = schedule();
        proposed[0].crew_id = "crew-y".into();
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(proposed, 41.0),
            ),
            Err(RepairScheduleCertificateError::CrewNotQualified { .. })
        ));
    }

    #[test]
    fn rejects_overlapping_jobs_on_one_crew() {
        let mut proposed = schedule();
        proposed[1].start_step = 1;
        proposed[1].finish_step = 5;
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(proposed, 41.0),
            ),
            Err(RepairScheduleCertificateError::CrewOverlap { .. })
        ));
    }

    #[test]
    fn rejects_prerequisite_completed_after_dependent_starts() {
        let jobs = vec![job("power", "x", 2, 1.0, &[]), job("pump", "y", 3, 2.0, &["power"])];
        let proposed = vec![
            ExactMultiCrewRepair { job_id: "power".into(), crew_id: "crew-x".into(), start_step: 2, finish_step: 4 },
            ExactMultiCrewRepair { job_id: "pump".into(), crew_id: "crew-y".into(), start_step: 0, finish_step: 3 },
        ];
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs, &crews(), &certificate(proposed, 10.0),
            ),
            Err(RepairScheduleCertificateError::PrecedenceViolation { .. })
        ));
    }

    #[test]
    fn rejects_duration_that_disagrees_with_the_job() {
        let mut proposed = schedule();
        proposed[0].finish_step = 3;
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(proposed, 51.0),
            ),
            Err(RepairScheduleCertificateError::DurationMismatch(_))
        ));
    }

    #[test]
    fn rejects_a_falsified_objective_claim() {
        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(schedule(), 40.0),
            ),
            Err(RepairScheduleCertificateError::ObjectiveMismatch { .. })
        ));
    }

    #[test]
    fn rejects_out_of_scope_stock_and_late_availability() {
        let mut with_parts = jobs();
        with_parts[0].required_parts = vec![RepairPartRequirement::new("fuse", 1.0)];
        assert!(matches!(
            verify_repair_schedule_certificate(
                &with_parts, &crews(), &certificate(schedule(), 41.0),
            ),
            Err(RepairScheduleCertificateError::UnsupportedParts(_))
        ));

        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &[crew("late", &["x", "y"], 2), crew("crew-y", &["y"], 0)],
                &certificate(schedule(), 41.0),
            ),
            Err(RepairScheduleCertificateError::UnsupportedCrewAvailability(_))
        ));
    }

    #[test]
    fn rejects_nonfinite_service_weights_and_nonfinite_claims() {
        let mut invalid_weight = jobs();
        invalid_weight[0].recovered_service_value_per_step = f64::NAN;
        assert!(matches!(
            verify_repair_schedule_certificate(
                &invalid_weight, &crews(), &certificate(schedule(), 41.0),
            ),
            Err(RepairScheduleCertificateError::InvalidServiceValue(_))
        ));

        assert!(matches!(
            verify_repair_schedule_certificate(
                &jobs(), &crews(), &certificate(schedule(), f64::INFINITY),
            ),
            Err(RepairScheduleCertificateError::InvalidClaimedObjective)
        ));
    }

    #[test]
    fn rejects_unknown_and_duplicate_prerequisites() {
        let unknown = vec![job("pump", "y", 3, 2.0, &["missing"])];
        assert!(matches!(
            verify_repair_schedule_certificate(
                &unknown, &crews(), &certificate(vec![], 0.0),
            ),
            Err(RepairScheduleCertificateError::UnknownDependency { .. })
        ));

        let duplicate = vec![job("power", "x", 2, 1.0, &[]), job("pump", "y", 3, 2.0, &["power", "power"])];
        assert!(matches!(
            verify_repair_schedule_certificate(
                &duplicate, &crews(), &certificate(vec![], 0.0),
            ),
            Err(RepairScheduleCertificateError::DuplicateDependency { .. })
        ));
    }

    #[test]
    fn independently_verifies_a_schedule_returned_by_the_exact_oracle() {
        let input_jobs = jobs();
        let input_crews = crews();
        let exact = crate::multi_crew_repair_oracle::exhaustive_multi_crew_repair_optimum(
            &input_jobs,
            &input_crews,
        )
        .unwrap();
        let checked = verify_repair_schedule_certificate(
            &input_jobs,
            &input_crews,
            &certificate(
                exact.optimal_schedule.clone(),
                exact.minimum_weighted_completion_loss,
            ),
        )
        .unwrap();

        assert_eq!(checked.jobs_verified, exact.jobs_evaluated);
        assert!((checked.weighted_completion_loss
            - exact.minimum_weighted_completion_loss).abs() < 1.0e-9);
    }


    #[test]
    fn deterministic_mutation_campaign_rejects_all_10000_invalid_certificates() {
        let mut rejected = 0usize;
        for mutation in 0..10_000usize {
            let input_jobs = jobs();
            let input_crews = crews();
            let mut proposed = schedule();
            let mut claimed_loss = 41.0;

            let result = match mutation % 10 {
                0 => {
                    proposed[0].job_id = format!("ghost-job-{mutation}");
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                1 => {
                    proposed[0].crew_id = format!("ghost-crew-{mutation}");
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                2 => {
                    proposed[0].crew_id = "crew-y".into();
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                3 => {
                    proposed[0].finish_step += 1;
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                4 => {
                    proposed[1].start_step = mutation % 2;
                    proposed[1].finish_step = proposed[1].start_step + 4;
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                5 => {
                    proposed.push(proposed[0].clone());
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                6 => {
                    proposed.pop();
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                7 => {
                    claimed_loss += 1.0 + mutation as f64 * 1.0e-6;
                    verify_repair_schedule_certificate(
                        &input_jobs, &input_crews, &certificate(proposed, claimed_loss),
                    )
                }
                8 => {
                    let unknown_dependency = format!("ghost-dependency-{mutation}");
                    let invalid_jobs = vec![job(
                        "pump", "y", 3, 2.0, &[unknown_dependency.as_str()],
                    )];
                    verify_repair_schedule_certificate(
                        &invalid_jobs,
                        &input_crews,
                        &certificate(Vec::new(), 0.0),
                    )
                }
                _ => {
                    let dependency_jobs = vec![
                        job("power", "x", 2, 1.0, &[]),
                        job("pump", "y", 3, 2.0, &["power"]),
                    ];
                    let out_of_order = vec![
                        ExactMultiCrewRepair {
                            job_id: "power".into(),
                            crew_id: "crew-x".into(),
                            start_step: 2,
                            finish_step: 4,
                        },
                        ExactMultiCrewRepair {
                            job_id: "pump".into(),
                            crew_id: "crew-y".into(),
                            start_step: 0,
                            finish_step: 3,
                        },
                    ];
                    verify_repair_schedule_certificate(
                        &dependency_jobs,
                        &input_crews,
                        &certificate(out_of_order, 10.0),
                    )
                }
            };
            if result.is_err() {
                rejected += 1;
            }
        }
        assert_eq!(rejected, 10_000, "every invalid schedule mutation must be rejected");
    }

}
