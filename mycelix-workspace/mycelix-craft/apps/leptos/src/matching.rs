// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Client-Side Job Matching Engine
//!
//! Runs entirely in the Leptos frontend (WASM). Zomes provide anchor-based
//! queries; this module scores and ranks results locally.
//!
//! ## Architectural Decision
//!
//! Matching runs client-side, NOT in Holochain coordinator zomes.
//! Global DHT aggregation inside WASM would hit memory limits and
//! network timeouts. Zomes provide `search_jobs_by_skill` (anchor-indexed);
//! the client fetches subsets and scores locally.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A job posting as returned by the zome (simplified view for matching).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobView {
    pub title: String,
    pub organization: String,
    pub required_skills: Vec<String>,
    pub preferred_skills: Vec<String>,
    pub education_level: Option<String>,
    pub remote_ok: bool,
    /// Minimum credential vitality required (0-1000 permille). None = no requirement.
    pub vitality_minimum: Option<u16>,
}

/// A scored job match result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobMatch {
    pub job: JobView,
    pub score: f32,
    pub matched_required: Vec<String>,
    pub matched_preferred: Vec<String>,
    pub credential_bonus: bool,
}

/// Compute match score for a candidate against a job posting.
///
/// Score components:
/// - Skill overlap: `|agent_skills ∩ required| / |required|` (0-0.6)
/// - Credential match: +0.2 if education_level matches
/// - Preferred skills: +0.02 per match (max 0.2)
pub fn score_job(
    job: &JobView,
    agent_skills: &HashSet<String>,
    agent_education: Option<&str>,
) -> JobMatch {
    // Required skill overlap (biggest weight)
    let matched_required: Vec<String> = job
        .required_skills
        .iter()
        .filter(|s| agent_skills.contains(s.as_str()))
        .cloned()
        .collect();

    let required_score = if job.required_skills.is_empty() {
        0.3 // No requirements = moderate default
    } else {
        (matched_required.len() as f32 / job.required_skills.len() as f32) * 0.6
    };

    // Preferred skill bonus
    let matched_preferred: Vec<String> = job
        .preferred_skills
        .iter()
        .filter(|s| agent_skills.contains(s.as_str()))
        .cloned()
        .collect();
    let preferred_score = (matched_preferred.len() as f32 * 0.02).min(0.2);

    // Credential/education match bonus
    let credential_bonus = match (&job.education_level, agent_education) {
        (Some(required), Some(agent)) => {
            required.to_lowercase() == agent.to_lowercase() || agent.to_lowercase() == "phd"
        }
        (None, _) => true, // No requirement = bonus
        _ => false,
    };
    let credential_score = if credential_bonus { 0.2 } else { 0.0 };

    let total = required_score + preferred_score + credential_score;

    JobMatch {
        job: job.clone(),
        score: total.min(1.0),
        matched_required,
        matched_preferred,
        credential_bonus,
    }
}

/// Rank jobs by match score (descending), filter to minimum threshold.
/// If `agent_avg_vitality` is provided, jobs requiring higher vitality are excluded.
pub fn rank_jobs(
    jobs: &[JobView],
    agent_skills: &HashSet<String>,
    agent_education: Option<&str>,
    agent_avg_vitality: Option<u16>,
    min_score: f32,
    limit: usize,
) -> Vec<JobMatch> {
    let mut matches: Vec<JobMatch> = jobs
        .iter()
        .filter(|j| {
            // Skip jobs whose vitality requirement exceeds the agent's average
            match (j.vitality_minimum, agent_avg_vitality) {
                (Some(min), Some(agent)) => agent >= min,
                (Some(_), None) => false, // Job requires vitality but agent has no credentials
                _ => true,
            }
        })
        .map(|j| score_job(j, agent_skills, agent_education))
        .filter(|m| m.score >= min_score)
        .collect();

    matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    matches.truncate(limit);
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust_dev_job() -> JobView {
        JobView {
            title: "Rust Developer".into(),
            organization: "Luminous Dynamics".into(),
            required_skills: vec!["rust".into(), "holochain".into(), "wasm".into()],
            preferred_skills: vec!["leptos".into(), "nix".into()],
            education_level: Some("Bachelor's".into()),
            remote_ok: true,
            vitality_minimum: None,
        }
    }

    #[test]
    fn test_perfect_match() {
        let skills: HashSet<String> =
            ["rust", "holochain", "wasm", "leptos", "nix"]
                .iter().map(|s| s.to_string()).collect();
        let m = score_job(&rust_dev_job(), &skills, Some("Bachelor's"));
        // 3/3 required = 0.6, 2 preferred = 0.04, credential = 0.2 => 0.84
        assert!(m.score > 0.8);
        assert_eq!(m.matched_required.len(), 3);
        assert!(m.credential_bonus);
    }

    #[test]
    fn test_no_match() {
        let skills: HashSet<String> =
            ["python", "django"].iter().map(|s| s.to_string()).collect();
        let m = score_job(&rust_dev_job(), &skills, None);
        assert!(m.score < 0.1);
        assert!(m.matched_required.is_empty());
        assert!(!m.credential_bonus);
    }

    #[test]
    fn test_partial_match() {
        let skills: HashSet<String> =
            ["rust", "wasm"].iter().map(|s| s.to_string()).collect();
        let m = score_job(&rust_dev_job(), &skills, Some("Bachelor's"));
        // 2/3 required = 0.4, credential = 0.2 -> 0.6
        assert!(m.score > 0.5 && m.score < 0.8);
    }

    #[test]
    fn test_rank_ordering() {
        let jobs = vec![
            JobView {
                title: "Python Dev".into(),
                organization: "A".into(),
                required_skills: vec!["python".into()],
                preferred_skills: vec![],
                education_level: None,
                remote_ok: true,
                vitality_minimum: None,
            },
            rust_dev_job(),
        ];
        let skills: HashSet<String> =
            ["rust", "holochain", "wasm"].iter().map(|s| s.to_string()).collect();
        let ranked = rank_jobs(&jobs, &skills, Some("Bachelor's"), None, 0.0, 10);
        assert_eq!(ranked[0].job.title, "Rust Developer");
    }

    #[test]
    fn test_min_score_filter() {
        let jobs = vec![rust_dev_job()];
        let skills: HashSet<String> = HashSet::new();
        let ranked = rank_jobs(&jobs, &skills, None, None, 0.5, 10);
        assert!(ranked.is_empty()); // No skills -> low score -> filtered
    }

    #[test]
    fn test_vitality_filter() {
        let job = JobView {
            title: "Senior Rust Dev".into(),
            organization: "Luminous Dynamics".into(),
            required_skills: vec!["rust".into()],
            preferred_skills: vec![],
            education_level: None,
            remote_ok: true,
            vitality_minimum: Some(700),
        };
        let jobs = vec![job];
        let skills: HashSet<String> =
            ["rust"].iter().map(|s| s.to_string()).collect();

        // Agent vitality below minimum -- job should be filtered out
        let ranked = rank_jobs(&jobs, &skills, None, Some(500), 0.0, 10);
        assert!(ranked.is_empty(), "Job should be excluded when agent vitality < minimum");

        // Agent vitality above minimum -- job should appear
        let ranked = rank_jobs(&jobs, &skills, None, Some(800), 0.0, 10);
        assert_eq!(ranked.len(), 1, "Job should appear when agent vitality >= minimum");
        assert_eq!(ranked[0].job.title, "Senior Rust Dev");
    }
}
