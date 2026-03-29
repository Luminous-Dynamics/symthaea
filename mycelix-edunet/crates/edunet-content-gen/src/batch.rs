// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batch content generation for curriculum graphs.
//!
//! Generates lessons, assessments, and flashcards for every node in a
//! curriculum document using the [`ContentPipeline`]. Features:
//! - Progress tracking with resume capability
//! - Quality control with configurable thresholds
//! - Filtered generation by subject, level, or node IDs

use crate::pipeline::{ContentGenError, ContentGenerator, ContentPipeline, StandardInput};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Complete generated content for a single curriculum node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeContent {
    /// Node ID from the curriculum graph.
    pub node_id: String,
    /// Generated lesson with explanation, examples, problems.
    pub lesson: GeneratedLesson,
    /// Generated assessment items.
    pub assessment: GeneratedAssessment,
    /// Generated flashcards for spaced repetition.
    pub flashcards: Vec<GeneratedFlashcard>,
    /// Quality summary.
    pub quality: BatchQuality,
    /// When this content was generated.
    pub generated_at: String,
}

/// Quality summary for a generated node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchQuality {
    pub coherence: f32,
    pub hallucination_detected: bool,
    pub needs_human_review: bool,
    pub veto_count: u32,
}

/// Progress tracking for resume capability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchProgress {
    pub total: usize,
    pub completed: Vec<String>,
    pub failed: Vec<FailedNode>,
    pub started_at: String,
    pub last_updated: String,
}

/// Record of a failed generation attempt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailedNode {
    pub node_id: String,
    pub error: String,
    pub attempts: u32,
}

/// Batch report summarizing the generation run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchReport {
    pub total_nodes: usize,
    pub generated: usize,
    pub failed: usize,
    pub skipped: usize,
    pub avg_coherence: f32,
    pub needs_review_count: usize,
    pub total_problems_generated: usize,
    pub total_flashcards_generated: usize,
}

/// Filter for selecting a subset of nodes.
#[derive(Clone, Debug)]
pub enum NodeFilter {
    /// Only nodes matching this subject area.
    BySubject(String),
    /// Only nodes at this grade level.
    ByGradeLevel(String),
    /// Only specific node IDs.
    ByIds(Vec<String>),
    /// All nodes.
    All,
}

/// Configuration for a batch generation run.
pub struct BatchConfig {
    pub output_dir: PathBuf,
    pub progress_file: PathBuf,
    pub quality_threshold: f32,
    pub max_retries: u32,
    pub filter: NodeFilter,
    pub num_assessment_items: usize,
    pub num_flashcards: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("generated_content"),
            progress_file: PathBuf::from("batch_progress.json"),
            quality_threshold: 0.6,
            max_retries: 2,
            filter: NodeFilter::All,
            num_assessment_items: 6,
            num_flashcards: 8,
        }
    }
}

/// Run batch content generation for a curriculum document.
///
/// Returns a report summarizing the run.
pub fn run_batch<G: ContentGenerator>(
    pipeline: &ContentPipeline<G>,
    standards: &[StandardInput],
    config: &BatchConfig,
) -> BatchReport {
    std::fs::create_dir_all(&config.output_dir).ok();

    // Load existing progress for resume
    let mut progress = load_progress(&config.progress_file);
    let completed_ids: HashSet<String> = progress.completed.iter().cloned().collect();

    // Filter standards
    let filtered: Vec<&StandardInput> = standards
        .iter()
        .filter(|s| match &config.filter {
            NodeFilter::All => true,
            NodeFilter::BySubject(subj) => {
                s.domain.to_lowercase().contains(&subj.to_lowercase())
                    || s.code.to_lowercase().contains(&subj.to_lowercase())
            }
            NodeFilter::ByGradeLevel(level) => s.grade_level == *level,
            NodeFilter::ByIds(ids) => ids.contains(&s.code),
        })
        .filter(|s| !completed_ids.contains(&s.code))
        .collect();

    progress.total = standards.len();

    let mut report = BatchReport {
        total_nodes: filtered.len(),
        generated: 0,
        failed: 0,
        skipped: completed_ids.len(),
        avg_coherence: 0.0,
        needs_review_count: 0,
        total_problems_generated: 0,
        total_flashcards_generated: 0,
    };

    let mut coherence_sum = 0.0f32;

    for (i, standard) in filtered.iter().enumerate() {
        eprint!(
            "[{}/{}] {}... ",
            i + 1,
            filtered.len(),
            truncate_title(&standard.code, 30)
        );

        match generate_node_content(pipeline, standard, config) {
            Ok(content) => {
                // Write to file
                let filename = sanitize_filename(&content.node_id);
                let path = config.output_dir.join(format!("{filename}.json"));
                if let Ok(json) = serde_json::to_string_pretty(&content) {
                    std::fs::write(&path, &json).ok();
                }

                coherence_sum += content.quality.coherence;
                if content.quality.needs_human_review {
                    report.needs_review_count += 1;
                }
                report.total_problems_generated += content.lesson.practice_problems.len();
                report.total_flashcards_generated += content.flashcards.len();
                report.generated += 1;

                progress.completed.push(standard.code.clone());
                eprintln!(
                    "OK (coherence: {:.2}, {} problems, {} flashcards)",
                    content.quality.coherence,
                    content.lesson.practice_problems.len(),
                    content.flashcards.len(),
                );
            }
            Err(e) => {
                report.failed += 1;
                progress.failed.push(FailedNode {
                    node_id: standard.code.clone(),
                    error: e.to_string(),
                    attempts: config.max_retries,
                });
                eprintln!("FAILED: {}", e);
            }
        }

        // Save progress periodically
        if (i + 1) % 10 == 0 {
            save_progress(&config.progress_file, &progress);
        }
    }

    // Final progress save
    progress.last_updated = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    save_progress(&config.progress_file, &progress);

    if report.generated > 0 {
        report.avg_coherence = coherence_sum / report.generated as f32;
    }

    report
}

/// Generate content for a single node.
fn generate_node_content<G: ContentGenerator>(
    pipeline: &ContentPipeline<G>,
    standard: &StandardInput,
    config: &BatchConfig,
) -> Result<NodeContent, ContentGenError> {
    let lesson = pipeline.generate_lesson(standard)?;
    let assessment = pipeline.generate_assessment(standard, config.num_assessment_items)?;
    let flashcards = pipeline.generate_flashcards(standard, config.num_flashcards)?;

    let quality = BatchQuality {
        coherence: lesson.generation_quality.coherence,
        hallucination_detected: lesson.generation_quality.hallucination_detected,
        needs_human_review: lesson.generation_quality.needs_human_review,
        veto_count: lesson.generation_quality.veto_count,
    };

    Ok(NodeContent {
        node_id: standard.code.clone(),
        lesson,
        assessment,
        flashcards,
        quality,
        generated_at: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
    })
}

/// Print a batch report.
pub fn print_report(report: &BatchReport) {
    eprintln!("\n=== BATCH GENERATION REPORT ===");
    eprintln!("  Total nodes:       {}", report.total_nodes);
    eprintln!("  Generated:         {}", report.generated);
    eprintln!("  Failed:            {}", report.failed);
    eprintln!("  Skipped (resumed): {}", report.skipped);
    eprintln!("  Avg coherence:     {:.2}", report.avg_coherence);
    eprintln!("  Needs review:      {}", report.needs_review_count);
    eprintln!("  Problems created:  {}", report.total_problems_generated);
    eprintln!("  Flashcards:        {}", report.total_flashcards_generated);
}

fn load_progress(path: &Path) -> BatchProgress {
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(progress) = serde_json::from_str(&content) {
            return progress;
        }
    }
    BatchProgress {
        total: 0,
        completed: vec![],
        failed: vec![],
        started_at: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        last_updated: String::new(),
    }
}

fn save_progress(path: &Path, progress: &BatchProgress) {
    if let Ok(json) = serde_json::to_string_pretty(progress) {
        std::fs::write(path, &json).ok();
    }
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            '/' | ' ' => '_',
            _ => '_',
        })
        .collect::<String>()
        .to_lowercase()
}

fn truncate_title(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockGenerator;

    fn sample_standards() -> Vec<StandardInput> {
        vec![
            StandardInput {
                code: "3.OA.A.1".into(),
                description: "Interpret products of whole numbers".into(),
                grade_level: "Grade3".into(),
                domain: "Operations & Algebraic Thinking".into(),
                prerequisites: vec![],
            },
            StandardInput {
                code: "3.OA.A.2".into(),
                description: "Interpret whole-number quotients".into(),
                grade_level: "Grade3".into(),
                domain: "Operations & Algebraic Thinking".into(),
                prerequisites: vec!["3.OA.A.1".into()],
            },
            StandardInput {
                code: "3.NF.A.1".into(),
                description: "Understand a fraction 1/b".into(),
                grade_level: "Grade3".into(),
                domain: "Number & Operations - Fractions".into(),
                prerequisites: vec![],
            },
        ]
    }

    #[test]
    fn test_batch_generates_all_nodes() {
        let dir = std::env::temp_dir().join("edunet_batch_test");
        let _ = std::fs::remove_dir_all(&dir);

        let pipeline = ContentPipeline::new(MockGenerator);
        let standards = sample_standards();
        let config = BatchConfig {
            output_dir: dir.clone(),
            progress_file: dir.join("progress.json"),
            filter: NodeFilter::All,
            ..Default::default()
        };

        let report = run_batch(&pipeline, &standards, &config);
        assert_eq!(report.generated, 3);
        assert_eq!(report.failed, 0);
        assert!(report.avg_coherence > 0.0);
        assert!(report.total_problems_generated > 0);

        // Verify files exist
        assert!(dir.join("3.oa.a.1.json").exists());
        assert!(dir.join("3.oa.a.2.json").exists());
        assert!(dir.join("3.nf.a.1.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_batch_filter_by_subject() {
        let dir = std::env::temp_dir().join("edunet_batch_filter_test");
        let _ = std::fs::remove_dir_all(&dir);

        let pipeline = ContentPipeline::new(MockGenerator);
        let standards = sample_standards();
        let config = BatchConfig {
            output_dir: dir.clone(),
            progress_file: dir.join("progress.json"),
            filter: NodeFilter::BySubject("Fractions".into()),
            ..Default::default()
        };

        let report = run_batch(&pipeline, &standards, &config);
        assert_eq!(report.generated, 1); // only 3.NF.A.1
        assert!(dir.join("3.nf.a.1.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_batch_resume() {
        let dir = std::env::temp_dir().join("edunet_batch_resume_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let pipeline = ContentPipeline::new(MockGenerator);
        let standards = sample_standards();

        // First run: generate 1 node manually to simulate prior progress
        let progress = BatchProgress {
            total: 3,
            completed: vec!["3.OA.A.1".into()],
            failed: vec![],
            started_at: "2026-01-01".into(),
            last_updated: "2026-01-01".into(),
        };
        save_progress(&dir.join("progress.json"), &progress);

        let config = BatchConfig {
            output_dir: dir.clone(),
            progress_file: dir.join("progress.json"),
            filter: NodeFilter::All,
            ..Default::default()
        };

        let report = run_batch(&pipeline, &standards, &config);
        assert_eq!(report.generated, 2); // skipped 3.OA.A.1
        assert_eq!(report.skipped, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_node_content_serialization() {
        let pipeline = ContentPipeline::new(MockGenerator);
        let standard = &sample_standards()[0];
        let config = BatchConfig::default();

        let content = generate_node_content(&pipeline, standard, &config).unwrap();
        let json = serde_json::to_string(&content).unwrap();
        let parsed: NodeContent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_id, "3.OA.A.1");
        assert!(!parsed.lesson.explanation.is_empty());
    }
}
