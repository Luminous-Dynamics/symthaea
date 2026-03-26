// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content types produced by the generation pipeline.

use serde::{Deserialize, Serialize};

/// A complete lesson generated for a standard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedLesson {
    /// Standard code (e.g., "3.OA.A.1")
    pub standard_code: String,
    /// Lesson title
    pub title: String,
    /// Target grade level (e.g., "Grade3")
    pub grade_level: String,
    /// Bloom's taxonomy level targeted
    pub bloom_level: String,
    /// Main teaching text explaining the concept
    pub explanation: String,
    /// Step-by-step worked examples
    pub examples: Vec<WorkedExample>,
    /// Practice problems for the student
    pub practice_problems: Vec<PracticeProblem>,
    /// Progressive hints (vague to specific)
    pub hints: Vec<String>,
    /// Key vocabulary terms with definitions
    pub key_vocabulary: Vec<VocabularyTerm>,
    /// Common misconceptions and corrections
    pub misconceptions: Vec<Misconception>,
    /// Quality metrics from the generation process
    pub generation_quality: GenerationQuality,
}

/// A worked example with step-by-step solution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkedExample {
    /// The problem statement
    pub problem: String,
    /// Ordered solution steps
    pub steps: Vec<SolutionStep>,
    /// Final answer
    pub answer: String,
    /// Optional description of a diagram or visual aid
    pub visual_description: Option<String>,
}

/// A single step in a worked example solution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolutionStep {
    /// What to do in this step
    pub instruction: String,
    /// The result after this step
    pub result: String,
}

/// A practice problem for the student.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PracticeProblem {
    /// The question text
    pub question: String,
    /// The correct answer
    pub answer: String,
    /// Difficulty as permille (0-1000), matching EduNet's convention
    pub difficulty_permille: u16,
    /// Bloom's taxonomy level
    pub bloom_level: String,
    /// Progressive hints (from vague to specific)
    pub hints: Vec<String>,
    /// Explanation of why the answer is correct
    pub explanation: String,
    /// Wrong answers for multiple-choice format
    pub distractors: Vec<String>,
}

/// Vocabulary term with definition and example.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VocabularyTerm {
    /// The term
    pub term: String,
    /// Clear definition appropriate for grade level
    pub definition: String,
    /// Example usage in context
    pub example_usage: String,
}

/// Common misconception and how to address it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Misconception {
    /// What students commonly think (incorrectly)
    pub misconception: String,
    /// The correct understanding
    pub correction: String,
    /// Why students develop this misconception
    pub why_students_think_this: String,
}

/// Quality metrics from the generation process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationQuality {
    /// Coherence score (0.0-1.0)
    pub coherence: f32,
    /// Whether hallucination was detected in any output
    pub hallucination_detected: bool,
    /// Number of self-corrections (veto events) during generation
    pub veto_count: u32,
    /// Overall epistemic confidence (0.0-1.0)
    pub epistemic_confidence: f32,
    /// True if quality is below threshold and needs human review
    pub needs_human_review: bool,
}

/// An assessment generated for a standard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedAssessment {
    /// Standard code
    pub standard_code: String,
    /// Assessment title
    pub title: String,
    /// Assessment items (questions)
    pub items: Vec<AssessmentItem>,
    /// Total points available
    pub total_points: u32,
    /// Estimated completion time in minutes
    pub estimated_minutes: u32,
    /// Distribution of Bloom's levels: (level_name, count)
    pub bloom_distribution: Vec<(String, u32)>,
}

/// A single assessment question.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssessmentItem {
    /// The question text
    pub question: String,
    /// Type of question
    pub item_type: AssessmentItemType,
    /// The correct answer
    pub correct_answer: String,
    /// Points for this item
    pub points: u32,
    /// Bloom's taxonomy level
    pub bloom_level: String,
    /// Rubric for open-ended items
    pub rubric: Option<String>,
    /// Optional hint
    pub hint: Option<String>,
}

/// Types of assessment questions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AssessmentItemType {
    /// Multiple choice with a list of options
    MultipleChoice { options: Vec<String> },
    /// Short text answer
    ShortAnswer,
    /// Open-ended response with rubric criteria
    OpenEnded { rubric_criteria: Vec<String> },
    /// True or false
    TrueFalse,
    /// Fill in the blank with a template containing `___`
    FillInBlank { template: String },
}

/// SRS flashcard generated from lesson content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedFlashcard {
    /// Front of card (question/prompt)
    pub front: String,
    /// Back of card (answer)
    pub back: String,
    /// Tags for deck organization
    pub tags: Vec<String>,
    /// Difficulty as permille (0-1000)
    pub difficulty_permille: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generated_lesson_roundtrip() {
        let lesson = GeneratedLesson {
            standard_code: "3.OA.A.1".to_string(),
            title: "Interpret Products".to_string(),
            grade_level: "Grade3".to_string(),
            bloom_level: "Understand".to_string(),
            explanation: "Multiplication means equal groups.".to_string(),
            examples: vec![WorkedExample {
                problem: "What is 3 x 4?".to_string(),
                steps: vec![SolutionStep {
                    instruction: "Draw 3 groups of 4".to_string(),
                    result: "3 groups with 4 in each".to_string(),
                }],
                answer: "12".to_string(),
                visual_description: Some("3 rows of 4 dots".to_string()),
            }],
            practice_problems: vec![],
            hints: vec!["Think about equal groups.".to_string()],
            key_vocabulary: vec![VocabularyTerm {
                term: "product".to_string(),
                definition: "The result of multiplication.".to_string(),
                example_usage: "The product of 3 and 4 is 12.".to_string(),
            }],
            misconceptions: vec![],
            generation_quality: GenerationQuality {
                coherence: 0.9,
                hallucination_detected: false,
                veto_count: 0,
                epistemic_confidence: 0.95,
                needs_human_review: false,
            },
        };

        let json = serde_json::to_string(&lesson).unwrap();
        let round: GeneratedLesson = serde_json::from_str(&json).unwrap();
        assert_eq!(round.standard_code, "3.OA.A.1");
        assert_eq!(round.examples.len(), 1);
        assert_eq!(round.examples[0].steps.len(), 1);
        assert_eq!(round.key_vocabulary[0].term, "product");
    }

    #[test]
    fn test_assessment_roundtrip() {
        let assessment = GeneratedAssessment {
            standard_code: "3.OA.A.1".to_string(),
            title: "Multiplication Quiz".to_string(),
            items: vec![
                AssessmentItem {
                    question: "What is 5 x 3?".to_string(),
                    item_type: AssessmentItemType::MultipleChoice {
                        options: vec![
                            "8".to_string(),
                            "15".to_string(),
                            "12".to_string(),
                            "53".to_string(),
                        ],
                    },
                    correct_answer: "15".to_string(),
                    points: 1,
                    bloom_level: "Remember".to_string(),
                    rubric: None,
                    hint: Some("Think: 5 groups of 3".to_string()),
                },
                AssessmentItem {
                    question: "Is 4 x 6 the same as 6 x 4?".to_string(),
                    item_type: AssessmentItemType::TrueFalse,
                    correct_answer: "True".to_string(),
                    points: 1,
                    bloom_level: "Understand".to_string(),
                    rubric: None,
                    hint: None,
                },
            ],
            total_points: 2,
            estimated_minutes: 5,
            bloom_distribution: vec![
                ("Remember".to_string(), 1),
                ("Understand".to_string(), 1),
            ],
        };

        let json = serde_json::to_string(&assessment).unwrap();
        let round: GeneratedAssessment = serde_json::from_str(&json).unwrap();
        assert_eq!(round.items.len(), 2);
        assert_eq!(round.total_points, 2);
    }

    #[test]
    fn test_flashcard_roundtrip() {
        let card = GeneratedFlashcard {
            front: "What does 4 x 5 mean?".to_string(),
            back: "4 groups of 5 objects, which equals 20.".to_string(),
            tags: vec!["multiplication".to_string(), "grade3".to_string()],
            difficulty_permille: 300,
        };

        let json = serde_json::to_string(&card).unwrap();
        let round: GeneratedFlashcard = serde_json::from_str(&json).unwrap();
        assert_eq!(round.front, "What does 4 x 5 mean?");
        assert_eq!(round.difficulty_permille, 300);
        assert_eq!(round.tags.len(), 2);
    }

    #[test]
    fn test_assessment_item_types_roundtrip() {
        let items = vec![
            AssessmentItemType::MultipleChoice {
                options: vec!["a".to_string(), "b".to_string()],
            },
            AssessmentItemType::ShortAnswer,
            AssessmentItemType::OpenEnded {
                rubric_criteria: vec!["Correctness".to_string(), "Explanation".to_string()],
            },
            AssessmentItemType::TrueFalse,
            AssessmentItemType::FillInBlank {
                template: "3 x ___ = 12".to_string(),
            },
        ];

        for item in &items {
            let json = serde_json::to_string(item).unwrap();
            let _round: AssessmentItemType = serde_json::from_str(&json).unwrap();
        }
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn test_generation_quality_flags() {
        let good = GenerationQuality {
            coherence: 0.85,
            hallucination_detected: false,
            veto_count: 0,
            epistemic_confidence: 0.9,
            needs_human_review: false,
        };
        assert!(!good.needs_human_review);
        assert!(!good.hallucination_detected);

        let bad = GenerationQuality {
            coherence: 0.3,
            hallucination_detected: true,
            veto_count: 2,
            epistemic_confidence: 0.4,
            needs_human_review: true,
        };
        assert!(bad.needs_human_review);
        assert!(bad.hallucination_detected);
    }
}
