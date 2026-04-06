// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content generation pipeline orchestration.
//!
//! The [`ContentPipeline`] takes a [`StandardInput`] and produces complete
//! teaching materials (lessons, assessments, flashcards) by making multiple
//! calls to a [`ContentGenerator`] with appropriate [`ContentChannels`].

use serde::{Deserialize, Serialize};

use crate::channels::ContentChannels;
use crate::templates;
use crate::types::*;

/// Trait for content generators.
///
/// Implementations convert [`ContentChannels`] + context into generated text.
/// The real implementation maps channels to Broca's 43-channel ThoughtChannels;
/// for development, use [`MockGenerator`](crate::mock::MockGenerator).
pub trait ContentGenerator {
    /// Generate text given channel configuration, context prompt, and token budget.
    fn generate(&self, channels: &ContentChannels, context: &str, max_tokens: usize) -> GenerationOutput;
}

/// Output from a single generation call.
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    /// Generated text
    pub text: String,
    /// Coherence score (0.0-1.0)
    pub coherence: f32,
    /// Whether hallucination was detected
    pub hallucination_flag: bool,
    /// Number of self-correction veto events
    pub veto_count: u32,
}

/// Input describing a standard to generate content for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardInput {
    /// Standard code (e.g., "3.OA.A.1" or "CCSS.MATH.CONTENT.3.OA.A.1")
    pub code: String,
    /// Human-readable description
    pub description: String,
    /// Target grade level (e.g., "Grade3")
    pub grade_level: String,
    /// Subject domain (e.g., "Operations & Algebraic Thinking")
    pub domain: String,
    /// What the student already knows
    pub prerequisites: Vec<String>,
}

/// Errors from the content generation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ContentGenError {
    /// All retry attempts produced content below the quality threshold.
    #[error("content quality below threshold after {0} attempts")]
    QualityBelowThreshold(u32),
    /// The generator returned empty text.
    #[error("generator returned empty text")]
    EmptyGeneration,
}

/// Content generation pipeline.
///
/// Orchestrates multiple generation calls to produce complete teaching
/// materials from a single standard definition.
pub struct ContentPipeline<G: ContentGenerator> {
    generator: G,
    /// Minimum coherence to accept (default: 0.6)
    quality_threshold: f32,
    /// Maximum retry attempts on low quality (default: 3)
    max_retries: u32,
}

impl<G: ContentGenerator> ContentPipeline<G> {
    /// Create a new pipeline with default quality settings.
    pub fn new(generator: G) -> Self {
        Self {
            generator,
            quality_threshold: 0.6,
            max_retries: 3,
        }
    }

    /// Set the minimum coherence threshold for accepting generated content.
    pub fn with_quality_threshold(mut self, threshold: f32) -> Self {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum retry count.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Parse grade ordinal from grade_level string (e.g., "Grade3" -> 4).
    fn grade_ordinal(grade_level: &str) -> u8 {
        match grade_level {
            "PreK" => 0,
            "Kindergarten" => 1,
            s if s.starts_with("Grade") => {
                s[5..].parse::<u8>().unwrap_or(6) + 1
            }
            "College" => 14,
            "Undergraduate" => 15,
            "Graduate" => 16,
            "Doctoral" => 17,
            "PostDoctoral" => 18,
            "Professional" => 19,
            "Adult" => 20,
            _ => 8, // default to middle school
        }
    }

    /// Generate a complete lesson for a standard.
    pub fn generate_lesson(&self, standard: &StandardInput) -> Result<GeneratedLesson, ContentGenError> {
        let grade_ord = Self::grade_ordinal(&standard.grade_level);

        // 1. Main explanation
        let explanation_output = self.generate_with_retry(
            &ContentChannels::teaching_factual().for_grade(grade_ord),
            &templates::lesson_context(standard),
            512,
        )?;

        // 2. Worked examples (3)
        let examples = (0..3)
            .map(|i| self.generate_worked_example(standard, grade_ord, i))
            .collect::<Result<Vec<_>, _>>()?;

        // 3. Practice problems (7, spread across difficulty levels)
        let problems = (0..7)
            .map(|i| {
                let difficulty = ((i as u16 + 1) * 1000) / 8; // ~125, 250, ..., 875
                self.generate_practice_problem(standard, grade_ord, difficulty)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 4. Progressive hints (3 levels)
        let hints = (1..=3u8)
            .map(|level| self.generate_hint(standard, grade_ord, level))
            .collect::<Result<Vec<_>, _>>()?;

        // 5. Vocabulary
        let vocab_output = self.generate_with_retry(
            &ContentChannels::teaching_factual().for_grade(grade_ord),
            &templates::vocabulary_context(standard),
            256,
        )?;
        let key_vocabulary = parse_vocabulary(&vocab_output.text);

        // 6. Misconceptions
        let misconception_output = self.generate_with_retry(
            &ContentChannels::misconception_correction().for_grade(grade_ord),
            &templates::misconception_context(standard),
            256,
        )?;
        let misconceptions = parse_misconceptions(&misconception_output.text);

        // 7. Compute aggregate quality
        let all_coherences = [
            explanation_output.coherence,
            vocab_output.coherence,
            misconception_output.coherence,
        ];
        let avg_coherence = all_coherences.iter().sum::<f32>() / all_coherences.len() as f32;
        let any_hallucination = explanation_output.hallucination_flag
            || vocab_output.hallucination_flag
            || misconception_output.hallucination_flag;
        let total_vetos = explanation_output.veto_count
            + vocab_output.veto_count
            + misconception_output.veto_count;

        // Determine Bloom's level from the standard code heuristic
        let bloom_level = infer_bloom_level(&standard.code, &standard.description);

        Ok(GeneratedLesson {
            standard_code: standard.code.clone(),
            title: format!("{}: {}", standard.code, truncate(&standard.description, 60)),
            grade_level: standard.grade_level.clone(),
            bloom_level,
            explanation: explanation_output.text,
            examples,
            practice_problems: problems,
            hints,
            key_vocabulary,
            misconceptions,
            generation_quality: GenerationQuality {
                coherence: avg_coherence,
                hallucination_detected: any_hallucination,
                veto_count: total_vetos,
                epistemic_confidence: avg_coherence, // proxy
                needs_human_review: avg_coherence < self.quality_threshold || any_hallucination,
            },
        })
    }

    /// Generate an assessment for a standard.
    pub fn generate_assessment(
        &self,
        standard: &StandardInput,
        num_items: usize,
    ) -> Result<GeneratedAssessment, ContentGenError> {
        let grade_ord = Self::grade_ordinal(&standard.grade_level);
        let blooms = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];

        let items: Vec<AssessmentItem> = (0..num_items)
            .map(|i| {
                let bloom = blooms[i % blooms.len()];
                self.generate_assessment_item(standard, grade_ord, bloom)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let total_points: u32 = items.iter().map(|it| it.points).sum();

        // Count Bloom's distribution
        let mut bloom_counts = std::collections::HashMap::new();
        for item in &items {
            *bloom_counts.entry(item.bloom_level.clone()).or_insert(0u32) += 1;
        }
        let bloom_distribution: Vec<(String, u32)> = bloom_counts.into_iter().collect();

        Ok(GeneratedAssessment {
            standard_code: standard.code.clone(),
            title: format!("Assessment: {}", standard.description),
            items,
            total_points,
            estimated_minutes: (num_items as u32) * 3, // ~3 min per item
            bloom_distribution,
        })
    }

    /// Generate flashcards for a standard.
    pub fn generate_flashcards(
        &self,
        standard: &StandardInput,
        num_cards: usize,
    ) -> Result<Vec<GeneratedFlashcard>, ContentGenError> {
        let grade_ord = Self::grade_ordinal(&standard.grade_level);

        (0..num_cards)
            .map(|i| {
                let difficulty = ((i as u16 + 1) * 1000) / (num_cards as u16 + 1);
                let output = self.generate_with_retry(
                    &ContentChannels::practice_problem().for_grade(grade_ord),
                    &templates::flashcard_context(standard),
                    128,
                )?;

                let (front, back) = split_flashcard(&output.text);
                Ok(GeneratedFlashcard {
                    front,
                    back,
                    tags: vec![
                        standard.code.clone(),
                        standard.grade_level.clone(),
                        standard.domain.clone(),
                    ],
                    difficulty_permille: difficulty,
                })
            })
            .collect()
    }

    /// Generate content for all standards in a curriculum.
    pub fn generate_curriculum(
        &self,
        standards: &[StandardInput],
    ) -> Result<Vec<GeneratedLesson>, ContentGenError> {
        standards.iter().map(|s| self.generate_lesson(s)).collect()
    }

    // ---- Internal helpers ----

    fn generate_with_retry(
        &self,
        channels: &ContentChannels,
        context: &str,
        max_tokens: usize,
    ) -> Result<GenerationOutput, ContentGenError> {
        for _attempt in 0..self.max_retries {
            let output = self.generator.generate(channels, context, max_tokens);
            if output.text.is_empty() {
                continue;
            }
            if output.coherence >= self.quality_threshold && !output.hallucination_flag {
                return Ok(output);
            }
        }
        Err(ContentGenError::QualityBelowThreshold(self.max_retries))
    }

    fn generate_worked_example(
        &self,
        standard: &StandardInput,
        grade_ord: u8,
        index: usize,
    ) -> Result<WorkedExample, ContentGenError> {
        let output = self.generate_with_retry(
            &ContentChannels::worked_example().for_grade(grade_ord),
            &templates::worked_example_context(standard, index),
            384,
        )?;

        Ok(parse_worked_example(&output.text))
    }

    fn generate_practice_problem(
        &self,
        standard: &StandardInput,
        grade_ord: u8,
        difficulty: u16,
    ) -> Result<PracticeProblem, ContentGenError> {
        let output = self.generate_with_retry(
            &ContentChannels::practice_problem().for_grade(grade_ord),
            &templates::practice_problem_context(standard, difficulty),
            256,
        )?;

        Ok(parse_practice_problem(&output.text, difficulty))
    }

    fn generate_hint(
        &self,
        standard: &StandardInput,
        grade_ord: u8,
        level: u8,
    ) -> Result<String, ContentGenError> {
        let output = self.generate_with_retry(
            &ContentChannels::hint(level).for_grade(grade_ord),
            &templates::hint_context(standard, level),
            128,
        )?;
        Ok(output.text)
    }

    fn generate_assessment_item(
        &self,
        standard: &StandardInput,
        grade_ord: u8,
        bloom: &str,
    ) -> Result<AssessmentItem, ContentGenError> {
        let output = self.generate_with_retry(
            &ContentChannels::practice_problem().for_grade(grade_ord),
            &templates::assessment_item_context(standard, bloom),
            256,
        )?;

        Ok(parse_assessment_item(&output.text, bloom))
    }
}

// ---- Parsing helpers ----

/// Parse vocabulary from generated text. In mock mode, text is pre-formatted
/// as "term: definition | example" lines. With real Broca, a more sophisticated
/// parser would be needed.
fn parse_vocabulary(text: &str) -> Vec<VocabularyTerm> {
    text.lines()
        .filter(|l| l.contains(':'))
        .filter_map(|line| {
            let mut parts = line.splitn(2, ':');
            let term = parts.next()?.trim().to_string();
            let rest = parts.next()?.trim();

            let (definition, example) = if rest.contains('|') {
                let mut sub = rest.splitn(2, '|');
                (
                    sub.next().unwrap_or("").trim().to_string(),
                    sub.next().unwrap_or("").trim().to_string(),
                )
            } else {
                (rest.to_string(), String::new())
            };

            if term.is_empty() {
                return None;
            }

            Some(VocabularyTerm {
                term,
                definition,
                example_usage: example,
            })
        })
        .collect()
}

/// Parse a misconception block. Mock format: "WRONG: ... | RIGHT: ... | WHY: ..."
fn parse_misconceptions(text: &str) -> Vec<Misconception> {
    // Simple approach: split on double newlines for multiple misconceptions
    let blocks: Vec<&str> = if text.contains("\n\n") {
        text.split("\n\n").collect()
    } else {
        vec![text]
    };

    blocks
        .iter()
        .filter(|b| !b.trim().is_empty())
        .map(|block| {
            let mut misconception = String::new();
            let mut correction = String::new();
            let mut why = String::new();

            for line in block.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("WRONG:") {
                    misconception = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("RIGHT:") {
                    correction = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("WHY:") {
                    why = rest.trim().to_string();
                } else if misconception.is_empty() {
                    misconception = line.to_string();
                }
            }

            Misconception {
                misconception,
                correction,
                why_students_think_this: why,
            }
        })
        .collect()
}

/// Parse a worked example from generated text.
fn parse_worked_example(text: &str) -> WorkedExample {
    let lines: Vec<&str> = text.lines().collect();

    let problem = lines.first().unwrap_or(&"").to_string();
    let answer = lines.last().unwrap_or(&"").to_string();

    let steps: Vec<SolutionStep> = lines
        .iter()
        .skip(1)
        .take(lines.len().saturating_sub(2))
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.splitn(2, "->").collect();
            if parts.len() == 2 {
                SolutionStep {
                    instruction: parts[0].trim().to_string(),
                    result: parts[1].trim().to_string(),
                }
            } else {
                SolutionStep {
                    instruction: line.trim().to_string(),
                    result: String::new(),
                }
            }
        })
        .collect();

    WorkedExample {
        problem,
        steps,
        answer,
        visual_description: None,
    }
}

/// Parse a practice problem from generated text.
fn parse_practice_problem(text: &str, difficulty: u16) -> PracticeProblem {
    let lines: Vec<&str> = text.lines().collect();

    PracticeProblem {
        question: lines.first().unwrap_or(&"").to_string(),
        answer: lines.get(1).unwrap_or(&"").to_string(),
        difficulty_permille: difficulty,
        bloom_level: "Apply".to_string(),
        hints: lines.iter().skip(4).map(|l| l.to_string()).collect(),
        explanation: lines.get(2).unwrap_or(&"").to_string(),
        distractors: lines.get(3).map(|l| {
            l.split(',').map(|s| s.trim().to_string()).collect()
        }).unwrap_or_default(),
    }
}

/// Parse an assessment item from generated text.
fn parse_assessment_item(text: &str, bloom: &str) -> AssessmentItem {
    let lines: Vec<&str> = text.lines().collect();
    let question = lines.first().unwrap_or(&"").to_string();
    let answer = lines.get(1).unwrap_or(&"").to_string();

    AssessmentItem {
        question,
        item_type: AssessmentItemType::ShortAnswer,
        correct_answer: answer,
        points: match bloom {
            "Remember" => 1,
            "Understand" => 2,
            "Apply" => 2,
            "Analyze" => 3,
            "Evaluate" => 3,
            "Create" => 4,
            _ => 1,
        },
        bloom_level: bloom.to_string(),
        rubric: None,
        hint: lines.get(2).map(|s| s.to_string()),
    }
}

/// Split flashcard text into front/back (separated by " | " or newline).
fn split_flashcard(text: &str) -> (String, String) {
    if text.contains(" | ") {
        let mut parts = text.splitn(2, " | ");
        (
            parts.next().unwrap_or("").to_string(),
            parts.next().unwrap_or("").to_string(),
        )
    } else {
        let lines: Vec<&str> = text.lines().collect();
        (
            lines.first().unwrap_or(&"").to_string(),
            lines.get(1).unwrap_or(&"").to_string(),
        )
    }
}

/// Infer Bloom's level from standard code and description.
fn infer_bloom_level(code: &str, description: &str) -> String {
    let desc_lower = description.to_lowercase();
    if desc_lower.contains("create") || desc_lower.contains("design") || desc_lower.contains("construct") {
        "Create".to_string()
    } else if desc_lower.contains("evaluate") || desc_lower.contains("judge") || desc_lower.contains("justify") {
        "Evaluate".to_string()
    } else if desc_lower.contains("analyze") || desc_lower.contains("compare") || desc_lower.contains("distinguish") {
        "Analyze".to_string()
    } else if desc_lower.contains("solve") || desc_lower.contains("apply") || desc_lower.contains("use")
        || desc_lower.contains("determine") || desc_lower.contains("multiply") || desc_lower.contains("divide")
    {
        "Apply".to_string()
    } else if desc_lower.contains("interpret") || desc_lower.contains("explain") || desc_lower.contains("understand")
        || desc_lower.contains("describe") || desc_lower.contains("represent")
    {
        "Understand".to_string()
    } else if code.contains(".7") || desc_lower.contains("fluently") || desc_lower.contains("know") {
        "Remember".to_string()
    } else {
        "Understand".to_string()
    }
}

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockGenerator;

    fn sample_standard() -> StandardInput {
        StandardInput {
            code: "3.OA.A.1".to_string(),
            description: "Interpret products of whole numbers".to_string(),
            grade_level: "Grade3".to_string(),
            domain: "Operations & Algebraic Thinking".to_string(),
            prerequisites: vec!["Addition within 100".to_string()],
        }
    }

    #[test]
    fn test_generate_lesson_full() {
        let pipeline = ContentPipeline::new(MockGenerator);
        let lesson = pipeline.generate_lesson(&sample_standard()).unwrap();

        assert_eq!(lesson.standard_code, "3.OA.A.1");
        assert!(!lesson.explanation.is_empty());
        assert_eq!(lesson.examples.len(), 3);
        assert_eq!(lesson.practice_problems.len(), 7);
        assert_eq!(lesson.hints.len(), 3);
        assert!(!lesson.key_vocabulary.is_empty());
        assert!(!lesson.misconceptions.is_empty());
        assert!(!lesson.generation_quality.needs_human_review);
        assert!(!lesson.generation_quality.hallucination_detected);
    }

    #[test]
    fn test_generate_assessment() {
        let pipeline = ContentPipeline::new(MockGenerator);
        let assessment = pipeline.generate_assessment(&sample_standard(), 6).unwrap();

        assert_eq!(assessment.standard_code, "3.OA.A.1");
        assert_eq!(assessment.items.len(), 6);
        assert!(assessment.total_points > 0);
        assert!(assessment.estimated_minutes > 0);
        assert!(!assessment.bloom_distribution.is_empty());
    }

    #[test]
    fn test_generate_flashcards() {
        let pipeline = ContentPipeline::new(MockGenerator);
        let cards = pipeline.generate_flashcards(&sample_standard(), 5).unwrap();

        assert_eq!(cards.len(), 5);
        for card in &cards {
            assert!(!card.front.is_empty());
            assert!(!card.back.is_empty());
            assert!(card.difficulty_permille <= 1000);
            assert!(!card.tags.is_empty());
        }
    }

    #[test]
    fn test_generate_curriculum_multiple() {
        let pipeline = ContentPipeline::new(MockGenerator);
        let standards = vec![
            sample_standard(),
            StandardInput {
                code: "3.NF.A.1".to_string(),
                description: "Understand a fraction 1/b as the quantity formed by 1 part".to_string(),
                grade_level: "Grade3".to_string(),
                domain: "Number & Operations - Fractions".to_string(),
                prerequisites: vec!["Equal parts".to_string()],
            },
        ];

        let lessons = pipeline.generate_curriculum(&standards).unwrap();
        assert_eq!(lessons.len(), 2);
        assert_eq!(lessons[0].standard_code, "3.OA.A.1");
        assert_eq!(lessons[1].standard_code, "3.NF.A.1");
    }

    #[test]
    fn test_quality_threshold_rejection() {
        /// Generator that always returns low quality.
        struct LowQualityGen;
        impl ContentGenerator for LowQualityGen {
            fn generate(&self, _: &ContentChannels, _: &str, _: usize) -> GenerationOutput {
                GenerationOutput {
                    text: "bad content".to_string(),
                    coherence: 0.2,
                    hallucination_flag: false,
                    veto_count: 0,
                }
            }
        }

        let pipeline = ContentPipeline::new(LowQualityGen);
        let result = pipeline.generate_lesson(&sample_standard());
        assert!(result.is_err());
    }

    #[test]
    fn test_hallucination_rejection() {
        /// Generator that always flags hallucination.
        struct HallucinatingGen;
        impl ContentGenerator for HallucinatingGen {
            fn generate(&self, _: &ContentChannels, _: &str, _: usize) -> GenerationOutput {
                GenerationOutput {
                    text: "hallucinated content".to_string(),
                    coherence: 0.9,
                    hallucination_flag: true,
                    veto_count: 1,
                }
            }
        }

        let pipeline = ContentPipeline::new(HallucinatingGen);
        let result = pipeline.generate_lesson(&sample_standard());
        assert!(result.is_err());
    }

    #[test]
    fn test_grade_ordinal_parsing() {
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("PreK"), 0);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Kindergarten"), 1);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Grade1"), 2);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Grade3"), 4);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Grade12"), 13);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("College"), 14);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Undergraduate"), 15);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Graduate"), 16);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Doctoral"), 17);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("PostDoctoral"), 18);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Professional"), 19);
        assert_eq!(ContentPipeline::<MockGenerator>::grade_ordinal("Adult"), 20);
    }

    #[test]
    fn test_infer_bloom_level() {
        assert_eq!(infer_bloom_level("3.OA.A.1", "Interpret products"), "Understand");
        assert_eq!(infer_bloom_level("3.OA.A.3", "Solve word problems"), "Apply");
        assert_eq!(infer_bloom_level("3.OA.C.7", "Fluently multiply and divide"), "Apply");
        assert_eq!(infer_bloom_level("3.OA.C.7", "Know from memory all products of two one-digit numbers"), "Remember");
        assert_eq!(infer_bloom_level("3.OA.A.4", "Determine the unknown"), "Apply");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world this is long", 10), "hello w...");
    }

    #[test]
    fn test_parse_vocabulary() {
        let text = "product: The result of multiplication | The product of 3 and 4 is 12.\n\
                    factor: A number being multiplied | 3 and 4 are factors of 12.";
        let vocab = parse_vocabulary(text);
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab[0].term, "product");
        assert!(vocab[0].definition.contains("result of multiplication"));
        assert!(vocab[0].example_usage.contains("product of 3"));
    }

    #[test]
    fn test_parse_misconceptions() {
        let text = "WRONG: Multiplication always makes numbers bigger.\n\
                    RIGHT: Multiplication by 1 gives the same number, and by 0 gives 0.\n\
                    WHY: Students only see multiplication with numbers greater than 1 in early examples.";
        let ms = parse_misconceptions(text);
        assert_eq!(ms.len(), 1);
        assert!(ms[0].misconception.contains("bigger"));
        assert!(ms[0].correction.contains("same number"));
        assert!(ms[0].why_students_think_this.contains("early examples"));
    }

    #[test]
    fn test_split_flashcard() {
        let (f, b) = split_flashcard("What is 3 x 4? | 12, because 3 groups of 4 = 12.");
        assert!(f.contains("3 x 4"));
        assert!(b.contains("12"));

        let (f2, b2) = split_flashcard("What is 3 x 4?\n12");
        assert!(f2.contains("3 x 4"));
        assert_eq!(b2, "12");
    }

    #[test]
    fn test_with_quality_threshold() {
        let pipeline = ContentPipeline::new(MockGenerator)
            .with_quality_threshold(0.9)
            .with_max_retries(5);
        // MockGenerator returns 0.85 coherence, so this should still pass
        // because the mock is designed to be acceptable
        let lesson = pipeline.generate_lesson(&sample_standard());
        // With 0.9 threshold and 0.85 mock coherence, this should fail
        assert!(lesson.is_err());
    }
}
