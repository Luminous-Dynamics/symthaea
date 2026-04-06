// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Context templates for Broca content generation.
//!
//! These functions produce the context string that is passed alongside
//! [`ContentChannels`](crate::channels::ContentChannels) to the content
//! generator. The context provides subject-matter grounding so Broca
//! (or the mock generator) knows *what* to generate.

use crate::pipeline::StandardInput;

/// Context for generating the main lesson explanation.
pub fn lesson_context(standard: &StandardInput) -> String {
    let prereqs = if standard.prerequisites.is_empty() {
        "basic number sense".to_string()
    } else {
        standard.prerequisites.join(", ")
    };

    format!(
        "You are teaching {domain} to a {grade} student.\n\
         Standard: {code} - {description}\n\
         Prerequisites the student already knows: {prereqs}\n\
         Explain this concept clearly with concrete, real-world examples.\n\
         Use simple language appropriate for this grade level.\n\
         Include visual descriptions where helpful.",
        domain = standard.domain,
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
        prereqs = prereqs,
    )
}

/// Context for generating a worked example.
///
/// `index` is 0-based; each example should illustrate a different facet.
pub fn worked_example_context(standard: &StandardInput, index: usize) -> String {
    let variation = match index {
        0 => "Use a simple, concrete scenario (e.g., objects in groups).",
        1 => "Use a different representation (e.g., an array or number line).",
        2 => "Use a real-world word problem.",
        _ => "Use a creative scenario the student might find fun.",
    };

    format!(
        "Create a step-by-step worked example for {grade} students.\n\
         Standard: {code} - {description}\n\
         Example #{num}: {variation}\n\
         Show each step clearly. State the final answer.",
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
        num = index + 1,
        variation = variation,
    )
}

/// Context for generating a practice problem.
///
/// `difficulty` is 0-1000 permille; the generator should produce a harder
/// problem at higher values.
pub fn practice_problem_context(standard: &StandardInput, difficulty: u16) -> String {
    let level_desc = match difficulty {
        0..=300 => "easy (single-step, small numbers)",
        301..=600 => "medium (may require two steps or larger numbers)",
        601..=800 => "challenging (multi-step or requires reasoning)",
        _ => "advanced (complex scenario, requires transfer)",
    };

    format!(
        "Create a {level} practice problem for a {grade} student.\n\
         Standard: {code} - {description}\n\
         Include the correct answer, an explanation of why it is correct,\n\
         and 3 plausible but incorrect distractors for multiple-choice format.\n\
         Also provide 2 progressive hints.",
        level = level_desc,
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
    )
}

/// Context for generating a hint at a given specificity level.
///
/// Level 1 = vague nudge, level 2 = moderate guidance, level 3 = nearly
/// reveals the answer.
pub fn hint_context(standard: &StandardInput, level: u8) -> String {
    let specificity = match level {
        1 => "Give a vague nudge that points the student in the right direction \
              without revealing the method. Ask a guiding question.",
        2 => "Give moderate guidance. Name the strategy or representation that \
              would help, but don't show the answer.",
        _ => "Give a very specific hint that nearly reveals the answer. \
              Walk through the first step or provide the setup.",
    };

    format!(
        "Provide a level-{level} hint for a {grade} student working on:\n\
         Standard: {code} - {description}\n\
         {specificity}",
        level = level,
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
        specificity = specificity,
    )
}

/// Context for identifying and correcting a misconception.
pub fn misconception_context(standard: &StandardInput) -> String {
    format!(
        "Identify a common misconception that {grade} students have about:\n\
         Standard: {code} - {description}\n\
         Explain: (1) what students incorrectly believe, (2) the correct understanding, \
         and (3) why students develop this misconception.\n\
         Be empathetic and constructive.",
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
    )
}

/// Context for generating an assessment item at a specific Bloom's level.
pub fn assessment_item_context(standard: &StandardInput, bloom: &str) -> String {
    let bloom_desc = match bloom {
        "Remember" => "recall a fact or definition",
        "Understand" => "explain or interpret the concept",
        "Apply" => "use the concept to solve a new problem",
        "Analyze" => "break down a problem and identify its parts",
        "Evaluate" => "judge or compare different approaches",
        "Create" => "design an original problem or solution",
        _ => "demonstrate understanding",
    };

    format!(
        "Create an assessment question for a {grade} student that requires them to {bloom_desc}.\n\
         Standard: {code} - {description}\n\
         Bloom's level: {bloom}\n\
         Provide the correct answer, point value, and an optional hint.",
        grade = standard.grade_level,
        bloom_desc = bloom_desc,
        code = standard.code,
        description = standard.description,
        bloom = bloom,
    )
}

/// Context for generating a flashcard.
pub fn flashcard_context(standard: &StandardInput) -> String {
    format!(
        "Create a flashcard for a {grade} student studying:\n\
         Standard: {code} - {description}\n\
         The front should be a clear question or prompt.\n\
         The back should be a concise, correct answer.\n\
         Keep both sides brief (1-2 sentences max).",
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
    )
}

/// Context for generating vocabulary terms.
pub fn vocabulary_context(standard: &StandardInput) -> String {
    format!(
        "List the key vocabulary terms a {grade} student needs for:\n\
         Standard: {code} - {description}\n\
         For each term, provide a grade-appropriate definition \
         and an example of how to use it in a sentence.",
        grade = standard.grade_level,
        code = standard.code,
        description = standard.description,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_standard() -> StandardInput {
        StandardInput {
            code: "3.OA.A.1".to_string(),
            description: "Interpret products of whole numbers".to_string(),
            grade_level: "Grade3".to_string(),
            domain: "Operations & Algebraic Thinking".to_string(),
            prerequisites: vec!["Addition within 100".to_string(), "Skip counting".to_string()],
        }
    }

    #[test]
    fn test_lesson_context_contains_standard() {
        let ctx = lesson_context(&sample_standard());
        assert!(ctx.contains("3.OA.A.1"));
        assert!(ctx.contains("Grade3"));
        assert!(ctx.contains("Operations & Algebraic Thinking"));
        assert!(ctx.contains("Addition within 100"));
    }

    #[test]
    fn test_lesson_context_empty_prereqs() {
        let mut s = sample_standard();
        s.prerequisites.clear();
        let ctx = lesson_context(&s);
        assert!(ctx.contains("basic number sense"));
    }

    #[test]
    fn test_worked_example_variations() {
        let s = sample_standard();
        let ctx0 = worked_example_context(&s, 0);
        let ctx1 = worked_example_context(&s, 1);
        let ctx2 = worked_example_context(&s, 2);
        // Each should have different instruction
        assert!(ctx0.contains("concrete scenario"));
        assert!(ctx1.contains("different representation"));
        assert!(ctx2.contains("real-world word problem"));
    }

    #[test]
    fn test_practice_problem_difficulty_levels() {
        let s = sample_standard();
        let easy = practice_problem_context(&s, 200);
        let hard = practice_problem_context(&s, 700);
        assert!(easy.contains("easy"));
        assert!(hard.contains("challenging"));
    }

    #[test]
    fn test_hint_levels() {
        let s = sample_standard();
        let h1 = hint_context(&s, 1);
        let h3 = hint_context(&s, 3);
        assert!(h1.contains("vague nudge"));
        assert!(h3.contains("very specific"));
    }

    #[test]
    fn test_assessment_bloom_levels() {
        let s = sample_standard();
        let remember = assessment_item_context(&s, "Remember");
        let create = assessment_item_context(&s, "Create");
        assert!(remember.contains("recall"));
        assert!(create.contains("design"));
    }

    #[test]
    fn test_flashcard_context_fields() {
        let s = sample_standard();
        let ctx = flashcard_context(&s);
        assert!(ctx.contains("3.OA.A.1"));
        assert!(ctx.contains("flashcard"));
    }

    #[test]
    fn test_vocabulary_context_fields() {
        let s = sample_standard();
        let ctx = vocabulary_context(&s);
        assert!(ctx.contains("vocabulary"));
        assert!(ctx.contains("definition"));
    }

    #[test]
    fn test_misconception_context_fields() {
        let s = sample_standard();
        let ctx = misconception_context(&s);
        assert!(ctx.contains("misconception"));
        assert!(ctx.contains("empathetic"));
    }
}
