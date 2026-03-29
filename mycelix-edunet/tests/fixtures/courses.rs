// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Course test fixtures
//
// Generators for course data, modules, learning outcomes, and related structures

use super::{random_hex, random_string, seeded_rng, DEFAULT_SEED};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Course {
    pub course_id: String,
    pub title: String,
    pub description: String,
    pub instructor: String,
    pub tags: Vec<String>,
    pub difficulty: String,
    pub modules: Vec<Module>,
    pub total_hours: u32,
    pub prerequisites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub module_id: String,
    pub title: String,
    pub description: String,
    pub learning_outcomes: Vec<String>,
    pub duration_hours: u32,
}

/// Course builder for flexible test data creation
pub struct CourseBuilder {
    seed: u64,
    module_count: usize,
    difficulty: String,
    tags: Vec<String>,
}

impl Default for CourseBuilder {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            module_count: 5,
            difficulty: "intermediate".to_string(),
            tags: vec!["programming".to_string()],
        }
    }
}

impl CourseBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn module_count(mut self, count: usize) -> Self {
        self.module_count = count;
        self
    }

    pub fn difficulty(mut self, difficulty: impl Into<String>) -> Self {
        self.difficulty = difficulty.into();
        self
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn build(self) -> Course {
        let mut rng = seeded_rng(self.seed);

        let course_id = format!("course-{}", random_hex(&mut rng, 8));
        let modules: Vec<Module> = (0..self.module_count)
            .map(|i| Module {
                module_id: format!("module-{}-{}", i + 1, random_hex(&mut rng, 4)),
                title: format!("Module {}: {}", i + 1, random_course_topic(&mut rng)),
                description: random_description(&mut rng),
                learning_outcomes: (0..3)
                    .map(|_| random_learning_outcome(&mut rng))
                    .collect(),
                duration_hours: rng.gen_range(2..10),
            })
            .collect();

        let total_hours: u32 = modules.iter().map(|m| m.duration_hours).sum();

        Course {
            course_id,
            title: format!("{} Course", random_course_topic(&mut rng)),
            description: random_description(&mut rng),
            instructor: format!("Prof. {}", random_string(&mut rng, 8)),
            tags: self.tags,
            difficulty: self.difficulty,
            modules,
            total_hours,
            prerequisites: vec![],
        }
    }
}

/// Generate a random course with default settings
pub fn random() -> Course {
    CourseBuilder::new().build()
}

/// Generate a course with specific seed for reproducibility
pub fn with_seed(seed: u64) -> Course {
    CourseBuilder::new().seed(seed).build()
}

/// Generate a course with specific number of modules
pub fn with_modules(count: usize) -> Course {
    CourseBuilder::new().module_count(count).build()
}

/// Generate a beginner-level course
pub fn beginner() -> Course {
    CourseBuilder::new()
        .difficulty("beginner")
        .module_count(3)
        .build()
}

/// Generate an advanced-level course
pub fn advanced() -> Course {
    CourseBuilder::new()
        .difficulty("advanced")
        .module_count(8)
        .tags(vec![
            "advanced".to_string(),
            "research".to_string(),
            "theory".to_string(),
        ])
        .build()
}

fn random_course_topic(rng: &mut rand_chacha::ChaCha8Rng) -> String {
    let topics = [
        "Rust Fundamentals",
        "Machine Learning",
        "Web Development",
        "Data Structures",
        "System Design",
        "Cryptography",
        "Distributed Systems",
        "Algorithm Analysis",
        "Quantum Computing",
        "Blockchain Technology",
    ];
    topics[rng.gen_range(0..topics.len())].to_string()
}

fn random_description(rng: &mut rand_chacha::ChaCha8Rng) -> String {
    let descriptions = [
        "A comprehensive introduction to the subject",
        "Learn the fundamentals and advanced concepts",
        "Master the skills needed for professional development",
        "From basics to expert-level understanding",
        "Practical hands-on learning experience",
    ];
    descriptions[rng.gen_range(0..descriptions.len())].to_string()
}

fn random_learning_outcome(rng: &mut rand_chacha::ChaCha8Rng) -> String {
    let outcomes = [
        "Understand core concepts and principles",
        "Apply theoretical knowledge to practical problems",
        "Analyze complex scenarios and propose solutions",
        "Design and implement efficient solutions",
        "Evaluate trade-offs and make informed decisions",
    ];
    outcomes[rng.gen_range(0..outcomes.len())].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_course() {
        let course = random();
        assert!(!course.course_id.is_empty());
        assert!(!course.title.is_empty());
        assert!(!course.modules.is_empty());
    }

    #[test]
    fn test_course_builder() {
        let course = CourseBuilder::new()
            .module_count(3)
            .difficulty("beginner")
            .tags(vec!["test".to_string()])
            .build();

        assert_eq!(course.modules.len(), 3);
        assert_eq!(course.difficulty, "beginner");
        assert_eq!(course.tags, vec!["test".to_string()]);
    }

    #[test]
    fn test_deterministic_generation() {
        let course1 = with_seed(42);
        let course2 = with_seed(42);

        assert_eq!(course1.course_id, course2.course_id);
        assert_eq!(course1.modules.len(), course2.modules.len());
    }

    #[test]
    fn test_beginner_course() {
        let course = beginner();
        assert_eq!(course.difficulty, "beginner");
        assert_eq!(course.modules.len(), 3);
    }

    #[test]
    fn test_advanced_course() {
        let course = advanced();
        assert_eq!(course.difficulty, "advanced");
        assert_eq!(course.modules.len(), 8);
        assert!(course.tags.contains(&"advanced".to_string()));
    }
}
