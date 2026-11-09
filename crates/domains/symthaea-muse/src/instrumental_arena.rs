// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, source-bound prompt bank for the instrumental preference arena.
//!
//! The prompt bank is a candidate input set, not a frozen experiment. It does
//! not carry model outputs, listener responses, randomized schedules, or
//! confirmatory authority. Those remain in the existing sealed study pipeline.

use crate::Style;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

pub const INSTRUMENTAL_ARENA_PROMPT_BANK_SCHEMA: &str =
    "muse-instrumental-arena-prompt-bank-v1";
pub const INSTRUMENTAL_ARENA_PROMPT_BANK_STATUS: &str =
    "pilot_candidate_not_confirmatory";
pub const INSTRUMENTAL_ARENA_PROMPT_BANK_PATH: &str =
    "MUSE_INSTRUMENTAL_ARENA_PROMPTS_2026-10-31.json";

const REQUIRED_CONTENT: [&str; 4] = [
    "instrumental_only",
    "no_singing_or_spoken_words",
    "coherent_motif_or_process",
    "clear_lead_and_support_roles",
];
const REQUIRED_DENSITIES: [&str; 3] = ["sparse", "moderate", "dense"];

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InstrumentalArenaPromptBank {
    pub schema: String,
    pub status: String,
    pub created_on: String,
    pub source: PromptBankSource,
    pub design: PromptBankDesign,
    pub balance: PromptBankBalance,
    pub cards: Vec<InstrumentalArenaPromptCard>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PromptBankSource {
    pub repository: String,
    pub style_registry_path: String,
    pub style_registry_blob_sha: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PromptBankDesign {
    pub target_cards: usize,
    pub strata_count: usize,
    pub task_families_per_style: usize,
    pub density_levels_per_task: usize,
    pub cards_per_style: usize,
    pub cards_per_density: usize,
    pub target_duration_seconds: u64,
    pub duration_policy: String,
    pub generation_policy: String,
    pub prompt_policy: String,
    pub review_policy: String,
    pub exclusions: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PromptBankBalance {
    pub cards_per_style: BTreeMap<String, usize>,
    pub cards_per_density: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InstrumentalArenaPromptCard {
    pub id: String,
    pub style_name: String,
    pub style_family: String,
    pub task_id: String,
    pub target_form: String,
    pub density: String,
    pub target_duration_seconds: u64,
    pub requested_instruments: Vec<String>,
    pub required_content: Vec<String>,
    pub task_description: String,
    pub style_detail: String,
    pub density_description: String,
    pub prompt: String,
    pub review_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptBankIssue {
    WrongSchema { found: String },
    WrongStatus { found: String },
    EmptyField { field: String },
    InvalidStyleRegistryDigest,
    TargetCardCountMismatch { expected: usize, found: usize },
    StyleStrataCountMismatch { expected: usize, found: usize },
    DensityLevelCountMismatch { expected: usize, found: usize },
    WrongPerStyleCardCount {
        style: String,
        expected: usize,
        found: usize,
    },
    WrongPerDensityCardCount {
        density: String,
        expected: usize,
        found: usize,
    },
    DuplicateCardId { id: String },
    DuplicatePrompt { id: String },
    UnknownStyle { id: String, style: String },
    UnknownDensity { id: String, density: String },
    DuplicateTaskDensityCell { id: String, task: String, density: String },
    WrongTaskCount { style: String, expected: usize, found: usize },
    IncompleteTaskDensityGrid { style: String, task: String },
    WrongDuration { id: String, expected: u64, found: u64 },
    MissingInstrument { id: String },
    DuplicateInstrument { id: String, instrument: String },
    MissingRequiredContent { id: String, requirement: String },
    MissingInstrumentalGuard { id: String },
    MissingPractitionerReview { style: String },
    BalanceMismatch { field: String },
}

impl InstrumentalArenaPromptBank {
    pub fn parse(source: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(source)
    }

    /// Validate source identity, style support, cardinality and the complete
    /// style × task × density design. This is a manifest check only; it does
    /// not validate rendered audio, provider access, or musical authenticity.
    pub fn validate(&self) -> Vec<PromptBankIssue> {
        let mut issues = Vec::new();
        if self.schema != INSTRUMENTAL_ARENA_PROMPT_BANK_SCHEMA {
            issues.push(PromptBankIssue::WrongSchema {
                found: self.schema.clone(),
            });
        }
        if self.status != INSTRUMENTAL_ARENA_PROMPT_BANK_STATUS {
            issues.push(PromptBankIssue::WrongStatus {
                found: self.status.clone(),
            });
        }

        for (field, value) in [
            ("created_on", self.created_on.as_str()),
            ("source.repository", self.source.repository.as_str()),
            ("source.style_registry_path", self.source.style_registry_path.as_str()),
            ("source.note", self.source.note.as_str()),
            ("design.duration_policy", self.design.duration_policy.as_str()),
            ("design.generation_policy", self.design.generation_policy.as_str()),
            ("design.prompt_policy", self.design.prompt_policy.as_str()),
            ("design.review_policy", self.design.review_policy.as_str()),
            ("design.exclusions", self.design.exclusions.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(PromptBankIssue::EmptyField {
                    field: field.into(),
                });
            }
        }
        if !is_sha1(&self.source.style_registry_blob_sha) {
            issues.push(PromptBankIssue::InvalidStyleRegistryDigest);
        }

        if self.design.target_cards != self.cards.len() {
            issues.push(PromptBankIssue::TargetCardCountMismatch {
                expected: self.design.target_cards,
                found: self.cards.len(),
            });
        }

        let mut card_ids = BTreeSet::new();
        let mut prompt_texts = BTreeSet::new();
        let mut styles = BTreeSet::new();
        let mut densities = BTreeSet::new();
        let mut counts_by_style = BTreeMap::<String, usize>::new();
        let mut counts_by_density = BTreeMap::<String, usize>::new();
        let mut tasks_by_style = BTreeMap::<String, BTreeSet<String>>::new();
        let mut cells = BTreeSet::<(String, String, String)>::new();

        let known_styles: BTreeSet<String> =
            Style::ALL.into_iter().map(|style| format!("{style:?}")).collect();

        for card in &self.cards {
            if card.id.trim().is_empty() {
                issues.push(PromptBankIssue::EmptyField {
                    field: "card.id".into(),
                });
            } else if !card_ids.insert(card.id.clone()) {
                issues.push(PromptBankIssue::DuplicateCardId {
                    id: card.id.clone(),
                });
            }
            if card.prompt.trim().is_empty() {
                issues.push(PromptBankIssue::EmptyField {
                    field: format!("card.{}.prompt", card.id),
                });
            } else if !prompt_texts.insert(card.prompt.clone()) {
                issues.push(PromptBankIssue::DuplicatePrompt {
                    id: card.id.clone(),
                });
            }
            if !known_styles.contains(&card.style_name) {
                issues.push(PromptBankIssue::UnknownStyle {
                    id: card.id.clone(),
                    style: card.style_name.clone(),
                });
            }
            if !REQUIRED_DENSITIES.contains(&card.density.as_str()) {
                issues.push(PromptBankIssue::UnknownDensity {
                    id: card.id.clone(),
                    density: card.density.clone(),
                });
            }
            if card.target_duration_seconds != self.design.target_duration_seconds {
                issues.push(PromptBankIssue::WrongDuration {
                    id: card.id.clone(),
                    expected: self.design.target_duration_seconds,
                    found: card.target_duration_seconds,
                });
            }
            if card.requested_instruments.is_empty() {
                issues.push(PromptBankIssue::MissingInstrument {
                    id: card.id.clone(),
                });
            }
            let mut instruments = BTreeSet::new();
            for instrument in &card.requested_instruments {
                if instrument.trim().is_empty() || !instruments.insert(instrument) {
                    issues.push(PromptBankIssue::DuplicateInstrument {
                        id: card.id.clone(),
                        instrument: instrument.clone(),
                    });
                }
            }
            for required in REQUIRED_CONTENT {
                if !card.required_content.iter().any(|found| found.as_str() == required) {
                    issues.push(PromptBankIssue::MissingRequiredContent {
                        id: card.id.clone(),
                        requirement: required.into(),
                    });
                }
            }
            let prompt_lower = card.prompt.to_ascii_lowercase();
            if !prompt_lower.contains("strictly instrumental")
                || !prompt_lower.contains("no singing")
                || !prompt_lower.contains("spoken words")
            {
                issues.push(PromptBankIssue::MissingInstrumentalGuard {
                    id: card.id.clone(),
                });
            }
            if matches!(
                card.style_name.as_str(),
                "AfroCuban" | "BossaNova" | "Flamenco"
            ) && !card.review_status.contains("practitioner_review_required")
            {
                issues.push(PromptBankIssue::MissingPractitionerReview {
                    style: card.style_name.clone(),
                });
            }

            *counts_by_style.entry(card.style_name.clone()).or_default() += 1;
            *counts_by_density.entry(card.density.clone()).or_default() += 1;
            styles.insert(card.style_name.clone());
            densities.insert(card.density.clone());
            tasks_by_style
                .entry(card.style_name.clone())
                .or_default()
                .insert(card.task_id.clone());

            let cell = (
                card.style_name.clone(),
                card.task_id.clone(),
                card.density.clone(),
            );
            if !cells.insert(cell) {
                issues.push(PromptBankIssue::DuplicateTaskDensityCell {
                    id: card.id.clone(),
                    task: card.task_id.clone(),
                    density: card.density.clone(),
                });
            }
        }

        if styles.len() != self.design.strata_count {
            issues.push(PromptBankIssue::StyleStrataCountMismatch {
                expected: self.design.strata_count,
                found: styles.len(),
            });
        }
        if densities.len() != self.design.density_levels_per_task {
            issues.push(PromptBankIssue::DensityLevelCountMismatch {
                expected: self.design.density_levels_per_task,
                found: densities.len(),
            });
        }

        for style in &styles {
            let found = counts_by_style.get(style).copied().unwrap_or(0);
            if found != self.design.cards_per_style {
                issues.push(PromptBankIssue::WrongPerStyleCardCount {
                    style: style.clone(),
                    expected: self.design.cards_per_style,
                    found,
                });
            }
            let tasks = tasks_by_style.get(style).map_or(0, BTreeSet::len);
            if tasks != self.design.task_families_per_style {
                issues.push(PromptBankIssue::WrongTaskCount {
                    style: style.clone(),
                    expected: self.design.task_families_per_style,
                    found: tasks,
                });
            }
            for task in tasks_by_style.get(style).into_iter().flat_map(|set| set.iter()) {
                for density in REQUIRED_DENSITIES {
                    if !cells.contains(&(style.clone(), task.clone(), density.into())) {
                        issues.push(PromptBankIssue::IncompleteTaskDensityGrid {
                            style: style.clone(),
                            task: task.clone(),
                        });
                    }
                }
            }
        }
        for density in REQUIRED_DENSITIES {
            let found = counts_by_density.get(density).copied().unwrap_or(0);
            if found != self.design.cards_per_density {
                issues.push(PromptBankIssue::WrongPerDensityCardCount {
                    density: density.into(),
                    expected: self.design.cards_per_density,
                    found,
                });
            }
        }

        if self.balance.cards_per_style != counts_by_style {
            issues.push(PromptBankIssue::BalanceMismatch {
                field: "cards_per_style".into(),
            });
        }
        if self.balance.cards_per_density != counts_by_density {
            issues.push(PromptBankIssue::BalanceMismatch {
                field: "cards_per_density".into(),
            });
        }
        issues
    }
}

fn is_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT_BANK_JSON: &str =
        include_str!("../MUSE_INSTRUMENTAL_ARENA_PROMPTS_2026-10-31.json");
    const DURATION_MATCHED_PROMPT_BANK_JSON: &str =
        include_str!("../MUSE_INSTRUMENTAL_ARENA_PROMPTS_180S_2026-10-31.json");

    #[test]
    fn checked_in_prompt_bank_is_balanced_and_pilot_only() {
        let bank = InstrumentalArenaPromptBank::parse(PROMPT_BANK_JSON)
            .expect("checked-in prompt bank must be valid JSON");
        assert_eq!(
            bank.validate(),
            Vec::<PromptBankIssue>::new(),
            "prompt-bank contract must remain exact"
        );
        assert_eq!(bank.cards.len(), 150);
        assert_eq!(bank.design.strata_count, 10);
        assert_eq!(bank.design.cards_per_style, 15);
        assert_eq!(bank.design.cards_per_density, 50);
        assert_eq!(bank.status, INSTRUMENTAL_ARENA_PROMPT_BANK_STATUS);
    }

    #[test]
    fn checked_in_180_second_prompt_bank_is_balanced_and_pilot_only() {
        let bank = InstrumentalArenaPromptBank::parse(DURATION_MATCHED_PROMPT_BANK_JSON)
            .expect("checked-in 180-second prompt bank must be valid JSON");
        assert_eq!(
            bank.validate(),
            Vec::<PromptBankIssue>::new(),
            "duration-matched prompt-bank contract must remain exact"
        );
        assert_eq!(bank.cards.len(), 150);
        assert_eq!(bank.design.target_duration_seconds, 180);
        assert_eq!(bank.design.cards_per_style, 15);
        assert_eq!(bank.design.cards_per_density, 50);
        assert_eq!(bank.status, INSTRUMENTAL_ARENA_PROMPT_BANK_STATUS);
    }

    #[test]
    fn confirmatory_status_cannot_be_smuggled_into_candidate_prompt_bank() {
        let mut bank = InstrumentalArenaPromptBank::parse(PROMPT_BANK_JSON).unwrap();
        bank.status = "confirmatory".into();
        assert!(bank
            .validate()
            .contains(&PromptBankIssue::WrongStatus {
                found: "confirmatory".into()
            }));
    }

    #[test]
    fn non_registry_styles_are_rejected() {
        let mut bank = InstrumentalArenaPromptBank::parse(PROMPT_BANK_JSON).unwrap();
        bank.cards[0].style_name = "ImaginaryStyle".into();
        assert!(bank.validate().contains(&PromptBankIssue::UnknownStyle {
            id: bank.cards[0].id.clone(),
            style: "ImaginaryStyle".into()
        }));
    }
}
