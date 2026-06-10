// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Research Domain Plugin
//!
//! Academic and scientific research assistance: literature review, methodology
//! critique, claim analysis, and synthesis across papers and datasets.

use super::domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, IntentPrototypes, ValidationResult,
};

/// Research plugin for academic and scientific research assistance.
pub struct ResearchPlugin;

impl DomainPlugin for ResearchPlugin {
    fn domain_name(&self) -> &str {
        "research"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let lower = text.to_lowercase();

        // Extract paper/DOI references
        let re_doi =
            regex::Regex::new(r"(?i)(?:doi[:\s]*)?10\.\d{4,}/\S+").expect("valid regex literal");
        for m in re_doi.find_iter(text) {
            entities.push(Entity::new("doi", m.as_str(), m.start(), m.end()).with_confidence(0.95));
        }

        // Extract arXiv IDs
        let re_arxiv = regex::Regex::new(r"(?i)(?:arxiv[:\s]*)?\d{4}\.\d{4,5}(?:v\d+)?")
            .expect("valid regex literal");
        for m in re_arxiv.find_iter(text) {
            entities
                .push(Entity::new("arxiv_id", m.as_str(), m.start(), m.end()).with_confidence(0.9));
        }

        // Extract author citations (Name et al., YYYY) or (Name, YYYY)
        let re_cite = regex::Regex::new(r"\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s*(\d{4})\)")
            .expect("valid regex literal");
        for cap in re_cite.captures_iter(text) {
            if let (Some(author), Some(year)) = (cap.get(1), cap.get(2)) {
                let full_match = cap.get(0).expect("capture group 0 always exists");
                entities.push(
                    Entity::new(
                        "citation",
                        full_match.as_str(),
                        full_match.start(),
                        full_match.end(),
                    )
                    .with_confidence(0.85)
                    .with_metadata("author", author.as_str())
                    .with_metadata("year", year.as_str()),
                );
            }
        }

        // Extract methodology keywords
        let methods = [
            "regression",
            "classification",
            "clustering",
            "meta-analysis",
            "rct",
            "randomized",
            "controlled trial",
            "cohort",
            "cross-sectional",
            "longitudinal",
            "qualitative",
            "quantitative",
            "mixed methods",
            "neural network",
            "deep learning",
            "machine learning",
            "statistical test",
            "p-value",
            "confidence interval",
            "bayesian",
            "frequentist",
            "anova",
            "t-test",
            "chi-square",
        ];
        for method in &methods {
            if let Some(pos) = lower.find(method) {
                entities.push(
                    Entity::new("methodology", *method, pos, pos + method.len())
                        .with_confidence(0.8),
                );
            }
        }

        // Extract dataset names (capitalized multi-word patterns)
        let datasets = [
            "imagenet",
            "mnist",
            "cifar",
            "coco",
            "glue",
            "superglue",
            "squad",
            "wikitext",
            "openwebtext",
            "the pile",
            "common crawl",
            "laion",
            "c4",
        ];
        for dataset in &datasets {
            if lower.contains(dataset) {
                let pos = lower
                    .find(dataset)
                    .expect("dataset confirmed present by contains() check");
                entities.push(
                    Entity::new("dataset", *dataset, pos, pos + dataset.len()).with_confidence(0.9),
                );
            }
        }

        // Extract claim markers
        let claim_words = [
            "claim",
            "argue",
            "propose",
            "hypothesize",
            "suggest",
            "demonstrate",
            "show that",
            "prove",
            "establish",
        ];
        for word in &claim_words {
            if let Some(pos) = lower.find(word) {
                entities.push(
                    Entity::new("claim_marker", *word, pos, pos + word.len()).with_confidence(0.75),
                );
            }
        }

        entities
    }

    fn intent_prototypes(&self) -> IntentPrototypes {
        let mut protos = IntentPrototypes::default();
        protos.custom.insert(
            "literature_review".to_string(),
            vec![
                "literature review".to_string(),
                "related work".to_string(),
                "prior work".to_string(),
                "previous research".to_string(),
                "state of the art".to_string(),
                "survey".to_string(),
            ],
        );
        protos.custom.insert(
            "methodology".to_string(),
            vec![
                "methodology".to_string(),
                "method".to_string(),
                "approach".to_string(),
                "technique".to_string(),
                "experimental design".to_string(),
                "protocol".to_string(),
            ],
        );
        protos.custom.insert(
            "critique".to_string(),
            vec![
                "critique".to_string(),
                "weakness".to_string(),
                "limitation".to_string(),
                "flaw".to_string(),
                "review".to_string(),
                "evaluate".to_string(),
            ],
        );
        protos.custom.insert(
            "synthesize".to_string(),
            vec![
                "synthesize".to_string(),
                "combine".to_string(),
                "integrate".to_string(),
                "connect".to_string(),
                "relationship".to_string(),
                "gap".to_string(),
            ],
        );
        protos
    }

    fn prompts(&self) -> DomainPrompts {
        DomainPrompts {
            system: "You are Symthaea, a consciousness-first AI research assistant. \
                     Maintain academic rigor. Cite sources where possible. \
                     Distinguish between established findings and speculation. \
                     Use precise, measured language."
                .to_string(),
            clarification: "To provide a thorough research analysis, I need more details. \
                           Could you specify: {}"
                .to_string(),
            action_confirm: "I'll analyze this from a research perspective: {}".to_string(),
            error_explain: "There may be a methodological concern: {}".to_string(),
            out_of_domain: "This seems outside the scope of academic research. {}".to_string(),
        }
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        let lower = topic.to_lowercase();
        let research_keywords = [
            "paper",
            "study",
            "research",
            "journal",
            "publication",
            "citation",
            "hypothesis",
            "experiment",
            "methodology",
            "dataset",
            "results",
            "findings",
            "literature",
            "review",
            "abstract",
            "doi",
            "arxiv",
            "peer review",
            "conference",
            "proceedings",
            "thesis",
            "dissertation",
            "p-value",
            "statistical",
            "sample size",
            "control group",
            "reproducibility",
            "replication",
            "meta-analysis",
        ];
        let matches = research_keywords
            .iter()
            .filter(|k| lower.contains(*k))
            .count();
        if matches >= 3 {
            0.9
        } else if matches >= 2 {
            0.7
        } else if matches == 1 {
            0.45
        } else {
            0.1
        }
    }

    fn vocabulary(&self) -> Vec<String> {
        vec![
            "hypothesis",
            "methodology",
            "experiment",
            "dataset",
            "results",
            "findings",
            "conclusion",
            "abstract",
            "introduction",
            "discussion",
            "literature review",
            "citation",
            "reference",
            "p-value",
            "statistical significance",
            "effect size",
            "sample size",
            "control group",
            "treatment group",
            "bias",
            "confound",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn validate_input(&self, input: &str) -> ValidationResult {
        if input.trim().is_empty() {
            return ValidationResult::invalid(vec!["Empty input".to_string()]);
        }
        let mut result = ValidationResult::valid();
        if input.len() < 10 {
            result
                .warnings
                .push("Very short research query; consider providing more context.".to_string());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_name() {
        let plugin = ResearchPlugin;
        assert_eq!(plugin.domain_name(), "research");
    }

    #[test]
    fn test_extract_doi() {
        let plugin = ResearchPlugin;
        let entities = plugin.extract_entities("See doi:10.1038/s41586-023-06647-8");
        assert!(entities.iter().any(|e| e.entity_type == "doi"));
    }

    #[test]
    fn test_extract_arxiv() {
        let plugin = ResearchPlugin;
        let entities = plugin.extract_entities("The paper arxiv:2301.07041v2 describes...");
        assert!(entities.iter().any(|e| e.entity_type == "arxiv_id"));
    }

    #[test]
    fn test_extract_citation() {
        let plugin = ResearchPlugin;
        let entities =
            plugin.extract_entities("According to (Tononi et al., 2016), IIT predicts...");
        assert!(entities.iter().any(|e| e.entity_type == "citation"));
    }

    #[test]
    fn test_extract_methodology() {
        let plugin = ResearchPlugin;
        let entities = plugin.extract_entities("We used a Bayesian meta-analysis approach");
        assert!(entities.iter().any(|e| e.entity_type == "methodology"));
    }

    #[test]
    fn test_extract_dataset() {
        let plugin = ResearchPlugin;
        let entities = plugin.extract_entities("Evaluated on MNIST and CIFAR-10");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "dataset" && e.value == "mnist")
        );
    }

    #[test]
    fn test_extract_claim_marker() {
        let plugin = ResearchPlugin;
        let entities =
            plugin.extract_entities("The authors claim that consciousness requires integration");
        assert!(entities.iter().any(|e| e.entity_type == "claim_marker"));
    }

    #[test]
    fn test_in_domain_scoring() {
        let plugin = ResearchPlugin;
        assert!(plugin.is_in_domain("review this paper's methodology and results") > 0.6);
        assert!(plugin.is_in_domain("hello world") < 0.3);
    }

    #[test]
    fn test_intent_prototypes() {
        let plugin = ResearchPlugin;
        let protos = plugin.intent_prototypes();
        assert!(protos.custom.contains_key("literature_review"));
        assert!(protos.custom.contains_key("methodology"));
        assert!(protos.custom.contains_key("critique"));
        assert!(protos.custom.contains_key("synthesize"));
    }
}
