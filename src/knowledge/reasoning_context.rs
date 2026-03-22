// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reasoning Context — Knowledge-Grounded Inference Support
//!
//! Provides structured context from the knowledge engine to any reasoning
//! process (cognitive loop, LLM organ, MCTS dream engine, etc.). Collects
//! relevant facts, causal chains, contradictions, and ontology concepts
//! into a single `ReasoningContext` that can be injected into working memory.
//!
//! Science: Baddeley (2000) working memory model, Anderson (1983) ACT-R
//!          spreading activation, Tulving (1972) episodic vs semantic memory

use super::graph::FactSearchResult;
use super::manager::{KnowledgeManager, KnowledgeSignals};
use crate::cognitive_loop::thresholds;

// ── Types ──────────────────────────────────────────────────────────────────

/// A grounded reasoning context assembled from the knowledge engine.
///
/// Contains everything a reasoning process needs to make informed
/// decisions without hallucination: relevant facts (with confidence),
/// causal chains (with direction), contradictions (flagged for attention),
/// and uncertainty estimates.
#[derive(Debug, Clone)]
pub struct ReasoningContext {
    /// Top-k relevant facts from the knowledge graph (by HDC similarity)
    pub relevant_facts: Vec<GroundedFact>,
    /// Causal chains traced from entities in the query
    pub causal_chains: Vec<CausalChain>,
    /// Active contradictions that should modulate confidence
    pub contradictions: Vec<String>,
    /// Overall epistemic state
    pub epistemic_state: EpistemicState,
    /// Natural language summary of the context (for LLM injection)
    pub summary: String,
    /// Cross-domain analogies: (source_fact_text, similarity_score).
    /// Science: Gentner (1983) — structure mapping theory.
    pub analogies: Vec<(String, f32)>,
    /// Counterfactual necessity scores: (cause, necessity_score).
    /// Science: Pearl (2000) — do-calculus for causal counterfactuals.
    pub counterfactuals: Vec<(String, f32)>,
}

/// A fact from the knowledge graph with provenance
#[derive(Debug, Clone)]
pub struct GroundedFact {
    /// Human-readable fact text
    pub text: String,
    /// Confidence score (decayed over time)
    pub confidence: f32,
    /// HDC similarity to the query
    pub similarity: f32,
    /// Domain tag (if any)
    pub domain: Option<String>,
    /// Whether this fact contains causal claims
    pub is_causal: bool,
}

/// A traced causal chain with named steps
#[derive(Debug, Clone)]
pub struct CausalChain {
    /// The entity this chain starts from
    pub root: String,
    /// Ordered steps in the chain
    pub steps: Vec<String>,
    /// Total chain depth
    pub depth: usize,
}

/// Epistemic state summary for reasoning confidence modulation
#[derive(Debug, Clone)]
pub struct EpistemicState {
    /// How uncertain the knowledge engine is about the current topic (0-1)
    pub uncertainty: f64,
    /// How novel the current input is (0-1)
    pub novelty: f64,
    /// How many contradictions are active
    pub contradiction_count: usize,
    /// Whether we have any grounded knowledge about this topic
    pub has_grounding: bool,
    /// Recommended confidence multiplier for reasoning outputs
    /// High grounding + low uncertainty → 1.0
    /// No grounding + high uncertainty → 0.3
    pub confidence_multiplier: f64,
}

/// Lightweight query result for cross-module consumers (Broca, EthicsEngine).
/// Avoids pulling in the full ReasoningContext machinery.
#[derive(Debug, Clone)]
pub struct KnowledgeQueryResult {
    /// Top-k relevant grounded facts
    pub facts: Vec<GroundedFact>,
    /// Causal chains traced from query entities
    pub causal_chains: Vec<CausalChain>,
    /// Overall grounding score [0.0, 1.0]
    pub grounding_score: f64,
}

impl Default for KnowledgeQueryResult {
    fn default() -> Self {
        Self {
            facts: Vec::new(),
            causal_chains: Vec::new(),
            grounding_score: 0.0,
        }
    }
}

impl KnowledgeQueryResult {
    /// Confidence multiplier for consumers — maps grounding score to [0.3, 1.3].
    pub fn confidence_multiplier(&self) -> f64 {
        0.3 + self.grounding_score
    }
}

impl Default for ReasoningContext {
    fn default() -> Self {
        Self {
            relevant_facts: Vec::new(),
            causal_chains: Vec::new(),
            contradictions: Vec::new(),
            epistemic_state: EpistemicState {
                uncertainty: 1.0,
                novelty: 1.0,
                contradiction_count: 0,
                has_grounding: false,
                confidence_multiplier: 0.3,
            },
            summary: String::new(),
            analogies: Vec::new(),
            counterfactuals: Vec::new(),
        }
    }
}

// ── Context Assembly ───────────────────────────────────────────────────────

impl ReasoningContext {
    /// Assemble a reasoning context from the knowledge manager's current state.
    ///
    /// Call after `KnowledgeManager::process()` to get a context snapshot
    /// that can be injected into LLM prompts, MCTS planning, or the
    /// reasoning engine's working memory.
    pub fn from_manager(manager: &KnowledgeManager, query: &str) -> Self {
        let signals = manager.signals();
        let search_results = manager.last_search_results();
        let alerts: &[(); 0] = &[]; // Alerts are drained separately

        // 1. Convert search results to grounded facts
        let relevant_facts: Vec<GroundedFact> = search_results
            .iter()
            .map(|r| GroundedFact {
                text: format!("fact:{}", r.fact_id),
                confidence: r.confidence,
                similarity: r.similarity,
                domain: None,
                is_causal: false,
            })
            .collect();

        // 2. Trace causal chains for key terms in the query
        let causal_chains = trace_query_chains(manager, query);

        // 3. Compute epistemic state
        let has_grounding =
            !relevant_facts.is_empty() && relevant_facts.iter().any(|f| f.similarity > 0.1);

        let confidence_multiplier = compute_confidence_multiplier(signals, has_grounding);

        let epistemic_state = EpistemicState {
            uncertainty: signals.uncertainty,
            novelty: signals.novelty,
            contradiction_count: alerts.len(),
            has_grounding,
            confidence_multiplier,
        };

        // 4. Generate natural language summary
        let summary = generate_summary(&relevant_facts, &causal_chains, &epistemic_state);

        // 5. Counterfactual necessity for causal chain roots
        let counterfactuals: Vec<(String, f32)> = causal_chains
            .iter()
            .take(3)
            .flat_map(|chain| {
                manager
                    .causal_bridge()
                    .counterfactual_necessity(&chain.root, 3)
            })
            .collect();

        // 6. Detect confounders between causal chain roots (Pearl backdoor criterion)
        let mut contradictions: Vec<String> = Vec::new();
        let chain_roots: Vec<&str> = causal_chains.iter().map(|c| c.root.as_str()).collect();
        for i in 0..chain_roots.len() {
            for j in (i + 1)..chain_roots.len() {
                let confounders = manager.detect_confounders(chain_roots[i], chain_roots[j]);
                for (confounder, strength) in &confounders {
                    if *strength > thresholds::KNOWLEDGE_CONFOUNDER_STRENGTH_THRESHOLD {
                        contradictions.push(format!(
                            "Confounder: '{}' (strength {:.2}) may cause both '{}' and '{}'",
                            confounder, strength, chain_roots[i], chain_roots[j]
                        ));
                    }
                }
            }
        }

        let mut epistemic_state = epistemic_state;
        epistemic_state.contradiction_count = contradictions.len();

        Self {
            relevant_facts,
            causal_chains,
            contradictions,
            epistemic_state,
            summary,
            analogies: Vec::new(), // Populated by cycle.rs analogical transfer
            counterfactuals,
        }
    }

    /// Whether this context provides useful grounding for reasoning
    pub fn is_grounded(&self) -> bool {
        self.epistemic_state.has_grounding
    }

    /// Number of relevant facts
    pub fn fact_count(&self) -> usize {
        self.relevant_facts.len()
    }

    /// Best fact confidence
    pub fn best_confidence(&self) -> f32 {
        self.relevant_facts
            .iter()
            .map(|f| f.confidence)
            .fold(0.0_f32, f32::max)
    }

    /// Total causal chain depth across all chains
    pub fn total_causal_depth(&self) -> usize {
        self.causal_chains.iter().map(|c| c.depth).sum()
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn trace_query_chains(manager: &KnowledgeManager, query: &str) -> Vec<CausalChain> {
    let mut chains = Vec::new();

    // Extract simple terms from the query (split on whitespace, lowercase)
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| w.len() > 2) // Skip tiny words
        .collect();

    for term in &terms {
        let traced = manager.causal_bridge().trace_chain(term, 5);
        for chain_steps in traced {
            if !chain_steps.is_empty() {
                chains.push(CausalChain {
                    root: term.clone(),
                    steps: chain_steps.clone(),
                    depth: chain_steps.len(),
                });
            }
        }
    }

    chains
}

fn compute_confidence_multiplier(signals: &KnowledgeSignals, has_grounding: bool) -> f64 {
    if !has_grounding {
        // No relevant knowledge → low confidence in reasoning
        return 0.3;
    }

    // Base confidence from relevance (how well did stored knowledge match?)
    let relevance_factor = signals.relevance.clamp(0.0, 1.0);

    // Penalty for contradictions
    let contradiction_penalty = 1.0 - (signals.contradiction_signal * 0.5).min(0.5);

    // Penalty for high uncertainty
    let certainty_factor = 1.0 - (signals.uncertainty * 0.3);

    // Bonus for causal depth (deeper chains = more reasoning support)
    let causal_bonus = 1.0 + (signals.causal_depth * 0.1).min(0.2);

    (relevance_factor * contradiction_penalty * certainty_factor * causal_bonus).clamp(0.3, 1.0)
}

fn generate_summary(
    facts: &[GroundedFact],
    chains: &[CausalChain],
    epistemic: &EpistemicState,
) -> String {
    let mut parts = Vec::new();

    // Facts summary
    if !facts.is_empty() {
        let top_facts: Vec<&str> = facts.iter().take(3).map(|f| f.text.as_str()).collect();
        parts.push(format!(
            "Known facts ({} total, best confidence {:.0}%): {}",
            facts.len(),
            facts.iter().map(|f| f.confidence).fold(0.0_f32, f32::max) * 100.0,
            top_facts.join("; "),
        ));
    }

    // Causal chains summary
    if !chains.is_empty() {
        let chain_strs: Vec<String> = chains
            .iter()
            .take(2)
            .map(|c| format!("{} → {}", c.root, c.steps.join(" → ")))
            .collect();
        parts.push(format!("Causal chains: {}", chain_strs.join("; ")));
    }

    // Epistemic warning
    if epistemic.uncertainty > 0.7 {
        parts.push("WARNING: High epistemic uncertainty — limited grounding available.".into());
    }
    if epistemic.contradiction_count > 0 {
        parts.push(format!(
            "ALERT: {} active contradiction(s) detected.",
            epistemic.contradiction_count,
        ));
    }

    if parts.is_empty() {
        "No relevant knowledge found for this query.".into()
    } else {
        parts.join(" | ")
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::manager::KnowledgeManagerConfig;

    #[test]
    fn test_empty_context() {
        let ctx = ReasoningContext::default();
        assert!(!ctx.is_grounded());
        assert_eq!(ctx.fact_count(), 0);
        assert_eq!(ctx.total_causal_depth(), 0);
        assert!(ctx.epistemic_state.confidence_multiplier < 0.5);
    }

    #[test]
    fn test_from_manager_no_knowledge() {
        let mgr = KnowledgeManager::default();
        let ctx = ReasoningContext::from_manager(&mgr, "something unknown");
        assert!(!ctx.is_grounded());
        assert!(ctx.summary.contains("No relevant knowledge"));
    }

    #[test]
    fn test_from_manager_with_knowledge() {
        let mut mgr = KnowledgeManager::default();
        mgr.bootstrap_entities();

        // Insert some knowledge
        mgr.process("Iran caused sanctions to escalate.", 1);
        mgr.process("Sanctions caused oil shortage.", 2);

        let ctx = ReasoningContext::from_manager(&mgr, "Iran sanctions");
        // Context should have been assembled (may or may not be grounded
        // depending on HDC similarity thresholds)
        assert!(ctx.epistemic_state.uncertainty >= 0.0);
        assert!(ctx.epistemic_state.uncertainty <= 1.0);
    }

    #[test]
    fn test_confidence_multiplier_no_grounding() {
        let signals = KnowledgeSignals::default();
        let mult = compute_confidence_multiplier(&signals, false);
        assert!((mult - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_confidence_multiplier_with_grounding() {
        let signals = KnowledgeSignals {
            uncertainty: 0.2,
            contradiction_signal: 0.0,
            relevance: 0.8,
            novelty: 0.2,
            causal_depth: 0.5,
            epistemic_surprise: 0.0,
        };
        let mult = compute_confidence_multiplier(&signals, true);
        assert!(mult > 0.5); // Good grounding → high confidence
    }

    #[test]
    fn test_summary_generation() {
        let facts = vec![GroundedFact {
            text: "Iran sanctioned.".into(),
            confidence: 0.9,
            similarity: 0.7,
            domain: Some("geopolitics".into()),
            is_causal: true,
        }];
        let chains = vec![CausalChain {
            root: "sanctions".into(),
            steps: vec!["oil shortage".into(), "price spike".into()],
            depth: 2,
        }];
        let epistemic = EpistemicState {
            uncertainty: 0.3,
            novelty: 0.3,
            contradiction_count: 0,
            has_grounding: true,
            confidence_multiplier: 0.8,
        };

        let summary = generate_summary(&facts, &chains, &epistemic);
        assert!(summary.contains("Known facts"));
        assert!(summary.contains("Causal chains"));
        assert!(!summary.contains("WARNING"));
    }

    #[test]
    fn test_summary_high_uncertainty_warning() {
        let epistemic = EpistemicState {
            uncertainty: 0.9,
            novelty: 0.9,
            contradiction_count: 2,
            has_grounding: false,
            confidence_multiplier: 0.3,
        };

        let summary = generate_summary(&[], &[], &epistemic);
        // High uncertainty + contradictions should produce warnings
        assert!(summary.contains("WARNING") || summary.contains("ALERT"));
    }

    #[test]
    fn test_grounded_fact_accessors() {
        let mut ctx = ReasoningContext::default();
        ctx.relevant_facts.push(GroundedFact {
            text: "test".into(),
            confidence: 0.85,
            similarity: 0.6,
            domain: None,
            is_causal: false,
        });
        ctx.epistemic_state.has_grounding = true;

        assert!(ctx.is_grounded());
        assert_eq!(ctx.fact_count(), 1);
        assert!((ctx.best_confidence() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_causal_chain_depth() {
        let mut ctx = ReasoningContext::default();
        ctx.causal_chains.push(CausalChain {
            root: "a".into(),
            steps: vec!["b".into(), "c".into()],
            depth: 2,
        });
        ctx.causal_chains.push(CausalChain {
            root: "x".into(),
            steps: vec!["y".into()],
            depth: 1,
        });

        assert_eq!(ctx.total_causal_depth(), 3);
    }
}
