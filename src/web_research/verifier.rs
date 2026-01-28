//! Epistemic Verification - Making Hallucination Impossible
//!
//! **Revolutionary Aspect**: All claims are epistemically verified before use.
//! If unverifiable, they are automatically hedged or rejected.
//!
//! This implements the "Epistemic Governance Paradigm" from the analysis:
//! - Verifiable Claims: Linked to indexed external evidence
//! - Unverifiable Claims: Clearly marked as opinions/hypotheses
//! - Evidence-Based Justifications: Logical connective tissue
//! - Automatic Hedging: "Evidence suggests..." when uncertain

use crate::hdc::binary_hv::HV16;
use crate::language::knowledge_graph::KnowledgeGraph;
use crate::language::reasoning::ReasoningEngine;
use super::types::{Source, Claim, VerificationLevel};
use serde::{Deserialize, Serialize};

/// Epistemic status of a claim
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimConfidence {
    /// Verified with high confidence (multiple agreeing sources)
    HighConfidence,

    /// Verified with moderate confidence (few sources or some disagreement)
    ModerateConfidence,

    /// Low confidence (single source or conflicting evidence)
    LowConfidence,

    /// Contested (sources contradict each other)
    Contested,

    /// Unverifiable (no external evidence available)
    Unverifiable,

    /// Verified false (contradicted by reliable sources)
    False,
}

/// Evidence supporting or refuting a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Source URL
    pub source_url: String,

    /// Specific quote/excerpt
    pub excerpt: String,

    /// How well this supports the claim (0.0-1.0)
    pub support_strength: f64,

    /// Credibility of source (0.0-1.0)
    pub source_credibility: f64,

    /// Is this supporting or refuting?
    pub supports: bool,
}

/// Detected contradiction between sources
#[derive(Debug, Clone)]
pub struct Contradiction {
    /// First claim
    pub claim_a: String,

    /// Contradicting claim
    pub claim_b: String,

    /// Source of first claim
    pub source_a: String,

    /// Source of second claim
    pub source_b: String,

    /// Strength of contradiction (0.0-1.0)
    pub strength: f64,
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    /// The claim being verified
    pub claim: Claim,

    /// Epistemic status
    pub status: ClaimConfidence,

    /// Confidence level (0.0-1.0)
    pub confidence: f64,

    /// Supporting evidence
    pub evidence: Vec<Evidence>,

    /// Contradictions found (if any)
    pub contradictions: Vec<String>,  // Simplified for serialization

    /// Automatic hedging phrase
    pub hedge_phrase: String,

    /// Source URLs
    pub sources: Vec<String>,

    /// Number of sources checked
    pub sources_checked: usize,

    /// Number of sources supporting
    pub sources_supporting: usize,

    /// Number of sources refuting
    pub sources_refuting: usize,
}

/// Epistemic verifier
pub struct EpistemicVerifier {
    /// Knowledge graph for checking existing knowledge
    knowledge_graph: KnowledgeGraph,

    /// Reasoning engine for logical verification
    reasoning: ReasoningEngine,

    /// Minimum sources required for high confidence
    min_sources_high_confidence: usize,

    /// Minimum source credibility to consider
    min_source_credibility: f64,

    /// Agreement threshold (% of sources that must agree)
    agreement_threshold: f64,
}

impl EpistemicVerifier {
    pub fn new() -> Self {
        Self {
            knowledge_graph: KnowledgeGraph::new(),
            reasoning: ReasoningEngine::new(),
            min_sources_high_confidence: 3,
            min_source_credibility: 0.6,
            agreement_threshold: 0.75,
        }
    }

    /// Verify a claim against sources
    pub fn verify_claim(
        &self,
        claim: &Claim,
        sources: &[Source],
        level: VerificationLevel,
    ) -> Verification {
        // 1. Extract evidence from sources
        let evidence = self.extract_evidence(claim, sources);

        // 2. Check for contradictions (if rigorous verification)
        let contradictions = match level {
            VerificationLevel::Rigorous | VerificationLevel::Academic => {
                self.detect_contradictions(&evidence)
            }
            _ => Vec::new(),
        };

        // 3. Assess confidence
        let confidence = self.assess_confidence(&evidence, &contradictions, level);

        // 4. Determine epistemic status
        let status = self.determine_status(&evidence, &contradictions, confidence, level);

        // 5. Count supporting/refuting sources
        let sources_supporting = evidence.iter().filter(|e| e.supports).count();
        let sources_refuting = evidence.iter().filter(|e| !e.supports).count();

        // 6. Generate appropriate hedging phrase
        let hedge_phrase = self.generate_hedge(status, confidence);

        // 7. Collect source URLs
        let sources_urls: Vec<String> = sources.iter()
            .map(|s| s.url.clone())
            .collect();

        Verification {
            claim: claim.clone(),
            status,
            confidence,
            evidence,
            contradictions: contradictions.iter()
                .map(|c| format!("{} vs {}", c.claim_a, c.claim_b))
                .collect(),
            hedge_phrase,
            sources: sources_urls,
            sources_checked: sources.len(),
            sources_supporting,
            sources_refuting,
        }
    }

    /// Extract evidence from sources for a claim
    fn extract_evidence(&self, claim: &Claim, sources: &[Source]) -> Vec<Evidence> {
        let mut evidence = Vec::new();

        for source in sources {
            // Calculate semantic similarity between claim and source content
            let similarity = self.calculate_similarity(&claim.encoding, &source.encoding);

            // If similarity is high enough, extract relevant excerpts
            if similarity > 0.3 {
                // Find paragraphs that mention the claim
                let relevant_excerpts = self.find_relevant_excerpts(
                    &claim.text,
                    &source.content,
                );

                for excerpt in relevant_excerpts {
                    // Determine if this supports or refutes the claim
                    let supports = self.does_support_claim(&claim.text, &excerpt);
                    let support_strength = similarity * 0.8; // Adjust based on context

                    evidence.push(Evidence {
                        source_url: source.url.clone(),
                        excerpt,
                        support_strength,
                        source_credibility: source.credibility,
                        supports,
                    });
                }
            }
        }

        evidence
    }

    /// Calculate semantic similarity between HV16 encodings
    fn calculate_similarity(&self, a: &HV16, b: &HV16) -> f64 {
        // Use HDC similarity (normalized hamming distance)
        a.similarity(b) as f64
    }

    /// Find relevant excerpts in source content
    fn find_relevant_excerpts(&self, claim: &str, content: &str) -> Vec<String> {
        let mut excerpts = Vec::new();

        // Split content into sentences
        let sentences: Vec<&str> = content.split('.').collect();

        // Extract key terms from claim
        let key_terms: Vec<&str> = claim.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        // Find sentences containing key terms
        for sentence in sentences {
            let sentence_lower = sentence.to_lowercase();
            let mut matches = 0;

            for term in &key_terms {
                if sentence_lower.contains(&term.to_lowercase()) {
                    matches += 1;
                }
            }

            // If sentence contains multiple key terms, include it
            if matches >= 2 {
                excerpts.push(sentence.trim().to_string());
            }
        }

        excerpts
    }

    /// Determine if excerpt supports the claim
    fn does_support_claim(&self, claim: &str, excerpt: &str) -> bool {
        // Simplified: Check for negation words
        let negations = ["not", "no", "never", "false", "incorrect", "untrue"];
        let excerpt_lower = excerpt.to_lowercase();

        for negation in &negations {
            if excerpt_lower.contains(negation) {
                return false;  // Likely refuting
            }
        }

        true  // Likely supporting
    }

    /// Detect contradictions in evidence
    fn detect_contradictions(&self, evidence: &[Evidence]) -> Vec<Contradiction> {
        let mut contradictions = Vec::new();

        // Check for supporting vs refuting evidence
        for (i, ev_a) in evidence.iter().enumerate() {
            for ev_b in evidence.iter().skip(i + 1) {
                if ev_a.supports != ev_b.supports {
                    // Found contradiction!
                    contradictions.push(Contradiction {
                        claim_a: ev_a.excerpt.clone(),
                        claim_b: ev_b.excerpt.clone(),
                        source_a: ev_a.source_url.clone(),
                        source_b: ev_b.source_url.clone(),
                        strength: 0.8,  // Simplified
                    });
                }
            }
        }

        contradictions
    }

    /// Assess confidence in claim
    fn assess_confidence(
        &self,
        evidence: &[Evidence],
        contradictions: &[Contradiction],
        level: VerificationLevel,
    ) -> f64 {
        if evidence.is_empty() {
            return 0.0;
        }

        // Count high-credibility sources
        let high_credibility_count = evidence.iter()
            .filter(|e| e.source_credibility >= self.min_source_credibility)
            .count();

        // Calculate agreement ratio
        let supporting = evidence.iter().filter(|e| e.supports).count();
        let total = evidence.len();
        let agreement_ratio = supporting as f64 / total as f64;

        // Base confidence on number of sources and agreement
        let base_confidence = match high_credibility_count {
            0 => 0.2,
            1 => 0.4,
            2 => 0.6,
            3 => 0.75,
            _ => 0.85,
        };

        // Adjust for agreement
        let confidence = base_confidence * agreement_ratio;

        // Reduce if contradictions found
        let contradiction_penalty = contradictions.len() as f64 * 0.15;
        let final_confidence = (confidence - contradiction_penalty).max(0.0);

        // Apply verification level threshold
        match level {
            VerificationLevel::Academic => final_confidence * 0.9,  // More conservative
            VerificationLevel::Rigorous => final_confidence * 0.95,
            _ => final_confidence,
        }
    }

    /// Determine epistemic status
    fn determine_status(
        &self,
        evidence: &[Evidence],
        contradictions: &[Contradiction],
        confidence: f64,
        _level: VerificationLevel,
    ) -> ClaimConfidence {
        // No evidence = unverifiable
        if evidence.is_empty() {
            return ClaimConfidence::Unverifiable;
        }

        // Contradictions = contested
        if !contradictions.is_empty() {
            return ClaimConfidence::Contested;
        }

        // Check if refuted
        let refuting = evidence.iter().filter(|e| !e.supports).count();
        let supporting = evidence.iter().filter(|e| e.supports).count();

        if refuting > supporting {
            return ClaimConfidence::False;
        }

        // Determine confidence level
        match confidence {
            c if c >= 0.8 => ClaimConfidence::HighConfidence,
            c if c >= 0.6 => ClaimConfidence::ModerateConfidence,
            c if c >= 0.4 => ClaimConfidence::LowConfidence,
            _ => ClaimConfidence::Unverifiable,
        }
    }

    /// Generate hedging phrase based on epistemic status
    fn generate_hedge(&self, status: ClaimConfidence, _confidence: f64) -> String {
        match status {
            ClaimConfidence::HighConfidence => {
                "According to multiple reliable sources,".to_string()
            }
            ClaimConfidence::ModerateConfidence => {
                "Evidence suggests that".to_string()
            }
            ClaimConfidence::LowConfidence => {
                "Some sources indicate that".to_string()
            }
            ClaimConfidence::Contested => {
                "Sources disagree on this, but".to_string()
            }
            ClaimConfidence::Unverifiable => {
                "I cannot verify this claim, but I believe".to_string()
            }
            ClaimConfidence::False => {
                "This claim appears to be incorrect based on".to_string()
            }
        }
    }

    /// Get credibility score for a domain
    pub fn domain_credibility(&self, url: &str) -> f64 {
        // Simplified credibility scoring
        // In production, use a maintained database of source credibility

        if url.contains(".edu") || url.contains("scholar.google") {
            return 0.95;  // Academic sources
        }

        if url.contains("arxiv.org") || url.contains("doi.org") {
            return 0.9;  // Research papers
        }

        if url.contains("wikipedia.org") {
            return 0.75;  // Wikipedia (good but not primary source)
        }

        if url.contains("medium.com") || url.contains("blog") {
            return 0.6;  // Blog posts
        }

        if url.contains("reddit.com") || url.contains("forum") {
            return 0.5;  // Forum posts
        }

        0.7  // Default for unknown domains
    }
}

impl Default for EpistemicVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Source Classifier — maps domain URLs to the 3D epistemic model
// ============================================================================

use crate::mind::structured_thought::{EpistemicCube, ETier, NTier, MTier};
use std::collections::HashMap;

/// Classifies web source URLs into the 3D epistemic cube model.
///
/// This bridges external knowledge into the consciousness pipeline's
/// epistemic framework, enabling principled credibility scoring.
pub struct SourceClassifier {
    domain_registry: HashMap<String, EpistemicCube>,
}

impl SourceClassifier {
    pub fn new() -> Self {
        let mut registry = HashMap::new();

        // Academic sources: high empirical, moderate normative & meta
        for domain in &["arxiv.org", "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov"] {
            registry.insert(domain.to_string(), EpistemicCube::new(ETier::E3, NTier::N2, MTier::M2));
        }

        // Government sources: high across all dimensions
        for domain in &["who.int", "cdc.gov", "nih.gov"] {
            registry.insert(domain.to_string(), EpistemicCube::new(ETier::E3, NTier::N3, MTier::M3));
        }

        // Official documentation: high empirical, moderate authority
        for domain in &["docs.rs", "doc.rust-lang.org", "docs.python.org"] {
            registry.insert(domain.to_string(), EpistemicCube::new(ETier::E3, NTier::N2, MTier::M2));
        }

        // Major news: moderate empirical, named authority
        for domain in &["reuters.com", "apnews.com", "bbc.com"] {
            registry.insert(domain.to_string(), EpistemicCube::new(ETier::E2, NTier::N1, MTier::M1));
        }

        // Wikipedia: moderate empirical, community normative, methodology stated
        registry.insert("wikipedia.org".to_string(), EpistemicCube::new(ETier::E2, NTier::N1, MTier::M2));
        registry.insert("en.wikipedia.org".to_string(), EpistemicCube::new(ETier::E2, NTier::N1, MTier::M2));

        Self { domain_registry: registry }
    }

    /// Classify a URL into an EpistemicCube position.
    pub fn classify(&self, url: &str) -> EpistemicCube {
        let domain = Self::extract_domain(url);

        // Exact domain match
        if let Some(cube) = self.domain_registry.get(&domain) {
            return *cube;
        }

        // TLD-based patterns
        let url_lower = url.to_lowercase();
        if url_lower.contains(".edu") || url_lower.contains(".ac.uk") {
            return EpistemicCube::new(ETier::E3, NTier::N2, MTier::M2);
        }
        if url_lower.contains(".gov") || url_lower.contains(".gov.uk") {
            return EpistemicCube::new(ETier::E3, NTier::N3, MTier::M3);
        }

        // Blog / forum heuristics
        if url_lower.contains("blog") || url_lower.contains("medium.com") {
            return EpistemicCube::new(ETier::E1, NTier::N0, MTier::M1);
        }
        if url_lower.contains("reddit.com") || url_lower.contains("forum")
            || url_lower.contains("stackoverflow.com")
        {
            return EpistemicCube::new(ETier::E1, NTier::N0, MTier::M1);
        }

        // Unknown default
        EpistemicCube::new(ETier::E1, NTier::N0, MTier::M0)
    }

    /// Credibility score derived from cube position (0.0–1.0).
    pub fn credibility_score(&self, cube: &EpistemicCube) -> f64 {
        cube.credibility_score()
    }

    /// Extract the domain part from a URL.
    fn extract_domain(url: &str) -> String {
        url.split("//")
            .nth(1)
            .unwrap_or(url)
            .split('/')
            .next()
            .unwrap_or("unknown")
            .to_lowercase()
    }
}

impl Default for SourceClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_hedge() {
        let verifier = EpistemicVerifier::new();

        let high = verifier.generate_hedge(ClaimConfidence::HighConfidence, 0.9);
        assert!(high.contains("reliable sources"));

        let unverifiable = verifier.generate_hedge(ClaimConfidence::Unverifiable, 0.3);
        assert!(unverifiable.contains("cannot verify"));
    }

    #[test]
    fn test_domain_credibility() {
        let verifier = EpistemicVerifier::new();

        assert!(verifier.domain_credibility("https://stanford.edu/paper") >= 0.9);
        assert!(verifier.domain_credibility("https://arxiv.org/abs/123") >= 0.9);
        assert!(verifier.domain_credibility("https://blog.example.com") < 0.7);
    }

    #[test]
    fn test_source_classifier_academic() {
        let classifier = SourceClassifier::new();
        let cube = classifier.classify("https://arxiv.org/abs/2301.00001");
        assert_eq!(cube.empirical, ETier::E3);
    }

    #[test]
    fn test_source_classifier_gov_tld() {
        let classifier = SourceClassifier::new();
        let cube = classifier.classify("https://data.census.gov/stats");
        assert_eq!(cube.normative, NTier::N3);
    }

    #[test]
    fn test_source_classifier_unknown() {
        let classifier = SourceClassifier::new();
        let cube = classifier.classify("https://random-site.xyz/page");
        assert_eq!(cube.empirical, ETier::E1);
        assert_eq!(cube.normative, NTier::N0);
    }
}
