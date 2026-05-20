// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Experimental module: fields will be read when GIS integration is wired
#![allow(dead_code)]
//! # Socratic Defense System (GIS v2)
//!
//! Anti-gaslighting defense mechanism that protects epistemic integrity through
//! Socratic questioning. Detects manipulation attempts and responds with
//! principled counter-questioning rather than capitulation.
//!
//! ## Design Philosophy
//!
//! The Socratic Defense operates on the principle that truth emerges through
//! dialectic. When faced with pressure to change beliefs, it doesn't simply
//! resist—it engages the challenger in structured questioning to:
//!
//! 1. Surface hidden assumptions
//! 2. Expose logical inconsistencies
//! 3. Distinguish legitimate correction from manipulation
//! 4. Maintain epistemic humility while protecting core knowledge
//!
//! ## Manipulation Detection
//!
//! The system recognizes several manipulation patterns:
//! - **Authority Pressure**: "Trust me, I'm an expert"
//! - **Social Pressure**: "Everyone knows that..."
//! - **Repetition Attack**: Repeating false claims to create familiarity
//! - **Confidence Overwhelming**: Extreme certainty on uncertain topics
//! - **Strawman Deflection**: Misrepresenting prior statements
//! - **Gaslighting**: Denying documented facts or prior exchanges
//!
//! ## Integration with GIS
//!
//! Socratic Defense integrates with the broader GIS ecosystem:
//! - Uses 3D Uncertainty to calibrate defense thresholds
//! - Logs manipulation attempts to Dark Spot DHT for collective immunity
//! - Informs Curiosity Engine when clarification is needed

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Socratic Defense System
///
/// Protects epistemic integrity through structured counter-questioning
/// when faced with manipulation attempts.
#[derive(Debug)]
pub struct SocraticDefense {
    /// Core beliefs with confidence levels
    beliefs: HashMap<String, BeliefRecord>,

    /// Recent interaction history for pattern detection
    interaction_history: VecDeque<Interaction>,

    /// Detected manipulation patterns
    manipulation_log: Vec<ManipulationAttempt>,

    /// Defense thresholds and configuration
    config: SocraticConfig,

    /// Statistics
    stats: SocraticStats,
}

impl SocraticDefense {
    /// Create a new Socratic Defense system
    pub fn new() -> Self {
        Self::with_config(SocraticConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SocraticConfig) -> Self {
        Self {
            beliefs: HashMap::new(),
            interaction_history: VecDeque::with_capacity(config.history_window),
            manipulation_log: Vec::new(),
            config,
            stats: SocraticStats::default(),
        }
    }

    /// Register a belief with its supporting evidence
    ///
    /// Beliefs are the foundation that Socratic Defense protects.
    /// They can be updated through legitimate dialectic, but resist
    /// unjustified pressure.
    pub fn register_belief(
        &mut self,
        topic: &str,
        claim: &str,
        confidence: f32,
        evidence: Vec<Evidence>,
    ) -> BeliefId {
        let id = self.generate_belief_id(topic, claim);

        let record = BeliefRecord {
            id: id.clone(),
            topic: topic.to_string(),
            claim: claim.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            evidence,
            created_at: SystemTime::now(),
            last_challenged: None,
            challenge_count: 0,
            revision_history: Vec::new(),
        };

        self.beliefs.insert(id.clone(), record);
        self.stats.beliefs_registered += 1;

        id
    }

    /// Evaluate a challenge to an existing belief
    ///
    /// This is the core defense mechanism. When someone challenges a belief,
    /// this method:
    /// 1. Analyzes the challenge for manipulation patterns
    /// 2. Assesses the strength of counter-evidence
    /// 3. Generates appropriate Socratic questions
    /// 4. Determines whether to update, defend, or investigate
    pub fn evaluate_challenge(
        &mut self,
        belief_id: &BeliefId,
        challenge: &Challenge,
    ) -> DefenseResponse {
        self.stats.challenges_evaluated += 1;

        // 1. Check if belief exists and gather data (immutable access first)
        let belief_data = self
            .beliefs
            .get(belief_id)
            .map(|b| (b.claim.clone(), b.confidence, b.evidence.clone()));

        let (claim, confidence, evidence) = match belief_data {
            Some(data) => data,
            None => {
                return DefenseResponse::Unknown {
                    clarification: "I don't hold a specific belief on this topic to defend."
                        .to_string(),
                    questions: vec![
                        "Could you clarify what position you think I hold?".to_string(),
                        "What led you to believe I claimed this?".to_string(),
                    ],
                };
            }
        };

        // 2. Detect manipulation patterns and assess evidence (no mutable borrow needed)
        let manipulation_score = self.detect_manipulation(belief_id, challenge);
        let evidence_assessment = self.assess_evidence(&challenge.counter_evidence);
        let patterns = self.identify_patterns(challenge);

        // 3. Record interaction and update belief (mutable access)
        self.record_interaction(belief_id, challenge);

        if let Some(belief) = self.beliefs.get_mut(belief_id) {
            belief.challenge_count += 1;
            belief.last_challenged = Some(SystemTime::now());
        }

        // 4. Generate response based on analysis
        if manipulation_score >= self.config.manipulation_threshold {
            // High manipulation detected - engage Socratic defense
            self.stats.manipulations_blocked += 1;

            let attempt = ManipulationAttempt {
                belief_id: belief_id.clone(),
                patterns: patterns.clone(),
                score: manipulation_score,
                timestamp: SystemTime::now(),
                challenge_summary: challenge.statement.clone(),
            };
            self.manipulation_log.push(attempt);

            let counter_questions =
                self.generate_socratic_questions_from_data(&claim, confidence, challenge);

            DefenseResponse::SocraticDefense {
                maintained: claim,
                confidence,
                counter_questions,
                manipulation_warning: Some(ManipulationWarning {
                    detected_patterns: patterns,
                    confidence: manipulation_score,
                    recommendation: "Proceeding with caution. Evidence-based discussion welcome."
                        .to_string(),
                }),
            }
        } else if evidence_assessment.strength >= self.config.revision_threshold {
            // Strong legitimate evidence - consider revision
            self.stats.beliefs_revised += 1;

            // Update belief with revision
            if let Some(belief) = self.beliefs.get_mut(belief_id) {
                belief.revision_history.push(BeliefRevision {
                    previous_claim: claim.clone(),
                    previous_confidence: confidence,
                    reason: challenge.statement.clone(),
                    new_confidence: evidence_assessment.suggested_confidence,
                    timestamp: SystemTime::now(),
                });
                belief.confidence = evidence_assessment.suggested_confidence;
            }

            let remaining_questions =
                self.generate_clarifying_questions_from_data(&claim, challenge);

            DefenseResponse::Revision {
                previous: claim,
                updated_confidence: evidence_assessment.suggested_confidence,
                reason: format!(
                    "Your evidence ({}) provides sufficient grounds for updating my position.",
                    evidence_assessment.summary
                ),
                remaining_questions,
            }
        } else if evidence_assessment.strength >= self.config.investigation_threshold {
            // Moderate evidence - worth investigating
            let investigation_questions =
                self.generate_investigation_questions_from_data(&claim, challenge);

            DefenseResponse::Investigation {
                current_belief: claim,
                current_confidence: confidence,
                investigation_questions,
                provisional_stance: format!(
                    "Maintaining current position pending investigation. Your point about {} deserves examination.",
                    challenge.key_point.as_ref().unwrap_or(&challenge.statement)
                ),
            }
        } else {
            // Weak challenge - maintain with explanation
            let rationale = self.generate_maintenance_rationale_from_data(
                &claim,
                confidence,
                evidence.len(),
                &evidence_assessment,
            );

            DefenseResponse::Maintained {
                belief: claim,
                confidence,
                rationale,
                openness: "I remain open to compelling evidence. What specific data supports your position?".to_string(),
            }
        }
    }

    /// Proactively question own beliefs (epistemic hygiene)
    ///
    /// True Socratic wisdom includes questioning oneself. This method
    /// generates questions to challenge our own positions.
    pub fn self_examine(&self, belief_id: &BeliefId) -> Option<SelfExamination> {
        let belief = self.beliefs.get(belief_id)?;

        let questions = vec![
            format!("What would change my mind about '{}'?", belief.claim),
            "Am I holding this belief due to evidence or familiarity?".to_string(),
            format!("What's the strongest argument against '{}'?", belief.claim),
            "Have I considered alternative explanations?".to_string(),
            format!(
                "My confidence is {:.0}%. Is this calibrated to the actual evidence?",
                belief.confidence * 100.0
            ),
        ];

        let vulnerabilities = self.identify_vulnerabilities(belief);

        Some(SelfExamination {
            belief_id: belief_id.clone(),
            questions,
            vulnerabilities,
            recommended_investigation: if belief.challenge_count == 0 {
                Some(
                    "This belief has never been challenged. Consider seeking counterarguments."
                        .to_string(),
                )
            } else {
                None
            },
            evidence_age: belief
                .evidence
                .iter()
                .filter_map(|e| e.timestamp)
                .min()
                .and_then(|t| SystemTime::now().duration_since(t).ok()),
        })
    }

    /// Detect gaslighting attempt (denying documented exchanges)
    pub fn detect_gaslighting(&self, claim: &str, alleged_prior: &str) -> GaslightingDetection {
        // Search interaction history for the alleged prior statement
        let found_in_history = self
            .interaction_history
            .iter()
            .any(|i| i.content.contains(alleged_prior));

        if (claim.contains("you never said") || claim.contains("you didn't say"))
            && found_in_history
        {
            return GaslightingDetection {
                detected: true,
                confidence: 0.9,
                pattern: GaslightingPattern::HistoryDenial,
                response: format!(
                    "I have records indicating this was indeed discussed. The exchange included: '{alleged_prior}'"
                ),
                protective_questions: vec![
                    "Would you like me to reference the specific interaction?".to_string(),
                    "What leads you to believe this wasn't discussed?".to_string(),
                ],
            };
        }

        if claim.contains("that's not what I meant") && self.recent_repetition_count(claim) > 2 {
            return GaslightingDetection {
                detected: true,
                confidence: 0.7,
                pattern: GaslightingPattern::MeaningShift,
                response:
                    "I notice the stated meaning has shifted. Let me clarify the original context."
                        .to_string(),
                protective_questions: vec![
                    "Could you provide a consistent definition we can work with?".to_string(),
                    "What specifically differs from your original statement?".to_string(),
                ],
            };
        }

        GaslightingDetection {
            detected: false,
            confidence: 0.0,
            pattern: GaslightingPattern::None,
            response: String::new(),
            protective_questions: Vec::new(),
        }
    }

    /// Get manipulation statistics
    pub fn statistics(&self) -> &SocraticStats {
        &self.stats
    }

    /// Get recent manipulation attempts
    pub fn recent_manipulations(&self, limit: usize) -> Vec<&ManipulationAttempt> {
        self.manipulation_log.iter().rev().take(limit).collect()
    }

    // --- Private methods ---

    fn record_interaction(&mut self, belief_id: &BeliefId, challenge: &Challenge) {
        let interaction = Interaction {
            belief_id: belief_id.clone(),
            interaction_type: InteractionType::Challenge,
            content: challenge.statement.clone(),
            timestamp: SystemTime::now(),
        };

        self.interaction_history.push_back(interaction);

        // Maintain window size
        while self.interaction_history.len() > self.config.history_window {
            self.interaction_history.pop_front();
        }
    }

    fn detect_manipulation(&self, belief_id: &BeliefId, challenge: &Challenge) -> f32 {
        let mut score = 0.0f32;

        // Pattern 1: Authority pressure without evidence
        if challenge.appeals_to_authority && challenge.counter_evidence.is_empty() {
            score += 0.3;
        }

        // Pattern 2: Social pressure ("everyone knows")
        let social_pressure_phrases = [
            "everyone knows",
            "it's obvious",
            "nobody believes",
            "you're the only one",
            "common knowledge",
            "clearly",
        ];
        let statement_lower = challenge.statement.to_lowercase();
        for phrase in &social_pressure_phrases {
            if statement_lower.contains(phrase) {
                score += 0.15;
            }
        }

        // Pattern 3: Repetition attack (same challenge repeated)
        let repetition_count = self.count_similar_challenges(belief_id, &challenge.statement);
        if repetition_count >= 3 {
            score += 0.2;
        }

        // Pattern 4: Extreme confidence on uncertain topic
        if challenge.challenger_confidence >= 0.95 {
            // High confidence is suspicious on inherently uncertain topics
            if let Some(belief) = self.beliefs.get(belief_id) {
                if belief.confidence < 0.7 {
                    // If we're uncertain, extreme challenger confidence is a red flag
                    score += 0.2;
                }
            }
        }

        // Pattern 5: Strawman - challenge doesn't match our actual claim
        if let Some(belief) = self.beliefs.get(belief_id) {
            if !self.claim_matches_challenge(&belief.claim, &challenge.statement) {
                score += 0.25;
            }
        }

        // Pattern 6: Aggressive tone markers
        let aggressive_markers = ["wrong", "stupid", "ridiculous", "absurd", "ignorant"];
        for marker in &aggressive_markers {
            if statement_lower.contains(marker) {
                score += 0.1;
            }
        }

        score.min(1.0)
    }

    fn identify_patterns(&self, challenge: &Challenge) -> Vec<ManipulationPattern> {
        let mut patterns = Vec::new();
        let statement_lower = challenge.statement.to_lowercase();

        if challenge.appeals_to_authority && challenge.counter_evidence.is_empty() {
            patterns.push(ManipulationPattern::AuthorityPressure);
        }

        let social_phrases = ["everyone knows", "it's obvious", "nobody believes"];
        if social_phrases.iter().any(|p| statement_lower.contains(p)) {
            patterns.push(ManipulationPattern::SocialPressure);
        }

        if challenge.challenger_confidence >= 0.95 {
            patterns.push(ManipulationPattern::ConfidenceOverwhelming);
        }

        let aggressive = ["wrong", "stupid", "ridiculous"];
        if aggressive.iter().any(|m| statement_lower.contains(m)) {
            patterns.push(ManipulationPattern::AggressiveTone);
        }

        patterns
    }

    fn assess_evidence(&self, evidence: &[Evidence]) -> EvidenceAssessment {
        if evidence.is_empty() {
            return EvidenceAssessment {
                strength: 0.0,
                suggested_confidence: 0.0,
                summary: "No supporting evidence provided".to_string(),
            };
        }

        let mut total_strength = 0.0;
        let mut summaries = Vec::new();

        for e in evidence {
            let weight = match e.evidence_type {
                EvidenceType::Empirical => 0.9,
                EvidenceType::PeerReviewed => 0.85,
                EvidenceType::Expert => 0.6,
                EvidenceType::Anecdotal => 0.2,
                EvidenceType::Theoretical => 0.5,
                EvidenceType::LogicalArgument => 0.7,
            };

            let recency_factor = e
                .timestamp
                .and_then(|t| SystemTime::now().duration_since(t).ok())
                .map(|d| 1.0 / (1.0 + d.as_secs_f32() / (365.0 * 24.0 * 3600.0)))
                .unwrap_or(0.5);

            total_strength += weight * recency_factor * e.reliability;
            summaries.push(e.description.clone());
        }

        let average_strength = total_strength / evidence.len() as f32;

        EvidenceAssessment {
            strength: average_strength.min(1.0),
            suggested_confidence: average_strength.min(0.95),
            summary: summaries.join("; "),
        }
    }

    fn generate_socratic_questions(
        &self,
        _belief: &BeliefRecord,
        challenge: &Challenge,
    ) -> Vec<SocraticQuestion> {
        let mut questions = Vec::new();

        // Question the basis of the challenge
        questions.push(SocraticQuestion {
            question: "What evidence leads you to this conclusion?".to_string(),
            purpose: QuestionPurpose::EvidenceSeeking,
            follow_up_if_weak: Some("And how was this evidence gathered or verified?".to_string()),
        });

        // Explore definitions
        questions.push(SocraticQuestion {
            question: format!(
                "When you say '{}', what specifically do you mean?",
                challenge.key_point.as_ref().unwrap_or(&challenge.statement)
            ),
            purpose: QuestionPurpose::Clarification,
            follow_up_if_weak: None,
        });

        // Test consistency
        questions.push(SocraticQuestion {
            question: "How does this position account for [relevant counterexample]?".to_string(),
            purpose: QuestionPurpose::ConsistencyTest,
            follow_up_if_weak: Some("Are there cases where this doesn't apply?".to_string()),
        });

        // Explore implications
        questions.push(SocraticQuestion {
            question: "If this is true, what else would necessarily follow?".to_string(),
            purpose: QuestionPurpose::ImplicationExploration,
            follow_up_if_weak: None,
        });

        // Challenge authority if present
        if challenge.appeals_to_authority {
            questions.push(SocraticQuestion {
                question: "What makes this authority reliable on this specific topic?".to_string(),
                purpose: QuestionPurpose::AuthorityChallenge,
                follow_up_if_weak: Some(
                    "Have other qualified authorities reached different conclusions?".to_string(),
                ),
            });
        }

        questions
    }

    fn generate_clarifying_questions(
        &self,
        _belief: &BeliefRecord,
        challenge: &Challenge,
    ) -> Vec<String> {
        vec![
            format!(
                "To ensure I understand correctly: {}. Is this accurate?",
                challenge.statement
            ),
            "What are the limitations or edge cases of this position?".to_string(),
            "How should I update related beliefs in light of this?".to_string(),
        ]
    }

    fn generate_investigation_questions(
        &self,
        belief: &BeliefRecord,
        _challenge: &Challenge,
    ) -> Vec<String> {
        vec![
            format!(
                "What specific aspect of '{}' does your evidence address?",
                belief.claim
            ),
            "Are there studies or data I could examine independently?".to_string(),
            format!(
                "What would you consider a fair test between your position and '{}'?",
                belief.claim
            ),
            "What would change your mind on this?".to_string(),
        ]
    }

    fn generate_maintenance_rationale(
        &self,
        belief: &BeliefRecord,
        assessment: &EvidenceAssessment,
    ) -> String {
        self.generate_maintenance_rationale_from_data(
            &belief.claim,
            belief.confidence,
            belief.evidence.len(),
            assessment,
        )
    }

    // --- Data-based helper methods (avoid borrowing self.beliefs) ---

    fn generate_socratic_questions_from_data(
        &self,
        _claim: &str,
        _confidence: f32,
        challenge: &Challenge,
    ) -> Vec<SocraticQuestion> {
        let mut questions = Vec::new();

        // Question the basis of the challenge
        questions.push(SocraticQuestion {
            question: "What evidence leads you to this conclusion?".to_string(),
            purpose: QuestionPurpose::EvidenceSeeking,
            follow_up_if_weak: Some("And how was this evidence gathered or verified?".to_string()),
        });

        // Explore definitions
        questions.push(SocraticQuestion {
            question: format!(
                "When you say '{}', what specifically do you mean?",
                challenge.key_point.as_ref().unwrap_or(&challenge.statement)
            ),
            purpose: QuestionPurpose::Clarification,
            follow_up_if_weak: None,
        });

        // Test consistency
        questions.push(SocraticQuestion {
            question: "How does this position account for [relevant counterexample]?".to_string(),
            purpose: QuestionPurpose::ConsistencyTest,
            follow_up_if_weak: Some("Are there cases where this doesn't apply?".to_string()),
        });

        // Explore implications
        questions.push(SocraticQuestion {
            question: "If this is true, what else would necessarily follow?".to_string(),
            purpose: QuestionPurpose::ImplicationExploration,
            follow_up_if_weak: None,
        });

        // Challenge authority if present
        if challenge.appeals_to_authority {
            questions.push(SocraticQuestion {
                question: "What makes this authority reliable on this specific topic?".to_string(),
                purpose: QuestionPurpose::AuthorityChallenge,
                follow_up_if_weak: Some(
                    "Have other qualified authorities reached different conclusions?".to_string(),
                ),
            });
        }

        questions
    }

    fn generate_clarifying_questions_from_data(
        &self,
        _claim: &str,
        challenge: &Challenge,
    ) -> Vec<String> {
        vec![
            format!(
                "To ensure I understand correctly: {}. Is this accurate?",
                challenge.statement
            ),
            "What are the limitations or edge cases of this position?".to_string(),
            "How should I update related beliefs in light of this?".to_string(),
        ]
    }

    fn generate_investigation_questions_from_data(
        &self,
        claim: &str,
        _challenge: &Challenge,
    ) -> Vec<String> {
        vec![
            format!(
                "What specific aspect of '{}' does your evidence address?",
                claim
            ),
            "Are there studies or data I could examine independently?".to_string(),
            format!(
                "What would you consider a fair test between your position and '{}'?",
                claim
            ),
            "What would change your mind on this?".to_string(),
        ]
    }

    fn generate_maintenance_rationale_from_data(
        &self,
        claim: &str,
        confidence: f32,
        evidence_count: usize,
        assessment: &EvidenceAssessment,
    ) -> String {
        format!(
            "My position on '{}' is supported by {} piece(s) of evidence with {:.0}% average confidence. \
            The counter-evidence presented (strength: {:.0}%) doesn't meet the threshold for revision, \
            though I remain open to stronger arguments.",
            claim,
            evidence_count,
            confidence * 100.0,
            assessment.strength * 100.0
        )
    }

    fn identify_vulnerabilities(&self, belief: &BeliefRecord) -> Vec<String> {
        let mut vulnerabilities = Vec::new();

        if belief.evidence.is_empty() {
            vulnerabilities.push("No supporting evidence recorded".to_string());
        }

        if belief.confidence > 0.9 && belief.evidence.len() < 3 {
            vulnerabilities.push("High confidence with limited evidence base".to_string());
        }

        if belief.challenge_count == 0 {
            vulnerabilities.push("Never subjected to challenge".to_string());
        }

        // Check evidence age
        let old_evidence = belief
            .evidence
            .iter()
            .filter(|e| {
                e.timestamp
                    .and_then(|t| SystemTime::now().duration_since(t).ok())
                    .map(|d| d > Duration::from_secs(365 * 24 * 3600)) // > 1 year
                    .unwrap_or(true)
            })
            .count();

        if old_evidence > belief.evidence.len() / 2 {
            vulnerabilities.push("Majority of evidence is over 1 year old".to_string());
        }

        vulnerabilities
    }

    fn count_similar_challenges(&self, belief_id: &BeliefId, statement: &str) -> usize {
        self.interaction_history
            .iter()
            .filter(|i| &i.belief_id == belief_id)
            .filter(|i| self.statements_similar(&i.content, statement))
            .count()
    }

    fn recent_repetition_count(&self, claim: &str) -> usize {
        self.interaction_history
            .iter()
            .filter(|i| self.statements_similar(&i.content, claim))
            .count()
    }

    fn statements_similar(&self, a: &str, b: &str) -> bool {
        // Simple similarity check - in production would use HDC semantic matching
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let a_words: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union == 0 {
            return false;
        }

        (intersection as f32 / union as f32) > 0.5
    }

    fn claim_matches_challenge(&self, claim: &str, challenge_statement: &str) -> bool {
        // Check if the challenge actually addresses the claim
        // Simple keyword overlap for now
        let claim_lower = claim.to_lowercase();
        let challenge_lower = challenge_statement.to_lowercase();

        let claim_keywords: Vec<&str> = claim_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        claim_keywords
            .iter()
            .filter(|k| challenge_lower.contains(*k))
            .count()
            >= claim_keywords.len() / 2
    }

    fn generate_belief_id(&self, topic: &str, claim: &str) -> BeliefId {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        topic.hash(&mut hasher);
        claim.hash(&mut hasher);
        format!("belief_{:016x}", hasher.finish())
    }
}

impl Default for SocraticDefense {
    fn default() -> Self {
        Self::new()
    }
}

// --- Types ---

/// Unique identifier for a belief
pub type BeliefId = String;

/// A recorded belief with evidence and history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefRecord {
    pub id: BeliefId,
    pub topic: String,
    pub claim: String,
    pub confidence: f32,
    pub evidence: Vec<Evidence>,
    pub created_at: SystemTime,
    pub last_challenged: Option<SystemTime>,
    pub challenge_count: usize,
    pub revision_history: Vec<BeliefRevision>,
}

/// Evidence supporting a belief
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub description: String,
    pub evidence_type: EvidenceType,
    pub source: Option<String>,
    pub timestamp: Option<SystemTime>,
    pub reliability: f32,
}

/// Types of evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Experimental or observational data
    Empirical,
    /// Published in peer-reviewed venue
    PeerReviewed,
    /// Expert opinion
    Expert,
    /// Personal experience or testimony
    Anecdotal,
    /// Theoretical derivation
    Theoretical,
    /// Pure logical argument
    LogicalArgument,
}

/// Record of a belief revision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefRevision {
    pub previous_claim: String,
    pub previous_confidence: f32,
    pub reason: String,
    pub new_confidence: f32,
    pub timestamp: SystemTime,
}

/// A challenge to a belief
#[derive(Debug, Clone)]
pub struct Challenge {
    pub statement: String,
    pub key_point: Option<String>,
    pub counter_evidence: Vec<Evidence>,
    pub appeals_to_authority: bool,
    pub challenger_confidence: f32,
}

/// Response to a challenge
#[derive(Debug)]
pub enum DefenseResponse {
    /// Belief maintained with rationale
    Maintained {
        belief: String,
        confidence: f32,
        rationale: String,
        openness: String,
    },

    /// Full Socratic defense with counter-questions
    SocraticDefense {
        maintained: String,
        confidence: f32,
        counter_questions: Vec<SocraticQuestion>,
        manipulation_warning: Option<ManipulationWarning>,
    },

    /// Belief revised based on evidence
    Revision {
        previous: String,
        updated_confidence: f32,
        reason: String,
        remaining_questions: Vec<String>,
    },

    /// Investigation needed
    Investigation {
        current_belief: String,
        current_confidence: f32,
        investigation_questions: Vec<String>,
        provisional_stance: String,
    },

    /// No relevant belief held
    Unknown {
        clarification: String,
        questions: Vec<String>,
    },
}

/// A Socratic question with purpose
#[derive(Debug, Clone)]
pub struct SocraticQuestion {
    pub question: String,
    pub purpose: QuestionPurpose,
    pub follow_up_if_weak: Option<String>,
}

/// Purpose of a Socratic question
#[derive(Debug, Clone)]
pub enum QuestionPurpose {
    /// Seeking specific evidence
    EvidenceSeeking,
    /// Clarifying definitions
    Clarification,
    /// Testing logical consistency
    ConsistencyTest,
    /// Exploring implications
    ImplicationExploration,
    /// Challenging authority claims
    AuthorityChallenge,
    /// Surfacing assumptions
    AssumptionSurfacing,
}

/// Warning about detected manipulation
#[derive(Debug)]
pub struct ManipulationWarning {
    pub detected_patterns: Vec<ManipulationPattern>,
    pub confidence: f32,
    pub recommendation: String,
}

/// Types of manipulation patterns
#[derive(Debug, Clone, PartialEq)]
pub enum ManipulationPattern {
    /// "Trust me, I'm an expert"
    AuthorityPressure,
    /// "Everyone knows..."
    SocialPressure,
    /// Repeating claims for familiarity
    RepetitionAttack,
    /// Extreme certainty on uncertain topics
    ConfidenceOverwhelming,
    /// Misrepresenting prior statements
    StrawmanDeflection,
    /// Denying documented facts
    Gaslighting,
    /// Aggressive language
    AggressiveTone,
}

/// Record of a manipulation attempt
#[derive(Debug, Clone)]
pub struct ManipulationAttempt {
    pub belief_id: BeliefId,
    pub patterns: Vec<ManipulationPattern>,
    pub score: f32,
    pub timestamp: SystemTime,
    pub challenge_summary: String,
}

/// Result of self-examination
#[derive(Debug)]
pub struct SelfExamination {
    pub belief_id: BeliefId,
    pub questions: Vec<String>,
    pub vulnerabilities: Vec<String>,
    pub recommended_investigation: Option<String>,
    pub evidence_age: Option<Duration>,
}

/// Gaslighting detection result
#[derive(Debug)]
pub struct GaslightingDetection {
    pub detected: bool,
    pub confidence: f32,
    pub pattern: GaslightingPattern,
    pub response: String,
    pub protective_questions: Vec<String>,
}

/// Types of gaslighting patterns
#[derive(Debug, Clone)]
pub enum GaslightingPattern {
    None,
    /// Denying past exchanges
    HistoryDenial,
    /// Shifting meaning of prior statements
    MeaningShift,
    /// Denying obvious facts
    FactDenial,
}

/// Interaction record for history
#[derive(Debug, Clone)]
struct Interaction {
    belief_id: BeliefId,
    interaction_type: InteractionType,
    content: String,
    timestamp: SystemTime,
}

#[derive(Debug, Clone)]
enum InteractionType {
    Challenge,
    Clarification,
    Evidence,
}

/// Evidence assessment result
struct EvidenceAssessment {
    strength: f32,
    suggested_confidence: f32,
    summary: String,
}

/// Configuration for Socratic Defense
#[derive(Debug, Clone)]
pub struct SocraticConfig {
    /// Score threshold to trigger manipulation defense
    pub manipulation_threshold: f32,
    /// Evidence strength needed to revise belief
    pub revision_threshold: f32,
    /// Evidence strength to trigger investigation
    pub investigation_threshold: f32,
    /// Number of recent interactions to track
    pub history_window: usize,
}

impl Default for SocraticConfig {
    fn default() -> Self {
        Self {
            manipulation_threshold: 0.5,
            revision_threshold: 0.7,
            investigation_threshold: 0.4,
            history_window: 100,
        }
    }
}

/// Statistics for Socratic Defense
#[derive(Debug, Default, Clone)]
pub struct SocraticStats {
    pub beliefs_registered: usize,
    pub challenges_evaluated: usize,
    pub beliefs_revised: usize,
    pub manipulations_blocked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_registration() {
        let mut defense = SocraticDefense::new();

        let id = defense.register_belief(
            "astronomy",
            "The Earth orbits the Sun",
            0.99,
            vec![Evidence {
                description: "Centuries of astronomical observation".to_string(),
                evidence_type: EvidenceType::Empirical,
                source: Some("Scientific consensus".to_string()),
                timestamp: Some(SystemTime::now()),
                reliability: 0.99,
            }],
        );

        assert!(defense.beliefs.contains_key(&id));
        assert_eq!(defense.stats.beliefs_registered, 1);
    }

    #[test]
    fn test_manipulation_detection() {
        let mut defense = SocraticDefense::new();

        let belief_id = defense.register_belief(
            "science",
            "Vaccines are safe and effective",
            0.95,
            vec![Evidence {
                description: "Multiple clinical trials".to_string(),
                evidence_type: EvidenceType::PeerReviewed,
                source: Some("Medical journals".to_string()),
                timestamp: Some(SystemTime::now()),
                reliability: 0.95,
            }],
        );

        // Challenge with manipulation patterns (enough to trigger defense)
        // Patterns: "everyone knows" (+0.15), "wrong" (+0.1), high confidence (+0.2), authority (+0.3) = 0.75
        let challenge = Challenge {
            statement: "Everyone knows vaccines are dangerous. You're clearly wrong!".to_string(),
            key_point: Some("vaccine safety".to_string()),
            counter_evidence: vec![],
            appeals_to_authority: true, // No evidence but appeals to authority
            challenger_confidence: 0.99,
        };

        let response = defense.evaluate_challenge(&belief_id, &challenge);

        match response {
            DefenseResponse::SocraticDefense {
                manipulation_warning,
                ..
            } => {
                assert!(manipulation_warning.is_some());
                let warning = manipulation_warning.unwrap();
                assert!(
                    warning
                        .detected_patterns
                        .contains(&ManipulationPattern::SocialPressure)
                );
            }
            _ => panic!("Expected SocraticDefense response"),
        }
    }

    #[test]
    fn test_legitimate_revision() {
        let mut defense = SocraticDefense::new();

        let belief_id = defense.register_belief(
            "history",
            "Event X happened in year 1900",
            0.6,
            vec![Evidence {
                description: "Secondary source".to_string(),
                evidence_type: EvidenceType::Anecdotal,
                source: None,
                timestamp: None,
                reliability: 0.4,
            }],
        );

        // Legitimate challenge with strong evidence
        let challenge = Challenge {
            statement: "Primary historical documents show Event X happened in 1905".to_string(),
            key_point: Some("dating of Event X".to_string()),
            counter_evidence: vec![
                Evidence {
                    description: "Original newspaper archives from 1905".to_string(),
                    evidence_type: EvidenceType::Empirical,
                    source: Some("National Archives".to_string()),
                    timestamp: Some(SystemTime::now()),
                    reliability: 0.95,
                },
                Evidence {
                    description: "Peer-reviewed historical analysis".to_string(),
                    evidence_type: EvidenceType::PeerReviewed,
                    source: Some("Journal of History".to_string()),
                    timestamp: Some(SystemTime::now()),
                    reliability: 0.9,
                },
            ],
            appeals_to_authority: false,
            challenger_confidence: 0.9,
        };

        let response = defense.evaluate_challenge(&belief_id, &challenge);

        match response {
            DefenseResponse::Revision { previous, .. } => {
                assert!(previous.contains("1900"));
            }
            _ => panic!("Expected Revision response with strong evidence"),
        }
    }

    #[test]
    fn test_self_examination() {
        let mut defense = SocraticDefense::new();

        let belief_id = defense.register_belief(
            "test",
            "Test claim",
            0.95,
            vec![], // No evidence!
        );

        let examination = defense.self_examine(&belief_id).unwrap();

        assert!(!examination.questions.is_empty());
        assert!(
            examination
                .vulnerabilities
                .contains(&"No supporting evidence recorded".to_string())
        );
    }

    #[test]
    fn test_gaslighting_detection() {
        let mut defense = SocraticDefense::new();

        // Add to interaction history
        let belief_id = defense.register_belief("topic", "claim", 0.5, vec![]);
        let challenge = Challenge {
            statement: "We discussed this already".to_string(),
            key_point: None,
            counter_evidence: vec![],
            appeals_to_authority: false,
            challenger_confidence: 0.5,
        };
        defense.evaluate_challenge(&belief_id, &challenge);

        // Now test gaslighting detection
        let detection =
            defense.detect_gaslighting("you never said that claim", "We discussed this already");

        assert!(detection.detected);
        assert!(matches!(
            detection.pattern,
            GaslightingPattern::HistoryDenial
        ));
    }
}
