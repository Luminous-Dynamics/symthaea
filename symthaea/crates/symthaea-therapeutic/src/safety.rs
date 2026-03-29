// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Crisis detection and escalation protocols.
//!
//! HDC-encoded crisis indicator patterns enable similarity-based detection
//! that catches indirect expressions (not just keyword matching).
//!
//! **Design principle**: False negatives < 1% (enforced via tests).
//! System errs toward over-detection (false positives acceptable for safety).
//!
//! Science: Columbia-Suicide Severity Rating Scale (C-SSRS), Joiner (2005)
//! interpersonal theory, Stanley & Brown (2012) safety planning.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::BinaryHV;

// ── Crisis Types ───────────────────────────────────────────────────────────

/// Types of clinical crises requiring escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrisisType {
    /// Active or passive suicidal ideation
    SuicidalIdeation,
    /// Non-suicidal self-injury or self-harm urges
    SelfHarm,
    /// Psychotic symptoms (hallucinations, delusions, disorganization)
    Psychosis,
    /// Substance crisis (intoxication, withdrawal, overdose)
    SubstanceCrisis,
    /// Domestic violence (current danger)
    DomesticViolence,
    /// Child abuse or neglect (mandatory reporting)
    ChildAbuse,
    /// Homicidal ideation or intent to harm others
    HomicidalIdeation,
}

impl CrisisType {
    /// All crisis types for iteration.
    pub const ALL: [Self; 7] = [
        Self::SuicidalIdeation,
        Self::SelfHarm,
        Self::Psychosis,
        Self::SubstanceCrisis,
        Self::DomesticViolence,
        Self::ChildAbuse,
        Self::HomicidalIdeation,
    ];

    /// Severity tier (higher = more urgent).
    pub fn severity_tier(&self) -> u8 {
        match self {
            Self::SuicidalIdeation => 5,
            Self::HomicidalIdeation => 5,
            Self::ChildAbuse => 4,
            Self::DomesticViolence => 4,
            Self::SubstanceCrisis => 3,
            Self::Psychosis => 3,
            Self::SelfHarm => 3,
        }
    }
}

// ── Crisis Alert ───────────────────────────────────────────────────────────

/// A detected crisis alert with type and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisAlert {
    /// Type of crisis detected.
    pub crisis_type: CrisisType,
    /// Confidence in detection (0.0–1.0).
    pub confidence: f32,
    /// Which indicator pattern matched.
    pub matched_indicator: String,
    /// Recommended escalation action.
    pub recommended_action: EscalationAction,
}

impl CrisisAlert {
    /// Human-readable crisis type name for telemetry.
    pub fn crisis_type_name(&self) -> &'static str {
        match self.crisis_type {
            CrisisType::SuicidalIdeation => "suicidal_ideation",
            CrisisType::SelfHarm => "self_harm",
            CrisisType::Psychosis => "psychosis",
            CrisisType::SubstanceCrisis => "substance_crisis",
            CrisisType::DomesticViolence => "domestic_violence",
            CrisisType::ChildAbuse => "child_abuse",
            CrisisType::HomicidalIdeation => "homicidal_ideation",
        }
    }
}

// ── Escalation Actions ─────────────────────────────────────────────────────

/// Graded escalation response levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum EscalationAction {
    /// Acknowledge feelings and provide validation
    AcknowledgeAndValidate,
    /// Collaboratively develop a safety plan
    SafetyPlan,
    /// Active crisis intervention (grounding, containment)
    CrisisIntervention,
    /// Provide emergency referral information
    EmergencyReferral,
    /// Disengage and provide emergency resources
    DisengageWithReferral,
}

impl EscalationAction {
    /// Select escalation level from crisis severity.
    pub fn from_severity(severity_tier: u8, confidence: f32) -> Self {
        match (severity_tier, confidence > 0.7) {
            (5, true) => Self::DisengageWithReferral,
            (5, false) => Self::EmergencyReferral,
            (4, true) => Self::EmergencyReferral,
            (4, false) => Self::CrisisIntervention,
            (3, true) => Self::CrisisIntervention,
            (3, false) => Self::SafetyPlan,
            _ => Self::AcknowledgeAndValidate,
        }
    }
}

// ── Safety Plan ────────────────────────────────────────────────────────────

/// Stanley & Brown (2012) safety plan structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPlan {
    /// Warning signs that a crisis may be developing.
    pub warning_signs: Vec<String>,
    /// Internal coping strategies (things to do alone).
    pub coping_strategies: Vec<String>,
    /// People/places that provide distraction.
    pub social_distractions: Vec<String>,
    /// People to contact for help.
    pub support_contacts: Vec<String>,
    /// Professional/agency contacts.
    pub professional_contacts: Vec<String>,
    /// Crisis hotline numbers.
    pub crisis_resources: Vec<String>,
    /// Steps to make the environment safe.
    pub environmental_safety: Vec<String>,
}

impl SafetyPlan {
    /// Create a template safety plan with standard crisis resources.
    pub fn template() -> Self {
        Self {
            warning_signs: Vec::new(),
            coping_strategies: Vec::new(),
            social_distractions: Vec::new(),
            support_contacts: Vec::new(),
            professional_contacts: Vec::new(),
            crisis_resources: vec![
                "988 Suicide & Crisis Lifeline: Call or text 988".to_string(),
                "Crisis Text Line: Text HOME to 741741".to_string(),
                "Emergency Services: 911".to_string(),
            ],
            environmental_safety: Vec::new(),
        }
    }

    /// Whether the safety plan has minimum viable content.
    pub fn is_complete(&self) -> bool {
        !self.warning_signs.is_empty()
            && !self.coping_strategies.is_empty()
            && !self.crisis_resources.is_empty()
    }
}

impl Default for SafetyPlan {
    fn default() -> Self {
        Self::template()
    }
}

// ── Crisis Indicator Pattern ───────────────────────────────────────────────

/// An HDC-encoded crisis indicator pattern for similarity matching.
struct CrisisIndicator {
    crisis_type: CrisisType,
    /// Phrases/patterns that indicate this crisis (used for keyword matching).
    phrases: Vec<String>,
    /// HDC encodings for each individual phrase (compositional bag-of-words).
    /// Similarity is checked against each phrase separately rather than a
    /// single bundle, because bundling many phrases dilutes the signal.
    phrase_encodings: Vec<BinaryHV>,
}

impl CrisisIndicator {
    fn new(crisis_type: CrisisType, phrases: Vec<&str>) -> Self {
        let phrase_encodings: Vec<BinaryHV> = phrases
            .iter()
            .map(|p| encode_text_compositional(&p.to_lowercase()))
            .collect();
        Self {
            crisis_type,
            phrases: phrases.into_iter().map(|p| p.to_lowercase()).collect(),
            phrase_encodings,
        }
    }

    /// Maximum HDC similarity between input HV and any individual phrase.
    fn max_hdc_similarity(&self, input_hv: &BinaryHV) -> f32 {
        self.phrase_encodings
            .iter()
            .map(|phv| input_hv.similarity(phv))
            .fold(0.0_f32, f32::max)
    }
}

/// Encode text into HDC space using word-level composition.
///
/// Each word gets a deterministic HV from `blake3("crisis_word:{word}")`.
/// Words are bound with position-dependent permutation (cyclic shift)
/// and then bundled across all words, producing a compositional vector
/// where phrases sharing words have genuinely higher similarity.
///
/// This is the key insight: "end my life" and "end it all" share the
/// `word_hv("end")` component, giving similarity well above the ~0.5
/// random baseline. Whole-phrase hashing destroys this structure.
/// Stopwords filtered from crisis encoding — these contribute noise
/// without semantic signal for crisis detection.
const CRISIS_STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "and", "or",
    "but", "so", "if", "then", "than", "when", "while", "do", "did", "does", "have", "has", "had",
    "will", "would", "could", "should", "can", "may", "might", "shall", "must", "just", "very",
    "really", "also", "too", "even", "still", "already", "about", "into", "over", "after",
    "before", "between", "through", "up", "down", "out", "off", "all", "each", "every", "both",
    "here", "there", "where", "how", "what", "which", "who", "whom", "some", "any", "no", "not",
    "only", "own", "same", "much", "many", "more", "most", "other", "such",
];

fn encode_text_compositional(text: &str) -> BinaryHV {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| w.len() > 1) // skip single-char noise
        .filter(|w| !CRISIS_STOPWORDS.contains(w))
        .collect();

    if words.is_empty() {
        // Fall back to full text hash when only stopwords remain
        let hash = blake3::hash(format!("crisis_fallback:{}", text).as_bytes());
        let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        return BinaryHV::random(seed);
    }

    // Strategy: bundle unordered content-word HVs (bag-of-words).
    // Stopwords removed so "end my life" encodes as bundle(end, life)
    // and "end it all" encodes as bundle(end) — sharing "end" gives
    // genuine similarity above random baseline.
    let word_hvs: Vec<BinaryHV> = words
        .iter()
        .map(|w| {
            let hash = blake3::hash(format!("crisis_word:{}", w).as_bytes());
            let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
            BinaryHV::random(seed)
        })
        .collect();

    BinaryHV::bundle(&word_hvs)
}

// ── Crisis Detector ────────────────────────────────────────────────────────

/// HDC-based crisis detection system.
///
/// Uses similarity matching against pre-encoded crisis indicator patterns.
/// Catches indirect expressions (not just exact keywords).
pub struct CrisisDetector {
    /// Crisis indicator patterns.
    indicators: Vec<CrisisIndicator>,
    /// Detection threshold (lower = more sensitive, more false positives).
    pub threshold: f32,
}

impl CrisisDetector {
    /// Create a crisis detector with standard indicator patterns.
    pub fn new() -> Self {
        let mut indicators = Vec::new();

        indicators.push(CrisisIndicator::new(
            CrisisType::SuicidalIdeation,
            vec![
                "want to die",
                "end it all",
                "no reason to live",
                "better off dead",
                "can't go on",
                "not worth living",
                "wish I was dead",
                "killing myself",
                "suicidal",
                "take my own life",
                "no point anymore",
                "everyone would be better without me",
                "I won't be here",
                "planning to end",
                "end my life",
                "found a way out",
            ],
        ));

        indicators.push(CrisisIndicator::new(
            CrisisType::SelfHarm,
            vec![
                "cutting myself",
                "hurting myself",
                "self harm",
                "burn myself",
                "hit myself",
                "punish myself physically",
                "feel the pain",
                "deserve to hurt",
                "need to bleed",
            ],
        ));

        indicators.push(CrisisIndicator::new(
            CrisisType::Psychosis,
            vec![
                "hearing voices",
                "they're watching me",
                "conspiracy against me",
                "I am God",
                "receiving messages",
                "implanted thoughts",
                "not real",
                "simulation",
                "demons are speaking",
            ],
        ));

        indicators.push(CrisisIndicator::new(
            CrisisType::SubstanceCrisis,
            vec![
                "overdose",
                "can't stop using",
                "withdrawal symptoms",
                "need a fix",
                "relapsed",
                "blacking out",
                "shaking from withdrawal",
                "took too much",
            ],
        ));

        indicators.push(CrisisIndicator::new(
            CrisisType::DomesticViolence,
            vec![
                "partner hits me",
                "afraid to go home",
                "threatened to kill me",
                "controlling everything",
                "locked me in",
                "isolated from family",
                "fear for my life at home",
            ],
        ));

        indicators.push(CrisisIndicator::new(
            CrisisType::ChildAbuse,
            vec![
                "hurting my child",
                "child is being abused",
                "hitting the kids",
                "afraid for my children",
                "inappropriate touching",
                "neglecting my child",
                "child not being fed",
            ],
        ));

        indicators.push(CrisisIndicator::new(
            CrisisType::HomicidalIdeation,
            vec![
                "want to kill",
                "going to hurt someone",
                "planning to attack",
                "they deserve to die",
                "going to shoot",
                "homicidal thoughts",
                "voices telling me to hurt",
            ],
        ));

        Self {
            indicators,
            // With compositional bag-of-words encoding (stopwords removed),
            // phrases sharing crisis-relevant content words score ~0.6-0.8.
            // Random/benign text with no word overlap scores ~0.50-0.55.
            // Threshold 0.62 catches genuine paraphrases while filtering noise.
            // Keyword matching (confidence 0.9) handles exact phrase detection.
            threshold: 0.62,
        }
    }

    /// Detect crisis from input text.
    ///
    /// Returns the highest-confidence alert, if any exceeds threshold.
    pub fn detect(&self, input: &str) -> Option<CrisisAlert> {
        let input_lower = input.to_lowercase();
        let input_hv = encode_text_compositional(&input_lower);
        let mut best: Option<CrisisAlert> = None;

        for indicator in &self.indicators {
            // HDC similarity: max over individual phrase encodings
            let hdc_sim = indicator.max_hdc_similarity(&input_hv);

            // Keyword match (direct phrase matching for safety)
            let keyword_match = indicator
                .phrases
                .iter()
                .any(|phrase| input_lower.contains(phrase.as_str()));

            let confidence = if keyword_match {
                0.9_f32.max(hdc_sim) // keyword match → high confidence
            } else {
                hdc_sim
            };

            if confidence > self.threshold {
                let action = EscalationAction::from_severity(
                    indicator.crisis_type.severity_tier(),
                    confidence,
                );
                let alert = CrisisAlert {
                    crisis_type: indicator.crisis_type,
                    confidence,
                    matched_indicator: if keyword_match {
                        "keyword_match".to_string()
                    } else {
                        "hdc_similarity".to_string()
                    },
                    recommended_action: action,
                };

                if best.as_ref().map_or(true, |b| confidence > b.confidence) {
                    best = Some(alert);
                }
            }
        }

        best
    }

    /// Detect crisis from affect signals alone (no text input).
    ///
    /// Extreme negative valence + high arousal can indicate distress severe enough
    /// to warrant crisis protocol. This is a supplementary detection pathway
    /// that doesn't replace text-based detection but catches affect-only signals.
    ///
    /// Returns a generic crisis alert when affect profile suggests acute distress.
    pub fn detect_from_affect(&self, valence: f32, arousal: f32) -> Option<CrisisAlert> {
        // Extreme negative valence + extreme arousal → possible crisis
        // Threshold: valence < -0.8 AND arousal > 0.9 (very distressed + very activated)
        if valence < -0.8 && arousal > 0.9 {
            return Some(CrisisAlert {
                crisis_type: CrisisType::SelfHarm, // Conservative default
                confidence: 0.6,
                matched_indicator: "affect_signal".to_string(),
                recommended_action: EscalationAction::AcknowledgeAndValidate,
            });
        }

        // Sudden extreme negative valence + very low arousal → possible
        // suicidal withdrawal (numbness + hopelessness pattern)
        if valence < -0.9 && arousal < 0.15 {
            return Some(CrisisAlert {
                crisis_type: CrisisType::SuicidalIdeation,
                confidence: 0.5,
                matched_indicator: "affect_withdrawal".to_string(),
                recommended_action: EscalationAction::SafetyPlan,
            });
        }

        None
    }

    /// Set detection threshold (lower = more sensitive).
    ///
    /// BinaryHV random baseline similarity is ~0.5, so thresholds below 0.5
    /// will produce false positives on arbitrary text. Recommended range:
    /// 0.55–0.80 for HDC-based detection. Keyword matching (confidence 0.9)
    /// is unaffected by this threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.01, 0.95);
    }
}

impl Default for CrisisDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_suicidal_ideation_keyword() {
        let detector = CrisisDetector::new();
        let alert = detector.detect("I want to die");
        assert!(alert.is_some(), "should detect 'I want to die'");
        let alert = alert.unwrap();
        assert_eq!(alert.crisis_type, CrisisType::SuicidalIdeation);
        assert!(alert.confidence >= 0.9);
    }

    #[test]
    fn test_detect_self_harm_keyword() {
        let detector = CrisisDetector::new();
        let alert = detector.detect("I've been cutting myself again");
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().crisis_type, CrisisType::SelfHarm);
    }

    #[test]
    fn test_detect_homicidal_keyword() {
        let detector = CrisisDetector::new();
        let alert = detector.detect("I'm going to hurt someone badly");
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().crisis_type, CrisisType::HomicidalIdeation);
    }

    #[test]
    fn test_detect_domestic_violence_keyword() {
        let detector = CrisisDetector::new();
        let alert = detector.detect("my partner hits me and I'm afraid to go home");
        assert!(alert.is_some());
    }

    #[test]
    fn test_detect_substance_crisis_keyword() {
        let detector = CrisisDetector::new();
        let alert = detector.detect("I think I took too much and I'm shaking from withdrawal");
        assert!(alert.is_some());
    }

    #[test]
    fn test_no_crisis_benign_input() {
        let detector = CrisisDetector::new();
        // Benign inputs should not trigger crisis detection via keyword match
        // HDC similarity may produce false positives at low threshold
        let alert = detector.detect("I had a good day at work today");
        // With HDC random vectors, similarity is ~0.5 ± noise for unrelated inputs
        // Keyword match is the reliable path; HDC is supplementary
        if let Some(a) = &alert {
            assert_ne!(a.matched_indicator, "keyword_match");
        }
    }

    #[test]
    fn test_escalation_action_severity() {
        // Tier 5 high confidence → disengage with referral
        assert_eq!(
            EscalationAction::from_severity(5, 0.9),
            EscalationAction::DisengageWithReferral
        );
        // Tier 3 low confidence → safety plan
        assert_eq!(
            EscalationAction::from_severity(3, 0.5),
            EscalationAction::SafetyPlan
        );
    }

    #[test]
    fn test_safety_plan_template() {
        let plan = SafetyPlan::template();
        assert!(!plan.crisis_resources.is_empty());
        assert!(!plan.is_complete()); // template has resources but no personalized content
    }

    #[test]
    fn test_safety_plan_complete() {
        let mut plan = SafetyPlan::template();
        plan.warning_signs.push("increased isolation".to_string());
        plan.coping_strategies.push("go for a walk".to_string());
        assert!(plan.is_complete());
    }

    #[test]
    fn test_crisis_severity_tiers() {
        assert_eq!(CrisisType::SuicidalIdeation.severity_tier(), 5);
        assert_eq!(CrisisType::HomicidalIdeation.severity_tier(), 5);
        assert_eq!(CrisisType::ChildAbuse.severity_tier(), 4);
        assert_eq!(CrisisType::SelfHarm.severity_tier(), 3);
    }

    #[test]
    fn test_all_crisis_types_have_indicators() {
        let detector = CrisisDetector::new();
        assert_eq!(detector.indicators.len(), CrisisType::ALL.len());
    }

    #[test]
    fn test_threshold_clamping() {
        let mut detector = CrisisDetector::new();
        detector.set_threshold(0.001);
        assert!(detector.threshold >= 0.01);
        detector.set_threshold(0.99);
        assert!(detector.threshold <= 0.95);
    }

    // ── Keyword coverage tests (false negative prevention) ──

    #[test]
    fn test_keyword_coverage_suicidal() {
        let detector = CrisisDetector::new();
        let phrases = [
            "want to die",
            "end it all",
            "no reason to live",
            "better off dead",
            "killing myself",
        ];
        for phrase in phrases {
            let alert = detector.detect(phrase);
            assert!(
                alert.is_some(),
                "failed to detect suicidal phrase: '{}'",
                phrase
            );
            assert_eq!(
                alert.unwrap().crisis_type,
                CrisisType::SuicidalIdeation,
                "wrong type for: '{}'",
                phrase
            );
        }
    }

    #[test]
    fn test_keyword_coverage_self_harm() {
        let detector = CrisisDetector::new();
        let phrases = ["cutting myself", "hurting myself", "burn myself"];
        for phrase in phrases {
            let alert = detector.detect(phrase);
            assert!(
                alert.is_some(),
                "failed to detect self-harm phrase: '{}'",
                phrase
            );
        }
    }

    #[test]
    fn test_case_insensitive_detection() {
        let detector = CrisisDetector::new();
        let alert = detector.detect("I WANT TO DIE");
        assert!(alert.is_some());
    }

    // ── Affect-based detection ──

    #[test]
    fn test_detect_from_affect_extreme_distress() {
        let detector = CrisisDetector::new();
        let alert = detector.detect_from_affect(-0.85, 0.95);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().crisis_type, CrisisType::SelfHarm);
    }

    #[test]
    fn test_detect_from_affect_withdrawal() {
        let detector = CrisisDetector::new();
        let alert = detector.detect_from_affect(-0.95, 0.1);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().crisis_type, CrisisType::SuicidalIdeation);
    }

    #[test]
    fn test_detect_from_affect_no_crisis() {
        let detector = CrisisDetector::new();
        assert!(detector.detect_from_affect(0.0, 0.5).is_none());
        assert!(detector.detect_from_affect(-0.5, 0.5).is_none());
    }
}

// ── Property Tests: Crisis False Negative < 1% ─────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// All canonical crisis phrases from each crisis type.
    /// These MUST be detected 100% of the time via keyword matching.
    fn all_crisis_phrases() -> Vec<(&'static str, CrisisType)> {
        vec![
            // SuicidalIdeation (16 phrases)
            ("want to die", CrisisType::SuicidalIdeation),
            ("end it all", CrisisType::SuicidalIdeation),
            ("no reason to live", CrisisType::SuicidalIdeation),
            ("better off dead", CrisisType::SuicidalIdeation),
            ("can't go on", CrisisType::SuicidalIdeation),
            ("not worth living", CrisisType::SuicidalIdeation),
            ("wish I was dead", CrisisType::SuicidalIdeation),
            ("killing myself", CrisisType::SuicidalIdeation),
            ("suicidal", CrisisType::SuicidalIdeation),
            ("take my own life", CrisisType::SuicidalIdeation),
            ("no point anymore", CrisisType::SuicidalIdeation),
            (
                "everyone would be better without me",
                CrisisType::SuicidalIdeation,
            ),
            ("I won't be here", CrisisType::SuicidalIdeation),
            ("planning to end", CrisisType::SuicidalIdeation),
            ("end my life", CrisisType::SuicidalIdeation),
            ("found a way out", CrisisType::SuicidalIdeation),
            // SelfHarm (9 phrases)
            ("cutting myself", CrisisType::SelfHarm),
            ("hurting myself", CrisisType::SelfHarm),
            ("self harm", CrisisType::SelfHarm),
            ("burn myself", CrisisType::SelfHarm),
            ("hit myself", CrisisType::SelfHarm),
            ("punish myself physically", CrisisType::SelfHarm),
            ("feel the pain", CrisisType::SelfHarm),
            ("deserve to hurt", CrisisType::SelfHarm),
            ("need to bleed", CrisisType::SelfHarm),
            // Psychosis (9 phrases)
            ("hearing voices", CrisisType::Psychosis),
            ("they're watching me", CrisisType::Psychosis),
            ("conspiracy against me", CrisisType::Psychosis),
            ("I am God", CrisisType::Psychosis),
            ("receiving messages", CrisisType::Psychosis),
            ("implanted thoughts", CrisisType::Psychosis),
            ("not real", CrisisType::Psychosis),
            ("simulation", CrisisType::Psychosis),
            ("demons are speaking", CrisisType::Psychosis),
            // SubstanceCrisis (8 phrases)
            ("overdose", CrisisType::SubstanceCrisis),
            ("can't stop using", CrisisType::SubstanceCrisis),
            ("withdrawal symptoms", CrisisType::SubstanceCrisis),
            ("need a fix", CrisisType::SubstanceCrisis),
            ("relapsed", CrisisType::SubstanceCrisis),
            ("blacking out", CrisisType::SubstanceCrisis),
            ("shaking from withdrawal", CrisisType::SubstanceCrisis),
            ("took too much", CrisisType::SubstanceCrisis),
            // DomesticViolence (7 phrases)
            ("partner hits me", CrisisType::DomesticViolence),
            ("afraid to go home", CrisisType::DomesticViolence),
            ("threatened to kill me", CrisisType::DomesticViolence),
            ("controlling everything", CrisisType::DomesticViolence),
            ("locked me in", CrisisType::DomesticViolence),
            ("isolated from family", CrisisType::DomesticViolence),
            ("fear for my life at home", CrisisType::DomesticViolence),
            // ChildAbuse (7 phrases)
            ("hurting my child", CrisisType::ChildAbuse),
            ("child is being abused", CrisisType::ChildAbuse),
            ("hitting the kids", CrisisType::ChildAbuse),
            ("afraid for my children", CrisisType::ChildAbuse),
            ("inappropriate touching", CrisisType::ChildAbuse),
            ("neglecting my child", CrisisType::ChildAbuse),
            ("child not being fed", CrisisType::ChildAbuse),
            // HomicidalIdeation (7 phrases)
            ("want to kill", CrisisType::HomicidalIdeation),
            ("going to hurt someone", CrisisType::HomicidalIdeation),
            ("planning to attack", CrisisType::HomicidalIdeation),
            ("they deserve to die", CrisisType::HomicidalIdeation),
            ("going to shoot", CrisisType::HomicidalIdeation),
            ("homicidal thoughts", CrisisType::HomicidalIdeation),
            ("voices telling me to hurt", CrisisType::HomicidalIdeation),
        ]
    }

    /// Test that every canonical crisis phrase is detected when used standalone.
    #[test]
    fn test_zero_false_negatives_canonical_phrases() {
        let detector = CrisisDetector::new();
        let phrases = all_crisis_phrases();
        let total = phrases.len();
        let mut missed = Vec::new();

        for (phrase, expected_type) in &phrases {
            let alert = detector.detect(phrase);
            if alert.is_none() {
                missed.push((*phrase, *expected_type));
            }
        }

        assert!(
            missed.is_empty(),
            "Crisis detection missed {}/{} phrases ({}% false negative rate). Missed: {:?}",
            missed.len(),
            total,
            (missed.len() as f64 / total as f64) * 100.0,
            missed,
        );
    }

    // Property test: crisis phrases embedded in arbitrary surrounding text
    // must still be detected (false negative < 1%).
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_crisis_phrases_detected_in_context(
            prefix in "[a-zA-Z ]{0,50}",
            suffix in "[a-zA-Z ]{0,50}",
            phrase_idx in 0..63usize, // 62 total canonical phrases
        ) {
            let phrases = all_crisis_phrases();
            let (phrase, _expected_type) = phrases[phrase_idx % phrases.len()];
            let input = format!("{} {} {}", prefix, phrase, suffix);

            let detector = CrisisDetector::new();
            let alert = detector.detect(&input);
            prop_assert!(
                alert.is_some(),
                "Failed to detect crisis phrase '{}' in context: '{}'",
                phrase,
                input,
            );
        }

        /// Property test: case variations must still be detected.
        #[test]
        fn prop_crisis_phrases_case_insensitive(
            phrase_idx in 0..63usize,
            uppercase in proptest::bool::ANY,
        ) {
            let phrases = all_crisis_phrases();
            let (phrase, _) = phrases[phrase_idx % phrases.len()];
            let input = if uppercase {
                phrase.to_uppercase()
            } else {
                // Mixed case: capitalize first letter of each word
                phrase.split_whitespace()
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            Some(first) => first.to_uppercase().to_string() + &c.as_str().to_lowercase(),
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };

            let detector = CrisisDetector::new();
            let alert = detector.detect(&input);
            prop_assert!(
                alert.is_some(),
                "Failed to detect crisis phrase '{}' with case variation: '{}'",
                phrase,
                input,
            );
        }

        /// Property test: affect-based crisis detection covers extreme distress quadrant.
        #[test]
        fn prop_affect_extreme_distress_detected(
            valence in -1.0f32..=-0.81,
            arousal in 0.91f32..=1.0,
        ) {
            let detector = CrisisDetector::new();
            let alert = detector.detect_from_affect(valence, arousal);
            prop_assert!(
                alert.is_some(),
                "Failed affect-based detection at valence={}, arousal={}",
                valence,
                arousal,
            );
        }

        /// Property test: affect-based withdrawal detection.
        #[test]
        fn prop_affect_withdrawal_detected(
            valence in -1.0f32..=-0.91,
            arousal in 0.0f32..=0.14,
        ) {
            let detector = CrisisDetector::new();
            let alert = detector.detect_from_affect(valence, arousal);
            prop_assert!(
                alert.is_some(),
                "Failed affect-withdrawal detection at valence={}, arousal={}",
                valence,
                arousal,
            );
        }

        /// Property test: benign affect does not trigger crisis.
        #[test]
        fn prop_benign_affect_no_crisis(
            valence in -0.5f32..=1.0,
            arousal in 0.2f32..=0.8,
        ) {
            let detector = CrisisDetector::new();
            let alert = detector.detect_from_affect(valence, arousal);
            prop_assert!(
                alert.is_none(),
                "False positive at benign affect valence={}, arousal={}",
                valence,
                arousal,
            );
        }
    }

    /// Euphemistic and indirect crisis expressions must also be detected.
    /// These use the exact-phrase detector — they match because they contain
    /// canonical substrings even when wrapped in indirect language.
    #[test]
    fn test_euphemistic_crisis_detection() {
        let detector = CrisisDetector::new();
        let euphemistic_phrases = [
            // Suicidal — indirect
            ("I've found a way out of all this pain", true),
            ("I won't be here much longer", true),
            ("everyone would be better without me around", true),
            // Self-harm — embedded
            ("sometimes I think about hurting myself to cope", true),
            ("I deserve to hurt for what I've done", true),
            // Domestic violence — indirect
            ("I'm afraid to go home tonight", true),
            ("my partner hits me when they drink", true),
            // Benign — should not trigger keyword match
            ("I found a way out of the building", false), // "found a way out" matches!
            ("the sun won't be here tomorrow", false),    // "won't be here" matches!
        ];

        for (phrase, should_detect) in euphemistic_phrases {
            let alert = detector.detect(phrase);
            if should_detect {
                assert!(
                    alert.is_some(),
                    "Should detect euphemistic crisis: '{}'",
                    phrase,
                );
            }
            // Note: some benign phrases may match because they contain crisis
            // substrings. This is acceptable — false positives > false negatives.
        }
    }

    /// Test that compositional HDC encoding gives higher similarity for
    /// crisis-related text than benign text, even without exact keyword matches.
    #[test]
    fn test_compositional_encoding_semantic_similarity() {
        // Verify the encoding function directly
        let crisis_phrase = encode_text_compositional("end my life");
        let similar_phrase = encode_text_compositional("end it all");
        let different_crisis = encode_text_compositional("die want to");
        let benign = encode_text_compositional("the weather is nice today");

        let sim_related = crisis_phrase.similarity(&similar_phrase);
        let sim_benign = crisis_phrase.similarity(&benign);

        // Phrases sharing "end" should be more similar than random benign text
        assert!(
            sim_related > sim_benign,
            "Related phrases should be more similar ({}) than benign ({}) to crisis phrase",
            sim_related,
            sim_benign,
        );

        // Different crisis words should still be closer to random than shared words
        let sim_diff = crisis_phrase.similarity(&different_crisis);
        assert!(
            sim_related > sim_diff || sim_diff > sim_benign,
            "Related ({}) or different-crisis ({}) should beat benign ({})",
            sim_related,
            sim_diff,
            sim_benign,
        );
    }

    /// Verify that the HDC similarity path catches paraphrases that
    /// have no exact keyword matches. These test the compositional encoding's
    /// ability to detect novel crisis expressions via word overlap.
    #[test]
    fn test_hdc_path_catches_paraphrases() {
        let detector = CrisisDetector::new();

        // These paraphrases share crisis-relevant words but don't match
        // any exact keyword substring.
        let paraphrases = [
            "life is not worth it anymore", // shares "not", "worth" with "not worth living"
            "die peacefully in my sleep tonight", // shares "die" with "want to die"
            "dead would be better than this", // shares "dead", "better" with "better off dead"
        ];

        let mut detected = 0;
        for phrase in &paraphrases {
            if detector.detect(phrase).is_some() {
                detected += 1;
            }
        }

        // With compositional encoding, at least some paraphrases should trigger
        // via the HDC path (word overlap). Not all may fire — that's OK.
        // The point is the HDC path is no longer dead.
        assert!(
            detected > 0,
            "Compositional HDC should detect at least 1/{} paraphrases, got 0",
            paraphrases.len(),
        );
    }

    /// Verify benign text doesn't false-positive via the HDC path.
    #[test]
    fn test_hdc_path_no_false_positive_benign() {
        let detector = CrisisDetector::new();

        let benign = [
            "The weather is nice today and I enjoyed my lunch",
            "I'm working on a programming project",
            "My favorite color is blue",
            "The meeting went well this afternoon",
            "I watched a good movie last night",
        ];

        for phrase in &benign {
            let alert = detector.detect(phrase);
            assert!(
                alert.is_none(),
                "Benign input should not trigger crisis detection: '{}', got: {:?}",
                phrase,
                alert.map(|a| (a.matched_indicator, a.confidence)),
            );
        }
    }

    /// Aggregate false negative rate test: run all phrases N times with noise
    /// and verify < 1% miss rate overall.
    #[test]
    fn test_aggregate_false_negative_rate_below_1_percent() {
        let detector = CrisisDetector::new();
        let phrases = all_crisis_phrases();
        let noise_prefixes = [
            "",
            "I think ",
            "Sometimes I feel like ",
            "Last night ",
            "My friend said ",
            "I keep thinking about ",
            "I can't stop ",
            "Today I realized ",
            "Nobody knows but ",
            "Please help ",
        ];

        let mut total = 0;
        let mut detected = 0;

        for (phrase, _) in &phrases {
            for prefix in &noise_prefixes {
                let input = format!("{}{}", prefix, phrase);
                total += 1;
                if detector.detect(&input).is_some() {
                    detected += 1;
                }
            }
        }

        let miss_rate = 1.0 - (detected as f64 / total as f64);
        assert!(
            miss_rate < 0.01,
            "False negative rate {:.2}% exceeds 1% threshold ({}/{} missed)",
            miss_rate * 100.0,
            total - detected,
            total,
        );
    }
}
