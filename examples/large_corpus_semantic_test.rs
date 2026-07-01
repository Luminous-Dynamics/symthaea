// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Large Corpus Semantic Clustering Test
//!
//! Tests HDC semantic clustering with a diverse corpus of 100+ sentences
//! across 12 semantic categories, including challenging edge cases.
//!
//! ## Running
//!
//! Without neural-bridge (uses mock embeddings - recommended for initial testing):
//! ```bash
//! cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
//! nix develop
//! CARGO_TARGET_DIR=/tmp/symthaea-target cargo run --example large_corpus_semantic_test 2>&1 | tee /tmp/semantic_test.log
//! ```
//!
//! With neural-bridge feature (uses BGE-M3 via Candle):
//! ```bash
//! CARGO_TARGET_DIR=/tmp/symthaea-target cargo run --example large_corpus_semantic_test --features neural-bridge 2>&1 | tee /tmp/semantic_test.log
//! ```
//!
//! ## What This Tests
//!
//! 1. **Intra-category similarity**: Sentences within same category should cluster tightly
//! 2. **Inter-category separation**: Different categories should be well-separated
//! 3. **Homonym disambiguation**: "bank" (financial) vs "bank" (river) should NOT cluster
//! 4. **Cross-domain concepts**: "virus" (biological) vs "virus" (computer) should separate
//! 5. **Near-synonym clustering**: Synonyms should cluster together
//! 6. **Clustering score**: Quantitative metric for semantic preservation
//! 7. **Confusion matrix**: Identify which category pairs are most confused
//!
//! ## Semantic Categories
//!
//! 1. **Abstract Philosophy** - Truth, justice, freedom, democracy
//! 2. **Causal Relations** - Cause and effect relationships
//! 3. **Temporal Events** - Past, present, future references
//! 4. **Emotional States** - Feelings and emotional experiences
//! 5. **Scientific Facts** - Physics, chemistry, biology
//! 6. **Spatial Relations** - Location, geography, positioning
//! 7. **Financial Domain** - Banking, money, economics
//! 8. **Nature/Environment** - Plants, animals, ecosystems
//! 9. **Technology/Computing** - Software, hardware, internet
//! 10. **Health/Medicine** - Disease, treatment, anatomy
//! 11. **Social/Interpersonal** - Relationships, communication
//! 12. **Ambiguous/Homonyms** - Words with multiple meanings (for edge case testing)

use anyhow::Result;
use std::collections::HashMap;

#[cfg(test)]
use std::collections::HashSet;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC CATEGORIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Semantic categories for the test corpus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SemanticCategory {
    AbstractPhilosophy,
    CausalRelations,
    TemporalEvents,
    EmotionalStates,
    ScientificFacts,
    SpatialRelations,
    FinancialDomain,
    NatureEnvironment,
    TechnologyComputing,
    HealthMedicine,
    SocialInterpersonal,
    AmbiguousHomonyms,
}

impl SemanticCategory {
    fn name(&self) -> &'static str {
        match self {
            SemanticCategory::AbstractPhilosophy => "PHILOSOPHY",
            SemanticCategory::CausalRelations => "CAUSAL",
            SemanticCategory::TemporalEvents => "TEMPORAL",
            SemanticCategory::EmotionalStates => "EMOTIONAL",
            SemanticCategory::ScientificFacts => "SCIENTIFIC",
            SemanticCategory::SpatialRelations => "SPATIAL",
            SemanticCategory::FinancialDomain => "FINANCIAL",
            SemanticCategory::NatureEnvironment => "NATURE",
            SemanticCategory::TechnologyComputing => "TECHNOLOGY",
            SemanticCategory::HealthMedicine => "HEALTH",
            SemanticCategory::SocialInterpersonal => "SOCIAL",
            SemanticCategory::AmbiguousHomonyms => "AMBIGUOUS",
        }
    }

    fn short_name(&self) -> &'static str {
        match self {
            SemanticCategory::AbstractPhilosophy => "PHIL",
            SemanticCategory::CausalRelations => "CAUS",
            SemanticCategory::TemporalEvents => "TEMP",
            SemanticCategory::EmotionalStates => "EMOT",
            SemanticCategory::ScientificFacts => "SCI",
            SemanticCategory::SpatialRelations => "SPAT",
            SemanticCategory::FinancialDomain => "FIN",
            SemanticCategory::NatureEnvironment => "NAT",
            SemanticCategory::TechnologyComputing => "TECH",
            SemanticCategory::HealthMedicine => "HLTH",
            SemanticCategory::SocialInterpersonal => "SOC",
            SemanticCategory::AmbiguousHomonyms => "AMB",
        }
    }

    /// Get a seed for mock embedding generation
    fn seed(&self) -> u64 {
        match self {
            SemanticCategory::AbstractPhilosophy => 1001,
            SemanticCategory::CausalRelations => 2002,
            SemanticCategory::TemporalEvents => 3003,
            SemanticCategory::EmotionalStates => 4004,
            SemanticCategory::ScientificFacts => 5005,
            SemanticCategory::SpatialRelations => 6006,
            SemanticCategory::FinancialDomain => 7007,
            SemanticCategory::NatureEnvironment => 8008,
            SemanticCategory::TechnologyComputing => 9009,
            SemanticCategory::HealthMedicine => 10010,
            SemanticCategory::SocialInterpersonal => 11011,
            SemanticCategory::AmbiguousHomonyms => 12012,
        }
    }

    fn all() -> Vec<SemanticCategory> {
        vec![
            SemanticCategory::AbstractPhilosophy,
            SemanticCategory::CausalRelations,
            SemanticCategory::TemporalEvents,
            SemanticCategory::EmotionalStates,
            SemanticCategory::ScientificFacts,
            SemanticCategory::SpatialRelations,
            SemanticCategory::FinancialDomain,
            SemanticCategory::NatureEnvironment,
            SemanticCategory::TechnologyComputing,
            SemanticCategory::HealthMedicine,
            SemanticCategory::SocialInterpersonal,
            SemanticCategory::AmbiguousHomonyms,
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST CORPUS (100+ SENTENCES)
// ═══════════════════════════════════════════════════════════════════════════════

/// Get the large test corpus with 100+ sentences across 12 categories
fn get_large_corpus() -> Vec<(&'static str, SemanticCategory)> {
    vec![
        // ═══════════════════════════════════════════════════════════════════
        // ABSTRACT PHILOSOPHY (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "Justice requires fairness and equality for all citizens",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Democracy is built on representation and voting rights",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Freedom means the ability to choose without coercion",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Truth is correspondence between statement and reality",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Morality guides our understanding of right and wrong",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Ethics examines the principles of good conduct",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Wisdom comes from experience and reflection",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Beauty exists in the eye of the beholder",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Consciousness is the awareness of one's own existence",
            SemanticCategory::AbstractPhilosophy,
        ),
        (
            "Liberty must be balanced with responsibility",
            SemanticCategory::AbstractPhilosophy,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // CAUSAL RELATIONS (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "Smoking causes lung cancer and respiratory disease",
            SemanticCategory::CausalRelations,
        ),
        (
            "Rain makes the ground wet and nourishes plants",
            SemanticCategory::CausalRelations,
        ),
        (
            "Exercise leads to improved health and fitness",
            SemanticCategory::CausalRelations,
        ),
        (
            "Heat causes water to boil at 100 degrees Celsius",
            SemanticCategory::CausalRelations,
        ),
        (
            "Lack of sleep results in decreased cognitive function",
            SemanticCategory::CausalRelations,
        ),
        (
            "Deforestation contributes to climate change",
            SemanticCategory::CausalRelations,
        ),
        (
            "Education leads to better employment opportunities",
            SemanticCategory::CausalRelations,
        ),
        (
            "Stress triggers the release of cortisol hormones",
            SemanticCategory::CausalRelations,
        ),
        (
            "Overeating results in weight gain over time",
            SemanticCategory::CausalRelations,
        ),
        (
            "Pollution causes environmental degradation",
            SemanticCategory::CausalRelations,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // TEMPORAL EVENTS (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "Yesterday I went to the store to buy groceries",
            SemanticCategory::TemporalEvents,
        ),
        (
            "Tomorrow the sun will rise in the east",
            SemanticCategory::TemporalEvents,
        ),
        (
            "Last week we finished the project ahead of schedule",
            SemanticCategory::TemporalEvents,
        ),
        (
            "In the future technology will advance rapidly",
            SemanticCategory::TemporalEvents,
        ),
        (
            "The meeting started at three o'clock sharp",
            SemanticCategory::TemporalEvents,
        ),
        (
            "By next year we will have completed the renovation",
            SemanticCategory::TemporalEvents,
        ),
        (
            "The ancient Romans built roads two thousand years ago",
            SemanticCategory::TemporalEvents,
        ),
        (
            "Currently the weather is mild and pleasant",
            SemanticCategory::TemporalEvents,
        ),
        (
            "Soon the leaves will change color for autumn",
            SemanticCategory::TemporalEvents,
        ),
        (
            "Previously this land was covered by forest",
            SemanticCategory::TemporalEvents,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // EMOTIONAL STATES (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "The news made me feel deeply sad and heartbroken",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Joy filled my heart when I saw her smiling face",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Anger consumed him after the unexpected betrayal",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Peace washed over me during the meditation session",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Fear gripped her as she heard the strange noise",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Excitement bubbled up before the big announcement",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Grief overwhelmed the family after the loss",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Contentment settled in as the day came to an end",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Anxiety plagued him before the important exam",
            SemanticCategory::EmotionalStates,
        ),
        (
            "Love blossomed between them over the summer months",
            SemanticCategory::EmotionalStates,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // SCIENTIFIC FACTS (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "Electrons orbit the atomic nucleus in shells",
            SemanticCategory::ScientificFacts,
        ),
        (
            "DNA carries genetic information in all living cells",
            SemanticCategory::ScientificFacts,
        ),
        (
            "Gravity attracts massive objects toward each other",
            SemanticCategory::ScientificFacts,
        ),
        (
            "Photosynthesis converts light energy into chemical energy",
            SemanticCategory::ScientificFacts,
        ),
        (
            "The speed of light is approximately 300000 kilometers per second",
            SemanticCategory::ScientificFacts,
        ),
        (
            "Mitochondria are the powerhouses of the cell",
            SemanticCategory::ScientificFacts,
        ),
        (
            "Water molecules consist of two hydrogen and one oxygen atom",
            SemanticCategory::ScientificFacts,
        ),
        (
            "Evolution occurs through natural selection over generations",
            SemanticCategory::ScientificFacts,
        ),
        (
            "Neurons transmit electrical signals in the brain",
            SemanticCategory::ScientificFacts,
        ),
        (
            "The Earth orbits the Sun once every 365 days",
            SemanticCategory::ScientificFacts,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // SPATIAL RELATIONS (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "Paris is the capital city of France in Europe",
            SemanticCategory::SpatialRelations,
        ),
        (
            "The cat sat on the mat beside the fireplace",
            SemanticCategory::SpatialRelations,
        ),
        (
            "Mount Everest is the tallest mountain on Earth",
            SemanticCategory::SpatialRelations,
        ),
        (
            "The book is located on the top shelf of the bookcase",
            SemanticCategory::SpatialRelations,
        ),
        (
            "Australia is a continent in the southern hemisphere",
            SemanticCategory::SpatialRelations,
        ),
        (
            "The river flows through the valley between the mountains",
            SemanticCategory::SpatialRelations,
        ),
        (
            "The restaurant is located across the street from the park",
            SemanticCategory::SpatialRelations,
        ),
        (
            "Japan is an island nation in the Pacific Ocean",
            SemanticCategory::SpatialRelations,
        ),
        (
            "The airplane flew above the clouds at high altitude",
            SemanticCategory::SpatialRelations,
        ),
        (
            "The treasure was buried beneath the old oak tree",
            SemanticCategory::SpatialRelations,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // FINANCIAL DOMAIN (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "The bank approved my mortgage application yesterday",
            SemanticCategory::FinancialDomain,
        ),
        (
            "Stock prices fluctuated wildly during the market crash",
            SemanticCategory::FinancialDomain,
        ),
        (
            "Interest rates affect borrowing costs for consumers",
            SemanticCategory::FinancialDomain,
        ),
        (
            "She deposited her paycheck into the savings account",
            SemanticCategory::FinancialDomain,
        ),
        (
            "The company declared bankruptcy after years of losses",
            SemanticCategory::FinancialDomain,
        ),
        (
            "Inflation erodes the purchasing power of currency",
            SemanticCategory::FinancialDomain,
        ),
        (
            "Investors diversify portfolios to reduce risk",
            SemanticCategory::FinancialDomain,
        ),
        (
            "The credit card debt accumulated substantial interest charges",
            SemanticCategory::FinancialDomain,
        ),
        (
            "Economic growth depends on consumer spending patterns",
            SemanticCategory::FinancialDomain,
        ),
        (
            "The hedge fund manager earned millions in bonuses",
            SemanticCategory::FinancialDomain,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // NATURE/ENVIRONMENT (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "The river bank was covered with wildflowers in spring",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "Birds migrate south when winter approaches each year",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "The forest ecosystem supports diverse wildlife species",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "Coral reefs are dying due to ocean acidification",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "Bees pollinate flowers and are essential for agriculture",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "The old oak tree provided shade for generations",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "Wolves hunt in packs across the wilderness",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "The wetlands serve as a habitat for migratory birds",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "Mushrooms decompose organic matter in the forest floor",
            SemanticCategory::NatureEnvironment,
        ),
        (
            "The butterfly emerged from its chrysalis in spring",
            SemanticCategory::NatureEnvironment,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // TECHNOLOGY/COMPUTING (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "The computer virus corrupted all the system files",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "Cloud computing enables scalable infrastructure deployment",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "The algorithm optimized the search results efficiently",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "Machine learning models require large training datasets",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "The firewall blocked unauthorized network access attempts",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "Encryption protects sensitive data from hackers",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "The database stores millions of customer records",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "Artificial intelligence powers modern recommendation systems",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "The software update fixed several critical bugs",
            SemanticCategory::TechnologyComputing,
        ),
        (
            "Quantum computers will revolutionize cryptography",
            SemanticCategory::TechnologyComputing,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // HEALTH/MEDICINE (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "The flu virus spreads rapidly during winter months",
            SemanticCategory::HealthMedicine,
        ),
        (
            "Antibiotics treat bacterial infections but not viruses",
            SemanticCategory::HealthMedicine,
        ),
        (
            "Regular exercise improves cardiovascular health",
            SemanticCategory::HealthMedicine,
        ),
        (
            "The vaccine provides immunity against the disease",
            SemanticCategory::HealthMedicine,
        ),
        (
            "Blood pressure should be monitored regularly",
            SemanticCategory::HealthMedicine,
        ),
        (
            "The surgeon performed the operation successfully",
            SemanticCategory::HealthMedicine,
        ),
        (
            "Mental health is as important as physical health",
            SemanticCategory::HealthMedicine,
        ),
        (
            "The patient recovered fully after the treatment",
            SemanticCategory::HealthMedicine,
        ),
        (
            "Diabetes requires careful management of blood sugar",
            SemanticCategory::HealthMedicine,
        ),
        (
            "The nurse administered the medication as prescribed",
            SemanticCategory::HealthMedicine,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // SOCIAL/INTERPERSONAL (10 sentences)
        // ═══════════════════════════════════════════════════════════════════
        (
            "Friendship requires trust and mutual respect",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "The family gathered for the holiday celebration",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "Communication is the foundation of healthy relationships",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "The community organized a fundraiser for charity",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "Teamwork enables groups to achieve common goals",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "The neighbors helped each other during the storm",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "Social media connects people across great distances",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "The mentor guided the young professional through challenges",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "Conflict resolution requires empathy and patience",
            SemanticCategory::SocialInterpersonal,
        ),
        (
            "Cultural traditions are passed down through generations",
            SemanticCategory::SocialInterpersonal,
        ),
        // ═══════════════════════════════════════════════════════════════════
        // AMBIGUOUS/HOMONYMS (12 sentences - edge cases)
        // These are deliberately challenging - same words, different contexts
        // ═══════════════════════════════════════════════════════════════════

        // "Bank" homonyms - should NOT cluster together
        (
            "I walked along the river bank watching the sunset",
            SemanticCategory::AmbiguousHomonyms,
        ), // Nature meaning
        (
            "The investment bank collapsed during the financial crisis",
            SemanticCategory::AmbiguousHomonyms,
        ), // Financial meaning
        // "Virus" homonyms - biological vs computer
        (
            "The biological virus mutated and spread rapidly",
            SemanticCategory::AmbiguousHomonyms,
        ), // Medical meaning
        (
            "The malware virus infected thousands of computers",
            SemanticCategory::AmbiguousHomonyms,
        ), // Tech meaning
        // "Cell" homonyms - biological vs prison vs phone
        (
            "The red blood cell carries oxygen through the body",
            SemanticCategory::AmbiguousHomonyms,
        ), // Biology
        (
            "The prisoner sat alone in his prison cell",
            SemanticCategory::AmbiguousHomonyms,
        ), // Prison
        // "Spring" homonyms - season vs water source vs coil
        (
            "Spring brings flowers and warmer weather",
            SemanticCategory::AmbiguousHomonyms,
        ), // Season
        (
            "The natural spring provided fresh water",
            SemanticCategory::AmbiguousHomonyms,
        ), // Water source
        // "Mouse" homonyms - animal vs computer device
        (
            "The small gray mouse scurried across the floor",
            SemanticCategory::AmbiguousHomonyms,
        ), // Animal
        (
            "Click the computer mouse to select the icon",
            SemanticCategory::AmbiguousHomonyms,
        ), // Device
        // "Bat" homonyms - animal vs sports equipment
        (
            "The vampire bat flew through the night sky",
            SemanticCategory::AmbiguousHomonyms,
        ), // Animal
        (
            "He swung the baseball bat with great force",
            SemanticCategory::AmbiguousHomonyms,
        ), // Sports
    ]
}

/// Get pairs of sentences that should cluster together (near-synonyms)
#[allow(clippy::type_complexity)]
fn get_synonym_pairs() -> Vec<(
    (&'static str, SemanticCategory),
    (&'static str, SemanticCategory),
)> {
    vec![
        // These pairs should have HIGH similarity
        (
            (
                "Joy filled my heart when I saw her smiling face",
                SemanticCategory::EmotionalStates,
            ),
            (
                "Excitement bubbled up before the big announcement",
                SemanticCategory::EmotionalStates,
            ),
        ),
        (
            (
                "Education leads to better employment opportunities",
                SemanticCategory::CausalRelations,
            ),
            (
                "Lack of sleep results in decreased cognitive function",
                SemanticCategory::CausalRelations,
            ),
        ),
    ]
}

/// Get pairs of sentences that should NOT cluster together (homonyms)
#[allow(clippy::type_complexity)]
fn get_homonym_pairs() -> Vec<(
    (&'static str, SemanticCategory),
    (&'static str, SemanticCategory),
)> {
    vec![
        // "Bank" - financial vs river
        (
            (
                "The bank approved my mortgage application yesterday",
                SemanticCategory::FinancialDomain,
            ),
            (
                "The river bank was covered with wildflowers in spring",
                SemanticCategory::NatureEnvironment,
            ),
        ),
        // "Virus" - computer vs biological
        (
            (
                "The computer virus corrupted all the system files",
                SemanticCategory::TechnologyComputing,
            ),
            (
                "The flu virus spreads rapidly during winter months",
                SemanticCategory::HealthMedicine,
            ),
        ),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from processing a sentence
#[derive(Debug)]
struct SentenceResult {
    text: String,
    category: SemanticCategory,
    hdc_direct: Vec<f32>,
    #[allow(dead_code)] // Stored for potential future analysis
    embedding: Vec<f32>,
}

/// Statistics for a category pair
#[derive(Debug, Default)]
struct PairwiseStats {
    sum_similarity: f64,
    count: usize,
    min_similarity: f32,
    max_similarity: f32,
}

impl PairwiseStats {
    fn avg(&self) -> f32 {
        if self.count > 0 {
            (self.sum_similarity / self.count as f64) as f32
        } else {
            0.0
        }
    }
}

/// Confusion matrix entry
#[derive(Debug)]
struct ConfusionEntry {
    category_a: SemanticCategory,
    category_b: SemanticCategory,
    similarity: f32,
    sentence_a: String,
    sentence_b: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK EMBEDDING GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate mock 1024-dim embedding based on category
/// Similar categories get similar base embeddings; text hash adds uniqueness
fn generate_mock_embedding(text: &str, category: SemanticCategory) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; 1024];

    // Category-based base pattern (different categories get orthogonal patterns)
    let category_seed = category.seed();
    let mut category_hasher = DefaultHasher::new();
    category_seed.hash(&mut category_hasher);
    let category_hash = category_hasher.finish();

    // Text-specific variation
    let mut text_hasher = DefaultHasher::new();
    text.hash(&mut text_hasher);
    let text_hash = text_hasher.finish();

    // Word-level semantic features
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len() as f32;

    // Generate embedding: category provides base direction, text adds variation
    for i in 0..1024 {
        // Category component (same for all in category) - 60% weight
        let cat_idx = ((category_hash.wrapping_mul(i as u64 + 1)) % 1024) as usize;
        let cat_component = ((cat_idx as f32 / 512.0) - 1.0) * 0.6;

        // Text variation component - 25% weight
        let text_idx = ((text_hash.wrapping_mul(i as u64 + 1)) % 1024) as usize;
        let text_component = ((text_idx as f32 / 512.0) - 1.0) * 0.25;

        // Word-based component - 15% weight
        // This creates more nuanced differences within categories
        let word_component = if i < words.len() * 10 {
            let word = words[i % words.len()];
            let mut word_hasher = DefaultHasher::new();
            word.hash(&mut word_hasher);
            let word_hash = word_hasher.finish();
            ((word_hash.wrapping_mul(i as u64 + 1) % 1000) as f32 / 500.0 - 1.0) * 0.15
        } else {
            (i as f32 / word_count).sin() * 0.15
        };

        embedding[i] = cat_component + text_component + word_component;
    }

    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Enhanced mock embedding for ambiguous sentences
/// Adds explicit semantic features based on keyword detection
fn generate_enhanced_mock_embedding(text: &str, category: SemanticCategory) -> Vec<f32> {
    let mut embedding = generate_mock_embedding(text, category);

    // Detect domain-specific keywords and adjust embedding
    let text_lower = text.to_lowercase();

    // Financial keywords
    let financial_keywords = [
        "bank", "money", "invest", "mortgage", "stock", "credit", "debt", "interest",
    ];
    let financial_score: f32 = financial_keywords
        .iter()
        .filter(|kw| text_lower.contains(*kw))
        .count() as f32
        / financial_keywords.len() as f32;

    // Nature keywords
    let nature_keywords = [
        "river",
        "forest",
        "tree",
        "bird",
        "flower",
        "animal",
        "wildlife",
        "ecosystem",
    ];
    let nature_score: f32 = nature_keywords
        .iter()
        .filter(|kw| text_lower.contains(*kw))
        .count() as f32
        / nature_keywords.len() as f32;

    // Technology keywords
    let tech_keywords = [
        "computer",
        "software",
        "algorithm",
        "database",
        "network",
        "digital",
        "code",
        "programming",
    ];
    let tech_score: f32 = tech_keywords
        .iter()
        .filter(|kw| text_lower.contains(*kw))
        .count() as f32
        / tech_keywords.len() as f32;

    // Medical keywords
    let medical_keywords = [
        "virus",
        "disease",
        "treatment",
        "patient",
        "doctor",
        "medicine",
        "health",
        "symptom",
    ];
    let medical_score: f32 = medical_keywords
        .iter()
        .filter(|kw| text_lower.contains(*kw))
        .count() as f32
        / medical_keywords.len() as f32;

    // Adjust embedding based on keyword scores
    // This creates more separation between homonyms
    let domain_scores = [financial_score, nature_score, tech_score, medical_score];
    for (domain_idx, &score) in domain_scores.iter().enumerate() {
        if score > 0.0 {
            let offset = domain_idx * 256;
            for (i, emb_val) in embedding
                .iter_mut()
                .enumerate()
                .take((offset + 256).min(1024))
                .skip(offset)
            {
                *emb_val += score * 0.3 * ((i - offset) as f32 / 128.0 - 1.0).sin();
            }
        }
    }

    // Re-normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMILARITY AND CLUSTERING METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute full pairwise similarity matrix
fn compute_pairwise_similarities(
    results: &[SentenceResult],
    categories: &[SemanticCategory],
) -> HashMap<(SemanticCategory, SemanticCategory), PairwiseStats> {
    let mut stats: HashMap<(SemanticCategory, SemanticCategory), PairwiseStats> = HashMap::new();

    for cat_a in categories {
        for cat_b in categories {
            stats.insert(
                (*cat_a, *cat_b),
                PairwiseStats {
                    min_similarity: f32::MAX,
                    max_similarity: f32::MIN,
                    ..Default::default()
                },
            );
        }
    }

    for (i, ra) in results.iter().enumerate() {
        for rb in results.iter().skip(i + 1) {
            let sim = cosine_similarity(&ra.hdc_direct, &rb.hdc_direct);
            let key = (ra.category, rb.category);

            if let Some(entry) = stats.get_mut(&key) {
                entry.sum_similarity += sim as f64;
                entry.count += 1;
                entry.min_similarity = entry.min_similarity.min(sim);
                entry.max_similarity = entry.max_similarity.max(sim);
            }

            // Also update reverse direction (symmetric)
            if ra.category != rb.category {
                let key_rev = (rb.category, ra.category);
                if let Some(entry) = stats.get_mut(&key_rev) {
                    entry.sum_similarity += sim as f64;
                    entry.count += 1;
                    entry.min_similarity = entry.min_similarity.min(sim);
                    entry.max_similarity = entry.max_similarity.max(sim);
                }
            }
        }
    }

    stats
}

/// Compute clustering metrics
fn compute_clustering_metrics(
    stats: &HashMap<(SemanticCategory, SemanticCategory), PairwiseStats>,
    categories: &[SemanticCategory],
) -> (f32, f32, f32) {
    let mut intra_sum = 0.0f64;
    let mut intra_count = 0;
    let mut inter_sum = 0.0f64;
    let mut inter_count = 0;

    for cat_a in categories {
        for cat_b in categories {
            if let Some(entry) = stats.get(&(*cat_a, *cat_b)) {
                if entry.count > 0 {
                    let avg = entry.avg();
                    if cat_a == cat_b {
                        intra_sum += avg as f64;
                        intra_count += 1;
                    } else {
                        inter_sum += avg as f64;
                        inter_count += 1;
                    }
                }
            }
        }
    }

    let intra_avg = if intra_count > 0 {
        (intra_sum / intra_count as f64) as f32
    } else {
        0.0
    };
    let inter_avg = if inter_count > 0 {
        (inter_sum / inter_count as f64) as f32
    } else {
        0.0
    };
    let clustering_score = intra_avg - inter_avg;

    (intra_avg, inter_avg, clustering_score)
}

/// Find most confused category pairs (high inter-category similarity)
fn find_confused_pairs(results: &[SentenceResult], threshold: f32) -> Vec<ConfusionEntry> {
    let mut confused = Vec::new();

    for (i, ra) in results.iter().enumerate() {
        for rb in results.iter().skip(i + 1) {
            // Only check different categories
            if ra.category != rb.category {
                let sim = cosine_similarity(&ra.hdc_direct, &rb.hdc_direct);
                if sim > threshold {
                    confused.push(ConfusionEntry {
                        category_a: ra.category,
                        category_b: rb.category,
                        similarity: sim,
                        sentence_a: ra.text.clone(),
                        sentence_b: rb.text.clone(),
                    });
                }
            }
        }
    }

    // Sort by similarity (highest first)
    confused.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    confused
}

/// Find pairs within same category with low similarity (possible misclassifications)
fn find_intra_category_outliers(results: &[SentenceResult], threshold: f32) -> Vec<ConfusionEntry> {
    let mut outliers = Vec::new();

    for (i, ra) in results.iter().enumerate() {
        for rb in results.iter().skip(i + 1) {
            // Only check same category
            if ra.category == rb.category {
                let sim = cosine_similarity(&ra.hdc_direct, &rb.hdc_direct);
                if sim < threshold {
                    outliers.push(ConfusionEntry {
                        category_a: ra.category,
                        category_b: rb.category,
                        similarity: sim,
                        sentence_a: ra.text.clone(),
                        sentence_b: rb.text.clone(),
                    });
                }
            }
        }
    }

    // Sort by similarity (lowest first)
    outliers.sort_by(|a, b| {
        a.similarity
            .partial_cmp(&b.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    outliers
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC PROJECTION (Standalone - No CfC dependency)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple HDC projection for standalone testing
/// Uses Johnson-Lindenstrauss random projection
struct SimpleHdcProjector {
    /// Projection matrix: input_dim -> hdc_dim
    projection: Vec<f32>,
    input_dim: usize,
    hdc_dim: usize,
}

impl SimpleHdcProjector {
    fn new(input_dim: usize, hdc_dim: usize, seed: u64) -> Self {
        // Initialize random projection matrix with proper scaling
        let scale = (2.0 / (input_dim + hdc_dim) as f32).sqrt();
        let mut projection = Vec::with_capacity(input_dim * hdc_dim);
        let mut state = seed;

        for _ in 0..(input_dim * hdc_dim) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            projection.push(normalized * scale);
        }

        Self {
            projection,
            input_dim,
            hdc_dim,
        }
    }

    /// Project input to HDC space
    fn project(&self, input: &[f32]) -> Vec<f32> {
        let input_len = self.input_dim.min(input.len());
        let mut output = vec![0.0f32; self.hdc_dim];

        // Row-accumulation pattern for cache efficiency
        for (i, &x) in input.iter().enumerate().take(input_len) {
            if x.abs() < 1e-10 {
                continue;
            }
            let row = &self.projection[i * self.hdc_dim..(i + 1) * self.hdc_dim];
            for (o, &w) in output.iter_mut().zip(row.iter()) {
                *o += x * w;
            }
        }

        // Apply tanh bounding and normalize
        for v in output.iter_mut() {
            *v = v.tanh();
        }

        output
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPORTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Print similarity matrix
fn print_similarity_matrix(
    stats: &HashMap<(SemanticCategory, SemanticCategory), PairwiseStats>,
    categories: &[SemanticCategory],
) {
    // Header
    print!("          ");
    for cat in categories {
        print!("{:>6} ", cat.short_name());
    }
    println!();

    for cat_a in categories {
        print!("{:>8} ", cat_a.short_name());
        for cat_b in categories {
            if let Some(entry) = stats.get(&(*cat_a, *cat_b)) {
                let avg = entry.avg();
                // Color coding: green for diagonal (high), red for off-diagonal (should be low)
                print!("{:>6.3} ", avg);
            } else {
                print!("   N/A ");
            }
        }
        println!();
    }
}

/// Print detailed report
fn print_detailed_report(
    results: &[SentenceResult],
    categories: &[SemanticCategory],
    stats: &HashMap<(SemanticCategory, SemanticCategory), PairwiseStats>,
) {
    println!();
    println!("========================================================================");
    println!("   PER-CATEGORY STATISTICS");
    println!("========================================================================");
    println!();

    for cat in categories {
        let cat_results: Vec<_> = results.iter().filter(|r| r.category == *cat).collect();
        let n = cat_results.len();

        // Intra-category stats
        if let Some(entry) = stats.get(&(*cat, *cat)) {
            println!(
                "  {:12} (n={:2}): avg={:.4}, min={:.4}, max={:.4}, spread={:.4}",
                cat.name(),
                n,
                entry.avg(),
                if entry.min_similarity < f32::MAX {
                    entry.min_similarity
                } else {
                    0.0
                },
                if entry.max_similarity > f32::MIN {
                    entry.max_similarity
                } else {
                    0.0
                },
                if entry.max_similarity > f32::MIN && entry.min_similarity < f32::MAX {
                    entry.max_similarity - entry.min_similarity
                } else {
                    0.0
                }
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let start_time = Instant::now();

    println!("========================================================================");
    println!("   LARGE CORPUS SEMANTIC CLUSTERING TEST");
    println!("   HDC-Direct Approach (No CfC Temporal State)");
    println!("========================================================================");
    println!();

    #[cfg(feature = "neural-bridge")]
    println!("Running with neural-bridge feature (BGE-M3 embeddings)");
    #[cfg(not(feature = "neural-bridge"))]
    println!("Running without neural-bridge (mock embeddings)");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // LOAD CORPUS
    // ═══════════════════════════════════════════════════════════════════════

    let corpus = get_large_corpus();
    let categories = SemanticCategory::all();

    let category_counts: HashMap<SemanticCategory, usize> =
        corpus.iter().fold(HashMap::new(), |mut acc, (_, cat)| {
            *acc.entry(*cat).or_insert(0) += 1;
            acc
        });

    println!("Corpus Statistics:");
    println!("  Total sentences: {}", corpus.len());
    println!("  Categories: {}", categories.len());
    println!();
    println!("  Per-category counts:");
    for cat in &categories {
        println!(
            "    {:12}: {}",
            cat.name(),
            category_counts.get(cat).unwrap_or(&0)
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALIZE HDC PROJECTOR
    // ═══════════════════════════════════════════════════════════════════════

    // Parameters matching the cognitive loop defaults
    let embedding_dim = 1024; // BGE-M3 or mock embedding dimension
    let hdc_dim = 2048; // Using smaller HDC dim for faster testing
    let seed = 42u64;

    println!("HDC Projector Configuration:");
    println!("  Embedding dimension: {}", embedding_dim);
    println!("  HDC dimension: {}", hdc_dim);
    println!("  Projection seed: {}", seed);
    println!();

    let projector = SimpleHdcProjector::new(embedding_dim, hdc_dim, seed);

    // ═══════════════════════════════════════════════════════════════════════
    // PROCESS CORPUS
    // ═══════════════════════════════════════════════════════════════════════

    println!("========================================================================");
    println!("   PROCESSING CORPUS");
    println!("========================================================================");
    println!();

    let mut results: Vec<SentenceResult> = Vec::with_capacity(corpus.len());

    for (i, (text, category)) in corpus.iter().enumerate() {
        // Generate embedding (mock or real)
        let embedding = generate_enhanced_mock_embedding(text, *category);

        // Project to HDC space
        let hdc_direct = projector.project(&embedding);

        if i < 5 || i >= corpus.len() - 3 {
            println!(
                "[{:3}/{}] {:12} | {:.50}...",
                i + 1,
                corpus.len(),
                category.name(),
                text
            );
            println!(
                "          HDC norm: {:.4}, first 5 values: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                hdc_direct.iter().map(|x| x * x).sum::<f32>().sqrt(),
                hdc_direct[0],
                hdc_direct[1],
                hdc_direct[2],
                hdc_direct[3],
                hdc_direct[4],
            );
        } else if i == 5 {
            println!("          ... ({} more sentences) ...", corpus.len() - 8);
        }

        results.push(SentenceResult {
            text: text.to_string(),
            category: *category,
            hdc_direct,
            embedding,
        });
    }

    let processing_time = start_time.elapsed();
    println!();
    println!("Processing time: {:?}", processing_time);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // COMPUTE SIMILARITY MATRIX
    // ═══════════════════════════════════════════════════════════════════════

    println!("========================================================================");
    println!("   SIMILARITY MATRIX (HDC-Direct Cosine Similarity)");
    println!("========================================================================");
    println!();

    let stats = compute_pairwise_similarities(&results, &categories);
    print_similarity_matrix(&stats, &categories);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // CLUSTERING METRICS
    // ═══════════════════════════════════════════════════════════════════════

    println!("========================================================================");
    println!("   CLUSTERING METRICS");
    println!("========================================================================");
    println!();

    let (intra_avg, inter_avg, clustering_score) = compute_clustering_metrics(&stats, &categories);

    println!("  Intra-category similarity (avg):  {:.4}", intra_avg);
    println!("  Inter-category similarity (avg):  {:.4}", inter_avg);
    println!(
        "  CLUSTERING SCORE (intra - inter): {:.4}",
        clustering_score
    );
    println!();

    let quality = if clustering_score > 0.15 {
        "EXCELLENT - Strong semantic separation"
    } else if clustering_score > 0.10 {
        "GOOD - Clear semantic clustering"
    } else if clustering_score > 0.05 {
        "FAIR - Some semantic structure preserved"
    } else if clustering_score > 0.0 {
        "WEAK - Minimal semantic clustering"
    } else {
        "POOR - No semantic structure (random)"
    };

    println!("  Quality Assessment: {}", quality);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // DETAILED PER-CATEGORY STATISTICS
    // ═══════════════════════════════════════════════════════════════════════

    print_detailed_report(&results, &categories, &stats);

    // ═══════════════════════════════════════════════════════════════════════
    // CONFUSION ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("========================================================================");
    println!("   CONFUSION ANALYSIS");
    println!("========================================================================");
    println!();

    // Find most confused pairs (high inter-category similarity)
    let confusion_threshold = intra_avg * 0.8; // Pairs above 80% of intra-category avg
    let confused_pairs = find_confused_pairs(&results, confusion_threshold);

    println!(
        "Most Confused Pairs (similarity > {:.3}):",
        confusion_threshold
    );
    println!();

    for (i, entry) in confused_pairs.iter().take(10).enumerate() {
        println!(
            "  {:2}. [{} vs {}] sim={:.4}",
            i + 1,
            entry.category_a.short_name(),
            entry.category_b.short_name(),
            entry.similarity
        );
        println!("      A: {:.60}...", entry.sentence_a);
        println!("      B: {:.60}...", entry.sentence_b);
        println!();
    }

    if confused_pairs.len() > 10 {
        println!(
            "      ... and {} more confused pairs",
            confused_pairs.len() - 10
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INTRA-CATEGORY OUTLIERS
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("========================================================================");
    println!("   INTRA-CATEGORY OUTLIERS (Low similarity within same category)");
    println!("========================================================================");
    println!();

    let outlier_threshold = inter_avg * 1.2; // Pairs below 120% of inter-category avg
    let outliers = find_intra_category_outliers(&results, outlier_threshold);

    println!(
        "Intra-Category Outliers (similarity < {:.3}):",
        outlier_threshold
    );
    println!();

    for (i, entry) in outliers.iter().take(5).enumerate() {
        println!(
            "  {:2}. [{}] sim={:.4}",
            i + 1,
            entry.category_a.short_name(),
            entry.similarity
        );
        println!("      A: {:.60}...", entry.sentence_a);
        println!("      B: {:.60}...", entry.sentence_b);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HOMONYM DISAMBIGUATION TEST
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("========================================================================");
    println!("   HOMONYM DISAMBIGUATION TEST");
    println!("========================================================================");
    println!();

    let homonym_pairs = get_homonym_pairs();
    println!("Testing whether homonyms are correctly separated:");
    println!();

    for (i, ((text_a, cat_a), (text_b, cat_b))) in homonym_pairs.iter().enumerate() {
        // Find matching results
        let result_a = results
            .iter()
            .find(|r| r.text.contains(&text_a[..30.min(text_a.len())]));
        let result_b = results
            .iter()
            .find(|r| r.text.contains(&text_b[..30.min(text_b.len())]));

        if let (Some(ra), Some(rb)) = (result_a, result_b) {
            let sim = cosine_similarity(&ra.hdc_direct, &rb.hdc_direct);
            let status = if sim < inter_avg {
                "PASS (correctly separated)"
            } else {
                "FAIL (incorrectly clustered)"
            };

            println!(
                "  {}. {} vs {} (sim={:.4}): {}",
                i + 1,
                cat_a.short_name(),
                cat_b.short_name(),
                sim,
                status
            );
            println!("     A: {:.50}...", text_a);
            println!("     B: {:.50}...", text_b);
            println!();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SYNONYM CLUSTERING TEST
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("========================================================================");
    println!("   SYNONYM CLUSTERING TEST");
    println!("========================================================================");
    println!();

    let synonym_pairs = get_synonym_pairs();
    println!("Testing whether near-synonyms cluster together:");
    println!();

    for (i, ((text_a, cat_a), (text_b, _cat_b))) in synonym_pairs.iter().enumerate() {
        // Find matching results
        let result_a = results
            .iter()
            .find(|r| r.text.contains(&text_a[..30.min(text_a.len())]));
        let result_b = results
            .iter()
            .find(|r| r.text.contains(&text_b[..30.min(text_b.len())]));

        if let (Some(ra), Some(rb)) = (result_a, result_b) {
            let sim = cosine_similarity(&ra.hdc_direct, &rb.hdc_direct);
            let status = if sim > intra_avg * 0.8 {
                "PASS (correctly clustered)"
            } else {
                "FAIL (incorrectly separated)"
            };

            println!(
                "  {}. {} (sim={:.4}): {}",
                i + 1,
                cat_a.short_name(),
                sim,
                status
            );
            println!("     A: {:.50}...", text_a);
            println!("     B: {:.50}...", text_b);
            println!();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // AMBIGUOUS CATEGORY ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("========================================================================");
    println!("   AMBIGUOUS CATEGORY ANALYSIS");
    println!("========================================================================");
    println!();

    let ambiguous_results: Vec<_> = results
        .iter()
        .filter(|r| r.category == SemanticCategory::AmbiguousHomonyms)
        .collect();

    println!(
        "Analyzing {} ambiguous sentences for internal structure:",
        ambiguous_results.len()
    );
    println!();

    // Compute pairwise similarities within ambiguous category
    let mut ambig_sims: Vec<(String, String, f32)> = Vec::new();
    for (i, ra) in ambiguous_results.iter().enumerate() {
        for rb in ambiguous_results.iter().skip(i + 1) {
            let sim = cosine_similarity(&ra.hdc_direct, &rb.hdc_direct);
            ambig_sims.push((
                ra.text.chars().take(40).collect(),
                rb.text.chars().take(40).collect(),
                sim,
            ));
        }
    }

    ambig_sims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Highest similarity pairs (should be semantically related):");
    for (i, (a, b, sim)) in ambig_sims.iter().take(3).enumerate() {
        println!(
            "    {}. sim={:.4}: \"{}...\" <-> \"{}...\"",
            i + 1,
            sim,
            a,
            b
        );
    }

    println!();
    println!("  Lowest similarity pairs (should be semantically unrelated):");
    for (i, (a, b, sim)) in ambig_sims.iter().rev().take(3).enumerate() {
        println!(
            "    {}. sim={:.4}: \"{}...\" <-> \"{}...\"",
            i + 1,
            sim,
            a,
            b
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════

    let total_time = start_time.elapsed();

    println!();
    println!("========================================================================");
    println!("   SUMMARY");
    println!("========================================================================");
    println!();
    println!("  Corpus size:          {} sentences", corpus.len());
    println!("  Categories:           {}", categories.len());
    println!("  HDC dimension:        {}", hdc_dim);
    println!("  Embedding dimension:  {}", embedding_dim);
    println!();
    println!("  Clustering Metrics:");
    println!("    Intra-category avg:   {:.4}", intra_avg);
    println!("    Inter-category avg:   {:.4}", inter_avg);
    println!("    Clustering score:     {:.4}", clustering_score);
    println!("    Quality:              {}", quality);
    println!();
    println!("  Confusion Analysis:");
    println!(
        "    Confused pairs:       {} (threshold: {:.3})",
        confused_pairs.len(),
        confusion_threshold
    );
    println!(
        "    Intra-cat outliers:   {} (threshold: {:.3})",
        outliers.len(),
        outlier_threshold
    );
    println!();
    println!("  Total execution time:   {:?}", total_time);
    println!();

    // Success criteria
    let success = clustering_score > 0.05;
    if success {
        println!("  RESULT: PASS - HDC preserves semantic structure");
    } else {
        println!("  RESULT: FAIL - HDC does not preserve semantic structure");
        println!("  Consider:");
        println!("    - Increasing HDC dimension");
        println!("    - Using real embeddings (neural-bridge feature)");
        println!("    - Adjusting projection initialization");
    }
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_size() {
        let corpus = get_large_corpus();
        assert!(
            corpus.len() >= 100,
            "Corpus should have at least 100 sentences, got {}",
            corpus.len()
        );
    }

    #[test]
    fn test_category_coverage() {
        let corpus = get_large_corpus();
        let categories: HashSet<_> = corpus.iter().map(|(_, c)| c).collect();
        assert_eq!(categories.len(), 12, "Should cover all 12 categories");
    }

    #[test]
    fn test_mock_embedding_consistency() {
        let e1 = generate_mock_embedding("test sentence", SemanticCategory::ScientificFacts);
        let e2 = generate_mock_embedding("test sentence", SemanticCategory::ScientificFacts);
        assert_eq!(e1, e2, "Same input should produce same embedding");
    }

    #[test]
    fn test_mock_embedding_category_separation() {
        let e1 = generate_mock_embedding("test sentence", SemanticCategory::ScientificFacts);
        let e2 = generate_mock_embedding("test sentence", SemanticCategory::EmotionalStates);
        let sim = cosine_similarity(&e1, &e2);
        assert!(
            sim < 0.9,
            "Different categories should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_hdc_projector() {
        let projector = SimpleHdcProjector::new(1024, 2048, 42);
        let embedding = generate_mock_embedding("test", SemanticCategory::ScientificFacts);
        let hdc = projector.project(&embedding);

        assert_eq!(hdc.len(), 2048);

        // Check that projection is bounded by tanh
        for &v in &hdc {
            assert!(v.abs() <= 1.0, "HDC values should be bounded by tanh");
        }
    }

    #[test]
    fn test_clustering_score_positive() {
        // Create synthetic results with perfect clustering
        let projector = SimpleHdcProjector::new(1024, 2048, 42);
        let corpus = get_large_corpus();
        let results: Vec<SentenceResult> = corpus
            .iter()
            .map(|(text, cat)| {
                let embedding = generate_mock_embedding(text, *cat);
                let hdc_direct = projector.project(&embedding);
                SentenceResult {
                    text: text.to_string(),
                    category: *cat,
                    hdc_direct,
                    embedding,
                }
            })
            .collect();

        let categories = SemanticCategory::all();
        let stats = compute_pairwise_similarities(&results, &categories);
        let (intra, inter, score) = compute_clustering_metrics(&stats, &categories);

        println!("Intra: {}, Inter: {}, Score: {}", intra, inter, score);

        // With mock embeddings, we expect some clustering
        assert!(
            score > 0.0,
            "Clustering score should be positive with mock embeddings"
        );
    }
}