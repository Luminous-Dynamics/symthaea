// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Universal educational taxonomy aligned to ISCED-F (UNESCO).
//!
//! This module provides the canonical subject classification, education
//! levels, and cognitive complexity frameworks used throughout EduNet.
//! All classifications align to international standards (ISCED, EQF,
//! Bloom, Webb DOK, Dreyfus).

use serde::{Deserialize, Serialize};

// ============================================================
// ISCED Education Levels (UNESCO)
// ============================================================

/// ISCED 2011 education levels (0-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IscedLevel {
    /// Level 0: Early childhood education
    EarlyChildhood = 0,
    /// Level 1: Primary education (Grades 1-5)
    Primary = 1,
    /// Level 2: Lower secondary (Grades 6-8)
    LowerSecondary = 2,
    /// Level 3: Upper secondary (Grades 9-12)
    UpperSecondary = 3,
    /// Level 4: Post-secondary non-tertiary (certificates)
    PostSecondaryNonTertiary = 4,
    /// Level 5: Short-cycle tertiary (associate degrees)
    ShortCycleTertiary = 5,
    /// Level 6: Bachelor's or equivalent
    Bachelors = 6,
    /// Level 7: Master's or equivalent
    Masters = 7,
    /// Level 8: Doctoral or equivalent
    Doctoral = 8,
}

impl IscedLevel {
    /// Convert from our GradeLevel string to ISCED level.
    pub fn from_grade_level(grade: &str) -> Self {
        match grade {
            "PreK" => IscedLevel::EarlyChildhood,
            "Kindergarten" => IscedLevel::Primary,
            g if g.starts_with("Grade") => {
                let num: u8 = g[5..].parse().unwrap_or(6);
                match num {
                    1..=5 => IscedLevel::Primary,
                    6..=8 => IscedLevel::LowerSecondary,
                    9..=12 => IscedLevel::UpperSecondary,
                    _ => IscedLevel::LowerSecondary,
                }
            }
            "College" | "Undergraduate" => IscedLevel::Bachelors,
            "Graduate" => IscedLevel::Masters,
            "Doctoral" | "PostDoctoral" => IscedLevel::Doctoral,
            "Professional" => IscedLevel::Masters,
            "Adult" => IscedLevel::PostSecondaryNonTertiary,
            _ => IscedLevel::LowerSecondary,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            IscedLevel::EarlyChildhood => "Early Childhood (ISCED 0)",
            IscedLevel::Primary => "Primary (ISCED 1)",
            IscedLevel::LowerSecondary => "Lower Secondary (ISCED 2)",
            IscedLevel::UpperSecondary => "Upper Secondary (ISCED 3)",
            IscedLevel::PostSecondaryNonTertiary => "Post-Secondary Non-Tertiary (ISCED 4)",
            IscedLevel::ShortCycleTertiary => "Short-Cycle Tertiary (ISCED 5)",
            IscedLevel::Bachelors => "Bachelor's (ISCED 6)",
            IscedLevel::Masters => "Master's (ISCED 7)",
            IscedLevel::Doctoral => "Doctoral (ISCED 8)",
        }
    }

    /// Typical Bloom level for this education level.
    pub fn typical_bloom(&self) -> &str {
        match self {
            IscedLevel::EarlyChildhood => "Remember",
            IscedLevel::Primary => "Understand",
            IscedLevel::LowerSecondary => "Apply",
            IscedLevel::UpperSecondary => "Analyze",
            IscedLevel::PostSecondaryNonTertiary => "Apply",
            IscedLevel::ShortCycleTertiary => "Apply",
            IscedLevel::Bachelors => "Analyze",
            IscedLevel::Masters => "Evaluate",
            IscedLevel::Doctoral => "Create",
        }
    }
}

// ============================================================
// ISCED-F Subject Classification (UNESCO Fields of Education)
// ============================================================

/// Broad field of education (ISCED-F 2013, 11 fields + our addition).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BroadField {
    Education,
    ArtsHumanities,
    SocialSciences,
    BusinessLaw,
    NaturalSciencesMath,
    ICT,
    EngineeringManufacturing,
    AgricultureEnvironment,
    HealthWelfare,
    Services,
    /// Our addition: meta-learning about consciousness computing
    MetaLearning,
}

impl BroadField {
    /// ISCED-F 2-digit code.
    pub fn code(&self) -> &str {
        match self {
            BroadField::Education => "01",
            BroadField::ArtsHumanities => "02",
            BroadField::SocialSciences => "03",
            BroadField::BusinessLaw => "04",
            BroadField::NaturalSciencesMath => "05",
            BroadField::ICT => "06",
            BroadField::EngineeringManufacturing => "07",
            BroadField::AgricultureEnvironment => "08",
            BroadField::HealthWelfare => "09",
            BroadField::Services => "10",
            BroadField::MetaLearning => "11",
        }
    }

    /// Classify a subject string into a broad field.
    pub fn from_subject(subject: &str) -> Self {
        let s = subject.to_lowercase();
        if s.contains("education") || s.contains("teaching") || s.contains("pedagogy") {
            BroadField::Education
        } else if s.contains("literature") || s.contains("philosophy") || s.contains("history")
            || s.contains("art") || s.contains("music") || s.contains("language")
            || s.contains("linguistics") || s.contains("religion") || s.contains("english")
        {
            BroadField::ArtsHumanities
        } else if s.contains("politic") || s.contains("sociology") || s.contains("econom")
            || s.contains("psychology") || s.contains("geography") || s.contains("anthropo")
            || s.contains("social")
        {
            BroadField::SocialSciences
        } else if s.contains("business") || s.contains("management") || s.contains("law")
            || s.contains("finance") || s.contains("accounting") || s.contains("marketing")
        {
            BroadField::BusinessLaw
        } else if s.contains("computer") || s.contains("software") || s.contains("data science")
            || s.contains("cyber") || s.contains("information") || s.contains("ict")
            || s.contains("cs2013") || s.contains("algorithm") || s.contains("programming")
        {
            BroadField::ICT
        } else if s.contains("math") || s.contains("physics") || s.contains("chemistry")
            || s.contains("biology") || s.contains("science") || s.contains("statistics")
            || s.contains("astronomy") || s.contains("earth")
        {
            BroadField::NaturalSciencesMath
        } else if s.contains("engineer") || s.contains("manufactur") || s.contains("construct") {
            BroadField::EngineeringManufacturing
        } else if s.contains("agricult") || s.contains("environment") || s.contains("forest") {
            BroadField::AgricultureEnvironment
        } else if s.contains("health") || s.contains("medic") || s.contains("nurs")
            || s.contains("pharm") || s.contains("public health")
        {
            BroadField::HealthWelfare
        } else if s.contains("sport") || s.contains("physical education") || s.contains("hotel")
            || s.contains("tourism") || s.contains("culinary")
        {
            BroadField::Services
        } else if s.contains("symthaea") || s.contains("mycelix") || s.contains("consciousness computing")
            || s.contains("holographic") || s.contains("hdc")
        {
            BroadField::MetaLearning
        } else {
            // Default to NaturalSciencesMath for unclassified
            BroadField::NaturalSciencesMath
        }
    }

    pub fn label(&self) -> &str {
        match self {
            BroadField::Education => "Education",
            BroadField::ArtsHumanities => "Arts & Humanities",
            BroadField::SocialSciences => "Social Sciences",
            BroadField::BusinessLaw => "Business & Law",
            BroadField::NaturalSciencesMath => "Natural Sciences & Mathematics",
            BroadField::ICT => "Information & Communication Technologies",
            BroadField::EngineeringManufacturing => "Engineering & Manufacturing",
            BroadField::AgricultureEnvironment => "Agriculture & Environment",
            BroadField::HealthWelfare => "Health & Welfare",
            BroadField::Services => "Services & Recreation",
            BroadField::MetaLearning => "Meta-Learning (Consciousness Computing)",
        }
    }
}

// ============================================================
// Webb's Depth of Knowledge
// ============================================================

/// Webb's Depth of Knowledge levels (1-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WebbDOK {
    /// Level 1: Recall and Reproduction
    Recall = 1,
    /// Level 2: Skills and Concepts
    SkillConcept = 2,
    /// Level 3: Strategic Thinking
    StrategicThinking = 3,
    /// Level 4: Extended Thinking
    ExtendedThinking = 4,
}

impl WebbDOK {
    /// Infer DOK from Bloom level.
    pub fn from_bloom(bloom: &str) -> Self {
        match bloom {
            "Remember" => WebbDOK::Recall,
            "Understand" => WebbDOK::SkillConcept,
            "Apply" => WebbDOK::SkillConcept,
            "Analyze" => WebbDOK::StrategicThinking,
            "Evaluate" => WebbDOK::StrategicThinking,
            "Create" => WebbDOK::ExtendedThinking,
            _ => WebbDOK::SkillConcept,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            WebbDOK::Recall => "Recall & Reproduction",
            WebbDOK::SkillConcept => "Skills & Concepts",
            WebbDOK::StrategicThinking => "Strategic Thinking",
            WebbDOK::ExtendedThinking => "Extended Thinking",
        }
    }
}

// ============================================================
// Dreyfus Skill Acquisition Model
// ============================================================

/// Dreyfus model stages (1-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DreyfusStage {
    Novice = 1,
    AdvancedBeginner = 2,
    Competent = 3,
    Proficient = 4,
    Expert = 5,
}

impl DreyfusStage {
    /// Map from our difficulty level.
    pub fn from_difficulty(difficulty: &str) -> Self {
        match difficulty {
            "Beginner" => DreyfusStage::Novice,
            "Intermediate" => DreyfusStage::AdvancedBeginner,
            "Advanced" => DreyfusStage::Competent,
            "Expert" => DreyfusStage::Proficient,
            _ => DreyfusStage::AdvancedBeginner,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            DreyfusStage::Novice => "Novice",
            DreyfusStage::AdvancedBeginner => "Advanced Beginner",
            DreyfusStage::Competent => "Competent",
            DreyfusStage::Proficient => "Proficient",
            DreyfusStage::Expert => "Expert",
        }
    }
}

// ============================================================
// Enriched Node Metadata
// ============================================================

/// Extended taxonomy metadata that can be computed for any CurriculumNode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyMetadata {
    /// ISCED education level (0-8)
    pub isced_level: u8,
    /// ISCED-F broad field code (01-11)
    pub isced_field: String,
    /// ISCED-F broad field label
    pub isced_field_label: String,
    /// Webb's Depth of Knowledge (1-4)
    pub webb_dok: u8,
    /// Dreyfus skill stage (1-5)
    pub dreyfus_stage: u8,
    /// Bloom's taxonomy level (existing field, included for completeness)
    pub bloom_level: String,
}

/// Compute taxonomy metadata for a node from its existing fields.
pub fn enrich_node(
    grade_level: &str,
    subject_area: &str,
    bloom_level: &str,
    difficulty: &str,
) -> TaxonomyMetadata {
    let isced = IscedLevel::from_grade_level(grade_level);
    let field = BroadField::from_subject(subject_area);
    let dok = WebbDOK::from_bloom(bloom_level);
    let dreyfus = DreyfusStage::from_difficulty(difficulty);

    TaxonomyMetadata {
        isced_level: isced as u8,
        isced_field: field.code().to_string(),
        isced_field_label: field.label().to_string(),
        webb_dok: dok as u8,
        dreyfus_stage: dreyfus as u8,
        bloom_level: bloom_level.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isced_from_grade() {
        assert_eq!(IscedLevel::from_grade_level("PreK"), IscedLevel::EarlyChildhood);
        assert_eq!(IscedLevel::from_grade_level("Grade3"), IscedLevel::Primary);
        assert_eq!(IscedLevel::from_grade_level("Grade7"), IscedLevel::LowerSecondary);
        assert_eq!(IscedLevel::from_grade_level("Grade10"), IscedLevel::UpperSecondary);
        assert_eq!(IscedLevel::from_grade_level("Undergraduate"), IscedLevel::Bachelors);
        assert_eq!(IscedLevel::from_grade_level("Graduate"), IscedLevel::Masters);
        assert_eq!(IscedLevel::from_grade_level("Doctoral"), IscedLevel::Doctoral);
    }

    #[test]
    fn test_broad_field_classification() {
        assert_eq!(BroadField::from_subject("Mathematics"), BroadField::NaturalSciencesMath);
        assert_eq!(BroadField::from_subject("Computer Science"), BroadField::ICT);
        assert_eq!(BroadField::from_subject("Literature"), BroadField::ArtsHumanities);
        assert_eq!(BroadField::from_subject("Economics"), BroadField::SocialSciences);
        assert_eq!(BroadField::from_subject("Engineering"), BroadField::EngineeringManufacturing);
        assert_eq!(BroadField::from_subject("EnglishLanguageArts"), BroadField::ArtsHumanities);
        assert_eq!(BroadField::from_subject("Symthaea Architecture"), BroadField::MetaLearning);
    }

    #[test]
    fn test_webb_dok_from_bloom() {
        assert_eq!(WebbDOK::from_bloom("Remember"), WebbDOK::Recall);
        assert_eq!(WebbDOK::from_bloom("Apply"), WebbDOK::SkillConcept);
        assert_eq!(WebbDOK::from_bloom("Analyze"), WebbDOK::StrategicThinking);
        assert_eq!(WebbDOK::from_bloom("Create"), WebbDOK::ExtendedThinking);
    }

    #[test]
    fn test_dreyfus_from_difficulty() {
        assert_eq!(DreyfusStage::from_difficulty("Beginner"), DreyfusStage::Novice);
        assert_eq!(DreyfusStage::from_difficulty("Expert"), DreyfusStage::Proficient);
    }

    #[test]
    fn test_enrich_node() {
        let meta = enrich_node("Grade3", "Mathematics", "Apply", "Beginner");
        assert_eq!(meta.isced_level, 1); // Primary
        assert_eq!(meta.isced_field, "05"); // Natural Sciences & Math
        assert_eq!(meta.webb_dok, 2); // Skills & Concepts
        assert_eq!(meta.dreyfus_stage, 1); // Novice
    }

    #[test]
    fn test_isced_ordering() {
        assert!(IscedLevel::Doctoral > IscedLevel::Masters);
        assert!(IscedLevel::Masters > IscedLevel::Bachelors);
        assert!(IscedLevel::Bachelors > IscedLevel::UpperSecondary);
    }

    #[test]
    fn test_all_broad_fields_have_codes() {
        let fields = [
            BroadField::Education, BroadField::ArtsHumanities, BroadField::SocialSciences,
            BroadField::BusinessLaw, BroadField::NaturalSciencesMath, BroadField::ICT,
            BroadField::EngineeringManufacturing, BroadField::AgricultureEnvironment,
            BroadField::HealthWelfare, BroadField::Services, BroadField::MetaLearning,
        ];
        for f in &fields {
            assert!(!f.code().is_empty());
            assert!(!f.label().is_empty());
        }
        // Codes should be unique
        let codes: Vec<&str> = fields.iter().map(|f| f.code()).collect();
        let unique: std::collections::HashSet<&str> = codes.iter().copied().collect();
        assert_eq!(codes.len(), unique.len());
    }
}
