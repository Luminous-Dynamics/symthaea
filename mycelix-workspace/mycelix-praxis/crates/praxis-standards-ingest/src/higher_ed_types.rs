// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Types for higher education curriculum modeling.
//!
//! These intermediate types represent degree programs, courses, and learning
//! outcomes before conversion into the unified `CurriculumDocument` format.

use serde::{Deserialize, Serialize};

/// Academic level of a program or course.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcademicLevel {
    /// Associate degree (2-year)
    Associate,
    /// Bachelor's degree (4-year undergraduate)
    Undergraduate,
    /// Master's degree
    Masters,
    /// Doctoral / PhD
    Doctoral,
    /// Post-doctoral research
    PostDoctoral,
    /// Professional degree (JD, MD, PharmD, etc.)
    Professional,
    /// Certificate or micro-credential
    Certificate,
}

impl AcademicLevel {
    /// Convert to a GradeLevel-compatible string.
    pub fn to_grade_level(&self) -> &str {
        match self {
            AcademicLevel::Associate => "Undergraduate",
            AcademicLevel::Undergraduate => "Undergraduate",
            AcademicLevel::Masters => "Graduate",
            AcademicLevel::Doctoral => "Doctoral",
            AcademicLevel::PostDoctoral => "PostDoctoral",
            AcademicLevel::Professional => "Professional",
            AcademicLevel::Certificate => "Professional",
        }
    }

    /// Map to difficulty level.
    pub fn to_difficulty(&self) -> &str {
        match self {
            AcademicLevel::Associate => "Intermediate",
            AcademicLevel::Undergraduate => "Advanced",
            AcademicLevel::Masters => "Expert",
            AcademicLevel::Doctoral | AcademicLevel::PostDoctoral => "Expert",
            AcademicLevel::Professional => "Expert",
            AcademicLevel::Certificate => "Advanced",
        }
    }
}

impl std::fmt::Display for AcademicLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_grade_level())
    }
}

/// Course level within a program (numbered by conventional course numbering).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CourseLevel {
    /// 100-level introductory courses
    Introductory,
    /// 200-300 level intermediate courses
    Intermediate,
    /// 300-400 level upper division
    UpperDivision,
    /// 500-600 level graduate
    Graduate,
    /// 700+ level advanced graduate / research seminars
    AdvancedGraduate,
    /// Thesis, dissertation, or independent research
    Research,
}

impl CourseLevel {
    /// Map to difficulty level.
    pub fn to_difficulty(&self) -> &str {
        match self {
            CourseLevel::Introductory => "Intermediate",
            CourseLevel::Intermediate => "Intermediate",
            CourseLevel::UpperDivision => "Advanced",
            CourseLevel::Graduate => "Expert",
            CourseLevel::AdvancedGraduate => "Expert",
            CourseLevel::Research => "Expert",
        }
    }

    /// Conventional course number range.
    pub fn number_range(&self) -> &str {
        match self {
            CourseLevel::Introductory => "100",
            CourseLevel::Intermediate => "200-300",
            CourseLevel::UpperDivision => "300-400",
            CourseLevel::Graduate => "500-600",
            CourseLevel::AdvancedGraduate => "700+",
            CourseLevel::Research => "900",
        }
    }
}

/// A degree program descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramDescriptor {
    /// Unique identifier (e.g., "CIP:11.0701" or "ACM:CS2013")
    pub id: String,
    /// Program title (e.g., "Computer Science, BS")
    pub title: String,
    /// Subject area for classification (e.g., "Computer Science").
    /// If None, falls back to title.
    pub subject_area: Option<String>,
    /// Academic level
    pub level: AcademicLevel,
    /// CIP code if applicable
    pub cip_code: Option<String>,
    /// Institution name (None for generic/framework programs)
    pub institution: Option<String>,
    /// Total credit hours for degree completion
    pub total_credits: Option<u16>,
    /// Duration in semesters
    pub duration_semesters: Option<u8>,
    /// Courses in this program
    pub courses: Vec<CourseDescriptor>,
}

/// A course within a program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseDescriptor {
    /// Course identifier (e.g., "CS101", "AL/BasicAnalysis")
    pub id: String,
    /// Course title
    pub title: String,
    /// Course description
    pub description: String,
    /// Credit hours (US semester credits)
    pub credits: Option<u8>,
    /// Course level
    pub level: CourseLevel,
    /// Prerequisite course IDs
    pub prerequisites: Vec<String>,
    /// Corequisite course IDs
    pub corequisites: Vec<String>,
    /// Learning outcomes
    pub outcomes: Vec<OutcomeDescriptor>,
    /// Topic keywords
    pub topics: Vec<String>,
    /// Estimated total hours (contact + study). If None, computed from credits.
    pub estimated_hours: Option<u32>,
}

impl CourseDescriptor {
    /// Compute estimated hours from credits if not explicitly set.
    /// 1 US semester credit ≈ 45 total hours (15 contact + 30 study).
    pub fn total_hours(&self) -> u32 {
        self.estimated_hours
            .unwrap_or_else(|| self.credits.unwrap_or(3) as u32 * 45)
    }
}

/// A learning outcome for a course.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDescriptor {
    /// Description of what the student should be able to do.
    pub description: String,
    /// Bloom's taxonomy level.
    pub bloom_level: String,
    /// How this outcome is assessed (exam, project, paper, etc.)
    pub assessment_method: Option<String>,
}

/// A milestone in a PhD program progression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhDMilestone {
    /// Milestone identifier
    pub id: String,
    /// Milestone title
    pub title: String,
    /// Description
    pub description: String,
    /// Milestone type
    pub milestone_type: PhDMilestoneType,
    /// Typical year in program (1-based)
    pub typical_year: u8,
    /// Estimated hours
    pub estimated_hours: u32,
    /// Prerequisites (other milestone IDs)
    pub prerequisites: Vec<String>,
}

/// Types of PhD milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhDMilestoneType {
    /// Required coursework
    Coursework,
    /// Teaching requirement
    Teaching,
    /// Qualifying / comprehensive examination
    QualifyingExam,
    /// Dissertation proposal / prospectus defense
    ProposalDefense,
    /// Independent research phase
    Research,
    /// Dissertation writing
    DissertationWriting,
    /// Final dissertation defense
    DissertationDefense,
    /// Publication requirement
    Publication,
}

impl PhDMilestoneType {
    /// Map to NodeType string for curriculum JSON.
    pub fn to_node_type(&self) -> &str {
        match self {
            PhDMilestoneType::Coursework => "Course",
            PhDMilestoneType::Teaching => "Skill",
            PhDMilestoneType::QualifyingExam => "Assessment",
            PhDMilestoneType::ProposalDefense => "Assessment",
            PhDMilestoneType::Research => "Project",
            PhDMilestoneType::DissertationWriting => "Project",
            PhDMilestoneType::DissertationDefense => "Assessment",
            PhDMilestoneType::Publication => "Project",
        }
    }

    /// Map to Bloom level.
    pub fn to_bloom_level(&self) -> &str {
        match self {
            PhDMilestoneType::Coursework => "Analyze",
            PhDMilestoneType::Teaching => "Apply",
            PhDMilestoneType::QualifyingExam => "Evaluate",
            PhDMilestoneType::ProposalDefense => "Evaluate",
            PhDMilestoneType::Research => "Create",
            PhDMilestoneType::DissertationWriting => "Create",
            PhDMilestoneType::DissertationDefense => "Evaluate",
            PhDMilestoneType::Publication => "Create",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_academic_level_grade_mapping() {
        assert_eq!(AcademicLevel::Undergraduate.to_grade_level(), "Undergraduate");
        assert_eq!(AcademicLevel::Doctoral.to_grade_level(), "Doctoral");
        assert_eq!(AcademicLevel::Professional.to_grade_level(), "Professional");
    }

    #[test]
    fn test_course_level_difficulty() {
        assert_eq!(CourseLevel::Introductory.to_difficulty(), "Intermediate");
        assert_eq!(CourseLevel::Graduate.to_difficulty(), "Expert");
        assert_eq!(CourseLevel::Research.to_difficulty(), "Expert");
    }

    #[test]
    fn test_course_total_hours() {
        let course = CourseDescriptor {
            id: "CS101".into(),
            title: "Intro CS".into(),
            description: "Intro".into(),
            credits: Some(3),
            level: CourseLevel::Introductory,
            prerequisites: vec![],
            corequisites: vec![],
            outcomes: vec![],
            topics: vec![],
            estimated_hours: None,
        };
        assert_eq!(course.total_hours(), 135); // 3 * 45

        let explicit = CourseDescriptor {
            estimated_hours: Some(100),
            ..course
        };
        assert_eq!(explicit.total_hours(), 100);
    }

    #[test]
    fn test_phd_milestone_type_mapping() {
        assert_eq!(PhDMilestoneType::Research.to_node_type(), "Project");
        assert_eq!(PhDMilestoneType::QualifyingExam.to_node_type(), "Assessment");
        assert_eq!(PhDMilestoneType::Research.to_bloom_level(), "Create");
        assert_eq!(PhDMilestoneType::QualifyingExam.to_bloom_level(), "Evaluate");
    }

    #[test]
    fn test_academic_level_display() {
        assert_eq!(format!("{}", AcademicLevel::Doctoral), "Doctoral");
        assert_eq!(format!("{}", AcademicLevel::Masters), "Graduate");
    }
}
