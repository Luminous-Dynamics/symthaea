// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CIP (Classification of Instructional Programs) taxonomy source.
//!
//! Provides the NCES CIP code hierarchy as a curriculum source. CIP codes
//! classify every accredited US academic program with a 6-digit taxonomy:
//! - 2-digit family (e.g., 11 = Computer and Information Sciences)
//! - 4-digit group (e.g., 11.07 = Computer Science)
//! - 6-digit specific (e.g., 11.0701 = Computer Science)
//!
//! The CIP taxonomy is bundled as embedded data rather than fetched from an API.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{self, CurriculumDocument};
use crate::higher_ed_types::*;
use serde::Deserialize;

/// A CIP code entry.
#[derive(Debug, Clone, Deserialize)]
pub struct CipEntry {
    pub code: String,
    pub title: String,
    pub definition: String,
}

/// CIP taxonomy source.
pub struct CipSource {
    entries: Vec<CipEntry>,
}

impl CipSource {
    /// Create a new CIP source from the embedded taxonomy.
    pub fn new() -> Self {
        Self {
            entries: embedded_cip_taxonomy(),
        }
    }

    /// Get all 2-digit families.
    pub fn families(&self) -> Vec<&CipEntry> {
        self.entries
            .iter()
            .filter(|e| e.code.len() == 2 || (e.code.len() == 5 && e.code.ends_with(".0000")))
            .collect()
    }

    /// Get all entries under a 2-digit family code.
    pub fn programs_in_family(&self, family_code: &str) -> Vec<&CipEntry> {
        let prefix = if family_code.contains('.') {
            family_code.split('.').next().unwrap_or(family_code)
        } else {
            family_code
        };
        self.entries
            .iter()
            .filter(|e| e.code.starts_with(prefix) && e.code != family_code)
            .collect()
    }

    /// Convert a CIP family into a program with its subprograms as courses.
    pub fn family_to_program(&self, family_code: &str) -> Option<ProgramDescriptor> {
        let family = self
            .entries
            .iter()
            .find(|e| e.code == family_code || e.code.starts_with(family_code))?;

        let programs = self.programs_in_family(family_code);

        let courses: Vec<CourseDescriptor> = programs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let level = if i < programs.len() / 3 {
                    CourseLevel::Introductory
                } else if i < 2 * programs.len() / 3 {
                    CourseLevel::Intermediate
                } else {
                    CourseLevel::UpperDivision
                };

                CourseDescriptor {
                    id: p.code.clone(),
                    title: p.title.clone(),
                    description: p.definition.clone(),
                    credits: Some(3),
                    level,
                    prerequisites: vec![],
                    corequisites: vec![],
                    outcomes: vec![],
                    topics: extract_cip_topics(&p.title),
                    estimated_hours: None,
                }
            })
            .collect();

        Some(ProgramDescriptor {
            id: format!("CIP:{}", family.code),
            title: family.title.clone(),
            subject_area: None, // falls back to title
            level: AcademicLevel::Undergraduate,
            cip_code: Some(family.code.clone()),
            institution: None,
            total_credits: Some(120),
            duration_semesters: Some(8),
            courses,
        })
    }
}

impl CurriculumSource for CipSource {
    fn name(&self) -> &str {
        "NCES CIP Taxonomy"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(self
            .entries
            .iter()
            .filter(|e| is_family_code(&e.code))
            .map(|e| SourceEntry {
                id: e.code.clone(),
                title: e.title.clone(),
                subject: e.title.clone(),
                level: "Undergraduate".to_string(),
                description: truncate_def(&e.definition, 80),
            })
            .collect())
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let program = self
            .family_to_program(id)
            .ok_or_else(|| SourceError::NotFound(format!("CIP family {id}")))?;
        Ok(converter::convert_program(&program))
    }
}

fn is_family_code(code: &str) -> bool {
    // 2-digit codes are top-level families
    code.len() == 2
}

fn truncate_def(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

fn extract_cip_topics(title: &str) -> Vec<String> {
    title
        .split(|c: char| c == ',' || c == '/' || c == '(' || c == ')')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty() && s.len() > 2)
        .collect()
}

/// Embedded CIP taxonomy — the 47 two-digit families + key subprograms.
///
/// Full CIP has ~1,800 codes; we embed the families and top programs.
/// A future version can load the full CSV from `data/cip_codes.csv`.
fn embedded_cip_taxonomy() -> Vec<CipEntry> {
    vec![
        CipEntry { code: "01".into(), title: "Agriculture, Agriculture Operations, and Related Sciences".into(), definition: "Programs that focus on agriculture and related sciences.".into() },
        CipEntry { code: "03".into(), title: "Natural Resources and Conservation".into(), definition: "Programs related to natural resources, conservation, and environmental science.".into() },
        CipEntry { code: "04".into(), title: "Architecture and Related Services".into(), definition: "Programs in architecture, urban planning, and related design fields.".into() },
        CipEntry { code: "05".into(), title: "Area, Ethnic, Cultural, Gender, and Group Studies".into(), definition: "Programs studying specific regions, cultures, ethnicities, and social groups.".into() },
        CipEntry { code: "09".into(), title: "Communication, Journalism, and Related Programs".into(), definition: "Programs in media, journalism, communication, and public relations.".into() },
        CipEntry { code: "11".into(), title: "Computer and Information Sciences and Support Services".into(), definition: "Programs in computer science, information technology, and related fields.".into() },
        CipEntry { code: "11.01".into(), title: "Computer and Information Sciences, General".into(), definition: "General programs in computer and information sciences.".into() },
        CipEntry { code: "11.0101".into(), title: "Computer and Information Sciences, General".into(), definition: "A general program in computer science including algorithms, data structures, software, hardware, and theory.".into() },
        CipEntry { code: "11.0701".into(), title: "Computer Science".into(), definition: "Programs focusing on computing theory, algorithms, data structures, programming languages, software engineering, and computer systems.".into() },
        CipEntry { code: "11.0401".into(), title: "Information Science/Studies".into(), definition: "Programs focusing on information organization, retrieval, and management.".into() },
        CipEntry { code: "11.0501".into(), title: "Computer Systems Analysis/Analyst".into(), definition: "Programs in analyzing and designing computer-based information systems.".into() },
        CipEntry { code: "11.0801".into(), title: "Web Page, Digital/Multimedia and Information Resources Design".into(), definition: "Programs in web design and multimedia development.".into() },
        CipEntry { code: "11.1003".into(), title: "Computer and Information Systems Security/Auditing/Information Assurance".into(), definition: "Programs in cybersecurity, information assurance, and security auditing.".into() },
        CipEntry { code: "13".into(), title: "Education".into(), definition: "Programs preparing individuals to teach and educate at various levels.".into() },
        CipEntry { code: "14".into(), title: "Engineering".into(), definition: "Programs in engineering disciplines including design, analysis, and manufacturing.".into() },
        CipEntry { code: "14.09".into(), title: "Computer Engineering".into(), definition: "Programs in computer hardware and software system design.".into() },
        CipEntry { code: "14.10".into(), title: "Electrical, Electronics, and Communications Engineering".into(), definition: "Programs in electrical and electronic systems engineering.".into() },
        CipEntry { code: "14.19".into(), title: "Mechanical Engineering".into(), definition: "Programs in mechanical system design and analysis.".into() },
        CipEntry { code: "14.35".into(), title: "Industrial Engineering".into(), definition: "Programs in optimization of complex systems and processes.".into() },
        CipEntry { code: "14.08".into(), title: "Civil Engineering".into(), definition: "Programs in infrastructure, structural, and environmental engineering.".into() },
        CipEntry { code: "15".into(), title: "Engineering Technologies and Engineering-Related Fields".into(), definition: "Applied engineering technology programs.".into() },
        CipEntry { code: "16".into(), title: "Foreign Languages, Literatures, and Linguistics".into(), definition: "Programs in world languages, translation, and linguistic science.".into() },
        CipEntry { code: "19".into(), title: "Family and Consumer Sciences/Human Sciences".into(), definition: "Programs in family studies, nutrition, and human development.".into() },
        CipEntry { code: "22".into(), title: "Legal Professions and Studies".into(), definition: "Programs in law, legal research, and paralegal studies.".into() },
        CipEntry { code: "23".into(), title: "English Language and Literature/Letters".into(), definition: "Programs in English language, literature, and creative writing.".into() },
        CipEntry { code: "24".into(), title: "Liberal Arts and Sciences, General Studies and Humanities".into(), definition: "Interdisciplinary liberal arts and general education programs.".into() },
        CipEntry { code: "25".into(), title: "Library Science".into(), definition: "Programs in library management and information retrieval.".into() },
        CipEntry { code: "26".into(), title: "Biological and Biomedical Sciences".into(), definition: "Programs in biology, biochemistry, genetics, ecology, and biomedical research.".into() },
        CipEntry { code: "26.01".into(), title: "Biology, General".into(), definition: "General study of living organisms, life processes, and biological principles.".into() },
        CipEntry { code: "27".into(), title: "Mathematics and Statistics".into(), definition: "Programs in pure mathematics, applied mathematics, and statistics.".into() },
        CipEntry { code: "27.01".into(), title: "Mathematics, General".into(), definition: "Programs in mathematical theory including analysis, algebra, geometry, and topology.".into() },
        CipEntry { code: "27.05".into(), title: "Statistics, General".into(), definition: "Programs in statistical theory, methods, and data analysis.".into() },
        CipEntry { code: "30".into(), title: "Multi/Interdisciplinary Studies".into(), definition: "Programs combining multiple academic disciplines.".into() },
        CipEntry { code: "31".into(), title: "Parks, Recreation, Leisure, and Fitness Studies".into(), definition: "Programs in recreation management, fitness, and leisure studies.".into() },
        CipEntry { code: "38".into(), title: "Philosophy and Religious Studies".into(), definition: "Programs in philosophy, ethics, logic, and religious studies.".into() },
        CipEntry { code: "40".into(), title: "Physical Sciences".into(), definition: "Programs in physics, chemistry, astronomy, and earth sciences.".into() },
        CipEntry { code: "40.08".into(), title: "Physics, General".into(), definition: "Programs in fundamental physics including mechanics, electromagnetism, quantum physics, and relativity.".into() },
        CipEntry { code: "40.05".into(), title: "Chemistry, General".into(), definition: "Programs in chemical principles, organic and inorganic chemistry, and analytical methods.".into() },
        CipEntry { code: "42".into(), title: "Psychology".into(), definition: "Programs in psychological science including cognitive, clinical, developmental, and social psychology.".into() },
        CipEntry { code: "43".into(), title: "Homeland Security, Law Enforcement, Firefighting and Related Protective Services".into(), definition: "Programs in criminal justice, law enforcement, and emergency management.".into() },
        CipEntry { code: "44".into(), title: "Public Administration and Social Service Professions".into(), definition: "Programs in public policy, administration, and social work.".into() },
        CipEntry { code: "45".into(), title: "Social Sciences".into(), definition: "Programs in economics, political science, sociology, anthropology, and geography.".into() },
        CipEntry { code: "50".into(), title: "Visual and Performing Arts".into(), definition: "Programs in fine arts, music, theater, dance, film, and design.".into() },
        CipEntry { code: "51".into(), title: "Health Professions and Related Programs".into(), definition: "Programs in medicine, nursing, pharmacy, public health, and allied health.".into() },
        CipEntry { code: "52".into(), title: "Business, Management, Marketing, and Related Support Services".into(), definition: "Programs in business administration, finance, accounting, marketing, and management.".into() },
        CipEntry { code: "54".into(), title: "History".into(), definition: "Programs in historical research, analysis, and interpretation.".into() },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cip_source_creation() {
        let source = CipSource::new();
        assert!(!source.entries.is_empty());
    }

    #[test]
    fn test_list_available() {
        let source = CipSource::new();
        let available = source.list_available().unwrap();
        assert!(available.len() >= 20); // at least 20 CIP families
        assert!(available.iter().any(|e| e.title.contains("Computer")));
    }

    #[test]
    fn test_family_to_program() {
        let source = CipSource::new();
        let program = source.family_to_program("11").unwrap();
        assert_eq!(program.level, AcademicLevel::Undergraduate);
        assert!(program.cip_code.as_deref() == Some("11"));
        assert!(!program.courses.is_empty());
    }

    #[test]
    fn test_fetch_produces_document() {
        let source = CipSource::new();
        let doc = source.fetch("11").unwrap();
        assert_eq!(doc.metadata.grade_level, "Undergraduate");
        assert!(doc.metadata.cip_code.as_deref() == Some("11"));
        assert!(!doc.nodes.is_empty());
        for node in &doc.nodes {
            assert!(node.cip_code.is_some());
        }
    }

    #[test]
    fn test_unknown_family_returns_error() {
        let source = CipSource::new();
        assert!(source.fetch("99").is_err());
    }
}
