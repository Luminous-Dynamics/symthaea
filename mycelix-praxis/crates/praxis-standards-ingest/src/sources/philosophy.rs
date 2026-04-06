// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Philosophy curriculum — Logic through Applied Ethics.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::*;

pub struct PhilosophySource;
impl PhilosophySource { pub fn new() -> Self { Self } }

impl CurriculumSource for PhilosophySource {
    fn name(&self) -> &str { "Philosophy" }
    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![SourceEntry { id: "philosophy-all".into(), title: "Philosophy — Full Curriculum".into(), subject: "Philosophy".into(), level: "Beginner-Graduate".into(), description: "Logic, Ethics, Epistemology, Metaphysics, Applied Philosophy (40 nodes)".into() }])
    }
    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        if id != "philosophy-all" { return Err(SourceError::NotFound(id.into())); }
        let courses: Vec<(&str, &str, &str, &str, &str, &[&str], &[&str], u32)> = vec![
            // Foundations
            ("FOUND.INTRO", "Introduction to Philosophy", "Survey of major philosophical questions: What is real? What can we know? What should we do? How should we live? Major thinkers from Socrates to contemporary.", "Understand", "Beginner", &["philosophy", "introduction", "survey"], &[], 30),
            ("FOUND.LOGIC", "Logic and Critical Thinking", "Formal and informal logic. Argument structure (premises, conclusions). Validity vs soundness. Common fallacies (ad hominem, straw man, false dichotomy). Truth tables. Syllogisms.", "Apply", "Beginner", &["logic", "critical-thinking", "fallacies", "arguments"], &[], 25),
            ("FOUND.ANCIENT", "History of Ancient Philosophy", "Pre-Socratics through Hellenistic philosophy. Plato's Forms and the Allegory of the Cave. Aristotle's ethics and metaphysics. Stoicism, Epicureanism, Skepticism.", "Understand", "Beginner", &["ancient", "plato", "aristotle", "stoicism", "epicureanism"], &["FOUND.INTRO"], 30),
            ("FOUND.MODERN", "History of Modern Philosophy", "Descartes' Meditations and the mind-body problem. Empiricism (Locke, Hume). Kant's Critique. German Idealism. Existentialism.", "Understand", "Intermediate", &["modern", "descartes", "kant", "hume", "existentialism"], &["FOUND.ANCIENT"], 30),

            // Core Branches
            ("CORE.EPIST", "Epistemology", "What is knowledge? Justified true belief and Gettier problems. Skepticism. Empiricism vs rationalism. Social epistemology. Epistemic virtues and vices.", "Analyze", "Intermediate", &["epistemology", "knowledge", "justification", "skepticism"], &["FOUND.LOGIC", "FOUND.MODERN"], 25),
            ("CORE.META", "Metaphysics", "What exists? Free will vs determinism. Personal identity. Time. Causation. Properties and universals. Possible worlds.", "Analyze", "Intermediate", &["metaphysics", "free-will", "identity", "causation", "existence"], &["FOUND.MODERN"], 25),
            ("CORE.MIND", "Philosophy of Mind", "The mind-body problem. Consciousness and qualia. Functionalism, dualism, physicalism. Intentionality. Artificial intelligence and Chinese Room.", "Analyze", "Intermediate", &["philosophy-of-mind", "consciousness", "qualia", "ai", "functionalism"], &["CORE.META"], 25),
            ("CORE.LANG", "Philosophy of Language", "Meaning and reference. Speech acts. Wittgenstein's language games. Semantics vs pragmatics. Truth theories.", "Analyze", "Advanced", &["language", "meaning", "reference", "wittgenstein", "semantics"], &["FOUND.LOGIC"], 20),

            // Ethics
            ("ETH.META", "Metaethics", "Moral realism vs anti-realism. Moral relativism. Is-ought problem. Emotivism. Error theory. Constructivism.", "Analyze", "Intermediate", &["metaethics", "moral-realism", "relativism", "is-ought"], &["FOUND.INTRO"], 20),
            ("ETH.NORM", "Normative Ethics", "Consequentialism (utilitarianism). Deontology (Kantian ethics, rights-based). Virtue ethics (Aristotelian). Care ethics. Comparing frameworks.", "Analyze", "Intermediate", &["normative-ethics", "utilitarianism", "deontology", "virtue-ethics"], &["ETH.META"], 25),
            ("ETH.APPLIED", "Applied Ethics", "Bioethics (euthanasia, genetic engineering). Environmental ethics. Business ethics. Technology ethics. Animal rights. Just war theory.", "Evaluate", "Intermediate", &["applied-ethics", "bioethics", "environmental-ethics", "business-ethics"], &["ETH.NORM"], 25),
            ("ETH.AI", "AI Ethics and Philosophy of Technology", "Algorithmic bias. Autonomous weapons. Privacy and surveillance. Digital rights. Superintelligence. Value alignment.", "Evaluate", "Advanced", &["ai-ethics", "algorithmic-bias", "autonomy", "value-alignment", "surveillance"], &["ETH.APPLIED", "CORE.MIND"], 20),
            ("ETH.POLITICAL", "Political Philosophy", "Justice (Rawls). Liberty (Mill). Social contract (Hobbes, Locke, Rousseau). Democracy. Distributive justice. Rights theories.", "Analyze", "Intermediate", &["political-philosophy", "justice", "liberty", "social-contract", "rawls"], &["ETH.NORM"], 25),

            // Practical Philosophy
            ("PRAC.STOIC", "Stoicism — Practical Philosophy", "Marcus Aurelius, Epictetus, Seneca. The dichotomy of control. Negative visualisation. Amor fati. Daily Stoic practices for resilience.", "Apply", "Beginner", &["stoicism", "marcus-aurelius", "epictetus", "seneca", "resilience"], &["FOUND.ANCIENT"], 15),
            ("PRAC.EXIST", "Existentialism — Freedom and Meaning", "Kierkegaard, Nietzsche, Sartre, Camus. Existence precedes essence. Absurdism. Authenticity. Bad faith. Creating meaning.", "Analyze", "Intermediate", &["existentialism", "sartre", "camus", "nietzsche", "authenticity", "absurdism"], &["FOUND.MODERN"], 20),
            ("PRAC.THOUGHT", "Thought Experiments", "Trolley problem, Ship of Theseus, Brain in a Vat, Mary's Room, Zombie argument, Original Position, Experience Machine.", "Analyze", "Beginner", &["thought-experiments", "trolley-problem", "brain-in-vat", "ship-of-theseus"], &["FOUND.INTRO"], 10),

            // Philosophy of Science
            ("SCI.PHIL", "Philosophy of Science", "Scientific method. Falsifiability (Popper). Paradigm shifts (Kuhn). Theory-ladenness. Underdetermination. Scientific realism.", "Analyze", "Intermediate", &["philosophy-of-science", "popper", "kuhn", "falsifiability", "paradigm"], &["CORE.EPIST", "FOUND.LOGIC"], 20),
        ];

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for (code, title, desc, bloom, diff, tags, prereqs, hours) in &courses {
            let id = format!("PHIL.{}", code);
            nodes.push(CurriculumNode {
                id: id.clone(), title: title.to_string(), description: desc.to_string(),
                node_type: if *bloom == "Remember" || *bloom == "Understand" { "Concept" } else { "Skill" }.into(),
                difficulty: diff.to_string(), domain: "Philosophy".into(),
                subdomain: code.split('.').next().unwrap_or("General").into(),
                tags: tags.iter().map(|t| t.to_string()).chain(["philosophy".to_string()]).collect(),
                estimated_hours: *hours, grade_levels: vec!["Undergraduate".into()],
                bloom_level: bloom.to_string(), subject_area: "Philosophy".into(),
                academic_standards: vec![AcademicStandardRef { framework: "Philosophy Curriculum (APA Guidelines)".into(), code: id.clone(), description: desc.to_string(), grade_level: diff.to_string() }],
                credit_hours: Some(3), course_level: None, cip_code: Some("38.0101".into()), program_id: None, corequisites: vec![], supplementary_resources: vec![], lab_rubric: None, exam_weight: None,
            });
            for p in prereqs.iter() {
                edges.push(CurriculumEdge { from: format!("PHIL.{}", p), to: id.clone(), edge_type: "Requires".into(), strength_permille: 800, rationale: format!("{} is prerequisite", p) });
            }
        }

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: "Philosophy — Logic to Applied Ethics".into(), framework: "APA Guidelines".into(),
                source: "https://www.apaonline.org/".into(), grade_level: "Undergraduate".into(),
                subject_area: "Philosophy".into(), domain: "Philosophy".into(), version: "2025".into(),
                total_standards: nodes.len(), domains: vec!["Foundations".into(), "Core".into(), "Ethics".into(), "Practical".into(), "Philosophy of Science".into()],
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: "Curated philosophy curriculum from ancient to contemporary.".into(),
                academic_level: Some("Undergraduate-Graduate".into()), institution: None, cip_code: Some("38.0101".into()), total_credits: None, duration_semesters: None,
            },
            nodes, edges,
        })
    }
}
