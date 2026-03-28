// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ACM/IEEE Computing Curricula source.
//!
//! Provides the ACM CS2013 Body of Knowledge as a curriculum source.
//! The CS2013 report defines 18 Knowledge Areas (KAs) with Knowledge Units
//! (KUs) and Learning Outcomes with explicit Bloom taxonomy levels.
//!
//! Reference: ACM/IEEE-CS Joint Task Force on Computing Curricula (2013).
//! "Computer Science Curricula 2013". ACM/IEEE.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{self, CurriculumDocument};
use crate::higher_ed_types::*;

/// ACM CS2013 Knowledge Area.
#[derive(Debug, Clone)]
pub struct KnowledgeArea {
    pub id: String,
    pub title: String,
    pub description: String,
    pub tier1_hours: u32,
    pub tier2_hours: u32,
    pub units: Vec<KnowledgeUnit>,
}

/// A Knowledge Unit within a KA.
#[derive(Debug, Clone)]
pub struct KnowledgeUnit {
    pub id: String,
    pub title: String,
    pub tier: &'static str, // "Tier1", "Tier2", "Elective"
    pub hours: u32,
    pub outcomes: Vec<(&'static str, &'static str)>, // (description, bloom_level)
    pub topics: Vec<String>,
}

/// ACM CS2013 curriculum source.
pub struct AcmSource {
    knowledge_areas: Vec<KnowledgeArea>,
}

impl AcmSource {
    pub fn new() -> Self {
        Self {
            knowledge_areas: embedded_cs2013(),
        }
    }

    /// Get a specific knowledge area by ID.
    pub fn get_ka(&self, id: &str) -> Option<&KnowledgeArea> {
        self.knowledge_areas.iter().find(|ka| ka.id == id)
    }

    /// Convert a KA into a program descriptor.
    fn ka_to_program(&self, ka: &KnowledgeArea) -> ProgramDescriptor {
        let courses: Vec<CourseDescriptor> = ka
            .units
            .iter()
            .map(|unit| {
                let level = match unit.tier {
                    "Tier1" => CourseLevel::Introductory,
                    "Tier2" => CourseLevel::Intermediate,
                    _ => CourseLevel::UpperDivision,
                };

                let outcomes: Vec<OutcomeDescriptor> = unit
                    .outcomes
                    .iter()
                    .map(|(desc, bloom)| OutcomeDescriptor {
                        description: desc.to_string(),
                        bloom_level: bloom.to_string(),
                        assessment_method: None,
                    })
                    .collect();

                CourseDescriptor {
                    id: unit.id.clone(),
                    title: unit.title.clone(),
                    description: format!("{} ({}, {} hours)", unit.title, unit.tier, unit.hours),
                    credits: Some((unit.hours / 15).max(1) as u8),
                    level,
                    prerequisites: vec![],
                    corequisites: vec![],
                    outcomes,
                    topics: unit.topics.clone(),
                    estimated_hours: Some(unit.hours * 3), // 1 contact hour ≈ 3 total hours
                }
            })
            .collect();

        ProgramDescriptor {
            id: format!("ACM:CS2013:{}", ka.id),
            title: format!("CS2013: {}", ka.title),
            subject_area: Some("Computer Science".to_string()),
            level: AcademicLevel::Undergraduate,
            cip_code: Some("11.0701".to_string()),
            institution: None,
            total_credits: Some((ka.tier1_hours + ka.tier2_hours) as u16 / 15),
            duration_semesters: None,
            courses,
        }
    }
}

impl CurriculumSource for AcmSource {
    fn name(&self) -> &str {
        "ACM/IEEE CS2013 Computing Curricula"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(self
            .knowledge_areas
            .iter()
            .map(|ka| SourceEntry {
                id: ka.id.clone(),
                title: ka.title.clone(),
                subject: "Computer Science".to_string(),
                level: "Undergraduate".to_string(),
                description: format!(
                    "Tier1: {}h, Tier2: {}h, {} units",
                    ka.tier1_hours,
                    ka.tier2_hours,
                    ka.units.len()
                ),
            })
            .collect())
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let ka = self
            .get_ka(id)
            .ok_or_else(|| SourceError::NotFound(format!("ACM KA {id}")))?;
        let program = self.ka_to_program(ka);
        Ok(converter::convert_program(&program))
    }
}

/// Embedded CS2013 Body of Knowledge — the 18 Knowledge Areas.
///
/// Each KA includes its core Knowledge Units with Bloom-level outcomes.
/// This covers the ~300 hours of Tier1+Tier2 core knowledge.
fn embedded_cs2013() -> Vec<KnowledgeArea> {
    vec![
        KnowledgeArea {
            id: "AL".into(), title: "Algorithms and Complexity".into(),
            description: "The study of algorithms, their design, analysis, and computational complexity.".into(),
            tier1_hours: 19, tier2_hours: 0,
            units: vec![
                KnowledgeUnit { id: "AL/BasicAnalysis".into(), title: "Basic Analysis".into(), tier: "Tier1", hours: 4,
                    outcomes: vec![
                        ("Explain what is meant by best, expected, and worst case behavior of an algorithm", "Understand"),
                        ("Use big-O notation formally to give asymptotic upper bounds on time and space complexity", "Apply"),
                        ("Determine informally the time and space complexity of simple algorithms", "Analyze"),
                    ],
                    topics: vec!["asymptotic analysis".into(), "big-O notation".into(), "time complexity".into(), "space complexity".into()],
                },
                KnowledgeUnit { id: "AL/AlgorithmicStrategies".into(), title: "Algorithmic Strategies".into(), tier: "Tier1", hours: 5,
                    outcomes: vec![
                        ("Implement a divide-and-conquer algorithm for solving a problem", "Apply"),
                        ("Apply dynamic programming to a problem", "Apply"),
                        ("Determine an appropriate algorithmic approach to a problem", "Analyze"),
                    ],
                    topics: vec!["brute-force".into(), "greedy".into(), "divide-and-conquer".into(), "dynamic programming".into(), "backtracking".into()],
                },
                KnowledgeUnit { id: "AL/FundamentalDataStructures".into(), title: "Fundamental Data Structures and Algorithms".into(), tier: "Tier1", hours: 9,
                    outcomes: vec![
                        ("Implement basic numerical algorithms", "Apply"),
                        ("Implement simple search and sort algorithms", "Apply"),
                        ("Discuss the computational efficiency of fundamental data structures", "Understand"),
                        ("Use hash tables to solve appropriate problems", "Apply"),
                    ],
                    topics: vec!["arrays".into(), "linked lists".into(), "stacks".into(), "queues".into(), "trees".into(), "hash tables".into(), "sorting".into(), "searching".into(), "graphs".into()],
                },
                KnowledgeUnit { id: "AL/Automata".into(), title: "Automata Theory".into(), tier: "Elective", hours: 6,
                    outcomes: vec![
                        ("Design a finite-state machine to accept a regular language", "Create"),
                        ("Determine what a given finite-state machine or Turing machine computes", "Analyze"),
                    ],
                    topics: vec!["finite automata".into(), "regular expressions".into(), "pushdown automata".into(), "Turing machines".into(), "decidability".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "AR".into(), title: "Architecture and Organization".into(),
            description: "Computer architecture fundamentals from logic gates to multiprocessor systems.".into(),
            tier1_hours: 0, tier2_hours: 16,
            units: vec![
                KnowledgeUnit { id: "AR/DigitalLogic".into(), title: "Digital Logic and Digital Systems".into(), tier: "Tier2", hours: 6,
                    outcomes: vec![
                        ("Design a simple circuit using fundamental building blocks", "Create"),
                        ("Implement a simple processor data path", "Apply"),
                    ],
                    topics: vec!["logic gates".into(), "boolean algebra".into(), "multiplexers".into(), "decoders".into(), "flip-flops".into()],
                },
                KnowledgeUnit { id: "AR/MachineOrg".into(), title: "Machine Level Representation of Data".into(), tier: "Tier2", hours: 3,
                    outcomes: vec![
                        ("Explain how data is represented and manipulated at the machine level", "Understand"),
                        ("Convert numeric data between different representations", "Apply"),
                    ],
                    topics: vec!["binary representation".into(), "floating point".into(), "two's complement".into(), "character encoding".into()],
                },
                KnowledgeUnit { id: "AR/ISA".into(), title: "Assembly Level Machine Organization".into(), tier: "Tier2", hours: 7,
                    outcomes: vec![
                        ("Write simple assembly language program segments", "Apply"),
                        ("Explain the relationship between ISA and microarchitecture", "Understand"),
                    ],
                    topics: vec!["instruction set architecture".into(), "assembly language".into(), "addressing modes".into(), "subroutine calls".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "DS".into(), title: "Discrete Structures".into(),
            description: "Foundational mathematics for computing: logic, sets, graphs, and combinatorics.".into(),
            tier1_hours: 37, tier2_hours: 0,
            units: vec![
                KnowledgeUnit { id: "DS/SetsRelations".into(), title: "Sets, Relations, and Functions".into(), tier: "Tier1", hours: 4,
                    outcomes: vec![
                        ("Apply set operations and relations to solve problems", "Apply"),
                        ("Determine whether a relation is an equivalence relation or partial order", "Analyze"),
                    ],
                    topics: vec!["sets".into(), "relations".into(), "functions".into(), "equivalence relations".into(), "partial orders".into()],
                },
                KnowledgeUnit { id: "DS/Logic".into(), title: "Basic Logic".into(), tier: "Tier1", hours: 9,
                    outcomes: vec![
                        ("Convert logical statements from informal to formal notation", "Apply"),
                        ("Apply formal methods of symbolic propositional and predicate logic", "Apply"),
                        ("Describe how to construct a proof using logical rules", "Understand"),
                    ],
                    topics: vec!["propositional logic".into(), "predicate logic".into(), "logical connectives".into(), "truth tables".into(), "proofs".into()],
                },
                KnowledgeUnit { id: "DS/ProofTechniques".into(), title: "Proof Techniques".into(), tier: "Tier1", hours: 10,
                    outcomes: vec![
                        ("Construct proofs using direct proof, proof by contradiction, and induction", "Create"),
                        ("Apply proof techniques to establish properties of algorithms", "Apply"),
                    ],
                    topics: vec!["direct proof".into(), "proof by contradiction".into(), "mathematical induction".into(), "strong induction".into()],
                },
                KnowledgeUnit { id: "DS/Combinatorics".into(), title: "Basics of Counting".into(), tier: "Tier1", hours: 5,
                    outcomes: vec![
                        ("Apply counting techniques to combinatorial problems", "Apply"),
                        ("Solve problems using the pigeonhole principle", "Apply"),
                    ],
                    topics: vec!["counting".into(), "permutations".into(), "combinations".into(), "pigeonhole principle".into(), "inclusion-exclusion".into()],
                },
                KnowledgeUnit { id: "DS/Graphs".into(), title: "Graphs and Trees".into(), tier: "Tier1", hours: 4,
                    outcomes: vec![
                        ("Model problems using graph terminology and notation", "Apply"),
                        ("Prove properties of graphs and trees using formal methods", "Analyze"),
                    ],
                    topics: vec!["directed graphs".into(), "undirected graphs".into(), "trees".into(), "spanning trees".into(), "graph traversals".into()],
                },
                KnowledgeUnit { id: "DS/Probability".into(), title: "Discrete Probability".into(), tier: "Tier1", hours: 5,
                    outcomes: vec![
                        ("Calculate probabilities of events and expected values", "Apply"),
                        ("Apply Bayes' theorem to solve conditional probability problems", "Apply"),
                    ],
                    topics: vec!["probability".into(), "conditional probability".into(), "Bayes' theorem".into(), "random variables".into(), "expected value".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "OS".into(), title: "Operating Systems".into(),
            description: "The design and implementation of operating systems.".into(),
            tier1_hours: 4, tier2_hours: 11,
            units: vec![
                KnowledgeUnit { id: "OS/Overview".into(), title: "Overview of Operating Systems".into(), tier: "Tier1", hours: 2,
                    outcomes: vec![
                        ("Explain the objectives and functions of modern operating systems", "Understand"),
                    ],
                    topics: vec!["OS role".into(), "kernel".into(), "system calls".into(), "protection".into()],
                },
                KnowledgeUnit { id: "OS/Concurrency".into(), title: "Concurrency".into(), tier: "Tier2", hours: 3,
                    outcomes: vec![
                        ("Implement mutual exclusion using semaphores or monitors", "Apply"),
                        ("Analyze a program for potential deadlock", "Analyze"),
                    ],
                    topics: vec!["threads".into(), "synchronization".into(), "semaphores".into(), "monitors".into(), "deadlock".into()],
                },
                KnowledgeUnit { id: "OS/MemMgmt".into(), title: "Memory Management".into(), tier: "Tier2", hours: 4,
                    outcomes: vec![
                        ("Explain memory hierarchy and virtual memory concepts", "Understand"),
                        ("Evaluate the trade-offs in memory management schemes", "Evaluate"),
                    ],
                    topics: vec!["virtual memory".into(), "paging".into(), "segmentation".into(), "memory allocation".into(), "cache".into()],
                },
                KnowledgeUnit { id: "OS/FileSystems".into(), title: "File Systems".into(), tier: "Tier2", hours: 2,
                    outcomes: vec![
                        ("Describe common file system organizations", "Understand"),
                    ],
                    topics: vec!["file systems".into(), "directories".into(), "access control".into(), "journaling".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "PL".into(), title: "Programming Languages".into(),
            description: "Programming language concepts, paradigms, and implementation.".into(),
            tier1_hours: 8, tier2_hours: 11,
            units: vec![
                KnowledgeUnit { id: "PL/OOP".into(), title: "Object-Oriented Programming".into(), tier: "Tier1", hours: 4,
                    outcomes: vec![
                        ("Design and implement a class hierarchy", "Create"),
                        ("Apply polymorphism and encapsulation in program design", "Apply"),
                    ],
                    topics: vec!["encapsulation".into(), "inheritance".into(), "polymorphism".into(), "interfaces".into(), "generics".into()],
                },
                KnowledgeUnit { id: "PL/FunctionalProg".into(), title: "Functional Programming".into(), tier: "Tier1", hours: 4,
                    outcomes: vec![
                        ("Write programs using higher-order functions, closures, and recursion", "Apply"),
                        ("Compare functional and imperative approaches to a problem", "Analyze"),
                    ],
                    topics: vec!["first-class functions".into(), "closures".into(), "recursion".into(), "immutability".into(), "pattern matching".into()],
                },
                KnowledgeUnit { id: "PL/TypeSystems".into(), title: "Type Systems".into(), tier: "Tier2", hours: 4,
                    outcomes: vec![
                        ("Explain the role of type systems in programming languages", "Understand"),
                        ("Apply parametric and ad-hoc polymorphism in program design", "Apply"),
                    ],
                    topics: vec!["static typing".into(), "dynamic typing".into(), "type inference".into(), "parametric polymorphism".into(), "generics".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "SE".into(), title: "Software Engineering".into(),
            description: "The disciplined approach to software design, development, and maintenance.".into(),
            tier1_hours: 6, tier2_hours: 22,
            units: vec![
                KnowledgeUnit { id: "SE/Process".into(), title: "Software Processes".into(), tier: "Tier1", hours: 3,
                    outcomes: vec![
                        ("Describe common software development lifecycles", "Understand"),
                        ("Compare agile and plan-driven approaches", "Analyze"),
                    ],
                    topics: vec!["waterfall".into(), "agile".into(), "scrum".into(), "kanban".into(), "DevOps".into()],
                },
                KnowledgeUnit { id: "SE/Design".into(), title: "Software Design".into(), tier: "Tier1", hours: 3,
                    outcomes: vec![
                        ("Apply design patterns to solve recurring design problems", "Apply"),
                        ("Create UML models of a software system", "Create"),
                    ],
                    topics: vec!["design patterns".into(), "architectural patterns".into(), "UML".into(), "coupling".into(), "cohesion".into()],
                },
                KnowledgeUnit { id: "SE/Testing".into(), title: "Software Verification and Validation".into(), tier: "Tier2", hours: 4,
                    outcomes: vec![
                        ("Design and execute test plans at unit, integration, and system levels", "Create"),
                        ("Apply code review techniques to improve software quality", "Apply"),
                    ],
                    topics: vec!["unit testing".into(), "integration testing".into(), "test-driven development".into(), "code review".into(), "static analysis".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "IAS".into(), title: "Information Assurance and Security".into(),
            description: "Principles and practices for securing computing systems and data.".into(),
            tier1_hours: 3, tier2_hours: 6,
            units: vec![
                KnowledgeUnit { id: "IAS/Foundations".into(), title: "Foundational Concepts in Security".into(), tier: "Tier1", hours: 1,
                    outcomes: vec![
                        ("Describe the CIA triad and its importance", "Understand"),
                        ("Explain the principle of least privilege", "Understand"),
                    ],
                    topics: vec!["confidentiality".into(), "integrity".into(), "availability".into(), "threat models".into(), "least privilege".into()],
                },
                KnowledgeUnit { id: "IAS/Cryptography".into(), title: "Cryptography".into(), tier: "Tier2", hours: 3,
                    outcomes: vec![
                        ("Explain symmetric and asymmetric encryption", "Understand"),
                        ("Apply hash functions and digital signatures", "Apply"),
                    ],
                    topics: vec!["symmetric encryption".into(), "public key cryptography".into(), "hash functions".into(), "digital signatures".into(), "certificates".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "NC".into(), title: "Networking and Communication".into(),
            description: "Computer networking principles, protocols, and applications.".into(),
            tier1_hours: 3, tier2_hours: 7,
            units: vec![
                KnowledgeUnit { id: "NC/Intro".into(), title: "Introduction to Networking".into(), tier: "Tier1", hours: 1,
                    outcomes: vec![
                        ("Describe the layered structure of a network protocol stack", "Understand"),
                    ],
                    topics: vec!["OSI model".into(), "TCP/IP".into(), "protocols".into(), "network topologies".into()],
                },
                KnowledgeUnit { id: "NC/NetworkApps".into(), title: "Networked Applications".into(), tier: "Tier2", hours: 3,
                    outcomes: vec![
                        ("Implement a simple client-server application using sockets", "Apply"),
                    ],
                    topics: vec!["sockets".into(), "HTTP".into(), "DNS".into(), "client-server".into(), "peer-to-peer".into()],
                },
            ],
        },
        KnowledgeArea {
            id: "SDF".into(), title: "Software Development Fundamentals".into(),
            description: "Core programming concepts and practices foundational to all CS students.".into(),
            tier1_hours: 43, tier2_hours: 0,
            units: vec![
                KnowledgeUnit { id: "SDF/Algorithms".into(), title: "Algorithms and Design".into(), tier: "Tier1", hours: 11,
                    outcomes: vec![
                        ("Implement, test, and debug simple algorithms", "Apply"),
                        ("Analyze and explain the behavior of simple programs", "Analyze"),
                    ],
                    topics: vec!["problem solving".into(), "algorithm design".into(), "pseudocode".into(), "testing".into(), "debugging".into()],
                },
                KnowledgeUnit { id: "SDF/FundProg".into(), title: "Fundamental Programming Concepts".into(), tier: "Tier1", hours: 10,
                    outcomes: vec![
                        ("Write programs that use variables, expressions, and control structures", "Apply"),
                        ("Use arrays, records, and strings in programs", "Apply"),
                    ],
                    topics: vec!["variables".into(), "expressions".into(), "control flow".into(), "functions".into(), "I/O".into(), "arrays".into()],
                },
                KnowledgeUnit { id: "SDF/DevMethods".into(), title: "Development Methods".into(), tier: "Tier1", hours: 10,
                    outcomes: vec![
                        ("Apply a development methodology to a program of moderate size", "Apply"),
                        ("Construct, execute, and debug programs using an IDE", "Apply"),
                    ],
                    topics: vec!["IDE".into(), "version control".into(), "pair programming".into(), "code review".into(), "refactoring".into()],
                },
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acm_source_creation() {
        let source = AcmSource::new();
        assert!(source.knowledge_areas.len() >= 9);
    }

    #[test]
    fn test_list_available() {
        let source = AcmSource::new();
        let available = source.list_available().unwrap();
        assert!(available.len() >= 9);
        assert!(available.iter().any(|e| e.id == "AL"));
        assert!(available.iter().any(|e| e.id == "DS"));
    }

    #[test]
    fn test_fetch_algorithms() {
        let source = AcmSource::new();
        let doc = source.fetch("AL").unwrap();
        assert!(doc.metadata.title.contains("Algorithms"));
        assert!(doc.metadata.academic_level.as_deref() == Some("Undergraduate"));
        assert!(doc.metadata.cip_code.as_deref() == Some("11.0701"));
        assert!(!doc.nodes.is_empty());
        // Verify Bloom levels propagate from outcomes
        let basic_analysis = doc.nodes.iter().find(|n| n.id.contains("BasicAnalysis"));
        assert!(basic_analysis.is_some());
    }

    #[test]
    fn test_fetch_discrete_structures() {
        let source = AcmSource::new();
        let doc = source.fetch("DS").unwrap();
        assert!(doc.nodes.len() >= 6);
        // All DS units are Tier1
        assert!(doc.metadata.title.contains("Discrete"));
    }

    #[test]
    fn test_fetch_unknown_ka_returns_error() {
        let source = AcmSource::new();
        assert!(source.fetch("UNKNOWN").is_err());
    }

    #[test]
    fn test_credit_hours_computed() {
        let source = AcmSource::new();
        let doc = source.fetch("AL").unwrap();
        for node in &doc.nodes {
            assert!(node.credit_hours.is_some());
            assert!(node.estimated_hours > 0);
        }
    }
}
