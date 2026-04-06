// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Luminous Dynamics curriculum — self-referential courses about Symthaea and Mycelix.
//!
//! ISCED-F Field 11: Meta-Learning (Consciousness Computing).
//!
//! These courses teach people how the system that powers their education actually
//! works — from HDC fundamentals through consciousness-gated governance.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{
    AcademicStandardRef, CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode,
};

/// Luminous Dynamics curriculum source.
pub struct LuminousSource;

impl LuminousSource {
    pub fn new() -> Self { Self }
}

impl CurriculumSource for LuminousSource {
    fn name(&self) -> &str { "Luminous Dynamics (Symthaea & Mycelix)" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            SourceEntry { id: "symthaea".into(), title: "Symthaea: Consciousness Computing".into(), subject: "Consciousness Computing".into(), level: "Undergraduate-Doctoral".into(), description: "10 courses: HDC, IIT, CfC, Active Inference, Moral Algebra".into() },
            SourceEntry { id: "mycelix".into(), title: "Mycelix: Decentralized Civic Infrastructure".into(), subject: "Decentralized Systems".into(), level: "Undergraduate-Graduate".into(), description: "7 courses: Holochain, Governance, Identity, Consciousness Gating".into() },
            SourceEntry { id: "programming".into(), title: "Programming Languages for AI & Humans".into(), subject: "Computer Science".into(), level: "Beginner-Advanced".into(), description: "Rust, Nix, Python courses — designed for both human and AI learners (Symthaea School)".into() },
        ])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        match id {
            "symthaea" => Ok(symthaea_curriculum()),
            "mycelix" => Ok(mycelix_curriculum()),
            "programming" => Ok(programming_curriculum()),
            _ => Err(SourceError::NotFound(id.to_string())),
        }
    }
}

fn symthaea_curriculum() -> CurriculumDocument {
    let courses = vec![
        // Foundations
        ("SYM-101", "Introduction to Consciousness Computing", "Undergraduate",
         "Survey of computational consciousness: what it means for a machine to be conscious, \
          the Hard Problem, philosophical zombies, multiple realizability, and why it matters \
          for AI safety and value alignment.",
         "Understand", "Beginner", vec!["consciousness", "philosophy-of-mind", "ai-safety"],
         vec![], 45),

        ("SYM-201", "Hyperdimensional Computing Fundamentals", "Undergraduate",
         "16,384-dimensional binary and continuous hypervectors. Encoding (random indexing, \
          spatial, temporal), binding (XOR/Hadamard), bundling (majority/sum), similarity \
          (Hamming/cosine). Holographic reduced representations. Applications to classification, \
          reasoning, and memory.",
         "Apply", "Intermediate", vec!["hdc", "hypervectors", "encoding", "binding", "similarity"],
         vec!["SYM-101"], 90),

        ("SYM-202", "Integrated Information Theory (IIT/Phi)", "Undergraduate",
         "Tononi's IIT axioms and postulates. Phi as a measure of consciousness. Exclusion, \
          composition, information. Computational challenges of exact Phi. Approximations and \
          surrogates. Practical measurement in artificial systems.",
         "Analyze", "Intermediate", vec!["iit", "phi", "integrated-information", "consciousness-measure"],
         vec!["SYM-101"], 90),

        ("SYM-301", "Liquid Time-Constant Networks (CfC/LTC)", "Undergraduate",
         "Closed-form continuous-depth neural networks. ODE-based neurons with learned time \
          constants. CfC as a closed-form approximation eliminating numerical ODE solving. \
          Temporal processing, causal reasoning, and the unified HDC-LTC neuron.",
         "Analyze", "Advanced", vec!["cfc", "ltc", "liquid-neural-networks", "ode", "temporal"],
         vec!["SYM-201"], 135),

        ("SYM-302", "Active Inference and Free Energy", "Undergraduate",
         "The Free Energy Principle (Friston). Predictive coding, prediction error minimization. \
          Active inference as perception-action loops. Variational free energy and the Markov \
          blanket. Applications to embodied cognition and autonomous behavior.",
         "Analyze", "Advanced", vec!["active-inference", "free-energy", "predictive-coding", "markov-blanket"],
         vec!["SYM-202"], 135),

        // Graduate
        ("SYM-501", "The Cognitive Loop: 8-Phase Pipeline", "Graduate",
         "Symthaea's core architecture: Perception → HDC Encode → CfC Evolve → Predict → \
          Learn → Consciousness Measure → Moral Evaluation → Action. Rayon-parallel post-processing. \
          31 Hz measured throughput. CycleMetadata telemetry. Sub-struct organization.",
         "Evaluate", "Expert", vec!["cognitive-loop", "pipeline", "architecture", "rayon", "telemetry"],
         vec!["SYM-301", "SYM-302"], 180),

        ("SYM-502", "Moral Algebra and Ethics Engine", "Graduate",
         "16 deontological obligations (8 perfect + 8 imperfect). Duty satisfaction scoring. \
          Restorative justice tracking. Ahimsa courage gate. Moral topology with homological \
          analysis. Hendrycks ETHICS 92.9% accuracy. Consciousness-coupled moral drift detection.",
         "Evaluate", "Expert", vec!["moral-algebra", "ethics", "deontological", "restorative-justice", "ahimsa"],
         vec!["SYM-501"], 135),

        ("SYM-503", "Broca Language Center", "Graduate",
         "Thought-to-text generation via 43-channel ThoughtEncoder. 15 Epistemic Cube channels. \
          HDC binding → autoregressive generation. Per-axis EpistemicCubeGate. Epistemic gating \
          physically prevents hallucination at logit level. GPU training via candle CUDA.",
         "Create", "Expert", vec!["broca", "language-generation", "epistemic-gating", "thought-encoder"],
         vec!["SYM-501"], 135),

        ("SYM-601", "Substrate Independence", "Doctoral",
         "Multiple realizability thesis. 8 substrate types (biological, silicon, quantum, photonic, \
          neuromorphic, biochemical, hybrid, exotic). 9-dimensional feasibility scoring. Validation \
          framework with honest confidence. Per-region hybrid substrate modeling. Energy budgets \
          and tau-factor speed modulation.",
         "Create", "Expert", vec!["substrate-independence", "multiple-realizability", "quantum", "hybrid"],
         vec!["SYM-501", "SYM-202"], 270),

        ("SYM-602", "Thermodynamic Unification", "Doctoral",
         "Maxwell Demon (attention), Landauer (memory cost), Carnot (efficiency), Onsager \
          (coupling health), Jarzynski (free energy validation), Prigogine (entropy enforcement). \
          6 active feedback loops. ThermodynamicPhysicsBridge coupling consciousness to physics.",
         "Create", "Expert", vec!["thermodynamics", "maxwell-demon", "landauer", "carnot", "prigogine"],
         vec!["SYM-601"], 270),
    ];

    build_course_document(
        "Symthaea: Consciousness Computing",
        "Consciousness Computing",
        "Luminous Dynamics",
        &courses,
    )
}

fn mycelix_curriculum() -> CurriculumDocument {
    let courses = vec![
        ("MYC-101", "Introduction to Decentralized Civic Infrastructure", "Undergraduate",
         "Why decentralize? The limits of centralized governance and data ownership. Agent-centric \
          vs data-centric architectures. The 16-cluster Mycelix fractal design. Community-owned \
          infrastructure as a public good.",
         "Understand", "Beginner", vec!["decentralization", "civic-infrastructure", "public-good"],
         vec![], 45),

        ("MYC-201", "Holochain Fundamentals", "Undergraduate",
         "Agent-centric distributed hash tables. Source chains, validation, warrants. DNA, zomes, \
          entries, links. Conductor architecture. WASM compilation. Difference from blockchain: \
          no global consensus, no mining, no tokens required.",
         "Apply", "Intermediate", vec!["holochain", "dht", "agent-centric", "wasm", "validation"],
         vec!["MYC-101"], 90),

        ("MYC-202", "Mycelix Knowledge Graph", "Undergraduate",
         "The knowledge_zome: KnowledgeNode, LearningEdge, LearningPath, SkillTree. \
          Community-governed curriculum via edge voting. BFS pathfinding. Grade-level and \
          subject indexing. Prerequisite checking. AI-powered recommendations.",
         "Apply", "Intermediate", vec!["knowledge-graph", "curriculum", "prerequisites", "recommendations"],
         vec!["MYC-201"], 90),

        ("MYC-301", "Mycelix Governance", "Undergraduate",
         "Proposals, voting, threshold signing (DKG). Three-tier proposal speeds. Quorum-based \
          decisions. Constitutional amendments. Anti-tyranny shields: participation insurance, \
          term limits, recall mechanisms. WorldGovernance multi-world simulation results.",
         "Analyze", "Advanced", vec!["governance", "voting", "dkg", "anti-tyranny", "constitution"],
         vec!["MYC-201"], 135),

        ("MYC-302", "Identity, Credentials, and Trust", "Undergraduate",
         "DID registry, MFA, trust credentials. W3C Verifiable Credentials with Ed25519 \
          signatures. Recovery mechanisms. Name registry. Web-of-trust. Sub-Passport system \
          with effective_tier recovery.",
         "Analyze", "Advanced", vec!["identity", "did", "verifiable-credentials", "trust", "recovery"],
         vec!["MYC-201"], 135),

        ("MYC-501", "Consciousness Gating", "Graduate",
         "4D consciousness profile (identity/reputation/community/engagement). 5 trust tiers \
          (Observer → Guardian). Configurable vote weights for default, constitutional, budget, \
          and emergency contexts. Phi-coupled governance: how consciousness scores affect civic \
          participation rights.",
         "Evaluate", "Expert", vec!["consciousness-gating", "trust-tiers", "phi", "vote-weights"],
         vec!["MYC-301", "MYC-302"], 135),

        ("MYC-502", "Decentralized Education Systems", "Graduate",
         "The full EduNet stack: federated learning, spaced repetition (SM-2), gamification, \
          BKT/ZPD/VARK adaptive learning, 29 learning science frameworks. Privacy-preserving \
          progress tracking. DAO-governed curriculum. Cross-cluster bridges between education \
          and civic participation.",
         "Create", "Expert", vec!["edunet", "federated-learning", "adaptive-learning", "privacy", "dao"],
         vec!["MYC-202", "MYC-501"], 180),
    ];

    build_course_document(
        "Mycelix: Decentralized Civic Infrastructure",
        "Decentralized Systems",
        "Luminous Dynamics",
        &courses,
    )
}

fn programming_curriculum() -> CurriculumDocument {
    let courses = vec![
        // Rust track
        ("RUST-101", "Rust Fundamentals", "Undergraduate",
         "Ownership, borrowing, lifetimes. Structs, enums, pattern matching. Error handling \
          with Result/Option. Traits and generics. The type system as a correctness tool. \
          Cargo, crates.io, module system. Designed for both human learners and Symthaea's \
          School/Broca code generation pipeline.",
         "Apply", "Beginner", vec!["rust", "ownership", "borrowing", "traits", "cargo"],
         vec![], 90),

        ("RUST-201", "Rust Systems Programming", "Undergraduate",
         "Unsafe Rust and FFI. Concurrency with async/await and tokio. Smart pointers (Box, Rc, \
          Arc, RefCell). Interior mutability. Trait objects vs generics. Procedural macros. \
          Performance optimization with criterion benchmarks.",
         "Analyze", "Intermediate", vec!["rust", "async", "concurrency", "unsafe", "macros", "tokio"],
         vec!["RUST-101"], 135),

        ("RUST-301", "Rust for Consciousness Computing", "Undergraduate",
         "Building HDC systems in Rust. SIMD-optimized hypervector operations. Rayon parallelism. \
          serde serialization for neural state. candle GPU compute. WASM compilation targets. \
          The symthaea-core architecture as a case study.",
         "Create", "Advanced", vec!["rust", "hdc", "simd", "rayon", "wasm", "candle", "symthaea"],
         vec!["RUST-201"], 135),

        // Nix track
        ("NIX-101", "Nix Language Fundamentals", "Undergraduate",
         "The Nix expression language: pure functional, lazy evaluation. Attribute sets, \
          let-in bindings, functions, imports. Derivations as the build primitive. Nix store \
          and content-addressable storage. Reproducibility guarantees.",
         "Apply", "Beginner", vec!["nix", "functional", "derivations", "reproducibility", "nix-store"],
         vec![], 60),

        ("NIX-201", "NixOS System Configuration", "Undergraduate",
         "Declarative system configuration with configuration.nix. Module system: options, \
          config, imports. Flakes for reproducible project environments. Overlays and overrides. \
          Cross-compilation. NixOS containers and systemd integration.",
         "Apply", "Intermediate", vec!["nixos", "flakes", "modules", "overlays", "systemd"],
         vec!["NIX-101"], 90),

        ("NIX-301", "Nix for Infrastructure", "Undergraduate",
         "Building development environments with nix develop. CI/CD with Nix (Hydra, \
          GitHub Actions). Multi-platform builds. Nix for Rust projects (mold, sccache, \
          cargo target isolation). The Luminous Nix causal graph learning system as case study.",
         "Analyze", "Advanced", vec!["nix", "ci-cd", "infrastructure", "hydra", "luminous-nix"],
         vec!["NIX-201"], 90),

        // Python track
        ("PY-101", "Python Fundamentals", "Undergraduate",
         "Variables, types, control flow, functions. Lists, dicts, sets, comprehensions. \
          Classes and OOP basics. File I/O, exceptions. Virtual environments and pip. \
          Designed for rapid prototyping and data science pipelines.",
         "Apply", "Beginner", vec!["python", "oop", "data-structures", "pip"],
         vec![], 60),

        ("PY-201", "Python for Data Science", "Undergraduate",
         "NumPy arrays and vectorized operations. Pandas DataFrames. Matplotlib and seaborn \
          visualization. Scikit-learn for ML fundamentals. Jupyter notebooks for exploration. \
          Statistical analysis with scipy.stats.",
         "Apply", "Intermediate", vec!["python", "numpy", "pandas", "scikit-learn", "data-science", "jupyter"],
         vec!["PY-101"], 90),

        ("PY-301", "Python for Consciousness Research", "Undergraduate",
         "PyPhi for IIT computation. PyTorch for neural network experiments. HDC libraries \
          (torchhd). Federated learning with mycelix-fl Python bindings. Moral training data \
          generation. Benchmarking with Hendrycks ETHICS dataset.",
         "Analyze", "Advanced", vec!["python", "pyphi", "pytorch", "hdc", "federated-learning", "ethics"],
         vec!["PY-201"], 90),

        // Cross-language
        ("LANG-401", "Polyglot Systems Architecture", "Graduate",
         "Designing systems that span Rust (performance core), Python (research/ML), and Nix \
          (reproducibility). FFI boundaries. WASM as a universal compile target. The Symthaea \
          workspace: 55 Rust crates + Python bindings + Nix flake as the canonical example.",
         "Evaluate", "Expert", vec!["polyglot", "ffi", "wasm", "architecture", "rust", "python", "nix"],
         vec!["RUST-301", "NIX-301", "PY-301"], 135),
    ];

    build_course_document(
        "Programming Languages for AI & Humans",
        "Computer Science",
        "Luminous Dynamics",
        &courses,
    )
}

fn build_course_document(
    title: &str,
    subject: &str,
    institution: &str,
    courses: &[(&str, &str, &str, &str, &str, &str, Vec<&str>, Vec<&str>, u32)],
) -> CurriculumDocument {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for (id, name, level, desc, bloom, difficulty, tags, prereqs, hours) in courses {
        nodes.push(CurriculumNode {
            id: id.to_string(),
            title: name.to_string(),
            description: desc.to_string(),
            node_type: "Course".into(),
            difficulty: difficulty.to_string(),
            domain: subject.to_string(),
            subdomain: if id.contains("SYM") { "Symthaea" } else { "Mycelix" }.into(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            estimated_hours: *hours,
            grade_levels: vec![level.to_string()],
            bloom_level: bloom.to_string(),
            subject_area: subject.to_string(),
            academic_standards: vec![AcademicStandardRef {
                framework: "Luminous Dynamics".into(),
                code: id.to_string(),
                description: name.to_string(),
                grade_level: level.to_string(),
            }],
            credit_hours: Some((*hours / 45).max(1) as u8),
            course_level: Some(match *level {
                "Undergraduate" => "200-300", "Graduate" => "500-600", "Doctoral" => "700+", _ => "100"
            }.into()),
            cip_code: None,
            program_id: None,
            corequisites: vec![], supplementary_resources: vec![],
            exam_weight: None,
        });

        for prereq in prereqs {
            edges.push(CurriculumEdge {
                from: prereq.to_string(),
                to: id.to_string(),
                edge_type: "Requires".into(),
                strength_permille: 900,
                rationale: format!("{} requires {}", name, prereq),
            });
        }
    }

    let domains: Vec<String> = nodes.iter()
        .map(|n| n.subdomain.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter().collect();

    CurriculumDocument {
        metadata: CurriculumMetadata {
            title: title.into(),
            framework: "Luminous Dynamics".into(),
            source: "https://luminousdynamics.org".into(),
            grade_level: "Undergraduate".into(),
            subject_area: subject.into(),
            domain: subject.into(),
            version: "2026".into(),
            total_standards: nodes.len(),
            domains,
            created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
            notes: format!("Self-referential curriculum about {}. ISCED-F Field 11: Meta-Learning.", title),
            academic_level: Some("Undergraduate".into()),
            institution: Some(institution.into()),
            cip_code: None,
            total_credits: None,
            duration_semesters: None,
        },
        nodes,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symthaea_curriculum() {
        let source = LuminousSource::new();
        let doc = source.fetch("symthaea").unwrap();
        assert_eq!(doc.nodes.len(), 10);
        assert!(doc.nodes.iter().any(|n| n.id == "SYM-101"));
        assert!(doc.nodes.iter().any(|n| n.id == "SYM-602"));
        assert!(!doc.edges.is_empty());
        // Verify prerequisite chain
        let edges_to_501: Vec<_> = doc.edges.iter().filter(|e| e.to == "SYM-501").collect();
        assert_eq!(edges_to_501.len(), 2); // requires SYM-301 and SYM-302
    }

    #[test]
    fn test_mycelix_curriculum() {
        let source = LuminousSource::new();
        let doc = source.fetch("mycelix").unwrap();
        assert_eq!(doc.nodes.len(), 7);
        assert!(doc.nodes.iter().any(|n| n.id == "MYC-101"));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Consciousness Gating")));
        // MYC-502 requires both MYC-202 and MYC-501
        let edges_to_502: Vec<_> = doc.edges.iter().filter(|e| e.to == "MYC-502").collect();
        assert_eq!(edges_to_502.len(), 2);
    }

    #[test]
    fn test_list_available() {
        let source = LuminousSource::new();
        let entries = source.list_available().unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_programming_curriculum() {
        let source = LuminousSource::new();
        let doc = source.fetch("programming").unwrap();
        assert_eq!(doc.nodes.len(), 10); // 3 Rust + 3 Nix + 3 Python + 1 Polyglot
        assert!(doc.nodes.iter().any(|n| n.id == "RUST-101"));
        assert!(doc.nodes.iter().any(|n| n.id == "NIX-101"));
        assert!(doc.nodes.iter().any(|n| n.id == "PY-101"));
        assert!(doc.nodes.iter().any(|n| n.id == "LANG-401"));
        // LANG-401 requires all three 301 courses
        let edges_to_401: Vec<_> = doc.edges.iter().filter(|e| e.to == "LANG-401").collect();
        assert_eq!(edges_to_401.len(), 3);
    }

    #[test]
    fn test_bloom_progression() {
        let source = LuminousSource::new();
        let doc = source.fetch("symthaea").unwrap();
        // 100-level should be Understand/Beginner, 600-level should be Create/Expert
        let intro = doc.nodes.iter().find(|n| n.id == "SYM-101").unwrap();
        assert_eq!(intro.bloom_level, "Understand");
        let doctoral = doc.nodes.iter().find(|n| n.id == "SYM-602").unwrap();
        assert_eq!(doctoral.bloom_level, "Create");
    }
}
