// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock data for Knowledge cluster — SA-contextualized claims and fact-checks.

use knowledge_leptos_types::*;

pub fn claims() -> Vec<ClaimView> {
    vec![
        ClaimView {
            id: "CLM-001".into(), content: "South Africa's renewable energy capacity exceeded 10 GW in 2024".into(),
            classification: EpistemicClassificationView { empirical: EmpiricalLevel::E3, normative: NormativeLevel::N2, materiality: MaterialityLevel::M2, confidence: 0.85 },
            author: "did:mycelix:researcher-001".into(), sources: vec!["DMRE Annual Report 2024".into()],
            tags: vec!["energy".into(), "south-africa".into(), "renewable".into()],
            claim_type: ClaimType::Fact, created: 1704067200, version: 1,
        },
        ClaimView {
            id: "CLM-002".into(), content: "Community-owned microgrids reduce energy poverty more effectively than national grid expansion".into(),
            classification: EpistemicClassificationView { empirical: EmpiricalLevel::E2, normative: NormativeLevel::N2, materiality: MaterialityLevel::M3, confidence: 0.72 },
            author: "did:mycelix:activist-001".into(), sources: vec!["IRENA Report 2023".into(), "Local case studies".into()],
            tags: vec!["energy".into(), "community".into(), "poverty".into()],
            claim_type: ClaimType::Hypothesis, created: 1711929600, version: 2,
        },
        ClaimView {
            id: "CLM-003".into(), content: "Decentralized identity systems will replace national ID by 2035".into(),
            classification: EpistemicClassificationView { empirical: EmpiricalLevel::E0, normative: NormativeLevel::N1, materiality: MaterialityLevel::M1, confidence: 0.35 },
            author: "did:mycelix:futurist-001".into(), sources: vec![],
            tags: vec!["identity".into(), "prediction".into(), "governance".into()],
            claim_type: ClaimType::Prediction, created: 1719792000, version: 1,
        },
        ClaimView {
            id: "CLM-004".into(), content: "Ubuntu philosophy aligns with distributed consensus mechanisms".into(),
            classification: EpistemicClassificationView { empirical: EmpiricalLevel::E1, normative: NormativeLevel::N3, materiality: MaterialityLevel::M2, confidence: 0.68 },
            author: "did:mycelix:philosopher-001".into(), sources: vec!["Ramose 1999".into(), "Metz 2007".into()],
            tags: vec!["ubuntu".into(), "governance".into(), "philosophy".into()],
            claim_type: ClaimType::Normative, created: 1696118400, version: 1,
        },
    ]
}

pub fn fact_checks() -> Vec<FactCheckResultView> {
    vec![
        FactCheckResultView { id: "FC-001".into(), statement: "SA renewable capacity exceeded 10 GW".into(), verdict: FactCheckVerdict::True, verdict_confidence: 0.92, credibility_score: 0.88, evidence_count: 5 },
        FactCheckResultView { id: "FC-002".into(), statement: "Microgrids reduce poverty better than grid".into(), verdict: FactCheckVerdict::MostlyTrue, verdict_confidence: 0.74, credibility_score: 0.71, evidence_count: 3 },
    ]
}

pub fn inferences() -> Vec<InferenceView> {
    vec![
        InferenceView { id: "INF-001".into(), inference_type: InferenceType::Pattern, source_claims: vec!["CLM-001".into(), "CLM-002".into()], conclusion: "Energy decentralization trend accelerating in SA".into(), confidence: 0.78, reasoning: "Multiple independent sources confirm shift toward distributed energy".into(), verified: true },
        InferenceView { id: "INF-002".into(), inference_type: InferenceType::Similarity, source_claims: vec!["CLM-004".into()], conclusion: "Ubuntu-consensus alignment may inform governance design".into(), confidence: 0.55, reasoning: "Philosophical alignment detected between ubuntu values and consensus protocols".into(), verified: false },
    ]
}

pub fn graph_stats() -> GraphStatsView {
    GraphStatsView { relationship_count: 47 }
}
