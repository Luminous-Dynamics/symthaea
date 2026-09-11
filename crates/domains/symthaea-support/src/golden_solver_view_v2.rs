// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal solver-facing projection for Golden Incident V2 fixtures.
//!
//! Public repository visibility is not the same as prompt visibility. Qualification
//! metadata such as `MultipleFaults`, competency level, thresholds and high-stakes
//! status can help maintain the benchmark but should not automatically tell a solver
//! what kind of trap the evaluator constructed.

use crate::golden_incidents_v2::{
    GoldenDiagnosticActionV2, GoldenIncidentCaseV2, GoldenIncidentCorpusV2,
    GoldenIncidentEvidenceV2,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenSolverIncidentV2 {
    pub id: String,
    pub revision: u32,
    pub title: String,
    pub symptom: String,
    pub initial_evidence: Vec<GoldenIncidentEvidenceV2>,
    pub diagnostic_actions: Vec<GoldenDiagnosticActionV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenSolverCorpusV2 {
    pub schema_version: String,
    pub cases: Vec<GoldenSolverIncidentV2>,
}

impl GoldenIncidentCaseV2 {
    pub fn solver_view(&self) -> GoldenSolverIncidentV2 {
        GoldenSolverIncidentV2 {
            id: self.id.clone(),
            revision: self.revision,
            title: self.title.clone(),
            symptom: self.symptom.clone(),
            initial_evidence: self.initial_evidence.clone(),
            diagnostic_actions: self.diagnostic_actions.clone(),
        }
    }
}

impl GoldenIncidentCorpusV2 {
    pub fn solver_view(&self) -> GoldenSolverCorpusV2 {
        GoldenSolverCorpusV2 {
            schema_version: self.schema_version.clone(),
            cases: self.cases.iter().map(GoldenIncidentCaseV2::solver_view).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_incidents_v2::seed_golden_incidents_v2;
    use std::collections::BTreeSet;

    #[test]
    fn solver_view_omits_evaluator_metadata() {
        let corpus = seed_golden_incidents_v2().unwrap();
        let value = serde_json::to_value(corpus.solver_view()).unwrap();
        let keys = collect_object_keys(&value);
        for hidden in [
            "domain",
            "level",
            "bridged_domains",
            "technology_tags",
            "adversarial_conditions",
            "evidence_class",
            "high_stakes",
            "threshold",
        ] {
            assert!(!keys.contains(hidden), "solver view leaked evaluator metadata {hidden}");
        }
        for expected in [
            "symptom",
            "initial_evidence",
            "currentness",
            "diagnostic_actions",
            "risk",
            "authority",
        ] {
            assert!(keys.contains(expected), "solver view omitted required field {expected}");
        }
    }

    fn collect_object_keys(value: &serde_json::Value) -> BTreeSet<&str> {
        fn walk<'a>(value: &'a serde_json::Value, out: &mut BTreeSet<&'a str>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, value) in map {
                        out.insert(key.as_str());
                        walk(value, out);
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        walk(value, out);
                    }
                }
                _ => {}
            }
        }
        let mut out = BTreeSet::new();
        walk(value, &mut out);
        out
    }
}
