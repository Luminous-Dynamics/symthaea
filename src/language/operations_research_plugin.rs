// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Operations-research domain plugin — economic order quantity (Wilson EOQ).

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::labeled;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_operations_research::inventory::economic_order_quantity;

pub struct OperationsResearchDomainPlugin;

const CUES: &[&str] = &[
    "economic order quantity",
    "eoq",
    "order quantity",
    "reorder",
];

impl OperationsResearchDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for OperationsResearchDomainPlugin {
    fn domain_name(&self) -> &str {
        "operations_research"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        ["eoq", "inventory", "demand", "order", "holding", "quantity"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        // Drop the connective word "cost" so each role word ("order"/"holding")
        // directly precedes its number, which is what `labeled` keys on.
        let flat = input.to_lowercase().replace("cost", " ");
        let demand = labeled(&flat, &["demand"])?;
        let order_cost = labeled(&flat, &["order", "setup", "ordering"])?;
        let holding = labeled(&flat, &["holding", "carrying"])?;
        if demand <= 0.0 || order_cost <= 0.0 || holding <= 0.0 {
            return None;
        }
        let q = economic_order_quantity(demand, order_cost, holding);
        Some(ComputedResult {
            answer: format!(
                "Economic order quantity (Wilson EOQ) for annual demand {demand}, order cost \
                 {order_cost}, holding cost {holding}/unit: {q:.1} units per order."
            ),
            cube: EpistemicCube {
                e: ETier::E4,
                n: NTier::N3,
                m: MTier::M3,
                h: None,
            },
            psi: 0.0,
            proof_available: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_eoq() {
        let p = OperationsResearchDomainPlugin;
        // D=1200, S=50, H=3 → EOQ = sqrt(2*1200*50/3) = 200.
        let r = p
            .compute(
                "economic order quantity for annual demand 1200 units, order cost 50, holding cost 3",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("200.0"), "{}", r.answer);
    }

    #[test]
    fn needs_all_three() {
        let p = OperationsResearchDomainPlugin;
        assert!(
            p.compute("eoq with annual demand 1200 and order cost 50", &[])
                .is_none()
        );
    }
}
