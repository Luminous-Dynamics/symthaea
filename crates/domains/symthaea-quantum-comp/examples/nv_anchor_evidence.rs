//! Frozen room-temperature NV-diamond evidence anchors for QCOMP-RT-001B (#5809).
//!
//! This example deliberately keeps the first physical-register evidence surface
//! outside the crate's stable public API. It is a scoped benchmark input registry,
//! not a canonical quantity/provenance ontology and not a hardware backend.
//!
//! Claim boundary:
//! - one row == one source-bound literature observation/input;
//! - reported value != independently reproduced result;
//! - room-temperature qubit host != fully room-temperature quantum computer;
//! - model agreement != hardware access, scalability, or quantum advantage.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractionState {
    DirectlyReported,
    DerivedFromReportedValues,
    DigitizedOrReconstructed,
    BoundedNuisance,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceAnchor {
    id: &'static str,
    title: &'static str,
    doi: &'static str,
    publication_date: &'static str,
    system_profile: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParameterEvidence {
    id: &'static str,
    source_id: &'static str,
    value_text: Option<&'static str>,
    unit_text: Option<&'static str>,
    uncertainty_text: Option<&'static str>,
    extraction_state: ExtractionState,
    caveat: &'static str,
}

const SOURCE_JAEGER_NV_QV: &str = "jaeger-nv-qv-2026";
const SOURCE_MINNELLA_GHZ: &str = "minnella-nv-ghz-2026";

fn sources() -> [SourceAnchor; 2] {
    [
        SourceAnchor {
            id: SOURCE_JAEGER_NV_QV,
            title: "Modeling quantum volume using randomized benchmarking of Room-Temperature NV center quantum registers",
            doi: "10.1038/s41534-025-01164-0",
            publication_date: "2025-12-29",
            system_profile: "room-temperature NV electron + nuclear-spin register",
        },
        SourceAnchor {
            id: SOURCE_MINNELLA_GHZ,
            title: "Single-gate, multipartite entanglement on a room-temperature quantum register",
            doi: "10.1038/s41565-026-02254-6",
            publication_date: "2026-09-14",
            system_profile: "room-temperature NV electron + nearby nuclear-spin register",
        },
    ]
}

fn extraction_states() -> [ExtractionState; 5] {
    [
        ExtractionState::DirectlyReported,
        ExtractionState::DerivedFromReportedValues,
        ExtractionState::DigitizedOrReconstructed,
        ExtractionState::BoundedNuisance,
        ExtractionState::Unknown,
    ]
}

fn parameters() -> [ParameterEvidence; 9] {
    [
        ParameterEvidence {
            id: "jaeger-modeled-quantum-volume",
            source_id: SOURCE_JAEGER_NV_QV,
            value_text: Some("8"),
            unit_text: None,
            uncertainty_text: None,
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Modeled from an experimentally calibrated error model; not a direct execution of 10,000 quantum-volume circuits on physical hardware.",
        },
        ParameterEvidence {
            id: "jaeger-register-nuclear-qubits",
            source_id: SOURCE_JAEGER_NV_QV,
            value_text: Some("3"),
            unit_text: Some("nuclear_qubits"),
            uncertainty_text: None,
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Register-size statement is profile-specific and does not imply scalable register construction.",
        },
        ParameterEvidence {
            id: "jaeger-qv-simulated-circuit-instances",
            source_id: SOURCE_JAEGER_NV_QV,
            value_text: Some("10000"),
            unit_text: Some("circuit_instances"),
            uncertainty_text: None,
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Simulation repetition count used for statistical confidence in the reported quantum-volume estimate.",
        },
        ParameterEvidence {
            id: "jaeger-exact-host-temperature",
            source_id: SOURCE_JAEGER_NV_QV,
            value_text: None,
            unit_text: Some("kelvin"),
            uncertainty_text: None,
            extraction_state: ExtractionState::Unknown,
            caveat: "The frozen article summary establishes room-temperature operation but this registry does not invent an exact kelvin value.",
        },
        ParameterEvidence {
            id: "minnella-ghz-register-qubits",
            source_id: SOURCE_MINNELLA_GHZ,
            value_text: Some("4"),
            unit_text: Some("qubits"),
            uncertainty_text: None,
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Four-qubit GHZ observation does not establish logical qubits or scalable architecture.",
        },
        ParameterEvidence {
            id: "minnella-parallel-gate-duration",
            source_id: SOURCE_MINNELLA_GHZ,
            value_text: Some("14.8"),
            unit_text: Some("microseconds"),
            uncertainty_text: None,
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Duration of the reported parallel four-qubit entangling gate under the paper's experimental profile.",
        },
        ParameterEvidence {
            id: "minnella-parallel-four-qubit-fidelity",
            source_id: SOURCE_MINNELLA_GHZ,
            value_text: Some("0.92"),
            unit_text: None,
            uncertainty_text: Some("0.04"),
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Reported four-qubit parallel-gate fidelity; uncertainty preserved textually from 0.92(4).",
        },
        ParameterEvidence {
            id: "minnella-sequential-four-qubit-fidelity",
            source_id: SOURCE_MINNELLA_GHZ,
            value_text: Some("0.69"),
            unit_text: None,
            uncertainty_text: Some("0.03"),
            extraction_state: ExtractionState::DirectlyReported,
            caveat: "Reported sequential four-qubit comparison fidelity; not a universal sequential-gate baseline.",
        },
        ParameterEvidence {
            id: "minnella-exact-host-temperature",
            source_id: SOURCE_MINNELLA_GHZ,
            value_text: None,
            unit_text: Some("kelvin"),
            uncertainty_text: None,
            extraction_state: ExtractionState::Unknown,
            caveat: "Ambient/room-temperature operation is reported, but this registry intentionally leaves exact kelvin unspecified until extracted from exact methods/supplement text.",
        },
    ]
}

fn source_exists(source_id: &str) -> bool {
    sources().iter().any(|source| source.id == source_id)
}

fn validate_registry() -> Result<(), String> {
    let anchors = sources();
    if anchors[0].id == anchors[1].id {
        return Err("duplicate source ids".to_string());
    }

    for parameter in parameters() {
        if !source_exists(parameter.source_id) {
            return Err(format!("unresolved source for {}", parameter.id));
        }
        match parameter.extraction_state {
            ExtractionState::Unknown if parameter.value_text.is_some() => {
                return Err(format!("unknown value was silently filled for {}", parameter.id));
            }
            ExtractionState::Unknown => {}
            _ if parameter.value_text.is_none() => {
                return Err(format!("missing evidence value for {}", parameter.id));
            }
            _ => {}
        }
        if parameter.id.contains("fully-room-temperature")
            || parameter.id.contains("fully_room_temperature")
        {
            return Err(format!(
                "ambiguous whole-system room-temperature claim encoded by {}",
                parameter.id
            ));
        }
    }

    Ok(())
}

fn print_registry() {
    println!("# qcomp-rt-nv-anchor-registry-v1");
    println!("# source_id\tpublication_date\tdoi\ttitle\tsystem_profile");
    for source in sources() {
        println!(
            "# {}\t{}\t{}\t{}\t{}",
            source.id,
            source.publication_date,
            source.doi,
            source.title.replace('\t', " ").replace('\n', " "),
            source.system_profile.replace('\t', " ").replace('\n', " "),
        );
    }

    for state in extraction_states() {
        println!("# extraction_state={state:?}");
    }

    println!("source_id\tparameter_id\tvalue\tunit\tuncertainty\textraction_state\tcaveat");
    for parameter in parameters() {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:?}\t{}",
            parameter.source_id,
            parameter.id,
            parameter.value_text.unwrap_or("<missing>"),
            parameter.unit_text.unwrap_or("<dimensionless>"),
            parameter.uncertainty_text.unwrap_or("<missing>"),
            parameter.extraction_state,
            parameter.caveat.replace('\t', " ").replace('\n', " "),
        );
    }
}

fn main() -> Result<(), String> {
    validate_registry()?;
    print_registry();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_internally_valid() {
        assert!(validate_registry().is_ok());
    }

    #[test]
    fn unknown_values_remain_missing() {
        for parameter in parameters() {
            if parameter.extraction_state == ExtractionState::Unknown {
                assert!(parameter.value_text.is_none());
            }
        }
    }

    #[test]
    fn frozen_direct_values_match_anchor_contract() {
        let values: Vec<_> = parameters()
            .iter()
            .filter_map(|parameter| parameter.value_text.map(|value| (parameter.id, value)))
            .collect();
        assert!(values.contains(&("jaeger-modeled-quantum-volume", "8")));
        assert!(values.contains(&("minnella-parallel-gate-duration", "14.8")));
        assert!(values.contains(&("minnella-parallel-four-qubit-fidelity", "0.92")));
        assert!(values.contains(&("minnella-sequential-four-qubit-fidelity", "0.69")));
    }
}
