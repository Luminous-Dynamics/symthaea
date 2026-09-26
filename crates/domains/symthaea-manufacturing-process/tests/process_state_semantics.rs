use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_manufacturing_process::{ManufacturingProcessError, ProcessDefinitionId};

/// References one externally-owned engineering state subject without copying its physics or units.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct ProcessStateRefV1 {
    namespace: String,
    subject_id: String,
    semantic_version: String,
    state_class: ProcessStateClassV1,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum ProcessStateClassV1 {
    MaterialState,
    GeometryState,
    SurfaceState,
    AssemblyState,
    CleanlinessState,
    ThermalHistoryState,
    InspectionState,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ProcessTransformationContractV1 {
    process_id: ProcessDefinitionId,
    inputs: Vec<ProcessStateRefV1>,
    outputs: Vec<ProcessStateRefV1>,
    preserved_refs: Vec<ProcessStateRefV1>,
}

fn canonical_token(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control)
}

fn validate_state_ref(state: &ProcessStateRefV1) -> Result<(), &'static str> {
    if !canonical_token(&state.namespace)
        || !canonical_token(&state.subject_id)
        || !canonical_token(&state.semantic_version)
    {
        return Err("state reference must use canonical non-empty tokens");
    }
    if let ProcessStateClassV1::Custom(tag) = &state.state_class {
        if !canonical_token(tag) {
            return Err("custom state class must be canonical");
        }
    }
    Ok(())
}

fn validate_contract(contract: &ProcessTransformationContractV1) -> Result<(), &'static str> {
    if contract.inputs.is_empty() || contract.outputs.is_empty() {
        return Err("process transformation requires at least one input and one output state");
    }
    let mut seen_inputs = BTreeSet::new();
    let mut seen_outputs = BTreeSet::new();
    let mut seen_preserved = BTreeSet::new();
    for state in &contract.inputs {
        validate_state_ref(state)?;
        if !seen_inputs.insert(state.clone()) {
            return Err("duplicate input state reference");
        }
    }
    for state in &contract.outputs {
        validate_state_ref(state)?;
        if !seen_outputs.insert(state.clone()) {
            return Err("duplicate output state reference");
        }
    }
    for state in &contract.preserved_refs {
        validate_state_ref(state)?;
        if !seen_preserved.insert(state.clone()) {
            return Err("duplicate preserved state reference");
        }
    }
    Ok(())
}

fn state(class: ProcessStateClassV1, id: &str) -> ProcessStateRefV1 {
    ProcessStateRefV1 {
        namespace: "symthaea.material-state".into(),
        subject_id: id.into(),
        semantic_version: "1".into(),
        state_class: class,
    }
}

#[test]
fn process_state_refs_do_not_copy_quantities_or_material_properties() {
    let input = state(ProcessStateClassV1::MaterialState, "annealed-6061-stock");
    let output = state(ProcessStateClassV1::MaterialState, "machined-6061-article");
    let contract = ProcessTransformationContractV1 {
        process_id: ProcessDefinitionId("process-id".into()),
        inputs: vec![input],
        outputs: vec![output],
        preserved_refs: vec![],
    };
    assert!(validate_contract(&contract).is_ok());
    let json = serde_json::to_string(&contract).unwrap();
    assert!(!json.contains("kelvin"));
    assert!(!json.contains("millimeter"));
    assert!(!json.contains("yield_strength"));
}

#[test]
fn duplicate_state_refs_fail_closed() {
    let input = state(ProcessStateClassV1::MaterialState, "stock");
    let output = state(ProcessStateClassV1::GeometryState, "part");
    let contract = ProcessTransformationContractV1 {
        process_id: ProcessDefinitionId("process-id".into()),
        inputs: vec![input.clone(), input],
        outputs: vec![output],
        preserved_refs: vec![],
    };
    assert_eq!(validate_contract(&contract), Err("duplicate input state reference"));
}

#[test]
fn missing_input_or_output_fails_closed() {
    let output = state(ProcessStateClassV1::GeometryState, "part");
    let contract = ProcessTransformationContractV1 {
        process_id: ProcessDefinitionId("process-id".into()),
        inputs: vec![],
        outputs: vec![output],
        preserved_refs: vec![],
    };
    assert_eq!(
        validate_contract(&contract),
        Err("process transformation requires at least one input and one output state")
    );
}

#[test]
fn serialization_preserves_state_subject_identity() {
    let state = state(ProcessStateClassV1::SurfaceState, "polished-surface-v3");
    let encoded = serde_json::to_string(&state).unwrap();
    let decoded: ProcessStateRefV1 = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, state);
}

#[test]
fn compilation_anchor_uses_public_process_types() {
    let _ = std::mem::size_of::<ManufacturingProcessError>();
}
