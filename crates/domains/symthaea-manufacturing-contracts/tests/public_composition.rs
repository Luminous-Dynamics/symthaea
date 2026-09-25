use symthaea_manufacturing_contracts::{
    CapabilityEvidenceClassV1, CapabilityMatchV1, CapabilityValidityV1, PlanEdgeKindV1,
    PlanNodeKindV1, ProcessCapabilityProfileV1, ProcessCapabilityRequirementV1, ProcessPlanEdgeV1,
    ProcessPlanNodeV1, ProcessPlanV1, ProcessRecipeV1, ProcessStateClassV1, ProcessStateRefV1,
    ProcessTransformationContractV1, RecipeDisclosureV1,
};
use symthaea_manufacturing_process::{
    ManufacturingOperationModeV1, ProcessDefinitionV1, ProcessFamilyRef, TransformationEffectV1,
};

fn process_definition() -> ProcessDefinitionV1 {
    ProcessDefinitionV1 {
        namespace: "luminous".into(),
        process_name: "benign-precision-coupon-milling".into(),
        semantic_version: "1.0.0".into(),
        family_refs: vec![ProcessFamilyRef("luminous:cnc-milling/v1".into())],
        operation_modes: vec![ManufacturingOperationModeV1::Discrete],
        transformation_effects: vec![TransformationEffectV1::MaterialRemoval],
        extension_profile_refs: vec![],
        display_name: Some("Synthetic precision coupon milling".into()),
        description: Some("Qualification fixture only; no machine execution authority".into()),
    }
}

fn state(class: ProcessStateClassV1, subject: &str) -> ProcessStateRefV1 {
    ProcessStateRefV1 {
        namespace: "synthetic.fixture".into(),
        subject_id: subject.into(),
        semantic_version: "1".into(),
        state_class: class,
    }
}

#[test]
fn exported_contracts_compose_without_authority_laundering() {
    let process = process_definition();
    let process_id = process.process_id().unwrap();

    let stock = state(ProcessStateClassV1::MaterialState, "coupon-stock-a");
    let machined = state(ProcessStateClassV1::GeometryState, "coupon-machined-a");
    let inspected = state(ProcessStateClassV1::InspectionState, "coupon-inspected-a");

    let transformation = ProcessTransformationContractV1 {
        process_id: process_id.clone(),
        inputs: vec![stock.clone()],
        outputs: vec![machined.clone()],
        preserved_refs: vec![],
    };
    let transformation_id = transformation.contract_id().unwrap();
    assert_eq!(transformation_id.len(), 64);

    let requirement = ProcessCapabilityRequirementV1 {
        process_id: process_id.clone(),
        envelope_profile_ref: "synthetic:envelope:coupon-mill-v1".into(),
        minimum_evidence: CapabilityEvidenceClassV1::QualifiedUnderProfile,
    };
    let requirement_id = requirement.requirement_id().unwrap();

    let capability = ProcessCapabilityProfileV1 {
        resource_subject_ref: "eng-catalog:synthetic-resource:mill-a".into(),
        process_id: process_id.clone(),
        envelope_profile_ref: requirement.envelope_profile_ref.clone(),
        configuration_profile_ref: "synthetic:configuration:fixture-a".into(),
        evidence_class: CapabilityEvidenceClassV1::QualifiedUnderProfile,
        evidence_ref: "synthetic:evidence:qualification-a".into(),
        validity: CapabilityValidityV1 {
            revision: "r1".into(),
            valid_from_unix_s: 100,
            valid_until_unix_s: 200,
        },
        display_label: Some("Synthetic Mill A".into()),
    };
    assert_eq!(
        requirement.evaluate(&capability, 150, true),
        CapabilityMatchV1::ExactAdmitted
    );

    let recipe = ProcessRecipeV1 {
        process_id: process_id.clone(),
        semantic_version: "1.0.0".into(),
        input_state_refs: vec![stock.coordinate().unwrap()],
        output_state_refs: vec![machined.coordinate().unwrap()],
        parameter_bundle_refs: vec!["synthetic:parameter-bundle:coupon-v1".into()],
        configuration_refs: vec!["synthetic:configuration:fixture-a".into()],
        environment_refs: vec!["synthetic:environment:lab-default".into()],
        inspection_refs: vec!["synthetic:inspection:coupon-v1".into()],
        model_profile_refs: vec!["synthetic:model:reduced-order-v1".into()],
        qualification_profile_refs: vec!["synthetic:qualification:coupon-v1".into()],
        display_label: Some("Synthetic coupon recipe".into()),
        notes: Some("No execution authority".into()),
    };
    let recipe_id = recipe.recipe_id().unwrap();
    let commitment = recipe
        .commitment(
            "synthetic:recipe-schema:v1".into(),
            "synthetic:owner:test-fixture".into(),
            RecipeDisclosureV1::CommitmentOnly,
            None,
        )
        .unwrap();
    assert!(commitment.matches_recipe(&recipe).unwrap());

    let process_node = ProcessPlanNodeV1 {
        node_id: "machine-coupon".into(),
        kind: PlanNodeKindV1::ProcessStep {
            process_id: process_id.clone(),
            recipe_or_commitment_ref: Some(format!("recipe-commitment:{}", commitment.recipe_id)),
            capability_requirement_ref: format!("capability-requirement:{requirement_id}"),
            input_state_refs: vec![stock.coordinate().unwrap()],
            output_state_refs: vec![machined.coordinate().unwrap()],
        },
        display_label: Some("Machine coupon".into()),
        ui_x: Some(10),
        ui_y: Some(10),
    };
    let inspect_node = ProcessPlanNodeV1 {
        node_id: "inspect-coupon".into(),
        kind: PlanNodeKindV1::Inspection {
            inspection_profile_ref: "synthetic:inspection:coupon-v1".into(),
            state_ref: inspected.coordinate().unwrap(),
        },
        display_label: Some("Inspect coupon".into()),
        ui_x: Some(20),
        ui_y: Some(10),
    };
    let plan = ProcessPlanV1 {
        semantic_version: "1".into(),
        nodes: vec![process_node, inspect_node],
        edges: vec![ProcessPlanEdgeV1 {
            from: "machine-coupon".into(),
            to: "inspect-coupon".into(),
            kind: PlanEdgeKindV1::Normal,
            condition_ref: None,
            max_iterations: None,
            disposition_ref: None,
        }],
        display_label: Some("Synthetic coupon qualification plan".into()),
    };
    let plan_id = plan.plan_id().unwrap();

    assert_eq!(transformation.process_id, process_id);
    assert_eq!(recipe.process_id, process_id);
    assert_eq!(capability.process_id, process_id);
    assert_eq!(recipe_id.len(), 64);
    assert_eq!(plan_id.len(), 64);

    // These IDs prove semantic composition only. No type in this facade represents
    // a scheduler placement, machine command, execution receipt, or output acceptance.
    assert_ne!(transformation_id, recipe_id);
    assert_ne!(recipe_id, plan_id);
}

#[test]
fn weaker_capability_does_not_become_admitted_because_the_rest_of_the_chain_is_valid() {
    let process_id = process_definition().process_id().unwrap();
    let requirement = ProcessCapabilityRequirementV1 {
        process_id: process_id.clone(),
        envelope_profile_ref: "synthetic:envelope:coupon-mill-v1".into(),
        minimum_evidence: CapabilityEvidenceClassV1::QualifiedUnderProfile,
    };
    let capability = ProcessCapabilityProfileV1 {
        resource_subject_ref: "eng-catalog:synthetic-resource:mill-a".into(),
        process_id,
        envelope_profile_ref: requirement.envelope_profile_ref.clone(),
        configuration_profile_ref: "synthetic:configuration:fixture-a".into(),
        evidence_class: CapabilityEvidenceClassV1::Declared,
        evidence_ref: "synthetic:evidence:self-declaration".into(),
        validity: CapabilityValidityV1 {
            revision: "r1".into(),
            valid_from_unix_s: 100,
            valid_until_unix_s: 200,
        },
        display_label: None,
    };
    assert_eq!(
        requirement.evaluate(&capability, 150, true),
        CapabilityMatchV1::CompatibleButWeakerEvidence
    );
}
