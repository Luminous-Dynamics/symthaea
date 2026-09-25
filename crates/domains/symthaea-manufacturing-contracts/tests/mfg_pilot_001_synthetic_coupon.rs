use serde::{Deserialize, Serialize};
use symthaea_manufacturing_contracts::{
    CapabilityEvidenceClassV1, CapabilityMatchV1, CapabilityValidityV1, PlanEdgeKindV1,
    PlanNodeKindV1, ProcessCapabilityProfileV1, ProcessCapabilityRequirementV1, ProcessPlanEdgeV1,
    ProcessPlanNodeV1, ProcessPlanV1, ProcessRecipeV1, ProcessStateClassV1, ProcessStateRefV1,
    ProcessTransformationContractV1, RecipeDisclosureV1,
};
use symthaea_manufacturing_process::{
    ManufacturingOperationModeV1, ProcessDefinitionV1, ProcessFamilyRef, TransformationEffectV1,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum ExternalAuthorityPlaneV0 {
    MycelixPlanInstance,
    MycelixResourceAssignment,
    SyntheticExecutionReceipt,
    SyntheticFieldInspection,
    SyntheticCapabilityHistory,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ExternalEvidenceRefV0 {
    plane: ExternalAuthorityPlaneV0,
    subject_ref: String,
    content_blake3: String,
}

impl ExternalEvidenceRefV0 {
    fn validate(&self) -> Result<(), &'static str> {
        if self.subject_ref.is_empty()
            || self.subject_ref.trim() != self.subject_ref
            || self.subject_ref.chars().any(char::is_control)
        {
            return Err("external evidence subject ref must be canonical");
        }
        if self.content_blake3.len() != 64
            || !self
                .content_blake3
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err("external evidence digest must be lowercase 64-hex");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyntheticPilotStageV0 {
    EngineeringCompositionComplete,
    SyntheticExecutionEvidencePresent,
    SyntheticInspectionEvidencePresent,
    SyntheticHistoryLinked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SyntheticPilotChainV0 {
    plan_id: String,
    plan_instance: ExternalEvidenceRefV0,
    assignment: ExternalEvidenceRefV0,
    execution: Option<ExternalEvidenceRefV0>,
    inspection: Option<ExternalEvidenceRefV0>,
    capability_history: Option<ExternalEvidenceRefV0>,
}

impl SyntheticPilotChainV0 {
    fn validate(&self) -> Result<(), &'static str> {
        validate_digest(&self.plan_id)?;
        self.plan_instance.validate()?;
        self.assignment.validate()?;
        if self.plan_instance.plane != ExternalAuthorityPlaneV0::MycelixPlanInstance {
            return Err("wrong authority plane for plan instance");
        }
        if self.assignment.plane != ExternalAuthorityPlaneV0::MycelixResourceAssignment {
            return Err("wrong authority plane for assignment");
        }
        if let Some(execution) = &self.execution {
            execution.validate()?;
            if execution.plane != ExternalAuthorityPlaneV0::SyntheticExecutionReceipt {
                return Err("wrong authority plane for execution receipt");
            }
        }
        if let Some(inspection) = &self.inspection {
            inspection.validate()?;
            if inspection.plane != ExternalAuthorityPlaneV0::SyntheticFieldInspection {
                return Err("wrong authority plane for inspection receipt");
            }
            if self.execution.is_none() {
                return Err("synthetic inspection cannot precede execution evidence");
            }
        }
        if let Some(history) = &self.capability_history {
            history.validate()?;
            if history.plane != ExternalAuthorityPlaneV0::SyntheticCapabilityHistory {
                return Err("wrong authority plane for capability history");
            }
            if self.inspection.is_none() {
                return Err("capability history requires prior inspection evidence");
            }
        }
        Ok(())
    }

    fn stage(&self) -> Result<SyntheticPilotStageV0, &'static str> {
        self.validate()?;
        Ok(if self.capability_history.is_some() {
            SyntheticPilotStageV0::SyntheticHistoryLinked
        } else if self.inspection.is_some() {
            SyntheticPilotStageV0::SyntheticInspectionEvidencePresent
        } else if self.execution.is_some() {
            SyntheticPilotStageV0::SyntheticExecutionEvidencePresent
        } else {
            SyntheticPilotStageV0::EngineeringCompositionComplete
        })
    }

    fn physical_qualification_claimed(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
struct SyntheticPlanInstanceV0 {
    canonical_plan_id: String,
    production_scope_ref: String,
    scheduler_slot: Option<String>,
    display_label: Option<String>,
}

impl SyntheticPlanInstanceV0 {
    fn engineering_id(&self) -> String {
        digest_fields(
            "mycelix.synthetic-plan-instance.v0",
            &[&self.canonical_plan_id, &self.production_scope_ref],
        )
    }
}

#[derive(Clone, Debug)]
struct SyntheticAssignmentV0 {
    plan_instance_id: String,
    node_id: String,
    resource_ref: String,
    capability_profile_ref: String,
    capability_evidence_ref: String,
    scheduler_slot: Option<String>,
}

impl SyntheticAssignmentV0 {
    fn engineering_id(&self) -> String {
        digest_fields(
            "mycelix.synthetic-assignment.v0",
            &[
                &self.plan_instance_id,
                &self.node_id,
                &self.resource_ref,
                &self.capability_profile_ref,
                &self.capability_evidence_ref,
            ],
        )
    }
}

fn validate_digest(value: &str) -> Result<(), &'static str> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("expected lowercase 64-hex digest");
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_fields(domain: &str, fields: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, domain);
    for field in fields {
        hash_field(&mut hasher, field);
    }
    hasher.finalize().to_hex().to_string()
}

fn repeated_digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn state(class: ProcessStateClassV1, subject: &str) -> ProcessStateRefV1 {
    ProcessStateRefV1 {
        namespace: "synthetic.fixture".into(),
        subject_id: subject.into(),
        semantic_version: "1".into(),
        state_class: class,
    }
}

fn fixture() -> (
    ProcessPlanV1,
    ProcessCapabilityRequirementV1,
    ProcessCapabilityProfileV1,
) {
    let process = ProcessDefinitionV1 {
        namespace: "luminous".into(),
        process_name: "synthetic-precision-coupon-milling".into(),
        semantic_version: "1.0.0".into(),
        family_refs: vec![ProcessFamilyRef("luminous:cnc-milling/v1".into())],
        operation_modes: vec![ManufacturingOperationModeV1::Discrete],
        transformation_effects: vec![TransformationEffectV1::MaterialRemoval],
        extension_profile_refs: vec![],
        display_name: Some("Synthetic coupon milling".into()),
        description: Some("Evidence-chain fixture only".into()),
    };
    let process_id = process.process_id().unwrap();
    let stock = state(ProcessStateClassV1::MaterialState, "coupon-stock-a");
    let machined = state(ProcessStateClassV1::GeometryState, "coupon-machined-a");

    let transformation = ProcessTransformationContractV1 {
        process_id: process_id.clone(),
        inputs: vec![stock.clone()],
        outputs: vec![machined.clone()],
        preserved_refs: vec![],
    };
    assert_eq!(transformation.contract_id().unwrap().len(), 64);

    let requirement = ProcessCapabilityRequirementV1 {
        process_id: process_id.clone(),
        envelope_profile_ref: "synthetic:envelope:coupon-v1".into(),
        minimum_evidence: CapabilityEvidenceClassV1::QualifiedUnderProfile,
    };
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
        notes: None,
    };
    let commitment = recipe
        .commitment(
            "synthetic:recipe-schema:v1".into(),
            "synthetic:owner:fixture".into(),
            RecipeDisclosureV1::CommitmentOnly,
            None,
        )
        .unwrap();

    let plan = ProcessPlanV1 {
        semantic_version: "1".into(),
        nodes: vec![
            ProcessPlanNodeV1 {
                node_id: "machine-coupon".into(),
                kind: PlanNodeKindV1::ProcessStep {
                    process_id,
                    recipe_or_commitment_ref: Some(format!(
                        "recipe-commitment:{}",
                        commitment.recipe_id
                    )),
                    capability_requirement_ref: format!(
                        "capability-requirement:{}",
                        requirement.requirement_id().unwrap()
                    ),
                    input_state_refs: vec![stock.coordinate().unwrap()],
                    output_state_refs: vec![machined.coordinate().unwrap()],
                },
                display_label: Some("Machine coupon".into()),
                ui_x: Some(1),
                ui_y: Some(1),
            },
            ProcessPlanNodeV1 {
                node_id: "inspect-coupon".into(),
                kind: PlanNodeKindV1::Inspection {
                    inspection_profile_ref: "synthetic:inspection:coupon-v1".into(),
                    state_ref: machined.coordinate().unwrap(),
                },
                display_label: Some("Inspect coupon".into()),
                ui_x: Some(2),
                ui_y: Some(1),
            },
        ],
        edges: vec![ProcessPlanEdgeV1 {
            from: "machine-coupon".into(),
            to: "inspect-coupon".into(),
            kind: PlanEdgeKindV1::Normal,
            condition_ref: None,
            max_iterations: None,
            disposition_ref: None,
        }],
        display_label: Some("Synthetic pilot plan".into()),
    };
    (plan, requirement, capability)
}

fn base_chain(plan_id: String) -> SyntheticPilotChainV0 {
    SyntheticPilotChainV0 {
        plan_id,
        plan_instance: ExternalEvidenceRefV0 {
            plane: ExternalAuthorityPlaneV0::MycelixPlanInstance,
            subject_ref: "mycelix:synthetic-plan-instance:001".into(),
            content_blake3: repeated_digest('a'),
        },
        assignment: ExternalEvidenceRefV0 {
            plane: ExternalAuthorityPlaneV0::MycelixResourceAssignment,
            subject_ref: "mycelix:synthetic-assignment:001".into(),
            content_blake3: repeated_digest('b'),
        },
        execution: None,
        inspection: None,
        capability_history: None,
    }
}

#[test]
fn positive_chain_reaches_history_link_without_physical_qualification_claim() {
    let (plan, _, _) = fixture();
    let mut chain = base_chain(plan.plan_id().unwrap());
    chain.execution = Some(ExternalEvidenceRefV0 {
        plane: ExternalAuthorityPlaneV0::SyntheticExecutionReceipt,
        subject_ref: "synthetic:execution:001".into(),
        content_blake3: repeated_digest('c'),
    });
    chain.inspection = Some(ExternalEvidenceRefV0 {
        plane: ExternalAuthorityPlaneV0::SyntheticFieldInspection,
        subject_ref: "synthetic:inspection:001".into(),
        content_blake3: repeated_digest('d'),
    });
    chain.capability_history = Some(ExternalEvidenceRefV0 {
        plane: ExternalAuthorityPlaneV0::SyntheticCapabilityHistory,
        subject_ref: "synthetic:history:001".into(),
        content_blake3: repeated_digest('e'),
    });
    assert_eq!(chain.stage().unwrap(), SyntheticPilotStageV0::SyntheticHistoryLinked);
    assert!(!chain.physical_qualification_claimed());
}

#[test]
fn missing_execution_never_becomes_executed_and_inspection_cannot_precede_it() {
    let (plan, _, _) = fixture();
    let mut chain = base_chain(plan.plan_id().unwrap());
    assert_eq!(
        chain.stage().unwrap(),
        SyntheticPilotStageV0::EngineeringCompositionComplete
    );
    chain.inspection = Some(ExternalEvidenceRefV0 {
        plane: ExternalAuthorityPlaneV0::SyntheticFieldInspection,
        subject_ref: "synthetic:inspection:001".into(),
        content_blake3: repeated_digest('d'),
    });
    assert!(chain.validate().is_err());
}

#[test]
fn schedule_changes_do_not_change_engineering_identity_but_resource_changes_do() {
    let (plan, requirement, capability) = fixture();
    let plan_id = plan.plan_id().unwrap();
    let mut instance = SyntheticPlanInstanceV0 {
        canonical_plan_id: plan_id.clone(),
        production_scope_ref: "mycelix:synthetic-work-order:001".into(),
        scheduler_slot: Some("slot-a".into()),
        display_label: Some("First slot".into()),
    };
    let instance_id = instance.engineering_id();
    instance.scheduler_slot = Some("slot-b".into());
    instance.display_label = Some("Rescheduled".into());
    assert_eq!(instance_id, instance.engineering_id());

    let mut assignment = SyntheticAssignmentV0 {
        plan_instance_id: instance_id,
        node_id: "machine-coupon".into(),
        resource_ref: capability.resource_subject_ref.clone(),
        capability_profile_ref: capability.capability_id().unwrap(),
        capability_evidence_ref: capability.evidence_ref.clone(),
        scheduler_slot: Some("slot-a".into()),
    };
    let assignment_id = assignment.engineering_id();
    assignment.scheduler_slot = Some("slot-b".into());
    assert_eq!(assignment_id, assignment.engineering_id());
    assignment.resource_ref = "eng-catalog:synthetic-resource:mill-b".into();
    assert_ne!(assignment_id, assignment.engineering_id());
    assert_eq!(plan_id, plan.plan_id().unwrap());
    assert_eq!(requirement.evaluate(&capability, 150, true), CapabilityMatchV1::ExactAdmitted);
}

#[test]
fn changed_recipe_commitment_changes_plan_identity() {
    let (plan, _, _) = fixture();
    let before = plan.plan_id().unwrap();
    let mut changed = plan.clone();
    let node = changed
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "machine-coupon")
        .unwrap();
    let PlanNodeKindV1::ProcessStep {
        recipe_or_commitment_ref,
        ..
    } = &mut node.kind
    else {
        panic!("fixture process node changed kind")
    };
    *recipe_or_commitment_ref = Some(format!(
        "recipe-commitment:{}",
        repeated_digest('f')
    ));
    assert_ne!(before, changed.plan_id().unwrap());
}
