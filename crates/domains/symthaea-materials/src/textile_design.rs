// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Static garment component/joint topology checks for repair and recovery design.
//!
//! A passing dossier is a coherent design declaration, not physical proof of
//! repairability or disassembly. Production claims require physical tests and
//! independent receiving acceptance through textile_recovery.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Product component role in a garment or textile article.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentRole {
    /// Outer textile panel or shell fabric.
    ShellPanel,
    /// Lining, interlining or inner layer.
    Lining,
    /// Closure or fastening component.
    Fastener,
    /// Brand, size, care or composition label.
    Label,
    /// Decorative or functional trim.
    Trim,
    /// Padding, foam or reinforcement.
    Padding,
    /// Electronic module, sensor or battery.
    Electronics,
    /// Component that does not fit the current vocabulary.
    Other,
}

/// Declared construction method joining one or more components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoiningMethod {
    /// Lockstitch or straight stitch.
    Lockstitch,
    /// Chain stitch.
    Chainstitch,
    /// Overlock/serger seam.
    Overlock,
    /// Reinforcing bartack.
    Bartack,
    /// Adhesive bond or seam tape.
    AdhesiveBond,
    /// Heat or ultrasonic fusion.
    ThermalFusion,
    /// Rivet, eyelet or permanent mechanical fastener.
    Rivet,
    /// Snap, hook-and-eye, button or zipper closure.
    MechanicalClosure,
    /// Hook-and-loop closure.
    HookAndLoop,
    /// Unspecified or unknown joining method.
    Unknown,
    /// Other method, with details in the dossier.
    Other,
}

/// Design declaration for how a joint can be separated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointReversibility {
    /// Intended to be removed/rejoined without destructive panel damage.
    Reversible,
    /// Removable only under a declared constraint or with a controlled cut.
    ConditionallyReversible,
    /// Intended to remain permanently joined in the declared design.
    Permanent,
    /// Reversibility has not been determined.
    Unknown,
}

/// Textile substrate architecture recorded in the product-design revision.
///
/// These are design/specification descriptors, not measured opening results.
/// Physical opening and recovered-fibre quality are qualified separately by
/// the route/process evidence contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FabricStructureKind {
    /// Interlaced warp/weft construction.
    Woven,
    /// Loop-based weft- or warp-knitted construction.
    Knitted,
    /// Fibre web bonded without conventional spun yarns.
    Nonwoven,
    /// Two or more layers bonded into a laminate.
    Laminated,
    /// A construction outside the current typed vocabulary.
    Other,
    /// Structure has not been established.
    Unknown,
}

/// Declared woven pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeavePattern {
    /// Plain weave.
    Plain,
    /// Twill weave.
    Twill,
    /// Satin or sateen weave.
    Satin,
    /// Dobby weave.
    Dobby,
    /// Jacquard weave.
    Jacquard,
    /// Other documented weave.
    Other,
    /// Weave pattern is unresolved.
    Unknown,
}

/// Declared knitted construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnitStructure {
    /// Single jersey.
    SingleJersey,
    /// Rib knit.
    Rib,
    /// Interlock knit.
    Interlock,
    /// Purl knit.
    Purl,
    /// Warp knit.
    WarpKnit,
    /// Other documented knit structure.
    Other,
    /// Knit structure is unresolved.
    Unknown,
}

/// Yarn system used in a woven/knitted fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum YarnSystem {
    /// Ring-spun staple yarn.
    RingSpun,
    /// Rotor/open-end spun staple yarn.
    RotorSpun,
    /// Air-jet spun yarn.
    AirJetSpun,
    /// Vortex-spun yarn.
    VortexSpun,
    /// Continuous-filament yarn.
    Filament,
    /// Yarn system is unresolved.
    Unknown,
}

/// Bonding process for a nonwoven fibre web.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NonwovenBonding {
    /// Needle-punched/mechanically entangled.
    NeedlePunched,
    /// Hydroentangled/spunlaced.
    Hydroentangled,
    /// Thermally bonded.
    Thermal,
    /// Chemically bonded.
    Chemical,
    /// Spunbond process.
    Spunbond,
    /// Meltblown process.
    Meltblown,
    /// Other documented method.
    Other,
    /// Bonding process is unresolved.
    Unknown,
}

/// Bonding method for a laminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaminationMethod {
    /// Adhesive lamination.
    Adhesive,
    /// Thermal/fusion lamination.
    Thermal,
    /// Flame lamination.
    Flame,
    /// Mechanical lamination.
    Mechanical,
    /// Other documented method.
    Other,
    /// Lamination method is unresolved.
    Unknown,
}

/// Type-specific, as-designed construction descriptors for one textile component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricStructure {
    /// Construction family.
    pub kind: FabricStructureKind,
    /// Exact pattern, mill specification or construction record reference.
    pub source_ref: Option<String>,
    /// Declared basis weight in grams per square metre.
    pub basis_weight_gsm: Option<u32>,
    /// Spinning system for woven/knitted constructions.
    pub yarn_system: Option<YarnSystem>,
    /// Nominal yarn linear density in tex.
    pub yarn_count_tex: Option<u32>,
    /// Woven pattern; required only for woven construction.
    pub weave_pattern: Option<WeavePattern>,
    /// Warp thread density per centimetre; required only for woven construction.
    pub warp_density_per_cm: Option<u32>,
    /// Weft thread density per centimetre; required only for woven construction.
    pub weft_density_per_cm: Option<u32>,
    /// Declared weave intersections per square centimetre, from a named construction method.
    pub interlacement_per_cm2: Option<u32>,
    /// Knit construction; required only for knitted construction.
    pub knit_structure: Option<KnitStructure>,
    /// Wale density per centimetre; required only for knitted construction.
    pub wales_per_cm: Option<u32>,
    /// Course density per centimetre; required only for knitted construction.
    pub courses_per_cm: Option<u32>,
    /// Nonwoven bonding process; required only for nonwovens.
    pub nonwoven_bonding: Option<NonwovenBonding>,
    /// Lamination process; required only for laminated structures.
    pub lamination_method: Option<LaminationMethod>,
    /// Reference to the declared substrate architecture beneath the laminate.
    pub substrate_structure_ref: Option<String>,
}

/// A component declared in one exact product-design revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignComponent {
    /// Stable component identifier within the dossier.
    pub component_id: String,
    /// Component role.
    pub role: ComponentRole,
    /// Material identity or versioned material record reference.
    pub material_ref: Option<String>,
    /// Declared component mass in milligrams, when measured/estimated.
    pub declared_mass_mg: Option<u64>,
    /// Whether the design intends the component to be replaceable.
    pub replaceable: bool,
    /// Repair or replacement instruction reference, if replaceable.
    pub replacement_instructions_ref: Option<String>,
    /// Optional as-designed fabric architecture for textile-bearing components.
    /// Required for textile components when OpeningCharacterized intent is declared.
    pub fabric_structure: Option<FabricStructure>,
}

/// A declared joint between components, with removal dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignJoint {
    /// Stable joint identifier within the dossier.
    pub joint_id: String,
    /// Components connected by this joint; at least two are required.
    pub component_ids: Vec<String>,
    /// Whether this joint must be traversed for the declared disassembly/component-recovery intent.
    /// Permanently joined construction outside the target recovery path is allowed.
    pub recovery_relevant: bool,
    /// Joining method.
    pub joining_method: JoiningMethod,
    /// Reversibility declaration.
    pub reversibility: JointReversibility,
    /// Accessible product zone, such as side seam, placket or cuff.
    pub access_zone: Option<String>,
    /// Declared repair/disassembly operation.
    pub removal_operation: Option<String>,
    /// Tools required for the declared operation.
    pub required_tools: Vec<String>,
    /// Joints that must be removed before this joint can be accessed.
    pub predecessor_joint_ids: Vec<String>,
    /// Conditions/limitations for conditional separation.
    pub conditional_notes: Option<String>,
    /// Optional source instruction/pattern reference.
    pub construction_ref: Option<String>,
}

/// Claims the designer wants the dossier to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesignIntent {
    /// Product includes replaceable or repairable modules.
    Repairable,
    /// All relevant joints are designed for component/panel separation.
    Disassemblable,
    /// Components can be recovered as separately identified modules.
    ComponentRecovery,
    /// Every component has an explicit material identity reference.
    MaterialTraceable,
    /// Textile-bearing components include type-appropriate construction descriptors
    /// needed to stratify opening trials; this is not a physical recyclability claim.
    OpeningCharacterized,
}

/// Versioned component/joint topology dossier for one product design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextileDesignDossier {
    /// Product/design family identity.
    pub product_id: String,
    /// Exact design revision; a changed pattern or joining method needs a new revision.
    pub design_revision: String,
    /// Product family or intended function.
    pub product_family: String,
    /// Declared component inventory.
    pub components: Vec<DesignComponent>,
    /// Declared joint topology.
    pub joints: Vec<DesignJoint>,
    /// Intended design properties to be audited.
    pub intents: Vec<DesignIntent>,
}

/// Static design-dossier audit outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesignAuditDisposition {
    /// All declared structures are internally coherent for the requested intents.
    Complete,
    /// Required information is unresolved or missing.
    Incomplete,
    /// Topology or identity is contradictory/invalid.
    Blocked,
}

/// Stable reasons for an incomplete or blocked static design audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DesignAuditReason {
    /// Product identifier is absent.
    MissingProductIdentity,
    /// Design revision is absent.
    MissingDesignRevision,
    /// Product family/functional context is absent.
    MissingProductFamily,
    /// No components were declared.
    NoComponents,
    /// Component identifiers are duplicated.
    DuplicateComponentIdentity,
    /// Joint identifiers are duplicated.
    DuplicateJointIdentity,
    /// A joint has fewer than two components.
    InvalidJointCardinality,
    /// A joint references an undeclared component.
    UnknownComponentReference,
    /// A joint repeats the same component ID.
    DuplicateComponentInJoint,
    /// Material identity is absent or unknown.
    MaterialIdentityUnknown,
    /// Material traceability intent cannot be supported by component metadata.
    MaterialTraceabilityIncomplete,
    /// A replaceable component lacks repair/replacement instructions.
    ReplaceableComponentMissingInstructions,
    /// Repair intent has no replaceable component.
    RepairIntentWithoutReplaceablePart,
    /// Disassembly/component-recovery intent has no declared joints.
    DisassemblyIntentWithoutJoints,
    /// Disassembly/component-recovery intent identifies no recovery-relevant joints.
    DisassemblyIntentWithoutTargetJoints,
    /// A reversible joint has no accessible product zone.
    ReversibleJointMissingAccess,
    /// A reversible joint has no operation/method.
    ReversibleJointMissingOperation,
    /// A reversible joint has no tool declaration.
    ReversibleJointMissingTools,
    /// A conditional joint has no declared conditions.
    ConditionalJointMissingNotes,
    /// A declared predecessor joint does not exist.
    UnknownPredecessorJoint,
    /// Disassembly dependency graph contains a cycle.
    CyclicDisassemblyDependencies,
    /// A permanent joint conflicts with a disassembly/component-recovery intent.
    PermanentJointConflictsWithIntent,
    /// Reversibility is unknown for a joint covered by a disassembly claim.
    JointReversibilityUnknown,
    /// Joining method remains unknown.
    JoiningMethodUnknown,
    /// A textile-bearing component has no fabric-structure descriptor under opening intent.
    FabricStructureMissing,
    /// Fabric structure is unknown or outside the current qualified vocabulary.
    FabricStructureUnknown,
    /// Fabric structure lacks a source specification/pattern reference.
    FabricStructureMissingSource,
    /// Fabric structure lacks a positive basis weight.
    FabricStructureMissingBasisWeight,
    /// Woven structure lacks pattern, yarn or density/interlacement descriptors.
    FabricStructureMissingWovenMetrics,
    /// Knitted structure lacks knit, yarn, wale or course descriptors.
    FabricStructureMissingKnittedMetrics,
    /// Nonwoven structure lacks a declared bonding process.
    FabricStructureMissingNonwovenBonding,
    /// Laminated structure lacks a declared bonding method or substrate reference.
    FabricStructureMissingLaminationDetails,
}

/// Result of a static design topology audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignAudit {
    /// Static audit disposition.
    pub disposition: DesignAuditDisposition,
    /// Stable audit reason codes.
    pub reasons: Vec<DesignAuditReason>,
    /// Number of declared components.
    pub component_count: usize,
    /// Number of declared joints.
    pub joint_count: usize,
    /// Number of joints declared reversible.
    pub reversible_joint_count: usize,
    /// Number of joints declared conditionally reversible.
    pub conditional_joint_count: usize,
    /// Number of permanent joints.
    pub permanent_joint_count: usize,
    /// Number of textile components with declared fabric architecture.
    pub fabric_structure_count: usize,
}

/// Checks whether an identifier is non-empty after trimming.
fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

/// Detects cycles in a joint-removal dependency graph.
fn has_dependency_cycle(joints: &[DesignJoint], known_ids: &BTreeSet<&str>) -> bool {
    fn visit(
        joint_id: &str,
        joints_by_id: &BTreeMap<&str, &DesignJoint>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
    ) -> bool {
        if visited.contains(joint_id) {
            return false;
        }
        if !visiting.insert(joint_id.to_owned()) {
            return true;
        }
        if let Some(joint) = joints_by_id.get(joint_id) {
            for predecessor in &joint.predecessor_joint_ids {
                if visit(predecessor, joints_by_id, visiting, visited) {
                    return true;
                }
            }
        }
        visiting.remove(joint_id);
        visited.insert(joint_id.to_owned());
        false
    }

    let by_id = joints
        .iter()
        .map(|joint| (joint.joint_id.as_str(), joint))
        .collect::<BTreeMap<_, _>>();
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    known_ids
        .iter()
        .any(|joint_id| visit(joint_id, &by_id, &mut visiting, &mut visited))
}

/// Audits design declarations without claiming physical repair/disassembly success.
pub fn audit_textile_design(dossier: &TextileDesignDossier) -> DesignAudit {
    let mut reasons = BTreeSet::new();
    if !nonempty(&dossier.product_id) {
        reasons.insert(DesignAuditReason::MissingProductIdentity);
    }
    if !nonempty(&dossier.design_revision) {
        reasons.insert(DesignAuditReason::MissingDesignRevision);
    }
    if !nonempty(&dossier.product_family) {
        reasons.insert(DesignAuditReason::MissingProductFamily);
    }
    if dossier.components.is_empty() {
        reasons.insert(DesignAuditReason::NoComponents);
    }

    let mut component_ids = BTreeSet::new();
    for component in &dossier.components {
        if !nonempty(&component.component_id)
            || !component_ids.insert(component.component_id.as_str())
        {
            reasons.insert(DesignAuditReason::DuplicateComponentIdentity);
        }
        if component
            .material_ref
            .as_deref()
            .is_none_or(|reference| !nonempty(reference))
        {
            reasons.insert(DesignAuditReason::MaterialIdentityUnknown);
        }
        if component.replaceable
            && component
                .replacement_instructions_ref
                .as_deref()
                .is_none_or(|reference| !nonempty(reference))
        {
            reasons.insert(DesignAuditReason::ReplaceableComponentMissingInstructions);
        }
    }

    let mut joint_ids = BTreeSet::new();
    let mut reversible_joint_count = 0;
    let mut conditional_joint_count = 0;
    let mut permanent_joint_count = 0;
    let has_disassembly_intent = dossier.intents.iter().any(|intent| {
        matches!(
            intent,
            DesignIntent::Disassemblable | DesignIntent::ComponentRecovery
        )
    });
    let has_material_traceability_intent =
        dossier.intents.contains(&DesignIntent::MaterialTraceable);
    let has_repair_intent = dossier.intents.contains(&DesignIntent::Repairable);

    if has_disassembly_intent && dossier.joints.is_empty() {
        reasons.insert(DesignAuditReason::DisassemblyIntentWithoutJoints);
    }

    for joint in &dossier.joints {
        if !nonempty(&joint.joint_id) || !joint_ids.insert(joint.joint_id.as_str()) {
            reasons.insert(DesignAuditReason::DuplicateJointIdentity);
        }
        if joint.component_ids.len() < 2 {
            reasons.insert(DesignAuditReason::InvalidJointCardinality);
        }
        let mut seen = BTreeSet::new();
        for component_id in &joint.component_ids {
            if !seen.insert(component_id.as_str()) {
                reasons.insert(DesignAuditReason::DuplicateComponentInJoint);
            }
            if !component_ids.contains(component_id.as_str()) {
                reasons.insert(DesignAuditReason::UnknownComponentReference);
            }
        }
        if joint.joining_method == JoiningMethod::Unknown {
            reasons.insert(DesignAuditReason::JoiningMethodUnknown);
        }
        match joint.reversibility {
            JointReversibility::Reversible => {
                reversible_joint_count += 1;
                if joint
                    .access_zone
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
                {
                    reasons.insert(DesignAuditReason::ReversibleJointMissingAccess);
                }
                if joint
                    .removal_operation
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
                {
                    reasons.insert(DesignAuditReason::ReversibleJointMissingOperation);
                }
                if joint.required_tools.is_empty()
                    || joint.required_tools.iter().any(|tool| !nonempty(tool))
                {
                    reasons.insert(DesignAuditReason::ReversibleJointMissingTools);
                }
            }
            JointReversibility::ConditionallyReversible => {
                conditional_joint_count += 1;
                if joint
                    .access_zone
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
                {
                    reasons.insert(DesignAuditReason::ReversibleJointMissingAccess);
                }
                if joint
                    .removal_operation
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
                {
                    reasons.insert(DesignAuditReason::ReversibleJointMissingOperation);
                }
                if joint.required_tools.is_empty()
                    || joint.required_tools.iter().any(|tool| !nonempty(tool))
                {
                    reasons.insert(DesignAuditReason::ReversibleJointMissingTools);
                }
                if joint
                    .conditional_notes
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
                {
                    reasons.insert(DesignAuditReason::ConditionalJointMissingNotes);
                }
            }
            JointReversibility::Permanent => {
                permanent_joint_count += 1;
                if has_disassembly_intent && joint.recovery_relevant {
                    reasons.insert(DesignAuditReason::PermanentJointConflictsWithIntent);
                }
            }
            JointReversibility::Unknown => {
                if has_disassembly_intent && joint.recovery_relevant {
                    reasons.insert(DesignAuditReason::JointReversibilityUnknown);
                }
            }
        }
    }

    if has_disassembly_intent && !dossier.joints.iter().any(|joint| joint.recovery_relevant) {
        reasons.insert(DesignAuditReason::DisassemblyIntentWithoutTargetJoints);
    }

    for joint in &dossier.joints {
        for predecessor in &joint.predecessor_joint_ids {
            if !joint_ids.contains(predecessor.as_str()) {
                reasons.insert(DesignAuditReason::UnknownPredecessorJoint);
            }
        }
    }
    if has_dependency_cycle(&dossier.joints, &joint_ids) {
        reasons.insert(DesignAuditReason::CyclicDisassemblyDependencies);
    }

    if has_material_traceability_intent
        && dossier.components.iter().any(|component| {
            component
                .material_ref
                .as_deref()
                .is_none_or(|reference| !nonempty(reference))
        })
    {
        reasons.insert(DesignAuditReason::MaterialTraceabilityIncomplete);
    }
    if has_repair_intent
        && !dossier
            .components
            .iter()
            .any(|component| component.replaceable)
    {
        reasons.insert(DesignAuditReason::RepairIntentWithoutReplaceablePart);
    }

    let blocked_reasons = [
        DesignAuditReason::DuplicateComponentIdentity,
        DesignAuditReason::DuplicateJointIdentity,
        DesignAuditReason::InvalidJointCardinality,
        DesignAuditReason::UnknownComponentReference,
        DesignAuditReason::DuplicateComponentInJoint,
        DesignAuditReason::UnknownPredecessorJoint,
        DesignAuditReason::CyclicDisassemblyDependencies,
        DesignAuditReason::PermanentJointConflictsWithIntent,
    ];
    let disposition = if reasons.is_empty() {
        DesignAuditDisposition::Complete
    } else if reasons
        .iter()
        .any(|reason| blocked_reasons.contains(reason))
    {
        DesignAuditDisposition::Blocked
    } else {
        DesignAuditDisposition::Incomplete
    };

    DesignAudit {
        disposition,
        reasons: reasons.into_iter().collect(),
        component_count: dossier.components.len(),
        joint_count: dossier.joints.len(),
        reversible_joint_count,
        conditional_joint_count,
        permanent_joint_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component(id: &str, replaceable: bool) -> DesignComponent {
        DesignComponent {
            component_id: id.to_owned(),
            role: ComponentRole::ShellPanel,
            material_ref: Some("MATERIAL-COTTON-01".to_owned()),
            declared_mass_mg: Some(100_000),
            replaceable,
            replacement_instructions_ref: replaceable.then(|| "REPAIR-CARD-V1".to_owned()),
        }
    }

    fn reversible_joint(id: &str, left: &str, right: &str) -> DesignJoint {
        DesignJoint {
            joint_id: id.to_owned(),
            component_ids: vec![left.to_owned(), right.to_owned()],
            recovery_relevant: true,
            joining_method: JoiningMethod::Lockstitch,
            reversibility: JointReversibility::Reversible,
            access_zone: Some("SIDE-SEAM".to_owned()),
            removal_operation: Some("UNPICK-LOCKSTITCH".to_owned()),
            required_tools: vec!["SEAM-RIPPER".to_owned()],
            predecessor_joint_ids: Vec::new(),
            conditional_notes: None,
            construction_ref: Some("PATTERN-V1".to_owned()),
        }
    }

    fn valid_dossier() -> TextileDesignDossier {
        TextileDesignDossier {
            product_id: "OVERSHIRT-01".to_owned(),
            design_revision: "REV-A".to_owned(),
            product_family: "WOVEN-OVERSHIRT".to_owned(),
            components: vec![
                component("BODY-PANEL", false),
                component("CUFF-MODULE", true),
            ],
            joints: vec![reversible_joint(
                "BODY-CUFF-JOINT",
                "BODY-PANEL",
                "CUFF-MODULE",
            )],
            intents: vec![
                DesignIntent::Repairable,
                DesignIntent::Disassemblable,
                DesignIntent::MaterialTraceable,
            ],
        }
    }

    #[test]
    fn complete_modular_dossier_passes_static_topology_audit() {
        let result = audit_textile_design(&valid_dossier());
        assert_eq!(result.disposition, DesignAuditDisposition::Complete);
        assert_eq!(result.component_count, 2);
        assert_eq!(result.joint_count, 1);
        assert_eq!(result.reversible_joint_count, 1);
    }

    #[test]
    fn duplicate_component_identity_blocks_dossier() {
        let mut dossier = valid_dossier();
        dossier.components.push(component("BODY-PANEL", false));
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Blocked);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::DuplicateComponentIdentity)
        );
    }

    #[test]
    fn dangling_joint_component_reference_blocks_dossier() {
        let mut dossier = valid_dossier();
        dossier.joints[0].component_ids[1] = "MISSING-COMPONENT".to_owned();
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Blocked);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::UnknownComponentReference)
        );
    }

    #[test]
    fn reversible_joint_without_tool_is_incomplete() {
        let mut dossier = valid_dossier();
        dossier.joints[0].required_tools.clear();
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::ReversibleJointMissingTools)
        );
    }

    #[test]
    fn permanent_joint_conflicts_with_disassembly_intent() {
        let mut dossier = valid_dossier();
        dossier.joints[0].reversibility = JointReversibility::Permanent;
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Blocked);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::PermanentJointConflictsWithIntent)
        );
    }

    #[test]
    fn repair_claim_requires_replaceable_module_and_instruction() {
        let mut dossier = valid_dossier();
        dossier.components[1].replaceable = false;
        dossier.components[1].replacement_instructions_ref = None;
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::RepairIntentWithoutReplaceablePart)
        );
    }

    #[test]
    fn circular_disassembly_dependency_blocks_dossier() {
        let mut dossier = valid_dossier();
        let mut second = reversible_joint("CUFF-TRIM-JOINT", "CUFF-MODULE", "BODY-PANEL");
        dossier.joints[0].predecessor_joint_ids = vec!["CUFF-TRIM-JOINT".to_owned()];
        second.predecessor_joint_ids = vec!["BODY-CUFF-JOINT".to_owned()];
        dossier.joints.push(second);
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Blocked);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::CyclicDisassemblyDependencies)
        );
    }

    #[test]
    fn disassembly_intent_requires_joint_topology() {
        let mut dossier = valid_dossier();
        dossier.joints.clear();
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::DisassemblyIntentWithoutJoints)
        );
    }

    #[test]
    fn replaceable_component_requires_instructions() {
        let mut dossier = valid_dossier();
        dossier.components[1].replacement_instructions_ref = None;
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::ReplaceableComponentMissingInstructions)
        );
    }

    #[test]
    fn permanent_non_target_seam_is_allowed_when_target_path_is_declared() {
        let mut dossier = valid_dossier();
        dossier.joints[0].recovery_relevant = false;
        dossier.joints[0].reversibility = JointReversibility::Permanent;
        let mut target = reversible_joint("REPLACEABLE-CUFF-JOINT", "BODY-PANEL", "CUFF-MODULE");
        target.recovery_relevant = true;
        dossier.joints.push(target);
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Complete);
        assert_eq!(result.permanent_joint_count, 1);
        assert_eq!(result.reversible_joint_count, 1);
    }

    #[test]
    fn disassembly_intent_must_name_recovery_relevant_joints() {
        let mut dossier = valid_dossier();
        dossier.joints[0].recovery_relevant = false;
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::DisassemblyIntentWithoutTargetJoints)
        );
    }

    #[test]
    fn unknown_joining_method_is_not_promoted_to_reversible_design() {
        let mut dossier = valid_dossier();
        dossier.joints[0].joining_method = JoiningMethod::Unknown;
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::JoiningMethodUnknown)
        );
    }

    #[test]
    fn unknown_joint_reversibility_does_not_support_disassembly_claim() {
        let mut dossier = valid_dossier();
        dossier.joints[0].reversibility = JointReversibility::Unknown;
        let result = audit_textile_design(&dossier);
        assert_eq!(result.disposition, DesignAuditDisposition::Incomplete);
        assert!(
            result
                .reasons
                .contains(&DesignAuditReason::JointReversibilityUnknown)
        );
    }

    fn exercise_design_case(case_id: &str) -> (&'static str, Option<&'static str>) {
        match case_id {
            "DESIGN-001-COMPLETE-MODULAR" => {
                complete_modular_dossier_passes_static_topology_audit();
                ("complete", None)
            }
            "DESIGN-002-DUPLICATE-COMPONENT" => {
                duplicate_component_identity_blocks_dossier();
                ("blocked", Some("DuplicateComponentIdentity"))
            }
            "DESIGN-003-DANGLING-COMPONENT" => {
                dangling_joint_component_reference_blocks_dossier();
                ("blocked", Some("UnknownComponentReference"))
            }
            "DESIGN-004-REVERSIBLE-NO-TOOL" => {
                reversible_joint_without_tool_is_incomplete();
                ("incomplete", Some("ReversibleJointMissingTools"))
            }
            "DESIGN-005-PERMANENT-CONFLICT" => {
                permanent_joint_conflicts_with_disassembly_intent();
                ("blocked", Some("PermanentJointConflictsWithIntent"))
            }
            "DESIGN-006-REPAIR-NO-MODULE" => {
                repair_claim_requires_replaceable_module_and_instruction();
                ("incomplete", Some("RepairIntentWithoutReplaceablePart"))
            }
            "DESIGN-007-DEPENDENCY-CYCLE" => {
                circular_disassembly_dependency_blocks_dossier();
                ("blocked", Some("CyclicDisassemblyDependencies"))
            }
            "DESIGN-008-JOINT-UNKNOWN" => {
                unknown_joint_reversibility_does_not_support_disassembly_claim();
                ("incomplete", Some("JointReversibilityUnknown"))
            }
            "DESIGN-009-DISASSEMBLY-NO-JOINTS" => {
                disassembly_intent_requires_joint_topology();
                ("incomplete", Some("DisassemblyIntentWithoutJoints"))
            }
            "DESIGN-010-REPLACEABLE-NO-INSTRUCTIONS" => {
                replaceable_component_requires_instructions();
                (
                    "incomplete",
                    Some("ReplaceableComponentMissingInstructions"),
                )
            }
            "DESIGN-011-UNKNOWN-JOINING-METHOD" => {
                unknown_joining_method_is_not_promoted_to_reversible_design();
                ("incomplete", Some("JoiningMethodUnknown"))
            }
            "DESIGN-012-PERMANENT-NON-TARGET-SEAM" => {
                permanent_non_target_seam_is_allowed_when_target_path_is_declared();
                ("complete", None)
            }
            "DESIGN-013-NO-TARGET-JOINTS" => {
                disassembly_intent_must_name_recovery_relevant_joints();
                ("incomplete", Some("DisassemblyIntentWithoutTargetJoints"))
            }
            other => panic!("unknown textile design corpus case: {other}"),
        }
    }

    #[test]
    fn synthetic_design_corpus_executes_all_topology_regressions() {
        const CORPUS: &str = include_str!(
            "../../../../docs/research/fixtures/textile_sort_fusion_v0_1/design_cases.example.json"
        );
        const SCHEMA: &str = include_str!(
            "../../../../docs/research/fixtures/textile_sort_fusion_v0_1/design_cases.schema.json"
        );

        let corpus: serde_json::Value =
            serde_json::from_str(CORPUS).expect("design corpus must parse as JSON");
        let schema: serde_json::Value =
            serde_json::from_str(SCHEMA).expect("design schema must parse as JSON");
        assert_eq!(schema["properties"]["schema_version"]["const"], "0.1.0");
        assert_eq!(schema["properties"]["cases"]["minItems"].as_u64(), Some(13));
        assert_eq!(corpus["schema_version"], "0.1.0");
        assert_eq!(corpus["corpus_id"], "TEXTILE-DESIGN-CASES-001");
        assert_eq!(corpus["fixture_class"], "synthetic");

        let cases = corpus["cases"]
            .as_array()
            .expect("design cases must be an array");
        assert_eq!(cases.len(), 13, "review design corpus count changes");
        let mut seen = BTreeSet::new();
        for case in cases {
            let case_id = case["case_id"].as_str().expect("case_id must be a string");
            assert!(seen.insert(case_id), "duplicate design case {case_id}");
            assert!(
                case["mutation"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "case {case_id} must declare a mutation"
            );
            let expected = &case["expected"];
            let expected_disposition = expected["disposition"]
                .as_str()
                .expect("expected disposition must be a string");
            let expected_reason = expected["reason"].as_str();
            let (disposition, reason) = exercise_design_case(case_id);
            assert_eq!(
                disposition, expected_disposition,
                "case {case_id} disposition"
            );
            assert_eq!(reason, expected_reason, "case {case_id} reason");
        }
        assert_eq!(seen.len(), 13);
    }
}
