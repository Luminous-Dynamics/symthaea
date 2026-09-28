// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical, non-authoritative robot design subjects.
//!
//! This crate identifies **as-designed engineering intent**. A [`RobotDesignId`]
//! is not a fabricated article, commissioned embodiment, qualified design,
//! safety result, or actuation capability.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Domain separator for V1 robot-design identity transcripts.
pub const ROBOT_DESIGN_SCHEMA_ID: &str = "symthaea.robot-design.v1";
pub const ROBOT_DESIGN_SCHEMA_VERSION: u32 = 1;

/// Ordinary content digest used to reference exact external artifacts/profiles.
///
/// A digest is an identity commitment, not proof that the referenced bytes are
/// correct, qualified, physically present, or authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentDigest([u8; 32]);

impl ContentDigest {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}

impl fmt::Display for ContentDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

macro_rules! semantic_id_type {
    ($name:ident, $kind:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, RobotDesignError> {
                let value = value.into();
                validate_semantic_id($kind, &value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

semantic_id_type!(DesignComponentId, "component");
semantic_id_type!(DesignJointId, "joint");
semantic_id_type!(GeometryDesignId, "geometry");
semantic_id_type!(ActuatorSlotId, "actuator-slot");
semantic_id_type!(SensorSlotId, "sensor-slot");
semantic_id_type!(ControlInterfaceId, "control-interface");

/// Content identity of one validated robot design subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RobotDesignId(ContentDigest);

impl RobotDesignId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }

    pub const fn from_digest(digest: ContentDigest) -> Self { Self(digest) }
    pub const fn from_bytes(bytes: [u8; 32]) -> Self { Self(ContentDigest::from_bytes(bytes)) }
    pub fn from_hex(value: &str) -> Result<Self, RobotDesignError> {
        Ok(Self(ContentDigest::from_bytes(parse_digest_hex(value)?)))
    }
}

impl fmt::Display for RobotDesignId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Provenance relationship to earlier design subjects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobotDesignLineageV1 {
    Independent,
    Derived { parent: RobotDesignId },
    Synthesis { parents: Vec<RobotDesignId> },
}

/// One semantic component of the as-designed morphology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignComponentV1 {
    pub id: DesignComponentId,
    /// Stable semantic role, e.g. `base`, `upper-arm`, `left-foot`.
    pub semantic_role: String,
    /// Human-facing metadata only. Deliberately excluded from design identity.
    pub display_name: Option<String>,
}

/// One directed kinematic relation between design components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignJointV1 {
    pub id: DesignJointId,
    pub parent: DesignComponentId,
    pub child: DesignComponentId,
    /// Exact semantic joint/profile name. Numeric parameter epistemics arrive in 001B.
    pub joint_profile: String,
}

/// Minimal as-designed morphology for V1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotMorphologyDesignV1 {
    /// Versioned morphology/profile identity, not a physical robot identity.
    pub profile: String,
    pub components: Vec<DesignComponentV1>,
    pub joints: Vec<DesignJointV1>,
}

/// Geometry representation kind. Different representations are not silently equivalent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeometryRepresentationV1 {
    PrimitiveCsg,
    Parametric,
    TriangleMesh,
    ImportedCad,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryDesignRefV1 {
    pub id: GeometryDesignId,
    pub component: DesignComponentId,
    pub representation: GeometryRepresentationV1,
    /// Exact artifact/definition identity for the selected representation.
    pub artifact_digest: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialAssignmentV1 {
    pub component: DesignComponentId,
    /// Exact material/profile identity. This does not prove material truth.
    pub material_digest: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActuatorDesignSlotV1 {
    pub id: ActuatorSlotId,
    pub joint: DesignJointId,
    /// Exact actuator/model identity; physical parameter semantics are separate evidence.
    pub actuator_digest: ContentDigest,
    /// Explicit command/actuation semantic profile, never inferred from numeric range.
    pub actuation_profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorDesignSlotV1 {
    pub id: SensorSlotId,
    pub component: DesignComponentId,
    pub sensor_digest: ContentDigest,
    pub observation_profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlInterfaceDesignV1 {
    pub id: ControlInterfaceId,
    /// Semantic interface/profile identity only; never a live ControlLease/capability.
    pub semantic_profile: String,
}

/// Exact positive designer-chosen length in micrometres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0LengthUmV1(u64);

impl C0LengthUmV1 {
    pub const fn new(value_um: u64) -> Result<Self, RobotDesignError> {
        if value_um == 0 { Err(RobotDesignError::InvalidDesignLength) } else { Ok(Self(value_um)) }
    }
    pub const fn value_um(self) -> u64 { self.0 }
    pub const fn validate(self) -> Result<(), RobotDesignError> {
        if self.0 == 0 { Err(RobotDesignError::InvalidDesignLength) } else { Ok(()) }
    }
    pub fn millimetres(self) -> ExactRationalV1 {
        ExactRationalV1::new(self.0 as u128, 1_000).expect("fixed non-zero denominator")
    }
    pub fn f32_bits(self) -> Result<u32, RobotDesignError> {
        let value = self.0 as f32;
        if value.is_finite() { Ok(value.to_bits()) } else { Err(RobotDesignError::FloatConversionOverflow) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExactRationalV1 { pub numerator: u128, pub denominator: u128 }

impl ExactRationalV1 {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, RobotDesignError> {
        if denominator == 0 { return Err(RobotDesignError::InvalidRational); }
        let divisor = gcd_u128(numerator, denominator);
        Ok(Self { numerator: numerator / divisor, denominator: denominator / divisor })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParameterIdV1(String);

impl ParameterIdV1 {
    pub fn new(value: impl Into<String>) -> Result<Self, RobotDesignError> {
        let value = value.into();
        validate_parameter_id(&value)?;
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl fmt::Display for ParameterIdV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.0) }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDesignLengthParameterV1 { pub id: ParameterIdV1, pub value: C0LengthUmV1 }

pub const EXACT_PARAMETER_SCHEMA_ID: &str = "symthaea.robot-design.exact-parameters.v1";
pub const EXACT_PARAMETER_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDesignParameterSetV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub parameters: Vec<ExactDesignLengthParameterV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParameterSetId(ContentDigest);

impl ParameterSetId {
    pub const fn from_digest(digest: ContentDigest) -> Self { Self(digest) }
    pub const fn from_bytes(bytes: [u8; 32]) -> Self { Self(ContentDigest::from_bytes(bytes)) }
    pub fn from_hex(value: &str) -> Result<Self, RobotDesignError> {
        Ok(Self(ContentDigest::from_bytes(parse_digest_hex(value)?)))
    }
    pub const fn digest(self) -> ContentDigest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

impl fmt::Display for ParameterSetId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.to_hex()) }
}

impl ExactDesignParameterSetV1 {
    pub fn new(parameters: Vec<ExactDesignLengthParameterV1>) -> Result<Self, RobotDesignError> {
        let value = Self {
            schema_id: EXACT_PARAMETER_SCHEMA_ID.to_string(),
            schema_version: EXACT_PARAMETER_SCHEMA_VERSION,
            parameters,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), RobotDesignError> {
        if self.schema_id != EXACT_PARAMETER_SCHEMA_ID || self.schema_version != EXACT_PARAMETER_SCHEMA_VERSION {
            return Err(RobotDesignError::UnsupportedParameterSchema {
                schema_id: self.schema_id.clone(), schema_version: self.schema_version
            });
        }
        if self.parameters.is_empty() { return Err(RobotDesignError::EmptyCollection("parameters")); }
        let mut ids = BTreeSet::new();
        for parameter in &self.parameters {
            validate_parameter_id(parameter.id.as_str())?;
            parameter.value.validate()?;
            if !ids.insert(parameter.id.clone()) {
                return Err(RobotDesignError::DuplicateParameterId(parameter.id.to_string()));
            }
        }
        Ok(())
    }
    pub fn canonical_transcript(&self) -> Result<Vec<u8>, RobotDesignError> {
        self.validate()?;
        let mut parameters = self.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.id.cmp(&b.id));
        let mut out = Vec::new();
        put_str(&mut out, EXACT_PARAMETER_SCHEMA_ID);
        put_u32(&mut out, EXACT_PARAMETER_SCHEMA_VERSION);
        put_len(&mut out, parameters.len());
        for parameter in parameters {
            put_str(&mut out, parameter.id.as_str());
            put_u64(&mut out, parameter.value.value_um());
        }
        Ok(out)
    }
    pub fn id(&self) -> Result<ParameterSetId, RobotDesignError> {
        Ok(ParameterSetId(hash_transcript(&self.canonical_transcript()?)))
    }
    pub fn parameter(&self, id: &ParameterIdV1) -> Option<C0LengthUmV1> {
        self.parameters.iter().find(|p| &p.id == id).map(|p| p.value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactLengthDomainV1 {
    Values(Vec<C0LengthUmV1>),
    Range { lower: C0LengthUmV1, upper: C0LengthUmV1, step_um: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDesignParameterDomainV1 { pub id: ParameterIdV1, pub domain: ExactLengthDomainV1 }

pub const EXACT_SEARCH_DOMAIN_SCHEMA_ID: &str = "symthaea.robot-design.search-domain.v1";
pub const EXACT_SEARCH_DOMAIN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDesignSearchDomainV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub parameters: Vec<ExactDesignParameterDomainV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SearchDomainId(ContentDigest);

impl SearchDomainId {
    pub const fn from_digest(digest: ContentDigest) -> Self { Self(digest) }
    pub const fn from_bytes(bytes: [u8; 32]) -> Self { Self(ContentDigest::from_bytes(bytes)) }
    pub fn from_hex(value: &str) -> Result<Self, RobotDesignError> {
        Ok(Self(ContentDigest::from_bytes(parse_digest_hex(value)?)))
    }
    pub const fn digest(self) -> ContentDigest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

impl fmt::Display for SearchDomainId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.to_hex()) }
}

impl ExactDesignSearchDomainV1 {
    pub fn new(parameters: Vec<ExactDesignParameterDomainV1>) -> Result<Self, RobotDesignError> {
        let value = Self {
            schema_id: EXACT_SEARCH_DOMAIN_SCHEMA_ID.to_string(),
            schema_version: EXACT_SEARCH_DOMAIN_SCHEMA_VERSION,
            parameters,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), RobotDesignError> {
        if self.schema_id != EXACT_SEARCH_DOMAIN_SCHEMA_ID || self.schema_version != EXACT_SEARCH_DOMAIN_SCHEMA_VERSION {
            return Err(RobotDesignError::UnsupportedSearchDomainSchema {
                schema_id: self.schema_id.clone(), schema_version: self.schema_version
            });
        }
        if self.parameters.is_empty() { return Err(RobotDesignError::EmptyCollection("search-domain parameters")); }
        let mut ids = BTreeSet::new();
        for parameter in &self.parameters {
            validate_parameter_id(parameter.id.as_str())?;
            if !ids.insert(parameter.id.clone()) {
                return Err(RobotDesignError::DuplicateParameterId(parameter.id.to_string()));
            }
            match &parameter.domain {
                ExactLengthDomainV1::Values(values) => {
                    if values.is_empty() { return Err(RobotDesignError::EmptyCollection("domain values")); }
                    for value in values { value.validate()?; }
                    let mut sorted = values.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    if sorted.len() != values.len() {
                        return Err(RobotDesignError::DuplicateDomainValue(parameter.id.to_string()));
                    }
                    if sorted != *values {
                        return Err(RobotDesignError::NonCanonicalDomainOrder(parameter.id.to_string()));
                    }
                }
                ExactLengthDomainV1::Range { lower, upper, step_um } => {
                    lower.validate()?;
                    upper.validate()?;
                    if *step_um == 0 || lower > upper {
                        return Err(RobotDesignError::InvalidDomainRange(parameter.id.to_string()));
                    }
                    if (upper.value_um() - lower.value_um()) % *step_um != 0 {
                        return Err(RobotDesignError::NonTerminatingDomainRange(parameter.id.to_string()));
                    }
                }
            }
        }
        Ok(())
    }
    pub fn canonical_transcript(&self) -> Result<Vec<u8>, RobotDesignError> {
        self.validate()?;
        let mut parameters = self.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.id.cmp(&b.id));
        let mut out = Vec::new();
        put_str(&mut out, EXACT_SEARCH_DOMAIN_SCHEMA_ID);
        put_u32(&mut out, EXACT_SEARCH_DOMAIN_SCHEMA_VERSION);
        put_len(&mut out, parameters.len());
        for parameter in parameters {
            put_str(&mut out, parameter.id.as_str());
            match &parameter.domain {
                ExactLengthDomainV1::Values(values) => {
                    put_u8(&mut out, 0);
                    put_len(&mut out, values.len());
                    for value in values { put_u64(&mut out, value.value_um()); }
                }
                ExactLengthDomainV1::Range { lower, upper, step_um } => {
                    put_u8(&mut out, 1);
                    put_u64(&mut out, lower.value_um());
                    put_u64(&mut out, upper.value_um());
                    put_u64(&mut out, *step_um);
                }
            }
        }
        Ok(out)
    }
    pub fn id(&self) -> Result<SearchDomainId, RobotDesignError> {
        Ok(SearchDomainId(hash_transcript(&self.canonical_transcript()?)))
    }
    pub fn enumerate(&self) -> Result<Vec<ExactDesignParameterSetV1>, RobotDesignError> {
        self.validate()?;
        let mut dimensions = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let values = match &parameter.domain {
                ExactLengthDomainV1::Values(values) => values.clone(),
                ExactLengthDomainV1::Range { lower, upper, step_um } => {
                    let span = upper.value_um() - lower.value_um();
                    let count = span.checked_div(*step_um).and_then(|n| n.checked_add(1))
                        .ok_or(RobotDesignError::DomainEnumerationOverflow)?;
                    let count = usize::try_from(count).map_err(|_| RobotDesignError::DomainEnumerationOverflow)?;
                    let mut values = Vec::with_capacity(count);
                    for index in 0..count {
                        let offset = (*step_um as u128).checked_mul(index as u128)
                            .ok_or(RobotDesignError::DomainEnumerationOverflow)?;
                        let value = (lower.value_um() as u128).checked_add(offset)
                            .ok_or(RobotDesignError::DomainEnumerationOverflow)?;
                        values.push(C0LengthUmV1::new(u64::try_from(value)
                            .map_err(|_| RobotDesignError::DomainEnumerationOverflow)?)?);
                    }
                    values
                }
            };
            dimensions.push((parameter.id.clone(), values));
        }
        let mut output = vec![Vec::<ExactDesignLengthParameterV1>::new()];
        for (id, values) in dimensions {
            let mut next = Vec::new();
            for partial in &output {
                for value in &values {
                    let mut candidate = partial.clone();
                    candidate.push(ExactDesignLengthParameterV1 { id: id.clone(), value: *value });
                    next.push(candidate);
                }
            }
            output = next;
        }
        output.into_iter().map(ExactDesignParameterSetV1::new).collect()
    }
}

/// As-designed engineering subject. It intentionally contains no physical authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotDesignSubjectV1 {
    pub schema_id: String,
    pub schema_version: u32,
    /// Human-facing metadata only. Deliberately excluded from design identity.
    pub display_name: Option<String>,
    pub lineage: RobotDesignLineageV1,
    /// Exact accepted requirement/configuration snapshot owned by ETK/SE semantics.
    pub requirement_snapshot: ContentDigest,
    pub morphology: RobotMorphologyDesignV1,
    pub geometry: Vec<GeometryDesignRefV1>,
    pub material_assignments: Vec<MaterialAssignmentV1>,
    pub actuator_slots: Vec<ActuatorDesignSlotV1>,
    pub sensor_slots: Vec<SensorDesignSlotV1>,
    pub control_interfaces: Vec<ControlInterfaceDesignV1>,
    /// Exact fabrication-constraint profile identity. Not fabrication authority.
    pub fabrication_constraints: ContentDigest,
    /// Exact selected design-parameter-set identity. Search-domain identity is intentionally a different type.
    pub parameter_set: ParameterSetId,
}

impl RobotDesignSubjectV1 {
    pub fn new(
        requirement_snapshot: ContentDigest,
        morphology: RobotMorphologyDesignV1,
        fabrication_constraints: ContentDigest,
        parameter_set: ParameterSetId,
    ) -> Self {
        Self {
            schema_id: ROBOT_DESIGN_SCHEMA_ID.to_string(),
            schema_version: ROBOT_DESIGN_SCHEMA_VERSION,
            display_name: None,
            lineage: RobotDesignLineageV1::Independent,
            requirement_snapshot,
            morphology,
            geometry: Vec::new(),
            material_assignments: Vec::new(),
            actuator_slots: Vec::new(),
            sensor_slots: Vec::new(),
            control_interfaces: Vec::new(),
            fabrication_constraints,
            parameter_set,
        }
    }

    /// Validate referential integrity and V1 canonicalization preconditions.
    pub fn validate(&self) -> Result<(), RobotDesignError> {
        if self.schema_id != ROBOT_DESIGN_SCHEMA_ID
            || self.schema_version != ROBOT_DESIGN_SCHEMA_VERSION
        {
            return Err(RobotDesignError::UnsupportedSchema {
                schema_id: self.schema_id.clone(),
                schema_version: self.schema_version,
            });
        }

        validate_semantic_text("morphology profile", &self.morphology.profile)?;
        validate_optional_display_name(self.display_name.as_deref())?;
        validate_lineage(&self.lineage)?;

        let mut component_ids = BTreeSet::new();
        for component in &self.morphology.components {
            validate_semantic_id("component", component.id.as_str())?;
            validate_semantic_text("component semantic role", &component.semantic_role)?;
            validate_optional_display_name(component.display_name.as_deref())?;
            if !component_ids.insert(component.id.clone()) {
                return Err(RobotDesignError::DuplicateId {
                    kind: "component",
                    id: component.id.to_string(),
                });
            }
        }
        if component_ids.is_empty() {
            return Err(RobotDesignError::EmptyCollection("components"));
        }

        let mut joint_ids = BTreeSet::new();
        for joint in &self.morphology.joints {
            validate_semantic_id("joint", joint.id.as_str())?;
            validate_semantic_text("joint profile", &joint.joint_profile)?;
            if !joint_ids.insert(joint.id.clone()) {
                return Err(RobotDesignError::DuplicateId {
                    kind: "joint",
                    id: joint.id.to_string(),
                });
            }
            require_component(&component_ids, &joint.parent, "joint parent")?;
            require_component(&component_ids, &joint.child, "joint child")?;
            if joint.parent == joint.child {
                return Err(RobotDesignError::SelfJoint(joint.id.to_string()));
            }
        }
        validate_kinematic_acyclic(&component_ids, &self.morphology.joints)?;

        let mut geometry_ids = BTreeSet::new();
        for geometry in &self.geometry {
            validate_semantic_id("geometry", geometry.id.as_str())?;
            if !geometry_ids.insert(geometry.id.clone()) {
                return Err(RobotDesignError::DuplicateId {
                    kind: "geometry",
                    id: geometry.id.to_string(),
                });
            }
            require_component(&component_ids, &geometry.component, "geometry component")?;
            if let GeometryRepresentationV1::Other(label) = &geometry.representation {
                validate_semantic_text("geometry representation", label)?;
            }
        }

        let mut material_components = BTreeSet::new();
        for assignment in &self.material_assignments {
            require_component(
                &component_ids,
                &assignment.component,
                "material assignment component",
            )?;
            if !material_components.insert(assignment.component.clone()) {
                return Err(RobotDesignError::DuplicateMaterialAssignment(
                    assignment.component.to_string(),
                ));
            }
        }

        let mut actuator_ids = BTreeSet::new();
        for actuator in &self.actuator_slots {
            validate_semantic_id("actuator-slot", actuator.id.as_str())?;
            validate_semantic_text("actuation profile", &actuator.actuation_profile)?;
            if !actuator_ids.insert(actuator.id.clone()) {
                return Err(RobotDesignError::DuplicateId {
                    kind: "actuator-slot",
                    id: actuator.id.to_string(),
                });
            }
            if !joint_ids.contains(&actuator.joint) {
                return Err(RobotDesignError::MissingJoint {
                    context: "actuator slot",
                    id: actuator.joint.to_string(),
                });
            }
        }

        let mut sensor_ids = BTreeSet::new();
        for sensor in &self.sensor_slots {
            validate_semantic_id("sensor-slot", sensor.id.as_str())?;
            validate_semantic_text("observation profile", &sensor.observation_profile)?;
            if !sensor_ids.insert(sensor.id.clone()) {
                return Err(RobotDesignError::DuplicateId {
                    kind: "sensor-slot",
                    id: sensor.id.to_string(),
                });
            }
            require_component(&component_ids, &sensor.component, "sensor slot component")?;
        }

        let mut interface_ids = BTreeSet::new();
        for interface in &self.control_interfaces {
            validate_semantic_id("control-interface", interface.id.as_str())?;
            validate_semantic_text("control interface profile", &interface.semantic_profile)?;
            if !interface_ids.insert(interface.id.clone()) {
                return Err(RobotDesignError::DuplicateId {
                    kind: "control-interface",
                    id: interface.id.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Canonical, language-neutral transcript used for design identity.
    ///
    /// Set-like collections are sorted by stable semantic IDs. Human-facing
    /// `display_name` fields are deliberately absent from this transcript.
    pub fn canonical_transcript(&self) -> Result<Vec<u8>, RobotDesignError> {
        self.validate()?;

        let mut out = Vec::new();
        put_str(&mut out, ROBOT_DESIGN_SCHEMA_ID);
        put_u32(&mut out, ROBOT_DESIGN_SCHEMA_VERSION);

        match &self.lineage {
            RobotDesignLineageV1::Independent => put_u8(&mut out, 0),
            RobotDesignLineageV1::Derived { parent } => {
                put_u8(&mut out, 1);
                put_digest(&mut out, parent.digest());
            }
            RobotDesignLineageV1::Synthesis { parents } => {
                put_u8(&mut out, 2);
                let mut parents = parents.to_vec();
                parents.sort_unstable();
                put_len(&mut out, parents.len());
                for parent in parents {
                    put_digest(&mut out, parent.digest());
                }
            }
        }

        put_digest(&mut out, self.requirement_snapshot);
        put_str(&mut out, &self.morphology.profile);

        let mut components = self.morphology.components.iter().collect::<Vec<_>>();
        components.sort_by(|left, right| left.id.cmp(&right.id));
        put_len(&mut out, components.len());
        for component in components {
            put_str(&mut out, component.id.as_str());
            put_str(&mut out, &component.semantic_role);
        }

        let mut joints = self.morphology.joints.iter().collect::<Vec<_>>();
        joints.sort_by(|left, right| left.id.cmp(&right.id));
        put_len(&mut out, joints.len());
        for joint in joints {
            put_str(&mut out, joint.id.as_str());
            put_str(&mut out, joint.parent.as_str());
            put_str(&mut out, joint.child.as_str());
            put_str(&mut out, &joint.joint_profile);
        }

        let mut geometry = self.geometry.iter().collect::<Vec<_>>();
        geometry.sort_by(|left, right| left.id.cmp(&right.id));
        put_len(&mut out, geometry.len());
        for item in geometry {
            put_str(&mut out, item.id.as_str());
            put_str(&mut out, item.component.as_str());
            match &item.representation {
                GeometryRepresentationV1::PrimitiveCsg => put_u8(&mut out, 0),
                GeometryRepresentationV1::Parametric => put_u8(&mut out, 1),
                GeometryRepresentationV1::TriangleMesh => put_u8(&mut out, 2),
                GeometryRepresentationV1::ImportedCad => put_u8(&mut out, 3),
                GeometryRepresentationV1::Other(label) => {
                    put_u8(&mut out, 4);
                    put_str(&mut out, label);
                }
            }
            put_digest(&mut out, item.artifact_digest);
        }

        let mut materials = self.material_assignments.iter().collect::<Vec<_>>();
        materials.sort_by(|left, right| left.component.cmp(&right.component));
        put_len(&mut out, materials.len());
        for assignment in materials {
            put_str(&mut out, assignment.component.as_str());
            put_digest(&mut out, assignment.material_digest);
        }

        let mut actuators = self.actuator_slots.iter().collect::<Vec<_>>();
        actuators.sort_by(|left, right| left.id.cmp(&right.id));
        put_len(&mut out, actuators.len());
        for actuator in actuators {
            put_str(&mut out, actuator.id.as_str());
            put_str(&mut out, actuator.joint.as_str());
            put_digest(&mut out, actuator.actuator_digest);
            put_str(&mut out, &actuator.actuation_profile);
        }

        let mut sensors = self.sensor_slots.iter().collect::<Vec<_>>();
        sensors.sort_by(|left, right| left.id.cmp(&right.id));
        put_len(&mut out, sensors.len());
        for sensor in sensors {
            put_str(&mut out, sensor.id.as_str());
            put_str(&mut out, sensor.component.as_str());
            put_digest(&mut out, sensor.sensor_digest);
            put_str(&mut out, &sensor.observation_profile);
        }

        let mut interfaces = self.control_interfaces.iter().collect::<Vec<_>>();
        interfaces.sort_by(|left, right| left.id.cmp(&right.id));
        put_len(&mut out, interfaces.len());
        for interface in interfaces {
            put_str(&mut out, interface.id.as_str());
            put_str(&mut out, &interface.semantic_profile);
        }

        put_digest(&mut out, self.fabrication_constraints);
        put_digest(&mut out, self.parameter_set.digest());
        Ok(out)
    }

    pub fn design_id(&self) -> Result<RobotDesignId, RobotDesignError> {
        let transcript = self.canonical_transcript()?;
        let digest: [u8; 32] = Sha256::digest(transcript).into();
        Ok(RobotDesignId(ContentDigest::from_bytes(digest)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobotDesignError {
    UnsupportedSchema {
        schema_id: String,
        schema_version: u32,
    },
    InvalidSemanticId {
        kind: &'static str,
        value: String,
    },
    InvalidSemanticText(&'static str),
    EmptyCollection(&'static str),
    DuplicateId {
        kind: &'static str,
        id: String,
    },
    MissingComponent {
        context: &'static str,
        id: String,
    },
    MissingJoint {
        context: &'static str,
        id: String,
    },
    SelfJoint(String),
    KinematicCycle,
    DuplicateMaterialAssignment(String),
    InvalidLineage(&'static str),
    InvalidDesignLength,
    FloatConversionOverflow,
    InvalidRational,
    UnsupportedParameterSchema { schema_id: String, schema_version: u32 },
    DuplicateParameterId(String),
    UnsupportedSearchDomainSchema { schema_id: String, schema_version: u32 },
    DuplicateDomainValue(String),
    NonCanonicalDomainOrder(String),
    InvalidDomainRange(String),
    NonTerminatingDomainRange(String),
    DomainEnumerationOverflow,
}

impl fmt::Display for RobotDesignError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema {
                schema_id,
                schema_version,
            } => write!(
                formatter,
                "unsupported robot-design schema {schema_id}@{schema_version}"
            ),
            Self::InvalidSemanticId { kind, value } => {
                write!(formatter, "invalid {kind} semantic id: {value:?}")
            }
            Self::InvalidSemanticText(field) => write!(formatter, "invalid {field}"),
            Self::EmptyCollection(field) => write!(formatter, "{field} must not be empty"),
            Self::DuplicateId { kind, id } => write!(formatter, "duplicate {kind} id: {id}"),
            Self::MissingComponent { context, id } => {
                write!(formatter, "{context} references missing component: {id}")
            }
            Self::MissingJoint { context, id } => {
                write!(formatter, "{context} references missing joint: {id}")
            }
            Self::SelfJoint(id) => write!(formatter, "joint {id} connects a component to itself"),
            Self::KinematicCycle => {
                formatter.write_str("kinematic component graph contains a cycle")
            }
            Self::DuplicateMaterialAssignment(id) => {
                write!(
                    formatter,
                    "duplicate material assignment for component: {id}"
                )
            }
            Self::InvalidLineage(message) => write!(formatter, "invalid design lineage: {message}"),
            Self::InvalidDesignLength => formatter.write_str("design length must be positive"),
            Self::FloatConversionOverflow => formatter.write_str("design length cannot be represented as finite f32"),
            Self::InvalidRational => formatter.write_str("rational denominator must be non-zero"),
            Self::UnsupportedParameterSchema { schema_id, schema_version } => write!(formatter, "unsupported exact-parameter schema {schema_id}@{schema_version}"),
            Self::DuplicateParameterId(id) => write!(formatter, "duplicate parameter id: {id}"),
            Self::UnsupportedSearchDomainSchema { schema_id, schema_version } => write!(formatter, "unsupported search-domain schema {schema_id}@{schema_version}"),
            Self::DuplicateDomainValue(id) => write!(formatter, "duplicate domain value for parameter: {id}"),
            Self::NonCanonicalDomainOrder(id) => write!(formatter, "domain values for parameter {id} are not canonically sorted"),
            Self::InvalidDomainRange(id) => write!(formatter, "invalid domain range for parameter: {id}"),
            Self::NonTerminatingDomainRange(id) => write!(formatter, "domain range for parameter {id} does not terminate exactly"),
            Self::DomainEnumerationOverflow => formatter.write_str("domain enumeration exceeds representable capacity"),
        }
    }
}

impl std::error::Error for RobotDesignError {}

fn validate_semantic_id(kind: &'static str, value: &str) -> Result<(), RobotDesignError> {
    let valid = !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        });
    if valid {
        Ok(())
    } else {
        Err(RobotDesignError::InvalidSemanticId {
            kind,
            value: value.to_string(),
        })
    }
}

fn validate_semantic_text(field: &'static str, value: &str) -> Result<(), RobotDesignError> {
    if value.trim().is_empty() || value.len() > 256 || value.contains('\0') {
        Err(RobotDesignError::InvalidSemanticText(field))
    } else {
        Ok(())
    }
}

fn validate_optional_display_name(value: Option<&str>) -> Result<(), RobotDesignError> {
    if let Some(value) = value {
        validate_semantic_text("display name", value)?;
    }
    Ok(())
}

fn validate_lineage(lineage: &RobotDesignLineageV1) -> Result<(), RobotDesignError> {
    if let RobotDesignLineageV1::Synthesis { parents } = lineage {
        if parents.len() < 2 {
            return Err(RobotDesignError::InvalidLineage(
                "synthesis requires at least two distinct parents",
            ));
        }
        let distinct = parents.iter().copied().collect::<BTreeSet<_>>();
        if distinct.len() != parents.len() {
            return Err(RobotDesignError::InvalidLineage(
                "synthesis parent identities must be distinct",
            ));
        }
    }
    Ok(())
}

fn require_component(
    components: &BTreeSet<DesignComponentId>,
    component: &DesignComponentId,
    context: &'static str,
) -> Result<(), RobotDesignError> {
    if components.contains(component) {
        Ok(())
    } else {
        Err(RobotDesignError::MissingComponent {
            context,
            id: component.to_string(),
        })
    }
}

fn validate_kinematic_acyclic(
    components: &BTreeSet<DesignComponentId>,
    joints: &[DesignJointV1],
) -> Result<(), RobotDesignError> {
    let mut indegree = components
        .iter()
        .cloned()
        .map(|component| (component, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<DesignComponentId, Vec<DesignComponentId>>::new();

    for joint in joints {
        *indegree
            .get_mut(&joint.child)
            .expect("joint child was validated before cycle check") += 1;
        outgoing
            .entry(joint.parent.clone())
            .or_default()
            .push(joint.child.clone());
    }

    let mut ready = indegree
        .iter()
        .filter_map(|(component, degree)| (*degree == 0).then_some(component.clone()))
        .collect::<VecDeque<_>>();
    let mut visited = 0usize;

    while let Some(component) = ready.pop_front() {
        visited += 1;
        if let Some(children) = outgoing.get(&component) {
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .expect("joint child was validated before cycle check");
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(child.clone());
                }
            }
        }
    }

    if visited == components.len() {
        Ok(())
    } else {
        Err(RobotDesignError::KinematicCycle)
    }
}

fn validate_parameter_id(value: &str) -> Result<(), RobotDesignError> {
    let valid = !value.is_empty() && value.len() <= 128
        && value.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-'));
    if valid { Ok(()) } else {
        Err(RobotDesignError::InvalidSemanticId { kind: "parameter", value: value.to_string() })
    }
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 { let remainder = left % right; left = right; right = remainder; }
    left.max(1)
}

fn hash_transcript(transcript: &[u8]) -> ContentDigest {
    ContentDigest::from_bytes(Sha256::digest(transcript).into())
}

fn parse_digest_hex(value: &str) -> Result<[u8; 32], RobotDesignError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(RobotDesignError::InvalidSemanticText("digest hex"));
    }
    let mut bytes = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(bytes)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => unreachable!("validated hexadecimal input"),
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_be_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn put_digest(out: &mut Vec<u8>, digest: ContentDigest) {
    out.extend_from_slice(&digest.into_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(fill: u8) -> ContentDigest {
        ContentDigest::from_bytes([fill; 32])
    }

    fn component(id: &str, role: &str) -> DesignComponentV1 {
        DesignComponentV1 {
            id: DesignComponentId::new(id).unwrap(),
            semantic_role: role.to_string(),
            display_name: None,
        }
    }

    fn base_subject() -> RobotDesignSubjectV1 {
        let base = DesignComponentId::new("base").unwrap();
        let link = DesignComponentId::new("link-1").unwrap();
        let joint = DesignJointId::new("joint-1").unwrap();

        let morphology = RobotMorphologyDesignV1 {
            profile: "single-rotary-joint-v1".to_string(),
            components: vec![
                component(base.as_str(), "base"),
                component(link.as_str(), "moving-link"),
            ],
            joints: vec![DesignJointV1 {
                id: joint.clone(),
                parent: base.clone(),
                child: link.clone(),
                joint_profile: "revolute-radians-v1".to_string(),
            }],
        };

        let mut subject = RobotDesignSubjectV1::new(
            digest(1),
            morphology,
            digest(8),
            ParameterSetId::from_digest(digest(9)),
        );
        subject.geometry = vec![
            GeometryDesignRefV1 {
                id: GeometryDesignId::new("geometry-base").unwrap(),
                component: base.clone(),
                representation: GeometryRepresentationV1::Parametric,
                artifact_digest: digest(2),
            },
            GeometryDesignRefV1 {
                id: GeometryDesignId::new("geometry-link").unwrap(),
                component: link.clone(),
                representation: GeometryRepresentationV1::TriangleMesh,
                artifact_digest: digest(3),
            },
        ];
        subject.material_assignments = vec![
            MaterialAssignmentV1 {
                component: base.clone(),
                material_digest: digest(4),
            },
            MaterialAssignmentV1 {
                component: link.clone(),
                material_digest: digest(5),
            },
        ];
        subject.actuator_slots = vec![ActuatorDesignSlotV1 {
            id: ActuatorSlotId::new("actuator-1").unwrap(),
            joint,
            actuator_digest: digest(6),
            actuation_profile: "torque-newton-metres-v1".to_string(),
        }];
        subject.sensor_slots = vec![SensorDesignSlotV1 {
            id: SensorSlotId::new("encoder-1").unwrap(),
            component: link,
            sensor_digest: digest(7),
            observation_profile: "joint-position-radians-v1".to_string(),
        }];
        subject.control_interfaces = vec![ControlInterfaceDesignV1 {
            id: ControlInterfaceId::new("joint-control").unwrap(),
            semantic_profile: "bounded-joint-command-v1".to_string(),
        }];
        subject
    }

    #[test]
    fn set_like_insertion_order_does_not_change_identity() {
        let subject = base_subject();
        let expected = subject.design_id().unwrap();

        let mut reordered = subject.clone();
        reordered.morphology.components.reverse();
        reordered.geometry.reverse();
        reordered.material_assignments.reverse();
        reordered.actuator_slots.reverse();
        reordered.sensor_slots.reverse();
        reordered.control_interfaces.reverse();

        assert_eq!(reordered.design_id().unwrap(), expected);
    }

    #[test]
    fn semantic_material_change_changes_identity() {
        let subject = base_subject();
        let original = subject.design_id().unwrap();
        let mut changed = subject.clone();
        changed.material_assignments[0].material_digest = digest(42);
        assert_ne!(changed.design_id().unwrap(), original);
    }

    #[test]
    fn actuator_change_changes_identity() {
        let subject = base_subject();
        let original = subject.design_id().unwrap();
        let mut changed = subject.clone();
        changed.actuator_slots[0].actuator_digest = digest(43);
        assert_ne!(changed.design_id().unwrap(), original);
    }

    #[test]
    fn geometry_change_changes_identity() {
        let subject = base_subject();
        let original = subject.design_id().unwrap();
        let mut changed = subject.clone();
        changed.geometry[0].artifact_digest = digest(44);
        assert_ne!(changed.design_id().unwrap(), original);
    }

    #[test]
    fn requirement_snapshot_change_changes_identity() {
        let subject = base_subject();
        let original = subject.design_id().unwrap();
        let mut changed = subject.clone();
        changed.requirement_snapshot = digest(45);
        assert_ne!(changed.design_id().unwrap(), original);
    }

    #[test]
    fn fabrication_constraint_change_changes_identity() {
        let subject = base_subject();
        let original = subject.design_id().unwrap();
        let mut changed = subject.clone();
        changed.fabrication_constraints = digest(46);
        assert_ne!(changed.design_id().unwrap(), original);
    }

    #[test]
    fn human_facing_labels_do_not_change_identity() {
        let subject = base_subject();
        let original = subject.design_id().unwrap();
        let mut relabeled = subject.clone();
        relabeled.display_name = Some("JointLab Mk I".to_string());
        relabeled.morphology.components[0].display_name = Some("Pretty Base".to_string());
        assert_eq!(relabeled.design_id().unwrap(), original);
    }

    #[test]
    fn missing_component_reference_fails_closed() {
        let mut subject = base_subject();
        subject.sensor_slots[0].component = DesignComponentId::new("missing").unwrap();
        assert!(matches!(
            subject.validate(),
            Err(RobotDesignError::MissingComponent {
                context: "sensor slot component",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_semantic_ids_are_rejected() {
        let mut subject = base_subject();
        subject
            .morphology
            .components
            .push(subject.morphology.components[0].clone());
        assert!(matches!(
            subject.validate(),
            Err(RobotDesignError::DuplicateId {
                kind: "component",
                ..
            })
        ));
    }

    #[test]
    fn cyclic_kinematic_graph_is_rejected() {
        let mut subject = base_subject();
        subject.morphology.joints.push(DesignJointV1 {
            id: DesignJointId::new("joint-return").unwrap(),
            parent: DesignComponentId::new("link-1").unwrap(),
            child: DesignComponentId::new("base").unwrap(),
            joint_profile: "revolute-radians-v1".to_string(),
        });
        assert_eq!(subject.validate(), Err(RobotDesignError::KinematicCycle));
    }

    #[test]
    fn synthesis_parent_order_is_nonsemantic() {
        let mut left = base_subject();
        let parent_a = RobotDesignId(digest(10));
        let parent_b = RobotDesignId(digest(11));
        left.lineage = RobotDesignLineageV1::Synthesis {
            parents: vec![parent_a, parent_b],
        };

        let mut right = left.clone();
        right.lineage = RobotDesignLineageV1::Synthesis {
            parents: vec![parent_b, parent_a],
        };

        assert_eq!(left.design_id().unwrap(), right.design_id().unwrap());
    }

    #[test]
    fn synthesis_requires_distinct_parents() {
        let mut subject = base_subject();
        let parent = RobotDesignId(digest(10));
        subject.lineage = RobotDesignLineageV1::Synthesis {
            parents: vec![parent, parent],
        };
        assert_eq!(
            subject.validate(),
            Err(RobotDesignError::InvalidLineage(
                "synthesis parent identities must be distinct"
            ))
        );
    }

    #[test]
    fn wrong_schema_is_rejected() {
        let mut subject = base_subject();
        subject.schema_version = 2;
        assert!(matches!(
            subject.design_id(),
            Err(RobotDesignError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn exact_parameter_identity_has_golden_cross_language_vector() {
        let width = ParameterIdV1::new("section_width").unwrap();
        let height = ParameterIdV1::new("section_height").unwrap();
        let set = ExactDesignParameterSetV1::new(vec![
            ExactDesignLengthParameterV1 { id: width, value: C0LengthUmV1::new(20_000).unwrap() },
            ExactDesignLengthParameterV1 { id: height, value: C0LengthUmV1::new(30_000).unwrap() },
        ]).unwrap();
        assert_eq!(
            set.id().unwrap().to_hex(),
            "8212598d5c517ef004c6e29fe267d420fb273016f8f8415251cbb348b52bd463"
        );
    }

    #[test]
    fn exact_parameter_identity_ignores_insertion_order() {
        let width = ParameterIdV1::new("section_width").unwrap();
        let height = ParameterIdV1::new("section_height").unwrap();
        let a = ExactDesignParameterSetV1::new(vec![
            ExactDesignLengthParameterV1 { id: width.clone(), value: C0LengthUmV1::new(20_000).unwrap() },
            ExactDesignLengthParameterV1 { id: height.clone(), value: C0LengthUmV1::new(30_000).unwrap() },
        ]).unwrap();
        let b = ExactDesignParameterSetV1::new(vec![
            ExactDesignLengthParameterV1 { id: height, value: C0LengthUmV1::new(30_000).unwrap() },
            ExactDesignLengthParameterV1 { id: width, value: C0LengthUmV1::new(20_000).unwrap() },
        ]).unwrap();
        assert_eq!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn parameter_identity_changes_for_one_micrometre() {
        let id = ParameterIdV1::new("section_width").unwrap();
        let a = ExactDesignParameterSetV1::new(vec![ExactDesignLengthParameterV1 { id: id.clone(), value: C0LengthUmV1::new(20_000).unwrap() }]).unwrap();
        let b = ExactDesignParameterSetV1::new(vec![ExactDesignLengthParameterV1 { id, value: C0LengthUmV1::new(20_001).unwrap() }]).unwrap();
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn search_domain_identity_is_separate_from_parameter_identity() {
        let id = ParameterIdV1::new("section_width").unwrap();
        let selected = ExactDesignParameterSetV1::new(vec![ExactDesignLengthParameterV1 { id: id.clone(), value: C0LengthUmV1::new(20_000).unwrap() }]).unwrap();
        let narrow = ExactDesignSearchDomainV1::new(vec![ExactDesignParameterDomainV1 { id: id.clone(), domain: ExactLengthDomainV1::Values(vec![C0LengthUmV1::new(20_000).unwrap()]) }]).unwrap();
        let broad = ExactDesignSearchDomainV1::new(vec![ExactDesignParameterDomainV1 { id, domain: ExactLengthDomainV1::Values(vec![C0LengthUmV1::new(19_000).unwrap(), C0LengthUmV1::new(20_000).unwrap()]) }]).unwrap();
        assert_eq!(selected.parameter(&ParameterIdV1::new("section_width").unwrap()).unwrap().value_um(), 20_000);
        assert_ne!(narrow.id().unwrap(), broad.id().unwrap());
        assert_ne!(narrow.id().unwrap().digest(), selected.id().unwrap().digest());
    }

    #[test]
    fn robot_design_subject_binds_selected_parameter_set_id() {
        let id = ParameterIdV1::new("section_width").unwrap();
        let selected = ExactDesignParameterSetV1::new(vec![
            ExactDesignLengthParameterV1 { id, value: C0LengthUmV1::new(20_000).unwrap() },
        ]).unwrap();
        let parameter_set_id = selected.id().unwrap();
        let mut subject = base_subject();
        subject.parameter_set = parameter_set_id;
        assert_eq!(subject.parameter_set, parameter_set_id);
        assert_eq!(subject.parameter_set.digest(), parameter_set_id.digest());
    }

    #[test]
    fn search_domain_identity_cannot_be_substituted_for_parameter_set_identity() {
        let id = ParameterIdV1::new("section_width").unwrap();
        let domain = ExactDesignSearchDomainV1::new(vec![
            ExactDesignParameterDomainV1 {
                id,
                domain: ExactLengthDomainV1::Values(vec![C0LengthUmV1::new(20_000).unwrap()]),
            },
        ]).unwrap();
        let search_domain_id = domain.id().unwrap();
        let selected = ExactDesignParameterSetV1::new(vec![
            ExactDesignLengthParameterV1 {
                id: ParameterIdV1::new("section_width").unwrap(),
                value: C0LengthUmV1::new(20_000).unwrap(),
            },
        ]).unwrap();
        let parameter_set_id = selected.id().unwrap();
        assert_ne!(search_domain_id.digest(), parameter_set_id.digest());
        // SearchDomainId and ParameterSetId are distinct Rust types, so assignment
        // of one to the other's field is rejected at compile time.
        let _ = (search_domain_id, parameter_set_id);
    }

    #[test]
    fn exact_length_conversion_is_reduced_rational() {
        assert_eq!(C0LengthUmV1::new(20_000).unwrap().millimetres(), ExactRationalV1 { numerator: 20, denominator: 1 });
    }

    #[test]
    fn deserialized_zero_length_cannot_pass_parameter_validation() {
        let set = ExactDesignParameterSetV1 {
            schema_id: EXACT_PARAMETER_SCHEMA_ID.to_string(),
            schema_version: EXACT_PARAMETER_SCHEMA_VERSION,
            parameters: vec![ExactDesignLengthParameterV1 {
                id: ParameterIdV1::new("section_width").unwrap(),
                value: C0LengthUmV1(0),
            }],
        };
        assert_eq!(set.validate(), Err(RobotDesignError::InvalidDesignLength));
    }

    #[test]
    fn deserialized_invalid_parameter_id_cannot_pass_parameter_validation() {
        let set = ExactDesignParameterSetV1 {
            schema_id: EXACT_PARAMETER_SCHEMA_ID.to_string(),
            schema_version: EXACT_PARAMETER_SCHEMA_VERSION,
            parameters: vec![ExactDesignLengthParameterV1 {
                id: ParameterIdV1("section width".to_string()),
                value: C0LengthUmV1::new(20_000).unwrap(),
            }],
        };
        assert!(matches!(
            set.validate(),
            Err(RobotDesignError::InvalidSemanticId { kind: "parameter", .. })
        ));
    }

    #[test]
    fn deserialized_invalid_parameter_id_cannot_pass_domain_validation() {
        let domain = ExactDesignSearchDomainV1 {
            schema_id: EXACT_SEARCH_DOMAIN_SCHEMA_ID.to_string(),
            schema_version: EXACT_SEARCH_DOMAIN_SCHEMA_VERSION,
            parameters: vec![ExactDesignParameterDomainV1 {
                id: ParameterIdV1("section width".to_string()),
                domain: ExactLengthDomainV1::Values(vec![C0LengthUmV1::new(20_000).unwrap()]),
            }],
        };
        assert!(matches!(
            domain.validate(),
            Err(RobotDesignError::InvalidSemanticId { kind: "parameter", .. })
        ));
    }

    #[test]
    fn deserialized_zero_length_cannot_pass_domain_validation() {
        let domain = ExactDesignSearchDomainV1 {
            schema_id: EXACT_SEARCH_DOMAIN_SCHEMA_ID.to_string(),
            schema_version: EXACT_SEARCH_DOMAIN_SCHEMA_VERSION,
            parameters: vec![ExactDesignParameterDomainV1 {
                id: ParameterIdV1::new("section_width").unwrap(),
                domain: ExactLengthDomainV1::Values(vec![C0LengthUmV1(0)]),
            }],
        };
        assert_eq!(domain.validate(), Err(RobotDesignError::InvalidDesignLength));
    }

    #[test]
    fn domain_enumeration_is_exact_and_deterministic() {
        let id = ParameterIdV1::new("section_width").unwrap();
        let domain = ExactDesignSearchDomainV1::new(vec![ExactDesignParameterDomainV1 {
            id,
            domain: ExactLengthDomainV1::Range { lower: C0LengthUmV1::new(20_000).unwrap(), upper: C0LengthUmV1::new(22_000).unwrap(), step_um: 1_000 },
        }]).unwrap();
        let sets = domain.enumerate().unwrap();
        assert_eq!(sets.len(), 3);
        assert_eq!(sets[1].parameter(&ParameterIdV1::new("section_width").unwrap()).unwrap().value_um(), 21_000);
    }

    #[test]
    fn identity_helpers_round_trip() {
        let id = ParameterIdV1::new("section_width").unwrap();
        let set = ExactDesignParameterSetV1::new(vec![ExactDesignLengthParameterV1 { id, value: C0LengthUmV1::new(20_000).unwrap() }]).unwrap();
        let parameter_id = set.id().unwrap();
        assert_eq!(ParameterSetId::from_hex(&parameter_id.to_hex()).unwrap(), parameter_id);
        let robot_id = base_subject().design_id().unwrap();
        assert_eq!(RobotDesignId::from_hex(&robot_id.to_hex()).unwrap(), robot_id);
    }

    #[test]
    fn invalid_semantic_id_is_rejected_at_construction() {
        assert!(matches!(
            DesignComponentId::new("contains spaces"),
            Err(RobotDesignError::InvalidSemanticId {
                kind: "component",
                ..
            })
        ));
    }
}
