use crate::{canonical_token, hash_field, ManufacturingContractError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_manufacturing_process::ProcessDefinitionId;

const STATE_DOMAIN: &str = "symthaea-manufacturing-contracts::state-ref-v1";
const TRANSFORMATION_DOMAIN: &str = "symthaea-manufacturing-contracts::transformation-v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessStateClassV1 {
    MaterialState,
    GeometryState,
    SurfaceState,
    AssemblyState,
    CleanlinessState,
    ThermalHistoryState,
    InspectionState,
    Custom(String),
}

impl ProcessStateClassV1 {
    fn tag(&self) -> String {
        match self {
            Self::MaterialState => "material".into(),
            Self::GeometryState => "geometry".into(),
            Self::SurfaceState => "surface".into(),
            Self::AssemblyState => "assembly".into(),
            Self::CleanlinessState => "cleanliness".into(),
            Self::ThermalHistoryState => "thermal-history".into(),
            Self::InspectionState => "inspection".into(),
            Self::Custom(tag) => format!("custom:{tag}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProcessStateRefV1 {
    pub namespace: String,
    pub subject_id: String,
    pub semantic_version: String,
    pub state_class: ProcessStateClassV1,
}

impl ProcessStateRefV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("state.namespace", &self.namespace)?;
        canonical_token("state.subject_id", &self.subject_id)?;
        canonical_token("state.semantic_version", &self.semantic_version)?;
        if let ProcessStateClassV1::Custom(tag) = &self.state_class {
            canonical_token("state.custom_class", tag)?;
        }
        Ok(())
    }

    pub fn state_id(&self) -> Result<String, ManufacturingContractError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, STATE_DOMAIN);
        hash_field(&mut hasher, &self.namespace);
        hash_field(&mut hasher, &self.subject_id);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(&mut hasher, &self.state_class.tag());
        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Stable external coordinate. Human-readable namespace/subject fields remain in the
    /// structured object; the coordinate binds their exact tuple without delimiter ambiguity.
    pub fn coordinate(&self) -> Result<String, ManufacturingContractError> {
        Ok(format!("state:{}", self.state_id()?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessTransformationContractV1 {
    pub process_id: ProcessDefinitionId,
    pub inputs: Vec<ProcessStateRefV1>,
    pub outputs: Vec<ProcessStateRefV1>,
    #[serde(default)]
    pub preserved_refs: Vec<ProcessStateRefV1>,
}

impl ProcessTransformationContractV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("transformation.process_id", &self.process_id.0)?;
        if self.inputs.is_empty() || self.outputs.is_empty() {
            return Err(ManufacturingContractError::Invalid(
                "process transformation requires at least one input and one output state",
            ));
        }
        validate_unique_states("inputs", &self.inputs)?;
        validate_unique_states("outputs", &self.outputs)?;
        validate_unique_states("preserved_refs", &self.preserved_refs)?;
        Ok(())
    }

    pub fn contract_id(&self) -> Result<String, ManufacturingContractError> {
        self.validate()?;
        let mut inputs = self.inputs.clone();
        let mut outputs = self.outputs.clone();
        let mut preserved = self.preserved_refs.clone();
        inputs.sort();
        outputs.sort();
        preserved.sort();

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, TRANSFORMATION_DOMAIN);
        hash_field(&mut hasher, &self.process_id.0);
        for state in inputs {
            hash_field(&mut hasher, &state.state_id()?);
        }
        hash_field(&mut hasher, "outputs");
        for state in outputs {
            hash_field(&mut hasher, &state.state_id()?);
        }
        hash_field(&mut hasher, "preserved");
        for state in preserved {
            hash_field(&mut hasher, &state.state_id()?);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_unique_states(
    field: &'static str,
    values: &[ProcessStateRefV1],
) -> Result<(), ManufacturingContractError> {
    let mut seen = BTreeSet::new();
    for value in values {
        value.validate()?;
        let id = value.state_id()?;
        if !seen.insert(id.clone()) {
            return Err(ManufacturingContractError::DuplicateReference { field, value: id });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delimiter_ambiguous_human_tokens_do_not_collide() {
        let a = ProcessStateRefV1 {
            namespace: "a:b".into(),
            subject_id: "c".into(),
            semantic_version: "1".into(),
            state_class: ProcessStateClassV1::MaterialState,
        };
        let b = ProcessStateRefV1 {
            namespace: "a".into(),
            subject_id: "b:c".into(),
            semantic_version: "1".into(),
            state_class: ProcessStateClassV1::MaterialState,
        };
        assert_ne!(a.state_id().unwrap(), b.state_id().unwrap());
        assert_ne!(a.coordinate().unwrap(), b.coordinate().unwrap());
    }
}
