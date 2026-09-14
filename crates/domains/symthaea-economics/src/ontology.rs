// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Theory-neutral identifiers and state-domain vocabulary for economic science.
//!
//! These types identify *what a model talks about*. They do not assert that a
//! quantity is observed, causally identified, normatively desirable, or
//! authorized for governance use.

use crate::error::{EconomicsError, Result};

const MAX_SYMBOLIC_ID_LEN: usize = 128;

fn validate_symbolic_id(value: &str, context: &'static str) -> Result<()> {
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_SYMBOLIC_ID_LEN {
        return Err(EconomicsError::InvalidParameter { context });
    }
    if !bytes[0].is_ascii_alphanumeric() {
        return Err(EconomicsError::InvalidParameter { context });
    }
    if !bytes.iter().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(*byte, b'-' | b'_' | b'.' | b':' | b'/')
    }) {
        return Err(EconomicsError::InvalidParameter { context });
    }
    Ok(())
}

macro_rules! symbolic_id {
    ($name:ident, $context:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                validate_symbolic_id(&value, $context)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

symbolic_id!(EntityId, "economic entity id");
symbolic_id!(AccountId, "economic account id");
symbolic_id!(StockId, "economic stock id");
symbolic_id!(VariableId, "economic variable id");
symbolic_id!(UnitId, "economic unit id");
symbolic_id!(ClaimId, "economic claim id");
symbolic_id!(PredictionId, "economic prediction id");
symbolic_id!(MechanismId, "economic mechanism id");
symbolic_id!(TheoryId, "economic theory id");
symbolic_id!(ModelId, "economic model id");

/// Coarse state-space domains. A variable may be represented by one primary
/// domain in ETIR v1; cross-domain mechanisms connect domains explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateDomain {
    Physical,
    Financial,
    Network,
    Institutional,
    Cognitive,
    Ecological,
}

/// A declared economic state variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EconomicVariable {
    id: VariableId,
    domain: StateDomain,
    unit: UnitId,
    description: String,
}

impl EconomicVariable {
    pub fn new(
        id: VariableId,
        domain: StateDomain,
        unit: UnitId,
        description: impl Into<String>,
    ) -> Result<Self> {
        let description = description.into();
        if description.trim().is_empty() {
            return Err(EconomicsError::InvalidParameter {
                context: "economic variable description",
            });
        }
        Ok(Self {
            id,
            domain,
            unit,
            description,
        })
    }

    pub fn id(&self) -> &VariableId {
        &self.id
    }

    pub fn domain(&self) -> StateDomain {
        self.domain
    }

    pub fn unit(&self) -> &UnitId {
        &self.unit
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_ids_are_closed_and_stable() {
        assert!(VariableId::new("labor.unemployment_rate").is_ok());
        assert!(VariableId::new("prices:cpi/all-items").is_ok());
        assert!(VariableId::new("").is_err());
        assert!(VariableId::new(" bad").is_err());
        assert!(VariableId::new("bad id").is_err());
    }

    #[test]
    fn variable_requires_explicit_domain_unit_and_description() {
        let variable = EconomicVariable::new(
            VariableId::new("prices:cpi").unwrap(),
            StateDomain::Financial,
            UnitId::new("index:2015_100").unwrap(),
            "Consumer price index",
        )
        .unwrap();
        assert_eq!(variable.domain(), StateDomain::Financial);
        assert_eq!(variable.unit().as_str(), "index:2015_100");
        assert!(EconomicVariable::new(
            VariableId::new("x").unwrap(),
            StateDomain::Physical,
            UnitId::new("unit").unwrap(),
            "   ",
        )
        .is_err());
    }
}
