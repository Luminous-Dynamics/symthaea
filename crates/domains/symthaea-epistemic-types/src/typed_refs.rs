use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! define_ref {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, RefValidationError> {
                let value = value.into();
                if value.is_empty() { return Err(RefValidationError::Empty); }
                if value.chars().any(char::is_control) { return Err(RefValidationError::ControlCharacter); }
                Ok(Self(value))
            }
            pub fn as_str(&self) -> &str { &self.0 }
            pub fn into_inner(self) -> String { self.0 }
        }

        impl TryFrom<String> for $name {
            type Error = RefValidationError;
            fn try_from(value: String) -> Result<Self, Self::Error> { Self::new(value) }
        }

        impl TryFrom<&str> for $name {
            type Error = RefValidationError;
            fn try_from(value: &str) -> Result<Self, Self::Error> { Self::new(value) }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefValidationError { Empty, ControlCharacter }

impl fmt::Display for RefValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "reference must not be empty"),
            Self::ControlCharacter => write!(f, "reference must not contain control characters"),
        }
    }
}

impl std::error::Error for RefValidationError {}

define_ref!(CanonicalArtifactRef);
define_ref!(StatementRef);
define_ref!(ProvenanceFamilyRef);
define_ref!(FrontierRef);
define_ref!(DerivationRef);
define_ref!(ModelRef);
define_ref!(RetrievalIndexRef);
define_ref!(SourceEventRef);
define_ref!(EpistemicStateRef);
define_ref!(ClaimCeilingRef);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_values() {
        assert_eq!(CanonicalArtifactRef::try_from("").unwrap_err(), RefValidationError::Empty);
    }

    #[test]
    fn rejects_control_characters() {
        assert_eq!(FrontierRef::try_from("frontier\n1").unwrap_err(), RefValidationError::ControlCharacter);
    }

    #[test]
    fn round_trips_through_serde() {
        let reference = ModelRef::try_from("model:symthaea@1").unwrap();
        let encoded = serde_json::to_string(&reference).unwrap();
        let decoded: ModelRef = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, reference);
    }
}
