// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated identifiers shared by the legal-reasoning calculi.
//!
//! Strings remain appropriate at parsing and interchange boundaries, but the
//! kernel should not silently accept empty identifiers, surrounding whitespace,
//! or control characters as legal actors, actions, atoms, or source references.

use std::error::Error;
use std::fmt;

/// Why a legal identifier failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentifierError {
    Empty,
    SurroundingWhitespace,
    ControlCharacter,
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentifierError::Empty => f.write_str("legal identifier cannot be empty"),
            IdentifierError::SurroundingWhitespace => {
                f.write_str("legal identifier cannot contain surrounding whitespace")
            }
            IdentifierError::ControlCharacter => {
                f.write_str("legal identifier cannot contain control characters")
            }
        }
    }
}

impl Error for IdentifierError {}

fn validate_identifier(value: &str) -> Result<(), IdentifierError> {
    if value.is_empty() {
        return Err(IdentifierError::Empty);
    }
    if value.trim() != value {
        return Err(IdentifierError::SurroundingWhitespace);
    }
    if value.chars().any(char::is_control) {
        return Err(IdentifierError::ControlCharacter);
    }
    Ok(())
}

macro_rules! legal_identifier {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, IdentifierError> {
                let value = value.into();
                validate_identifier(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdentifierError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = IdentifierError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

legal_identifier!(
    /// A proposition name used by a formal rule theory.
    Atom
);
legal_identifier!(
    /// A stable identifier for a natural or legal person, office, or group.
    PartyId
);
legal_identifier!(
    /// A stable identifier for an action or legally relevant event.
    ActionId
);
legal_identifier!(
    /// A stable identifier for a rule within a rule pack.
    RuleId
);
legal_identifier!(
    /// A stable identifier for a versioned collection of formal rules.
    RulePackId
);
legal_identifier!(
    /// A stable identifier for one temporal revision of a formal object.
    RevisionId
);
legal_identifier!(
    /// A stable identifier for a legally relevant recorded event.
    EventId
);
legal_identifier!(
    /// A stable identifier for a formal query recorded in evidence.
    QueryId
);
legal_identifier!(
    /// A stable identifier for a selected inference semantic profile.
    SemanticProfileId
);
legal_identifier!(
    /// A stable identifier for a formally adjudicated issue.
    IssueId
);
legal_identifier!(
    /// A stable identifier for a claim or cause of action.
    ClaimId
);
legal_identifier!(
    /// A stable identifier for one stage in an explicit burden-shifting plan.
    BurdenStageId
);
legal_identifier!(
    /// A stable identifier for one authority or precedent record.
    AuthorityRecordId
);
legal_identifier!(
    /// A stable identifier for a requested or awarded remedy.
    RemedyId
);
legal_identifier!(
    /// A stable identifier for one monetary remedy component.
    RemedyComponentId
);
legal_identifier!(
    /// A stable identifier for a formal adjudication record.
    DecisionId
);
legal_identifier!(
    /// A stable identifier for a currency unit used by exact monetary arithmetic.
    CurrencyId
);
legal_identifier!(
    /// A stable identifier for a legal source document.
    DocumentId
);
legal_identifier!(
    /// A stable identifier for a provision within a legal source document.
    ProvisionId
);
legal_identifier!(
    /// A stable identifier for a jurisdiction.
    JurisdictionId
);
legal_identifier!(
    /// A stable identifier for an issuing or controlling authority.
    AuthorityId
);

/// An explicitly positive or negative proposition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Literal {
    Positive(Atom),
    Negative(Atom),
}

impl Literal {
    pub fn atom(&self) -> &Atom {
        match self {
            Literal::Positive(atom) | Literal::Negative(atom) => atom,
        }
    }

    pub fn is_positive(&self) -> bool {
        matches!(self, Literal::Positive(_))
    }

    pub fn opposite(&self) -> Literal {
        match self {
            Literal::Positive(atom) => Literal::Negative(atom.clone()),
            Literal::Negative(atom) => Literal::Positive(atom.clone()),
        }
    }
}

/// Traceability from a formal object back to an exact legal provision.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceRef {
    pub document: DocumentId,
    pub provision: ProvisionId,
}

impl SourceRef {
    pub fn new(document: DocumentId, provision: ProvisionId) -> Self {
        Self {
            document,
            provision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_reject_ambiguous_boundary_strings() {
        assert_eq!(Atom::new(""), Err(IdentifierError::Empty));
        assert_eq!(
            PartyId::new(" claimant"),
            Err(IdentifierError::SurroundingWhitespace)
        );
        assert_eq!(
            RuleId::new("rule\n1"),
            Err(IdentifierError::ControlCharacter)
        );
    }

    #[test]
    fn identifiers_preserve_exact_valid_spelling() {
        let action = ActionId::new("file_appeal").unwrap();
        assert_eq!(action.as_str(), "file_appeal");
        assert_eq!(action.to_string(), "file_appeal");
    }

    #[test]
    fn literal_opposition_is_an_involution() {
        let literal = Literal::Positive(Atom::new("liable").unwrap());
        assert_eq!(literal.opposite().opposite(), literal);
    }
}
