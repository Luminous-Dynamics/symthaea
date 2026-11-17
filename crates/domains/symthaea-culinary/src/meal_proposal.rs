// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed, evidence-carrying meal proposals for the Symthaea-to-Symtropy handoff.
//!
//! A proposal expresses user intent, candidate ingredients and culinary steps.
//! It does not select appliances, claim household inventory, create a schedule,
//! issue device commands, or authorize execution. Symtropy remains responsible
//! for task-graph grounding, exact module admission and resource scheduling.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MEAL_PROPOSAL_SCHEMA_VERSION: u32 = 1;
pub const MEAL_PROPOSAL_CLAIM_CEILING: &str =
    "A user-reviewable culinary proposal with scoped source references and explicit uncertainty; not verified household inventory, nutrition/health advice, food-safety approval, equipment compatibility, a schedule, or actuation authority.";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityDimensionV1 {
    Mass,
    Volume,
    Count,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityUnitV1 {
    Milligram,
    Gram,
    Kilogram,
    Millilitre,
    Litre,
    Piece,
    Count,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExactIngredientQuantityV1 {
    pub dimension: QuantityDimensionV1,
    pub value: u64,
    pub decimal_places: u8,
    pub unit: QuantityUnitV1,
}

impl ExactIngredientQuantityV1 {
    pub fn is_valid(&self) -> bool {
        if self.decimal_places > 6 {
            return false;
        }
        matches!(
            (self.dimension, self.unit),
            (QuantityDimensionV1::Mass, QuantityUnitV1::Milligram)
                | (QuantityDimensionV1::Mass, QuantityUnitV1::Gram)
                | (QuantityDimensionV1::Mass, QuantityUnitV1::Kilogram)
                | (QuantityDimensionV1::Volume, QuantityUnitV1::Millilitre)
                | (QuantityDimensionV1::Volume, QuantityUnitV1::Litre)
                | (QuantityDimensionV1::Count, QuantityUnitV1::Piece)
                | (QuantityDimensionV1::Count, QuantityUnitV1::Count)
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalEvidenceClassV1 {
    UserSpecified,
    Observed,
    Measured,
    Published,
    VendorDeclared,
    Modelled,
    Simulated,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FactScopeV1 {
    GenericIngredient,
    ProductSku,
    SupplierLot,
    RecipeProcess,
    MarketBasket,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CulinaryFactKindV1 {
    Price,
    Nutrient,
    IngredientIdentity,
    Package,
    AllergenDeclaration,
    Availability,
    Quality,
    ProcessProperty,
    Other,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CulinaryFactV1 {
    pub fact_id: String,
    pub ingredient_id: Option<String>,
    pub kind: CulinaryFactKindV1,
    pub scope: FactScopeV1,
    pub value: serde_json::Value,
    pub unit: Option<String>,
    pub basis: Option<String>,
    pub source_ref: String,
    pub source_record_id: String,
    pub source_version: String,
    pub evidence_class: ProposalEvidenceClassV1,
    pub observed_at_epoch_ms: u64,
    pub valid_until_epoch_ms: Option<u64>,
    pub market_ref: Option<String>,
    pub license_ref: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IngredientQuantityStateV1 {
    Explicit,
    Unknown,
    NotRequired,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IngredientFoodStateV1 {
    Raw,
    Cooked,
    Dried,
    Frozen,
    Drained,
    Rehydrated,
    Fermented,
    Other,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProposalIngredientV1 {
    pub ingredient_id: String,
    pub display_name: String,
    pub quantity_state: IngredientQuantityStateV1,
    pub quantity: Option<ExactIngredientQuantityV1>,
    pub food_state: IngredientFoodStateV1,
    /// A canonical lot reference is optional and must never be inferred from
    /// an ingredient name, supplier listing or generic product catalogue row.
    pub product_lot_ref: Option<String>,
    pub fact_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProposedTaskV1 {
    pub task_id: String,
    /// Natural-language description for later grounding by Symtropy's action
    /// ontology. It is not an executable device instruction.
    pub description: String,
    pub action_label: Option<String>,
    pub ingredient_refs: Vec<String>,
    pub depends_on: Vec<String>,
    pub estimated_duration_min_ms: Option<u64>,
    pub estimated_duration_max_ms: Option<u64>,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UserMealConstraintsV1 {
    pub servings: u32,
    pub time_budget_ms: Option<u64>,
    pub budget_minor_units: Option<u64>,
    pub currency_code: Option<String>,
    pub preferred_cuisine_tags: Vec<String>,
    /// User-authored ingredient exclusions or preferences; not a diagnosis,
    /// allergy determination, or personalized medical recommendation.
    pub user_declared_exclusions: Vec<String>,
    pub user_declared_preferences: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalReviewStateV1 {
    Proposed,
    HumanReviewed,
    Accepted,
    Rejected,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyImpactV1 {
    PlanningBlocker,
    CandidateScoringOnly,
    Informational,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProposalUncertaintyV1 {
    pub field_path: String,
    pub reason: String,
    pub impact: UncertaintyImpactV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CulinaryProposalSourceV1 {
    pub source_ref: String,
    pub content_digest_sha256: Option<String>,
    pub license_ref: Option<String>,
    pub attribution: Option<String>,
    pub evidence_class: ProposalEvidenceClassV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MealProposalV1 {
    pub schema_version: u32,
    pub proposal_id: String,
    pub revision: u32,
    pub title: String,
    pub source: CulinaryProposalSourceV1,
    pub constraints: UserMealConstraintsV1,
    pub ingredients: Vec<ProposalIngredientV1>,
    pub tasks: Vec<ProposedTaskV1>,
    pub facts: Vec<CulinaryFactV1>,
    pub uncertainties: Vec<ProposalUncertaintyV1>,
    pub review_state: ProposalReviewStateV1,
    pub claim_ceiling: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MealProposalViolationCodeV1 {
    UnsupportedSchemaVersion,
    MissingRequiredValue,
    InvalidRevision,
    InvalidSourceDigest,
    InvalidQuantity,
    QuantityStateMismatch,
    MissingUnknownReason,
    DuplicateIngredientId,
    DuplicateTaskId,
    UnknownIngredientReference,
    UnknownTaskDependency,
    SelfDependency,
    DuplicateTaskDependency,
    InvalidDurationWindow,
    DuplicateFactId,
    UnknownFactIngredient,
    MissingFactSource,
    InvalidFactTimeWindow,
    UnknownFactReference,
    DuplicateUncertainty,
    ClaimCeilingMismatch,
    HardwareBindingForbidden,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MealProposalViolationV1 {
    pub code: MealProposalViolationCodeV1,
    pub reference: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MealProposalValidationReportV1 {
    pub proposal_id: String,
    pub violations: Vec<MealProposalViolationV1>,
}

impl MealProposalValidationReportV1 {
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn violation(
    output: &mut Vec<MealProposalViolationV1>,
    code: MealProposalViolationCodeV1,
    reference: Option<&str>,
    detail: impl Into<String>,
) {
    output.push(MealProposalViolationV1 {
        code,
        reference: reference.map(str::to_owned),
        detail: detail.into(),
    });
}

pub fn validate_meal_proposal(proposal: &MealProposalV1) -> MealProposalValidationReportV1 {
    let mut violations = Vec::new();

    if proposal.schema_version != MEAL_PROPOSAL_SCHEMA_VERSION {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::UnsupportedSchemaVersion,
            None,
            "unsupported meal proposal schema version",
        );
    }
    if !nonempty(&proposal.proposal_id) || !nonempty(&proposal.title) {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::MissingRequiredValue,
            None,
            "proposal ID and title must be non-empty",
        );
    }
    if proposal.revision == 0 {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::InvalidRevision,
            None,
            "proposal revision must be positive",
        );
    }
    if !nonempty(&proposal.source.source_ref) {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::MissingFactSource,
            Some("source"),
            "proposal source reference must be non-empty",
        );
    }
    if let Some(digest) = &proposal.source.content_digest_sha256 {
        if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::InvalidSourceDigest,
                Some("source"),
                "source digest must be 64 hexadecimal characters",
            );
        }
    }
    if proposal.constraints.servings == 0 {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::MissingRequiredValue,
            Some("constraints.servings"),
            "servings must be greater than zero",
        );
    }
    if proposal.constraints.currency_code.is_some()
        && proposal.constraints.currency_code.as_deref().is_none_or(|value| {
            value.len() != 3 || !value.bytes().all(|b| b.is_ascii_uppercase())
        })
    {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::MissingRequiredValue,
            Some("constraints.currency_code"),
            "currency code must be a three-letter uppercase code",
        );
    }

    let mut ingredient_ids = BTreeSet::new();
    for ingredient in &proposal.ingredients {
        if !nonempty(&ingredient.ingredient_id) || !nonempty(&ingredient.display_name) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::MissingRequiredValue,
                Some(&ingredient.ingredient_id),
                "ingredient ID and display name must be non-empty",
            );
        }
        if !ingredient_ids.insert(ingredient.ingredient_id.as_str()) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::DuplicateIngredientId,
                Some(&ingredient.ingredient_id),
                "ingredient ID is duplicated",
            );
        }
        match (ingredient.quantity_state, ingredient.quantity) {
            (IngredientQuantityStateV1::Explicit, Some(quantity)) if !quantity.is_valid() => {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::InvalidQuantity,
                    Some(&ingredient.ingredient_id),
                    "quantity dimension, unit and decimal scale do not agree",
                );
            }
            (IngredientQuantityStateV1::Explicit, None)
            | (IngredientQuantityStateV1::Unknown, Some(_))
            | (IngredientQuantityStateV1::NotRequired, Some(_)) => {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::QuantityStateMismatch,
                    Some(&ingredient.ingredient_id),
                    "quantity state and quantity value do not agree",
                );
            }
            _ => {}
        }
        if ingredient.quantity_state == IngredientQuantityStateV1::Unknown
            && !proposal.uncertainties.iter().any(|item| {
                item.field_path == format!("ingredients.{}.quantity", ingredient.ingredient_id)
                    && nonempty(&item.reason)
            })
        {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::MissingUnknownReason,
                Some(&ingredient.ingredient_id),
                "unknown ingredient quantity requires a matching uncertainty record",
            );
        }
        if ingredient
            .product_lot_ref
            .as_deref()
            .is_some_and(|reference| !nonempty(reference))
        {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::MissingRequiredValue,
                Some(&ingredient.ingredient_id),
                "product lot reference cannot be an empty string",
            );
        }
    }

    let mut task_ids = BTreeSet::new();
    for task in &proposal.tasks {
        if !nonempty(&task.task_id) || !nonempty(&task.description) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::MissingRequiredValue,
                Some(&task.task_id),
                "task ID and natural-language description must be non-empty",
            );
        }
        if !task_ids.insert(task.task_id.as_str()) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::DuplicateTaskId,
                Some(&task.task_id),
                "task ID is duplicated",
            );
        }
        for ingredient_ref in &task.ingredient_refs {
            if !ingredient_ids.contains(ingredient_ref.as_str()) {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::UnknownIngredientReference,
                    Some(&task.task_id),
                    format!("unknown ingredient reference: {ingredient_ref}"),
                );
            }
        }
        let mut dependencies = BTreeSet::new();
        for dependency in &task.depends_on {
            if dependency == &task.task_id {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::SelfDependency,
                    Some(&task.task_id),
                    "task cannot depend on itself",
                );
            }
            if !dependencies.insert(dependency.as_str()) {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::DuplicateTaskDependency,
                    Some(&task.task_id),
                    "task dependency is duplicated",
                );
            }
        }
        if let (Some(minimum), Some(maximum)) = (
            task.estimated_duration_min_ms,
            task.estimated_duration_max_ms,
        ) {
            if minimum > maximum {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::InvalidDurationWindow,
                    Some(&task.task_id),
                    "estimated duration minimum exceeds maximum",
                );
            }
        }
    }
    for task in &proposal.tasks {
        for dependency in &task.depends_on {
            if !task_ids.contains(dependency.as_str()) {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::UnknownTaskDependency,
                    Some(&task.task_id),
                    format!("unknown task dependency: {dependency}"),
                );
            }
        }
    }

    let mut fact_ids = BTreeSet::new();
    for fact in &proposal.facts {
        if !nonempty(&fact.fact_id) || !fact_ids.insert(fact.fact_id.as_str()) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::DuplicateFactId,
                Some(&fact.fact_id),
                "fact ID is empty or duplicated",
            );
        }
        if !nonempty(&fact.source_ref) || !nonempty(&fact.source_record_id) || !nonempty(&fact.source_version) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::MissingFactSource,
                Some(&fact.fact_id),
                "fact source, source record ID and source version are required",
            );
        }
        if fact.valid_until_epoch_ms.is_some_and(|expiry| expiry < fact.observed_at_epoch_ms) {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::InvalidFactTimeWindow,
                Some(&fact.fact_id),
                "fact validity end precedes observation time",
            );
        }
        if let Some(ingredient_id) = &fact.ingredient_id {
            if !ingredient_ids.contains(ingredient_id.as_str()) {
                violation(
                    &mut violations,
                    MealProposalViolationCodeV1::UnknownFactIngredient,
                    Some(&fact.fact_id),
                    "fact references an ingredient absent from the proposal",
                );
            }
        }
    }

    let mut uncertainty_paths = BTreeSet::new();
    for item in &proposal.uncertainties {
        if !nonempty(&item.field_path)
            || !nonempty(&item.reason)
            || !uncertainty_paths.insert(item.field_path.as_str())
        {
            violation(
                &mut violations,
                MealProposalViolationCodeV1::DuplicateUncertainty,
                Some(&item.field_path),
                "uncertainty path/reason must be non-empty and each path unique",
            );
        }
    }

    if proposal.claim_ceiling != MEAL_PROPOSAL_CLAIM_CEILING {
        violation(
            &mut violations,
            MealProposalViolationCodeV1::ClaimCeilingMismatch,
            None,
            "claim ceiling does not match the fixed proposal-only statement",
        );
    }

    violations.sort_by(|a, b| {
        (a.code, a.reference.as_deref(), a.detail.as_str()).cmp(&(
            b.code,
            b.reference.as_deref(),
            b.detail.as_str(),
        ))
    });

    MealProposalValidationReportV1 {
        proposal_id: proposal.proposal_id.clone(),
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_proposal() -> MealProposalV1 {
        MealProposalV1 {
            schema_version: MEAL_PROPOSAL_SCHEMA_VERSION,
            proposal_id: "fixture:meal-proposal-001".to_owned(),
            revision: 1,
            title: "Rice, vegetables and beans".to_owned(),
            source: CulinaryProposalSourceV1 {
                source_ref: "fixture:user-request".to_owned(),
                content_digest_sha256: None,
                license_ref: Some("user_authored".to_owned()),
                attribution: Some("User-authored intent".to_owned()),
                evidence_class: ProposalEvidenceClassV1::UserSpecified,
            },
            constraints: UserMealConstraintsV1 {
                servings: 4,
                time_budget_ms: Some(3_600_000),
                budget_minor_units: Some(20_000),
                currency_code: Some("ZAR".to_owned()),
                preferred_cuisine_tags: vec!["south_african_inspired".to_owned()],
                user_declared_exclusions: vec![],
                user_declared_preferences: vec!["mild".to_owned()],
            },
            ingredients: vec![
                ProposalIngredientV1 {
                    ingredient_id: "rice".to_owned(),
                    display_name: "Brown rice".to_owned(),
                    quantity_state: IngredientQuantityStateV1::Explicit,
                    quantity: Some(ExactIngredientQuantityV1 {
                        dimension: QuantityDimensionV1::Mass,
                        value: 300,
                        decimal_places: 0,
                        unit: QuantityUnitV1::Gram,
                    }),
                    food_state: IngredientFoodStateV1::Raw,
                    product_lot_ref: None,
                    fact_refs: vec![],
                },
                ProposalIngredientV1 {
                    ingredient_id: "beans".to_owned(),
                    display_name: "Beans".to_owned(),
                    quantity_state: IngredientQuantityStateV1::Unknown,
                    quantity: None,
                    food_state: IngredientFoodStateV1::Cooked,
                    product_lot_ref: None,
                    fact_refs: vec![],
                },
            ],
            tasks: vec![
                ProposedTaskV1 {
                    task_id: "prepare".to_owned(),
                    description: "Rinse and prepare the ingredients".to_owned(),
                    action_label: Some("prepare".to_owned()),
                    ingredient_refs: vec!["rice".to_owned(), "beans".to_owned()],
                    depends_on: vec![],
                    estimated_duration_min_ms: Some(300_000),
                    estimated_duration_max_ms: Some(600_000),
                    evidence_refs: vec!["fixture:recipe".to_owned()],
                },
                ProposedTaskV1 {
                    task_id: "cook".to_owned(),
                    description: "Cook the rice and combine with vegetables".to_owned(),
                    action_label: Some("cook".to_owned()),
                    ingredient_refs: vec!["rice".to_owned()],
                    depends_on: vec!["prepare".to_owned()],
                    estimated_duration_min_ms: Some(900_000),
                    estimated_duration_max_ms: Some(1_500_000),
                    evidence_refs: vec![],
                },
            ],
            facts: vec![],
            uncertainties: vec![ProposalUncertaintyV1 {
                field_path: "ingredients.beans.quantity".to_owned(),
                reason: "Exact household quantity has not been observed".to_owned(),
                impact: UncertaintyImpactV1::PlanningBlocker,
            }],
            review_state: ProposalReviewStateV1::Proposed,
            claim_ceiling: MEAL_PROPOSAL_CLAIM_CEILING.to_owned(),
        }
    }

    #[test]
    fn explicit_human_reviewable_proposal_validates() {
        assert!(validate_meal_proposal(&valid_proposal()).is_valid());
    }

    #[test]
    fn incompatible_quantity_dimension_and_unit_is_rejected() {
        let mut proposal = valid_proposal();
        proposal.ingredients[0].quantity = Some(ExactIngredientQuantityV1 {
            dimension: QuantityDimensionV1::Mass,
            value: 2,
            decimal_places: 0,
            unit: QuantityUnitV1::Litre,
        });
        let report = validate_meal_proposal(&proposal);
        assert!(report.violations.iter().any(|v| {
            v.code == MealProposalViolationCodeV1::InvalidQuantity
        }));
    }

    #[test]
    fn unknown_quantity_requires_a_reasoned_uncertainty() {
        let mut proposal = valid_proposal();
        proposal.uncertainties.clear();
        let report = validate_meal_proposal(&proposal);
        assert!(report.violations.iter().any(|v| {
            v.code == MealProposalViolationCodeV1::MissingUnknownReason
        }));
    }

    #[test]
    fn unknown_lot_is_not_inferred_from_ingredient_name() {
        let proposal = valid_proposal();
        assert!(proposal.ingredients.iter().all(|i| i.product_lot_ref.is_none()));
        assert!(validate_meal_proposal(&proposal).is_valid());
    }

    #[test]
    fn unknown_ingredient_and_task_dependency_are_rejected() {
        let mut proposal = valid_proposal();
        proposal.tasks[0].ingredient_refs.push("imaginary".to_owned());
        proposal.tasks[1].depends_on.push("missing-task".to_owned());
        let report = validate_meal_proposal(&proposal);
        assert!(report.violations.iter().any(|v| {
            v.code == MealProposalViolationCodeV1::UnknownIngredientReference
        }));
        assert!(report.violations.iter().any(|v| {
            v.code == MealProposalViolationCodeV1::UnknownTaskDependency
        }));
    }

    #[test]
    fn invalid_duration_window_and_fact_time_are_rejected() {
        let mut proposal = valid_proposal();
        proposal.tasks[0].estimated_duration_min_ms = Some(90);
        proposal.tasks[0].estimated_duration_max_ms = Some(20);
        proposal.facts.push(CulinaryFactV1 {
            fact_id: "fact:price-1".to_owned(),
            ingredient_id: Some("rice".to_owned()),
            kind: CulinaryFactKindV1::Price,
            scope: FactScopeV1::ProductSku,
            value: serde_json::json!({"minor_units": 2500}),
            unit: Some("ZAR".to_owned()),
            basis: Some("per_1kg_package".to_owned()),
            source_ref: "fixture:retailer".to_owned(),
            source_record_id: "sku-01".to_owned(),
            source_version: "2026-11-17".to_owned(),
            evidence_class: ProposalEvidenceClassV1::Observed,
            observed_at_epoch_ms: 200,
            valid_until_epoch_ms: Some(100),
            market_ref: Some("ZA".to_owned()),
            license_ref: None,
        });
        let report = validate_meal_proposal(&proposal);
        assert!(report.violations.iter().any(|v| {
            v.code == MealProposalViolationCodeV1::InvalidDurationWindow
        }));
        assert!(report.violations.iter().any(|v| {
            v.code == MealProposalViolationCodeV1::InvalidFactTimeWindow
        }));
    }
}
