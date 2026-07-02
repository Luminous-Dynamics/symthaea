// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Projects Integrity Zome
//!
//! Defines entry types and validation for climate projects.
//! Uses HDI 0.7.0-dev.1 with FlatOp validation pattern.

use hdi::prelude::*;

/// Anchor entry for creating deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Anchor(pub String);

/// Type of climate project
#[hdk_entry_helper]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProjectType {
    /// Tree planting and forest restoration
    Reforestation,
    /// Solar, wind, hydro, etc.
    RenewableEnergy,
    /// Capturing methane from landfills, farms, etc.
    MethaneCapture,
    /// Coastal and marine ecosystem restoration
    OceanRestoration,
    /// Direct air capture of CO2
    DirectAirCapture,
}

/// Status of a climate project
#[hdk_entry_helper]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProjectStatus {
    /// Initial proposal submitted
    Proposed,
    /// Verified by third party
    Verified,
    /// Actively generating credits
    Active,
    /// Project has finished its term
    Completed,
}

/// Geographic location for a project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Location {
    /// Country code (ISO 3166-1 alpha-2)
    pub country_code: String,
    /// Region/state/province
    pub region: Option<String>,
    /// Latitude in decimal degrees
    pub latitude: f64,
    /// Longitude in decimal degrees
    pub longitude: f64,
}

/// A climate project that generates carbon credits
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClimateProject {
    /// Unique project identifier
    pub id: String,
    /// Human-readable project name
    pub name: String,
    /// Type of climate project
    pub project_type: ProjectType,
    /// Geographic location
    pub location: Location,
    /// Expected total credits over project lifetime
    pub expected_credits: f64,
    /// Project start date (Unix timestamp)
    pub start_date: i64,
    /// DID of the verifying organization
    pub verifier_did: Option<String>,
    /// Current project status
    pub status: ProjectStatus,
}

/// A milestone in a climate project's lifecycle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProjectMilestone {
    /// ID of the associated project
    pub project_id: String,
    /// Milestone title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Target date (Unix timestamp)
    pub target_date: i64,
    /// Actual completion date (Unix timestamp, if completed)
    pub completed_at: Option<i64>,
    /// Credits issued upon completion
    pub credits_issued: Option<f64>,
    /// DID of the verifier who approved this milestone
    pub verified_by: Option<String>,
}

/// Entry types for the projects zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    ClimateProject(ClimateProject),
    #[entry_type(visibility = "public")]
    ProjectMilestone(ProjectMilestone),
}

/// Link types for the projects zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all projects
    AnchorToProjects,
    /// Anchor to projects by type
    TypeToProjects,
    /// Anchor to projects by status
    StatusToProjects,
    /// Project to its milestones
    ProjectToMilestones,
    /// Project updates chain
    ProjectUpdates,
    /// Verifier to projects they verified
    VerifierToProjects,
}

/// Validate DIDs have proper format
fn validate_did(did: &str) -> ExternResult<ValidateCallbackResult> {
    if did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("DID cannot be empty".to_string()));
    }
    if !did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:' prefix".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate a Location
fn validate_location(location: &Location) -> ExternResult<ValidateCallbackResult> {
    // Validate country code (simple check for 2 uppercase letters)
    if location.country_code.len() != 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Country code must be 2 characters (ISO 3166-1 alpha-2)".to_string(),
        ));
    }

    // Validate latitude
    if location.latitude < -90.0 || location.latitude > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".to_string(),
        ));
    }

    // Validate longitude
    if location.longitude < -180.0 || location.longitude > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a ClimateProject entry
fn validate_climate_project(project: &ClimateProject) -> ExternResult<ValidateCallbackResult> {
    // Validate project ID
    if project.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Project ID cannot be empty".to_string(),
        ));
    }

    // Validate project name
    if project.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Project name cannot be empty".to_string(),
        ));
    }

    // Validate location
    let location_result = validate_location(&project.location)?;
    if let ValidateCallbackResult::Invalid(_) = location_result {
        return Ok(location_result);
    }

    // Validate expected credits are non-negative
    if project.expected_credits < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Expected credits cannot be negative".to_string(),
        ));
    }

    // Validate verifier DID if present
    if let Some(ref verifier) = project.verifier_did {
        let verifier_result = validate_did(verifier)?;
        if let ValidateCallbackResult::Invalid(_) = verifier_result {
            return Ok(verifier_result);
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a ProjectMilestone entry
fn validate_milestone(milestone: &ProjectMilestone) -> ExternResult<ValidateCallbackResult> {
    // Validate project ID
    if milestone.project_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Project ID cannot be empty".to_string(),
        ));
    }

    // Validate title
    if milestone.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Milestone title cannot be empty".to_string(),
        ));
    }

    // Validate credits issued are non-negative if present
    if let Some(credits) = milestone.credits_issued {
        if credits < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Credits issued cannot be negative".to_string(),
            ));
        }
    }

    // Validate verifier DID if present
    if let Some(ref verifier) = milestone.verified_by {
        let verifier_result = validate_did(verifier)?;
        if let ValidateCallbackResult::Invalid(_) = verifier_result {
            return Ok(verifier_result);
        }
    }

    // Validate completed_at if present
    if let Some(completed) = milestone.completed_at {
        if completed < milestone.target_date - 31536000 {
            // More than 1 year before target
            return Ok(ValidateCallbackResult::Invalid(
                "Completion date seems unreasonably early".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::ClimateProject(project) => validate_climate_project(&project),
                    EntryTypes::ProjectMilestone(milestone) => validate_milestone(&milestone),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToProjects
            | LinkTypes::TypeToProjects
            | LinkTypes::StatusToProjects
            | LinkTypes::ProjectToMilestones
            | LinkTypes::ProjectUpdates
            | LinkTypes::VerifierToProjects => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { link_type, .. } => match link_type {
            LinkTypes::ProjectToMilestones => Ok(ValidateCallbackResult::Invalid(
                "Milestone links cannot be deleted to preserve project history".to_string(),
            )),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
