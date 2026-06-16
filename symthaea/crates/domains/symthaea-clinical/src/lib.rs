#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(clippy::dbg_macro, clippy::todo, clippy::unimplemented)]

//! # Symthaea Clinical Knowledge Ontology
//!
//! Standalone knowledge crate encoding clinical taxonomies as HDC hypervectors.
//! Covers DSM-5/ICD-11 diagnostic categories, NIMH RDoC dimensional framework,
//! therapeutic modalities (CBT/ACT/DBT/Narrative/Somatic/MI/EMDR/IFS/Psychodynamic),
//! and individual symptom encoding.
//!
//! No dependency on main symthaea crate — uses only `symthaea-core` for HDC types.
//!
//! Science: APA DSM-5 (2013), WHO ICD-11 (2019), Insel et al. (2010) RDoC,
//! Barlow (2014) unified protocol, Linehan (1993) DBT, Hayes (2006) ACT.

pub mod nosology;
pub mod rdoc;
pub mod symptom_encoding;
pub mod therapeutic_modalities;

pub use nosology::{DiagnosticCategory, DiagnosticProfile, Severity, Specifier};
pub use rdoc::{RDocDomain, RDocProfile};
pub use symptom_encoding::{SymptomEncoding, SymptomProfile};
pub use therapeutic_modalities::{
    InterventionLibrary, TherapeuticIntervention, TherapeuticModality,
};
