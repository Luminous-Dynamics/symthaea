// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Guard the boundary between the top-level application crate and symthaea-core.

use std::any::TypeId;

#[test]
fn hdc_facade_uses_core_hypervector_types() {
    assert_eq!(
        TypeId::of::<symthaea::hdc::BinaryHV>(),
        TypeId::of::<symthaea_core::hdc::binary_hv::BinaryHV>(),
        "symthaea::hdc::BinaryHV must remain a symthaea-core re-export"
    );
    assert_eq!(
        TypeId::of::<symthaea::hdc::ContinuousHV>(),
        TypeId::of::<symthaea_core::hdc::unified_hv::ContinuousHV>(),
        "symthaea::hdc::ContinuousHV must remain a symthaea-core re-export"
    );
    assert_eq!(
        TypeId::of::<symthaea::hdc::HV>(),
        TypeId::of::<symthaea_core::hdc::unified_hv::HV>(),
        "symthaea::hdc::HV must remain a symthaea-core re-export"
    );
}

#[test]
fn hdc_facade_uses_core_dimensions_and_phi_engine() {
    assert_eq!(
        symthaea::hdc::HDC_DIM,
        symthaea_core::hdc::unified_hv::HDC_DIMENSION
    );
    assert_eq!(
        symthaea::hdc::HDC_DIMENSION,
        symthaea_core::hdc::unified_hv::HDC_DIMENSION
    );
    assert_eq!(
        TypeId::of::<symthaea::hdc::PhiEngine>(),
        TypeId::of::<symthaea_core::phi_engine::PhiEngine>(),
        "symthaea::hdc::PhiEngine must remain a symthaea-core re-export"
    );
}
