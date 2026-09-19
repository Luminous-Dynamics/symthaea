// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-NATIVEINGRESS-521: checked versioned C ABI for assurance-bearing Soma input.
//!
//! This module wires QUAL-FFIINPUT-520 into actual native entrypoints without
//! changing the legacy `native_ffi` ABI. Accepted requests are validated against
//! the strict platform-ingress contract before any framebuffer slice is formed or
//! any touch event is delivered to `SomaEngine`.
//!
//! Raw pointer provenance/lifetime remains an explicit unsafe caller assumption:
//! C ABIs cannot prove that a non-null address names live memory. What 521 does
//! establish is that nulls, ABI-layout mismatches, dimension/length mismatches,
//! oversized frames, malformed platform/profile metadata, and invalid touch
//! semantics fail closed before engine mutation.

use core::{mem, ptr, slice};

use symthaea_core::assurance_interaction_continuity::InputAttesterId;
use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_platform_ingress::{
    PlatformIngressKind, PlatformIngressProfile, PlatformIngressProfileId,
};
use crate::assurance_soma_interaction::{ScreenCaptureProfileId, TouchInputProfileId};
use crate::engine::SomaEngine;
use crate::touch_body::{TouchAction, TouchEvent};

/// Absolute transport ceiling for the checked native path. Qualified profiles
/// may impose a smaller bound but can never raise this implementation ceiling.
pub const NATIVE_HARD_MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeIngressStatus {
    Accepted = 0,
    NullPointer = 1,
    StructSizeMismatch = 2,
    InvalidPlatform = 3,
    ProfileRejected = 4,
    HardFrameLimitExceeded = 5,
    LengthNotRepresentable = 6,
    IngressRejected = 7,
    ReservedFieldNonZero = 8,
}

impl NativeIngressStatus {
    pub const fn code(self) -> u32 {
        self as u32
    }
}

/// C-layout profile payload. `struct_size` is mandatory so ABI drift fails closed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SomaAssuranceIngressProfileV1 {
    pub struct_size: u32,
    pub platform: u8,
    pub reserved0: [u8; 3],
    pub abi_version: u32,
    pub reserved1: u32,
    pub max_frame_bytes: u64,
    pub capture_profile_id: [u8; 32],
    pub touch_input_profile_id: [u8; 32],
}

impl SomaAssuranceIngressProfileV1 {
    pub fn expected_struct_size() -> u32 {
        mem::size_of::<Self>() as u32
    }

    fn decode(&self) -> Result<PlatformIngressProfile, NativeIngressStatus> {
        if self.struct_size != Self::expected_struct_size() {
            return Err(NativeIngressStatus::StructSizeMismatch);
        }
        if self.reserved0 != [0; 3] || self.reserved1 != 0 {
            return Err(NativeIngressStatus::ReservedFieldNonZero);
        }
        let platform = match self.platform {
            1 => PlatformIngressKind::AndroidJni,
            2 => PlatformIngressKind::IosCAbi,
            _ => return Err(NativeIngressStatus::InvalidPlatform),
        };
        let profile = PlatformIngressProfile {
            platform,
            abi_version: self.abi_version,
            capture_profile_id: ScreenCaptureProfileId(self.capture_profile_id),
            touch_input_profile_id: TouchInputProfileId(self.touch_input_profile_id),
            max_frame_bytes: self.max_frame_bytes,
        };
        profile
            .validate()
            .map_err(|_| NativeIngressStatus::ProfileRejected)?;
        if profile.max_frame_bytes > NATIVE_HARD_MAX_FRAME_BYTES {
            return Err(NativeIngressStatus::HardFrameLimitExceeded);
        }
        Ok(profile)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SomaAssuranceFrameRequestV1 {
    pub struct_size: u32,
    pub reserved0: u32,
    pub profile: SomaAssuranceIngressProfileV1,
    pub data: *const u8,
    pub data_len: u64,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub reserved1: u32,
    pub surface_id: [u8; 32],
    pub surface_generation: u64,
    pub interaction_generation: u64,
    pub frame_sequence: u64,
}

impl SomaAssuranceFrameRequestV1 {
    pub fn expected_struct_size() -> u32 {
        mem::size_of::<Self>() as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SomaAssuranceTouchRequestV1 {
    pub struct_size: u32,
    pub reserved0: u32,
    pub profile: SomaAssuranceIngressProfileV1,
    pub input_attester_id: [u8; 32],
    pub interaction_generation: u64,
    pub input_sequence: u64,
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
    pub action: u8,
    pub reserved1: [u8; 3],
    pub timestamp_ms: u64,
}

impl SomaAssuranceTouchRequestV1 {
    pub fn expected_struct_size() -> u32 {
        mem::size_of::<Self>() as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SomaAssuranceFrameResultV1 {
    pub struct_size: u32,
    pub status: u32,
    pub profile_id: [u8; 32],
    pub frame_observation_id: [u8; 32],
    pub ingress_receipt_id: [u8; 32],
    pub surprise: f32,
    pub reserved0: u32,
}

impl SomaAssuranceFrameResultV1 {
    pub fn expected_struct_size() -> u32 {
        mem::size_of::<Self>() as u32
    }

    fn rejected(status: NativeIngressStatus) -> Self {
        Self {
            struct_size: Self::expected_struct_size(),
            status: status.code(),
            profile_id: [0; 32],
            frame_observation_id: [0; 32],
            ingress_receipt_id: [0; 32],
            surprise: 0.0,
            reserved0: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SomaAssuranceTouchResultV1 {
    pub struct_size: u32,
    pub status: u32,
    pub profile_id: [u8; 32],
    pub touch_observation_id: [u8; 32],
    pub ingress_receipt_id: [u8; 32],
}

impl SomaAssuranceTouchResultV1 {
    pub fn expected_struct_size() -> u32 {
        mem::size_of::<Self>() as u32
    }

    fn rejected(status: NativeIngressStatus) -> Self {
        Self {
            struct_size: Self::expected_struct_size(),
            status: status.code(),
            profile_id: [0; 32],
            touch_observation_id: [0; 32],
            ingress_receipt_id: [0; 32],
        }
    }
}

unsafe fn write_frame_result(
    out: *mut SomaAssuranceFrameResultV1,
    value: SomaAssuranceFrameResultV1,
) {
    // SAFETY: caller contract requires `out` to be writable for one result object.
    unsafe { ptr::write_unaligned(out, value) };
}

unsafe fn write_touch_result(
    out: *mut SomaAssuranceTouchResultV1,
    value: SomaAssuranceTouchResultV1,
) {
    // SAFETY: caller contract requires `out` to be writable for one result object.
    unsafe { ptr::write_unaligned(out, value) };
}

/// Strict checked framebuffer injection.
///
/// # Safety
/// `engine`, `request`, and `out` must point to live objects for the duration of
/// the call. When metadata validation succeeds, `request.data` must point to at
/// least `request.data_len` readable bytes. The function rejects null pointers,
/// structural/semantic mismatches, and lengths that cannot be represented before
/// constructing the Rust slice.
#[cfg(feature = "screen-vision")]
#[no_mangle]
pub unsafe extern "C" fn soma_assurance_inject_frame_v1(
    engine: *mut SomaEngine,
    request: *const SomaAssuranceFrameRequestV1,
    out: *mut SomaAssuranceFrameResultV1,
) -> u32 {
    if out.is_null() {
        return NativeIngressStatus::NullPointer.code();
    }
    if engine.is_null() || request.is_null() {
        unsafe {
            write_frame_result(
                out,
                SomaAssuranceFrameResultV1::rejected(NativeIngressStatus::NullPointer),
            )
        };
        return NativeIngressStatus::NullPointer.code();
    }

    // SAFETY: caller contract requires `request` to be readable for one request.
    let request = unsafe { ptr::read_unaligned(request) };
    if request.struct_size != SomaAssuranceFrameRequestV1::expected_struct_size() {
        unsafe {
            write_frame_result(
                out,
                SomaAssuranceFrameResultV1::rejected(NativeIngressStatus::StructSizeMismatch),
            )
        };
        return NativeIngressStatus::StructSizeMismatch.code();
    }
    if request.reserved0 != 0 || request.reserved1 != 0 {
        unsafe {
            write_frame_result(
                out,
                SomaAssuranceFrameResultV1::rejected(NativeIngressStatus::ReservedFieldNonZero),
            )
        };
        return NativeIngressStatus::ReservedFieldNonZero.code();
    }

    let profile = match request.profile.decode() {
        Ok(profile) => profile,
        Err(status) => {
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };

    let expected_len = match profile.expected_rgb_len(request.width, request.height, request.channels)
    {
        Ok(len) => len,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };
    if request.data_len != expected_len || request.data.is_null() {
        let status = if request.data.is_null() {
            NativeIngressStatus::NullPointer
        } else {
            NativeIngressStatus::IngressRejected
        };
        unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
        return status.code();
    }
    let data_len = match usize::try_from(request.data_len) {
        Ok(len) => len,
        Err(_) => {
            let status = NativeIngressStatus::LengthNotRepresentable;
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };

    // SAFETY: only after exact metadata/length checks; pointer liveness remains the
    // explicit unsafe caller assumption documented above.
    let frame = unsafe { slice::from_raw_parts(request.data, data_len) };
    let (observation, receipt) = match profile.accept_frame(
        TrustedSurfaceId(request.surface_id),
        request.surface_generation,
        request.interaction_generation,
        request.frame_sequence,
        request.width,
        request.height,
        request.channels,
        frame,
    ) {
        Ok(value) => value,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };

    // Finish every identity/certificate computation before mutating Soma.
    let profile_id = match profile.profile_id() {
        Ok(id) => id,
        Err(_) => {
            let status = NativeIngressStatus::ProfileRejected;
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };
    let observation_id = match observation.observation_id() {
        Ok(id) => id,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };
    let receipt_id = match receipt.receipt_id() {
        Ok(id) => id,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_frame_result(out, SomaAssuranceFrameResultV1::rejected(status)) };
            return status.code();
        }
    };

    // SAFETY: engine was checked non-null; liveness/exclusive access remains caller contract.
    let engine = unsafe { &mut *engine };
    let perception = engine.inject_frame(frame, request.width, request.height);

    let result = SomaAssuranceFrameResultV1 {
        struct_size: SomaAssuranceFrameResultV1::expected_struct_size(),
        status: NativeIngressStatus::Accepted.code(),
        profile_id: profile_id.0,
        frame_observation_id: observation_id.0,
        ingress_receipt_id: receipt_id.0,
        surprise: perception.surprise_level,
        reserved0: 0,
    };
    unsafe { write_frame_result(out, result) };
    NativeIngressStatus::Accepted.code()
}

/// Strict checked touch injection.
///
/// # Safety
/// `engine`, `request`, and `out` must point to live objects for the duration of
/// the call. No pointer-derived payload is dereferenced beyond the request/result
/// objects themselves.
#[cfg(feature = "screen-vision")]
#[no_mangle]
pub unsafe extern "C" fn soma_assurance_touch_event_v1(
    engine: *mut SomaEngine,
    request: *const SomaAssuranceTouchRequestV1,
    out: *mut SomaAssuranceTouchResultV1,
) -> u32 {
    if out.is_null() {
        return NativeIngressStatus::NullPointer.code();
    }
    if engine.is_null() || request.is_null() {
        unsafe {
            write_touch_result(
                out,
                SomaAssuranceTouchResultV1::rejected(NativeIngressStatus::NullPointer),
            )
        };
        return NativeIngressStatus::NullPointer.code();
    }

    // SAFETY: caller contract requires `request` to be readable for one request.
    let request = unsafe { ptr::read_unaligned(request) };
    if request.struct_size != SomaAssuranceTouchRequestV1::expected_struct_size() {
        unsafe {
            write_touch_result(
                out,
                SomaAssuranceTouchResultV1::rejected(NativeIngressStatus::StructSizeMismatch),
            )
        };
        return NativeIngressStatus::StructSizeMismatch.code();
    }
    if request.reserved0 != 0 || request.reserved1 != [0; 3] {
        unsafe {
            write_touch_result(
                out,
                SomaAssuranceTouchResultV1::rejected(NativeIngressStatus::ReservedFieldNonZero),
            )
        };
        return NativeIngressStatus::ReservedFieldNonZero.code();
    }

    let profile = match request.profile.decode() {
        Ok(profile) => profile,
        Err(status) => {
            unsafe { write_touch_result(out, SomaAssuranceTouchResultV1::rejected(status)) };
            return status.code();
        }
    };

    let action = match request.action {
        0 => TouchAction::Down,
        1 => TouchAction::Move,
        2 => TouchAction::Up,
        3 => TouchAction::Cancel,
        _ => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_touch_result(out, SomaAssuranceTouchResultV1::rejected(status)) };
            return status.code();
        }
    };

    let (observation, receipt) = match profile.accept_touch(
        InputAttesterId(request.input_attester_id),
        request.interaction_generation,
        request.input_sequence,
        request.x,
        request.y,
        request.action,
        request.pressure,
        request.timestamp_ms,
    ) {
        Ok(value) => value,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_touch_result(out, SomaAssuranceTouchResultV1::rejected(status)) };
            return status.code();
        }
    };

    // Finish every identity/certificate computation before mutating Soma.
    let profile_id: PlatformIngressProfileId = match profile.profile_id() {
        Ok(id) => id,
        Err(_) => {
            let status = NativeIngressStatus::ProfileRejected;
            unsafe { write_touch_result(out, SomaAssuranceTouchResultV1::rejected(status)) };
            return status.code();
        }
    };
    let observation_id = match observation.observation_id() {
        Ok(id) => id,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_touch_result(out, SomaAssuranceTouchResultV1::rejected(status)) };
            return status.code();
        }
    };
    let receipt_id = match receipt.receipt_id() {
        Ok(id) => id,
        Err(_) => {
            let status = NativeIngressStatus::IngressRejected;
            unsafe { write_touch_result(out, SomaAssuranceTouchResultV1::rejected(status)) };
            return status.code();
        }
    };

    let event = TouchEvent {
        x: request.x,
        y: request.y,
        action,
        pressure: request.pressure,
        timestamp_ms: request.timestamp_ms,
    };
    // SAFETY: engine was checked non-null; liveness/exclusive access remains caller contract.
    unsafe { &mut *engine }.on_touch(event);

    let result = SomaAssuranceTouchResultV1 {
        struct_size: SomaAssuranceTouchResultV1::expected_struct_size(),
        status: NativeIngressStatus::Accepted.code(),
        profile_id: profile_id.0,
        touch_observation_id: observation_id.0,
        ingress_receipt_id: receipt_id.0,
    };
    unsafe { write_touch_result(out, result) };
    NativeIngressStatus::Accepted.code()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> SomaAssuranceIngressProfileV1 {
        SomaAssuranceIngressProfileV1 {
            struct_size: SomaAssuranceIngressProfileV1::expected_struct_size(),
            platform: 1,
            reserved0: [0; 3],
            abi_version: 1,
            reserved1: 0,
            max_frame_bytes: 1024,
            capture_profile_id: [1; 32],
            touch_input_profile_id: [2; 32],
        }
    }

    #[test]
    fn profile_rejects_unknown_platform() {
        let mut p = profile();
        p.platform = 9;
        assert_eq!(p.decode().unwrap_err(), NativeIngressStatus::InvalidPlatform);
    }

    #[test]
    fn profile_rejects_struct_size_drift() {
        let mut p = profile();
        p.struct_size -= 1;
        assert_eq!(p.decode().unwrap_err(), NativeIngressStatus::StructSizeMismatch);
    }

    #[test]
    fn reserved_profile_fields_are_canonical_zero() {
        let mut p = profile();
        p.reserved1 = 1;
        assert_eq!(
            p.decode().unwrap_err(),
            NativeIngressStatus::ReservedFieldNonZero
        );
    }

    #[test]
    fn hard_frame_ceiling_is_not_policy_expandable() {
        let mut p = profile();
        p.max_frame_bytes = NATIVE_HARD_MAX_FRAME_BYTES + 1;
        assert_eq!(
            p.decode().unwrap_err(),
            NativeIngressStatus::HardFrameLimitExceeded
        );
    }

    #[test]
    fn result_structs_have_nonzero_versioned_size() {
        assert!(SomaAssuranceIngressProfileV1::expected_struct_size() > 0);
        assert!(SomaAssuranceFrameRequestV1::expected_struct_size() > 0);
        assert!(SomaAssuranceTouchRequestV1::expected_struct_size() > 0);
        assert!(SomaAssuranceFrameResultV1::expected_struct_size() > 0);
        assert!(SomaAssuranceTouchResultV1::expected_struct_size() > 0);
    }
}
