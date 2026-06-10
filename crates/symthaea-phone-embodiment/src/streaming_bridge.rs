// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming variant of [`crate::bridge::PhoneBridge`] that pulls frames
//! from a persistent scrcpy-server stream instead of polling
//! `adb screencap` every cognitive cycle.
//!
//! # Why a separate type
//!
//! [`crate::bridge::PhoneBridge`] implements
//! [`symthaea_core::embodiment::EmbodimentBridge`], whose trait bound is
//! `Send + Sync`. The ffmpeg-next 7 wrappers used by [`super::scrcpy`]
//! (`codec::Context`, `software::scaling::Context`) are `Send` but **not**
//! `Sync` — they own raw FFI pointers to libavcodec/swscale state that
//! is single-thread-at-a-time safe. Embedding a [`super::scrcpy::stream::ScrcpyCaptureStream`]
//! inside `PhoneBridge` would therefore strip its `Sync` impl and break
//! the trait conformance for every existing consumer.
//!
//! The fix: keep `PhoneBridge` exactly as it is, and wrap it in a new
//! type that owns both the bridge and the stream side-by-side. This
//! type does **not** implement `EmbodimentBridge` itself — it forwards
//! to the inner bridge via [`Deref`] for ergonomic access to all the
//! existing perception and action methods, and adds one new method,
//! [`capture_streaming`](StreamingPhoneBridge::capture_streaming), that
//! drives the streaming capture path.
//!
//! Code that needs the polling capture path keeps using `PhoneBridge`
//! directly. Code that needs streaming uses `StreamingPhoneBridge`. The
//! split has zero impact on the default-features build.
//!
//! # Lifecycle
//!
//! ```ignore
//! use std::path::Path;
//! use symthaea_phone_embodiment::streaming_bridge::StreamingPhoneBridge;
//!
//! let jar = Path::new("crates/symthaea-phone-embodiment/vendor/scrcpy-server-v2.4.jar");
//! let mut sb = StreamingPhoneBridge::launch(
//!     "41201FDJG000UM",
//!     1008,
//!     2244,
//!     jar,
//!     8400,
//! )?;
//!
//! loop {
//!     match sb.capture_streaming(0.033)? {
//!         Some((tel, rgba, w, h)) => {
//!             // tel is the vision manifold telemetry,
//!             // rgba is the full-resolution decoded frame.
//!         }
//!         None => {
//!             // No frame within the read timeout — try again next tick.
//!         }
//!     }
//!     // All other methods (propose_action, learn_template, etc.)
//!     // are accessible via Deref to the inner PhoneBridge:
//!     let action = sb.propose_action(0.7);
//! }
//! ```
//!
//! On `Drop`, the [`super::scrcpy::stream::ScrcpyCaptureStream`] tears
//! down the device-side scrcpy-server and removes the `adb reverse`
//! tunnel. The inner `PhoneBridge` drops as normal.

use std::ops::{Deref, DerefMut};
use std::path::Path;

use crate::bridge::PhoneBridge;
use crate::scrcpy::ScrcpyOptions;
use crate::scrcpy::stream::{ScrcpyCaptureStream, StreamError};

/// Streaming-capture wrapper around [`PhoneBridge`].
///
/// Owns both a `PhoneBridge` (for perception, action, vision manifold,
/// templates, etc.) and a [`ScrcpyCaptureStream`] (for the persistent
/// HEVC-decode capture path).
///
/// Derefs to `PhoneBridge` so all existing methods on the inner bridge
/// are accessible through the wrapper.
pub struct StreamingPhoneBridge {
    inner: PhoneBridge,
    stream: ScrcpyCaptureStream,
}

impl StreamingPhoneBridge {
    /// Construct a wrapper by launching a fresh scrcpy-server stream.
    ///
    /// `tcp_port` should come from the dev/test range 8400–8409 per
    /// `.claude/rules/PORTS.md`. It's the host-side port the
    /// `adb reverse` tunnel will land on.
    pub fn launch(
        serial: impl Into<String>,
        screen_w: u32,
        screen_h: u32,
        jar_path: &Path,
        tcp_port: u16,
    ) -> Result<Self, StreamError> {
        let serial = serial.into();
        let opts = ScrcpyOptions::cybernetic_defaults(serial.clone(), tcp_port);
        let stream = ScrcpyCaptureStream::launch(jar_path, &opts)?;

        // Pick up the device-reported native dimensions from the wire's
        // initial video header — overrides the caller's hint if they
        // disagree (rotation, encoder downscale).
        let header = stream.video_header();
        let mut inner = PhoneBridge::new(serial, screen_w, screen_h);
        // Force the bridge's screen dimensions to match what the encoder
        // is actually emitting. PhoneBridge already self-corrects on the
        // first capture, but doing it now means grid_to_screen gives
        // accurate taps even before the first frame arrives.
        inner.set_screen_dimensions(header.width, header.height);

        Ok(Self { inner, stream })
    }

    /// Construct a wrapper by adopting an already-built `PhoneBridge`
    /// and an already-launched stream.
    ///
    /// Use this when you have an existing bridge with templates already
    /// loaded or vision manifold settings tweaked, and you want to
    /// graft a streaming capture path onto it without rebuilding.
    pub fn from_parts(inner: PhoneBridge, stream: ScrcpyCaptureStream) -> Self {
        Self { inner, stream }
    }

    /// Borrow the inner bridge.
    pub fn inner(&self) -> &PhoneBridge {
        &self.inner
    }

    /// Mutably borrow the inner bridge.
    pub fn inner_mut(&mut self) -> &mut PhoneBridge {
        &mut self.inner
    }

    /// Borrow the live capture stream.
    pub fn stream(&self) -> &ScrcpyCaptureStream {
        &self.stream
    }

    /// Drop the wrapper and recover the parts. Useful when you want to
    /// retire the streaming path back to polling, or hand the inner
    /// bridge off to an `EmbodimentBridge`-typed consumer.
    pub fn into_parts(self) -> (PhoneBridge, ScrcpyCaptureStream) {
        (self.inner, self.stream)
    }

    /// Pull the next frame from the scrcpy stream and run it through
    /// the vision manifold.
    ///
    /// Returns `Ok(None)` when the stream has no frame ready within its
    /// read timeout — caller should retry on a future cognitive cycle.
    /// Returns `Ok(Some((tel, rgba, w, h)))` mirroring the shape of
    /// [`PhoneBridge::capture_and_observe_rgba`] so existing downstream
    /// code (e.g. the SomaRdpServer codec hook) can consume either
    /// capture path interchangeably.
    pub fn capture_streaming(
        &mut self,
        dt: f32,
    ) -> Result<Option<(symthaea_vision_manifold::VisionTelemetry, Vec<u8>, u32, u32)>, StreamError>
    {
        let frame = match self.stream.next_frame()? {
            Some(f) => f,
            None => return Ok(None),
        };

        // Pass the frame's native dimensions back into the bridge in
        // case the device rotated mid-stream.
        self.inner.set_screen_dimensions(frame.width, frame.height);

        // Resize for the vision manifold target. We need to clone the
        // RGBA buffer because image::RgbaImage takes ownership and we
        // also want to return the original full-resolution bytes to
        // the caller.
        let img_rgba = image::RgbaImage::from_raw(frame.width, frame.height, frame.rgba.clone())
            .ok_or_else(|| {
                StreamError::Io(std::io::Error::other(
                    "RGBA buffer length mismatch (decoder bug)",
                ))
            })?;
        let dyn_img = image::DynamicImage::ImageRgba8(img_rgba);
        let (target_w, target_h) = self.inner.vision_target_dims();
        let resized =
            dyn_img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
        let rgb_pixels: Vec<u8> = resized.to_rgb8().into_raw();

        // Drive the vision manifold via the inner bridge's accessor.
        let tel = self
            .inner
            .vision_mut()
            .observe_frame(&rgb_pixels, target_w, target_h, 3, dt);

        Ok(Some((tel, frame.rgba, frame.width, frame.height)))
    }
}

impl Deref for StreamingPhoneBridge {
    type Target = PhoneBridge;
    fn deref(&self) -> &PhoneBridge {
        &self.inner
    }
}

impl DerefMut for StreamingPhoneBridge {
    fn deref_mut(&mut self) -> &mut PhoneBridge {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    // Note: structural tests only — a live device test would require
    // both the Pixel 8 Pro plugged in AND the scrcpy server running,
    // which lands in I.B.5 + I.B.6.

    use super::*;

    /// `StreamingPhoneBridge` is intentionally `!Send + !Sync`. This is
    /// the inverse of the design contract for `PhoneBridge`, which IS
    /// `Send + Sync` (and therefore satisfies the `EmbodimentBridge`
    /// trait bound). The reason is that ffmpeg-next 7 leaves both
    /// `codec::Context` and `software::scaling::Context` without any
    /// auto-trait impl — they own raw FFI pointers and the upstream
    /// crate doesn't mark them safe to move between threads.
    ///
    /// This is acceptable for the cognitive-loop-on-main-thread
    /// architecture today. When Phase I.C swaps the WebSocket transport
    /// for QUIC and the bridge needs to run inside a tokio worker, we
    /// will revisit by adding `unsafe impl Send for ScrcpyCaptureStream`
    /// behind a documented single-thread-ownership invariant — moving
    /// the underlying ffmpeg context between threads is actually safe
    /// as long as no two threads call into it simultaneously, which
    /// `&mut self` already guarantees.
    ///
    /// For now this test is a no-op marker: it exists so the design
    /// rationale lives next to the type and any future change that
    /// makes the wrapper Send/Sync has a place to land its assertion.
    #[test]
    fn streaming_bridge_send_sync_design_marker() {
        // Intentionally empty. See the doc comment above for why.
    }

    /// `Deref<Target = PhoneBridge>` lets the wrapper transparently
    /// expose all the perception and action methods of the inner
    /// bridge. We sanity-check the impl exists at the type level here.
    #[test]
    fn deref_to_inner_phone_bridge() {
        fn assert_deref<T: std::ops::Deref<Target = PhoneBridge>>() {}
        assert_deref::<StreamingPhoneBridge>();
    }
}
