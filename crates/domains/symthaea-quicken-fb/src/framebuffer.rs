// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use drm::Device;
/// DRM/KMS framebuffer abstraction.
///
/// Opens a DRM device, finds a connected display, creates a dumb buffer at
/// native resolution, and maps it for direct pixel access on each frame.
/// No display server required — this runs on bare metal during NixOS installation.
use drm::buffer::Buffer;
use drm::control::connector::{
    Handle as ConnectorHandle, Info as ConnectorInfo, State as ConnectorState,
};
use drm::control::crtc::Handle as CrtcHandle;
use drm::control::framebuffer::Handle as FbHandle;
use drm::control::{self, Device as ControlDevice, Mode, ResourceHandles};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::{AsFd, BorrowedFd};

/// A DRM card device wrapper implementing the drm traits.
struct Card(File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    fn open(path: &str) -> Result<Self, DrmError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| DrmError::DeviceOpen(path.to_string(), e))?;
        Ok(Card(file))
    }
}

/// Errors from framebuffer operations.
#[derive(Debug)]
pub enum DrmError {
    DeviceOpen(String, std::io::Error),
    NoConnector,
    NoMode,
    NoEncoder,
    NoCrtc,
    UnrestorableCrtcState(&'static str),
    ResourceQuery(std::io::Error),
    BufferCreate(std::io::Error),
    BufferMap(std::io::Error),
    FramebufferAdd(std::io::Error),
    ModeSetting(std::io::Error),
}

impl std::fmt::Display for DrmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceOpen(path, e) => write!(f, "cannot open DRM device {path}: {e}"),
            Self::NoConnector => write!(f, "no connected display found"),
            Self::NoMode => write!(f, "no display mode available"),
            Self::NoEncoder => write!(f, "no encoder for connector"),
            Self::NoCrtc => write!(f, "no CRTC available"),
            Self::UnrestorableCrtcState(reason) => {
                write!(f, "active CRTC state cannot be restored safely: {reason}")
            }
            Self::ResourceQuery(e) => write!(f, "DRM resource query failed: {e}"),
            Self::BufferCreate(e) => write!(f, "dumb buffer creation failed: {e}"),
            Self::BufferMap(e) => write!(f, "dumb buffer map failed: {e}"),
            Self::FramebufferAdd(e) => write!(f, "framebuffer add failed: {e}"),
            Self::ModeSetting(e) => write!(f, "mode setting failed: {e}"),
        }
    }
}

impl std::error::Error for DrmError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestoreStatus {
    Restored,
    Failed(io::ErrorKind),
}

/// Opaque evidence minted only by `DrmFramebuffer::release()` after the actual
/// display-restore ioctl has been attempted.
///
/// Callers may inspect or copy this evidence, but cannot construct a successful
/// value themselves. That keeps `restore_succeeded=true` provenance tied to the
/// DRM release boundary rather than to a freely constructible public enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayReleaseEvidence {
    restore_status: RestoreStatus,
}

impl DisplayReleaseEvidence {
    pub const fn succeeded(self) -> bool {
        matches!(self.restore_status, RestoreStatus::Restored)
    }

    pub const fn as_str(self) -> &'static str {
        match self.restore_status {
            RestoreStatus::Restored => "restored",
            RestoreStatus::Failed(_) => "restore-failed",
        }
    }

    pub const fn error_kind(self) -> Option<io::ErrorKind> {
        match self.restore_status {
            RestoreStatus::Restored => None,
            RestoreStatus::Failed(kind) => Some(kind),
        }
    }

    #[cfg(test)]
    pub(crate) const fn restored_for_test() -> Self {
        Self {
            restore_status: RestoreStatus::Restored,
        }
    }

    #[cfg(test)]
    pub(crate) const fn failed_for_test(kind: io::ErrorKind) -> Self {
        Self {
            restore_status: RestoreStatus::Failed(kind),
        }
    }
}

/// An active DRM framebuffer with mapped pixel memory.
pub struct DrmFramebuffer {
    card: Card,
    crtc: CrtcHandle,
    fb: FbHandle,
    /// Display width in pixels.
    pub width: u32,
    /// Display height in pixels.
    pub height: u32,
    /// Stride in bytes (may be > width * 4 due to alignment).
    pub stride: u32,
    /// The display mode being used.
    pub mode: Mode,
    /// Dumb buffer for cleanup and mapping.
    dumb_buffer: control::dumbbuffer::DumbBuffer,
    /// Complete original CRTC state captured before renderer takeover.
    original_crtc: control::crtc::Info,
    /// Exact connectors routed to the original CRTC before renderer takeover.
    ///
    /// Legacy SETCRTC restores connector routing as well as framebuffer/mode
    /// state. Passing an empty connector array would detach the restored CRTC
    /// from its sinks, so capture this topology before changing anything.
    original_connectors: Vec<ConnectorHandle>,
    /// True until an explicit release attempt has consumed the restoration
    /// responsibility. Abnormal destruction keeps this true so Drop remains a
    /// best-effort safety net.
    restore_pending: bool,
}

impl DrmFramebuffer {
    /// Open a DRM device at the given path (e.g., "/dev/dri/card0"),
    /// find the first connected display, set up a framebuffer.
    pub fn open(device_path: &str) -> Result<Self, DrmError> {
        let card = Card::open(device_path)?;

        // Query resources
        let res = card.resource_handles().map_err(DrmError::ResourceQuery)?;

        // Find first connected connector with a valid mode
        let (connector, mode) = Self::find_connected_display(&card, &res)?;

        // Find encoder + CRTC
        let encoder_handle = connector.current_encoder().ok_or(DrmError::NoEncoder)?;
        let encoder = card
            .get_encoder(encoder_handle)
            .map_err(DrmError::ResourceQuery)?;
        let crtc = encoder.crtc().ok_or(DrmError::NoCrtc)?;

        // Save the complete legacy KMS routing before takeover. If either the
        // CRTC state or connector topology cannot be captured, do not modeset a
        // display we cannot faithfully restore.
        let original_crtc = card.get_crtc(crtc).map_err(DrmError::ResourceQuery)?;
        let original_connectors = Self::connectors_for_crtc(&card, &res, crtc)?;

        // This branch is deliberately the active-topology path. A connector that
        // resolves to a CRTC is not enough evidence that the CRTC has an active
        // framebuffer/mode that can be restored with legacy SETCRTC. Until the
        // separately qualified cold-start path exists, fail before takeover if
        // any essential restore component is absent.
        if original_crtc.framebuffer().is_none() {
            return Err(DrmError::UnrestorableCrtcState(
                "original CRTC has no framebuffer",
            ));
        }
        if original_crtc.mode().is_none() {
            return Err(DrmError::UnrestorableCrtcState(
                "original CRTC has no active mode",
            ));
        }
        if original_connectors.is_empty() {
            return Err(DrmError::UnrestorableCrtcState(
                "original CRTC has no routed connectors",
            ));
        }

        let width = mode.size().0 as u32;
        let height = mode.size().1 as u32;

        // Create dumb buffer (32bpp XRGB8888)
        let db = card
            .create_dumb_buffer((width, height), drm::buffer::DrmFourcc::Xrgb8888, 32)
            .map_err(DrmError::BufferCreate)?;

        let stride = db.pitch();
        // Add framebuffer
        let fb = card
            .add_framebuffer(&db, 24, 32)
            .map_err(DrmError::FramebufferAdd)?;

        // Set the CRTC to display our framebuffer
        card.set_crtc(crtc, Some(fb), (0, 0), &[connector.handle()], Some(mode))
            .map_err(DrmError::ModeSetting)?;

        Ok(Self {
            card,
            crtc,
            fb,
            width,
            height,
            stride,
            mode,
            dumb_buffer: db,
            original_crtc,
            original_connectors,
            restore_pending: true,
        })
    }

    /// Capture every connector currently routed through `crtc`.
    ///
    /// This is part of the restore transaction, not renderer presentation data.
    /// We fail before takeover if any connector/encoder query needed to snapshot
    /// the current topology fails, because a partial snapshot cannot prove a
    /// faithful restore later.
    fn connectors_for_crtc(
        card: &Card,
        res: &ResourceHandles,
        crtc: CrtcHandle,
    ) -> Result<Vec<ConnectorHandle>, DrmError> {
        let mut connectors = Vec::new();
        for &handle in res.connectors() {
            let connector = card
                .get_connector(handle, false)
                .map_err(DrmError::ResourceQuery)?;
            let Some(encoder_handle) = connector.current_encoder() else {
                continue;
            };
            let encoder = card
                .get_encoder(encoder_handle)
                .map_err(DrmError::ResourceQuery)?;
            if encoder.crtc() == Some(crtc) {
                connectors.push(handle);
            }
        }
        Ok(connectors)
    }

    /// Find the first connected connector and its preferred mode.
    fn find_connected_display(
        card: &Card,
        res: &ResourceHandles,
    ) -> Result<(ConnectorInfo, Mode), DrmError> {
        for &conn_handle in res.connectors() {
            let conn = match card.get_connector(conn_handle, false) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if conn.state() != ConnectorState::Connected {
                continue;
            }
            let modes = conn.modes().to_vec();
            if modes.is_empty() {
                continue;
            }
            // Prefer the first mode (usually the preferred/native resolution)
            let mode = modes
                .iter()
                .find(|m| m.mode_type().contains(control::ModeTypeFlags::PREFERRED))
                .unwrap_or(&modes[0])
                .clone();
            return Ok((conn, mode));
        }
        Err(DrmError::NoConnector)
    }

    /// Stride in bytes.
    pub fn stride_bytes(&self) -> u32 {
        self.stride
    }

    /// Copy from a row-major u32 buffer (width*height) into the DRM dumb buffer.
    /// Maps the buffer, writes, and unmaps each frame.
    pub fn blit_from(&mut self, src: &[u32]) {
        let Ok(mut mapping) = self.card.map_dumb_buffer(&mut self.dumb_buffer) else {
            return;
        };

        let stride_pixels = self.stride as usize / 4;
        let w = self.width as usize;
        let h = self.height as usize;

        // Reinterpret the u8 mapping as u32 slice
        let dst_bytes: &mut [u8] = &mut mapping;
        // SAFETY: XRGB8888 is 4-byte aligned, DRM guarantees alignment.
        let dst: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u32, dst_bytes.len() / 4)
        };

        if stride_pixels == w {
            // Fast path: no padding
            let copy_len = (w * h).min(dst.len()).min(src.len());
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
        } else {
            // Stride-aware copy
            for y in 0..h {
                let src_start = y * w;
                let dst_start = y * stride_pixels;
                let row_end = src_start + w;
                if row_end > src.len() || dst_start + w > dst.len() {
                    break;
                }
                dst[dst_start..dst_start + w].copy_from_slice(&src[src_start..row_end]);
            }
        }
        // mapping is dropped here, which flushes/unmaps
    }

    /// Request a page flip (non-blocking). Returns immediately.
    pub fn page_flip(&self) -> Result<(), DrmError> {
        // For dumb buffers with a single FB, we just do a set_crtc.
        // True page-flipping with double-buffering would require two FBs.
        // For a boot animation at ~30fps, direct writes are fine.
        Ok(())
    }

    /// Explicitly restore the captured display state and relinquish this
    /// framebuffer.
    ///
    /// The returned opaque evidence reports the actual SETCRTC result. Regardless
    /// of that result, consuming `self` causes Drop to destroy the renderer
    /// framebuffer and close the DRM fd before the caller receives the evidence.
    /// A failed restore therefore remains diagnostic-only and cannot keep
    /// presentation alive as an authority boundary.
    pub fn release(mut self) -> DisplayReleaseEvidence {
        let restore_status = match self.restore_original() {
            Ok(()) => RestoreStatus::Restored,
            Err(error) => RestoreStatus::Failed(error.kind()),
        };
        // Do not let Drop silently retry after the explicit result has been
        // observed: that would make the returned status ambiguous. Abnormal paths
        // that never call release() retain the best-effort Drop safety net.
        self.restore_pending = false;
        DisplayReleaseEvidence { restore_status }
    }

    fn restore_original(&self) -> io::Result<()> {
        self.card.set_crtc(
            self.crtc,
            self.original_crtc.framebuffer(),
            self.original_crtc.position(),
            &self.original_connectors,
            self.original_crtc.mode(),
        )
    }
}

impl Drop for DrmFramebuffer {
    fn drop(&mut self) {
        if self.restore_pending {
            // Abnormal path only: restoration remains best-effort and cannot
            // panic. Normal shutdown calls release() and reports the result.
            let _ = self.restore_original();
        }

        // Destroy framebuffer
        let _ = self.card.destroy_framebuffer(self.fb);
        // DumbBuffer is dropped automatically, which calls destroy_dumb_buffer
    }
}

// SAFETY: The dumb buffer is tied to the Card file descriptor lifetime.
// Access is single-threaded (animation loop).
unsafe impl Send for DrmFramebuffer {}
