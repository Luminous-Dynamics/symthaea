// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Encrypted file transfer for Sovereign RDP sessions.
//!
//! Chunked transfer over a dedicated QUIC stream with BLAKE3 integrity
//! verification. Consciousness-gated: Tier 2+ required.

use serde::{Deserialize, Serialize};

/// Chunk size for file transfer (64 KB).
pub const FILE_CHUNK_SIZE: usize = 64 * 1024;

/// File transfer message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileTransferMessage {
    /// Sender offers a file.
    Offer {
        transfer_id: u64,
        name: String,
        size: u64,
        blake3_hash: [u8; 32],
    },
    /// Receiver accepts the offer.
    Accept { transfer_id: u64 },
    /// Receiver rejects the offer.
    Reject { transfer_id: u64, reason: String },
    /// A chunk of file data.
    Chunk {
        transfer_id: u64,
        offset: u64,
        data: Vec<u8>,
    },
    /// Transfer complete (sender side).
    Complete { transfer_id: u64 },
    /// Integrity verification result (receiver side).
    Verified { transfer_id: u64, ok: bool },
}

/// Transfer state tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferState {
    Offered,
    Accepted,
    InProgress,
    Complete,
    Verified,
    Failed,
}

/// Tracks an in-progress file transfer.
pub struct FileTransfer {
    pub transfer_id: u64,
    pub name: String,
    pub total_size: u64,
    pub expected_hash: [u8; 32],
    pub state: TransferState,
    pub bytes_transferred: u64,
    /// Received chunks (for receiver side).
    pub buffer: Vec<u8>,
    /// Is this the sending or receiving side?
    pub is_sender: bool,
}

impl FileTransfer {
    /// Create a new outbound transfer (sender side).
    pub fn new_send(transfer_id: u64, name: String, data: &[u8]) -> (Self, [u8; 32]) {
        let hash = *blake3::hash(data).as_bytes();
        let transfer = Self {
            transfer_id,
            name,
            total_size: data.len() as u64,
            expected_hash: hash,
            state: TransferState::Offered,
            bytes_transferred: 0,
            buffer: data.to_vec(),
            is_sender: true,
        };
        (transfer, hash)
    }

    /// Create a new inbound transfer (receiver side).
    pub fn new_receive(transfer_id: u64, name: String, size: u64, hash: [u8; 32]) -> Self {
        Self {
            transfer_id,
            name,
            total_size: size,
            expected_hash: hash,
            state: TransferState::Offered,
            bytes_transferred: 0,
            buffer: Vec::with_capacity(size as usize),
            is_sender: false,
        }
    }

    /// Generate chunks for sending.
    pub fn chunks(&self) -> Vec<FileTransferMessage> {
        self.buffer
            .chunks(FILE_CHUNK_SIZE)
            .enumerate()
            .map(|(i, chunk)| FileTransferMessage::Chunk {
                transfer_id: self.transfer_id,
                offset: i as u64 * FILE_CHUNK_SIZE as u64,
                data: chunk.to_vec(),
            })
            .collect()
    }

    /// Apply a received chunk.
    pub fn apply_chunk(&mut self, offset: u64, data: &[u8]) {
        let off = offset as usize;
        if off + data.len() > self.buffer.len() {
            self.buffer.resize(off + data.len(), 0);
        }
        self.buffer[off..off + data.len()].copy_from_slice(data);
        self.bytes_transferred += data.len() as u64;
        self.state = TransferState::InProgress;
    }

    /// Verify integrity after all chunks received.
    pub fn verify(&mut self) -> bool {
        let actual = *blake3::hash(&self.buffer).as_bytes();
        let ok = actual == self.expected_hash;
        self.state = if ok {
            TransferState::Verified
        } else {
            TransferState::Failed
        };
        ok
    }

    /// Progress as fraction (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_size == 0 {
            return 1.0;
        }
        self.bytes_transferred as f32 / self.total_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_receive_roundtrip() {
        let data = b"Hello, Sovereign RDP file transfer!";
        let (sender, hash) = FileTransfer::new_send(1, "test.txt".into(), data);
        let mut receiver = FileTransfer::new_receive(1, "test.txt".into(), data.len() as u64, hash);

        // Send all chunks.
        let chunks = sender.chunks();
        for chunk in chunks {
            if let FileTransferMessage::Chunk {
                offset, data: d, ..
            } = chunk
            {
                receiver.apply_chunk(offset, &d);
            }
        }

        assert!(receiver.verify());
        assert_eq!(receiver.buffer, data);
    }

    #[test]
    fn test_integrity_failure() {
        let data = b"original data";
        let (_, hash) = FileTransfer::new_send(1, "test.txt".into(), data);
        let mut receiver = FileTransfer::new_receive(1, "test.txt".into(), data.len() as u64, hash);

        // Receive tampered data.
        receiver.apply_chunk(0, b"tampered data");
        assert!(!receiver.verify());
        assert_eq!(receiver.state, TransferState::Failed);
    }

    #[test]
    fn test_progress_tracking() {
        let data = vec![0u8; 200_000]; // 200KB = ~4 chunks
        let (sender, hash) = FileTransfer::new_send(1, "big.bin".into(), &data);
        let mut receiver = FileTransfer::new_receive(1, "big.bin".into(), data.len() as u64, hash);

        let chunks = sender.chunks();
        assert!(chunks.len() >= 3);

        // Apply first chunk.
        if let FileTransferMessage::Chunk {
            offset, data: d, ..
        } = &chunks[0]
        {
            receiver.apply_chunk(*offset, d);
        }

        let progress = receiver.progress();
        assert!(progress > 0.0 && progress < 1.0);
    }
}
