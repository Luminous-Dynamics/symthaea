// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

pub const MAX_SIGNATURE_ALGORITHM_NAME_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    Ed25519,
    MlDsa65,
    MlDsa87,
    Other(String),
}

impl SignatureAlgorithm {
    pub fn is_canonical(&self) -> bool {
        match self {
            Self::Ed25519 | Self::MlDsa65 | Self::MlDsa87 => true,
            Self::Other(name) => {
                !name.trim().is_empty()
                    && name == name.trim()
                    && name.len() <= MAX_SIGNATURE_ALGORITHM_NAME_BYTES
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetachedSignature {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}
