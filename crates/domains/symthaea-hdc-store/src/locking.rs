// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Path-stable advisory locking across atomic store-file replacement.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use crate::HdcStoreError;

/// Stable coordination path associated with a canonical store path.
pub fn store_lock_path(store_path: impl AsRef<Path>) -> PathBuf {
    let mut path = store_path.as_ref().as_os_str().to_os_string();
    path.push(".lock");
    PathBuf::from(path)
}

/// Held advisory lock on the path-stable coordination inode.
pub(crate) struct StoreLock {
    #[allow(dead_code)]
    file: File,
}

impl StoreLock {
    pub(crate) fn exclusive(store_path: &Path) -> Result<Self, HdcStoreError> {
        Self::acquire(store_path, true)
    }

    pub(crate) fn shared(store_path: &Path) -> Result<Self, HdcStoreError> {
        Self::acquire(store_path, false)
    }

    fn acquire(store_path: &Path, exclusive: bool) -> Result<Self, HdcStoreError> {
        let lock_path = store_lock_path(store_path);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)?;
        // std's File locking API (stable since 1.89): try_lock() is the
        // exclusive variant. The fs2 crate's trait methods collide with the
        // std inherent methods and made the two branches return different
        // error types — use std consistently.
        let result = if exclusive {
            file.try_lock()
        } else {
            file.try_lock_shared()
        };
        match result {
            Ok(()) => Ok(Self { file }),
            Err(std::fs::TryLockError::WouldBlock) => Err(HdcStoreError::StoreLocked {
                path: store_path.to_path_buf(),
            }),
            Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }
}
