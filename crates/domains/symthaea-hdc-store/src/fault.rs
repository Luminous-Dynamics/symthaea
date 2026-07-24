// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic one-shot failpoints for crash-boundary tests.

use crate::HdcStoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum FailPoint {
    AfterAppendEntryFlush = 1,
    AfterDeleteStatusFlush = 2,
    AfterBatchJournalSync = 3,
    AfterBatchDataFlush = 4,
    AfterBatchHeaderCommit = 5,
}

#[cfg(test)]
thread_local! {
    static ACTIVE_FAILPOINT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn arm(point: FailPoint) {
    ACTIVE_FAILPOINT.with(|active| active.set(point as u8));
}

#[cfg(test)]
pub(crate) fn clear() {
    ACTIVE_FAILPOINT.with(|active| active.set(0));
}

pub(crate) fn check(point: FailPoint) -> Result<(), HdcStoreError> {
    #[cfg(test)]
    {
        let triggered = ACTIVE_FAILPOINT.with(|active| {
            if active.get() == point as u8 {
                active.set(0);
                true
            } else {
                false
            }
        });
        if triggered {
            return Err(std::io::Error::other(format!(
                "injected crash-boundary fault at {point:?}"
            ))
            .into());
        }
    }

    #[cfg(not(test))]
    let _ = point;
    Ok(())
}

#[cfg(test)]
pub(crate) struct FailPointGuard;

#[cfg(test)]
impl FailPointGuard {
    pub(crate) fn arm(point: FailPoint) -> Self {
        arm(point);
        Self
    }
}

#[cfg(test)]
impl Drop for FailPointGuard {
    fn drop(&mut self) {
        clear();
    }
}
