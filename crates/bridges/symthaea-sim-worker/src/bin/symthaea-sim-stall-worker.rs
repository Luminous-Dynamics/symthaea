// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deliberately non-responsive worker used only to qualify supervisor timeout.

fn main() {
    std::thread::sleep(std::time::Duration::from_secs(60));
}
