// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ultra-minimal WASM module for Mycelix demonstration
#![no_std]

#[no_mangle]
pub extern "C" fn register_agent(agent_id: i32) -> i32 {
    // Simple agent registration logic
    // Returns 1 for success
    if agent_id > 0 {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn get_agent_count() -> i32 {
    // Return a simulated count for demonstration
    42
}

#[no_mangle]
pub extern "C" fn update_reputation(agent_id: i32, score: i32) -> i32 {
    // Update agent reputation
    // Returns new reputation score
    agent_id + score
}

// Panic handler for no_std
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}