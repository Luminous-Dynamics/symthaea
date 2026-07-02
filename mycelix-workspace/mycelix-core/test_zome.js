// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Simple test of H-FL zome functions
const { AdminWebsocket, AppWebsocket } = require('@holochain/client');

async function test() {
    try {
        // Connect to conductor
        const adminWs = await AdminWebsocket.connect('ws://localhost:65000');
        
        console.log('Connected to admin interface');
        
        // Install app would go here
        // For now just test connection
        
        console.log('✅ Connection successful!');
        
        await adminWs.close();
    } catch (error) {
        console.error('Connection test failed:', error);
    }
}

test();
