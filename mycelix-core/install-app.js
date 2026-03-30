#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Script to install FL hApp via WebSocket admin interface

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

const ADMIN_PORT = 38888;
const ADMIN_URL = `ws://localhost:${ADMIN_PORT}`;

// Load the hApp bundle
const hAppPath = path.join(__dirname, 'federated-learning/happ/fl_coordinator.happ');
const hAppBytes = fs.readFileSync(hAppPath);
const hAppBase64 = hAppBytes.toString('base64');

async function installApp() {
    const ws = new WebSocket(ADMIN_URL);
    
    return new Promise((resolve, reject) => {
        ws.on('open', () => {
            console.log('Connected to conductor admin interface');
            
            // Generate agent pub key first
            const genAgentRequest = {
                id: 1,
                type: 'generate_agent_pub_key',
                data: null
            };
            
            ws.send(JSON.stringify(genAgentRequest));
        });
        
        ws.on('message', (data) => {
            const response = JSON.parse(data.toString());
            console.log('Response:', response);
            
            if (response.id === 1 && response.type === 'agent_pub_key_generated') {
                // Now install app with generated key
                const agentKey = response.data;
                console.log('Agent key generated:', agentKey);
                
                const installRequest = {
                    id: 2,
                    type: 'install_app',
                    data: {
                        app_id: 'federated-learning',
                        agent_key: agentKey,
                        bundle: {
                            manifest: {
                                manifest_version: "0",
                                name: "federated_learning",
                                description: "Federated Learning on Holochain"
                            },
                            resources: {
                                "fl_coordinator.dna": hAppBase64
                            }
                        },
                        membrane_proofs: {}
                    }
                };
                
                ws.send(JSON.stringify(installRequest));
            } else if (response.id === 2) {
                if (response.type === 'app_installed') {
                    console.log('✅ App installed successfully!');
                    console.log('App info:', response.data);
                    ws.close();
                    resolve(response.data);
                } else if (response.type === 'error') {
                    console.error('❌ Install failed:', response.data);
                    ws.close();
                    reject(new Error(response.data));
                }
            }
        });
        
        ws.on('error', (err) => {
            console.error('WebSocket error:', err);
            reject(err);
        });
        
        ws.on('close', () => {
            console.log('Connection closed');
        });
    });
}

// Run the installation
console.log('Installing Federated Learning hApp...');
installApp()
    .then((result) => {
        console.log('Installation complete:', result);
        process.exit(0);
    })
    .catch((err) => {
        console.error('Installation failed:', err);
        process.exit(1);
    });