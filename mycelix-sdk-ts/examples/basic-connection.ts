// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Basic Mycelix SDK usage example.
 *
 * Demonstrates connecting to a conductor, calling zome functions,
 * and checking consciousness state. Works in both real and mock mode.
 *
 * Run:
 *   npx ts-node examples/basic-connection.ts
 *
 * Or in mock mode (no conductor needed):
 *   MOCK=1 npx ts-node examples/basic-connection.ts
 */

import { MycelixClient, deriveTier, combinedScore } from "../src";
import type { ConsciousnessProfile } from "../src";

async function main() {
    const isMock = process.env.MOCK === "1";

    // Create client — mock mode for testing, real conductor for production
    const client = new MycelixClient(
        isMock
            ? {
                mock: true,
                mockHandler: async (role, zome, fnName, _input) => {
                    console.log(`[mock] ${role}/${zome}/${fnName}`);
                    // Return mock data based on the function name
                    if (fnName === "list_properties") {
                        return [
                            { id: "prop-1", name: "Community Garden", status: "Active" },
                            { id: "prop-2", name: "Workshop Space", status: "Available" },
                        ];
                    }
                    if (fnName === "get_consciousness_credential") {
                        return {
                            identity: 0.6,
                            reputation: 0.5,
                            community: 0.7,
                            engagement: 0.4,
                        } as ConsciousnessProfile;
                    }
                    return null;
                },
            }
            : {
                conductor: {
                    url: "ws://localhost:8300",
                    installedAppId: "mycelix-unified",
                },
            },
    );

    // Connect (no-op in mock mode)
    if (!isMock) {
        await client.connect();
    }
    console.log(`Connection state: ${client.connectionState}`);

    // Call a zome function
    interface Property {
        id: string;
        name: string;
        status: string;
    }

    const result = await client.callZome<null, Property[]>(
        "commons",
        "property_registry",
        "list_properties",
        null,
    );

    if (result.ok && result.data) {
        console.log(`Found ${result.data.length} properties:`);
        for (const prop of result.data) {
            console.log(`  - ${prop.name} (${prop.status})`);
        }
    } else {
        console.log(`Error: ${result.error}`);
    }

    // Check consciousness profile
    const profile = await client.getConsciousnessProfile();
    if (profile) {
        const score = combinedScore(profile);
        const tier = deriveTier(profile);
        console.log(`\nConsciousness Profile:`);
        console.log(`  Identity:   ${profile.identity.toFixed(2)}`);
        console.log(`  Reputation: ${profile.reputation.toFixed(2)}`);
        console.log(`  Community:  ${profile.community.toFixed(2)}`);
        console.log(`  Engagement: ${profile.engagement.toFixed(2)}`);
        console.log(`  Combined:   ${score.toFixed(2)}`);
        console.log(`  Tier:       ${tier}`);
    }

    // Disconnect
    await client.disconnect();
    console.log("\nDone.");
}

main().catch(console.error);
