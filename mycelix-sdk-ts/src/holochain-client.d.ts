// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Type declarations for @holochain/client (peer dependency).
// Provides minimal types needed by MycelixClient.
// Install @holochain/client for full type safety.

declare module "@holochain/client" {
    export class AppWebsocket {
        static connect(options: {
            url: URL;
            installed_app_id: string;
            token?: Uint8Array;
            defaultTimeout?: number;
        }): Promise<AppWebsocket>;

        callZome(input: {
            role_name: string;
            zome_name: string;
            fn_name: string;
            payload: unknown;
        }): Promise<unknown>;

        close(): Promise<void>;
    }
}
