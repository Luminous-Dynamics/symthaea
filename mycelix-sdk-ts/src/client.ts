// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Mycelix client for connecting to Holochain conductors.
 *
 * Provides a typed `callZome` method that works with any cluster's
 * zome functions. Supports both real conductor connections (via
 * `@holochain/client`) and mock mode for development/testing.
 */

import type { ConductorConfig, ZomeCallResult } from "./types";
import type { ConsciousnessProfile } from "./consciousness";

/**
 * Connection state of the client.
 */
export type ConnectionState = "disconnected" | "connecting" | "connected" | "error";

/**
 * Options for creating a MycelixClient.
 */
export interface MycelixClientOptions {
  /** Conductor configuration. Omit for mock mode. */
  conductor?: ConductorConfig;
  /** Enable mock mode (no real conductor connection). */
  mock?: boolean;
  /** Mock data provider for testing. */
  mockHandler?: <I, O>(role: string, zome: string, fnName: string, input: I) => Promise<O>;
}

/**
 * Generic client for interacting with Mycelix Holochain conductors.
 *
 * Usage with a real conductor:
 * ```ts
 * const client = new MycelixClient({
 *   conductor: { url: "ws://localhost:8300", installedAppId: "mycelix" },
 * });
 * await client.connect();
 * const result = await client.callZome<CreateInput, Record>("commons", "property", "create_property", input);
 * ```
 *
 * Usage in mock mode:
 * ```ts
 * const client = new MycelixClient({ mock: true });
 * ```
 */
export class MycelixClient {
  private state: ConnectionState = "disconnected";
  private options: MycelixClientOptions;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private appClient: any = null;

  constructor(options: MycelixClientOptions) {
    this.options = options;
    if (options.mock) {
      this.state = "connected";
    }
  }

  /** Current connection state. */
  get connectionState(): ConnectionState {
    return this.state;
  }

  /** Whether the client is connected (or in mock mode). */
  get isConnected(): boolean {
    return this.state === "connected";
  }

  /**
   * Connect to the Holochain conductor.
   *
   * In mock mode this is a no-op. Otherwise, attempts to connect
   * via WebSocket using `@holochain/client`.
   *
   * @throws Error if conductor config is missing and not in mock mode
   */
  async connect(): Promise<void> {
    if (this.options.mock) {
      this.state = "connected";
      return;
    }

    const config = this.options.conductor;
    if (!config) {
      throw new Error(
        "ConductorConfig required when not in mock mode. " +
        "Pass { conductor: { url, installedAppId } } or { mock: true }.",
      );
    }

    this.state = "connecting";

    try {
      // Dynamic import so @holochain/client is only needed for real connections
      const { AppWebsocket } = await import("@holochain/client");
      this.appClient = await AppWebsocket.connect({
        url: new URL(config.url),
        installed_app_id: config.installedAppId,
        token: config.token ? (config.token as unknown as Uint8Array) : undefined,
        defaultTimeout: config.timeoutMs ?? 15000,
      });
      this.state = "connected";
    } catch (err) {
      this.state = "error";
      throw new Error(`Failed to connect to conductor at ${config.url}: ${err}`);
    }
  }

  /**
   * Disconnect from the conductor.
   */
  async disconnect(): Promise<void> {
    if (this.appClient && typeof this.appClient.close === "function") {
      await this.appClient.close();
    }
    this.appClient = null;
    this.state = "disconnected";
  }

  /**
   * Call a zome function on the conductor.
   *
   * @typeParam I - Input payload type
   * @typeParam O - Output (return) type
   * @param role - The hApp role name (e.g., "commons", "civic", "health")
   * @param zome - The zome name within the role (e.g., "property", "care")
   * @param fnName - The extern function name (e.g., "create_property")
   * @param input - The input payload (will be serialized via MessagePack)
   * @returns The zome call result wrapped in ZomeCallResult
   */
  async callZome<I, O>(
    role: string,
    zome: string,
    fnName: string,
    input: I,
  ): Promise<ZomeCallResult<O>> {
    if (!this.isConnected) {
      return { ok: false, error: "Client is not connected" };
    }

    // Mock mode: delegate to handler or return empty success
    if (this.options.mock) {
      if (this.options.mockHandler) {
        try {
          const data = await this.options.mockHandler<I, O>(role, zome, fnName, input);
          return { ok: true, data };
        } catch (err) {
          return { ok: false, error: `Mock handler error: ${err}` };
        }
      }
      return { ok: true, data: undefined as unknown as O };
    }

    // Real conductor call
    try {
      const result = await this.appClient.callZome({
        role_name: role,
        zome_name: zome,
        fn_name: fnName,
        payload: input,
      });
      return { ok: true, data: result as O };
    } catch (err) {
      return { ok: false, error: `Zome call failed [${role}/${zome}/${fnName}]: ${err}` };
    }
  }

  /**
   * Fetch the caller's consciousness profile from the identity bridge.
   *
   * Convenience method that calls the standard
   * `get_consciousness_credential` function on the identity bridge.
   *
   * @param role - The role containing the bridge coordinator (defaults to "commons")
   * @param zome - The bridge zome name (defaults to "commons_bridge")
   * @returns The caller's consciousness profile, or null if unavailable
   */
  async getConsciousnessProfile(
    role = "commons",
    zome = "commons_bridge",
  ): Promise<ConsciousnessProfile | null> {
    const result = await this.callZome<null, ConsciousnessProfile>(
      role,
      zome,
      "get_consciousness_credential",
      null,
    );
    return result.ok && result.data ? result.data : null;
  }
}
