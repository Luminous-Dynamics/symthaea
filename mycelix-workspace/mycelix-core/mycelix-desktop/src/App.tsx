// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { createSignal, onMount, Show, For } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

type StatusType = "disconnected" | "connecting" | "connected" | "error";

function App() {
  const [greetMsg, setGreetMsg] = createSignal("");
  const [name, setName] = createSignal("World");
  const [status, setStatus] = createSignal("Disconnected");
  const [holochainStatus, setHolochainStatus] = createSignal("Not Started");
  const [isTauriMode, setIsTauriMode] = createSignal(false);
  const [isLoading, setIsLoading] = createSignal(false);
  const [connectionStatus, setConnectionStatus] = createSignal<StatusType>("disconnected");
  const [holochainState, setHolochainState] = createSignal<StatusType>("disconnected");
  const [peerCount, setPeerCount] = createSignal(0);
  const [notifications, setNotifications] = createSignal<Array<{id: number, message: string, type: string}>>([]);
  const [installedApps, setInstalledApps] = createSignal<Array<any>>([]);
  const [cells, setCells] = createSignal<Array<any>>([]);
  const [showAppManagement, setShowAppManagement] = createSignal(false);
  const [testResults, setTestResults] = createSignal<Record<string, string>>({});
  const [echoInput, setEchoInput] = createSignal("");

  // Message broadcasting state
  const [messages, setMessages] = createSignal<Array<any>>([]);
  const [messageInput, setMessageInput] = createSignal("");
  const [isSending, setIsSending] = createSignal(false);

  // Add notification helper
  const addNotification = (message: string, type: string = "info") => {
    const id = Date.now();
    setNotifications([...notifications(), { id, message, type }]);
    setTimeout(() => {
      setNotifications(notifications().filter(n => n.id !== id));
    }, 5000);
  };

  // Check if running in Tauri context
  onMount(() => {
    // Debug: Log what we can see
    console.log("🔍 Checking for Tauri runtime...");
    console.log("window.__TAURI__:", (window as any).__TAURI__);
    console.log("window.location:", window.location);
    console.log("All window keys:", Object.keys(window));

    // Try multiple times to detect Tauri (in case it loads late)
    let attempts = 0;
    const maxAttempts = 10;

    const checkTauri = () => {
      const tauri = (window as any).__TAURI__;
      console.log(`Attempt ${attempts + 1}/${maxAttempts}: Tauri found?`, !!tauri);

      if (tauri) {
        setIsTauriMode(true);
        greet();
        setStatus("Ready (Tauri Mode)");
        setConnectionStatus("connected");
        addNotification("Tauri runtime active - All features available", "success");
        return true;
      }

      attempts++;
      if (attempts < maxAttempts) {
        setTimeout(checkTauri, 100); // Try again in 100ms
        return false;
      } else {
        // Failed to detect after all attempts
        console.error("❌ Failed to detect Tauri runtime after", maxAttempts, "attempts");
        setIsTauriMode(false);
        setGreetMsg("⚠️ Browser-only mode - Tauri commands unavailable");
        setStatus("Browser-only mode");
        setHolochainStatus("Requires Tauri runtime");
        setHolochainState("error");
        addNotification("Running in browser mode - Tauri features unavailable", "warning");
        return false;
      }
    };

    checkTauri();
  });

  async function greet() {
    if (!isTauriMode()) {
      setGreetMsg("❌ Tauri not available - please run via 'npm run tauri dev'");
      addNotification("Tauri runtime required for this action", "error");
      return;
    }

    setIsLoading(true);
    try {
      const msg = await invoke<string>("greet", { name: name() });
      setGreetMsg(msg);
      addNotification(`Greeted ${name()}!`, "success");
    } catch (error) {
      setGreetMsg(`Error: ${error}`);
      addNotification("Failed to send greeting", "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function startHolochain() {
    if (!isTauriMode()) {
      setHolochainStatus("❌ Requires Tauri runtime");
      addNotification("Tauri runtime required for Holochain", "error");
      return;
    }

    setIsLoading(true);
    setHolochainState("connecting");
    setHolochainStatus("Starting Holochain conductor...");
    addNotification("Starting Holochain conductor...", "info");

    try {
      const result = await invoke<string>("start_holochain");
      setHolochainStatus(result);
      setHolochainState("connected");
      addNotification("Holochain conductor started", "success");
    } catch (error) {
      setHolochainStatus(`Error: ${error}`);
      setHolochainState("error");
      addNotification("Failed to start Holochain", "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function connectNetwork() {
    if (!isTauriMode()) {
      setStatus("❌ Requires Tauri runtime");
      addNotification("Tauri runtime required for P2P connection", "error");
      return;
    }

    setIsLoading(true);
    setConnectionStatus("connecting");
    setStatus("Connecting to P2P network...");
    addNotification("Connecting to P2P network...", "info");

    try {
      const result = await invoke<string>("connect_to_network");
      setStatus(result);
      setConnectionStatus("connected");
      setPeerCount(Math.floor(Math.random() * 5) + 1); // Mock peer count
      addNotification("Connected to P2P network", "success");
    } catch (error) {
      setStatus(`Error: ${error}`);
      setConnectionStatus("error");
      addNotification("Failed to connect to network", "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function loadInstalledApps() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    try {
      const result = await invoke<string>("get_installed_apps");
      const parsed = JSON.parse(result);
      setInstalledApps(Array.isArray(parsed) ? parsed : []);
      addNotification("Loaded installed apps", "success");
    } catch (error) {
      addNotification(`Failed to load apps: ${error}`, "error");
    }
  }

  async function loadCells() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    try {
      const result = await invoke<string>("get_cells");
      const parsed = JSON.parse(result);
      setCells(Array.isArray(parsed) ? parsed : []);
      addNotification("Loaded cells", "success");
    } catch (error) {
      addNotification(`Failed to load cells: ${error}`, "error");
    }
  }

  async function toggleAppStatus(appId: string, currentlyEnabled: boolean) {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    setIsLoading(true);
    try {
      if (currentlyEnabled) {
        await invoke<string>("disable_app", { appId });
        addNotification(`Disabled app: ${appId}`, "success");
      } else {
        await invoke<string>("enable_app", { appId });
        addNotification(`Enabled app: ${appId}`, "success");
      }
      await loadInstalledApps();
    } catch (error) {
      addNotification(`Failed to toggle app: ${error}`, "error");
    } finally {
      setIsLoading(false);
    }
  }

  const getStatusColor = (status: StatusType) => {
    switch (status) {
      case "connected": return "var(--color-success)";
      case "connecting": return "var(--color-warning)";
      case "error": return "var(--color-error)";
      default: return "var(--color-text-dim)";
    }
  };

  const getStatusIcon = (status: StatusType) => {
    switch (status) {
      case "connected": return "🟢";
      case "connecting": return "🟡";
      case "error": return "🔴";
      default: return "⚪";
    }
  };

  // Zome function tests
  async function testHello() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    setIsLoading(true);
    try {
      const result = await invoke<string>("call_hello");
      setTestResults({ ...testResults(), hello: result });
      addNotification("Hello function called successfully", "success");
    } catch (error) {
      setTestResults({ ...testResults(), hello: `Error: ${error}` });
      addNotification(`Failed to call hello: ${error}`, "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function testWhoami() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    setIsLoading(true);
    try {
      const result = await invoke<string>("call_whoami");
      setTestResults({ ...testResults(), whoami: result });
      addNotification("Whoami function called successfully", "success");
    } catch (error) {
      setTestResults({ ...testResults(), whoami: `Error: ${error}` });
      addNotification(`Failed to call whoami: ${error}`, "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function testEcho() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    if (!echoInput().trim()) {
      addNotification("Please enter text to echo", "warning");
      return;
    }

    setIsLoading(true);
    try {
      const result = await invoke<string>("call_echo", { input: echoInput() });
      setTestResults({ ...testResults(), echo: result });
      addNotification("Echo function called successfully", "success");
    } catch (error) {
      setTestResults({ ...testResults(), echo: `Error: ${error}` });
      addNotification(`Failed to call echo: ${error}`, "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function testGetAgentInfo() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    setIsLoading(true);
    try {
      const result = await invoke<string>("call_get_agent_info");
      setTestResults({ ...testResults(), agentInfo: result });
      addNotification("Get agent info called successfully", "success");
    } catch (error) {
      setTestResults({ ...testResults(), agentInfo: `Error: ${error}` });
      addNotification(`Failed to get agent info: ${error}`, "error");
    } finally {
      setIsLoading(false);
    }
  }

  // Message broadcasting functions
  async function sendMessage() {
    if (!isTauriMode()) {
      addNotification("Tauri runtime required", "error");
      return;
    }

    if (!messageInput().trim()) {
      addNotification("Please enter a message", "warning");
      return;
    }

    setIsSending(true);
    try {
      const result = await invoke<string>("send_message", { content: messageInput() });
      addNotification("Message sent!", "success");
      setMessageInput("");

      // Refresh messages after sending
      setTimeout(() => fetchMessages(), 1000);
    } catch (error) {
      addNotification(`Failed to send message: ${error}`, "error");
    } finally {
      setIsSending(false);
    }
  }

  async function fetchMessages() {
    if (!isTauriMode()) {
      return;
    }

    try {
      const result = await invoke<string>("get_messages");
      const parsed = JSON.parse(result);
      setMessages(Array.isArray(parsed) ? parsed : []);
    } catch (error) {
      console.error("Failed to fetch messages:", error);
    }
  }

  // Auto-refresh messages every 5 seconds when connected
  onMount(() => {
    const interval = setInterval(() => {
      if (isTauriMode() && holochainState() === "connected") {
        fetchMessages();
      }
    }, 5000);

    return () => clearInterval(interval);
  });

  return (
    <div class="container">
      {/* Toast Notifications */}
      <div class="toast-container">
        <For each={notifications()}>
          {(notif) => (
            <div class={`toast toast-${notif.type}`}>
              {notif.message}
            </div>
          )}
        </For>
      </div>

      {/* Status Bar */}
      <div class="status-bar">
        <div class="status-bar-item">
          <span class="status-icon">{getStatusIcon(connectionStatus())}</span>
          <span>Network: {connectionStatus()}</span>
        </div>
        <div class="status-bar-item">
          <span class="status-icon">{getStatusIcon(holochainState())}</span>
          <span>Holochain: {holochainState()}</span>
        </div>
        <Show when={peerCount() > 0}>
          <div class="status-bar-item">
            <span class="status-icon">👥</span>
            <span>{peerCount()} peers</span>
          </div>
        </Show>
        <div class="status-bar-item">
          <span class="status-icon">{isTauriMode() ? "⚡" : "🌐"}</span>
          <span>{isTauriMode() ? "Tauri" : "Browser"}</span>
        </div>
      </div>

      <header>
        <div class="logo-container">
          <div class="mycelial-network"></div>
          <h1>🍄 Mycelix Desktop</h1>
        </div>
        <p class="subtitle">P2P Consciousness Network</p>
        <Show when={!isTauriMode()}>
          <div class="warning-banner">
            ⚠️ Browser-only mode - Running via Vite dev server<br/>
            For full functionality, run: <code>GDK_BACKEND=x11 npm run tauri dev</code>
          </div>
        </Show>
        <Show when={isTauriMode()}>
          <div class="success-banner">
            ✅ Tauri mode active - All features available
          </div>
        </Show>
      </header>

      <main>
        {/* Welcome Card */}
        <div class="card card-interactive">
          <div class="card-header">
            <div class="card-icon">👋</div>
            <h2>Welcome</h2>
          </div>
          <div class="card-content">
            <div class="input-group">
              <input
                id="name-input"
                placeholder="Enter your name..."
                value={name()}
                onInput={(e) => setName(e.currentTarget.value)}
                disabled={isLoading()}
              />
              <button
                onClick={greet}
                class="btn-primary"
                disabled={isLoading() || !name().trim()}
              >
                <Show when={isLoading()} fallback="Greet">
                  <span class="spinner"></span> Greeting...
                </Show>
              </button>
            </div>
            <Show when={greetMsg()}>
              <div class="greeting-message">
                <div class="message-bubble">{greetMsg()}</div>
              </div>
            </Show>
          </div>
        </div>

        {/* Network Status Card */}
        <div class="card card-interactive">
          <div class="card-header">
            <div class="card-icon">🌐</div>
            <h2>Network Status</h2>
          </div>
          <div class="card-content">
            <div class="status-grid">
              <div class="status-item-enhanced">
                <div class="status-item-header">
                  <span class="status-dot" style={{ "background-color": getStatusColor(holochainState()) }}></span>
                  <span class="status-label">Holochain Conductor</span>
                </div>
                <span class="status-value">{holochainStatus()}</span>
                <Show when={holochainState() === "connecting"}>
                  <div class="progress-bar">
                    <div class="progress-fill"></div>
                  </div>
                </Show>
              </div>
              <div class="status-item-enhanced">
                <div class="status-item-header">
                  <span class="status-dot" style={{ "background-color": getStatusColor(connectionStatus()) }}></span>
                  <span class="status-label">P2P Network</span>
                </div>
                <span class="status-value">{status()}</span>
                <Show when={connectionStatus() === "connecting"}>
                  <div class="progress-bar">
                    <div class="progress-fill"></div>
                  </div>
                </Show>
              </div>
            </div>
            <div class="button-group">
              <button
                onClick={startHolochain}
                class="btn-primary"
                disabled={isLoading() || holochainState() === "connected"}
              >
                <span class="btn-icon">🚀</span>
                <Show when={holochainState() === "connecting"} fallback="Start Holochain">
                  <span class="spinner"></span> Starting...
                </Show>
              </button>
              <button
                onClick={connectNetwork}
                class="btn-secondary"
                disabled={isLoading() || connectionStatus() === "connected"}
              >
                <span class="btn-icon">🔗</span>
                <Show when={connectionStatus() === "connecting"} fallback="Connect to Network">
                  <span class="spinner"></span> Connecting...
                </Show>
              </button>
            </div>
          </div>
        </div>

        {/* App Management Card */}
        <Show when={holochainState() === "connected"}>
          <div class="card card-interactive">
            <div class="card-header">
              <div class="card-icon">🧬</div>
              <h2>App Management</h2>
            </div>
            <div class="card-content">
              <div class="button-group">
                <button
                  onClick={loadInstalledApps}
                  class="btn-secondary"
                  disabled={isLoading()}
                >
                  <span class="btn-icon">📦</span>
                  Load Apps
                </button>
                <button
                  onClick={loadCells}
                  class="btn-secondary"
                  disabled={isLoading()}
                >
                  <span class="btn-icon">🔬</span>
                  Load Cells
                </button>
              </div>

              {/* Installed Apps List */}
              <Show when={installedApps().length > 0}>
                <div class="status-grid" style={{ "margin-top": "1rem" }}>
                  <h3 style={{ "grid-column": "1 / -1", "margin": "0.5rem 0" }}>
                    Installed Apps ({installedApps().length})
                  </h3>
                  <For each={installedApps()}>
                    {(app: any) => (
                      <div class="status-item-enhanced">
                        <div class="status-item-header">
                          <span class="status-dot" style={{
                            "background-color": app.status === "running" ? "var(--color-success)" : "var(--color-text-dim)"
                          }}></span>
                          <span class="status-label">{app.installed_app_id || "Unknown App"}</span>
                        </div>
                        <span class="status-value" style={{ "font-size": "0.85rem" }}>
                          Status: {app.status || "unknown"}
                        </span>
                        <button
                          onClick={() => toggleAppStatus(app.installed_app_id, app.status === "running")}
                          class="btn-secondary"
                          style={{ "margin-top": "0.5rem", "padding": "0.4rem 0.8rem", "font-size": "0.85rem" }}
                          disabled={isLoading()}
                        >
                          {app.status === "running" ? "Disable" : "Enable"}
                        </button>
                      </div>
                    )}
                  </For>
                </div>
              </Show>

              {/* Cells List */}
              <Show when={cells().length > 0}>
                <div class="status-grid" style={{ "margin-top": "1rem" }}>
                  <h3 style={{ "grid-column": "1 / -1", "margin": "0.5rem 0" }}>
                    Active Cells ({cells().length})
                  </h3>
                  <For each={cells()}>
                    {(cell: any) => (
                      <div class="status-item-enhanced">
                        <div class="status-item-header">
                          <span class="status-dot" style={{ "background-color": "var(--color-success)" }}></span>
                          <span class="status-label" style={{ "font-size": "0.85rem" }}>
                            {cell.cell_id ? String(cell.cell_id).substring(0, 16) + "..." : "Cell"}
                          </span>
                        </div>
                        <span class="status-value" style={{ "font-size": "0.75rem", "word-break": "break-all" }}>
                          DNA: {cell.dna_hash ? String(cell.dna_hash).substring(0, 20) + "..." : "N/A"}
                        </span>
                      </div>
                    )}
                  </For>
                </div>
              </Show>

              <Show when={installedApps().length === 0 && cells().length === 0}>
                <p style={{ "text-align": "center", "color": "var(--color-text-dim)", "margin-top": "1rem" }}>
                  Click "Load Apps" or "Load Cells" to view installed applications
                </p>
              </Show>
            </div>
          </div>
        </Show>

        {/* Test Functions Card */}
        <Show when={holochainState() === "connected"}>
          <div class="card card-interactive">
            <div class="card-header">
              <div class="card-icon">🧪</div>
              <h2>Test Functions</h2>
            </div>
            <div class="card-content">
              <p style={{ "margin-bottom": "1rem", "color": "var(--color-text-dim)" }}>
                Test the 4 zome functions from the hello zome
              </p>

              {/* Test Buttons Grid */}
              <div class="button-group" style={{ "display": "grid", "grid-template-columns": "repeat(2, 1fr)", "gap": "0.75rem" }}>
                <button
                  onClick={testHello}
                  class="btn-primary"
                  disabled={isLoading()}
                >
                  <span class="btn-icon">👋</span>
                  Test hello()
                </button>

                <button
                  onClick={testWhoami}
                  class="btn-primary"
                  disabled={isLoading()}
                >
                  <span class="btn-icon">👤</span>
                  Test whoami()
                </button>

                <button
                  onClick={testGetAgentInfo}
                  class="btn-primary"
                  disabled={isLoading()}
                  style={{ "grid-column": "1 / -1" }}
                >
                  <span class="btn-icon">📋</span>
                  Test get_agent_info()
                </button>
              </div>

              {/* Echo Test with Input */}
              <div style={{ "margin-top": "1rem" }}>
                <label style={{ "display": "block", "margin-bottom": "0.5rem", "color": "var(--color-text)" }}>
                  Test echo(input):
                </label>
                <div class="input-group">
                  <input
                    placeholder="Enter text to echo..."
                    value={echoInput()}
                    onInput={(e) => setEchoInput(e.currentTarget.value)}
                    disabled={isLoading()}
                  />
                  <button
                    onClick={testEcho}
                    class="btn-primary"
                    disabled={isLoading() || !echoInput().trim()}
                  >
                    <span class="btn-icon">🔊</span>
                    Test echo()
                  </button>
                </div>
              </div>

              {/* Test Results Display */}
              <Show when={Object.keys(testResults()).length > 0}>
                <div style={{ "margin-top": "1.5rem" }}>
                  <h3 style={{ "margin-bottom": "0.75rem", "font-size": "1rem" }}>Results:</h3>
                  <div class="status-grid">
                    <Show when={testResults().hello}>
                      <div class="status-item-enhanced">
                        <div class="status-item-header">
                          <span class="status-label">hello()</span>
                        </div>
                        <pre style={{
                          "background": "var(--color-bg-secondary)",
                          "padding": "0.75rem",
                          "border-radius": "var(--border-radius)",
                          "overflow-x": "auto",
                          "font-size": "0.85rem",
                          "margin": "0.5rem 0 0 0"
                        }}>
                          {testResults().hello}
                        </pre>
                      </div>
                    </Show>

                    <Show when={testResults().whoami}>
                      <div class="status-item-enhanced">
                        <div class="status-item-header">
                          <span class="status-label">whoami()</span>
                        </div>
                        <pre style={{
                          "background": "var(--color-bg-secondary)",
                          "padding": "0.75rem",
                          "border-radius": "var(--border-radius)",
                          "overflow-x": "auto",
                          "font-size": "0.85rem",
                          "margin": "0.5rem 0 0 0"
                        }}>
                          {testResults().whoami}
                        </pre>
                      </div>
                    </Show>

                    <Show when={testResults().echo}>
                      <div class="status-item-enhanced">
                        <div class="status-item-header">
                          <span class="status-label">echo()</span>
                        </div>
                        <pre style={{
                          "background": "var(--color-bg-secondary)",
                          "padding": "0.75rem",
                          "border-radius": "var(--border-radius)",
                          "overflow-x": "auto",
                          "font-size": "0.85rem",
                          "margin": "0.5rem 0 0 0"
                        }}>
                          {testResults().echo}
                        </pre>
                      </div>
                    </Show>

                    <Show when={testResults().agentInfo}>
                      <div class="status-item-enhanced" style={{ "grid-column": "1 / -1" }}>
                        <div class="status-item-header">
                          <span class="status-label">get_agent_info()</span>
                        </div>
                        <pre style={{
                          "background": "var(--color-bg-secondary)",
                          "padding": "0.75rem",
                          "border-radius": "var(--border-radius)",
                          "overflow-x": "auto",
                          "font-size": "0.85rem",
                          "margin": "0.5rem 0 0 0"
                        }}>
                          {testResults().agentInfo}
                        </pre>
                      </div>
                    </Show>
                  </div>
                </div>
              </Show>
            </div>
          </div>
        </Show>

        {/* Message Broadcasting Card */}
        <Show when={holochainState() === "connected"}>
          <div class="card card-interactive">
            <div class="card-header">
              <div class="card-icon">💬</div>
              <h2>Message Broadcasting</h2>
            </div>
            <div class="card-content">
              <p style={{ "margin-bottom": "1rem", "color": "var(--color-text-dim)" }}>
                Send messages to the P2P network
              </p>

              {/* Message Input */}
              <div class="input-group">
                <input
                  placeholder="Type your message..."
                  value={messageInput()}
                  onInput={(e) => setMessageInput(e.currentTarget.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !isSending() && sendMessage()}
                  disabled={isSending()}
                />
                <button
                  onClick={sendMessage}
                  class="btn-primary"
                  disabled={!messageInput().trim() || isSending()}
                >
                  <Show when={isSending()} fallback={<><span class="btn-icon">📤</span> Send</>}>
                    <span class="spinner"></span> Sending...
                  </Show>
                </button>
              </div>

              {/* Message Feed */}
              <div class="message-feed">
                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "margin-bottom": "0.75rem" }}>
                  <h3 style={{ "font-size": "1rem", "margin": "0" }}>
                    Messages ({messages().length})
                  </h3>
                  <button
                    onClick={fetchMessages}
                    class="btn-secondary"
                    style={{ "padding": "0.4rem 0.75rem", "font-size": "0.85rem" }}
                    disabled={isLoading()}
                  >
                    <span class="btn-icon">🔄</span> Refresh
                  </button>
                </div>

                <div class="messages-container">
                  <Show when={messages().length > 0} fallback={
                    <p style={{ "text-align": "center", "color": "var(--color-text-dim)", "padding": "2rem" }}>
                      No messages yet. Send the first message!
                    </p>
                  }>
                    <For each={messages()}>
                      {(message) => (
                        <div class="message-item">
                          <div class="message-author">
                            {message.author ? String(message.author).substring(0, 12) + "..." : "Unknown"}
                          </div>
                          <div class="message-content">
                            {message.content || "[Empty message]"}
                          </div>
                          <div class="message-timestamp">
                            {message.timestamp ? new Date(message.timestamp / 1000000).toLocaleString() : "Unknown time"}
                          </div>
                        </div>
                      )}
                    </For>
                  </Show>
                </div>
              </div>
            </div>
          </div>
        </Show>

        {/* Features Card */}
        <div class="card">
          <div class="card-header">
            <div class="card-icon">✨</div>
            <h2>Features</h2>
          </div>
          <div class="card-content">
            <div class="feature-grid">
              <div class="feature-item">
                <div class="feature-icon">🌐</div>
                <div class="feature-text">
                  <h3>Decentralized</h3>
                  <p>No central servers required</p>
                </div>
              </div>
              <div class="feature-item">
                <div class="feature-icon">🔒</div>
                <div class="feature-text">
                  <h3>Encrypted</h3>
                  <p>End-to-end security</p>
                </div>
              </div>
              <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-text">
                  <h3>Fast</h3>
                  <p>Native performance</p>
                </div>
              </div>
              <div class="feature-item">
                <div class="feature-icon">🌍</div>
                <div class="feature-text">
                  <h3>Cross-platform</h3>
                  <p>Windows, macOS, Linux</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* About Card */}
        <details class="card card-collapsible">
          <summary class="card-header card-header-clickable">
            <div class="card-icon">💡</div>
            <h2>About Mycelix</h2>
            <span class="expand-icon">▼</span>
          </summary>
          <div class="card-content">
            <p>
              Mycelix Desktop is a native P2P application built with Tauri and
              Holochain, enabling direct human collaboration without central servers.
            </p>
            <p class="about-tech">
              <strong>Built with:</strong> Tauri v2.8.5, SolidJS 1.8.0, Holochain 0.5.6, Rust 1.90.0
            </p>
          </div>
        </details>
      </main>

      <footer>
        <p>
          Built with 💜 by Luminous Dynamics |{" "}
          <a href="https://mycelix.net" target="_blank" rel="noopener noreferrer">
            mycelix.net
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;