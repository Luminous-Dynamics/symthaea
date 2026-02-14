<script lang="ts">
  import { connectionStatus, connectionError, connect, disconnect } from '../stores/holochain';
  import { loadThoughts } from '../stores/thoughts';

  let wsUrl = 'ws://localhost:8888';
  let isConnecting = false;

  async function handleConnect() {
    isConnecting = true;
    await connect(wsUrl);
    if ($connectionStatus === 'connected') {
      await loadThoughts();
    }
    isConnecting = false;
  }

  async function handleDisconnect() {
    await disconnect();
  }
</script>

<div class="connection-status">
  {#if $connectionStatus === 'connected'}
    <span class="status connected">● Connected</span>
    <button class="disconnect-btn" on:click={handleDisconnect}>Disconnect</button>
  {:else if $connectionStatus === 'connecting' || isConnecting}
    <span class="status connecting">◐ Connecting...</span>
  {:else}
    <div class="connect-form">
      <input
        type="text"
        bind:value={wsUrl}
        placeholder="ws://localhost:8888"
        disabled={isConnecting}
      />
      <button on:click={handleConnect} disabled={isConnecting}>
        Connect
      </button>
    </div>
    {#if $connectionStatus === 'error' && $connectionError}
      <span class="error">{$connectionError}</span>
    {/if}
  {/if}
</div>

<style>
  .connection-status {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .status {
    font-size: 0.85rem;
    padding: 4px 12px;
    border-radius: 4px;
  }

  .status.connected {
    color: #34d399;
    background: rgba(52, 211, 153, 0.1);
  }

  .status.connecting {
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.1);
  }

  .connect-form {
    display: flex;
    gap: 8px;
  }

  input {
    padding: 6px 10px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    color: #e5e5e5;
    font-size: 0.85rem;
    width: 200px;
  }

  input:focus {
    outline: none;
    border-color: #7c3aed;
  }

  button {
    padding: 6px 14px;
    background: #7c3aed;
    border: none;
    border-radius: 4px;
    color: white;
    font-size: 0.85rem;
    cursor: pointer;
  }

  button:hover:not(:disabled) {
    background: #6d28d9;
  }

  button:disabled {
    background: #4a4a6e;
    cursor: not-allowed;
  }

  .disconnect-btn {
    background: #3a3a5e;
  }

  .disconnect-btn:hover {
    background: #4a4a6e;
  }

  .error {
    color: #f87171;
    font-size: 0.8rem;
  }
</style>
