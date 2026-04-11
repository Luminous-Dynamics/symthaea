// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track W: Offline & Mobile Excellence
 *
 * PWA shell, offline sync status, smart prefetch, gesture controls,
 * wearable companion, cross-device continuity, and bandwidth optimization.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// Types
// ============================================================================

interface SyncStatus {
  isOnline: boolean;
  lastSyncTime?: string;
  pendingChanges: number;
  syncProgress?: number;
  currentVersion: number;
  serverVersion: number;
  conflictCount: number;
}

interface OfflineEmail {
  id: string;
  subject: string;
  from: string;
  snippet: string;
  receivedAt: string;
  isRead: boolean;
  isCached: boolean;
  cacheSize: number;
}

interface PrefetchConfig {
  enabled: boolean;
  maxCacheSize: number;
  currentCacheSize: number;
  prefetchOnWifi: boolean;
  prefetchThreadDepth: number;
  prefetchAttachments: boolean;
  maxAttachmentSize: number;
  smartPrefetchEnabled: boolean;
}

interface GestureConfig {
  swipeLeftAction: string;
  swipeRightAction: string;
  longPressAction: string;
  doubleTapAction: string;
  pinchAction: string;
  shakeAction: string;
  enabled: boolean;
}

interface WearableDevice {
  id: string;
  name: string;
  deviceType: 'Watch' | 'Ring' | 'Glasses' | 'Band';
  connected: boolean;
  batteryLevel?: number;
  lastSeen?: string;
  capabilities: string[];
}

interface WearableNotification {
  id: string;
  emailId: string;
  subject: string;
  from: string;
  sentAt: string;
  priority: 'High' | 'Normal' | 'Low';
  dismissed: boolean;
}

interface ContinuitySession {
  sessionId: string;
  sourceDevice: string;
  targetDevice: string;
  context: {
    emailId?: string;
    draftId?: string;
    scrollPosition?: number;
    composeState?: object;
  };
  createdAt: string;
  expiresAt: string;
}

interface DeviceInfo {
  deviceId: string;
  deviceName: string;
  platform: string;
  lastActive: string;
  isCurrent: boolean;
}

interface BandwidthStats {
  currentMode: 'Full' | 'Reduced' | 'Minimal' | 'TextOnly';
  dataUsedToday: number;
  dataSavedToday: number;
  compressionRatio: number;
  imageQuality: number;
}

// ============================================================================
// Hooks
// ============================================================================

function useSyncStatus() {
  const [status, setStatus] = useState<SyncStatus>({
    isOnline: navigator.onLine,
    pendingChanges: 0,
    currentVersion: 0,
    serverVersion: 0,
    conflictCount: 0,
  });
  const [syncing, setSyncing] = useState(false);

  useEffect(() => {
    const handleOnline = () => setStatus(prev => ({ ...prev, isOnline: true }));
    const handleOffline = () => setStatus(prev => ({ ...prev, isOnline: false }));

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Initial fetch
    fetch('/api/mobile/sync/status')
      .then(res => res.json())
      .then(data => setStatus(prev => ({ ...prev, ...data })));

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const triggerSync = useCallback(async () => {
    setSyncing(true);
    try {
      const response = await fetch('/api/mobile/sync/trigger', { method: 'POST' });
      const result = await response.json();
      setStatus(prev => ({ ...prev, ...result }));
    } finally {
      setSyncing(false);
    }
  }, []);

  const resolveConflicts = useCallback(async (resolutions: { id: string; choice: 'local' | 'remote' }[]) => {
    await fetch('/api/mobile/sync/resolve-conflicts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resolutions }),
    });
    setStatus(prev => ({ ...prev, conflictCount: prev.conflictCount - resolutions.length }));
  }, []);

  return { status, syncing, triggerSync, resolveConflicts };
}

function useOfflineEmails() {
  const [emails, setEmails] = useState<OfflineEmail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/mobile/offline/emails')
      .then(res => res.json())
      .then(setEmails)
      .finally(() => setLoading(false));
  }, []);

  const cacheEmail = useCallback(async (emailId: string) => {
    await fetch(`/api/mobile/offline/cache/${emailId}`, { method: 'POST' });
    setEmails(prev => prev.map(e => e.id === emailId ? { ...e, isCached: true } : e));
  }, []);

  const uncacheEmail = useCallback(async (emailId: string) => {
    await fetch(`/api/mobile/offline/cache/${emailId}`, { method: 'DELETE' });
    setEmails(prev => prev.map(e => e.id === emailId ? { ...e, isCached: false } : e));
  }, []);

  const clearCache = useCallback(async () => {
    await fetch('/api/mobile/offline/cache', { method: 'DELETE' });
    setEmails(prev => prev.map(e => ({ ...e, isCached: false })));
  }, []);

  return { emails, loading, cacheEmail, uncacheEmail, clearCache };
}

function usePrefetch() {
  const [config, setConfig] = useState<PrefetchConfig | null>(null);
  const [predictions, setPredictions] = useState<{ emailId: string; probability: number }[]>([]);

  useEffect(() => {
    fetch('/api/mobile/prefetch/config')
      .then(res => res.json())
      .then(setConfig);

    fetch('/api/mobile/prefetch/predictions')
      .then(res => res.json())
      .then(setPredictions);
  }, []);

  const updateConfig = useCallback(async (updates: Partial<PrefetchConfig>) => {
    await fetch('/api/mobile/prefetch/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setConfig(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  const triggerPrefetch = useCallback(async () => {
    await fetch('/api/mobile/prefetch/trigger', { method: 'POST' });
  }, []);

  return { config, predictions, updateConfig, triggerPrefetch };
}

function useGestures() {
  const [config, setConfig] = useState<GestureConfig | null>(null);

  useEffect(() => {
    fetch('/api/mobile/gestures/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const updateConfig = useCallback(async (updates: Partial<GestureConfig>) => {
    await fetch('/api/mobile/gestures/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setConfig(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  return { config, updateConfig };
}

function useWearables() {
  const [devices, setDevices] = useState<WearableDevice[]>([]);
  const [notifications, setNotifications] = useState<WearableNotification[]>([]);

  useEffect(() => {
    fetch('/api/mobile/wearables/devices')
      .then(res => res.json())
      .then(setDevices);

    fetch('/api/mobile/wearables/notifications')
      .then(res => res.json())
      .then(setNotifications);
  }, []);

  const pairDevice = useCallback(async (pairingCode: string) => {
    const response = await fetch('/api/mobile/wearables/pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pairingCode }),
    });
    const device = await response.json();
    setDevices(prev => [...prev, device]);
    return device;
  }, []);

  const unpairDevice = useCallback(async (deviceId: string) => {
    await fetch(`/api/mobile/wearables/devices/${deviceId}`, { method: 'DELETE' });
    setDevices(prev => prev.filter(d => d.id !== deviceId));
  }, []);

  const dismissNotification = useCallback(async (notificationId: string) => {
    await fetch(`/api/mobile/wearables/notifications/${notificationId}/dismiss`, { method: 'POST' });
    setNotifications(prev => prev.map(n => n.id === notificationId ? { ...n, dismissed: true } : n));
  }, []);

  return { devices, notifications, pairDevice, unpairDevice, dismissNotification };
}

function useContinuity() {
  const [sessions, setSessions] = useState<ContinuitySession[]>([]);
  const [devices, setDevices] = useState<DeviceInfo[]>([]);

  useEffect(() => {
    fetch('/api/mobile/continuity/sessions')
      .then(res => res.json())
      .then(setSessions);

    fetch('/api/mobile/continuity/devices')
      .then(res => res.json())
      .then(setDevices);
  }, []);

  const handoff = useCallback(async (targetDeviceId: string, context: object) => {
    const response = await fetch('/api/mobile/continuity/handoff', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ targetDeviceId, context }),
    });
    return await response.json();
  }, []);

  const acceptHandoff = useCallback(async (sessionId: string) => {
    const response = await fetch(`/api/mobile/continuity/sessions/${sessionId}/accept`, {
      method: 'POST',
    });
    return await response.json();
  }, []);

  return { sessions, devices, handoff, acceptHandoff };
}

function useBandwidth() {
  const [stats, setStats] = useState<BandwidthStats | null>(null);

  useEffect(() => {
    fetch('/api/mobile/bandwidth/stats')
      .then(res => res.json())
      .then(setStats);
  }, []);

  const setMode = useCallback(async (mode: BandwidthStats['currentMode']) => {
    await fetch('/api/mobile/bandwidth/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    setStats(prev => prev ? { ...prev, currentMode: mode } : null);
  }, []);

  return { stats, setMode };
}

// ============================================================================
// Components
// ============================================================================

function SyncStatusPanel() {
  const { status, syncing, triggerSync, resolveConflicts } = useSyncStatus();

  return (
    <div className="sync-status-panel">
      <header>
        <h2>Sync Status</h2>
        <div className={`connection-indicator ${status.isOnline ? 'online' : 'offline'}`}>
          {status.isOnline ? 'Online' : 'Offline'}
        </div>
      </header>

      <div className="sync-info">
        <div className="stat">
          <span className="label">Last Sync</span>
          <span className="value">
            {status.lastSyncTime
              ? new Date(status.lastSyncTime).toLocaleString()
              : 'Never'}
          </span>
        </div>
        <div className="stat">
          <span className="label">Pending Changes</span>
          <span className="value">{status.pendingChanges}</span>
        </div>
        <div className="stat">
          <span className="label">Version</span>
          <span className="value">
            {status.currentVersion} / {status.serverVersion}
          </span>
        </div>
      </div>

      {status.syncProgress !== undefined && status.syncProgress < 100 && (
        <div className="sync-progress">
          <div className="progress-bar">
            <div className="fill" style={{ width: `${status.syncProgress}%` }} />
          </div>
          <span>{Math.round(status.syncProgress)}% synced</span>
        </div>
      )}

      {status.conflictCount > 0 && (
        <div className="conflicts-warning">
          <span className="icon">⚠️</span>
          <span>{status.conflictCount} conflicts need resolution</span>
          <button onClick={() => resolveConflicts([])}>Resolve</button>
        </div>
      )}

      <button
        className="sync-button"
        onClick={triggerSync}
        disabled={syncing || !status.isOnline}
      >
        {syncing ? 'Syncing...' : 'Sync Now'}
      </button>
    </div>
  );
}

function OfflineCachePanel() {
  const { emails, loading, cacheEmail, uncacheEmail, clearCache } = useOfflineEmails();
  const { config } = usePrefetch();

  const totalCacheSize = emails.reduce((sum, e) => sum + (e.isCached ? e.cacheSize : 0), 0);
  const cachedCount = emails.filter(e => e.isCached).length;

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <div className="offline-cache-panel">
      <header>
        <h2>Offline Cache</h2>
        <button onClick={clearCache} className="clear-btn">Clear All</button>
      </header>

      <div className="cache-stats">
        <div className="stat">
          <span className="value">{cachedCount}</span>
          <span className="label">Emails Cached</span>
        </div>
        <div className="stat">
          <span className="value">{formatSize(totalCacheSize)}</span>
          <span className="label">Cache Size</span>
        </div>
        {config && (
          <div className="stat">
            <span className="value">{formatSize(config.maxCacheSize)}</span>
            <span className="label">Max Cache</span>
          </div>
        )}
      </div>

      {config && (
        <div className="cache-usage">
          <div className="progress-bar">
            <div
              className="fill"
              style={{ width: `${(totalCacheSize / config.maxCacheSize) * 100}%` }}
            />
          </div>
          <span>{Math.round((totalCacheSize / config.maxCacheSize) * 100)}% used</span>
        </div>
      )}

      {loading ? (
        <div className="loading">Loading cached emails...</div>
      ) : (
        <div className="email-list">
          {emails.slice(0, 20).map(email => (
            <div key={email.id} className={`email-item ${email.isCached ? 'cached' : ''}`}>
              <div className="email-info">
                <span className="from">{email.from}</span>
                <span className="subject">{email.subject}</span>
                <span className="date">{new Date(email.receivedAt).toLocaleDateString()}</span>
              </div>
              <div className="cache-controls">
                {email.isCached && <span className="size">{formatSize(email.cacheSize)}</span>}
                <button onClick={() => email.isCached ? uncacheEmail(email.id) : cacheEmail(email.id)}>
                  {email.isCached ? 'Remove' : 'Cache'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PrefetchSettings() {
  const { config, predictions, updateConfig, triggerPrefetch } = usePrefetch();

  if (!config) {
    return <div className="loading">Loading prefetch settings...</div>;
  }

  const formatSize = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(0)} MB`;

  return (
    <div className="prefetch-settings">
      <header>
        <h2>Smart Prefetch</h2>
        <label className="toggle">
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={e => updateConfig({ enabled: e.target.checked })}
          />
          <span className="slider" />
        </label>
      </header>

      {config.enabled && (
        <>
          <div className="settings-group">
            <div className="setting">
              <label>
                <input
                  type="checkbox"
                  checked={config.smartPrefetchEnabled}
                  onChange={e => updateConfig({ smartPrefetchEnabled: e.target.checked })}
                />
                ML-based smart prefetch
              </label>
              <span className="hint">Predict which emails you'll open</span>
            </div>

            <div className="setting">
              <label>
                <input
                  type="checkbox"
                  checked={config.prefetchOnWifi}
                  onChange={e => updateConfig({ prefetchOnWifi: e.target.checked })}
                />
                Only prefetch on Wi-Fi
              </label>
            </div>

            <div className="setting">
              <label>
                <input
                  type="checkbox"
                  checked={config.prefetchAttachments}
                  onChange={e => updateConfig({ prefetchAttachments: e.target.checked })}
                />
                Prefetch attachments
              </label>
            </div>
          </div>

          <div className="slider-setting">
            <label>Max cache size: {formatSize(config.maxCacheSize)}</label>
            <input
              type="range"
              min={52428800}
              max={1073741824}
              step={52428800}
              value={config.maxCacheSize}
              onChange={e => updateConfig({ maxCacheSize: parseInt(e.target.value) })}
            />
          </div>

          <div className="slider-setting">
            <label>Thread depth: {config.prefetchThreadDepth} emails</label>
            <input
              type="range"
              min={1}
              max={20}
              value={config.prefetchThreadDepth}
              onChange={e => updateConfig({ prefetchThreadDepth: parseInt(e.target.value) })}
            />
          </div>

          {config.smartPrefetchEnabled && predictions.length > 0 && (
            <div className="predictions">
              <h3>Predicted Next Opens</h3>
              {predictions.slice(0, 5).map(p => (
                <div key={p.emailId} className="prediction">
                  <span className="id">{p.emailId.slice(0, 8)}...</span>
                  <div className="probability-bar">
                    <div className="fill" style={{ width: `${p.probability * 100}%` }} />
                  </div>
                  <span className="percent">{Math.round(p.probability * 100)}%</span>
                </div>
              ))}
            </div>
          )}

          <button onClick={triggerPrefetch} className="prefetch-btn">
            Prefetch Now
          </button>
        </>
      )}
    </div>
  );
}

function GestureSettings() {
  const { config, updateConfig } = useGestures();

  if (!config) {
    return <div className="loading">Loading gesture settings...</div>;
  }

  const actions = [
    'Archive',
    'Delete',
    'MarkRead',
    'MarkUnread',
    'Star',
    'Snooze',
    'Reply',
    'Forward',
    'MoveTo',
    'None',
  ];

  const gestures = [
    { key: 'swipeLeftAction', label: 'Swipe Left', icon: '👈' },
    { key: 'swipeRightAction', label: 'Swipe Right', icon: '👉' },
    { key: 'longPressAction', label: 'Long Press', icon: '👆' },
    { key: 'doubleTapAction', label: 'Double Tap', icon: '👆👆' },
    { key: 'pinchAction', label: 'Pinch', icon: '🤏' },
    { key: 'shakeAction', label: 'Shake', icon: '📱' },
  ] as const;

  return (
    <div className="gesture-settings">
      <header>
        <h2>Gesture Controls</h2>
        <label className="toggle">
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={e => updateConfig({ enabled: e.target.checked })}
          />
          <span className="slider" />
        </label>
      </header>

      {config.enabled && (
        <div className="gesture-list">
          {gestures.map(gesture => (
            <div key={gesture.key} className="gesture-item">
              <div className="gesture-info">
                <span className="icon">{gesture.icon}</span>
                <span className="label">{gesture.label}</span>
              </div>
              <select
                value={config[gesture.key]}
                onChange={e => updateConfig({ [gesture.key]: e.target.value })}
              >
                {actions.map(action => (
                  <option key={action} value={action}>{action}</option>
                ))}
              </select>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WearablePanel() {
  const { devices, notifications, pairDevice, unpairDevice, dismissNotification } = useWearables();
  const [showPair, setShowPair] = useState(false);
  const [pairingCode, setPairingCode] = useState('');

  const deviceIcons: Record<string, string> = {
    Watch: '⌚',
    Ring: '💍',
    Glasses: '👓',
    Band: '📿',
  };

  const handlePair = async () => {
    if (pairingCode) {
      await pairDevice(pairingCode);
      setShowPair(false);
      setPairingCode('');
    }
  };

  return (
    <div className="wearable-panel">
      <header>
        <h2>Wearable Devices</h2>
        <button onClick={() => setShowPair(true)}>Pair Device</button>
      </header>

      {devices.length === 0 ? (
        <div className="empty-state">
          <p>No wearable devices paired</p>
          <button onClick={() => setShowPair(true)}>Pair your first device</button>
        </div>
      ) : (
        <div className="device-list">
          {devices.map(device => (
            <div key={device.id} className={`device-item ${device.connected ? 'connected' : ''}`}>
              <span className="icon">{deviceIcons[device.deviceType] || '📱'}</span>
              <div className="device-info">
                <span className="name">{device.name}</span>
                <span className="type">{device.deviceType}</span>
                {device.batteryLevel !== undefined && (
                  <span className="battery">🔋 {device.batteryLevel}%</span>
                )}
              </div>
              <span className={`status ${device.connected ? 'connected' : 'disconnected'}`}>
                {device.connected ? 'Connected' : 'Disconnected'}
              </span>
              <button onClick={() => unpairDevice(device.id)} className="unpair-btn">
                Unpair
              </button>
            </div>
          ))}
        </div>
      )}

      {notifications.filter(n => !n.dismissed).length > 0 && (
        <div className="wearable-notifications">
          <h3>Recent Notifications</h3>
          {notifications.filter(n => !n.dismissed).slice(0, 5).map(notif => (
            <div key={notif.id} className={`notification-item ${notif.priority.toLowerCase()}`}>
              <div className="content">
                <span className="from">{notif.from}</span>
                <span className="subject">{notif.subject}</span>
              </div>
              <button onClick={() => dismissNotification(notif.id)}>Dismiss</button>
            </div>
          ))}
        </div>
      )}

      {showPair && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Pair Wearable Device</h3>
            <p>Enter the pairing code shown on your device</p>
            <input
              type="text"
              value={pairingCode}
              onChange={e => setPairingCode(e.target.value)}
              placeholder="Enter pairing code"
            />
            <div className="modal-actions">
              <button onClick={() => setShowPair(false)}>Cancel</button>
              <button onClick={handlePair} disabled={!pairingCode}>Pair</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ContinuityPanel() {
  const { sessions, devices, handoff, acceptHandoff } = useContinuity();

  const pendingSessions = sessions.filter(s =>
    new Date(s.expiresAt) > new Date()
  );

  return (
    <div className="continuity-panel">
      <header>
        <h2>Cross-Device Continuity</h2>
      </header>

      <div className="devices-section">
        <h3>Your Devices</h3>
        <div className="device-list">
          {devices.map(device => (
            <div key={device.deviceId} className={`device-item ${device.isCurrent ? 'current' : ''}`}>
              <span className="platform-icon">
                {device.platform === 'ios' ? '📱' :
                 device.platform === 'android' ? '📱' :
                 device.platform === 'macos' ? '💻' :
                 device.platform === 'windows' ? '🖥️' : '🌐'}
              </span>
              <div className="device-info">
                <span className="name">{device.deviceName}</span>
                <span className="platform">{device.platform}</span>
                <span className="last-active">
                  {device.isCurrent ? 'Current device' : `Last active: ${new Date(device.lastActive).toLocaleString()}`}
                </span>
              </div>
              {!device.isCurrent && (
                <button onClick={() => handoff(device.deviceId, {})}>
                  Handoff
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {pendingSessions.length > 0 && (
        <div className="pending-sessions">
          <h3>Incoming Handoffs</h3>
          {pendingSessions.map(session => (
            <div key={session.sessionId} className="session-item">
              <div className="session-info">
                <span className="from">From: {session.sourceDevice}</span>
                <span className="context">
                  {session.context.emailId && 'Viewing email'}
                  {session.context.draftId && 'Composing draft'}
                </span>
                <span className="expires">
                  Expires: {new Date(session.expiresAt).toLocaleTimeString()}
                </span>
              </div>
              <button onClick={() => acceptHandoff(session.sessionId)}>
                Accept
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BandwidthPanel() {
  const { stats, setMode } = useBandwidth();

  if (!stats) {
    return <div className="loading">Loading bandwidth stats...</div>;
  }

  const modes = [
    { id: 'Full', label: 'Full Quality', description: 'All images and content at full resolution' },
    { id: 'Reduced', label: 'Reduced', description: 'Compressed images, lazy loading' },
    { id: 'Minimal', label: 'Minimal', description: 'Thumbnails only, defer attachments' },
    { id: 'TextOnly', label: 'Text Only', description: 'No images or rich content' },
  ] as const;

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <div className="bandwidth-panel">
      <header>
        <h2>Bandwidth Optimization</h2>
      </header>

      <div className="stats">
        <div className="stat">
          <span className="value">{formatBytes(stats.dataUsedToday)}</span>
          <span className="label">Used Today</span>
        </div>
        <div className="stat">
          <span className="value">{formatBytes(stats.dataSavedToday)}</span>
          <span className="label">Saved Today</span>
        </div>
        <div className="stat">
          <span className="value">{Math.round(stats.compressionRatio * 100)}%</span>
          <span className="label">Compression</span>
        </div>
      </div>

      <div className="mode-selector">
        <h3>Data Mode</h3>
        {modes.map(mode => (
          <label
            key={mode.id}
            className={`mode-option ${stats.currentMode === mode.id ? 'active' : ''}`}
          >
            <input
              type="radio"
              name="bandwidthMode"
              checked={stats.currentMode === mode.id}
              onChange={() => setMode(mode.id)}
            />
            <div className="mode-info">
              <span className="label">{mode.label}</span>
              <span className="description">{mode.description}</span>
            </div>
          </label>
        ))}
      </div>

      {stats.currentMode !== 'Full' && (
        <div className="image-quality">
          <label>Image Quality: {stats.imageQuality}%</label>
          <div className="quality-bar">
            <div className="fill" style={{ width: `${stats.imageQuality}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface MobileAppProps {
  initialTab?: string;
}

export function MobileApp({ initialTab = 'sync' }: MobileAppProps) {
  const [activeTab, setActiveTab] = useState(initialTab);

  const tabs = [
    { id: 'sync', label: 'Sync', icon: '🔄' },
    { id: 'offline', label: 'Offline', icon: '📴' },
    { id: 'prefetch', label: 'Prefetch', icon: '⚡' },
    { id: 'gestures', label: 'Gestures', icon: '👆' },
    { id: 'wearables', label: 'Wearables', icon: '⌚' },
    { id: 'continuity', label: 'Continuity', icon: '🔗' },
    { id: 'bandwidth', label: 'Data', icon: '📶' },
  ];

  return (
    <div className="mobile-app">
      <nav className="mobile-nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? 'active' : ''}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="icon">{tab.icon}</span>
            <span className="label">{tab.label}</span>
          </button>
        ))}
      </nav>

      <div className="mobile-content">
        {activeTab === 'sync' && <SyncStatusPanel />}
        {activeTab === 'offline' && <OfflineCachePanel />}
        {activeTab === 'prefetch' && <PrefetchSettings />}
        {activeTab === 'gestures' && <GestureSettings />}
        {activeTab === 'wearables' && <WearablePanel />}
        {activeTab === 'continuity' && <ContinuityPanel />}
        {activeTab === 'bandwidth' && <BandwidthPanel />}
      </div>
    </div>
  );
}

export default MobileApp;
