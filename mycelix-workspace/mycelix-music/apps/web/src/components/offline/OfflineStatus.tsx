// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Offline Status Component
 *
 * Shows offline/online status and cache management UI.
 */

import { useCallback, useEffect, useState } from 'react';
import { useOfflineStorage, type CachedTrack } from '@/hooks/useOfflineStorage';

// === Types ===

interface OfflineStatusProps {
  compact?: boolean;
  onDownloadTrack?: (trackId: string) => void;
}

// === Helpers ===

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// === Component ===

export function OfflineStatus({ compact = false, onDownloadTrack }: OfflineStatusProps) {
  const storage = useOfflineStorage();
  const [cachedTracks, setCachedTracks] = useState<CachedTrack[]>([]);
  const [showDetails, setShowDetails] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  // Load cached tracks
  useEffect(() => {
    if (storage.isIndexedDBReady) {
      storage.getAllCachedTracks().then(setCachedTracks);
    }
  }, [storage.isIndexedDBReady, storage.cachedTracksCount]);

  // Process sync when coming online
  useEffect(() => {
    if (storage.isOnline && storage.syncQueueCount > 0) {
      storage.processSyncQueue();
    }
  }, [storage.isOnline, storage.syncQueueCount]);

  // Clear all caches
  const handleClearCache = useCallback(async () => {
    setIsClearing(true);
    await storage.clearAllCaches();
    setCachedTracks([]);
    setIsClearing(false);
  }, [storage]);

  // Remove single track
  const handleRemoveTrack = useCallback(async (trackId: string) => {
    await storage.uncacheTrack(trackId);
    setCachedTracks((prev) => prev.filter((t) => t.id !== trackId));
  }, [storage]);

  // Compact view
  if (compact) {
    return (
      <div style={styles.compactContainer}>
        <div
          style={{
            ...styles.statusDot,
            backgroundColor: storage.isOnline ? '#22c55e' : '#ef4444',
          }}
        />
        <span style={styles.compactText}>
          {storage.isOnline ? 'Online' : 'Offline'}
        </span>
        {storage.cachedTracksCount > 0 && (
          <span style={styles.compactBadge}>
            {storage.cachedTracksCount} cached
          </span>
        )}
        {storage.syncQueueCount > 0 && (
          <span style={styles.syncBadge}>
            {storage.syncQueueCount} pending
          </span>
        )}
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Status Header */}
      <div style={styles.header}>
        <div style={styles.statusRow}>
          <div
            style={{
              ...styles.statusIndicator,
              backgroundColor: storage.isOnline ? '#22c55e' : '#ef4444',
            }}
          />
          <span style={styles.statusText}>
            {storage.isOnline ? 'Online' : 'Offline Mode'}
          </span>
        </div>
        <button
          style={styles.detailsButton}
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide' : 'Details'}
        </button>
      </div>

      {/* Offline Banner */}
      {!storage.isOnline && (
        <div style={styles.offlineBanner}>
          <span style={styles.offlineIcon}>\u{1F4F4}</span>
          <span>You're offline. Cached music is available.</span>
        </div>
      )}

      {/* Sync Status */}
      {storage.syncQueueCount > 0 && (
        <div style={styles.syncStatus}>
          <span style={styles.syncIcon}>\u{1F504}</span>
          <span>{storage.syncQueueCount} actions pending sync</span>
          {storage.isOnline && (
            <button
              style={styles.syncButton}
              onClick={() => storage.processSyncQueue()}
            >
              Sync Now
            </button>
          )}
        </div>
      )}

      {/* Cache Stats */}
      <div style={styles.stats}>
        <div style={styles.stat}>
          <span style={styles.statValue}>{storage.cachedTracksCount}</span>
          <span style={styles.statLabel}>Cached Tracks</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statValue}>{formatBytes(storage.cacheSize)}</span>
          <span style={styles.statLabel}>Storage Used</span>
        </div>
        <div style={styles.stat}>
          <div style={styles.readyIndicators}>
            <span
              style={{
                ...styles.readyDot,
                backgroundColor: storage.isServiceWorkerReady ? '#22c55e' : '#6b7280',
              }}
              title="Service Worker"
            />
            <span
              style={{
                ...styles.readyDot,
                backgroundColor: storage.isIndexedDBReady ? '#22c55e' : '#6b7280',
              }}
              title="IndexedDB"
            />
          </div>
          <span style={styles.statLabel}>System Ready</span>
        </div>
      </div>

      {/* Details Panel */}
      {showDetails && (
        <div style={styles.details}>
          <div style={styles.detailsHeader}>
            <h4 style={styles.detailsTitle}>Cached Tracks</h4>
            <button
              style={styles.clearButton}
              onClick={handleClearCache}
              disabled={isClearing || cachedTracks.length === 0}
            >
              {isClearing ? 'Clearing...' : 'Clear All'}
            </button>
          </div>

          {cachedTracks.length === 0 ? (
            <div style={styles.emptyState}>
              <span style={styles.emptyIcon}>\u{1F4E5}</span>
              <p>No tracks cached yet</p>
              <p style={styles.emptyHint}>
                Tap the download icon on tracks to cache them for offline listening
              </p>
            </div>
          ) : (
            <div style={styles.trackList}>
              {cachedTracks.map((track) => (
                <div key={track.id} style={styles.trackItem}>
                  {track.coverUrl && (
                    <img
                      src={track.coverUrl}
                      alt=""
                      style={styles.trackCover}
                    />
                  )}
                  <div style={styles.trackInfo}>
                    <span style={styles.trackTitle}>{track.title}</span>
                    <span style={styles.trackArtist}>{track.artist}</span>
                    <span style={styles.trackMeta}>
                      {formatBytes(track.size)} • Cached {formatTime(track.cachedAt)}
                    </span>
                  </div>
                  <button
                    style={styles.removeButton}
                    onClick={() => handleRemoveTrack(track.id)}
                    title="Remove from cache"
                  >
                    X
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Last Sync */}
          {storage.lastSyncTime && (
            <div style={styles.lastSync}>
              Last synced: {formatTime(storage.lastSyncTime)}
            </div>
          )}
        </div>
      )}

      {/* Install Prompt */}
      {storage.isServiceWorkerReady && !window.matchMedia('(display-mode: standalone)').matches && (
        <div style={styles.installPrompt}>
          <span>Install Mycelix for the best offline experience</span>
        </div>
      )}
    </div>
  );
}

// === Styles ===

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: '12px',
    padding: '16px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    color: '#ffffff',
  },
  compactContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '20px',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  compactText: {
    fontSize: '12px',
    color: '#ffffff',
  },
  compactBadge: {
    fontSize: '11px',
    color: '#9ca3af',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: '2px 6px',
    borderRadius: '10px',
  },
  syncBadge: {
    fontSize: '11px',
    color: '#eab308',
    backgroundColor: 'rgba(234, 179, 8, 0.1)',
    padding: '2px 6px',
    borderRadius: '10px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  statusIndicator: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  statusText: {
    fontSize: '14px',
    fontWeight: 500,
  },
  detailsButton: {
    background: 'none',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '6px',
    padding: '4px 12px',
    color: '#9ca3af',
    fontSize: '12px',
    cursor: 'pointer',
  },
  offlineBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: '8px',
    fontSize: '13px',
    color: '#fca5a5',
    marginBottom: '12px',
  },
  offlineIcon: {
    fontSize: '16px',
  },
  syncStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    backgroundColor: 'rgba(234, 179, 8, 0.1)',
    borderRadius: '8px',
    fontSize: '13px',
    color: '#fde047',
    marginBottom: '12px',
  },
  syncIcon: {
    fontSize: '14px',
  },
  syncButton: {
    marginLeft: 'auto',
    background: 'none',
    border: '1px solid currentColor',
    borderRadius: '4px',
    padding: '4px 8px',
    color: 'inherit',
    fontSize: '11px',
    cursor: 'pointer',
  },
  stats: {
    display: 'flex',
    gap: '16px',
    padding: '12px 0',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  },
  stat: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
  },
  statValue: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#818cf8',
  },
  statLabel: {
    fontSize: '11px',
    color: '#9ca3af',
  },
  readyIndicators: {
    display: 'flex',
    gap: '4px',
  },
  readyDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  details: {
    marginTop: '16px',
  },
  detailsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  detailsTitle: {
    fontSize: '14px',
    fontWeight: 500,
    margin: 0,
  },
  clearButton: {
    background: 'none',
    border: '1px solid rgba(239, 68, 68, 0.5)',
    borderRadius: '4px',
    padding: '4px 10px',
    color: '#ef4444',
    fontSize: '11px',
    cursor: 'pointer',
  },
  emptyState: {
    textAlign: 'center',
    padding: '24px',
    color: '#9ca3af',
  },
  emptyIcon: {
    fontSize: '32px',
    display: 'block',
    marginBottom: '12px',
  },
  emptyHint: {
    fontSize: '12px',
    color: '#6b7280',
    marginTop: '8px',
  },
  trackList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    maxHeight: '300px',
    overflowY: 'auto',
  },
  trackItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: '8px',
  },
  trackCover: {
    width: '40px',
    height: '40px',
    borderRadius: '4px',
    objectFit: 'cover',
  },
  trackInfo: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    overflow: 'hidden',
  },
  trackTitle: {
    fontSize: '13px',
    fontWeight: 500,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  trackArtist: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  trackMeta: {
    fontSize: '10px',
    color: '#6b7280',
  },
  removeButton: {
    background: 'none',
    border: 'none',
    color: '#6b7280',
    fontSize: '14px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  lastSync: {
    textAlign: 'center',
    fontSize: '11px',
    color: '#6b7280',
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  },
  installPrompt: {
    marginTop: '12px',
    padding: '10px',
    backgroundColor: 'rgba(129, 140, 248, 0.1)',
    borderRadius: '8px',
    textAlign: 'center',
    fontSize: '12px',
    color: '#818cf8',
  },
};

export default OfflineStatus;
