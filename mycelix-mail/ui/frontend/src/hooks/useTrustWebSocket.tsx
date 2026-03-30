// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust WebSocket Hook
 *
 * Real-time updates for trust-related events:
 * - New attestations
 * - Trust score changes
 * - Introduction requests
 * - Network updates
 *
 * Connects to the backend WebSocket and dispatches events.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTrustNotificationStore } from '@/components/notifications/TrustNotifications';

// WebSocket event types
export type TrustEventType =
  | 'attestation_created'
  | 'attestation_revoked'
  | 'trust_score_updated'
  | 'introduction_received'
  | 'introduction_accepted'
  | 'introduction_declined'
  | 'credential_verified'
  | 'credential_expired'
  | 'network_sync'
  | 'connection_status';

export interface TrustEvent {
  type: TrustEventType;
  timestamp: string;
  data: {
    // Attestation events
    attestorDid?: string;
    subjectDid?: string;
    attestationId?: string;
    trustScore?: number;
    previousScore?: number;
    relationship?: string;
    message?: string;

    // Introduction events
    introductionId?: string;
    introducerDid?: string;
    introducerName?: string;

    // Credential events
    credentialType?: string;
    credentialId?: string;
    expiresAt?: string;

    // Network events
    connectionCount?: number;
    newConnections?: number;
  };
}

export interface UseTrustWebSocketOptions {
  userDid: string;
  enabled?: boolean;
  onEvent?: (event: TrustEvent) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface UseTrustWebSocketResult {
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  lastEvent: TrustEvent | null;
  reconnect: () => void;
  disconnect: () => void;
}

export function useTrustWebSocket({
  userDid,
  enabled = true,
  onEvent,
  reconnectInterval = 5000,
  maxReconnectAttempts = 10,
}: UseTrustWebSocketOptions): UseTrustWebSocketResult {
  const queryClient = useQueryClient();
  const { addNotification } = useTrustNotificationStore();

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastEvent, setLastEvent] = useState<TrustEvent | null>(null);

  // Handle incoming WebSocket events
  const handleEvent = useCallback(
    (event: TrustEvent) => {
      setLastEvent(event);
      onEvent?.(event);

      // Create notifications for relevant events
      switch (event.type) {
        case 'attestation_created':
          addNotification({
            type: 'attestation_received',
            title: 'New Trust Attestation',
            message: `${event.data.attestorDid?.slice(-12)} has attested trust for you`,
            actionable: true,
            data: {
              contactDid: event.data.attestorDid,
              trustScore: event.data.trustScore,
            },
          });
          // Invalidate trust queries
          queryClient.invalidateQueries({ queryKey: ['trust'] });
          queryClient.invalidateQueries({ queryKey: ['attestations'] });
          break;

        case 'trust_score_updated':
          if (event.data.previousScore !== undefined && event.data.trustScore !== undefined) {
            const change = event.data.trustScore - event.data.previousScore;
            if (Math.abs(change) >= 0.1) {
              addNotification({
                type: 'trust_score_change',
                title: 'Trust Score Changed',
                message: `Trust with ${event.data.subjectDid?.slice(-12)} ${change > 0 ? 'increased' : 'decreased'} to ${Math.round(event.data.trustScore * 100)}%`,
                actionable: false,
                data: {
                  contactDid: event.data.subjectDid,
                  trustScore: event.data.trustScore,
                  previousScore: event.data.previousScore,
                },
              });
            }
          }
          queryClient.invalidateQueries({ queryKey: ['trust'] });
          break;

        case 'introduction_received':
          addNotification({
            type: 'introduction_request',
            title: 'Introduction Request',
            message: `${event.data.introducerName || event.data.introducerDid?.slice(-12)} wants to introduce you to someone`,
            actionable: true,
            data: {
              introductionId: event.data.introductionId,
              contactDid: event.data.introducerDid,
              contactName: event.data.introducerName,
            },
          });
          queryClient.invalidateQueries({ queryKey: ['introductions'] });
          break;

        case 'introduction_accepted':
          addNotification({
            type: 'introduction_accepted',
            title: 'Introduction Accepted',
            message: 'Your introduction was accepted',
            actionable: false,
          });
          queryClient.invalidateQueries({ queryKey: ['trust'] });
          queryClient.invalidateQueries({ queryKey: ['introductions'] });
          break;

        case 'credential_verified':
          addNotification({
            type: 'credential_verified',
            title: 'Credential Verified',
            message: `Your ${event.data.credentialType} has been verified`,
            actionable: false,
            data: {
              credentialType: event.data.credentialType,
            },
          });
          queryClient.invalidateQueries({ queryKey: ['credentials'] });
          break;

        case 'credential_expired':
          addNotification({
            type: 'credential_expired',
            title: 'Credential Expiring',
            message: `Your ${event.data.credentialType} will expire soon`,
            actionable: true,
            data: {
              credentialType: event.data.credentialType,
            },
          });
          break;

        case 'network_sync':
          if (event.data.newConnections && event.data.newConnections > 0) {
            addNotification({
              type: 'network_update',
              title: 'Network Updated',
              message: `${event.data.newConnections} new connection${event.data.newConnections > 1 ? 's' : ''} in your trust network`,
              actionable: false,
            });
          }
          queryClient.invalidateQueries({ queryKey: ['trust'] });
          break;
      }
    },
    [addNotification, onEvent, queryClient]
  );

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled || !userDid) return;

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/trust?did=${encodeURIComponent(userDid)}`;

    setIsConnecting(true);
    setError(null);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
        reconnectAttemptsRef.current = 0;
        console.log('[TrustWS] Connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TrustEvent;
          handleEvent(data);
        } catch (err) {
          console.error('[TrustWS] Failed to parse message:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('[TrustWS] Error:', event);
        setError(new Error('WebSocket connection error'));
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null;
        console.log('[TrustWS] Disconnected:', event.code, event.reason);

        // Attempt reconnect if not intentional close
        if (event.code !== 1000 && enabled) {
          if (reconnectAttemptsRef.current < maxReconnectAttempts) {
            reconnectAttemptsRef.current++;
            console.log(`[TrustWS] Reconnecting in ${reconnectInterval}ms (attempt ${reconnectAttemptsRef.current})`);
            reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
          } else {
            setError(new Error('Max reconnect attempts reached'));
          }
        }
      };
    } catch (err) {
      setIsConnecting(false);
      setError(err instanceof Error ? err : new Error('Failed to connect'));
    }
  }, [enabled, userDid, handleEvent, reconnectInterval, maxReconnectAttempts]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect');
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  // Reconnect handler
  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect, disconnect]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    error,
    lastEvent,
    reconnect,
    disconnect,
  };
}

// Connection status indicator component
export function TrustConnectionStatus({
  isConnected,
  isConnecting,
  error,
  onReconnect,
}: {
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  onReconnect: () => void;
}) {
  if (isConnecting) {
    return (
      <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
        <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
        <span>Connecting...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400">
        <div className="w-2 h-2 rounded-full bg-red-500" />
        <span>Disconnected</span>
        <button
          onClick={onReconnect}
          className="underline hover:no-underline"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isConnected) {
    return (
      <div className="flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400">
        <div className="w-2 h-2 rounded-full bg-emerald-500" />
        <span>Live</span>
      </div>
    );
  }

  return null;
}

export default useTrustWebSocket;
