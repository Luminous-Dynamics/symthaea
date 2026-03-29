// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useCollaboration Hook
 *
 * Real-time collaboration for the Creation Studio.
 * Handles WebSocket connection and state synchronization.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from './useAuth';

interface Participant {
  id: string;
  name: string;
  color: string;
  cursor?: { stemId: string; position: number };
}

interface CollaborationEvent {
  type: string;
  payload: any;
  senderId: string;
  timestamp: number;
}

interface UseCollaborationReturn {
  participants: Participant[];
  isConnected: boolean;
  connectionError: string | null;
  sendMessage: (message: { type: string; payload: any }) => void;
  lastEvent: CollaborationEvent | null;
  updateCursor: (cursor: { stemId: string; position: number }) => void;
}

export function useCollaboration(projectId: string): UseCollaborationReturn {
  const { user } = useAuth();
  const [participants, setParticipants] = useState<Participant[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [lastEvent, setLastEvent] = useState<CollaborationEvent | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    if (!user) return;

    const wsUrl = new URL(process.env.NEXT_PUBLIC_COLLABORATION_WS_URL || 'ws://localhost:8081');
    wsUrl.searchParams.set('projectId', projectId);
    wsUrl.searchParams.set('userId', user.id);
    wsUrl.searchParams.set('userName', user.displayName || 'Anonymous');

    const ws = new WebSocket(wsUrl.toString());
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setConnectionError(null);
      reconnectAttemptsRef.current = 0;
      console.log('Collaboration connected');
    };

    ws.onmessage = (event) => {
      try {
        const message: CollaborationEvent = JSON.parse(event.data);

        switch (message.type) {
          case 'room:joined':
            setParticipants(message.payload.participants);
            break;

          case 'participant:joined':
            setParticipants((prev) => [...prev, message.payload]);
            break;

          case 'participant:left':
            setParticipants((prev) =>
              prev.filter((p) => p.id !== message.payload.id)
            );
            break;

          case 'cursor:moved':
            setParticipants((prev) =>
              prev.map((p) =>
                p.id === message.payload.userId
                  ? { ...p, cursor: message.payload.cursor }
                  : p
              )
            );
            break;

          default:
            setLastEvent(message);
        }
      } catch (e) {
        console.error('Failed to parse collaboration message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('Collaboration WebSocket error:', error);
      setConnectionError('Connection error');
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;

      // Attempt reconnection with exponential backoff
      const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
      reconnectAttemptsRef.current++;

      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('Attempting to reconnect...');
        connect();
      }, delay);
    };
  }, [projectId, user]);

  // Connect on mount
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Send message
  const sendMessage = useCallback((message: { type: string; payload: any }) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && user) {
      wsRef.current.send(JSON.stringify({
        ...message,
        senderId: user.id,
        timestamp: Date.now(),
      }));
    }
  }, [user]);

  // Update cursor position
  const updateCursor = useCallback((cursor: { stemId: string; position: number }) => {
    sendMessage({ type: 'cursor:move', payload: cursor });
  }, [sendMessage]);

  return {
    participants,
    isConnected,
    connectionError,
    sendMessage,
    lastEvent,
    updateCursor,
  };
}
