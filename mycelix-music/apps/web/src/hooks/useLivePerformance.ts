// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Live Performance Hook
 *
 * Virtual concert and live performance features:
 * - Virtual concert platform
 * - Crowd interaction
 * - Multi-performer sync
 * - Ticketed events
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Types
export interface VirtualConcert {
  id: string;
  title: string;
  description: string;
  artist: ArtistInfo;
  venue: VirtualVenue;
  scheduledAt: Date;
  duration: number;  // minutes
  status: ConcertStatus;
  ticketTiers: TicketTier[];
  maxAttendees: number;
  currentAttendees: number;
  features: ConcertFeature[];
  chatEnabled: boolean;
  reactionsEnabled: boolean;
  recordingAvailable: boolean;
}

export type ConcertStatus = 'scheduled' | 'live' | 'ended' | 'cancelled';

export interface ArtistInfo {
  id: string;
  name: string;
  avatar: string;
  verified: boolean;
  genre: string;
  followers: number;
}

export interface VirtualVenue {
  id: string;
  name: string;
  type: 'club' | 'arena' | 'stadium' | 'intimate' | 'festival' | 'custom';
  capacity: number;
  theme: string;
  backgroundUrl?: string;
  model3dUrl?: string;
  ambiance: {
    lighting: 'dynamic' | 'static';
    effects: string[];
    interactiveElements: string[];
  };
}

export interface TicketTier {
  id: string;
  name: string;
  price: number;
  currency: string;
  perks: string[];
  maxQuantity: number;
  soldCount: number;
  accessLevel: 'general' | 'vip' | 'backstage' | 'meet-greet';
}

export type ConcertFeature =
  | 'vr-support'
  | 'spatial-audio'
  | 'chat'
  | 'reactions'
  | 'song-requests'
  | 'tipping'
  | 'merchandise'
  | 'recording'
  | 'multi-cam'
  | 'backstage';

export interface Attendee {
  id: string;
  username: string;
  avatar?: string;
  ticketTier: string;
  position?: { x: number; y: number; z: number };
  joinedAt: Date;
  reactions: number;
  tipped: number;
}

export interface CrowdReaction {
  type: 'cheer' | 'clap' | 'heart' | 'fire' | 'wave' | 'lighter' | 'custom';
  count: number;
  intensity: number;  // 0-1
  position?: { x: number; y: number };
}

export interface SongRequest {
  id: string;
  songTitle: string;
  requestedBy: Attendee;
  votes: number;
  status: 'pending' | 'accepted' | 'played' | 'declined';
  requestedAt: Date;
}

export interface PerformerState {
  id: string;
  name: string;
  role: 'lead' | 'support' | 'dj' | 'vj';
  isLive: boolean;
  audioTrack: number;
  videoFeed?: string;
  instruments: string[];
  position: { x: number; y: number; z: number };
}

export interface MultiPerformerSession {
  id: string;
  performers: PerformerState[];
  syncLatency: number;
  masterClock: number;
  chatEnabled: boolean;
}

export interface ConcertAnalytics {
  peakAttendees: number;
  totalReactions: number;
  totalTips: number;
  averageWatchTime: number;
  engagementRate: number;
  topSongRequests: SongRequest[];
  geographicDistribution: { country: string; count: number }[];
  revenueBreakdown: {
    tickets: number;
    tips: number;
    merchandise: number;
  };
}

export interface LivePerformanceState {
  isInitialized: boolean;
  currentConcert: VirtualConcert | null;
  isPerforming: boolean;
  isAttending: boolean;
  attendees: Attendee[];
  reactions: CrowdReaction[];
  songRequests: SongRequest[];
  performerSession: MultiPerformerSession | null;
  chatMessages: ChatMessage[];
  analytics: ConcertAnalytics | null;
  error: string | null;
}

export interface ChatMessage {
  id: string;
  user: { id: string; username: string; avatar?: string };
  text: string;
  timestamp: Date;
  type: 'message' | 'tip' | 'reaction' | 'system';
  metadata?: any;
}

export function useLivePerformance() {
  const [state, setState] = useState<LivePerformanceState>({
    isInitialized: false,
    currentConcert: null,
    isPerforming: false,
    isAttending: false,
    attendees: [],
    reactions: [],
    songRequests: [],
    performerSession: null,
    chatMessages: [],
    analytics: null,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const peerConnectionsRef = useRef<Map<string, RTCPeerConnection>>(new Map());

  /**
   * Create a new virtual concert
   */
  const createConcert = useCallback(async (
    config: {
      title: string;
      description: string;
      scheduledAt: Date;
      duration: number;
      venue: VirtualVenue;
      ticketTiers: Omit<TicketTier, 'id' | 'soldCount'>[];
      features: ConcertFeature[];
    }
  ): Promise<VirtualConcert | null> => {
    try {
      const concert: VirtualConcert = {
        id: `concert-${Date.now()}`,
        title: config.title,
        description: config.description,
        artist: {
          id: 'current-user',
          name: 'Artist Name',
          avatar: '/avatar.jpg',
          verified: true,
          genre: 'Electronic',
          followers: 10000,
        },
        venue: config.venue,
        scheduledAt: config.scheduledAt,
        duration: config.duration,
        status: 'scheduled',
        ticketTiers: config.ticketTiers.map((tier, i) => ({
          ...tier,
          id: `tier-${i}`,
          soldCount: 0,
        })),
        maxAttendees: config.venue.capacity,
        currentAttendees: 0,
        features: config.features,
        chatEnabled: config.features.includes('chat'),
        reactionsEnabled: config.features.includes('reactions'),
        recordingAvailable: config.features.includes('recording'),
      };

      return concert;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to create concert',
      }));
      return null;
    }
  }, []);

  /**
   * Start performing (go live)
   */
  const startPerformance = useCallback(async (
    concertId: string
  ): Promise<boolean> => {
    try {
      // Get media stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: true,
      });

      mediaStreamRef.current = stream;

      // Connect to signaling server
      wsRef.current = new WebSocket(`wss://live.mycelix.music/perform/${concertId}`);

      wsRef.current.onopen = () => {
        wsRef.current?.send(JSON.stringify({
          type: 'start-broadcast',
          streamId: stream.id,
        }));
      };

      wsRef.current.onmessage = (event) => {
        handleSignalingMessage(JSON.parse(event.data));
      };

      setState(prev => ({
        ...prev,
        isPerforming: true,
        currentConcert: prev.currentConcert
          ? { ...prev.currentConcert, status: 'live' }
          : null,
      }));

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to start performance',
      }));
      return false;
    }
  }, []);

  /**
   * Stop performing
   */
  const stopPerformance = useCallback(async (): Promise<boolean> => {
    try {
      // Stop media tracks
      mediaStreamRef.current?.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;

      // Close WebSocket
      wsRef.current?.close();
      wsRef.current = null;

      // Close peer connections
      peerConnectionsRef.current.forEach(pc => pc.close());
      peerConnectionsRef.current.clear();

      setState(prev => ({
        ...prev,
        isPerforming: false,
        currentConcert: prev.currentConcert
          ? { ...prev.currentConcert, status: 'ended' }
          : null,
      }));

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, error: 'Failed to stop performance' }));
      return false;
    }
  }, []);

  /**
   * Join concert as attendee
   */
  const joinConcert = useCallback(async (
    concertId: string,
    ticketId: string
  ): Promise<boolean> => {
    try {
      // Verify ticket
      const verified = await verifyTicket(concertId, ticketId);
      if (!verified) {
        throw new Error('Invalid or expired ticket');
      }

      // Connect to concert stream
      wsRef.current = new WebSocket(`wss://live.mycelix.music/watch/${concertId}`);

      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleConcertMessage(message);
      };

      setState(prev => ({
        ...prev,
        isAttending: true,
      }));

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to join concert',
      }));
      return false;
    }
  }, []);

  /**
   * Leave concert
   */
  const leaveConcert = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;

    setState(prev => ({
      ...prev,
      isAttending: false,
      currentConcert: null,
    }));
  }, []);

  /**
   * Send reaction
   */
  const sendReaction = useCallback((
    type: CrowdReaction['type']
  ) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({
      type: 'reaction',
      reactionType: type,
      timestamp: Date.now(),
    }));

    // Optimistic update
    setState(prev => {
      const existing = prev.reactions.find(r => r.type === type);
      if (existing) {
        return {
          ...prev,
          reactions: prev.reactions.map(r =>
            r.type === type ? { ...r, count: r.count + 1 } : r
          ),
        };
      }
      return {
        ...prev,
        reactions: [...prev.reactions, { type, count: 1, intensity: 0.5 }],
      };
    });
  }, []);

  /**
   * Request song
   */
  const requestSong = useCallback((songTitle: string): SongRequest | null => {
    if (!wsRef.current) return null;

    const request: SongRequest = {
      id: `req-${Date.now()}`,
      songTitle,
      requestedBy: {
        id: 'current-user',
        username: 'User',
        ticketTier: 'general',
        joinedAt: new Date(),
        reactions: 0,
        tipped: 0,
      },
      votes: 1,
      status: 'pending',
      requestedAt: new Date(),
    };

    wsRef.current.send(JSON.stringify({
      type: 'song-request',
      request,
    }));

    setState(prev => ({
      ...prev,
      songRequests: [...prev.songRequests, request],
    }));

    return request;
  }, []);

  /**
   * Vote for song request
   */
  const voteForSong = useCallback((requestId: string) => {
    if (!wsRef.current) return;

    wsRef.current.send(JSON.stringify({
      type: 'song-vote',
      requestId,
    }));

    setState(prev => ({
      ...prev,
      songRequests: prev.songRequests.map(r =>
        r.id === requestId ? { ...r, votes: r.votes + 1 } : r
      ),
    }));
  }, []);

  /**
   * Send chat message
   */
  const sendChatMessage = useCallback((text: string) => {
    if (!wsRef.current) return;

    const message: ChatMessage = {
      id: `msg-${Date.now()}`,
      user: { id: 'current-user', username: 'User' },
      text,
      timestamp: new Date(),
      type: 'message',
    };

    wsRef.current.send(JSON.stringify({
      type: 'chat',
      message,
    }));

    setState(prev => ({
      ...prev,
      chatMessages: [...prev.chatMessages, message],
    }));
  }, []);

  /**
   * Send tip
   */
  const sendTip = useCallback(async (
    amount: number,
    currency: string,
    message?: string
  ): Promise<boolean> => {
    if (!wsRef.current) return false;

    try {
      // Process payment
      const paymentResult = await processPayment(amount, currency);
      if (!paymentResult.success) {
        throw new Error('Payment failed');
      }

      wsRef.current.send(JSON.stringify({
        type: 'tip',
        amount,
        currency,
        message,
        transactionId: paymentResult.transactionId,
      }));

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, error: 'Tip failed' }));
      return false;
    }
  }, []);

  /**
   * Start multi-performer session
   */
  const startMultiPerformerSession = useCallback(async (
    performerIds: string[]
  ): Promise<MultiPerformerSession | null> => {
    try {
      const session: MultiPerformerSession = {
        id: `session-${Date.now()}`,
        performers: performerIds.map((id, i) => ({
          id,
          name: `Performer ${i + 1}`,
          role: i === 0 ? 'lead' : 'support',
          isLive: false,
          audioTrack: i,
          instruments: [],
          position: { x: i * 2, y: 0, z: 0 },
        })),
        syncLatency: 0,
        masterClock: Date.now(),
        chatEnabled: true,
      };

      setState(prev => ({ ...prev, performerSession: session }));
      return session;
    } catch (error) {
      setState(prev => ({ ...prev, error: 'Failed to start session' }));
      return null;
    }
  }, []);

  /**
   * Sync with other performers
   */
  const syncPerformers = useCallback(async (): Promise<number> => {
    if (!state.performerSession) return -1;

    // Calculate latency to each performer
    const latencies = await Promise.all(
      state.performerSession.performers.map(async (p) => {
        const start = Date.now();
        await fetch(`/api/performers/${p.id}/ping`);
        return Date.now() - start;
      })
    );

    const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;

    setState(prev => ({
      ...prev,
      performerSession: prev.performerSession
        ? { ...prev.performerSession, syncLatency: avgLatency }
        : null,
    }));

    return avgLatency;
  }, [state.performerSession]);

  /**
   * Get concert analytics
   */
  const getAnalytics = useCallback(async (
    concertId: string
  ): Promise<ConcertAnalytics | null> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const analytics: ConcertAnalytics = {
        peakAttendees: 1250,
        totalReactions: 15420,
        totalTips: 2340,
        averageWatchTime: 45,
        engagementRate: 0.78,
        topSongRequests: state.songRequests.slice(0, 5),
        geographicDistribution: [
          { country: 'US', count: 450 },
          { country: 'UK', count: 230 },
          { country: 'DE', count: 180 },
          { country: 'JP', count: 120 },
        ],
        revenueBreakdown: {
          tickets: 12500,
          tips: 2340,
          merchandise: 890,
        },
      };

      setState(prev => ({ ...prev, analytics }));
      return analytics;
    } catch (error) {
      setState(prev => ({ ...prev, error: 'Failed to load analytics' }));
      return null;
    }
  }, [state.songRequests]);

  /**
   * Handle signaling messages
   */
  const handleSignalingMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'attendee-joined':
        setState(prev => ({
          ...prev,
          attendees: [...prev.attendees, message.attendee],
        }));
        break;
      case 'attendee-left':
        setState(prev => ({
          ...prev,
          attendees: prev.attendees.filter(a => a.id !== message.attendeeId),
        }));
        break;
      case 'reaction-burst':
        setState(prev => ({
          ...prev,
          reactions: message.reactions,
        }));
        break;
    }
  }, []);

  /**
   * Handle concert messages
   */
  const handleConcertMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'chat':
        setState(prev => ({
          ...prev,
          chatMessages: [...prev.chatMessages, message.message],
        }));
        break;
      case 'reaction':
        // Update reaction counts
        break;
      case 'song-played':
        setState(prev => ({
          ...prev,
          songRequests: prev.songRequests.map(r =>
            r.id === message.requestId ? { ...r, status: 'played' } : r
          ),
        }));
        break;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      mediaStreamRef.current?.getTracks().forEach(track => track.stop());
      peerConnectionsRef.current.forEach(pc => pc.close());
    };
  }, []);

  return {
    ...state,
    createConcert,
    startPerformance,
    stopPerformance,
    joinConcert,
    leaveConcert,
    sendReaction,
    requestSong,
    voteForSong,
    sendChatMessage,
    sendTip,
    startMultiPerformerSession,
    syncPerformers,
    getAnalytics,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

async function verifyTicket(concertId: string, ticketId: string): Promise<boolean> {
  // Would verify ticket on backend
  return true;
}

async function processPayment(amount: number, currency: string): Promise<{
  success: boolean;
  transactionId?: string;
}> {
  // Would process payment
  return { success: true, transactionId: `tx-${Date.now()}` };
}

export default useLivePerformance;
