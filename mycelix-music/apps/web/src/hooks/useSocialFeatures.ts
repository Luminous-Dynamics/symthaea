// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Social Features Hook
 *
 * Community and social features:
 * - Artist collaboration matching
 * - Remix challenges/competitions
 * - Tipping and micro-payments
 * - Social interactions
 */

import { useState, useCallback, useEffect, useRef } from 'react';

// Types
export interface UserProfile {
  id: string;
  username: string;
  displayName: string;
  avatar?: string;
  bio?: string;
  genres: string[];
  skills: ArtistSkill[];
  instruments: string[];
  daw: string[];
  collaborationStatus: 'open' | 'busy' | 'closed';
  followers: number;
  following: number;
  tracks: number;
  collabs: number;
  verified: boolean;
  socialLinks?: {
    twitter?: string;
    instagram?: string;
    soundcloud?: string;
    spotify?: string;
  };
}

export type ArtistSkill =
  | 'production'
  | 'mixing'
  | 'mastering'
  | 'vocals'
  | 'songwriting'
  | 'djing'
  | 'sound-design'
  | 'beatmaking'
  | 'arrangement'
  | 'live-performance';

export interface CollaborationMatch {
  user: UserProfile;
  matchScore: number;
  matchReasons: string[];
  sharedGenres: string[];
  complementarySkills: ArtistSkill[];
  mutualConnections: number;
}

export interface CollaborationRequest {
  id: string;
  from: UserProfile;
  to: UserProfile;
  message: string;
  projectIdea?: string;
  status: 'pending' | 'accepted' | 'declined' | 'expired';
  createdAt: Date;
  expiresAt: Date;
}

export interface Challenge {
  id: string;
  title: string;
  description: string;
  type: 'remix' | 'beat-battle' | 'sample-flip' | 'original';
  status: 'upcoming' | 'active' | 'voting' | 'ended';
  host: UserProfile;
  startsAt: Date;
  endsAt: Date;
  votingEndsAt?: Date;
  prizes?: Prize[];
  rules: string[];
  samplePackUrl?: string;
  stemPackUrl?: string;
  submissions: ChallengeSubmission[];
  participantCount: number;
  featured: boolean;
}

export interface Prize {
  place: number;
  description: string;
  value?: number;
  currency?: string;
  type: 'cash' | 'gear' | 'exposure' | 'nft' | 'other';
}

export interface ChallengeSubmission {
  id: string;
  user: UserProfile;
  trackUrl: string;
  title: string;
  description?: string;
  votes: number;
  submittedAt: Date;
  place?: number;
}

export interface Tip {
  id: string;
  from: UserProfile;
  to: UserProfile;
  amount: number;
  currency: string;
  message?: string;
  trackId?: string;
  createdAt: Date;
  status: 'pending' | 'completed' | 'failed';
  transactionHash?: string;
}

export interface SocialInteraction {
  id: string;
  type: 'like' | 'repost' | 'comment' | 'follow' | 'collab-request' | 'tip';
  from: UserProfile;
  targetType: 'track' | 'user' | 'playlist' | 'challenge';
  targetId: string;
  message?: string;
  createdAt: Date;
}

export interface Comment {
  id: string;
  user: UserProfile;
  text: string;
  timestamp?: number;  // For track comments at specific time
  createdAt: Date;
  likes: number;
  replies: Comment[];
  isLikedByMe: boolean;
}

export interface SocialFeaturesState {
  isLoading: boolean;
  collaborationMatches: CollaborationMatch[];
  incomingRequests: CollaborationRequest[];
  outgoingRequests: CollaborationRequest[];
  activeChallenges: Challenge[];
  myChallenges: Challenge[];
  recentTips: Tip[];
  feed: SocialInteraction[];
  error: string | null;
}

// Collaboration matching weights
const MATCH_WEIGHTS = {
  sharedGenres: 30,
  complementarySkills: 25,
  mutualConnections: 20,
  activityLevel: 15,
  responseRate: 10,
};

export function useSocialFeatures(userId?: string) {
  const [state, setState] = useState<SocialFeaturesState>({
    isLoading: false,
    collaborationMatches: [],
    incomingRequests: [],
    outgoingRequests: [],
    activeChallenges: [],
    myChallenges: [],
    recentTips: [],
    feed: [],
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);

  /**
   * Find collaboration matches
   */
  const findCollaborators = useCallback(async (filters?: {
    genres?: string[];
    skills?: ArtistSkill[];
    minMatchScore?: number;
  }): Promise<CollaborationMatch[]> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Simulated API call
      await new Promise(resolve => setTimeout(resolve, 800));

      const matches: CollaborationMatch[] = [
        {
          user: {
            id: '1',
            username: 'beatmaker_pro',
            displayName: 'BeatMaker Pro',
            genres: ['hiphop', 'trap', 'lofi'],
            skills: ['beatmaking', 'production', 'mixing'],
            instruments: ['mpc', 'keys'],
            daw: ['ableton', 'fl-studio'],
            collaborationStatus: 'open',
            followers: 5420,
            following: 312,
            tracks: 89,
            collabs: 12,
            verified: true,
          },
          matchScore: 92,
          matchReasons: ['Similar style', 'Complementary skills', 'Active collaborator'],
          sharedGenres: ['hiphop', 'lofi'],
          complementarySkills: ['vocals', 'songwriting'],
          mutualConnections: 8,
        },
        {
          user: {
            id: '2',
            username: 'synth_wizard',
            displayName: 'Synth Wizard',
            genres: ['electronic', 'synthwave', 'ambient'],
            skills: ['sound-design', 'production', 'arrangement'],
            instruments: ['synthesizer', 'modular'],
            daw: ['ableton'],
            collaborationStatus: 'open',
            followers: 3210,
            following: 156,
            tracks: 45,
            collabs: 7,
            verified: false,
          },
          matchScore: 85,
          matchReasons: ['Unique sound palette', 'Looking for vocalists'],
          sharedGenres: ['electronic'],
          complementarySkills: ['mixing', 'mastering'],
          mutualConnections: 3,
        },
      ];

      setState(prev => ({
        ...prev,
        isLoading: false,
        collaborationMatches: matches,
      }));

      return matches;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to find collaborators',
      }));
      return [];
    }
  }, []);

  /**
   * Send collaboration request
   */
  const sendCollabRequest = useCallback(async (
    toUserId: string,
    message: string,
    projectIdea?: string
  ): Promise<CollaborationRequest | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const request: CollaborationRequest = {
        id: `req-${Date.now()}`,
        from: { id: userId || 'me' } as UserProfile,
        to: { id: toUserId } as UserProfile,
        message,
        projectIdea,
        status: 'pending',
        createdAt: new Date(),
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        outgoingRequests: [...prev.outgoingRequests, request],
      }));

      return request;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to send request' }));
      return null;
    }
  }, [userId]);

  /**
   * Respond to collaboration request
   */
  const respondToRequest = useCallback(async (
    requestId: string,
    accept: boolean,
    message?: string
  ): Promise<boolean> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 300));

      setState(prev => ({
        ...prev,
        incomingRequests: prev.incomingRequests.map(r =>
          r.id === requestId ? { ...r, status: accept ? 'accepted' : 'declined' } : r
        ),
      }));

      return true;
    } catch (error) {
      return false;
    }
  }, []);

  /**
   * Get active challenges
   */
  const getChallenges = useCallback(async (filters?: {
    type?: Challenge['type'];
    status?: Challenge['status'];
    featured?: boolean;
  }): Promise<Challenge[]> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 600));

      const challenges: Challenge[] = [
        {
          id: 'challenge-1',
          title: 'Lo-Fi Flip Challenge',
          description: 'Flip the provided sample into a chill lo-fi beat',
          type: 'sample-flip',
          status: 'active',
          host: { id: 'host-1', username: 'lofi_central', displayName: 'Lo-Fi Central' } as UserProfile,
          startsAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
          endsAt: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000),
          prizes: [
            { place: 1, description: 'Featured on playlist', type: 'exposure' },
            { place: 2, description: '$50 plugin voucher', value: 50, currency: 'USD', type: 'gear' },
          ],
          rules: ['Use only provided sample', 'BPM: 70-90', 'Max length: 3 minutes'],
          samplePackUrl: '/samples/lofi-challenge.zip',
          submissions: [],
          participantCount: 127,
          featured: true,
        },
        {
          id: 'challenge-2',
          title: 'Beat Battle #42',
          description: 'Create the hardest trap beat from scratch',
          type: 'beat-battle',
          status: 'active',
          host: { id: 'host-2', username: 'beat_arena', displayName: 'Beat Arena' } as UserProfile,
          startsAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
          endsAt: new Date(Date.now() + 6 * 24 * 60 * 60 * 1000),
          prizes: [
            { place: 1, description: '0.1 ETH', value: 0.1, currency: 'ETH', type: 'cash' },
          ],
          rules: ['Original production only', 'Must include 808s', 'Max length: 2 minutes'],
          submissions: [],
          participantCount: 89,
          featured: false,
        },
      ];

      setState(prev => ({
        ...prev,
        isLoading: false,
        activeChallenges: challenges,
      }));

      return challenges;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to load challenges' }));
      return [];
    }
  }, []);

  /**
   * Submit to challenge
   */
  const submitToChallenge = useCallback(async (
    challengeId: string,
    trackUrl: string,
    title: string,
    description?: string
  ): Promise<ChallengeSubmission | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const submission: ChallengeSubmission = {
        id: `sub-${Date.now()}`,
        user: { id: userId || 'me' } as UserProfile,
        trackUrl,
        title,
        description,
        votes: 0,
        submittedAt: new Date(),
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        activeChallenges: prev.activeChallenges.map(c =>
          c.id === challengeId
            ? { ...c, submissions: [...c.submissions, submission], participantCount: c.participantCount + 1 }
            : c
        ),
      }));

      return submission;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to submit' }));
      return null;
    }
  }, [userId]);

  /**
   * Vote for submission
   */
  const voteForSubmission = useCallback(async (
    challengeId: string,
    submissionId: string
  ): Promise<boolean> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 200));

      setState(prev => ({
        ...prev,
        activeChallenges: prev.activeChallenges.map(c =>
          c.id === challengeId
            ? {
                ...c,
                submissions: c.submissions.map(s =>
                  s.id === submissionId ? { ...s, votes: s.votes + 1 } : s
                ),
              }
            : c
        ),
      }));

      return true;
    } catch (error) {
      return false;
    }
  }, []);

  /**
   * Send tip to artist
   */
  const sendTip = useCallback(async (
    toUserId: string,
    amount: number,
    currency: string,
    message?: string,
    trackId?: string
  ): Promise<Tip | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const tip: Tip = {
        id: `tip-${Date.now()}`,
        from: { id: userId || 'me' } as UserProfile,
        to: { id: toUserId } as UserProfile,
        amount,
        currency,
        message,
        trackId,
        createdAt: new Date(),
        status: 'completed',
        transactionHash: `0x${Math.random().toString(16).slice(2, 66)}`,
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        recentTips: [tip, ...prev.recentTips],
      }));

      return tip;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Tip failed' }));
      return null;
    }
  }, [userId]);

  /**
   * Follow user
   */
  const followUser = useCallback(async (targetUserId: string): Promise<boolean> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 200));

      const interaction: SocialInteraction = {
        id: `int-${Date.now()}`,
        type: 'follow',
        from: { id: userId || 'me' } as UserProfile,
        targetType: 'user',
        targetId: targetUserId,
        createdAt: new Date(),
      };

      setState(prev => ({
        ...prev,
        feed: [interaction, ...prev.feed],
      }));

      return true;
    } catch (error) {
      return false;
    }
  }, [userId]);

  /**
   * Like track
   */
  const likeTrack = useCallback(async (trackId: string): Promise<boolean> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 100));

      const interaction: SocialInteraction = {
        id: `int-${Date.now()}`,
        type: 'like',
        from: { id: userId || 'me' } as UserProfile,
        targetType: 'track',
        targetId: trackId,
        createdAt: new Date(),
      };

      setState(prev => ({
        ...prev,
        feed: [interaction, ...prev.feed],
      }));

      return true;
    } catch (error) {
      return false;
    }
  }, [userId]);

  /**
   * Repost track
   */
  const repostTrack = useCallback(async (trackId: string, message?: string): Promise<boolean> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 200));

      const interaction: SocialInteraction = {
        id: `int-${Date.now()}`,
        type: 'repost',
        from: { id: userId || 'me' } as UserProfile,
        targetType: 'track',
        targetId: trackId,
        message,
        createdAt: new Date(),
      };

      setState(prev => ({
        ...prev,
        feed: [interaction, ...prev.feed],
      }));

      return true;
    } catch (error) {
      return false;
    }
  }, [userId]);

  /**
   * Comment on track
   */
  const commentOnTrack = useCallback(async (
    trackId: string,
    text: string,
    timestamp?: number
  ): Promise<Comment | null> => {
    try {
      await new Promise(resolve => setTimeout(resolve, 300));

      const comment: Comment = {
        id: `comment-${Date.now()}`,
        user: { id: userId || 'me' } as UserProfile,
        text,
        timestamp,
        createdAt: new Date(),
        likes: 0,
        replies: [],
        isLikedByMe: false,
      };

      return comment;
    } catch (error) {
      return null;
    }
  }, [userId]);

  /**
   * Get social feed
   */
  const getFeed = useCallback(async (limit: number = 50): Promise<SocialInteraction[]> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 400));

      // Simulated feed
      const feed: SocialInteraction[] = [];

      setState(prev => ({
        ...prev,
        isLoading: false,
        feed,
      }));

      return feed;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to load feed' }));
      return [];
    }
  }, []);

  return {
    ...state,
    findCollaborators,
    sendCollabRequest,
    respondToRequest,
    getChallenges,
    submitToChallenge,
    voteForSubmission,
    sendTip,
    followUser,
    likeTrack,
    repostTrack,
    commentOnTrack,
    getFeed,
  };
}

export default useSocialFeatures;
