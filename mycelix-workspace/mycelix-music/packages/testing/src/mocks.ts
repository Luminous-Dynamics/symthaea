// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mock Service Worker Handlers
 *
 * API mocks for testing without real backend.
 */

import { http, HttpResponse } from 'msw';
import {
  createSong,
  createSongs,
  createPlaylist,
  createDashboardData,
  createSearchResults,
  createUser,
} from './fixtures';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// ==================== Songs Handlers ====================

export const songHandlers = [
  // Get trending songs
  http.get(`${API_BASE}/api/v2/songs/trending`, ({ request }) => {
    const url = new URL(request.url);
    const limit = parseInt(url.searchParams.get('limit') || '20', 10);
    return HttpResponse.json({
      success: true,
      data: createSongs(limit),
    });
  }),

  // Get new releases
  http.get(`${API_BASE}/api/v2/songs/new`, ({ request }) => {
    const url = new URL(request.url);
    const limit = parseInt(url.searchParams.get('limit') || '20', 10);
    return HttpResponse.json({
      success: true,
      data: createSongs(limit),
    });
  }),

  // Get single song
  http.get(`${API_BASE}/api/v2/songs/:id`, ({ params }) => {
    return HttpResponse.json({
      success: true,
      data: createSong({ id: params.id as string }),
    });
  }),

  // Record play
  http.post(`${API_BASE}/api/v2/plays`, () => {
    return HttpResponse.json({ success: true });
  }),

  // Upload song
  http.post(`${API_BASE}/api/v2/songs/upload`, () => {
    return HttpResponse.json({
      success: true,
      data: {
        songId: 'new-song-id',
        status: 'processing',
      },
    });
  }),
];

// ==================== Recommendations Handlers ====================

export const recommendationHandlers = [
  http.get(`${API_BASE}/api/v2/recommendations`, ({ request }) => {
    const url = new URL(request.url);
    const limit = parseInt(url.searchParams.get('limit') || '20', 10);
    return HttpResponse.json({
      success: true,
      data: createSongs(limit),
    });
  }),

  http.get(`${API_BASE}/api/v2/recommendations/similar/:id`, () => {
    return HttpResponse.json({
      success: true,
      data: createSongs(10),
    });
  }),
];

// ==================== Playlist Handlers ====================

export const playlistHandlers = [
  // Get user playlists
  http.get(`${API_BASE}/api/v2/playlists/user`, () => {
    return HttpResponse.json({
      success: true,
      data: Array.from({ length: 5 }, () => createPlaylist()),
    });
  }),

  // Get single playlist
  http.get(`${API_BASE}/api/v2/playlists/:id`, ({ params }) => {
    return HttpResponse.json({
      success: true,
      data: {
        ...createPlaylist({ id: params.id as string }),
        songs: createSongs(10),
      },
    });
  }),

  // Create playlist
  http.post(`${API_BASE}/api/v2/playlists`, async ({ request }) => {
    const body = await request.json() as { name: string };
    return HttpResponse.json({
      success: true,
      data: createPlaylist({ name: body.name }),
    });
  }),

  // Update playlist
  http.patch(`${API_BASE}/api/v2/playlists/:id`, async ({ params, request }) => {
    const body = await request.json() as { name?: string; description?: string };
    return HttpResponse.json({
      success: true,
      data: createPlaylist({ id: params.id as string, ...body }),
    });
  }),

  // Delete playlist
  http.delete(`${API_BASE}/api/v2/playlists/:id`, () => {
    return HttpResponse.json({ success: true });
  }),

  // Add song to playlist
  http.post(`${API_BASE}/api/v2/playlists/:id/songs`, () => {
    return HttpResponse.json({ success: true });
  }),

  // Remove song from playlist
  http.delete(`${API_BASE}/api/v2/playlists/:playlistId/songs/:songId`, () => {
    return HttpResponse.json({ success: true });
  }),
];

// ==================== Artist Handlers ====================

export const artistHandlers = [
  // Get artist
  http.get(`${API_BASE}/api/v2/artists/:address`, ({ params }) => {
    const user = createUser({ address: params.address as string, isArtist: true });
    return HttpResponse.json({
      success: true,
      data: {
        address: user.address,
        name: user.displayName,
        avatar: user.avatar,
        bio: 'Multi-genre artist pushing the boundaries of sound.',
        followerCount: 12500,
        totalStreams: 500000,
        isFollowing: false,
        isVerified: true,
      },
    });
  }),

  // Get artist songs
  http.get(`${API_BASE}/api/v2/artists/:address/songs`, () => {
    return HttpResponse.json({
      success: true,
      data: createSongs(10),
    });
  }),

  // Follow artist
  http.post(`${API_BASE}/api/v2/artists/:address/follow`, () => {
    return HttpResponse.json({ success: true });
  }),

  // Unfollow artist
  http.delete(`${API_BASE}/api/v2/artists/:address/follow`, () => {
    return HttpResponse.json({ success: true });
  }),
];

// ==================== Social Handlers ====================

export const socialHandlers = [
  // Like song
  http.post(`${API_BASE}/api/v2/social/like`, () => {
    return HttpResponse.json({ success: true });
  }),

  // Get liked songs
  http.get(`${API_BASE}/api/v2/social/likes`, () => {
    return HttpResponse.json({
      success: true,
      data: createSongs(20),
    });
  }),

  // Get following
  http.get(`${API_BASE}/api/v2/social/following`, () => {
    return HttpResponse.json({
      success: true,
      data: Array.from({ length: 10 }, () => {
        const user = createUser({ isArtist: true });
        return {
          address: user.address,
          name: user.displayName,
          avatar: user.avatar,
          followerCount: 5000,
        };
      }),
    });
  }),

  // Get feed
  http.get(`${API_BASE}/api/v2/social/feed`, () => {
    return HttpResponse.json({
      success: true,
      data: [],
    });
  }),
];

// ==================== Search Handlers ====================

export const searchHandlers = [
  http.get(`${API_BASE}/api/v2/search`, ({ request }) => {
    const url = new URL(request.url);
    const query = url.searchParams.get('q') || '';

    if (query.length < 2) {
      return HttpResponse.json({
        success: true,
        data: { songs: [], artists: [], playlists: [] },
      });
    }

    return HttpResponse.json({
      success: true,
      data: createSearchResults(),
    });
  }),
];

// ==================== Dashboard Handlers ====================

export const dashboardHandlers = [
  http.get(`${API_BASE}/api/v2/dashboard`, () => {
    return HttpResponse.json({
      success: true,
      data: createDashboardData(),
    });
  }),

  http.get(`${API_BASE}/api/v2/dashboard/songs`, () => {
    return HttpResponse.json({
      success: true,
      data: createSongs(10),
    });
  }),
];

// ==================== Profile Handlers ====================

export const profileHandlers = [
  http.get(`${API_BASE}/api/v2/profile`, () => {
    const user = createUser();
    return HttpResponse.json({
      success: true,
      data: {
        address: user.address,
        displayName: user.displayName,
        avatar: user.avatar,
        bio: 'Music lover and creator.',
        email: user.email,
        isArtist: user.isArtist,
      },
    });
  }),

  http.patch(`${API_BASE}/api/v2/profile`, async ({ request }) => {
    const body = await request.json();
    const user = createUser();
    return HttpResponse.json({
      success: true,
      data: { ...user, ...body },
    });
  }),

  http.get(`${API_BASE}/api/v2/profile/stats`, () => {
    return HttpResponse.json({
      success: true,
      data: {
        songsPlayed: 1234,
        likedSongs: 89,
        following: 45,
        playlists: 12,
      },
    });
  }),

  http.post(`${API_BASE}/api/v2/profile/image`, () => {
    return HttpResponse.json({
      success: true,
      data: { url: 'https://cdn.mycelix.io/avatars/new-avatar.jpg' },
    });
  }),
];

// ==================== Settings Handlers ====================

export const settingsHandlers = [
  http.get(`${API_BASE}/api/v2/settings`, () => {
    return HttpResponse.json({
      success: true,
      data: {
        isArtist: true,
        notifyNewFollowers: true,
        notifyNewReleases: true,
        notifyEarnings: true,
        emailNotifications: false,
        pushNotifications: true,
        audioQuality: 'high',
        normalizeVolume: true,
        crossfade: false,
        autoplay: true,
        theme: 'dark',
        accentColor: 'purple',
        privateProfile: false,
        hideRecentlyPlayed: false,
        allowExplicit: true,
      },
    });
  }),

  http.patch(`${API_BASE}/api/v2/settings`, async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      success: true,
      data: body,
    });
  }),
];

// ==================== All Handlers ====================

export const handlers = [
  ...songHandlers,
  ...recommendationHandlers,
  ...playlistHandlers,
  ...artistHandlers,
  ...socialHandlers,
  ...searchHandlers,
  ...dashboardHandlers,
  ...profileHandlers,
  ...settingsHandlers,
];
