// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Client for Mycelix Backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export async function fetchJSON<T = any>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ==================== Types ====================

export interface Song {
  id: string;
  title: string;
  artist: string;
  artistAddress: string;
  artistAvatar?: string;
  album?: string;
  genre?: string;
  duration: number;
  coverArt: string;
  audioUrl: string;
  playCount: number;
  likeCount: number;
  isLiked?: boolean;
  bpm?: number;
  key?: string;
  lyrics?: string;
  releaseDate?: string;
  dominantColor?: string;
  nftTokenId?: string;
  contractAddress?: string;
  createdAt: string;
}

export interface Playlist {
  id: string;
  name: string;
  description?: string;
  coverImage?: string;
  ownerAddress: string;
  ownerName: string;
  songCount: number;
  duration: number;
  isPublic: boolean;
  followers?: number;
  dominantColor?: string;
  songs: Song[];
}

export interface Artist {
  address: string;
  name: string;
  avatar?: string;
  bannerImage?: string;
  bio?: string;
  followerCount: number;
  songCount: number;
  totalStreams?: number;
  isFollowing?: boolean;
  isVerified?: boolean;
  dominantColor?: string;
  socialLinks?: Record<string, string>;
}

export interface UserProfile {
  address: string;
  displayName?: string;
  avatar?: string;
  bannerImage?: string;
  bio?: string;
  email?: string;
  isArtist?: boolean;
  socialLinks?: Record<string, string>;
}

export interface ProfileStats {
  songsPlayed: number;
  likedSongs: number;
  following: number;
  playlists: number;
}

export interface UserSettings {
  isArtist: boolean;
  notifyNewFollowers: boolean;
  notifyNewReleases: boolean;
  notifyEarnings: boolean;
  emailNotifications: boolean;
  pushNotifications: boolean;
  audioQuality: 'low' | 'normal' | 'high' | 'lossless';
  normalizeVolume: boolean;
  crossfade: boolean;
  autoplay: boolean;
  theme: 'dark' | 'light' | 'system';
  accentColor: string;
  privateProfile: boolean;
  hideRecentlyPlayed: boolean;
  allowExplicit: boolean;
}

export interface Activity {
  id: string;
  type: string;
  actorName: string;
  actorAvatar?: string;
  targetTitle: string;
  targetImage?: string;
  createdAt: string;
}

export interface DashboardData {
  streams: { totalStreams: number; uniqueListeners: number };
  revenue: { totalRevenue: number; pendingPayout: number };
  engagement: { followers: number; followerGrowth: number };
  topSongs: Array<{ songId: string; title: string; streams: number }>;
  streamHistory: Array<{ date: string; value: number }>;
}

// ==================== API Client ====================

class ApiClient {
  private token: string | null = null;

  setToken(token: string | null) {
    this.token = token;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const headers: HeadersInit = { 'Content-Type': 'application/json', ...options.headers };
    if (this.token) (headers as Record<string, string>)['Authorization'] = `Bearer ${this.token}`;

    const response = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });
    const json = await response.json();
    if (!response.ok) throw new Error(json.error?.message || 'Request failed');
    return json.data;
  }

  // Songs
  async getSongs(params?: { limit?: number; genre?: string }): Promise<Song[]> {
    const query = new URLSearchParams(params as Record<string, string>);
    return this.request(`/api/v2/songs?${query}`);
  }

  async getSong(id: string): Promise<Song> {
    return this.request(`/api/v2/songs/${id}`);
  }

  async getTrendingSongs(limit = 20): Promise<Song[]> {
    return this.request(`/api/v2/songs/trending?limit=${limit}`);
  }

  async getNewReleases(limit = 20): Promise<Song[]> {
    return this.request(`/api/v2/songs/new?limit=${limit}`);
  }

  // Recommendations
  async getRecommendations(limit = 20): Promise<Song[]> {
    return this.request(`/api/v2/recommendations?limit=${limit}`);
  }

  async getSimilarSongs(songId: string): Promise<Song[]> {
    return this.request(`/api/v2/recommendations/similar/${songId}`);
  }

  // Player
  async recordPlay(songId: string, duration: number): Promise<void> {
    await this.request('/api/v2/plays', { method: 'POST', body: JSON.stringify({ songId, duration }) });
  }

  // Playlists
  async getPlaylists(): Promise<Playlist[]> {
    return this.request('/api/v2/playlists');
  }

  async createPlaylist(data: { name: string; description?: string }): Promise<Playlist> {
    return this.request('/api/v2/playlists', { method: 'POST', body: JSON.stringify(data) });
  }

  async addToPlaylist(playlistId: string, songId: string): Promise<void> {
    await this.request(`/api/v2/playlists/${playlistId}/songs`, { method: 'POST', body: JSON.stringify({ songId }) });
  }

  // Social
  async follow(address: string): Promise<void> {
    await this.request('/api/v2/social/follow', { method: 'POST', body: JSON.stringify({ address }) });
  }

  async likeSong(songId: string): Promise<void> {
    await this.request('/api/v2/social/like', { method: 'POST', body: JSON.stringify({ targetId: songId, targetType: 'song' }) });
  }

  async getFeed(): Promise<Activity[]> {
    return this.request('/api/v2/social/feed');
  }

  async getLikedSongs(): Promise<Song[]> {
    return this.request('/api/v2/social/likes');
  }

  // Search
  async search(query: string): Promise<{ songs: Song[]; artists: Artist[]; playlists: Playlist[] }> {
    return this.request(`/api/v2/search?q=${encodeURIComponent(query)}`);
  }

  // Dashboard
  async getDashboard(period = '30d'): Promise<DashboardData> {
    return this.request(`/api/v2/dashboard?period=${period}`);
  }

  async getArtistSongs(): Promise<Song[]> {
    return this.request('/api/v2/dashboard/songs');
  }

  // Playlist Management
  async getPlaylist(id: string): Promise<Playlist> {
    return this.request(`/api/v2/playlists/${id}`);
  }

  async getUserPlaylists(): Promise<Playlist[]> {
    return this.request('/api/v2/playlists/user');
  }

  async updatePlaylist(id: string, data: { name?: string; description?: string }): Promise<Playlist> {
    return this.request(`/api/v2/playlists/${id}`, { method: 'PATCH', body: JSON.stringify(data) });
  }

  async deletePlaylist(id: string): Promise<void> {
    await this.request(`/api/v2/playlists/${id}`, { method: 'DELETE' });
  }

  async removeFromPlaylist(playlistId: string, songId: string): Promise<void> {
    await this.request(`/api/v2/playlists/${playlistId}/songs/${songId}`, { method: 'DELETE' });
  }

  // Artists
  async getArtist(address: string): Promise<Artist> {
    return this.request(`/api/v2/artists/${address}`);
  }

  async getArtistPublicSongs(address: string): Promise<Song[]> {
    return this.request(`/api/v2/artists/${address}/songs`);
  }

  async followArtist(address: string): Promise<void> {
    await this.request(`/api/v2/artists/${address}/follow`, { method: 'POST' });
  }

  async unfollowArtist(address: string): Promise<void> {
    await this.request(`/api/v2/artists/${address}/follow`, { method: 'DELETE' });
  }

  async getFollowedArtists(): Promise<Artist[]> {
    return this.request('/api/v2/social/following');
  }

  // Profile
  async getProfile(): Promise<UserProfile> {
    return this.request('/api/v2/profile');
  }

  async getProfileStats(): Promise<ProfileStats> {
    return this.request('/api/v2/profile/stats');
  }

  async updateProfile(data: { displayName?: string; bio?: string; socialLinks?: Record<string, string> }): Promise<UserProfile> {
    return this.request('/api/v2/profile', { method: 'PATCH', body: JSON.stringify(data) });
  }

  async uploadProfileImage(file: File, type: 'avatar' | 'banner'): Promise<{ url: string }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    const headers: HeadersInit = {};
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

    const response = await fetch(`${API_BASE}/api/v2/profile/image`, {
      method: 'POST',
      headers,
      body: formData,
    });

    const json = await response.json();
    if (!response.ok) throw new Error(json.error?.message || 'Upload failed');
    return json.data;
  }

  // Settings
  async getSettings(): Promise<UserSettings> {
    return this.request('/api/v2/settings');
  }

  async updateSettings(data: Partial<UserSettings>): Promise<UserSettings> {
    return this.request('/api/v2/settings', { method: 'PATCH', body: JSON.stringify(data) });
  }
}

export const api = new ApiClient();
export default api

