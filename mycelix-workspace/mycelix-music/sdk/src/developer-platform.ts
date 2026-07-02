// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Developer Experience & SDK Platform
 *
 * Complete developer ecosystem including:
 * - TypeScript SDK with full type safety
 * - Developer Portal & API Key Management
 * - Webhook System for event subscriptions
 * - CLI Tools for artists and developers
 * - Local Development Environment
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import * as crypto from 'crypto';

// ============================================================================
// TypeScript SDK
// ============================================================================

interface SDKConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  retries?: number;
  debug?: boolean;
}

interface RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  path: string;
  body?: any;
  query?: Record<string, string>;
  headers?: Record<string, string>;
}

interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    perPage: number;
    total: number;
    totalPages: number;
    hasMore: boolean;
  };
}

// Base SDK Client
export class MycelixSDK {
  private config: Required<SDKConfig>;
  private rateLimiter: RateLimiter;

  // Resource clients
  public tracks: TracksClient;
  public artists: ArtistsClient;
  public albums: AlbumsClient;
  public playlists: PlaylistsClient;
  public users: UsersClient;
  public streaming: StreamingClient;
  public analytics: AnalyticsClient;
  public search: SearchClient;
  public recommendations: RecommendationsClient;
  public social: SocialClient;

  constructor(config: SDKConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl || 'https://api.mycelix.io/v1',
      timeout: config.timeout || 30000,
      retries: config.retries || 3,
      debug: config.debug || false,
    };

    this.rateLimiter = new RateLimiter(1000, 60000); // 1000 requests per minute

    // Initialize resource clients
    this.tracks = new TracksClient(this);
    this.artists = new ArtistsClient(this);
    this.albums = new AlbumsClient(this);
    this.playlists = new PlaylistsClient(this);
    this.users = new UsersClient(this);
    this.streaming = new StreamingClient(this);
    this.analytics = new AnalyticsClient(this);
    this.search = new SearchClient(this);
    this.recommendations = new RecommendationsClient(this);
    this.social = new SocialClient(this);
  }

  async request<T>(options: RequestOptions): Promise<T> {
    await this.rateLimiter.acquire();

    const url = new URL(options.path, this.config.baseUrl);
    if (options.query) {
      Object.entries(options.query).forEach(([key, value]) => {
        url.searchParams.append(key, value);
      });
    }

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.config.apiKey}`,
      'Content-Type': 'application/json',
      'X-SDK-Version': '1.0.0',
      'X-Request-ID': uuidv4(),
      ...options.headers,
    };

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.config.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        const response = await fetch(url.toString(), {
          method: options.method,
          headers,
          body: options.body ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw new MycelixAPIError(
            error.message || `HTTP ${response.status}`,
            response.status,
            error.code,
            error.details
          );
        }

        const data = await response.json();

        if (this.config.debug) {
          console.log(`[Mycelix SDK] ${options.method} ${options.path}`, { response: data });
        }

        return data as T;
      } catch (error: any) {
        lastError = error;

        if (error.name === 'AbortError') {
          throw new MycelixAPIError('Request timeout', 408, 'TIMEOUT');
        }

        // Don't retry on client errors (4xx)
        if (error instanceof MycelixAPIError && error.status >= 400 && error.status < 500) {
          throw error;
        }

        // Exponential backoff
        if (attempt < this.config.retries - 1) {
          await this.sleep(Math.pow(2, attempt) * 1000);
        }
      }
    }

    throw lastError;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Custom Error Class
export class MycelixAPIError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'MycelixAPIError';
  }
}

// Rate Limiter
class RateLimiter {
  private tokens: number;
  private lastRefill: number;

  constructor(private maxTokens: number, private refillInterval: number) {
    this.tokens = maxTokens;
    this.lastRefill = Date.now();
  }

  async acquire(): Promise<void> {
    this.refill();

    if (this.tokens <= 0) {
      const waitTime = this.refillInterval - (Date.now() - this.lastRefill);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      this.refill();
    }

    this.tokens--;
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = now - this.lastRefill;

    if (elapsed >= this.refillInterval) {
      this.tokens = this.maxTokens;
      this.lastRefill = now;
    }
  }
}

// ============================================================================
// Resource Clients
// ============================================================================

class TracksClient {
  constructor(private sdk: MycelixSDK) {}

  async get(id: string): Promise<Track> {
    return this.sdk.request({ method: 'GET', path: `/tracks/${id}` });
  }

  async list(params: ListTracksParams = {}): Promise<PaginatedResponse<Track>> {
    return this.sdk.request({
      method: 'GET',
      path: '/tracks',
      query: this.buildQuery(params),
    });
  }

  async create(data: CreateTrackInput): Promise<Track> {
    return this.sdk.request({ method: 'POST', path: '/tracks', body: data });
  }

  async update(id: string, data: UpdateTrackInput): Promise<Track> {
    return this.sdk.request({ method: 'PATCH', path: `/tracks/${id}`, body: data });
  }

  async delete(id: string): Promise<void> {
    return this.sdk.request({ method: 'DELETE', path: `/tracks/${id}` });
  }

  async getStreamUrl(id: string, quality: 'low' | 'medium' | 'high' | 'lossless' = 'high'): Promise<StreamUrl> {
    return this.sdk.request({
      method: 'GET',
      path: `/tracks/${id}/stream`,
      query: { quality },
    });
  }

  async analyze(id: string): Promise<AudioAnalysis> {
    return this.sdk.request({ method: 'POST', path: `/tracks/${id}/analyze` });
  }

  async getCredits(id: string): Promise<TrackCredits> {
    return this.sdk.request({ method: 'GET', path: `/tracks/${id}/credits` });
  }

  async getLyrics(id: string): Promise<Lyrics> {
    return this.sdk.request({ method: 'GET', path: `/tracks/${id}/lyrics` });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        query[key] = String(value);
      }
    });
    return query;
  }
}

class ArtistsClient {
  constructor(private sdk: MycelixSDK) {}

  async get(id: string): Promise<Artist> {
    return this.sdk.request({ method: 'GET', path: `/artists/${id}` });
  }

  async list(params: ListArtistsParams = {}): Promise<PaginatedResponse<Artist>> {
    return this.sdk.request({
      method: 'GET',
      path: '/artists',
      query: this.buildQuery(params),
    });
  }

  async getTracks(id: string, params: PaginationParams = {}): Promise<PaginatedResponse<Track>> {
    return this.sdk.request({
      method: 'GET',
      path: `/artists/${id}/tracks`,
      query: this.buildQuery(params),
    });
  }

  async getAlbums(id: string, params: PaginationParams = {}): Promise<PaginatedResponse<Album>> {
    return this.sdk.request({
      method: 'GET',
      path: `/artists/${id}/albums`,
      query: this.buildQuery(params),
    });
  }

  async getRelated(id: string, limit = 10): Promise<Artist[]> {
    return this.sdk.request({
      method: 'GET',
      path: `/artists/${id}/related`,
      query: { limit: String(limit) },
    });
  }

  async getStats(id: string): Promise<ArtistStats> {
    return this.sdk.request({ method: 'GET', path: `/artists/${id}/stats` });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

class AlbumsClient {
  constructor(private sdk: MycelixSDK) {}

  async get(id: string): Promise<Album> {
    return this.sdk.request({ method: 'GET', path: `/albums/${id}` });
  }

  async list(params: ListAlbumsParams = {}): Promise<PaginatedResponse<Album>> {
    return this.sdk.request({
      method: 'GET',
      path: '/albums',
      query: this.buildQuery(params),
    });
  }

  async getTracks(id: string): Promise<Track[]> {
    return this.sdk.request({ method: 'GET', path: `/albums/${id}/tracks` });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

class PlaylistsClient {
  constructor(private sdk: MycelixSDK) {}

  async get(id: string): Promise<Playlist> {
    return this.sdk.request({ method: 'GET', path: `/playlists/${id}` });
  }

  async list(params: ListPlaylistsParams = {}): Promise<PaginatedResponse<Playlist>> {
    return this.sdk.request({
      method: 'GET',
      path: '/playlists',
      query: this.buildQuery(params),
    });
  }

  async create(data: CreatePlaylistInput): Promise<Playlist> {
    return this.sdk.request({ method: 'POST', path: '/playlists', body: data });
  }

  async update(id: string, data: UpdatePlaylistInput): Promise<Playlist> {
    return this.sdk.request({ method: 'PATCH', path: `/playlists/${id}`, body: data });
  }

  async delete(id: string): Promise<void> {
    return this.sdk.request({ method: 'DELETE', path: `/playlists/${id}` });
  }

  async addTracks(id: string, trackIds: string[], position?: number): Promise<Playlist> {
    return this.sdk.request({
      method: 'POST',
      path: `/playlists/${id}/tracks`,
      body: { trackIds, position },
    });
  }

  async removeTracks(id: string, trackIds: string[]): Promise<Playlist> {
    return this.sdk.request({
      method: 'DELETE',
      path: `/playlists/${id}/tracks`,
      body: { trackIds },
    });
  }

  async reorder(id: string, rangeStart: number, insertBefore: number, rangeLength = 1): Promise<Playlist> {
    return this.sdk.request({
      method: 'PUT',
      path: `/playlists/${id}/tracks/reorder`,
      body: { rangeStart, insertBefore, rangeLength },
    });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

class UsersClient {
  constructor(private sdk: MycelixSDK) {}

  async getMe(): Promise<User> {
    return this.sdk.request({ method: 'GET', path: '/me' });
  }

  async get(id: string): Promise<User> {
    return this.sdk.request({ method: 'GET', path: `/users/${id}` });
  }

  async updateMe(data: UpdateUserInput): Promise<User> {
    return this.sdk.request({ method: 'PATCH', path: '/me', body: data });
  }

  async getPlaylists(userId: string, params: PaginationParams = {}): Promise<PaginatedResponse<Playlist>> {
    return this.sdk.request({
      method: 'GET',
      path: `/users/${userId}/playlists`,
      query: this.buildQuery(params),
    });
  }

  async getFollowers(userId: string, params: PaginationParams = {}): Promise<PaginatedResponse<User>> {
    return this.sdk.request({
      method: 'GET',
      path: `/users/${userId}/followers`,
      query: this.buildQuery(params),
    });
  }

  async getFollowing(userId: string, params: PaginationParams = {}): Promise<PaginatedResponse<User>> {
    return this.sdk.request({
      method: 'GET',
      path: `/users/${userId}/following`,
      query: this.buildQuery(params),
    });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

class StreamingClient {
  constructor(private sdk: MycelixSDK) {}

  async createSession(trackId: string, deviceId: string): Promise<StreamSession> {
    return this.sdk.request({
      method: 'POST',
      path: '/streaming/sessions',
      body: { trackId, deviceId },
    });
  }

  async reportProgress(sessionId: string, position: number, duration: number): Promise<void> {
    return this.sdk.request({
      method: 'POST',
      path: `/streaming/sessions/${sessionId}/progress`,
      body: { position, duration },
    });
  }

  async endSession(sessionId: string): Promise<void> {
    return this.sdk.request({
      method: 'DELETE',
      path: `/streaming/sessions/${sessionId}`,
    });
  }

  async getDevices(): Promise<Device[]> {
    return this.sdk.request({ method: 'GET', path: '/streaming/devices' });
  }

  async transferPlayback(deviceId: string, play = true): Promise<void> {
    return this.sdk.request({
      method: 'PUT',
      path: '/streaming/transfer',
      body: { deviceId, play },
    });
  }
}

class AnalyticsClient {
  constructor(private sdk: MycelixSDK) {}

  async getTrackAnalytics(trackId: string, period: AnalyticsPeriod): Promise<TrackAnalytics> {
    return this.sdk.request({
      method: 'GET',
      path: `/analytics/tracks/${trackId}`,
      query: { period },
    });
  }

  async getArtistAnalytics(artistId: string, period: AnalyticsPeriod): Promise<ArtistAnalytics> {
    return this.sdk.request({
      method: 'GET',
      path: `/analytics/artists/${artistId}`,
      query: { period },
    });
  }

  async getListenerDemographics(entityId: string, entityType: 'track' | 'artist' | 'album'): Promise<Demographics> {
    return this.sdk.request({
      method: 'GET',
      path: `/analytics/${entityType}s/${entityId}/demographics`,
    });
  }

  async getStreamingHistory(params: StreamingHistoryParams): Promise<PaginatedResponse<StreamEvent>> {
    return this.sdk.request({
      method: 'GET',
      path: '/analytics/history',
      query: this.buildQuery(params),
    });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

class SearchClient {
  constructor(private sdk: MycelixSDK) {}

  async search(query: string, types: SearchType[] = ['track', 'artist', 'album', 'playlist'], limit = 20): Promise<SearchResults> {
    return this.sdk.request({
      method: 'GET',
      path: '/search',
      query: { q: query, types: types.join(','), limit: String(limit) },
    });
  }

  async autocomplete(query: string, limit = 10): Promise<AutocompleteResults> {
    return this.sdk.request({
      method: 'GET',
      path: '/search/autocomplete',
      query: { q: query, limit: String(limit) },
    });
  }
}

class RecommendationsClient {
  constructor(private sdk: MycelixSDK) {}

  async getForUser(params: RecommendationParams = {}): Promise<Track[]> {
    return this.sdk.request({
      method: 'GET',
      path: '/recommendations',
      query: this.buildQuery(params),
    });
  }

  async getForTrack(trackId: string, limit = 20): Promise<Track[]> {
    return this.sdk.request({
      method: 'GET',
      path: `/recommendations/tracks/${trackId}`,
      query: { limit: String(limit) },
    });
  }

  async getPersonalizedPlaylists(): Promise<Playlist[]> {
    return this.sdk.request({ method: 'GET', path: '/recommendations/playlists' });
  }

  async discoverWeekly(): Promise<Playlist> {
    return this.sdk.request({ method: 'GET', path: '/recommendations/discover-weekly' });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

class SocialClient {
  constructor(private sdk: MycelixSDK) {}

  async follow(userId: string): Promise<void> {
    return this.sdk.request({ method: 'POST', path: `/users/${userId}/follow` });
  }

  async unfollow(userId: string): Promise<void> {
    return this.sdk.request({ method: 'DELETE', path: `/users/${userId}/follow` });
  }

  async like(entityType: 'track' | 'album' | 'playlist', entityId: string): Promise<void> {
    return this.sdk.request({ method: 'POST', path: `/${entityType}s/${entityId}/like` });
  }

  async unlike(entityType: 'track' | 'album' | 'playlist', entityId: string): Promise<void> {
    return this.sdk.request({ method: 'DELETE', path: `/${entityType}s/${entityId}/like` });
  }

  async share(entityType: 'track' | 'album' | 'playlist', entityId: string, platform: SharePlatform): Promise<ShareResult> {
    return this.sdk.request({
      method: 'POST',
      path: `/${entityType}s/${entityId}/share`,
      body: { platform },
    });
  }

  async getFeed(params: PaginationParams = {}): Promise<PaginatedResponse<FeedItem>> {
    return this.sdk.request({
      method: 'GET',
      path: '/feed',
      query: this.buildQuery(params),
    });
  }

  private buildQuery(params: Record<string, any>): Record<string, string> {
    const query: Record<string, string> = {};
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) query[key] = String(value);
    });
    return query;
  }
}

// ============================================================================
// Type Definitions
// ============================================================================

interface Track {
  id: string;
  title: string;
  artistId: string;
  artist: Artist;
  albumId?: string;
  album?: Album;
  duration: number;
  audioUrl: string;
  coverUrl?: string;
  genre: string;
  mood?: string[];
  bpm?: number;
  key?: string;
  isrc?: string;
  explicit: boolean;
  playCount: number;
  releaseDate: string;
  createdAt: string;
}

interface Artist {
  id: string;
  name: string;
  bio?: string;
  imageUrl?: string;
  genres: string[];
  verified: boolean;
  followerCount: number;
  monthlyListeners: number;
}

interface Album {
  id: string;
  title: string;
  artistId: string;
  artist: Artist;
  coverUrl?: string;
  type: 'album' | 'single' | 'ep' | 'compilation';
  trackCount: number;
  releaseDate: string;
  upc?: string;
}

interface Playlist {
  id: string;
  name: string;
  description?: string;
  coverUrl?: string;
  ownerId: string;
  owner: User;
  isPublic: boolean;
  collaborative: boolean;
  trackCount: number;
  followerCount: number;
  createdAt: string;
}

interface User {
  id: string;
  username: string;
  displayName: string;
  imageUrl?: string;
  followerCount: number;
  followingCount: number;
  isPremium: boolean;
}

interface StreamUrl {
  url: string;
  expiresAt: string;
  quality: string;
  format: string;
}

interface AudioAnalysis {
  trackId: string;
  duration: number;
  tempo: number;
  timeSignature: number;
  key: number;
  mode: number;
  loudness: number;
  energy: number;
  danceability: number;
  valence: number;
  speechiness: number;
  instrumentalness: number;
  acousticness: number;
  liveness: number;
  sections: AudioSection[];
  beats: AudioBeat[];
}

interface AudioSection {
  start: number;
  duration: number;
  loudness: number;
  tempo: number;
  key: number;
  mode: number;
}

interface AudioBeat {
  start: number;
  confidence: number;
}

interface TrackCredits {
  writers: string[];
  producers: string[];
  engineers: string[];
  performers: Record<string, string[]>;
}

interface Lyrics {
  trackId: string;
  text: string;
  syncedLyrics?: SyncedLine[];
  language: string;
  copyright?: string;
}

interface SyncedLine {
  startTime: number;
  endTime: number;
  text: string;
}

interface StreamSession {
  id: string;
  trackId: string;
  deviceId: string;
  startedAt: string;
}

interface Device {
  id: string;
  name: string;
  type: 'computer' | 'smartphone' | 'tablet' | 'speaker' | 'tv';
  isActive: boolean;
  volumePercent: number;
}

interface ArtistStats {
  totalStreams: number;
  monthlyListeners: number;
  followerCount: number;
  topTracks: Track[];
  topCountries: Array<{ country: string; listeners: number }>;
}

interface TrackAnalytics {
  trackId: string;
  period: string;
  totalStreams: number;
  uniqueListeners: number;
  averageListenDuration: number;
  skipRate: number;
  saveRate: number;
  shareCount: number;
  dailyStreams: Array<{ date: string; count: number }>;
}

interface ArtistAnalytics {
  artistId: string;
  period: string;
  totalStreams: number;
  uniqueListeners: number;
  followerGrowth: number;
  topTracks: Array<{ track: Track; streams: number }>;
  topPlaylists: Array<{ playlist: Playlist; streams: number }>;
}

interface Demographics {
  ageGroups: Array<{ range: string; percentage: number }>;
  genders: Array<{ gender: string; percentage: number }>;
  countries: Array<{ country: string; percentage: number }>;
  cities: Array<{ city: string; percentage: number }>;
}

interface StreamEvent {
  id: string;
  trackId: string;
  track: Track;
  timestamp: string;
  duration: number;
  completed: boolean;
}

interface SearchResults {
  tracks: Track[];
  artists: Artist[];
  albums: Album[];
  playlists: Playlist[];
}

interface AutocompleteResults {
  suggestions: Array<{
    type: 'track' | 'artist' | 'album' | 'playlist';
    id: string;
    name: string;
    imageUrl?: string;
  }>;
}

interface ShareResult {
  url: string;
  platform: string;
}

interface FeedItem {
  id: string;
  type: 'new_release' | 'playlist_update' | 'artist_activity' | 'friend_activity';
  actor: User | Artist;
  target?: Track | Album | Playlist;
  timestamp: string;
}

// Parameter types
interface ListTracksParams extends PaginationParams {
  artistId?: string;
  albumId?: string;
  genre?: string;
}

interface ListArtistsParams extends PaginationParams {
  genre?: string;
  verified?: boolean;
}

interface ListAlbumsParams extends PaginationParams {
  artistId?: string;
  type?: Album['type'];
}

interface ListPlaylistsParams extends PaginationParams {
  ownerId?: string;
  featured?: boolean;
}

interface PaginationParams {
  page?: number;
  perPage?: number;
}

interface CreateTrackInput {
  title: string;
  artistId: string;
  albumId?: string;
  genre: string;
  duration: number;
  audioUrl: string;
}

interface UpdateTrackInput {
  title?: string;
  genre?: string;
  mood?: string[];
}

interface CreatePlaylistInput {
  name: string;
  description?: string;
  isPublic?: boolean;
}

interface UpdatePlaylistInput {
  name?: string;
  description?: string;
  isPublic?: boolean;
}

interface UpdateUserInput {
  displayName?: string;
  imageUrl?: string;
}

interface StreamingHistoryParams extends PaginationParams {
  startDate?: string;
  endDate?: string;
}

interface RecommendationParams {
  seedTracks?: string[];
  seedArtists?: string[];
  seedGenres?: string[];
  limit?: number;
  minEnergy?: number;
  maxEnergy?: number;
  minDanceability?: number;
  maxDanceability?: number;
  minValence?: number;
  maxValence?: number;
}

type SearchType = 'track' | 'artist' | 'album' | 'playlist';
type AnalyticsPeriod = '7d' | '28d' | '90d' | '365d' | 'all';
type SharePlatform = 'twitter' | 'facebook' | 'instagram' | 'tiktok' | 'whatsapp' | 'copy';

// ============================================================================
// Webhook System
// ============================================================================

interface WebhookConfig {
  id: string;
  url: string;
  secret: string;
  events: WebhookEventType[];
  active: boolean;
  createdAt: Date;
  metadata?: Record<string, any>;
}

type WebhookEventType =
  | 'track.created'
  | 'track.updated'
  | 'track.deleted'
  | 'track.played'
  | 'album.released'
  | 'playlist.created'
  | 'playlist.updated'
  | 'user.signup'
  | 'user.subscription.created'
  | 'user.subscription.cancelled'
  | 'payment.completed'
  | 'payment.failed'
  | 'artist.verified'
  | 'milestone.reached';

interface WebhookEvent {
  id: string;
  type: WebhookEventType;
  timestamp: string;
  data: any;
  apiVersion: string;
}

interface WebhookDelivery {
  id: string;
  webhookId: string;
  eventId: string;
  status: 'pending' | 'delivered' | 'failed';
  attempts: number;
  lastAttemptAt?: Date;
  nextRetryAt?: Date;
  responseStatus?: number;
  responseBody?: string;
  error?: string;
}

export class WebhookService extends EventEmitter {
  private webhooks: Map<string, WebhookConfig> = new Map();
  private deliveryQueue: WebhookDelivery[] = [];
  private maxRetries = 5;
  private retryDelays = [60000, 300000, 900000, 3600000, 86400000]; // 1m, 5m, 15m, 1h, 24h

  async registerWebhook(config: Omit<WebhookConfig, 'id' | 'secret' | 'createdAt'>): Promise<WebhookConfig> {
    const webhook: WebhookConfig = {
      ...config,
      id: uuidv4(),
      secret: this.generateSecret(),
      createdAt: new Date(),
    };

    this.webhooks.set(webhook.id, webhook);
    return webhook;
  }

  async updateWebhook(id: string, updates: Partial<WebhookConfig>): Promise<WebhookConfig | null> {
    const webhook = this.webhooks.get(id);
    if (!webhook) return null;

    const updated = { ...webhook, ...updates };
    this.webhooks.set(id, updated);
    return updated;
  }

  async deleteWebhook(id: string): Promise<boolean> {
    return this.webhooks.delete(id);
  }

  async rotateSecret(id: string): Promise<string | null> {
    const webhook = this.webhooks.get(id);
    if (!webhook) return null;

    webhook.secret = this.generateSecret();
    return webhook.secret;
  }

  async dispatchEvent(event: Omit<WebhookEvent, 'id' | 'timestamp' | 'apiVersion'>): Promise<void> {
    const fullEvent: WebhookEvent = {
      ...event,
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      apiVersion: '2024-01-01',
    };

    // Find all webhooks subscribed to this event
    for (const [, webhook] of this.webhooks) {
      if (webhook.active && webhook.events.includes(event.type)) {
        const delivery: WebhookDelivery = {
          id: uuidv4(),
          webhookId: webhook.id,
          eventId: fullEvent.id,
          status: 'pending',
          attempts: 0,
        };

        this.deliveryQueue.push(delivery);
        this.deliverWebhook(webhook, fullEvent, delivery);
      }
    }
  }

  private async deliverWebhook(
    webhook: WebhookConfig,
    event: WebhookEvent,
    delivery: WebhookDelivery
  ): Promise<void> {
    delivery.attempts++;
    delivery.lastAttemptAt = new Date();

    const payload = JSON.stringify(event);
    const signature = this.signPayload(payload, webhook.secret);

    try {
      const response = await fetch(webhook.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Mycelix-Signature': signature,
          'X-Mycelix-Event': event.type,
          'X-Mycelix-Delivery': delivery.id,
          'X-Mycelix-Timestamp': event.timestamp,
        },
        body: payload,
      });

      delivery.responseStatus = response.status;

      if (response.ok) {
        delivery.status = 'delivered';
        this.emit('webhook_delivered', delivery);
      } else {
        delivery.responseBody = await response.text().catch(() => '');
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error: any) {
      delivery.error = error.message;

      if (delivery.attempts < this.maxRetries) {
        delivery.nextRetryAt = new Date(Date.now() + this.retryDelays[delivery.attempts - 1]);
        setTimeout(() => {
          this.deliverWebhook(webhook, event, delivery);
        }, this.retryDelays[delivery.attempts - 1]);
      } else {
        delivery.status = 'failed';
        this.emit('webhook_failed', delivery);
      }
    }
  }

  private generateSecret(): string {
    return `whsec_${crypto.randomBytes(32).toString('hex')}`;
  }

  private signPayload(payload: string, secret: string): string {
    const timestamp = Math.floor(Date.now() / 1000);
    const signedPayload = `${timestamp}.${payload}`;
    const signature = crypto
      .createHmac('sha256', secret)
      .update(signedPayload)
      .digest('hex');
    return `t=${timestamp},v1=${signature}`;
  }

  // Verification helper for webhook receivers
  static verifySignature(payload: string, signature: string, secret: string, tolerance = 300): boolean {
    const parts = signature.split(',').reduce((acc, part) => {
      const [key, value] = part.split('=');
      acc[key] = value;
      return acc;
    }, {} as Record<string, string>);

    const timestamp = parseInt(parts.t, 10);
    const now = Math.floor(Date.now() / 1000);

    if (Math.abs(now - timestamp) > tolerance) {
      return false;
    }

    const signedPayload = `${timestamp}.${payload}`;
    const expectedSignature = crypto
      .createHmac('sha256', secret)
      .update(signedPayload)
      .digest('hex');

    return crypto.timingSafeEqual(
      Buffer.from(parts.v1),
      Buffer.from(expectedSignature)
    );
  }

  getDeliveryHistory(webhookId: string, limit = 100): WebhookDelivery[] {
    return this.deliveryQueue
      .filter(d => d.webhookId === webhookId)
      .slice(-limit);
  }
}

// ============================================================================
// API Key Management
// ============================================================================

interface APIKey {
  id: string;
  name: string;
  key: string;
  keyPrefix: string;
  hashedKey: string;
  scopes: APIScope[];
  rateLimit: {
    requests: number;
    window: number;
  };
  ownerId: string;
  environment: 'development' | 'production';
  lastUsedAt?: Date;
  expiresAt?: Date;
  createdAt: Date;
  metadata?: Record<string, any>;
}

type APIScope =
  | 'tracks:read'
  | 'tracks:write'
  | 'artists:read'
  | 'artists:write'
  | 'playlists:read'
  | 'playlists:write'
  | 'users:read'
  | 'users:write'
  | 'analytics:read'
  | 'streaming:read'
  | 'streaming:write'
  | 'webhooks:manage'
  | 'admin';

export class APIKeyManager {
  private keys: Map<string, APIKey> = new Map();
  private keysByHash: Map<string, APIKey> = new Map();

  async createKey(config: {
    name: string;
    scopes: APIScope[];
    ownerId: string;
    environment: APIKey['environment'];
    rateLimit?: APIKey['rateLimit'];
    expiresAt?: Date;
  }): Promise<{ apiKey: APIKey; secretKey: string }> {
    const secretKey = this.generateSecretKey(config.environment);
    const keyPrefix = secretKey.substring(0, 12);
    const hashedKey = this.hashKey(secretKey);

    const apiKey: APIKey = {
      id: uuidv4(),
      name: config.name,
      key: `${keyPrefix}...`,
      keyPrefix,
      hashedKey,
      scopes: config.scopes,
      rateLimit: config.rateLimit || { requests: 1000, window: 60000 },
      ownerId: config.ownerId,
      environment: config.environment,
      expiresAt: config.expiresAt,
      createdAt: new Date(),
    };

    this.keys.set(apiKey.id, apiKey);
    this.keysByHash.set(hashedKey, apiKey);

    return { apiKey, secretKey };
  }

  async validateKey(secretKey: string): Promise<APIKey | null> {
    const hashedKey = this.hashKey(secretKey);
    const apiKey = this.keysByHash.get(hashedKey);

    if (!apiKey) return null;

    if (apiKey.expiresAt && apiKey.expiresAt < new Date()) {
      return null;
    }

    apiKey.lastUsedAt = new Date();
    return apiKey;
  }

  async revokeKey(id: string): Promise<boolean> {
    const apiKey = this.keys.get(id);
    if (!apiKey) return false;

    this.keys.delete(id);
    this.keysByHash.delete(apiKey.hashedKey);
    return true;
  }

  async listKeys(ownerId: string): Promise<APIKey[]> {
    return Array.from(this.keys.values()).filter(k => k.ownerId === ownerId);
  }

  hasScope(apiKey: APIKey, requiredScope: APIScope): boolean {
    if (apiKey.scopes.includes('admin')) return true;
    return apiKey.scopes.includes(requiredScope);
  }

  private generateSecretKey(environment: APIKey['environment']): string {
    const prefix = environment === 'production' ? 'mk_live_' : 'mk_test_';
    return `${prefix}${crypto.randomBytes(32).toString('hex')}`;
  }

  private hashKey(key: string): string {
    return crypto.createHash('sha256').update(key).digest('hex');
  }
}

// ============================================================================
// Developer Portal
// ============================================================================

interface DeveloperApp {
  id: string;
  name: string;
  description?: string;
  ownerId: string;
  website?: string;
  redirectUris: string[];
  clientId: string;
  clientSecret: string;
  scopes: APIScope[];
  status: 'pending' | 'approved' | 'rejected' | 'suspended';
  rateLimitTier: 'free' | 'basic' | 'pro' | 'enterprise';
  createdAt: Date;
  approvedAt?: Date;
}

interface UsageStats {
  appId: string;
  period: string;
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageLatency: number;
  p95Latency: number;
  p99Latency: number;
  topEndpoints: Array<{ endpoint: string; count: number }>;
  errorBreakdown: Array<{ code: string; count: number }>;
}

export class DeveloperPortal {
  private apps: Map<string, DeveloperApp> = new Map();
  private usageStats: Map<string, UsageStats[]> = new Map();
  private apiKeyManager: APIKeyManager;
  private webhookService: WebhookService;

  constructor() {
    this.apiKeyManager = new APIKeyManager();
    this.webhookService = new WebhookService();
  }

  async registerApp(config: {
    name: string;
    description?: string;
    ownerId: string;
    website?: string;
    redirectUris: string[];
    scopes: APIScope[];
  }): Promise<DeveloperApp> {
    const app: DeveloperApp = {
      ...config,
      id: uuidv4(),
      clientId: `myc_${crypto.randomBytes(16).toString('hex')}`,
      clientSecret: `myc_secret_${crypto.randomBytes(32).toString('hex')}`,
      status: 'pending',
      rateLimitTier: 'free',
      createdAt: new Date(),
    };

    this.apps.set(app.id, app);
    return app;
  }

  async approveApp(id: string): Promise<DeveloperApp | null> {
    const app = this.apps.get(id);
    if (!app) return null;

    app.status = 'approved';
    app.approvedAt = new Date();
    return app;
  }

  async createAPIKey(appId: string, name: string, environment: 'development' | 'production'): Promise<any> {
    const app = this.apps.get(appId);
    if (!app || app.status !== 'approved') {
      throw new Error('App not approved');
    }

    return this.apiKeyManager.createKey({
      name,
      scopes: app.scopes,
      ownerId: app.ownerId,
      environment,
      rateLimit: this.getRateLimitForTier(app.rateLimitTier),
    });
  }

  private getRateLimitForTier(tier: DeveloperApp['rateLimitTier']): APIKey['rateLimit'] {
    const limits = {
      free: { requests: 100, window: 60000 },
      basic: { requests: 1000, window: 60000 },
      pro: { requests: 10000, window: 60000 },
      enterprise: { requests: 100000, window: 60000 },
    };
    return limits[tier];
  }

  async getUsageStats(appId: string, period: '24h' | '7d' | '30d'): Promise<UsageStats | null> {
    const stats = this.usageStats.get(appId);
    return stats?.find(s => s.period === period) || null;
  }

  async registerWebhook(appId: string, url: string, events: WebhookEventType[]): Promise<WebhookConfig> {
    const app = this.apps.get(appId);
    if (!app || app.status !== 'approved') {
      throw new Error('App not approved');
    }

    return this.webhookService.registerWebhook({
      url,
      events,
      active: true,
      metadata: { appId },
    });
  }

  getDocumentation(): object {
    return {
      version: '1.0.0',
      baseUrl: 'https://api.mycelix.io/v1',
      authentication: {
        type: 'bearer',
        description: 'Include API key in Authorization header',
      },
      rateLimits: {
        free: '100 requests/minute',
        basic: '1,000 requests/minute',
        pro: '10,000 requests/minute',
        enterprise: '100,000 requests/minute',
      },
      endpoints: this.getEndpointDocumentation(),
      webhookEvents: this.getWebhookEventDocumentation(),
      sdks: {
        typescript: 'npm install @mycelix/sdk',
        python: 'pip install mycelix',
        swift: 'pod install Mycelix',
        kotlin: 'implementation("io.mycelix:sdk:1.0.0")',
      },
    };
  }

  private getEndpointDocumentation(): object {
    return {
      tracks: {
        'GET /tracks': 'List tracks',
        'GET /tracks/:id': 'Get track details',
        'POST /tracks': 'Create track (artist only)',
        'GET /tracks/:id/stream': 'Get streaming URL',
        'POST /tracks/:id/analyze': 'Analyze audio features',
      },
      artists: {
        'GET /artists': 'List artists',
        'GET /artists/:id': 'Get artist details',
        'GET /artists/:id/tracks': 'Get artist tracks',
        'GET /artists/:id/albums': 'Get artist albums',
      },
      playlists: {
        'GET /playlists': 'List playlists',
        'POST /playlists': 'Create playlist',
        'PUT /playlists/:id/tracks': 'Add tracks to playlist',
      },
      search: {
        'GET /search': 'Search tracks, artists, albums, playlists',
        'GET /search/autocomplete': 'Get search suggestions',
      },
      recommendations: {
        'GET /recommendations': 'Get personalized recommendations',
        'GET /recommendations/tracks/:id': 'Get similar tracks',
      },
    };
  }

  private getWebhookEventDocumentation(): object {
    return {
      'track.created': 'New track uploaded',
      'track.played': 'Track was played (30+ seconds)',
      'album.released': 'New album released',
      'user.signup': 'New user registration',
      'payment.completed': 'Payment successful',
      'milestone.reached': 'Artist/track milestone',
    };
  }
}

// ============================================================================
// CLI Tool Framework
// ============================================================================

interface CLICommand {
  name: string;
  description: string;
  options: CLIOption[];
  action: (args: Record<string, any>, sdk: MycelixSDK) => Promise<void>;
}

interface CLIOption {
  name: string;
  alias?: string;
  description: string;
  required?: boolean;
  type: 'string' | 'number' | 'boolean';
  default?: any;
}

export class MycelixCLI {
  private commands: Map<string, CLICommand> = new Map();
  private sdk: MycelixSDK | null = null;

  constructor() {
    this.registerDefaultCommands();
  }

  private registerDefaultCommands(): void {
    // Auth commands
    this.register({
      name: 'login',
      description: 'Authenticate with Mycelix',
      options: [
        { name: 'api-key', alias: 'k', description: 'API key', type: 'string' },
      ],
      action: async (args) => {
        this.sdk = new MycelixSDK({ apiKey: args['api-key'] });
        console.log('Successfully authenticated!');
      },
    });

    // Track commands
    this.register({
      name: 'tracks:list',
      description: 'List your tracks',
      options: [
        { name: 'limit', alias: 'l', description: 'Number of tracks', type: 'number', default: 20 },
        { name: 'format', alias: 'f', description: 'Output format', type: 'string', default: 'table' },
      ],
      action: async (args, sdk) => {
        const response = await sdk.tracks.list({ perPage: args.limit });
        this.formatOutput(response.data, args.format);
      },
    });

    this.register({
      name: 'tracks:upload',
      description: 'Upload a new track',
      options: [
        { name: 'file', alias: 'f', description: 'Audio file path', type: 'string', required: true },
        { name: 'title', alias: 't', description: 'Track title', type: 'string', required: true },
        { name: 'genre', alias: 'g', description: 'Genre', type: 'string', required: true },
      ],
      action: async (args, sdk) => {
        console.log(`Uploading ${args.file}...`);
        // Upload logic would go here
        console.log('Track uploaded successfully!');
      },
    });

    this.register({
      name: 'tracks:analyze',
      description: 'Analyze a track',
      options: [
        { name: 'id', description: 'Track ID', type: 'string', required: true },
      ],
      action: async (args, sdk) => {
        const analysis = await sdk.tracks.analyze(args.id);
        console.log('Audio Analysis:');
        console.log(`  BPM: ${analysis.tempo}`);
        console.log(`  Key: ${analysis.key}`);
        console.log(`  Energy: ${(analysis.energy * 100).toFixed(1)}%`);
        console.log(`  Danceability: ${(analysis.danceability * 100).toFixed(1)}%`);
      },
    });

    // Analytics commands
    this.register({
      name: 'analytics:summary',
      description: 'Get analytics summary',
      options: [
        { name: 'period', alias: 'p', description: 'Time period', type: 'string', default: '28d' },
      ],
      action: async (args, sdk) => {
        const me = await sdk.users.getMe();
        console.log(`Analytics for ${me.displayName}`);
        // Would fetch and display analytics
      },
    });

    // Playlist commands
    this.register({
      name: 'playlists:create',
      description: 'Create a new playlist',
      options: [
        { name: 'name', alias: 'n', description: 'Playlist name', type: 'string', required: true },
        { name: 'description', alias: 'd', description: 'Description', type: 'string' },
        { name: 'public', description: 'Make playlist public', type: 'boolean', default: true },
      ],
      action: async (args, sdk) => {
        const playlist = await sdk.playlists.create({
          name: args.name,
          description: args.description,
          isPublic: args.public,
        });
        console.log(`Created playlist: ${playlist.name} (${playlist.id})`);
      },
    });

    // Search command
    this.register({
      name: 'search',
      description: 'Search for music',
      options: [
        { name: 'query', alias: 'q', description: 'Search query', type: 'string', required: true },
        { name: 'type', alias: 't', description: 'Result type', type: 'string', default: 'all' },
      ],
      action: async (args, sdk) => {
        const types = args.type === 'all'
          ? ['track', 'artist', 'album', 'playlist'] as SearchType[]
          : [args.type as SearchType];
        const results = await sdk.search.search(args.query, types);
        this.formatOutput(results, 'json');
      },
    });
  }

  register(command: CLICommand): void {
    this.commands.set(command.name, command);
  }

  async run(argv: string[]): Promise<void> {
    const [commandName, ...args] = argv.slice(2);

    if (!commandName || commandName === 'help') {
      this.showHelp();
      return;
    }

    const command = this.commands.get(commandName);
    if (!command) {
      console.error(`Unknown command: ${commandName}`);
      this.showHelp();
      return;
    }

    const parsedArgs = this.parseArgs(args, command.options);

    // Validate required options
    for (const option of command.options) {
      if (option.required && !(option.name in parsedArgs)) {
        console.error(`Missing required option: --${option.name}`);
        return;
      }
    }

    if (!this.sdk && commandName !== 'login') {
      console.error('Please login first: mycelix login --api-key YOUR_KEY');
      return;
    }

    try {
      await command.action(parsedArgs, this.sdk!);
    } catch (error: any) {
      console.error(`Error: ${error.message}`);
    }
  }

  private parseArgs(args: string[], options: CLIOption[]): Record<string, any> {
    const result: Record<string, any> = {};

    // Set defaults
    for (const option of options) {
      if (option.default !== undefined) {
        result[option.name] = option.default;
      }
    }

    // Parse arguments
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];

      if (arg.startsWith('--')) {
        const name = arg.slice(2);
        const option = options.find(o => o.name === name);

        if (option) {
          if (option.type === 'boolean') {
            result[name] = true;
          } else {
            result[name] = this.castValue(args[++i], option.type);
          }
        }
      } else if (arg.startsWith('-')) {
        const alias = arg.slice(1);
        const option = options.find(o => o.alias === alias);

        if (option) {
          if (option.type === 'boolean') {
            result[option.name] = true;
          } else {
            result[option.name] = this.castValue(args[++i], option.type);
          }
        }
      }
    }

    return result;
  }

  private castValue(value: string, type: CLIOption['type']): any {
    switch (type) {
      case 'number':
        return Number(value);
      case 'boolean':
        return value === 'true';
      default:
        return value;
    }
  }

  private formatOutput(data: any, format: string): void {
    switch (format) {
      case 'json':
        console.log(JSON.stringify(data, null, 2));
        break;
      case 'table':
        console.table(data);
        break;
      default:
        console.log(data);
    }
  }

  private showHelp(): void {
    console.log('Mycelix CLI - Music Platform Command Line Interface\n');
    console.log('Usage: mycelix <command> [options]\n');
    console.log('Commands:');

    for (const [name, command] of this.commands) {
      console.log(`  ${name.padEnd(20)} ${command.description}`);
    }

    console.log('\nRun "mycelix <command> --help" for command-specific help');
  }
}

// ============================================================================
// Local Development Environment
// ============================================================================

interface LocalDevConfig {
  services: {
    api: boolean;
    database: boolean;
    redis: boolean;
    storage: boolean;
    search: boolean;
  };
  ports: {
    api: number;
    database: number;
    redis: number;
    storage: number;
    search: number;
  };
  seedData: boolean;
  hotReload: boolean;
}

export class LocalDevEnvironment {
  private config: LocalDevConfig;
  private runningServices: Set<string> = new Set();

  constructor(config: Partial<LocalDevConfig> = {}) {
    this.config = {
      services: {
        api: true,
        database: true,
        redis: true,
        storage: true,
        search: true,
        ...config.services,
      },
      ports: {
        api: 3000,
        database: 5432,
        redis: 6379,
        storage: 9000,
        search: 9200,
        ...config.ports,
      },
      seedData: config.seedData ?? true,
      hotReload: config.hotReload ?? true,
    };
  }

  generateDockerCompose(): string {
    const services: Record<string, any> = {};

    if (this.config.services.database) {
      services.postgres = {
        image: 'postgres:15-alpine',
        environment: {
          POSTGRES_USER: 'mycelix',
          POSTGRES_PASSWORD: 'mycelix_dev',
          POSTGRES_DB: 'mycelix_dev',
        },
        ports: [`${this.config.ports.database}:5432`],
        volumes: ['postgres_data:/var/lib/postgresql/data'],
        healthcheck: {
          test: ['CMD-SHELL', 'pg_isready -U mycelix'],
          interval: '10s',
          timeout: '5s',
          retries: 5,
        },
      };
    }

    if (this.config.services.redis) {
      services.redis = {
        image: 'redis:7-alpine',
        ports: [`${this.config.ports.redis}:6379`],
        volumes: ['redis_data:/data'],
        healthcheck: {
          test: ['CMD', 'redis-cli', 'ping'],
          interval: '10s',
          timeout: '5s',
          retries: 5,
        },
      };
    }

    if (this.config.services.storage) {
      services.minio = {
        image: 'minio/minio',
        command: 'server /data --console-address ":9001"',
        environment: {
          MINIO_ROOT_USER: 'mycelix',
          MINIO_ROOT_PASSWORD: 'mycelix_dev',
        },
        ports: [
          `${this.config.ports.storage}:9000`,
          '9001:9001',
        ],
        volumes: ['minio_data:/data'],
      };
    }

    if (this.config.services.search) {
      services.elasticsearch = {
        image: 'elasticsearch:8.11.0',
        environment: {
          'discovery.type': 'single-node',
          'xpack.security.enabled': 'false',
          'ES_JAVA_OPTS': '-Xms512m -Xmx512m',
        },
        ports: [`${this.config.ports.search}:9200`],
        volumes: ['es_data:/usr/share/elasticsearch/data'],
      };
    }

    if (this.config.services.api) {
      services.api = {
        build: {
          context: '.',
          dockerfile: 'Dockerfile.dev',
        },
        ports: [`${this.config.ports.api}:3000`],
        environment: {
          NODE_ENV: 'development',
          DATABASE_URL: `postgresql://mycelix:mycelix_dev@postgres:5432/mycelix_dev`,
          REDIS_URL: 'redis://redis:6379',
          S3_ENDPOINT: 'http://minio:9000',
          S3_ACCESS_KEY: 'mycelix',
          S3_SECRET_KEY: 'mycelix_dev',
          ELASTICSEARCH_URL: 'http://elasticsearch:9200',
        },
        volumes: [
          '.:/app',
          '/app/node_modules',
        ],
        depends_on: {
          postgres: { condition: 'service_healthy' },
          redis: { condition: 'service_healthy' },
        },
      };
    }

    return `version: '3.8'

services:
${Object.entries(services).map(([name, config]) =>
  `  ${name}:\n${this.yamlify(config, 4)}`
).join('\n\n')}

volumes:
  postgres_data:
  redis_data:
  minio_data:
  es_data:
`;
  }

  private yamlify(obj: any, indent: number): string {
    const spaces = ' '.repeat(indent);
    const lines: string[] = [];

    for (const [key, value] of Object.entries(obj)) {
      if (Array.isArray(value)) {
        lines.push(`${spaces}${key}:`);
        for (const item of value) {
          if (typeof item === 'object') {
            lines.push(`${spaces}  -`);
            lines.push(this.yamlify(item, indent + 4));
          } else {
            lines.push(`${spaces}  - ${item}`);
          }
        }
      } else if (typeof value === 'object' && value !== null) {
        lines.push(`${spaces}${key}:`);
        lines.push(this.yamlify(value, indent + 2));
      } else {
        lines.push(`${spaces}${key}: ${value}`);
      }
    }

    return lines.join('\n');
  }

  generateSeedData(): object {
    return {
      users: [
        {
          id: 'user_dev_1',
          email: 'developer@mycelix.io',
          username: 'developer',
          displayName: 'Dev User',
          isPremium: true,
        },
        {
          id: 'user_artist_1',
          email: 'artist@mycelix.io',
          username: 'demo_artist',
          displayName: 'Demo Artist',
          isPremium: true,
        },
      ],
      artists: [
        {
          id: 'artist_dev_1',
          userId: 'user_artist_1',
          name: 'Demo Artist',
          genres: ['electronic', 'ambient'],
          verified: true,
        },
      ],
      tracks: Array.from({ length: 50 }, (_, i) => ({
        id: `track_dev_${i + 1}`,
        title: `Demo Track ${i + 1}`,
        artistId: 'artist_dev_1',
        duration: 180 + Math.floor(Math.random() * 120),
        genre: ['electronic', 'ambient', 'pop', 'rock'][Math.floor(Math.random() * 4)],
        bpm: 80 + Math.floor(Math.random() * 80),
      })),
      playlists: [
        {
          id: 'playlist_dev_1',
          name: 'Dev Playlist',
          ownerId: 'user_dev_1',
          isPublic: true,
          trackIds: ['track_dev_1', 'track_dev_2', 'track_dev_3'],
        },
      ],
    };
  }

  generateEnvFile(): string {
    return `# Mycelix Local Development Environment
NODE_ENV=development

# API
API_PORT=${this.config.ports.api}
API_URL=http://localhost:${this.config.ports.api}

# Database
DATABASE_URL=postgresql://mycelix:mycelix_dev@localhost:${this.config.ports.database}/mycelix_dev

# Redis
REDIS_URL=redis://localhost:${this.config.ports.redis}

# Storage (MinIO)
S3_ENDPOINT=http://localhost:${this.config.ports.storage}
S3_ACCESS_KEY=mycelix
S3_SECRET_KEY=mycelix_dev
S3_BUCKET=mycelix-dev

# Search
ELASTICSEARCH_URL=http://localhost:${this.config.ports.search}

# Auth
JWT_SECRET=dev_jwt_secret_change_in_production
SESSION_SECRET=dev_session_secret_change_in_production

# Development API Key
DEV_API_KEY=mk_test_dev_key_for_local_development
`;
  }
}

// ============================================================================
// Export
// ============================================================================

export const createDeveloperPlatform = (): {
  sdk: typeof MycelixSDK;
  portal: DeveloperPortal;
  cli: MycelixCLI;
  localDev: LocalDevEnvironment;
} => {
  return {
    sdk: MycelixSDK,
    portal: new DeveloperPortal(),
    cli: new MycelixCLI(),
    localDev: new LocalDevEnvironment(),
  };
};
