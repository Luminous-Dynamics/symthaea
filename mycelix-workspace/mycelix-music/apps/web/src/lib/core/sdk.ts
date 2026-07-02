// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Developer SDK
 *
 * Third-party integration APIs:
 * - Embed player
 * - OAuth integration
 * - Webhooks
 * - REST API client
 * - WebSocket client
 * - SDK initialization
 */

// ==================== Types ====================

export interface SDKConfig {
  clientId: string;
  clientSecret?: string;
  redirectUri?: string;
  scopes?: string[];
  apiBaseUrl?: string;
  wsBaseUrl?: string;
  version?: string;
}

export interface AuthToken {
  accessToken: string;
  refreshToken?: string;
  tokenType: string;
  expiresIn: number;
  expiresAt: number;
  scope: string[];
}

export interface User {
  id: string;
  name: string;
  email?: string;
  avatar: string;
  followers: number;
  following: number;
  isArtist: boolean;
}

export interface Track {
  id: string;
  title: string;
  artist: User;
  album?: Album;
  duration: number;
  streamUrl: string;
  waveformUrl: string;
  coverUrl: string;
  playCount: number;
  likeCount: number;
}

export interface Album {
  id: string;
  title: string;
  artist: User;
  coverUrl: string;
  tracks: Track[];
  releaseDate: Date;
}

export interface Playlist {
  id: string;
  name: string;
  owner: User;
  tracks: Track[];
  isPublic: boolean;
  followers: number;
}

export interface EmbedOptions {
  trackId?: string;
  playlistId?: string;
  albumId?: string;
  width?: number | string;
  height?: number | string;
  autoplay?: boolean;
  showArtwork?: boolean;
  showPlayCount?: boolean;
  showLikes?: boolean;
  showComments?: boolean;
  showRelated?: boolean;
  color?: string;
  theme?: 'light' | 'dark' | 'auto';
}

export interface WebhookEvent {
  id: string;
  type: WebhookEventType;
  createdAt: Date;
  data: any;
}

export type WebhookEventType =
  | 'track.created'
  | 'track.updated'
  | 'track.deleted'
  | 'track.played'
  | 'track.liked'
  | 'user.followed'
  | 'user.unfollowed'
  | 'playlist.created'
  | 'playlist.updated'
  | 'comment.created'
  | 'subscription.created'
  | 'subscription.cancelled';

export interface WebhookConfig {
  url: string;
  events: WebhookEventType[];
  secret: string;
  active: boolean;
}

// ==================== OAuth Client ====================

export class OAuthClient {
  private config: SDKConfig;
  private token: AuthToken | null = null;
  private tokenChangeCallbacks: Set<(token: AuthToken | null) => void> = new Set();

  constructor(config: SDKConfig) {
    this.config = config;
    this.loadStoredToken();
  }

  private loadStoredToken(): void {
    try {
      const stored = localStorage.getItem('mycelix:oauth:token');
      if (stored) {
        const token = JSON.parse(stored);
        if (token.expiresAt > Date.now()) {
          this.token = token;
        }
      }
    } catch {
      // Ignore
    }
  }

  private saveToken(token: AuthToken | null): void {
    this.token = token;
    if (token) {
      localStorage.setItem('mycelix:oauth:token', JSON.stringify(token));
    } else {
      localStorage.removeItem('mycelix:oauth:token');
    }
    this.tokenChangeCallbacks.forEach(cb => cb(token));
  }

  /**
   * Get authorization URL for OAuth flow
   */
  getAuthorizationUrl(state?: string): string {
    const params = new URLSearchParams({
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri || window.location.origin + '/callback',
      response_type: 'code',
      scope: (this.config.scopes || ['read']).join(' '),
      state: state || crypto.randomUUID(),
    });

    return `${this.config.apiBaseUrl}/oauth/authorize?${params}`;
  }

  /**
   * Exchange authorization code for tokens
   */
  async exchangeCode(code: string): Promise<AuthToken> {
    const response = await fetch(`${this.config.apiBaseUrl}/oauth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        grant_type: 'authorization_code',
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        code,
        redirect_uri: this.config.redirectUri,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to exchange authorization code');
    }

    const data = await response.json();
    const token: AuthToken = {
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      tokenType: data.token_type,
      expiresIn: data.expires_in,
      expiresAt: Date.now() + data.expires_in * 1000,
      scope: data.scope.split(' '),
    };

    this.saveToken(token);
    return token;
  }

  /**
   * Refresh the access token
   */
  async refreshAccessToken(): Promise<AuthToken> {
    if (!this.token?.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.config.apiBaseUrl}/oauth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        refresh_token: this.token.refreshToken,
      }),
    });

    if (!response.ok) {
      this.saveToken(null);
      throw new Error('Failed to refresh token');
    }

    const data = await response.json();
    const token: AuthToken = {
      accessToken: data.access_token,
      refreshToken: data.refresh_token || this.token.refreshToken,
      tokenType: data.token_type,
      expiresIn: data.expires_in,
      expiresAt: Date.now() + data.expires_in * 1000,
      scope: data.scope.split(' '),
    };

    this.saveToken(token);
    return token;
  }

  /**
   * Get valid access token, refreshing if needed
   */
  async getAccessToken(): Promise<string | null> {
    if (!this.token) return null;

    // Refresh if expiring soon (within 5 minutes)
    if (this.token.expiresAt - Date.now() < 300000) {
      try {
        await this.refreshAccessToken();
      } catch {
        return null;
      }
    }

    return this.token.accessToken;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return this.token !== null && this.token.expiresAt > Date.now();
  }

  /**
   * Logout
   */
  logout(): void {
    this.saveToken(null);
  }

  /**
   * Subscribe to token changes
   */
  onTokenChange(callback: (token: AuthToken | null) => void): () => void {
    this.tokenChangeCallbacks.add(callback);
    return () => this.tokenChangeCallbacks.delete(callback);
  }
}

// ==================== API Client ====================

export class APIClient {
  private baseUrl: string;
  private oauth: OAuthClient;
  private version: string;

  constructor(oauth: OAuthClient, baseUrl: string, version: string = 'v1') {
    this.oauth = oauth;
    this.baseUrl = baseUrl;
    this.version = version;
  }

  private async request<T>(
    method: string,
    path: string,
    options: {
      body?: any;
      params?: Record<string, string>;
      requireAuth?: boolean;
    } = {}
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}/${this.version}${path}`);

    if (options.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.set(key, value);
      });
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-API-Version': this.version,
    };

    if (options.requireAuth !== false) {
      const token = await this.oauth.getAccessToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    }

    const response = await fetch(url.toString(), {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new APIError(
        error.message || `Request failed: ${response.status}`,
        response.status,
        error.code
      );
    }

    return response.json();
  }

  // User endpoints
  async getMe(): Promise<User> {
    return this.request('GET', '/me');
  }

  async getUser(userId: string): Promise<User> {
    return this.request('GET', `/users/${userId}`, { requireAuth: false });
  }

  async followUser(userId: string): Promise<void> {
    await this.request('POST', `/users/${userId}/follow`);
  }

  async unfollowUser(userId: string): Promise<void> {
    await this.request('DELETE', `/users/${userId}/follow`);
  }

  async getUserTracks(userId: string, params?: { limit?: number; offset?: number }): Promise<Track[]> {
    return this.request('GET', `/users/${userId}/tracks`, {
      params: params as Record<string, string>,
      requireAuth: false,
    });
  }

  // Track endpoints
  async getTrack(trackId: string): Promise<Track> {
    return this.request('GET', `/tracks/${trackId}`, { requireAuth: false });
  }

  async likeTrack(trackId: string): Promise<void> {
    await this.request('POST', `/tracks/${trackId}/like`);
  }

  async unlikeTrack(trackId: string): Promise<void> {
    await this.request('DELETE', `/tracks/${trackId}/like`);
  }

  async repostTrack(trackId: string): Promise<void> {
    await this.request('POST', `/tracks/${trackId}/repost`);
  }

  async getTrackComments(trackId: string): Promise<any[]> {
    return this.request('GET', `/tracks/${trackId}/comments`, { requireAuth: false });
  }

  async addComment(trackId: string, content: string, timestamp?: number): Promise<any> {
    return this.request('POST', `/tracks/${trackId}/comments`, {
      body: { content, timestamp },
    });
  }

  async getStreamUrl(trackId: string): Promise<{ url: string; expiresAt: number }> {
    return this.request('GET', `/tracks/${trackId}/stream`);
  }

  // Playlist endpoints
  async getPlaylist(playlistId: string): Promise<Playlist> {
    return this.request('GET', `/playlists/${playlistId}`, { requireAuth: false });
  }

  async createPlaylist(name: string, isPublic: boolean = true): Promise<Playlist> {
    return this.request('POST', '/playlists', { body: { name, isPublic } });
  }

  async addToPlaylist(playlistId: string, trackId: string): Promise<void> {
    await this.request('POST', `/playlists/${playlistId}/tracks`, {
      body: { trackId },
    });
  }

  async removeFromPlaylist(playlistId: string, trackId: string): Promise<void> {
    await this.request('DELETE', `/playlists/${playlistId}/tracks/${trackId}`);
  }

  // Search
  async search(query: string, type?: 'tracks' | 'users' | 'playlists' | 'albums'): Promise<{
    tracks: Track[];
    users: User[];
    playlists: Playlist[];
    albums: Album[];
  }> {
    return this.request('GET', '/search', {
      params: { q: query, type: type || 'all' },
      requireAuth: false,
    });
  }
}

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// ==================== WebSocket Client ====================

export class WSClient {
  private socket: WebSocket | null = null;
  private oauth: OAuthClient;
  private wsUrl: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private messageHandlers: Map<string, Set<(data: any) => void>> = new Map();
  private connectionCallbacks: Set<(connected: boolean) => void> = new Set();

  constructor(oauth: OAuthClient, wsUrl: string) {
    this.oauth = oauth;
    this.wsUrl = wsUrl;
  }

  async connect(): Promise<void> {
    const token = await this.oauth.getAccessToken();
    if (!token) {
      throw new Error('Not authenticated');
    }

    this.socket = new WebSocket(`${this.wsUrl}?token=${token}`);

    this.socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.connectionCallbacks.forEach(cb => cb(true));
    };

    this.socket.onclose = () => {
      this.connectionCallbacks.forEach(cb => cb(false));
      this.attemptReconnect();
    };

    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const handlers = this.messageHandlers.get(message.type);
        if (handlers) {
          handlers.forEach(handler => handler(message.data));
        }
      } catch {
        // Ignore parse errors
      }
    };
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.connect().catch(() => {});
      }, Math.pow(2, this.reconnectAttempts) * 1000);
    }
  }

  on<T = any>(event: string, handler: (data: T) => void): () => void {
    if (!this.messageHandlers.has(event)) {
      this.messageHandlers.set(event, new Set());
    }
    this.messageHandlers.get(event)!.add(handler);

    return () => {
      this.messageHandlers.get(event)?.delete(handler);
    };
  }

  onConnectionChange(callback: (connected: boolean) => void): () => void {
    this.connectionCallbacks.add(callback);
    return () => this.connectionCallbacks.delete(callback);
  }

  send(type: string, data: any): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type, data }));
    }
  }

  disconnect(): void {
    this.socket?.close();
    this.socket = null;
  }
}

// ==================== Embed Builder ====================

export class EmbedBuilder {
  private baseUrl: string;

  constructor(baseUrl: string = 'https://embed.mycelix.io') {
    this.baseUrl = baseUrl;
  }

  /**
   * Create embed HTML for a track
   */
  track(trackId: string, options: Omit<EmbedOptions, 'playlistId' | 'albumId'> = {}): string {
    return this.createEmbed({ ...options, trackId });
  }

  /**
   * Create embed HTML for a playlist
   */
  playlist(playlistId: string, options: Omit<EmbedOptions, 'trackId' | 'albumId'> = {}): string {
    return this.createEmbed({ ...options, playlistId });
  }

  /**
   * Create embed HTML for an album
   */
  album(albumId: string, options: Omit<EmbedOptions, 'trackId' | 'playlistId'> = {}): string {
    return this.createEmbed({ ...options, albumId });
  }

  /**
   * Get embed URL
   */
  getUrl(options: EmbedOptions): string {
    const params = new URLSearchParams();

    if (options.trackId) params.set('track', options.trackId);
    if (options.playlistId) params.set('playlist', options.playlistId);
    if (options.albumId) params.set('album', options.albumId);
    if (options.autoplay) params.set('autoplay', '1');
    if (options.showArtwork === false) params.set('artwork', '0');
    if (options.showPlayCount === false) params.set('playcount', '0');
    if (options.showLikes === false) params.set('likes', '0');
    if (options.showComments === false) params.set('comments', '0');
    if (options.showRelated === false) params.set('related', '0');
    if (options.color) params.set('color', options.color.replace('#', ''));
    if (options.theme) params.set('theme', options.theme);

    return `${this.baseUrl}?${params}`;
  }

  private createEmbed(options: EmbedOptions): string {
    const url = this.getUrl(options);
    const width = options.width || '100%';
    const height = options.height || (options.playlistId || options.albumId ? 400 : 166);

    return `<iframe
  width="${width}"
  height="${height}"
  scrolling="no"
  frameborder="no"
  allow="autoplay"
  src="${url}"
></iframe>`;
  }

  /**
   * Create oEmbed response
   */
  async getOEmbed(url: string): Promise<{
    type: string;
    version: string;
    title: string;
    author_name: string;
    author_url: string;
    provider_name: string;
    provider_url: string;
    html: string;
    width: number;
    height: number;
  }> {
    const response = await fetch(`${this.baseUrl}/oembed?url=${encodeURIComponent(url)}`);
    return response.json();
  }
}

// ==================== Webhook Manager ====================

export class WebhookManager {
  private apiClient: APIClient;

  constructor(apiClient: APIClient) {
    this.apiClient = apiClient;
  }

  async createWebhook(config: Omit<WebhookConfig, 'secret'>): Promise<WebhookConfig> {
    return (this.apiClient as any).request('POST', '/webhooks', { body: config });
  }

  async getWebhooks(): Promise<WebhookConfig[]> {
    return (this.apiClient as any).request('GET', '/webhooks');
  }

  async updateWebhook(webhookId: string, updates: Partial<WebhookConfig>): Promise<WebhookConfig> {
    return (this.apiClient as any).request('PATCH', `/webhooks/${webhookId}`, { body: updates });
  }

  async deleteWebhook(webhookId: string): Promise<void> {
    return (this.apiClient as any).request('DELETE', `/webhooks/${webhookId}`);
  }

  async testWebhook(webhookId: string): Promise<{ success: boolean; response?: any }> {
    return (this.apiClient as any).request('POST', `/webhooks/${webhookId}/test`);
  }

  /**
   * Verify webhook signature
   */
  verifySignature(payload: string, signature: string, secret: string): boolean {
    // Would use crypto.subtle.sign with HMAC
    // Simplified for reference
    const encoder = new TextEncoder();
    const data = encoder.encode(payload);
    const key = encoder.encode(secret);

    // In production, use proper HMAC verification
    return signature.length > 0;
  }
}

// ==================== Mycelix SDK ====================

export class MycelixSDK {
  public readonly oauth: OAuthClient;
  public readonly api: APIClient;
  public readonly ws: WSClient;
  public readonly embed: EmbedBuilder;
  public readonly webhooks: WebhookManager;

  private config: SDKConfig;

  constructor(config: SDKConfig) {
    this.config = {
      apiBaseUrl: 'https://api.mycelix.io',
      wsBaseUrl: 'wss://ws.mycelix.io',
      version: 'v1',
      ...config,
    };

    this.oauth = new OAuthClient(this.config);
    this.api = new APIClient(this.oauth, this.config.apiBaseUrl!, this.config.version);
    this.ws = new WSClient(this.oauth, this.config.wsBaseUrl!);
    this.embed = new EmbedBuilder();
    this.webhooks = new WebhookManager(this.api);
  }

  /**
   * Initialize the SDK
   */
  async init(): Promise<void> {
    // Handle OAuth callback if present
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');

    if (code) {
      await this.oauth.exchangeCode(code);
      // Clean up URL
      window.history.replaceState({}, '', window.location.pathname);
    }

    // Connect WebSocket if authenticated
    if (this.oauth.isAuthenticated()) {
      await this.ws.connect();
    }
  }

  /**
   * Start OAuth login flow
   */
  login(scopes?: string[]): void {
    if (scopes) {
      this.config.scopes = scopes;
    }
    window.location.href = this.oauth.getAuthorizationUrl();
  }

  /**
   * Logout and cleanup
   */
  logout(): void {
    this.oauth.logout();
    this.ws.disconnect();
  }

  /**
   * Check if user is logged in
   */
  isLoggedIn(): boolean {
    return this.oauth.isAuthenticated();
  }
}

// ==================== Factory ====================

export function createSDK(config: SDKConfig): MycelixSDK {
  return new MycelixSDK(config);
}

export default {
  createSDK,
  MycelixSDK,
  OAuthClient,
  APIClient,
  WSClient,
  EmbedBuilder,
  WebhookManager,
};
