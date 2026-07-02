// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Unified API Gateway
 * GraphQL Federation, REST proxy, WebSocket handling
 * Service mesh integration and request routing
 */

import { EventEmitter } from 'events';

// ============================================================
// CONFIGURATION
// ============================================================

interface GatewayConfig {
  port: number;
  services: ServiceRegistry;
  security: SecurityConfig;
  rateLimit: RateLimitConfig;
  cache: CacheConfig;
  circuit: CircuitBreakerConfig;
  tracing: TracingConfig;
}

interface ServiceRegistry {
  services: Map<string, ServiceDefinition>;
  discovery: ServiceDiscoveryConfig;
  loadBalancing: LoadBalancingStrategy;
}

interface ServiceDefinition {
  name: string;
  version: string;
  endpoints: ServiceEndpoint[];
  healthCheck: HealthCheckConfig;
  timeout: number;
  retries: number;
  circuitBreaker: boolean;
}

interface ServiceEndpoint {
  url: string;
  weight: number;
  region: string;
  healthy: boolean;
  lastCheck: Date;
}

interface HealthCheckConfig {
  path: string;
  interval: number;
  timeout: number;
  unhealthyThreshold: number;
  healthyThreshold: number;
}

interface ServiceDiscoveryConfig {
  type: 'static' | 'consul' | 'kubernetes' | 'etcd';
  refreshInterval: number;
  config: Record<string, any>;
}

type LoadBalancingStrategy = 'round_robin' | 'least_connections' | 'random' | 'weighted' | 'ip_hash';

interface SecurityConfig {
  jwt: JWTConfig;
  apiKey: APIKeyConfig;
  cors: CORSConfig;
  helmet: HelmetConfig;
}

interface JWTConfig {
  secret: string;
  issuer: string;
  audience: string;
  expiresIn: string;
  refreshExpiresIn: string;
}

interface APIKeyConfig {
  enabled: boolean;
  headerName: string;
  queryParam: string;
}

interface CORSConfig {
  origins: string[];
  methods: string[];
  headers: string[];
  credentials: boolean;
  maxAge: number;
}

interface HelmetConfig {
  contentSecurityPolicy: boolean;
  crossOriginEmbedderPolicy: boolean;
  crossOriginOpenerPolicy: boolean;
  crossOriginResourcePolicy: boolean;
  dnsPrefetchControl: boolean;
  frameguard: boolean;
  hidePoweredBy: boolean;
  hsts: boolean;
  ieNoOpen: boolean;
  noSniff: boolean;
  originAgentCluster: boolean;
  permittedCrossDomainPolicies: boolean;
  referrerPolicy: boolean;
  xssFilter: boolean;
}

interface RateLimitConfig {
  enabled: boolean;
  windowMs: number;
  maxRequests: number;
  keyGenerator: 'ip' | 'user' | 'api_key' | 'custom';
  skipSuccessfulRequests: boolean;
  tiers: RateLimitTier[];
}

interface RateLimitTier {
  name: string;
  maxRequests: number;
  windowMs: number;
  condition: string;
}

interface CacheConfig {
  enabled: boolean;
  type: 'memory' | 'redis' | 'memcached';
  ttl: number;
  maxSize: number;
  strategies: CacheStrategy[];
}

interface CacheStrategy {
  pattern: string;
  ttl: number;
  tags: string[];
  invalidateOn: string[];
}

interface CircuitBreakerConfig {
  enabled: boolean;
  threshold: number;
  timeout: number;
  resetTimeout: number;
  monitorInterval: number;
}

interface TracingConfig {
  enabled: boolean;
  serviceName: string;
  samplingRate: number;
  exporter: 'jaeger' | 'zipkin' | 'otlp';
  endpoint: string;
}

// ============================================================
// REQUEST/RESPONSE TYPES
// ============================================================

interface GatewayRequest {
  id: string;
  method: string;
  path: string;
  headers: Map<string, string>;
  query: Map<string, string>;
  body: any;
  user?: AuthenticatedUser;
  metadata: RequestMetadata;
}

interface AuthenticatedUser {
  id: string;
  email: string;
  roles: string[];
  permissions: string[];
  tier: string;
  metadata: Record<string, any>;
}

interface RequestMetadata {
  timestamp: Date;
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  clientIp: string;
  userAgent: string;
  region: string;
}

interface GatewayResponse {
  statusCode: number;
  headers: Map<string, string>;
  body: any;
  metadata: ResponseMetadata;
}

interface ResponseMetadata {
  duration: number;
  cached: boolean;
  service: string;
  endpoint: string;
  traceId: string;
}

// ============================================================
// API GATEWAY
// ============================================================

export class APIGateway extends EventEmitter {
  private config: GatewayConfig;
  private router: RequestRouter;
  private authenticator: Authenticator;
  private rateLimiter: RateLimiter;
  private cacheManager: CacheManager;
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private serviceHealth: Map<string, ServiceHealthStatus> = new Map();
  private metrics: GatewayMetrics;

  constructor(config: GatewayConfig) {
    super();
    this.config = config;
    this.router = new RequestRouter(config.services);
    this.authenticator = new Authenticator(config.security);
    this.rateLimiter = new RateLimiter(config.rateLimit);
    this.cacheManager = new CacheManager(config.cache);
    this.metrics = new GatewayMetrics();
    this.initializeCircuitBreakers();
    this.startHealthChecks();
  }

  async handleRequest(request: GatewayRequest): Promise<GatewayResponse> {
    const startTime = Date.now();
    const traceId = request.metadata.traceId;

    try {
      // 1. Rate limiting
      const rateLimitResult = await this.rateLimiter.check(request);
      if (!rateLimitResult.allowed) {
        return this.createRateLimitResponse(rateLimitResult);
      }

      // 2. Authentication
      const authResult = await this.authenticator.authenticate(request);
      if (!authResult.authenticated && this.requiresAuth(request.path)) {
        return this.createUnauthorizedResponse();
      }
      request.user = authResult.user;

      // 3. Authorization
      if (request.user && !this.authorize(request)) {
        return this.createForbiddenResponse();
      }

      // 4. Cache check
      const cachedResponse = await this.cacheManager.get(request);
      if (cachedResponse) {
        this.metrics.recordCacheHit(request.path);
        return {
          ...cachedResponse,
          metadata: {
            ...cachedResponse.metadata,
            cached: true,
            duration: Date.now() - startTime
          }
        };
      }

      // 5. Route to service
      const route = this.router.resolve(request);
      if (!route) {
        return this.createNotFoundResponse();
      }

      // 6. Circuit breaker check
      const circuitBreaker = this.circuitBreakers.get(route.service);
      if (circuitBreaker && !circuitBreaker.allowRequest()) {
        return this.createServiceUnavailableResponse(route.service);
      }

      // 7. Forward request
      const response = await this.forwardRequest(request, route);

      // 8. Update circuit breaker
      if (circuitBreaker) {
        if (response.statusCode >= 500) {
          circuitBreaker.recordFailure();
        } else {
          circuitBreaker.recordSuccess();
        }
      }

      // 9. Cache response
      if (this.shouldCache(request, response)) {
        await this.cacheManager.set(request, response);
      }

      // 10. Record metrics
      this.metrics.recordRequest(request, response, Date.now() - startTime);

      return {
        ...response,
        metadata: {
          ...response.metadata,
          duration: Date.now() - startTime,
          traceId
        }
      };

    } catch (error) {
      this.metrics.recordError(request, error as Error);
      return this.createErrorResponse(error as Error);
    }
  }

  private initializeCircuitBreakers(): void {
    for (const [name, service] of this.config.services.services) {
      if (service.circuitBreaker) {
        this.circuitBreakers.set(name, new CircuitBreaker({
          threshold: this.config.circuit.threshold,
          timeout: this.config.circuit.timeout,
          resetTimeout: this.config.circuit.resetTimeout
        }));
      }
    }
  }

  private startHealthChecks(): void {
    setInterval(() => {
      this.checkAllServicesHealth();
    }, 30000); // Every 30 seconds
  }

  private async checkAllServicesHealth(): Promise<void> {
    for (const [name, service] of this.config.services.services) {
      for (const endpoint of service.endpoints) {
        const health = await this.checkEndpointHealth(endpoint, service.healthCheck);
        endpoint.healthy = health.healthy;
        endpoint.lastCheck = new Date();
      }
    }
  }

  private async checkEndpointHealth(
    endpoint: ServiceEndpoint,
    config: HealthCheckConfig
  ): Promise<{ healthy: boolean; latency: number }> {
    const startTime = Date.now();
    try {
      const response = await fetch(`${endpoint.url}${config.path}`, {
        method: 'GET',
        signal: AbortSignal.timeout(config.timeout)
      });
      return {
        healthy: response.ok,
        latency: Date.now() - startTime
      };
    } catch {
      return { healthy: false, latency: Date.now() - startTime };
    }
  }

  private requiresAuth(path: string): boolean {
    const publicPaths = ['/health', '/metrics', '/api/v1/auth/login', '/api/v1/auth/register'];
    return !publicPaths.some(p => path.startsWith(p));
  }

  private authorize(request: GatewayRequest): boolean {
    if (!request.user) return false;

    const requiredPermission = this.getRequiredPermission(request);
    if (!requiredPermission) return true;

    return request.user.permissions.includes(requiredPermission) ||
           request.user.roles.includes('admin');
  }

  private getRequiredPermission(request: GatewayRequest): string | null {
    const permissionMap: Record<string, string> = {
      'POST:/api/v1/tracks': 'tracks:create',
      'DELETE:/api/v1/tracks': 'tracks:delete',
      'POST:/api/v1/playlists': 'playlists:create',
      'PUT:/api/v1/users': 'users:update'
    };

    const key = `${request.method}:${request.path.split('/').slice(0, 4).join('/')}`;
    return permissionMap[key] || null;
  }

  private async forwardRequest(
    request: GatewayRequest,
    route: ResolvedRoute
  ): Promise<GatewayResponse> {
    const endpoint = this.selectEndpoint(route.service);
    if (!endpoint) {
      throw new Error(`No healthy endpoints for service: ${route.service}`);
    }

    const url = `${endpoint.url}${route.path}`;
    const headers: Record<string, string> = {};
    request.headers.forEach((value, key) => {
      headers[key] = value;
    });

    // Add tracing headers
    headers['X-Trace-ID'] = request.metadata.traceId;
    headers['X-Span-ID'] = request.metadata.spanId;
    if (request.user) {
      headers['X-User-ID'] = request.user.id;
      headers['X-User-Roles'] = request.user.roles.join(',');
    }

    const fetchOptions: RequestInit = {
      method: request.method,
      headers,
      body: request.method !== 'GET' ? JSON.stringify(request.body) : undefined
    };

    const service = this.config.services.services.get(route.service);
    const timeout = service?.timeout || 30000;

    const response = await fetch(url, {
      ...fetchOptions,
      signal: AbortSignal.timeout(timeout)
    });

    const responseBody = await response.json().catch(() => null);

    return {
      statusCode: response.status,
      headers: new Map(Object.entries(Object.fromEntries(response.headers))),
      body: responseBody,
      metadata: {
        duration: 0,
        cached: false,
        service: route.service,
        endpoint: endpoint.url,
        traceId: request.metadata.traceId
      }
    };
  }

  private selectEndpoint(serviceName: string): ServiceEndpoint | null {
    const service = this.config.services.services.get(serviceName);
    if (!service) return null;

    const healthyEndpoints = service.endpoints.filter(e => e.healthy);
    if (healthyEndpoints.length === 0) return null;

    switch (this.config.services.loadBalancing) {
      case 'round_robin':
        return this.roundRobinSelect(healthyEndpoints);
      case 'weighted':
        return this.weightedSelect(healthyEndpoints);
      case 'random':
        return healthyEndpoints[Math.floor(Math.random() * healthyEndpoints.length)];
      default:
        return healthyEndpoints[0];
    }
  }

  private roundRobinSelect(endpoints: ServiceEndpoint[]): ServiceEndpoint {
    // Simple round-robin (would track index per service in production)
    return endpoints[Date.now() % endpoints.length];
  }

  private weightedSelect(endpoints: ServiceEndpoint[]): ServiceEndpoint {
    const totalWeight = endpoints.reduce((sum, e) => sum + e.weight, 0);
    let random = Math.random() * totalWeight;

    for (const endpoint of endpoints) {
      random -= endpoint.weight;
      if (random <= 0) return endpoint;
    }

    return endpoints[0];
  }

  private shouldCache(request: GatewayRequest, response: GatewayResponse): boolean {
    if (request.method !== 'GET') return false;
    if (response.statusCode !== 200) return false;
    if (!this.config.cache.enabled) return false;

    const noCacheHeader = request.headers.get('Cache-Control');
    if (noCacheHeader?.includes('no-cache')) return false;

    return true;
  }

  private createRateLimitResponse(result: RateLimitResult): GatewayResponse {
    return {
      statusCode: 429,
      headers: new Map([
        ['Retry-After', String(result.retryAfter)],
        ['X-RateLimit-Limit', String(result.limit)],
        ['X-RateLimit-Remaining', String(result.remaining)]
      ]),
      body: { error: 'Too Many Requests', retryAfter: result.retryAfter },
      metadata: { duration: 0, cached: false, service: 'gateway', endpoint: '', traceId: '' }
    };
  }

  private createUnauthorizedResponse(): GatewayResponse {
    return {
      statusCode: 401,
      headers: new Map([['WWW-Authenticate', 'Bearer']]),
      body: { error: 'Unauthorized' },
      metadata: { duration: 0, cached: false, service: 'gateway', endpoint: '', traceId: '' }
    };
  }

  private createForbiddenResponse(): GatewayResponse {
    return {
      statusCode: 403,
      headers: new Map(),
      body: { error: 'Forbidden' },
      metadata: { duration: 0, cached: false, service: 'gateway', endpoint: '', traceId: '' }
    };
  }

  private createNotFoundResponse(): GatewayResponse {
    return {
      statusCode: 404,
      headers: new Map(),
      body: { error: 'Not Found' },
      metadata: { duration: 0, cached: false, service: 'gateway', endpoint: '', traceId: '' }
    };
  }

  private createServiceUnavailableResponse(service: string): GatewayResponse {
    return {
      statusCode: 503,
      headers: new Map([['Retry-After', '30']]),
      body: { error: 'Service Unavailable', service },
      metadata: { duration: 0, cached: false, service: 'gateway', endpoint: '', traceId: '' }
    };
  }

  private createErrorResponse(error: Error): GatewayResponse {
    return {
      statusCode: 500,
      headers: new Map(),
      body: { error: 'Internal Server Error', message: error.message },
      metadata: { duration: 0, cached: false, service: 'gateway', endpoint: '', traceId: '' }
    };
  }
}

// ============================================================
// GRAPHQL FEDERATION
// ============================================================

export class GraphQLFederationGateway extends EventEmitter {
  private schemas: Map<string, FederatedSchema> = new Map();
  private queryPlanner: QueryPlanner;
  private executor: FederatedExecutor;

  constructor() {
    super();
    this.queryPlanner = new QueryPlanner();
    this.executor = new FederatedExecutor();
    this.registerSubgraphs();
  }

  private registerSubgraphs(): void {
    // Register all Mycelix service subgraphs
    const subgraphs: SubgraphConfig[] = [
      {
        name: 'users',
        url: 'http://users-service:4001/graphql',
        schema: this.getUsersSchema()
      },
      {
        name: 'tracks',
        url: 'http://tracks-service:4002/graphql',
        schema: this.getTracksSchema()
      },
      {
        name: 'playlists',
        url: 'http://playlists-service:4003/graphql',
        schema: this.getPlaylistsSchema()
      },
      {
        name: 'streaming',
        url: 'http://streaming-service:4004/graphql',
        schema: this.getStreamingSchema()
      },
      {
        name: 'social',
        url: 'http://social-service:4005/graphql',
        schema: this.getSocialSchema()
      },
      {
        name: 'analytics',
        url: 'http://analytics-service:4006/graphql',
        schema: this.getAnalyticsSchema()
      },
      {
        name: 'ai',
        url: 'http://ai-service:4007/graphql',
        schema: this.getAISchema()
      },
      {
        name: 'marketplace',
        url: 'http://marketplace-service:4008/graphql',
        schema: this.getMarketplaceSchema()
      }
    ];

    for (const subgraph of subgraphs) {
      this.schemas.set(subgraph.name, {
        name: subgraph.name,
        url: subgraph.url,
        schema: subgraph.schema,
        entities: this.extractEntities(subgraph.schema)
      });
    }
  }

  async executeQuery(query: GraphQLQuery): Promise<GraphQLResponse> {
    // Plan query execution across subgraphs
    const plan = await this.queryPlanner.plan(query, this.schemas);

    // Execute plan
    const results = await this.executor.execute(plan);

    // Merge results
    const mergedData = this.mergeResults(results);

    return {
      data: mergedData,
      errors: results.flatMap(r => r.errors || []),
      extensions: {
        queryPlan: plan,
        timing: results.map(r => ({ service: r.service, duration: r.duration }))
      }
    };
  }

  private getUsersSchema(): string {
    return `
      type User @key(fields: "id") {
        id: ID!
        email: String!
        username: String!
        displayName: String
        avatar: String
        bio: String
        createdAt: DateTime!
        followers: [User!]!
        following: [User!]!
        playlists: [Playlist!]!
        likedTracks: [Track!]!
        subscription: Subscription
        preferences: UserPreferences!
      }

      type Subscription {
        tier: SubscriptionTier!
        expiresAt: DateTime
        features: [String!]!
      }

      enum SubscriptionTier {
        FREE
        PREMIUM
        ARTIST
        LABEL
      }

      type UserPreferences {
        theme: String!
        language: String!
        audioQuality: AudioQuality!
        notifications: NotificationSettings!
      }

      type Query {
        me: User
        user(id: ID!): User
        users(filter: UserFilter, limit: Int, offset: Int): [User!]!
      }

      type Mutation {
        updateProfile(input: UpdateProfileInput!): User!
        followUser(userId: ID!): User!
        unfollowUser(userId: ID!): User!
      }
    `;
  }

  private getTracksSchema(): string {
    return `
      type Track @key(fields: "id") {
        id: ID!
        title: String!
        artist: Artist!
        album: Album
        duration: Int!
        audioUrl: String!
        waveformUrl: String
        coverArt: String
        genres: [Genre!]!
        mood: [String!]!
        bpm: Int
        key: String
        releaseDate: DateTime
        playCount: Int!
        likeCount: Int!
        isExplicit: Boolean!
        lyrics: Lyrics
        credits: [Credit!]!
        audioFeatures: AudioFeatures
      }

      type Artist @key(fields: "id") {
        id: ID!
        name: String!
        image: String
        verified: Boolean!
        tracks: [Track!]!
        albums: [Album!]!
        monthlyListeners: Int!
      }

      type Album @key(fields: "id") {
        id: ID!
        title: String!
        artist: Artist!
        tracks: [Track!]!
        coverArt: String!
        releaseDate: DateTime!
        type: AlbumType!
      }

      type AudioFeatures {
        danceability: Float!
        energy: Float!
        valence: Float!
        acousticness: Float!
        instrumentalness: Float!
        speechiness: Float!
        liveness: Float!
      }

      type Query {
        track(id: ID!): Track
        tracks(filter: TrackFilter, limit: Int, offset: Int): [Track!]!
        searchTracks(query: String!, limit: Int): [Track!]!
        recommendations(trackId: ID!, limit: Int): [Track!]!
      }
    `;
  }

  private getPlaylistsSchema(): string {
    return `
      type Playlist @key(fields: "id") {
        id: ID!
        name: String!
        description: String
        owner: User!
        coverImage: String
        isPublic: Boolean!
        collaborative: Boolean!
        tracks: [PlaylistTrack!]!
        followers: Int!
        duration: Int!
        createdAt: DateTime!
        updatedAt: DateTime!
      }

      type PlaylistTrack {
        track: Track!
        addedAt: DateTime!
        addedBy: User!
        position: Int!
      }

      type Query {
        playlist(id: ID!): Playlist
        myPlaylists: [Playlist!]!
        featuredPlaylists: [Playlist!]!
        categoryPlaylists(category: String!): [Playlist!]!
      }

      type Mutation {
        createPlaylist(input: CreatePlaylistInput!): Playlist!
        updatePlaylist(id: ID!, input: UpdatePlaylistInput!): Playlist!
        addTrackToPlaylist(playlistId: ID!, trackId: ID!): Playlist!
        removeTrackFromPlaylist(playlistId: ID!, trackId: ID!): Playlist!
        reorderPlaylist(playlistId: ID!, positions: [Int!]!): Playlist!
      }
    `;
  }

  private getStreamingSchema(): string {
    return `
      type PlaybackState {
        isPlaying: Boolean!
        currentTrack: Track
        position: Int!
        duration: Int!
        volume: Float!
        repeat: RepeatMode!
        shuffle: Boolean!
        queue: [Track!]!
        device: Device!
      }

      type Device {
        id: ID!
        name: String!
        type: DeviceType!
        isActive: Boolean!
        volumePercent: Int!
      }

      enum DeviceType {
        COMPUTER
        MOBILE
        SPEAKER
        TV
        CAR
        WATCH
      }

      type Query {
        playbackState: PlaybackState
        availableDevices: [Device!]!
        recentlyPlayed(limit: Int): [Track!]!
      }

      type Mutation {
        play(trackId: ID, context: PlayContextInput): PlaybackState!
        pause: PlaybackState!
        next: PlaybackState!
        previous: PlaybackState!
        seek(position: Int!): PlaybackState!
        setVolume(volume: Float!): PlaybackState!
        setRepeat(mode: RepeatMode!): PlaybackState!
        setShuffle(enabled: Boolean!): PlaybackState!
        transferPlayback(deviceId: ID!): PlaybackState!
        addToQueue(trackId: ID!): PlaybackState!
      }

      type Subscription {
        playbackStateChanged: PlaybackState!
      }
    `;
  }

  private getSocialSchema(): string {
    return `
      type Activity {
        id: ID!
        user: User!
        type: ActivityType!
        target: ActivityTarget!
        timestamp: DateTime!
      }

      union ActivityTarget = Track | Playlist | Album | User

      enum ActivityType {
        PLAYED
        LIKED
        ADDED_TO_PLAYLIST
        FOLLOWED
        SHARED
        COMMENTED
      }

      type Comment {
        id: ID!
        user: User!
        content: String!
        timestamp: DateTime!
        likes: Int!
        replies: [Comment!]!
      }

      type Query {
        feed(limit: Int, offset: Int): [Activity!]!
        friendActivity(limit: Int): [Activity!]!
        comments(targetType: String!, targetId: ID!): [Comment!]!
      }

      type Mutation {
        shareTrack(trackId: ID!, message: String): Activity!
        addComment(targetType: String!, targetId: ID!, content: String!): Comment!
        likeComment(commentId: ID!): Comment!
      }
    `;
  }

  private getAnalyticsSchema(): string {
    return `
      type ListeningStats {
        totalMinutes: Int!
        trackCount: Int!
        artistCount: Int!
        genreDistribution: [GenreStat!]!
        topTracks: [TrackStat!]!
        topArtists: [ArtistStat!]!
        listeningHistory: [ListeningSession!]!
        trends: ListeningTrends!
      }

      type GenreStat {
        genre: String!
        percentage: Float!
        minutes: Int!
      }

      type TrackStat {
        track: Track!
        playCount: Int!
        totalMinutes: Int!
      }

      type ArtistStat {
        artist: Artist!
        playCount: Int!
        totalMinutes: Int!
      }

      type ArtistAnalytics {
        streams: Int!
        listeners: Int!
        followers: Int!
        saves: Int!
        demographics: Demographics!
        geographics: [GeographicStat!]!
        revenue: RevenueStats!
      }

      type Query {
        myListeningStats(period: StatsPeriod!): ListeningStats!
        artistAnalytics(artistId: ID!, period: StatsPeriod!): ArtistAnalytics!
        trackAnalytics(trackId: ID!, period: StatsPeriod!): TrackAnalytics!
      }
    `;
  }

  private getAISchema(): string {
    return `
      type Recommendation {
        tracks: [Track!]!
        reason: String!
        confidence: Float!
      }

      type GeneratedPlaylist {
        name: String!
        description: String!
        tracks: [Track!]!
        mood: String!
        energy: Float!
      }

      type MusicAnalysis {
        track: Track!
        mood: [String!]!
        genres: [GenrePrediction!]!
        instruments: [String!]!
        vocals: VocalAnalysis
        structure: StructureAnalysis!
      }

      type Query {
        personalRecommendations(limit: Int): Recommendation!
        similarTracks(trackId: ID!, limit: Int): [Track!]!
        moodBasedTracks(mood: String!, limit: Int): [Track!]!
        discoverWeekly: GeneratedPlaylist!
        analyzeTrack(trackId: ID!): MusicAnalysis!
      }

      type Mutation {
        generatePlaylist(prompt: String!): GeneratedPlaylist!
        provideFeedback(trackId: ID!, liked: Boolean!): Boolean!
      }
    `;
  }

  private getMarketplaceSchema(): string {
    return `
      type License {
        id: ID!
        track: Track!
        type: LicenseType!
        price: Money!
        terms: String!
        usageRights: [String!]!
      }

      type NFT {
        id: ID!
        track: Track!
        tokenId: String!
        owner: User!
        edition: Int!
        totalEditions: Int!
        price: Money!
        royaltyPercentage: Float!
      }

      type Merchandise {
        id: ID!
        artist: Artist!
        name: String!
        description: String!
        price: Money!
        images: [String!]!
        variants: [MerchVariant!]!
      }

      type Query {
        licenses(trackId: ID!): [License!]!
        nfts(filter: NFTFilter): [NFT!]!
        merchandise(artistId: ID!): [Merchandise!]!
        myPurchases: [Purchase!]!
      }

      type Mutation {
        purchaseLicense(licenseId: ID!): Purchase!
        purchaseNFT(nftId: ID!): Purchase!
        listNFT(nftId: ID!, price: MoneyInput!): NFT!
      }
    `;
  }

  private extractEntities(schema: string): string[] {
    const entityRegex = /type\s+(\w+)\s+@key/g;
    const entities: string[] = [];
    let match;
    while ((match = entityRegex.exec(schema)) !== null) {
      entities.push(match[1]);
    }
    return entities;
  }

  private mergeResults(results: ExecutionResult[]): any {
    const merged: any = {};
    for (const result of results) {
      if (result.data) {
        Object.assign(merged, result.data);
      }
    }
    return merged;
  }
}

// ============================================================
// EVENT BUS & MESSAGE QUEUE
// ============================================================

export class EventBus extends EventEmitter {
  private subscribers: Map<string, EventSubscriber[]> = new Map();
  private deadLetterQueue: DeadLetterQueue;
  private retryPolicy: RetryPolicy;

  constructor() {
    super();
    this.deadLetterQueue = new DeadLetterQueue();
    this.retryPolicy = new RetryPolicy();
  }

  async publish(event: DomainEvent): Promise<void> {
    const subscribers = this.subscribers.get(event.type) || [];

    for (const subscriber of subscribers) {
      try {
        await this.deliverWithRetry(event, subscriber);
      } catch (error) {
        await this.deadLetterQueue.add(event, subscriber, error as Error);
      }
    }

    this.emit('eventPublished', event);
  }

  subscribe(eventType: string, handler: EventHandler): Subscription {
    const subscriber: EventSubscriber = {
      id: this.generateSubscriberId(),
      eventType,
      handler,
      filter: null,
      createdAt: new Date()
    };

    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, []);
    }
    this.subscribers.get(eventType)!.push(subscriber);

    return {
      unsubscribe: () => {
        const subs = this.subscribers.get(eventType) || [];
        const index = subs.findIndex(s => s.id === subscriber.id);
        if (index > -1) subs.splice(index, 1);
      }
    };
  }

  private async deliverWithRetry(event: DomainEvent, subscriber: EventSubscriber): Promise<void> {
    let attempts = 0;
    const maxAttempts = this.retryPolicy.maxAttempts;

    while (attempts < maxAttempts) {
      try {
        await subscriber.handler(event);
        return;
      } catch (error) {
        attempts++;
        if (attempts >= maxAttempts) throw error;
        await this.delay(this.retryPolicy.getDelay(attempts));
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private generateSubscriberId(): string {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

interface DomainEvent {
  id: string;
  type: string;
  aggregateId: string;
  aggregateType: string;
  payload: any;
  metadata: EventMetadata;
  timestamp: Date;
}

interface EventMetadata {
  correlationId: string;
  causationId?: string;
  userId?: string;
  version: number;
}

type EventHandler = (event: DomainEvent) => Promise<void>;

interface EventSubscriber {
  id: string;
  eventType: string;
  handler: EventHandler;
  filter: EventFilter | null;
  createdAt: Date;
}

interface EventFilter {
  aggregateType?: string;
  aggregateId?: string;
  metadata?: Record<string, any>;
}

interface Subscription {
  unsubscribe: () => void;
}

// ============================================================
// HELPER CLASSES
// ============================================================

class RequestRouter {
  private routes: Map<string, RouteDefinition> = new Map();

  constructor(services: ServiceRegistry) {
    this.buildRoutes(services);
  }

  private buildRoutes(services: ServiceRegistry): void {
    const routeMappings: [string, string][] = [
      ['/api/v1/users', 'users'],
      ['/api/v1/auth', 'auth'],
      ['/api/v1/tracks', 'tracks'],
      ['/api/v1/albums', 'tracks'],
      ['/api/v1/artists', 'tracks'],
      ['/api/v1/playlists', 'playlists'],
      ['/api/v1/streaming', 'streaming'],
      ['/api/v1/social', 'social'],
      ['/api/v1/analytics', 'analytics'],
      ['/api/v1/ai', 'ai'],
      ['/api/v1/marketplace', 'marketplace'],
      ['/api/v1/search', 'search']
    ];

    for (const [path, service] of routeMappings) {
      this.routes.set(path, { prefix: path, service });
    }
  }

  resolve(request: GatewayRequest): ResolvedRoute | null {
    for (const [prefix, route] of this.routes) {
      if (request.path.startsWith(prefix)) {
        return {
          service: route.service,
          path: request.path
        };
      }
    }
    return null;
  }
}

interface RouteDefinition {
  prefix: string;
  service: string;
}

interface ResolvedRoute {
  service: string;
  path: string;
}

class Authenticator {
  constructor(private config: SecurityConfig) {}

  async authenticate(request: GatewayRequest): Promise<AuthResult> {
    const authHeader = request.headers.get('Authorization');

    if (authHeader?.startsWith('Bearer ')) {
      const token = authHeader.substring(7);
      return this.verifyJWT(token);
    }

    const apiKey = request.headers.get(this.config.apiKey.headerName) ||
                   request.query.get(this.config.apiKey.queryParam);

    if (apiKey) {
      return this.verifyAPIKey(apiKey);
    }

    return { authenticated: false, user: undefined };
  }

  private async verifyJWT(token: string): Promise<AuthResult> {
    // JWT verification would happen here
    try {
      const payload = this.decodeToken(token);
      return {
        authenticated: true,
        user: {
          id: payload.sub,
          email: payload.email,
          roles: payload.roles || [],
          permissions: payload.permissions || [],
          tier: payload.tier || 'free',
          metadata: payload.metadata || {}
        }
      };
    } catch {
      return { authenticated: false, user: undefined };
    }
  }

  private async verifyAPIKey(apiKey: string): Promise<AuthResult> {
    // API key verification would happen here
    return { authenticated: false, user: undefined };
  }

  private decodeToken(token: string): any {
    // Simplified - would use proper JWT library
    const parts = token.split('.');
    if (parts.length !== 3) throw new Error('Invalid token');
    return JSON.parse(Buffer.from(parts[1], 'base64').toString());
  }
}

interface AuthResult {
  authenticated: boolean;
  user?: AuthenticatedUser;
}

class RateLimiter {
  private windows: Map<string, RateLimitWindow> = new Map();

  constructor(private config: RateLimitConfig) {}

  async check(request: GatewayRequest): Promise<RateLimitResult> {
    if (!this.config.enabled) {
      return { allowed: true, limit: 0, remaining: 0, retryAfter: 0 };
    }

    const key = this.getKey(request);
    const window = this.getOrCreateWindow(key);

    window.count++;

    const tier = this.getTier(request);
    const limit = tier?.maxRequests || this.config.maxRequests;

    if (window.count > limit) {
      return {
        allowed: false,
        limit,
        remaining: 0,
        retryAfter: Math.ceil((window.expiresAt - Date.now()) / 1000)
      };
    }

    return {
      allowed: true,
      limit,
      remaining: limit - window.count,
      retryAfter: 0
    };
  }

  private getKey(request: GatewayRequest): string {
    switch (this.config.keyGenerator) {
      case 'user':
        return request.user?.id || request.metadata.clientIp;
      case 'api_key':
        return request.headers.get('X-API-Key') || request.metadata.clientIp;
      default:
        return request.metadata.clientIp;
    }
  }

  private getOrCreateWindow(key: string): RateLimitWindow {
    let window = this.windows.get(key);

    if (!window || window.expiresAt < Date.now()) {
      window = {
        count: 0,
        expiresAt: Date.now() + this.config.windowMs
      };
      this.windows.set(key, window);
    }

    return window;
  }

  private getTier(request: GatewayRequest): RateLimitTier | undefined {
    return this.config.tiers.find(tier => {
      if (tier.condition === 'premium' && request.user?.tier === 'premium') return true;
      if (tier.condition === 'authenticated' && request.user) return true;
      return false;
    });
  }
}

interface RateLimitWindow {
  count: number;
  expiresAt: number;
}

interface RateLimitResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  retryAfter: number;
}

class CacheManager {
  private cache: Map<string, CachedResponse> = new Map();

  constructor(private config: CacheConfig) {}

  async get(request: GatewayRequest): Promise<GatewayResponse | null> {
    if (!this.config.enabled) return null;

    const key = this.generateKey(request);
    const cached = this.cache.get(key);

    if (cached && cached.expiresAt > Date.now()) {
      return cached.response;
    }

    return null;
  }

  async set(request: GatewayRequest, response: GatewayResponse): Promise<void> {
    if (!this.config.enabled) return;

    const key = this.generateKey(request);
    const ttl = this.getTTL(request.path);

    this.cache.set(key, {
      response,
      expiresAt: Date.now() + ttl * 1000
    });

    // Cleanup old entries
    if (this.cache.size > this.config.maxSize) {
      this.evictOldest();
    }
  }

  private generateKey(request: GatewayRequest): string {
    const queryString = Array.from(request.query.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `${k}=${v}`)
      .join('&');

    return `${request.method}:${request.path}?${queryString}`;
  }

  private getTTL(path: string): number {
    for (const strategy of this.config.strategies) {
      if (path.match(new RegExp(strategy.pattern))) {
        return strategy.ttl;
      }
    }
    return this.config.ttl;
  }

  private evictOldest(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].expiresAt - b[1].expiresAt);

    const toRemove = entries.slice(0, Math.floor(this.config.maxSize * 0.1));
    for (const [key] of toRemove) {
      this.cache.delete(key);
    }
  }
}

interface CachedResponse {
  response: GatewayResponse;
  expiresAt: number;
}

class CircuitBreaker {
  private state: 'closed' | 'open' | 'half_open' = 'closed';
  private failures: number = 0;
  private lastFailure: number = 0;
  private successes: number = 0;

  constructor(private config: { threshold: number; timeout: number; resetTimeout: number }) {}

  allowRequest(): boolean {
    if (this.state === 'closed') return true;

    if (this.state === 'open') {
      if (Date.now() - this.lastFailure > this.config.resetTimeout) {
        this.state = 'half_open';
        return true;
      }
      return false;
    }

    return true; // half_open
  }

  recordSuccess(): void {
    if (this.state === 'half_open') {
      this.successes++;
      if (this.successes >= 3) {
        this.state = 'closed';
        this.failures = 0;
        this.successes = 0;
      }
    }
  }

  recordFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();

    if (this.failures >= this.config.threshold) {
      this.state = 'open';
    }
  }
}

class GatewayMetrics {
  private requests: MetricCounter = { total: 0, success: 0, error: 0 };
  private latencies: number[] = [];
  private cacheHits: number = 0;

  recordRequest(request: GatewayRequest, response: GatewayResponse, duration: number): void {
    this.requests.total++;
    if (response.statusCode < 400) {
      this.requests.success++;
    } else {
      this.requests.error++;
    }
    this.latencies.push(duration);
  }

  recordCacheHit(path: string): void {
    this.cacheHits++;
  }

  recordError(request: GatewayRequest, error: Error): void {
    this.requests.error++;
  }

  getMetrics(): GatewayMetricsReport {
    return {
      requests: this.requests,
      latency: {
        avg: this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length || 0,
        p50: this.percentile(50),
        p95: this.percentile(95),
        p99: this.percentile(99)
      },
      cacheHitRate: this.cacheHits / this.requests.total || 0
    };
  }

  private percentile(p: number): number {
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const index = Math.floor(sorted.length * p / 100);
    return sorted[index] || 0;
  }
}

interface MetricCounter {
  total: number;
  success: number;
  error: number;
}

interface GatewayMetricsReport {
  requests: MetricCounter;
  latency: { avg: number; p50: number; p95: number; p99: number };
  cacheHitRate: number;
}

interface ServiceHealthStatus {
  service: string;
  healthy: boolean;
  lastCheck: Date;
  endpoints: { url: string; healthy: boolean; latency: number }[];
}

interface SubgraphConfig {
  name: string;
  url: string;
  schema: string;
}

interface FederatedSchema {
  name: string;
  url: string;
  schema: string;
  entities: string[];
}

interface GraphQLQuery {
  query: string;
  variables?: Record<string, any>;
  operationName?: string;
}

interface GraphQLResponse {
  data: any;
  errors: any[];
  extensions?: any;
}

interface ExecutionResult {
  service: string;
  data: any;
  errors: any[];
  duration: number;
}

class QueryPlanner {
  async plan(query: GraphQLQuery, schemas: Map<string, FederatedSchema>): Promise<QueryPlan> {
    return { steps: [] };
  }
}

interface QueryPlan {
  steps: QueryPlanStep[];
}

interface QueryPlanStep {
  service: string;
  query: string;
  dependsOn: string[];
}

class FederatedExecutor {
  async execute(plan: QueryPlan): Promise<ExecutionResult[]> {
    return [];
  }
}

class DeadLetterQueue {
  private queue: DeadLetter[] = [];

  async add(event: DomainEvent, subscriber: EventSubscriber, error: Error): Promise<void> {
    this.queue.push({
      event,
      subscriber: subscriber.id,
      error: error.message,
      timestamp: new Date(),
      retryCount: 0
    });
  }
}

interface DeadLetter {
  event: DomainEvent;
  subscriber: string;
  error: string;
  timestamp: Date;
  retryCount: number;
}

class RetryPolicy {
  maxAttempts = 3;

  getDelay(attempt: number): number {
    return Math.pow(2, attempt) * 1000; // Exponential backoff
  }
}
