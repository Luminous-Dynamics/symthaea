// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Platform - End-to-End Demo System
 *
 * Complete demonstration environment with:
 * - Realistic seed data generation
 * - Demo scenarios and workflows
 * - Interactive API playground
 * - Sample frontend components
 * - Performance testing harness
 */

import * as crypto from 'crypto';
import { EventEmitter } from 'events';

// ============================================================================
// SEED DATA GENERATION
// ============================================================================

interface SeedConfig {
  artists: number;
  albumsPerArtist: number;
  tracksPerAlbum: number;
  users: number;
  playlistsPerUser: number;
  tracksPerPlaylist: number;
}

interface GeneratedData {
  artists: Artist[];
  albums: Album[];
  tracks: Track[];
  users: User[];
  playlists: Playlist[];
  follows: Follow[];
  plays: Play[];
}

interface Artist {
  id: string;
  name: string;
  bio: string;
  genres: string[];
  country: string;
  imageUrl: string;
  verified: boolean;
  monthlyListeners: number;
  createdAt: Date;
}

interface Album {
  id: string;
  title: string;
  artistId: string;
  releaseDate: Date;
  albumType: 'album' | 'single' | 'ep';
  imageUrl: string;
  label: string;
  copyright: string;
}

interface Track {
  id: string;
  title: string;
  artistId: string;
  albumId: string;
  duration: number;
  trackNumber: number;
  discNumber: number;
  isExplicit: boolean;
  isrc: string;
  popularity: number;
  previewUrl: string;
}

interface User {
  id: string;
  email: string;
  displayName: string;
  avatarUrl: string;
  country: string;
  subscription: 'free' | 'premium' | 'family';
  createdAt: Date;
}

interface Playlist {
  id: string;
  name: string;
  description: string;
  ownerId: string;
  isPublic: boolean;
  imageUrl: string;
  trackIds: string[];
}

interface Follow {
  followerId: string;
  followeeId: string;
  followeeType: 'artist' | 'user' | 'playlist';
  createdAt: Date;
}

interface Play {
  id: string;
  userId: string;
  trackId: string;
  playedAt: Date;
  duration: number;
  context: 'album' | 'playlist' | 'artist' | 'search';
  device: string;
}

/**
 * Realistic seed data generator
 */
class SeedDataGenerator {
  private genres = [
    'Pop', 'Rock', 'Hip Hop', 'R&B', 'Electronic', 'Jazz', 'Classical',
    'Country', 'Folk', 'Metal', 'Punk', 'Indie', 'Alternative', 'Soul',
    'Funk', 'Reggae', 'Blues', 'Latin', 'World', 'Ambient'
  ];

  private adjectives = [
    'Electric', 'Midnight', 'Golden', 'Silver', 'Neon', 'Crystal', 'Shadow',
    'Velvet', 'Cosmic', 'Eternal', 'Wild', 'Silent', 'Burning', 'Frozen',
    'Broken', 'Sacred', 'Digital', 'Ancient', 'Modern', 'Infinite'
  ];

  private nouns = [
    'Dreams', 'Stars', 'Hearts', 'Nights', 'Days', 'Roads', 'Rivers',
    'Mountains', 'Cities', 'Skies', 'Waves', 'Flames', 'Echoes', 'Voices',
    'Memories', 'Shadows', 'Lights', 'Storms', 'Gardens', 'Oceans'
  ];

  private countries = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR', 'MX', 'KR'];
  private devices = ['mobile', 'desktop', 'smart_speaker', 'car', 'tv', 'web'];

  /**
   * Generate complete seed data
   */
  async generate(config: SeedConfig): Promise<GeneratedData> {
    console.log('Generating seed data...');

    const artists = this.generateArtists(config.artists);
    const albums = this.generateAlbums(artists, config.albumsPerArtist);
    const tracks = this.generateTracks(albums, config.tracksPerAlbum);
    const users = this.generateUsers(config.users);
    const playlists = this.generatePlaylists(users, tracks, config.playlistsPerUser, config.tracksPerPlaylist);
    const follows = this.generateFollows(users, artists, playlists);
    const plays = this.generatePlays(users, tracks, 10000);

    return { artists, albums, tracks, users, playlists, follows, plays };
  }

  private generateArtists(count: number): Artist[] {
    return Array.from({ length: count }, (_, i) => ({
      id: crypto.randomUUID(),
      name: this.generateArtistName(),
      bio: this.generateBio(),
      genres: this.pickRandom(this.genres, Math.floor(Math.random() * 3) + 1),
      country: this.pickRandom(this.countries, 1)[0],
      imageUrl: `https://picsum.photos/seed/artist${i}/400/400`,
      verified: Math.random() > 0.7,
      monthlyListeners: Math.floor(Math.random() * 10000000),
      createdAt: this.randomDate(new Date('2020-01-01'), new Date()),
    }));
  }

  private generateAlbums(artists: Artist[], perArtist: number): Album[] {
    const albums: Album[] = [];
    const albumTypes: Album['albumType'][] = ['album', 'single', 'ep'];

    for (const artist of artists) {
      for (let i = 0; i < perArtist; i++) {
        albums.push({
          id: crypto.randomUUID(),
          title: this.generateAlbumTitle(),
          artistId: artist.id,
          releaseDate: this.randomDate(new Date('2020-01-01'), new Date()),
          albumType: albumTypes[Math.floor(Math.random() * albumTypes.length)],
          imageUrl: `https://picsum.photos/seed/album${albums.length}/300/300`,
          label: `${this.pickRandom(this.adjectives, 1)[0]} Records`,
          copyright: `© ${new Date().getFullYear()} ${artist.name}`,
        });
      }
    }

    return albums;
  }

  private generateTracks(albums: Album[], perAlbum: number): Track[] {
    const tracks: Track[] = [];

    for (const album of albums) {
      const trackCount = album.albumType === 'single' ? 1 :
                        album.albumType === 'ep' ? Math.min(perAlbum, 5) :
                        perAlbum;

      for (let i = 0; i < trackCount; i++) {
        tracks.push({
          id: crypto.randomUUID(),
          title: this.generateTrackTitle(),
          artistId: album.artistId,
          albumId: album.id,
          duration: Math.floor(Math.random() * 180000) + 120000, // 2-5 minutes
          trackNumber: i + 1,
          discNumber: 1,
          isExplicit: Math.random() > 0.8,
          isrc: this.generateISRC(),
          popularity: Math.floor(Math.random() * 100),
          previewUrl: `https://audio.mycelix.music/preview/${tracks.length}.mp3`,
        });
      }
    }

    return tracks;
  }

  private generateUsers(count: number): User[] {
    const subscriptions: User['subscription'][] = ['free', 'premium', 'family'];
    const firstNames = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Quinn', 'Avery'];
    const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'];

    return Array.from({ length: count }, (_, i) => ({
      id: crypto.randomUUID(),
      email: `user${i}@example.com`,
      displayName: `${this.pickRandom(firstNames, 1)[0]} ${this.pickRandom(lastNames, 1)[0]}`,
      avatarUrl: `https://i.pravatar.cc/150?u=${i}`,
      country: this.pickRandom(this.countries, 1)[0],
      subscription: subscriptions[Math.floor(Math.random() * subscriptions.length)],
      createdAt: this.randomDate(new Date('2021-01-01'), new Date()),
    }));
  }

  private generatePlaylists(users: User[], tracks: Track[], perUser: number, tracksPerPlaylist: number): Playlist[] {
    const playlists: Playlist[] = [];
    const playlistThemes = ['Chill', 'Workout', 'Focus', 'Party', 'Road Trip', 'Mood', 'Discover', 'Throwback'];

    for (const user of users) {
      for (let i = 0; i < perUser; i++) {
        const theme = this.pickRandom(playlistThemes, 1)[0];
        playlists.push({
          id: crypto.randomUUID(),
          name: `${user.displayName}'s ${theme} Mix`,
          description: `A collection of ${theme.toLowerCase()} tracks curated by ${user.displayName}`,
          ownerId: user.id,
          isPublic: Math.random() > 0.3,
          imageUrl: `https://picsum.photos/seed/playlist${playlists.length}/300/300`,
          trackIds: this.pickRandom(tracks, tracksPerPlaylist).map(t => t.id),
        });
      }
    }

    return playlists;
  }

  private generateFollows(users: User[], artists: Artist[], playlists: Playlist[]): Follow[] {
    const follows: Follow[] = [];

    for (const user of users) {
      // Follow some artists
      const artistsToFollow = this.pickRandom(artists, Math.floor(Math.random() * 20) + 5);
      for (const artist of artistsToFollow) {
        follows.push({
          followerId: user.id,
          followeeId: artist.id,
          followeeType: 'artist',
          createdAt: this.randomDate(user.createdAt, new Date()),
        });
      }

      // Follow some users
      const usersToFollow = this.pickRandom(users.filter(u => u.id !== user.id), Math.floor(Math.random() * 10));
      for (const otherUser of usersToFollow) {
        follows.push({
          followerId: user.id,
          followeeId: otherUser.id,
          followeeType: 'user',
          createdAt: this.randomDate(user.createdAt, new Date()),
        });
      }

      // Follow some playlists
      const playlistsToFollow = this.pickRandom(
        playlists.filter(p => p.ownerId !== user.id),
        Math.floor(Math.random() * 15)
      );
      for (const playlist of playlistsToFollow) {
        follows.push({
          followerId: user.id,
          followeeId: playlist.id,
          followeeType: 'playlist',
          createdAt: this.randomDate(user.createdAt, new Date()),
        });
      }
    }

    return follows;
  }

  private generatePlays(users: User[], tracks: Track[], count: number): Play[] {
    const contexts: Play['context'][] = ['album', 'playlist', 'artist', 'search'];

    return Array.from({ length: count }, () => {
      const user = this.pickRandom(users, 1)[0];
      const track = this.pickRandom(tracks, 1)[0];

      return {
        id: crypto.randomUUID(),
        userId: user.id,
        trackId: track.id,
        playedAt: this.randomDate(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), new Date()),
        duration: Math.floor(Math.random() * track.duration),
        context: contexts[Math.floor(Math.random() * contexts.length)],
        device: this.pickRandom(this.devices, 1)[0],
      };
    });
  }

  // Helper methods
  private generateArtistName(): string {
    const formats = [
      () => `The ${this.pickRandom(this.adjectives, 1)[0]} ${this.pickRandom(this.nouns, 1)[0]}`,
      () => `${this.pickRandom(this.adjectives, 1)[0]}${this.pickRandom(this.nouns, 1)[0]}`,
      () => this.pickRandom(['DJ', 'MC', 'Dr.', 'Kid', 'Lil', 'Young', 'Big'], 1)[0] + ' ' + this.generateWord(),
      () => this.generateWord() + ' & The ' + this.pickRandom(this.nouns, 1)[0],
    ];
    return formats[Math.floor(Math.random() * formats.length)]();
  }

  private generateAlbumTitle(): string {
    return `${this.pickRandom(this.adjectives, 1)[0]} ${this.pickRandom(this.nouns, 1)[0]}`;
  }

  private generateTrackTitle(): string {
    const formats = [
      () => `${this.pickRandom(this.adjectives, 1)[0]} ${this.pickRandom(this.nouns, 1)[0]}`,
      () => `The ${this.pickRandom(this.nouns, 1)[0]}`,
      () => `${this.pickRandom(this.nouns, 1)[0]} of ${this.pickRandom(this.nouns, 1)[0]}`,
      () => this.pickRandom(this.nouns, 1)[0],
    ];
    return formats[Math.floor(Math.random() * formats.length)]();
  }

  private generateBio(): string {
    const templates = [
      'An innovative artist pushing the boundaries of {genre} music.',
      'Known for their unique blend of {genre} and {genre2}, they have captivated audiences worldwide.',
      'Rising star in the {genre} scene with a fresh perspective on modern music.',
      'Veteran {genre} artist with decades of experience and countless hits.',
    ];
    const template = templates[Math.floor(Math.random() * templates.length)];
    return template
      .replace('{genre}', this.pickRandom(this.genres, 1)[0])
      .replace('{genre2}', this.pickRandom(this.genres, 1)[0]);
  }

  private generateISRC(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    const country = 'US';
    const registrant = Array.from({ length: 3 }, () => chars[Math.floor(Math.random() * 26)]).join('');
    const year = new Date().getFullYear().toString().slice(-2);
    const designation = Array.from({ length: 5 }, () => Math.floor(Math.random() * 10)).join('');
    return `${country}${registrant}${year}${designation}`;
  }

  private generateWord(): string {
    const syllables = ['ka', 'ri', 'mo', 'na', 'te', 'su', 'mi', 'ro', 'la', 'ze', 'vo', 'xi'];
    const length = Math.floor(Math.random() * 2) + 2;
    return Array.from({ length }, () => syllables[Math.floor(Math.random() * syllables.length)])
      .join('')
      .replace(/^./, c => c.toUpperCase());
  }

  private pickRandom<T>(array: T[], count: number): T[] {
    const shuffled = [...array].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
  }

  private randomDate(start: Date, end: Date): Date {
    return new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));
  }
}

// ============================================================================
// DEMO SCENARIOS
// ============================================================================

interface DemoScenario {
  id: string;
  name: string;
  description: string;
  steps: DemoStep[];
  expectedDuration: number; // seconds
}

interface DemoStep {
  action: string;
  endpoint: string;
  method: string;
  body?: any;
  expectedStatus: number;
  delay?: number;
}

/**
 * Pre-built demo scenarios for showcasing the platform
 */
const demoScenarios: DemoScenario[] = [
  {
    id: 'user-onboarding',
    name: 'User Onboarding Flow',
    description: 'Complete user registration, profile setup, and first playlist creation',
    expectedDuration: 60,
    steps: [
      {
        action: 'Register new user',
        endpoint: '/api/auth/register',
        method: 'POST',
        body: { email: 'demo@example.com', password: 'demo123', displayName: 'Demo User' },
        expectedStatus: 201,
      },
      {
        action: 'Verify email',
        endpoint: '/api/auth/verify-email',
        method: 'POST',
        body: { token: '{{verificationToken}}' },
        expectedStatus: 200,
        delay: 2000,
      },
      {
        action: 'Login',
        endpoint: '/api/auth/login',
        method: 'POST',
        body: { email: 'demo@example.com', password: 'demo123' },
        expectedStatus: 200,
      },
      {
        action: 'Get personalized recommendations',
        endpoint: '/api/recommendations?limit=10',
        method: 'GET',
        expectedStatus: 200,
      },
      {
        action: 'Create first playlist',
        endpoint: '/api/playlists',
        method: 'POST',
        body: { name: 'My First Playlist', description: 'Getting started!' },
        expectedStatus: 201,
      },
      {
        action: 'Add tracks to playlist',
        endpoint: '/api/playlists/{{playlistId}}/tracks',
        method: 'POST',
        body: { trackIds: ['{{track1}}', '{{track2}}', '{{track3}}'] },
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'music-discovery',
    name: 'Music Discovery Journey',
    description: 'Search for music, explore artists, and discover new tracks',
    expectedDuration: 120,
    steps: [
      {
        action: 'Search for an artist',
        endpoint: '/api/search?q=electronic&type=artist&limit=5',
        method: 'GET',
        expectedStatus: 200,
      },
      {
        action: 'View artist profile',
        endpoint: '/api/artists/{{artistId}}',
        method: 'GET',
        expectedStatus: 200,
      },
      {
        action: 'Get artist top tracks',
        endpoint: '/api/artists/{{artistId}}/top-tracks?limit=10',
        method: 'GET',
        expectedStatus: 200,
      },
      {
        action: 'Get artist albums',
        endpoint: '/api/artists/{{artistId}}/albums',
        method: 'GET',
        expectedStatus: 200,
      },
      {
        action: 'Follow artist',
        endpoint: '/api/artists/{{artistId}}/follow',
        method: 'POST',
        expectedStatus: 200,
      },
      {
        action: 'Get similar artists',
        endpoint: '/api/artists/{{artistId}}/related',
        method: 'GET',
        expectedStatus: 200,
      },
      {
        action: 'Stream a track',
        endpoint: '/api/tracks/{{trackId}}/stream',
        method: 'GET',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'playlist-collaboration',
    name: 'Collaborative Playlist',
    description: 'Create and manage a collaborative playlist with friends',
    expectedDuration: 90,
    steps: [
      {
        action: 'Create collaborative playlist',
        endpoint: '/api/playlists',
        method: 'POST',
        body: { name: 'Party Mix', isCollaborative: true, isPublic: true },
        expectedStatus: 201,
      },
      {
        action: 'Invite collaborator',
        endpoint: '/api/playlists/{{playlistId}}/collaborators',
        method: 'POST',
        body: { userId: '{{friendId}}' },
        expectedStatus: 200,
      },
      {
        action: 'Add tracks',
        endpoint: '/api/playlists/{{playlistId}}/tracks',
        method: 'POST',
        body: { trackIds: ['{{track1}}', '{{track2}}'] },
        expectedStatus: 200,
      },
      {
        action: 'Reorder tracks',
        endpoint: '/api/playlists/{{playlistId}}/tracks/reorder',
        method: 'PUT',
        body: { rangeStart: 0, insertBefore: 2, rangeLength: 1 },
        expectedStatus: 200,
      },
      {
        action: 'Share playlist',
        endpoint: '/api/playlists/{{playlistId}}/share',
        method: 'POST',
        body: { platform: 'twitter' },
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'listening-session',
    name: 'Extended Listening Session',
    description: 'Simulate a realistic listening session with track plays and skips',
    expectedDuration: 300,
    steps: [
      {
        action: 'Start playback',
        endpoint: '/api/player/play',
        method: 'PUT',
        body: { trackId: '{{track1}}', context: 'playlist:{{playlistId}}' },
        expectedStatus: 200,
      },
      {
        action: 'Report playback progress',
        endpoint: '/api/player/progress',
        method: 'POST',
        body: { trackId: '{{track1}}', position: 30000, duration: 180000 },
        expectedStatus: 200,
        delay: 30000,
      },
      {
        action: 'Skip to next track',
        endpoint: '/api/player/next',
        method: 'POST',
        expectedStatus: 200,
        delay: 5000,
      },
      {
        action: 'Like current track',
        endpoint: '/api/tracks/{{track2}}/like',
        method: 'POST',
        expectedStatus: 200,
      },
      {
        action: 'Add to queue',
        endpoint: '/api/player/queue',
        method: 'POST',
        body: { trackId: '{{track3}}' },
        expectedStatus: 200,
      },
    ],
  },
];

// ============================================================================
// INTERACTIVE API PLAYGROUND
// ============================================================================

interface PlaygroundRequest {
  id: string;
  endpoint: string;
  method: string;
  headers: Record<string, string>;
  body?: any;
  timestamp: Date;
}

interface PlaygroundResponse {
  requestId: string;
  status: number;
  headers: Record<string, string>;
  body: any;
  duration: number;
  timestamp: Date;
}

/**
 * Interactive API playground for developers
 */
class APIPlayground extends EventEmitter {
  private history: Array<{ request: PlaygroundRequest; response: PlaygroundResponse }> = [];
  private savedRequests: Map<string, PlaygroundRequest> = new Map();
  private baseUrl: string;
  private authToken?: string;

  constructor(baseUrl: string = 'http://localhost:3000') {
    super();
    this.baseUrl = baseUrl;
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string): void {
    this.authToken = token;
  }

  /**
   * Execute an API request
   */
  async execute(
    endpoint: string,
    method: string = 'GET',
    body?: any,
    headers: Record<string, string> = {}
  ): Promise<PlaygroundResponse> {
    const request: PlaygroundRequest = {
      id: crypto.randomUUID(),
      endpoint,
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(this.authToken ? { 'Authorization': `Bearer ${this.authToken}` } : {}),
        ...headers,
      },
      body,
      timestamp: new Date(),
    };

    this.emit('request', request);

    const startTime = Date.now();

    try {
      const fetchResponse = await fetch(`${this.baseUrl}${endpoint}`, {
        method,
        headers: request.headers,
        body: body ? JSON.stringify(body) : undefined,
      });

      const responseBody = await fetchResponse.json().catch(() => null);
      const responseHeaders: Record<string, string> = {};
      fetchResponse.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      const response: PlaygroundResponse = {
        requestId: request.id,
        status: fetchResponse.status,
        headers: responseHeaders,
        body: responseBody,
        duration: Date.now() - startTime,
        timestamp: new Date(),
      };

      this.history.push({ request, response });
      this.emit('response', response);

      return response;
    } catch (error) {
      const response: PlaygroundResponse = {
        requestId: request.id,
        status: 0,
        headers: {},
        body: { error: (error as Error).message },
        duration: Date.now() - startTime,
        timestamp: new Date(),
      };

      this.history.push({ request, response });
      this.emit('error', { request, error });

      return response;
    }
  }

  /**
   * Save a request for later use
   */
  saveRequest(name: string, request: Omit<PlaygroundRequest, 'id' | 'timestamp'>): void {
    this.savedRequests.set(name, {
      ...request,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    });
  }

  /**
   * Get saved request
   */
  getSavedRequest(name: string): PlaygroundRequest | undefined {
    return this.savedRequests.get(name);
  }

  /**
   * Get request history
   */
  getHistory(): Array<{ request: PlaygroundRequest; response: PlaygroundResponse }> {
    return [...this.history];
  }

  /**
   * Clear history
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * Generate curl command for a request
   */
  toCurl(request: PlaygroundRequest): string {
    let curl = `curl -X ${request.method} '${this.baseUrl}${request.endpoint}'`;

    for (const [key, value] of Object.entries(request.headers)) {
      curl += ` \\\n  -H '${key}: ${value}'`;
    }

    if (request.body) {
      curl += ` \\\n  -d '${JSON.stringify(request.body)}'`;
    }

    return curl;
  }

  /**
   * Generate code snippet for a request
   */
  toCode(request: PlaygroundRequest, language: 'javascript' | 'python' | 'go'): string {
    switch (language) {
      case 'javascript':
        return this.toJavaScript(request);
      case 'python':
        return this.toPython(request);
      case 'go':
        return this.toGo(request);
    }
  }

  private toJavaScript(request: PlaygroundRequest): string {
    return `
const response = await fetch('${this.baseUrl}${request.endpoint}', {
  method: '${request.method}',
  headers: ${JSON.stringify(request.headers, null, 2)},
  ${request.body ? `body: JSON.stringify(${JSON.stringify(request.body, null, 2)}),` : ''}
});

const data = await response.json();
console.log(data);
    `.trim();
  }

  private toPython(request: PlaygroundRequest): string {
    return `
import requests

response = requests.${request.method.toLowerCase()}(
    '${this.baseUrl}${request.endpoint}',
    headers=${JSON.stringify(request.headers)},
    ${request.body ? `json=${JSON.stringify(request.body)},` : ''}
)

print(response.json())
    `.trim();
  }

  private toGo(request: PlaygroundRequest): string {
    return `
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    ${request.body ? `body, _ := json.Marshal(${JSON.stringify(request.body)})
    req, _ := http.NewRequest("${request.method}", "${this.baseUrl}${request.endpoint}", bytes.NewBuffer(body))` :
    `req, _ := http.NewRequest("${request.method}", "${this.baseUrl}${request.endpoint}", nil)`}

    ${Object.entries(request.headers).map(([k, v]) => `req.Header.Set("${k}", "${v}")`).join('\n    ')}

    client := &http.Client{}
    resp, _ := client.Do(req)
    defer resp.Body.Close()

    fmt.Println(resp.Status)
}
    `.trim();
  }
}

// ============================================================================
// SAMPLE FRONTEND COMPONENTS
// ============================================================================

/**
 * React component templates for demo frontend
 */
const frontendComponents = {
  TrackCard: `
import React from 'react';

interface Track {
  id: string;
  title: string;
  artist: { name: string };
  album: { imageUrl: string };
  duration: number;
}

interface TrackCardProps {
  track: Track;
  onPlay: (trackId: string) => void;
  onAddToPlaylist: (trackId: string) => void;
}

export function TrackCard({ track, onPlay, onAddToPlaylist }: TrackCardProps) {
  const formatDuration = (ms: number) => {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return \`\${minutes}:\${seconds.toString().padStart(2, '0')}\`;
  };

  return (
    <div className="flex items-center gap-4 p-3 hover:bg-gray-100 rounded-lg group">
      <img
        src={track.album.imageUrl}
        alt={track.title}
        className="w-12 h-12 rounded"
      />

      <div className="flex-1 min-w-0">
        <p className="font-medium truncate">{track.title}</p>
        <p className="text-sm text-gray-500 truncate">{track.artist.name}</p>
      </div>

      <span className="text-sm text-gray-400">
        {formatDuration(track.duration)}
      </span>

      <div className="opacity-0 group-hover:opacity-100 flex gap-2">
        <button
          onClick={() => onPlay(track.id)}
          className="p-2 hover:bg-gray-200 rounded-full"
        >
          <PlayIcon className="w-5 h-5" />
        </button>
        <button
          onClick={() => onAddToPlaylist(track.id)}
          className="p-2 hover:bg-gray-200 rounded-full"
        >
          <PlusIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}
  `,

  PlayerControls: `
import React, { useState, useEffect } from 'react';

interface PlayerControlsProps {
  currentTrack: Track | null;
  isPlaying: boolean;
  progress: number;
  onPlayPause: () => void;
  onNext: () => void;
  onPrevious: () => void;
  onSeek: (position: number) => void;
}

export function PlayerControls({
  currentTrack,
  isPlaying,
  progress,
  onPlayPause,
  onNext,
  onPrevious,
  onSeek,
}: PlayerControlsProps) {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t p-4">
      <div className="max-w-screen-xl mx-auto flex items-center gap-4">
        {/* Track Info */}
        {currentTrack && (
          <div className="flex items-center gap-3 w-1/4">
            <img
              src={currentTrack.album.imageUrl}
              alt={currentTrack.title}
              className="w-14 h-14 rounded"
            />
            <div>
              <p className="font-medium">{currentTrack.title}</p>
              <p className="text-sm text-gray-500">{currentTrack.artist.name}</p>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="flex-1 flex flex-col items-center">
          <div className="flex items-center gap-4 mb-2">
            <button onClick={onPrevious}>
              <SkipBackIcon className="w-5 h-5" />
            </button>
            <button
              onClick={onPlayPause}
              className="w-10 h-10 bg-black text-white rounded-full flex items-center justify-center"
            >
              {isPlaying ? <PauseIcon /> : <PlayIcon />}
            </button>
            <button onClick={onNext}>
              <SkipForwardIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Progress Bar */}
          <div className="w-full max-w-md flex items-center gap-2">
            <span className="text-xs text-gray-500">
              {formatTime(progress)}
            </span>
            <input
              type="range"
              min={0}
              max={currentTrack?.duration || 100}
              value={progress}
              onChange={(e) => onSeek(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-xs text-gray-500">
              {formatTime(currentTrack?.duration || 0)}
            </span>
          </div>
        </div>

        {/* Volume */}
        <div className="w-1/4 flex justify-end">
          <VolumeSlider />
        </div>
      </div>
    </div>
  );
}
  `,

  SearchBar: `
import React, { useState, useCallback } from 'react';
import { debounce } from 'lodash';

interface SearchBarProps {
  onSearch: (query: string) => void;
  onResults: (results: SearchResults) => void;
}

export function SearchBar({ onSearch, onResults }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const debouncedSearch = useCallback(
    debounce(async (q: string) => {
      if (q.length < 2) return;

      setIsLoading(true);
      try {
        const response = await fetch(\`/api/search?q=\${encodeURIComponent(q)}\`);
        const results = await response.json();
        onResults(results);
      } finally {
        setIsLoading(false);
      }
    }, 300),
    [onResults]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    debouncedSearch(value);
  };

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={handleChange}
        placeholder="Search tracks, artists, albums..."
        className="w-full px-4 py-2 pl-10 bg-gray-100 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
      {isLoading && (
        <Spinner className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5" />
      )}
    </div>
  );
}
  `,
};

// ============================================================================
// DEMO RUNNER
// ============================================================================

/**
 * Demo scenario runner
 */
class DemoRunner extends EventEmitter {
  private playground: APIPlayground;
  private variables: Map<string, string> = new Map();

  constructor(baseUrl: string = 'http://localhost:3000') {
    super();
    this.playground = new APIPlayground(baseUrl);
  }

  /**
   * Set a variable for template replacement
   */
  setVariable(name: string, value: string): void {
    this.variables.set(name, value);
  }

  /**
   * Run a demo scenario
   */
  async runScenario(scenario: DemoScenario): Promise<void> {
    this.emit('scenarioStart', scenario);

    for (let i = 0; i < scenario.steps.length; i++) {
      const step = scenario.steps[i];
      this.emit('stepStart', { scenario, step, index: i });

      // Template replacement
      const endpoint = this.replaceTemplates(step.endpoint);
      const body = step.body ? this.replaceTemplates(JSON.stringify(step.body)) : undefined;

      // Execute step
      const response = await this.playground.execute(
        endpoint,
        step.method,
        body ? JSON.parse(body) : undefined
      );

      // Check expected status
      if (response.status !== step.expectedStatus) {
        this.emit('stepFailed', {
          scenario,
          step,
          expected: step.expectedStatus,
          actual: response.status,
        });
        throw new Error(`Step failed: expected ${step.expectedStatus}, got ${response.status}`);
      }

      // Extract variables from response
      this.extractVariables(response.body);

      this.emit('stepComplete', { scenario, step, response, index: i });

      // Delay if specified
      if (step.delay) {
        await this.delay(step.delay);
      }
    }

    this.emit('scenarioComplete', scenario);
  }

  private replaceTemplates(text: string): string {
    return text.replace(/\{\{(\w+)\}\}/g, (_, name) => {
      return this.variables.get(name) || `{{${name}}}`;
    });
  }

  private extractVariables(body: any): void {
    if (!body || typeof body !== 'object') return;

    // Extract common IDs
    if (body.data?.id) {
      const type = body.data.type || 'item';
      this.variables.set(`${type}Id`, body.data.id);
    }

    if (body.accessToken) {
      this.variables.set('accessToken', body.accessToken);
      this.playground.setAuthToken(body.accessToken);
    }

    if (body.verificationToken) {
      this.variables.set('verificationToken', body.verificationToken);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  // Seed data
  SeedDataGenerator,
  SeedConfig,
  GeneratedData,
  Artist,
  Album,
  Track,
  User,
  Playlist,
  Follow,
  Play,

  // Demo scenarios
  demoScenarios,
  DemoScenario,
  DemoStep,

  // API Playground
  APIPlayground,
  PlaygroundRequest,
  PlaygroundResponse,

  // Frontend components
  frontendComponents,

  // Demo runner
  DemoRunner,
};

/**
 * Create the complete demo system
 */
export async function createDemoSystem(baseUrl: string = 'http://localhost:3000'): Promise<{
  generator: SeedDataGenerator;
  playground: APIPlayground;
  runner: DemoRunner;
  scenarios: DemoScenario[];
  components: typeof frontendComponents;
}> {
  const generator = new SeedDataGenerator();
  const playground = new APIPlayground(baseUrl);
  const runner = new DemoRunner(baseUrl);

  return {
    generator,
    playground,
    runner,
    scenarios: demoScenarios,
    components: frontendComponents,
  };
}

/**
 * Generate and seed demo data
 */
export async function seedDemoData(config?: Partial<SeedConfig>): Promise<GeneratedData> {
  const generator = new SeedDataGenerator();

  const defaultConfig: SeedConfig = {
    artists: 100,
    albumsPerArtist: 3,
    tracksPerAlbum: 10,
    users: 500,
    playlistsPerUser: 5,
    tracksPerPlaylist: 20,
  };

  return generator.generate({ ...defaultConfig, ...config });
}
