// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * GraphQL Layer
 *
 * Alternative query interface for flexible data fetching.
 * Supports queries, mutations, and subscriptions.
 */

import { Pool } from 'pg';
import { createBatchLoader } from '../resilience/request-coalescing';
import { getLogger } from '../logging';

const logger = getLogger();

/**
 * GraphQL Schema Definition
 */
export const typeDefs = `
  scalar DateTime
  scalar JSON

  type Query {
    # Songs
    song(id: ID!): Song
    songs(
      filter: SongFilter
      pagination: PaginationInput
      sort: SongSort
    ): SongConnection!

    # Artists
    artist(address: String!): Artist
    artists(
      pagination: PaginationInput
      sort: ArtistSort
    ): ArtistConnection!

    # Plays
    plays(
      filter: PlayFilter
      pagination: PaginationInput
    ): PlayConnection!

    # Search
    search(query: String!, types: [SearchType!], limit: Int): SearchResults!

    # Analytics
    analytics(
      artistAddress: String
      dateFrom: DateTime
      dateTo: DateTime
    ): Analytics!

    # Trending
    trending(
      period: TrendingPeriod
      limit: Int
    ): TrendingResults!
  }

  type Mutation {
    # Songs
    createSong(input: CreateSongInput!): Song!
    updateSong(id: ID!, input: UpdateSongInput!): Song!
    deleteSong(id: ID!): Boolean!

    # Plays
    recordPlay(input: RecordPlayInput!): Play!

    # Webhooks
    createWebhook(input: CreateWebhookInput!): Webhook!
    updateWebhook(id: ID!, input: UpdateWebhookInput!): Webhook!
    deleteWebhook(id: ID!): Boolean!
  }

  type Subscription {
    playRecorded(songId: ID, artistAddress: String): Play!
    songUpdated(artistAddress: String): Song!
  }

  # Types
  type Song {
    id: ID!
    title: String!
    artistAddress: String!
    artistName: String
    ipfsHash: String!
    genre: String
    duration: Int
    playCount: Int!
    metadata: JSON
    createdAt: DateTime!
    updatedAt: DateTime!

    # Relations
    artist: Artist
    recentPlays(limit: Int): [Play!]!
    analytics: SongAnalytics
  }

  type Artist {
    id: String!
    address: String!
    name: String
    songCount: Int!
    totalPlays: Int!
    uniqueListeners: Int!

    # Relations
    songs(pagination: PaginationInput): SongConnection!
    topSongs(limit: Int): [Song!]!
  }

  type Play {
    id: ID!
    songId: ID!
    walletAddress: String!
    duration: Int
    playedAt: DateTime!
    source: String

    # Relations
    song: Song
  }

  type Webhook {
    id: ID!
    url: String!
    events: [String!]!
    active: Boolean!
    createdAt: DateTime!
  }

  type SongAnalytics {
    totalPlays: Int!
    uniqueListeners: Int!
    averagePlayDuration: Float
    playsByDay: [DailyStats!]!
  }

  type DailyStats {
    date: DateTime!
    count: Int!
  }

  type Analytics {
    totalPlays: Int!
    uniqueListeners: Int!
    uniqueSongs: Int!
    topGenres: [GenreStats!]!
    playsByDay: [DailyStats!]!
  }

  type GenreStats {
    genre: String!
    playCount: Int!
  }

  type TrendingResults {
    songs: [Song!]!
    artists: [Artist!]!
    genres: [GenreStats!]!
  }

  type SearchResults {
    songs: [Song!]!
    artists: [Artist!]!
    total: Int!
  }

  # Connections (for pagination)
  type SongConnection {
    edges: [SongEdge!]!
    pageInfo: PageInfo!
    totalCount: Int!
  }

  type SongEdge {
    node: Song!
    cursor: String!
  }

  type ArtistConnection {
    edges: [ArtistEdge!]!
    pageInfo: PageInfo!
    totalCount: Int!
  }

  type ArtistEdge {
    node: Artist!
    cursor: String!
  }

  type PlayConnection {
    edges: [PlayEdge!]!
    pageInfo: PageInfo!
    totalCount: Int!
  }

  type PlayEdge {
    node: Play!
    cursor: String!
  }

  type PageInfo {
    hasNextPage: Boolean!
    hasPreviousPage: Boolean!
    startCursor: String
    endCursor: String
  }

  # Inputs
  input SongFilter {
    artistAddress: String
    genre: String
    minPlays: Int
    createdAfter: DateTime
    createdBefore: DateTime
  }

  input PlayFilter {
    songId: ID
    walletAddress: String
    artistAddress: String
    playedAfter: DateTime
    playedBefore: DateTime
  }

  input PaginationInput {
    first: Int
    after: String
    last: Int
    before: String
  }

  input SongSort {
    field: SongSortField!
    direction: SortDirection!
  }

  input ArtistSort {
    field: ArtistSortField!
    direction: SortDirection!
  }

  input CreateSongInput {
    title: String!
    artistAddress: String!
    ipfsHash: String!
    genre: String
    duration: Int
    metadata: JSON
  }

  input UpdateSongInput {
    title: String
    genre: String
    metadata: JSON
  }

  input RecordPlayInput {
    songId: ID!
    walletAddress: String!
    duration: Int
    source: String
  }

  input CreateWebhookInput {
    url: String!
    events: [String!]!
  }

  input UpdateWebhookInput {
    url: String
    events: [String!]
    active: Boolean
  }

  # Enums
  enum SongSortField {
    CREATED_AT
    PLAY_COUNT
    TITLE
  }

  enum ArtistSortField {
    TOTAL_PLAYS
    SONG_COUNT
    NAME
  }

  enum SortDirection {
    ASC
    DESC
  }

  enum SearchType {
    SONG
    ARTIST
  }

  enum TrendingPeriod {
    DAY
    WEEK
    MONTH
  }
`;

/**
 * Create GraphQL resolvers
 */
export function createResolvers(pool: Pool) {
  // DataLoaders for batching
  const songLoader = createBatchLoader<string, any>(
    'songs',
    async (ids) => {
      const result = await pool.query(
        'SELECT * FROM songs WHERE id = ANY($1) AND deleted_at IS NULL',
        [ids]
      );
      return new Map(result.rows.map(row => [row.id, row]));
    }
  );

  const artistLoader = createBatchLoader<string, any>(
    'artists',
    async (addresses) => {
      const result = await pool.query(`
        SELECT
          artist_address as address,
          artist_name as name,
          COUNT(*) as song_count,
          SUM(play_count) as total_plays
        FROM songs
        WHERE artist_address = ANY($1) AND deleted_at IS NULL
        GROUP BY artist_address, artist_name
      `, [addresses]);
      return new Map(result.rows.map(row => [row.address, row]));
    }
  );

  return {
    Query: {
      // Song queries
      song: async (_: any, { id }: { id: string }) => {
        return songLoader.load(id);
      },

      songs: async (_: any, { filter, pagination, sort }: any) => {
        let query = 'SELECT * FROM songs WHERE deleted_at IS NULL';
        const params: unknown[] = [];

        if (filter?.artistAddress) {
          params.push(filter.artistAddress.toLowerCase());
          query += ` AND artist_address = $${params.length}`;
        }
        if (filter?.genre) {
          params.push(filter.genre);
          query += ` AND genre = $${params.length}`;
        }
        if (filter?.minPlays) {
          params.push(filter.minPlays);
          query += ` AND play_count >= $${params.length}`;
        }

        // Sorting
        const sortField = sort?.field === 'PLAY_COUNT' ? 'play_count' :
                         sort?.field === 'TITLE' ? 'title' : 'created_at';
        const sortDir = sort?.direction === 'ASC' ? 'ASC' : 'DESC';
        query += ` ORDER BY ${sortField} ${sortDir}`;

        // Pagination
        const limit = pagination?.first || 20;
        params.push(limit);
        query += ` LIMIT $${params.length}`;

        const result = await pool.query(query, params);
        const countResult = await pool.query(
          'SELECT COUNT(*) FROM songs WHERE deleted_at IS NULL'
        );

        return {
          edges: result.rows.map((row: any) => ({
            node: row,
            cursor: Buffer.from(row.id).toString('base64'),
          })),
          pageInfo: {
            hasNextPage: result.rows.length === limit,
            hasPreviousPage: false,
            startCursor: result.rows[0] ? Buffer.from(result.rows[0].id).toString('base64') : null,
            endCursor: result.rows.length > 0 ? Buffer.from(result.rows[result.rows.length - 1].id).toString('base64') : null,
          },
          totalCount: parseInt(countResult.rows[0].count),
        };
      },

      // Artist queries
      artist: async (_: any, { address }: { address: string }) => {
        return artistLoader.load(address.toLowerCase());
      },

      artists: async (_: any, { pagination, sort }: any) => {
        const sortField = sort?.field === 'SONG_COUNT' ? 'song_count' :
                         sort?.field === 'NAME' ? 'artist_name' : 'total_plays';
        const sortDir = sort?.direction === 'ASC' ? 'ASC' : 'DESC';

        const result = await pool.query(`
          SELECT
            artist_address as address,
            artist_name as name,
            COUNT(*) as song_count,
            SUM(play_count) as total_plays
          FROM songs
          WHERE deleted_at IS NULL
          GROUP BY artist_address, artist_name
          ORDER BY ${sortField} ${sortDir}
          LIMIT $1
        `, [pagination?.first || 20]);

        return {
          edges: result.rows.map((row: any) => ({
            node: { ...row, id: row.address },
            cursor: Buffer.from(row.address).toString('base64'),
          })),
          pageInfo: {
            hasNextPage: result.rows.length === (pagination?.first || 20),
            hasPreviousPage: false,
          },
          totalCount: result.rows.length,
        };
      },

      // Search
      search: async (_: any, { query, types, limit }: any) => {
        const searchQuery = `%${query}%`;
        const maxResults = limit || 20;

        const songs = await pool.query(`
          SELECT * FROM songs
          WHERE deleted_at IS NULL
          AND (title ILIKE $1 OR artist_name ILIKE $1)
          ORDER BY play_count DESC
          LIMIT $2
        `, [searchQuery, maxResults]);

        return {
          songs: songs.rows,
          artists: [], // Would need separate artist table
          total: songs.rows.length,
        };
      },

      // Analytics
      analytics: async (_: any, { artistAddress, dateFrom, dateTo }: any) => {
        let query = `
          SELECT
            COUNT(*) as total_plays,
            COUNT(DISTINCT wallet_address) as unique_listeners,
            COUNT(DISTINCT song_id) as unique_songs
          FROM plays p
          JOIN songs s ON p.song_id = s.id
          WHERE 1=1
        `;
        const params: unknown[] = [];

        if (artistAddress) {
          params.push(artistAddress.toLowerCase());
          query += ` AND s.artist_address = $${params.length}`;
        }
        if (dateFrom) {
          params.push(dateFrom);
          query += ` AND p.played_at >= $${params.length}`;
        }
        if (dateTo) {
          params.push(dateTo);
          query += ` AND p.played_at <= $${params.length}`;
        }

        const result = await pool.query(query, params);
        return result.rows[0];
      },

      // Trending
      trending: async (_: any, { period, limit }: any) => {
        const interval = period === 'DAY' ? '1 day' :
                        period === 'WEEK' ? '7 days' : '30 days';
        const maxResults = limit || 10;

        const songs = await pool.query(`
          SELECT s.*, COUNT(p.id) as recent_plays
          FROM songs s
          JOIN plays p ON s.id = p.song_id
          WHERE p.played_at > NOW() - INTERVAL '${interval}'
          AND s.deleted_at IS NULL
          GROUP BY s.id
          ORDER BY recent_plays DESC
          LIMIT $1
        `, [maxResults]);

        return {
          songs: songs.rows,
          artists: [],
          genres: [],
        };
      },
    },

    Mutation: {
      createSong: async (_: any, { input }: any) => {
        const result = await pool.query(`
          INSERT INTO songs (title, artist_address, ipfs_hash, genre, duration, metadata)
          VALUES ($1, $2, $3, $4, $5, $6)
          RETURNING *
        `, [
          input.title,
          input.artistAddress.toLowerCase(),
          input.ipfsHash,
          input.genre,
          input.duration,
          input.metadata || {},
        ]);
        return result.rows[0];
      },

      updateSong: async (_: any, { id, input }: any) => {
        const setClauses: string[] = [];
        const params: unknown[] = [id];

        if (input.title) {
          params.push(input.title);
          setClauses.push(`title = $${params.length}`);
        }
        if (input.genre) {
          params.push(input.genre);
          setClauses.push(`genre = $${params.length}`);
        }
        if (input.metadata) {
          params.push(JSON.stringify(input.metadata));
          setClauses.push(`metadata = $${params.length}`);
        }

        if (setClauses.length === 0) {
          return songLoader.load(id);
        }

        const result = await pool.query(`
          UPDATE songs SET ${setClauses.join(', ')}, updated_at = NOW()
          WHERE id = $1 AND deleted_at IS NULL
          RETURNING *
        `, params);

        return result.rows[0];
      },

      deleteSong: async (_: any, { id }: any) => {
        const result = await pool.query(
          'UPDATE songs SET deleted_at = NOW() WHERE id = $1 AND deleted_at IS NULL',
          [id]
        );
        return result.rowCount > 0;
      },

      recordPlay: async (_: any, { input }: any) => {
        const result = await pool.query(`
          INSERT INTO plays (song_id, wallet_address, duration, source)
          VALUES ($1, $2, $3, $4)
          RETURNING *
        `, [
          input.songId,
          input.walletAddress.toLowerCase(),
          input.duration,
          input.source || 'graphql',
        ]);

        // Update play count
        await pool.query(
          'UPDATE songs SET play_count = play_count + 1 WHERE id = $1',
          [input.songId]
        );

        return result.rows[0];
      },
    },

    // Field resolvers
    Song: {
      artist: async (song: any) => {
        return artistLoader.load(song.artist_address);
      },
      recentPlays: async (song: any, { limit }: any) => {
        const result = await pool.query(
          'SELECT * FROM plays WHERE song_id = $1 ORDER BY played_at DESC LIMIT $2',
          [song.id, limit || 10]
        );
        return result.rows;
      },
    },

    Artist: {
      songs: async (artist: any, { pagination }: any) => {
        const result = await pool.query(`
          SELECT * FROM songs
          WHERE artist_address = $1 AND deleted_at IS NULL
          ORDER BY play_count DESC
          LIMIT $2
        `, [artist.address, pagination?.first || 20]);

        return {
          edges: result.rows.map((row: any) => ({ node: row, cursor: row.id })),
          pageInfo: { hasNextPage: false, hasPreviousPage: false },
          totalCount: result.rows.length,
        };
      },
      topSongs: async (artist: any, { limit }: any) => {
        const result = await pool.query(`
          SELECT * FROM songs
          WHERE artist_address = $1 AND deleted_at IS NULL
          ORDER BY play_count DESC
          LIMIT $2
        `, [artist.address, limit || 5]);
        return result.rows;
      },
    },

    Play: {
      song: async (play: any) => {
        return songLoader.load(play.song_id);
      },
    },
  };
}

/**
 * Simple GraphQL executor (for use without full GraphQL server)
 */
export async function executeGraphQL(
  pool: Pool,
  query: string,
  variables?: Record<string, unknown>
): Promise<{ data?: unknown; errors?: unknown[] }> {
  // This is a simplified executor - in production use graphql-js or Apollo
  logger.info('GraphQL query executed', { query: query.slice(0, 100) });

  // Would need actual GraphQL execution here
  return { data: null, errors: [{ message: 'Full GraphQL execution requires graphql-js' }] };
}

export default { typeDefs, createResolvers, executeGraphQL };
