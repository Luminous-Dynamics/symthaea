// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { GraphQLScalarType, Kind } from 'graphql';
import { PubSub, withFilter } from 'graphql-subscriptions';
import { AuthenticationError, ForbiddenError, UserInputError } from 'apollo-server-errors';
import GraphQLUpload from 'graphql-upload/GraphQLUpload.js';

// Services (to be injected)
interface Context {
  user?: {
    id: string;
    role: string;
  };
  dataSources: {
    users: any;
    tracks: any;
    albums: any;
    playlists: any;
    collaborations: any;
    notifications: any;
    analytics: any;
    streaming: any;
    storage: any;
    search: any;
    royalties: any;
    nft: any;
  };
  pubsub: PubSub;
}

// Custom Scalars
const DateTimeScalar = new GraphQLScalarType({
  name: 'DateTime',
  description: 'ISO-8601 formatted date-time string',
  serialize(value: unknown): string {
    if (value instanceof Date) {
      return value.toISOString();
    }
    throw new Error('DateTime must be a Date object');
  },
  parseValue(value: unknown): Date {
    if (typeof value === 'string') {
      return new Date(value);
    }
    throw new Error('DateTime must be a string');
  },
  parseLiteral(ast): Date | null {
    if (ast.kind === Kind.STRING) {
      return new Date(ast.value);
    }
    return null;
  },
});

const JSONScalar = new GraphQLScalarType({
  name: 'JSON',
  description: 'Arbitrary JSON value',
  serialize(value: unknown): unknown {
    return value;
  },
  parseValue(value: unknown): unknown {
    return value;
  },
  parseLiteral(ast, variables): unknown {
    switch (ast.kind) {
      case Kind.STRING:
      case Kind.BOOLEAN:
        return ast.value;
      case Kind.INT:
      case Kind.FLOAT:
        return parseFloat(ast.value);
      case Kind.OBJECT: {
        const result: Record<string, unknown> = {};
        ast.fields.forEach((field) => {
          result[field.name.value] = JSONScalar.parseLiteral!(field.value, variables);
        });
        return result;
      }
      case Kind.LIST:
        return ast.values.map((v) => JSONScalar.parseLiteral!(v, variables));
      case Kind.NULL:
        return null;
      default:
        return undefined;
    }
  },
});

// Auth helpers
function requireAuth(context: Context) {
  if (!context.user) {
    throw new AuthenticationError('You must be logged in');
  }
  return context.user;
}

function requireRole(context: Context, roles: string[]) {
  const user = requireAuth(context);
  if (!roles.includes(user.role)) {
    throw new ForbiddenError('You do not have permission to perform this action');
  }
  return user;
}

// Subscription events
const EVENTS = {
  NOW_PLAYING: 'NOW_PLAYING',
  COLLABORATION_UPDATED: 'COLLABORATION_UPDATED',
  COLLABORATION_MESSAGE: 'COLLABORATION_MESSAGE',
  COLLABORATION_FILE: 'COLLABORATION_FILE',
  NEW_FOLLOWER: 'NEW_FOLLOWER',
  NEW_LIKE: 'NEW_LIKE',
  NEW_COMMENT: 'NEW_COMMENT',
  NOTIFICATION: 'NOTIFICATION',
  LIVE_STREAM_STARTED: 'LIVE_STREAM_STARTED',
  LIVE_STREAM_ENDED: 'LIVE_STREAM_ENDED',
  LIVE_STREAM_VIEWERS: 'LIVE_STREAM_VIEWERS',
  TRACK_PROCESSING: 'TRACK_PROCESSING',
};

export const resolvers = {
  // Scalars
  DateTime: DateTimeScalar,
  JSON: JSONScalar,
  Upload: GraphQLUpload,

  // Type Resolvers
  User: {
    followers: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getFollowers(parent.id);
    },
    following: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getFollowing(parent.id);
    },
    tracks: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getByArtist(parent.id);
    },
    albums: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.albums.getByArtist(parent.id);
    },
    playlists: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.playlists.getByOwner(parent.id);
    },
    likedTracks: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getLikedByUser(parent.id);
    },
    collaborations: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.collaborations.getByUser(parent.id);
    },
    royalties: async (parent: any, _: any, { dataSources, user }: Context) => {
      if (user?.id !== parent.id) return null;
      return dataSources.royalties.getSummary(parent.id);
    },
  },

  Track: {
    artist: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getById(parent.artistId);
    },
    featuredArtists: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getByIds(parent.featuredArtistIds || []);
    },
    album: async (parent: any, _: any, { dataSources }: Context) => {
      return parent.albumId ? dataSources.albums.getById(parent.albumId) : null;
    },
    genre: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getGenre(parent.genreId);
    },
    comments: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getComments(parent.id);
    },
    audioFiles: async (parent: any, _: any, { dataSources, user }: Context) => {
      // Filter based on user subscription
      const quality = user ? await dataSources.users.getMaxQuality(user.id) : 'STANDARD_256';
      return dataSources.streaming.getAudioFiles(parent.id, quality);
    },
  },

  Album: {
    artist: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getById(parent.artistId);
    },
    tracks: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getByAlbum(parent.id);
    },
    genre: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getGenre(parent.genreId);
    },
    label: async (parent: any, _: any, { dataSources }: Context) => {
      return parent.labelId ? dataSources.users.getLabel(parent.labelId) : null;
    },
  },

  Playlist: {
    owner: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getById(parent.ownerId);
    },
    collaborators: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getByIds(parent.collaboratorIds || []);
    },
    tracks: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.playlists.getTracks(parent.id);
    },
  },

  Comment: {
    author: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.users.getById(parent.authorId);
    },
    track: async (parent: any, _: any, { dataSources }: Context) => {
      return parent.trackId ? dataSources.tracks.getById(parent.trackId) : null;
    },
    replies: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.tracks.getCommentReplies(parent.id);
    },
  },

  Collaboration: {
    participants: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.collaborations.getParticipants(parent.id);
    },
    messages: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.collaborations.getMessages(parent.id);
    },
    files: async (parent: any, _: any, { dataSources }: Context) => {
      return dataSources.collaborations.getFiles(parent.id);
    },
  },

  FeedItem: {
    __resolveType(obj: any) {
      if (obj.duration !== undefined) return 'Track';
      if (obj.trackCount !== undefined && obj.tracks !== undefined) return 'Album';
      if (obj.visibility !== undefined) return 'Playlist';
      return 'ActivityItem';
    },
  },

  FeedItemTarget: {
    __resolveType(obj: any) {
      if (obj.duration !== undefined) return 'Track';
      if (obj.trackCount !== undefined) return 'Album';
      if (obj.visibility !== undefined) return 'Playlist';
      return 'User';
    },
  },

  // Queries
  Query: {
    // User queries
    me: (_: any, __: any, context: Context) => {
      return context.user
        ? context.dataSources.users.getById(context.user.id)
        : null;
    },

    user: async (_: any, { id, username }: any, { dataSources }: Context) => {
      if (id) return dataSources.users.getById(id);
      if (username) return dataSources.users.getByUsername(username);
      throw new UserInputError('Must provide id or username');
    },

    users: async (_: any, args: any, { dataSources }: Context) => {
      return dataSources.users.search(args);
    },

    // Track queries
    track: async (_: any, { id, slug }: any, { dataSources }: Context) => {
      if (id) return dataSources.tracks.getById(id);
      if (slug) return dataSources.tracks.getBySlug(slug);
      throw new UserInputError('Must provide id or slug');
    },

    tracks: async (_: any, args: any, { dataSources }: Context) => {
      return dataSources.tracks.list(args);
    },

    trendingTracks: async (_: any, { genreId, timeRange, limit }: any, { dataSources }: Context) => {
      return dataSources.tracks.getTrending({ genreId, timeRange, limit: limit || 50 });
    },

    newReleases: async (_: any, { genreId, limit }: any, { dataSources }: Context) => {
      return dataSources.tracks.getNewReleases({ genreId, limit: limit || 50 });
    },

    recommendedTracks: async (_: any, { limit }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.tracks.getRecommended(user.id, limit || 50);
    },

    // Album queries
    album: async (_: any, { id, slug }: any, { dataSources }: Context) => {
      if (id) return dataSources.albums.getById(id);
      if (slug) return dataSources.albums.getBySlug(slug);
      throw new UserInputError('Must provide id or slug');
    },

    albums: async (_: any, args: any, { dataSources }: Context) => {
      return dataSources.albums.list(args);
    },

    // Playlist queries
    playlist: async (_: any, { id, slug }: any, { dataSources }: Context) => {
      if (id) return dataSources.playlists.getById(id);
      if (slug) return dataSources.playlists.getBySlug(slug);
      throw new UserInputError('Must provide id or slug');
    },

    playlists: async (_: any, args: any, { dataSources }: Context) => {
      return dataSources.playlists.list(args);
    },

    featuredPlaylists: async (_: any, { limit }: any, { dataSources }: Context) => {
      return dataSources.playlists.getFeatured(limit || 20);
    },

    // Genre queries
    genre: async (_: any, { id, slug }: any, { dataSources }: Context) => {
      if (id) return dataSources.tracks.getGenre(id);
      if (slug) return dataSources.tracks.getGenreBySlug(slug);
      throw new UserInputError('Must provide id or slug');
    },

    genres: async (_: any, { parentId }: any, { dataSources }: Context) => {
      return dataSources.tracks.getGenres(parentId);
    },

    // Search
    search: async (_: any, { input, pagination }: any, { dataSources }: Context) => {
      return dataSources.search.search({ ...input, ...pagination });
    },

    autocomplete: async (_: any, { query, limit }: any, { dataSources }: Context) => {
      return dataSources.search.autocomplete(query, limit || 10);
    },

    // Feed
    feed: async (_: any, { pagination }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.users.getFeed(user.id, pagination);
    },

    // Collaborations
    collaboration: async (_: any, { id }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.collaborations.getById(id, user.id);
    },

    myCollaborations: async (_: any, { status }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.collaborations.getByUser(user.id, status);
    },

    // Analytics
    trackAnalytics: async (_: any, { trackId, dateRange }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.analytics.getTrackAnalytics(trackId, user.id, dateRange);
    },

    artistAnalytics: async (_: any, { dateRange }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.analytics.getArtistAnalytics(user.id, dateRange);
    },

    // Notifications
    notifications: async (_: any, { unreadOnly, pagination }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.notifications.list(user.id, { unreadOnly, ...pagination });
    },

    unreadNotificationCount: async (_: any, __: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.notifications.getUnreadCount(user.id);
    },

    // Royalties
    royalties: async (_: any, { dateRange }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.royalties.getSummary(user.id, dateRange);
    },

    // Streaming URLs
    streamUrl: async (_: any, { trackId, quality }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.streaming.getStreamUrl(trackId, user.id, quality);
    },

    downloadUrl: async (_: any, { trackId, quality }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.streaming.getDownloadUrl(trackId, user.id, quality);
    },
  },

  // Mutations
  Mutation: {
    // Auth
    signUp: async (_: any, { email, password, username }: any, { dataSources }: Context) => {
      return dataSources.users.signUp({ email, password, username });
    },

    signIn: async (_: any, { email, password }: any, { dataSources }: Context) => {
      return dataSources.users.signIn({ email, password });
    },

    signOut: async (_: any, __: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.users.signOut(user.id);
    },

    refreshToken: async (_: any, { refreshToken }: any, { dataSources }: Context) => {
      return dataSources.users.refreshToken(refreshToken);
    },

    // User mutations
    updateUser: async (_: any, { input }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.users.update(user.id, input);
    },

    updateSettings: async (_: any, { input }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.users.updateSettings(user.id, input);
    },

    updateAvatar: async (_: any, { file }: any, context: Context) => {
      const user = requireAuth(context);
      const { createReadStream, filename, mimetype } = await file;
      const url = await context.dataSources.storage.uploadImage(createReadStream(), filename, mimetype);
      return context.dataSources.users.update(user.id, { avatar: url });
    },

    // Social
    followUser: async (_: any, { userId }: any, context: Context) => {
      const user = requireAuth(context);
      const result = await context.dataSources.users.follow(user.id, userId);
      context.pubsub.publish(EVENTS.NEW_FOLLOWER, { newFollower: user, userId });
      return result;
    },

    unfollowUser: async (_: any, { userId }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.users.unfollow(user.id, userId);
    },

    // Track mutations
    createTrack: async (_: any, { input, audioFile, coverArt }: any, context: Context) => {
      const user = requireAuth(context);

      // Upload audio file
      const { createReadStream, filename, mimetype } = await audioFile;
      const audioUrl = await context.dataSources.storage.uploadAudio(createReadStream(), filename, mimetype);

      // Upload cover art if provided
      let coverArtUrl;
      if (coverArt) {
        const cover = await coverArt;
        coverArtUrl = await context.dataSources.storage.uploadImage(
          cover.createReadStream(),
          cover.filename,
          cover.mimetype
        );
      }

      // Create track
      const track = await context.dataSources.tracks.create({
        ...input,
        artistId: user.id,
        audioUrl,
        coverArt: coverArtUrl,
      });

      // Start audio processing
      context.pubsub.publish(EVENTS.TRACK_PROCESSING, {
        trackProcessingStatus: { trackId: track.id, status: 'PROCESSING', progress: 0 },
      });

      return track;
    },

    updateTrack: async (_: any, { id, input }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.tracks.update(id, user.id, input);
    },

    deleteTrack: async (_: any, { id }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.tracks.delete(id, user.id);
    },

    likeTrack: async (_: any, { trackId }: any, context: Context) => {
      const user = requireAuth(context);
      const track = await context.dataSources.tracks.like(trackId, user.id);
      context.pubsub.publish(EVENTS.NEW_LIKE, { newLike: track, trackId });
      return track;
    },

    unlikeTrack: async (_: any, { trackId }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.tracks.unlike(trackId, user.id);
    },

    // Playlist mutations
    createPlaylist: async (_: any, { input }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.playlists.create({ ...input, ownerId: user.id });
    },

    updatePlaylist: async (_: any, { id, input }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.playlists.update(id, user.id, input);
    },

    deletePlaylist: async (_: any, { id }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.playlists.delete(id, user.id);
    },

    addTracksToPlaylist: async (_: any, { playlistId, trackIds }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.playlists.addTracks(playlistId, user.id, trackIds);
    },

    removeTracksFromPlaylist: async (_: any, { playlistId, trackIds }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.playlists.removeTracks(playlistId, user.id, trackIds);
    },

    // Comments
    createComment: async (_: any, { trackId, content, timestamp, parentId }: any, context: Context) => {
      const user = requireAuth(context);
      const comment = await context.dataSources.tracks.createComment({
        trackId,
        content,
        timestamp,
        parentId,
        authorId: user.id,
      });
      context.pubsub.publish(EVENTS.NEW_COMMENT, { newComment: comment, trackId });
      return comment;
    },

    // Collaborations
    createCollaboration: async (_: any, { input }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.collaborations.create({ ...input, creatorId: user.id });
    },

    sendCollaborationMessage: async (_: any, { collaborationId, content, timestamp }: any, context: Context) => {
      const user = requireAuth(context);
      const message = await context.dataSources.collaborations.sendMessage({
        collaborationId,
        content,
        timestamp,
        authorId: user.id,
      });
      context.pubsub.publish(EVENTS.COLLABORATION_MESSAGE, {
        newCollaborationMessage: message,
        collaborationId,
      });
      return message;
    },

    // Playback
    recordPlay: async (_: any, { trackId, duration, quality, device }: any, context: Context) => {
      const user = requireAuth(context);
      await context.dataSources.analytics.recordPlay({
        trackId,
        userId: user.id,
        duration,
        quality,
        device,
      });
      return true;
    },

    updateNowPlaying: async (_: any, { trackId }: any, context: Context) => {
      const user = requireAuth(context);
      await context.dataSources.users.updateNowPlaying(user.id, trackId);
      const track = await context.dataSources.tracks.getById(trackId);
      context.pubsub.publish(EVENTS.NOW_PLAYING, { nowPlaying: track, userId: user.id });
      return true;
    },

    // Notifications
    markNotificationRead: async (_: any, { id }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.notifications.markRead(id, user.id);
    },

    markAllNotificationsRead: async (_: any, __: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.notifications.markAllRead(user.id);
    },

    // NFT
    mintTrackNFT: async (_: any, { trackId, price, royaltyPercentage }: any, context: Context) => {
      const user = requireAuth(context);
      return context.dataSources.nft.mintTrack(trackId, user.id, price, royaltyPercentage);
    },
  },

  // Subscriptions
  Subscription: {
    nowPlaying: {
      subscribe: withFilter(
        (_: any, __: any, { pubsub }: Context) => pubsub.asyncIterator([EVENTS.NOW_PLAYING]),
        (payload, variables) => payload.userId === variables.userId
      ),
    },

    collaborationUpdated: {
      subscribe: withFilter(
        (_: any, __: any, { pubsub }: Context) => pubsub.asyncIterator([EVENTS.COLLABORATION_UPDATED]),
        (payload, variables) => payload.collaborationId === variables.collaborationId
      ),
    },

    newCollaborationMessage: {
      subscribe: withFilter(
        (_: any, __: any, { pubsub }: Context) => pubsub.asyncIterator([EVENTS.COLLABORATION_MESSAGE]),
        (payload, variables) => payload.collaborationId === variables.collaborationId
      ),
    },

    newFollower: {
      subscribe: (_: any, __: any, context: Context) => {
        const user = requireAuth(context);
        return withFilter(
          () => context.pubsub.asyncIterator([EVENTS.NEW_FOLLOWER]),
          (payload) => payload.userId === user.id
        )();
      },
    },

    newLike: {
      subscribe: withFilter(
        (_: any, __: any, { pubsub }: Context) => pubsub.asyncIterator([EVENTS.NEW_LIKE]),
        (payload, variables) => !variables.trackId || payload.trackId === variables.trackId
      ),
    },

    newComment: {
      subscribe: withFilter(
        (_: any, __: any, { pubsub }: Context) => pubsub.asyncIterator([EVENTS.NEW_COMMENT]),
        (payload, variables) => !variables.trackId || payload.trackId === variables.trackId
      ),
    },

    notificationReceived: {
      subscribe: (_: any, __: any, context: Context) => {
        const user = requireAuth(context);
        return withFilter(
          () => context.pubsub.asyncIterator([EVENTS.NOTIFICATION]),
          (payload) => payload.userId === user.id
        )();
      },
    },

    trackProcessingStatus: {
      subscribe: withFilter(
        (_: any, __: any, { pubsub }: Context) => pubsub.asyncIterator([EVENTS.TRACK_PROCESSING]),
        (payload, variables) => payload.trackProcessingStatus.trackId === variables.trackId
      ),
    },
  },
};

export default resolvers;
