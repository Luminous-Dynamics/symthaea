// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { gql } from 'graphql-tag';

export const typeDefs = gql`
  # Scalars
  scalar DateTime
  scalar JSON
  scalar Upload

  # Enums
  enum UserRole {
    USER
    ARTIST
    PRODUCER
    LABEL
    ADMIN
  }

  enum TrackStatus {
    DRAFT
    PROCESSING
    PUBLISHED
    ARCHIVED
  }

  enum PlaylistVisibility {
    PUBLIC
    PRIVATE
    UNLISTED
  }

  enum SubscriptionTier {
    FREE
    PREMIUM
    ARTIST_PRO
    LABEL_ENTERPRISE
  }

  enum AudioQuality {
    LOW_128
    STANDARD_256
    HIGH_320
    LOSSLESS_FLAC
    HI_RES_24BIT
  }

  enum NotificationType {
    FOLLOW
    LIKE
    COMMENT
    RELEASE
    COLLABORATION
    ROYALTY
    SYSTEM
  }

  # Types
  type User {
    id: ID!
    username: String!
    email: String!
    displayName: String
    avatar: String
    bio: String
    role: UserRole!
    isVerified: Boolean!
    subscription: SubscriptionTier!
    followers: [User!]!
    following: [User!]!
    followerCount: Int!
    followingCount: Int!
    tracks: [Track!]!
    albums: [Album!]!
    playlists: [Playlist!]!
    likedTracks: [Track!]!
    collaborations: [Collaboration!]!
    royalties: RoyaltySummary
    socialLinks: SocialLinks
    settings: UserSettings!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type SocialLinks {
    website: String
    twitter: String
    instagram: String
    spotify: String
    soundcloud: String
    bandcamp: String
  }

  type UserSettings {
    emailNotifications: Boolean!
    pushNotifications: Boolean!
    privateProfile: Boolean!
    showListeningActivity: Boolean!
    audioQuality: AudioQuality!
    autoplay: Boolean!
    crossfade: Int!
    normalizeVolume: Boolean!
  }

  type Track {
    id: ID!
    title: String!
    slug: String!
    artist: User!
    featuredArtists: [User!]!
    album: Album
    duration: Int!
    bpm: Float
    key: String
    genre: Genre!
    subGenres: [Genre!]!
    tags: [String!]!
    description: String
    lyrics: String
    coverArt: String!
    waveformData: JSON
    audioFiles: [AudioFile!]!
    stems: [Stem!]
    status: TrackStatus!
    isExplicit: Boolean!
    isStreamable: Boolean!
    isDownloadable: Boolean!
    playCount: Int!
    likeCount: Int!
    commentCount: Int!
    repostCount: Int!
    comments: [Comment!]!
    credits: [Credit!]!
    license: License
    price: Float
    nftToken: NFTToken
    releaseDate: DateTime
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type AudioFile {
    id: ID!
    quality: AudioQuality!
    format: String!
    bitrate: Int!
    sampleRate: Int!
    bitDepth: Int
    size: Int!
    url: String!
  }

  type Stem {
    id: ID!
    name: String!
    type: String!
    audioFile: AudioFile!
  }

  type Album {
    id: ID!
    title: String!
    slug: String!
    artist: User!
    tracks: [Track!]!
    trackCount: Int!
    totalDuration: Int!
    coverArt: String!
    releaseDate: DateTime!
    genre: Genre!
    description: String
    type: AlbumType!
    label: Label
    upc: String
    credits: [Credit!]!
    playCount: Int!
    likeCount: Int!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  enum AlbumType {
    ALBUM
    EP
    SINGLE
    COMPILATION
    MIXTAPE
  }

  type Playlist {
    id: ID!
    name: String!
    slug: String!
    description: String
    coverArt: String
    owner: User!
    collaborators: [User!]!
    tracks: [PlaylistTrack!]!
    trackCount: Int!
    totalDuration: Int!
    visibility: PlaylistVisibility!
    followerCount: Int!
    playCount: Int!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type PlaylistTrack {
    track: Track!
    addedAt: DateTime!
    addedBy: User!
    position: Int!
  }

  type Genre {
    id: ID!
    name: String!
    slug: String!
    description: String
    coverImage: String
    trackCount: Int!
    parentGenre: Genre
    subGenres: [Genre!]!
  }

  type Label {
    id: ID!
    name: String!
    slug: String!
    logo: String
    description: String
    website: String
    artists: [User!]!
    releases: [Album!]!
    verified: Boolean!
  }

  type Comment {
    id: ID!
    content: String!
    author: User!
    track: Track
    timestamp: Int
    parentComment: Comment
    replies: [Comment!]!
    likeCount: Int!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type Credit {
    id: ID!
    user: User
    name: String!
    role: String!
  }

  type License {
    id: ID!
    type: String!
    name: String!
    description: String!
    permissions: [String!]!
    restrictions: [String!]!
    price: Float
  }

  type NFTToken {
    id: ID!
    contractAddress: String!
    tokenId: String!
    blockchain: String!
    owner: User!
    price: Float
    currency: String!
    royaltyPercentage: Float!
    metadata: JSON
  }

  type Collaboration {
    id: ID!
    project: CollaborationProject!
    participants: [CollaborationParticipant!]!
    status: CollaborationStatus!
    messages: [CollaborationMessage!]!
    files: [CollaborationFile!]!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type CollaborationProject {
    id: ID!
    name: String!
    description: String
    bpm: Float
    key: String
    genre: Genre
  }

  type CollaborationParticipant {
    user: User!
    role: String!
    splitPercentage: Float!
    joinedAt: DateTime!
  }

  enum CollaborationStatus {
    ACTIVE
    COMPLETED
    ARCHIVED
  }

  type CollaborationMessage {
    id: ID!
    author: User!
    content: String!
    timestamp: Int
    createdAt: DateTime!
  }

  type CollaborationFile {
    id: ID!
    name: String!
    type: String!
    size: Int!
    url: String!
    uploadedBy: User!
    version: Int!
    createdAt: DateTime!
  }

  type RoyaltySummary {
    totalEarnings: Float!
    pendingPayout: Float!
    lastPayout: Float
    lastPayoutDate: DateTime
    monthlyEarnings: [MonthlyEarning!]!
    trackRoyalties: [TrackRoyalty!]!
  }

  type MonthlyEarning {
    month: String!
    streams: Int!
    earnings: Float!
  }

  type TrackRoyalty {
    track: Track!
    streams: Int!
    earnings: Float!
  }

  type Notification {
    id: ID!
    type: NotificationType!
    title: String!
    message: String!
    data: JSON
    read: Boolean!
    createdAt: DateTime!
  }

  type SearchResults {
    tracks: [Track!]!
    artists: [User!]!
    albums: [Album!]!
    playlists: [Playlist!]!
    genres: [Genre!]!
  }

  type Feed {
    items: [FeedItem!]!
    cursor: String
    hasMore: Boolean!
  }

  union FeedItem = Track | Album | Playlist | ActivityItem

  type ActivityItem {
    id: ID!
    type: String!
    user: User!
    target: FeedItemTarget
    createdAt: DateTime!
  }

  union FeedItemTarget = Track | Album | Playlist | User

  type PlaybackSession {
    id: ID!
    track: Track!
    user: User!
    startedAt: DateTime!
    duration: Int!
    quality: AudioQuality!
    device: String
    location: String
  }

  type Analytics {
    plays: Int!
    uniqueListeners: Int!
    likes: Int!
    reposts: Int!
    comments: Int!
    followers: Int!
    topTracks: [Track!]!
    topCountries: [CountryStat!]!
    demographics: Demographics!
    timeSeriesData: [TimeSeriesPoint!]!
  }

  type CountryStat {
    country: String!
    plays: Int!
    percentage: Float!
  }

  type Demographics {
    ageGroups: [AgeGroupStat!]!
    genderDistribution: [GenderStat!]!
  }

  type AgeGroupStat {
    group: String!
    percentage: Float!
  }

  type GenderStat {
    gender: String!
    percentage: Float!
  }

  type TimeSeriesPoint {
    date: DateTime!
    value: Int!
  }

  # Inputs
  input CreateTrackInput {
    title: String!
    genreId: ID!
    subGenreIds: [ID!]
    tags: [String!]
    description: String
    lyrics: String
    bpm: Float
    key: String
    isExplicit: Boolean
    releaseDate: DateTime
    featuredArtistIds: [ID!]
    credits: [CreditInput!]
    licenseId: ID
    price: Float
  }

  input UpdateTrackInput {
    title: String
    genreId: ID
    subGenreIds: [ID!]
    tags: [String!]
    description: String
    lyrics: String
    bpm: Float
    key: String
    isExplicit: Boolean
    releaseDate: DateTime
    status: TrackStatus
  }

  input CreateAlbumInput {
    title: String!
    genreId: ID!
    description: String
    type: AlbumType!
    releaseDate: DateTime!
    trackIds: [ID!]!
    labelId: ID
    upc: String
    credits: [CreditInput!]
  }

  input CreatePlaylistInput {
    name: String!
    description: String
    visibility: PlaylistVisibility
    trackIds: [ID!]
  }

  input UpdatePlaylistInput {
    name: String
    description: String
    visibility: PlaylistVisibility
  }

  input CreditInput {
    userId: ID
    name: String!
    role: String!
  }

  input CollaborationInput {
    name: String!
    description: String
    participantIds: [ID!]!
    bpm: Float
    key: String
    genreId: ID
  }

  input UpdateUserInput {
    displayName: String
    bio: String
    socialLinks: SocialLinksInput
  }

  input SocialLinksInput {
    website: String
    twitter: String
    instagram: String
    spotify: String
    soundcloud: String
    bandcamp: String
  }

  input UpdateSettingsInput {
    emailNotifications: Boolean
    pushNotifications: Boolean
    privateProfile: Boolean
    showListeningActivity: Boolean
    audioQuality: AudioQuality
    autoplay: Boolean
    crossfade: Int
    normalizeVolume: Boolean
  }

  input SearchInput {
    query: String!
    types: [String!]
    genreIds: [ID!]
    minBpm: Float
    maxBpm: Float
    key: String
    minDuration: Int
    maxDuration: Int
    releasedAfter: DateTime
    releasedBefore: DateTime
    sortBy: String
    sortOrder: String
  }

  input PaginationInput {
    limit: Int
    cursor: String
    offset: Int
  }

  input DateRangeInput {
    start: DateTime!
    end: DateTime!
  }

  # Queries
  type Query {
    # User
    me: User
    user(id: ID, username: String): User
    users(search: String, role: UserRole, pagination: PaginationInput): [User!]!

    # Tracks
    track(id: ID, slug: String): Track
    tracks(
      artistId: ID
      genreId: ID
      status: TrackStatus
      pagination: PaginationInput
    ): [Track!]!
    trendingTracks(genreId: ID, timeRange: String, limit: Int): [Track!]!
    newReleases(genreId: ID, limit: Int): [Track!]!
    recommendedTracks(limit: Int): [Track!]!

    # Albums
    album(id: ID, slug: String): Album
    albums(artistId: ID, labelId: ID, pagination: PaginationInput): [Album!]!

    # Playlists
    playlist(id: ID, slug: String): Playlist
    playlists(ownerId: ID, pagination: PaginationInput): [Playlist!]!
    featuredPlaylists(limit: Int): [Playlist!]!
    curatedPlaylists(genreId: ID, mood: String, limit: Int): [Playlist!]!

    # Genres
    genre(id: ID, slug: String): Genre
    genres(parentId: ID): [Genre!]!

    # Labels
    label(id: ID, slug: String): Label
    labels(pagination: PaginationInput): [Label!]!

    # Search
    search(input: SearchInput!, pagination: PaginationInput): SearchResults!
    autocomplete(query: String!, limit: Int): [String!]!

    # Feed
    feed(pagination: PaginationInput): Feed!
    activityFeed(userId: ID!, pagination: PaginationInput): Feed!

    # Collaborations
    collaboration(id: ID!): Collaboration
    myCollaborations(status: CollaborationStatus): [Collaboration!]!

    # Analytics
    trackAnalytics(trackId: ID!, dateRange: DateRangeInput): Analytics!
    artistAnalytics(dateRange: DateRangeInput): Analytics!

    # Notifications
    notifications(unreadOnly: Boolean, pagination: PaginationInput): [Notification!]!
    unreadNotificationCount: Int!

    # Royalties
    royalties(dateRange: DateRangeInput): RoyaltySummary!

    # Streaming
    streamUrl(trackId: ID!, quality: AudioQuality): String!
    downloadUrl(trackId: ID!, quality: AudioQuality): String!
  }

  # Mutations
  type Mutation {
    # Auth
    signUp(email: String!, password: String!, username: String!): AuthPayload!
    signIn(email: String!, password: String!): AuthPayload!
    signOut: Boolean!
    refreshToken(refreshToken: String!): AuthPayload!
    forgotPassword(email: String!): Boolean!
    resetPassword(token: String!, password: String!): Boolean!
    verifyEmail(token: String!): Boolean!

    # User
    updateUser(input: UpdateUserInput!): User!
    updateSettings(input: UpdateSettingsInput!): UserSettings!
    updateAvatar(file: Upload!): User!
    deleteAccount: Boolean!

    # Social
    followUser(userId: ID!): User!
    unfollowUser(userId: ID!): User!

    # Tracks
    createTrack(input: CreateTrackInput!, audioFile: Upload!, coverArt: Upload): Track!
    updateTrack(id: ID!, input: UpdateTrackInput!): Track!
    deleteTrack(id: ID!): Boolean!
    uploadStem(trackId: ID!, file: Upload!, name: String!, type: String!): Stem!
    likeTrack(trackId: ID!): Track!
    unlikeTrack(trackId: ID!): Track!
    repostTrack(trackId: ID!): Track!
    unrepostTrack(trackId: ID!): Track!

    # Albums
    createAlbum(input: CreateAlbumInput!, coverArt: Upload!): Album!
    updateAlbum(id: ID!, input: CreateAlbumInput!): Album!
    deleteAlbum(id: ID!): Boolean!

    # Playlists
    createPlaylist(input: CreatePlaylistInput!): Playlist!
    updatePlaylist(id: ID!, input: UpdatePlaylistInput!): Playlist!
    deletePlaylist(id: ID!): Boolean!
    addTracksToPlaylist(playlistId: ID!, trackIds: [ID!]!): Playlist!
    removeTracksFromPlaylist(playlistId: ID!, trackIds: [ID!]!): Playlist!
    reorderPlaylistTracks(playlistId: ID!, trackIds: [ID!]!): Playlist!
    followPlaylist(playlistId: ID!): Playlist!
    unfollowPlaylist(playlistId: ID!): Playlist!

    # Comments
    createComment(trackId: ID!, content: String!, timestamp: Int, parentId: ID): Comment!
    updateComment(id: ID!, content: String!): Comment!
    deleteComment(id: ID!): Boolean!
    likeComment(commentId: ID!): Comment!
    unlikeComment(commentId: ID!): Comment!

    # Collaborations
    createCollaboration(input: CollaborationInput!): Collaboration!
    inviteToCollaboration(collaborationId: ID!, userId: ID!, role: String!): Collaboration!
    leaveCollaboration(collaborationId: ID!): Boolean!
    sendCollaborationMessage(collaborationId: ID!, content: String!, timestamp: Int): CollaborationMessage!
    uploadCollaborationFile(collaborationId: ID!, file: Upload!, name: String!): CollaborationFile!

    # Playback
    recordPlay(trackId: ID!, duration: Int!, quality: AudioQuality!, device: String): Boolean!
    updateNowPlaying(trackId: ID!): Boolean!
    scrobble(trackId: ID!): Boolean!

    # Notifications
    markNotificationRead(id: ID!): Notification!
    markAllNotificationsRead: Boolean!
    deleteNotification(id: ID!): Boolean!

    # NFT
    mintTrackNFT(trackId: ID!, price: Float!, royaltyPercentage: Float!): NFTToken!
    purchaseNFT(tokenId: ID!): NFTToken!

    # Subscription
    upgradeSubscription(tier: SubscriptionTier!, paymentMethodId: String!): User!
    cancelSubscription: User!

    # Royalties
    requestPayout(amount: Float!): Boolean!
    updatePayoutMethod(type: String!, details: JSON!): Boolean!
  }

  # Subscriptions (Real-time)
  type Subscription {
    # Playback sync
    nowPlaying(userId: ID!): Track

    # Collaboration
    collaborationUpdated(collaborationId: ID!): Collaboration!
    newCollaborationMessage(collaborationId: ID!): CollaborationMessage!
    collaborationFileUploaded(collaborationId: ID!): CollaborationFile!

    # Social
    newFollower: User!
    newLike(trackId: ID): Track!
    newComment(trackId: ID): Comment!

    # Notifications
    notificationReceived: Notification!

    # Live streaming
    liveStreamStarted(artistId: ID!): LiveStream!
    liveStreamEnded(artistId: ID!): LiveStream!
    liveStreamViewerCount(streamId: ID!): Int!

    # Processing
    trackProcessingStatus(trackId: ID!): ProcessingStatus!
  }

  type AuthPayload {
    user: User!
    accessToken: String!
    refreshToken: String!
    expiresAt: DateTime!
  }

  type LiveStream {
    id: ID!
    artist: User!
    title: String!
    description: String
    viewerCount: Int!
    startedAt: DateTime!
    streamUrl: String!
    chatEnabled: Boolean!
  }

  type ProcessingStatus {
    trackId: ID!
    status: String!
    progress: Int!
    message: String
    completedAt: DateTime
  }
`;

export default typeDefs;
