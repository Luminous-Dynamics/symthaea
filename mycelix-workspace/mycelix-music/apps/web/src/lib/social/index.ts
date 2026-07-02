// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Social Platform
 *
 * Complete social features:
 * - User profiles with verification
 * - Activity feeds
 * - Comments and reactions
 * - Following/followers
 * - Collaborative playlists
 * - Direct messaging
 * - Notifications
 */

// ==================== Types ====================

export interface UserProfile {
  id: string;
  username: string;
  displayName: string;
  bio: string;
  avatar: string;
  banner?: string;
  verified: boolean;
  role: 'user' | 'artist' | 'label' | 'admin';
  stats: {
    followers: number;
    following: number;
    tracks: number;
    playlists: number;
    plays: number;
  };
  links: {
    website?: string;
    twitter?: string;
    instagram?: string;
    spotify?: string;
  };
  createdAt: Date;
  isFollowing?: boolean;
  isFollowedBy?: boolean;
}

export interface FeedItem {
  id: string;
  type: 'track_upload' | 'track_like' | 'playlist_create' | 'playlist_add' | 'follow' | 'repost' | 'comment' | 'milestone';
  actor: UserProfile;
  target?: UserProfile | Track | Playlist;
  content?: string;
  createdAt: Date;
  reactions: ReactionSummary;
  comments: number;
}

export interface Track {
  id: string;
  title: string;
  artist: UserProfile;
  coverArt: string;
  duration: number;
  plays: number;
  likes: number;
  reposts: number;
  waveform?: number[];
  isLiked?: boolean;
  isReposted?: boolean;
}

export interface Playlist {
  id: string;
  name: string;
  description: string;
  coverArt: string;
  owner: UserProfile;
  collaborators: UserProfile[];
  isPublic: boolean;
  isCollaborative: boolean;
  trackCount: number;
  duration: number;
  likes: number;
  isLiked?: boolean;
}

export interface Comment {
  id: string;
  author: UserProfile;
  content: string;
  timestamp?: number; // For timestamped comments on tracks
  createdAt: Date;
  likes: number;
  replies: Comment[];
  isLiked?: boolean;
}

export interface Reaction {
  type: 'like' | 'love' | 'fire' | 'wow' | 'sad';
  count: number;
  hasReacted: boolean;
}

export type ReactionSummary = Record<Reaction['type'], Reaction>;

export interface Notification {
  id: string;
  type: 'follow' | 'like' | 'comment' | 'mention' | 'repost' | 'collab_invite' | 'message' | 'milestone';
  actor: UserProfile;
  target?: Track | Playlist | Comment;
  message: string;
  read: boolean;
  createdAt: Date;
}

export interface DirectMessage {
  id: string;
  sender: UserProfile;
  recipient: UserProfile;
  content: string;
  attachments?: { type: 'track' | 'playlist' | 'image'; id: string }[];
  read: boolean;
  createdAt: Date;
}

export interface Conversation {
  id: string;
  participants: UserProfile[];
  lastMessage: DirectMessage;
  unreadCount: number;
  updatedAt: Date;
}

// ==================== Profile Service ====================

export class ProfileService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getProfile(userId: string): Promise<UserProfile> {
    const response = await fetch(`${this.baseUrl}/users/${userId}`);
    return response.json();
  }

  async getProfileByUsername(username: string): Promise<UserProfile> {
    const response = await fetch(`${this.baseUrl}/users/by-username/${username}`);
    return response.json();
  }

  async updateProfile(updates: Partial<UserProfile>): Promise<UserProfile> {
    const response = await fetch(`${this.baseUrl}/users/me`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return response.json();
  }

  async uploadAvatar(file: File): Promise<string> {
    const formData = new FormData();
    formData.append('avatar', file);
    const response = await fetch(`${this.baseUrl}/users/me/avatar`, {
      method: 'POST',
      body: formData,
    });
    const { url } = await response.json();
    return url;
  }

  async getFollowers(userId: string, cursor?: string): Promise<{ users: UserProfile[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/users/${userId}/followers?${params}`);
    return response.json();
  }

  async getFollowing(userId: string, cursor?: string): Promise<{ users: UserProfile[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/users/${userId}/following?${params}`);
    return response.json();
  }

  async follow(userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/users/${userId}/follow`, { method: 'POST' });
  }

  async unfollow(userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/users/${userId}/follow`, { method: 'DELETE' });
  }

  async searchUsers(query: string): Promise<UserProfile[]> {
    const response = await fetch(`${this.baseUrl}/users/search?q=${encodeURIComponent(query)}`);
    return response.json();
  }
}

// ==================== Feed Service ====================

export class FeedService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getHomeFeed(cursor?: string): Promise<{ items: FeedItem[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/feed?${params}`);
    return response.json();
  }

  async getUserFeed(userId: string, cursor?: string): Promise<{ items: FeedItem[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/users/${userId}/feed?${params}`);
    return response.json();
  }

  async getExploreFeed(category?: string): Promise<{ items: FeedItem[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (category) params.set('category', category);
    const response = await fetch(`${this.baseUrl}/explore?${params}`);
    return response.json();
  }

  async getTrendingTracks(timeRange: 'day' | 'week' | 'month' = 'week'): Promise<Track[]> {
    const response = await fetch(`${this.baseUrl}/trending/tracks?range=${timeRange}`);
    return response.json();
  }

  async getTrendingArtists(): Promise<UserProfile[]> {
    const response = await fetch(`${this.baseUrl}/trending/artists`);
    return response.json();
  }
}

// ==================== Interaction Service ====================

export class InteractionService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  // Likes
  async likeTrack(trackId: string): Promise<void> {
    await fetch(`${this.baseUrl}/tracks/${trackId}/like`, { method: 'POST' });
  }

  async unlikeTrack(trackId: string): Promise<void> {
    await fetch(`${this.baseUrl}/tracks/${trackId}/like`, { method: 'DELETE' });
  }

  async likePlaylist(playlistId: string): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/like`, { method: 'POST' });
  }

  async unlikePlaylist(playlistId: string): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/like`, { method: 'DELETE' });
  }

  async likeComment(commentId: string): Promise<void> {
    await fetch(`${this.baseUrl}/comments/${commentId}/like`, { method: 'POST' });
  }

  // Reposts
  async repostTrack(trackId: string): Promise<void> {
    await fetch(`${this.baseUrl}/tracks/${trackId}/repost`, { method: 'POST' });
  }

  async unrepostTrack(trackId: string): Promise<void> {
    await fetch(`${this.baseUrl}/tracks/${trackId}/repost`, { method: 'DELETE' });
  }

  // Reactions
  async addReaction(targetType: 'track' | 'comment', targetId: string, reaction: Reaction['type']): Promise<void> {
    await fetch(`${this.baseUrl}/${targetType}s/${targetId}/reactions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: reaction }),
    });
  }

  async removeReaction(targetType: 'track' | 'comment', targetId: string): Promise<void> {
    await fetch(`${this.baseUrl}/${targetType}s/${targetId}/reactions`, { method: 'DELETE' });
  }
}

// ==================== Comment Service ====================

export class CommentService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getComments(
    targetType: 'track' | 'playlist',
    targetId: string,
    cursor?: string
  ): Promise<{ comments: Comment[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/${targetType}s/${targetId}/comments?${params}`);
    return response.json();
  }

  async addComment(
    targetType: 'track' | 'playlist',
    targetId: string,
    content: string,
    timestamp?: number
  ): Promise<Comment> {
    const response = await fetch(`${this.baseUrl}/${targetType}s/${targetId}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, timestamp }),
    });
    return response.json();
  }

  async replyToComment(commentId: string, content: string): Promise<Comment> {
    const response = await fetch(`${this.baseUrl}/comments/${commentId}/replies`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
    return response.json();
  }

  async deleteComment(commentId: string): Promise<void> {
    await fetch(`${this.baseUrl}/comments/${commentId}`, { method: 'DELETE' });
  }

  async reportComment(commentId: string, reason: string): Promise<void> {
    await fetch(`${this.baseUrl}/comments/${commentId}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
  }
}

// ==================== Playlist Service ====================

export class PlaylistService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async createPlaylist(data: {
    name: string;
    description?: string;
    isPublic?: boolean;
    isCollaborative?: boolean;
  }): Promise<Playlist> {
    const response = await fetch(`${this.baseUrl}/playlists`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return response.json();
  }

  async getPlaylist(playlistId: string): Promise<Playlist> {
    const response = await fetch(`${this.baseUrl}/playlists/${playlistId}`);
    return response.json();
  }

  async updatePlaylist(playlistId: string, updates: Partial<Playlist>): Promise<Playlist> {
    const response = await fetch(`${this.baseUrl}/playlists/${playlistId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return response.json();
  }

  async deletePlaylist(playlistId: string): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}`, { method: 'DELETE' });
  }

  async addTrack(playlistId: string, trackId: string, position?: number): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/tracks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trackId, position }),
    });
  }

  async removeTrack(playlistId: string, trackId: string): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/tracks/${trackId}`, { method: 'DELETE' });
  }

  async reorderTracks(playlistId: string, trackIds: string[]): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/tracks/reorder`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trackIds }),
    });
  }

  async inviteCollaborator(playlistId: string, userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/collaborators`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId }),
    });
  }

  async removeCollaborator(playlistId: string, userId: string): Promise<void> {
    await fetch(`${this.baseUrl}/playlists/${playlistId}/collaborators/${userId}`, { method: 'DELETE' });
  }
}

// ==================== Notification Service ====================

export class NotificationService {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  public onNotification?: (notification: Notification) => void;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getNotifications(cursor?: string): Promise<{ notifications: Notification[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/notifications?${params}`);
    return response.json();
  }

  async markAsRead(notificationId: string): Promise<void> {
    await fetch(`${this.baseUrl}/notifications/${notificationId}/read`, { method: 'POST' });
  }

  async markAllAsRead(): Promise<void> {
    await fetch(`${this.baseUrl}/notifications/read-all`, { method: 'POST' });
  }

  async getUnreadCount(): Promise<number> {
    const response = await fetch(`${this.baseUrl}/notifications/unread-count`);
    const { count } = await response.json();
    return count;
  }

  connectRealtime(wsUrl: string): void {
    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (event) => {
      const notification = JSON.parse(event.data) as Notification;
      this.onNotification?.(notification);
    };

    this.ws.onclose = () => {
      // Reconnect after delay
      setTimeout(() => this.connectRealtime(wsUrl), 5000);
    };
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }
}

// ==================== Messaging Service ====================

export class MessagingService {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  public onMessage?: (message: DirectMessage) => void;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getConversations(): Promise<Conversation[]> {
    const response = await fetch(`${this.baseUrl}/messages/conversations`);
    return response.json();
  }

  async getMessages(conversationId: string, cursor?: string): Promise<{ messages: DirectMessage[]; nextCursor?: string }> {
    const params = new URLSearchParams();
    if (cursor) params.set('cursor', cursor);
    const response = await fetch(`${this.baseUrl}/messages/conversations/${conversationId}?${params}`);
    return response.json();
  }

  async sendMessage(recipientId: string, content: string, attachments?: { type: string; id: string }[]): Promise<DirectMessage> {
    const response = await fetch(`${this.baseUrl}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipientId, content, attachments }),
    });
    return response.json();
  }

  async markConversationAsRead(conversationId: string): Promise<void> {
    await fetch(`${this.baseUrl}/messages/conversations/${conversationId}/read`, { method: 'POST' });
  }

  async deleteConversation(conversationId: string): Promise<void> {
    await fetch(`${this.baseUrl}/messages/conversations/${conversationId}`, { method: 'DELETE' });
  }

  connectRealtime(wsUrl: string): void {
    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data) as DirectMessage;
      this.onMessage?.(message);
    };
  }

  disconnect(): void {
    this.ws?.close();
  }
}

// ==================== Social Manager ====================

export class SocialManager {
  public readonly profiles: ProfileService;
  public readonly feed: FeedService;
  public readonly interactions: InteractionService;
  public readonly comments: CommentService;
  public readonly playlists: PlaylistService;
  public readonly notifications: NotificationService;
  public readonly messaging: MessagingService;

  constructor(baseUrl = '/api') {
    this.profiles = new ProfileService(baseUrl);
    this.feed = new FeedService(baseUrl);
    this.interactions = new InteractionService(baseUrl);
    this.comments = new CommentService(baseUrl);
    this.playlists = new PlaylistService(baseUrl);
    this.notifications = new NotificationService(baseUrl);
    this.messaging = new MessagingService(baseUrl);
  }

  dispose(): void {
    this.notifications.disconnect();
    this.messaging.disconnect();
  }
}

// ==================== Singleton ====================

let socialManager: SocialManager | null = null;

export function getSocialManager(baseUrl?: string): SocialManager {
  if (!socialManager) {
    socialManager = new SocialManager(baseUrl);
  }
  return socialManager;
}

export default {
  SocialManager,
  getSocialManager,
  ProfileService,
  FeedService,
  InteractionService,
  CommentService,
  PlaylistService,
  NotificationService,
  MessagingService,
};
