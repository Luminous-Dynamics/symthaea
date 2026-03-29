// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, endpoints } from '@/lib/api';
import { useAuth } from './use-auth';

interface Playlist {
  id: string;
  name: string;
  description?: string;
  coverUrl?: string;
  trackCount: number;
  isPublic: boolean;
  ownerId: string;
}

interface Track {
  id: string;
  title: string;
  duration: number;
  coverUrl?: string;
  artist: {
    id: string;
    name: string;
  };
}

interface Album {
  id: string;
  title: string;
  coverUrl?: string;
  artist: {
    id: string;
    name: string;
  };
  trackCount: number;
  releaseDate: string;
}

interface Artist {
  id: string;
  name: string;
  imageUrl?: string;
  followers: number;
}

export function useLibrary() {
  const { isAuthenticated } = useAuth();
  const queryClient = useQueryClient();

  // Fetch playlists
  const playlistsQuery = useQuery({
    queryKey: ['library', 'playlists'],
    queryFn: () => api.get<Playlist[]>(endpoints.library.playlists).then(res => res.data),
    enabled: isAuthenticated,
  });

  // Fetch liked songs
  const likedSongsQuery = useQuery({
    queryKey: ['library', 'liked'],
    queryFn: () => api.get<Track[]>(endpoints.library.liked).then(res => res.data),
    enabled: isAuthenticated,
  });

  // Fetch saved albums
  const albumsQuery = useQuery({
    queryKey: ['library', 'albums'],
    queryFn: () => api.get<Album[]>(endpoints.library.albums).then(res => res.data),
    enabled: isAuthenticated,
  });

  // Fetch followed artists
  const artistsQuery = useQuery({
    queryKey: ['library', 'artists'],
    queryFn: () => api.get<Artist[]>(endpoints.library.artists).then(res => res.data),
    enabled: isAuthenticated,
  });

  // Like/unlike track
  const likeTrackMutation = useMutation({
    mutationFn: ({ trackId, isLiked }: { trackId: string; isLiked: boolean }) =>
      isLiked
        ? api.delete(`${endpoints.library.liked}/${trackId}`)
        : api.post(`${endpoints.library.liked}/${trackId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['library', 'liked'] });
    },
  });

  // Save/unsave album
  const saveAlbumMutation = useMutation({
    mutationFn: ({ albumId, isSaved }: { albumId: string; isSaved: boolean }) =>
      isSaved
        ? api.delete(`${endpoints.library.albums}/${albumId}`)
        : api.post(`${endpoints.library.albums}/${albumId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['library', 'albums'] });
    },
  });

  // Follow/unfollow artist
  const followArtistMutation = useMutation({
    mutationFn: ({ artistId, isFollowing }: { artistId: string; isFollowing: boolean }) =>
      isFollowing
        ? api.delete(`${endpoints.library.artists}/${artistId}`)
        : api.post(`${endpoints.library.artists}/${artistId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['library', 'artists'] });
    },
  });

  // Create playlist
  const createPlaylistMutation = useMutation({
    mutationFn: (data: { name: string; description?: string; isPublic?: boolean }) =>
      api.post<Playlist>(endpoints.playlists.create, data).then(res => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['library', 'playlists'] });
    },
  });

  // Add track to playlist
  const addToPlaylistMutation = useMutation({
    mutationFn: ({ playlistId, trackId }: { playlistId: string; trackId: string }) =>
      api.post(`${endpoints.playlists.tracks(playlistId)}/${trackId}`),
    onSuccess: (_, { playlistId }) => {
      queryClient.invalidateQueries({ queryKey: ['playlist', playlistId] });
      queryClient.invalidateQueries({ queryKey: ['library', 'playlists'] });
    },
  });

  return {
    playlists: playlistsQuery.data,
    likedSongs: likedSongsQuery.data,
    albums: albumsQuery.data,
    artists: artistsQuery.data,
    isLoading:
      playlistsQuery.isLoading ||
      likedSongsQuery.isLoading ||
      albumsQuery.isLoading ||
      artistsQuery.isLoading,
    likeTrack: likeTrackMutation.mutate,
    saveAlbum: saveAlbumMutation.mutate,
    followArtist: followArtistMutation.mutate,
    createPlaylist: createPlaylistMutation.mutateAsync,
    addToPlaylist: addToPlaylistMutation.mutate,
  };
}

export function useIsLiked(trackId: string) {
  const { likedSongs } = useLibrary();
  return likedSongs?.some(track => track.id === trackId) ?? false;
}

export function useIsSaved(albumId: string) {
  const { albums } = useLibrary();
  return albums?.some(album => album.id === albumId) ?? false;
}

export function useIsFollowing(artistId: string) {
  const { artists } = useLibrary();
  return artists?.some(artist => artist.id === artistId) ?? false;
}
