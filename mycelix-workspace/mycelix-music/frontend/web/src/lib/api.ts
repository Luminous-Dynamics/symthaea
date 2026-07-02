// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3100';

export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

    // Handle 401 - attempt refresh
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refreshToken,
          });
          const { token } = response.data;
          localStorage.setItem('auth_token', token);

          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${token}`;
          }
          return api(originalRequest);
        }
      } catch {
        // Refresh failed, clear tokens
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// API helper functions
export const apiGet = <T>(url: string, config?: AxiosRequestConfig) =>
  api.get<T>(url, config).then((res) => res.data);

export const apiPost = <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
  api.post<T>(url, data, config).then((res) => res.data);

export const apiPut = <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
  api.put<T>(url, data, config).then((res) => res.data);

export const apiPatch = <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
  api.patch<T>(url, data, config).then((res) => res.data);

export const apiDelete = <T>(url: string, config?: AxiosRequestConfig) =>
  api.delete<T>(url, config).then((res) => res.data);

// Upload helper with progress
export const uploadFile = (
  url: string,
  file: File,
  onProgress?: (progress: number) => void
): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);

  return api.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  });
};

// Streaming fetch for large responses
export const streamFetch = async (
  url: string,
  onChunk: (chunk: string) => void
): Promise<void> => {
  const token = localStorage.getItem('auth_token');
  const response = await fetch(`${API_BASE_URL}${url}`, {
    headers: {
      Authorization: token ? `Bearer ${token}` : '',
    },
  });

  if (!response.body) throw new Error('No response body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk(decoder.decode(value, { stream: true }));
  }
};

// Error handling helper
export const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ message?: string; error?: string }>;
    return (
      axiosError.response?.data?.message ||
      axiosError.response?.data?.error ||
      axiosError.message ||
      'An error occurred'
    );
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

// Typed API endpoints
export const endpoints = {
  // Auth
  auth: {
    login: '/auth/login',
    signup: '/auth/signup',
    logout: '/auth/logout',
    refresh: '/auth/refresh',
    me: '/auth/me',
    walletNonce: '/auth/wallet/nonce',
    walletVerify: '/auth/wallet/verify',
  },

  // Users
  users: {
    me: '/users/me',
    profile: (id: string) => `/users/${id}`,
    followers: (id: string) => `/users/${id}/followers`,
    following: (id: string) => `/users/${id}/following`,
  },

  // Tracks
  tracks: {
    list: '/tracks',
    get: (id: string) => `/tracks/${id}`,
    search: '/tracks/search',
    trending: '/tracks/trending',
    stream: (id: string) => `/tracks/${id}/stream`,
  },

  // Artists
  artists: {
    list: '/artists',
    get: (id: string) => `/artists/${id}`,
    topTracks: (id: string) => `/artists/${id}/top-tracks`,
    albums: (id: string) => `/artists/${id}/albums`,
    trending: '/artists/trending',
  },

  // Albums
  albums: {
    list: '/albums',
    get: (id: string) => `/albums/${id}`,
    tracks: (id: string) => `/albums/${id}/tracks`,
    newReleases: '/albums/new-releases',
  },

  // Playlists
  playlists: {
    list: '/playlists',
    get: (id: string) => `/playlists/${id}`,
    tracks: (id: string) => `/playlists/${id}/tracks`,
    featured: '/playlists/featured',
    create: '/playlists',
  },

  // Library
  library: {
    tracks: '/library/tracks',
    albums: '/library/albums',
    artists: '/library/artists',
    playlists: '/library/playlists',
    history: '/library/history',
    liked: '/library/liked',
  },

  // Search
  search: {
    all: '/search',
    tracks: '/search/tracks',
    artists: '/search/artists',
    albums: '/search/albums',
    playlists: '/search/playlists',
  },

  // Live
  live: {
    streams: '/live/streams',
    stream: (id: string) => `/live/streams/${id}`,
    schedule: '/live/schedule',
  },

  // Social
  social: {
    feed: '/social/feed',
    activity: '/social/activity',
    friends: '/social/friends',
    notifications: '/social/notifications',
  },

  // Discover
  discover: {
    featured: '/discover/featured',
    personalized: '/discover/personalized',
    genres: '/discover/genres',
  },
} as const;
