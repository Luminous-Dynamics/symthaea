// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CDN Helper Functions for Mycelix Music
 * Handles audio delivery from both local demo files and Bunny CDN
 */

import { Song } from '../../data/mockSongs';

// Bunny CDN configuration
const BUNNY_CDN_HOSTNAME = process.env.NEXT_PUBLIC_BUNNY_CDN_HOSTNAME || 'mycelix-music-cdn.b-cdn.net';
const USE_CDN = process.env.NEXT_PUBLIC_USE_CDN === 'true';

/**
 * Get CDN URL for any asset
 * @param path - Relative path to the asset
 * @returns Full CDN URL
 */
export function getCDNUrl(path: string): string {
  // Remove leading slash if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return `https://${BUNNY_CDN_HOSTNAME}/${cleanPath}`;
}

/**
 * Get audio URL for a song - handles local demo files and CDN
 * @param song - Song object
 * @returns Audio URL (local or CDN)
 */
export function getAudioUrl(song: Song): string {
  // Development: Use local demo files
  if (!USE_CDN && song.audioUrl && song.audioUrl.startsWith('/demo-music/')) {
    return song.audioUrl;
  }

  // Production: Use CDN if configured
  if (USE_CDN && song.audioUrl) {
    // If audioUrl starts with /, it's a local path - convert to CDN
    if (song.audioUrl.startsWith('/')) {
      return getCDNUrl(song.audioUrl);
    }
    // Otherwise it's already a full URL
    return song.audioUrl;
  }

  // Fallback: Try local first, then construct CDN path from IPFS hash
  if (song.audioUrl) {
    return song.audioUrl;
  }

  // Last resort: construct from artist and song ID
  const artistSlug = song.artist.toLowerCase().replace(/\s+/g, '-');
  const cdnPath = `music/${artistSlug}/${song.id}.mp3`;

  return USE_CDN ? getCDNUrl(cdnPath) : `/demo-music/${song.id}.mp3`;
}

/**
 * Get cover art URL with CDN fallback
 * @param song - Song object
 * @returns Cover art URL
 */
export function getCoverArtUrl(song: Song): string {
  // If it's already an external URL (Unsplash, etc), return as-is
  if (song.coverArt.startsWith('http://') || song.coverArt.startsWith('https://')) {
    return song.coverArt;
  }

  // Otherwise, treat as CDN path
  return USE_CDN ? getCDNUrl(song.coverArt) : song.coverArt;
}

/**
 * Check if a file exists at the given URL
 * @param url - URL to check
 * @returns Promise<boolean>
 */
export async function fileExists(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get audio URL with fallback
 * Tries local demo first, then CDN, then returns undefined
 * @param song - Song object
 * @returns Promise<string | undefined>
 */
export async function getAudioUrlWithFallback(song: Song): Promise<string | undefined> {
  // Try local demo file first
  if (song.audioUrl && song.audioUrl.startsWith('/demo-music/')) {
    if (await fileExists(song.audioUrl)) {
      return song.audioUrl;
    }
  }

  // Try CDN
  const cdnUrl = getAudioUrl(song);
  if (await fileExists(cdnUrl)) {
    return cdnUrl;
  }

  // No audio available
  console.warn(`No audio file found for song: ${song.title}`);
  return undefined;
}

/**
 * Upload a file to Bunny CDN Storage
 * Server-side only (requires API key)
 * @param file - File to upload
 * @param path - Destination path in storage zone
 * @returns Promise<boolean>
 */
export async function uploadToBunnyCDN(
  file: File | Uint8Array,
  path: string
): Promise<boolean> {
  const storageApiKey = process.env.BUNNY_STORAGE_API_KEY;
  const storageZone = process.env.BUNNY_CDN_STORAGE_ZONE || 'mycelix-music';

  if (!storageApiKey) {
    throw new Error('BUNNY_STORAGE_API_KEY not configured');
  }

  try {
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    const url = `https://storage.bunnycdn.com/${storageZone}/${cleanPath}`;

    // Convert to Blob for fetch body (File extends Blob, binary data needs wrapping)
    let body: Blob;
    if (file instanceof Blob) {
      body = file;
    } else {
      // For Uint8Array, create a new Uint8Array copy to ensure ArrayBuffer type
      body = new Blob([new Uint8Array(file)], { type: 'application/octet-stream' });
    }

    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'AccessKey': storageApiKey,
        'Content-Type': 'application/octet-stream',
      },
      body,
    });

    return response.ok;
  } catch (error) {
    console.error('Failed to upload to Bunny CDN:', error);
    return false;
  }
}

/**
 * Delete a file from Bunny CDN Storage
 * Server-side only (requires API key)
 * @param path - Path to file in storage zone
 * @returns Promise<boolean>
 */
export async function deleteFromBunnyCDN(path: string): Promise<boolean> {
  const storageApiKey = process.env.BUNNY_STORAGE_API_KEY;
  const storageZone = process.env.BUNNY_CDN_STORAGE_ZONE || 'mycelix-music';

  if (!storageApiKey) {
    throw new Error('BUNNY_STORAGE_API_KEY not configured');
  }

  try {
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    const url = `https://storage.bunnycdn.com/${storageZone}/${cleanPath}`;

    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'AccessKey': storageApiKey,
      },
    });

    return response.ok;
  } catch (error) {
    console.error('Failed to delete from Bunny CDN:', error);
    return false;
  }
}

/**
 * Get CDN statistics
 * Requires Bunny CDN API key
 * @returns Promise<any>
 */
export async function getCDNStats(): Promise<any> {
  const apiKey = process.env.BUNNY_CDN_API_KEY;

  if (!apiKey) {
    throw new Error('BUNNY_CDN_API_KEY not configured');
  }

  try {
    const response = await fetch('https://api.bunny.net/pullzone', {
      headers: {
        'AccessKey': apiKey,
      },
    });

    return await response.json();
  } catch (error) {
    console.error('Failed to get CDN stats:', error);
    return null;
  }
}
