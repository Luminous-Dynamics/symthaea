// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { format, formatDistanceToNow, isToday, isYesterday, isThisYear } from 'date-fns';

/**
 * Format seconds into mm:ss or hh:mm:ss
 */
export function formatTime(seconds: number): string {
  if (!seconds || isNaN(seconds)) return '0:00';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format duration in milliseconds to human readable string
 */
export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

/**
 * Format a number with proper suffix (K, M, B)
 */
export function formatNumber(num: number): string {
  if (num >= 1_000_000_000) {
    return (num / 1_000_000_000).toFixed(1).replace(/\.0$/, '') + 'B';
  }
  if (num >= 1_000_000) {
    return (num / 1_000_000).toFixed(1).replace(/\.0$/, '') + 'M';
  }
  if (num >= 1_000) {
    return (num / 1_000).toFixed(1).replace(/\.0$/, '') + 'K';
  }
  return num.toString();
}

/**
 * Format number with commas
 */
export function formatNumberWithCommas(num: number): string {
  return num.toLocaleString();
}

/**
 * Format currency
 */
export function formatCurrency(
  amount: number,
  currency = 'USD',
  locale = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(amount);
}

/**
 * Format date in a human readable way
 */
export function formatDate(date: Date | string): string {
  const d = new Date(date);

  if (isToday(d)) {
    return 'Today';
  }
  if (isYesterday(d)) {
    return 'Yesterday';
  }
  if (isThisYear(d)) {
    return format(d, 'MMM d');
  }
  return format(d, 'MMM d, yyyy');
}

/**
 * Format date with time
 */
export function formatDateTime(date: Date | string): string {
  const d = new Date(date);
  return format(d, 'MMM d, yyyy h:mm a');
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: Date | string): string {
  return formatDistanceToNow(new Date(date), { addSuffix: true });
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format bitrate
 */
export function formatBitrate(bitsPerSecond: number): string {
  if (bitsPerSecond >= 1_000_000) {
    return `${(bitsPerSecond / 1_000_000).toFixed(1)} Mbps`;
  }
  if (bitsPerSecond >= 1_000) {
    return `${(bitsPerSecond / 1_000).toFixed(0)} kbps`;
  }
  return `${bitsPerSecond} bps`;
}

/**
 * Format percentage
 */
export function formatPercentage(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format track count
 */
export function formatTrackCount(count: number): string {
  if (count === 1) return '1 track';
  return `${formatNumber(count)} tracks`;
}

/**
 * Format follower count
 */
export function formatFollowerCount(count: number): string {
  if (count === 1) return '1 follower';
  return `${formatNumber(count)} followers`;
}

/**
 * Format listener count
 */
export function formatListenerCount(count: number): string {
  if (count === 1) return '1 listener';
  return `${formatNumber(count)} listeners`;
}

/**
 * Format play count
 */
export function formatPlayCount(count: number): string {
  if (count === 1) return '1 play';
  return `${formatNumber(count)} plays`;
}

/**
 * Format album type with track info
 */
export function formatAlbumInfo(type: string, trackCount: number, totalDuration: number): string {
  const typeLabel = type === 'single' ? 'Single' : type === 'ep' ? 'EP' : 'Album';
  const tracks = formatTrackCount(trackCount);
  const duration = formatDuration(totalDuration);
  return `${typeLabel} · ${tracks} · ${duration}`;
}

/**
 * Format audio quality
 */
export function formatAudioQuality(quality: string): string {
  const qualities: Record<string, string> = {
    low: 'Normal (128 kbps)',
    normal: 'High (256 kbps)',
    high: 'Very High (320 kbps)',
    lossless: 'Lossless (FLAC)',
    hires: 'Hi-Res (24-bit)',
  };
  return qualities[quality] || quality;
}

/**
 * Format genre list
 */
export function formatGenres(genres: string[], limit = 3): string {
  if (genres.length === 0) return '';
  if (genres.length <= limit) return genres.join(', ');
  return `${genres.slice(0, limit).join(', ')} +${genres.length - limit}`;
}

/**
 * Format BPM
 */
export function formatBPM(bpm: number): string {
  return `${Math.round(bpm)} BPM`;
}

/**
 * Format musical key
 */
export function formatKey(key: string): string {
  // Convert Camelot notation or pitch class
  const keyMap: Record<string, string> = {
    '1A': 'Ab minor',
    '1B': 'B major',
    '2A': 'Eb minor',
    '2B': 'Gb major',
    '3A': 'Bb minor',
    '3B': 'Db major',
    '4A': 'F minor',
    '4B': 'Ab major',
    '5A': 'C minor',
    '5B': 'Eb major',
    '6A': 'G minor',
    '6B': 'Bb major',
    '7A': 'D minor',
    '7B': 'F major',
    '8A': 'A minor',
    '8B': 'C major',
    '9A': 'E minor',
    '9B': 'G major',
    '10A': 'B minor',
    '10B': 'D major',
    '11A': 'Gb minor',
    '11B': 'A major',
    '12A': 'Db minor',
    '12B': 'E major',
  };
  return keyMap[key] || key;
}
