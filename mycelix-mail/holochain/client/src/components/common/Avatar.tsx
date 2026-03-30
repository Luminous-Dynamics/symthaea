// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Avatar Component
 *
 * Displays user avatar with fallback to initials.
 */

import React, { useMemo, useState } from 'react';

export interface AvatarProps {
  /** User's name for generating initials */
  name?: string;
  /** Email address for gravatar */
  email?: string;
  /** Direct image URL */
  src?: string;
  /** Size in pixels */
  size?: number;
  /** Whether user is online */
  online?: boolean;
  /** Custom className */
  className?: string;
}

const COLORS = [
  '#f87171', '#fb923c', '#fbbf24', '#a3e635', '#4ade80',
  '#34d399', '#22d3d8', '#38bdf8', '#60a5fa', '#818cf8',
  '#a78bfa', '#c084fc', '#e879f9', '#f472b6', '#fb7185',
];

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
}

function getColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return COLORS[Math.abs(hash) % COLORS.length];
}

function getGravatarUrl(email: string, size: number): string {
  // Simple hash - in production use MD5
  const hash = email.toLowerCase().trim();
  return `https://www.gravatar.com/avatar/${hash}?s=${size}&d=404`;
}

export const Avatar: React.FC<AvatarProps> = ({
  name = '',
  email,
  src,
  size = 40,
  online,
  className = '',
}) => {
  const [imgError, setImgError] = useState(false);

  const initials = useMemo(() => getInitials(name || email || '?'), [name, email]);
  const bgColor = useMemo(() => getColor(email || name || ''), [email, name]);

  const imageUrl = useMemo(() => {
    if (src) return src;
    if (email && !imgError) return getGravatarUrl(email, size * 2);
    return null;
  }, [src, email, size, imgError]);

  return (
    <div
      className={`relative inline-flex items-center justify-center rounded-full font-medium ${className}`}
      style={{
        width: size,
        height: size,
        fontSize: size * 0.4,
        backgroundColor: bgColor,
        color: '#fff',
      }}
    >
      {imageUrl && !imgError ? (
        <img
          src={imageUrl}
          alt={name || email || 'Avatar'}
          className="w-full h-full rounded-full object-cover"
          onError={() => setImgError(true)}
        />
      ) : (
        <span>{initials}</span>
      )}

      {online !== undefined && (
        <span
          className={`absolute bottom-0 right-0 block rounded-full ring-2 ring-white ${
            online ? 'bg-green-400' : 'bg-gray-400'
          }`}
          style={{
            width: size * 0.25,
            height: size * 0.25,
          }}
        />
      )}
    </div>
  );
};

export default Avatar;
