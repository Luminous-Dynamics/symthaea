// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import Image from 'next/image';
import {
  Music2,
  Radio,
  Users,
  Mic2,
  Calendar,
  MapPin,
  Clock,
  Sparkles,
  Star,
  Shield,
  Crown,
  Heart,
  Zap,
} from 'lucide-react';

export type PresenceType =
  | 'LIVE_CONCERT'
  | 'VIRTUAL_CONCERT'
  | 'ALBUM_RELEASE'
  | 'FIRST_LISTEN'
  | 'LISTENING_CIRCLE'
  | 'ARTIST_MILESTONE'
  | 'COLLABORATIVE_SESSION'
  | 'COMMUNITY_EVENT'
  | 'SEASONAL_GATHERING'
  | 'PATRONAGE_RENEWAL';

export interface Presence {
  id: string;
  type: PresenceType;
  eventId: string;
  eventName: string;
  timestamp: Date;
  duration: number; // seconds
  artistId: string;
  artistName: string;
  artistAvatar?: string;
  collaborators?: { id: string; name: string }[];
  location: string;
  resonanceLevel: number; // 1-100
  imageUrl?: string;
  abilities?: string[];
}

interface PresenceTokenProps {
  presence: Presence;
  size?: 'small' | 'medium' | 'large';
  showDetails?: boolean;
}

const TYPE_CONFIG: Record<PresenceType, {
  icon: typeof Music2;
  label: string;
  gradient: string;
  glow: string;
}> = {
  LIVE_CONCERT: {
    icon: Mic2,
    label: 'Live Concert',
    gradient: 'from-red-500 via-orange-500 to-yellow-500',
    glow: 'shadow-red-500/50',
  },
  VIRTUAL_CONCERT: {
    icon: Radio,
    label: 'Virtual Concert',
    gradient: 'from-purple-500 via-pink-500 to-red-500',
    glow: 'shadow-purple-500/50',
  },
  ALBUM_RELEASE: {
    icon: Music2,
    label: 'Album Release',
    gradient: 'from-blue-500 via-indigo-500 to-purple-500',
    glow: 'shadow-blue-500/50',
  },
  FIRST_LISTEN: {
    icon: Sparkles,
    label: 'First Listen',
    gradient: 'from-yellow-400 via-amber-500 to-orange-500',
    glow: 'shadow-yellow-500/50',
  },
  LISTENING_CIRCLE: {
    icon: Users,
    label: 'Listening Circle',
    gradient: 'from-green-500 via-emerald-500 to-teal-500',
    glow: 'shadow-green-500/50',
  },
  ARTIST_MILESTONE: {
    icon: Crown,
    label: 'Artist Milestone',
    gradient: 'from-amber-400 via-yellow-500 to-amber-600',
    glow: 'shadow-amber-500/50',
  },
  COLLABORATIVE_SESSION: {
    icon: Users,
    label: 'Collaboration',
    gradient: 'from-cyan-500 via-blue-500 to-indigo-500',
    glow: 'shadow-cyan-500/50',
  },
  COMMUNITY_EVENT: {
    icon: Heart,
    label: 'Community Event',
    gradient: 'from-pink-500 via-rose-500 to-red-500',
    glow: 'shadow-pink-500/50',
  },
  SEASONAL_GATHERING: {
    icon: Star,
    label: 'Seasonal Gathering',
    gradient: 'from-violet-500 via-purple-500 to-fuchsia-500',
    glow: 'shadow-violet-500/50',
  },
  PATRONAGE_RENEWAL: {
    icon: Shield,
    label: 'Patronage Renewal',
    gradient: 'from-emerald-400 via-green-500 to-teal-500',
    glow: 'shadow-emerald-500/50',
  },
};

export function PresenceToken({ presence, size = 'medium', showDetails = false }: PresenceTokenProps) {
  const [isHovered, setIsHovered] = useState(false);
  const config = TYPE_CONFIG[presence.type];
  const Icon = config.icon;

  const sizeClasses = {
    small: 'w-20 h-20',
    medium: 'w-32 h-32',
    large: 'w-48 h-48',
  };

  const iconSizes = {
    small: 'w-6 h-6',
    medium: 'w-10 h-10',
    large: 'w-14 h-14',
  };

  return (
    <div
      className={`relative ${sizeClasses[size]} group cursor-pointer`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Outer glow ring */}
      <div
        className={`absolute inset-0 rounded-full bg-gradient-to-br ${config.gradient} opacity-30 blur-xl transition-opacity ${
          isHovered ? 'opacity-60' : 'opacity-30'
        }`}
      />

      {/* Token body */}
      <div
        className={`relative w-full h-full rounded-full bg-gradient-to-br ${config.gradient} p-0.5 shadow-lg ${config.glow} transition-transform ${
          isHovered ? 'scale-105' : ''
        }`}
      >
        <div className="w-full h-full rounded-full bg-gray-900 flex items-center justify-center overflow-hidden">
          {/* Background image or gradient */}
          {presence.imageUrl ? (
            <Image
              src={presence.imageUrl}
              alt={presence.eventName}
              fill
              className="object-cover opacity-40"
            />
          ) : (
            <div className={`absolute inset-0 bg-gradient-to-br ${config.gradient} opacity-20`} />
          )}

          {/* Center icon */}
          <div className="relative z-10 flex flex-col items-center">
            <Icon className={`${iconSizes[size]} mb-1`} />
            {size !== 'small' && (
              <span className="text-xs font-medium text-center px-2 opacity-80">
                {config.label}
              </span>
            )}
          </div>

          {/* Resonance ring */}
          <svg
            className="absolute inset-0 w-full h-full -rotate-90"
            viewBox="0 0 100 100"
          >
            <circle
              cx="50"
              cy="50"
              r="48"
              fill="none"
              stroke="currentColor"
              strokeWidth="1"
              className="text-white/10"
            />
            <circle
              cx="50"
              cy="50"
              r="48"
              fill="none"
              stroke="url(#resonance-gradient)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeDasharray={`${presence.resonanceLevel * 3.01} 301`}
              className="transition-all duration-1000"
            />
            <defs>
              <linearGradient id="resonance-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8B5CF6" />
                <stop offset="100%" stopColor="#EC4899" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </div>

      {/* Resonance badge */}
      <div className="absolute -bottom-1 -right-1 w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-xs font-bold shadow-lg">
        {presence.resonanceLevel}
      </div>

      {/* Soulbound indicator */}
      <div className="absolute -top-1 -left-1 w-6 h-6 rounded-full bg-gray-900 border-2 border-purple-500 flex items-center justify-center" title="Soulbound - Cannot be transferred">
        <Shield className="w-3 h-3 text-purple-400" />
      </div>

      {/* Hover details */}
      {showDetails && isHovered && (
        <div className="absolute left-full top-0 ml-4 w-64 bg-gray-900 rounded-xl p-4 shadow-xl border border-white/10 z-50">
          <h4 className="font-semibold mb-2">{presence.eventName}</h4>
          <p className="text-sm text-muted-foreground mb-3">{presence.artistName}</p>

          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Calendar className="w-4 h-4" />
              <span>{presence.timestamp.toLocaleDateString()}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="w-4 h-4" />
              <span>{presence.location}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock className="w-4 h-4" />
              <span>{Math.round(presence.duration / 60)} min present</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400" />
              <span className="text-purple-400">Resonance: {presence.resonanceLevel}%</span>
            </div>
          </div>

          {presence.abilities && presence.abilities.length > 0 && (
            <div className="mt-3 pt-3 border-t border-white/10">
              <p className="text-xs text-muted-foreground mb-2">Unlocked Abilities:</p>
              <div className="flex flex-wrap gap-1">
                {presence.abilities.map((ability) => (
                  <span
                    key={ability}
                    className="px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded text-xs"
                  >
                    {ability}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Grid display of multiple presence tokens
interface PresenceCollectionProps {
  presences: Presence[];
  title?: string;
}

export function PresenceCollection({ presences, title }: PresenceCollectionProps) {
  return (
    <div>
      {title && <h3 className="text-lg font-semibold mb-4">{title}</h3>}
      <div className="flex flex-wrap gap-4">
        {presences.map((presence) => (
          <PresenceToken
            key={presence.id}
            presence={presence}
            size="medium"
            showDetails
          />
        ))}
      </div>
    </div>
  );
}
