// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { Song } from '@/lib/api';
import { usePlayerStore } from '@/store/playerStore';
import { formatDuration } from '@/lib/utils';
import { Play, Pause, MoreHorizontal, Heart } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';

interface SongCardProps {
  song: Song;
  showArtist?: boolean;
  variant?: 'card' | 'row';
  index?: number;
}

export function SongCard({ song, showArtist = true, variant = 'card', index }: SongCardProps) {
  const { currentSong, isPlaying, play, pause, resume } = usePlayerStore();
  const [isHovered, setIsHovered] = useState(false);

  const isCurrentSong = currentSong?.id === song.id;
  const isCurrentlyPlaying = isCurrentSong && isPlaying;

  const handlePlay = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (isCurrentSong) {
      if (isPlaying) pause();
      else resume();
    } else {
      play(song);
    }
  };

  if (variant === 'row') {
    return (
      <div
        className={`group flex items-center gap-4 px-4 py-2 rounded-md hover:bg-white/10 transition-colors ${
          isCurrentSong ? 'bg-white/5' : ''
        }`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Index / Play Button */}
        <div className="w-8 flex items-center justify-center">
          {isHovered || isCurrentlyPlaying ? (
            <button
              onClick={handlePlay}
              className="w-8 h-8 flex items-center justify-center"
            >
              {isCurrentlyPlaying ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4 ml-0.5" />
              )}
            </button>
          ) : (
            <span className={`text-sm ${isCurrentSong ? 'text-primary' : 'text-muted-foreground'}`}>
              {index !== undefined ? index + 1 : ''}
            </span>
          )}
        </div>

        {/* Cover Art */}
        <div className="relative w-10 h-10 rounded overflow-hidden flex-shrink-0">
          <Image
            src={song.coverArt || '/placeholder-album.png'}
            alt={song.title}
            fill
            className="object-cover"
          />
        </div>

        {/* Song Info */}
        <div className="flex-1 min-w-0">
          <Link
            href={`/song/${song.id}`}
            className={`block text-sm font-medium truncate hover:underline ${
              isCurrentSong ? 'text-primary' : ''
            }`}
          >
            {song.title}
          </Link>
          {showArtist && (
            <Link
              href={`/artist/${song.artistAddress}`}
              className="block text-xs text-muted-foreground truncate hover:underline"
            >
              {song.artist}
            </Link>
          )}
        </div>

        {/* Like Button */}
        <button className="p-2 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-white">
          <Heart className={`w-4 h-4 ${song.isLiked ? 'fill-primary text-primary' : ''}`} />
        </button>

        {/* Duration */}
        <span className="text-sm text-muted-foreground w-12 text-right">
          {formatDuration(song.duration)}
        </span>

        {/* More Button */}
        <button className="p-2 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-white">
          <MoreHorizontal className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      className="group relative p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors cursor-pointer"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => play(song)}
    >
      {/* Cover Art */}
      <div className="relative aspect-square rounded-md overflow-hidden mb-4 shadow-lg">
        <Image
          src={song.coverArt || '/placeholder-album.png'}
          alt={song.title}
          fill
          className="object-cover"
        />

        {/* Play Button Overlay */}
        <button
          onClick={handlePlay}
          className={`absolute bottom-2 right-2 w-12 h-12 rounded-full bg-primary shadow-xl flex items-center justify-center transition-all ${
            isHovered || isCurrentlyPlaying
              ? 'opacity-100 translate-y-0'
              : 'opacity-0 translate-y-2'
          }`}
        >
          {isCurrentlyPlaying ? (
            <Pause className="w-6 h-6 text-black" />
          ) : (
            <Play className="w-6 h-6 text-black ml-1" />
          )}
        </button>
      </div>

      {/* Song Info */}
      <Link href={`/song/${song.id}`} className="block">
        <h3 className="font-medium truncate hover:underline">{song.title}</h3>
      </Link>
      {showArtist && (
        <Link
          href={`/artist/${song.artistAddress}`}
          className="text-sm text-muted-foreground truncate hover:underline block"
        >
          {song.artist}
        </Link>
      )}
    </div>
  );
}

export default SongCard;
