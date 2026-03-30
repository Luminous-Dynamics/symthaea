// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * ReactionDisplay Component
 *
 * Displays floating emoji reactions that animate upward.
 * Used in listening parties for real-time engagement.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { Reaction } from '@/hooks/useListeningParty';

export interface ReactionDisplayProps {
  /** Active reactions to display */
  reactions: Reaction[];
  /** Callback to send a new reaction */
  onReact?: (emoji: string) => void;
  /** Available reaction emojis */
  availableReactions?: string[];
  /** Whether reactions are enabled */
  enabled?: boolean;
  /** Additional CSS class */
  className?: string;
}

const DEFAULT_REACTIONS = [
  '\uD83D\uDD25', // fire
  '\u2764\uFE0F', // heart
  '\uD83D\uDC4F', // clap
  '\uD83D\uDE0D', // heart eyes
  '\uD83C\uDFB5', // music note
  '\uD83D\uDE80', // rocket
  '\uD83E\uDD73', // party face
  '\uD83C\uDF89', // party popper
];

export function ReactionDisplay({
  reactions,
  onReact,
  availableReactions = DEFAULT_REACTIONS,
  enabled = true,
  className = '',
}: ReactionDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showPicker, setShowPicker] = useState(false);

  const handleReact = useCallback((emoji: string) => {
    if (onReact && enabled) {
      onReact(emoji);
      setShowPicker(false);
    }
  }, [onReact, enabled]);

  // Close picker when clicking outside
  useEffect(() => {
    if (!showPicker) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowPicker(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showPicker]);

  return (
    <div ref={containerRef} className={`reaction-display relative ${className}`}>
      {/* Floating reactions */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {reactions.map((reaction) => (
          <FloatingReaction key={reaction.id} reaction={reaction} />
        ))}
      </div>

      {/* Reaction trigger button */}
      {onReact && enabled && (
        <div className="absolute bottom-4 right-4">
          <button
            onClick={() => setShowPicker(!showPicker)}
            className="w-12 h-12 rounded-full bg-gray-800/80 backdrop-blur-sm flex items-center justify-center text-2xl hover:bg-gray-700/80 transition-colors shadow-lg"
            title="Send reaction"
          >
            {showPicker ? '\u2715' : '\uD83D\uDE00'}
          </button>

          {/* Reaction picker */}
          {showPicker && (
            <div className="absolute bottom-16 right-0 bg-gray-800/95 backdrop-blur-sm rounded-xl p-2 shadow-xl">
              <div className="grid grid-cols-4 gap-1">
                {availableReactions.map((emoji) => (
                  <button
                    key={emoji}
                    onClick={() => handleReact(emoji)}
                    className="w-10 h-10 rounded-lg hover:bg-gray-700 flex items-center justify-center text-xl transition-colors"
                  >
                    {emoji}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Individual floating reaction animation
interface FloatingReactionProps {
  reaction: Reaction;
}

function FloatingReaction({ reaction }: FloatingReactionProps) {
  const [style, setStyle] = useState<React.CSSProperties>({
    left: `${reaction.x * 100}%`,
    bottom: '0%',
    opacity: 1,
  });

  useEffect(() => {
    // Animate upward and fade out
    const animationFrame = requestAnimationFrame(() => {
      setStyle({
        left: `${reaction.x * 100}%`,
        bottom: '100%',
        opacity: 0,
        transition: 'all 3s ease-out',
      });
    });

    return () => cancelAnimationFrame(animationFrame);
  }, [reaction.x]);

  return (
    <div
      className="floating-reaction absolute text-3xl pointer-events-none transform -translate-x-1/2"
      style={style}
    >
      <span className="drop-shadow-lg">{reaction.emoji}</span>
      <span
        className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-xs whitespace-nowrap px-2 py-0.5 rounded-full bg-black/50 text-white"
        style={{ opacity: 0.8 }}
      >
        {reaction.displayName}
      </span>
    </div>
  );
}

// Quick reaction bar (alternative to picker)
export interface QuickReactionBarProps {
  onReact: (emoji: string) => void;
  reactions?: string[];
  enabled?: boolean;
  className?: string;
}

export function QuickReactionBar({
  onReact,
  reactions = DEFAULT_REACTIONS.slice(0, 5),
  enabled = true,
  className = '',
}: QuickReactionBarProps) {
  if (!enabled) return null;

  return (
    <div className={`quick-reaction-bar flex gap-1 ${className}`}>
      {reactions.map((emoji) => (
        <button
          key={emoji}
          onClick={() => onReact(emoji)}
          className="w-9 h-9 rounded-full bg-gray-800/60 hover:bg-gray-700/80 flex items-center justify-center text-lg transition-all hover:scale-110"
        >
          {emoji}
        </button>
      ))}
    </div>
  );
}

export default ReactionDisplay;
