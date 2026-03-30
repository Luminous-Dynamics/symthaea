// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePlayerStore } from '@/store/playerStore';
import {
  Users,
  MessageCircle,
  Heart,
  Share2,
  Crown,
  Music2,
  Send,
  Smile,
  ThumbsUp,
  Flame,
  Sparkles,
  X,
  Volume2,
  VolumeX,
  SkipForward,
  ListMusic,
  UserPlus,
  Link2,
  Settings,
} from 'lucide-react';
import Image from 'next/image';

interface Participant {
  id: string;
  name: string;
  avatar?: string;
  isHost: boolean;
  reaction?: string;
  joinedAt: Date;
}

interface ChatMessage {
  id: string;
  userId: string;
  userName: string;
  userColor: string;
  content: string;
  timestamp: number; // Position in song when sent
  createdAt: Date;
}

interface TimedReaction {
  id: string;
  userId: string;
  emoji: string;
  timestamp: number;
  position: { x: number; y: number };
}

interface ListeningCircleProps {
  circleId: string;
  circleName: string;
  isHost: boolean;
  onLeave: () => void;
}

const REACTIONS = ['❤️', '🔥', '✨', '🎵', '👏', '🙌', '💫', '🌟'];

export function ListeningCircle({
  circleId,
  circleName,
  isHost,
  onLeave,
}: ListeningCircleProps) {
  const { currentSong, position, isPlaying, toggle, next } = usePlayerStore();

  const [participants, setParticipants] = useState<Participant[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [timedReactions, setTimedReactions] = useState<TimedReaction[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [showChat, setShowChat] = useState(true);
  const [showQueue, setShowQueue] = useState(false);
  const [showInvite, setShowInvite] = useState(false);

  const chatRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Connect to circle WebSocket
  useEffect(() => {
    const ws = new WebSocket(`${process.env.NEXT_PUBLIC_CIRCLES_WS_URL}?circleId=${circleId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'participants':
          setParticipants(data.payload);
          break;
        case 'participant:joined':
          setParticipants(prev => [...prev, data.payload]);
          break;
        case 'participant:left':
          setParticipants(prev => prev.filter(p => p.id !== data.payload.id));
          break;
        case 'chat:message':
          setMessages(prev => [...prev, data.payload]);
          break;
        case 'reaction':
          addTimedReaction(data.payload);
          break;
        case 'sync:playback':
          // Sync playback state from host
          break;
      }
    };

    return () => {
      ws.close();
    };
  }, [circleId]);

  // Auto-scroll chat
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // Add timed reaction with animation
  const addTimedReaction = (reaction: TimedReaction) => {
    setTimedReactions(prev => [...prev, reaction]);
    setTimeout(() => {
      setTimedReactions(prev => prev.filter(r => r.id !== reaction.id));
    }, 3000);
  };

  // Send message
  const sendMessage = () => {
    if (!newMessage.trim()) return;

    wsRef.current?.send(JSON.stringify({
      type: 'chat:message',
      payload: {
        content: newMessage,
        timestamp: position,
      },
    }));

    setNewMessage('');
  };

  // Send reaction
  const sendReaction = (emoji: string) => {
    wsRef.current?.send(JSON.stringify({
      type: 'reaction',
      payload: {
        emoji,
        timestamp: position,
        position: {
          x: Math.random() * 80 + 10,
          y: Math.random() * 60 + 20,
        },
      },
    }));
  };

  // Copy invite link
  const copyInviteLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}/circles/join/${circleId}`);
    setShowInvite(false);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black flex">
      {/* Main Area - Album Art & Reactions */}
      <div className="flex-1 relative flex items-center justify-center overflow-hidden">
        {/* Blurred Background */}
        {currentSong && (
          <div className="absolute inset-0">
            <Image
              src={currentSong.coverArt || '/placeholder-album.png'}
              alt=""
              fill
              className="object-cover blur-3xl opacity-30 scale-110"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent" />
          </div>
        )}

        {/* Floating Reactions */}
        {timedReactions.map(reaction => (
          <div
            key={reaction.id}
            className="absolute text-4xl animate-float-up pointer-events-none"
            style={{
              left: `${reaction.position.x}%`,
              top: `${reaction.position.y}%`,
            }}
          >
            {reaction.emoji}
          </div>
        ))}

        {/* Center Content */}
        <div className="relative z-10 text-center">
          {/* Album Art */}
          {currentSong && (
            <div className="relative w-80 h-80 mx-auto mb-8 rounded-2xl overflow-hidden shadow-2xl">
              <Image
                src={currentSong.coverArt || '/placeholder-album.png'}
                alt={currentSong.title}
                fill
                className="object-cover"
              />
            </div>
          )}

          {/* Song Info */}
          <h2 className="text-3xl font-bold mb-2">{currentSong?.title || 'No song playing'}</h2>
          <p className="text-lg text-muted-foreground mb-8">{currentSong?.artist}</p>

          {/* Reaction Bar */}
          <div className="flex items-center justify-center gap-2 mb-8">
            {REACTIONS.map(emoji => (
              <button
                key={emoji}
                onClick={() => sendReaction(emoji)}
                className="w-12 h-12 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-2xl transition-transform hover:scale-110 active:scale-95"
              >
                {emoji}
              </button>
            ))}
          </div>

          {/* Participants Ring */}
          <div className="flex items-center justify-center -space-x-3">
            {participants.slice(0, 8).map((p, i) => (
              <div
                key={p.id}
                className={`relative w-12 h-12 rounded-full border-2 border-black overflow-hidden ${
                  p.isHost ? 'ring-2 ring-yellow-400' : ''
                }`}
                style={{ zIndex: participants.length - i }}
                title={p.name}
              >
                {p.avatar ? (
                  <Image src={p.avatar} alt={p.name} fill className="object-cover" />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-lg font-bold">
                    {p.name[0].toUpperCase()}
                  </div>
                )}
                {p.isHost && (
                  <Crown className="absolute -top-1 -right-1 w-4 h-4 text-yellow-400" />
                )}
                {p.reaction && (
                  <div className="absolute -bottom-1 -right-1 text-sm">
                    {p.reaction}
                  </div>
                )}
              </div>
            ))}
            {participants.length > 8 && (
              <div className="w-12 h-12 rounded-full bg-gray-700 border-2 border-black flex items-center justify-center text-sm font-medium">
                +{participants.length - 8}
              </div>
            )}
          </div>

          <p className="mt-4 text-sm text-muted-foreground">
            {participants.length} listening together
          </p>
        </div>

        {/* Header Controls */}
        <div className="absolute top-0 left-0 right-0 p-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onLeave}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20"
            >
              <X className="w-5 h-5" />
            </button>
            <div>
              <h3 className="font-semibold">{circleName}</h3>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                <Users className="w-3 h-3" />
                {participants.length} listeners
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowInvite(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500 hover:bg-purple-600 text-sm font-medium"
            >
              <UserPlus className="w-4 h-4" />
              Invite
            </button>
            {isHost && (
              <button className="p-2 rounded-full bg-white/10 hover:bg-white/20">
                <Settings className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Host Controls */}
        {isHost && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-4 px-6 py-3 rounded-full bg-white/10 backdrop-blur-lg">
            <button onClick={toggle} className="p-3 rounded-full bg-white text-black">
              {isPlaying ? '⏸' : '▶️'}
            </button>
            <button onClick={next} className="p-2 rounded-full hover:bg-white/20">
              <SkipForward className="w-5 h-5" />
            </button>
            <button
              onClick={() => setShowQueue(!showQueue)}
              className={`p-2 rounded-full ${showQueue ? 'bg-purple-500' : 'hover:bg-white/20'}`}
            >
              <ListMusic className="w-5 h-5" />
            </button>
          </div>
        )}
      </div>

      {/* Chat Sidebar */}
      {showChat && (
        <div className="w-80 bg-gray-900 border-l border-white/10 flex flex-col">
          {/* Chat Header */}
          <div className="p-4 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageCircle className="w-5 h-5 text-purple-400" />
              <span className="font-semibold">Live Chat</span>
            </div>
            <button
              onClick={() => setShowChat(false)}
              className="p-1 rounded hover:bg-white/10"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Messages */}
          <div ref={chatRef} className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map(msg => (
              <div key={msg.id} className="group">
                <div className="flex items-start gap-2">
                  <div
                    className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0"
                    style={{ backgroundColor: msg.userColor }}
                  >
                    {msg.userName[0].toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2">
                      <span className="text-sm font-medium">{msg.userName}</span>
                      <span className="text-xs text-muted-foreground">
                        at {formatTimestamp(msg.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-300 break-words">{msg.content}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Message Input */}
          <div className="p-4 border-t border-white/10">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Say something..."
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm focus:outline-none focus:border-purple-500"
              />
              <button
                onClick={sendMessage}
                disabled={!newMessage.trim()}
                className="p-2 rounded-lg bg-purple-500 hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Chat Toggle (when hidden) */}
      {!showChat && (
        <button
          onClick={() => setShowChat(true)}
          className="absolute right-4 bottom-4 p-3 rounded-full bg-purple-500 hover:bg-purple-600 shadow-lg"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      )}

      {/* Invite Modal */}
      {showInvite && (
        <div className="fixed inset-0 z-60 bg-black/80 flex items-center justify-center">
          <div className="bg-gray-900 rounded-2xl p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold mb-4">Invite Friends</h3>
            <p className="text-muted-foreground mb-6">
              Share this link to invite others to your listening circle:
            </p>
            <div className="flex items-center gap-2 p-3 bg-white/5 rounded-lg mb-6">
              <Link2 className="w-5 h-5 text-muted-foreground" />
              <span className="flex-1 text-sm truncate">
                {window.location.origin}/circles/join/{circleId}
              </span>
              <button
                onClick={copyInviteLink}
                className="px-3 py-1 bg-purple-500 rounded text-sm font-medium"
              >
                Copy
              </button>
            </div>
            <button
              onClick={() => setShowInvite(false)}
              className="w-full py-2 bg-white/10 rounded-lg hover:bg-white/20"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* CSS for floating animation */}
      <style jsx>{`
        @keyframes float-up {
          0% {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
          100% {
            opacity: 0;
            transform: translateY(-100px) scale(1.5);
          }
        }
        .animate-float-up {
          animation: float-up 3s ease-out forwards;
        }
      `}</style>
    </div>
  );
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
