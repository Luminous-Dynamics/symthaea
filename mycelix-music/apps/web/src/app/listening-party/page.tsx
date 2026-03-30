// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { ListeningParty } from '@/components/listening-party';
import { Users, Plus, Music, Globe, Lock, Clock } from 'lucide-react';

// Mock active parties
const activeParties = [
  { id: '1', name: 'Friday Night Vibes', host: 'DJ Aurora', participants: 12, genre: 'House', isPrivate: false, nowPlaying: 'Summer Nights - Horizons' },
  { id: '2', name: 'Chill Study Session', host: 'LofiMaster', participants: 8, genre: 'Lo-Fi', isPrivate: false, nowPlaying: 'Rainy Days - CloudWalker' },
  { id: '3', name: 'Album Listening: Dark Side', host: 'ClassicFan', participants: 24, genre: 'Rock', isPrivate: false, nowPlaying: 'Money - Pink Floyd' },
];

export default function ListeningPartyPage() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [activeParty, setActiveParty] = useState<string | null>(null);
  const [newParty, setNewParty] = useState({
    name: '',
    isPrivate: false,
  });

  if (activeParty) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black p-4">
        <div className="max-w-6xl mx-auto">
          <button
            onClick={() => setActiveParty(null)}
            className="mb-4 flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            ← Leave Party
          </button>
          <ListeningParty
            partyId={activeParty}
            onLeave={() => setActiveParty(null)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black">
      {/* Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 via-transparent to-purple-500/20" />
        <div className="relative max-w-6xl mx-auto px-4 py-12">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                <Users className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Listening Parties</h1>
                <p className="text-gray-400">Listen together with friends in real-time</p>
              </div>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 text-white font-medium hover:opacity-90 transition-opacity"
            >
              <Plus className="w-5 h-5" />
              Create Party
            </button>
          </div>
        </div>
      </div>

      {/* Active Parties */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <h2 className="text-xl font-semibold text-white mb-6">Active Parties</h2>

        {activeParties.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {activeParties.map((party) => (
              <div
                key={party.id}
                className="rounded-xl bg-gray-900/50 border border-gray-800 hover:border-gray-700 transition-all overflow-hidden cursor-pointer"
                onClick={() => setActiveParty(party.id)}
              >
                {/* Party Header */}
                <div className="p-4 border-b border-gray-800">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-white">{party.name}</h3>
                    {party.isPrivate ? (
                      <Lock className="w-4 h-4 text-gray-500" />
                    ) : (
                      <Globe className="w-4 h-4 text-green-400" />
                    )}
                  </div>
                  <p className="text-sm text-gray-400">Hosted by {party.host}</p>
                </div>

                {/* Now Playing */}
                <div className="p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                      <Music className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-gray-400">Now Playing</p>
                      <p className="text-sm text-white truncate">{party.nowPlaying}</p>
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div className="p-4 flex items-center justify-between">
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <span className="flex items-center gap-1">
                      <Users className="w-4 h-4" />
                      {party.participants}
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-gray-800 text-xs">
                      {party.genre}
                    </span>
                  </div>
                  <button className="px-4 py-2 rounded-lg bg-purple-500 text-white text-sm font-medium hover:bg-purple-600 transition-colors">
                    Join
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <Users className="w-16 h-16 text-gray-700 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-400 mb-2">No active parties</h3>
            <p className="text-gray-500 mb-6">Start one and invite your friends!</p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-3 rounded-xl bg-purple-500 text-white font-medium hover:bg-purple-600 transition-colors"
            >
              Create Party
            </button>
          </div>
        )}
      </div>

      {/* How It Works */}
      <div className="max-w-6xl mx-auto px-4 pb-8">
        <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">How Listening Parties Work</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-blue-400 font-bold">1</span>
              </div>
              <div>
                <h4 className="font-medium text-white mb-1">Create or Join</h4>
                <p className="text-sm text-gray-400">Start a new party or join an existing one</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-purple-400 font-bold">2</span>
              </div>
              <div>
                <h4 className="font-medium text-white mb-1">Listen Together</h4>
                <p className="text-sm text-gray-400">Everyone hears the same music in perfect sync</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-fuchsia-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-fuchsia-400 font-bold">3</span>
              </div>
              <div>
                <h4 className="font-medium text-white mb-1">Chat & React</h4>
                <p className="text-sm text-gray-400">Share reactions and chat with other listeners</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Create Party Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 w-full max-w-md">
            <h3 className="text-xl font-semibold text-white mb-6">Create Listening Party</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Party Name</label>
                <input
                  type="text"
                  value={newParty.name}
                  onChange={(e) => setNewParty(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Friday Night Vibes"
                  className="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50">
                <div>
                  <p className="text-white font-medium">Private Party</p>
                  <p className="text-sm text-gray-400">Only people with the link can join</p>
                </div>
                <button
                  onClick={() => setNewParty(prev => ({ ...prev, isPrivate: !prev.isPrivate }))}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    newParty.isPrivate ? 'bg-purple-500' : 'bg-gray-700'
                  }`}
                >
                  <div
                    className={`w-5 h-5 rounded-full bg-white transition-transform ${
                      newParty.isPrivate ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-3 rounded-lg border border-gray-700 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setActiveParty('new-' + Date.now());
                  setShowCreateModal(false);
                }}
                className="flex-1 px-4 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white font-medium hover:opacity-90 transition-opacity"
              >
                Create Party
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
