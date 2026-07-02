// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useMemo } from 'react';
import { usePrivy } from '@privy-io/react-auth';
import { ethers } from 'ethers';
import { Play, Heart, Gift, TrendingUp, Filter, Search, Loader2 } from 'lucide-react';
import { EconomicStrategySDK, PaymentModel } from '@/lib';
import { type Song } from '../data/mockSongs';
import { useSongs } from '../src/hooks/useSongs';
import MusicPlayer from '../components/MusicPlayer';
import Navigation from '../components/Navigation';

export default function DiscoverPage() {
  const { authenticated, user } = usePrivy();
  const [selectedGenre, setSelectedGenre] = useState<string>('all');
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [debouncedQuery, setDebouncedQuery] = useState<string>('');
  const [playingSong, setPlayingSong] = useState<Song | null>(null);
  const [currentSongIndex, setCurrentSongIndex] = useState<number>(0);

  // Debounce search query
  useMemo(() => {
    const timer = setTimeout(() => setDebouncedQuery(searchQuery), 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Fetch songs from API with filters
  const { songs, total, isLoading, error } = useSongs({
    genre: selectedGenre,
    model: selectedModel,
    q: debouncedQuery,
    limit: 50,
  });

  // API handles filtering, but we still filter locally for client-side search responsiveness
  const filteredSongs = songs;

  const handleOpenPlayer = (song: Song, index: number) => {
    setPlayingSong(song);
    setCurrentSongIndex(index);
  };

  const handleClosePlayer = () => {
    setPlayingSong(null);
  };

  const handleNextSong = () => {
    if (filteredSongs && currentSongIndex < filteredSongs.length - 1) {
      const nextIndex = currentSongIndex + 1;
      setCurrentSongIndex(nextIndex);
      setPlayingSong(filteredSongs[nextIndex]);
    }
  };

  const handlePreviousSong = () => {
    if (currentSongIndex > 0) {
      const prevIndex = currentSongIndex - 1;
      setCurrentSongIndex(prevIndex);
      setPlayingSong(filteredSongs[prevIndex]);
    }
  };

  const handleStreamSong = async (song: Song) => {
    if (!authenticated) {
      alert('Please connect your wallet to stream music');
      return;
    }

    try {
      const provider = new ethers.BrowserProvider((window as any).ethereum);
      const signer = await provider.getSigner();

      const sdk = new EconomicStrategySDK(
        provider,
        process.env.NEXT_PUBLIC_ROUTER_ADDRESS!,
        signer
      );

      // Check if payment required
      if (song.paymentModel === PaymentModel.PAY_PER_STREAM) {
        await sdk.streamSong(song.id, '0.01'); // $0.01 payment
      } else {
        // Free streaming (gift economy)
        await sdk.streamSong(song.id, '0');
      }

      // Update play count
      await fetch(`/api/songs/${song.id}/play`, { method: 'POST' });
    } catch (error) {
      console.error('Stream failed:', error);
      alert('Failed to stream song: ' + (error as Error).message);
    }
  };

  const handleTip = async (song: Song) => {
    if (!authenticated) {
      alert('Please connect your wallet to tip artists');
      return;
    }

    const tipAmount = prompt('Enter tip amount (FLOW):', '1.0');
    if (!tipAmount) return;

    try {
      const provider = new ethers.BrowserProvider((window as any).ethereum);
      const signer = await provider.getSigner();

      const sdk = new EconomicStrategySDK(
        provider,
        process.env.NEXT_PUBLIC_ROUTER_ADDRESS!,
        signer
      );

      await sdk.tipArtist(song.id, tipAmount);

      alert(`Successfully tipped ${tipAmount} FLOW! 🎉`);
    } catch (error) {
      console.error('Tip failed:', error);
      alert('Failed to tip artist: ' + (error as Error).message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <Navigation />
      <div className="max-w-7xl mx-auto px-4 py-12 pt-24">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Discover Music</h1>
          <p className="text-gray-400">Explore songs from independent artists worldwide</p>
        </div>

        {/* Search Bar */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4 mb-4">
          <div className="flex items-center space-x-3">
            <Search className="w-5 h-5 text-purple-400" />
            <input
              type="text"
              placeholder="Search by song title, artist, or genre..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 bg-transparent border-none text-white placeholder-gray-500 focus:outline-none"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="text-gray-400 hover:text-white transition text-sm"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 mb-8">
          <div className="flex items-center space-x-4">
            <Filter className="w-5 h-5 text-purple-400" />
            <div className="flex-1 grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-400 text-sm mb-2">Genre</label>
                <select
                  value={selectedGenre}
                  onChange={(e) => setSelectedGenre(e.target.value)}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:border-purple-400 focus:outline-none [&>option]:bg-gray-900 [&>option]:text-white"
                >
                  <option value="all">All Genres</option>
                  <option value="Electronic">Electronic</option>
                  <option value="Rock">Rock</option>
                  <option value="Classical">Classical</option>
                  <option value="Hip-Hop">Hip-Hop</option>
                  <option value="Jazz">Jazz</option>
                  <option value="Ambient">Ambient</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-2">Economic Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:border-purple-400 focus:outline-none [&>option]:bg-gray-900 [&>option]:text-white"
                >
                  <option value="all">All Models</option>
                  <option value={PaymentModel.PAY_PER_STREAM}>Pay Per Stream ($0.01)</option>
                  <option value={PaymentModel.FREEMIUM}>Freemium (3 free plays)</option>
                  <option value={PaymentModel.PAY_WHAT_YOU_WANT}>Pay What You Want (Tips)</option>
                  <option value={PaymentModel.PATRONAGE}>Patronage (Subscription)</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Songs Grid */}
        {error && (
          <div className="text-center py-12">
            <p className="text-red-400">Failed to load songs from server - showing cached data</p>
          </div>
        )}

        {isLoading && (
          <div className="text-center py-12">
            <Loader2 className="inline-block h-12 w-12 text-purple-400 animate-spin" />
            <p className="text-gray-400 mt-4">Loading songs...</p>
          </div>
        )}

        {/* Total count */}
        {!isLoading && total > 0 && (
          <p className="text-gray-500 text-sm mb-4">{total} songs found</p>
        )}

        {filteredSongs && filteredSongs.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-400">No songs found with these filters</p>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4 md:gap-6">
          {filteredSongs?.map((song, index) => (
            <div
              key={song.id}
              className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 hover:border-purple-400/50 transition group"
            >
              {/* Song Cover - Clickable to open player */}
              <div
                onClick={() => handleOpenPlayer(song, index)}
                className="w-full aspect-square rounded-lg mb-4 cursor-pointer relative overflow-hidden group"
              >
                <img
                  src={song.coverArt}
                  alt={`${song.title} by ${song.artist}`}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                  <Play className="w-20 h-20 text-white" />
                </div>
              </div>

              {/* Song Info */}
              <h3 className="text-xl font-bold text-white mb-1 truncate">{song.title}</h3>
              <p className="text-gray-400 mb-3 truncate">{song.artist}</p>

              {/* Stats */}
              <div className="flex items-center space-x-4 text-sm text-gray-500 mb-4">
                <span>{song.genre}</span>
                <span>•</span>
                <span>{song.plays} plays</span>
              </div>

              {/* Economic Model Badge */}
              <div className="mb-4">
                {song.paymentModel === PaymentModel.PAY_PER_STREAM && (
                  <div className="inline-flex items-center space-x-2 px-3 py-1 bg-blue-500/20 border border-blue-500/30 rounded-full text-blue-400 text-sm">
                    <TrendingUp className="w-4 h-4" />
                    <span>$0.01 per stream</span>
                  </div>
                )}
                {song.paymentModel === PaymentModel.FREEMIUM && (
                  <div className="inline-flex items-center space-x-2 px-3 py-1 bg-orange-500/20 border border-orange-500/30 rounded-full text-orange-400 text-sm">
                    <Gift className="w-4 h-4" />
                    <span>{song.freePlaysRemaining || 3} free plays</span>
                  </div>
                )}
                {song.paymentModel === PaymentModel.PAY_WHAT_YOU_WANT && (
                  <div className="inline-flex items-center space-x-2 px-3 py-1 bg-green-500/20 border border-green-500/30 rounded-full text-green-400 text-sm">
                    <Heart className="w-4 h-4" />
                    <span>Free + Tip what you want</span>
                  </div>
                )}
                {song.paymentModel === PaymentModel.PATRONAGE && (
                  <div className="inline-flex items-center space-x-2 px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded-full text-purple-400 text-sm">
                    <TrendingUp className="w-4 h-4" />
                    <span>Monthly subscription</span>
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex space-x-2">
                <button
                  onClick={() => handleStreamSong(song)}
                  disabled={playingSong?.id === song.id}
                  className={`flex-1 px-4 py-2 rounded-lg font-medium transition ${
                    playingSong?.id === song.id
                      ? 'bg-green-600 text-white'
                      : 'bg-purple-600 hover:bg-purple-700 text-white'
                  }`}
                >
                  {playingSong?.id === song.id ? (
                    <>
                      <Play className="w-4 h-4 inline mr-2" />
                      Playing...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 inline mr-2" />
                      {song.paymentModel === PaymentModel.PAY_PER_STREAM && 'Stream $0.01'}
                      {song.paymentModel === PaymentModel.FREEMIUM && 'Try Free'}
                      {song.paymentModel === PaymentModel.PAY_WHAT_YOU_WANT && 'Listen Free'}
                      {song.paymentModel === PaymentModel.PATRONAGE && 'Subscribe'}
                    </>
                  )}
                </button>
                <button
                  onClick={() => handleTip(song)}
                  className="px-4 py-2 border border-purple-400 text-purple-400 hover:bg-purple-400/10 rounded-lg transition"
                >
                  <Heart className="w-4 h-4" />
                </button>
              </div>

              {/* Model-specific info */}
              {song.paymentModel === PaymentModel.FREEMIUM && song.freePlaysRemaining && (
                <div className="mt-3 p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                  <p className="text-orange-400 text-sm">
                    🎵 {song.freePlaysRemaining} free plays remaining, then $0.01/stream
                  </p>
                </div>
              )}
              {song.paymentModel === PaymentModel.PAY_WHAT_YOU_WANT && song.tipAmount && (
                <div className="mt-3 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <p className="text-green-400 text-sm">
                    💚 Free to listen • Listeners tip {song.tipAmount} on average
                  </p>
                </div>
              )}
              {song.paymentModel === PaymentModel.PATRONAGE && (
                <div className="mt-3 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                  <p className="text-purple-400 text-sm">
                    ⭐ Unlimited plays for patrons • Support this artist monthly
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Music Player Modal */}
        {playingSong && (
          <MusicPlayer
            song={playingSong}
            onClose={handleClosePlayer}
            onNext={currentSongIndex < (filteredSongs?.length || 0) - 1 ? handleNextSong : undefined}
            onPrevious={currentSongIndex > 0 ? handlePreviousSong : undefined}
          />
        )}
      </div>
    </div>
  );
}
