// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { SongCard } from '@/components/ui/SongCard';
import { Search, X, TrendingUp, Clock, User, Music2, ListMusic } from 'lucide-react';
import { debounce, cn } from '@/lib/utils';
import Link from 'next/link';
import Image from 'next/image';

type SearchFilter = 'all' | 'songs' | 'artists' | 'playlists';

export default function SearchPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('q') || '';

  const [query, setQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const [filter, setFilter] = useState<SearchFilter>('all');

  // Debounce search query
  const debouncedSetQuery = useCallback(
    debounce((value: string) => {
      setDebouncedQuery(value);
      if (value) {
        router.push(`/search?q=${encodeURIComponent(value)}`, { scroll: false });
      } else {
        router.push('/search', { scroll: false });
      }
    }, 300),
    [router]
  );

  useEffect(() => {
    debouncedSetQuery(query);
  }, [query, debouncedSetQuery]);

  // Search query
  const { data: results, isLoading } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () => api.search(debouncedQuery),
    enabled: debouncedQuery.length >= 2,
  });

  // Recent searches (from localStorage)
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('recentSearches');
    if (stored) {
      setRecentSearches(JSON.parse(stored));
    }
  }, []);

  const addRecentSearch = (term: string) => {
    const updated = [term, ...recentSearches.filter((s) => s !== term)].slice(0, 10);
    setRecentSearches(updated);
    localStorage.setItem('recentSearches', JSON.stringify(updated));
  };

  const clearRecentSearches = () => {
    setRecentSearches([]);
    localStorage.removeItem('recentSearches');
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      addRecentSearch(query.trim());
    }
  };

  const filters: { value: SearchFilter; label: string; icon: React.ReactNode }[] = [
    { value: 'all', label: 'All', icon: null },
    { value: 'songs', label: 'Songs', icon: <Music2 className="w-4 h-4" /> },
    { value: 'artists', label: 'Artists', icon: <User className="w-4 h-4" /> },
    { value: 'playlists', label: 'Playlists', icon: <ListMusic className="w-4 h-4" /> },
  ];

  const hasResults = results && (
    results.songs.length > 0 ||
    results.artists.length > 0 ||
    results.playlists.length > 0
  );

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Search Input */}
          <form onSubmit={handleSearchSubmit} className="mb-6">
            <div className="relative max-w-xl">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What do you want to listen to?"
                className="w-full h-12 pl-12 pr-12 bg-white/10 rounded-full text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-white/20"
                autoFocus
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery('')}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
          </form>

          {/* Filter Tabs */}
          {debouncedQuery && (
            <div className="flex gap-2 mb-6">
              {filters.map((f) => (
                <button
                  key={f.value}
                  onClick={() => setFilter(f.value)}
                  className={cn(
                    'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-colors',
                    filter === f.value
                      ? 'bg-white text-black'
                      : 'bg-white/10 hover:bg-white/20'
                  )}
                >
                  {f.icon}
                  {f.label}
                </button>
              ))}
            </div>
          )}

          {/* Results or Browse */}
          {debouncedQuery ? (
            isLoading ? (
              <div className="flex items-center justify-center py-20">
                <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              </div>
            ) : hasResults ? (
              <div className="space-y-8">
                {/* Songs */}
                {(filter === 'all' || filter === 'songs') && results.songs.length > 0 && (
                  <section>
                    <h2 className="text-xl font-bold mb-4">Songs</h2>
                    <div className="bg-white/5 rounded-lg">
                      {results.songs.slice(0, filter === 'songs' ? 50 : 5).map((song, i) => (
                        <SongCard key={song.id} song={song} variant="row" index={i} />
                      ))}
                    </div>
                  </section>
                )}

                {/* Artists */}
                {(filter === 'all' || filter === 'artists') && results.artists.length > 0 && (
                  <section>
                    <h2 className="text-xl font-bold mb-4">Artists</h2>
                    <div className="grid grid-cols-6 gap-4">
                      {results.artists.slice(0, filter === 'artists' ? 30 : 6).map((artist) => (
                        <Link
                          key={artist.address}
                          href={`/artist/${artist.address}`}
                          className="group p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-center"
                        >
                          <div className="relative w-32 h-32 mx-auto rounded-full overflow-hidden mb-4">
                            {artist.avatar ? (
                              <Image
                                src={artist.avatar}
                                alt={artist.name}
                                fill
                                className="object-cover"
                              />
                            ) : (
                              <div className="w-full h-full bg-gradient-to-br from-purple-500 to-fuchsia-600 flex items-center justify-center">
                                <User className="w-12 h-12" />
                              </div>
                            )}
                          </div>
                          <h3 className="font-medium truncate">{artist.name}</h3>
                          <p className="text-sm text-muted-foreground">Artist</p>
                        </Link>
                      ))}
                    </div>
                  </section>
                )}

                {/* Playlists */}
                {(filter === 'all' || filter === 'playlists') && results.playlists.length > 0 && (
                  <section>
                    <h2 className="text-xl font-bold mb-4">Playlists</h2>
                    <div className="grid grid-cols-5 gap-4">
                      {results.playlists.slice(0, filter === 'playlists' ? 25 : 5).map((playlist) => (
                        <Link
                          key={playlist.id}
                          href={`/playlist/${playlist.id}`}
                          className="group p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                        >
                          <div className="relative aspect-square rounded-md overflow-hidden mb-4">
                            {playlist.coverImage ? (
                              <Image
                                src={playlist.coverImage}
                                alt={playlist.name}
                                fill
                                className="object-cover"
                              />
                            ) : (
                              <div className="w-full h-full bg-gradient-to-br from-purple-500 to-fuchsia-600 flex items-center justify-center">
                                <ListMusic className="w-12 h-12" />
                              </div>
                            )}
                          </div>
                          <h3 className="font-medium truncate">{playlist.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            By {playlist.ownerName} · {playlist.songCount} songs
                          </p>
                        </Link>
                      ))}
                    </div>
                  </section>
                )}
              </div>
            ) : (
              <div className="text-center py-20">
                <p className="text-xl text-muted-foreground">
                  No results found for "{debouncedQuery}"
                </p>
                <p className="text-sm text-muted-foreground mt-2">
                  Try searching for something else
                </p>
              </div>
            )
          ) : (
            /* Browse State */
            <div className="space-y-8">
              {/* Recent Searches */}
              {recentSearches.length > 0 && (
                <section>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold">Recent Searches</h2>
                    <button
                      onClick={clearRecentSearches}
                      className="text-sm text-muted-foreground hover:text-white"
                    >
                      Clear all
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {recentSearches.map((term) => (
                      <button
                        key={term}
                        onClick={() => setQuery(term)}
                        className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
                      >
                        <Clock className="w-4 h-4 text-muted-foreground" />
                        {term}
                      </button>
                    ))}
                  </div>
                </section>
              )}

              {/* Browse Categories */}
              <section>
                <h2 className="text-xl font-bold mb-4">Browse All</h2>
                <div className="grid grid-cols-5 gap-4">
                  {browseCategories.map((category) => (
                    <Link
                      key={category.name}
                      href={`/genre/${category.slug}`}
                      className="relative aspect-square rounded-lg overflow-hidden"
                      style={{ backgroundColor: category.color }}
                    >
                      <div className="absolute inset-0 bg-gradient-to-br from-transparent to-black/30" />
                      <h3 className="absolute bottom-4 left-4 text-xl font-bold">
                        {category.name}
                      </h3>
                    </Link>
                  ))}
                </div>
              </section>
            </div>
          )}
        </div>
      </main>

      <Player />
    </div>
  );
}

const browseCategories = [
  { name: 'Electronic', slug: 'electronic', color: '#8B5CF6' },
  { name: 'Hip Hop', slug: 'hip-hop', color: '#EC4899' },
  { name: 'Pop', slug: 'pop', color: '#3B82F6' },
  { name: 'Rock', slug: 'rock', color: '#EF4444' },
  { name: 'Jazz', slug: 'jazz', color: '#F59E0B' },
  { name: 'Classical', slug: 'classical', color: '#10B981' },
  { name: 'R&B', slug: 'rnb', color: '#6366F1' },
  { name: 'Country', slug: 'country', color: '#F97316' },
  { name: 'Ambient', slug: 'ambient', color: '#14B8A6' },
  { name: 'Lo-Fi', slug: 'lofi', color: '#A855F7' },
];
