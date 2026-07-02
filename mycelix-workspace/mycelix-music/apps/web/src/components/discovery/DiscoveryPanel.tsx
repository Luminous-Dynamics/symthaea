// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Discovery Panel Component
 *
 * Smart music discovery UI with mood/genre exploration and similar tracks.
 */

import { useCallback, useState } from 'react';
import {
  useDiscovery,
  MOOD_LABELS,
  GENRE_LABELS,
  type MoodLabel,
  type GenreLabel,
  type SimilarityResult,
} from '@/hooks/useDiscovery';

// === Types ===

type DiscoveryMode = 'similar' | 'mood' | 'genre' | 'radio';

interface DiscoveryPanelProps {
  currentTrackId?: string;
  onTrackSelect?: (trackId: string) => void;
  onCreatePlaylist?: (tracks: SimilarityResult[]) => void;
}

// === Mood Icons ===

const MOOD_ICONS: Record<MoodLabel, string> = {
  happy: '\u{1F60A}',
  sad: '\u{1F622}',
  energetic: '\u{26A1}',
  calm: '\u{1F33F}',
  aggressive: '\u{1F525}',
  romantic: '\u{2764}',
  melancholic: '\u{1F319}',
  uplifting: '\u{2600}',
  dark: '\u{1F311}',
  dreamy: '\u{2601}',
};

// === Genre Icons ===

const GENRE_ICONS: Record<GenreLabel, string> = {
  electronic: '\u{1F3B9}',
  rock: '\u{1F3B8}',
  pop: '\u{1F3A4}',
  'hip-hop': '\u{1F399}',
  jazz: '\u{1F3B7}',
  classical: '\u{1F3BB}',
  folk: '\u{1FA95}',
  'r&b': '\u{1F3B6}',
  metal: '\u{1F918}',
  ambient: '\u{1F30C}',
  indie: '\u{1F3B5}',
  punk: '\u{1F47B}',
  blues: '\u{1F3B9}',
  country: '\u{1F3A0}',
  reggae: '\u{1F334}',
};

// === Component ===

export function DiscoveryPanel({
  currentTrackId,
  onTrackSelect,
  onCreatePlaylist,
}: DiscoveryPanelProps) {
  const discovery = useDiscovery();
  const [mode, setMode] = useState<DiscoveryMode>('similar');
  const [selectedMoods, setSelectedMoods] = useState<Set<MoodLabel>>(new Set());
  const [selectedGenres, setSelectedGenres] = useState<Set<GenreLabel>>(new Set());
  const [seedTracks, setSeedTracks] = useState<string[]>([]);

  // Find similar to current track
  const handleFindSimilar = useCallback(async () => {
    if (currentTrackId) {
      await discovery.findSimilar([currentTrackId]);
    } else if (seedTracks.length > 0) {
      await discovery.findSimilar(seedTracks);
    }
  }, [currentTrackId, seedTracks, discovery]);

  // Mood selection
  const toggleMood = useCallback((mood: MoodLabel) => {
    setSelectedMoods(prev => {
      const next = new Set(prev);
      if (next.has(mood)) {
        next.delete(mood);
      } else {
        next.add(mood);
      }
      return next;
    });
  }, []);

  // Genre selection
  const toggleGenre = useCallback((genre: GenreLabel) => {
    setSelectedGenres(prev => {
      const next = new Set(prev);
      if (next.has(genre)) {
        next.delete(genre);
      } else {
        next.add(genre);
      }
      return next;
    });
  }, []);

  // Create mood playlist
  const handleMoodPlaylist = useCallback(async () => {
    if (selectedMoods.size > 0) {
      await discovery.createMoodPlaylist([...selectedMoods]);
    }
  }, [selectedMoods, discovery]);

  // Create genre playlist
  const handleGenrePlaylist = useCallback(async () => {
    if (selectedGenres.size > 0) {
      await discovery.createGenrePlaylist([...selectedGenres]);
    }
  }, [selectedGenres, discovery]);

  // Create radio station
  const handleRadioStation = useCallback(async () => {
    const seeds = currentTrackId ? [currentTrackId, ...seedTracks] : seedTracks;
    if (seeds.length > 0) {
      await discovery.createRadioStation(seeds, { limit: 50 });
    }
  }, [currentTrackId, seedTracks, discovery]);

  // Save as playlist
  const handleSavePlaylist = useCallback(() => {
    if (discovery.results.length > 0 && onCreatePlaylist) {
      onCreatePlaylist(discovery.results);
    }
  }, [discovery.results, onCreatePlaylist]);

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h2 style={styles.title}>Discover</h2>
        <div style={styles.indexStatus}>
          <span style={styles.indexDot} />
          <span style={styles.indexText}>
            {discovery.indexSize} tracks indexed
          </span>
        </div>
      </div>

      {/* Mode Tabs */}
      <div style={styles.tabs}>
        {[
          { id: 'similar' as const, label: 'Similar', icon: '\u{1F50D}' },
          { id: 'mood' as const, label: 'Mood', icon: '\u{1F3A8}' },
          { id: 'genre' as const, label: 'Genre', icon: '\u{1F3B5}' },
          { id: 'radio' as const, label: 'Radio', icon: '\u{1F4FB}' },
        ].map(tab => (
          <button
            key={tab.id}
            style={{
              ...styles.tab,
              ...(mode === tab.id && styles.tabActive),
            }}
            onClick={() => setMode(tab.id)}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={styles.content}>
        {/* Similar Mode */}
        {mode === 'similar' && (
          <div style={styles.modeContent}>
            <p style={styles.modeDescription}>
              Find tracks similar to your current selection
            </p>
            {currentTrackId ? (
              <div style={styles.seedDisplay}>
                <span style={styles.seedLabel}>Seed track:</span>
                <span style={styles.seedTrack}>{currentTrackId}</span>
              </div>
            ) : (
              <p style={styles.hint}>Play a track to find similar music</p>
            )}
            <button
              style={styles.actionButton}
              onClick={handleFindSimilar}
              disabled={!currentTrackId && seedTracks.length === 0}
            >
              Find Similar Tracks
            </button>
          </div>
        )}

        {/* Mood Mode */}
        {mode === 'mood' && (
          <div style={styles.modeContent}>
            <p style={styles.modeDescription}>
              Select moods to create a playlist
            </p>
            <div style={styles.chipGrid}>
              {MOOD_LABELS.map(mood => (
                <button
                  key={mood}
                  style={{
                    ...styles.chip,
                    ...(selectedMoods.has(mood) && styles.chipSelected),
                  }}
                  onClick={() => toggleMood(mood)}
                >
                  <span>{MOOD_ICONS[mood]}</span>
                  <span>{mood}</span>
                </button>
              ))}
            </div>
            <button
              style={styles.actionButton}
              onClick={handleMoodPlaylist}
              disabled={selectedMoods.size === 0}
            >
              Create Mood Playlist
            </button>
          </div>
        )}

        {/* Genre Mode */}
        {mode === 'genre' && (
          <div style={styles.modeContent}>
            <p style={styles.modeDescription}>
              Explore by genre
            </p>
            <div style={styles.chipGrid}>
              {GENRE_LABELS.map(genre => (
                <button
                  key={genre}
                  style={{
                    ...styles.chip,
                    ...(selectedGenres.has(genre) && styles.chipSelected),
                  }}
                  onClick={() => toggleGenre(genre)}
                >
                  <span>{GENRE_ICONS[genre]}</span>
                  <span>{genre}</span>
                </button>
              ))}
            </div>
            <button
              style={styles.actionButton}
              onClick={handleGenrePlaylist}
              disabled={selectedGenres.size === 0}
            >
              Create Genre Playlist
            </button>
          </div>
        )}

        {/* Radio Mode */}
        {mode === 'radio' && (
          <div style={styles.modeContent}>
            <p style={styles.modeDescription}>
              Create an endless radio station based on your favorites
            </p>
            {currentTrackId && (
              <div style={styles.seedDisplay}>
                <span style={styles.seedLabel}>Starting from:</span>
                <span style={styles.seedTrack}>{currentTrackId}</span>
              </div>
            )}
            <button
              style={styles.actionButton}
              onClick={handleRadioStation}
              disabled={!currentTrackId && seedTracks.length === 0}
            >
              Start Radio Station
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      {discovery.results.length > 0 && (
        <div style={styles.results}>
          <div style={styles.resultsHeader}>
            <h3 style={styles.resultsTitle}>
              Results ({discovery.results.length})
            </h3>
            <button style={styles.saveButton} onClick={handleSavePlaylist}>
              Save as Playlist
            </button>
          </div>
          <div style={styles.resultsList}>
            {discovery.results.slice(0, 10).map((result, idx) => (
              <div
                key={result.trackId}
                style={styles.resultItem}
                onClick={() => onTrackSelect?.(result.trackId)}
              >
                <span style={styles.resultRank}>#{idx + 1}</span>
                <div style={styles.resultInfo}>
                  <span style={styles.resultTrack}>
                    {result.track?.title || result.trackId}
                  </span>
                  {result.track?.artist && (
                    <span style={styles.resultArtist}>
                      {result.track.artist}
                    </span>
                  )}
                </div>
                <span style={styles.resultScore}>
                  {Math.round(result.score * 100)}%
                </span>
              </div>
            ))}
            {discovery.results.length > 10 && (
              <div style={styles.moreResults}>
                +{discovery.results.length - 10} more tracks
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading State */}
      {discovery.isSearching && (
        <div style={styles.loading}>
          <div style={styles.spinner} />
          <span>Searching...</span>
        </div>
      )}

      {/* Error State */}
      {discovery.error && (
        <div style={styles.error}>
          {discovery.error}
        </div>
      )}
    </div>
  );
}

// === Styles ===

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: '12px',
    padding: '16px',
    width: '100%',
    maxWidth: '450px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    color: '#ffffff',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  title: {
    fontSize: '18px',
    fontWeight: 600,
    margin: 0,
  },
  indexStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  indexDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#22c55e',
  },
  indexText: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  tabs: {
    display: 'flex',
    gap: '4px',
    marginBottom: '16px',
  },
  tab: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
    padding: '10px 8px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid transparent',
    borderRadius: '8px',
    color: '#9ca3af',
    cursor: 'pointer',
    fontSize: '12px',
    transition: 'all 0.2s',
  },
  tabActive: {
    backgroundColor: 'rgba(129, 140, 248, 0.1)',
    borderColor: '#818cf8',
    color: '#818cf8',
  },
  content: {
    marginBottom: '16px',
  },
  modeContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  modeDescription: {
    fontSize: '14px',
    color: '#9ca3af',
    margin: 0,
  },
  hint: {
    fontSize: '13px',
    color: '#6b7280',
    fontStyle: 'italic',
    margin: 0,
  },
  seedDisplay: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    backgroundColor: 'rgba(129, 140, 248, 0.1)',
    borderRadius: '6px',
  },
  seedLabel: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  seedTrack: {
    fontSize: '13px',
    color: '#818cf8',
    fontWeight: 500,
  },
  chipGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
  },
  chip: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 12px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '16px',
    color: '#9ca3af',
    cursor: 'pointer',
    fontSize: '12px',
    transition: 'all 0.2s',
  },
  chipSelected: {
    backgroundColor: 'rgba(129, 140, 248, 0.2)',
    borderColor: '#818cf8',
    color: '#ffffff',
  },
  actionButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#818cf8',
    border: 'none',
    borderRadius: '8px',
    color: '#ffffff',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  results: {
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
    paddingTop: '16px',
  },
  resultsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  resultsTitle: {
    fontSize: '14px',
    fontWeight: 500,
    margin: 0,
  },
  saveButton: {
    padding: '6px 12px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '6px',
    color: '#ffffff',
    fontSize: '12px',
    cursor: 'pointer',
  },
  resultsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    maxHeight: '300px',
    overflowY: 'auto',
  },
  resultItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '8px 12px',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  resultRank: {
    fontSize: '12px',
    color: '#6b7280',
    width: '24px',
  },
  resultInfo: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    overflow: 'hidden',
  },
  resultTrack: {
    fontSize: '13px',
    fontWeight: 500,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  resultArtist: {
    fontSize: '11px',
    color: '#9ca3af',
  },
  resultScore: {
    fontSize: '12px',
    color: '#818cf8',
    fontWeight: 500,
  },
  moreResults: {
    textAlign: 'center',
    fontSize: '12px',
    color: '#6b7280',
    padding: '8px',
  },
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    padding: '24px',
    color: '#9ca3af',
  },
  spinner: {
    width: '20px',
    height: '20px',
    border: '2px solid rgba(255, 255, 255, 0.1)',
    borderTopColor: '#818cf8',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  error: {
    padding: '12px',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: '6px',
    color: '#ef4444',
    fontSize: '13px',
    textAlign: 'center',
  },
};

export default DiscoveryPanel;
