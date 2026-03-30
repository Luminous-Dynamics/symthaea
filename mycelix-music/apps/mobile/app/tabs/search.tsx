// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Search Screen
 *
 * Search for tracks, artists, albums, and circles.
 */

import { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  ScrollView,
  StyleSheet,
  Pressable,
  Keyboard,
  Dimensions,
} from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
} from 'react-native-reanimated';
import { useTheme } from '../../src/theme/ThemeContext';
import { usePlayer } from '../../src/components/Player/PlayerContext';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface GenreCardProps {
  name: string;
  color: string;
  imageUri: string;
  onPress: () => void;
}

function GenreCard({ name, color, imageUri, onPress }: GenreCardProps) {
  return (
    <Pressable
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
      style={({ pressed }) => [
        styles.genreCard,
        { opacity: pressed ? 0.8 : 1 },
      ]}
    >
      <LinearGradient
        colors={[color, color + '80']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.genreGradient}
      >
        <Text style={styles.genreName}>{name}</Text>
        <Image
          source={{ uri: imageUri }}
          style={styles.genreImage}
          contentFit="cover"
        />
      </LinearGradient>
    </Pressable>
  );
}

interface SearchResultProps {
  type: 'track' | 'artist' | 'album' | 'circle';
  title: string;
  subtitle: string;
  imageUri: string;
  onPress: () => void;
}

function SearchResult({ type, title, subtitle, imageUri, onPress }: SearchResultProps) {
  const { colors } = useTheme();

  return (
    <Pressable
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
      style={({ pressed }) => [
        styles.searchResult,
        { opacity: pressed ? 0.8 : 1 },
      ]}
    >
      <Image
        source={{ uri: imageUri }}
        style={[
          styles.resultImage,
          type === 'artist' && styles.roundImage,
        ]}
        contentFit="cover"
      />
      <View style={styles.resultInfo}>
        <Text style={[styles.resultTitle, { color: colors.text }]} numberOfLines={1}>
          {title}
        </Text>
        <View style={styles.resultMeta}>
          <Text style={[styles.resultType, { color: colors.primary }]}>
            {type.charAt(0).toUpperCase() + type.slice(1)}
          </Text>
          <Text style={[styles.resultSubtitle, { color: colors.textMuted }]}>
            {subtitle}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

export default function SearchScreen() {
  const { colors } = useTheme();
  const { playTrack } = usePlayer();
  const router = useRouter();
  const inputRef = useRef<TextInput>(null);

  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [recentSearches] = useState(['ambient', 'jazz fusion', 'electronic']);

  const searchBarScale = useSharedValue(1);

  const searchBarStyle = useAnimatedStyle(() => ({
    transform: [{ scale: searchBarScale.value }],
  }));

  const handleFocus = useCallback(() => {
    setIsFocused(true);
    searchBarScale.value = withTiming(1.02, { duration: 200 });
  }, []);

  const handleBlur = useCallback(() => {
    setIsFocused(false);
    searchBarScale.value = withTiming(1, { duration: 200 });
  }, []);

  const handleSearch = useCallback((text: string) => {
    setQuery(text);
    // Implement search logic here
  }, []);

  const clearSearch = useCallback(() => {
    setQuery('');
    inputRef.current?.focus();
  }, []);

  const genres = [
    { name: 'Ambient', color: '#4A90D9', imageUri: 'https://picsum.photos/seed/ambient/200' },
    { name: 'Electronic', color: '#9B59B6', imageUri: 'https://picsum.photos/seed/electronic/200' },
    { name: 'Jazz', color: '#E67E22', imageUri: 'https://picsum.photos/seed/jazz/200' },
    { name: 'Classical', color: '#2ECC71', imageUri: 'https://picsum.photos/seed/classical/200' },
    { name: 'World', color: '#E74C3C', imageUri: 'https://picsum.photos/seed/world/200' },
    { name: 'Experimental', color: '#1ABC9C', imageUri: 'https://picsum.photos/seed/experimental/200' },
  ];

  // Mock search results
  const searchResults = query.length > 0 ? [
    { type: 'track' as const, title: `${query} Dreams`, subtitle: 'Various Artists', imageUri: 'https://picsum.photos/seed/search1/200' },
    { type: 'artist' as const, title: `The ${query} Collective`, subtitle: '1.2K patrons', imageUri: 'https://picsum.photos/seed/search2/200' },
    { type: 'album' as const, title: `${query} Sessions`, subtitle: 'Artist Name', imageUri: 'https://picsum.photos/seed/search3/200' },
    { type: 'circle' as const, title: `${query} Listening Circle`, subtitle: '24 listening now', imageUri: 'https://picsum.photos/seed/search4/200' },
  ] : [];

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {/* Search Bar */}
      <View style={styles.searchBarContainer}>
        <Animated.View style={[styles.searchBar, searchBarStyle, { backgroundColor: colors.border + '40' }]}>
          <Text style={styles.searchIcon}>⌕</Text>
          <TextInput
            ref={inputRef}
            style={[styles.searchInput, { color: colors.text }]}
            placeholder="Search music, artists, circles..."
            placeholderTextColor={colors.textMuted}
            value={query}
            onChangeText={handleSearch}
            onFocus={handleFocus}
            onBlur={handleBlur}
            returnKeyType="search"
            autoCapitalize="none"
            autoCorrect={false}
          />
          {query.length > 0 && (
            <Pressable onPress={clearSearch} style={styles.clearButton}>
              <Text style={[styles.clearIcon, { color: colors.textMuted }]}>✕</Text>
            </Pressable>
          )}
        </Animated.View>
        {isFocused && (
          <Pressable
            onPress={() => {
              Keyboard.dismiss();
              setQuery('');
            }}
            style={styles.cancelButton}
          >
            <Text style={[styles.cancelText, { color: colors.primary }]}>Cancel</Text>
          </Pressable>
        )}
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {query.length === 0 ? (
          <>
            {/* Recent Searches */}
            {recentSearches.length > 0 && (
              <View style={styles.section}>
                <Text style={[styles.sectionTitle, { color: colors.text }]}>Recent Searches</Text>
                <View style={styles.recentList}>
                  {recentSearches.map((search, index) => (
                    <Pressable
                      key={index}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setQuery(search);
                      }}
                      style={[styles.recentItem, { backgroundColor: colors.border + '30' }]}
                    >
                      <Text style={[styles.recentText, { color: colors.text }]}>{search}</Text>
                    </Pressable>
                  ))}
                </View>
              </View>
            )}

            {/* Browse Genres */}
            <View style={styles.section}>
              <Text style={[styles.sectionTitle, { color: colors.text }]}>Browse by Genre</Text>
              <View style={styles.genreGrid}>
                {genres.map((genre, index) => (
                  <GenreCard
                    key={index}
                    name={genre.name}
                    color={genre.color}
                    imageUri={genre.imageUri}
                    onPress={() => router.push(`/genre/${genre.name.toLowerCase()}`)}
                  />
                ))}
              </View>
            </View>

            {/* Trending Searches */}
            <View style={styles.section}>
              <Text style={[styles.sectionTitle, { color: colors.text }]}>Trending</Text>
              {['collaborative ambient', 'live session recordings', 'meditation music'].map((trend, index) => (
                <Pressable
                  key={index}
                  onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    setQuery(trend);
                  }}
                  style={({ pressed }) => [
                    styles.trendingItem,
                    { opacity: pressed ? 0.7 : 1 },
                  ]}
                >
                  <Text style={[styles.trendingNumber, { color: colors.primary }]}>{index + 1}</Text>
                  <Text style={[styles.trendingText, { color: colors.text }]}>{trend}</Text>
                </Pressable>
              ))}
            </View>
          </>
        ) : (
          /* Search Results */
          <View style={styles.resultsContainer}>
            {searchResults.map((result, index) => (
              <SearchResult
                key={index}
                type={result.type}
                title={result.title}
                subtitle={result.subtitle}
                imageUri={result.imageUri}
                onPress={() => {
                  if (result.type === 'track') {
                    playTrack('mock-id');
                  } else {
                    router.push(`/${result.type}/mock-id`);
                  }
                }}
              />
            ))}
          </View>
        )}

        <View style={{ height: 150 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 60,
  },
  searchBarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 20,
    gap: 12,
  },
  searchBar: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 12,
    paddingHorizontal: 16,
    height: 48,
  },
  searchIcon: {
    fontSize: 20,
    marginRight: 12,
    color: '#888',
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Regular',
  },
  clearButton: {
    padding: 4,
  },
  clearIcon: {
    fontSize: 16,
  },
  cancelButton: {
    paddingVertical: 8,
  },
  cancelText: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    paddingHorizontal: 20,
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontFamily: 'SpaceGrotesk-Bold',
    marginBottom: 16,
  },
  recentList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  recentItem: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  recentText: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  genreGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  genreCard: {
    width: (SCREEN_WIDTH - 52) / 2,
    height: 100,
    borderRadius: 12,
    overflow: 'hidden',
  },
  genreGradient: {
    flex: 1,
    padding: 16,
    justifyContent: 'flex-end',
  },
  genreName: {
    color: '#fff',
    fontSize: 18,
    fontFamily: 'SpaceGrotesk-Bold',
    zIndex: 1,
  },
  genreImage: {
    position: 'absolute',
    right: -20,
    bottom: -20,
    width: 80,
    height: 80,
    borderRadius: 8,
    transform: [{ rotate: '15deg' }],
    opacity: 0.4,
  },
  trendingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    gap: 16,
  },
  trendingNumber: {
    fontSize: 18,
    fontFamily: 'SpaceGrotesk-Bold',
    width: 24,
  },
  trendingText: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  resultsContainer: {
    gap: 12,
  },
  searchResult: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  resultImage: {
    width: 56,
    height: 56,
    borderRadius: 8,
  },
  roundImage: {
    borderRadius: 28,
  },
  resultInfo: {
    flex: 1,
  },
  resultTitle: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  resultMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 4,
  },
  resultType: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  resultSubtitle: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
  },
});
