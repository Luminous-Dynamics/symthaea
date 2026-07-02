// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Home Screen
 *
 * Personalized feed with contextual recommendations.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Pressable,
  Dimensions,
} from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import * as Location from 'expo-location';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useTheme } from '../../src/theme/ThemeContext';
import { useIdentityContext } from '../../src/providers/IdentityProvider';
import { usePlayer } from '../../src/components/Player/PlayerContext';
import { SectionHeader } from '../../src/components/SectionHeader';
import { TrackCard } from '../../src/components/TrackCard';
import { CircleCard } from '../../src/components/CircleCard';
import { ArtistCard } from '../../src/components/ArtistCard';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface WeatherContext {
  condition: string;
  temperature: number;
  timeOfDay: 'dawn' | 'morning' | 'afternoon' | 'evening' | 'night';
}

export default function HomeScreen() {
  const { colors, spacing } = useTheme();
  const { soul, isAuthenticated } = useIdentityContext();
  const { playTrack } = usePlayer();
  const router = useRouter();

  const [refreshing, setRefreshing] = useState(false);
  const [weatherContext, setWeatherContext] = useState<WeatherContext | null>(null);

  // Fetch weather context for recommendations
  useEffect(() => {
    async function fetchWeatherContext() {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') return;

        const location = await Location.getCurrentPositionAsync({});
        // In production, this would call a weather API
        const hour = new Date().getHours();
        const timeOfDay =
          hour < 6 ? 'night' :
          hour < 9 ? 'dawn' :
          hour < 12 ? 'morning' :
          hour < 17 ? 'afternoon' :
          hour < 20 ? 'evening' : 'night';

        setWeatherContext({
          condition: 'clear', // Would come from API
          temperature: 72,
          timeOfDay,
        });
      } catch (error) {
        console.log('Weather context unavailable');
      }
    }

    fetchWeatherContext();
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    // Refresh data here
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const getContextualMessage = () => {
    if (!weatherContext) return 'Discover something new';

    const messages: Record<string, string> = {
      dawn: 'Music for gentle awakenings',
      morning: 'Energize your morning',
      afternoon: 'Afternoon vibes',
      evening: 'Wind down with these sounds',
      night: 'Late night listening',
    };

    return messages[weatherContext.timeOfDay] || 'Curated for you';
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={colors.primary}
        />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={[styles.greeting, { color: colors.textMuted }]}>
            {getGreeting()}
          </Text>
          <Text style={[styles.name, { color: colors.text }]}>
            {soul?.profile.displayName || 'Explorer'}
          </Text>
        </View>
        <Pressable
          onPress={() => router.push('/tabs/profile')}
          style={({ pressed }) => [styles.avatar, { opacity: pressed ? 0.8 : 1 }]}
        >
          {soul?.profile.avatarUri ? (
            <Image
              source={{ uri: soul.profile.avatarUri }}
              style={styles.avatarImage}
              contentFit="cover"
            />
          ) : (
            <LinearGradient
              colors={[colors.primary, colors.secondary]}
              style={styles.avatarImage}
            />
          )}
        </Pressable>
      </View>

      {/* Contextual Hero */}
      <View style={styles.heroSection}>
        <LinearGradient
          colors={[colors.primary + '40', colors.background]}
          style={styles.heroGradient}
        >
          <Text style={[styles.heroTitle, { color: colors.text }]}>
            {getContextualMessage()}
          </Text>
          <Text style={[styles.heroSubtitle, { color: colors.textMuted }]}>
            Based on your time and location
          </Text>
        </LinearGradient>
      </View>

      {/* Active Circles */}
      <SectionHeader
        title="Live Circles"
        subtitle="Join others listening now"
        onSeeAll={() => router.push('/circles')}
      />
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.horizontalList}
      >
        {/* Mock circle data */}
        {[1, 2, 3].map((i) => (
          <CircleCard
            key={i}
            id={`circle-${i}`}
            name={`Evening Ambient Circle`}
            listenerCount={12 + i * 3}
            currentTrack="Floating Dreams"
            artworkUri="https://picsum.photos/seed/circle1/200"
            isLive
          />
        ))}
      </ScrollView>

      {/* Your Resonance */}
      {isAuthenticated && soul && (
        <>
          <SectionHeader
            title="Your Resonance"
            subtitle="Tracks that move you"
          />
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.horizontalList}
          >
            {[1, 2, 3, 4].map((i) => (
              <TrackCard
                key={i}
                id={`track-${i}`}
                title={`Resonant Track ${i}`}
                artist="Various Artists"
                artworkUri={`https://picsum.photos/seed/track${i}/200`}
                duration={240}
                onPlay={() => playTrack(`track-${i}`)}
              />
            ))}
          </ScrollView>
        </>
      )}

      {/* Discover Artists */}
      <SectionHeader
        title="Artists on the Rise"
        subtitle="Support emerging creators"
        onSeeAll={() => router.push('/discover/artists')}
      />
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.horizontalList}
      >
        {[1, 2, 3, 4, 5].map((i) => (
          <ArtistCard
            key={i}
            id={`artist-${i}`}
            name={`Artist ${i}`}
            avatarUri={`https://picsum.photos/seed/artist${i}/200`}
            patronCount={150 + i * 50}
          />
        ))}
      </ScrollView>

      {/* Recent Releases */}
      <SectionHeader
        title="Fresh Releases"
        subtitle="New music from your network"
        onSeeAll={() => router.push('/discover/releases')}
      />
      <View style={styles.releaseGrid}>
        {[1, 2, 3, 4].map((i) => (
          <Pressable
            key={i}
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              router.push(`/album/${i}`);
            }}
            style={({ pressed }) => [
              styles.releaseCard,
              { opacity: pressed ? 0.8 : 1 },
            ]}
          >
            <Image
              source={{ uri: `https://picsum.photos/seed/album${i}/300` }}
              style={styles.releaseArtwork}
              contentFit="cover"
            />
            <Text
              style={[styles.releaseTitle, { color: colors.text }]}
              numberOfLines={1}
            >
              Album Title {i}
            </Text>
            <Text
              style={[styles.releaseArtist, { color: colors.textMuted }]}
              numberOfLines={1}
            >
              Artist Name
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Bottom spacing for tab bar */}
      <View style={{ height: 150 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    paddingTop: 60,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  greeting: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Regular',
  },
  name: {
    fontSize: 24,
    fontFamily: 'SpaceGrotesk-Bold',
    marginTop: 2,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    overflow: 'hidden',
  },
  avatarImage: {
    width: '100%',
    height: '100%',
  },
  heroSection: {
    marginHorizontal: 20,
    marginBottom: 32,
    borderRadius: 16,
    overflow: 'hidden',
  },
  heroGradient: {
    padding: 24,
    borderRadius: 16,
  },
  heroTitle: {
    fontSize: 28,
    fontFamily: 'SpaceGrotesk-Bold',
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Regular',
  },
  horizontalList: {
    paddingHorizontal: 20,
    gap: 16,
  },
  releaseGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 20,
    gap: 16,
  },
  releaseCard: {
    width: (SCREEN_WIDTH - 56) / 2,
  },
  releaseArtwork: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 12,
    marginBottom: 8,
  },
  releaseTitle: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  releaseArtist: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
    marginTop: 2,
  },
});
