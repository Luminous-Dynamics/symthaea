// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Circle Card Component
 *
 * Displays a listening circle with live status.
 */

import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';
import { useEffect } from 'react';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/ThemeContext';

const CARD_WIDTH = 200;

interface CircleCardProps {
  id: string;
  name: string;
  listenerCount: number;
  currentTrack?: string;
  artworkUri: string;
  isLive?: boolean;
}

export function CircleCard({
  id,
  name,
  listenerCount,
  currentTrack,
  artworkUri,
  isLive = false,
}: CircleCardProps) {
  const { colors } = useTheme();
  const router = useRouter();
  const scale = useSharedValue(1);
  const pulseOpacity = useSharedValue(1);

  // Pulse animation for live indicator
  useEffect(() => {
    if (isLive) {
      pulseOpacity.value = withRepeat(
        withTiming(0.4, { duration: 1000 }),
        -1,
        true
      );
    }
  }, [isLive]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const pulseStyle = useAnimatedStyle(() => ({
    opacity: pulseOpacity.value,
  }));

  const handlePress = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    router.push(`/circle/${id}`);
  };

  return (
    <Pressable
      onPressIn={() => {
        scale.value = withSpring(0.95);
      }}
      onPressOut={() => {
        scale.value = withSpring(1);
      }}
      onPress={handlePress}
    >
      <Animated.View style={[styles.container, animatedStyle]}>
        <Image
          source={{ uri: artworkUri }}
          style={styles.backgroundImage}
          contentFit="cover"
          blurRadius={20}
        />
        <LinearGradient
          colors={['rgba(0,0,0,0.3)', 'rgba(0,0,0,0.8)']}
          style={styles.gradient}
        >
          {/* Live indicator */}
          {isLive && (
            <View style={styles.liveContainer}>
              <Animated.View
                style={[styles.livePulse, { backgroundColor: colors.error }, pulseStyle]}
              />
              <View style={[styles.liveDot, { backgroundColor: colors.error }]} />
              <Text style={styles.liveText}>LIVE</Text>
            </View>
          )}

          {/* Content */}
          <View style={styles.content}>
            <Text style={styles.name} numberOfLines={2}>
              {name}
            </Text>

            {currentTrack && (
              <View style={styles.nowPlaying}>
                <Text style={styles.nowPlayingLabel}>Now playing</Text>
                <Text style={styles.trackName} numberOfLines={1}>
                  {currentTrack}
                </Text>
              </View>
            )}

            <View style={styles.footer}>
              <View style={styles.listeners}>
                <View style={styles.avatarStack}>
                  {[1, 2, 3].map((i) => (
                    <View
                      key={i}
                      style={[
                        styles.miniAvatar,
                        { marginLeft: i > 1 ? -8 : 0, backgroundColor: colors.primary },
                      ]}
                    />
                  ))}
                </View>
                <Text style={styles.listenerCount}>
                  {listenerCount} listening
                </Text>
              </View>

              <View style={[styles.joinButton, { backgroundColor: colors.primary }]}>
                <Text style={styles.joinText}>Join</Text>
              </View>
            </View>
          </View>
        </LinearGradient>
      </Animated.View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    width: CARD_WIDTH,
    height: 180,
    borderRadius: 16,
    overflow: 'hidden',
  },
  backgroundImage: {
    ...StyleSheet.absoluteFillObject,
  },
  gradient: {
    flex: 1,
    padding: 14,
    justifyContent: 'space-between',
  },
  liveContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
  },
  livePulse: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderRadius: 10,
    left: -4,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  liveText: {
    color: '#fff',
    fontSize: 11,
    fontFamily: 'SpaceGrotesk-Bold',
    letterSpacing: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  name: {
    color: '#fff',
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Bold',
    marginBottom: 8,
  },
  nowPlaying: {
    marginBottom: 12,
  },
  nowPlayingLabel: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 10,
    fontFamily: 'SpaceGrotesk-Medium',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  trackName: {
    color: '#fff',
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
    marginTop: 2,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  listeners: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  avatarStack: {
    flexDirection: 'row',
  },
  miniAvatar: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#0a0a0a',
  },
  listenerCount: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 11,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  joinButton: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 14,
  },
  joinText: {
    color: '#0a0a0a',
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Bold',
  },
});
