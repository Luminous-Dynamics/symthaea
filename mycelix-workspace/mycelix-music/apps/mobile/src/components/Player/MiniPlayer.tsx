// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mini Player Component
 *
 * Compact player shown above tab bar, expands to full screen.
 */

import { Pressable, View, Text, StyleSheet, Dimensions } from 'react-native';
import { Image } from 'expo-image';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { useTheme } from '../../theme/ThemeContext';
import { usePlayer } from './PlayerContext';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const MINI_PLAYER_HEIGHT = 64;
const TAB_BAR_HEIGHT = 85;

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function MiniPlayer() {
  const { colors } = useTheme();
  const {
    currentTrack,
    isPlaying,
    isLoading,
    position,
    duration,
    isExpanded,
    pause,
    resume,
    toggleExpanded,
    skipNext,
    skipPrevious,
    seekTo,
    toggleRepeat,
    toggleShuffle,
    repeatMode,
    shuffleEnabled,
  } = usePlayer();

  const translateY = useSharedValue(0);
  const expandProgress = useSharedValue(isExpanded ? 1 : 0);

  // Handle expansion animation
  const containerStyle = useAnimatedStyle(() => {
    const height = interpolate(
      expandProgress.value,
      [0, 1],
      [MINI_PLAYER_HEIGHT, SCREEN_HEIGHT],
      Extrapolation.CLAMP
    );

    const bottom = interpolate(
      expandProgress.value,
      [0, 1],
      [TAB_BAR_HEIGHT, 0],
      Extrapolation.CLAMP
    );

    return {
      height,
      bottom,
    };
  });

  const miniContentStyle = useAnimatedStyle(() => ({
    opacity: interpolate(expandProgress.value, [0, 0.3], [1, 0]),
    transform: [{ scale: interpolate(expandProgress.value, [0, 0.3], [1, 0.9]) }],
  }));

  const fullContentStyle = useAnimatedStyle(() => ({
    opacity: interpolate(expandProgress.value, [0.5, 1], [0, 1]),
    transform: [{ scale: interpolate(expandProgress.value, [0.5, 1], [0.95, 1]) }],
  }));

  // Swipe gesture to expand/collapse
  const gesture = Gesture.Pan()
    .onUpdate((event) => {
      if (!isExpanded && event.translationY < 0) {
        // Swiping up to expand
        expandProgress.value = Math.min(1, -event.translationY / SCREEN_HEIGHT);
      } else if (isExpanded && event.translationY > 0) {
        // Swiping down to collapse
        expandProgress.value = Math.max(0, 1 - event.translationY / SCREEN_HEIGHT);
      }
    })
    .onEnd((event) => {
      if (!isExpanded && event.translationY < -100) {
        expandProgress.value = withSpring(1, { damping: 20 });
        toggleExpanded();
      } else if (isExpanded && event.translationY > 100) {
        expandProgress.value = withSpring(0, { damping: 20 });
        toggleExpanded();
      } else {
        expandProgress.value = withSpring(isExpanded ? 1 : 0, { damping: 20 });
      }
    });

  if (!currentTrack) return null;

  const progress = duration > 0 ? position / duration : 0;

  return (
    <GestureDetector gesture={gesture}>
      <Animated.View style={[styles.container, containerStyle]}>
        <BlurView intensity={90} tint="dark" style={StyleSheet.absoluteFill} />

        {/* Progress bar (mini mode) */}
        <View style={styles.progressContainer}>
          <View style={[styles.progressBar, { backgroundColor: colors.border }]}>
            <View
              style={[
                styles.progressFill,
                { backgroundColor: colors.primary, width: `${progress * 100}%` },
              ]}
            />
          </View>
        </View>

        {/* Mini Player Content */}
        <Animated.View style={[styles.miniContent, miniContentStyle]}>
          <Pressable onPress={toggleExpanded} style={styles.miniTrackInfo}>
            <Image
              source={{ uri: currentTrack.artworkUri }}
              style={styles.miniArtwork}
              contentFit="cover"
            />
            <View style={styles.miniTextContainer}>
              <Text style={[styles.miniTitle, { color: colors.text }]} numberOfLines={1}>
                {currentTrack.title}
              </Text>
              <Text style={[styles.miniArtist, { color: colors.textMuted }]} numberOfLines={1}>
                {currentTrack.artist}
              </Text>
            </View>
          </Pressable>

          <View style={styles.miniControls}>
            <Pressable
              onPress={() => (isPlaying ? pause() : resume())}
              style={styles.playButton}
            >
              <Text style={[styles.playIcon, { color: colors.text }]}>
                {isLoading ? '⟳' : isPlaying ? '⏸' : '▶'}
              </Text>
            </Pressable>
            <Pressable onPress={skipNext} style={styles.skipButton}>
              <Text style={[styles.skipIcon, { color: colors.text }]}>⏭</Text>
            </Pressable>
          </View>
        </Animated.View>

        {/* Full Player Content */}
        <Animated.View style={[styles.fullContent, fullContentStyle]}>
          <View style={styles.fullHeader}>
            <Pressable onPress={toggleExpanded} style={styles.collapseHandle}>
              <View style={[styles.handleBar, { backgroundColor: colors.border }]} />
            </Pressable>
          </View>

          <View style={styles.fullArtworkContainer}>
            <Image
              source={{ uri: currentTrack.artworkUri }}
              style={styles.fullArtwork}
              contentFit="cover"
            />
          </View>

          <View style={styles.fullTrackInfo}>
            <Text style={[styles.fullTitle, { color: colors.text }]}>
              {currentTrack.title}
            </Text>
            <Text style={[styles.fullArtist, { color: colors.textMuted }]}>
              {currentTrack.artist}
            </Text>
          </View>

          {/* Full Progress Bar */}
          <View style={styles.fullProgressContainer}>
            <Pressable
              onPress={(e) => {
                const x = e.nativeEvent.locationX;
                const newPosition = (x / (SCREEN_WIDTH - 40)) * duration;
                seekTo(newPosition);
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              }}
              style={styles.fullProgressTrack}
            >
              <View style={[styles.fullProgressBar, { backgroundColor: colors.border }]}>
                <View
                  style={[
                    styles.fullProgressFill,
                    { backgroundColor: colors.primary, width: `${progress * 100}%` },
                  ]}
                />
                <View
                  style={[
                    styles.progressKnob,
                    { backgroundColor: colors.primary, left: `${progress * 100}%` },
                  ]}
                />
              </View>
            </Pressable>
            <View style={styles.timeContainer}>
              <Text style={[styles.timeText, { color: colors.textMuted }]}>
                {formatTime(position)}
              </Text>
              <Text style={[styles.timeText, { color: colors.textMuted }]}>
                {formatTime(duration)}
              </Text>
            </View>
          </View>

          {/* Full Controls */}
          <View style={styles.fullControls}>
            <Pressable
              onPress={toggleShuffle}
              style={styles.secondaryButton}
            >
              <Text
                style={[
                  styles.secondaryIcon,
                  { color: shuffleEnabled ? colors.primary : colors.textMuted },
                ]}
              >
                ⤮
              </Text>
            </Pressable>

            <Pressable onPress={skipPrevious} style={styles.controlButton}>
              <Text style={[styles.controlIcon, { color: colors.text }]}>⏮</Text>
            </Pressable>

            <Pressable
              onPress={() => (isPlaying ? pause() : resume())}
              style={[styles.mainPlayButton, { backgroundColor: colors.primary }]}
            >
              <Text style={styles.mainPlayIcon}>
                {isLoading ? '⟳' : isPlaying ? '⏸' : '▶'}
              </Text>
            </Pressable>

            <Pressable onPress={skipNext} style={styles.controlButton}>
              <Text style={[styles.controlIcon, { color: colors.text }]}>⏭</Text>
            </Pressable>

            <Pressable
              onPress={toggleRepeat}
              style={styles.secondaryButton}
            >
              <Text
                style={[
                  styles.secondaryIcon,
                  { color: repeatMode !== 'off' ? colors.primary : colors.textMuted },
                ]}
              >
                {repeatMode === 'one' ? '🔂' : '🔁'}
              </Text>
            </Pressable>
          </View>

          {/* Additional actions */}
          <View style={styles.additionalActions}>
            <Pressable style={styles.actionButton}>
              <Text style={[styles.actionIcon, { color: colors.textMuted }]}>♡</Text>
            </Pressable>
            <Pressable style={styles.actionButton}>
              <Text style={[styles.actionIcon, { color: colors.textMuted }]}>⋮</Text>
            </Pressable>
          </View>
        </Animated.View>
      </Animated.View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    left: 0,
    right: 0,
    overflow: 'hidden',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  progressContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
  },
  progressBar: {
    height: '100%',
  },
  progressFill: {
    height: '100%',
  },
  miniContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    paddingTop: 10,
  },
  miniTrackInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  miniArtwork: {
    width: 44,
    height: 44,
    borderRadius: 8,
  },
  miniTextContainer: {
    flex: 1,
  },
  miniTitle: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  miniArtist: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
    marginTop: 2,
  },
  miniControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  playButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  playIcon: {
    fontSize: 20,
  },
  skipButton: {
    width: 36,
    height: 36,
    justifyContent: 'center',
    alignItems: 'center',
  },
  skipIcon: {
    fontSize: 16,
  },
  fullContent: {
    flex: 1,
    paddingHorizontal: 20,
  },
  fullHeader: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 20,
  },
  collapseHandle: {
    padding: 10,
  },
  handleBar: {
    width: 40,
    height: 4,
    borderRadius: 2,
  },
  fullArtworkContainer: {
    alignItems: 'center',
    marginBottom: 32,
  },
  fullArtwork: {
    width: SCREEN_WIDTH - 80,
    height: SCREEN_WIDTH - 80,
    borderRadius: 16,
  },
  fullTrackInfo: {
    alignItems: 'center',
    marginBottom: 24,
  },
  fullTitle: {
    fontSize: 22,
    fontFamily: 'SpaceGrotesk-Bold',
    textAlign: 'center',
  },
  fullArtist: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
    marginTop: 8,
  },
  fullProgressContainer: {
    marginBottom: 24,
  },
  fullProgressTrack: {
    paddingVertical: 10,
  },
  fullProgressBar: {
    height: 4,
    borderRadius: 2,
    position: 'relative',
  },
  fullProgressFill: {
    height: '100%',
    borderRadius: 2,
  },
  progressKnob: {
    position: 'absolute',
    top: -4,
    width: 12,
    height: 12,
    borderRadius: 6,
    marginLeft: -6,
  },
  timeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  timeText: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
  },
  fullControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 20,
    marginBottom: 32,
  },
  secondaryButton: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  secondaryIcon: {
    fontSize: 22,
  },
  controlButton: {
    width: 56,
    height: 56,
    justifyContent: 'center',
    alignItems: 'center',
  },
  controlIcon: {
    fontSize: 28,
  },
  mainPlayButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: 'center',
    alignItems: 'center',
  },
  mainPlayIcon: {
    fontSize: 32,
    color: '#0a0a0a',
  },
  additionalActions: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 40,
  },
  actionButton: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  actionIcon: {
    fontSize: 24,
  },
});
