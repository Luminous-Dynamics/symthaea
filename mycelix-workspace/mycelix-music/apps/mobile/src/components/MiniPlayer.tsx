// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mini Player Component
 *
 * Compact player bar shown at bottom of screens.
 * Expandable to full player view with gesture.
 */

import React, { useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  Dimensions,
  Pressable,
} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  useAnimatedGestureHandler,
  withSpring,
  interpolate,
  runOnJS,
} from 'react-native-reanimated';
import { PanGestureHandler, PanGestureHandlerGestureEvent } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';

import { usePlayerStore } from '../store/playerStore';
import { PlayIcon, PauseIcon, SkipForwardIcon, HeartIcon } from './icons';
import { colors, spacing, typography } from '../theme';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const MINI_PLAYER_HEIGHT = 64;
const FULL_PLAYER_HEIGHT = SCREEN_HEIGHT;
const SNAP_THRESHOLD = 100;

interface MiniPlayerProps {
  onExpand?: () => void;
}

export const MiniPlayer: React.FC<MiniPlayerProps> = ({ onExpand }) => {
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();

  const {
    currentSong,
    isPlaying,
    position,
    duration,
    pause,
    resume,
    next,
  } = usePlayerStore();

  const translateY = useSharedValue(0);
  const isExpanded = useSharedValue(false);

  // Progress percentage
  const progress = useMemo(() => {
    if (duration === 0) return 0;
    return (position / duration) * 100;
  }, [position, duration]);

  // Gesture handler
  const gestureHandler = useAnimatedGestureHandler<
    PanGestureHandlerGestureEvent,
    { startY: number }
  >({
    onStart: (_, ctx) => {
      ctx.startY = translateY.value;
    },
    onActive: (event, ctx) => {
      translateY.value = Math.max(
        -(FULL_PLAYER_HEIGHT - MINI_PLAYER_HEIGHT),
        Math.min(0, ctx.startY + event.translationY)
      );
    },
    onEnd: (event) => {
      if (event.velocityY < -500 || translateY.value < -SNAP_THRESHOLD) {
        translateY.value = withSpring(-(FULL_PLAYER_HEIGHT - MINI_PLAYER_HEIGHT));
        isExpanded.value = true;
        if (onExpand) {
          runOnJS(onExpand)();
        }
      } else {
        translateY.value = withSpring(0);
        isExpanded.value = false;
      }
    },
  });

  // Animated styles
  const animatedContainerStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  const animatedMiniStyle = useAnimatedStyle(() => ({
    opacity: interpolate(
      translateY.value,
      [0, -(FULL_PLAYER_HEIGHT - MINI_PLAYER_HEIGHT) / 2],
      [1, 0]
    ),
  }));

  const animatedFullStyle = useAnimatedStyle(() => ({
    opacity: interpolate(
      translateY.value,
      [-(FULL_PLAYER_HEIGHT - MINI_PLAYER_HEIGHT) / 2, -(FULL_PLAYER_HEIGHT - MINI_PLAYER_HEIGHT)],
      [0, 1]
    ),
  }));

  // Handlers
  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      resume();
    }
  }, [isPlaying, pause, resume]);

  const handleNext = useCallback(() => {
    next();
  }, [next]);

  const handlePress = useCallback(() => {
    navigation.navigate('Player' as never);
  }, [navigation]);

  // Don't render if no song
  if (!currentSong) {
    return null;
  }

  return (
    <PanGestureHandler onGestureEvent={gestureHandler}>
      <Animated.View
        style={[
          styles.container,
          { paddingBottom: insets.bottom },
          animatedContainerStyle,
        ]}
      >
        {/* Mini Player */}
        <Animated.View style={[styles.miniPlayer, animatedMiniStyle]}>
          {/* Progress bar */}
          <View style={styles.progressContainer}>
            <View style={[styles.progressBar, { width: `${progress}%` }]} />
          </View>

          <Pressable style={styles.content} onPress={handlePress}>
            {/* Album art */}
            <Image
              source={{ uri: currentSong.coverArt }}
              style={styles.albumArt}
            />

            {/* Song info */}
            <View style={styles.songInfo}>
              <Text style={styles.title} numberOfLines={1}>
                {currentSong.title}
              </Text>
              <Text style={styles.artist} numberOfLines={1}>
                {currentSong.artist}
              </Text>
            </View>

            {/* Controls */}
            <View style={styles.controls}>
              <TouchableOpacity
                style={styles.likeButton}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <HeartIcon size={20} color={colors.textSecondary} />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.playButton}
                onPress={handlePlayPause}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                {isPlaying ? (
                  <PauseIcon size={24} color={colors.text} />
                ) : (
                  <PlayIcon size={24} color={colors.text} />
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.nextButton}
                onPress={handleNext}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <SkipForwardIcon size={20} color={colors.textSecondary} />
              </TouchableOpacity>
            </View>
          </Pressable>
        </Animated.View>

        {/* Full Player (shown when expanded) */}
        <Animated.View style={[styles.fullPlayer, animatedFullStyle]}>
          {/* Drag handle */}
          <View style={styles.dragHandle}>
            <View style={styles.dragIndicator} />
          </View>

          {/* Full player content would go here */}
          <View style={styles.fullPlayerContent}>
            <Image
              source={{ uri: currentSong.coverArt }}
              style={styles.fullAlbumArt}
            />

            <View style={styles.fullSongInfo}>
              <Text style={styles.fullTitle}>{currentSong.title}</Text>
              <Text style={styles.fullArtist}>{currentSong.artist}</Text>
            </View>

            {/* Full controls would go here */}
          </View>
        </Animated.View>
      </Animated.View>
    </PanGestureHandler>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: FULL_PLAYER_HEIGHT,
    backgroundColor: colors.background,
  },
  miniPlayer: {
    height: MINI_PLAYER_HEIGHT,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  progressContainer: {
    height: 2,
    backgroundColor: colors.backgroundSecondary,
  },
  progressBar: {
    height: '100%',
    backgroundColor: colors.primary,
  },
  content: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
  },
  albumArt: {
    width: 48,
    height: 48,
    borderRadius: 4,
    backgroundColor: colors.backgroundSecondary,
  },
  songInfo: {
    flex: 1,
    marginLeft: spacing.md,
    marginRight: spacing.sm,
  },
  title: {
    ...typography.bodyMedium,
    color: colors.text,
  },
  artist: {
    ...typography.bodySmall,
    color: colors.textSecondary,
    marginTop: 2,
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  likeButton: {
    padding: spacing.xs,
  },
  playButton: {
    padding: spacing.xs,
  },
  nextButton: {
    padding: spacing.xs,
  },
  fullPlayer: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: colors.background,
  },
  dragHandle: {
    alignItems: 'center',
    paddingTop: spacing.md,
  },
  dragIndicator: {
    width: 40,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.border,
  },
  fullPlayerContent: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.xl,
  },
  fullAlbumArt: {
    width: SCREEN_WIDTH - spacing.xl * 2,
    height: SCREEN_WIDTH - spacing.xl * 2,
    borderRadius: 8,
    backgroundColor: colors.backgroundSecondary,
  },
  fullSongInfo: {
    alignItems: 'center',
    marginTop: spacing.xl,
  },
  fullTitle: {
    ...typography.h2,
    color: colors.text,
    textAlign: 'center',
  },
  fullArtist: {
    ...typography.body,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
});

export default MiniPlayer;
