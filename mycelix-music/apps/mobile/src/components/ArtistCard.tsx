// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Artist Card Component
 *
 * Displays an artist with avatar and patron count.
 */

import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import { useTheme } from '../theme/ThemeContext';

const CARD_SIZE = 120;

interface ArtistCardProps {
  id: string;
  name: string;
  avatarUri: string;
  patronCount: number;
  isVerified?: boolean;
}

export function ArtistCard({
  id,
  name,
  avatarUri,
  patronCount,
  isVerified = false,
}: ArtistCardProps) {
  const { colors } = useTheme();
  const router = useRouter();
  const scale = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePress = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    router.push(`/artist/${id}`);
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
        <View style={styles.avatarContainer}>
          <Image
            source={{ uri: avatarUri }}
            style={styles.avatar}
            contentFit="cover"
          />
          <LinearGradient
            colors={[colors.primary + '40', colors.secondary + '40']}
            style={styles.avatarRing}
          />
          {isVerified && (
            <View style={[styles.verifiedBadge, { backgroundColor: colors.primary }]}>
              <Text style={styles.verifiedIcon}>✓</Text>
            </View>
          )}
        </View>
        <Text style={[styles.name, { color: colors.text }]} numberOfLines={1}>
          {name}
        </Text>
        <Text style={[styles.patronCount, { color: colors.textMuted }]}>
          {patronCount.toLocaleString()} patrons
        </Text>
      </Animated.View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    width: CARD_SIZE,
    alignItems: 'center',
  },
  avatarContainer: {
    position: 'relative',
    marginBottom: 10,
  },
  avatar: {
    width: CARD_SIZE - 16,
    height: CARD_SIZE - 16,
    borderRadius: (CARD_SIZE - 16) / 2,
  },
  avatarRing: {
    position: 'absolute',
    top: -4,
    left: -4,
    right: -4,
    bottom: -4,
    borderRadius: (CARD_SIZE - 8) / 2,
    zIndex: -1,
  },
  verifiedBadge: {
    position: 'absolute',
    bottom: 4,
    right: 4,
    width: 22,
    height: 22,
    borderRadius: 11,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#0a0a0a',
  },
  verifiedIcon: {
    color: '#0a0a0a',
    fontSize: 12,
    fontWeight: 'bold',
  },
  name: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
    textAlign: 'center',
  },
  patronCount: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
    marginTop: 2,
    textAlign: 'center',
  },
});
