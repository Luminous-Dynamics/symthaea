// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Profile / Soul Screen
 *
 * Displays user's soul, resonance, presences, and connections.
 */

import { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Pressable,
  RefreshControl,
  Dimensions,
} from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../../src/theme/ThemeContext';
import { useIdentityContext } from '../../src/providers/IdentityProvider';
import { SectionHeader } from '../../src/components/SectionHeader';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface StatCardProps {
  label: string;
  value: string | number;
  icon: string;
  color: string;
}

function StatCard({ label, value, icon, color }: StatCardProps) {
  const { colors } = useTheme();
  const scale = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <Pressable
      onPressIn={() => {
        scale.value = withSpring(0.95);
      }}
      onPressOut={() => {
        scale.value = withSpring(1);
      }}
    >
      <Animated.View style={[styles.statCard, animatedStyle]}>
        <BlurView intensity={40} tint="dark" style={StyleSheet.absoluteFill} />
        <Text style={[styles.statIcon, { color }]}>{icon}</Text>
        <Text style={[styles.statValue, { color: colors.text }]}>{value}</Text>
        <Text style={[styles.statLabel, { color: colors.textMuted }]}>{label}</Text>
      </Animated.View>
    </Pressable>
  );
}

interface PresenceBadgeProps {
  type: string;
  name: string;
  date: string;
  resonance: number;
}

function PresenceBadge({ type, name, date, resonance }: PresenceBadgeProps) {
  const { colors } = useTheme();

  const getPresenceColor = () => {
    switch (type) {
      case 'LIVE_CONCERT': return '#FF6B6B';
      case 'ALBUM_RELEASE': return '#4ECDC4';
      case 'LISTENING_CIRCLE': return '#9B59B6';
      case 'FIRST_LISTEN': return '#F1C40F';
      default: return colors.primary;
    }
  };

  return (
    <Pressable
      onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
      style={styles.presenceBadge}
    >
      <LinearGradient
        colors={[getPresenceColor() + '40', getPresenceColor() + '10']}
        style={styles.presenceGradient}
      >
        <View style={[styles.presenceIcon, { backgroundColor: getPresenceColor() }]} />
        <View style={styles.presenceInfo}>
          <Text style={[styles.presenceName, { color: colors.text }]}>{name}</Text>
          <Text style={[styles.presenceDate, { color: colors.textMuted }]}>{date}</Text>
        </View>
        <View style={styles.presenceResonance}>
          <Text style={[styles.resonanceValue, { color: getPresenceColor() }]}>
            {resonance}
          </Text>
          <Text style={[styles.resonanceLabel, { color: colors.textMuted }]}>RES</Text>
        </View>
      </LinearGradient>
    </Pressable>
  );
}

export default function ProfileScreen() {
  const { colors } = useTheme();
  const { soul, isAuthenticated, connect, disconnect } = useIdentityContext();
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  if (!isAuthenticated || !soul) {
    return (
      <View style={[styles.container, styles.centered, { backgroundColor: colors.background }]}>
        <LinearGradient
          colors={[colors.primary + '20', colors.background]}
          style={styles.connectGradient}
        >
          <Text style={[styles.connectTitle, { color: colors.text }]}>
            Awaken Your Soul
          </Text>
          <Text style={[styles.connectSubtitle, { color: colors.textMuted }]}>
            Connect your wallet to create your Mycelix identity and start building resonance
          </Text>
          <Pressable
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
              connect();
            }}
            style={({ pressed }) => [
              styles.connectButton,
              { backgroundColor: colors.primary, opacity: pressed ? 0.8 : 1 },
            ]}
          >
            <Text style={styles.connectButtonText}>Connect Wallet</Text>
          </Pressable>
          <Text style={[styles.orText, { color: colors.textMuted }]}>or</Text>
          <Pressable
            onPress={() => router.push('/(auth)/social')}
            style={({ pressed }) => [
              styles.socialButton,
              { borderColor: colors.border, opacity: pressed ? 0.8 : 1 },
            ]}
          >
            <Text style={[styles.socialButtonText, { color: colors.text }]}>
              Continue with Email
            </Text>
          </Pressable>
        </LinearGradient>
      </View>
    );
  }

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
      {/* Profile Header */}
      <View style={styles.header}>
        <LinearGradient
          colors={[colors.primary + '30', colors.background]}
          style={styles.headerGradient}
        >
          <View style={styles.avatarContainer}>
            {soul.profile.avatarUri ? (
              <Image
                source={{ uri: soul.profile.avatarUri }}
                style={styles.avatar}
                contentFit="cover"
              />
            ) : (
              <LinearGradient
                colors={[colors.primary, colors.secondary]}
                style={styles.avatar}
              />
            )}
            <View style={[styles.soulBadge, { backgroundColor: colors.primary }]}>
              <Text style={styles.soulBadgeText}>#{soul.id.slice(-4)}</Text>
            </View>
          </View>

          <Text style={[styles.displayName, { color: colors.text }]}>
            {soul.profile.displayName}
          </Text>
          {soul.profile.bio && (
            <Text style={[styles.bio, { color: colors.textMuted }]}>
              {soul.profile.bio}
            </Text>
          )}

          <View style={styles.statsRow}>
            <StatCard
              label="Resonance"
              value={soul.resonance.total.toLocaleString()}
              icon="◈"
              color={colors.primary}
            />
            <StatCard
              label="Connections"
              value={soul.connections.length}
              icon="⬡"
              color={colors.secondary}
            />
            <StatCard
              label="Presences"
              value={12}
              icon="◉"
              color={colors.accent}
            />
          </View>
        </LinearGradient>
      </View>

      {/* Resonance Breakdown */}
      <SectionHeader title="Resonance Breakdown" />
      <View style={styles.resonanceBreakdown}>
        {Object.entries(soul.resonance.categories).map(([category, value]) => (
          <View key={category} style={styles.resonanceItem}>
            <View style={styles.resonanceItemHeader}>
              <Text style={[styles.resonanceCategory, { color: colors.text }]}>
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </Text>
              <Text style={[styles.resonanceItemValue, { color: colors.primary }]}>
                {value}
              </Text>
            </View>
            <View style={[styles.resonanceBar, { backgroundColor: colors.border }]}>
              <View
                style={[
                  styles.resonanceFill,
                  {
                    backgroundColor: colors.primary,
                    width: `${Math.min(100, (value / soul.resonance.total) * 100)}%`,
                  },
                ]}
              />
            </View>
          </View>
        ))}
      </View>

      {/* Recent Presences */}
      <SectionHeader
        title="Proof of Presence"
        subtitle="Your music journey, on-chain"
        onSeeAll={() => router.push('/presences')}
      />
      <View style={styles.presencesList}>
        <PresenceBadge
          type="LIVE_CONCERT"
          name="Ambient Night @ The Warehouse"
          date="Dec 15, 2024"
          resonance={450}
        />
        <PresenceBadge
          type="ALBUM_RELEASE"
          name="First Listen: Ethereal Waves"
          date="Dec 10, 2024"
          resonance={200}
        />
        <PresenceBadge
          type="LISTENING_CIRCLE"
          name="Late Night Ambient Circle"
          date="Dec 8, 2024"
          resonance={75}
        />
      </View>

      {/* Settings */}
      <SectionHeader title="Settings" />
      <View style={styles.settingsList}>
        <Pressable
          onPress={() => router.push('/settings/profile')}
          style={({ pressed }) => [
            styles.settingsItem,
            { backgroundColor: pressed ? colors.border + '20' : 'transparent' },
          ]}
        >
          <Text style={[styles.settingsItemText, { color: colors.text }]}>
            Edit Profile
          </Text>
        </Pressable>
        <Pressable
          onPress={() => router.push('/settings/wallet')}
          style={({ pressed }) => [
            styles.settingsItem,
            { backgroundColor: pressed ? colors.border + '20' : 'transparent' },
          ]}
        >
          <Text style={[styles.settingsItemText, { color: colors.text }]}>
            Wallet & Identity
          </Text>
        </Pressable>
        <Pressable
          onPress={() => router.push('/settings/notifications')}
          style={({ pressed }) => [
            styles.settingsItem,
            { backgroundColor: pressed ? colors.border + '20' : 'transparent' },
          ]}
        >
          <Text style={[styles.settingsItemText, { color: colors.text }]}>
            Notifications
          </Text>
        </Pressable>
        <Pressable
          onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            disconnect();
          }}
          style={({ pressed }) => [
            styles.settingsItem,
            { backgroundColor: pressed ? colors.border + '20' : 'transparent' },
          ]}
        >
          <Text style={[styles.settingsItemText, { color: '#FF6B6B' }]}>
            Disconnect
          </Text>
        </Pressable>
      </View>

      <View style={{ height: 150 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    paddingTop: 60,
  },
  connectGradient: {
    padding: 40,
    alignItems: 'center',
    borderRadius: 24,
    margin: 20,
  },
  connectTitle: {
    fontSize: 28,
    fontFamily: 'SpaceGrotesk-Bold',
    marginBottom: 12,
    textAlign: 'center',
  },
  connectSubtitle: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Regular',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
  },
  connectButton: {
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 30,
  },
  connectButtonText: {
    color: '#0a0a0a',
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  orText: {
    marginVertical: 16,
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Regular',
  },
  socialButton: {
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 30,
    borderWidth: 1,
  },
  socialButtonText: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  header: {
    marginBottom: 24,
  },
  headerGradient: {
    paddingTop: 20,
    paddingBottom: 32,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  avatarContainer: {
    position: 'relative',
    marginBottom: 16,
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
  },
  soulBadge: {
    position: 'absolute',
    bottom: 0,
    right: -8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  soulBadgeText: {
    color: '#0a0a0a',
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  displayName: {
    fontSize: 24,
    fontFamily: 'SpaceGrotesk-Bold',
    marginBottom: 8,
  },
  bio: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Regular',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 40,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  statCard: {
    width: (SCREEN_WIDTH - 76) / 3,
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    overflow: 'hidden',
  },
  statIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 20,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  statLabel: {
    fontSize: 11,
    fontFamily: 'SpaceGrotesk-Medium',
    marginTop: 4,
  },
  resonanceBreakdown: {
    paddingHorizontal: 20,
    gap: 16,
    marginBottom: 24,
  },
  resonanceItem: {
    gap: 8,
  },
  resonanceItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  resonanceCategory: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  resonanceItemValue: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  resonanceBar: {
    height: 6,
    borderRadius: 3,
  },
  resonanceFill: {
    height: '100%',
    borderRadius: 3,
  },
  presencesList: {
    paddingHorizontal: 20,
    gap: 12,
    marginBottom: 24,
  },
  presenceBadge: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  presenceGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    gap: 12,
  },
  presenceIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
  },
  presenceInfo: {
    flex: 1,
  },
  presenceName: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  presenceDate: {
    fontSize: 12,
    fontFamily: 'SpaceGrotesk-Regular',
    marginTop: 2,
  },
  presenceResonance: {
    alignItems: 'center',
  },
  resonanceValue: {
    fontSize: 18,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  resonanceLabel: {
    fontSize: 10,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  settingsList: {
    paddingHorizontal: 20,
    gap: 4,
  },
  settingsItem: {
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderRadius: 12,
  },
  settingsItemText: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
});
