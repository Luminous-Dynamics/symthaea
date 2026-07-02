// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Library Screen
 *
 * User's saved music, playlists, and listening history.
 */

import { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Pressable,
  RefreshControl,
} from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useTheme } from '../../src/theme/ThemeContext';
import { useIdentityContext } from '../../src/providers/IdentityProvider';
import { SectionHeader } from '../../src/components/SectionHeader';

type LibraryTab = 'playlists' | 'albums' | 'artists' | 'history';

interface TabButtonProps {
  label: string;
  active: boolean;
  onPress: () => void;
}

function TabButton({ label, active, onPress }: TabButtonProps) {
  const { colors } = useTheme();

  return (
    <Pressable
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
      style={[
        styles.tabButton,
        {
          backgroundColor: active ? colors.primary : colors.border + '30',
        },
      ]}
    >
      <Text
        style={[
          styles.tabButtonText,
          { color: active ? '#0a0a0a' : colors.text },
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

interface LibraryItemProps {
  title: string;
  subtitle: string;
  imageUri: string;
  onPress: () => void;
  type?: 'square' | 'round';
}

function LibraryItem({ title, subtitle, imageUri, onPress, type = 'square' }: LibraryItemProps) {
  const { colors } = useTheme();

  return (
    <Pressable
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
      style={({ pressed }) => [
        styles.libraryItem,
        { opacity: pressed ? 0.8 : 1 },
      ]}
    >
      <Image
        source={{ uri: imageUri }}
        style={[styles.libraryItemImage, type === 'round' && styles.roundImage]}
        contentFit="cover"
      />
      <View style={styles.libraryItemInfo}>
        <Text style={[styles.libraryItemTitle, { color: colors.text }]} numberOfLines={1}>
          {title}
        </Text>
        <Text style={[styles.libraryItemSubtitle, { color: colors.textMuted }]} numberOfLines={1}>
          {subtitle}
        </Text>
      </View>
    </Pressable>
  );
}

export default function LibraryScreen() {
  const { colors } = useTheme();
  const { isAuthenticated } = useIdentityContext();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<LibraryTab>('playlists');
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  if (!isAuthenticated) {
    return (
      <View style={[styles.container, styles.centered, { backgroundColor: colors.background }]}>
        <Text style={[styles.emptyTitle, { color: colors.text }]}>
          Your Library Awaits
        </Text>
        <Text style={[styles.emptySubtitle, { color: colors.textMuted }]}>
          Connect to access your saved music
        </Text>
        <Pressable
          onPress={() => router.push('/tabs/profile')}
          style={({ pressed }) => [
            styles.connectButton,
            { backgroundColor: colors.primary, opacity: pressed ? 0.8 : 1 },
          ]}
        >
          <Text style={styles.connectButtonText}>Connect</Text>
        </Pressable>
      </View>
    );
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'playlists':
        return (
          <View style={styles.listContent}>
            {/* Create Playlist Card */}
            <Pressable
              onPress={() => router.push('/playlist/create')}
              style={({ pressed }) => [
                styles.createCard,
                { borderColor: colors.primary, opacity: pressed ? 0.8 : 1 },
              ]}
            >
              <LinearGradient
                colors={[colors.primary + '20', colors.primary + '05']}
                style={styles.createCardGradient}
              >
                <Text style={[styles.createCardIcon, { color: colors.primary }]}>+</Text>
                <Text style={[styles.createCardText, { color: colors.primary }]}>
                  Create Playlist
                </Text>
              </LinearGradient>
            </Pressable>

            {/* Playlists */}
            {[1, 2, 3, 4, 5].map((i) => (
              <LibraryItem
                key={i}
                title={`My Playlist ${i}`}
                subtitle={`${10 + i * 3} tracks`}
                imageUri={`https://picsum.photos/seed/playlist${i}/200`}
                onPress={() => router.push(`/playlist/${i}`)}
              />
            ))}
          </View>
        );

      case 'albums':
        return (
          <View style={styles.listContent}>
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <LibraryItem
                key={i}
                title={`Saved Album ${i}`}
                subtitle="Artist Name"
                imageUri={`https://picsum.photos/seed/savedalbum${i}/200`}
                onPress={() => router.push(`/album/${i}`)}
              />
            ))}
          </View>
        );

      case 'artists':
        return (
          <View style={styles.listContent}>
            {[1, 2, 3, 4, 5].map((i) => (
              <LibraryItem
                key={i}
                title={`Artist ${i}`}
                subtitle={`${100 + i * 50} patrons`}
                imageUri={`https://picsum.photos/seed/followedartist${i}/200`}
                onPress={() => router.push(`/artist/${i}`)}
                type="round"
              />
            ))}
          </View>
        );

      case 'history':
        return (
          <View style={styles.listContent}>
            <SectionHeader title="Today" />
            {[1, 2, 3].map((i) => (
              <LibraryItem
                key={`today-${i}`}
                title={`Recent Track ${i}`}
                subtitle="Artist Name"
                imageUri={`https://picsum.photos/seed/history${i}/200`}
                onPress={() => {}}
              />
            ))}
            <SectionHeader title="Yesterday" />
            {[4, 5, 6].map((i) => (
              <LibraryItem
                key={`yesterday-${i}`}
                title={`Listened Track ${i}`}
                subtitle="Artist Name"
                imageUri={`https://picsum.photos/seed/history${i}/200`}
                onPress={() => {}}
              />
            ))}
          </View>
        );
    }
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
        <Text style={[styles.title, { color: colors.text }]}>Your Library</Text>
      </View>

      {/* Tab Bar */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.tabBar}
      >
        <TabButton
          label="Playlists"
          active={activeTab === 'playlists'}
          onPress={() => setActiveTab('playlists')}
        />
        <TabButton
          label="Albums"
          active={activeTab === 'albums'}
          onPress={() => setActiveTab('albums')}
        />
        <TabButton
          label="Artists"
          active={activeTab === 'artists'}
          onPress={() => setActiveTab('artists')}
        />
        <TabButton
          label="History"
          active={activeTab === 'history'}
          onPress={() => setActiveTab('history')}
        />
      </ScrollView>

      {renderContent()}

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
    padding: 40,
  },
  content: {
    paddingTop: 60,
  },
  header: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  tabBar: {
    paddingHorizontal: 20,
    gap: 12,
    marginBottom: 24,
  },
  tabButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  tabButtonText: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  listContent: {
    paddingHorizontal: 20,
    gap: 12,
  },
  createCard: {
    borderWidth: 1,
    borderStyle: 'dashed',
    borderRadius: 12,
    overflow: 'hidden',
  },
  createCardGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    gap: 12,
  },
  createCardIcon: {
    fontSize: 24,
    fontFamily: 'SpaceGrotesk-Bold',
  },
  createCardText: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  libraryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  libraryItemImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  roundImage: {
    borderRadius: 30,
  },
  libraryItemInfo: {
    flex: 1,
  },
  libraryItemTitle: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Medium',
  },
  libraryItemSubtitle: {
    fontSize: 14,
    fontFamily: 'SpaceGrotesk-Regular',
    marginTop: 2,
  },
  emptyTitle: {
    fontSize: 24,
    fontFamily: 'SpaceGrotesk-Bold',
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Regular',
    textAlign: 'center',
    marginBottom: 24,
  },
  connectButton: {
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 25,
  },
  connectButtonText: {
    color: '#0a0a0a',
    fontSize: 16,
    fontFamily: 'SpaceGrotesk-Bold',
  },
});
