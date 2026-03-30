// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile Inbox Screen
 *
 * Main email list view with:
 * - Pull to refresh
 * - Swipe actions (archive, delete, snooze)
 * - Search
 * - Trust indicators
 * - Offline support
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  StyleSheet,
  TextInput,
  Animated,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Swipeable, GestureHandlerRootView } from 'react-native-gesture-handler';

import { emailService } from '../services/email';
import { useOfflineStore } from '../services/offline';
import { TrustBadge } from '../components/TrustBadge';
import { EmptyState } from '../components/EmptyState';
import { EmailSkeleton } from '../components/EmailSkeleton';

// ============================================================================
// Types
// ============================================================================

interface Email {
  id: string;
  threadId: string;
  from: { email: string; name: string };
  subject: string;
  preview: string;
  receivedAt: string;
  isRead: boolean;
  isStarred: boolean;
  hasAttachments: boolean;
  trustScore: number;
  labels: string[];
}

interface InboxScreenProps {
  folder?: string;
}

// ============================================================================
// Email Item Component
// ============================================================================

const EmailItem: React.FC<{
  email: Email;
  onPress: () => void;
  onArchive: () => void;
  onDelete: () => void;
  onStar: () => void;
}> = ({ email, onPress, onArchive, onDelete, onStar }) => {
  const swipeableRef = React.useRef<Swipeable>(null);

  const renderLeftActions = (
    progress: Animated.AnimatedInterpolation<number>,
    dragX: Animated.AnimatedInterpolation<number>
  ) => {
    const scale = dragX.interpolate({
      inputRange: [0, 100],
      outputRange: [0.5, 1],
      extrapolate: 'clamp',
    });

    return (
      <View style={styles.swipeActionLeft}>
        <Animated.View style={{ transform: [{ scale }] }}>
          <Ionicons name="archive-outline" size={24} color="white" />
          <Text style={styles.swipeActionText}>Archive</Text>
        </Animated.View>
      </View>
    );
  };

  const renderRightActions = (
    progress: Animated.AnimatedInterpolation<number>,
    dragX: Animated.AnimatedInterpolation<number>
  ) => {
    const scale = dragX.interpolate({
      inputRange: [-100, 0],
      outputRange: [1, 0.5],
      extrapolate: 'clamp',
    });

    return (
      <View style={styles.swipeActionRight}>
        <Animated.View style={{ transform: [{ scale }] }}>
          <Ionicons name="trash-outline" size={24} color="white" />
          <Text style={styles.swipeActionText}>Delete</Text>
        </Animated.View>
      </View>
    );
  };

  const handleSwipeableOpen = (direction: 'left' | 'right') => {
    if (direction === 'left') {
      onArchive();
    } else {
      onDelete();
    }
    swipeableRef.current?.close();
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
    if (diff < 604800000) return date.toLocaleDateString('en-US', { weekday: 'short' });
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <Swipeable
      ref={swipeableRef}
      renderLeftActions={renderLeftActions}
      renderRightActions={renderRightActions}
      onSwipeableOpen={handleSwipeableOpen}
      overshootLeft={false}
      overshootRight={false}
    >
      <TouchableOpacity
        style={[styles.emailItem, !email.isRead && styles.emailItemUnread]}
        onPress={onPress}
        activeOpacity={0.7}
      >
        <View style={styles.emailLeft}>
          <View style={[styles.avatar, { backgroundColor: getAvatarColor(email.from.email) }]}>
            <Text style={styles.avatarText}>
              {(email.from.name || email.from.email)[0].toUpperCase()}
            </Text>
          </View>
          <TrustBadge score={email.trustScore} size="small" />
        </View>

        <View style={styles.emailContent}>
          <View style={styles.emailHeader}>
            <Text
              style={[styles.emailFrom, !email.isRead && styles.emailFromUnread]}
              numberOfLines={1}
            >
              {email.from.name || email.from.email}
            </Text>
            <Text style={styles.emailDate}>{formatDate(email.receivedAt)}</Text>
          </View>

          <Text
            style={[styles.emailSubject, !email.isRead && styles.emailSubjectUnread]}
            numberOfLines={1}
          >
            {email.subject || '(No subject)'}
          </Text>

          <Text style={styles.emailPreview} numberOfLines={1}>
            {email.preview}
          </Text>

          <View style={styles.emailMeta}>
            {email.hasAttachments && (
              <Ionicons name="attach" size={14} color="#666" style={styles.metaIcon} />
            )}
            {email.labels.slice(0, 2).map((label) => (
              <View key={label} style={styles.label}>
                <Text style={styles.labelText}>{label}</Text>
              </View>
            ))}
          </View>
        </View>

        <TouchableOpacity onPress={onStar} style={styles.starButton}>
          <Ionicons
            name={email.isStarred ? 'star' : 'star-outline'}
            size={20}
            color={email.isStarred ? '#FFB800' : '#CCC'}
          />
        </TouchableOpacity>
      </TouchableOpacity>
    </Swipeable>
  );
};

// ============================================================================
// Main Screen Component
// ============================================================================

export default function InboxScreen({ folder = 'inbox' }: InboxScreenProps) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { isOffline, queueAction } = useOfflineStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchFocused, setIsSearchFocused] = useState(false);

  // Fetch emails
  const {
    data: emails,
    isLoading,
    isRefetching,
    refetch,
    error,
  } = useQuery({
    queryKey: ['emails', folder],
    queryFn: () => emailService.getEmails(folder),
    staleTime: 30000,
  });

  // Mutations
  const archiveMutation = useMutation({
    mutationFn: emailService.archiveEmails,
    onMutate: async (ids) => {
      await queryClient.cancelQueries({ queryKey: ['emails', folder] });
      const previous = queryClient.getQueryData(['emails', folder]);

      queryClient.setQueryData(['emails', folder], (old: Email[] | undefined) =>
        old?.filter((e) => !ids.includes(e.id))
      );

      return { previous };
    },
    onError: (err, ids, context) => {
      queryClient.setQueryData(['emails', folder], context?.previous);
      if (isOffline) {
        queueAction({ type: 'archive', ids });
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: emailService.deleteEmails,
    onMutate: async (ids) => {
      await queryClient.cancelQueries({ queryKey: ['emails', folder] });
      const previous = queryClient.getQueryData(['emails', folder]);

      queryClient.setQueryData(['emails', folder], (old: Email[] | undefined) =>
        old?.filter((e) => !ids.includes(e.id))
      );

      return { previous };
    },
    onError: (err, ids, context) => {
      queryClient.setQueryData(['emails', folder], context?.previous);
    },
  });

  const starMutation = useMutation({
    mutationFn: ({ id, starred }: { id: string; starred: boolean }) =>
      emailService.updateEmail(id, { isStarred: starred }),
    onMutate: async ({ id, starred }) => {
      await queryClient.cancelQueries({ queryKey: ['emails', folder] });
      const previous = queryClient.getQueryData(['emails', folder]);

      queryClient.setQueryData(['emails', folder], (old: Email[] | undefined) =>
        old?.map((e) => (e.id === id ? { ...e, isStarred: starred } : e))
      );

      return { previous };
    },
    onError: (err, vars, context) => {
      queryClient.setQueryData(['emails', folder], context?.previous);
    },
  });

  // Handlers
  const handleRefresh = useCallback(() => {
    refetch();
  }, [refetch]);

  const handleEmailPress = useCallback(
    (email: Email) => {
      router.push(`/email/${email.id}`);
    },
    [router]
  );

  const handleArchive = useCallback(
    (id: string) => {
      archiveMutation.mutate([id]);
    },
    [archiveMutation]
  );

  const handleDelete = useCallback(
    (id: string) => {
      Alert.alert('Delete Email', 'Are you sure you want to delete this email?', [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => deleteMutation.mutate([id]),
        },
      ]);
    },
    [deleteMutation]
  );

  const handleStar = useCallback(
    (email: Email) => {
      starMutation.mutate({ id: email.id, starred: !email.isStarred });
    },
    [starMutation]
  );

  const handleCompose = useCallback(() => {
    router.push('/compose');
  }, [router]);

  // Filtered emails
  const filteredEmails = useMemo(() => {
    if (!emails) return [];
    if (!searchQuery.trim()) return emails;

    const query = searchQuery.toLowerCase();
    return emails.filter(
      (email: Email) =>
        email.subject.toLowerCase().includes(query) ||
        email.from.name?.toLowerCase().includes(query) ||
        email.from.email.toLowerCase().includes(query) ||
        email.preview.toLowerCase().includes(query)
    );
  }, [emails, searchQuery]);

  // Render
  const renderEmail = useCallback(
    ({ item }: { item: Email }) => (
      <EmailItem
        email={item}
        onPress={() => handleEmailPress(item)}
        onArchive={() => handleArchive(item.id)}
        onDelete={() => handleDelete(item.id)}
        onStar={() => handleStar(item)}
      />
    ),
    [handleEmailPress, handleArchive, handleDelete, handleStar]
  );

  const keyExtractor = useCallback((item: Email) => item.id, []);

  return (
    <GestureHandlerRootView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>
          {folder.charAt(0).toUpperCase() + folder.slice(1)}
        </Text>
        {isOffline && (
          <View style={styles.offlineBadge}>
            <Ionicons name="cloud-offline" size={14} color="#FF9800" />
            <Text style={styles.offlineText}>Offline</Text>
          </View>
        )}
      </View>

      {/* Search Bar */}
      <View style={[styles.searchContainer, isSearchFocused && styles.searchContainerFocused]}>
        <Ionicons name="search" size={20} color="#666" />
        <TextInput
          style={styles.searchInput}
          placeholder="Search emails..."
          value={searchQuery}
          onChangeText={setSearchQuery}
          onFocus={() => setIsSearchFocused(true)}
          onBlur={() => setIsSearchFocused(false)}
          returnKeyType="search"
        />
        {searchQuery.length > 0 && (
          <TouchableOpacity onPress={() => setSearchQuery('')}>
            <Ionicons name="close-circle" size={20} color="#666" />
          </TouchableOpacity>
        )}
      </View>

      {/* Email List */}
      {isLoading ? (
        <View style={styles.loadingContainer}>
          {[...Array(5)].map((_, i) => (
            <EmailSkeleton key={i} />
          ))}
        </View>
      ) : error ? (
        <EmptyState
          icon="alert-circle-outline"
          title="Unable to load emails"
          description="Please check your connection and try again"
          actionLabel="Retry"
          onAction={handleRefresh}
        />
      ) : filteredEmails.length === 0 ? (
        <EmptyState
          icon={searchQuery ? 'search' : 'mail-outline'}
          title={searchQuery ? 'No results found' : 'No emails yet'}
          description={
            searchQuery
              ? `No emails matching "${searchQuery}"`
              : 'Your inbox is empty. Time to relax!'
          }
        />
      ) : (
        <FlatList
          data={filteredEmails}
          renderItem={renderEmail}
          keyExtractor={keyExtractor}
          refreshControl={
            <RefreshControl
              refreshing={isRefetching}
              onRefresh={handleRefresh}
              tintColor="#667eea"
            />
          }
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          initialNumToRender={10}
          maxToRenderPerBatch={10}
          windowSize={5}
        />
      )}

      {/* Compose FAB */}
      <TouchableOpacity style={styles.fab} onPress={handleCompose} activeOpacity={0.8}>
        <Ionicons name="create-outline" size={24} color="white" />
      </TouchableOpacity>
    </GestureHandlerRootView>
  );
}

// ============================================================================
// Helper Functions
// ============================================================================

function getAvatarColor(email: string): string {
  const colors = [
    '#667eea',
    '#764ba2',
    '#f093fb',
    '#f5576c',
    '#4facfe',
    '#00f2fe',
    '#43e97b',
    '#fa709a',
    '#fee140',
    '#30cfd0',
  ];
  let hash = 0;
  for (let i = 0; i < email.length; i++) {
    hash = email.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
}

// ============================================================================
// Styles
// ============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
    backgroundColor: 'white',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  offlineBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF3E0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  offlineText: {
    fontSize: 12,
    color: '#FF9800',
    marginLeft: 4,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    marginHorizontal: 16,
    marginVertical: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  searchContainerFocused: {
    borderColor: '#667eea',
  },
  searchInput: {
    flex: 1,
    marginLeft: 8,
    fontSize: 16,
    color: '#333',
  },
  loadingContainer: {
    flex: 1,
    padding: 16,
  },
  listContent: {
    paddingBottom: 100,
  },
  emailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#e0e0e0',
  },
  emailItemUnread: {
    backgroundColor: '#f8f9ff',
  },
  emailLeft: {
    alignItems: 'center',
    marginRight: 12,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  avatarText: {
    fontSize: 18,
    fontWeight: '600',
    color: 'white',
  },
  emailContent: {
    flex: 1,
  },
  emailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  emailFrom: {
    fontSize: 15,
    color: '#333',
    flex: 1,
    marginRight: 8,
  },
  emailFromUnread: {
    fontWeight: '600',
  },
  emailDate: {
    fontSize: 12,
    color: '#999',
  },
  emailSubject: {
    fontSize: 14,
    color: '#333',
    marginBottom: 2,
  },
  emailSubjectUnread: {
    fontWeight: '600',
  },
  emailPreview: {
    fontSize: 13,
    color: '#666',
    marginBottom: 4,
  },
  emailMeta: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  metaIcon: {
    marginRight: 8,
  },
  label: {
    backgroundColor: '#e8eaf6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginRight: 4,
  },
  labelText: {
    fontSize: 10,
    color: '#3f51b5',
  },
  starButton: {
    padding: 8,
  },
  swipeActionLeft: {
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    width: 100,
  },
  swipeActionRight: {
    backgroundColor: '#F44336',
    justifyContent: 'center',
    alignItems: 'center',
    width: 100,
  },
  swipeActionText: {
    color: 'white',
    fontSize: 12,
    marginTop: 4,
  },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 30,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#667eea',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
});
