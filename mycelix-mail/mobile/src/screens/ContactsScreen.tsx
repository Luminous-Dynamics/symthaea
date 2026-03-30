// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contacts Screen
 *
 * Contact list with trust scores, groups, and search
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  FlatList,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  SectionList,
  Alert,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as Haptics from 'expo-haptics';

import { useTheme } from '../providers/ThemeProvider';
import { RootStackParamList } from '../types/navigation';
import { api } from '../services/api';
import { Avatar } from '../components/Avatar';
import { TrustBadge } from '../components/TrustBadge';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

interface Contact {
  id: string;
  email: string;
  name?: string;
  phone?: string;
  company?: string;
  trustScore?: number;
  isFavorite: boolean;
  lastContacted?: string;
  avatar?: string;
}

interface Section {
  title: string;
  data: Contact[];
}

export function ContactsScreen() {
  const { colors } = useTheme();
  const navigation = useNavigation<NavigationProp>();
  const queryClient = useQueryClient();

  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  // Fetch contacts
  const { data: contacts = [], isLoading } = useQuery({
    queryKey: ['contacts'],
    queryFn: () => api.getContacts(),
  });

  // Toggle favorite mutation
  const favoriteMutation = useMutation({
    mutationFn: ({ id, favorite }: { id: string; favorite: boolean }) =>
      api.setContactFavorite(id, favorite),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contacts'] });
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    },
  });

  // Delete contact mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteContact(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contacts'] });
    },
  });

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await queryClient.invalidateQueries({ queryKey: ['contacts'] });
    setRefreshing(false);
  }, [queryClient]);

  const handleContactPress = useCallback((contact: Contact) => {
    navigation.navigate('Compose', { to: [contact.email] });
  }, [navigation]);

  const handleContactLongPress = useCallback((contact: Contact) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      contact.name || contact.email,
      'Choose an action',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Send Email',
          onPress: () => navigation.navigate('Compose', { to: [contact.email] }),
        },
        {
          text: contact.isFavorite ? 'Remove from Favorites' : 'Add to Favorites',
          onPress: () => favoriteMutation.mutate({ id: contact.id, favorite: !contact.isFavorite }),
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            Alert.alert(
              'Delete Contact',
              'Are you sure you want to delete this contact?',
              [
                { text: 'Cancel', style: 'cancel' },
                { text: 'Delete', style: 'destructive', onPress: () => deleteMutation.mutate(contact.id) },
              ]
            );
          },
        },
      ]
    );
  }, [navigation, favoriteMutation, deleteMutation]);

  // Filter and group contacts
  const sections = useMemo(() => {
    let filtered = contacts;

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = contacts.filter(
        c =>
          c.email.toLowerCase().includes(query) ||
          c.name?.toLowerCase().includes(query) ||
          c.company?.toLowerCase().includes(query)
      );
    }

    // Sort alphabetically
    const sorted = [...filtered].sort((a, b) => {
      const nameA = (a.name || a.email).toLowerCase();
      const nameB = (b.name || b.email).toLowerCase();
      return nameA.localeCompare(nameB);
    });

    // Group by first letter
    const groups: Record<string, Contact[]> = {};
    const favorites: Contact[] = [];

    sorted.forEach(contact => {
      if (contact.isFavorite) {
        favorites.push(contact);
      }

      const letter = (contact.name || contact.email)[0].toUpperCase();
      if (!groups[letter]) {
        groups[letter] = [];
      }
      groups[letter].push(contact);
    });

    const result: Section[] = [];

    if (favorites.length > 0 && !searchQuery) {
      result.push({ title: '★ Favorites', data: favorites });
    }

    Object.keys(groups)
      .sort()
      .forEach(letter => {
        result.push({ title: letter, data: groups[letter] });
      });

    return result;
  }, [contacts, searchQuery]);

  const renderContact = useCallback(
    ({ item }: { item: Contact }) => (
      <TouchableOpacity
        style={[styles.contactItem, { backgroundColor: colors.surface }]}
        onPress={() => handleContactPress(item)}
        onLongPress={() => handleContactLongPress(item)}
      >
        <Avatar email={item.email} name={item.name} size={44} />
        <View style={styles.contactInfo}>
          <View style={styles.contactNameRow}>
            <Text style={[styles.contactName, { color: colors.text }]}>
              {item.name || item.email}
            </Text>
            {item.trustScore !== undefined && (
              <TrustBadge score={item.trustScore} size="small" />
            )}
          </View>
          {item.name && (
            <Text style={[styles.contactEmail, { color: colors.textSecondary }]}>
              {item.email}
            </Text>
          )}
          {item.company && (
            <Text style={[styles.contactCompany, { color: colors.textSecondary }]}>
              {item.company}
            </Text>
          )}
        </View>
        <View style={styles.contactActions}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => favoriteMutation.mutate({ id: item.id, favorite: !item.isFavorite })}
          >
            <Ionicons
              name={item.isFavorite ? 'star' : 'star-outline'}
              size={20}
              color={item.isFavorite ? '#f59e0b' : colors.textSecondary}
            />
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    ),
    [colors, handleContactPress, handleContactLongPress, favoriteMutation]
  );

  const renderSectionHeader = useCallback(
    ({ section }: { section: Section }) => (
      <View style={[styles.sectionHeader, { backgroundColor: colors.background }]}>
        <Text style={[styles.sectionTitle, { color: colors.textSecondary }]}>
          {section.title}
        </Text>
      </View>
    ),
    [colors]
  );

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    header: {
      padding: 16,
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    searchContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: colors.background,
      borderRadius: 10,
      paddingHorizontal: 12,
    },
    searchIcon: {
      marginRight: 8,
    },
    searchInput: {
      flex: 1,
      height: 40,
      fontSize: 16,
      color: colors.text,
    },
    clearButton: {
      padding: 4,
    },
    addButton: {
      position: 'absolute',
      bottom: 24,
      right: 24,
      width: 56,
      height: 56,
      borderRadius: 28,
      backgroundColor: colors.primary,
      alignItems: 'center',
      justifyContent: 'center',
      elevation: 4,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.25,
      shadowRadius: 4,
    },
    list: {
      flex: 1,
    },
    sectionHeader: {
      paddingHorizontal: 16,
      paddingVertical: 8,
    },
    sectionTitle: {
      fontSize: 13,
      fontWeight: '600',
      textTransform: 'uppercase',
    },
    contactItem: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: 12,
      paddingHorizontal: 16,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border,
    },
    contactInfo: {
      flex: 1,
      marginLeft: 12,
    },
    contactNameRow: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    contactName: {
      fontSize: 16,
      fontWeight: '500',
    },
    contactEmail: {
      fontSize: 14,
      marginTop: 2,
    },
    contactCompany: {
      fontSize: 13,
      marginTop: 2,
    },
    contactActions: {
      flexDirection: 'row',
    },
    actionButton: {
      padding: 8,
    },
    emptyContainer: {
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
      padding: 40,
    },
    emptyText: {
      fontSize: 16,
      color: colors.textSecondary,
      textAlign: 'center',
      marginTop: 16,
    },
    indexContainer: {
      position: 'absolute',
      right: 0,
      top: 100,
      bottom: 100,
      justifyContent: 'center',
      paddingHorizontal: 4,
    },
    indexLetter: {
      fontSize: 11,
      fontWeight: '600',
      color: colors.primary,
      paddingVertical: 1,
    },
  });

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      {/* Search Header */}
      <View style={styles.header}>
        <View style={styles.searchContainer}>
          <Ionicons
            name="search"
            size={20}
            color={colors.textSecondary}
            style={styles.searchIcon}
          />
          <TextInput
            style={styles.searchInput}
            placeholder="Search contacts..."
            placeholderTextColor={colors.textSecondary}
            value={searchQuery}
            onChangeText={setSearchQuery}
            autoCorrect={false}
            autoCapitalize="none"
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity
              style={styles.clearButton}
              onPress={() => setSearchQuery('')}
            >
              <Ionicons name="close-circle" size={20} color={colors.textSecondary} />
            </TouchableOpacity>
          )}
        </View>
      </View>

      {/* Contact List */}
      {sections.length > 0 ? (
        <SectionList
          style={styles.list}
          sections={sections}
          keyExtractor={item => item.id}
          renderItem={renderContact}
          renderSectionHeader={renderSectionHeader}
          stickySectionHeadersEnabled
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={handleRefresh}
              tintColor={colors.primary}
            />
          }
        />
      ) : (
        <View style={styles.emptyContainer}>
          <Ionicons name="people-outline" size={64} color={colors.textSecondary} />
          <Text style={styles.emptyText}>
            {searchQuery ? 'No contacts match your search' : 'No contacts yet'}
          </Text>
        </View>
      )}

      {/* Add Contact FAB */}
      <TouchableOpacity
        style={styles.addButton}
        onPress={() => Alert.alert('Add Contact', 'Contact creation coming soon!')}
      >
        <Ionicons name="add" size={28} color="#fff" />
      </TouchableOpacity>
    </SafeAreaView>
  );
}
