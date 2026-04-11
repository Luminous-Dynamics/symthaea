// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Pulse - React Native Mobile App
 *
 * Cross-platform iOS and Android email client
 */

import React, { useEffect, useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StatusBar,
  RefreshControl,
  TextInput,
  ActivityIndicator,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

// Types
interface Email {
  id: string;
  from: string;
  fromName?: string;
  subject: string;
  preview: string;
  receivedAt: string;
  isRead: boolean;
  isStarred: boolean;
  hasAttachments: boolean;
  labels: string[];
}

interface Folder {
  id: string;
  name: string;
  unreadCount: number;
  icon: string;
}

// API Client
const API_BASE = 'https://api.mycelixmail.com';

async function apiRequest<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      // Auth token would be added here
      ...options?.headers,
    },
  });
  if (!response.ok) throw new Error('API request failed');
  return response.json();
}

// Navigation
const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  async function checkAuth() {
    // Check for stored auth token
    // For now, assume authenticated
    setIsAuthenticated(true);
    setIsLoading(false);
  }

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  if (!isAuthenticated) {
    return <LoginScreen onLogin={() => setIsAuthenticated(true)} />;
  }

  return (
    <NavigationContainer>
      <StatusBar barStyle="dark-content" />
      <Tab.Navigator
        screenOptions={{
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: '#8E8E93',
          headerShown: false,
        }}
      >
        <Tab.Screen
          name="Inbox"
          component={InboxStack}
          options={{
            tabBarIcon: ({ color }) => <TabIcon name="inbox" color={color} />,
            tabBarBadge: 3,
          }}
        />
        <Tab.Screen
          name="Search"
          component={SearchScreen}
          options={{
            tabBarIcon: ({ color }) => <TabIcon name="search" color={color} />,
          }}
        />
        <Tab.Screen
          name="Calendar"
          component={CalendarScreen}
          options={{
            tabBarIcon: ({ color }) => <TabIcon name="calendar" color={color} />,
          }}
        />
        <Tab.Screen
          name="Settings"
          component={SettingsScreen}
          options={{
            tabBarIcon: ({ color }) => <TabIcon name="settings" color={color} />,
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

function TabIcon({ name, color }: { name: string; color: string }) {
  const icons: Record<string, string> = {
    inbox: '📥',
    search: '🔍',
    calendar: '📅',
    settings: '⚙️',
  };
  return <Text style={{ fontSize: 20 }}>{icons[name]}</Text>;
}

// Inbox Stack Navigator
function InboxStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="EmailList"
        component={EmailListScreen}
        options={{ title: 'Inbox' }}
      />
      <Stack.Screen
        name="EmailDetail"
        component={EmailDetailScreen}
        options={{ title: 'Email' }}
      />
      <Stack.Screen
        name="Compose"
        component={ComposeScreen}
        options={{ title: 'New Email', presentation: 'modal' }}
      />
    </Stack.Navigator>
  );
}

// Login Screen
function LoginScreen({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleLogin() {
    setLoading(true);
    try {
      // Authenticate
      onLogin();
    } catch (error) {
      console.error('Login failed:', error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.loginContainer}>
        <Text style={styles.logo}>Mycelix Pulse</Text>
        <Text style={styles.tagline}>Decentralized communication for the future</Text>

        <TextInput
          style={styles.input}
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
        <TextInput
          style={styles.input}
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={handleLogin}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Sign In</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity style={styles.linkButton}>
          <Text style={styles.linkText}>Create Account</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

// Email List Screen
function EmailListScreen({ navigation }: any) {
  const [emails, setEmails] = useState<Email[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEmails();
  }, []);

  async function fetchEmails() {
    try {
      // Mock data for now
      const mockEmails: Email[] = [
        {
          id: '1',
          from: 'alice@example.com',
          fromName: 'Alice Smith',
          subject: 'Project Update',
          preview: 'Hi, I wanted to share the latest progress on our project...',
          receivedAt: new Date().toISOString(),
          isRead: false,
          isStarred: true,
          hasAttachments: true,
          labels: ['work'],
        },
        {
          id: '2',
          from: 'bob@example.com',
          fromName: 'Bob Jones',
          subject: 'Meeting Tomorrow',
          preview: 'Can we reschedule our meeting to 3pm instead?',
          receivedAt: new Date(Date.now() - 3600000).toISOString(),
          isRead: true,
          isStarred: false,
          hasAttachments: false,
          labels: [],
        },
        {
          id: '3',
          from: 'newsletter@tech.com',
          fromName: 'Tech Weekly',
          subject: 'This Week in Tech',
          preview: 'The top stories in technology this week including...',
          receivedAt: new Date(Date.now() - 86400000).toISOString(),
          isRead: true,
          isStarred: false,
          hasAttachments: false,
          labels: ['newsletter'],
        },
      ];
      setEmails(mockEmails);
    } catch (error) {
      console.error('Failed to fetch emails:', error);
    } finally {
      setLoading(false);
    }
  }

  async function onRefresh() {
    setRefreshing(true);
    await fetchEmails();
    setRefreshing(false);
  }

  function formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  }

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <FlatList
        data={emails}
        keyExtractor={(item) => item.id}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[styles.emailRow, !item.isRead && styles.emailUnread]}
            onPress={() => navigation.navigate('EmailDetail', { email: item })}
          >
            <View style={styles.emailHeader}>
              <Text style={[styles.emailFrom, !item.isRead && styles.textBold]}>
                {item.fromName || item.from}
              </Text>
              <Text style={styles.emailDate}>{formatDate(item.receivedAt)}</Text>
            </View>
            <Text style={[styles.emailSubject, !item.isRead && styles.textBold]} numberOfLines={1}>
              {item.subject}
            </Text>
            <Text style={styles.emailPreview} numberOfLines={2}>
              {item.preview}
            </Text>
            <View style={styles.emailMeta}>
              {item.isStarred && <Text>⭐</Text>}
              {item.hasAttachments && <Text>📎</Text>}
              {item.labels.map((label) => (
                <View key={label} style={styles.label}>
                  <Text style={styles.labelText}>{label}</Text>
                </View>
              ))}
            </View>
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No emails</Text>
          </View>
        }
      />
      <TouchableOpacity
        style={styles.fab}
        onPress={() => navigation.navigate('Compose')}
      >
        <Text style={styles.fabText}>+</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

// Email Detail Screen
function EmailDetailScreen({ route, navigation }: any) {
  const { email } = route.params as { email: Email };

  async function handleReply() {
    navigation.navigate('Compose', { replyTo: email });
  }

  async function handleForward() {
    navigation.navigate('Compose', { forward: email });
  }

  async function handleDelete() {
    // Delete email
    navigation.goBack();
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.emailDetail}>
        <View style={styles.emailDetailHeader}>
          <Text style={styles.emailDetailSubject}>{email.subject}</Text>
          <View style={styles.emailDetailMeta}>
            <Text style={styles.emailDetailFrom}>
              From: {email.fromName || email.from}
            </Text>
            <Text style={styles.emailDetailDate}>
              {new Date(email.receivedAt).toLocaleString()}
            </Text>
          </View>
        </View>

        <View style={styles.emailDetailBody}>
          <Text style={styles.emailDetailText}>{email.preview}</Text>
        </View>

        <View style={styles.emailActions}>
          <TouchableOpacity style={styles.actionButton} onPress={handleReply}>
            <Text style={styles.actionButtonText}>↩️ Reply</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={handleForward}>
            <Text style={styles.actionButtonText}>↗️ Forward</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.actionButton, styles.actionButtonDanger]}
            onPress={handleDelete}
          >
            <Text style={[styles.actionButtonText, styles.actionButtonTextDanger]}>
              🗑️ Delete
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

// Compose Screen
function ComposeScreen({ route, navigation }: any) {
  const replyTo = route.params?.replyTo;
  const forward = route.params?.forward;

  const [to, setTo] = useState(replyTo ? replyTo.from : '');
  const [subject, setSubject] = useState(
    replyTo ? `Re: ${replyTo.subject}` : forward ? `Fwd: ${forward.subject}` : ''
  );
  const [body, setBody] = useState('');
  const [sending, setSending] = useState(false);

  async function handleSend() {
    if (!to || !subject) return;

    setSending(true);
    try {
      // Send email
      navigation.goBack();
    } catch (error) {
      console.error('Failed to send:', error);
    } finally {
      setSending(false);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.composeContainer}>
        <View style={styles.composeField}>
          <Text style={styles.composeLabel}>To:</Text>
          <TextInput
            style={styles.composeInput}
            value={to}
            onChangeText={setTo}
            placeholder="recipient@example.com"
            keyboardType="email-address"
            autoCapitalize="none"
          />
        </View>
        <View style={styles.composeField}>
          <Text style={styles.composeLabel}>Subject:</Text>
          <TextInput
            style={styles.composeInput}
            value={subject}
            onChangeText={setSubject}
            placeholder="Email subject"
          />
        </View>
        <TextInput
          style={styles.composeBody}
          value={body}
          onChangeText={setBody}
          placeholder="Write your message..."
          multiline
          textAlignVertical="top"
        />
        <TouchableOpacity
          style={[styles.sendButton, sending && styles.buttonDisabled]}
          onPress={handleSend}
          disabled={sending || !to}
        >
          {sending ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.sendButtonText}>Send</Text>
          )}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

// Search Screen
function SearchScreen() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Email[]>([]);
  const [searching, setSearching] = useState(false);

  async function handleSearch() {
    if (!query.trim()) return;

    setSearching(true);
    try {
      // Search emails
      setResults([]);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setSearching(false);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.searchContainer}>
        <View style={styles.searchBar}>
          <TextInput
            style={styles.searchInput}
            value={query}
            onChangeText={setQuery}
            placeholder="Search emails..."
            returnKeyType="search"
            onSubmitEditing={handleSearch}
          />
          {searching && <ActivityIndicator size="small" />}
        </View>

        {results.length === 0 && query && !searching && (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No results found</Text>
          </View>
        )}

        <FlatList
          data={results}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <View style={styles.emailRow}>
              <Text style={styles.emailSubject}>{item.subject}</Text>
              <Text style={styles.emailPreview} numberOfLines={1}>
                {item.preview}
              </Text>
            </View>
          )}
        />
      </View>
    </SafeAreaView>
  );
}

// Calendar Screen
function CalendarScreen() {
  const [selectedDate, setSelectedDate] = useState(new Date());

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.calendarContainer}>
        <Text style={styles.calendarTitle}>Calendar</Text>
        <Text style={styles.calendarDate}>
          {selectedDate.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </Text>
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No events today</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

// Settings Screen
function SettingsScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.settingsContainer}>
        <Text style={styles.settingsTitle}>Settings</Text>

        <View style={styles.settingsSection}>
          <Text style={styles.settingsSectionTitle}>Account</Text>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Profile</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Notifications</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Privacy & Security</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.settingsSection}>
          <Text style={styles.settingsSectionTitle}>Mail</Text>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Signature</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Swipe Actions</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Default Account</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.settingsSection}>
          <Text style={styles.settingsSectionTitle}>Encryption</Text>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Manage Keys</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.settingsRow}>
            <Text>Import Key</Text>
            <Text style={styles.settingsArrow}>›</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.logoutButton}>
          <Text style={styles.logoutText}>Sign Out</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  // Login
  loginContainer: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  logo: {
    fontSize: 32,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  tagline: {
    fontSize: 16,
    color: '#8E8E93',
    textAlign: 'center',
    marginBottom: 48,
  },
  input: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  linkButton: {
    marginTop: 16,
    alignItems: 'center',
  },
  linkText: {
    color: '#007AFF',
    fontSize: 16,
  },
  // Email List
  emailRow: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#C6C6C8',
  },
  emailUnread: {
    backgroundColor: '#E8F4FF',
  },
  emailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  emailFrom: {
    fontSize: 16,
    color: '#000',
  },
  emailDate: {
    fontSize: 14,
    color: '#8E8E93',
  },
  emailSubject: {
    fontSize: 15,
    marginBottom: 4,
  },
  emailPreview: {
    fontSize: 14,
    color: '#8E8E93',
    lineHeight: 20,
  },
  emailMeta: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  textBold: {
    fontWeight: '600',
  },
  label: {
    backgroundColor: '#E5E5EA',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  labelText: {
    fontSize: 12,
    color: '#636366',
  },
  fab: {
    position: 'absolute',
    right: 24,
    bottom: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  fabText: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '300',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 48,
  },
  emptyText: {
    fontSize: 16,
    color: '#8E8E93',
  },
  // Email Detail
  emailDetail: {
    flex: 1,
    backgroundColor: '#fff',
  },
  emailDetailHeader: {
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#C6C6C8',
  },
  emailDetailSubject: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
  },
  emailDetailMeta: {
    gap: 4,
  },
  emailDetailFrom: {
    fontSize: 15,
  },
  emailDetailDate: {
    fontSize: 14,
    color: '#8E8E93',
  },
  emailDetailBody: {
    flex: 1,
    padding: 16,
  },
  emailDetailText: {
    fontSize: 16,
    lineHeight: 24,
  },
  emailActions: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#C6C6C8',
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#F2F2F7',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  actionButtonDanger: {
    backgroundColor: '#FFEBEB',
  },
  actionButtonText: {
    fontSize: 14,
  },
  actionButtonTextDanger: {
    color: '#FF3B30',
  },
  // Compose
  composeContainer: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 16,
  },
  composeField: {
    flexDirection: 'row',
    alignItems: 'center',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#C6C6C8',
    paddingVertical: 12,
  },
  composeLabel: {
    width: 60,
    fontSize: 16,
    color: '#8E8E93',
  },
  composeInput: {
    flex: 1,
    fontSize: 16,
  },
  composeBody: {
    flex: 1,
    fontSize: 16,
    paddingTop: 16,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  // Search
  searchContainer: {
    flex: 1,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    margin: 16,
    padding: 12,
    borderRadius: 12,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
  },
  // Calendar
  calendarContainer: {
    flex: 1,
    padding: 16,
  },
  calendarTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  calendarDate: {
    fontSize: 16,
    color: '#8E8E93',
    marginBottom: 24,
  },
  // Settings
  settingsContainer: {
    flex: 1,
    padding: 16,
  },
  settingsTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 24,
  },
  settingsSection: {
    marginBottom: 24,
  },
  settingsSectionTitle: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 8,
    marginLeft: 16,
    textTransform: 'uppercase',
  },
  settingsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#C6C6C8',
  },
  settingsArrow: {
    fontSize: 20,
    color: '#C7C7CC',
  },
  logoutButton: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 24,
  },
  logoutText: {
    color: '#FF3B30',
    fontSize: 16,
  },
});
