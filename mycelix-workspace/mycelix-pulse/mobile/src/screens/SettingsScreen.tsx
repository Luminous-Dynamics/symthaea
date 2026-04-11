// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Settings Screen
 *
 * App settings, account management, and preferences
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Switch,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as LocalAuthentication from 'expo-local-authentication';
import * as Haptics from 'expo-haptics';

import { useTheme, ThemeMode } from '../providers/ThemeProvider';
import { useAuth } from '../providers/AuthProvider';

type IconName = keyof typeof Ionicons.glyphMap;

interface SettingItem {
  id: string;
  icon: IconName;
  title: string;
  subtitle?: string;
  type: 'navigate' | 'toggle' | 'select' | 'action';
  value?: boolean | string;
  options?: { label: string; value: string }[];
  onPress?: () => void;
  onToggle?: (value: boolean) => void;
  destructive?: boolean;
}

interface SettingSection {
  title: string;
  items: SettingItem[];
}

export function SettingsScreen() {
  const { colors, mode, setMode } = useTheme();
  const { user, logout } = useAuth();

  const [biometricEnabled, setBiometricEnabled] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [pushEnabled, setPushEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [vibrationEnabled, setVibrationEnabled] = useState(true);
  const [autoDownloadAttachments, setAutoDownloadAttachments] = useState(false);
  const [readReceipts, setReadReceipts] = useState(true);
  const [compactMode, setCompactMode] = useState(false);
  const [swipeActions, setSwipeActions] = useState(true);

  const handleBiometricToggle = useCallback(async (value: boolean) => {
    if (value) {
      const hasHardware = await LocalAuthentication.hasHardwareAsync();
      if (!hasHardware) {
        Alert.alert('Not Available', 'Biometric authentication is not available on this device');
        return;
      }

      const isEnrolled = await LocalAuthentication.isEnrolledAsync();
      if (!isEnrolled) {
        Alert.alert('Not Set Up', 'Please set up biometric authentication in your device settings');
        return;
      }

      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Enable biometric login',
      });

      if (result.success) {
        setBiometricEnabled(true);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }
    } else {
      setBiometricEnabled(false);
    }
  }, []);

  const handleLogout = useCallback(() => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Sign Out', style: 'destructive', onPress: logout },
      ]
    );
  }, [logout]);

  const handleDeleteAccount = useCallback(() => {
    Alert.alert(
      'Delete Account',
      'This will permanently delete your account and all associated data. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete Account',
          style: 'destructive',
          onPress: () => {
            Alert.alert(
              'Confirm Deletion',
              'Type DELETE to confirm account deletion',
              [{ text: 'Cancel', style: 'cancel' }]
            );
          },
        },
      ]
    );
  }, []);

  const handleExportData = useCallback(() => {
    Alert.alert(
      'Export Data',
      'We will prepare your data export and send you an email when it\'s ready.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Export', onPress: () => Alert.alert('Export Started', 'You will receive an email when your data is ready.') },
      ]
    );
  }, []);

  const sections: SettingSection[] = [
    {
      title: 'Account',
      items: [
        {
          id: 'profile',
          icon: 'person-outline',
          title: 'Profile',
          subtitle: user?.email,
          type: 'navigate',
        },
        {
          id: 'linked-accounts',
          icon: 'link-outline',
          title: 'Linked Accounts',
          subtitle: '2 accounts',
          type: 'navigate',
        },
        {
          id: 'signature',
          icon: 'create-outline',
          title: 'Email Signature',
          type: 'navigate',
        },
        {
          id: 'vacation',
          icon: 'airplane-outline',
          title: 'Vacation Responder',
          subtitle: 'Off',
          type: 'navigate',
        },
      ],
    },
    {
      title: 'Security',
      items: [
        {
          id: 'biometric',
          icon: 'finger-print-outline',
          title: 'Biometric Lock',
          subtitle: 'Require Face ID or Touch ID to open app',
          type: 'toggle',
          value: biometricEnabled,
          onToggle: handleBiometricToggle,
        },
        {
          id: 'change-password',
          icon: 'key-outline',
          title: 'Change Password',
          type: 'navigate',
        },
        {
          id: 'two-factor',
          icon: 'shield-checkmark-outline',
          title: 'Two-Factor Authentication',
          subtitle: 'Enabled',
          type: 'navigate',
        },
        {
          id: 'trusted-devices',
          icon: 'phone-portrait-outline',
          title: 'Trusted Devices',
          subtitle: '3 devices',
          type: 'navigate',
        },
      ],
    },
    {
      title: 'Notifications',
      items: [
        {
          id: 'notifications',
          icon: 'notifications-outline',
          title: 'Notifications',
          type: 'toggle',
          value: notificationsEnabled,
          onToggle: setNotificationsEnabled,
        },
        {
          id: 'push',
          icon: 'phone-portrait-outline',
          title: 'Push Notifications',
          type: 'toggle',
          value: pushEnabled,
          onToggle: setPushEnabled,
        },
        {
          id: 'sound',
          icon: 'volume-high-outline',
          title: 'Sound',
          type: 'toggle',
          value: soundEnabled,
          onToggle: setSoundEnabled,
        },
        {
          id: 'vibration',
          icon: 'phone-portrait-outline',
          title: 'Vibration',
          type: 'toggle',
          value: vibrationEnabled,
          onToggle: setVibrationEnabled,
        },
      ],
    },
    {
      title: 'Appearance',
      items: [
        {
          id: 'theme',
          icon: 'color-palette-outline',
          title: 'Theme',
          type: 'select',
          value: mode,
          options: [
            { label: 'Light', value: 'light' },
            { label: 'Dark', value: 'dark' },
            { label: 'System', value: 'system' },
          ],
        },
        {
          id: 'compact',
          icon: 'list-outline',
          title: 'Compact Mode',
          subtitle: 'Show more emails on screen',
          type: 'toggle',
          value: compactMode,
          onToggle: setCompactMode,
        },
        {
          id: 'swipe-actions',
          icon: 'swap-horizontal-outline',
          title: 'Swipe Actions',
          subtitle: 'Archive and delete with swipe',
          type: 'toggle',
          value: swipeActions,
          onToggle: setSwipeActions,
        },
      ],
    },
    {
      title: 'Email Settings',
      items: [
        {
          id: 'auto-download',
          icon: 'download-outline',
          title: 'Auto-Download Attachments',
          subtitle: 'Download on Wi-Fi only',
          type: 'toggle',
          value: autoDownloadAttachments,
          onToggle: setAutoDownloadAttachments,
        },
        {
          id: 'read-receipts',
          icon: 'checkmark-done-outline',
          title: 'Read Receipts',
          type: 'toggle',
          value: readReceipts,
          onToggle: setReadReceipts,
        },
        {
          id: 'filters',
          icon: 'funnel-outline',
          title: 'Filters & Blocked',
          type: 'navigate',
        },
        {
          id: 'labels',
          icon: 'pricetag-outline',
          title: 'Manage Labels',
          type: 'navigate',
        },
      ],
    },
    {
      title: 'Trust Network',
      items: [
        {
          id: 'trust-settings',
          icon: 'shield-outline',
          title: 'Trust Settings',
          subtitle: 'Configure trust thresholds',
          type: 'navigate',
        },
        {
          id: 'attestations',
          icon: 'ribbon-outline',
          title: 'My Attestations',
          subtitle: '12 issued, 8 received',
          type: 'navigate',
        },
        {
          id: 'key-management',
          icon: 'key-outline',
          title: 'Key Management',
          subtitle: 'Manage your cryptographic keys',
          type: 'navigate',
        },
      ],
    },
    {
      title: 'Privacy & Data',
      items: [
        {
          id: 'privacy',
          icon: 'eye-off-outline',
          title: 'Privacy Settings',
          type: 'navigate',
        },
        {
          id: 'export',
          icon: 'cloud-download-outline',
          title: 'Export My Data',
          type: 'action',
          onPress: handleExportData,
        },
        {
          id: 'delete-account',
          icon: 'trash-outline',
          title: 'Delete Account',
          type: 'action',
          destructive: true,
          onPress: handleDeleteAccount,
        },
      ],
    },
    {
      title: 'About',
      items: [
        {
          id: 'version',
          icon: 'information-circle-outline',
          title: 'Version',
          subtitle: '1.0.0 (Build 42)',
          type: 'navigate',
        },
        {
          id: 'help',
          icon: 'help-circle-outline',
          title: 'Help & Support',
          type: 'navigate',
        },
        {
          id: 'terms',
          icon: 'document-text-outline',
          title: 'Terms of Service',
          type: 'navigate',
        },
        {
          id: 'privacy-policy',
          icon: 'shield-outline',
          title: 'Privacy Policy',
          type: 'navigate',
        },
        {
          id: 'logout',
          icon: 'log-out-outline',
          title: 'Sign Out',
          type: 'action',
          destructive: true,
          onPress: handleLogout,
        },
      ],
    },
  ];

  const handleThemeChange = useCallback((value: string) => {
    setMode(value as ThemeMode);
  }, [setMode]);

  const renderItem = (item: SettingItem) => {
    return (
      <TouchableOpacity
        key={item.id}
        style={[
          styles.item,
          { backgroundColor: colors.surface, borderBottomColor: colors.border },
        ]}
        onPress={() => {
          if (item.type === 'action' && item.onPress) {
            item.onPress();
          } else if (item.type === 'select') {
            Alert.alert(
              item.title,
              'Select an option',
              item.options?.map(opt => ({
                text: opt.label,
                onPress: () => handleThemeChange(opt.value),
              }))
            );
          }
        }}
        disabled={item.type === 'toggle'}
      >
        <View style={[styles.iconContainer, { backgroundColor: colors.background }]}>
          <Ionicons
            name={item.icon}
            size={22}
            color={item.destructive ? colors.error : colors.primary}
          />
        </View>
        <View style={styles.itemContent}>
          <Text
            style={[
              styles.itemTitle,
              { color: item.destructive ? colors.error : colors.text },
            ]}
          >
            {item.title}
          </Text>
          {item.subtitle && (
            <Text style={[styles.itemSubtitle, { color: colors.textSecondary }]}>
              {item.subtitle}
            </Text>
          )}
        </View>
        {item.type === 'toggle' && (
          <Switch
            value={item.value as boolean}
            onValueChange={item.onToggle}
            trackColor={{ false: colors.border, true: colors.primary }}
          />
        )}
        {item.type === 'navigate' && (
          <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
        )}
        {item.type === 'select' && (
          <View style={styles.selectValue}>
            <Text style={[styles.selectText, { color: colors.textSecondary }]}>
              {item.options?.find(o => o.value === item.value)?.label}
            </Text>
            <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
          </View>
        )}
      </TouchableOpacity>
    );
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    scrollView: {
      flex: 1,
    },
    section: {
      marginTop: 24,
    },
    sectionTitle: {
      fontSize: 13,
      fontWeight: '600',
      color: colors.textSecondary,
      paddingHorizontal: 16,
      paddingBottom: 8,
      textTransform: 'uppercase',
    },
    item: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: 12,
      paddingHorizontal: 16,
      borderBottomWidth: StyleSheet.hairlineWidth,
    },
    iconContainer: {
      width: 36,
      height: 36,
      borderRadius: 8,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 12,
    },
    itemContent: {
      flex: 1,
    },
    itemTitle: {
      fontSize: 16,
    },
    itemSubtitle: {
      fontSize: 13,
      marginTop: 2,
    },
    selectValue: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    selectText: {
      fontSize: 15,
      marginRight: 4,
    },
  });

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView}>
        {sections.map(section => (
          <View key={section.title} style={styles.section}>
            <Text style={styles.sectionTitle}>{section.title}</Text>
            {section.items.map(renderItem)}
          </View>
        ))}
        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}
