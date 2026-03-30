// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Detail Screen
 *
 * Full email view with thread support, trust indicators, and actions
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Share,
  Alert,
  ActivityIndicator,
  Linking,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as Haptics from 'expo-haptics';
import RenderHtml from 'react-native-render-html';
import { useWindowDimensions } from 'react-native';

import { useTheme } from '../providers/ThemeProvider';
import { RootStackParamList } from '../types/navigation';
import { api } from '../services/api';
import { TrustBadge } from '../components/TrustBadge';
import { AttachmentList } from '../components/AttachmentList';
import { Avatar } from '../components/Avatar';

type EmailDetailRouteProp = RouteProp<RootStackParamList, 'EmailDetail'>;
type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

export function EmailDetailScreen() {
  const { colors } = useTheme();
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<EmailDetailRouteProp>();
  const { width } = useWindowDimensions();
  const queryClient = useQueryClient();

  const { id } = route.params;

  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [showFullHeaders, setShowFullHeaders] = useState(false);

  // Fetch email details
  const { data: email, isLoading, error } = useQuery({
    queryKey: ['email', id],
    queryFn: () => api.getEmail(id),
  });

  // Fetch thread if it exists
  const { data: thread } = useQuery({
    queryKey: ['thread', email?.threadId],
    queryFn: () => api.getThread(email!.threadId!),
    enabled: !!email?.threadId,
  });

  // Mark as read mutation
  const markReadMutation = useMutation({
    mutationFn: () => api.markAsRead(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
    },
  });

  // Star mutation
  const starMutation = useMutation({
    mutationFn: (starred: boolean) => api.starEmail(id, starred),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['email', id] });
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    },
  });

  // Archive mutation
  const archiveMutation = useMutation({
    mutationFn: () => api.archiveEmail(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      navigation.goBack();
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: () => api.deleteEmail(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      navigation.goBack();
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    },
  });

  // Mark as read when email loads
  React.useEffect(() => {
    if (email && !email.isRead) {
      markReadMutation.mutate();
    }
  }, [email?.id]);

  const handleReply = useCallback(() => {
    navigation.navigate('Compose', {
      replyTo: email,
    });
  }, [navigation, email]);

  const handleReplyAll = useCallback(() => {
    navigation.navigate('Compose', {
      replyTo: email,
      replyAll: true,
    });
  }, [navigation, email]);

  const handleForward = useCallback(() => {
    navigation.navigate('Compose', {
      forwardFrom: email,
    });
  }, [navigation, email]);

  const handleShare = useCallback(async () => {
    if (!email) return;
    try {
      await Share.share({
        message: `${email.subject}\n\n${email.bodyText}`,
        title: email.subject,
      });
    } catch (error) {
      console.error('Share failed:', error);
    }
  }, [email]);

  const handleDelete = useCallback(() => {
    Alert.alert(
      'Delete Email',
      'Are you sure you want to delete this email?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: () => deleteMutation.mutate() },
      ]
    );
  }, [deleteMutation]);

  const handleStar = useCallback(() => {
    starMutation.mutate(!email?.isStarred);
  }, [starMutation, email?.isStarred]);

  const formatDate = useCallback((dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
    }
  }, []);

  const htmlSource = useMemo(() => {
    if (!email?.bodyHtml) return { html: `<p>${email?.bodyText || ''}</p>` };
    return { html: email.bodyHtml };
  }, [email]);

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    loadingContainer: {
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
    },
    errorContainer: {
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
      padding: 20,
    },
    errorText: {
      color: colors.error,
      fontSize: 16,
      textAlign: 'center',
    },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: 16,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
      backgroundColor: colors.surface,
    },
    headerLeft: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    headerRight: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    headerButton: {
      padding: 8,
      marginLeft: 8,
    },
    scrollView: {
      flex: 1,
    },
    emailContainer: {
      padding: 16,
    },
    subject: {
      fontSize: 22,
      fontWeight: '700',
      color: colors.text,
      marginBottom: 16,
    },
    senderRow: {
      flexDirection: 'row',
      alignItems: 'flex-start',
    },
    senderInfo: {
      flex: 1,
      marginLeft: 12,
    },
    senderNameRow: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    senderName: {
      fontSize: 16,
      fontWeight: '600',
      color: colors.text,
    },
    senderEmail: {
      fontSize: 14,
      color: colors.textSecondary,
      marginTop: 2,
    },
    recipientRow: {
      flexDirection: 'row',
      alignItems: 'center',
      marginTop: 4,
    },
    recipientLabel: {
      fontSize: 13,
      color: colors.textSecondary,
    },
    recipientText: {
      fontSize: 13,
      color: colors.text,
      marginLeft: 4,
    },
    dateText: {
      fontSize: 13,
      color: colors.textSecondary,
      marginTop: 4,
    },
    showHeadersButton: {
      marginTop: 8,
    },
    showHeadersText: {
      fontSize: 13,
      color: colors.primary,
    },
    fullHeaders: {
      marginTop: 12,
      padding: 12,
      backgroundColor: colors.surface,
      borderRadius: 8,
    },
    headerLine: {
      fontSize: 12,
      color: colors.textSecondary,
      fontFamily: 'monospace',
    },
    bodyContainer: {
      marginTop: 20,
      paddingTop: 16,
      borderTopWidth: 1,
      borderTopColor: colors.border,
    },
    attachmentsContainer: {
      marginTop: 16,
      padding: 16,
      backgroundColor: colors.surface,
      borderRadius: 12,
    },
    attachmentsTitle: {
      fontSize: 14,
      fontWeight: '600',
      color: colors.text,
      marginBottom: 12,
    },
    toolbar: {
      flexDirection: 'row',
      justifyContent: 'space-around',
      padding: 12,
      borderTopWidth: 1,
      borderTopColor: colors.border,
      backgroundColor: colors.surface,
    },
    toolbarButton: {
      alignItems: 'center',
      padding: 8,
    },
    toolbarButtonText: {
      fontSize: 12,
      color: colors.textSecondary,
      marginTop: 4,
    },
    threadDivider: {
      flexDirection: 'row',
      alignItems: 'center',
      marginVertical: 20,
    },
    threadLine: {
      flex: 1,
      height: 1,
      backgroundColor: colors.border,
    },
    threadText: {
      paddingHorizontal: 12,
      fontSize: 13,
      color: colors.textSecondary,
    },
  });

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    );
  }

  if (error || !email) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="alert-circle-outline" size={48} color={colors.error} />
        <Text style={styles.errorText}>Failed to load email</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="arrow-back" size={24} color={colors.text} />
          </TouchableOpacity>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.headerButton} onPress={handleStar}>
            <Ionicons
              name={email.isStarred ? 'star' : 'star-outline'}
              size={24}
              color={email.isStarred ? '#f59e0b' : colors.textSecondary}
            />
          </TouchableOpacity>
          <TouchableOpacity style={styles.headerButton} onPress={() => archiveMutation.mutate()}>
            <Ionicons name="archive-outline" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.headerButton} onPress={handleDelete}>
            <Ionicons name="trash-outline" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.headerButton} onPress={handleShare}>
            <Ionicons name="share-outline" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView style={styles.scrollView}>
        <View style={styles.emailContainer}>
          {/* Subject */}
          <Text style={styles.subject}>{email.subject}</Text>

          {/* Sender Info */}
          <View style={styles.senderRow}>
            <Avatar email={email.from} size={44} />
            <View style={styles.senderInfo}>
              <View style={styles.senderNameRow}>
                <Text style={styles.senderName}>{email.fromName || email.from}</Text>
                <TrustBadge
                  score={email.trustScore}
                  style={{ marginLeft: 8 }}
                />
              </View>
              <Text style={styles.senderEmail}>{email.from}</Text>
              <View style={styles.recipientRow}>
                <Text style={styles.recipientLabel}>to:</Text>
                <Text style={styles.recipientText}>
                  {email.to.length > 1 ? `${email.to[0]} +${email.to.length - 1}` : email.to[0]}
                </Text>
              </View>
              <Text style={styles.dateText}>{formatDate(email.date)}</Text>

              <TouchableOpacity
                style={styles.showHeadersButton}
                onPress={() => setShowFullHeaders(!showFullHeaders)}
              >
                <Text style={styles.showHeadersText}>
                  {showFullHeaders ? 'Hide details' : 'Show details'}
                </Text>
              </TouchableOpacity>

              {showFullHeaders && (
                <View style={styles.fullHeaders}>
                  <Text style={styles.headerLine}>From: {email.from}</Text>
                  <Text style={styles.headerLine}>To: {email.to.join(', ')}</Text>
                  {email.cc?.length > 0 && (
                    <Text style={styles.headerLine}>Cc: {email.cc.join(', ')}</Text>
                  )}
                  <Text style={styles.headerLine}>Date: {new Date(email.date).toISOString()}</Text>
                  <Text style={styles.headerLine}>Message-ID: {email.messageId}</Text>
                </View>
              )}
            </View>
          </View>

          {/* Body */}
          <View style={styles.bodyContainer}>
            <RenderHtml
              contentWidth={width - 32}
              source={htmlSource}
              baseStyle={{ color: colors.text, fontSize: 16, lineHeight: 24 }}
              tagsStyles={{
                a: { color: colors.primary },
                p: { marginVertical: 8 },
              }}
              renderersProps={{
                a: {
                  onPress: (_, href) => {
                    if (href) Linking.openURL(href);
                  },
                },
              }}
            />
          </View>

          {/* Attachments */}
          {email.attachments?.length > 0 && (
            <View style={styles.attachmentsContainer}>
              <Text style={styles.attachmentsTitle}>
                {email.attachments.length} Attachment{email.attachments.length > 1 ? 's' : ''}
              </Text>
              <AttachmentList
                attachments={email.attachments}
                onPress={(attachment) => api.downloadAttachment(attachment.id)}
              />
            </View>
          )}

          {/* Thread Messages */}
          {thread && thread.messages.length > 1 && (
            <>
              <View style={styles.threadDivider}>
                <View style={styles.threadLine} />
                <Text style={styles.threadText}>
                  {thread.messages.length - 1} earlier message{thread.messages.length > 2 ? 's' : ''}
                </Text>
                <View style={styles.threadLine} />
              </View>
            </>
          )}
        </View>
      </ScrollView>

      {/* Bottom Toolbar */}
      <View style={styles.toolbar}>
        <TouchableOpacity style={styles.toolbarButton} onPress={handleReply}>
          <Ionicons name="arrow-undo-outline" size={24} color={colors.primary} />
          <Text style={styles.toolbarButtonText}>Reply</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.toolbarButton} onPress={handleReplyAll}>
          <Ionicons name="arrow-undo" size={24} color={colors.primary} />
          <Text style={styles.toolbarButtonText}>Reply All</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.toolbarButton} onPress={handleForward}>
          <Ionicons name="arrow-redo-outline" size={24} color={colors.primary} />
          <Text style={styles.toolbarButtonText}>Forward</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}
