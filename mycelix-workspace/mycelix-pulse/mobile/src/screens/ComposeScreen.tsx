// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Compose Email Screen
 *
 * Full-featured email composition with attachments, trust indicators
 */

import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import { useMutation, useQuery } from '@tanstack/react-query';

import { useTheme } from '../providers/ThemeProvider';
import { RootStackParamList } from '../types/navigation';
import { api } from '../services/api';
import { ContactPicker } from '../components/ContactPicker';
import { AttachmentList } from '../components/AttachmentList';
import { TrustIndicator } from '../components/TrustIndicator';

type ComposeScreenRouteProp = RouteProp<RootStackParamList, 'Compose'>;
type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
  uri: string;
}

export function ComposeScreen() {
  const { colors } = useTheme();
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<ComposeScreenRouteProp>();

  const { replyTo, forwardFrom, draftId } = route.params || {};

  const [to, setTo] = useState<string[]>(replyTo?.from ? [replyTo.from] : []);
  const [cc, setCc] = useState<string[]>([]);
  const [bcc, setBcc] = useState<string[]>([]);
  const [subject, setSubject] = useState(
    replyTo ? `Re: ${replyTo.subject}` :
    forwardFrom ? `Fwd: ${forwardFrom.subject}` : ''
  );
  const [body, setBody] = useState(
    forwardFrom ? `\n\n---------- Forwarded message ----------\n${forwardFrom.body}` : ''
  );
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [showCcBcc, setShowCcBcc] = useState(false);
  const [showContactPicker, setShowContactPicker] = useState<'to' | 'cc' | 'bcc' | null>(null);

  const bodyInputRef = useRef<TextInput>(null);

  // Check trust for recipients
  const { data: recipientTrust } = useQuery({
    queryKey: ['recipientTrust', to],
    queryFn: () => api.getTrustScores(to),
    enabled: to.length > 0,
  });

  // Send mutation
  const sendMutation = useMutation({
    mutationFn: () => api.sendEmail({
      to,
      cc,
      bcc,
      subject,
      body,
      attachments: attachments.map(a => a.id),
      replyToId: replyTo?.id,
    }),
    onSuccess: () => {
      navigation.goBack();
    },
    onError: (error: Error) => {
      Alert.alert('Error', error.message || 'Failed to send email');
    },
  });

  // Save draft mutation
  const saveDraftMutation = useMutation({
    mutationFn: () => api.saveDraft({
      id: draftId,
      to,
      cc,
      bcc,
      subject,
      body,
      attachments: attachments.map(a => a.id),
    }),
    onSuccess: () => {
      Alert.alert('Saved', 'Draft saved successfully');
    },
  });

  const handleSend = useCallback(() => {
    if (to.length === 0) {
      Alert.alert('Error', 'Please add at least one recipient');
      return;
    }
    if (!subject.trim()) {
      Alert.alert(
        'No Subject',
        'Send without a subject?',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Send', onPress: () => sendMutation.mutate() },
        ]
      );
      return;
    }
    sendMutation.mutate();
  }, [to, subject, sendMutation]);

  const handleAttach = useCallback(async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: '*/*',
        multiple: true,
      });

      if (result.canceled) return;

      const newAttachments: Attachment[] = result.assets.map(asset => ({
        id: Math.random().toString(36).substring(7),
        name: asset.name,
        size: asset.size || 0,
        type: asset.mimeType || 'application/octet-stream',
        uri: asset.uri,
      }));

      setAttachments(prev => [...prev, ...newAttachments]);
    } catch (error) {
      Alert.alert('Error', 'Failed to attach file');
    }
  }, []);

  const handleRemoveAttachment = useCallback((id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  }, []);

  const handleAddRecipient = useCallback((field: 'to' | 'cc' | 'bcc', email: string) => {
    const setter = field === 'to' ? setTo : field === 'cc' ? setCc : setBcc;
    setter(prev => [...new Set([...prev, email])]);
  }, []);

  const handleRemoveRecipient = useCallback((field: 'to' | 'cc' | 'bcc', email: string) => {
    const setter = field === 'to' ? setTo : field === 'cc' ? setCc : setBcc;
    setter(prev => prev.filter(e => e !== email));
  }, []);

  const handleDiscard = useCallback(() => {
    if (to.length > 0 || subject || body) {
      Alert.alert(
        'Discard Draft?',
        'Your changes will be lost.',
        [
          { text: 'Keep Editing', style: 'cancel' },
          { text: 'Save Draft', onPress: () => saveDraftMutation.mutate() },
          { text: 'Discard', style: 'destructive', onPress: () => navigation.goBack() },
        ]
      );
    } else {
      navigation.goBack();
    }
  }, [to, subject, body, navigation, saveDraftMutation]);

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
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
    headerButton: {
      padding: 8,
    },
    headerTitle: {
      fontSize: 18,
      fontWeight: '600',
      color: colors.text,
    },
    sendButton: {
      backgroundColor: colors.primary,
      paddingHorizontal: 16,
      paddingVertical: 8,
      borderRadius: 20,
      flexDirection: 'row',
      alignItems: 'center',
    },
    sendButtonDisabled: {
      opacity: 0.5,
    },
    sendButtonText: {
      color: '#fff',
      fontWeight: '600',
      marginLeft: 4,
    },
    form: {
      flex: 1,
    },
    fieldRow: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 16,
      paddingVertical: 12,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    fieldLabel: {
      width: 60,
      color: colors.textSecondary,
      fontSize: 16,
    },
    fieldContent: {
      flex: 1,
      flexDirection: 'row',
      flexWrap: 'wrap',
      alignItems: 'center',
    },
    recipientChip: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: colors.surface,
      borderRadius: 16,
      paddingVertical: 4,
      paddingHorizontal: 8,
      marginRight: 8,
      marginBottom: 4,
    },
    recipientText: {
      color: colors.text,
      fontSize: 14,
    },
    recipientRemove: {
      marginLeft: 4,
    },
    fieldInput: {
      flex: 1,
      fontSize: 16,
      color: colors.text,
      minWidth: 100,
    },
    addButton: {
      padding: 4,
    },
    ccBccToggle: {
      color: colors.primary,
      fontSize: 14,
    },
    subjectInput: {
      flex: 1,
      fontSize: 16,
      color: colors.text,
    },
    bodyContainer: {
      flex: 1,
      padding: 16,
    },
    bodyInput: {
      flex: 1,
      fontSize: 16,
      color: colors.text,
      textAlignVertical: 'top',
    },
    attachmentsContainer: {
      padding: 16,
      borderTopWidth: 1,
      borderTopColor: colors.border,
    },
    toolbar: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: 12,
      borderTopWidth: 1,
      borderTopColor: colors.border,
      backgroundColor: colors.surface,
    },
    toolbarButton: {
      padding: 8,
      marginRight: 16,
    },
  });

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.headerButton} onPress={handleDiscard}>
            <Ionicons name="close" size={24} color={colors.text} />
          </TouchableOpacity>

          <Text style={styles.headerTitle}>
            {replyTo ? 'Reply' : forwardFrom ? 'Forward' : 'New Email'}
          </Text>

          <TouchableOpacity
            style={[styles.sendButton, sendMutation.isPending && styles.sendButtonDisabled]}
            onPress={handleSend}
            disabled={sendMutation.isPending}
          >
            {sendMutation.isPending ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <>
                <Ionicons name="send" size={18} color="#fff" />
                <Text style={styles.sendButtonText}>Send</Text>
              </>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.form} keyboardShouldPersistTaps="handled">
          {/* To Field */}
          <View style={styles.fieldRow}>
            <Text style={styles.fieldLabel}>To:</Text>
            <View style={styles.fieldContent}>
              {to.map(email => (
                <View key={email} style={styles.recipientChip}>
                  <TrustIndicator
                    score={recipientTrust?.[email]?.score}
                    size="small"
                  />
                  <Text style={styles.recipientText}>{email}</Text>
                  <TouchableOpacity
                    style={styles.recipientRemove}
                    onPress={() => handleRemoveRecipient('to', email)}
                  >
                    <Ionicons name="close-circle" size={18} color={colors.textSecondary} />
                  </TouchableOpacity>
                </View>
              ))}
              <TouchableOpacity
                style={styles.addButton}
                onPress={() => setShowContactPicker('to')}
              >
                <Ionicons name="add-circle-outline" size={24} color={colors.primary} />
              </TouchableOpacity>
            </View>
            {!showCcBcc && (
              <TouchableOpacity onPress={() => setShowCcBcc(true)}>
                <Text style={styles.ccBccToggle}>Cc/Bcc</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Cc Field */}
          {showCcBcc && (
            <View style={styles.fieldRow}>
              <Text style={styles.fieldLabel}>Cc:</Text>
              <View style={styles.fieldContent}>
                {cc.map(email => (
                  <View key={email} style={styles.recipientChip}>
                    <Text style={styles.recipientText}>{email}</Text>
                    <TouchableOpacity
                      style={styles.recipientRemove}
                      onPress={() => handleRemoveRecipient('cc', email)}
                    >
                      <Ionicons name="close-circle" size={18} color={colors.textSecondary} />
                    </TouchableOpacity>
                  </View>
                ))}
                <TouchableOpacity
                  style={styles.addButton}
                  onPress={() => setShowContactPicker('cc')}
                >
                  <Ionicons name="add-circle-outline" size={24} color={colors.primary} />
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* Bcc Field */}
          {showCcBcc && (
            <View style={styles.fieldRow}>
              <Text style={styles.fieldLabel}>Bcc:</Text>
              <View style={styles.fieldContent}>
                {bcc.map(email => (
                  <View key={email} style={styles.recipientChip}>
                    <Text style={styles.recipientText}>{email}</Text>
                    <TouchableOpacity
                      style={styles.recipientRemove}
                      onPress={() => handleRemoveRecipient('bcc', email)}
                    >
                      <Ionicons name="close-circle" size={18} color={colors.textSecondary} />
                    </TouchableOpacity>
                  </View>
                ))}
                <TouchableOpacity
                  style={styles.addButton}
                  onPress={() => setShowContactPicker('bcc')}
                >
                  <Ionicons name="add-circle-outline" size={24} color={colors.primary} />
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* Subject Field */}
          <View style={styles.fieldRow}>
            <Text style={styles.fieldLabel}>Subject:</Text>
            <TextInput
              style={styles.subjectInput}
              value={subject}
              onChangeText={setSubject}
              placeholder="Subject"
              placeholderTextColor={colors.textSecondary}
              returnKeyType="next"
              onSubmitEditing={() => bodyInputRef.current?.focus()}
            />
          </View>

          {/* Body */}
          <View style={styles.bodyContainer}>
            <TextInput
              ref={bodyInputRef}
              style={styles.bodyInput}
              value={body}
              onChangeText={setBody}
              placeholder="Compose email..."
              placeholderTextColor={colors.textSecondary}
              multiline
              textAlignVertical="top"
            />
          </View>

          {/* Attachments */}
          {attachments.length > 0 && (
            <View style={styles.attachmentsContainer}>
              <AttachmentList
                attachments={attachments}
                onRemove={handleRemoveAttachment}
                editable
              />
            </View>
          )}
        </ScrollView>

        {/* Toolbar */}
        <View style={styles.toolbar}>
          <TouchableOpacity style={styles.toolbarButton} onPress={handleAttach}>
            <Ionicons name="attach" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.toolbarButton}>
            <Ionicons name="camera-outline" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.toolbarButton}>
            <Ionicons name="link-outline" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.toolbarButton}
            onPress={() => saveDraftMutation.mutate()}
          >
            <Ionicons name="save-outline" size={24} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>

      {/* Contact Picker Modal */}
      {showContactPicker && (
        <ContactPicker
          visible={!!showContactPicker}
          onClose={() => setShowContactPicker(null)}
          onSelect={(email) => {
            handleAddRecipient(showContactPicker, email);
            setShowContactPicker(null);
          }}
        />
      )}
    </SafeAreaView>
  );
}
