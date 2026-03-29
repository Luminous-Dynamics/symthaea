// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Holochain Types
 *
 * TypeScript types matching the Rust zome entry types
 */

import type { ActionHash, AgentPubKey, EntryHash, Timestamp } from '@holochain/client';

// ==================== MESSAGES ZOME ====================

export interface Email {
  message_id: string;
  thread_id?: string;
  sender: AgentPubKey;
  encrypted_subject: Uint8Array;
  encrypted_body: Uint8Array;
  encrypted_body_html?: Uint8Array;
  encrypted_attachments: EncryptedAttachment[];
  timestamp: Timestamp;
  priority: EmailPriority;
  read_receipt_requested: boolean;
  encryption_metadata: EncryptionMetadata;
  in_reply_to?: string;
  references: string[];
}

export interface EncryptedAttachment {
  id: string;
  encrypted_filename: Uint8Array;
  encrypted_content: Uint8Array;
  content_type: string;
  size: number;
}

export type EmailPriority = 'High' | 'Normal' | 'Low';

export interface EncryptionMetadata {
  algorithm: string;
  key_version: number;
  recipient_key_hash: Uint8Array;
}

export interface EmailState {
  email_hash: ActionHash;
  owner: AgentPubKey;
  is_read: boolean;
  is_starred: boolean;
  folders: ActionHash[];
  encrypted_labels: Uint8Array[];
  snoozed_until?: Timestamp;
  is_archived: boolean;
  is_trashed: boolean;
  trashed_at?: Timestamp;
}

export interface EmailStateUpdate {
  is_read?: boolean;
  is_starred?: boolean;
  is_archived?: boolean;
  is_trashed?: boolean;
  folders?: ActionHash[];
}

export interface EmailFolder {
  id: string;
  name: string;
  folder_type: FolderType;
  parent_id?: string;
  created_at: Timestamp;
}

export type FolderType = 'System' | 'User';

export interface EmailThread {
  thread_id: string;
  encrypted_subject: Uint8Array;
  participants: AgentPubKey[];
  created_at: Timestamp;
  updated_at: Timestamp;
  email_count: number;
}

export interface ReadReceipt {
  email_hash: ActionHash;
  reader: AgentPubKey;
  read_at: Timestamp;
  signature: Uint8Array;
}

// ==================== CONTACTS ZOME ====================

export interface Contact {
  id: string;
  display_name: string;
  emails: ContactEmail[];
  phone_numbers: ContactPhone[];
  agent_pub_key?: AgentPubKey;
  avatar?: Uint8Array;
  notes?: string;
  groups: string[];
  created_at: Timestamp;
  updated_at: Timestamp;
}

export interface ContactEmail {
  email: string;
  label: string;
  is_primary: boolean;
}

export interface ContactPhone {
  number: string;
  label: string;
}

export interface ContactGroup {
  id: string;
  name: string;
  description?: string;
  color?: string;
  created_at: Timestamp;
}

// ==================== TRUST ZOME ====================

export interface TrustAttestation {
  truster: AgentPubKey;
  trustee: AgentPubKey;
  trust_level: number;
  category: TrustCategory;
  evidence?: string;
  reason?: string;
  created_at: Timestamp;
  expires_at?: Timestamp;
  signature: Uint8Array;
  revoked: boolean;
  stake?: number;
}

export type TrustCategory =
  | 'Email'
  | 'Identity'
  | 'Content'
  | 'Behavior'
  | 'Network'
  | 'Custom';

export interface TrustScore {
  agent: AgentPubKey;
  overall_score: number;
  category_scores: Record<string, number>;
  attestation_count: number;
  computed_at: Timestamp;
  confidence: number;
}

export interface CreateAttestationInput {
  trustee: AgentPubKey;
  trust_level: number;
  category: TrustCategory;
  evidence?: string;
  reason?: string;
  expires_at?: Timestamp;
  stake?: number;
}

// ==================== KEYS ZOME ====================

export interface KeyBundle {
  agent: AgentPubKey;
  kyber_public_key: Uint8Array;
  dilithium_public_key: Uint8Array;
  version: number;
  created_at: Timestamp;
  expires_at?: Timestamp;
  rotation_signature?: Uint8Array;
}

export interface KeyLookupResult {
  key_bundle: KeyBundle;
  action_hash: ActionHash;
  is_current: boolean;
  is_revoked: boolean;
}

// ==================== CAPABILITIES ZOME ====================

export interface MailboxCapability {
  id: string;
  grantor: AgentPubKey;
  grantee: AgentPubKey;
  access_type: MailboxAccessType;
  permissions: MailboxPermissions;
  restrictions?: AccessRestrictions;
  granted_at: Timestamp;
  expires_at?: Timestamp;
  revoked: boolean;
  revocation_reason?: string;
}

export type MailboxAccessType =
  | 'FullAccess'
  | 'ReadOnly'
  | 'SendAs'
  | 'ManageFilters'
  | 'ViewMetadata';

export interface MailboxPermissions {
  can_read: boolean;
  can_send: boolean;
  can_delete: boolean;
  can_move: boolean;
  can_label: boolean;
  can_archive: boolean;
}

export interface AccessRestrictions {
  folders?: string[];
  labels?: string[];
  time_window?: TimeWindow;
  rate_limit?: number;
}

export interface TimeWindow {
  start: Timestamp;
  end: Timestamp;
}

// ==================== SEARCH ZOME ====================

export interface SearchQuery {
  query: string;
  filters?: SearchFilters;
  limit?: number;
  offset?: number;
}

export interface SearchFilters {
  folders?: ActionHash[];
  labels?: string[];
  from?: AgentPubKey;
  to?: AgentPubKey;
  date_from?: Timestamp;
  date_to?: Timestamp;
  has_attachments?: boolean;
  is_read?: boolean;
  is_starred?: boolean;
}

export interface SearchResult {
  email_hash: ActionHash;
  score: number;
  highlights: SearchHighlight[];
  matched_terms: string[];
}

export interface SearchHighlight {
  field: string;
  snippet: string;
  positions: [number, number][];
}

// ==================== BACKUP ZOME ====================

export interface BackupManifest {
  backup_id: string;
  agent: AgentPubKey;
  backup_type: BackupType;
  contents: BackupContents;
  created_at: Timestamp;
  size_bytes: number;
  entry_count: number;
  checksum: string;
  encryption?: BackupEncryption;
  status: BackupStatus;
  metadata: Record<string, string>;
}

export type BackupType = 'Full' | 'Incremental' | 'Selective';
export type BackupStatus = 'InProgress' | 'Completed' | 'Failed' | 'Verified';

export interface BackupContents {
  emails: boolean;
  contacts: boolean;
  folders: boolean;
  labels: boolean;
  settings: boolean;
  keys: boolean;
  trust_data: boolean;
}

export interface BackupEncryption {
  algorithm: string;
  key_derivation: string;
  salt: Uint8Array;
}

// ==================== SYNC ZOME ====================

export interface SyncState {
  agent: AgentPubKey;
  vector_clock: VectorClock;
  last_sync: Timestamp;
  pending_operations: number;
}

export interface VectorClock {
  clocks: Record<string, number>;
}

export interface SyncOperation {
  id: string;
  operation_type: OperationType;
  entity_hash: ActionHash;
  timestamp: Timestamp;
  data?: Uint8Array;
}

export type OperationType = 'Create' | 'Update' | 'Delete';

// ==================== SIGNALS ====================

export interface MailSignal {
  type: 'EmailReceived' | 'EmailStateChanged' | 'ReadReceiptReceived' | 'TypingIndicator';
  data: unknown;
}

export interface TrustSignal {
  type: 'AttestationReceived' | 'TrustScoreUpdated' | 'IntroductionReceived';
  data: unknown;
}

export interface CapabilitySignal {
  type: 'CapabilityGranted' | 'CapabilityRevoked' | 'AddedToSharedMailbox';
  data: unknown;
}

// ==================== INPUT TYPES ====================

export interface SendEmailInput {
  recipients: AgentPubKey[];
  encrypted_subject: Uint8Array;
  encrypted_body: Uint8Array;
  encrypted_body_html?: Uint8Array;
  encrypted_attachments: EncryptedAttachment[];
  priority: EmailPriority;
  read_receipt_requested: boolean;
  in_reply_to?: string;
  thread_id?: string;
  encryption_metadata: EncryptionMetadata;
}

export interface SendEmailOutput {
  email_hash: ActionHash;
  delivered_to: AgentPubKey[];
  failed_deliveries: [AgentPubKey, string][];
}
