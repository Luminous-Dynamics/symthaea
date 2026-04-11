// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Holochain Types
 * Auto-generated TypeScript types matching Rust structures
 */

import type { ActionHash, AgentPubKey, DnaHash, Timestamp } from '@holochain/client';

// ==================== COMMON ====================

export type HoloHash = Uint8Array;

export interface SignalPayload<T = unknown> {
  type: string;
  data: T;
}

// ==================== MESSAGES ZOME ====================

export type CryptoSuite =
  | 'X25519ChaCha20'
  | 'KyberAesGcm'
  | 'HybridKyberX25519';

export type EmailPriority = 'Low' | 'Normal' | 'High' | 'Urgent';

export type EmailState =
  | 'Unread'
  | 'Read'
  | 'Archived'
  | 'Deleted'
  | 'Spam'
  | 'Draft';

export interface EncryptedEmail {
  sender: AgentPubKey;
  recipient: AgentPubKey;
  encrypted_subject: Uint8Array;
  encrypted_body: Uint8Array;
  encrypted_attachments: Uint8Array;
  ephemeral_pubkey: Uint8Array;
  nonce: Uint8Array;
  signature: Uint8Array;
  crypto_suite: CryptoSuite;
  message_id: string;
  in_reply_to: string | null;
  references: string[];
  timestamp: Timestamp;
  priority: EmailPriority;
  read_receipt_requested: boolean;
  expires_at: Timestamp | null;
}

export interface DecryptedEmail {
  hash: ActionHash;
  sender: AgentPubKey;
  recipient: AgentPubKey;
  subject: string;
  body: string;
  attachments: EmailAttachment[];
  message_id: string;
  in_reply_to: string | null;
  references: string[];
  timestamp: Date;
  priority: EmailPriority;
  state: EmailState;
  labels: string[];
  folder: ActionHash | null;
  thread_id: string | null;
}

export interface EmailAttachment {
  filename: string;
  mime_type: string;
  size: number;
  encrypted_data: Uint8Array;
  hash: string;
}

export interface EmailFolder {
  id: string;
  name: string;
  parent: ActionHash | null;
  color: string | null;
  icon: string | null;
  sort_order: number;
  is_system: boolean;
}

export interface EmailThread {
  thread_id: string;
  subject: string;
  participants: AgentPubKey[];
  email_count: number;
  unread_count: number;
  last_activity: Timestamp;
  labels: string[];
}

export interface SendEmailInput {
  recipient: AgentPubKey;
  subject: string;
  body: string;
  attachments?: EmailAttachment[];
  priority?: EmailPriority;
  in_reply_to?: string;
  request_read_receipt?: boolean;
  encrypt_with?: CryptoSuite;
}

export interface SendEmailOutput {
  email_hash: ActionHash;
  message_id: string;
  delivered: boolean;
}

// ==================== TRUST ZOME (MATL) ====================

export type TrustCategory =
  | 'Identity'
  | 'Communication'
  | 'FileSharing'
  | 'Scheduling'
  | 'Financial'
  | 'CredentialIssuer'
  | 'Organization'
  | 'Personal'
  | 'Professional'
  | { Custom: string };

export type TrustLevel =
  | 'Untrusted'
  | 'Unknown'
  | 'Minimal'
  | 'Low'
  | 'Medium'
  | 'High'
  | 'Verified'
  | 'Absolute';

export type EvidenceType =
  | 'DirectInteraction'
  | 'SharedSecret'
  | 'VideoVerification'
  | 'InPersonMeeting'
  | 'OrganizationMembership'
  | 'CredentialPresentation'
  | 'SocialProof'
  | 'KeyCeremony'
  | 'BiometricMatch'
  | { Custom: string };

export interface TrustAttestation {
  id: string;
  attestor: AgentPubKey;
  subject: AgentPubKey;
  category: TrustCategory;
  level: TrustLevel;
  confidence: number; // 0.0 - 1.0
  evidence: TrustEvidence[];
  context: string | null;
  stake_amount: number;
  created_at: Timestamp;
  expires_at: Timestamp | null;
  revoked: boolean;
  revocation_reason: string | null;
}

export interface TrustEvidence {
  evidence_type: EvidenceType;
  description: string;
  timestamp: Timestamp;
  verifiable_claim: Uint8Array | null;
  witnesses: AgentPubKey[];
}

export interface TrustScore {
  subject: AgentPubKey;
  category: TrustCategory;
  direct_score: number;
  transitive_score: number;
  combined_score: number;
  confidence: number;
  attestation_count: number;
  last_updated: Timestamp;
  byzantine_flags: ByzantineFlag[];
}

export type ByzantineFlagType =
  | 'SybilSuspicion'
  | 'CollusionPattern'
  | 'TrustVolatility'
  | 'InconsistentAttestations'
  | 'SelfDealing'
  | 'AttestationFarming';

export interface ByzantineFlag {
  flag_type: ByzantineFlagType;
  severity: number;
  detected_at: Timestamp;
  evidence: string;
}

export interface CreateAttestationInput {
  subject: AgentPubKey;
  category: TrustCategory;
  level: TrustLevel;
  confidence: number;
  evidence?: TrustEvidence[];
  context?: string;
  stake_amount?: number;
  expires_in_days?: number;
}

export interface GetTrustScoreInput {
  subject: AgentPubKey;
  category?: TrustCategory;
  include_transitive?: boolean;
  max_depth?: number;
}

// ==================== CAPABILITIES ZOME ====================

export type MailboxAccessType =
  | 'FullAccess'
  | 'ReadOnly'
  | 'SendAs'
  | { FolderAccess: { folder_hash: ActionHash } }
  | { ThreadAccess: { thread_id: string } }
  | 'OutOfOffice'
  | 'OrganizationAdmin'
  | { Custom: string };

export interface MailboxPermissions {
  can_read: boolean;
  can_send: boolean;
  can_delete: boolean;
  can_move: boolean;
  can_create_folders: boolean;
  can_manage_labels: boolean;
  can_view_attachments: boolean;
  can_download_attachments: boolean;
  can_manage_rules: boolean;
  can_delegate: boolean;
  can_modify_settings: boolean;
  can_view_trust: boolean;
  can_modify_trust: boolean;
}

export interface AccessRestrictions {
  folder_whitelist: ActionHash[] | null;
  folder_blacklist: ActionHash[] | null;
  sender_whitelist: string[] | null;
  max_emails: number | null;
  date_from: Timestamp | null;
  date_to: Timestamp | null;
  network_restrictions: string[] | null;
  require_2fa: boolean;
  audit_required: boolean;
}

export interface MailboxCapability {
  id: string;
  grantor: AgentPubKey;
  grantee: AgentPubKey;
  access_type: MailboxAccessType;
  permissions: MailboxPermissions;
  restrictions: AccessRestrictions | null;
  granted_at: Timestamp;
  expires_at: Timestamp | null;
  revoked: boolean;
  revocation_reason: string | null;
}

export type SharedMailboxRole = 'Owner' | 'Admin' | 'Editor' | 'Contributor' | 'Viewer';

export interface SharedMailbox {
  id: string;
  name: string;
  owner: AgentPubKey;
  email_address: string;
  members: SharedMailboxMember[];
  created_at: Timestamp;
  is_active: boolean;
}

export interface SharedMailboxMember {
  agent: AgentPubKey;
  role: SharedMailboxRole;
  permissions: MailboxPermissions;
  added_at: Timestamp;
  added_by: AgentPubKey;
}

export interface GrantCapabilityInput {
  grantee: AgentPubKey;
  access_type: MailboxAccessType;
  permissions: MailboxPermissions;
  restrictions?: AccessRestrictions;
  expires_in_days?: number;
}

// ==================== SYNC ZOME (CRDT) ====================

export interface VectorClock {
  clocks: Array<[AgentPubKey, number]>;
}

export interface SyncState {
  agent: AgentPubKey;
  vector_clock: VectorClock;
  last_sync: Timestamp;
  sync_version: number;
  pending_ops: number;
  is_online: boolean;
}

export type SyncOpType =
  | 'EmailReceived'
  | 'EmailRead'
  | 'EmailDeleted'
  | 'EmailMoved'
  | 'EmailFlagged'
  | 'EmailLabeled'
  | 'FolderCreated'
  | 'FolderRenamed'
  | 'FolderDeleted'
  | 'DraftCreated'
  | 'DraftUpdated'
  | 'DraftSent'
  | 'SettingChanged'
  | { Custom: string };

export type ConflictType =
  | 'ConcurrentEdit'
  | 'DeleteUpdate'
  | 'ConcurrentMove'
  | 'CausalityViolation'
  | 'VersionMismatch';

export interface SyncConflict {
  conflict_id: string;
  operations: string[];
  conflict_type: ConflictType;
  detected_at: Timestamp;
  resolution: ConflictResolution;
}

export type ConflictResolution =
  | 'Pending'
  | { AutoResolved: { strategy: string } }
  | { LastWriterWins: { winner: AgentPubKey } }
  | { ManualResolution: { resolver: AgentPubKey } }
  | 'Merged'
  | 'BothKept';

export interface SyncResult {
  operations_sent: number;
  operations_received: number;
  conflicts_detected: number;
  conflicts_resolved: number;
  new_vector_clock: VectorClock;
}

// ==================== FEDERATION ZOME ====================

export type NetworkType =
  | 'HolochainNative'
  | 'SmtpBridge'
  | 'MatrixBridge'
  | 'ActivityPubBridge'
  | 'XmppBridge'
  | { CustomBridge: { protocol: string } }
  | 'OrganizationPrivate'
  | 'PublicOpen';

export interface FederatedNetwork {
  network_id: string;
  dna_hash: DnaHash;
  name: string;
  domain: string;
  network_type: NetworkType;
  is_active: boolean;
  capabilities: NetworkCapabilities;
}

export interface NetworkCapabilities {
  supports_e2e: boolean;
  supports_pqc: boolean;
  supports_attachments: boolean;
  max_attachment_size: number | null;
  supports_read_receipts: boolean;
  supports_threading: boolean;
  rate_limit_per_hour: number | null;
}

export type DeliveryStatus =
  | 'Queued'
  | { InTransit: { current_hop: string } }
  | { Delivered: { delivered_at: Timestamp } }
  | { DeliveredToRecipient: { delivered_at: Timestamp } }
  | { Failed: { error: string; failed_at: Timestamp } }
  | { Bounced: { reason: string; bounced_at: Timestamp } }
  | 'Expired';

export interface FederatedEnvelope {
  envelope_id: string;
  source_network: string;
  source_agent: string;
  dest_network: string;
  dest_agent: string;
  timestamp: Timestamp;
  status: DeliveryStatus;
  hop_count: number;
}

export interface SendFederatedInput {
  dest_network: string;
  dest_agent: string;
  payload: Uint8Array;
  priority?: 'Low' | 'Normal' | 'High' | 'Urgent';
  ttl?: number;
}

// ==================== SIGNALS ====================

export type MessageSignal =
  | { type: 'EmailReceived'; email_hash: ActionHash; sender: AgentPubKey; preview: string }
  | { type: 'DeliveryConfirmed'; email_hash: ActionHash; recipient: AgentPubKey }
  | { type: 'ReadReceiptReceived'; email_hash: ActionHash; reader: AgentPubKey }
  | { type: 'EmailStateChanged'; email_hash: ActionHash; new_state: EmailState }
  | { type: 'ThreadActivity'; thread_id: string; activity_type: string }
  | { type: 'TypingIndicator'; from: AgentPubKey; thread_id: string };

export type TrustSignal =
  | { type: 'AttestationCreated'; attestation_id: string; subject: AgentPubKey }
  | { type: 'AttestationRevoked'; attestation_id: string; reason: string }
  | { type: 'TrustScoreChanged'; subject: AgentPubKey; category: TrustCategory; new_score: number }
  | { type: 'ByzantineFlagRaised'; subject: AgentPubKey; flag_type: ByzantineFlagType };

export type SyncSignal =
  | { type: 'SyncStateChanged'; vector_clock: VectorClock; pending_ops: number }
  | { type: 'OperationReceived'; op_id: string; from_agent: AgentPubKey }
  | { type: 'ConflictDetected'; conflict_id: string; conflict_type: ConflictType }
  | { type: 'ConflictResolved'; conflict_id: string; resolution: ConflictResolution }
  | { type: 'Online' }
  | { type: 'Offline' }
  | { type: 'SyncProgress'; synced: number; total: number; peer: AgentPubKey };

export type FederationSignal =
  | { type: 'NetworkDiscovered'; network_id: string; network_type: NetworkType }
  | { type: 'EnvelopeReceived'; envelope_id: string; from_network: string }
  | { type: 'EnvelopeDelivered'; envelope_id: string; to_network: string }
  | { type: 'DeliveryFailed'; envelope_id: string; error: string }
  | { type: 'BridgeStatusChanged'; bridge_agent: AgentPubKey; is_active: boolean };

export type MycelixSignal = MessageSignal | TrustSignal | SyncSignal | FederationSignal;
