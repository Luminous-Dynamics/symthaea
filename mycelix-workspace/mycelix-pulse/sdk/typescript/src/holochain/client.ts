// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Holochain Client
 *
 * TypeScript client for direct Holochain zome interactions
 */

import {
  AppClient,
  AppWebsocket,
  ActionHash,
  AgentPubKey,
  CellId,
  encodeHashToBase64,
  decodeHashFromBase64,
} from '@holochain/client';

import type {
  Email,
  EmailState,
  EmailStateUpdate,
  EmailFolder,
  EmailThread,
  Contact,
  ContactGroup,
  TrustAttestation,
  TrustScore,
  CreateAttestationInput,
  KeyBundle,
  KeyLookupResult,
  MailboxCapability,
  SearchQuery,
  SearchResult,
  BackupManifest,
  SyncState,
  MailSignal,
  TrustSignal,
  CapabilitySignal,
  SendEmailInput,
  SendEmailOutput,
} from './types';

export interface MycelixHolochainConfig {
  appId: string;
  roleName?: string;
  url?: string;
}

type SignalHandler = (signal: MailSignal | TrustSignal | CapabilitySignal) => void;

/**
 * Main Holochain Client for Mycelix Mail
 */
export class MycelixHolochainClient {
  private client: AppClient | null = null;
  private cellId: CellId | null = null;
  private config: Required<MycelixHolochainConfig>;
  private signalHandlers: SignalHandler[] = [];

  public readonly messages: MessagesZomeClient;
  public readonly contacts: ContactsZomeClient;
  public readonly trust: TrustZomeClient;
  public readonly keys: KeysZomeClient;
  public readonly capabilities: CapabilitiesZomeClient;
  public readonly search: SearchZomeClient;
  public readonly backup: BackupZomeClient;
  public readonly sync: SyncZomeClient;

  constructor(config: MycelixHolochainConfig) {
    this.config = {
      appId: config.appId,
      roleName: config.roleName || 'mycelix_mail',
      url: config.url || 'ws://localhost:8888',
    };

    this.messages = new MessagesZomeClient(this);
    this.contacts = new ContactsZomeClient(this);
    this.trust = new TrustZomeClient(this);
    this.keys = new KeysZomeClient(this);
    this.capabilities = new CapabilitiesZomeClient(this);
    this.search = new SearchZomeClient(this);
    this.backup = new BackupZomeClient(this);
    this.sync = new SyncZomeClient(this);
  }

  /**
   * Connect to Holochain conductor
   */
  async connect(): Promise<void> {
    this.client = await AppWebsocket.connect(this.config.url);

    // Get app info and cell ID
    const appInfo = await this.client.appInfo();
    if (!appInfo) {
      throw new Error('Failed to get app info');
    }

    const cells = appInfo.cell_info[this.config.roleName];
    if (!cells || cells.length === 0) {
      throw new Error(`No cells found for role: ${this.config.roleName}`);
    }

    const cell = cells[0];
    if ('provisioned' in cell) {
      this.cellId = cell.provisioned.cell_id;
    } else if ('cloned' in cell) {
      this.cellId = cell.cloned.cell_id;
    } else {
      throw new Error('No valid cell found');
    }

    // Set up signal handler
    this.client.on('signal', (signal) => {
      if ('App' in signal) {
        const payload = signal.App.payload;
        this.signalHandlers.forEach((handler) => handler(payload as MailSignal));
      }
    });
  }

  /**
   * Disconnect from Holochain conductor
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
      this.cellId = null;
    }
  }

  /**
   * Get the current agent's public key
   */
  async getMyAgentPubKey(): Promise<AgentPubKey> {
    if (!this.cellId) {
      throw new Error('Not connected');
    }
    return this.cellId[1];
  }

  /**
   * Register a signal handler
   */
  onSignal(handler: SignalHandler): () => void {
    this.signalHandlers.push(handler);
    return () => {
      const index = this.signalHandlers.indexOf(handler);
      if (index > -1) {
        this.signalHandlers.splice(index, 1);
      }
    };
  }

  /**
   * Call a zome function
   */
  async callZome<T>(
    zomeName: string,
    fnName: string,
    payload: unknown = null
  ): Promise<T> {
    if (!this.client || !this.cellId) {
      throw new Error('Not connected. Call connect() first.');
    }

    const result = await this.client.callZome({
      cell_id: this.cellId,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    });

    return result as T;
  }
}

/**
 * Messages Zome Client
 */
class MessagesZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async sendEmail(input: SendEmailInput): Promise<SendEmailOutput> {
    return this.client.callZome('messages', 'send_email', input);
  }

  async getEmail(hash: ActionHash): Promise<Email | null> {
    return this.client.callZome('messages', 'get_email', hash);
  }

  async getInbox(): Promise<Email[]> {
    return this.client.callZome('messages', 'get_inbox', null);
  }

  async getSent(): Promise<Email[]> {
    return this.client.callZome('messages', 'get_sent', null);
  }

  async updateEmailState(
    emailHash: ActionHash,
    update: EmailStateUpdate
  ): Promise<ActionHash> {
    return this.client.callZome('messages', 'update_email_state', {
      email_hash: emailHash,
      ...update,
    });
  }

  async markAsRead(
    emailHash: ActionHash,
    sendReceipt: boolean = false
  ): Promise<ActionHash | null> {
    return this.client.callZome('messages', 'mark_as_read', [emailHash, sendReceipt]);
  }

  async getThread(threadId: string): Promise<Email[]> {
    return this.client.callZome('messages', 'get_thread', threadId);
  }

  async getFolders(): Promise<EmailFolder[]> {
    return this.client.callZome('messages', 'get_folders', null);
  }

  async createFolder(name: string, parentId?: string): Promise<ActionHash> {
    return this.client.callZome('messages', 'create_folder', { name, parent_id: parentId });
  }

  async deleteFolder(folderId: string): Promise<void> {
    return this.client.callZome('messages', 'delete_folder', folderId);
  }

  async moveToFolder(emailHash: ActionHash, folderId: ActionHash): Promise<void> {
    return this.client.callZome('messages', 'move_to_folder', [emailHash, folderId]);
  }

  async sendTypingIndicator(recipients: AgentPubKey[], threadId?: string): Promise<void> {
    return this.client.callZome('messages', 'send_typing_indicator', [recipients, threadId]);
  }
}

/**
 * Contacts Zome Client
 */
class ContactsZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async createContact(contact: Omit<Contact, 'created_at' | 'updated_at'>): Promise<ActionHash> {
    return this.client.callZome('contacts', 'create_contact', contact);
  }

  async getContact(hash: ActionHash): Promise<Contact | null> {
    return this.client.callZome('contacts', 'get_contact', hash);
  }

  async getAllContacts(): Promise<Contact[]> {
    return this.client.callZome('contacts', 'get_all_contacts', null);
  }

  async updateContact(hash: ActionHash, contact: Contact): Promise<ActionHash> {
    return this.client.callZome('contacts', 'update_contact', { hash, contact });
  }

  async deleteContact(hash: ActionHash): Promise<ActionHash> {
    return this.client.callZome('contacts', 'delete_contact', hash);
  }

  async getContactByEmail(email: string): Promise<Contact | null> {
    return this.client.callZome('contacts', 'get_contact_by_email', email);
  }

  async getContactByAgent(agent: AgentPubKey): Promise<Contact | null> {
    return this.client.callZome('contacts', 'get_contact_by_agent', agent);
  }

  async createGroup(group: Omit<ContactGroup, 'created_at'>): Promise<ActionHash> {
    return this.client.callZome('contacts', 'create_group', group);
  }

  async getAllGroups(): Promise<ContactGroup[]> {
    return this.client.callZome('contacts', 'get_all_groups', null);
  }

  async addToGroup(contactHash: ActionHash, groupId: string): Promise<ActionHash> {
    return this.client.callZome('contacts', 'add_to_group', {
      contact_hash: contactHash,
      group_id: groupId,
    });
  }

  async removeFromGroup(contactHash: ActionHash, groupId: string): Promise<void> {
    return this.client.callZome('contacts', 'remove_from_group', {
      contact_hash: contactHash,
      group_id: groupId,
    });
  }

  async blockContact(hash: ActionHash): Promise<ActionHash> {
    return this.client.callZome('contacts', 'block_contact', hash);
  }

  async unblockContact(hash: ActionHash): Promise<void> {
    return this.client.callZome('contacts', 'unblock_contact', hash);
  }

  async getBlockedContacts(): Promise<ActionHash[]> {
    return this.client.callZome('contacts', 'get_blocked_contacts', null);
  }
}

/**
 * Trust Zome Client
 */
class TrustZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async createAttestation(input: CreateAttestationInput): Promise<ActionHash> {
    return this.client.callZome('trust', 'create_attestation', input);
  }

  async revokeAttestation(hash: ActionHash): Promise<ActionHash> {
    return this.client.callZome('trust', 'revoke_attestation', hash);
  }

  async getAttestation(hash: ActionHash): Promise<TrustAttestation | null> {
    return this.client.callZome('trust', 'get_attestation', hash);
  }

  async getGivenAttestations(): Promise<TrustAttestation[]> {
    return this.client.callZome('trust', 'get_given_attestations', null);
  }

  async getReceivedAttestations(): Promise<TrustAttestation[]> {
    return this.client.callZome('trust', 'get_received_attestations', null);
  }

  async computeTrustScore(agent: AgentPubKey): Promise<TrustScore> {
    return this.client.callZome('trust', 'compute_trust_score', agent);
  }

  async getTrustScore(agent: AgentPubKey): Promise<TrustScore | null> {
    return this.client.callZome('trust', 'get_trust_score', agent);
  }

  async findTrustPath(from: AgentPubKey, to: AgentPubKey): Promise<AgentPubKey[]> {
    return this.client.callZome('trust', 'find_trust_path', [from, to]);
  }

  async createIntroduction(
    target: AgentPubKey,
    introduced: AgentPubKey,
    recommendation: number,
    message?: string
  ): Promise<ActionHash> {
    return this.client.callZome('trust', 'create_introduction', {
      target,
      introduced,
      recommendation_level: recommendation,
      message,
    });
  }
}

/**
 * Keys Zome Client
 */
class KeysZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async createKeyBundle(
    kyberPublicKey: Uint8Array,
    dilithiumPublicKey: Uint8Array,
    expiresAt?: number,
    email?: string
  ): Promise<ActionHash> {
    return this.client.callZome('keys', 'create_key_bundle', {
      kyber_public_key: kyberPublicKey,
      dilithium_public_key: dilithiumPublicKey,
      expires_at: expiresAt,
      email,
    });
  }

  async rotateKeys(
    newKyberPublicKey: Uint8Array,
    newDilithiumPublicKey: Uint8Array,
    rotationSignature: Uint8Array,
    expiresAt?: number
  ): Promise<ActionHash> {
    return this.client.callZome('keys', 'rotate_keys', {
      new_kyber_public_key: newKyberPublicKey,
      new_dilithium_public_key: newDilithiumPublicKey,
      rotation_signature: rotationSignature,
      expires_at: expiresAt,
    });
  }

  async getCurrentKeyBundle(agent: AgentPubKey): Promise<KeyLookupResult | null> {
    return this.client.callZome('keys', 'get_current_key_bundle', agent);
  }

  async getKeyBundleByEmail(email: string): Promise<KeyLookupResult | null> {
    return this.client.callZome('keys', 'get_key_bundle_by_email', email);
  }

  async revokeKey(
    keyBundleHash: ActionHash,
    reason: string,
    revocationSignature: Uint8Array
  ): Promise<ActionHash> {
    return this.client.callZome('keys', 'revoke_key', {
      key_bundle_hash: keyBundleHash,
      reason,
      revocation_signature: revocationSignature,
    });
  }

  async verifySignature(
    keyBundleHash: ActionHash,
    message: Uint8Array,
    signature: Uint8Array
  ): Promise<boolean> {
    return this.client.callZome('keys', 'verify_signature', {
      key_bundle_hash: keyBundleHash,
      message,
      signature,
    });
  }
}

/**
 * Capabilities Zome Client
 */
class CapabilitiesZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async grantCapability(
    grantee: AgentPubKey,
    accessType: string,
    permissions: object,
    restrictions?: object,
    expiresAt?: number
  ): Promise<ActionHash> {
    return this.client.callZome('capabilities', 'grant_capability', {
      grantee,
      access_type: accessType,
      permissions,
      restrictions,
      expires_at: expiresAt,
    });
  }

  async revokeCapability(hash: ActionHash, reason?: string): Promise<ActionHash> {
    return this.client.callZome('capabilities', 'revoke_capability', [hash, reason]);
  }

  async getGrantedCapabilities(): Promise<MailboxCapability[]> {
    return this.client.callZome('capabilities', 'get_granted_capabilities', null);
  }

  async getReceivedCapabilities(): Promise<MailboxCapability[]> {
    return this.client.callZome('capabilities', 'get_received_capabilities', null);
  }

  async verifyCapability(hash: ActionHash, action: string): Promise<boolean> {
    return this.client.callZome('capabilities', 'verify_capability', [hash, action]);
  }
}

/**
 * Search Zome Client
 */
class SearchZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async search(query: SearchQuery): Promise<SearchResult[]> {
    return this.client.callZome('search', 'search', query);
  }

  async indexDocument(
    documentHash: ActionHash,
    content: string,
    metadata: Record<string, string>
  ): Promise<number> {
    return this.client.callZome('search', 'index_document', {
      document_hash: documentHash,
      content,
      metadata,
    });
  }

  async removeFromIndex(documentHash: ActionHash): Promise<void> {
    return this.client.callZome('search', 'remove_from_index', documentHash);
  }

  async getSuggestions(prefix: string, limit?: number): Promise<string[]> {
    return this.client.callZome('search', 'get_suggestions', { prefix, limit });
  }
}

/**
 * Backup Zome Client
 */
class BackupZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async createBackup(
    backupType: string,
    contents: object,
    password?: string,
    metadata?: Record<string, string>
  ): Promise<ActionHash> {
    return this.client.callZome('backup', 'create_backup', {
      backup_type: backupType,
      contents,
      password,
      metadata,
    });
  }

  async getBackups(): Promise<BackupManifest[]> {
    return this.client.callZome('backup', 'get_backups', null);
  }

  async getBackup(hash: ActionHash): Promise<BackupManifest | null> {
    return this.client.callZome('backup', 'get_backup', hash);
  }

  async restoreBackup(
    backupHash: ActionHash,
    password?: string,
    options?: object
  ): Promise<object> {
    return this.client.callZome('backup', 'restore_backup', {
      backup_hash: backupHash,
      password,
      options,
    });
  }

  async deleteBackup(hash: ActionHash): Promise<void> {
    return this.client.callZome('backup', 'delete_backup', hash);
  }
}

/**
 * Sync Zome Client
 */
class SyncZomeClient {
  constructor(private client: MycelixHolochainClient) {}

  async getSyncState(): Promise<SyncState | null> {
    return this.client.callZome('sync', 'get_sync_state', null);
  }

  async setOnlineStatus(isOnline: boolean): Promise<void> {
    return this.client.callZome('sync', 'set_online_status', isOnline);
  }

  async queueOperation(
    operationType: string,
    entityHash: ActionHash,
    data?: Uint8Array
  ): Promise<ActionHash> {
    return this.client.callZome('sync', 'queue_operation', {
      operation_type: operationType,
      entity_hash: entityHash,
      data,
    });
  }

  async processPendingOperations(): Promise<number> {
    return this.client.callZome('sync', 'process_pending_operations', null);
  }

  async requestSyncFromPeer(peer: AgentPubKey): Promise<void> {
    return this.client.callZome('sync', 'request_sync_from_peer', peer);
  }

  async registerSyncPeer(peer: AgentPubKey): Promise<ActionHash> {
    return this.client.callZome('sync', 'register_sync_peer', peer);
  }
}

// Export factory function
export function createHolochainClient(
  config: MycelixHolochainConfig
): MycelixHolochainClient {
  return new MycelixHolochainClient(config);
}

export default MycelixHolochainClient;
