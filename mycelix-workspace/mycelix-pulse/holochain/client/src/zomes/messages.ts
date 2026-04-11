// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Messages Zome Client
 * P2P email delivery with DHT storage
 */

import type { AppClient, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  SendEmailInput,
  SendEmailOutput,
  DecryptedEmail,
  EmailFolder,
  EmailThread,
  EmailState,
  EmailPriority,
} from '../types';

export class MessagesZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'mycelix_mail',
    private zomeName: string = 'mail_messages'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ==================== EMAIL OPERATIONS ====================

  /**
   * Send an encrypted email via P2P
   */
  async sendEmail(input: SendEmailInput): Promise<SendEmailOutput> {
    return this.callZome('send_email', {
      recipient: input.recipient,
      encrypted_subject: new TextEncoder().encode(input.subject),
      encrypted_body: new TextEncoder().encode(input.body),
      encrypted_attachments: input.attachments
        ? new TextEncoder().encode(JSON.stringify(input.attachments))
        : new Uint8Array(),
      priority: input.priority ?? 'Normal',
      in_reply_to: input.in_reply_to ?? null,
      read_receipt_requested: input.request_read_receipt ?? false,
      crypto_suite: input.encrypt_with ?? 'HybridKyberX25519',
    });
  }

  /**
   * Get inbox emails
   */
  async getInbox(limit?: number): Promise<DecryptedEmail[]> {
    return this.callZome('get_inbox', { limit: limit ?? 50 });
  }

  /**
   * Get sent emails
   */
  async getSent(limit?: number): Promise<DecryptedEmail[]> {
    return this.callZome('get_sent', { limit: limit ?? 50 });
  }

  /**
   * Get drafts
   */
  async getDrafts(): Promise<DecryptedEmail[]> {
    return this.callZome('get_drafts', null);
  }

  /**
   * Get email by hash
   */
  async getEmail(emailHash: ActionHash): Promise<DecryptedEmail | null> {
    return this.callZome('get_email', emailHash);
  }

  /**
   * Mark email as read
   */
  async markAsRead(emailHash: ActionHash): Promise<void> {
    await this.callZome('mark_as_read', emailHash);
  }

  /**
   * Mark email as unread
   */
  async markAsUnread(emailHash: ActionHash): Promise<void> {
    await this.callZome('mark_as_unread', emailHash);
  }

  /**
   * Archive email
   */
  async archiveEmail(emailHash: ActionHash): Promise<void> {
    await this.callZome('archive_email', emailHash);
  }

  /**
   * Delete email (soft delete)
   */
  async deleteEmail(emailHash: ActionHash): Promise<void> {
    await this.callZome('delete_email', emailHash);
  }

  /**
   * Permanently delete email
   */
  async permanentlyDelete(emailHash: ActionHash): Promise<void> {
    await this.callZome('permanently_delete', emailHash);
  }

  /**
   * Move email to folder
   */
  async moveToFolder(emailHash: ActionHash, folderHash: ActionHash): Promise<void> {
    await this.callZome('move_to_folder', { email_hash: emailHash, folder_hash: folderHash });
  }

  /**
   * Add label to email
   */
  async addLabel(emailHash: ActionHash, label: string): Promise<void> {
    await this.callZome('add_label', { email_hash: emailHash, label });
  }

  /**
   * Remove label from email
   */
  async removeLabel(emailHash: ActionHash, label: string): Promise<void> {
    await this.callZome('remove_label', { email_hash: emailHash, label });
  }

  /**
   * Star/flag email
   */
  async starEmail(emailHash: ActionHash): Promise<void> {
    await this.callZome('star_email', emailHash);
  }

  /**
   * Unstar email
   */
  async unstarEmail(emailHash: ActionHash): Promise<void> {
    await this.callZome('unstar_email', emailHash);
  }

  /**
   * Send read receipt
   */
  async sendReadReceipt(emailHash: ActionHash): Promise<void> {
    await this.callZome('send_read_receipt', emailHash);
  }

  // ==================== DRAFTS ====================

  /**
   * Save draft
   */
  async saveDraft(input: Partial<SendEmailInput>): Promise<ActionHash> {
    return this.callZome('save_draft', {
      recipient: input.recipient ?? null,
      subject: input.subject ?? '',
      body: input.body ?? '',
      attachments: input.attachments ?? [],
      in_reply_to: input.in_reply_to ?? null,
    });
  }

  /**
   * Update draft
   */
  async updateDraft(draftHash: ActionHash, input: Partial<SendEmailInput>): Promise<ActionHash> {
    return this.callZome('update_draft', {
      draft_hash: draftHash,
      ...input,
    });
  }

  /**
   * Send draft
   */
  async sendDraft(draftHash: ActionHash): Promise<SendEmailOutput> {
    return this.callZome('send_draft', draftHash);
  }

  /**
   * Delete draft
   */
  async deleteDraft(draftHash: ActionHash): Promise<void> {
    await this.callZome('delete_draft', draftHash);
  }

  // ==================== FOLDERS ====================

  /**
   * Create folder
   */
  async createFolder(name: string, parent?: ActionHash): Promise<ActionHash> {
    return this.callZome('create_folder', { name, parent: parent ?? null });
  }

  /**
   * Get all folders
   */
  async getFolders(): Promise<EmailFolder[]> {
    return this.callZome('get_folders', null);
  }

  /**
   * Rename folder
   */
  async renameFolder(folderHash: ActionHash, newName: string): Promise<void> {
    await this.callZome('rename_folder', { folder_hash: folderHash, new_name: newName });
  }

  /**
   * Delete folder
   */
  async deleteFolder(folderHash: ActionHash): Promise<void> {
    await this.callZome('delete_folder', folderHash);
  }

  /**
   * Get emails in folder
   */
  async getEmailsInFolder(folderHash: ActionHash, limit?: number): Promise<DecryptedEmail[]> {
    return this.callZome('get_emails_in_folder', { folder_hash: folderHash, limit: limit ?? 50 });
  }

  // ==================== THREADS ====================

  /**
   * Get thread
   */
  async getThread(threadId: string): Promise<EmailThread | null> {
    return this.callZome('get_thread', threadId);
  }

  /**
   * Get emails in thread
   */
  async getThreadEmails(threadId: string): Promise<DecryptedEmail[]> {
    return this.callZome('get_thread_emails', threadId);
  }

  /**
   * Get all threads (paginated)
   */
  async getThreads(limit?: number, offset?: number): Promise<EmailThread[]> {
    return this.callZome('get_threads', { limit: limit ?? 20, offset: offset ?? 0 });
  }

  // ==================== SEARCH ====================

  /**
   * Search emails
   */
  async searchEmails(query: string, options?: {
    folder?: ActionHash;
    from?: AgentPubKey;
    labels?: string[];
    state?: EmailState;
    priority?: EmailPriority;
    date_from?: Date;
    date_to?: Date;
    limit?: number;
  }): Promise<DecryptedEmail[]> {
    return this.callZome('search_emails', {
      query,
      folder: options?.folder ?? null,
      from: options?.from ?? null,
      labels: options?.labels ?? null,
      state: options?.state ?? null,
      priority: options?.priority ?? null,
      date_from: options?.date_from?.getTime() ?? null,
      date_to: options?.date_to?.getTime() ?? null,
      limit: options?.limit ?? 50,
    });
  }

  // ==================== STATS ====================

  /**
   * Get mailbox stats
   */
  async getStats(): Promise<{
    total_emails: number;
    unread_count: number;
    sent_count: number;
    draft_count: number;
    storage_used: number;
  }> {
    return this.callZome('get_stats', null);
  }
}
