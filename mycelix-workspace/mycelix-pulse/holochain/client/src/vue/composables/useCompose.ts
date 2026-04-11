// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Vue Composable - useCompose
 *
 * Email composition with drafts, scheduling, and encryption.
 */

import { ref, computed, watch, onUnmounted } from 'vue';
import type { Ref, ComputedRef } from 'vue';
import { getMycelixClient } from '../../bootstrap';

export interface Recipient {
  address: string;
  name?: string;
  trustLevel?: number;
}

export interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
  data?: ArrayBuffer;
}

export interface Draft {
  id: string;
  to: Recipient[];
  cc: Recipient[];
  bcc: Recipient[];
  subject: string;
  body: string;
  bodyHtml?: string;
  attachments: Attachment[];
  encrypted: boolean;
  replyTo?: string;
  threadId?: string;
  savedAt: number;
}

export interface UseComposeOptions {
  replyTo?: string;
  threadId?: string;
  initialTo?: Recipient[];
  initialSubject?: string;
  initialBody?: string;
  autoSaveIntervalMs?: number;
}

export interface UseComposeReturn {
  // State
  to: Ref<Recipient[]>;
  cc: Ref<Recipient[]>;
  bcc: Ref<Recipient[]>;
  subject: Ref<string>;
  body: Ref<string>;
  bodyHtml: Ref<string | undefined>;
  attachments: Ref<Attachment[]>;
  encrypted: Ref<boolean>;
  sending: Ref<boolean>;
  error: Ref<string | null>;
  draftId: Ref<string | null>;
  isDirty: Ref<boolean>;

  // Computed
  isValid: ComputedRef<boolean>;
  recipientCount: ComputedRef<number>;
  attachmentSize: ComputedRef<number>;

  // Actions
  addRecipient: (field: 'to' | 'cc' | 'bcc', recipient: Recipient) => void;
  removeRecipient: (field: 'to' | 'cc' | 'bcc', email: string) => void;
  addAttachment: (file: File) => Promise<void>;
  removeAttachment: (id: string) => void;
  saveDraft: () => Promise<string>;
  loadDraft: (draftId: string) => Promise<void>;
  deleteDraft: () => Promise<void>;
  send: () => Promise<string>;
  scheduleSend: (sendAt: Date) => Promise<string>;
  clear: () => void;
}

export function useCompose(options: UseComposeOptions = {}): UseComposeReturn {
  const {
    replyTo,
    threadId,
    initialTo = [],
    initialSubject = '',
    initialBody = '',
    autoSaveIntervalMs = 30000,
  } = options;

  // State
  const to = ref<Recipient[]>(initialTo);
  const cc = ref<Recipient[]>([]);
  const bcc = ref<Recipient[]>([]);
  const subject = ref(initialSubject);
  const body = ref(initialBody);
  const bodyHtml = ref<string | undefined>();
  const attachments = ref<Attachment[]>([]);
  const encrypted = ref(true);
  const sending = ref(false);
  const error = ref<string | null>(null);
  const draftId = ref<string | null>(null);
  const isDirty = ref(false);

  let autoSaveInterval: ReturnType<typeof setInterval> | null = null;

  // Computed
  const isValid = computed(() => {
    return to.value.length > 0 && (subject.value.trim().length > 0 || body.value.trim().length > 0);
  });

  const recipientCount = computed(() => {
    return to.value.length + cc.value.length + bcc.value.length;
  });

  const attachmentSize = computed(() => {
    return attachments.value.reduce((sum, a) => sum + a.size, 0);
  });

  // Watch for changes to mark as dirty
  watch(
    [to, cc, bcc, subject, body, attachments],
    () => {
      isDirty.value = true;
    },
    { deep: true }
  );

  // Actions
  function addRecipient(field: 'to' | 'cc' | 'bcc', recipient: Recipient) {
    const list = field === 'to' ? to : field === 'cc' ? cc : bcc;
    if (!list.value.some((r) => r.address === recipient.address)) {
      list.value = [...list.value, recipient];
    }
  }

  function removeRecipient(field: 'to' | 'cc' | 'bcc', email: string) {
    const list = field === 'to' ? to : field === 'cc' ? cc : bcc;
    list.value = list.value.filter((r) => r.address !== email);
  }

  async function addAttachment(file: File): Promise<void> {
    const data = await file.arrayBuffer();
    attachments.value = [
      ...attachments.value,
      {
        id: `${Date.now()}-${file.name}`,
        name: file.name,
        size: file.size,
        type: file.type,
        data,
      },
    ];
  }

  function removeAttachment(id: string) {
    attachments.value = attachments.value.filter((a) => a.id !== id);
  }

  async function saveDraft(): Promise<string> {
    try {
      const client = getMycelixClient();
      const draft: Omit<Draft, 'id' | 'savedAt'> = {
        to: to.value,
        cc: cc.value,
        bcc: bcc.value,
        subject: subject.value,
        body: body.value,
        bodyHtml: bodyHtml.value,
        attachments: attachments.value,
        encrypted: encrypted.value,
        replyTo,
        threadId,
      };

      const id = await client.getServices().messages.saveDraft(draft, draftId.value);
      draftId.value = id;
      isDirty.value = false;

      return id;
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function loadDraft(id: string): Promise<void> {
    try {
      const client = getMycelixClient();
      const draft = await client.getServices().messages.getDraft(id);

      if (draft) {
        to.value = draft.to;
        cc.value = draft.cc;
        bcc.value = draft.bcc;
        subject.value = draft.subject;
        body.value = draft.body;
        bodyHtml.value = draft.bodyHtml;
        attachments.value = draft.attachments;
        encrypted.value = draft.encrypted;
        draftId.value = id;
        isDirty.value = false;
      }
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function deleteDraft(): Promise<void> {
    if (!draftId.value) return;

    try {
      const client = getMycelixClient();
      await client.getServices().messages.deleteDraft(draftId.value);

      draftId.value = null;
      clear();
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function send(): Promise<string> {
    if (!isValid.value) {
      error.value = 'Please add at least one recipient';
      throw new Error(error.value);
    }

    sending.value = true;
    error.value = null;

    try {
      const client = getMycelixClient();
      const services = client.getServices();

      // If encrypted, encrypt for each recipient
      let encryptedBody = body.value;
      let encryptedAttachments = attachments.value;

      if (encrypted.value) {
        // Encryption would happen here via KeyExchangeService
        // For now, we'll pass through to the messages service
      }

      const hash = await services.messages.send({
        to: to.value,
        cc: cc.value,
        bcc: bcc.value,
        subject: subject.value,
        body: encryptedBody,
        bodyHtml: bodyHtml.value,
        attachments: encryptedAttachments,
        encrypted: encrypted.value,
        replyTo,
        threadId,
      });

      // Delete draft if one exists
      if (draftId.value) {
        await services.messages.deleteDraft(draftId.value);
      }

      clear();
      return hash;
    } catch (e) {
      error.value = String(e);
      throw e;
    } finally {
      sending.value = false;
    }
  }

  async function scheduleSend(sendAt: Date): Promise<string> {
    if (!isValid.value) {
      error.value = 'Please add at least one recipient';
      throw new Error(error.value);
    }

    sending.value = true;
    error.value = null;

    try {
      const client = getMycelixClient();
      const hash = await client.getServices().scheduler.schedule({
        email: {
          to: to.value,
          cc: cc.value,
          bcc: bcc.value,
          subject: subject.value,
          body: body.value,
          bodyHtml: bodyHtml.value,
          attachments: attachments.value,
          encrypted: encrypted.value,
          replyTo,
          threadId,
        },
        sendAt: sendAt.getTime() * 1000, // Convert to microseconds
      });

      // Delete draft if one exists
      if (draftId.value) {
        await client.getServices().messages.deleteDraft(draftId.value);
      }

      clear();
      return hash;
    } catch (e) {
      error.value = String(e);
      throw e;
    } finally {
      sending.value = false;
    }
  }

  function clear() {
    to.value = [];
    cc.value = [];
    bcc.value = [];
    subject.value = '';
    body.value = '';
    bodyHtml.value = undefined;
    attachments.value = [];
    encrypted.value = true;
    draftId.value = null;
    isDirty.value = false;
    error.value = null;
  }

  // Auto-save
  autoSaveInterval = setInterval(() => {
    if (isDirty.value && isValid.value) {
      saveDraft().catch(console.error);
    }
  }, autoSaveIntervalMs);

  onUnmounted(() => {
    if (autoSaveInterval) clearInterval(autoSaveInterval);
  });

  return {
    to,
    cc,
    bcc,
    subject,
    body,
    bodyHtml,
    attachments,
    encrypted,
    sending,
    error,
    draftId,
    isDirty,
    isValid,
    recipientCount,
    attachmentSize,
    addRecipient,
    removeRecipient,
    addAttachment,
    removeAttachment,
    saveDraft,
    loadDraft,
    deleteDraft,
    send,
    scheduleSend,
    clear,
  };
}

export default useCompose;
