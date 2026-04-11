// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Messages Integration Tests
 *
 * Tests for email messaging functionality with Holochain zomes.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { createMycelixClient, MycelixClient } from '../../src/bootstrap';

describe('Messages Integration', () => {
  let client: MycelixClient;

  beforeAll(async () => {
    client = createMycelixClient({
      websocketUrl: process.env.HOLOCHAIN_URL || 'ws://localhost:8888',
      appId: 'mycelix-mail-test',
    });
    await client.initialize();
  });

  afterAll(async () => {
    await client.shutdown();
  });

  describe('Email CRUD Operations', () => {
    it('should send an email', async () => {
      const services = client.getServices();

      const hash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        cc: [],
        bcc: [],
        subject: 'Test Email',
        body: 'This is a test email body.',
        attachments: [],
        encrypted: false,
      });

      expect(hash).toBeDefined();
      expect(typeof hash).toBe('string');
      expect(hash.length).toBeGreaterThan(0);
    });

    it('should retrieve sent emails', async () => {
      const services = client.getServices();

      const result = await services.messages.getMessages({
        folder: 'sent',
        limit: 10,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.messages)).toBe(true);
    });

    it('should mark email as read', async () => {
      const services = client.getServices();

      // First, send an email
      const hash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        cc: [],
        bcc: [],
        subject: 'Read Test',
        body: 'Test body',
        attachments: [],
        encrypted: false,
      });

      // Mark as read
      await services.messages.markAsRead(hash);

      // Verify (would need to fetch and check)
      const email = await services.messages.getMessage(hash);
      expect(email?.read).toBe(true);
    });

    it('should star and unstar emails', async () => {
      const services = client.getServices();

      const hash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        subject: 'Star Test',
        body: 'Test body',
        cc: [],
        bcc: [],
        attachments: [],
        encrypted: false,
      });

      // Star
      await services.messages.star(hash);
      let email = await services.messages.getMessage(hash);
      expect(email?.starred).toBe(true);

      // Unstar
      await services.messages.unstar(hash);
      email = await services.messages.getMessage(hash);
      expect(email?.starred).toBe(false);
    });

    it('should archive emails', async () => {
      const services = client.getServices();

      const hash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        subject: 'Archive Test',
        body: 'Test body',
        cc: [],
        bcc: [],
        attachments: [],
        encrypted: false,
      });

      await services.messages.archive(hash);

      // Verify it's in archive folder
      const result = await services.messages.getMessages({
        folder: 'archive',
        limit: 50,
      });

      const found = result.messages.find((m: any) => m.hash === hash);
      expect(found).toBeDefined();
    });

    it('should move emails to trash', async () => {
      const services = client.getServices();

      const hash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        subject: 'Trash Test',
        body: 'Test body',
        cc: [],
        bcc: [],
        attachments: [],
        encrypted: false,
      });

      await services.messages.trash(hash);

      // Verify it's in trash folder
      const result = await services.messages.getMessages({
        folder: 'trash',
        limit: 50,
      });

      const found = result.messages.find((m: any) => m.hash === hash);
      expect(found).toBeDefined();
    });
  });

  describe('Email Threading', () => {
    it('should thread replies together', async () => {
      const services = client.getServices();

      // Send original email
      const originalHash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        subject: 'Thread Test',
        body: 'Original message',
        cc: [],
        bcc: [],
        attachments: [],
        encrypted: false,
      });

      // Send reply
      const replyHash = await services.messages.send({
        to: [{ address: 'recipient@test.local' }],
        subject: 'Re: Thread Test',
        body: 'Reply message',
        cc: [],
        bcc: [],
        attachments: [],
        encrypted: false,
        replyTo: originalHash,
      });

      // Get thread
      const thread = await services.thread.getThread(originalHash);

      expect(thread).toBeDefined();
      expect(thread.messages.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Batch Operations', () => {
    it('should batch mark emails as read', async () => {
      const services = client.getServices();

      // Send multiple emails
      const hashes = await Promise.all([
        services.messages.send({
          to: [{ address: 'recipient@test.local' }],
          subject: 'Batch Test 1',
          body: 'Test',
          cc: [],
          bcc: [],
          attachments: [],
          encrypted: false,
        }),
        services.messages.send({
          to: [{ address: 'recipient@test.local' }],
          subject: 'Batch Test 2',
          body: 'Test',
          cc: [],
          bcc: [],
          attachments: [],
          encrypted: false,
        }),
        services.messages.send({
          to: [{ address: 'recipient@test.local' }],
          subject: 'Batch Test 3',
          body: 'Test',
          cc: [],
          bcc: [],
          attachments: [],
          encrypted: false,
        }),
      ]);

      // Batch mark as read
      await services.batch.markManyAsRead(hashes);

      // Verify all are read
      for (const hash of hashes) {
        const email = await services.messages.getMessage(hash);
        expect(email?.read).toBe(true);
      }
    });
  });

  describe('Draft Management', () => {
    it('should save and retrieve drafts', async () => {
      const services = client.getServices();

      const draftId = await services.messages.saveDraft({
        to: [{ address: 'recipient@test.local' }],
        cc: [],
        bcc: [],
        subject: 'Draft Test',
        body: 'Draft content',
        attachments: [],
        encrypted: true,
      });

      expect(draftId).toBeDefined();

      const draft = await services.messages.getDraft(draftId);
      expect(draft).toBeDefined();
      expect(draft?.subject).toBe('Draft Test');
      expect(draft?.body).toBe('Draft content');
    });

    it('should update existing drafts', async () => {
      const services = client.getServices();

      const draftId = await services.messages.saveDraft({
        to: [{ address: 'recipient@test.local' }],
        cc: [],
        bcc: [],
        subject: 'Original Draft',
        body: 'Original content',
        attachments: [],
        encrypted: true,
      });

      // Update the draft
      await services.messages.saveDraft(
        {
          to: [{ address: 'recipient@test.local' }],
          cc: [],
          bcc: [],
          subject: 'Updated Draft',
          body: 'Updated content',
          attachments: [],
          encrypted: true,
        },
        draftId
      );

      const draft = await services.messages.getDraft(draftId);
      expect(draft?.subject).toBe('Updated Draft');
      expect(draft?.body).toBe('Updated content');
    });

    it('should delete drafts', async () => {
      const services = client.getServices();

      const draftId = await services.messages.saveDraft({
        to: [{ address: 'recipient@test.local' }],
        cc: [],
        bcc: [],
        subject: 'Delete Draft Test',
        body: 'Content',
        attachments: [],
        encrypted: true,
      });

      await services.messages.deleteDraft(draftId);

      const draft = await services.messages.getDraft(draftId);
      expect(draft).toBeNull();
    });
  });
});
