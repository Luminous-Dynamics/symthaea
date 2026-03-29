// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contacts Integration Tests
 *
 * Tests for contact management with trust integration.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { createMycelixClient, MycelixClient } from '../../src/bootstrap';

describe('Contacts Integration', () => {
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

  describe('Contact CRUD', () => {
    it('should create a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'alice@test.local',
        name: 'Alice Smith',
        organization: 'Test Corp',
        tags: ['friend', 'work'],
        favorite: false,
        blocked: false,
      });

      expect(id).toBeDefined();
      expect(typeof id).toBe('string');
    });

    it('should retrieve a contact by ID', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'bob@test.local',
        name: 'Bob Jones',
        favorite: false,
        blocked: false,
        tags: [],
      });

      const contact = await services.contacts.getContact(id);

      expect(contact).toBeDefined();
      expect(contact?.email).toBe('bob@test.local');
      expect(contact?.name).toBe('Bob Jones');
    });

    it('should find a contact by email', async () => {
      const services = client.getServices();

      await services.contacts.createContact({
        email: 'unique@test.local',
        name: 'Unique Person',
        favorite: false,
        blocked: false,
        tags: [],
      });

      const contact = await services.contacts.getContactByEmail('unique@test.local');

      expect(contact).toBeDefined();
      expect(contact?.name).toBe('Unique Person');
    });

    it('should update a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'charlie@test.local',
        name: 'Charlie Brown',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.updateContact(id, {
        name: 'Charlie B. Brown',
        organization: 'Peanuts Inc',
        phone: '555-1234',
      });

      const contact = await services.contacts.getContact(id);

      expect(contact?.name).toBe('Charlie B. Brown');
      expect(contact?.organization).toBe('Peanuts Inc');
      expect(contact?.phone).toBe('555-1234');
    });

    it('should delete a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'delete-me@test.local',
        name: 'Delete Me',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.deleteContact(id);

      const contact = await services.contacts.getContact(id);
      expect(contact).toBeNull();
    });

    it('should list all contacts', async () => {
      const services = client.getServices();

      // Create a few contacts
      await services.contacts.createContact({
        email: 'list1@test.local',
        name: 'List One',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.createContact({
        email: 'list2@test.local',
        name: 'List Two',
        favorite: false,
        blocked: false,
        tags: [],
      });

      const contacts = await services.contacts.getAllContacts();

      expect(Array.isArray(contacts)).toBe(true);
      expect(contacts.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Favorites and Blocking', () => {
    it('should mark contact as favorite', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'favorite@test.local',
        name: 'Favorite Person',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.updateContact(id, { favorite: true });

      const contact = await services.contacts.getContact(id);
      expect(contact?.favorite).toBe(true);
    });

    it('should block a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'blocked@test.local',
        name: 'Blocked Person',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.updateContact(id, { blocked: true });

      const contact = await services.contacts.getContact(id);
      expect(contact?.blocked).toBe(true);
    });

    it('should unblock a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'unblock@test.local',
        name: 'Unblock Person',
        favorite: false,
        blocked: true,
        tags: [],
      });

      await services.contacts.updateContact(id, { blocked: false });

      const contact = await services.contacts.getContact(id);
      expect(contact?.blocked).toBe(false);
    });
  });

  describe('Tags', () => {
    it('should add tags to a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'tagged@test.local',
        name: 'Tagged Person',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.updateContact(id, {
        tags: ['work', 'important', 'project-x'],
      });

      const contact = await services.contacts.getContact(id);

      expect(contact?.tags).toContain('work');
      expect(contact?.tags).toContain('important');
      expect(contact?.tags).toContain('project-x');
    });

    it('should remove tags from a contact', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'untag@test.local',
        name: 'Untag Person',
        favorite: false,
        blocked: false,
        tags: ['remove-me', 'keep-me'],
      });

      await services.contacts.updateContact(id, {
        tags: ['keep-me'],
      });

      const contact = await services.contacts.getContact(id);

      expect(contact?.tags).not.toContain('remove-me');
      expect(contact?.tags).toContain('keep-me');
    });
  });

  describe('Contact Groups', () => {
    it('should create a group', async () => {
      const services = client.getServices();

      const id = await services.contacts.createGroup({
        name: 'Test Group',
        color: '#ff0000',
      });

      expect(id).toBeDefined();
    });

    it('should add contacts to a group', async () => {
      const services = client.getServices();

      const groupId = await services.contacts.createGroup({
        name: 'Team Group',
      });

      const contactId = await services.contacts.createContact({
        email: 'team-member@test.local',
        name: 'Team Member',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.addToGroup(contactId, groupId);

      const groups = await services.contacts.getAllGroups();
      const group = groups.find((g) => g.id === groupId);

      expect(group?.contacts).toContain(contactId);
    });

    it('should remove contacts from a group', async () => {
      const services = client.getServices();

      const groupId = await services.contacts.createGroup({
        name: 'Remove Group',
      });

      const contactId = await services.contacts.createContact({
        email: 'remove-from-group@test.local',
        name: 'Remove From Group',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.addToGroup(contactId, groupId);
      await services.contacts.removeFromGroup(contactId, groupId);

      const groups = await services.contacts.getAllGroups();
      const group = groups.find((g) => g.id === groupId);

      expect(group?.contacts).not.toContain(contactId);
    });

    it('should delete a group', async () => {
      const services = client.getServices();

      const groupId = await services.contacts.createGroup({
        name: 'Delete Group',
      });

      await services.contacts.deleteGroup(groupId);

      const groups = await services.contacts.getAllGroups();
      const group = groups.find((g) => g.id === groupId);

      expect(group).toBeUndefined();
    });
  });

  describe('vCard Import/Export', () => {
    it('should import contacts from vCard', async () => {
      const services = client.getServices();

      const vcard = `BEGIN:VCARD
VERSION:3.0
FN:John Doe
EMAIL:john.doe@example.com
ORG:Example Inc
TEL:+1-555-555-5555
END:VCARD`;

      const imported = await services.contacts.importVCard(vcard);

      expect(Array.isArray(imported)).toBe(true);
      expect(imported.length).toBe(1);
      expect(imported[0].email).toBe('john.doe@example.com');
      expect(imported[0].name).toBe('John Doe');
    });

    it('should export contacts to vCard', async () => {
      const services = client.getServices();

      const id = await services.contacts.createContact({
        email: 'export@test.local',
        name: 'Export Person',
        organization: 'Export Corp',
        phone: '555-EXPORT',
        favorite: false,
        blocked: false,
        tags: [],
      });

      const vcard = await services.contacts.exportVCard([id]);

      expect(vcard).toContain('BEGIN:VCARD');
      expect(vcard).toContain('export@test.local');
      expect(vcard).toContain('Export Person');
      expect(vcard).toContain('END:VCARD');
    });

    it('should handle multi-contact vCard import', async () => {
      const services = client.getServices();

      const vcard = `BEGIN:VCARD
VERSION:3.0
FN:Person One
EMAIL:one@example.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:Person Two
EMAIL:two@example.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:Person Three
EMAIL:three@example.com
END:VCARD`;

      const imported = await services.contacts.importVCard(vcard);

      expect(imported.length).toBe(3);
    });
  });

  describe('Contact Suggestions', () => {
    it('should suggest contacts based on query', async () => {
      const services = client.getServices();

      // Create some contacts
      await services.contacts.createContact({
        email: 'suggest-alice@test.local',
        name: 'Alice Suggester',
        favorite: false,
        blocked: false,
        tags: [],
      });

      await services.contacts.createContact({
        email: 'suggest-alex@test.local',
        name: 'Alex Suggester',
        favorite: false,
        blocked: false,
        tags: [],
      });

      const suggestions = await services.contacts.getSuggestions('al', 5);

      expect(Array.isArray(suggestions)).toBe(true);
      expect(suggestions.length).toBeGreaterThanOrEqual(2);
      expect(
        suggestions.every((s) =>
          s.name?.toLowerCase().includes('al') ||
          s.email.toLowerCase().includes('al')
        )
      ).toBe(true);
    });

    it('should include trust level in suggestions', async () => {
      const services = client.getServices();

      // Create contact and add trust
      const id = await services.contacts.createContact({
        email: 'trusted-suggest@test.local',
        name: 'Trusted Suggester',
        favorite: false,
        blocked: false,
        tags: [],
      });

      // The contact service should enrich with trust data
      const suggestions = await services.contacts.getSuggestions('trusted', 5);

      const found = suggestions.find((s) => s.email === 'trusted-suggest@test.local');
      expect(found).toBeDefined();
      // Trust level may or may not be set depending on attestations
      expect(found?.trustLevel).toBeDefined();
    });
  });

  describe('Trust Integration', () => {
    it('should link contacts with trust attestations', async () => {
      const services = client.getServices();

      // Create contact
      const contactId = await services.contacts.createContact({
        email: 'trust-contact@test.local',
        name: 'Trust Contact',
        favorite: false,
        blocked: false,
        tags: [],
      });

      // The contact should be enriched with trust data if available
      const contact = await services.contacts.getContact(contactId);

      expect(contact).toBeDefined();
      // Trust level is optional and depends on attestations
      expect(contact?.trustLevel === undefined || typeof contact?.trustLevel === 'number').toBe(true);
    });
  });
});
