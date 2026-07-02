// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/identity - Store Tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createIdentityStore, type IdentityStore } from '../store';
import type { MycelixSoul, SoulCredential, SoulSession } from '../types';

describe('IdentityStore', () => {
  let store: ReturnType<typeof createIdentityStore>;

  beforeEach(() => {
    // Create fresh store for each test
    store = createIdentityStore('music');
  });

  describe('initialization', () => {
    it('should start with no soul', () => {
      expect(store.getState().soul).toBeNull();
      expect(store.getState().isAuthenticated).toBe(false);
    });

    it('should have correct initial app context', () => {
      expect(store.getState().currentApp).toBe('music');
    });

    it('should have wallet disconnected initially', () => {
      expect(store.getState().wallet.status).toBe('disconnected');
      expect(store.getState().wallet.connection).toBeNull();
    });
  });

  describe('soul management', () => {
    it('should create a soul with default values', async () => {
      const soul = await store.getState().createSoul({});

      expect(soul).toBeDefined();
      expect(soul.id).toMatch(/^soul_/);
      expect(soul.profile.name).toBe('Anonymous Soul');
      expect(soul.totalResonance).toBe(0);
      expect(soul.connections).toBe(0);
      expect(soul.bornAt).toBeInstanceOf(Date);
    });

    it('should create a soul with custom profile', async () => {
      const soul = await store.getState().createSoul({
        name: 'Test Soul',
        bio: 'A test soul',
        color: '#FF0000',
      });

      expect(soul.profile.name).toBe('Test Soul');
      expect(soul.profile.bio).toBe('A test soul');
      expect(soul.profile.color).toBe('#FF0000');
    });

    it('should set authenticated after soul creation', async () => {
      await store.getState().createSoul({});

      expect(store.getState().isAuthenticated).toBe(true);
      expect(store.getState().soul).not.toBeNull();
    });

    it('should update soul profile', async () => {
      await store.getState().createSoul({ name: 'Original' });

      await store.getState().updateSoul({ name: 'Updated', bio: 'New bio' });

      const soul = store.getState().soul!;
      expect(soul.profile.name).toBe('Updated');
      expect(soul.profile.bio).toBe('New bio');
    });

    it('should throw when updating without soul', async () => {
      await expect(store.getState().updateSoul({ name: 'Test' }))
        .rejects.toThrow('No soul to update');
    });
  });

  describe('resonance', () => {
    it('should add resonance to soul', async () => {
      await store.getState().createSoul({});

      await store.getState().addResonance(10, 'test');
      expect(store.getState().soul!.totalResonance).toBe(10);

      await store.getState().addResonance(15, 'test2');
      expect(store.getState().soul!.totalResonance).toBe(25);
    });

    it('should get resonance', async () => {
      await store.getState().createSoul({});
      await store.getState().addResonance(42, 'test');

      expect(store.getState().getResonance()).toBe(42);
    });

    it('should return 0 resonance when no soul', () => {
      expect(store.getState().getResonance()).toBe(0);
    });
  });

  describe('credentials', () => {
    it('should add credential to soul', async () => {
      await store.getState().createSoul({});

      const credential: SoulCredential = {
        type: 'artist_verified',
        issuer: 'mycelix',
        issuedAt: new Date(),
        data: { level: 'gold' },
      };

      await store.getState().addCredential(credential);

      const credentials = store.getState().getCredentials();
      expect(credentials).toHaveLength(1);
      expect(credentials[0].type).toBe('artist_verified');
    });

    it('should filter credentials by type', async () => {
      await store.getState().createSoul({});

      await store.getState().addCredential({
        type: 'artist_verified',
        issuer: 'mycelix',
        issuedAt: new Date(),
        data: {},
      });

      await store.getState().addCredential({
        type: 'patron_tier',
        issuer: 'mycelix',
        issuedAt: new Date(),
        data: { tier: 'grove' },
      });

      const artistCredentials = store.getState().getCredentials('artist_verified');
      expect(artistCredentials).toHaveLength(1);

      const patronCredentials = store.getState().getCredentials('patron_tier');
      expect(patronCredentials).toHaveLength(1);
    });

    it('should check if credential exists', async () => {
      await store.getState().createSoul({});

      await store.getState().addCredential({
        type: 'early_adopter',
        issuer: 'mycelix',
        issuedAt: new Date(),
        data: {},
      });

      expect(store.getState().hasCredential('early_adopter')).toBe(true);
      expect(store.getState().hasCredential('artist_verified')).toBe(false);
    });
  });

  describe('sessions', () => {
    it('should start session after soul creation', async () => {
      await store.getState().createSoul({});

      const sessions = store.getState().getActiveSessions();
      expect(sessions).toHaveLength(1);
      expect(sessions[0].appId).toBe('music');
    });

    it('should start new session in different app', async () => {
      await store.getState().createSoul({});

      await store.getState().startSession('studio');

      const sessions = store.getState().getActiveSessions();
      expect(sessions).toHaveLength(2);
      expect(sessions.map(s => s.appId)).toContain('music');
      expect(sessions.map(s => s.appId)).toContain('studio');
    });

    it('should replace session for same app', async () => {
      await store.getState().createSoul({});

      const firstSession = store.getState().getActiveSessions()[0];
      await store.getState().startSession('music');

      const sessions = store.getState().getActiveSessions();
      expect(sessions).toHaveLength(1);
      expect(sessions[0].id).not.toBe(firstSession.id);
    });

    it('should end session', async () => {
      await store.getState().createSoul({});

      const session = store.getState().getActiveSessions()[0];
      await store.getState().endSession(session.id);

      expect(store.getState().getActiveSessions()).toHaveLength(0);
    });
  });

  describe('wallet connection', () => {
    it('should connect wallet', async () => {
      const connection = await store.getState().connectWallet('metamask');

      expect(connection).toBeDefined();
      expect(connection.provider).toBe('metamask');
      expect(connection.address).toMatch(/^0x/);
      expect(store.getState().wallet.status).toBe('connected');
    });

    it('should create soul when connecting wallet without existing soul', async () => {
      await store.getState().connectWallet('metamask');

      expect(store.getState().soul).not.toBeNull();
      expect(store.getState().isAuthenticated).toBe(true);
    });

    it('should link wallet to existing soul', async () => {
      await store.getState().createSoul({ name: 'Existing Soul' });

      const connection = await store.getState().connectWallet('metamask');

      expect(store.getState().soul!.address).toBe(connection.address);
      expect(store.getState().soul!.profile.name).toBe('Existing Soul');
    });

    it('should disconnect wallet', async () => {
      await store.getState().connectWallet('metamask');
      await store.getState().disconnectWallet();

      expect(store.getState().wallet.status).toBe('disconnected');
      expect(store.getState().wallet.connection).toBeNull();
    });

    it('should identify smart account wallets', async () => {
      const emailConnection = await store.getState().connectWallet('email');
      expect(emailConnection.isSmartAccount).toBe(true);

      await store.getState().disconnectWallet();

      const metamaskConnection = await store.getState().connectWallet('metamask');
      expect(metamaskConnection.isSmartAccount).toBe(false);
    });
  });

  describe('authentication', () => {
    it('should sign in with wallet', async () => {
      const soul = await store.getState().signIn({ type: 'wallet', provider: 'metamask' });

      expect(soul).toBeDefined();
      expect(store.getState().isAuthenticated).toBe(true);
      expect(store.getState().authMethod?.type).toBe('wallet');
    });

    it('should sign in with email', async () => {
      const soul = await store.getState().signIn({ type: 'email', email: 'test@example.com' });

      expect(soul).toBeDefined();
      expect(store.getState().authMethod?.type).toBe('email');
    });

    it('should sign in with social', async () => {
      const soul = await store.getState().signIn({ type: 'social', provider: 'discord' });

      expect(soul).toBeDefined();
      expect(store.getState().authMethod?.type).toBe('social');
    });

    it('should sign out completely', async () => {
      await store.getState().signIn({ type: 'wallet', provider: 'metamask' });
      await store.getState().signOut();

      expect(store.getState().soul).toBeNull();
      expect(store.getState().isAuthenticated).toBe(false);
      expect(store.getState().authMethod).toBeNull();
      expect(store.getState().wallet.status).toBe('disconnected');
    });
  });

  describe('event system', () => {
    it('should emit events on soul creation', async () => {
      const handler = vi.fn();
      store.getState().subscribe(handler);

      await store.getState().createSoul({ name: 'Test' });

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'soul:created' })
      );
    });

    it('should emit events on resonance update', async () => {
      await store.getState().createSoul({});

      const handler = vi.fn();
      store.getState().subscribe(handler);

      await store.getState().addResonance(10, 'test');

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'resonance:updated',
          total: 10,
          delta: 10,
        })
      );
    });

    it('should unsubscribe correctly', async () => {
      const handler = vi.fn();
      const unsubscribe = store.getState().subscribe(handler);

      await store.getState().createSoul({});
      expect(handler).toHaveBeenCalled();

      handler.mockClear();
      unsubscribe();

      await store.getState().addResonance(10, 'test');
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('should set and clear errors', () => {
      store.getState().setError('Test error');
      expect(store.getState().error).toBe('Test error');

      store.getState().clearError();
      expect(store.getState().error).toBeNull();
    });
  });
});
