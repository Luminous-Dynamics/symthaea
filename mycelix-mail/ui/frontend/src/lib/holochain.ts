/**
 * Holochain Client Integration for Mycelix-Mail
 *
 * This module provides the bridge between the React UI and the Holochain hApp backend.
 * It replaces the traditional REST API / Prisma calls with Holochain zome calls.
 */

import { AppAgentWebsocket, AppAgentClient, ActionHash, AgentPubKey } from '@holochain/client';

// Types matching the Holochain DNA
export interface Email {
  action_hash: ActionHash;
  to: AgentPubKey;
  from: AgentPubKey;
  subject: string;
  body: string;
  timestamp: number;
  encrypted: boolean;
  trust_score?: number;
  spam_probability?: number;
}

export interface SendEmailInput {
  to: AgentPubKey;
  subject: string;
  body: string;
  encrypt?: boolean;
}

export interface Contact {
  agent_pubkey: AgentPubKey;
  nickname?: string;
  trust_level: number;
}

// Singleton client instance
let client: AppAgentClient | null = null;

/**
 * Initialize the Holochain client connection
 */
export async function initHolochainClient(): Promise<AppAgentClient> {
  if (client) return client;

  // Connect to the Holochain conductor
  // In development, this is typically ws://localhost:8888
  // The app_id must match what's installed in the conductor
  client = await AppAgentWebsocket.connect('', 'mycelix-mail');

  console.log('Connected to Holochain conductor');
  return client;
}

/**
 * Get the current agent's public key
 */
export async function getMyPubKey(): Promise<AgentPubKey> {
  const c = await initHolochainClient();
  return c.myPubKey;
}

// =============================================================================
// Email Operations (replace REST API calls)
// =============================================================================

/**
 * Send an email via Holochain
 * Replaces: POST /api/emails
 */
export async function sendEmail(input: SendEmailInput): Promise<ActionHash> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'mail',
    fn_name: 'send_email',
    payload: input
  });
}

/**
 * Get inbox emails
 * Replaces: GET /api/emails?folder=inbox
 */
export async function getInbox(): Promise<Email[]> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'mail',
    fn_name: 'get_inbox',
    payload: null
  });
}

/**
 * Get sent emails
 * Replaces: GET /api/emails?folder=sent
 */
export async function getSentEmails(): Promise<Email[]> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'mail',
    fn_name: 'get_sent',
    payload: null
  });
}

/**
 * Get a single email by action hash
 * Replaces: GET /api/emails/:id
 */
export async function getEmail(actionHash: ActionHash): Promise<Email | null> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'mail',
    fn_name: 'get_email',
    payload: actionHash
  });
}

/**
 * Delete an email
 * Replaces: DELETE /api/emails/:id
 */
export async function deleteEmail(actionHash: ActionHash): Promise<void> {
  const c = await initHolochainClient();

  await c.callZome({
    role_name: 'mail',
    zome_name: 'mail',
    fn_name: 'delete_email',
    payload: actionHash
  });
}

// =============================================================================
// Contact Operations
// =============================================================================

/**
 * Add a contact
 * Replaces: POST /api/contacts
 */
export async function addContact(contact: Omit<Contact, 'trust_level'>): Promise<ActionHash> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'contacts',
    fn_name: 'add_contact',
    payload: contact
  });
}

/**
 * Get all contacts
 * Replaces: GET /api/contacts
 */
export async function getContacts(): Promise<Contact[]> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'contacts',
    fn_name: 'get_contacts',
    payload: null
  });
}

// =============================================================================
// MATL Trust Operations (unique to Mycelix)
// =============================================================================

/**
 * Get trust score for a sender
 * Used for spam filtering
 */
export async function getTrustScore(agentPubKey: AgentPubKey): Promise<number> {
  const c = await initHolochainClient();

  return c.callZome({
    role_name: 'mail',
    zome_name: 'trust',
    fn_name: 'get_trust_score',
    payload: agentPubKey
  });
}

/**
 * Report spam (negative trust signal)
 */
export async function reportSpam(emailHash: ActionHash): Promise<void> {
  const c = await initHolochainClient();

  await c.callZome({
    role_name: 'mail',
    zome_name: 'trust',
    fn_name: 'report_spam',
    payload: emailHash
  });
}

/**
 * Mark as not spam (positive trust signal)
 */
export async function markNotSpam(emailHash: ActionHash): Promise<void> {
  const c = await initHolochainClient();

  await c.callZome({
    role_name: 'mail',
    zome_name: 'trust',
    fn_name: 'mark_not_spam',
    payload: emailHash
  });
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Subscribe to new emails (real-time updates)
 */
export async function subscribeToEmails(callback: (email: Email) => void): Promise<void> {
  const c = await initHolochainClient();

  // Holochain signals for real-time updates
  c.on('signal', (signal) => {
    if (signal.zome_name === 'mail' && signal.payload.type === 'new_email') {
      callback(signal.payload.email);
    }
  });
}

/**
 * Disconnect from Holochain
 */
export async function disconnect(): Promise<void> {
  if (client && 'close' in client) {
    await (client as any).close();
    client = null;
  }
}
