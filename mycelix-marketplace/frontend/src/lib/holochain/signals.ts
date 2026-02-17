import type { AppClient } from '@holochain/client';
import { markProofDenied, markProofFulfilled, notifications } from '$lib/stores';

type ProofSignal =
  | { type: 'proof_fulfilled'; agent_id: string; listing_hash?: string; message?: string }
  | { type: 'proof_denied'; agent_id: string; listing_hash?: string; message?: string };

function isProofSignal(payload: any): payload is ProofSignal {
  const p = payload?.data || payload?.payload || payload;
  return p?.type === 'proof_fulfilled' || p?.type === 'proof_denied';
}

const toastCooldownMs = 10_000;
const lastToastByKey = new Map<string, number>();

function shouldToast(key: string) {
  const now = Date.now();
  const last = lastToastByKey.get(key) || 0;
  if (now - last < toastCooldownMs) return false;
  lastToastByKey.set(key, now);
  return true;
}

/**
 * Attach signal handler to an existing AppClient.
 */
export function attachSignalHandler(client: AppClient) {
  // AppWebsocket.connect allows passing a signal handler; we attach here for clarity if needed
  if (typeof (client as any).on === 'function') {
    (client as any).on('signal', handleSignal);
  }
}

export function handleSignal(signal: any) {
  try {
    const payload = signal?.data || signal?.payload || signal;
    const body = payload?.data || payload?.payload || payload;
    if (!isProofSignal(payload)) return;

    const { type, agent_id, listing_hash, message } = body;
    const key = `${agent_id || 'unknown'}::${listing_hash || 'agent'}`;

    if (type === 'proof_fulfilled') {
      markProofFulfilled(agent_id, listing_hash);
      if (shouldToast(key)) {
        notifications.success('Proof fulfilled', message || 'New proof published');
      }
    } else if (type === 'proof_denied') {
      markProofDenied(agent_id, listing_hash, message);
      if (shouldToast(key)) {
        notifications.warning('Proof denied', message || 'Proof request was denied');
      }
    }
  } catch (error) {
    console.error('Error handling signal', error);
  }
}
