import { readable, writable } from 'svelte/store';
import type { ProofRequest, ProofRequestStatus } from '$types';

const STORAGE_KEY = 'mycelix_proof_requests';

function loadInitial(): ProofRequest[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ProofRequest[]) : [];
  } catch {
    return [];
  }
}

function persist(requests: ProofRequest[]) {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(requests));
  } catch {
    // ignore persistence failures (private mode, etc.)
  }
}

const store = writable<ProofRequest[]>(loadInitial());

store.subscribe((value) => persist(value));

function upsert(request: ProofRequest) {
  store.update((requests) => {
    const idx = requests.findIndex((r) => r.key === request.key);
    if (idx >= 0) {
      const next = [...requests];
      next[idx] = { ...requests[idx], ...request, updated_at: Date.now() };
      return next;
    }
    return [{ ...request, updated_at: Date.now() }, ...requests].slice(0, 100);
  });
}

function makeKey(agentId: string, listingHash?: string) {
  return `${agentId}::${listingHash || 'agent'}`;
}

export const proofRequests = readable<ProofRequest[]>([], (set) =>
  store.subscribe((value) => set(value))
);

export function markProofRequest(
  agentId: string,
  listingHash: string | undefined,
  status: ProofRequestStatus,
  message?: string,
  usingMock?: boolean
) {
  upsert({
    key: makeKey(agentId, listingHash),
    agent_id: agentId,
    listing_hash: listingHash,
    status,
    message,
    updated_at: Date.now(),
    usingMock,
  });
}

export function getProofKey(agentId: string, listingHash?: string) {
  return makeKey(agentId, listingHash);
}

export function getProofStatus(agentId: string, listingHash?: string): ProofRequestStatus | null {
  let current: ProofRequest[] = [];
  store.subscribe((value) => {
    current = value;
  })();
  const found = current.find((r) => r.key === makeKey(agentId, listingHash));
  return found?.status ?? null;
}

export function markProofFulfilled(agentId: string, listingHash?: string) {
  upsert({
    key: makeKey(agentId, listingHash),
    agent_id: agentId,
    listing_hash: listingHash,
    status: 'fulfilled',
    updated_at: Date.now(),
  });
}

export function markProofDenied(agentId: string, listingHash?: string, message?: string) {
  upsert({
    key: makeKey(agentId, listingHash),
    agent_id: agentId,
    listing_hash: listingHash,
    status: 'denied',
    message,
    updated_at: Date.now(),
  });
}
