import { writable, readable } from 'svelte/store';

export type GuardrailOverrideEntry = {
  id: string;
  created_at: number;
  note: string;
  item_hashes: string[];
  proof_states: Record<string, string | undefined>;
  risk_flags: Record<string, string | undefined>;
  transaction_ids?: string[];
};

const STORAGE_KEY = 'mycelix_guardrail_overrides';

function loadInitial(): GuardrailOverrideEntry[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as GuardrailOverrideEntry[]) : [];
  } catch {
    return [];
  }
}

function persist(entries: GuardrailOverrideEntry[]) {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // ignore persistence failures
  }
}

const store = writable<GuardrailOverrideEntry[]>(loadInitial());
store.subscribe((value) => persist(value));

export const guardrailOverrides = readable<GuardrailOverrideEntry[]>([], (set) =>
  store.subscribe((value) => set(value))
);

export function recordGuardrailOverride(entry: Omit<GuardrailOverrideEntry, 'id' | 'created_at'>) {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const created_at = Date.now();
  store.update((list) => [{ ...entry, id, created_at }, ...list].slice(0, 100));
  return id;
}

export function attachTransactions(id: string, transactionIds: string[]) {
  if (!transactionIds?.length) return;
  store.update((list) =>
    list.map((entry) => (entry.id === id ? { ...entry, transaction_ids: transactionIds } : entry))
  );
}
