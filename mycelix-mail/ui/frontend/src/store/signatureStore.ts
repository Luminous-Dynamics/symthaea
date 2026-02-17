import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Signature {
  id: string;
  accountId: string; // Associate with specific email account
  name: string;
  content: string; // HTML content
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

interface SignatureStore {
  signatures: Signature[];
  addSignature: (signature: Omit<Signature, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateSignature: (id: string, updates: Partial<Omit<Signature, 'id'>>) => void;
  deleteSignature: (id: string) => void;
  getSignatureForAccount: (accountId: string) => Signature | null;
  setDefaultSignature: (accountId: string, signatureId: string) => void;
}

const generateId = () => Math.random().toString(36).substring(2, 11);

export const useSignatureStore = create<SignatureStore>()(
  persist(
    (set, get) => ({
      signatures: [],

      addSignature: (signature) => {
        const now = new Date().toISOString();
        const newSignature: Signature = {
          ...signature,
          id: generateId(),
          createdAt: now,
          updatedAt: now,
        };

        set((state) => ({
          signatures: [...state.signatures, newSignature],
        }));
      },

      updateSignature: (id, updates) => {
        set((state) => ({
          signatures: state.signatures.map((sig) =>
            sig.id === id
              ? {
                  ...sig,
                  ...updates,
                  updatedAt: new Date().toISOString(),
                }
              : sig
          ),
        }));
      },

      deleteSignature: (id) => {
        set((state) => ({
          signatures: state.signatures.filter((sig) => sig.id !== id),
        }));
      },

      getSignatureForAccount: (accountId) => {
        const signatures = get().signatures.filter(
          (sig) => sig.accountId === accountId
        );

        // Return default signature for account, or first signature
        const defaultSig = signatures.find((sig) => sig.isDefault);
        return defaultSig || signatures[0] || null;
      },

      setDefaultSignature: (accountId, signatureId) => {
        set((state) => ({
          signatures: state.signatures.map((sig) => ({
            ...sig,
            // Set this signature as default, unset others for same account
            isDefault:
              sig.id === signatureId
                ? true
                : sig.accountId === accountId
                ? false
                : sig.isDefault,
          })),
        }));
      },
    }),
    {
      name: 'signature-storage',
    }
  )
);
