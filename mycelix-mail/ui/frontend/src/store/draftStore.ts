import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface EmailDraft {
  id: string;
  to: string;
  cc?: string;
  bcc?: string;
  subject: string;
  body: string;
  isHtml: boolean;
  createdAt: string;
  updatedAt: string;
  inReplyTo?: string; // Email ID this is a reply to
  mode?: 'new' | 'reply' | 'replyAll' | 'forward';
}

interface DraftState {
  drafts: EmailDraft[];
  currentDraftId: string | null;

  // Actions
  saveDraft: (draft: Omit<EmailDraft, 'id' | 'createdAt' | 'updatedAt'>) => string;
  updateDraft: (id: string, updates: Partial<Omit<EmailDraft, 'id' | 'createdAt'>>) => void;
  deleteDraft: (id: string) => void;
  getDraft: (id: string) => EmailDraft | undefined;
  setCurrentDraft: (id: string | null) => void;
  clearOldDrafts: (daysOld: number) => void;
}

export const useDraftStore = create<DraftState>()(
  persist(
    (set, get) => ({
      drafts: [],
      currentDraftId: null,

      saveDraft: (draft) => {
        const id = crypto.randomUUID();
        const now = new Date().toISOString();

        const newDraft: EmailDraft = {
          ...draft,
          id,
          createdAt: now,
          updatedAt: now,
          isHtml: true,
        };

        set((state) => ({
          drafts: [...state.drafts, newDraft],
          currentDraftId: id,
        }));

        return id;
      },

      updateDraft: (id, updates) => {
        set((state) => ({
          drafts: state.drafts.map((draft) =>
            draft.id === id
              ? {
                  ...draft,
                  ...updates,
                  updatedAt: new Date().toISOString(),
                }
              : draft
          ),
        }));
      },

      deleteDraft: (id) => {
        set((state) => ({
          drafts: state.drafts.filter((draft) => draft.id !== id),
          currentDraftId: state.currentDraftId === id ? null : state.currentDraftId,
        }));
      },

      getDraft: (id) => {
        return get().drafts.find((draft) => draft.id === id);
      },

      setCurrentDraft: (id) => {
        set({ currentDraftId: id });
      },

      clearOldDrafts: (daysOld) => {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - daysOld);

        set((state) => ({
          drafts: state.drafts.filter(
            (draft) => new Date(draft.updatedAt) > cutoffDate
          ),
        }));
      },
    }),
    {
      name: 'draft-storage',
      version: 1,
    }
  )
);

// Auto-clear old drafts on initialization (drafts older than 30 days)
if (typeof window !== 'undefined') {
  setTimeout(() => {
    useDraftStore.getState().clearOldDrafts(30);
  }, 1000);
}
