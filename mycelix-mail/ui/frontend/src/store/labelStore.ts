import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from './toastStore';

export interface Label {
  id: string;
  name: string;
  color: string; // hex color
  icon?: string; // optional emoji
  emailCount: number;
  createdAt: string;
  updatedAt: string;
}

interface EmailLabelMap {
  [emailId: string]: string[]; // emailId -> labelIds[]
}

interface LabelStore {
  labels: Label[];
  emailLabels: EmailLabelMap;
  addLabel: (label: Omit<Label, 'id' | 'emailCount' | 'createdAt' | 'updatedAt'>) => void;
  updateLabel: (id: string, updates: Partial<Omit<Label, 'id' | 'createdAt'>>) => void;
  deleteLabel: (id: string) => void;
  addLabelToEmail: (emailId: string, labelId: string) => void;
  removeLabelFromEmail: (emailId: string, labelId: string) => void;
  getLabelsForEmail: (emailId: string) => Label[];
  getEmailsWithLabel: (labelId: string) => string[];
  bulkLabelEmails: (emailIds: string[], labelId: string) => void;
  bulkUnlabelEmails: (emailIds: string[], labelId: string) => void;
  toggleLabelOnEmail: (emailId: string, labelId: string) => void;
}

const generateId = () => Math.random().toString(36).substring(2, 11);

// Default labels with color-coding
const defaultLabels: Omit<Label, 'emailCount'>[] = [
  {
    id: 'label_work',
    name: 'Work',
    color: '#3B82F6',
    icon: '🔧',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'label_personal',
    name: 'Personal',
    color: '#10B981',
    icon: '👤',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'label_important',
    name: 'Important',
    color: '#F59E0B',
    icon: '⭐',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'label_followup',
    name: 'Follow Up',
    color: '#EF4444',
    icon: '📌',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'label_later',
    name: 'Later',
    color: '#8B5CF6',
    icon: '📅',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'label_receipts',
    name: 'Receipts',
    color: '#6B7280',
    icon: '🧾',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];

export const useLabelStore = create<LabelStore>()(
  persist(
    (set, get) => ({
      labels: defaultLabels.map(label => ({ ...label, emailCount: 0 })),
      emailLabels: {},

      addLabel: (labelData) => {
        const newLabel: Label = {
          ...labelData,
          id: `label_${generateId()}`,
          emailCount: 0,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };

        set((state) => ({
          labels: [...state.labels, newLabel],
        }));

        toast.success(`Label "${newLabel.name}" created`);
      },

      updateLabel: (id, updates) => {
        set((state) => ({
          labels: state.labels.map((label) =>
            label.id === id
              ? { ...label, ...updates, updatedAt: new Date().toISOString() }
              : label
          ),
        }));

        toast.success('Label updated');
      },

      deleteLabel: (id) => {
        const label = get().labels.find((l) => l.id === id);

        if (!label) return;

        // Remove label from all emails
        set((state) => {
          const newEmailLabels = { ...state.emailLabels };
          Object.keys(newEmailLabels).forEach((emailId) => {
            newEmailLabels[emailId] = newEmailLabels[emailId].filter((labelId) => labelId !== id);
            if (newEmailLabels[emailId].length === 0) {
              delete newEmailLabels[emailId];
            }
          });

          return {
            labels: state.labels.filter((l) => l.id !== id),
            emailLabels: newEmailLabels,
          };
        });

        toast.success(`Label "${label.name}" deleted`);
      },

      addLabelToEmail: (emailId, labelId) => {
        set((state) => {
          const currentLabels = state.emailLabels[emailId] || [];

          // Don't add if already present
          if (currentLabels.includes(labelId)) {
            return state;
          }

          const newEmailLabels = {
            ...state.emailLabels,
            [emailId]: [...currentLabels, labelId],
          };

          // Update email count for the label
          const updatedLabels = state.labels.map((label) =>
            label.id === labelId
              ? { ...label, emailCount: label.emailCount + 1 }
              : label
          );

          return {
            emailLabels: newEmailLabels,
            labels: updatedLabels,
          };
        });

        const label = get().labels.find((l) => l.id === labelId);
        if (label) {
          toast.success(`Added label "${label.name}"`);
        }
      },

      removeLabelFromEmail: (emailId, labelId) => {
        set((state) => {
          const currentLabels = state.emailLabels[emailId] || [];
          const newLabels = currentLabels.filter((id) => id !== labelId);

          const newEmailLabels = { ...state.emailLabels };
          if (newLabels.length === 0) {
            delete newEmailLabels[emailId];
          } else {
            newEmailLabels[emailId] = newLabels;
          }

          // Update email count for the label
          const updatedLabels = state.labels.map((label) =>
            label.id === labelId
              ? { ...label, emailCount: Math.max(0, label.emailCount - 1) }
              : label
          );

          return {
            emailLabels: newEmailLabels,
            labels: updatedLabels,
          };
        });

        const label = get().labels.find((l) => l.id === labelId);
        if (label) {
          toast.success(`Removed label "${label.name}"`);
        }
      },

      getLabelsForEmail: (emailId) => {
        const labelIds = get().emailLabels[emailId] || [];
        return get().labels.filter((label) => labelIds.includes(label.id));
      },

      getEmailsWithLabel: (labelId) => {
        const emailLabels = get().emailLabels;
        return Object.keys(emailLabels).filter((emailId) =>
          emailLabels[emailId].includes(labelId)
        );
      },

      bulkLabelEmails: (emailIds, labelId) => {
        emailIds.forEach((emailId) => {
          get().addLabelToEmail(emailId, labelId);
        });

        const label = get().labels.find((l) => l.id === labelId);
        if (label) {
          toast.success(`Added "${label.name}" to ${emailIds.length} email${emailIds.length > 1 ? 's' : ''}`);
        }
      },

      bulkUnlabelEmails: (emailIds, labelId) => {
        emailIds.forEach((emailId) => {
          get().removeLabelFromEmail(emailId, labelId);
        });

        const label = get().labels.find((l) => l.id === labelId);
        if (label) {
          toast.success(`Removed "${label.name}" from ${emailIds.length} email${emailIds.length > 1 ? 's' : ''}`);
        }
      },

      toggleLabelOnEmail: (emailId, labelId) => {
        const currentLabels = get().emailLabels[emailId] || [];
        if (currentLabels.includes(labelId)) {
          get().removeLabelFromEmail(emailId, labelId);
        } else {
          get().addLabelToEmail(emailId, labelId);
        }
      },
    }),
    {
      name: 'label-storage',
    }
  )
);

// Predefined color palette for labels
export const LABEL_COLORS = [
  { name: 'Blue', value: '#3B82F6' },
  { name: 'Green', value: '#10B981' },
  { name: 'Yellow', value: '#F59E0B' },
  { name: 'Red', value: '#EF4444' },
  { name: 'Purple', value: '#8B5CF6' },
  { name: 'Pink', value: '#EC4899' },
  { name: 'Indigo', value: '#6366F1' },
  { name: 'Gray', value: '#6B7280' },
  { name: 'Orange', value: '#F97316' },
  { name: 'Teal', value: '#14B8A6' },
  { name: 'Cyan', value: '#06B6D4' },
  { name: 'Lime', value: '#84CC16' },
];

// Common label icons
export const LABEL_ICONS = [
  '🔧', '👤', '⭐', '📌', '📅', '🧾', '💼', '🏠', '📧', '🎯',
  '🔔', '📝', '💡', '🔥', '❤️', '🎉', '📊', '🔒', '⚡', '🌟',
];
