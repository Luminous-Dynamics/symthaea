import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Contact {
  email: string;
  name?: string;
  avatar?: string; // URL or data URI for custom avatar
  color: string; // Generated color for initials
  lastContactDate: string;
  emailCount: number;
  isVip?: boolean;
  notes?: string;
  organization?: string;
  createdAt: string;
  updatedAt: string;
}

interface ContactStore {
  contacts: Contact[];

  // Get contact by email
  getContact: (email: string) => Contact | undefined;

  // Add or update contact
  upsertContact: (email: string, data?: Partial<Contact>) => Contact;

  // Update contact
  updateContact: (email: string, data: Partial<Contact>) => void;

  // Delete contact
  deleteContact: (email: string) => void;

  // Mark as VIP
  toggleVip: (email: string) => void;

  // Record email interaction (increments count, updates last contact date)
  recordInteraction: (email: string, name?: string) => void;

  // Get all VIP contacts
  getVipContacts: () => Contact[];

  // Search contacts
  searchContacts: (query: string) => Contact[];

  // Get frequent contacts (sorted by email count)
  getFrequentContacts: (limit?: number) => Contact[];

  // Get recent contacts (sorted by last contact date)
  getRecentContacts: (limit?: number) => Contact[];
}

// Generate consistent color based on email
const getColorForEmail = (email: string): string => {
  const colors = [
    '#3B82F6', // Blue
    '#10B981', // Green
    '#F59E0B', // Amber
    '#EF4444', // Red
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#06B6D4', // Cyan
    '#F97316', // Orange
    '#14B8A6', // Teal
    '#6366F1', // Indigo
    '#84CC16', // Lime
    '#F43F5E', // Rose
  ];

  let hash = 0;
  for (let i = 0; i < email.length; i++) {
    hash = email.charCodeAt(i) + ((hash << 5) - hash);
  }

  return colors[Math.abs(hash) % colors.length];
};

export const useContactStore = create<ContactStore>()(
  persist(
    (set, get) => ({
      contacts: [],

      getContact: (email: string) => {
        return get().contacts.find((c) => c.email.toLowerCase() === email.toLowerCase());
      },

      upsertContact: (email: string, data?: Partial<Contact>) => {
        const existing = get().getContact(email);
        const now = new Date().toISOString();

        if (existing) {
          // Update existing contact
          set({
            contacts: get().contacts.map((c) =>
              c.email.toLowerCase() === email.toLowerCase()
                ? { ...c, ...data, updatedAt: now }
                : c
            ),
          });
          return get().getContact(email)!;
        } else {
          // Create new contact
          const newContact: Contact = {
            email,
            name: data?.name,
            avatar: data?.avatar,
            color: data?.color || getColorForEmail(email),
            lastContactDate: data?.lastContactDate || now,
            emailCount: data?.emailCount || 0,
            isVip: data?.isVip || false,
            notes: data?.notes,
            organization: data?.organization,
            createdAt: now,
            updatedAt: now,
          };

          set({
            contacts: [...get().contacts, newContact],
          });

          return newContact;
        }
      },

      updateContact: (email: string, data: Partial<Contact>) => {
        const now = new Date().toISOString();

        set({
          contacts: get().contacts.map((c) =>
            c.email.toLowerCase() === email.toLowerCase()
              ? { ...c, ...data, updatedAt: now }
              : c
          ),
        });
      },

      deleteContact: (email: string) => {
        set({
          contacts: get().contacts.filter(
            (c) => c.email.toLowerCase() !== email.toLowerCase()
          ),
        });
      },

      toggleVip: (email: string) => {
        const contact = get().getContact(email);
        if (contact) {
          get().updateContact(email, { isVip: !contact.isVip });
        } else {
          // Create contact and mark as VIP
          get().upsertContact(email, { isVip: true });
        }
      },

      recordInteraction: (email: string, name?: string) => {
        const existing = get().getContact(email);
        const now = new Date().toISOString();

        if (existing) {
          get().updateContact(email, {
            emailCount: existing.emailCount + 1,
            lastContactDate: now,
            name: name || existing.name, // Update name if provided
          });
        } else {
          get().upsertContact(email, {
            name,
            emailCount: 1,
            lastContactDate: now,
          });
        }
      },

      getVipContacts: () => {
        return get().contacts.filter((c) => c.isVip);
      },

      searchContacts: (query: string) => {
        const lowerQuery = query.toLowerCase();
        return get().contacts.filter(
          (c) =>
            c.email.toLowerCase().includes(lowerQuery) ||
            c.name?.toLowerCase().includes(lowerQuery) ||
            c.organization?.toLowerCase().includes(lowerQuery)
        );
      },

      getFrequentContacts: (limit = 10) => {
        return [...get().contacts]
          .sort((a, b) => b.emailCount - a.emailCount)
          .slice(0, limit);
      },

      getRecentContacts: (limit = 10) => {
        return [...get().contacts]
          .sort((a, b) =>
            new Date(b.lastContactDate).getTime() - new Date(a.lastContactDate).getTime()
          )
          .slice(0, limit);
      },
    }),
    {
      name: 'contact-storage',
    }
  )
);
