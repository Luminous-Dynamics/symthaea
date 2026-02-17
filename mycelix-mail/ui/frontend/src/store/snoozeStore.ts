import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from './toastStore';

export interface SnoozedEmail {
  emailId: string;
  snoozedUntil: string; // ISO date string
  originalFolderId: string;
  reminderSent: boolean;
  createdAt: string;
}

export type SnoozePreset = 'later-today' | 'tomorrow' | 'this-weekend' | 'next-week' | 'custom';

interface SnoozeStore {
  snoozedEmails: SnoozedEmail[];
  snoozeEmail: (emailId: string, until: Date, folderId: string) => void;
  unsnoozeEmail: (emailId: string) => void;
  checkDueEmails: () => SnoozedEmail[];
  getSnoozedEmail: (emailId: string) => SnoozedEmail | null;
  getAllSnoozed: () => SnoozedEmail[];
}

const generateId = () => Math.random().toString(36).substring(2, 11);

export const useSnoozeStore = create<SnoozeStore>()(
  persist(
    (set, get) => ({
      snoozedEmails: [],

      snoozeEmail: (emailId, until, folderId) => {
        // Remove existing snooze for this email if present
        const filtered = get().snoozedEmails.filter((se) => se.emailId !== emailId);

        const newSnooze: SnoozedEmail = {
          emailId,
          snoozedUntil: until.toISOString(),
          originalFolderId: folderId,
          reminderSent: false,
          createdAt: new Date().toISOString(),
        };

        set({ snoozedEmails: [...filtered, newSnooze] });
        toast.success(`Email snoozed until ${until.toLocaleDateString()} ${until.toLocaleTimeString()}`);
      },

      unsnoozeEmail: (emailId) => {
        set((state) => ({
          snoozedEmails: state.snoozedEmails.filter((se) => se.emailId !== emailId),
        }));
      },

      checkDueEmails: () => {
        const now = new Date();
        const dueEmails: SnoozedEmail[] = [];

        get().snoozedEmails.forEach((snoozed) => {
          const dueTime = new Date(snoozed.snoozedUntil);

          if (dueTime <= now && !snoozed.reminderSent) {
            dueEmails.push(snoozed);

            // Mark reminder as sent
            set((state) => ({
              snoozedEmails: state.snoozedEmails.map((se) =>
                se.emailId === snoozed.emailId
                  ? { ...se, reminderSent: true }
                  : se
              ),
            }));
          }
        });

        return dueEmails;
      },

      getSnoozedEmail: (emailId) => {
        return get().snoozedEmails.find((se) => se.emailId === emailId) || null;
      },

      getAllSnoozed: () => {
        return get().snoozedEmails;
      },
    }),
    {
      name: 'snooze-storage',
    }
  )
);

// Helper functions to calculate snooze dates
export const getSnoozeDate = (preset: SnoozePreset): Date | null => {
  const now = new Date();

  switch (preset) {
    case 'later-today': {
      // 4 hours from now
      const date = new Date(now);
      date.setHours(date.getHours() + 4);
      return date;
    }

    case 'tomorrow': {
      // Tomorrow at 9 AM
      const date = new Date(now);
      date.setDate(date.getDate() + 1);
      date.setHours(9, 0, 0, 0);
      return date;
    }

    case 'this-weekend': {
      // Next Saturday at 9 AM
      const date = new Date(now);
      const daysUntilSaturday = (6 - date.getDay() + 7) % 7 || 7;
      date.setDate(date.getDate() + daysUntilSaturday);
      date.setHours(9, 0, 0, 0);
      return date;
    }

    case 'next-week': {
      // Next Monday at 9 AM
      const date = new Date(now);
      const daysUntilMonday = (1 - date.getDay() + 7) % 7 || 7;
      date.setDate(date.getDate() + daysUntilMonday);
      date.setHours(9, 0, 0, 0);
      return date;
    }

    case 'custom':
      return null; // Let user pick

    default:
      return null;
  }
};

// Background checker - run this in your app initialization
if (typeof window !== 'undefined') {
  // Check for due emails every minute
  setInterval(() => {
    const { checkDueEmails } = useSnoozeStore.getState();
    const dueEmails = checkDueEmails();

    if (dueEmails.length > 0) {
      dueEmails.forEach((snoozed) => {
        toast.info(`Email returned from snooze`, 10000);

        // Show browser notification if permitted
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('Email Reminder', {
            body: 'A snoozed email is now ready for your attention',
            icon: '/vite.svg',
          });
        }
      });
    }
  }, 60000); // Check every minute
}
