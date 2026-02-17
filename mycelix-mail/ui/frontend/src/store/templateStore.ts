import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type TemplateCategory = 'greeting' | 'follow-up' | 'meeting' | 'support' | 'custom';

export interface EmailTemplate {
  id: string;
  name: string;
  subject: string;
  body: string;
  category: TemplateCategory;
  variables: string[]; // e.g., ['name', 'date', 'time']
  useCount: number;
  lastUsed?: string;
  createdAt: string;
}

interface TemplateStore {
  templates: EmailTemplate[];
  addTemplate: (template: Omit<EmailTemplate, 'id' | 'createdAt' | 'useCount' | 'lastUsed'>) => void;
  updateTemplate: (id: string, updates: Partial<Omit<EmailTemplate, 'id'>>) => void;
  deleteTemplate: (id: string) => void;
  getTemplatesByCategory: (category: TemplateCategory) => EmailTemplate[];
  incrementUseCount: (id: string) => void;
  getPopularTemplates: (limit?: number) => EmailTemplate[];
}

const generateId = () => Math.random().toString(36).substring(2, 11);

// Default templates
const defaultTemplates: Omit<EmailTemplate, 'id' | 'createdAt' | 'useCount' | 'lastUsed'>[] = [
  {
    name: 'Quick Thanks',
    subject: 'Thank you',
    body: 'Hi {{name}},\n\nThank you for your email. I appreciate you reaching out.\n\nBest regards',
    category: 'greeting',
    variables: ['name'],
  },
  {
    name: 'Meeting Follow-up',
    subject: 'Follow-up: {{meeting_topic}}',
    body: 'Hi {{name}},\n\nThank you for meeting with me {{date}} to discuss {{meeting_topic}}.\n\nAs discussed, the next steps are:\n- [Action item 1]\n- [Action item 2]\n- [Action item 3]\n\nLooking forward to working together.\n\nBest regards',
    category: 'meeting',
    variables: ['name', 'date', 'meeting_topic'],
  },
  {
    name: 'Out of Office',
    subject: 'Out of Office: Re: {{subject}}',
    body: 'Thank you for your email.\n\nI am currently out of the office and will have limited access to email until {{return_date}}. I will respond to your message as soon as possible upon my return.\n\nFor urgent matters, please contact {{alternate_contact}} at {{alternate_email}}.\n\nBest regards',
    category: 'custom',
    variables: ['subject', 'return_date', 'alternate_contact', 'alternate_email'],
  },
  {
    name: 'Schedule Meeting',
    subject: 'Meeting Request: {{meeting_topic}}',
    body: 'Hi {{name}},\n\nI hope this email finds you well. I would like to schedule a meeting to discuss {{meeting_topic}}.\n\nWould any of these times work for you?\n- {{time_option_1}}\n- {{time_option_2}}\n- {{time_option_3}}\n\nPlease let me know what works best for your schedule.\n\nBest regards',
    category: 'meeting',
    variables: ['name', 'meeting_topic', 'time_option_1', 'time_option_2', 'time_option_3'],
  },
  {
    name: 'Support Acknowledgment',
    subject: 'Re: Support Request - {{ticket_number}}',
    body: 'Dear {{name}},\n\nThank you for contacting support. We have received your request regarding {{issue}}.\n\nYour ticket number is: {{ticket_number}}\n\nOur team is reviewing your case and will respond within {{response_time}}. We appreciate your patience.\n\nBest regards,\nSupport Team',
    category: 'support',
    variables: ['name', 'issue', 'ticket_number', 'response_time'],
  },
  {
    name: 'Follow-up Request',
    subject: 'Following up: {{topic}}',
    body: 'Hi {{name}},\n\nI wanted to follow up on {{topic}}. Have you had a chance to review the information I shared?\n\nPlease let me know if you have any questions or need any additional information.\n\nLooking forward to hearing from you.\n\nBest regards',
    category: 'follow-up',
    variables: ['name', 'topic'],
  },
  {
    name: 'Introduction',
    subject: 'Introduction',
    body: 'Hi {{name}},\n\nMy name is {{my_name}} and I am reaching out regarding {{purpose}}.\n\n{{message_body}}\n\nI would love to connect and discuss this further. Would you be available for a brief call {{suggested_time}}?\n\nBest regards',
    category: 'greeting',
    variables: ['name', 'my_name', 'purpose', 'message_body', 'suggested_time'],
  },
  {
    name: 'Project Update',
    subject: 'Project Update: {{project_name}}',
    body: 'Hi Team,\n\nHere\'s an update on {{project_name}} for {{date}}:\n\n**Completed:**\n- [Item 1]\n- [Item 2]\n\n**In Progress:**\n- [Item 1]\n- [Item 2]\n\n**Upcoming:**\n- [Item 1]\n- [Item 2]\n\n**Blockers:**\n[Any blockers or concerns]\n\nNext update: {{next_update_date}}\n\nBest regards',
    category: 'custom',
    variables: ['project_name', 'date', 'next_update_date'],
  },
];

export const useTemplateStore = create<TemplateStore>()(
  persist(
    (set, get) => ({
      templates: defaultTemplates.map((template) => ({
        ...template,
        id: generateId(),
        createdAt: new Date().toISOString(),
        useCount: 0,
      })),

      addTemplate: (template) => {
        const newTemplate: EmailTemplate = {
          ...template,
          id: generateId(),
          createdAt: new Date().toISOString(),
          useCount: 0,
        };

        set((state) => ({
          templates: [...state.templates, newTemplate],
        }));
      },

      updateTemplate: (id, updates) => {
        set((state) => ({
          templates: state.templates.map((template) =>
            template.id === id ? { ...template, ...updates } : template
          ),
        }));
      },

      deleteTemplate: (id) => {
        set((state) => ({
          templates: state.templates.filter((template) => template.id !== id),
        }));
      },

      getTemplatesByCategory: (category) => {
        return get().templates.filter((template) => template.category === category);
      },

      incrementUseCount: (id) => {
        set((state) => ({
          templates: state.templates.map((template) =>
            template.id === id
              ? {
                  ...template,
                  useCount: template.useCount + 1,
                  lastUsed: new Date().toISOString(),
                }
              : template
          ),
        }));
      },

      getPopularTemplates: (limit = 5) => {
        return [...get().templates]
          .sort((a, b) => b.useCount - a.useCount)
          .slice(0, limit);
      },
    }),
    {
      name: 'template-storage',
    }
  )
);
