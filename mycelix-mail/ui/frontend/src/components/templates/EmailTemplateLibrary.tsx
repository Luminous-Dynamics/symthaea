// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Template Library
 *
 * Pre-built email templates for common use cases.
 */

import React, { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Search,
  Star,
  Clock,
  Briefcase,
  Heart,
  PartyPopper,
  AlertCircle,
  CheckCircle,
  Users,
  Calendar,
  FileText,
  Megaphone,
  Gift,
  Sparkles,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

interface EmailTemplate {
  id: string;
  name: string;
  description: string;
  category: TemplateCategory;
  subject: string;
  body: string;
  variables: TemplateVariable[];
  tags: string[];
  isFavorite?: boolean;
  usageCount: number;
  preview?: string;
}

interface TemplateVariable {
  name: string;
  description: string;
  defaultValue?: string;
  required?: boolean;
}

type TemplateCategory =
  | 'professional'
  | 'personal'
  | 'sales'
  | 'support'
  | 'marketing'
  | 'notifications'
  | 'celebrations'
  | 'meetings';

const categoryIcons: Record<TemplateCategory, React.ElementType> = {
  professional: Briefcase,
  personal: Heart,
  sales: Megaphone,
  support: AlertCircle,
  marketing: Sparkles,
  notifications: CheckCircle,
  celebrations: PartyPopper,
  meetings: Calendar,
};

const categoryColors: Record<TemplateCategory, string> = {
  professional: 'bg-blue-100 text-blue-800',
  personal: 'bg-pink-100 text-pink-800',
  sales: 'bg-green-100 text-green-800',
  support: 'bg-orange-100 text-orange-800',
  marketing: 'bg-purple-100 text-purple-800',
  notifications: 'bg-gray-100 text-gray-800',
  celebrations: 'bg-yellow-100 text-yellow-800',
  meetings: 'bg-cyan-100 text-cyan-800',
};

// ============================================================================
// Built-in Templates
// ============================================================================

const builtInTemplates: EmailTemplate[] = [
  // Professional Templates
  {
    id: 'prof-intro',
    name: 'Professional Introduction',
    description: 'Introduce yourself professionally to a new contact',
    category: 'professional',
    subject: 'Introduction - {{your_name}} from {{company}}',
    body: `Dear {{recipient_name}},

I hope this email finds you well. My name is {{your_name}}, and I am the {{your_title}} at {{company}}.

{{introduction_context}}

I would love the opportunity to connect and discuss how we might collaborate. Would you be available for a brief call next week?

Best regards,
{{your_name}}
{{your_title}}
{{company}}`,
    variables: [
      { name: 'your_name', description: 'Your full name', required: true },
      { name: 'your_title', description: 'Your job title', required: true },
      { name: 'company', description: 'Your company name', required: true },
      { name: 'recipient_name', description: "Recipient's name", required: true },
      { name: 'introduction_context', description: 'Context for reaching out', defaultValue: 'I came across your work and was impressed by your expertise in the field.' },
    ],
    tags: ['introduction', 'networking', 'business'],
    usageCount: 1250,
  },
  {
    id: 'prof-followup',
    name: 'Meeting Follow-up',
    description: 'Follow up after a meeting with next steps',
    category: 'professional',
    subject: 'Follow-up: {{meeting_topic}} - {{meeting_date}}',
    body: `Hi {{recipient_name}},

Thank you for taking the time to meet with me {{meeting_timeframe}}. I really enjoyed our discussion about {{meeting_topic}}.

As discussed, here are the key takeaways and next steps:

{{action_items}}

Please let me know if you have any questions or if there's anything else you'd like to discuss.

Looking forward to our continued collaboration.

Best regards,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Recipient's name", required: true },
      { name: 'meeting_topic', description: 'Topic of the meeting', required: true },
      { name: 'meeting_date', description: 'Date of meeting', required: true },
      { name: 'meeting_timeframe', description: 'e.g., "today", "yesterday"', defaultValue: 'today' },
      { name: 'action_items', description: 'List of next steps', required: true },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['follow-up', 'meeting', 'action items'],
    usageCount: 2340,
  },
  {
    id: 'prof-referral',
    name: 'Referral Request',
    description: 'Ask for a professional referral or recommendation',
    category: 'professional',
    subject: 'Referral Request - {{position_or_opportunity}}',
    body: `Dear {{recipient_name}},

I hope you're doing well. I'm reaching out because I'm currently {{situation}}, and I thought of you as someone who might be able to help.

{{context}}

Would you be comfortable providing a referral or introduction to {{target}}? I would greatly appreciate any guidance or connections you could offer.

Please let me know if you need any additional information from me.

Thank you so much for your time and consideration.

Warm regards,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Contact's name", required: true },
      { name: 'position_or_opportunity', description: 'What you are pursuing', required: true },
      { name: 'situation', description: 'Your current situation', required: true },
      { name: 'context', description: 'Additional context', defaultValue: '' },
      { name: 'target', description: 'Who you want to be referred to', required: true },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['referral', 'networking', 'career'],
    usageCount: 890,
  },

  // Sales Templates
  {
    id: 'sales-intro',
    name: 'Sales Introduction',
    description: 'Initial outreach to a potential customer',
    category: 'sales',
    subject: 'Quick question about {{company_pain_point}}',
    body: `Hi {{recipient_name}},

I noticed that {{company_name}} is {{observation}}. Many companies in {{industry}} face similar challenges with {{pain_point}}.

At {{your_company}}, we help businesses like yours {{value_proposition}}.

Would you be open to a 15-minute call to explore if we might be able to help?

Best,
{{your_name}}
{{your_title}}
{{your_company}}`,
    variables: [
      { name: 'recipient_name', description: "Prospect's name", required: true },
      { name: 'company_name', description: "Prospect's company", required: true },
      { name: 'company_pain_point', description: 'Their main challenge', required: true },
      { name: 'observation', description: 'What you observed about their company', required: true },
      { name: 'industry', description: 'Their industry', required: true },
      { name: 'pain_point', description: 'The problem you solve', required: true },
      { name: 'value_proposition', description: 'What you offer', required: true },
      { name: 'your_company', description: 'Your company name', required: true },
      { name: 'your_name', description: 'Your name', required: true },
      { name: 'your_title', description: 'Your job title', required: true },
    ],
    tags: ['sales', 'prospecting', 'cold email'],
    usageCount: 3450,
  },
  {
    id: 'sales-followup',
    name: 'Sales Follow-up',
    description: 'Follow up with a prospect after initial contact',
    category: 'sales',
    subject: 'Re: {{original_subject}}',
    body: `Hi {{recipient_name}},

I wanted to follow up on my previous email about {{topic}}. I understand you're busy, so I'll keep this brief.

{{new_value_add}}

Would {{day_time}} work for a quick 10-minute call?

Best,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Prospect's name", required: true },
      { name: 'original_subject', description: 'Subject of previous email', required: true },
      { name: 'topic', description: 'What you discussed before', required: true },
      { name: 'new_value_add', description: 'New information or value', defaultValue: "I thought you might find this case study relevant - it shows how a similar company achieved 40% improvement in their metrics." },
      { name: 'day_time', description: 'Suggested meeting time', defaultValue: 'Tuesday at 2pm' },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['sales', 'follow-up', 'persistence'],
    usageCount: 2100,
  },

  // Support Templates
  {
    id: 'support-ack',
    name: 'Support Ticket Acknowledgment',
    description: 'Acknowledge receipt of a support request',
    category: 'support',
    subject: 'Re: {{ticket_subject}} [Ticket #{{ticket_id}}]',
    body: `Hi {{customer_name}},

Thank you for contacting {{company_name}} support. We've received your request and a member of our team is looking into it.

Your ticket number is: #{{ticket_id}}

Here's what you can expect:
- We aim to provide an initial response within {{response_time}}
- You'll receive updates as we investigate
- You can reply to this email to add more information

In the meantime, you might find these resources helpful:
{{helpful_links}}

Thank you for your patience.

Best regards,
{{agent_name}}
{{company_name}} Support Team`,
    variables: [
      { name: 'customer_name', description: "Customer's name", required: true },
      { name: 'company_name', description: 'Your company name', required: true },
      { name: 'ticket_subject', description: 'Original subject', required: true },
      { name: 'ticket_id', description: 'Ticket number', required: true },
      { name: 'response_time', description: 'Expected response time', defaultValue: '24 hours' },
      { name: 'helpful_links', description: 'Links to relevant help articles', defaultValue: '' },
      { name: 'agent_name', description: 'Support agent name', required: true },
    ],
    tags: ['support', 'ticket', 'acknowledgment'],
    usageCount: 5600,
  },
  {
    id: 'support-resolution',
    name: 'Issue Resolution',
    description: 'Notify customer their issue has been resolved',
    category: 'support',
    subject: 'Resolved: {{ticket_subject}} [Ticket #{{ticket_id}}]',
    body: `Hi {{customer_name}},

Great news! We've resolved the issue you reported regarding {{issue_summary}}.

Here's what we did:
{{resolution_details}}

{{next_steps}}

If you experience any further issues, please don't hesitate to reach out. We're here to help!

Thank you for your patience throughout this process.

Best regards,
{{agent_name}}
{{company_name}} Support Team

---
Was this response helpful? Reply with "Yes" or "No" to let us know.`,
    variables: [
      { name: 'customer_name', description: "Customer's name", required: true },
      { name: 'company_name', description: 'Your company name', required: true },
      { name: 'ticket_subject', description: 'Original subject', required: true },
      { name: 'ticket_id', description: 'Ticket number', required: true },
      { name: 'issue_summary', description: 'Brief summary of the issue', required: true },
      { name: 'resolution_details', description: 'What was done to fix it', required: true },
      { name: 'next_steps', description: 'Any actions customer should take', defaultValue: '' },
      { name: 'agent_name', description: 'Support agent name', required: true },
    ],
    tags: ['support', 'resolution', 'ticket'],
    usageCount: 4200,
  },

  // Meeting Templates
  {
    id: 'meeting-invite',
    name: 'Meeting Invitation',
    description: 'Invite someone to a meeting',
    category: 'meetings',
    subject: 'Meeting Request: {{meeting_topic}}',
    body: `Hi {{recipient_name}},

I'd like to schedule a meeting to discuss {{meeting_topic}}.

Meeting Details:
- Date: {{proposed_date}}
- Time: {{proposed_time}}
- Duration: {{duration}}
- Location: {{location}}

Agenda:
{{agenda}}

Please let me know if this time works for you, or suggest an alternative that fits your schedule better.

Looking forward to our discussion.

Best regards,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Attendee's name", required: true },
      { name: 'meeting_topic', description: 'Meeting topic', required: true },
      { name: 'proposed_date', description: 'Proposed date', required: true },
      { name: 'proposed_time', description: 'Proposed time', required: true },
      { name: 'duration', description: 'Meeting length', defaultValue: '30 minutes' },
      { name: 'location', description: 'Meeting location or video link', required: true },
      { name: 'agenda', description: 'Meeting agenda', required: true },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['meeting', 'invitation', 'scheduling'],
    usageCount: 3800,
  },
  {
    id: 'meeting-reschedule',
    name: 'Meeting Reschedule Request',
    description: 'Request to reschedule a scheduled meeting',
    category: 'meetings',
    subject: 'Reschedule Request: {{meeting_topic}}',
    body: `Hi {{recipient_name}},

I apologize, but I need to reschedule our meeting originally planned for {{original_date}}.

{{reason}}

Would any of these alternative times work for you?
{{alternative_times}}

I apologize for any inconvenience this may cause. Please let me know what works best for you.

Thank you for your understanding.

Best regards,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Attendee's name", required: true },
      { name: 'meeting_topic', description: 'Meeting topic', required: true },
      { name: 'original_date', description: 'Original date/time', required: true },
      { name: 'reason', description: 'Reason for rescheduling', defaultValue: 'Due to an unexpected conflict in my schedule' },
      { name: 'alternative_times', description: 'Alternative time options', required: true },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['meeting', 'reschedule', 'scheduling'],
    usageCount: 1560,
  },

  // Celebration Templates
  {
    id: 'celebration-birthday',
    name: 'Birthday Wishes',
    description: 'Send birthday wishes to a colleague or friend',
    category: 'celebrations',
    subject: 'Happy Birthday, {{recipient_name}}! 🎂',
    body: `Dear {{recipient_name}},

Happy Birthday! 🎉

Wishing you a wonderful day filled with joy, laughter, and all your favorite things.

{{personal_message}}

May this year bring you success, happiness, and all the things you've been hoping for.

Best wishes,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Birthday person's name", required: true },
      { name: 'personal_message', description: 'Personal message', defaultValue: "It's been a pleasure working with you, and I hope you have an amazing birthday celebration!" },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['birthday', 'celebration', 'personal'],
    usageCount: 2100,
  },
  {
    id: 'celebration-congrats',
    name: 'Congratulations',
    description: 'Congratulate someone on an achievement',
    category: 'celebrations',
    subject: 'Congratulations on {{achievement}}! 🎉',
    body: `Dear {{recipient_name}},

Congratulations on {{achievement}}! This is such wonderful news, and I wanted to reach out to express my sincere congratulations.

{{personal_message}}

You absolutely deserve this, and I know this is just the beginning of many more successes to come.

Best wishes,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Person's name", required: true },
      { name: 'achievement', description: 'What they achieved', required: true },
      { name: 'personal_message', description: 'Personal message', defaultValue: 'Your hard work and dedication have really paid off.' },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['congratulations', 'celebration', 'achievement'],
    usageCount: 1780,
  },

  // Personal Templates
  {
    id: 'personal-thankyou',
    name: 'Thank You Note',
    description: 'Express gratitude to someone',
    category: 'personal',
    subject: 'Thank You, {{recipient_name}}!',
    body: `Dear {{recipient_name}},

I wanted to take a moment to express my sincere gratitude for {{reason_for_thanks}}.

{{personal_message}}

Your {{quality}} means so much to me, and I truly appreciate everything you've done.

Thank you again from the bottom of my heart.

Warmly,
{{your_name}}`,
    variables: [
      { name: 'recipient_name', description: "Person's name", required: true },
      { name: 'reason_for_thanks', description: 'Why you are thanking them', required: true },
      { name: 'personal_message', description: 'Personal message', defaultValue: '' },
      { name: 'quality', description: 'Quality you appreciate', defaultValue: 'kindness and generosity' },
      { name: 'your_name', description: 'Your name', required: true },
    ],
    tags: ['thank you', 'gratitude', 'personal'],
    usageCount: 3200,
  },

  // Marketing Templates
  {
    id: 'marketing-newsletter',
    name: 'Newsletter',
    description: 'Monthly/weekly newsletter template',
    category: 'marketing',
    subject: '{{newsletter_title}} - {{month_year}}',
    body: `Hi {{recipient_name}},

Welcome to this {{frequency}}'s edition of {{newsletter_name}}!

📰 WHAT'S NEW
{{whats_new}}

💡 TIP OF THE {{frequency.toUpperCase()}}
{{tip}}

📅 UPCOMING EVENTS
{{events}}

{{cta}}

As always, we'd love to hear from you. Reply to this email with your thoughts, questions, or topics you'd like us to cover.

Best,
{{sender_name}}
{{company_name}}

---
You're receiving this because you subscribed to {{newsletter_name}}.
[Unsubscribe]({{unsubscribe_link}}) | [Update Preferences]({{preferences_link}})`,
    variables: [
      { name: 'recipient_name', description: "Subscriber's name", defaultValue: 'there' },
      { name: 'newsletter_title', description: 'Newsletter title', required: true },
      { name: 'newsletter_name', description: 'Newsletter name', required: true },
      { name: 'month_year', description: 'Month and year', required: true },
      { name: 'frequency', description: 'week or month', defaultValue: 'month' },
      { name: 'whats_new', description: 'New updates and announcements', required: true },
      { name: 'tip', description: 'Helpful tip', required: true },
      { name: 'events', description: 'Upcoming events', defaultValue: 'No events scheduled this period.' },
      { name: 'cta', description: 'Call to action', defaultValue: '' },
      { name: 'sender_name', description: 'Sender name', required: true },
      { name: 'company_name', description: 'Company name', required: true },
      { name: 'unsubscribe_link', description: 'Unsubscribe URL', required: true },
      { name: 'preferences_link', description: 'Preferences URL', required: true },
    ],
    tags: ['newsletter', 'marketing', 'updates'],
    usageCount: 890,
  },

  // Notification Templates
  {
    id: 'notif-password',
    name: 'Password Reset',
    description: 'Password reset notification',
    category: 'notifications',
    subject: 'Reset Your {{app_name}} Password',
    body: `Hi {{user_name}},

We received a request to reset your password for your {{app_name}} account.

Click the button below to reset your password:

[Reset Password]({{reset_link}})

This link will expire in {{expiry_time}}.

If you didn't request this, you can safely ignore this email. Your password won't be changed.

If you need help, contact our support team at {{support_email}}.

Best,
The {{app_name}} Team

---
This is an automated message. Please do not reply directly to this email.`,
    variables: [
      { name: 'user_name', description: "User's name", required: true },
      { name: 'app_name', description: 'Application name', required: true },
      { name: 'reset_link', description: 'Password reset URL', required: true },
      { name: 'expiry_time', description: 'Link expiry time', defaultValue: '24 hours' },
      { name: 'support_email', description: 'Support email address', required: true },
    ],
    tags: ['notification', 'security', 'password'],
    usageCount: 12000,
  },
];

// ============================================================================
// Component
// ============================================================================

interface EmailTemplateLibraryProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTemplate: (template: EmailTemplate, values: Record<string, string>) => void;
}

export const EmailTemplateLibrary: React.FC<EmailTemplateLibraryProps> = ({
  isOpen,
  onClose,
  onSelectTemplate,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<TemplateCategory | 'all'>('all');
  const [selectedTemplate, setSelectedTemplate] = useState<EmailTemplate | null>(null);
  const [variableValues, setVariableValues] = useState<Record<string, string>>({});
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  const categories: Array<{ value: TemplateCategory | 'all'; label: string }> = [
    { value: 'all', label: 'All Templates' },
    { value: 'professional', label: 'Professional' },
    { value: 'personal', label: 'Personal' },
    { value: 'sales', label: 'Sales' },
    { value: 'support', label: 'Support' },
    { value: 'marketing', label: 'Marketing' },
    { value: 'notifications', label: 'Notifications' },
    { value: 'celebrations', label: 'Celebrations' },
    { value: 'meetings', label: 'Meetings' },
  ];

  const filteredTemplates = useMemo(() => {
    return builtInTemplates.filter(template => {
      const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
      const matchesSearch = searchQuery === '' ||
        template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));

      return matchesCategory && matchesSearch;
    }).sort((a, b) => {
      // Favorites first, then by usage count
      const aFav = favorites.has(a.id) ? 1 : 0;
      const bFav = favorites.has(b.id) ? 1 : 0;
      if (aFav !== bFav) return bFav - aFav;
      return b.usageCount - a.usageCount;
    });
  }, [searchQuery, selectedCategory, favorites]);

  const handleSelectTemplate = (template: EmailTemplate) => {
    setSelectedTemplate(template);
    // Initialize variable values with defaults
    const defaults: Record<string, string> = {};
    template.variables.forEach(v => {
      defaults[v.name] = v.defaultValue || '';
    });
    setVariableValues(defaults);
  };

  const handleUseTemplate = () => {
    if (selectedTemplate) {
      onSelectTemplate(selectedTemplate, variableValues);
      onClose();
    }
  };

  const toggleFavorite = (templateId: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(templateId)) {
        next.delete(templateId);
      } else {
        next.add(templateId);
      }
      return next;
    });
  };

  const renderPreview = () => {
    if (!selectedTemplate) return null;

    let subject = selectedTemplate.subject;
    let body = selectedTemplate.body;

    // Replace variables with values
    Object.entries(variableValues).forEach(([key, value]) => {
      const placeholder = `{{${key}}}`;
      subject = subject.replace(new RegExp(placeholder, 'g'), value || `[${key}]`);
      body = body.replace(new RegExp(placeholder, 'g'), value || `[${key}]`);
    });

    return (
      <div className="border rounded-lg p-4 bg-gray-50">
        <div className="text-sm text-gray-500 mb-1">Preview:</div>
        <div className="font-medium mb-2">Subject: {subject}</div>
        <div className="whitespace-pre-wrap text-sm">{body}</div>
      </div>
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Email Template Library
          </DialogTitle>
          <DialogDescription>
            Choose from our collection of professionally crafted email templates
          </DialogDescription>
        </DialogHeader>

        <div className="flex gap-4 h-[600px]">
          {/* Left Panel - Template List */}
          <div className="w-1/2 flex flex-col">
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search templates..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <Tabs value={selectedCategory} onValueChange={(v) => setSelectedCategory(v as TemplateCategory | 'all')}>
              <TabsList className="flex flex-wrap gap-1 h-auto mb-4">
                {categories.map(cat => (
                  <TabsTrigger key={cat.value} value={cat.value} className="text-xs px-2 py-1">
                    {cat.label}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>

            <ScrollArea className="flex-1">
              <div className="space-y-2 pr-4">
                {filteredTemplates.map(template => {
                  const Icon = categoryIcons[template.category];
                  const isSelected = selectedTemplate?.id === template.id;

                  return (
                    <div
                      key={template.id}
                      onClick={() => handleSelectTemplate(template)}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        isSelected
                          ? 'border-primary bg-primary/5'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4 text-gray-500" />
                          <span className="font-medium text-sm">{template.name}</span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFavorite(template.id);
                          }}
                          className="p-1 hover:bg-gray-100 rounded"
                        >
                          <Star
                            className={`h-4 w-4 ${
                              favorites.has(template.id) ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'
                            }`}
                          />
                        </button>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{template.description}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="secondary" className={`text-xs ${categoryColors[template.category]}`}>
                          {template.category}
                        </Badge>
                        <span className="text-xs text-gray-400 flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {template.usageCount.toLocaleString()} uses
                        </span>
                      </div>
                    </div>
                  );
                })}

                {filteredTemplates.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No templates found matching your search.
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Right Panel - Template Preview & Variables */}
          <div className="w-1/2 flex flex-col border-l pl-4">
            {selectedTemplate ? (
              <>
                <div className="mb-4">
                  <h3 className="font-medium">{selectedTemplate.name}</h3>
                  <p className="text-sm text-gray-500">{selectedTemplate.description}</p>
                </div>

                <ScrollArea className="flex-1">
                  <div className="space-y-4 pr-4">
                    <div>
                      <h4 className="text-sm font-medium mb-2">Fill in the details:</h4>
                      <div className="space-y-3">
                        {selectedTemplate.variables.map(variable => (
                          <div key={variable.name}>
                            <label className="text-xs text-gray-500 flex items-center gap-1">
                              {variable.description}
                              {variable.required && <span className="text-red-500">*</span>}
                            </label>
                            <Input
                              value={variableValues[variable.name] || ''}
                              onChange={(e) => setVariableValues(prev => ({
                                ...prev,
                                [variable.name]: e.target.value,
                              }))}
                              placeholder={variable.defaultValue || `Enter ${variable.name.replace(/_/g, ' ')}`}
                              className="mt-1"
                            />
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="pt-4">
                      {renderPreview()}
                    </div>
                  </div>
                </ScrollArea>

                <div className="pt-4 mt-4 border-t">
                  <Button onClick={handleUseTemplate} className="w-full">
                    Use This Template
                  </Button>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>Select a template to preview</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EmailTemplateLibrary;
