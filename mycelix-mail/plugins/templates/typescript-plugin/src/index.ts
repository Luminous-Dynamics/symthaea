// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Plugin Template
 *
 * This is a starter template for building Mycelix Mail plugins.
 * Plugins can extend functionality, add integrations, or customize the email experience.
 */

import {
  Plugin,
  PluginContext,
  Email,
  Contact,
  Hook,
  Command,
  SettingsPanel,
  MenuContribution,
} from '@mycelix/plugin-api';

// ============================================================================
// Plugin Configuration
// ============================================================================

interface MyPluginSettings {
  enabled: boolean;
  apiKey: string;
  customOption: string;
}

const defaultSettings: MyPluginSettings = {
  enabled: true,
  apiKey: '',
  customOption: 'default',
};

// ============================================================================
// Main Plugin Class
// ============================================================================

export default class MyPlugin implements Plugin {
  private context: PluginContext | null = null;
  private settings: MyPluginSettings = defaultSettings;

  /**
   * Plugin metadata - must match package.json
   */
  get id(): string {
    return 'com.example.my-plugin';
  }

  get name(): string {
    return 'My Plugin';
  }

  get version(): string {
    return '1.0.0';
  }

  /**
   * Called when the plugin is loaded
   * Use this to initialize resources, register hooks, and set up the plugin
   */
  async activate(context: PluginContext): Promise<void> {
    this.context = context;

    // Load saved settings
    this.settings = {
      ...defaultSettings,
      ...await context.settings.get<Partial<MyPluginSettings>>(),
    };

    // Register hooks
    context.hooks.register('onEmailReceived', this.handleEmailReceived.bind(this));
    context.hooks.register('onEmailSent', this.handleEmailSent.bind(this));
    context.hooks.register('onComposeOpen', this.handleComposeOpen.bind(this));

    // Register commands
    context.commands.register('my-plugin.run', {
      title: 'Run My Plugin',
      handler: this.runCommand.bind(this),
    });

    context.commands.register('my-plugin.settings', {
      title: 'Open My Plugin Settings',
      handler: () => context.ui.openSettings('my-plugin'),
    });

    // Register menu items
    context.menus.register('email.actions', {
      id: 'my-plugin.process-email',
      label: 'Process with My Plugin',
      icon: 'sparkles',
      handler: this.processSelectedEmail.bind(this),
    });

    // Register toolbar button
    context.ui.registerToolbarButton({
      id: 'my-plugin.toolbar',
      icon: 'zap',
      tooltip: 'My Plugin Action',
      onClick: this.toolbarAction.bind(this),
    });

    // Log activation
    context.logger.info('My Plugin activated successfully');
  }

  /**
   * Called when the plugin is unloaded
   * Clean up any resources, remove listeners, etc.
   */
  async deactivate(): Promise<void> {
    // Clean up resources
    this.context?.logger.info('My Plugin deactivated');
    this.context = null;
  }

  // ============================================================================
  // Hook Handlers
  // ============================================================================

  /**
   * Called when a new email is received
   */
  private async handleEmailReceived(email: Email): Promise<void> {
    if (!this.settings.enabled) return;

    this.context?.logger.debug(`Processing received email: ${email.subject}`);

    // Example: Check for specific patterns
    if (email.subject.includes('[IMPORTANT]')) {
      await this.context?.notifications.show({
        title: 'Important Email Received',
        body: `From: ${email.from}\n${email.subject}`,
        icon: 'alert-circle',
        actions: [
          { id: 'view', label: 'View Email' },
          { id: 'dismiss', label: 'Dismiss' },
        ],
      });
    }

    // Example: Add label based on content
    if (await this.shouldAutoLabel(email)) {
      await this.context?.emails.addLabel(email.id, 'processed-by-plugin');
    }
  }

  /**
   * Called when an email is sent
   */
  private async handleEmailSent(email: Email): Promise<void> {
    if (!this.settings.enabled) return;

    this.context?.logger.debug(`Email sent: ${email.subject}`);

    // Example: Log to external service
    if (this.settings.apiKey) {
      await this.logToExternalService(email);
    }
  }

  /**
   * Called when the compose window opens
   */
  private async handleComposeOpen(draft: Email): Promise<void> {
    if (!this.settings.enabled) return;

    // Example: Suggest recipients based on context
    if (draft.inReplyTo) {
      const suggestions = await this.getSuggestedRecipients(draft);
      if (suggestions.length > 0) {
        this.context?.ui.showSuggestions({
          type: 'recipients',
          items: suggestions,
        });
      }
    }
  }

  // ============================================================================
  // Command Handlers
  // ============================================================================

  /**
   * Main plugin command
   */
  private async runCommand(): Promise<void> {
    const selectedEmails = await this.context?.emails.getSelected();

    if (!selectedEmails || selectedEmails.length === 0) {
      this.context?.notifications.show({
        title: 'No emails selected',
        body: 'Please select one or more emails to process',
        type: 'warning',
      });
      return;
    }

    // Show progress
    const progress = this.context?.ui.showProgress({
      title: 'Processing emails...',
      total: selectedEmails.length,
    });

    try {
      for (let i = 0; i < selectedEmails.length; i++) {
        await this.processEmail(selectedEmails[i]);
        progress?.update(i + 1);
      }

      this.context?.notifications.show({
        title: 'Complete',
        body: `Processed ${selectedEmails.length} email(s)`,
        type: 'success',
      });
    } catch (error) {
      this.context?.logger.error('Failed to process emails', error);
      this.context?.notifications.show({
        title: 'Error',
        body: 'Failed to process some emails',
        type: 'error',
      });
    } finally {
      progress?.close();
    }
  }

  private async processSelectedEmail(): Promise<void> {
    const selected = await this.context?.emails.getSelected();
    if (selected && selected.length > 0) {
      await this.processEmail(selected[0]);
    }
  }

  private async toolbarAction(): Promise<void> {
    // Quick action from toolbar
    await this.runCommand();
  }

  // ============================================================================
  // Settings Panel
  // ============================================================================

  /**
   * Define the settings UI for this plugin
   */
  getSettingsPanel(): SettingsPanel {
    return {
      sections: [
        {
          id: 'general',
          title: 'General Settings',
          fields: [
            {
              key: 'enabled',
              type: 'toggle',
              label: 'Enable Plugin',
              description: 'Turn the plugin on or off',
            },
            {
              key: 'customOption',
              type: 'select',
              label: 'Processing Mode',
              description: 'Choose how emails should be processed',
              options: [
                { value: 'default', label: 'Default' },
                { value: 'aggressive', label: 'Aggressive' },
                { value: 'conservative', label: 'Conservative' },
              ],
            },
          ],
        },
        {
          id: 'integration',
          title: 'External Integration',
          fields: [
            {
              key: 'apiKey',
              type: 'password',
              label: 'API Key',
              description: 'Your API key for the external service',
              placeholder: 'Enter your API key',
            },
          ],
        },
      ],
      onSave: async (newSettings: Partial<MyPluginSettings>) => {
        this.settings = { ...this.settings, ...newSettings };
        await this.context?.settings.set(this.settings);
        this.context?.logger.info('Settings saved');
      },
    };
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  private async processEmail(email: Email): Promise<void> {
    // Implement your email processing logic here
    this.context?.logger.debug(`Processing email: ${email.id}`);

    // Example: Extract data
    const extractedData = this.extractData(email);

    // Example: Store in plugin storage
    await this.context?.storage.set(`processed:${email.id}`, {
      processedAt: new Date().toISOString(),
      data: extractedData,
    });
  }

  private extractData(email: Email): Record<string, unknown> {
    // Implement your data extraction logic
    return {
      subject: email.subject,
      from: email.from,
      wordCount: email.bodyText?.split(/\s+/).length || 0,
      hasAttachments: email.attachments && email.attachments.length > 0,
    };
  }

  private async shouldAutoLabel(email: Email): Promise<boolean> {
    // Implement your auto-labeling logic
    return email.subject.toLowerCase().includes('invoice');
  }

  private async logToExternalService(email: Email): Promise<void> {
    if (!this.settings.apiKey) return;

    try {
      await fetch('https://api.example.com/log', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.settings.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'email_sent',
          to: email.to,
          subject: email.subject,
          timestamp: new Date().toISOString(),
        }),
      });
    } catch (error) {
      this.context?.logger.error('Failed to log to external service', error);
    }
  }

  private async getSuggestedRecipients(draft: Email): Promise<Contact[]> {
    // Get contacts related to the thread
    const contacts = await this.context?.contacts.search({
      recentlyContacted: true,
      limit: 5,
    });

    return contacts || [];
  }
}

// ============================================================================
// Plugin API Types (for reference)
// ============================================================================

/*
 * These types are provided by @mycelix/plugin-api
 * They're included here for reference only

interface PluginContext {
  logger: Logger;
  settings: SettingsManager;
  storage: StorageManager;
  hooks: HookManager;
  commands: CommandManager;
  menus: MenuManager;
  ui: UIManager;
  emails: EmailManager;
  contacts: ContactManager;
  notifications: NotificationManager;
}

interface Email {
  id: string;
  messageId: string;
  from: string;
  to: string[];
  cc?: string[];
  bcc?: string[];
  subject: string;
  bodyText?: string;
  bodyHtml?: string;
  date: string;
  folder: string;
  labels: string[];
  isRead: boolean;
  isStarred: boolean;
  attachments?: Attachment[];
  inReplyTo?: string;
  threadId?: string;
  trustScore?: number;
}

interface Contact {
  id: string;
  email: string;
  name?: string;
  company?: string;
  trustScore?: number;
}
*/
