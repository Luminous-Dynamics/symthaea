export interface EmailProvider {
  id: string;
  name: string;
  description: string;
  logo?: string;
  imapHost: string;
  imapPort: number;
  imapSecure: boolean;
  smtpHost: string;
  smtpPort: number;
  smtpSecure: boolean;
  setupInstructions?: string;
}

export const EMAIL_PROVIDERS: EmailProvider[] = [
  {
    id: 'gmail',
    name: 'Gmail',
    description: 'Google Mail',
    imapHost: 'imap.gmail.com',
    imapPort: 993,
    imapSecure: true,
    smtpHost: 'smtp.gmail.com',
    smtpPort: 587,
    smtpSecure: true,
    setupInstructions: `
      To use Gmail with Mycelix-Mail:
      1. Enable 2-factor authentication on your Google account
      2. Generate an App Password: https://myaccount.google.com/apppasswords
      3. Use the App Password instead of your regular password
    `,
  },
  {
    id: 'outlook',
    name: 'Outlook/Hotmail',
    description: 'Microsoft Email',
    imapHost: 'outlook.office365.com',
    imapPort: 993,
    imapSecure: true,
    smtpHost: 'smtp.office365.com',
    smtpPort: 587,
    smtpSecure: true,
    setupInstructions: `
      To use Outlook with Mycelix-Mail:
      1. Enable IMAP in your Outlook settings
      2. Use your full email address as username
      3. Use your regular password or app password
    `,
  },
  {
    id: 'yahoo',
    name: 'Yahoo Mail',
    description: 'Yahoo Email',
    imapHost: 'imap.mail.yahoo.com',
    imapPort: 993,
    imapSecure: true,
    smtpHost: 'smtp.mail.yahoo.com',
    smtpPort: 587,
    smtpSecure: true,
    setupInstructions: `
      To use Yahoo Mail with Mycelix-Mail:
      1. Enable IMAP in Yahoo Mail settings
      2. Generate an App Password in Account Security
      3. Use the App Password for authentication
    `,
  },
  {
    id: 'icloud',
    name: 'iCloud Mail',
    description: 'Apple iCloud',
    imapHost: 'imap.mail.me.com',
    imapPort: 993,
    imapSecure: true,
    smtpHost: 'smtp.mail.me.com',
    smtpPort: 587,
    smtpSecure: true,
    setupInstructions: `
      To use iCloud Mail with Mycelix-Mail:
      1. Enable 2-factor authentication on your Apple ID
      2. Generate an app-specific password
      3. Use your iCloud email and app-specific password
    `,
  },
  {
    id: 'custom',
    name: 'Custom Provider',
    description: 'Other email provider',
    imapHost: '',
    imapPort: 993,
    imapSecure: true,
    smtpHost: '',
    smtpPort: 587,
    smtpSecure: true,
    setupInstructions: `
      For custom email providers:
      1. Contact your email provider for IMAP/SMTP settings
      2. Enter the server details manually
      3. Common IMAP port: 993 (SSL) or 143 (non-SSL)
      4. Common SMTP port: 587 (STARTTLS) or 465 (SSL)
    `,
  },
];

export function getProviderById(id: string): EmailProvider | undefined {
  return EMAIL_PROVIDERS.find((p) => p.id === id);
}
