# Getting Started with Mycelix Mail

Welcome to Mycelix Mail! This guide will help you set up your account and start using your new privacy-focused email client.

## First Login

1. Navigate to your Mycelix Mail instance
2. Click "Create Account" or "Sign In"
3. Enter your credentials
4. Complete the setup wizard

## Adding Email Accounts

### IMAP/SMTP Account

1. Go to **Settings** > **Accounts**
2. Click **Add Account**
3. Enter your email address
4. Mycelix will auto-detect settings for common providers
5. If needed, manually enter:
   - IMAP server and port
   - SMTP server and port
   - Username and password

### OAuth (Gmail, Outlook)

1. Go to **Settings** > **Accounts**
2. Click **Add Account**
3. Select your provider (Gmail, Outlook, etc.)
4. Click "Connect with [Provider]"
5. Authorize Mycelix in the popup

## Understanding the Interface

### Sidebar

- **Inbox** - Your primary inbox
- **Sent** - Emails you've sent
- **Drafts** - Unsent drafts
- **Archive** - Archived emails
- **Trash** - Deleted emails
- **Labels** - Custom labels/folders

### Email List

- **Unread indicator** - Blue dot for unread
- **Star** - Click to star important emails
- **Trust badge** - Shows sender trust level
- **Attachments** - Paper clip for attachments

### Email View

- **Reply/Forward** - Response actions
- **Archive/Delete** - Quick actions
- **Labels** - Assign labels
- **More** - Additional options

## Composing Emails

1. Click **Compose** or press `C`
2. Enter recipients in **To** field
3. Add **Subject**
4. Write your message
5. Click **Send** or press `Ctrl+Enter`

### Features

- **Attachments** - Drag & drop or click attach
- **Formatting** - Rich text editor
- **Templates** - Save common responses
- **Schedule** - Send later
- **Encryption** - Enable PGP/S-MIME

## Trust System

Mycelix Mail includes a unique trust-based approach to email.

### Trust Levels

- **High Trust** (Green) - Verified contacts you've attested
- **Medium Trust** (Yellow) - Some trust signals present
- **Low Trust** (Gray) - Unknown sender
- **Suspicious** (Red) - Potential phishing detected

### Creating Attestations

1. Open an email from someone you trust
2. Click the trust badge
3. Select trust level (1-5 stars)
4. Add context (colleague, friend, vendor)
5. Click "Create Attestation"

### Web of Trust

Your attestations contribute to a network of trust. When contacts you trust also trust someone, that person gains trust transitively.

## Search

### Basic Search

Type in the search bar to find emails by:
- Subject
- Sender
- Body content

### Advanced Search

Use search operators:
- `from:alice@example.com` - From specific sender
- `to:me` - Sent to you
- `subject:meeting` - Subject contains "meeting"
- `has:attachment` - Has attachments
- `is:unread` - Unread emails
- `after:2024-01-01` - After date
- `label:work` - Has label

### Natural Language

Type naturally:
- "Emails from John last week"
- "Attachments larger than 5MB"
- "Unread from important contacts"

## Labels & Folders

### Creating Labels

1. Go to **Settings** > **Labels**
2. Click **Create Label**
3. Enter name and choose color
4. Click **Save**

### Applying Labels

- **Single email** - Click label icon in email view
- **Multiple emails** - Select emails, click label in toolbar
- **Automatic** - Create filters to auto-label

## Filters & Rules

### Creating a Filter

1. Go to **Settings** > **Filters**
2. Click **Create Filter**
3. Set conditions:
   - From address
   - Subject contains
   - Has attachment
4. Set actions:
   - Apply label
   - Archive
   - Mark as read
   - Forward
5. Click **Save**

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Compose | `C` |
| Reply | `R` |
| Reply All | `A` |
| Forward | `F` |
| Archive | `E` |
| Delete | `#` |
| Search | `/` |
| Next email | `J` |
| Previous email | `K` |
| Open email | `Enter` |
| Back to list | `U` |
| Star | `S` |
| Mark unread | `Shift+U` |

Press `?` to see all shortcuts.

## AI Features

### Email Summarization

Long email threads? Click **Summarize** to get:
- Key points
- Action items
- Important dates
- Decision summary

### Smart Reply

1. Click **Smart Reply** when viewing an email
2. Choose a suggested response
3. Edit if needed
4. Send

### Meeting Detection

Mycelix automatically detects:
- Meeting requests
- Proposed times
- Attendees
- Locations

Click **Add to Calendar** to save.

## Privacy Features

### Tracking Protection

Mycelix blocks:
- Tracking pixels
- Link tracking
- Read receipts (unless you opt-in)

### Encryption

- **TLS** - All connections encrypted
- **PGP** - End-to-end encryption with keys
- **S/MIME** - Certificate-based encryption

### Quiet Hours

1. Go to **Settings** > **Quiet Hours**
2. Enable and set time range
3. Choose which days
4. Set exceptions for VIP contacts

## Mobile Access

### PWA (Progressive Web App)

1. Open Mycelix in mobile browser
2. Tap "Add to Home Screen"
3. Use like a native app

### Offline Mode

- Recent emails cached automatically
- Compose offline, sends when online
- Search cached emails

## Getting Help

- **In-app help**: Click `?` icon
- **Documentation**: [docs.mycelix.dev](https://docs.mycelix.dev)
- **Community**: [community.mycelix.dev](https://community.mycelix.dev)
- **Support**: support@mycelix.dev

## Tips & Tricks

1. **Inbox Zero** - Use archive liberally, search when needed
2. **Keyboard shortcuts** - Much faster than clicking
3. **Trust attestations** - Build your web of trust early
4. **Smart labels** - Let filters do the organizing
5. **Templates** - Save time on repetitive emails
6. **Quiet hours** - Protect your focus time
