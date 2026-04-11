# Mycelix-Mail

A modern, secure, and feature-rich email client built with React and Node.js.

## Features

### Core Features
- 📧 **Full IMAP/SMTP email support** - Connect any email provider
- 🔐 **Secure authentication** with JWT tokens
- 🎨 **Modern, responsive UI** with TailwindCSS
- 🌙 **Dark mode support** - System-aware theme switching
- 🔔 **Real-time notifications** via WebSocket
- 📎 **Attachment support** with size display
- 🔍 **Advanced search & filtering** - Search across all fields

### Power User Features
- ⌨️ **Comprehensive keyboard shortcuts** - Navigate without touching the mouse
- 🎯 **Bulk operations** - Select and act on multiple emails at once
- ⭐ **Quick actions bar** - Floating action bar with Reply, Star, Label, Archive, Delete
- 💬 **Conversation threading** - Group emails by subject with expandable timeline view
- 👤 **Contact avatars** - Gravatar integration with colorful initials fallback
- 📇 **Contact management** - Auto-tracking with VIP status and interaction history
- 🎯 **Importance detection** - Auto-prioritize emails with smart scoring
- 💾 **Draft autosave** - Never lose your work (auto-saves every 30s)
- 🖨️ **Print support** - Professional email printing
- 🎨 **Smart date formatting** - Contextual date display (Today, Yesterday, etc.)
- 📊 **Enhanced metadata** - Email size, attachment totals, recipient counts

### Productivity Features
- ✍️ **Email signatures** - Custom signatures per account with variable support
- 📝 **Quick reply templates** - 8+ pre-loaded templates, create unlimited custom ones
- ⏰ **Email snooze** - Defer emails with smart presets (Later Today, Tomorrow, Weekend, etc.)
- 🏷️ **Email labels** - Custom colored labels with 12-color palette and 20 emoji icons
- 📁 **Smart folders** - Virtual folders: All Mail, Starred, Important, Unread, Attachments
- 🎯 **Template categories** - Organized by greeting, follow-up, meeting, support, custom
- 📈 **Usage analytics** - Track most-used templates for optimization
- 🔄 **Variable substitution** - Dynamic content in signatures and templates ({{name}}, {{date}}, etc.)

### Accessibility
- ♿ **WCAG compliant** - Full ARIA label support
- ⌨️ **Keyboard navigation** - Complete keyboard control
- 🎯 **Screen reader friendly** - Semantic HTML and ARIA roles
- 🔊 **Visual feedback** - Clear status indicators and notifications

### Technical Features
- 📱 **Mobile-responsive design** - Works on all devices
- 🧪 **Comprehensive test coverage** - Unit and integration tests
- 🐳 **Docker support** - Easy deployment
- 📝 **Centralized error logging** - Debug and monitor issues
- ⚡ **Optimistic UI updates** - Instant feedback
- 🎯 **Smart caching** - React Query with intelligent stale-time strategy

## Tech Stack

### Frontend
- React 18 with TypeScript
- TailwindCSS for styling
- React Query for data fetching
- Zustand for state management
- React Router for navigation
- Vite for fast development

### Backend
- Node.js with Express
- TypeScript
- Prisma ORM with PostgreSQL
- JWT authentication
- nodemailer (SMTP) & imap (IMAP)
- WebSocket for real-time updates

### DevOps
- Docker & Docker Compose
- ESLint & Prettier
- Jest for testing
- GitHub Actions for CI/CD

## Project Structure

```
mycelix-mail/
├── backend/                 # Backend API server
│   ├── src/
│   │   ├── controllers/    # Route controllers
│   │   ├── services/       # Business logic
│   │   ├── middleware/     # Express middleware
│   │   ├── models/         # Database models
│   │   ├── routes/         # API routes
│   │   ├── utils/          # Utility functions
│   │   └── types/          # TypeScript types
│   ├── prisma/             # Database schema & migrations
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom hooks
│   │   ├── services/       # API services
│   │   ├── store/          # State management
│   │   ├── utils/          # Utility functions
│   │   └── types/          # TypeScript types
│   └── tests/              # Frontend tests
├── docker/                 # Docker configuration
└── docs/                   # Documentation

```

## Getting Started

### Prerequisites

- Node.js 18+
- PostgreSQL 14+
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Luminous-Dynamics/Mycelix-Mail.git
   cd Mycelix-Mail
   ```

2. **Install dependencies**
   ```bash
   # Install backend dependencies
   cd backend
   npm install

   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

3. **Set up environment variables**
   ```bash
   # Backend (.env)
   cp backend/.env.example backend/.env
   # Edit backend/.env with your configuration

   # Frontend (.env)
   cp frontend/.env.example frontend/.env
   # Edit frontend/.env with your configuration
   ```

4. **Set up the database**
   ```bash
   cd backend
   npx prisma migrate dev
   npx prisma generate
   ```

5. **Start the development servers**
   ```bash
   # Terminal 1 - Backend
   cd backend
   npm run dev

   # Terminal 2 - Frontend
   cd frontend
   npm run dev
   ```

6. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:3000

### Using Docker

```bash
# Build and start all services
docker-compose up -d

# Stop all services
docker-compose down
```

## Environment Variables

### Backend (.env)

```env
# Server
NODE_ENV=development
PORT=3000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mycelix_mail

# JWT
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRES_IN=7d

# CORS
CORS_ORIGIN=http://localhost:5173

# Email (for system emails)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
```

### Frontend (.env)

```env
# API Configuration
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000

# Application Configuration
VITE_APP_NAME=Mycelix Mail
VITE_APP_VERSION=1.0.0

# Feature Flags
VITE_ENABLE_DEBUG=false
VITE_ENABLE_DRAFT_AUTOSAVE=true

# Performance Settings
VITE_QUERY_STALE_TIME=30000
VITE_QUERY_CACHE_TIME=300000

# Trust (optional, aligns with backend)
# No frontend envs required for trust; backend handles TRUST_PROVIDER_URL and TTL.
# Frontend trust cache/TTL can be adjusted in Settings > Trust.
```

### Trust Provider (backend)

- Set `TRUST_PROVIDER_URL` (and optional `TRUST_PROVIDER_API_KEY`) to point the backend at your MATL/Holochain trust summary endpoint returning `summary` with score/tier/reasons/pathLength/decayAt/attestations/quarantined/fetchedAt.
- Adjust `TRUST_CACHE_TTL_MS` to match your desired refresh cadence; frontend TTL can be tuned in Settings > Trust.
- Health: `GET /api/trust/health` shows provider configured status, cache size, and TTL.
- Cache clear: `POST /api/trust/cache/clear` flushes backend cache; the Trust settings “Refresh summaries” button clears both frontend and backend caches.

## API Documentation

### Authentication

- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/me` - Get current user

### Email Accounts

- `GET /api/accounts` - Get all email accounts
- `POST /api/accounts` - Add email account
- `PUT /api/accounts/:id` - Update email account
- `DELETE /api/accounts/:id` - Delete email account

### Emails

- `GET /api/emails` - Get emails (with pagination & filters)
- `GET /api/emails/:id` - Get single email
- `POST /api/emails` - Send email
- `DELETE /api/emails/:id` - Delete email
- `PUT /api/emails/:id/read` - Mark as read/unread
- `PUT /api/emails/:id/star` - Star/unstar email

### Folders

- `GET /api/folders` - Get all folders
- `POST /api/folders` - Create folder
- `PUT /api/folders/:id` - Update folder
- `DELETE /api/folders/:id` - Delete folder

## Development

### Running Tests

```bash
# Backend tests
cd backend
npm test

# Frontend tests
cd frontend
npm test

# Test coverage
npm run test:coverage
```

### Linting

```bash
# Backend
cd backend
npm run lint
npm run lint:fix

# Frontend
cd frontend
npm run lint
npm run lint:fix
```

### Type Checking

```bash
# Backend
cd backend
npm run type-check

# Frontend
cd frontend
npm run type-check
```

## Security Features

- ✅ Password hashing with bcrypt
- ✅ JWT token authentication
- ✅ HTTP-only cookies
- ✅ CORS protection
- ✅ Rate limiting
- ✅ SQL injection prevention (Prisma)
- ✅ XSS protection
- ✅ CSRF protection
- ✅ Secure headers (Helmet.js)
- ✅ Input validation
- ✅ Email credentials encryption

## Keyboard Shortcuts

### Navigation
- `j` / `k` - Navigate emails up/down
- `↑` / `↓` - Navigate emails up/down
- `Enter` - Open selected email
- `Esc` - Close email or modal
- `/` - Focus search bar
- `?` - Show keyboard shortcuts help

### Email Actions
- `r` - Reply to email
- `a` - Reply all
- `f` - Forward email
- `e` / `u` - Mark as read/unread
- `s` - Star/unstar email
- `#` - Delete email
- `c` - Compose new email

### Bulk Operations
- `Ctrl/Cmd + A` - Select all emails
- `Shift + R` - Mark selected as read
- `Shift + U` - Mark selected as unread
- `Shift + S` - Star selected
- `Shift + D` - Deselect all
- `Shift + Delete` - Delete selected

### Compose Email
- `Ctrl/Cmd + Enter` - Send email
- `Esc` - Discard/cancel draft

## Performance Optimizations

- **Intelligent caching** - React Query with 30s stale time, 5min cache
- **Optimistic updates** - Instant UI feedback for star/read actions
- **Debounced search** - 300ms debounce to reduce API calls
- **Memoized callbacks** - Prevent unnecessary re-renders
- **Lazy loading** - Components loaded on demand
- **Code splitting** - Smaller initial bundle size
- **Virtual scrolling** - Handle large email lists efficiently
- **Image optimization** - Compressed and lazy-loaded images
- **Gzip compression** - Reduced payload sizes
- **Database indexing** - Fast queries
- **Connection pooling** - Efficient database connections
- **WebSocket** - Real-time updates without polling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Troubleshooting

### Common Issues

**Error: Cannot connect to email server**
- Verify your IMAP/SMTP credentials are correct
- Check that your email provider allows IMAP access
- Some providers (Gmail) require app-specific passwords
- Ensure firewall isn't blocking connections

**Email not syncing**
- Check WebSocket connection status (indicator in UI)
- Verify backend is running and accessible
- Check browser console for errors

**Dark mode not working**
- Clear browser cache and localStorage
- Check system theme preference is enabled
- Verify TailwindCSS dark mode classes are compiled

**Keyboard shortcuts not working**
- Ensure no input field is focused
- Check for browser extension conflicts
- Press `?` to see all available shortcuts

### Debug Mode

Enable debug mode to see detailed logging:

```env
# frontend/.env
VITE_ENABLE_DEBUG=true
```

This will show:
- API request/response details
- Error logging information
- WebSocket events
- State changes

## Support

For support, email support@mycelix.com or open an issue on GitHub.

## Acknowledgments

- Built with ❤️ by the Luminous Dynamics team
- Icons by [Lucide Icons](https://lucide.dev/)
- UI inspired by modern email clients

---

## Recent Updates

### Phase 14 - Smart Actions & Productivity Workflows (Latest) ⚡🚀

**Smart Email Actions:**
- ✅ **Delete and Next** - Delete email and automatically select next (#  key)
- ✅ **Next/Previous Navigation** - Navigate emails without returning to list (] and [ keys)
- ✅ **Smart Selection Logic** - Automatically selects next unread or wraps around
- ✅ **Fallback Behavior** - Closes email view when no next email available
- ✅ **useSmartActions Hook** - Reusable hook for "and next" patterns
- ✅ **Keyboard Shortcuts** - # (delete & next), ] (next), [ (previous)
- ✅ **Visual Navigation Buttons** - Previous/Next arrows in EmailView toolbar
- ✅ **Delete & Next Button** - Prominent button replaces simple delete
- ✅ **Tooltip Hints** - All buttons show keyboard shortcuts in tooltips
- ✅ **Integrated Undo** - Delete and next works with undo/redo system
- ✅ **State Management** - Proper cleanup and navigation state tracking

### Phase 13 - Rich Text Editor & Keyboard Navigation ⌨️✍️

**Rich Text Compose Editor:**
- ✅ **Full Formatting Toolbar** - Bold, Italic, Underline, Strikethrough text formatting
- ✅ **Headings Support** - H1, H2, and normal paragraph formatting
- ✅ **Lists** - Ordered (numbered) and unordered (bullet) lists
- ✅ **Text Alignment** - Left, center, and right alignment options
- ✅ **Insert Links** - Link dialog with URL validation (Ctrl+K shortcut)
- ✅ **Code Formatting** - Inline code blocks with monospace styling
- ✅ **Block Quotes** - Professional quote formatting for email replies
- ✅ **Clear Formatting** - Remove all formatting from selected text
- ✅ **Plain Text Toggle** - Switch between HTML rich text and plain text modes
- ✅ **Keyboard Shortcuts** - Ctrl+B (bold), Ctrl+I (italic), Ctrl+U (underline), Ctrl+K (link)
- ✅ **Auto-Save Drafts** - Automatic draft saving every 30 seconds
- ✅ **Draft Store** - Zustand-powered draft persistence with localStorage
- ✅ **Draft Management** - Auto-cleanup of old drafts (30+ days)
- ✅ **HTML Email Support** - Full HTML email composition with ContentEditable API
- ✅ **Dark Mode Editor** - Fully themed toolbar and editor for dark mode

**Gmail-Style Keyboard Navigation:**
- ✅ **j/k Navigation** - Navigate next/previous email with j/k keys (Gmail-style)
- ✅ **o/Enter to Open** - Open focused email with o or Enter key
- ✅ **x to Select** - Toggle checkbox selection for focused email with x key
- ✅ **u to Return** - Return to email list from email view with u key
- ✅ **Visual Focus Indicator** - Ring border highlights currently focused email
- ✅ **Smooth Scrolling** - Auto-scroll focused email into view
- ✅ **Focus Tracking** - Separate focus state from checkbox selection
- ✅ **Power User Efficiency** - Navigate entire inbox without touching mouse
- ✅ **Keyboard-First UX** - All email actions accessible via keyboard
- ✅ **No Conflicts** - Works alongside existing shortcuts (compose, search, etc.)
- ✅ **Dark Mode Support** - Focus indicator fully themed for dark mode

### Phase 12 - Advanced Search, Layout & Visual Enhancements 🔍

**Advanced Search with Operators:**
- ✅ **Gmail-Style Search Operators** - Powerful query syntax for precise email filtering
- ✅ **Operator Support** - from:, to:, subject:, has:, is:, before:, after:, label:
- ✅ **Negation Support** - Use `-` prefix to exclude (e.g., -from:spam@example.com)
- ✅ **Visual Query Builder** - AdvancedSearchPanel with dropdown selectors
- ✅ **Active Criteria Display** - See all active search filters with remove buttons
- ✅ **Query Preview** - Real-time preview of final search string
- ✅ **Combined Search** - Mix operators with free text for maximum flexibility
- ✅ **Search Examples** - has:attachment, is:unread, label:Important, before:2024-01-01
- ✅ **Client-Side Filtering** - Instant results without backend queries
- ✅ **Advanced Search Button** - Quick access from email list search bar
- ✅ **Search Hints** - Placeholder shows operator examples

**Reading Layout Modes:**
- ✅ **Vertical Split Layout** - Email list on left, preview pane on right (classic Gmail style)
- ✅ **Horizontal Split Layout** - Email list on top, preview pane on bottom (Outlook style)
- ✅ **No Preview Mode** - Full-screen email list, click to open full email view
- ✅ **Dynamic Layout Switching** - Instant layout changes without page reload
- ✅ **useLayout Hook** - Centralized layout state management with localStorage persistence
- ✅ **Settings UI** - Visual layout selector with icons and descriptions in General settings
- ✅ **Responsive Behavior** - Layouts adapt to different workflow preferences

**Email Density Options:**
- ✅ **Email Density Options** - Three customizable density levels
- ✅ **Comfortable Mode** - Spacious layout with maximum readability (~8 emails visible)
- ✅ **Cozy Mode** - Balanced spacing for good readability (~10 emails visible)
- ✅ **Compact Mode** - Dense layout to see more emails (~12 emails visible)
- ✅ **Settings Integration** - Visual density selector in General settings
- ✅ **Persistent Preference** - Saved to localStorage across sessions
- ✅ **Dynamic Application** - Applied to text size and padding in real-time

**Enhanced Visual Polish:**
- ✅ **Improved Empty States** - Animations, helpful tips, and better visual hierarchy
- ✅ **Tip Callouts** - Blue styled hints with icons for better guidance
- ✅ **Fade-in Animations** - Smooth entrance for empty state components
- ✅ **Enhanced Skeletons** - Realistic loading placeholders with shimmer effect
- ✅ **Avatar Placeholders** - Circular skeleton for avatars in list and view
- ✅ **Checkbox Placeholders** - Complete skeleton structure matching actual layout
- ✅ **8 Skeleton Items** - More realistic loading experience (was 5)
- ✅ **Dark Mode Support** - All skeletons and empty states fully themed

### Phase 11 - Undo/Redo & Animation System 🎨

**Undo/Redo Action History:**
- ✅ **Full Undo/Redo System** - Undo any destructive action with Ctrl/Cmd+Z
- ✅ **Toast Action Buttons** - Every action shows "UNDO" button in toast notification
- ✅ **10-Action History** - Buffer stores last 10 actions (configurable)
- ✅ **Redo Support** - Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y to redo
- ✅ **Smart Rollback** - Restores previous state accurately for star, read, delete operations
- ✅ **Action Descriptions** - Clear messages like "Email starred", "Marked as read"
- ✅ **Toast Integration** - 5-second dismiss with prominent undo button

**Animation Framework:**
- ✅ **60+ CSS Animations** - Fade, slide, scale, bounce, pulse, shimmer effects
- ✅ **Email List Animations** - Smooth enter/exit/hover transitions
- ✅ **Modal Animations** - Scale-in entrance, fade-out exit
- ✅ **Button Ripple Effects** - Material Design-style click feedback
- ✅ **Skeleton Loading** - Shimmer effect for all loading states
- ✅ **Micro-interactions** - Avatar pulse, label bounce, star rotate
- ✅ **GPU-Accelerated** - 60fps performance with hardware acceleration
- ✅ **Accessible** - Respects prefers-reduced-motion for users sensitive to motion
- ✅ **Page Transitions** - Smooth navigation between views
- ✅ **Card Lift Effects** - Subtle hover elevations
- ✅ **Icon Animations** - Rotation and scale effects

### Phase 10 - Professional Polish & Power Features 🚀

**Part 1 - Conversation Threading, Avatars & Contact Management:**
- ✅ **Conversation Threading** - Group emails into conversations by subject
- ✅ **Thread View** - Expandable/collapsible timeline view with all messages
- ✅ **Threading Toggle** - One-click switch between threaded and flat views
- ✅ **Thread Metadata** - Shows message count, unread count, participants, preview
- ✅ **Thread Importance** - Automatic scoring based on unread, stars, activity
- ✅ **Subject Normalization** - Removes Re:, Fwd:, etc. for intelligent grouping
- ✅ **Contact Management** - Auto-generates contacts from emails with tracking
- ✅ **Gravatar Integration** - Shows professional avatars with proper MD5 hashing
- ✅ **Avatar Fallbacks** - Colorful initials (Gmail-style) when Gravatar unavailable
- ✅ **Avatar Groups** - Multiple recipient avatars in compact display
- ✅ **Contact Store** - Tracks email count, last contact date, VIP status
- ✅ **VIP Contacts** - Mark important contacts for priority treatment
- ✅ **Contact Search** - Find contacts by name, email, or organization
- ✅ **Interaction Tracking** - Auto-records contact interactions from emails
- ✅ **Quick Actions Bar** - Material Design floating action bar at bottom of screen
- ✅ **Action Shortcuts** - Reply, Forward, Star, Label, Snooze, Archive, Delete
- ✅ **Keyboard Hints** - Tooltips show keyboard shortcuts for each action
- ✅ **Smooth Animations** - Slide-up entrance and fade-in effects
- ✅ **Email Importance Detection** - Advanced scoring algorithm for auto-priority
- ✅ **Importance Factors** - VIP sender, direct recipient, urgent keywords, work domain
- ✅ **Importance Levels** - Critical, High, Medium, Low with contextual badges
- ✅ **Auto-Labeling** - Automatically labels high-importance emails
- ✅ **Importance Sorting** - Sort emails by calculated importance score
- ✅ **Persistent Preferences** - Threading and contact data saved locally

### Phase 9 - Advanced Search & Organization System 🚀

**Part 1 - Labels & Tags System:**
- ✅ **Email Labels** - Organize emails with custom colored labels
- ✅ **6 Default Labels** - Work, Personal, Important, Follow Up, Later, Receipts
- ✅ **Label Manager** - Full CRUD for labels with color and icon customization
- ✅ **Label Picker** - Quick modal to add/remove labels from emails
- ✅ **Label Chips** - Visual colored badges on emails in list and view
- ✅ **Color Palette** - 12 predefined colors for visual distinction
- ✅ **Icon Library** - 20 emoji icons to customize labels
- ✅ **Bulk Labeling** - Apply labels to multiple emails at once
- ✅ **Label Display** - Shows up to 3 labels per email with "+N more" indicator
- ✅ **Persistent Storage** - All label data saved locally with Zustand
- ✅ **Data Export** - Labels included in data backup
- ✅ **Settings Integration** - Dedicated Labels tab in Settings page

**Part 2 - Smart Folders & Keyboard Shortcuts:**
- ✅ **Smart Folders** - 5 virtual folders: All Mail, Starred, Important, Unread, Attachments
- ✅ **Dynamic Filtering** - Smart folders filter emails in real-time
- ✅ **Smart Folder Icons** - Distinct icons for each smart folder (📧, ⭐, ❗, 🔵, 📎)
- ✅ **Label Shortcuts** - L key to label single email, Shift+L for bulk labeling
- ✅ **Keyboard Navigation** - Quick access to smart folders and labels
- ✅ **Empty States** - Beautiful empty states for each smart folder
- ✅ **Folder Count Display** - Shows email count for Important folder
- ✅ **Seamless Integration** - Works with existing folder structure
- ✅ **Help Documentation** - Updated keyboard shortcuts help

### Phase 8 - Advanced Productivity Suite 🚀

**Part 1 - Core Features:**
- ✅ **Email Signatures** - Per-account custom signatures with variables
- ✅ **Quick Reply Templates** - 8 pre-loaded templates + unlimited custom
- ✅ **Email Snooze** - Defer emails with smart presets (Later Today, Tomorrow, Weekend, Next Week)
- ✅ **Template Manager** - Full CRUD for templates with categories and analytics
- ✅ **Signature Manager** - Create, edit, delete signatures per account
- ✅ **Variable System** - Dynamic {{name}}, {{date}}, {{email}}, etc. in signatures/templates
- ✅ **Template Categories** - Organized: greeting, follow-up, meeting, support, custom
- ✅ **Usage Analytics** - Track most-used templates
- ✅ **Smart Presets** - Quick snooze options with auto-calculated dates

**Part 2 - Integration & Polish:**
- ✅ **Comprehensive Settings Page** - Centralized hub for all configurations with tabs
- ✅ **Snoozed Folder** - Virtual folder showing all snoozed emails with time-until-due indicators
- ✅ **Enhanced Keyboard Shortcuts** - Ctrl/Cmd+K for templates, Z for snooze, Ctrl/Cmd+, for settings
- ✅ **Settings Navigation** - Easy access from header and keyboard shortcut
- ✅ **Data Export** - Backup all settings, signatures, templates, and preferences
- ✅ **Notification Management** - Desktop notification settings for snooze reminders
- ✅ **Theme Integration** - Consistent dark mode across all new features
- ✅ **Unsnooze Actions** - Quick unsnooze from snoozed folder view
- ✅ **Time Display** - Smart time-until-due formatting (minutes, hours, days)

### Phase 7 - Production Readiness
- ✅ Environment variables configuration
- ✅ Enhanced date formatting utilities
- ✅ Improved email metadata display (size, folder, recipient counts)
- ✅ Centralized error logging service
- ✅ Updated comprehensive documentation

### Phase 6 - Power User Features & UX
- ✅ Comprehensive keyboard shortcuts
- ✅ Bulk operations (select, mark read, star, delete)
- ✅ Query optimization (30s stale time, smart caching)
- ✅ Draft autosave (every 30 seconds)
- ✅ Enhanced compose UX (Ctrl/Cmd+Enter to send, character counter)
- ✅ ARIA labels and accessibility improvements
- ✅ Dark mode for all components

### Phase 5 - Polish & Performance
- ✅ Optimistic UI updates
- ✅ Enhanced error boundaries
- ✅ Empty state components
- ✅ Loading skeletons
- ✅ Toast notification limits
- ✅ Selection state management

**Note**: This is a demonstration email client. For production use, ensure you:
- Use strong JWT secrets (min 32 characters)
- Enable SSL/TLS for all connections
- Implement rate limiting on API endpoints
- Set up proper logging and monitoring (use errorLogger service)
- Perform regular security audits
- Configure CSP headers
- Enable HTTPS in production
- Use environment-specific configurations
- Set up backup and disaster recovery
- Comply with email provider terms of service
- Follow GDPR/privacy regulations for user data
