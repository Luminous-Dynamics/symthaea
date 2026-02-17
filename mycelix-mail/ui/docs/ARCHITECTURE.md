# Mycelix-Mail Architecture

## Overview

Mycelix-Mail is a full-stack email client built with modern web technologies, following industry best practices for security, scalability, and maintainability.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   React    │  │  Zustand   │  │   React    │            │
│  │ Components │  │   Store    │  │   Query    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                          │
                    HTTP/WebSocket
                          │
┌─────────────────────────────────────────────────────────────┐
│                        Backend API                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Express  │  │Middleware  │  │ WebSocket  │            │
│  │  Routers   │  │   Layer    │  │   Server   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │Controllers │  │  Services  │  │   Utils    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                          │
                     Prisma ORM
                          │
┌─────────────────────────────────────────────────────────────┐
│                      PostgreSQL                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Users    │  │  Accounts  │  │   Emails   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  ┌────────────┐  ┌────────────┐                             │
│  │ IMAP Server│  │ SMTP Server│                             │
│  │  (Receive) │  │   (Send)   │                             │
│  └────────────┘  └────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## Backend Architecture

### Layered Architecture

1. **Routes Layer** (`src/routes/`)
   - Defines API endpoints
   - Handles request routing
   - Applies middleware (auth, validation, rate limiting)

2. **Controller Layer** (`src/controllers/`)
   - Handles HTTP request/response
   - Input validation
   - Error handling
   - Response formatting

3. **Service Layer** (`src/services/`)
   - Business logic implementation
   - Database operations
   - Third-party integrations (IMAP/SMTP)
   - Data transformation

4. **Middleware Layer** (`src/middleware/`)
   - Authentication (JWT)
   - Request validation (Joi)
   - Error handling
   - Rate limiting
   - Security headers

5. **Utils Layer** (`src/utils/`)
   - Encryption utilities (AES-256-GCM)
   - JWT token management
   - Database client (Prisma)
   - Helper functions

### Key Components

#### Authentication Flow
```
User Login Request
    ↓
Login Route (/api/auth/login)
    ↓
Validation Middleware (Joi)
    ↓
Auth Controller
    ↓
Auth Service
    ├── Find user in database
    ├── Verify password (bcrypt)
    └── Generate JWT token
    ↓
Return user + token
```

#### Email Sync Flow
```
Sync Request
    ↓
Email Route (/api/emails/sync)
    ↓
Authentication Middleware
    ↓
Email Controller
    ↓
Email Service
    ├── Decrypt account credentials
    ├── Connect to IMAP server
    ├── Fetch emails
    ├── Parse email content
    ├── Store in database
    └── Update folder counts
    ↓
WebSocket notification
    ↓
Frontend receives update
```

## Frontend Architecture

### Component Hierarchy

```
App (ErrorBoundary)
├── BrowserRouter
    ├── LoginPage
    ├── RegisterPage
    ├── DashboardPage
    │   ├── Layout
    │   │   └── Header (navigation, logout)
    │   ├── FolderList (sidebar)
    │   ├── EmailList (middle pane)
    │   ├── EmailView (right pane)
    │   └── ComposeEmail (modal)
    └── SettingsPage
        └── Layout
            └── Account Management
```

### State Management

1. **Zustand Store** (`src/store/`)
   - Authentication state
   - User profile
   - Global UI state

2. **React Query** (TanStack Query)
   - Server state management
   - Caching
   - Background refetching
   - Optimistic updates

### Data Flow

```
User Action
    ↓
Component Event Handler
    ↓
API Service Call
    ↓
React Query Mutation/Query
    ↓
Backend API Request
    ↓
Update Cache
    ↓
Re-render Components
```

## Database Schema

### Users Table
- Stores user accounts
- Password hashed with bcrypt
- One-to-many with email accounts

### EmailAccounts Table
- Stores email account configurations
- IMAP/SMTP credentials (encrypted)
- One-to-many with folders and emails

### Folders Table
- Stores mailbox folders
- Types: INBOX, SENT, DRAFTS, TRASH, SPAM, CUSTOM
- Tracks unread and total counts

### Emails Table
- Stores email messages
- JSON fields for addresses (from, to, cc, bcc)
- Flags: isRead, isStarred, isDraft

### Attachments Table
- Stores email attachments
- Binary data for small files
- References for large files

## Security Architecture

### Authentication & Authorization
- JWT tokens for stateless auth
- HTTP-only cookies (optional)
- Token expiration (configurable)
- Password hashing with bcrypt (12 rounds)

### Data Encryption
- AES-256-GCM for credential encryption
- PBKDF2 for key derivation
- Random salt and IV for each encryption
- Auth tags for integrity verification

### API Security
- Rate limiting (configurable per endpoint)
- CORS protection
- Helmet.js for security headers
- Input validation with Joi
- SQL injection prevention (Prisma)
- XSS protection

### Network Security
- TLS/SSL for all connections
- HTTPS enforcement in production
- Secure WebSocket (WSS)
- Environment-based configuration

## Performance Optimizations

### Backend
- Database connection pooling
- Indexed database queries
- Efficient pagination
- Background email sync
- WebSocket for real-time updates

### Frontend
- Code splitting (React lazy loading)
- React Query caching
- Virtual scrolling for email lists
- Optimistic UI updates
- Image lazy loading

## Deployment Architecture

### Docker Containers
```
┌──────────────┐
│  PostgreSQL  │ (Port 5432)
└──────────────┘
       ↑
       │
┌──────────────┐
│   Backend    │ (Port 3000)
│  (Node.js)   │
└──────────────┘
       ↑
       │
┌──────────────┐
│   Frontend   │ (Port 80)
│    (Nginx)   │
└──────────────┘
```

### Production Considerations
- Load balancing for multiple backend instances
- Database replication and backups
- Redis for session storage
- CDN for static assets
- Logging and monitoring
- Auto-scaling based on load

## Development Workflow

1. **Local Development**
   ```bash
   npm run dev  # Start both frontend and backend
   ```

2. **Testing**
   ```bash
   npm test  # Run all tests
   ```

3. **Building**
   ```bash
   npm run build  # Build for production
   ```

4. **Deployment**
   ```bash
   docker-compose up -d  # Deploy with Docker
   ```

## API Design

### RESTful Principles
- Resource-based URLs
- HTTP methods (GET, POST, PUT, DELETE)
- Status codes (200, 201, 400, 401, 404, 500)
- JSON responses
- Consistent error format

### API Versioning
- Version in URL (`/api/v1/...`)
- Backward compatibility
- Deprecation notices

## WebSocket Events

### Client → Server
- `authenticate`: Send JWT token
- `ping`: Keep-alive

### Server → Client
- `connected`: Connection established
- `new_email`: New email received
- `email_read`: Email marked as read
- `error`: Error occurred

## Error Handling Strategy

### Backend
- Custom AppError class
- Async error wrapper
- Global error handler
- Environment-specific messages
- Error logging

### Frontend
- Error boundaries
- Try-catch blocks
- User-friendly error messages
- Retry mechanisms
- Fallback UI

## Future Enhancements

1. **Features**
   - Email threading
   - Labels and filters
   - Search improvements
   - Calendar integration
   - Contact management

2. **Performance**
   - Email caching
   - Offline support
   - Progressive Web App (PWA)
   - Service workers

3. **Security**
   - Two-factor authentication
   - OAuth integration
   - End-to-end encryption
   - Security audit logging

4. **DevOps**
   - Kubernetes deployment
   - CI/CD improvements
   - Automated testing
   - Performance monitoring
