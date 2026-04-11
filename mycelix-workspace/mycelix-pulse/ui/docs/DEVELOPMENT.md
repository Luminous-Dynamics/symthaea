# Development Guide

## Getting Started

### Prerequisites

- **Node.js** 18+ ([Download](https://nodejs.org/))
- **PostgreSQL** 14+ ([Download](https://www.postgresql.org/download/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Docker** (optional) ([Download](https://www.docker.com/))

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Luminous-Dynamics/Mycelix-Mail.git
   cd Mycelix-Mail
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure environment variables**
   - Edit `backend/.env` with your database credentials
   - Edit `frontend/.env` if needed

4. **Set up the database**
   ```bash
   cd backend
   npx prisma migrate dev
   ```

5. **Start development servers**
   ```bash
   npm run dev
   ```

Visit:
- Frontend: http://localhost:5173
- Backend API: http://localhost:3000

## Manual Setup

### Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Generate Prisma client
npx prisma generate

# Run migrations
npx prisma migrate dev

# Start development server
npm run dev
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

## Environment Variables

### Backend (.env)

```env
# Required
NODE_ENV=development
PORT=3000
DATABASE_URL=postgresql://user:password@localhost:5432/mycelix_mail
JWT_SECRET=your-secret-key-here
ENCRYPTION_KEY=your-32-character-key-here

# Optional
JWT_EXPIRES_IN=7d
CORS_ORIGIN=http://localhost:5173
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000
```

## Database Management

### Creating Migrations

```bash
cd backend
npx prisma migrate dev --name your_migration_name
```

### Resetting Database

```bash
./scripts/reset-db.sh
```

Or manually:
```bash
cd backend
npx prisma migrate reset
```

### Prisma Studio (Database GUI)

```bash
cd backend
npx prisma studio
```

Opens at: http://localhost:5555

## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Production hotfixes

### Making Changes

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests
   ```bash
   npm test
   ```

4. Run linter
   ```bash
   npm run lint
   ```

5. Commit with meaningful message
   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. Push and create PR
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Frontend Tests

```bash
cd frontend

# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

### Writing Tests

#### Backend Test Example
```typescript
// src/services/__tests__/example.test.ts
import { ExampleService } from '../example.service';

describe('ExampleService', () => {
  it('should do something', async () => {
    const service = new ExampleService();
    const result = await service.doSomething();
    expect(result).toBe(expected);
  });
});
```

#### Frontend Test Example
```typescript
// src/components/__tests__/Example.test.tsx
import { render, screen } from '@testing-library/react';
import Example from '../Example';

describe('Example', () => {
  it('should render correctly', () => {
    render(<Example />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

## Code Style

### Prettier

Format all files:
```bash
npx prettier --write .
```

### ESLint

Lint all files:
```bash
npm run lint
```

Auto-fix issues:
```bash
npm run lint:fix
```

### Naming Conventions

- **Files**: `kebab-case.ts`, `PascalCase.tsx`
- **Components**: `PascalCase`
- **Functions**: `camelCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Interfaces/Types**: `PascalCase`

## Docker Development

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild images
docker-compose build

# Start specific service
docker-compose up -d postgres
```

### Database Only

If you only want PostgreSQL in Docker:

```bash
docker-compose up -d postgres
```

Then run backend and frontend locally.

## Debugging

### Backend Debugging (VSCode)

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Debug Backend",
      "runtimeExecutable": "npm",
      "runtimeArgs": ["run", "dev"],
      "cwd": "${workspaceFolder}/backend",
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    }
  ]
}
```

### Frontend Debugging

Use React DevTools browser extension.

### Database Debugging

Use Prisma Studio:
```bash
cd backend
npx prisma studio
```

## Common Tasks

### Adding a New API Endpoint

1. Define route in `backend/src/routes/`
2. Create controller in `backend/src/controllers/`
3. Add service logic in `backend/src/services/`
4. Add validation schema
5. Write tests
6. Update Swagger docs

### Adding a New Component

1. Create component in `frontend/src/components/`
2. Add TypeScript types
3. Style with TailwindCSS
4. Write tests
5. Export from index if needed

### Adding a New Database Model

1. Update `backend/prisma/schema.prisma`
2. Run migration:
   ```bash
   npx prisma migrate dev --name add_model_name
   ```
3. Update types and services
4. Add tests

## Troubleshooting

### Database Connection Issues

1. Check PostgreSQL is running:
   ```bash
   # macOS
   brew services list

   # Linux
   systemctl status postgresql
   ```

2. Verify DATABASE_URL in `.env`
3. Check database exists:
   ```bash
   psql -U postgres -l
   ```

### Port Already in Use

```bash
# Find process using port 3000
lsof -i :3000

# Kill process
kill -9 <PID>
```

### Prisma Issues

```bash
# Regenerate client
npx prisma generate

# Reset database
npx prisma migrate reset

# Clear Prisma cache
rm -rf node_modules/.prisma
```

### Node Modules Issues

```bash
# Clear all node_modules
npm run clean

# Reinstall
npm run install:all
```

## Performance Profiling

### Backend

Use Node.js built-in profiler:
```bash
node --prof dist/server.js
```

### Frontend

Use React DevTools Profiler tab.

## Resources

- [Express Documentation](https://expressjs.com/)
- [React Documentation](https://react.dev/)
- [Prisma Documentation](https://www.prisma.io/docs)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Review the architecture documentation
- Read the API documentation

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
