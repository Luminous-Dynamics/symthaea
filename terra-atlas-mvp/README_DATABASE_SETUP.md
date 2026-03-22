# 🗄️ Terra Atlas Database Setup Guide

## Quick Start

### 1. Install PostgreSQL with PostGIS

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib postgis

# macOS
brew install postgresql postgis

# NixOS
nix-env -iA nixpkgs.postgresql nixpkgs.postgis
```

### 2. Create Database and Enable Extensions

```bash
# Connect to PostgreSQL as superuser
sudo -u postgres psql

# Create database
CREATE DATABASE terra_atlas;

# Connect to the new database
\c terra_atlas;

# Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

# Create application user
CREATE USER terra_atlas_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE terra_atlas TO terra_atlas_user;

# Exit
\q
```

### 3. Run Database Schema

```bash
# Run the schema SQL
psql -U terra_atlas_user -d terra_atlas -f database/schema.sql
```

### 4. Install Node Dependencies

```bash
# Install Prisma and authentication packages
npm install @prisma/client prisma bcryptjs jsonwebtoken
npm install --save-dev @types/bcryptjs @types/jsonwebtoken

# Generate Prisma client
npx prisma generate

# Run Prisma migrations
npx prisma migrate dev --name init
```

### 5. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and update the DATABASE_URL
DATABASE_URL="postgresql://terra_atlas_user:your_secure_password@localhost:5432/terra_atlas?schema=public"

# Generate secure JWT secret
openssl rand -base64 32
# Copy the output and use as JWT_SECRET in .env
```

## 📊 Database Architecture

### Core Tables

#### Users
- Authentication and profile management
- Reputation scoring system
- Trust levels: novice → contributor → expert → guardian
- Validation accuracy tracking

#### DataPoints
- PostGIS geometry for geospatial data
- Trust scores calculated from validations
- Quality metrics from data sources
- Real-time denormalized counts for performance

#### Validations
- Three types: confirm, dispute, flag
- One vote per user per data point
- Evidence URLs and comments
- Anonymous tracking via fingerprinting

#### UserApiKeys
- Secure API key generation
- Scoped permissions (read, write, validate, admin)
- Rate limiting per key
- Automatic expiration

### Trust Score Algorithm

```sql
Trust Score = (Quality × 0.7) + (Community × 0.3) + Verification - Penalties

Where:
- Quality: Source data quality (0-100)
- Community: Confirm ratio from validations
- Verification: +5 to +20 bonus based on source type
- Penalties: -2 per flag after 5 flags
```

### Reputation System

```sql
Reputation = (Validations × 10) + (Confirms × 5) + (Flags × 15)

Trust Levels:
- Novice: 0-100 reputation
- Contributor: 101-500 reputation  
- Expert: 501-2000 reputation
- Guardian: 2000+ reputation
```

## 🔒 Security Features

### Password Security
- Bcrypt hashing with salt rounds
- Minimum 8 character passwords
- Password strength validation

### JWT Authentication
- 7-day access tokens
- 30-day refresh tokens
- Session tracking and management
- Automatic token rotation

### API Key Management
- SHA-256 hashed storage
- Prefix identification (ta_xxxxxxxx...)
- Scoped permissions
- Rate limiting per key
- Maximum 10 keys per user

### Activity Logging
- All user actions logged
- IP address tracking
- User agent recording
- Performance metrics
- Error logging

## 🚀 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh tokens
- `POST /api/auth/logout` - Logout

### Validations
- `GET /api/validations` - List validations
- `POST /api/validations` - Create/update validation
- `DELETE /api/validations` - Remove validation

### API Keys
- `GET /api/keys` - List user's API keys
- `POST /api/keys` - Create new API key
- `PUT /api/keys` - Update API key
- `DELETE /api/keys` - Revoke API key

### Data Management
- `GET /api/data/[layer]` - Get layer data
- `POST /api/data/import` - Import new data
- `GET /api/data/export` - Export data

## 📈 Performance Optimizations

### Indexes
- Spatial GIST index on geometry
- B-tree indexes on foreign keys
- Partial indexes for active records
- Trigram indexes for text search

### Denormalization
- Validation counts on data points
- Trust scores pre-calculated
- User stats cached

### Connection Pooling
```javascript
// Use Prisma connection pooling
const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL,
    },
  },
  log: ['query', 'info', 'warn', 'error'],
  connectionLimit: 10,
})
```

## 🧪 Testing

### Seed Data
```bash
# Run seed script
npx prisma db seed
```

### Test Authentication
```bash
# Register user
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"testuser","password":"SecurePass123"}'

# Login
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"emailOrUsername":"testuser","password":"SecurePass123"}'
```

### Test Validations
```bash
# Create validation (requires auth token)
curl -X POST http://localhost:3000/api/validations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"dataPointId":"uuid","validationType":"confirm"}'
```

## 🐳 Docker Setup (Optional)

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgis/postgis:14-3.2
    environment:
      POSTGRES_DB: terra_atlas
      POSTGRES_USER: terra_atlas_user
      POSTGRES_PASSWORD: your_secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql

volumes:
  postgres_data:
```

## 🔧 Maintenance

### Backup Database
```bash
pg_dump -U terra_atlas_user -h localhost terra_atlas > backup_$(date +%Y%m%d).sql
```

### Restore Database
```bash
psql -U terra_atlas_user -h localhost terra_atlas < backup_20250116.sql
```

### Update Trust Scores
```sql
-- Recalculate all trust scores
SELECT update_data_point_trust_score(id) FROM data_points;

-- Update all user reputations
UPDATE users SET reputation_score = calculate_user_reputation(id);
```

## 📊 Monitoring Queries

### Top Validators
```sql
SELECT * FROM top_validators LIMIT 10;
```

### Recent Trusted Data
```sql
SELECT * FROM recent_trusted_data LIMIT 100;
```

### System Health
```sql
-- Active users last 24h
SELECT COUNT(DISTINCT user_id) FROM activity_log 
WHERE created_at > NOW() - INTERVAL '24 hours';

-- Validation rate
SELECT COUNT(*) as validations_per_hour 
FROM validations 
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Data freshness
SELECT source_id, MAX(created_at) as last_update 
FROM data_points 
GROUP BY source_id;
```

## 🚨 Troubleshooting

### Connection Issues
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U terra_atlas_user -d terra_atlas -c "SELECT 1"
```

### Migration Issues
```bash
# Reset database (CAUTION: Deletes all data!)
npx prisma migrate reset

# Check migration status
npx prisma migrate status
```

### Performance Issues
```sql
-- Check slow queries
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Analyze tables
ANALYZE data_points;
ANALYZE validations;
```

---

**Next Steps:**
1. Set up PostgreSQL and run schema
2. Configure environment variables
3. Install npm packages
4. Run Prisma migrations
5. Test authentication endpoints
6. Begin collecting real data!

The database is now ready to persistently store all Terra Atlas data with full authentication, validation tracking, and API key management! 🌍🔒📊