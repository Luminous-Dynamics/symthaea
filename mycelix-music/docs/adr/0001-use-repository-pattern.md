# ADR 0001: Use Repository Pattern for Data Access

## Status
Accepted

## Context
The API needs a clean abstraction over database operations that provides:
- Type-safe queries
- Consistent error handling
- Transaction support
- Testability through dependency injection
- Protection against SQL injection

We considered several approaches:
1. **Direct pg queries** - Simple but leads to scattered SQL and poor testability
2. **Full ORM (TypeORM/Prisma)** - Heavy dependency, magic behavior, schema lock-in
3. **Query builder (Knex)** - Good middle ground but still requires careful abstraction
4. **Repository pattern with raw SQL** - Full control, explicit queries, easy to test

## Decision
We will use the **Repository Pattern** with parameterized raw SQL queries.

Each entity (Song, Play, etc.) gets a dedicated repository class that:
- Extends a `BaseRepository<T>` with common CRUD operations
- Contains domain-specific query methods
- Uses parameterized queries to prevent SQL injection
- Supports transaction passing via `PoolClient`

### Structure
```
repositories/
  base.repository.ts    # Generic CRUD operations
  song.repository.ts    # Song-specific queries
  play.repository.ts    # Play event queries
  index.ts              # Exports and factory
```

### Key Design Choices
- **Parameterized queries only** - No string interpolation of user input
- **Explicit SQL** - Queries are visible and optimizable
- **Transaction support** - Methods accept optional `PoolClient` for transactions
- **Pagination built-in** - Cursor and offset pagination in base class
- **Custom error class** - `RepositoryError` with context for debugging

## Consequences

### Positive
- Clear separation between business logic and data access
- Easy to unit test with mock repositories
- Full control over SQL queries for optimization
- No ORM migration/schema complexity
- Type-safe with TypeScript generics

### Negative
- More boilerplate than ORM-based solutions
- Manual query writing required
- Must maintain type definitions separately from schema
- N+1 queries possible if not careful (solved with specific methods)

### Mitigations
- Provide common query patterns in base class
- Use domain-specific methods for complex queries
- Add query logging for performance monitoring
