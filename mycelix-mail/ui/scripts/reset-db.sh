#!/bin/bash

# Reset database and run migrations

set -e

echo "⚠️  This will reset your database and delete all data!"
read -p "Are you sure? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

cd backend

echo "🗑️  Resetting database..."
npx prisma migrate reset --force

echo "✅ Database reset complete!"
