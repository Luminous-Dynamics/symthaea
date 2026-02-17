import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcrypt';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seed...');

  // Create demo user
  const hashedPassword = await bcrypt.hash('Password123!', 12);

  const user = await prisma.user.upsert({
    where: { email: 'demo@mycelix.com' },
    update: {},
    create: {
      email: 'demo@mycelix.com',
      password: hashedPassword,
      firstName: 'Demo',
      lastName: 'User',
    },
  });

  console.log('✅ Created demo user:', user.email);

  // Note: Email accounts with real IMAP/SMTP credentials should be added manually
  // through the UI or with environment-specific seed data

  console.log('🎉 Seed completed successfully!');
  console.log('');
  console.log('Demo Credentials:');
  console.log('  Email: demo@mycelix.com');
  console.log('  Password: Password123!');
  console.log('');
}

main()
  .catch((e) => {
    console.error('❌ Seed failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
