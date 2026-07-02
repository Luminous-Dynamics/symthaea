# 🚀 Mycelix Music: Complete Deployment Checklist

**Purpose**: Step-by-step checklist from local development to mainnet production
**Audience**: DevOps engineers, project leads, security auditors
**Time Estimate**: 2 weeks (local → testnet → mainnet)

---

## ✅ Phase 1: Local Development (Day 1)

### Prerequisites Installation
- [ ] Node.js 20+ installed (`node --version`)
- [ ] npm 10+ installed (`npm --version`)
- [ ] Foundry installed (`forge --version`)
- [ ] Docker & Docker Compose installed
- [ ] Git configured
- [ ] Code editor ready (VSCode recommended)

### Project Setup
- [ ] Clone repository
- [ ] Install root dependencies (`npm install`)
- [ ] Install workspace dependencies
  - [ ] `cd packages/sdk && npm install`
  - [ ] `cd apps/web && npm install`
  - [ ] `cd apps/api && npm install`
  - [ ] `cd contracts && npm install`
- [ ] Copy `.env.example` to `.env`
- [ ] Run validation script: `bash scripts/validate-setup.sh`

### Local Blockchain
- [ ] Start Anvil in Terminal 1: `anvil --block-time 1`
- [ ] Deploy contracts: `npm run contracts:deploy:local`
- [ ] Copy contract addresses to `.env`
- [ ] Verify deployment:
  ```bash
  cast call $ROUTER_ADDRESS "flowToken()" --rpc-url http://localhost:8545
  ```

### Backend Services
- [ ] Start Docker services: `npm run services:up`
- [ ] Verify PostgreSQL: `docker-compose ps | grep postgres`
- [ ] Verify Redis: `docker-compose ps | grep redis`
- [ ] Verify Ceramic: `curl http://localhost:7007/api/v0/node/healthcheck`
- [ ] Verify IPFS: `curl http://localhost:5001/api/v0/version`

### Seed Test Data
- [ ] Run seed script: `npm run seed:local`
- [ ] Verify songs in database:
  ```bash
  docker exec -it mycelix-music-postgres psql -U mycelix -d mycelix_music -c "SELECT COUNT(*) FROM songs;"
  ```
- [ ] Expected: 10 songs

### Frontend & Backend
- [ ] Start backend API in Terminal 2: `cd apps/api && npm run dev`
- [ ] Verify API: `curl http://localhost:3100/health`
- [ ] Start frontend in Terminal 3: `cd apps/web && npm run dev`
- [ ] Visit: http://localhost:3000
- [ ] Test homepage loads
- [ ] Test discover page shows songs
- [ ] Test upload wizard (mock wallet)

### Smart Contract Tests
- [ ] Run Foundry tests: `cd contracts && forge test -vvv`
- [ ] Expected: All 12 tests pass
- [ ] Check gas costs: `forge test --gas-report`
- [ ] Verify coverage: `forge coverage`

---

## 🧪 Phase 2: Testnet Deployment (Days 2-7)

### Service Accounts Setup
- [ ] Sign up for Privy (https://privy.io)
  - [ ] Create app
  - [ ] Get App ID and Secret
  - [ ] Add to `.env`: `PRIVY_APP_ID` and `PRIVY_APP_SECRET`
  - [ ] Configure authentication methods (wallet + email)
- [ ] Sign up for Web3.Storage (https://web3.storage)
  - [ ] Get API token
  - [ ] Add to `.env`: `WEB3_STORAGE_TOKEN`
- [ ] Set up Ceramic testnet
  - [ ] Use Clay testnet: `https://ceramic-clay.3boxlabs.com`
  - [ ] Generate admin seed
  - [ ] Add to `.env`: `CERAMIC_URL` and `CERAMIC_ADMIN_SEED`

### Testnet Preparation (Gnosis Chiado)
- [ ] Get Chiado RPC URL (https://rpc.chiadochain.net)
- [ ] Create deployment wallet
  - [ ] Generate new private key: `cast wallet new`
  - [ ] **SECURE IT**: Never commit to git!
  - [ ] Add to `.env`: `PRIVATE_KEY`
- [ ] Get testnet ETH
  - [ ] Visit: https://gnosisfaucet.com
  - [ ] Request Chiado testnet tokens
  - [ ] Verify balance:
    ```bash
    cast balance $DEPLOYER_ADDRESS --rpc-url https://rpc.chiadochain.net
    ```

### Replace Mocked Services
- [ ] **IPFS Upload** (apps/api/src/index.ts)
  - [ ] Replace mock with real Web3.Storage client
  - [ ] Test upload: Upload a small test file
  - [ ] Verify on IPFS gateway
- [ ] **DKG Claims** (apps/api/src/index.ts)
  - [ ] Replace mock with real Ceramic client
  - [ ] Create test claim
  - [ ] Verify stream ID returned
- [ ] **Wallet Authentication**
  - [ ] Configure Privy in `apps/web/pages/_app.tsx`
  - [ ] Test wallet connection
  - [ ] Test email authentication

### Deploy to Testnet
- [ ] Update contracts deployment script for testnet
- [ ] Deploy contracts to Chiado:
  ```bash
  cd contracts
  forge script script/DeployTestnet.s.sol \
    --rpc-url https://rpc.chiadochain.net \
    --private-key $PRIVATE_KEY \
    --broadcast \
    --verify
  ```
- [ ] Record deployed addresses:
  - [ ] FLOW Token: `_______________`
  - [ ] Router: `_______________`
  - [ ] PayPerStream: `_______________`
  - [ ] GiftEconomy: `_______________`
- [ ] Update frontend `.env` with testnet addresses
- [ ] Verify on Blockscout:
  - [ ] https://gnosis-chiado.blockscout.com

### Deploy Frontend (Vercel)
- [ ] Create Vercel account
- [ ] Connect GitHub repository
- [ ] Configure build settings:
  - [ ] Framework: Next.js
  - [ ] Root directory: `apps/web`
  - [ ] Build command: `npm run build`
  - [ ] Output directory: `.next`
- [ ] Add environment variables to Vercel:
  - [ ] `NEXT_PUBLIC_ROUTER_ADDRESS`
  - [ ] `NEXT_PUBLIC_FLOW_TOKEN_ADDRESS`
  - [ ] `NEXT_PUBLIC_CHAIN_ID=10200`
  - [ ] `NEXT_PUBLIC_RPC_URL=https://rpc.chiadochain.net`
  - [ ] `PRIVY_APP_ID`
  - [ ] `PRIVY_APP_SECRET`
- [ ] Deploy
- [ ] Visit deployed URL
- [ ] Test end-to-end flow

### Deploy Backend API (Fly.io or Railway)
- [ ] Choose hosting platform
- [ ] Create PostgreSQL database
- [ ] Create Redis instance
- [ ] Configure environment variables
- [ ] Deploy API
- [ ] Update frontend to use production API URL
- [ ] Test API endpoints

### Integration Testing
- [ ] Upload test song with real wallet
- [ ] Verify IPFS upload works
- [ ] Verify DKG claim creation works
- [ ] Stream song with testnet FLOW
- [ ] Tip artist
- [ ] Check artist receives payment
- [ ] Verify listener receives CGC (if gift economy)
- [ ] Test all pages load correctly
- [ ] Test mobile responsiveness

### Beta Tester Recruitment
- [ ] Create onboarding guide
- [ ] Invite 10 beta testers
- [ ] Set up feedback form
- [ ] Monitor usage for 1 week
- [ ] Collect bug reports
- [ ] Iterate on feedback

---

## 🔒 Phase 3: Security & Audit (Days 8-21)

### Code Review
- [ ] Internal code review by team
- [ ] Check for hardcoded secrets
- [ ] Review access controls
- [ ] Review error handling
- [ ] Check input validation
- [ ] Review gas optimizations

### Smart Contract Audit
- [ ] Choose auditor:
  - [ ] Option 1: OpenZeppelin ($15-20K)
  - [ ] Option 2: Trail of Bits ($20-25K)
  - [ ] Option 3: Consensys Diligence ($15-20K)
- [ ] Submit contracts for audit
- [ ] Review audit report
- [ ] Fix all critical issues
- [ ] Fix all high issues
- [ ] Fix medium issues (if feasible)
- [ ] Document accepted risks
- [ ] Get final approval from auditor

### Security Best Practices
- [ ] Enable rate limiting on API
- [ ] Add DDoS protection (Cloudflare)
- [ ] Enable CORS with whitelist
- [ ] Add request size limits
- [ ] Enable SQL injection protection
- [ ] Add XSS protection headers
- [ ] Enable HTTPS everywhere
- [ ] Set up security headers:
  ```
  Strict-Transport-Security: max-age=31536000
  Content-Security-Policy: default-src 'self'
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  ```

### Monitoring & Alerts
- [ ] Set up Sentry for error tracking
- [ ] Add Mixpanel/PostHog for analytics
- [ ] Configure uptime monitoring (UptimeRobot)
- [ ] Set up contract event monitoring
- [ ] Create alert rules:
  - [ ] Contract balance low
  - [ ] API error rate > 5%
  - [ ] Database connection failures
  - [ ] Unusual transaction patterns

### Backup & Recovery
- [ ] Set up database backups (daily)
- [ ] Test backup restoration
- [ ] Document recovery procedures
- [ ] Set up off-chain data backup
- [ ] Create disaster recovery plan

---

## 🌍 Phase 4: Mainnet Launch (Days 22-30)

### Final Pre-Launch Checks
- [ ] All audit issues resolved
- [ ] All tests passing
- [ ] Beta testing complete
- [ ] Documentation up to date
- [ ] Legal review complete
- [ ] Terms of Service finalized
- [ ] Privacy Policy finalized

### Mainnet Deployment
- [ ] Create mainnet deployment wallet
  - [ ] Generate with hardware wallet (recommended)
  - [ ] **CRITICAL**: Secure private keys in cold storage
- [ ] Get mainnet GNO tokens
- [ ] Deploy FLOW token (or connect to existing)
- [ ] Deploy strategies:
  ```bash
  cd contracts
  forge script script/DeployMainnet.s.sol \
    --rpc-url https://rpc.gnosischain.com \
    --private-key $MAINNET_PRIVATE_KEY \
    --broadcast \
    --verify
  ```
- [ ] Verify all contracts on Gnosisscan
- [ ] Transfer ownership to multisig
- [ ] Update frontend with mainnet addresses

### Deploy Production Infrastructure
- [ ] Deploy frontend to Vercel (production)
- [ ] Deploy API to production server
- [ ] Use production database (not testnet)
- [ ] Enable CDN for static assets
- [ ] Configure custom domain
- [ ] Enable SSL certificate
- [ ] Test all endpoints

### Launch Configuration
- [ ] Start with low protocol fee (0.5% for launch)
- [ ] Enable only 2 strategies initially (limit risk)
- [ ] Set up emergency pause mechanism
- [ ] Configure admin multisig (3-of-5 recommended)
- [ ] Document emergency procedures

### Soft Launch (First 48 Hours)
- [ ] Announce to beta testers only
- [ ] Limit to 100 users
- [ ] Monitor closely for issues
- [ ] Be ready for emergency pause
- [ ] Gather immediate feedback

### Public Launch
- [ ] Announce on social media
- [ ] Post on relevant forums (Reddit, Discord)
- [ ] Reach out to music press
- [ ] Create launch video
- [ ] Monitor metrics:
  - [ ] User signups
  - [ ] Songs uploaded
  - [ ] Transactions processed
  - [ ] Error rate
  - [ ] Gas costs

### Post-Launch Monitoring (Week 1)
- [ ] Daily check of all metrics
- [ ] Respond to user support requests
- [ ] Monitor contract events
- [ ] Track gas costs
- [ ] Monitor API performance
- [ ] Check database size growth
- [ ] Review error logs daily

---

## 📊 Success Metrics

### Technical Health
- [ ] API uptime > 99.9%
- [ ] Average response time < 500ms
- [ ] Error rate < 0.1%
- [ ] Gas costs < $0.10 per transaction
- [ ] Database queries < 100ms
- [ ] P2P success rate > 80%

### User Engagement
- [ ] 100+ artists in first month
- [ ] 1000+ listeners in first month
- [ ] 10,000+ streams in first month
- [ ] User retention > 40% (week 2)
- [ ] Average session > 10 minutes

### Economic Health
- [ ] Average artist earnings > $50/month
- [ ] Protocol revenue > costs by month 6
- [ ] CGC distribution working correctly
- [ ] Payment success rate > 99%
- [ ] Tip conversion rate > 5%

---

## 🚨 Emergency Procedures

### If Contract Vulnerability Found
1. [ ] Immediately pause contracts (if pause enabled)
2. [ ] Alert all users via website banner
3. [ ] Coordinate with auditor
4. [ ] Prepare fix and re-audit
5. [ ] Deploy fix to new contracts
6. [ ] Migrate user data if necessary

### If API Compromised
1. [ ] Shut down API immediately
2. [ ] Rotate all secrets and keys
3. [ ] Investigate breach
4. [ ] Patch vulnerability
5. [ ] Notify affected users
6. [ ] File security report

### If Database Lost
1. [ ] Restore from most recent backup
2. [ ] Verify data integrity
3. [ ] Check for missing transactions
4. [ ] Reconcile on-chain vs database
5. [ ] Communicate with users

---

## 📝 Legal & Compliance

### Before Mainnet Launch
- [ ] Consult with Web3 lawyer
- [ ] Review securities law implications
- [ ] Understand DMCA requirements
- [ ] Set up DMCA takedown process
- [ ] Create Terms of Service
- [ ] Create Privacy Policy
- [ ] Register business entity (LLC or DAO wrapper)
- [ ] Set up bank account
- [ ] Configure payment processing (if needed)

### Ongoing Compliance
- [ ] Monitor regulatory changes
- [ ] File required reports
- [ ] Respond to DMCA requests within 24h
- [ ] Maintain user data privacy (GDPR if EU users)
- [ ] Annual legal review

---

## ✅ Final Verification

Before marking mainnet launch complete:

- [ ] All contracts deployed and verified
- [ ] All services running smoothly
- [ ] Monitoring and alerts configured
- [ ] Backups tested and working
- [ ] Legal documentation complete
- [ ] Security audit complete
- [ ] Team trained on emergency procedures
- [ ] User support system ready
- [ ] Marketing materials ready
- [ ] Press kit available

---

## 🎉 Launch Complete!

**Date Launched**: _______________
**Initial Users**: _______________
**Initial Songs**: _______________
**Initial Transaction Volume**: _______________

**Post-Launch Tasks**:
1. Monitor daily for first 2 weeks
2. Gather user feedback
3. Plan feature iterations
4. Grow artist community
5. Add additional economic strategies based on demand

---

**Status**: Ready for deployment
**Estimated Timeline**: 30 days from start to mainnet
**Critical Path**: Audit (longest, 14 days)
**Total Cost Estimate**: $30-50K (audit + infrastructure + legal)

🎵 **Good luck with your launch!** 🎵
