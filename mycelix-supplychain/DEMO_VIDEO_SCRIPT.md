# 🎬 Mycelix ERP Demo Video Script

**Target Length**: 3 minutes
**Format**: Screen recording + voiceover
**Audience**: Business owners, CTOs, procurement managers

---

## 📋 Pre-Production Checklist

- [ ] Service running on localhost:8000
- [ ] Database initialized and clean
- [ ] Coffee demo script ready
- [ ] Screen recorder tested (OBS/Loom)
- [ ] Microphone checked
- [ ] Browser tabs closed (clean desktop)
- [ ] Practice run completed

---

## 🎯 Video Structure

### Opening (0:00-0:20)
**Visual**: Mycelix logo → Problem statement text overlay
**Voiceover**:

> "Traditional ERPs cost $100,000 to set up, take 6 months to deploy, and lack transparency. What if there was a better way?"

**On-screen text**:
- ❌ $100K+ setup cost
- ❌ 6-12 month deployment
- ❌ No blockchain verification

---

### Introduction (0:20-0:40)
**Visual**: Mycelix dashboard (clean UI)
**Voiceover**:

> "Meet Mycelix ERP - the world's first blockchain-auditable ERP system. Setup in one week for just $5,000. Let me show you how it works with a real example."

**On-screen text**:
- ✅ $5K setup
- ✅ 1 week deployment
- ✅ Blockchain-verified supply chain

---

### Demo Part 1: Supply Chain Tracking (0:40-1:20)
**Visual**: Terminal running coffee demo (first 3 events)
**Voiceover**:

> "Let's follow a batch of Ethiopian coffee from farm to customer.
>
> First, we record the harvest at the Koke Washing Station in Ethiopia. The system generates a cryptographic signature - proof this event actually happened.
>
> Next, the coffee ships from the Port of Djibouti. Each step is recorded on the blockchain, creating an immutable audit trail.
>
> Finally, it arrives at our roastery in San Francisco where it's processed. Every event is tamper-proof and verifiable."

**Key moments to highlight**:
- JSON request going in
- Cryptographic signature returned
- Event IDs being generated

**On-screen callouts**:
- "Cryptographic SHA-256 signature"
- "Blockchain hash: uhCkk..."
- "Tamper-proof audit trail"

---

### Demo Part 2: Complete Provenance (1:20-1:50)
**Visual**: Provenance query showing full journey
**Voiceover**:

> "With one API call, we can retrieve the complete journey of this coffee - from the exact farm in Ethiopia, through international shipping, to our roastery.
>
> This isn't just tracking - it's cryptographic proof. Perfect for organic certification, fair trade verification, or FDA compliance."

**Visual**: Show provenance JSON with events array
**On-screen callouts**:
- "Farm → Port → Roastery"
- "Verified: ✓ TRUE"
- "Perfect for regulatory compliance"

---

### Demo Part 3: Financial Operations (1:50-2:20)
**Visual**: Creating invoice and recording payment
**Voiceover**:

> "Now let's handle the financial side. We create an invoice for our wholesale customer - Blue Bottle Coffee.
>
> Mycelix automatically handles double-entry bookkeeping. When we create the invoice, it debits accounts receivable and credits revenue.
>
> When the payment comes in, it's recorded instantly. The system updates the general ledger, marks the invoice as paid, and keeps perfect books."

**Visual**: Show invoice creation → payment recording → reports
**On-screen callouts**:
- "Automatic double-entry"
- "Every transaction hashed"
- "Trial balance always correct"

---

### Demo Part 4: Financial Reports (2:20-2:40)
**Visual**: Trial balance, income statement, AR aging
**Voiceover**:

> "Financial reports are generated in real-time. Trial balance shows perfect accounting - debits equal credits. Income statement tracks revenue and profitability. And AR aging shows exactly who owes what."

**Visual**: Quickly flash through three reports
**On-screen text**:
- "Real-time reports"
- "Always audit-ready"
- "100% accurate"

---

### Closing: Value Proposition (2:40-3:00)
**Visual**: Split screen comparison table + call to action
**Voiceover**:

> "Mycelix ERP combines the best of both worlds - enterprise-grade financial management with blockchain-powered supply chain transparency. Setup in one week, not six months. $5,000, not $100,000. And every transaction is cryptographically verified.
>
> Ready to revolutionize your ERP? Visit mycelix.net to start your free pilot."

**On-screen final frame**:
```
Traditional ERP          Mycelix ERP
$100K+ setup         →   $5K setup
6-12 months          →   1 week
Manual audits        →   Blockchain-verified
$5K-$20K/month       →   $500/month

🚀 Start your pilot: mycelix.net
📧 sales@mycelix.net
```

---

## 🎥 Production Notes

### Screen Recording Setup

**Resolution**: 1920x1080 (1080p)
**Frame rate**: 30 fps
**Audio**: 48kHz, mono or stereo

**Terminal settings**:
```bash
# Use larger font for readability
Terminal font size: 16pt
Color scheme: Dark with good contrast
Window size: Full screen (hide menu bar)
```

**Preparation**:
```bash
# Before recording, run these to prepare
cd /srv/luminous-dynamics/mycelix-supplychain
./init-database.sh  # Fresh database
cd rust && cargo run --release &  # Start service

# Wait 10 seconds for service to start
sleep 10

# Test health check
curl http://localhost:8000/v1/health
# Should return {"status":"ok"}
```

### Voiceover Tips

- **Pace**: Slightly slower than normal conversation (120-140 words/minute)
- **Tone**: Professional but approachable, enthusiastic but not salesy
- **Emphasis**: Highlight key numbers ($5K, 1 week, blockchain-verified)
- **Pauses**: Brief pause after each major section
- **Energy**: Start strong, maintain energy, finish with clear CTA

### Visual Effects

**Transitions**:
- Fade between major sections (0.5s)
- No flashy effects - keep it professional

**On-screen callouts**:
- Use clean, modern font (Inter, Helvetica)
- Yellow/gold color for emphasis (#FFD700)
- Fade in/out (not pop)
- Position: Lower third or beside relevant element

**Cursor**:
- Use cursor highlighting (circle around cursor)
- Slow down cursor movement during key moments
- Pause cursor on important text for 1-2 seconds

---

## 📝 Script Variations

### 30-Second Version (Social Media)
```
Traditional ERPs cost $100K and take 6 months to set up.

Mycelix ERP? $5K and one week.

Plus, every transaction is blockchain-verified for complete transparency.

Setup in one week. $500/month. Blockchain-powered.

Start your pilot at mycelix.net
```

### 60-Second Version (LinkedIn/Twitter)
```
[0:00-0:10] Problem: Traditional ERPs are expensive and slow
[0:10-0:20] Solution: Mycelix ERP - $5K, 1 week, blockchain-verified
[0:20-0:40] Demo: Show quick coffee supply chain tracking
[0:40-0:55] Benefits: Real-time reports, automatic bookkeeping, audit-ready
[0:55-1:00] CTA: Visit mycelix.net
```

### 5-Minute Version (Sales Calls)
Include all sections above, plus:
- Competitive comparison (vs QuickBooks, SAP, NetSuite)
- Customer testimonials (when available)
- Technical architecture overview
- Security & compliance (SOC 2, GDPR)
- Pricing tiers breakdown
- Q&A teaser ("Common questions...")

---

## 🎨 Thumbnail Design

**Image**: Split screen
- Left: Frustrated person at desk with papers (traditional ERP chaos)
- Right: Happy person with clean dashboard (Mycelix simplicity)

**Text Overlay**:
```
OLD WAY:              NEW WAY:
$100K setup       →   $5K setup
6 months          →   1 week
Manual            →   BLOCKCHAIN
```

**Colors**: Blue (trust) + Gold (premium)
**Font**: Bold, modern, readable at small sizes

---

## 📊 Success Metrics

**Engagement**:
- Target: 80%+ retention through 3 minutes
- Click-through rate: 5%+ to mycelix.net
- Social shares: 50+ in first week

**Conversions**:
- Pilot signups: 10+ from video
- Demo requests: 20+ from video
- Email list growth: 100+ subscribers

---

## 🚀 Distribution Plan

### Week 1: Owned Channels
- Website: mycelix.net homepage hero
- GitHub: README.md embedded video
- LinkedIn: Company page + founder posts
- Twitter/X: Pinned tweet
- YouTube: Mycelix channel

### Week 2: Partner Channels
- Submit to Product Hunt
- Post in r/entrepreneur, r/smallbusiness
- Hacker News Show HN
- IndieHackers community

### Week 3: Paid Promotion
- LinkedIn ads (B2B audience)
- YouTube pre-roll (business content)
- Reddit sponsored post
- Twitter promoted tweet

---

## 🎬 Post-Production Checklist

- [ ] Color correction (consistent brightness/contrast)
- [ ] Audio leveling (consistent volume)
- [ ] Noise reduction on voiceover
- [ ] Add background music (subtle, professional)
- [ ] On-screen callouts timed correctly
- [ ] Transitions smooth
- [ ] Captions/subtitles added (accessibility + silent viewing)
- [ ] End screen with clear CTA
- [ ] Export in multiple formats (1080p, 720p, 480p)
- [ ] Upload to YouTube, Vimeo, Wistia
- [ ] Create GIF snippets for social media

---

## 💡 Additional Video Ideas

### Follow-up Videos (Series):
1. **Deep Dive: Supply Chain Module** (5 min)
   - Detailed walkthrough of all event types
   - Blockchain verification explained
   - Integration with existing systems

2. **Deep Dive: Finance Module** (5 min)
   - Double-entry bookkeeping basics
   - Chart of accounts customization
   - Report generation and analysis

3. **Industry Spotlights** (3 min each):
   - E-commerce with Mycelix
   - Manufacturing with Mycelix
   - Pharmaceuticals with Mycelix

4. **Customer Success Stories** (2 min each):
   - "How Acme Corp saved $80K/year"
   - "From setup to production in 5 days"

5. **Technical Deep Dive** (10 min):
   - API walkthrough for developers
   - Holochain integration explained
   - Security & compliance overview

---

## 📞 Call to Action Variations

**Primary CTA** (most effective):
> "Start your 3-month pilot for 50% off: mycelix.net/pilot"

**Secondary CTA** (lower commitment):
> "Schedule a 15-minute demo: mycelix.net/demo"

**Tertiary CTA** (lead generation):
> "Download the free ERP comparison guide: mycelix.net/guide"

---

**Version**: 1.0
**Last Updated**: December 30, 2025
**Status**: Ready for production

🎬 **Lights, camera, blockchain action!**
