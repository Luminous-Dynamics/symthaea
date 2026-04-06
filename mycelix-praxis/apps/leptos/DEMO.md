# Praxis Demo Walkthrough (5 minutes)

## Setup

- Terminal 1: `cd mycelix-praxis && nix develop` then `hc sandbox generate -a praxis happ-minimal/praxis-minimal.happ --run=8888`
- Terminal 2: `cd mycelix-praxis/apps/leptos && trunk serve --port 3001`
- Open browser: `http://localhost:3001`

The app runs in Mock mode by default. All pages are fully functional with simulated data.
If the conductor is not running, every zome call gracefully falls back to mock data.

---

## Act 1: Student Arrives (1 min)

1. **Home page** -- "Mycelix Praxis: Privacy-preserving decentralized education"
   - Four feature cards link to the core workflows:
     - Adaptive Learning (BKT, ZPD, VARK) -> /courses
     - Spaced Repetition (SM-2) -> /review
     - Verifiable Credentials (W3C standard) -> /credentials
     - Community Governance (quadratic voting) -> /governance
   - Connection badge in the top-right navbar shows "Mock" (orange) or "Connected" (green)

2. **Dashboard** -- Click "Dashboard" in the nav bar
   - **Consciousness card** -- live Phi, coherence, neuromodulators updating via coupled-oscillator simulation at 20 Hz
   - **XP & Level card** -- Level 7, 4,280 XP total, +120 today, +680 this week, progress bar to next level
   - **Streak card** -- 12 days current, 1.5x bonus multiplier, 1 freeze remaining, personal best 21 days
   - **Due Reviews card** -- 18 cards due (3 overdue, 5 new), "Start Review" button links to /review
   - **Top Skills card** -- five mastery bars: Rust Ownership 85%, Consensus Algorithms 72%, Soil Chemistry 63%, DHT Fundamentals 58%, Cooperative Governance 45%
   - **Recommendations** -- three cards suggesting next steps with reasoning ("Your Rust Ownership mastery is high -- ready for async")
   - **Recent Activity** -- timeline of last 5 events (course progress, review sessions, badge earned, level up, streak milestone)

---

## Act 2: Learning Session (1.5 min)

3. **Skill Map** -- Click "Skill Map" in the nav bar
   - Filter bar: Grade (K-5), Subject (Math/ELA/Science), View (Tree/List)
   - Default shows Grade 3 Mathematics with 5 tiers and 12 skill nodes:
     - Foundations: Place Value (95%, gold), Number Sense (88%, green)
     - Operations I: Addition (92%, gold), Subtraction (85%, green), Rounding (78%, green)
     - Operations II: Multiplication (65%, yellow), Division (48%, yellow), Patterns (56%, yellow)
     - Applications: Fractions (32%, yellow), Word Problems (41%, yellow)
     - Advanced: Measurement (18%, red), Geometry Basics (locked), Data & Graphs (locked)
   - Click "Multiplication" node to open the detail panel:
     - Mastery: 65% with progress bar
     - Bloom level: Apply
     - Standard: CCSS.MATH.3.OA.A.1
     - Prerequisites: Addition (done)
     - Three action buttons: Start Learning, Review, Take Assessment
   - Toggle to "List" view for a flat table with Bloom levels and standard codes

4. **Review** -- Click "Review" in the nav bar
   - "Loading due cards..." spinner, then first card appears
   - Progress bar: "Card 1 of 5"
   - Card front shows question with a topic tag (e.g., "Learning Science": "What is the SM-2 algorithm?")
   - Click "Show Answer" to flip the card
   - Back side shows question and answer separated by a divider
   - SM-2 quality rating grid: 0 (Blackout) through 5 (Perfect)
   - Rate each card -- progress bar advances
   - After card 5, session summary appears:
     - Cards Reviewed, Accuracy %, Avg Time per card, XP earned
     - Rating breakdown dots (color-coded per quality level)

---

## Act 3: Teacher View (1 min)

5. **Teacher Dashboard** -- Click "Teacher" in the nav bar
   - Class header: "Grade 3 Mathematics | 24 students | Fall 2025 | Mrs. Rodriguez"
   - **Mastery Heatmap** -- 8 students x 6 skills table, cells color-coded:
     - Gold (90%+), Green (70-89%), Yellow (30-69%), Red (<30%), Gray (not started)
     - Spot Bob's row: red cells in Subtraction (28%) and Multiplication (15%)
   - **At-Risk Students** panel -- two alerts:
     - Critical: "Bob: Struggling with Subtraction (28%) -- below grade-level expectations"
     - Warning: "Dan: Low engagement -- only 2 sessions this week, declining mastery trend"
   - **Class Stats** -- 68% avg mastery, 18/24 on track, 4/24 ahead, 2/24 behind
   - **Skill Averages** -- bar chart showing class-wide averages per skill
   - **Actions** -- Create Assessment, Assign Curriculum, Generate Report Cards, View Attendance

---

## Act 4: Governance (45 sec)

6. **Governance** -- Click "Governance" in the nav bar
   - "Community Governance: Participate in curriculum decisions through democratic voting."
   - "New Proposal" button opens a form (title, description, category, track, voting mode)
   - Three active proposals displayed as cards:
     - "Add Rust Programming Track" -- Curriculum, Normal track, QV badge (quadratic), 12 for / 3 against, 3 days remaining
     - "Extend Review Session Limit to 200 Cards" -- Policy, Fast track, 1p1v badge, 8 for / 5 against, 18 hours remaining
     - "Community Mentorship Program" -- Community, Slow track, 1p1v badge, 25 for / 2 against, 12 days remaining
   - Click a proposal to expand the detail view:
     - Full description, proposer DID, creation date, deadline
     - Vote breakdown bars (For / Against / Abstain with percentages)
     - Quadratic proposals show weighted tallies and a reputation allocation slider
     - Cast vote with For / Against / Abstain buttons
     - Confirmation message after voting

---

## Act 5: Credentials (45 sec)

7. **Credentials** -- Click "Credentials" in the nav bar
   - "W3C Verifiable Credentials earned through course completion. Portable, tamper-proof proof-of-learning."
   - Three earned credentials in a grid:
     - "Rust Fundamentals" -- Score: 92 (A), E3/N2/M2
     - "Distributed Systems" -- Score: 78 (B), E2/N1/M1
     - "Introduction to Holochain" -- Score: 95 (A+), E3/N2/M2
   - Epistemic badges on each card: E (Empirical), N (Normative), M (Materiality)
   - Click a credential to see full detail:
     - Score with band badge (A+, A, B, etc.)
     - Epistemic Classification with level bars (E0-E4, N0-N3, M0-M3)
     - W3C VC fields: issuer DID, course ID, issuance date, expiration, revocation status
     - Cryptographic Proof section: Ed25519Signature2020, verification method, purpose, signature
     - "Verify Credential" button -- click to see "[OK] Credential Verified: Signature valid. Not revoked. Epistemic classification confirmed."

---

## Closing

- Return to Dashboard
- Point out: consciousness card has been running continuously, showing how engagement fluctuates through the session
- Key message: "Every piece of learning data stays on the student's device. No cloud. No tracking. The Holochain DHT ensures peer-to-peer data integrity without any central server."
