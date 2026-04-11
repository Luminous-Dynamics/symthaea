# Mycelix Pulse Calendar — Plan

## The Paradigm Shift

Centralized calendars are **passive recorders of commitments you've already made**. They help you not double-book, but they don't protect your time.

Mycelix Pulse Calendar is an **active time sovereignty agent**. It protects your attention, respects your relationships, serves your community, and preserves your privacy — while syncing with Google/Outlook for everything else.

---

## What Already Exists (Infrastructure Audit)

| Component | Location | Status |
|-----------|----------|--------|
| Community calendar zome | mycelix-commons (10 extern functions, geohash, RSVP) | Production |
| Hearth rhythms zome | mycelix-hearth (6 extern functions, 5 rhythm types, mood, presence) | Production |
| Mail scheduler zome | mycelix-mail (9 functions, RFC 5545 recurrence, snooze) | Production |
| ICS parser | mycelix-mail/backend (VEVENT, attendees, PARTSTAT) | Production |
| Meeting scheduler | mycelix-mail/backend (availability, multi-participant slots, booking) | Production |
| Multi-provider sync | mycelix-mail/backend (Google, Outlook, Apple, CalDAV) | Production |
| React calendar UI | mycelix-mail/ui (day/week/month, time grid, event modals) | Production (React) |
| Event detection | mycelix-mail/backend (NLP regex for dates, times, locations) | Production |
| Leptos calendar page | mycelix-commons/apps/leptos (basic event list) | Minimal |
| Basic agenda view | mycelix-mail/apps/leptos (current, 4 mock events) | Built |

**Key gap**: No bridge between mail scheduler and commons calendar. No personal calendar zome. No day/week grid in Leptos.

---

## Three Calendar Layers

### Personal Calendar (The Self)
- **Storage**: Private source chain entries (never published to DHT)
- **Sync**: Bidirectional with Google, Outlook, Apple, CalDAV
- **Visibility**: Only free/busy windows optionally published
- **Features**: Events, reminders, focus time, working hours, travel time
- **Attention budget**: Local cognitive load tracking, auto-decline when depleted

### Community Calendar (The Commons)
- **Storage**: Public DHT entries, visible to community members
- **Governance**: Event creation gated by reputation/consciousness tier
- **Resources**: Linked to community resources (rooms, tools, vehicles)
- **Types**: Assemblies, work parties, celebrations, governance sessions
- **RSVP**: Public or anonymous, quorum tracking for governance
- **Cross-community**: Events publishable to multiple community calendars

### Hearth Calendar (The Home)
- **Storage**: Shared among hearth members, private from community
- **Rhythms**: Morning routine, dinner, bedtime — recurring family patterns
- **Presence**: "Who's home?" as calendar-integrated status
- **Care**: Childcare, eldercare rotation scheduling
- **Priority**: Hearth events always override community scheduling
- **Gentle time**: "After nap" as a valid event time (resolves to clock when needed)

### Cross-Layer Rules
- Hearth busy → Personal busy → Community unavailable
- Community accepted → Personal blocked
- Personal "focus time" → Chat delivery paused, mail batched
- Hearth "emergency" → Breaks through all layers

---

## Phase 1: Calendar Parity (Table Stakes)

Port existing React calendar to Leptos. Without these, nobody will try the product.

### P0 Features (14 items)

| # | Feature | Source | Complexity |
|---|---------|--------|-----------|
| 1 | Day view (time grid, hourly slots, event positioning) | React CalendarView | Medium |
| 2 | Week view (7-column grid, events spanning hours) | React CalendarView | Medium |
| 3 | Month view (date grid, 3 events per day, overflow) | React CalendarView | Medium |
| 4 | Event creation form (title, time, location, description, recurrence) | React QuickCreateModal | Low |
| 5 | Event detail view (attendees, RSVP status, location, edit/delete) | React EventDetailModal | Low |
| 6 | Multiple calendar sources with color coding + toggle | React sidebar | Low |
| 7 | RSVP (Going/Maybe/No) wired to commons calendar zome | Existing zome API | Low |
| 8 | Recurring events (Daily/Weekly/Monthly/Yearly) | Existing zome + scheduler | Low |
| 9 | Today/prev/next navigation | Standard pattern | Low |
| 10 | Time zone display (UTC storage, local rendering) | js_sys::Date | Low |
| 11 | ICS import (paste or file upload → parse → create events) | Existing ICS parser | Low |
| 12 | Working hours + out-of-office config | Settings page | Low |
| 13 | Hearth rhythms displayed as recurring entries | Existing hearth-rhythms zome | Low |
| 14 | Community + Personal + Hearth unified view with filters | Current agenda view extended | Low |

### Build Approach

The React `CalendarView.tsx` has day/week/month views with a time grid. Port the layout logic to Leptos components:

```
src/pages/calendar.rs          ← Extended with grid views
src/components/
  calendar_grid.rs             ← Day/week time grid (hourly slots)
  calendar_month.rs            ← Month date grid
  event_form.rs                ← Create/edit event modal
  event_detail.rs              ← Event detail popup
  mini_calendar.rs             ← Small month navigator in sidebar
```

---

## Phase 2: Integration (Wire the Bridges)

Connect calendar to mail, chat, and external providers.

| # | Feature | Complexity | What It Does |
|---|---------|-----------|-------------|
| 15 | ICS from email → auto-calendar event | Low | When email has .ics attachment, parse and offer "Add to calendar" |
| 16 | Chat event detection → calendar entry | Medium | "Let's meet Tuesday 2pm" → "Create event?" button in chat |
| 17 | Scheduled emails on calendar | Low | Scheduled sends appear as calendar entries |
| 18 | Inline RSVP in mail/chat | Low | Accept/Decline/Tentative buttons in mail reader |
| 19 | Google/Outlook sync setup UI | Medium | Settings page for OAuth connection, calendar selection |
| 20 | Mini calendar in sidebar | Low | Small month grid for quick navigation |
| 21 | Schedule-from-email button | Low | "Find a time" button pre-filled with email participants |
| 22 | Post-meeting summary form | Low | After event ends, prompt "What was decided?" structured form |

---

## Phase 3: Paradigm Features

Things Google Calendar structurally cannot do.

### Privacy-Preserving Availability (P3a)
- **v0 (build now)**: Encrypted availability windows published to DHT. Requester proposes times, agent auto-replies yes/no without exposing details.
- **v1 (later)**: Bloom filter of busy times. Plausible deniability built in.
- **Why Google can't**: They see all access patterns. DHT has no central observer.

### Attention Budget (P3b)
- Daily cognitive budget (configurable, default 100 points)
- Meeting types cost differently: 1:1 = 10, group = 20, presentation = 40
- Auto-decline when budget depleted ("attention budget exhausted")
- Budget regenerates after breaks (lunch = +20, walk = +10)
- Community-configurable maximum meeting load per member
- Budget remaining is local-only — never published to DHT

### Calendar-Aware Message Delivery (P3c)
- Focus time → messages held, delivered as batch after focus block
- DND → only urgent messages break through
- Priority routing: Hearth always, Community respects focus, Unknown queues
- Configurable per-calendar breakthrough rules

### Economic Scheduling (P3d)
- TEND token stake when requesting someone's time
- Higher stake = higher priority in queue
- No-show → stake forfeited to recipient
- Reputation-based discount: high-trust contacts need lower stake

### Governance-Integrated Scheduling (P3e)
- Auto-schedule when proposal reaches voting threshold
- Emergency sessions within 24h, overriding non-essential blocks
- Constitutional constraints: "Council meets monthly" auto-enforced
- Meeting outcomes linked to governance record

### Community Resource Booking (P3f)
- Rooms, vehicles, tools as DHT entries with calendar slots
- No double-booking (CAS on DHT)
- Governance-gated: some resources need approval or deposit
- Maintenance windows auto-block booking

---

## The "Gmail Moment" for Calendar

Gmail won with 1GB when everyone offered 4MB. 250x improvement on the dimension people cared about.

**Mycelix Pulse Calendar's "Gmail moment": Your calendar actually protects your time.**

No other calendar does this. Google Calendar records what you agreed to but does nothing to protect you. Mycelix Pulse:
- Says no for you when you're depleted
- Gates all communication during focus time (not just calendar invites)
- Shows meeting cost: "This meeting costs 3 people x 1 hour = 3 person-hours"
- Tracks meeting debt: "You owe Alice 2 meetings, Bob owes you 1"
- Suggests: "You've had 6 hours of meetings today. Decline the 4pm?"

---

## Migration Strategy

**Phase A: Read-only overlay** — Sync from Google/Outlook. Add community/hearth layers. Keep existing calendar.
**Phase B: Write-back** — Events in Pulse sync back to Google/Outlook. Pulse becomes primary interface.  
**Phase C: Sovereign** — Stop syncing. Calendar lives fully on Holochain.

Most users stay at Phase A-B. That's fine — value is in the community + family + attention layer.

---

## Build Order

1. **Calendar grid components** (day/week/month Leptos views)
2. **Event CRUD** (create/edit forms, detail view)
3. **Three-source unified view** (Personal + Community + Hearth with filters)
4. **Hearth rhythms integration** (display family routines as calendar entries)
5. **Mini calendar in sidebar** (quick date navigation)
6. **ICS from email integration** (auto-detect calendar invites)
7. **Chat event detection** ("let's meet Tuesday" → create event)
8. **External provider sync UI** (Google/Outlook OAuth in settings)
9. **Privacy-preserving availability** (encrypted free/busy windows)
10. **Attention budget** (cognitive load tracking + auto-decline)

---

## Naming

Personal Calendar → **My Time**
Community Calendar → **Commons Calendar**  
Hearth Calendar → **Hearth Rhythms**
Combined view → **Pulse Calendar**

---

*"A calendar should not just record your commitments. It should protect your capacity to make them."*
