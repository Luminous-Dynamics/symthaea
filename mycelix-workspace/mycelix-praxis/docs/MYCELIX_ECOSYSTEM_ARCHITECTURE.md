# Mycelix Ecosystem Architecture

**Last Updated**: 2025-12-11
**Status**: Design Document

---

## 🌐 Mycelix Ecosystem Overview

The Mycelix ecosystem consists of multiple interconnected Holochain applications:

1. **Mycelix Praxis** - Privacy-preserving education platform
2. **Mycelix FL** - Federated learning infrastructure
3. **Mycelix Core** - Identity and core services
4. **Mycelix DAO** - Governance and decision-making

---

## 🏗️ Conductor Architecture Strategies

### Strategy 1: Shared Conductor (Production - Recommended)

**Use Case**: Deployed applications with cross-app integration

```
┌─────────────────────────────────────────────────┐
│         Mycelix Shared Conductor                │
│                                                  │
│  Admin Interface: localhost:4444                │
│  App Interface:   localhost:8888                │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Praxis   │ │ FL Core  │ │DAO Core  │       │
│  │   DNA    │ │   DNA    │ │   DNA    │       │
│  └──────────┘ └──────────┘ └──────────┘       │
│                                                  │
│  Shared DHT Network                             │
│  Shared Agent Identity                          │
│  Single Lair Keystore                           │
└─────────────────────────────────────────────────┘
```

**Advantages**:
- ✅ Apps can call each other's zomes directly
- ✅ Shared agent identity (same keypair across apps)
- ✅ Single conductor process (lower resource usage)
- ✅ Simplified deployment
- ✅ Cross-app data sharing via DHT

**Configuration**: `conductor/mycelix-ecosystem-conductor.yaml`

### Strategy 2: Separate Conductors (Development - Recommended)

**Use Case**: Independent development of different apps

```
┌──────────────────────┐  ┌──────────────────────┐
│  Praxis Conductor    │  │  FL Conductor        │
│  Admin: 4444         │  │  Admin: 4445         │
│  App:   8888         │  │  App:   8889         │
│  ┌──────────┐        │  │  ┌──────────┐        │
│  │ Praxis   │        │  │  │ FL Core  │        │
│  │   DNA    │        │  │  │   DNA    │        │
│  └──────────┘        │  │  └──────────┘        │
└──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐
│  Core Conductor      │  │  DAO Conductor       │
│  Admin: 4446         │  │  Admin: 4447         │
│  App:   8890         │  │  App:   8891         │
│  ┌──────────┐        │  │  ┌──────────┐        │
│  │  Core    │        │  │  │   DAO    │        │
│  │   DNA    │        │  │  │   DNA    │        │
│  └──────────┘        │  │  └──────────┘        │
└──────────────────────┘  └──────────────────────┘
```

**Advantages**:
- ✅ Independent development and testing
- ✅ Easier debugging (isolated logs)
- ✅ No cross-app interference
- ✅ Can use different Holochain versions

**Disadvantages**:
- ❌ More resource intensive
- ❌ Complex cross-app communication
- ❌ Multiple keystores to manage

---

## 🔢 Port Allocation Strategy

### Reserved Port Ranges

```bash
# Mycelix Admin Interfaces (4444-4499)
4444  # Shared conductor OR Praxis dev conductor
4445  # FL dev conductor
4446  # Core dev conductor
4447  # DAO dev conductor
4448-4499  # Reserved for future Mycelix apps

# Mycelix App Interfaces (8888-8949)
8888  # Shared conductor OR Praxis dev conductor
8889  # FL dev conductor
8890  # Core dev conductor
8891  # DAO dev conductor
8892-8949  # Reserved for future Mycelix apps

# Avoid These Ports (Already in use)
3001  # The Weave
3333  # Sacred Core
3338  # Field Visualizer
7777  # Sacred Bridge
8000-8099  # General dev servers
8200  # OAuth callbacks
```

---

## 🔄 Cross-App Integration Patterns

### Pattern 1: Bridge Calls (Shared Conductor)

When using a shared conductor, apps can call each other's zomes:

```rust
// In Praxis coordinator zome
// Call FL zome to start federated learning round
let fl_response: ExternResult<FlRoundId> = call(
    CallTargetCell::OtherCell(fl_cell_id),
    "fl_coordinator",
    "create_round".into(),
    None,
    create_round_input,
)?;
```

### Pattern 2: Signals (Shared or Separate)

Apps can emit and listen to signals:

```typescript
// Praxis web client
client.on('signal', (signal) => {
  if (signal.type === 'fl_round_complete') {
    // Update UI when FL round completes
    refreshLearningProgress();
  }
});
```

### Pattern 3: DHT Sharing (Shared Conductor Only)

Apps can read each other's DHT entries if they're in the same conductor:

```rust
// Access credential from credential zome while in Praxis zome
let credential: Record = get(credential_hash, GetOptions::default())?;
```

---

## 📦 Multi-App Installation

### Shared Conductor Setup

```bash
# 1. Start shared conductor
./scripts/run-mycelix-ecosystem.sh

# 2. Install all apps
hc app install --admin-port 4444 \
  --app-id mycelix-praxis \
  --bundle dna/praxis.dna

hc app install --admin-port 4444 \
  --app-id mycelix-fl \
  --bundle ../Mycelix-Core/0TML/mycelix_fl/holochain/dna/fl.dna

hc app install --admin-port 4444 \
  --app-id mycelix-core \
  --bundle ../Mycelix-Core/0TML/core.dna

# 3. Enable all apps
hc app enable --admin-port 4444 mycelix-praxis
hc app enable --admin-port 4444 mycelix-fl
hc app enable --admin-port 4444 mycelix-core
```

### Separate Conductor Setup (Development)

```bash
# Terminal 1: Praxis
cd mycelix-praxis
./scripts/run-conductor.sh  # Ports: 4444, 8888

# Terminal 2: FL
cd ../Mycelix-Core/0TML/mycelix_fl/holochain
holochain -c conductor-config.yaml  # Ports: 4445, 8889

# Terminal 3: Core
cd ../core
holochain -c conductor-config.yaml  # Ports: 4446, 8890
```

---

## 🌊 Data Flow Between Apps

### Example: Federated Learning in Education Context

```
1. User completes course in Praxis
   └─> Praxis stores progress in its DHT

2. User opts into FL for course improvement
   └─> Praxis calls FL zome to register for round
       └─> FL zome creates FL round

3. FL round completes gradient aggregation
   └─> FL zome emits signal
       └─> Praxis receives signal
           └─> Praxis updates course recommendations

4. User earns credential for participation
   └─> Praxis calls Credential zome
       └─> Credential zome issues W3C VC
```

---

## 🔐 Security Considerations

### Shared Conductor Security

- ✅ **Single Keystore**: All apps use same agent identity
- ✅ **Zome Permissions**: Use capability grants to control cross-zome calls
- ⚠️ **Trust Boundary**: All apps trust each other (same conductor)

### Separate Conductor Security

- ✅ **Isolated Keystores**: Each app has separate identity
- ✅ **Network Boundary**: Apps communicate via DHT/signals only
- ✅ **Defense in Depth**: Compromise of one app doesn't affect others

---

## 📊 Resource Usage Comparison

### Shared Conductor
- **Memory**: ~500MB-1GB (one conductor process)
- **CPU**: 1-2 cores
- **Disk**: ~100MB (shared databases)
- **Network**: Single DHT connection

### Separate Conductors (4 apps)
- **Memory**: ~2-4GB (four conductor processes)
- **CPU**: 4-8 cores
- **Disk**: ~400MB (separate databases)
- **Network**: Four DHT connections

---

## 🎯 Recommendations

### For Development
**Use Separate Conductors**:
1. Easier to debug and test each app independently
2. Can restart one app without affecting others
3. Clear separation of concerns
4. Follows current directory structure

### For Production
**Use Shared Conductor**:
1. More efficient resource usage
2. Better cross-app integration
3. Simplified deployment
4. Single agent identity across ecosystem

### For Your Current Stage
**Recommendation**: **Continue with separate conductors** for now:

```bash
# Mycelix Praxis (current project)
Admin: 4444
App: 8888

# Mycelix FL (when integrated)
Admin: 4445
App: 8889

# Future integration to shared conductor in Phase 6-7
```

**Reasons**:
1. You're in Phase 2 of Praxis (still building core functionality)
2. FL integration planned for later phases
3. Easier to develop and test independently
4. Can consolidate to shared conductor when ready for production

---

## 🔄 Migration Path

### When to Migrate to Shared Conductor

**Triggers**:
- ✅ All individual apps are stable and tested
- ✅ Cross-app features are needed (FL + Praxis integration)
- ✅ Ready for production deployment
- ✅ Need shared agent identity across apps

### Migration Steps

1. **Test in staging environment**
2. **Create shared conductor config**
3. **Install all DNAs in shared conductor**
4. **Migrate keystore (use same lair root)**
5. **Test cross-app functionality**
6. **Update client connection logic**
7. **Deploy to production**

---

## 🛠️ Implementation Scripts

See:
- `scripts/run-mycelix-ecosystem.sh` - Start shared conductor
- `scripts/run-conductor.sh` - Start Praxis dev conductor
- `conductor/mycelix-ecosystem-conductor.yaml` - Shared conductor config
- `conductor/dev-conductor-config.yaml` - Praxis dev config

---

## 📚 References

- [Holochain Multi-App Patterns](https://developer.holochain.org/concepts/multi-app/)
- [Conductor Configuration Guide](https://developer.holochain.org/guides/conductor-config/)
- [Bridge Calls Documentation](https://developer.holochain.org/concepts/bridging/)

---

**Next Steps**:
1. Continue development with separate conductors
2. Plan cross-app integration points
3. Design shared data schemas
4. Prepare for shared conductor migration in production
