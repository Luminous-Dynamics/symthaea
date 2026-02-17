# 🔄 Holochain 0.6.0 Migration Planning

**Date**: November 12, 2025
**Current Version**: Holochain 0.5.x with @holochain/client
**Target Version**: Holochain 0.6.0 (releasing ~November 19, 2025)
**Status**: ⏳ **Planning Phase - Breaking Changes Expected**

---

## 📋 Breaking Changes Overview

According to the Holochain team, version 0.6.0 will introduce breaking changes in:

### 1. HDI/HDK (Holochain Development Interface/Kit)
**Impact**: Backend zome code
- Integrity zomes may require updates
- Coordinator zomes may require updates
- Type definitions may change

**Action Required**: Review and update zome code when migration guide is released

### 2. Conductor APIs
**Impact**: Backend conductor configuration and management
- Configuration format may change
- Conductor admin API may have breaking changes
- App installation process may differ

**Action Required**: Update conductor setup and admin scripts

### 3. JavaScript Client (@holochain/client)
**Impact**: Frontend integration (our current implementation)
- API method signatures may change
- Connection patterns may differ
- Type definitions will update

**Action Required**: Update all Holochain client calls in frontend

### 4. Network Protocol Incompatibilities
**Impact**: Network communication between nodes
- Nodes running 0.5.x cannot communicate with 0.6.0 nodes
- **Complete network upgrade required**

**Action Required**: Coordinate upgrade across all nodes simultaneously

### 5. Database Schema Incompatibilities
**Impact**: Data migration required
- 0.5.x databases cannot be read by 0.6.0 conductor
- **Data export/import will be necessary**

**Action Required**: Export data before upgrade, re-import after

---

## 🎯 Migration Strategy

### Phase 1: Preparation (Current - Nov 19)
- ✅ Document current API usage
- ✅ Create comprehensive test suite for all Holochain interactions
- ⏳ Monitor for official migration guide release
- ⏳ Review 0.6.0 release notes when available

### Phase 2: Development Testing (Nov 19-26)
- Install 0.6.0 in development environment
- Follow official migration guide
- Update zome code (HDI/HDK)
- Update conductor configuration
- Update frontend client code
- Run comprehensive tests

### Phase 3: Data Migration (Nov 26-30)
- Export existing data from 0.5.x
- Test data import into 0.6.0
- Validate data integrity
- Document migration procedure

### Phase 4: Production Deployment (Dec 1+)
- Schedule maintenance window
- Coordinate network-wide upgrade
- Execute data migration
- Deploy updated application
- Verify network connectivity

---

## 📍 Current Implementation Status

### Frontend Files Using @holochain/client

**Core Client Module**:
- `src/lib/holochain/client.ts` - Connection management
  - `AppWebsocket.connect()`
  - Client initialization
  - Store integration

**Zome Call Modules**:
- `src/lib/holochain/listings.ts` - Listing operations
- `src/lib/holochain/transactions.ts` - Transaction operations
- `src/lib/holochain/users.ts` - User/profile operations
- `src/lib/holochain/disputes.ts` - Dispute/arbitration operations

**Total Holochain Integration Points**: ~40 function calls across 4 modules

### Current API Patterns

```typescript
// Connection pattern (may change in 0.6.0)
const client = await AppWebsocket.connect(url as any);

// Zome call pattern (may change in 0.6.0)
const response = await client.callZome({
  role_name: ROLE_NAME,
  zome_name: ZOME_NAME,
  fn_name: 'function_name',
  payload: input,
});
```

---

## 🔍 Migration Checklist

### Before Upgrade
- [ ] Review official 0.6.0 migration guide
- [ ] Document all current API usage patterns
- [ ] Create backup of 0.5.x database
- [ ] Export all critical data
- [ ] Tag current codebase version in git
- [ ] Run full test suite on 0.5.x (baseline)

### During Upgrade - Backend
- [ ] Update HDI dependency version
- [ ] Update HDK dependency version
- [ ] Fix breaking changes in integrity zomes
- [ ] Fix breaking changes in coordinator zomes
- [ ] Update conductor configuration format
- [ ] Test zome functions in development

### During Upgrade - Frontend
- [ ] Update @holochain/client package version
- [ ] Fix breaking changes in client.ts
- [ ] Update all zome call patterns
- [ ] Fix type definition changes
- [ ] Update error handling if needed
- [ ] Run TypeScript compiler (verify 0 errors)

### Testing Phase
- [ ] Unit test all zome functions
- [ ] Integration test frontend ↔ backend
- [ ] Test data import/migration
- [ ] Verify network connectivity
- [ ] Load testing
- [ ] Security audit

### Production Deployment
- [ ] Schedule maintenance window
- [ ] Notify users of downtime
- [ ] Backup production database
- [ ] Deploy 0.6.0 conductor
- [ ] Deploy updated zomes
- [ ] Deploy updated frontend
- [ ] Import migrated data
- [ ] Verify all functionality
- [ ] Monitor for issues

---

## 📝 Known API Changes (To Be Updated)

**Note**: This section will be populated when the official migration guide is released.

### Expected Changes in @holochain/client

```typescript
// Current (0.5.x)
import { AppWebsocket } from '@holochain/client';
const client = await AppWebsocket.connect(url);

// Possible 0.6.0 pattern (speculation - wait for guide)
// May change to different connection API
```

### Expected Changes in Zome Calls

```typescript
// Current (0.5.x)
const response = await client.callZome({
  role_name: 'marketplace',
  zome_name: 'listings',
  fn_name: 'create_listing',
  payload: input,
});

// Possible 0.6.0 pattern (speculation - wait for guide)
// Method signature may change
```

---

## 🔗 Resources

### Official Documentation
- **0.5.6 Release**: https://github.com/holochain/holochain/releases/tag/holochain-0.5.6
- **0.6.0 Release**: (to be published ~Nov 19, 2025)
- **Migration Guide**: (to be published with 0.6.0)

### Internal Documentation
- **Current Implementation**: `/srv/luminous-dynamics/Mycelix-Core/mycelix-marketplace/frontend/src/lib/holochain/`
- **Phase 4 Complete**: `PHASE_4_COMPLETE_NOV_11_2025.md`
- **Type Definitions**: `src/types/`

---

## ⚠️ Risk Assessment

### High Risk Areas
1. **Network Protocol Breaking**: Complete network upgrade required simultaneously
2. **Database Migration**: Data loss risk if not done carefully
3. **API Changes**: Extensive code changes may introduce bugs

### Mitigation Strategies
1. **Comprehensive Testing**: Test suite before upgrade provides safety net
2. **Phased Rollout**: Dev → Staging → Production
3. **Rollback Plan**: Keep 0.5.x backup available
4. **Documentation**: Document every change for future reference

---

## 📅 Timeline

```
Nov 12, 2025  ← We are here
   ↓
Nov 19, 2025  ← Expected 0.6.0 release + migration guide
   ↓
Nov 19-26     ← Development migration & testing (1 week)
   ↓
Nov 26-30     ← Data migration testing (4 days)
   ↓
Dec 1+        ← Production deployment (coordinated)
```

---

## 💡 Recommendations

### Immediate Actions (This Week)
1. ✅ Create this migration planning document
2. ⏳ Set up monitoring for 0.6.0 release announcement
3. ⏳ Complete Phase 4 testing with current 0.5.x implementation
4. ⏳ Document all current Holochain integration points

### Next Week (When 0.6.0 Releases)
1. Review official migration guide thoroughly
2. Create migration branch in git
3. Begin updating development environment
4. Start frontend code migration

### Ongoing
1. Subscribe to Holochain GitHub releases
2. Join Holochain developer forum/Discord for updates
3. Share migration learnings with community

---

## 🎯 Success Criteria

Migration will be considered successful when:
- ✅ All TypeScript errors resolved (0 errors)
- ✅ All tests passing
- ✅ All 10 pages functional
- ✅ Data successfully migrated
- ✅ Network connectivity established
- ✅ No performance degradation
- ✅ All features working as in 0.5.x

---

**Status**: ⏳ **Planning Complete - Awaiting 0.6.0 Release**

**Next Action**: Monitor https://github.com/holochain/holochain/releases for 0.6.0 announcement

**Point of Contact**: Holochain community forum / GitHub issues for migration questions

---

*This document will be updated as the migration progresses and official guidance becomes available.*
